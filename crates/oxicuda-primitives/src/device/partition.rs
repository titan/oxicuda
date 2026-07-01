//! Device-wide flag-based partition and consecutive-unique compaction.
//!
//! * [`DevicePartitionTemplate`] — split an array into two outputs according to
//!   a predicate: elements satisfying the predicate are written, in stable
//!   order, to buffer **A**; the rest, in stable order, to buffer **B**.  This
//!   mirrors `cub::DevicePartition::If`.
//! * [`DeviceSelectUniqueTemplate`] — compact runs of consecutive duplicates,
//!   keeping the first element of each run (`cub::DeviceSelect::Unique`).
//!
//! # Partition pipeline
//!
//! 1. **flag** — `flag[i] = pred(in[i]) ? 1 : 0`.
//!    Then run an exclusive scan to obtain the *match-rank* `rank_a[i]`.
//!    The B-position is `i - rank_a[i]` (number of non-matches before `i`).
//! 2. **scatter** — matches go to `out_a[rank_a[i]]`, non-matches go to
//!    `out_b[i - rank_a[i]]`.  The total number of matches is
//!    `rank_a[n-1] + flag[n-1]`.
//!
//! # Select-unique pipeline
//!
//! 1. **head** — `head[i] = (i == 0 || in[i] != in[i-1]) ? 1 : 0`.
//!    Then exclusive-scan `head` → `out_idx`.
//! 2. **gather** — for each head element, `out[out_idx[i]] = in[i]`.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::device::partition::{
//!     DevicePartitionConfig, DevicePartitionTemplate,
//! };
//! use oxicuda_primitives::device::select::SelectPredicate;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = DevicePartitionConfig::new(PtxType::S32, SelectPredicate::Positive, 256)
//!     .expect("valid config");
//! let (flag, scatter) = DevicePartitionTemplate::new(cfg).generate(SmVersion::Sm80)
//!     .expect("PTX gen");
//! assert!(flag.contains("partition_flag_positive_s32_bs256"));
//! assert!(scatter.contains("partition_scatter_positive_s32_bs256"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::device::select::SelectPredicate;
use crate::error::{PrimitivesError, PrimitivesResult};
use crate::ptx_helpers::{ptx_header, ptx_type_str};

fn elem_bytes(ty: PtxType) -> u32 {
    match ty {
        PtxType::F64 | PtxType::U64 | PtxType::S64 | PtxType::B64 => 8,
        _ => 4,
    }
}

fn validate_block_size(block_size: u32) -> PrimitivesResult<()> {
    if !(32..=1024).contains(&block_size) || !block_size.is_power_of_two() {
        return Err(PrimitivesError::InvalidArgument(format!(
            "block_size must be a power of two in [32, 1024], got {block_size}"
        )));
    }
    Ok(())
}

/// PTX immediate zero literal for the given element type.
fn ptx_zero_literal(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F32 => "0f00000000",
        PtxType::F64 => "0d0000000000000000",
        _ => "0",
    }
}

/// PTX comparison emitting `%match_pred` from `%val` for a [`SelectPredicate`].
fn comparison_ptx(pred: SelectPredicate, ty: PtxType) -> String {
    let ty_str = ptx_type_str(ty);
    let zero = ptx_zero_literal(ty);
    let is_unsigned = matches!(ty, PtxType::U32 | PtxType::U64);
    match pred {
        SelectPredicate::NonZero => format!("    setp.ne.{ty_str} %match_pred, %val, {zero};"),
        SelectPredicate::Positive => {
            if is_unsigned {
                format!("    setp.ne.{ty_str} %match_pred, %val, {zero};")
            } else {
                format!("    setp.gt.{ty_str} %match_pred, %val, {zero};")
            }
        }
        SelectPredicate::Negative => {
            if is_unsigned {
                "    setp.ne.u32 %match_pred, 0, 0;".to_string()
            } else {
                format!("    setp.lt.{ty_str} %match_pred, %val, {zero};")
            }
        }
        SelectPredicate::FlagArray => "    setp.ne.u32 %match_pred, %flag_u32, 0;".to_string(),
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Partition
// ════════════════════════════════════════════════════════════════════════════

/// Configuration for device-wide flag-based partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePartitionConfig {
    /// Element type.
    pub ty: PtxType,
    /// Predicate selecting which elements go to buffer **A**.
    pub pred: SelectPredicate,
    /// Threads per block (power of 2, `32`–`1024`).
    pub block_size: u32,
}

impl DevicePartitionConfig {
    /// Create a configuration, validating `block_size`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] for an invalid `block_size`.
    pub fn new(ty: PtxType, pred: SelectPredicate, block_size: u32) -> PrimitivesResult<Self> {
        validate_block_size(block_size)?;
        Ok(Self {
            ty,
            pred,
            block_size,
        })
    }

    /// Kernel name for the flag pass.
    #[must_use]
    pub fn flag_kernel_name(&self) -> String {
        format!(
            "partition_flag_{}_{}_bs{}",
            self.pred.name(),
            ptx_type_str(self.ty),
            self.block_size
        )
    }

    /// Kernel name for the scatter pass.
    #[must_use]
    pub fn scatter_kernel_name(&self) -> String {
        format!(
            "partition_scatter_{}_{}_bs{}",
            self.pred.name(),
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX code generator for device-wide flag-based partition.
///
/// Produces `(flag_ptx, scatter_ptx)`.  Between them the caller runs an
/// exclusive prefix scan (sum) on the `u32` flag array; the scan output gives
/// each matching element's rank in buffer A.
pub struct DevicePartitionTemplate {
    /// Configuration.
    pub cfg: DevicePartitionConfig,
}

impl DevicePartitionTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: DevicePartitionConfig) -> Self {
        Self { cfg }
    }

    /// Generate `(flag_ptx, scatter_ptx)`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::PtxGeneration`] on formatting failure.
    pub fn generate(&self, sm: SmVersion) -> PrimitivesResult<(String, String)> {
        Ok((
            self.generate_flag_kernel(sm)?,
            self.generate_scatter_kernel(sm)?,
        ))
    }

    fn generate_flag_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.flag_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = elem_bytes(self.cfg.ty);
        let is_flagarr = self.cfg.pred == SelectPredicate::FlagArray;
        let cmp = comparison_ptx(self.cfg.pred, self.cfg.ty);
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("partition_flag", e);

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_flags,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        if is_flagarr {
            writeln!(out, "    .reg .u32    %flag_u32, %out_flag, %ltid, %bid;").map_err(ferr)?;
        } else {
            writeln!(out, "    .reg .{ty}   %val;").map_err(ferr)?;
            writeln!(out, "    .reg .u32    %out_flag, %ltid, %bid;").map_err(ferr)?;
        }
        writeln!(out, "    .reg .u64    %n, %gid, %ptr_in, %ptr_out, %addr;").map_err(ferr)?;
        writeln!(out, "    .reg .pred   %p, %match_pred;").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_out, [param_flags];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    @%p ret;").map_err(ferr)?;
        if is_flagarr {
            writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_in;").map_err(ferr)?;
            writeln!(out, "    ld.global.u32 %flag_u32, [%addr];").map_err(ferr)?;
        } else {
            writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(ferr)?;
            writeln!(out, "    ld.global.{ty} %val, [%addr];").map_err(ferr)?;
        }
        writeln!(out, "{cmp}").map_err(ferr)?;
        writeln!(out, "    selp.u32     %out_flag, 1, 0, %match_pred;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_out;").map_err(ferr)?;
        writeln!(out, "    st.global.u32 [%addr], %out_flag;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }

    fn generate_scatter_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.scatter_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = elem_bytes(self.cfg.ty);
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("partition_scatter", e);

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_out_a,\n    \
             .param .u64 param_out_b,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_flags,\n    \
             .param .u64 param_rank_a,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{ty}   %val;").map_err(ferr)?;
        writeln!(out, "    .reg .u32    %flag, %ltid, %bid;").map_err(ferr)?;
        writeln!(out, "    .reg .u64    %n, %gid, %rank, %pos_b, %addr;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %ptr_a, %ptr_b, %ptr_in, %ptr_flags, %ptr_rank;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .pred   %p, %is_match;").map_err(ferr)?;

        writeln!(out, "    ld.param.u64 %ptr_a,     [param_out_a];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_b,     [param_out_b];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,    [param_input];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_flags, [param_flags];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_rank,  [param_rank_a];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,          [param_n];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    @%p ret;").map_err(ferr)?;

        // Load value and per-element rank from the exclusive scan of flags.
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %val, [%addr];").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_flags;").map_err(ferr)?;
        writeln!(out, "    ld.global.u32 %flag, [%addr];").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, 8, %ptr_rank;").map_err(ferr)?;
        writeln!(out, "    ld.global.u64 %rank, [%addr];").map_err(ferr)?;

        // Match → out_a[rank]; non-match → out_b[gid - rank].
        writeln!(out, "    setp.ne.u32  %is_match, %flag, 0;").map_err(ferr)?;
        writeln!(out, "    @!%is_match bra PART_TO_B;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %rank, {eb}, %ptr_a;").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %val;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "PART_TO_B:").map_err(ferr)?;
        writeln!(out, "    sub.u64      %pos_b, %gid, %rank;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %pos_b, {eb}, %ptr_b;").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %val;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Select-unique
// ════════════════════════════════════════════════════════════════════════════

/// Configuration for consecutive-duplicate compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceSelectUniqueConfig {
    /// Element type.
    pub ty: PtxType,
    /// Threads per block (power of 2, `32`–`1024`).
    pub block_size: u32,
}

impl DeviceSelectUniqueConfig {
    /// Create a configuration, validating `block_size`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] for an invalid `block_size`.
    pub fn new(ty: PtxType, block_size: u32) -> PrimitivesResult<Self> {
        validate_block_size(block_size)?;
        Ok(Self { ty, block_size })
    }

    /// Kernel name for the head-flag pass.
    #[must_use]
    pub fn head_kernel_name(&self) -> String {
        format!(
            "select_unique_head_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }

    /// Kernel name for the gather pass.
    #[must_use]
    pub fn gather_kernel_name(&self) -> String {
        format!(
            "select_unique_gather_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX code generator for consecutive-duplicate compaction.
///
/// Produces `(head_ptx, gather_ptx)`.  Between them the caller runs an
/// exclusive prefix scan (sum) on the `u32` head-flag array; the scan output
/// gives each unique element's output index.
pub struct DeviceSelectUniqueTemplate {
    /// Configuration.
    pub cfg: DeviceSelectUniqueConfig,
}

impl DeviceSelectUniqueTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: DeviceSelectUniqueConfig) -> Self {
        Self { cfg }
    }

    /// Generate `(head_ptx, gather_ptx)`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::PtxGeneration`] on formatting failure.
    pub fn generate(&self, sm: SmVersion) -> PrimitivesResult<(String, String)> {
        Ok((
            self.generate_head_kernel(sm)?,
            self.generate_gather_kernel(sm)?,
        ))
    }

    fn generate_head_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.head_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = elem_bytes(self.cfg.ty);
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("select_unique_head", e);

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_heads,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{ty}   %cur, %prev;").map_err(ferr)?;
        writeln!(out, "    .reg .u32    %head, %ltid, %bid;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr_in, %ptr_out, %addr, %prev_idx;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .pred   %oob, %is_first, %diff;").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_out, [param_heads];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %oob, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    @%oob ret;").map_err(ferr)?;
        writeln!(out, "    setp.eq.u64  %is_first, %gid, 0;").map_err(ferr)?;
        writeln!(out, "    @%is_first bra SU_HEAD_ONE;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %cur, [%addr];").map_err(ferr)?;
        writeln!(out, "    sub.u64      %prev_idx, %gid, 1;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %prev_idx, {eb}, %ptr_in;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %prev, [%addr];").map_err(ferr)?;
        writeln!(out, "    setp.ne.{ty} %diff, %cur, %prev;").map_err(ferr)?;
        writeln!(out, "    selp.u32     %head, 1, 0, %diff;").map_err(ferr)?;
        writeln!(out, "    bra SU_HEAD_STORE;").map_err(ferr)?;
        writeln!(out, "SU_HEAD_ONE:").map_err(ferr)?;
        writeln!(out, "    mov.u32      %head, 1;").map_err(ferr)?;
        writeln!(out, "SU_HEAD_STORE:").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_out;").map_err(ferr)?;
        writeln!(out, "    st.global.u32 [%addr], %head;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }

    fn generate_gather_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.gather_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = elem_bytes(self.cfg.ty);
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("select_unique_gather", e);

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_output,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_heads,\n    \
             .param .u64 param_out_idx,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{ty}   %val;").map_err(ferr)?;
        writeln!(out, "    .reg .u32    %head, %ltid, %bid;").map_err(ferr)?;
        writeln!(out, "    .reg .u64    %n, %gid, %oidx, %addr;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %ptr_out, %ptr_in, %ptr_head, %ptr_oidx;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .pred   %oob, %keep;").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_out,  [param_output];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,   [param_input];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_head, [param_heads];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_oidx, [param_out_idx];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,         [param_n];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %oob, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    @%oob ret;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_head;").map_err(ferr)?;
        writeln!(out, "    ld.global.u32 %head, [%addr];").map_err(ferr)?;
        writeln!(out, "    setp.ne.u32  %keep, %head, 0;").map_err(ferr)?;
        writeln!(out, "    @!%keep ret;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, 8, %ptr_oidx;").map_err(ferr)?;
        writeln!(out, "    ld.global.u64 %oidx, [%addr];").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %val, [%addr];").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %oidx, {eb}, %ptr_out;").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %val;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }
}

// ─── CPU references ────────────────────────────────────────────────────────────

/// Host reference for flag-based partition.  `keep(x)` selects buffer A.
/// Returns `(buffer_a, buffer_b)` in stable order.
#[must_use]
pub fn reference_partition<T: Copy>(input: &[T], keep: impl Fn(T) -> bool) -> (Vec<T>, Vec<T>) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    for &x in input {
        if keep(x) {
            a.push(x);
        } else {
            b.push(x);
        }
    }
    (a, b)
}

/// Host reference for consecutive-duplicate compaction (keep first of each run).
#[must_use]
pub fn reference_select_unique<T: PartialEq + Copy>(input: &[T]) -> Vec<T> {
    let mut out = Vec::new();
    let mut iter = input.iter().copied();
    if let Some(first) = iter.next() {
        out.push(first);
        let mut prev = first;
        for x in iter {
            if x != prev {
                out.push(x);
                prev = x;
            }
        }
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    #[test]
    fn partition_config_validation_and_names() {
        assert!(DevicePartitionConfig::new(PtxType::S32, SelectPredicate::Positive, 100).is_err());
        let c = DevicePartitionConfig::new(PtxType::S32, SelectPredicate::Positive, 256)
            .expect("valid config");
        assert_eq!(c.flag_kernel_name(), "partition_flag_positive_s32_bs256");
        assert_eq!(
            c.scatter_kernel_name(),
            "partition_scatter_positive_s32_bs256"
        );
    }

    #[test]
    fn partition_flag_ptx_positive_s32() {
        let c = DevicePartitionConfig::new(PtxType::S32, SelectPredicate::Positive, 256)
            .expect("valid config");
        let (flag, _) = DevicePartitionTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(flag.contains("setp.gt.s32 %match_pred"), "PTX: {flag}");
        assert!(flag.contains("selp.u32     %out_flag, 1, 0"), "PTX: {flag}");
    }

    #[test]
    fn partition_scatter_ptx_routes_a_and_b() {
        let c = DevicePartitionConfig::new(PtxType::F32, SelectPredicate::NonZero, 128)
            .expect("valid config");
        let (_, scatter) = DevicePartitionTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(scatter.contains("PART_TO_B"), "PTX: {scatter}");
        assert!(
            scatter.contains("sub.u64      %pos_b, %gid, %rank"),
            "PTX: {scatter}"
        );
        assert!(scatter.contains("param_out_a"), "PTX: {scatter}");
        assert!(scatter.contains("param_out_b"), "PTX: {scatter}");
    }

    #[test]
    fn partition_flagarray_reads_u32() {
        let c = DevicePartitionConfig::new(PtxType::F32, SelectPredicate::FlagArray, 256)
            .expect("valid config");
        let (flag, _) = DevicePartitionTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(flag.contains("ld.global.u32 %flag_u32"), "PTX: {flag}");
        assert!(
            flag.contains("setp.ne.u32 %match_pred, %flag_u32, 0"),
            "PTX: {flag}"
        );
    }

    #[test]
    fn select_unique_config_and_names() {
        let c = DeviceSelectUniqueConfig::new(PtxType::U64, 512).expect("valid config");
        assert_eq!(c.head_kernel_name(), "select_unique_head_u64_bs512");
        assert_eq!(c.gather_kernel_name(), "select_unique_gather_u64_bs512");
    }

    #[test]
    fn select_unique_head_ptx_compares_neighbours() {
        let c = DeviceSelectUniqueConfig::new(PtxType::U32, 256).expect("valid config");
        let (head, gather) = DeviceSelectUniqueTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(head.contains("setp.ne.u32 %diff"), "PTX: {head}");
        assert!(head.contains("SU_HEAD_ONE"), "PTX: {head}");
        assert!(gather.contains("param_out_idx"), "PTX: {gather}");
        assert!(gather.contains("@!%keep ret"), "PTX: {gather}");
    }

    #[test]
    fn select_unique_gather_u64_uses_8byte_stride() {
        let c = DeviceSelectUniqueConfig::new(PtxType::F64, 256).expect("valid config");
        let (_, gather) = DeviceSelectUniqueTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(gather.contains("ld.global.f64"), "PTX: {gather}");
        assert!(gather.contains("st.global.f64"), "PTX: {gather}");
    }

    #[test]
    fn reference_partition_stable() {
        let (a, b) = reference_partition(&[1i32, -2, 3, -4, 5], |x| x > 0);
        assert_eq!(a, vec![1, 3, 5]);
        assert_eq!(b, vec![-2, -4]);
    }

    #[test]
    fn reference_partition_all_or_none() {
        let (a, b) = reference_partition(&[1i32, 2, 3], |x| x > 0);
        assert_eq!(a, vec![1, 2, 3]);
        assert!(b.is_empty());
        let (a2, b2) = reference_partition(&[1i32, 2, 3], |x| x < 0);
        assert!(a2.is_empty());
        assert_eq!(b2, vec![1, 2, 3]);
    }

    #[test]
    fn reference_unique_basic() {
        assert_eq!(
            reference_select_unique(&[1u32, 1, 2, 2, 2, 3, 1, 1]),
            vec![1, 2, 3, 1]
        );
        assert_eq!(reference_select_unique::<u32>(&[]), Vec::<u32>::new());
        assert_eq!(reference_select_unique(&[5u32]), vec![5]);
        assert_eq!(reference_select_unique(&[7u32, 7, 7]), vec![7]);
    }
}

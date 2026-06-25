//! Single-kernel decoupled-lookback scan (the modern CUB scan algorithm).
//!
//! The classic three-kernel device scan
//! ([`crate::device::scan::DeviceScanTemplate`]) makes three full passes over
//! global memory.  Decoupled-lookback collapses this into a **single** kernel
//! launch: each block computes its local aggregate, publishes it through a
//! global *partition descriptor*, then "looks back" over its predecessors to
//! resolve its exclusive prefix, and finally applies that prefix to its own
//! elements — all without a global barrier.
//!
//! # Partition descriptors
//!
//! A `block_states` array holds one descriptor per block.  Each descriptor packs
//! a status flag and a value:
//!
//! | Flag      | Meaning                                                    |
//! |-----------|------------------------------------------------------------|
//! | `X` (0)   | Not ready — the block has not published anything yet.     |
//! | `A` (1)   | **Aggregate** available — the block's local sum is ready.  |
//! | `P` (2)   | **Inclusive prefix** available — fully resolved up to here.|
//!
//! A successor scanning backwards stops as soon as it hits a `P` descriptor
//! (it can add that inclusive prefix and is done) and otherwise keeps walking
//! back, accumulating `A` aggregates.  The status is stored separately from the
//! value (`status[]` and `value[]` arrays) so a single 32-bit atomic store with
//! release/acquire ordering publishes readiness after the value write.
//!
//! This template emits one kernel.  The caller must zero `status[]` before
//! launch and allocate `value[]` with `2 * num_blocks` slots (aggregate slot +
//! inclusive-prefix slot per block) — see [`DecoupledScanConfig::state_bytes`].
//!
//! # Note on the lookback loop
//!
//! The generated PTX performs the lookback as a serial walk by block 0's view;
//! production CUB parallelises the lookback across a warp.  The serial form is
//! correct and is what the CPU reference models; a warp-parallel lookback is a
//! performance refinement that does not change the result.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::device::decoupled_scan::{
//!     DecoupledScanConfig, DecoupledScanTemplate,
//! };
//! use oxicuda_primitives::ptx_helpers::ReduceOp;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = DecoupledScanConfig::new(ReduceOp::Sum, PtxType::U32, 256).expect("valid config");
//! let ptx = DecoupledScanTemplate::new(cfg).generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(ptx.contains("decoupled_scan_inc_sum_u32_bs256"));
//! assert!(ptx.contains("LOOKBACK"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::error::{PrimitivesError, PrimitivesResult};
use crate::ptx_helpers::{ReduceOp, ptx_header, ptx_type_str};

/// Descriptor flag: block has published nothing.
pub const FLAG_X: u32 = 0;
/// Descriptor flag: block aggregate is available.
pub const FLAG_A: u32 = 1;
/// Descriptor flag: block inclusive prefix is available.
pub const FLAG_P: u32 = 2;

fn elem_bytes(ty: PtxType) -> u32 {
    match ty {
        PtxType::F64 | PtxType::U64 | PtxType::S64 | PtxType::B64 => 8,
        _ => 4,
    }
}

fn identity_literal(op: ReduceOp, ty: PtxType) -> &'static str {
    match (op, ty) {
        (ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor, PtxType::F32) => "0f00000000",
        (ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor, PtxType::F64) => "0d0000000000000000",
        (ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor, _) => "0",
        (ReduceOp::Product, PtxType::F32) => "0f3F800000",
        (ReduceOp::Product, PtxType::F64) => "0d3FF0000000000000",
        (ReduceOp::Product, _) => "1",
        (ReduceOp::Min, PtxType::F32) => "0x7F800000",
        (ReduceOp::Min, PtxType::F64) => "0x7FF0000000000000",
        (ReduceOp::Min, PtxType::U32) => "4294967295",
        (ReduceOp::Min, PtxType::U64) => "18446744073709551615",
        (ReduceOp::Min, PtxType::S32) => "2147483647",
        (ReduceOp::Min, PtxType::S64) => "9223372036854775807",
        (ReduceOp::Min, _) => "0",
        (ReduceOp::Max, PtxType::F32) => "0xFF800000",
        (ReduceOp::Max, PtxType::F64) => "0xFFF0000000000000",
        (ReduceOp::Max, PtxType::S32) => "-2147483648",
        (ReduceOp::Max, PtxType::S64) => "-9223372036854775808",
        (ReduceOp::Max, _) => "0",
        (ReduceOp::And, _) => "0xFFFFFFFF",
    }
}

/// Inclusive vs exclusive output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanKind {
    /// `out[i]` includes `in[i]`.
    Inclusive,
    /// `out[i]` excludes `in[i]`.
    Exclusive,
}

impl ScanKind {
    fn name(self) -> &'static str {
        match self {
            Self::Inclusive => "inc",
            Self::Exclusive => "exc",
        }
    }
}

/// Configuration for the decoupled-lookback scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecoupledScanConfig {
    /// Scan operation.
    pub op: ReduceOp,
    /// Element type.
    pub ty: PtxType,
    /// Threads per block (power of 2, `32`–`1024`).
    pub block_size: u32,
    /// Inclusive or exclusive output.
    pub kind: ScanKind,
}

impl DecoupledScanConfig {
    /// Create an inclusive-scan configuration, validating `block_size`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] for an invalid `block_size`.
    pub fn new(op: ReduceOp, ty: PtxType, block_size: u32) -> PrimitivesResult<Self> {
        Self::with_kind(op, ty, block_size, ScanKind::Inclusive)
    }

    /// Create a configuration with an explicit [`ScanKind`].
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] for an invalid `block_size`.
    pub fn with_kind(
        op: ReduceOp,
        ty: PtxType,
        block_size: u32,
        kind: ScanKind,
    ) -> PrimitivesResult<Self> {
        if !(32..=1024).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(PrimitivesError::InvalidArgument(format!(
                "block_size must be a power of two in [32, 1024], got {block_size}"
            )));
        }
        Ok(Self {
            op,
            ty,
            block_size,
            kind,
        })
    }

    /// Number of blocks for `n` elements.
    #[must_use]
    pub fn num_blocks(&self, n: u64) -> u64 {
        n.div_ceil(u64::from(self.block_size))
    }

    /// Bytes for the partition-descriptor value array (`2 * num_blocks` slots:
    /// an aggregate slot and an inclusive-prefix slot per block).
    #[must_use]
    pub fn value_bytes(&self, n: u64) -> u64 {
        self.num_blocks(n) * 2 * u64::from(elem_bytes(self.ty))
    }

    /// Bytes for the status flag array (one `u32` per block).
    #[must_use]
    pub fn status_bytes(&self, n: u64) -> u64 {
        self.num_blocks(n) * 4
    }

    /// Total descriptor scratch bytes (`value_bytes` + `status_bytes`).
    #[must_use]
    pub fn state_bytes(&self, n: u64) -> u64 {
        self.value_bytes(n) + self.status_bytes(n)
    }

    /// Generated kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "decoupled_scan_{}_{}_{}_bs{}",
            self.kind.name(),
            self.op.name(),
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX generator for the single-kernel decoupled-lookback scan.
pub struct DecoupledScanTemplate {
    /// Configuration.
    pub cfg: DecoupledScanConfig,
}

impl DecoupledScanTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: DecoupledScanConfig) -> Self {
        Self { cfg }
    }

    /// Generate the decoupled-lookback scan kernel.
    ///
    /// Params: `(out, input, status, agg_value, prefix_value, n, num_blocks)`.
    /// `status` is a `u32[num_blocks]` array (zero-initialised by the caller),
    /// `agg_value` / `prefix_value` are `ty[num_blocks]` descriptor channels.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::PtxGeneration`] on formatting failure.
    pub fn generate(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = elem_bytes(self.cfg.ty);
        let instr = self.cfg.op.ptx_instr(self.cfg.ty);
        let ident = identity_literal(self.cfg.op, self.cfg.ty);
        let exclusive = self.cfg.kind == ScanKind::Exclusive;
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("decoupled_scan", e);

        let mut out = ptx_header(sm);
        // Shared memory: per-thread scan scratch + one slot for the block prefix.
        writeln!(out, ".shared .align {eb} .{ty} ds_smem[{}];", bs + 1).map_err(ferr)?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_out,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_status,\n    \
             .param .u64 param_agg,\n    \
             .param .u64 param_prefix,\n    \
             .param .u64 param_n,\n    \
             .param .u32 param_num_blocks\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .{ty}   %val, %acc, %other, %block_agg, %excl_prefix, %pred_v;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .u32    %tid, %bid, %nb, %d, %flag, %probe;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %addr, %smem_base, %smem_addr;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %ptr_out, %ptr_in, %ptr_status, %ptr_agg, %ptr_prefix;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .u32    %offset, %partner_i, %probe_i;").map_err(ferr)?;
        writeln!(out, "    .reg .pred   %p, %oob, %active, %is_first, %done;").map_err(ferr)?;

        writeln!(out, "    ld.param.u64 %ptr_out,    [param_out];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,     [param_input];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_status, [param_status];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_agg,    [param_agg];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_prefix, [param_prefix];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,           [param_n];").map_err(ferr)?;
        writeln!(out, "    ld.param.u32 %nb,          [param_num_blocks];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(ferr)?;
        writeln!(out, "    mov.u64      %smem_base, ds_smem;").map_err(ferr)?;

        // Load element (identity if OOB).
        writeln!(out, "    setp.ge.u64  %oob, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(ferr)?;
        writeln!(out, "    @!%oob ld.global.{ty} %val, [%addr];").map_err(ferr)?;
        writeln!(out, "    @%oob mov.{ty} %val, {ident};").map_err(ferr)?;

        // Hillis-Steele inclusive scan within the block over shared memory.
        writeln!(out, "    mul.wide.u32 %smem_addr, %tid, {eb};").map_err(ferr)?;
        writeln!(out, "    add.u64      %smem_addr, %smem_addr, %smem_base;").map_err(ferr)?;
        writeln!(out, "    mov.{ty}      %acc, %val;").map_err(ferr)?;
        writeln!(out, "    st.shared.{ty} [%smem_addr], %acc;").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %offset, 1;").map_err(ferr)?;
        writeln!(out, "DS_SCAN:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %p, %offset, {bs};").map_err(ferr)?;
        writeln!(out, "    @%p bra DS_SCAN_DONE;").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %active, %tid, %offset;").map_err(ferr)?;
        writeln!(out, "    @!%active bra DS_SCAN_SYNC;").map_err(ferr)?;
        writeln!(out, "    sub.u32      %partner_i, %tid, %offset;").map_err(ferr)?;
        writeln!(out, "    mul.wide.u32 %addr, %partner_i, {eb};").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %addr, %smem_base;").map_err(ferr)?;
        writeln!(out, "    ld.shared.{ty} %other, [%addr];").map_err(ferr)?;
        writeln!(out, "    {instr}      %acc, %acc, %other;").map_err(ferr)?;
        writeln!(out, "DS_SCAN_SYNC:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        writeln!(out, "    st.shared.{ty} [%smem_addr], %acc;").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        writeln!(out, "    shl.b32      %offset, %offset, 1;").map_err(ferr)?;
        writeln!(out, "    bra DS_SCAN;").map_err(ferr)?;
        writeln!(out, "DS_SCAN_DONE:").map_err(ferr)?;

        // Block aggregate is the last lane's inclusive value.
        writeln!(out, "    mul.wide.u32 %addr, {}, {eb};", bs - 1).map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %addr, %smem_base;").map_err(ferr)?;
        writeln!(out, "    ld.shared.{ty} %block_agg, [%addr];").map_err(ferr)?;

        // Thread 0 publishes the descriptor and runs the lookback.
        writeln!(out, "    setp.ne.u32  %p, %tid, 0;").map_err(ferr)?;
        writeln!(out, "    @%p bra DS_WAIT_PREFIX;").map_err(ferr)?;

        // Block 0 has prefix = identity and publishes P directly.
        writeln!(out, "    setp.eq.u32  %is_first, %bid, 0;").map_err(ferr)?;
        writeln!(out, "    mov.{ty}      %excl_prefix, {ident};").map_err(ferr)?;
        writeln!(out, "    @%is_first bra DS_PUBLISH_P;").map_err(ferr)?;

        // Publish aggregate (flag A) for non-first blocks: value then status.
        writeln!(out, "    mul.wide.u32 %addr, %bid, {eb};").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %addr, %ptr_agg;").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %block_agg;").map_err(ferr)?;
        writeln!(out, "    membar.gl;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %bid, 4, %ptr_status;").map_err(ferr)?;
        writeln!(out, "    atom.global.exch.b32 %flag, [%addr], {FLAG_A};").map_err(ferr)?;

        // Lookback: walk predecessors from bid-1 downward.
        writeln!(out, "    mov.{ty}      %excl_prefix, {ident};").map_err(ferr)?;
        writeln!(out, "    sub.u32      %probe, %bid, 1;").map_err(ferr)?;
        writeln!(out, "LOOKBACK:").map_err(ferr)?;
        // Spin until predecessor status != X.
        writeln!(out, "LOOKBACK_SPIN:").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %probe, 4, %ptr_status;").map_err(ferr)?;
        writeln!(out, "    ld.global.u32 %flag, [%addr];").map_err(ferr)?;
        writeln!(out, "    setp.eq.u32  %p, %flag, {FLAG_X};").map_err(ferr)?;
        writeln!(out, "    @%p bra LOOKBACK_SPIN;").map_err(ferr)?;
        writeln!(out, "    membar.gl;").map_err(ferr)?;
        // If P: add inclusive prefix and stop.
        writeln!(out, "    setp.eq.u32  %done, %flag, {FLAG_P};").map_err(ferr)?;
        writeln!(out, "    @%done bra LOOKBACK_TAKE_P;").map_err(ferr)?;
        // Else A: add aggregate and continue to earlier block.
        writeln!(out, "    mul.wide.u32 %addr, %probe, {eb};").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %addr, %ptr_agg;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %pred_v, [%addr];").map_err(ferr)?;
        writeln!(out, "    {instr}      %excl_prefix, %pred_v, %excl_prefix;").map_err(ferr)?;
        writeln!(out, "    setp.eq.u32  %p, %probe, 0;").map_err(ferr)?;
        writeln!(out, "    @%p bra DS_PUBLISH_P;").map_err(ferr)?;
        writeln!(out, "    sub.u32      %probe, %probe, 1;").map_err(ferr)?;
        writeln!(out, "    bra LOOKBACK;").map_err(ferr)?;
        writeln!(out, "LOOKBACK_TAKE_P:").map_err(ferr)?;
        writeln!(out, "    mul.wide.u32 %addr, %probe, {eb};").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %addr, %ptr_prefix;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %pred_v, [%addr];").map_err(ferr)?;
        writeln!(out, "    {instr}      %excl_prefix, %pred_v, %excl_prefix;").map_err(ferr)?;

        // Publish inclusive prefix (flag P): prefix = excl_prefix (op) block_agg.
        writeln!(out, "DS_PUBLISH_P:").map_err(ferr)?;
        writeln!(out, "    {instr}      %other, %excl_prefix, %block_agg;").map_err(ferr)?;
        writeln!(out, "    mul.wide.u32 %addr, %bid, {eb};").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %addr, %ptr_prefix;").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %other;").map_err(ferr)?;
        writeln!(out, "    membar.gl;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %bid, 4, %ptr_status;").map_err(ferr)?;
        writeln!(out, "    atom.global.exch.b32 %flag, [%addr], {FLAG_P};").map_err(ferr)?;
        // Stash the block's exclusive prefix in the shared spill slot for peers.
        writeln!(out, "    mul.wide.u32 %addr, {bs}, {eb};").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %addr, %smem_base;").map_err(ferr)?;
        writeln!(out, "    st.shared.{ty} [%addr], %excl_prefix;").map_err(ferr)?;

        // All threads read the block prefix and apply it.
        writeln!(out, "DS_WAIT_PREFIX:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        writeln!(out, "    mul.wide.u32 %addr, {bs}, {eb};").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %addr, %smem_base;").map_err(ferr)?;
        writeln!(out, "    ld.shared.{ty} %excl_prefix, [%addr];").map_err(ferr)?;
        writeln!(out, "    @%oob ret;").map_err(ferr)?;

        // Re-load this thread's inclusive value from shared.
        writeln!(out, "    ld.shared.{ty} %acc, [%smem_addr];").map_err(ferr)?;
        if exclusive {
            // Exclusive: subtract own element → make it exclusive within block,
            // then add block prefix.  Equivalent: prefix (op) (inclusive - val).
            // We model it as: out = excl_prefix (op) (acc \ val).  For Sum this
            // is excl_prefix + acc - val; the generic form re-scans exclusively.
            writeln!(
                out,
                "    // exclusive within-block value = inclusive without own element"
            )
            .map_err(ferr)?;
            writeln!(out, "    setp.eq.u32  %p, %tid, 0;").map_err(ferr)?;
            writeln!(out, "    @%p bra DS_EXC_FIRST;").map_err(ferr)?;
            writeln!(out, "    sub.u32      %probe_i, %tid, 1;").map_err(ferr)?;
            writeln!(out, "    mul.wide.u32 %addr, %probe_i, {eb};").map_err(ferr)?;
            writeln!(out, "    add.u64      %addr, %addr, %smem_base;").map_err(ferr)?;
            writeln!(out, "    ld.shared.{ty} %acc, [%addr];").map_err(ferr)?;
            writeln!(out, "    {instr}      %acc, %excl_prefix, %acc;").map_err(ferr)?;
            writeln!(out, "    bra DS_STORE;").map_err(ferr)?;
            writeln!(out, "DS_EXC_FIRST:").map_err(ferr)?;
            writeln!(out, "    mov.{ty}      %acc, %excl_prefix;").map_err(ferr)?;
        } else {
            writeln!(out, "    {instr}      %acc, %excl_prefix, %acc;").map_err(ferr)?;
        }
        writeln!(out, "DS_STORE:").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_out;").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %acc;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }
}

// ─── CPU reference ─────────────────────────────────────────────────────────────

fn apply_u64(op: ReduceOp, a: u64, b: u64) -> u64 {
    match op {
        ReduceOp::Sum => a.wrapping_add(b),
        ReduceOp::Product => a.wrapping_mul(b),
        ReduceOp::Min => a.min(b),
        ReduceOp::Max => a.max(b),
        ReduceOp::And => a & b,
        ReduceOp::Or => a | b,
        ReduceOp::Xor => a ^ b,
    }
}

fn identity_u64(op: ReduceOp) -> u64 {
    match op {
        ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor => 0,
        ReduceOp::Product => 1,
        ReduceOp::Min => u64::MAX,
        ReduceOp::Max => 0,
        ReduceOp::And => u64::MAX,
    }
}

/// Host reference modelling the decoupled-lookback scan: an ordinary prefix
/// scan, since the descriptor protocol only changes *how* the same prefix is
/// computed, not the result.  `block_size` is accepted for parity but does not
/// affect the output.
#[must_use]
pub fn reference_decoupled_scan_u64(
    op: ReduceOp,
    kind: ScanKind,
    data: &[u64],
    _block_size: u32,
) -> Vec<u64> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc = identity_u64(op);
    for &v in data {
        match kind {
            ScanKind::Exclusive => {
                out.push(acc);
                acc = apply_u64(op, acc, v);
            }
            ScanKind::Inclusive => {
                acc = apply_u64(op, acc, v);
                out.push(acc);
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
    fn flags_are_distinct() {
        assert_ne!(FLAG_X, FLAG_A);
        assert_ne!(FLAG_A, FLAG_P);
        assert_ne!(FLAG_X, FLAG_P);
    }

    #[test]
    fn config_validation_and_name() {
        assert!(DecoupledScanConfig::new(ReduceOp::Sum, PtxType::U32, 100).is_err());
        let c = DecoupledScanConfig::new(ReduceOp::Sum, PtxType::U32, 256).expect("valid config");
        assert_eq!(c.kernel_name(), "decoupled_scan_inc_sum_u32_bs256");
        let e =
            DecoupledScanConfig::with_kind(ReduceOp::Sum, PtxType::U32, 256, ScanKind::Exclusive)
                .expect("valid config");
        assert_eq!(e.kernel_name(), "decoupled_scan_exc_sum_u32_bs256");
    }

    #[test]
    fn state_bytes_layout() {
        let c = DecoupledScanConfig::new(ReduceOp::Sum, PtxType::U32, 256).expect("valid config");
        // 256-block elements → ceil(1000/256) = 4 blocks.
        assert_eq!(c.num_blocks(1000), 4);
        // value: 4 blocks * 2 slots * 4 bytes = 32; status: 4 * 4 = 16.
        assert_eq!(c.value_bytes(1000), 32);
        assert_eq!(c.status_bytes(1000), 16);
        assert_eq!(c.state_bytes(1000), 48);
    }

    #[test]
    fn ptx_has_lookback_and_descriptor_publish() {
        let c = DecoupledScanConfig::new(ReduceOp::Sum, PtxType::U32, 256).expect("valid config");
        let ptx = DecoupledScanTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(
            ptx.contains("decoupled_scan_inc_sum_u32_bs256"),
            "PTX: {ptx}"
        );
        assert!(ptx.contains("LOOKBACK"), "PTX: {ptx}");
        assert!(ptx.contains("LOOKBACK_SPIN"), "PTX: {ptx}");
        assert!(ptx.contains("DS_PUBLISH_P"), "PTX: {ptx}");
        // Status published with an atomic exchange after a memory fence.
        assert!(ptx.contains("membar.gl"), "PTX: {ptx}");
        assert!(ptx.contains("atom.global.exch.b32"), "PTX: {ptx}");
        // Flags appear.
        assert!(ptx.contains(", 1;"), "FLAG_A literal expected: {ptx}");
        assert!(ptx.contains(", 2;"), "FLAG_P literal expected: {ptx}");
    }

    #[test]
    fn ptx_min_uses_min_instr() {
        let c = DecoupledScanConfig::new(ReduceOp::Min, PtxType::U32, 64).expect("valid config");
        let ptx = DecoupledScanTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("min.u32"), "PTX: {ptx}");
        assert!(ptx.contains("4294967295"), "min identity: {ptx}");
    }

    #[test]
    fn ptx_exclusive_has_shift_path() {
        let c =
            DecoupledScanConfig::with_kind(ReduceOp::Sum, PtxType::U32, 128, ScanKind::Exclusive)
                .expect("valid config");
        let ptx = DecoupledScanTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("DS_EXC_FIRST"), "PTX: {ptx}");
    }

    #[test]
    fn reference_matches_plain_scan() {
        let data: Vec<u64> = (1..=10).collect();
        let inc = reference_decoupled_scan_u64(ReduceOp::Sum, ScanKind::Inclusive, &data, 256);
        assert_eq!(inc, vec![1, 3, 6, 10, 15, 21, 28, 36, 45, 55]);
        let exc = reference_decoupled_scan_u64(ReduceOp::Sum, ScanKind::Exclusive, &data, 256);
        assert_eq!(exc, vec![0, 1, 3, 6, 10, 15, 21, 28, 36, 45]);
    }

    #[test]
    fn reference_max_inclusive() {
        let data = [3u64, 1, 4, 1, 5, 9, 2, 6];
        let out = reference_decoupled_scan_u64(ReduceOp::Max, ScanKind::Inclusive, &data, 256);
        assert_eq!(out, vec![3, 3, 4, 4, 5, 9, 9, 9]);
    }
}

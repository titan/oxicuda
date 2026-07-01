//! Device-wide segmented reduce and segmented scan.
//!
//! Both primitives take a flat data array plus a `segment offsets` array of
//! length `num_segments + 1`.  Segment `s` spans the half-open index range
//! `[offsets[s], offsets[s+1])`.  This is the layout used by CSR sparse
//! matrices, ragged / jagged batches (GraphRS, dynamic batching), and grouped
//! aggregations.
//!
//! * [`SegmentedReduceTemplate`] — one aggregate value per segment.
//! * [`SegmentedScanTemplate`] — per-segment inclusive / exclusive prefix op.
//!
//! # Kernel mapping
//!
//! * **Segmented reduce** launches one block per segment.  Each block strides
//!   over its segment, accumulating into a per-thread partial, then reduces the
//!   partials in shared memory to a single value written to `out[s]`.
//! * **Segmented scan** launches one thread per segment.  Each thread performs
//!   a sequential prefix scan over its (typically short) segment.  This favours
//!   the many-short-segments regime that dominates sparse / ragged workloads;
//!   long segments can instead reuse [`crate::device::scan::DeviceScanTemplate`]
//!   per-segment.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::device::segmented::{
//!     SegmentedReduceConfig, SegmentedReduceTemplate,
//! };
//! use oxicuda_primitives::ptx_helpers::ReduceOp;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = SegmentedReduceConfig::new(ReduceOp::Sum, PtxType::F32, 256).expect("valid config");
//! let ptx = SegmentedReduceTemplate::new(cfg).generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(ptx.contains("seg_reduce_sum_f32_bs256"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::error::{PrimitivesError, PrimitivesResult};
use crate::ptx_helpers::{ReduceOp, ptx_header, ptx_type_str};

// ─── Shared helpers ────────────────────────────────────────────────────────────

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

/// Identity literal for a reduce op on the given type, as a PTX immediate.
fn identity_literal(op: ReduceOp, ty: PtxType) -> &'static str {
    match (op, ty) {
        (ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor, PtxType::F32) => "0f00000000",
        (ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor, PtxType::F64) => "0d0000000000000000",
        (ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor, _) => "0",
        (ReduceOp::Product, PtxType::F32) => "0f3F800000", // 1.0f
        (ReduceOp::Product, PtxType::F64) => "0d3FF0000000000000", // 1.0
        (ReduceOp::Product, _) => "1",
        (ReduceOp::Min, PtxType::F32) => "0f7F800000",
        (ReduceOp::Min, PtxType::F64) => "0x7FF0000000000000",
        (ReduceOp::Min, PtxType::U32) => "4294967295",
        (ReduceOp::Min, PtxType::U64) => "18446744073709551615",
        (ReduceOp::Min, PtxType::S32) => "2147483647",
        (ReduceOp::Min, PtxType::S64) => "9223372036854775807",
        (ReduceOp::Min, _) => "0",
        (ReduceOp::Max, PtxType::F32) => "0fFF800000",
        (ReduceOp::Max, PtxType::F64) => "0xFFF0000000000000",
        (ReduceOp::Max, PtxType::S32) => "-2147483648",
        (ReduceOp::Max, PtxType::S64) => "-9223372036854775808",
        (ReduceOp::Max, _) => "0",
        (ReduceOp::And, _) => "0xFFFFFFFF",
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Segmented reduce
// ════════════════════════════════════════════════════════════════════════════

/// Configuration for device-wide segmented reduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentedReduceConfig {
    /// Reduction operation.
    pub op: ReduceOp,
    /// Element type of the data array.
    pub ty: PtxType,
    /// Threads per block (power of 2, `32`–`1024`).  One block per segment.
    pub block_size: u32,
}

impl SegmentedReduceConfig {
    /// Create a configuration, validating `block_size`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] for an invalid `block_size`.
    pub fn new(op: ReduceOp, ty: PtxType, block_size: u32) -> PrimitivesResult<Self> {
        validate_block_size(block_size)?;
        Ok(Self { op, ty, block_size })
    }

    /// Generated kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "seg_reduce_{}_{}_bs{}",
            self.op.name(),
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX code generator for device-wide segmented reduce.
pub struct SegmentedReduceTemplate {
    /// Configuration.
    pub cfg: SegmentedReduceConfig,
}

impl SegmentedReduceTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: SegmentedReduceConfig) -> Self {
        Self { cfg }
    }

    /// Generate the segmented-reduce PTX kernel.
    ///
    /// Launch with `grid = num_segments`, `block = block_size`.  Params:
    /// `(out, input, offsets, num_segments)`.
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
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("seg_reduce", e);

        let mut out = ptx_header(sm);
        // Shared scratch for the block-level partial reduction.
        writeln!(out, ".shared .align 8 .b8 seg_red_smem[{}];", bs * eb).map_err(ferr)?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_out,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_offsets,\n    \
             .param .u64 param_num_segments\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{ty}   %acc, %val, %other;").map_err(ferr)?;
        writeln!(out, "    .reg .u32    %ltid, %seg, %stride_t;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %nseg, %seg_beg, %seg_end, %i, %addr, %off_addr;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %ptr_out, %ptr_in, %ptr_off, %smem_base, %smem_addr;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .u32    %half, %partner;").map_err(ferr)?;
        writeln!(out, "    .reg .pred   %p, %loop_p, %active;").map_err(ferr)?;

        writeln!(out, "    ld.param.u64 %ptr_out, [param_out];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_off, [param_offsets];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %nseg,    [param_num_segments];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %seg, %ctaid.x;").map_err(ferr)?;
        // Out-of-range block guard (grid may be rounded up).
        writeln!(out, "    cvt.u64.u32  %i, %seg;").map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %p, %i, %nseg;").map_err(ferr)?;
        writeln!(out, "    @%p ret;").map_err(ferr)?;

        // Load segment bounds: offsets[seg], offsets[seg+1].
        writeln!(out, "    mad.lo.u64   %off_addr, %i, 8, %ptr_off;").map_err(ferr)?;
        writeln!(out, "    ld.global.u64 %seg_beg, [%off_addr];").map_err(ferr)?;
        writeln!(out, "    ld.global.u64 %seg_end, [%off_addr+8];").map_err(ferr)?;

        // Per-thread strided accumulation starting at identity.
        writeln!(out, "    mov.{ty}      %acc, {ident};").map_err(ferr)?;
        writeln!(out, "    cvt.u64.u32  %i, %ltid;").map_err(ferr)?;
        writeln!(out, "    add.u64      %i, %i, %seg_beg;").map_err(ferr)?;
        writeln!(out, "SEG_RED_LOOP:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %loop_p, %i, %seg_end;").map_err(ferr)?;
        writeln!(out, "    @%loop_p bra SEG_RED_REDUCE;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %i, {eb}, %ptr_in;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %val, [%addr];").map_err(ferr)?;
        writeln!(out, "    {instr}      %acc, %acc, %val;").map_err(ferr)?;
        writeln!(out, "    add.u64      %i, %i, {bs};").map_err(ferr)?;
        writeln!(out, "    bra SEG_RED_LOOP;").map_err(ferr)?;

        // Tree reduction of per-thread partials through shared memory.
        writeln!(out, "SEG_RED_REDUCE:").map_err(ferr)?;
        writeln!(out, "    mov.u64      %smem_base, seg_red_smem;").map_err(ferr)?;
        writeln!(out, "    mul.wide.u32 %smem_addr, %ltid, {eb};").map_err(ferr)?;
        writeln!(out, "    add.u64      %smem_addr, %smem_addr, %smem_base;").map_err(ferr)?;
        writeln!(out, "    st.shared.{ty} [%smem_addr], %acc;").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %half, {};", bs / 2).map_err(ferr)?;
        writeln!(out, "SEG_RED_TREE:").map_err(ferr)?;
        writeln!(out, "    setp.eq.u32  %p, %half, 0;").map_err(ferr)?;
        writeln!(out, "    @%p bra SEG_RED_WRITE;").map_err(ferr)?;
        writeln!(out, "    setp.lt.u32  %active, %ltid, %half;").map_err(ferr)?;
        writeln!(out, "    @!%active bra SEG_RED_TREE_SYNC;").map_err(ferr)?;
        writeln!(out, "    add.u32      %partner, %ltid, %half;").map_err(ferr)?;
        writeln!(out, "    mul.wide.u32 %addr, %partner, {eb};").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %addr, %smem_base;").map_err(ferr)?;
        writeln!(out, "    ld.shared.{ty} %other, [%addr];").map_err(ferr)?;
        writeln!(out, "    ld.shared.{ty} %acc, [%smem_addr];").map_err(ferr)?;
        writeln!(out, "    {instr}      %acc, %acc, %other;").map_err(ferr)?;
        writeln!(out, "    st.shared.{ty} [%smem_addr], %acc;").map_err(ferr)?;
        writeln!(out, "SEG_RED_TREE_SYNC:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        writeln!(out, "    shr.u32      %half, %half, 1;").map_err(ferr)?;
        writeln!(out, "    bra SEG_RED_TREE;").map_err(ferr)?;

        // Thread 0 writes the segment result.
        writeln!(out, "SEG_RED_WRITE:").map_err(ferr)?;
        writeln!(out, "    setp.ne.u32  %p, %ltid, 0;").map_err(ferr)?;
        writeln!(out, "    @%p ret;").map_err(ferr)?;
        writeln!(out, "    cvt.u64.u32  %i, %seg;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %i, {eb}, %ptr_out;").map_err(ferr)?;
        writeln!(out, "    ld.shared.{ty} %acc, [%smem_base];").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %acc;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;

        Ok(out)
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Segmented scan
// ════════════════════════════════════════════════════════════════════════════

/// Inclusive vs exclusive scan selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegScanKind {
    /// `out[i]` includes `in[i]`.
    Inclusive,
    /// `out[i]` excludes `in[i]`; first element of each segment is the identity.
    Exclusive,
}

impl SegScanKind {
    fn name(self) -> &'static str {
        match self {
            Self::Inclusive => "inc",
            Self::Exclusive => "exc",
        }
    }
}

/// Configuration for device-wide segmented scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentedScanConfig {
    /// Scan operation.
    pub op: ReduceOp,
    /// Element type.
    pub ty: PtxType,
    /// Inclusive or exclusive.
    pub kind: SegScanKind,
    /// Threads per block (power of 2, `32`–`1024`).  One thread per segment.
    pub block_size: u32,
}

impl SegmentedScanConfig {
    /// Create a configuration, validating `block_size`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] for an invalid `block_size`.
    pub fn new(
        op: ReduceOp,
        ty: PtxType,
        kind: SegScanKind,
        block_size: u32,
    ) -> PrimitivesResult<Self> {
        validate_block_size(block_size)?;
        Ok(Self {
            op,
            ty,
            kind,
            block_size,
        })
    }

    /// Generated kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "seg_scan_{}_{}_{}_bs{}",
            self.kind.name(),
            self.op.name(),
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX code generator for device-wide segmented scan (one thread / segment).
pub struct SegmentedScanTemplate {
    /// Configuration.
    pub cfg: SegmentedScanConfig,
}

impl SegmentedScanTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: SegmentedScanConfig) -> Self {
        Self { cfg }
    }

    /// Generate the segmented-scan PTX kernel.
    ///
    /// Launch with a 1-D grid covering `num_segments` threads.  Params:
    /// `(out, input, offsets, num_segments)`.  Output array is the same length
    /// as the input; each segment is scanned independently in place.
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
        let exclusive = self.cfg.kind == SegScanKind::Exclusive;
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("seg_scan", e);

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_out,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_offsets,\n    \
             .param .u64 param_num_segments\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{ty}   %acc, %val, %store;").map_err(ferr)?;
        writeln!(out, "    .reg .u32    %ltid, %bid;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %nseg, %seg, %seg_beg, %seg_end, %i, %addr, %off_addr;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .u64    %ptr_out, %ptr_in, %ptr_off;").map_err(ferr)?;
        writeln!(out, "    .reg .pred   %p, %loop_p;").map_err(ferr)?;

        writeln!(out, "    ld.param.u64 %ptr_out, [param_out];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_off, [param_offsets];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %nseg,    [param_num_segments];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(
            out,
            "    cvt.u64.u32   %seg, %ltid;
    mad.wide.u32   %seg, %bid, {bs}, %seg;"
        )
        .map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %p, %seg, %nseg;").map_err(ferr)?;
        writeln!(out, "    @%p ret;").map_err(ferr)?;

        // Load segment bounds.
        writeln!(out, "    mad.lo.u64   %off_addr, %seg, 8, %ptr_off;").map_err(ferr)?;
        writeln!(out, "    ld.global.u64 %seg_beg, [%off_addr];").map_err(ferr)?;
        writeln!(out, "    ld.global.u64 %seg_end, [%off_addr+8];").map_err(ferr)?;

        // Sequential prefix scan over [seg_beg, seg_end).
        writeln!(out, "    mov.{ty}      %acc, {ident};").map_err(ferr)?;
        writeln!(out, "    mov.u64      %i, %seg_beg;").map_err(ferr)?;
        writeln!(out, "SEG_SCAN_LOOP:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %loop_p, %i, %seg_end;").map_err(ferr)?;
        writeln!(out, "    @%loop_p bra SEG_SCAN_DONE;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %i, {eb}, %ptr_in;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %val, [%addr];").map_err(ferr)?;
        if exclusive {
            // Write running prefix BEFORE folding the current element.
            writeln!(out, "    mov.{ty}      %store, %acc;").map_err(ferr)?;
            writeln!(out, "    {instr}      %acc, %acc, %val;").map_err(ferr)?;
        } else {
            writeln!(out, "    {instr}      %acc, %acc, %val;").map_err(ferr)?;
            writeln!(out, "    mov.{ty}      %store, %acc;").map_err(ferr)?;
        }
        writeln!(out, "    mad.lo.u64   %addr, %i, {eb}, %ptr_out;").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %store;").map_err(ferr)?;
        writeln!(out, "    add.u64      %i, %i, 1;").map_err(ferr)?;
        writeln!(out, "    bra SEG_SCAN_LOOP;").map_err(ferr)?;
        writeln!(out, "SEG_SCAN_DONE:").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;

        Ok(out)
    }
}

// ─── CPU references ────────────────────────────────────────────────────────────

/// Apply a [`ReduceOp`] to two integer values (used by the integer CPU
/// references for cross-checking generated kernels).
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

/// Host reference for integer segmented reduce.
///
/// `offsets` has length `num_segments + 1`.  Returns one value per segment.
#[must_use]
pub fn reference_segmented_reduce_u64(op: ReduceOp, data: &[u64], offsets: &[u64]) -> Vec<u64> {
    if offsets.len() < 2 {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(offsets.len() - 1);
    for w in offsets.windows(2) {
        let beg = w[0] as usize;
        let end = w[1] as usize;
        let mut acc = identity_u64(op);
        for &v in &data[beg..end] {
            acc = apply_u64(op, acc, v);
        }
        result.push(acc);
    }
    result
}

/// Host reference for integer segmented scan (inclusive or exclusive).
///
/// Returns an array the same length as `data` with each segment scanned in
/// place.  Indices outside any segment are left as the input value.
#[must_use]
pub fn reference_segmented_scan_u64(
    op: ReduceOp,
    kind: SegScanKind,
    data: &[u64],
    offsets: &[u64],
) -> Vec<u64> {
    let mut out = data.to_vec();
    if offsets.len() < 2 {
        return out;
    }
    for w in offsets.windows(2) {
        let beg = w[0] as usize;
        let end = w[1] as usize;
        let mut acc = identity_u64(op);
        for i in beg..end {
            match kind {
                SegScanKind::Exclusive => {
                    let store = acc;
                    acc = apply_u64(op, acc, data[i]);
                    out[i] = store;
                }
                SegScanKind::Inclusive => {
                    acc = apply_u64(op, acc, data[i]);
                    out[i] = acc;
                }
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
    fn config_validation() {
        assert!(SegmentedReduceConfig::new(ReduceOp::Sum, PtxType::F32, 100).is_err());
        assert!(SegmentedReduceConfig::new(ReduceOp::Sum, PtxType::F32, 256).is_ok());
        assert!(
            SegmentedScanConfig::new(ReduceOp::Sum, PtxType::U32, SegScanKind::Inclusive, 7)
                .is_err()
        );
    }

    #[test]
    fn reduce_kernel_name() {
        let c = SegmentedReduceConfig::new(ReduceOp::Max, PtxType::U32, 128).expect("valid config");
        assert_eq!(c.kernel_name(), "seg_reduce_max_u32_bs128");
    }

    #[test]
    fn scan_kernel_name_inc_exc() {
        let inc =
            SegmentedScanConfig::new(ReduceOp::Sum, PtxType::F32, SegScanKind::Inclusive, 256)
                .expect("valid config");
        let exc =
            SegmentedScanConfig::new(ReduceOp::Sum, PtxType::F32, SegScanKind::Exclusive, 256)
                .expect("valid config");
        assert_eq!(inc.kernel_name(), "seg_scan_inc_sum_f32_bs256");
        assert_eq!(exc.kernel_name(), "seg_scan_exc_sum_f32_bs256");
    }

    #[test]
    fn reduce_ptx_loads_segment_bounds_and_reduces() {
        let c = SegmentedReduceConfig::new(ReduceOp::Sum, PtxType::F32, 256).expect("valid config");
        let ptx = SegmentedReduceTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("seg_reduce_sum_f32_bs256"), "PTX: {ptx}");
        assert!(ptx.contains("ld.global.u64 %seg_beg"), "PTX: {ptx}");
        assert!(ptx.contains("[%off_addr+8]"), "PTX: {ptx}");
        assert!(ptx.contains("add.f32      %acc"), "PTX: {ptx}");
        assert!(ptx.contains("bar.sync 0"), "PTX: {ptx}");
        assert!(ptx.contains("SEG_RED_TREE"), "PTX: {ptx}");
    }

    #[test]
    fn reduce_ptx_min_uses_min_instr_and_inf_identity() {
        let c = SegmentedReduceConfig::new(ReduceOp::Min, PtxType::F32, 64).expect("valid config");
        let ptx = SegmentedReduceTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("min.f32"), "PTX: {ptx}");
        assert!(ptx.contains("0f7F800000"), "PTX: {ptx}");
    }

    #[test]
    fn scan_ptx_inclusive_vs_exclusive_ordering() {
        let inc =
            SegmentedScanConfig::new(ReduceOp::Sum, PtxType::U32, SegScanKind::Inclusive, 256)
                .expect("valid config");
        let exc =
            SegmentedScanConfig::new(ReduceOp::Sum, PtxType::U32, SegScanKind::Exclusive, 256)
                .expect("valid config");
        let p_inc = SegmentedScanTemplate::new(inc)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        let p_exc = SegmentedScanTemplate::new(exc)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(p_inc.contains("SEG_SCAN_LOOP"), "PTX: {p_inc}");
        assert!(p_exc.contains("SEG_SCAN_LOOP"), "PTX: {p_exc}");
        // Both store; differ only in where %store is captured.
        assert!(p_inc.contains("st.global.u32"), "PTX: {p_inc}");
        assert!(p_exc.contains("st.global.u32"), "PTX: {p_exc}");
    }

    #[test]
    fn reference_reduce_sum_and_max() {
        let data = [1u64, 2, 3, 10, 20, 7];
        let offsets = [0u64, 3, 5, 6];
        let sums = reference_segmented_reduce_u64(ReduceOp::Sum, &data, &offsets);
        assert_eq!(sums, vec![6, 30, 7]);
        let maxs = reference_segmented_reduce_u64(ReduceOp::Max, &data, &offsets);
        assert_eq!(maxs, vec![3, 20, 7]);
    }

    #[test]
    fn reference_reduce_empty_segment() {
        // Segment 1 is empty (offsets 3..3).
        let data = [5u64, 6, 7];
        let offsets = [0u64, 3, 3];
        let sums = reference_segmented_reduce_u64(ReduceOp::Sum, &data, &offsets);
        assert_eq!(sums, vec![18, 0]); // empty segment → identity 0
        let prods = reference_segmented_reduce_u64(ReduceOp::Product, &data, &offsets);
        assert_eq!(prods, vec![210, 1]); // empty segment → identity 1
    }

    #[test]
    fn reference_scan_inclusive() {
        let data = [1u64, 2, 3, 10, 20];
        let offsets = [0u64, 3, 5];
        let out =
            reference_segmented_scan_u64(ReduceOp::Sum, SegScanKind::Inclusive, &data, &offsets);
        // segment 0: 1,3,6 ; segment 1: 10,30
        assert_eq!(out, vec![1, 3, 6, 10, 30]);
    }

    #[test]
    fn reference_scan_exclusive() {
        let data = [1u64, 2, 3, 10, 20];
        let offsets = [0u64, 3, 5];
        let out =
            reference_segmented_scan_u64(ReduceOp::Sum, SegScanKind::Exclusive, &data, &offsets);
        // segment 0: 0,1,3 ; segment 1: 0,10
        assert_eq!(out, vec![0, 1, 3, 0, 10]);
    }

    #[test]
    fn reference_scan_max_inclusive() {
        let data = [3u64, 1, 4, 1, 5, 9, 2];
        let offsets = [0u64, 4, 7];
        let out =
            reference_segmented_scan_u64(ReduceOp::Max, SegScanKind::Inclusive, &data, &offsets);
        // seg0: 3,3,4,4 ; seg1: 5,9,9
        assert_eq!(out, vec![3, 3, 4, 4, 5, 9, 9]);
    }
}

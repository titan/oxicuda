//! Per-segment sort — one block bitonic-sorts each segment in shared memory.
//!
//! Given a flat data array and a `segment offsets` array of length
//! `num_segments + 1` (segment `s` spans `[offsets[s], offsets[s+1])`), this
//! primitive sorts each segment independently in place.  It mirrors
//! `cub::DeviceSegmentedSort` / `DeviceSegmentedRadixSort` for the common case
//! where every segment fits in one thread block (`len ≤ block_size`).
//!
//! # Algorithm
//!
//! One block is launched per segment.  Each block:
//!
//! 1. loads its segment into shared memory, padding the tail up to the next
//!    power of two with the type's maximum value so out-of-segment slots sort to
//!    the end,
//! 2. runs a bitonic sorting network over the padded shared array (the same
//!    two-barrier-per-stage pattern as [`crate::sort::merge_sort`]'s block sort),
//! 3. writes the first `len` sorted elements back to global memory.
//!
//! Segments longer than `block_size` are not handled by this single-block kernel
//! (they need the multi-pass merge path); the CPU reference documents the exact
//! per-segment ordering this kernel produces.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::sort::segmented_sort::{
//!     SegmentedSortConfig, SegmentedSortTemplate,
//! };
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = SegmentedSortConfig::new(PtxType::U32, 256).expect("valid config");
//! let ptx = SegmentedSortTemplate::new(cfg).generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(ptx.contains("segmented_sort_u32_bs256"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::error::{PrimitivesError, PrimitivesResult};
use crate::ptx_helpers::{ptx_header, ptx_type_str};

/// PTX literal for the maximum value of a type (tail padding sorts to the end).
fn max_fill_literal(ty: PtxType) -> &'static str {
    match ty {
        PtxType::U32 => "0xFFFFFFFF",
        PtxType::S32 => "0x7FFFFFFF",
        PtxType::U64 => "0xFFFFFFFFFFFFFFFF",
        PtxType::S64 => "0x7FFFFFFFFFFFFFFF",
        PtxType::F32 => "0f7F800000",
        PtxType::F64 => "0d7FF0000000000000",
        _ => "0xFFFFFFFF",
    }
}

/// Configuration for per-segment block sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentedSortConfig {
    /// Element type.
    pub ty: PtxType,
    /// Threads per block (power of 2, `32`–`1024`); also the maximum segment
    /// length this single-block kernel handles.
    pub block_size: u32,
}

impl SegmentedSortConfig {
    /// Create a configuration, validating `block_size`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] for an invalid `block_size`.
    pub fn new(ty: PtxType, block_size: u32) -> PrimitivesResult<Self> {
        if !(32..=1024).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(PrimitivesError::InvalidArgument(format!(
                "block_size must be a power of two in [32, 1024], got {block_size}"
            )));
        }
        Ok(Self { ty, block_size })
    }

    /// log₂(block_size) — the number of bitonic sort stages.
    #[must_use]
    pub fn log2_block_size(&self) -> u32 {
        self.block_size.trailing_zeros()
    }

    /// Bytes per element.
    #[must_use]
    pub fn elem_bytes(&self) -> u32 {
        match self.ty {
            PtxType::F64 | PtxType::U64 | PtxType::S64 | PtxType::B64 => 8,
            _ => 4,
        }
    }

    /// Generated kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "segmented_sort_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX generator for per-segment block bitonic sort.
pub struct SegmentedSortTemplate {
    /// Configuration.
    pub cfg: SegmentedSortConfig,
}

impl SegmentedSortTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: SegmentedSortConfig) -> Self {
        Self { cfg }
    }

    /// Generate the segmented-sort PTX kernel.
    ///
    /// Launch with `grid = num_segments`, `block = block_size`.  Params:
    /// `(data, offsets, num_segments)`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::PtxGeneration`] on formatting failure.
    pub fn generate(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.elem_bytes();
        let log2 = self.cfg.log2_block_size();
        let fill = max_fill_literal(self.cfg.ty);
        let cmp = format!("setp.lt.{ty}");
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("segmented_sort", e);

        let mut out = ptx_header(sm);
        writeln!(out, ".shared .align {eb} .{ty} segsort_smem[{bs}];").map_err(ferr)?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_data,\n    \
             .param .u64 param_offsets,\n    \
             .param .u64 param_num_segments\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .{ty}   %a, %b, %write_val, %min_val, %max_val;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u32    %ltid, %seg, %partner, %dir, %low_bit, %low_int, %want_int, %seg_len;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %nseg, %seg_beg, %seg_end, %gidx, %smem_base, %tid_addr;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %ptr, %ptr_off, %off_addr, %glob_addr, %len64;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .pred   %p, %in_seg, %want_min, %gt, %is_low;"
        )
        .map_err(ferr)?;

        writeln!(out, "    ld.param.u64 %ptr,     [param_data];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_off, [param_offsets];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %nseg,    [param_num_segments];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %seg, %ctaid.x;").map_err(ferr)?;
        writeln!(out, "    cvt.u64.u32  %gidx, %seg;").map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %p, %gidx, %nseg;").map_err(ferr)?;
        writeln!(out, "    @%p ret;").map_err(ferr)?;

        // Segment bounds.
        writeln!(out, "    mad.lo.u64   %off_addr, %gidx, 8, %ptr_off;").map_err(ferr)?;
        writeln!(out, "    ld.global.u64 %seg_beg, [%off_addr];").map_err(ferr)?;
        writeln!(out, "    ld.global.u64 %seg_end, [%off_addr+8];").map_err(ferr)?;
        writeln!(out, "    sub.u64      %len64, %seg_end, %seg_beg;").map_err(ferr)?;
        writeln!(out, "    cvt.u32.u64  %seg_len, %len64;").map_err(ferr)?;

        writeln!(out, "    mov.u64      %smem_base, segsort_smem;").map_err(ferr)?;
        writeln!(
            out,
            "    mad.wide.u32   %tid_addr, %ltid, {eb}, %smem_base;"
        )
        .map_err(ferr)?;

        // Load this thread's element if within the segment, else pad with max.
        writeln!(out, "    setp.lt.u32  %in_seg, %ltid, %seg_len;").map_err(ferr)?;
        writeln!(out, "    cvt.u64.u32  %glob_addr, %ltid;").map_err(ferr)?;
        writeln!(out, "    add.u64      %glob_addr, %glob_addr, %seg_beg;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %glob_addr, %glob_addr, {eb}, %ptr;").map_err(ferr)?;
        writeln!(out, "    @%in_seg ld.global.{ty} %a, [%glob_addr];").map_err(ferr)?;
        writeln!(out, "    @!%in_seg mov.{ty} %a, {fill};").map_err(ferr)?;
        writeln!(out, "    st.shared.{ty} [%tid_addr], %a;").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;

        // Bitonic sort network over the padded power-of-two shared array.
        for stage in 1..=log2 {
            let log2_k = stage;
            for sub in (0..stage).rev() {
                let j: u32 = 1 << sub;
                writeln!(out, "    bar.sync 0;").map_err(ferr)?;
                writeln!(
                    out,
                    "    mad.wide.u32   %tid_addr, %ltid, {eb}, %smem_base;"
                )
                .map_err(ferr)?;
                writeln!(out, "    xor.b32      %partner, %ltid, {j};").map_err(ferr)?;
                writeln!(out, "    .reg .u64 %par_addr_{stage}_{sub};").map_err(ferr)?;
                writeln!(
                    out,
                    "    mad.wide.u32   %par_addr_{stage}_{sub}, %partner, {eb}, %smem_base;"
                )
                .map_err(ferr)?;
                writeln!(out, "    ld.shared.{ty} %a, [%tid_addr];").map_err(ferr)?;
                writeln!(out, "    ld.shared.{ty} %b, [%par_addr_{stage}_{sub}];").map_err(ferr)?;
                writeln!(out, "    bar.sync 0;").map_err(ferr)?;
                writeln!(out, "    shr.u32      %dir, %ltid, {log2_k};").map_err(ferr)?;
                writeln!(out, "    and.b32      %dir, %dir, 1;").map_err(ferr)?;
                // Compare-exchange where each thread writes only its own slot:
                // the low index of the pair (bit `j` of tid == 0) keeps the min
                // for an ascending block and the max for a descending one. The
                // previous code made both partners write the partner's value,
                // collapsing the segment to a single repeated element.
                writeln!(out, "    {cmp}        %gt, %b, %a;").map_err(ferr)?;
                writeln!(out, "    selp.{ty}    %min_val, %b, %a, %gt;").map_err(ferr)?;
                writeln!(out, "    selp.{ty}    %max_val, %a, %b, %gt;").map_err(ferr)?;
                writeln!(out, "    and.b32      %low_bit, %ltid, {j};").map_err(ferr)?;
                writeln!(out, "    setp.eq.u32  %is_low, %low_bit, 0;").map_err(ferr)?;
                writeln!(out, "    selp.u32     %low_int, 1, 0, %is_low;").map_err(ferr)?;
                writeln!(out, "    xor.b32      %want_int, %low_int, %dir;").map_err(ferr)?;
                writeln!(out, "    setp.ne.u32  %want_min, %want_int, 0;").map_err(ferr)?;
                writeln!(
                    out,
                    "    selp.{ty}    %write_val, %min_val, %max_val, %want_min;"
                )
                .map_err(ferr)?;
                writeln!(out, "    st.shared.{ty} [%tid_addr], %write_val;").map_err(ferr)?;
            }
        }

        // Write back the first seg_len sorted elements.
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        writeln!(out, "    setp.lt.u32  %in_seg, %ltid, %seg_len;").map_err(ferr)?;
        writeln!(out, "    @!%in_seg ret;").map_err(ferr)?;
        writeln!(out, "    ld.shared.{ty} %a, [%tid_addr];").map_err(ferr)?;
        writeln!(out, "    cvt.u64.u32  %glob_addr, %ltid;").map_err(ferr)?;
        writeln!(out, "    add.u64      %glob_addr, %glob_addr, %seg_beg;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %glob_addr, %glob_addr, {eb}, %ptr;").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%glob_addr], %a;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }
}

// ─── CPU reference ─────────────────────────────────────────────────────────────

/// Host reference for per-segment ascending sort over integer keys.  `offsets`
/// has length `num_segments + 1`; each segment is sorted in place, the rest of
/// the array untouched.
#[must_use]
pub fn reference_segmented_sort_u64(data: &[u64], offsets: &[u64]) -> Vec<u64> {
    let mut out = data.to_vec();
    if offsets.len() < 2 {
        return out;
    }
    for w in offsets.windows(2) {
        let beg = w[0] as usize;
        let end = w[1] as usize;
        out[beg..end].sort_unstable();
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    #[test]
    fn config_validation_and_name() {
        assert!(SegmentedSortConfig::new(PtxType::U32, 100).is_err());
        let c = SegmentedSortConfig::new(PtxType::U32, 256).expect("valid config");
        assert_eq!(c.kernel_name(), "segmented_sort_u32_bs256");
        assert_eq!(c.log2_block_size(), 8);
        assert_eq!(c.elem_bytes(), 4);
    }

    #[test]
    fn ptx_loads_bounds_pads_and_bitonic_sorts() {
        let c = SegmentedSortConfig::new(PtxType::U32, 256).expect("valid config");
        let ptx = SegmentedSortTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("segmented_sort_u32_bs256"), "PTX: {ptx}");
        assert!(ptx.contains("ld.global.u64 %seg_beg"), "PTX: {ptx}");
        assert!(ptx.contains("[%off_addr+8]"), "PTX: {ptx}");
        // Tail padding with max value.
        assert!(ptx.contains("mov.u32 %a, 0xFFFFFFFF"), "PTX: {ptx}");
        // Bitonic compare-swap and barriers.
        assert!(ptx.contains("xor.b32      %partner"), "PTX: {ptx}");
        assert!(ptx.contains("selp.u32"), "PTX: {ptx}");
        assert!(ptx.contains("bar.sync 0"), "PTX: {ptx}");
    }

    #[test]
    fn ptx_f64_uses_8byte_and_inf_pad() {
        let c = SegmentedSortConfig::new(PtxType::F64, 128).expect("valid config");
        let ptx = SegmentedSortTemplate::new(c)
            .generate(SmVersion::Sm90)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("ld.shared.f64"), "PTX: {ptx}");
        // f64 +inf must be a valid PTX double literal (`0d…`), not a hex int.
        assert!(ptx.contains("0d7FF0000000000000"), "PTX: {ptx}");
    }

    #[test]
    fn reference_sorts_each_segment() {
        let data = [5u64, 1, 3, 9, 2, 8, 4];
        let offsets = [0u64, 3, 5, 7];
        let out = reference_segmented_sort_u64(&data, &offsets);
        // seg0 [5,1,3]→[1,3,5]; seg1 [9,2]→[2,9]; seg2 [8,4]→[4,8]
        assert_eq!(out, vec![1, 3, 5, 2, 9, 4, 8]);
    }

    #[test]
    fn reference_single_and_empty_segments() {
        let out = reference_segmented_sort_u64(&[7u64], &[0, 1]);
        assert_eq!(out, vec![7]);
        // Empty segment (3..3) leaves nothing to sort.
        let out2 = reference_segmented_sort_u64(&[3u64, 1, 2], &[0, 3, 3]);
        assert_eq!(out2, vec![1, 2, 3]);
    }
}

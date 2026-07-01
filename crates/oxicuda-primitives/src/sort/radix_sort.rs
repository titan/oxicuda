//! 4-bit LSD (Least Significant Digit) radix sort for integer keys.
//!
//! Each pass of the sort processes 4 bits, for a total of
//! * **8 passes** for 32-bit keys
//! * **16 passes** for 64-bit keys
//!
//! The pass pipeline is three kernels that the caller launches in order for
//! each pass:
//!
//! | # | Kernel             | Purpose                                               |
//! |---|--------------------|-------------------------------------------------------|
//! | 1 | **count**          | Per-block histogram of the 4-bit digit at bit `shift` |
//! | 2 | **scan**           | Exclusive scan of the per-block histograms            |
//! | 3 | **scatter**        | Scatter keys using scanned offsets + atomic ranking   |
//!
//! # Data layout
//!
//! The count/scan array has shape `[num_blocks][RADIX_SIZE]` in row-major
//! order (element at `[b][d]` lives at flat index `b * 16 + d`).
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::sort::radix_sort::{RadixSortConfig, RadixSortTemplate};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = RadixSortConfig { ty: PtxType::U32, block_size: 256 };
//! let t   = RadixSortTemplate::new(cfg);
//! let (count_ptx, scan_ptx, scatter_ptx) = t.generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(count_ptx.contains("radix_count_u32_bs256"));
//! assert!(scan_ptx.contains("radix_scan_u32"));
//! assert!(scatter_ptx.contains("radix_scatter_u32_bs256"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::ptx_helpers::{ptx_header, ptx_type_str};

/// Number of digit buckets per radix pass (2^4 = 16).
pub const RADIX_SIZE: u32 = 16;

/// Configuration for 4-bit LSD radix sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RadixSortConfig {
    /// Key type.  Only `PtxType::U32` and `PtxType::U64` are supported.
    pub ty: PtxType,
    /// Threads per block (power of 2, 32–1024).
    pub block_size: u32,
}

impl RadixSortConfig {
    /// Number of radix passes required to sort a key of this type.
    #[must_use]
    pub fn passes(&self) -> u32 {
        match self.ty {
            PtxType::U64 => 16,
            _ => 8,
        }
    }

    /// Bytes per key element.
    #[must_use]
    pub fn elem_bytes(&self) -> u32 {
        match self.ty {
            PtxType::U64 => 8,
            _ => 4,
        }
    }

    /// Number of thread blocks for `n` elements.
    #[must_use]
    pub fn num_blocks(&self, n: u64) -> u64 {
        n.div_ceil(self.block_size as u64)
    }

    /// Bytes for the count/offset scratch array (num_blocks × RADIX_SIZE × 4).
    #[must_use]
    pub fn scratch_bytes(&self, n: u64) -> u64 {
        self.num_blocks(n) * u64::from(RADIX_SIZE) * 4
    }

    /// Kernel name for the histogram pass.
    #[must_use]
    pub fn count_kernel_name(&self) -> String {
        format!(
            "radix_count_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }

    /// Kernel name for the scan pass.
    #[must_use]
    pub fn scan_kernel_name(&self) -> String {
        format!("radix_scan_{}", ptx_type_str(self.ty))
    }

    /// Kernel name for the scatter pass.
    #[must_use]
    pub fn scatter_kernel_name(&self) -> String {
        format!(
            "radix_scatter_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX code generator for 4-bit LSD radix sort.
///
/// Call [`RadixSortTemplate::generate`] once to get all three kernel PTX
/// strings; then launch them per-pass from the driver.
pub struct RadixSortTemplate {
    /// Configuration.
    pub cfg: RadixSortConfig,
}

impl RadixSortTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: RadixSortConfig) -> Self {
        Self { cfg }
    }

    /// Generate all three PTX kernels as `(count_ptx, scan_ptx, scatter_ptx)`.
    ///
    /// # Errors
    ///
    /// Returns a `String` error on generation failure.
    pub fn generate(&self, sm: SmVersion) -> Result<(String, String, String), String> {
        let count = self.generate_count_kernel(sm)?;
        let scan = self.generate_scan_kernel(sm)?;
        let scatter = self.generate_scatter_kernel(sm)?;
        Ok((count, scan, scatter))
    }

    // ── Kernel 1: per-block histogram of the current 4-bit digit ────────────

    fn generate_count_kernel(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.count_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.elem_bytes();
        let is64 = self.cfg.ty == PtxType::U64;

        let mut out = ptx_header(sm);
        // Private histogram: 16 u32 bins per block.
        writeln!(out, ".shared .align 4 .u32 cnt_hist[{RADIX_SIZE}];")
            .map_err(|e| e.to_string())?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_counts,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_n,\n    \
             .param .u32 param_shift\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %key;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %ltid, %bid, %shift, %digit, %old;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %n, %gid, %ptr_in, %ptr_cnt, %addr;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %smem_base, %hist_addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;
        if is64 {
            writeln!(out, "    .reg .u64    %shift64, %key_shifted;").map_err(|e| e.to_string())?;
        }

        writeln!(out, "    ld.param.u64 %ptr_cnt, [param_counts];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %shift,    [param_shift];").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %smem_base, cnt_hist;").map_err(|e| e.to_string())?;

        // Phase 1: Init private histogram to zero (first 16 threads).
        writeln!(out, "    setp.ge.u32  %p, %ltid, {RADIX_SIZE};").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra CNT_INIT_DONE;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %hist_addr, %ltid, 4, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.u32 [%hist_addr], 0;").map_err(|e| e.to_string())?;
        writeln!(out, "CNT_INIT_DONE:").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        // Phase 2: Accumulate digit histogram.
        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra CNT_FLUSH;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.{ty} %key, [%addr];").map_err(|e| e.to_string())?;

        // Extract 4-bit digit at bit position `shift`.
        if is64 {
            writeln!(out, "    cvt.u64.u32  %shift64, %shift;").map_err(|e| e.to_string())?;
            writeln!(out, "    shr.u64      %key_shifted, %key, %shift64;")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    cvt.u32.u64  %digit, %key_shifted;").map_err(|e| e.to_string())?;
        } else {
            writeln!(out, "    shr.u32      %digit, %key, %shift;").map_err(|e| e.to_string())?;
        }
        writeln!(out, "    and.b32      %digit, %digit, 0xF;").map_err(|e| e.to_string())?;

        writeln!(out, "    mad.wide.u32   %hist_addr, %digit, 4, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    atom.shared.add.u32 %old, [%hist_addr], 1;")
            .map_err(|e| e.to_string())?;

        // Phase 3: Flush private histogram to global counts[bid][0..16].
        writeln!(out, "CNT_FLUSH:").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u32  %p, %ltid, {RADIX_SIZE};").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p ret;").map_err(|e| e.to_string())?;
        // counts[bid * 16 + tid]
        writeln!(out, "    .reg .u32 %flat_idx;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u32   %flat_idx, %bid, {RADIX_SIZE}, %ltid;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %addr, %flat_idx, 4, %ptr_cnt;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %hist_addr, %ltid, 4, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.shared.u32 %old, [%hist_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.u32 [%addr], %old;").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── Kernel 2: exclusive scan of per-block digit histograms ────────────────

    fn generate_scan_kernel(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.scan_kernel_name();

        let mut out = ptx_header(sm);
        // Launch with 1 block, RADIX_SIZE (16) threads: thread d scans digit d.
        //
        // The output offset for digit `d` in block `b` is the GLOBAL position
        // where that (block, digit) bucket starts:
        //   offset[b][d] = digit_base[d] + (count of digit d in blocks < b)
        // where `digit_base[d] = Σ_{d' < d} total[d']` is the start of digit `d`
        // in the sorted output. The previous version omitted `digit_base`, so the
        // buckets all started near 0 and overwrote each other — radix sort was
        // completely broken. We now compute it in three passes: per-digit totals
        // into shared memory, an exclusive scan across digits for `digit_base`,
        // then the cross-block exclusive scan seeded with `digit_base`.
        writeln!(out, ".shared .align 4 .u32 radix_totals[{RADIX_SIZE}];")
            .map_err(|e| e.to_string())?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_counts,\n    \
             .param .u32 param_num_blocks\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u32    %ltid, %nb, %b, %d, %cnt, %prefix, %base, %total, %flat_idx;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %ptr, %addr, %smem_base, %smem_addr;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr, [param_counts];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %nb,  [param_num_blocks];").map_err(|e| e.to_string())?;
        // tid = digit index (0..15)
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %smem_base, radix_totals;").map_err(|e| e.to_string())?;

        // Pass 1: total occurrences of this digit across all blocks.
        writeln!(out, "    mov.u32      %total, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %b, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "RADIX_SCAN_TOTAL:").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u32  %p, %b, %nb;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra RADIX_SCAN_TOTAL_DONE;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u32   %flat_idx, %b, {RADIX_SIZE}, %ltid;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %addr, %flat_idx, 4, %ptr;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.u32 %cnt, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u32      %total, %total, %cnt;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u32      %b, %b, 1;").map_err(|e| e.to_string())?;
        writeln!(out, "    bra RADIX_SCAN_TOTAL;").map_err(|e| e.to_string())?;
        writeln!(out, "RADIX_SCAN_TOTAL_DONE:").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %smem_addr, %ltid, 4, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.u32 [%smem_addr], %total;").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        // Pass 2: digit_base = exclusive prefix of totals over digits < tid.
        writeln!(out, "    mov.u32      %base, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %d, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "RADIX_SCAN_BASE:").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u32  %p, %d, %ltid;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra RADIX_SCAN_BASE_DONE;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %smem_addr, %d, 4, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.shared.u32 %cnt, [%smem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u32      %base, %base, %cnt;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u32      %d, %d, 1;").map_err(|e| e.to_string())?;
        writeln!(out, "    bra RADIX_SCAN_BASE;").map_err(|e| e.to_string())?;
        writeln!(out, "RADIX_SCAN_BASE_DONE:").map_err(|e| e.to_string())?;

        // Pass 3: cross-block exclusive scan seeded with digit_base.
        writeln!(out, "    mov.u32      %prefix, %base;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %b, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "SCAN_LOOP:").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u32  %p, %b, %nb;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra SCAN_DONE;").map_err(|e| e.to_string())?;
        // flat_idx = b * 16 + tid
        writeln!(out, "    mad.lo.u32   %flat_idx, %b, {RADIX_SIZE}, %ltid;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %addr, %flat_idx, 4, %ptr;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.u32 %cnt, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.u32 [%addr], %prefix;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u32      %prefix, %prefix, %cnt;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u32      %b, %b, 1;").map_err(|e| e.to_string())?;
        writeln!(out, "    bra SCAN_LOOP;").map_err(|e| e.to_string())?;
        writeln!(out, "SCAN_DONE:").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── Kernel 3: scatter keys to output using pre-scanned offsets ───────────
    //
    // Each block loads its pre-scanned offsets for all 16 digits into shared
    // memory, then each thread atomically increments its digit's counter to
    // claim a unique output slot.

    fn generate_scatter_kernel(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.scatter_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.elem_bytes();
        let is64 = self.cfg.ty == PtxType::U64;

        let mut out = ptx_header(sm);
        // `block_offs[d]` = global output offset where this block's digit-`d`
        // bucket starts (from the scan). `sct_digits[t]` caches each thread's
        // digit so the scatter can assign a STABLE within-block rank: LSD radix
        // is only correct if equal digits keep their input order across passes.
        // The previous version used `atom.shared.add`, whose completion order is
        // arbitrary, which destroyed stability and the sort.
        writeln!(out, ".shared .align 4 .u32 block_offs[{RADIX_SIZE}];")
            .map_err(|e| e.to_string())?;
        writeln!(out, ".shared .align 4 .u32 sct_digits[{bs}];").map_err(|e| e.to_string())?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_output,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_offsets,\n    \
             .param .u64 param_n,\n    \
             .param .u32 param_shift\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %key;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u32    %ltid, %bid, %shift, %digit, %out_pos, %rank, %tp, %other, %boff, %flat_init, %off_val;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr_in, %ptr_out, %ptr_off;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %addr, %smem_base, %dsmem_base, %smem_addr, %out64;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p, %oob, %eq;").map_err(|e| e.to_string())?;
        if is64 {
            writeln!(out, "    .reg .u64    %shift64, %key_shifted;").map_err(|e| e.to_string())?;
        }

        writeln!(out, "    ld.param.u64 %ptr_out, [param_output];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_off, [param_offsets];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %shift,    [param_shift];").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %smem_base, block_offs;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %dsmem_base, sct_digits;").map_err(|e| e.to_string())?;

        // Phase 1a: Load this block's pre-scanned offsets (first 16 threads).
        writeln!(out, "    setp.ge.u32  %p, %ltid, {RADIX_SIZE};").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra SCT_LOAD_DONE;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u32   %flat_init, %bid, {RADIX_SIZE}, %ltid;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %addr, %flat_init, 4, %ptr_off;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.u32 %off_val, [%addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %smem_addr, %ltid, 4, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.u32 [%smem_addr], %off_val;").map_err(|e| e.to_string())?;
        writeln!(out, "SCT_LOAD_DONE:").map_err(|e| e.to_string())?;

        // Phase 1b: Every thread caches its digit (sentinel 0xFFFFFFFF if OOB so
        // it never matches a real digit during ranking).
        writeln!(out, "    setp.ge.u64  %oob, %gid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%oob bra SCT_DIG_OOB;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.{ty} %key, [%addr];").map_err(|e| e.to_string())?;
        if is64 {
            writeln!(out, "    cvt.u64.u32  %shift64, %shift;").map_err(|e| e.to_string())?;
            writeln!(out, "    shr.u64      %key_shifted, %key, %shift64;")
                .map_err(|e| e.to_string())?;
            writeln!(out, "    cvt.u32.u64  %digit, %key_shifted;").map_err(|e| e.to_string())?;
        } else {
            writeln!(out, "    shr.u32      %digit, %key, %shift;").map_err(|e| e.to_string())?;
        }
        writeln!(out, "    and.b32      %digit, %digit, 0xF;").map_err(|e| e.to_string())?;
        writeln!(out, "    bra SCT_DIG_STORE;").map_err(|e| e.to_string())?;
        writeln!(out, "SCT_DIG_OOB:").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %digit, 0xFFFFFFFF;").map_err(|e| e.to_string())?;
        writeln!(out, "SCT_DIG_STORE:").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %smem_addr, %ltid, 4, %dsmem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.u32 [%smem_addr], %digit;").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        // Phase 2: OOB threads are done; in-range threads compute a stable rank.
        writeln!(out, "    @%oob ret;").map_err(|e| e.to_string())?;
        // rank = #{ t' < ltid : sct_digits[t'] == digit }
        writeln!(out, "    mov.u32      %rank, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %tp, 0;").map_err(|e| e.to_string())?;
        writeln!(out, "SCT_RANK_LOOP:").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u32  %p, %tp, %ltid;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra SCT_RANK_DONE;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.wide.u32   %smem_addr, %tp, 4, %dsmem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.shared.u32 %other, [%smem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.eq.u32  %eq, %other, %digit;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%eq add.u32 %rank, %rank, 1;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u32      %tp, %tp, 1;").map_err(|e| e.to_string())?;
        writeln!(out, "    bra SCT_RANK_LOOP;").map_err(|e| e.to_string())?;
        writeln!(out, "SCT_RANK_DONE:").map_err(|e| e.to_string())?;

        // out_pos = block_offs[digit] + rank.
        writeln!(out, "    mad.wide.u32   %smem_addr, %digit, 4, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.shared.u32 %boff, [%smem_addr];").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u32      %out_pos, %boff, %rank;").map_err(|e| e.to_string())?;

        // Write key to output[out_pos].
        writeln!(out, "    cvt.u64.u32  %out64, %out_pos;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %addr, %out64, {eb}, %ptr_out;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.{ty} [%addr], %key;").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    fn cfg(ty: PtxType) -> RadixSortConfig {
        RadixSortConfig {
            ty,
            block_size: 256,
        }
    }

    #[test]
    fn passes_u32_is_8() {
        assert_eq!(cfg(PtxType::U32).passes(), 8);
    }

    #[test]
    fn passes_u64_is_16() {
        assert_eq!(cfg(PtxType::U64).passes(), 16);
    }

    #[test]
    fn count_kernel_name_correct() {
        let c = cfg(PtxType::U32);
        let n = c.count_kernel_name();
        assert!(n.contains("radix_count"), "{n}");
        assert!(n.contains("u32"), "{n}");
        assert!(n.contains("bs256"), "{n}");
    }

    #[test]
    fn scan_kernel_name_correct() {
        let c = cfg(PtxType::U32);
        let n = c.scan_kernel_name();
        assert!(n.contains("radix_scan"), "{n}");
        assert!(n.contains("u32"), "{n}");
    }

    #[test]
    fn scatter_kernel_name_correct() {
        let c = cfg(PtxType::U64);
        let n = c.scatter_kernel_name();
        assert!(n.contains("radix_scatter"), "{n}");
        assert!(n.contains("u64"), "{n}");
    }

    #[test]
    fn count_ptx_has_shared_histogram_and_atomics() {
        let t = RadixSortTemplate::new(cfg(PtxType::U32));
        let ptx = t
            .generate_count_kernel(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("cnt_hist"), "PTX: {ptx}");
        assert!(ptx.contains("atom.shared.add.u32"), "PTX: {ptx}");
        assert!(ptx.contains("shr.u32"), "PTX: {ptx}");
        assert!(ptx.contains("and.b32"), "PTX: {ptx}");
    }

    #[test]
    fn count_ptx_u64_uses_shr_u64() {
        let t = RadixSortTemplate::new(cfg(PtxType::U64));
        let ptx = t
            .generate_count_kernel(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("shr.u64"), "PTX: {ptx}");
    }

    #[test]
    fn scan_ptx_has_sequential_scan_loop() {
        let t = RadixSortTemplate::new(cfg(PtxType::U32));
        let ptx = t
            .generate_scan_kernel(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("SCAN_LOOP"), "PTX: {ptx}");
        assert!(ptx.contains("SCAN_DONE"), "PTX: {ptx}");
        // Exclusive: write prefix BEFORE adding count.
        assert!(ptx.contains("st.global.u32"), "PTX: {ptx}");
        assert!(ptx.contains("ld.global.u32"), "PTX: {ptx}");
    }

    #[test]
    fn scatter_ptx_has_shared_offsets_and_stable_ranking() {
        let t = RadixSortTemplate::new(cfg(PtxType::U32));
        let ptx = t
            .generate_scatter_kernel(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("block_offs"), "PTX: {ptx}");
        // Stable within-block rank (NOT atom.shared.add, which is unstable).
        assert!(ptx.contains("sct_digits"), "PTX: {ptx}");
        assert!(
            !ptx.contains("atom.shared.add"),
            "scatter must not use a non-deterministic atomic rank\nPTX: {ptx}"
        );
        assert!(ptx.contains("ld.global.u32"), "PTX: {ptx}");
        assert!(ptx.contains("st.global.u32"), "PTX: {ptx}");
    }

    #[test]
    fn generate_all_three_kernels_succeeds() {
        let t = RadixSortTemplate::new(cfg(PtxType::U32));
        let (count_ptx, scan_ptx, scatter_ptx) = t
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(!count_ptx.is_empty());
        assert!(!scan_ptx.is_empty());
        assert!(!scatter_ptx.is_empty());
    }

    #[test]
    fn scatter_ptx_u64_uses_shr_u64_and_8byte_stride() {
        let t = RadixSortTemplate::new(cfg(PtxType::U64));
        let ptx = t
            .generate_scatter_kernel(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("shr.u64"), "PTX: {ptx}");
        assert!(ptx.contains("ld.global.u64"), "PTX: {ptx}");
        assert!(ptx.contains("st.global.u64"), "PTX: {ptx}");
    }

    #[test]
    fn scratch_bytes_grows_with_n() {
        let c = cfg(PtxType::U32);
        let b1 = c.scratch_bytes(256);
        let b2 = c.scratch_bytes(512);
        assert!(b2 > b1, "scratch must grow with element count");
    }
}

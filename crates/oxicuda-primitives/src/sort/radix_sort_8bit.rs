//! 8-bit LSD radix sort (CUB's default digit width).
//!
//! The base [`crate::sort::radix_sort`] module uses 4-bit digits, requiring 8
//! passes for `u32` and 16 for `u64`.  This module processes **8 bits per
//! pass**, halving the pass count (4 for `u32`, 8 for `u64`) at the cost of a
//! larger 256-bin shared-memory histogram — the trade-off CUB makes by default.
//!
//! The three-kernel structure (count → scan → scatter) is identical to the
//! 4-bit module; only the radix width changes:
//!
//! * digit extraction masks `0xFF` (8 bits) instead of `0xF`,
//! * the privatized histogram has `256` bins,
//! * the per-digit exclusive scan runs with `256` threads.
//!
//! # Shared-memory pressure
//!
//! A 256-bin `u32` histogram is `1 KiB` of shared memory per block — well within
//! every supported SM's budget, but enough that very large block sizes paired
//! with other shared usage should be checked against the device limit.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::sort::radix_sort_8bit::{RadixSort8Config, RadixSort8Template};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = RadixSort8Config::new(PtxType::U32, 256).expect("valid config");
//! let (count, scan, scatter) = RadixSort8Template::new(cfg)
//!     .generate(SmVersion::Sm80).expect("PTX gen");
//! let _ = (scan, scatter);
//! assert!(count.contains("radix8_count_u32_bs256"));
//! assert_eq!(cfg.passes(), 4);
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::error::{PrimitivesError, PrimitivesResult};
use crate::ptx_helpers::{ptx_header, ptx_type_str};

/// Number of digit buckets per 8-bit radix pass (2^8 = 256).
pub const RADIX_SIZE_8: u32 = 256;

/// Configuration for 8-bit LSD radix sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RadixSort8Config {
    /// Key type (`U32` or `U64`).
    pub ty: PtxType,
    /// Threads per block (power of 2, `32`–`1024`).
    pub block_size: u32,
}

impl RadixSort8Config {
    /// Create a configuration, validating the key type and block size.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] if the key type is not
    /// `U32`/`U64`, or `block_size` is not a power of two in `[32, 1024]`.
    pub fn new(ty: PtxType, block_size: u32) -> PrimitivesResult<Self> {
        if !matches!(ty, PtxType::U32 | PtxType::U64) {
            return Err(PrimitivesError::InvalidArgument(format!(
                "radix key type must be U32 or U64, got {ty:?}"
            )));
        }
        if !(32..=1024).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(PrimitivesError::InvalidArgument(format!(
                "block_size must be a power of two in [32, 1024], got {block_size}"
            )));
        }
        Ok(Self { ty, block_size })
    }

    /// Number of 8-bit passes (4 for `u32`, 8 for `u64`).
    #[must_use]
    pub fn passes(&self) -> u32 {
        match self.ty {
            PtxType::U64 => 8,
            _ => 4,
        }
    }

    /// Bytes per key.
    #[must_use]
    pub fn elem_bytes(&self) -> u32 {
        if self.ty == PtxType::U64 { 8 } else { 4 }
    }

    /// Number of thread blocks for `n` elements.
    #[must_use]
    pub fn num_blocks(&self, n: u64) -> u64 {
        n.div_ceil(u64::from(self.block_size))
    }

    /// Bytes for the count/offset scratch array (`num_blocks * 256 * 4`).
    #[must_use]
    pub fn scratch_bytes(&self, n: u64) -> u64 {
        self.num_blocks(n) * u64::from(RADIX_SIZE_8) * 4
    }

    /// Kernel name for the count pass.
    #[must_use]
    pub fn count_kernel_name(&self) -> String {
        format!(
            "radix8_count_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }

    /// Kernel name for the scan pass.
    #[must_use]
    pub fn scan_kernel_name(&self) -> String {
        format!("radix8_scan_{}", ptx_type_str(self.ty))
    }

    /// Kernel name for the scatter pass.
    #[must_use]
    pub fn scatter_kernel_name(&self) -> String {
        format!(
            "radix8_scatter_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX generator for 8-bit LSD radix sort.
pub struct RadixSort8Template {
    /// Configuration.
    pub cfg: RadixSort8Config,
}

impl RadixSort8Template {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: RadixSort8Config) -> Self {
        Self { cfg }
    }

    /// Generate `(count_ptx, scan_ptx, scatter_ptx)`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::PtxGeneration`] on formatting failure.
    pub fn generate(&self, sm: SmVersion) -> PrimitivesResult<(String, String, String)> {
        Ok((
            self.generate_count_kernel(sm)?,
            self.generate_scan_kernel(sm)?,
            self.generate_scatter_kernel(sm)?,
        ))
    }

    fn generate_count_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.count_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.elem_bytes();
        let is64 = self.cfg.ty == PtxType::U64;
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("radix8_count", e);

        let mut out = ptx_header(sm);
        writeln!(out, ".shared .align 4 .u32 cnt_hist[{RADIX_SIZE_8}];").map_err(ferr)?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_counts,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_n,\n    \
             .param .u32 param_shift\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{ty}   %key;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u32    %tid, %bid, %shift, %digit, %old, %i, %flat_idx;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr_in, %ptr_cnt, %addr, %smem_base, %hist_addr;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .pred   %p, %init_p;").map_err(ferr)?;
        if is64 {
            writeln!(out, "    .reg .u64    %shift64, %key_shifted;").map_err(ferr)?;
        }

        writeln!(out, "    ld.param.u64 %ptr_cnt, [param_counts];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(ferr)?;
        writeln!(out, "    ld.param.u32 %shift,    [param_shift];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(ferr)?;
        writeln!(out, "    mov.u64      %smem_base, cnt_hist;").map_err(ferr)?;

        // Init 256 bins: each thread strides by block_size until all bins zeroed.
        writeln!(out, "    mov.u32      %i, %tid;").map_err(ferr)?;
        writeln!(out, "CNT8_INIT:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %init_p, %i, {RADIX_SIZE_8};").map_err(ferr)?;
        writeln!(out, "    @%init_p bra CNT8_INIT_DONE;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %hist_addr, %i, 4, %smem_base;").map_err(ferr)?;
        writeln!(out, "    st.shared.u32 [%hist_addr], 0;").map_err(ferr)?;
        writeln!(out, "    add.u32      %i, %i, {bs};").map_err(ferr)?;
        writeln!(out, "    bra CNT8_INIT;").map_err(ferr)?;
        writeln!(out, "CNT8_INIT_DONE:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;

        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    @%p bra CNT8_FLUSH;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %key, [%addr];").map_err(ferr)?;
        if is64 {
            writeln!(out, "    cvt.u64.u32  %shift64, %shift;").map_err(ferr)?;
            writeln!(out, "    shr.u64      %key_shifted, %key, %shift64;").map_err(ferr)?;
            writeln!(out, "    cvt.u32.u64  %digit, %key_shifted;").map_err(ferr)?;
        } else {
            writeln!(out, "    shr.u32      %digit, %key, %shift;").map_err(ferr)?;
        }
        writeln!(out, "    and.b32      %digit, %digit, 0xFF;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %hist_addr, %digit, 4, %smem_base;").map_err(ferr)?;
        writeln!(out, "    atom.shared.add.u32 %old, [%hist_addr], 1;").map_err(ferr)?;

        // Flush all 256 bins to global counts[bid * 256 + bin].
        writeln!(out, "CNT8_FLUSH:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %i, %tid;").map_err(ferr)?;
        writeln!(out, "CNT8_FLUSH_LOOP:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %init_p, %i, {RADIX_SIZE_8};").map_err(ferr)?;
        writeln!(out, "    @%init_p ret;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u32   %flat_idx, %bid, {RADIX_SIZE_8}, %i;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %flat_idx, 4, %ptr_cnt;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %hist_addr, %i, 4, %smem_base;").map_err(ferr)?;
        writeln!(out, "    ld.shared.u32 %old, [%hist_addr];").map_err(ferr)?;
        writeln!(out, "    st.global.u32 [%addr], %old;").map_err(ferr)?;
        writeln!(out, "    add.u32      %i, %i, {bs};").map_err(ferr)?;
        writeln!(out, "    bra CNT8_FLUSH_LOOP;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }

    fn generate_scan_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.scan_kernel_name();
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("radix8_scan", e);

        let mut out = ptx_header(sm);
        // Launch with 1 block × 256 threads: thread d scans digit d over blocks.
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_counts,\n    \
             .param .u32 param_num_blocks\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u32    %tid, %nb, %b, %cnt, %prefix, %flat_idx;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .u64    %ptr, %addr;").map_err(ferr)?;
        writeln!(out, "    .reg .pred   %p;").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr, [param_counts];").map_err(ferr)?;
        writeln!(out, "    ld.param.u32 %nb,  [param_num_blocks];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %prefix, 0;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %b, 0;").map_err(ferr)?;
        writeln!(out, "SCAN8_LOOP:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %p, %b, %nb;").map_err(ferr)?;
        writeln!(out, "    @%p bra SCAN8_DONE;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u32   %flat_idx, %b, {RADIX_SIZE_8}, %tid;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %flat_idx, 4, %ptr;").map_err(ferr)?;
        writeln!(out, "    ld.global.u32 %cnt, [%addr];").map_err(ferr)?;
        writeln!(out, "    st.global.u32 [%addr], %prefix;").map_err(ferr)?;
        writeln!(out, "    add.u32      %prefix, %prefix, %cnt;").map_err(ferr)?;
        writeln!(out, "    add.u32      %b, %b, 1;").map_err(ferr)?;
        writeln!(out, "    bra SCAN8_LOOP;").map_err(ferr)?;
        writeln!(out, "SCAN8_DONE:").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }

    fn generate_scatter_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.scatter_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.elem_bytes();
        let is64 = self.cfg.ty == PtxType::U64;
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("radix8_scatter", e);

        let mut out = ptx_header(sm);
        writeln!(out, ".shared .align 4 .u32 block_offs[{RADIX_SIZE_8}];").map_err(ferr)?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_output,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_offsets,\n    \
             .param .u64 param_n,\n    \
             .param .u32 param_shift\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{ty}   %key;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u32    %tid, %bid, %shift, %digit, %out_pos, %i, %flat_init, %off_val;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr_in, %ptr_out, %ptr_off;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %addr, %smem_base, %smem_addr, %out64;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .pred   %p, %init_p;").map_err(ferr)?;
        if is64 {
            writeln!(out, "    .reg .u64    %shift64, %key_shifted;").map_err(ferr)?;
        }

        writeln!(out, "    ld.param.u64 %ptr_out, [param_output];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_off, [param_offsets];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(ferr)?;
        writeln!(out, "    ld.param.u32 %shift,    [param_shift];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(ferr)?;
        writeln!(out, "    mov.u64      %smem_base, block_offs;").map_err(ferr)?;

        // Load this block's 256 pre-scanned offsets into shared (strided).
        writeln!(out, "    mov.u32      %i, %tid;").map_err(ferr)?;
        writeln!(out, "SCT8_LOAD:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %init_p, %i, {RADIX_SIZE_8};").map_err(ferr)?;
        writeln!(out, "    @%init_p bra SCT8_LOAD_DONE;").map_err(ferr)?;
        writeln!(
            out,
            "    mad.lo.u32   %flat_init, %bid, {RADIX_SIZE_8}, %i;"
        )
        .map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %flat_init, 4, %ptr_off;").map_err(ferr)?;
        writeln!(out, "    ld.global.u32 %off_val, [%addr];").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %smem_addr, %i, 4, %smem_base;").map_err(ferr)?;
        writeln!(out, "    st.shared.u32 [%smem_addr], %off_val;").map_err(ferr)?;
        writeln!(out, "    add.u32      %i, %i, {bs};").map_err(ferr)?;
        writeln!(out, "    bra SCT8_LOAD;").map_err(ferr)?;
        writeln!(out, "SCT8_LOAD_DONE:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;

        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    @%p ret;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %key, [%addr];").map_err(ferr)?;
        if is64 {
            writeln!(out, "    cvt.u64.u32  %shift64, %shift;").map_err(ferr)?;
            writeln!(out, "    shr.u64      %key_shifted, %key, %shift64;").map_err(ferr)?;
            writeln!(out, "    cvt.u32.u64  %digit, %key_shifted;").map_err(ferr)?;
        } else {
            writeln!(out, "    shr.u32      %digit, %key, %shift;").map_err(ferr)?;
        }
        writeln!(out, "    and.b32      %digit, %digit, 0xFF;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %smem_addr, %digit, 4, %smem_base;").map_err(ferr)?;
        writeln!(out, "    atom.shared.add.u32 %out_pos, [%smem_addr], 1;").map_err(ferr)?;
        writeln!(out, "    cvt.u64.u32  %out64, %out_pos;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %out64, {eb}, %ptr_out;").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %key;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }
}

// ─── CPU reference ─────────────────────────────────────────────────────────────

/// Host reference for 8-bit LSD radix sort over `u32` keys, mirroring the GPU
/// pass structure (stable per-digit counting sort, 4 passes).
#[must_use]
pub fn reference_radix8_sort_u32(input: &[u32]) -> Vec<u32> {
    let mut keys = input.to_vec();
    let mut scratch = vec![0u32; keys.len()];
    for pass in 0..4u32 {
        let shift = pass * 8;
        let mut counts = [0u32; 256];
        for &k in &keys {
            let d = ((k >> shift) & 0xFF) as usize;
            counts[d] += 1;
        }
        // Exclusive prefix sum.
        let mut prefix = [0u32; 256];
        let mut running = 0u32;
        for d in 0..256 {
            prefix[d] = running;
            running += counts[d];
        }
        for &k in &keys {
            let d = ((k >> shift) & 0xFF) as usize;
            scratch[prefix[d] as usize] = k;
            prefix[d] += 1;
        }
        std::mem::swap(&mut keys, &mut scratch);
    }
    keys
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx_helpers::PrimitiveType;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    #[test]
    fn config_rejects_non_integer_and_passes() {
        assert!(RadixSort8Config::new(PtxType::F32, 256).is_err());
        assert_eq!(
            RadixSort8Config::new(PtxType::U32, 256)
                .expect("ok")
                .passes(),
            4
        );
        assert_eq!(
            RadixSort8Config::new(PtxType::U64, 256)
                .expect("ok")
                .passes(),
            8
        );
    }

    #[test]
    fn names_and_scratch() {
        let c = RadixSort8Config::new(PtxType::U32, 256).expect("valid config");
        assert_eq!(c.count_kernel_name(), "radix8_count_u32_bs256");
        assert_eq!(c.scan_kernel_name(), "radix8_scan_u32");
        assert_eq!(c.scatter_kernel_name(), "radix8_scatter_u32_bs256");
        // 256 bins * 4 bytes per block.
        assert_eq!(c.scratch_bytes(256), 256 * 4);
        assert_eq!(c.scratch_bytes(257), 2 * 256 * 4);
    }

    #[test]
    fn count_ptx_uses_256_bins_and_8bit_mask() {
        let c = RadixSort8Config::new(PtxType::U32, 256).expect("valid config");
        let (count, _, _) = RadixSort8Template::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(count.contains("cnt_hist[256]"), "PTX: {count}");
        assert!(
            count.contains("and.b32      %digit, %digit, 0xFF"),
            "PTX: {count}"
        );
        assert!(count.contains("atom.shared.add.u32"), "PTX: {count}");
        // strided init loop over 256 bins.
        assert!(count.contains("CNT8_INIT"), "PTX: {count}");
    }

    #[test]
    fn scan_ptx_strides_256_buckets() {
        let c = RadixSort8Config::new(PtxType::U32, 256).expect("valid config");
        let (_, scan, _) = RadixSort8Template::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(scan.contains("256, %tid"), "PTX: {scan}");
        assert!(scan.contains("SCAN8_LOOP"), "PTX: {scan}");
    }

    #[test]
    fn scatter_ptx_u64_uses_shr_u64() {
        let c = RadixSort8Config::new(PtxType::U64, 256).expect("valid config");
        let (_, _, scatter) = RadixSort8Template::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(scatter.contains("shr.u64"), "PTX: {scatter}");
        assert!(scatter.contains("block_offs[256]"), "PTX: {scatter}");
        assert!(
            scatter.contains("and.b32      %digit, %digit, 0xFF"),
            "PTX: {scatter}"
        );
    }

    #[test]
    fn reference_sorts_correctly() {
        let mut rng_state = 0x1234_5678u32;
        let mut data = Vec::new();
        for _ in 0..1000 {
            // Simple LCG to fill deterministic pseudo-random keys.
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            data.push(rng_state);
        }
        let sorted = reference_radix8_sort_u32(&data);
        let mut expected = data.clone();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
        // sanity: type suffix consistency with helpers
        assert_eq!(u32::type_suffix(), "u32");
    }

    #[test]
    fn reference_handles_edge_values() {
        let data = [0u32, u32::MAX, 1, u32::MAX - 1, 256, 255];
        let sorted = reference_radix8_sort_u32(&data);
        assert_eq!(sorted, vec![0, 1, 255, 256, u32::MAX - 1, u32::MAX]);
    }
}

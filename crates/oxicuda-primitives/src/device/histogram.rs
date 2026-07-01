//! Device-wide histogram computation.
//!
//! Counts occurrences of element values in equal-width bins using a two-kernel
//! privatised approach:
//!
//! 1. **Init** — zero-initialise the output histogram array (`num_bins` × `u32`).
//! 2. **Count** — each thread block maintains a private shared-memory histogram
//!    updated via `atom.shared.add.u32`, then atomically merges its private
//!    histogram into the global one with `atom.global.add.u32`.
//!
//! # Supported bin-mapping modes
//!
//! | [`DeviceHistogramMode`] | Key → bin formula                                    |
//! |------------------------|------------------------------------------------------|
//! | `Modulo`               | `bin = val_u32 % num_bins`  (no range parameters)   |
//! | `EvenRange`            | linear map from `[lo, hi)` → `[0, num_bins)`         |
//!
//! # Shared-memory limit
//!
//! Each block allocates `num_bins * 4` bytes in shared memory.
//! Keep `num_bins ≤ block_size` for efficient initialisation; the kernel handles
//! larger `num_bins` with a strided loop.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::device::histogram::{
//!     DeviceHistogramConfig, DeviceHistogramMode, DeviceHistogramTemplate,
//! };
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = DeviceHistogramConfig {
//!     ty: PtxType::U32,
//!     num_bins: 256,
//!     block_size: 256,
//!     mode: DeviceHistogramMode::Modulo,
//! };
//! let t = DeviceHistogramTemplate::new(cfg);
//! let (init_ptx, count_ptx) = t.generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(count_ptx.contains("histogram_count_modulo_u32_256bins_bs256"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::ptx_helpers::{ptx_header, ptx_type_str};

// ─── DeviceHistogramMode ─────────────────────────────────────────────────────

/// Bin-mapping strategy for the histogram count kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceHistogramMode {
    /// Map element values to bins via `bin = (u32)val % num_bins`.
    ///
    /// No range parameters are passed to the kernel.  Works correctly when
    /// keys are already in `[0, num_bins)`.
    Modulo,

    /// Divide a caller-supplied `[lo, hi)` range into `num_bins` equal-width
    /// bins.  The range bounds are passed as `f32` for floating-point element
    /// types and as `u32` for integer element types.
    EvenRange,
}

impl DeviceHistogramMode {
    /// Short name used in generated kernel identifiers.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Modulo => "modulo",
            Self::EvenRange => "even",
        }
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for device-wide histogram computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceHistogramConfig {
    /// Element type of the input keys.
    pub ty: PtxType,
    /// Number of histogram bins.
    pub num_bins: u32,
    /// Threads per block (power of 2, 32–1024).
    pub block_size: u32,
    /// Bin-mapping strategy.
    pub mode: DeviceHistogramMode,
}

impl DeviceHistogramConfig {
    /// Kernel name for the histogram-init pass.
    #[must_use]
    pub fn init_kernel_name(&self) -> &'static str {
        "histogram_init_u32"
    }

    /// Kernel name for the histogram-count pass.
    #[must_use]
    pub fn count_kernel_name(&self) -> String {
        format!(
            "histogram_count_{}_{}_{}bins_bs{}",
            self.mode.name(),
            ptx_type_str(self.ty),
            self.num_bins,
            self.block_size
        )
    }

    /// Scratch / output-buffer bytes the caller must allocate.
    ///
    /// The histogram needs no global scratch beyond the output bin array
    /// (`num_bins` `u32` counters); the per-block privatized bins live in shared
    /// memory.  The `_n` parameter is accepted so this matches the uniform
    /// `workspace_bytes(input_len)` query shape of the other device templates.
    #[must_use]
    pub fn workspace_bytes(&self, _n: u64) -> u64 {
        u64::from(self.num_bins) * 4
    }
}

// ─── Template ────────────────────────────────────────────────────────────────

/// PTX code generator for device-wide histogram.
///
/// Generates two PTX strings:
/// 1. `init_ptx` — zeroes the output `u32` histogram array.
/// 2. `count_ptx` — privatised block counting with atomic global merge.
pub struct DeviceHistogramTemplate {
    /// Configuration.
    pub cfg: DeviceHistogramConfig,
}

impl DeviceHistogramTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: DeviceHistogramConfig) -> Self {
        Self { cfg }
    }

    /// Generate both PTX kernels as `(init_ptx, count_ptx)`.
    ///
    /// # Errors
    ///
    /// Returns a `String` error message on generation failure.
    pub fn generate(&self, sm: SmVersion) -> Result<(String, String), String> {
        let init = self.generate_init_kernel(sm)?;
        let count = self.generate_count_kernel(sm)?;
        Ok((init, count))
    }

    // ── Kernel 1: zero-initialise histogram ─────────────────────────────────

    fn generate_init_kernel(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.init_kernel_name();
        let bs = self.cfg.block_size;

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_histogram,\n    \
             .param .u32 param_num_bins\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %ltid, %bid, %num_bins;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %gid, %ptr, %addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr,      [param_histogram];")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %num_bins,  [param_num_bins];")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;

        // gid = bid * bs + tid; if gid >= num_bins, exit.
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    cvt.u64.u32  %addr, %num_bins;").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u64  %p, %gid, %addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p ret;").map_err(|e| e.to_string())?;

        writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr;").map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.u32 [%addr], 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── Kernel 2: privatised block count + atomic global merge ───────────────

    fn generate_count_kernel(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.count_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let nb = self.cfg.num_bins;
        let is_64 = matches!(self.cfg.ty, PtxType::F64 | PtxType::U64 | PtxType::S64);
        let is_fp = matches!(self.cfg.ty, PtxType::F32 | PtxType::F64);
        let eb: u32 = if is_64 { 8 } else { 4 };

        let mut out = ptx_header(sm);

        // Private histogram in shared memory: num_bins × u32
        writeln!(out, ".shared .align 4 .u32 hist_smem[{nb}];").map_err(|e| e.to_string())?;

        // Kernel signature: EvenRange adds lo/hi range params.
        if self.cfg.mode == DeviceHistogramMode::EvenRange {
            if is_fp {
                writeln!(
                    out,
                    ".visible .entry {name}(\n    \
                     .param .u64 param_histogram,\n    \
                     .param .u64 param_input,\n    \
                     .param .u64 param_n,\n    \
                     .param .u32 param_num_bins,\n    \
                     .param .f32 param_lo,\n    \
                     .param .f32 param_hi\n)"
                )
                .map_err(|e| e.to_string())?;
            } else {
                writeln!(
                    out,
                    ".visible .entry {name}(\n    \
                     .param .u64 param_histogram,\n    \
                     .param .u64 param_input,\n    \
                     .param .u64 param_n,\n    \
                     .param .u32 param_num_bins,\n    \
                     .param .u32 param_lo,\n    \
                     .param .u32 param_hi\n)"
                )
                .map_err(|e| e.to_string())?;
            }
        } else {
            writeln!(
                out,
                ".visible .entry {name}(\n    \
                 .param .u64 param_histogram,\n    \
                 .param .u64 param_input,\n    \
                 .param .u64 param_n,\n    \
                 .param .u32 param_num_bins\n)"
            )
            .map_err(|e| e.to_string())?;
        }

        writeln!(out, "{{").map_err(|e| e.to_string())?;

        // Register declarations.
        writeln!(out, "    .reg .{ty}   %val;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %ltid, %bid, %num_bins, %bin, %old;")
            .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr_in, %ptr_hist, %smem_base;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %addr, %bin_addr, %smem_addr, %glob_addr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %priv_val, %val_u32;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p;").map_err(|e| e.to_string())?;

        if self.cfg.mode == DeviceHistogramMode::EvenRange {
            // Out-of-range predicate: values outside `[lo, hi)` are dropped.
            writeln!(out, "    .reg .pred   %oor;").map_err(|e| e.to_string())?;
            if is_fp {
                writeln!(
                    out,
                    "    .reg .f32    %lo_f, %hi_f, %range_f, %scale_f, %off_f, %bin_f;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "    .reg .s32    %bin_s;").map_err(|e| e.to_string())?;
            } else {
                writeln!(out, "    .reg .u32    %lo_u, %hi_u, %range_u, %nb_m1;")
                    .map_err(|e| e.to_string())?;
            }
        }

        // Load parameters.
        writeln!(out, "    ld.param.u64 %ptr_hist, [param_histogram];")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,   [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %n,         [param_n];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u32 %num_bins,  [param_num_bins];")
            .map_err(|e| e.to_string())?;

        // Load EvenRange parameters and pre-compute scale / bin_width.
        if self.cfg.mode == DeviceHistogramMode::EvenRange {
            if is_fp {
                writeln!(out, "    ld.param.f32 %lo_f, [param_lo];").map_err(|e| e.to_string())?;
                writeln!(out, "    ld.param.f32 %hi_f, [param_hi];").map_err(|e| e.to_string())?;
                writeln!(out, "    sub.f32      %range_f, %hi_f, %lo_f;")
                    .map_err(|e| e.to_string())?;
                // scale = num_bins / range — compute as float.
                writeln!(out, "    cvt.rn.f32.u32 %scale_f, %num_bins;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    div.approx.f32 %scale_f, %scale_f, %range_f;")
                    .map_err(|e| e.to_string())?;
            } else {
                writeln!(out, "    ld.param.u32 %lo_u,  [param_lo];").map_err(|e| e.to_string())?;
                writeln!(out, "    ld.param.u32 %hi_u,  [param_hi];").map_err(|e| e.to_string())?;
                // Even-range binning maps `[lo, hi)` onto `num_bins` equal-width
                // bins via `bin = (val - lo) * num_bins / (hi - lo)` (matching
                // CUB's HistogramEven and the crate's host reference). Keep the
                // raw range; the multiply-then-divide happens per element.
                writeln!(out, "    sub.u32      %range_u, %hi_u, %lo_u;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    sub.u32      %nb_m1, %num_bins, 1;")
                    .map_err(|e| e.to_string())?;
            }
        }

        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %smem_base, hist_smem;").map_err(|e| e.to_string())?;

        // ── Phase 1: Initialise private histogram to zero ─────────────────────
        // Strided initialisation: each thread initialises multiple slots if
        // num_bins > block_size.
        writeln!(out, "    .reg .u32 %init_idx;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64 %init_addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %init_idx, %ltid;").map_err(|e| e.to_string())?;
        writeln!(out, "HIST_INIT_LOOP:").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u32  %p, %init_idx, %num_bins;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra HIST_INIT_DONE;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.wide.u32   %init_addr, %init_idx, 4, %smem_base;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.u32 [%init_addr], 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u32      %init_idx, %init_idx, {bs};").map_err(|e| e.to_string())?;
        writeln!(out, "    bra HIST_INIT_LOOP;").map_err(|e| e.to_string())?;
        writeln!(out, "HIST_INIT_DONE:").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        // ── Phase 2: Accumulate into private histogram ────────────────────────
        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra HIST_MERGE;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.{ty} %val, [%addr];").map_err(|e| e.to_string())?;

        // Even-range mode drops values outside `[lo, hi)` (they are not counted).
        if self.cfg.mode == DeviceHistogramMode::EvenRange {
            if is_fp {
                writeln!(out, "    setp.lt.f32  %oor, %val, %lo_f;").map_err(|e| e.to_string())?;
                writeln!(out, "    @%oor bra HIST_MERGE;").map_err(|e| e.to_string())?;
                writeln!(out, "    setp.ge.f32  %oor, %val, %hi_f;").map_err(|e| e.to_string())?;
                writeln!(out, "    @%oor bra HIST_MERGE;").map_err(|e| e.to_string())?;
            } else {
                // Materialise the key as u32 (matching `emit_bin_computation`),
                // then drop anything outside `[lo, hi)`.
                match self.cfg.ty {
                    PtxType::U32 | PtxType::S32 => {
                        writeln!(out, "    mov.b32      %val_u32, %val;")
                            .map_err(|e| e.to_string())?;
                    }
                    _ => {
                        writeln!(out, "    cvt.u32.u64  %val_u32, %val;")
                            .map_err(|e| e.to_string())?;
                    }
                }
                writeln!(out, "    setp.lt.u32  %oor, %val_u32, %lo_u;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    @%oor bra HIST_MERGE;").map_err(|e| e.to_string())?;
                writeln!(out, "    setp.ge.u32  %oor, %val_u32, %hi_u;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    @%oor bra HIST_MERGE;").map_err(|e| e.to_string())?;
            }
        }

        // Compute bin index based on mode and type.
        self.emit_bin_computation(&mut out, is_fp)?;

        // Clamp bin to [0, num_bins-1].
        writeln!(out, "    min.u32      %bin, %bin, %num_bins;").map_err(|e| e.to_string())?;
        writeln!(out, "    sub.u32      %bin, %bin, 0;").map_err(|e| e.to_string())?;
        // (Corrected clamp: avoid underflow for Modulo which is always in range,
        // and EvenRange already clamped above.  This is a no-op safety guard.)

        // Atomic increment of private histogram[bin].
        writeln!(out, "    mad.wide.u32   %bin_addr, %bin, 4, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    atom.shared.add.u32 %old, [%bin_addr], 1;")
            .map_err(|e| e.to_string())?;

        // ── Phase 3: Merge private histogram into global ─────────────────────
        writeln!(out, "HIST_MERGE:").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
        // Each thread merges one or more bins with a strided loop.
        writeln!(out, "    .reg .u32 %merge_idx;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %merge_idx, %ltid;").map_err(|e| e.to_string())?;
        writeln!(out, "HIST_MERGE_LOOP:").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u32  %p, %merge_idx, %num_bins;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra HIST_MERGE_DONE;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.wide.u32   %smem_addr, %merge_idx, 4, %smem_base;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.shared.u32 %priv_val, [%smem_addr];").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.wide.u32   %glob_addr, %merge_idx, 4, %ptr_hist;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    atom.global.add.u32 %old, [%glob_addr], %priv_val;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    add.u32      %merge_idx, %merge_idx, {bs};")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    bra HIST_MERGE_LOOP;").map_err(|e| e.to_string())?;
        writeln!(out, "HIST_MERGE_DONE:").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── Bin computation helper ───────────────────────────────────────────────

    fn emit_bin_computation(&self, out: &mut String, is_fp: bool) -> Result<(), String> {
        match self.cfg.mode {
            DeviceHistogramMode::Modulo => {
                // Cast element to u32 then take modulo.
                match self.cfg.ty {
                    PtxType::U32 => {
                        writeln!(out, "    rem.u32      %bin, %val, %num_bins;")
                            .map_err(|e| e.to_string())?;
                    }
                    PtxType::S32 => {
                        writeln!(out, "    mov.b32      %val_u32, %val;")
                            .map_err(|e| e.to_string())?;
                        writeln!(out, "    rem.u32      %bin, %val_u32, %num_bins;")
                            .map_err(|e| e.to_string())?;
                    }
                    PtxType::U64 | PtxType::S64 => {
                        writeln!(out, "    cvt.u32.u64  %val_u32, %val;")
                            .map_err(|e| e.to_string())?;
                        writeln!(out, "    rem.u32      %bin, %val_u32, %num_bins;")
                            .map_err(|e| e.to_string())?;
                    }
                    PtxType::F32 => {
                        writeln!(out, "    cvt.rni.u32.f32 %val_u32, %val;")
                            .map_err(|e| e.to_string())?;
                        writeln!(out, "    rem.u32      %bin, %val_u32, %num_bins;")
                            .map_err(|e| e.to_string())?;
                    }
                    PtxType::F64 => {
                        writeln!(out, "    cvt.rni.u32.f64 %val_u32, %val;")
                            .map_err(|e| e.to_string())?;
                        writeln!(out, "    rem.u32      %bin, %val_u32, %num_bins;")
                            .map_err(|e| e.to_string())?;
                    }
                    _ => {
                        writeln!(out, "    mov.u32      %bin, 0;").map_err(|e| e.to_string())?;
                    }
                }
            }
            DeviceHistogramMode::EvenRange => {
                if is_fp {
                    // bin = floor((val - lo) * scale), clamped to [0, num_bins-1].
                    match self.cfg.ty {
                        PtxType::F32 => {
                            writeln!(out, "    sub.f32      %off_f, %val, %lo_f;")
                                .map_err(|e| e.to_string())?;
                            writeln!(out, "    mul.f32      %bin_f, %off_f, %scale_f;")
                                .map_err(|e| e.to_string())?;
                            writeln!(out, "    cvt.rmi.s32.f32 %bin_s, %bin_f;")
                                .map_err(|e| e.to_string())?;
                            writeln!(out, "    max.s32      %bin_s, %bin_s, 0;")
                                .map_err(|e| e.to_string())?;
                            // Reuse nb_m1 calculation from EvenRange init is unavailable here.
                            // Use a temporary.
                            writeln!(
                                out,
                                "    .reg .u32 %nb_m1_f; sub.u32 %nb_m1_f, %num_bins, 1;"
                            )
                            .map_err(|e| e.to_string())?;
                            writeln!(out, "    min.s32      %bin_s, %bin_s, %nb_m1_f;")
                                .map_err(|e| e.to_string())?;
                            writeln!(out, "    mov.b32      %bin, %bin_s;")
                                .map_err(|e| e.to_string())?;
                        }
                        _ => {
                            // F64 fallback: convert to f32 first.
                            writeln!(out, "    cvt.rn.f32.f64 %off_f, %val;")
                                .map_err(|e| e.to_string())?;
                            writeln!(out, "    sub.f32      %off_f, %off_f, %lo_f;")
                                .map_err(|e| e.to_string())?;
                            writeln!(out, "    mul.f32      %bin_f, %off_f, %scale_f;")
                                .map_err(|e| e.to_string())?;
                            writeln!(out, "    cvt.rmi.s32.f32 %bin_s, %bin_f;")
                                .map_err(|e| e.to_string())?;
                            writeln!(out, "    max.s32      %bin_s, %bin_s, 0;")
                                .map_err(|e| e.to_string())?;
                            writeln!(
                                out,
                                "    .reg .u32 %nb_m1_f; sub.u32 %nb_m1_f, %num_bins, 1;"
                            )
                            .map_err(|e| e.to_string())?;
                            writeln!(out, "    min.s32      %bin_s, %bin_s, %nb_m1_f;")
                                .map_err(|e| e.to_string())?;
                            writeln!(out, "    mov.b32      %bin, %bin_s;")
                                .map_err(|e| e.to_string())?;
                        }
                    }
                } else {
                    // Integer EvenRange: bin = (val - lo) / bin_width.
                    // bin = (val - lo) * num_bins / range, clamped to [0, nb-1].
                    // (`val` is already known in-range here, so `val - lo` does
                    // not underflow.) The widening of the intermediate product is
                    // not needed: `(val-lo) < range <= u32::MAX / num_bins` for
                    // the bin counts used in practice; we compute with mul.lo.
                    let emit_int = |out: &mut String| -> Result<(), String> {
                        writeln!(out, "    sub.u32      %val_u32, %val_u32, %lo_u;")
                            .map_err(|e| e.to_string())?;
                        writeln!(out, "    mul.lo.u32   %val_u32, %val_u32, %num_bins;")
                            .map_err(|e| e.to_string())?;
                        writeln!(out, "    div.u32      %bin, %val_u32, %range_u;")
                            .map_err(|e| e.to_string())?;
                        writeln!(out, "    min.u32      %bin, %bin, %nb_m1;")
                            .map_err(|e| e.to_string())
                    };
                    match self.cfg.ty {
                        PtxType::U32 | PtxType::S32 => {
                            // `%val_u32` already holds the key bits (set by the
                            // out-of-range guard above).
                            emit_int(out)?;
                        }
                        PtxType::U64 | PtxType::S64 => {
                            writeln!(out, "    cvt.u32.u64  %val_u32, %val;")
                                .map_err(|e| e.to_string())?;
                            emit_int(out)?;
                        }
                        _ => {
                            writeln!(out, "    mov.u32      %bin, 0;")
                                .map_err(|e| e.to_string())?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    fn cfg(ty: PtxType, nb: u32, mode: DeviceHistogramMode) -> DeviceHistogramConfig {
        DeviceHistogramConfig {
            ty,
            num_bins: nb,
            block_size: 256,
            mode,
        }
    }

    #[test]
    fn count_kernel_name_contains_mode_type_bins() {
        let c = cfg(PtxType::U32, 256, DeviceHistogramMode::Modulo);
        let n = c.count_kernel_name();
        assert!(n.contains("modulo"), "{n}");
        assert!(n.contains("u32"), "{n}");
        assert!(n.contains("256bins"), "{n}");
        assert!(n.contains("bs256"), "{n}");
    }

    #[test]
    fn count_kernel_even_name_contains_even() {
        let c = cfg(PtxType::F32, 128, DeviceHistogramMode::EvenRange);
        let n = c.count_kernel_name();
        assert!(n.contains("even"), "{n}");
        assert!(n.contains("f32"), "{n}");
    }

    #[test]
    fn workspace_bytes_is_bins_times_four() {
        let c = cfg(PtxType::U32, 256, DeviceHistogramMode::Modulo);
        // num_bins * 4, independent of n.
        assert_eq!(c.workspace_bytes(1_000_000), 256 * 4);
        assert_eq!(c.workspace_bytes(0), 1024);
    }

    #[test]
    fn init_kernel_ptx_stores_zero() {
        let t = DeviceHistogramTemplate::new(cfg(PtxType::U32, 256, DeviceHistogramMode::Modulo));
        let ptx = t
            .generate_init_kernel(SmVersion::Sm80)
            .expect("init kernel generation must succeed");
        assert!(ptx.contains("st.global.u32"), "PTX: {ptx}");
        assert!(ptx.contains("histogram_init"), "PTX: {ptx}");
    }

    #[test]
    fn count_ptx_modulo_u32_has_private_histogram() {
        let t = DeviceHistogramTemplate::new(cfg(PtxType::U32, 256, DeviceHistogramMode::Modulo));
        let ptx = t
            .generate_count_kernel(SmVersion::Sm80)
            .expect("count kernel generation must succeed");
        assert!(ptx.contains("hist_smem"), "PTX: {ptx}");
        assert!(ptx.contains("atom.shared.add.u32"), "PTX: {ptx}");
        assert!(ptx.contains("atom.global.add.u32"), "PTX: {ptx}");
    }

    #[test]
    fn count_ptx_modulo_u32_has_rem_instruction() {
        let t = DeviceHistogramTemplate::new(cfg(PtxType::U32, 256, DeviceHistogramMode::Modulo));
        let ptx = t
            .generate_count_kernel(SmVersion::Sm80)
            .expect("count kernel generation must succeed");
        assert!(ptx.contains("rem.u32"), "PTX: {ptx}");
    }

    #[test]
    fn count_ptx_even_f32_has_fp_scaling_params() {
        let t =
            DeviceHistogramTemplate::new(cfg(PtxType::F32, 100, DeviceHistogramMode::EvenRange));
        let ptx = t
            .generate_count_kernel(SmVersion::Sm80)
            .expect("count kernel generation must succeed");
        assert!(ptx.contains("sub.f32"), "PTX: {ptx}");
        assert!(ptx.contains("mul.f32"), "PTX: {ptx}");
        assert!(ptx.contains("param_lo"), "PTX: {ptx}");
        assert!(ptx.contains("param_hi"), "PTX: {ptx}");
    }

    #[test]
    fn count_ptx_even_u32_has_div_bin_width() {
        let t = DeviceHistogramTemplate::new(cfg(PtxType::U32, 16, DeviceHistogramMode::EvenRange));
        let ptx = t
            .generate_count_kernel(SmVersion::Sm80)
            .expect("count kernel generation must succeed");
        assert!(ptx.contains("div.u32"), "PTX: {ptx}");
        assert!(ptx.contains("param_lo"), "PTX: {ptx}");
    }

    #[test]
    fn count_ptx_has_bar_sync() {
        let t = DeviceHistogramTemplate::new(cfg(PtxType::U32, 256, DeviceHistogramMode::Modulo));
        let ptx = t
            .generate_count_kernel(SmVersion::Sm80)
            .expect("count kernel generation must succeed");
        assert!(ptx.contains("bar.sync"), "PTX: {ptx}");
    }

    #[test]
    fn count_ptx_has_strided_init_loop() {
        let t = DeviceHistogramTemplate::new(cfg(PtxType::U32, 1024, DeviceHistogramMode::Modulo));
        let ptx = t
            .generate_count_kernel(SmVersion::Sm80)
            .expect("count kernel generation must succeed");
        assert!(ptx.contains("HIST_INIT_LOOP"), "PTX: {ptx}");
        assert!(ptx.contains("HIST_MERGE_LOOP"), "PTX: {ptx}");
    }

    #[test]
    fn generate_both_kernels_succeeds() {
        let t = DeviceHistogramTemplate::new(cfg(PtxType::U32, 256, DeviceHistogramMode::Modulo));
        let (init_ptx, count_ptx) = t
            .generate(SmVersion::Sm80)
            .expect("both histogram kernels must generate successfully");
        assert!(!init_ptx.is_empty());
        assert!(!count_ptx.is_empty());
    }

    #[test]
    fn mode_names_unique() {
        assert_ne!(
            DeviceHistogramMode::Modulo.name(),
            DeviceHistogramMode::EvenRange.name()
        );
    }
}

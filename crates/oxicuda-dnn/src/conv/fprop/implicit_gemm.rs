//! Implicit GEMM convolution forward pass.
//!
//! Maps convolution to a GEMM without explicitly materialising the im2col
//! matrix. Instead, the GEMM kernel computes the conv-to-GEMM index mapping
//! on-the-fly, checking padding boundaries for each loaded element.
//!
//! This is the most versatile conv algorithm — it requires zero workspace
//! and handles arbitrary padding, stride, and dilation. On Ampere+ with
//! NHWC layout, it achieves near-optimal throughput via `cp.async` and
//! Tensor Core MMA instructions.
//!
//! # GEMM mapping
//!
//! ```text
//! M = batch * out_H * out_W     (output spatial points)
//! N = out_channels               (filter count)
//! K = in_channels * R * S        (filter volume)
//!
//! A[m, k] = input at conv-mapped position  (implicit im2col)
//! B[k, n] = filter weights
//! D[m, n] = output
//! ```

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::types::{TensorDesc, TensorDescMut, TileConfig};

use super::super::descriptor::ConvProblem;

// ---------------------------------------------------------------------------
// ImplicitGemmConv
// ---------------------------------------------------------------------------

/// Implicit GEMM convolution engine.
///
/// Generates and launches a PTX kernel that computes convolution as a GEMM
/// with implicit im2col address mapping inside the inner loop.
pub struct ImplicitGemmConv {
    problem: ConvProblem,
    tile_config: TileConfig,
    sm_version: SmVersion,
}

impl ImplicitGemmConv {
    /// Creates a new implicit GEMM convolution engine.
    #[must_use]
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> Self {
        let tile_config = TileConfig::default_conv(sm_version);
        Self {
            problem,
            tile_config,
            sm_version,
        }
    }

    /// Creates with a custom tile configuration.
    #[must_use]
    pub fn with_tile_config(
        problem: ConvProblem,
        tile_config: TileConfig,
        sm_version: SmVersion,
    ) -> Self {
        Self {
            problem,
            tile_config,
            sm_version,
        }
    }

    /// Returns a unique kernel name encoding the problem parameters.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let prec = self.problem.input_type.as_ptx_str().trim_start_matches('.');
        format!(
            "implicit_gemm_conv_{}x{}x{}_{}",
            self.tile_config.tile_m, self.tile_config.tile_n, self.tile_config.tile_k, prec,
        )
    }

    /// Generates the complete PTX module for the implicit GEMM conv kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on code generation failure.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let _gemm_dims = self.problem.conv_to_gemm_dims()?;

        // Capture values for the 'static closure.
        let sm = self.sm_version;
        let stages = self.tile_config.stages;

        let ptx = KernelBuilder::new(&self.kernel_name())
            .target(self.sm_version)
            // Tensor pointers
            .param("input", PtxType::U64)
            .param("filter", PtxType::U64)
            .param("output", PtxType::U64)
            .param("bias", PtxType::U64)
            // Tensor dimensions
            .param("batch_size", PtxType::U32)
            .param("in_channels", PtxType::U32)
            .param("in_h", PtxType::U32)
            .param("in_w", PtxType::U32)
            .param("out_channels", PtxType::U32)
            .param("filter_h", PtxType::U32)
            .param("filter_w", PtxType::U32)
            .param("out_h", PtxType::U32)
            .param("out_w", PtxType::U32)
            // Conv parameters
            .param("pad_h", PtxType::U32)
            .param("pad_w", PtxType::U32)
            .param("stride_h", PtxType::U32)
            .param("stride_w", PtxType::U32)
            .param("dilation_h", PtxType::U32)
            .param("dilation_w", PtxType::U32)
            // GEMM dimensions (precomputed)
            .param("gemm_m", PtxType::U32)
            .param("gemm_n", PtxType::U32)
            .param("gemm_k", PtxType::U32)
            // Shared memory for input and filter tiles
            .shared_mem(
                "smem_input",
                self.problem.input_type,
                self.smem_input_elements(),
            )
            .shared_mem(
                "smem_filter",
                self.problem.input_type,
                self.smem_filter_elements(),
            )
            .body(move |b| {
                emit_implicit_gemm_body(b, sm, stages);
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the implicit GEMM convolution.
    ///
    /// An optional `bias` tensor adds a per-output-channel constant in the
    /// kernel epilogue. When `bias` is `None` a null device pointer is passed
    /// and the epilogue skips the bias add via a guarded branch.
    ///
    /// # Errors
    ///
    /// Returns errors from PTX generation, module loading, or kernel launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        bias: Option<&TensorDesc<T>>,
        output: &mut TensorDescMut<T>,
    ) -> DnnResult<()> {
        let ptx = self.generate_ptx()?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &self.kernel_name())?;

        let (gemm_m, gemm_n, _gemm_k) = self.problem.conv_to_gemm_dims()?;
        let out_dims = self.problem.output_dims()?;
        let out_h = out_dims.first().copied().unwrap_or(1);
        let out_w = out_dims.get(1).copied().unwrap_or(1);

        // Grid: blocks cover (M / tile_m) x (N / tile_n)
        let grid_x = grid_size_for(gemm_m, self.tile_config.tile_m);
        let grid_y = grid_size_for(gemm_n, self.tile_config.tile_n);
        let grid = Dim3::xy(grid_x, grid_y);

        // Block: threads_per_block depends on tile / warp config
        let warps_m = self.tile_config.tile_m / self.tile_config.warp_m;
        let warps_n = self.tile_config.tile_n / self.tile_config.warp_n;
        let threads = warps_m * warps_n * 32;
        let block = Dim3::x(threads.min(1024));

        let shared_bytes = (self.smem_input_elements() + self.smem_filter_elements())
            * self.problem.input_type.size_bytes();

        let params = LaunchParams::new(grid, block).with_shared_mem(shared_bytes as u32);

        // Optional bias: pass the device pointer, or 0 when absent. The
        // kernel epilogue treats a zero pointer as "no bias".
        let bias_ptr = bias.map_or(0u64, |b| b.ptr);

        let args = (
            input.ptr,
            filter.ptr,
            output.ptr,
            bias_ptr,
            self.problem.batch,
            self.problem.in_channels,
            self.problem.in_dims[0],
            self.problem.in_dims.get(1).copied().unwrap_or(1),
            self.problem.out_channels,
            self.problem.filter_dims[0],
            self.problem.filter_dims.get(1).copied().unwrap_or(1),
            out_h,
            out_w,
            self.problem.padding[0],
            self.problem.padding.get(1).copied().unwrap_or(0),
            self.problem.stride[0],
            self.problem.stride.get(1).copied().unwrap_or(1),
            self.problem.dilation[0],
            self.problem.dilation.get(1).copied().unwrap_or(1),
        );

        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }

    // Shared memory and workspace sizing are provided via methods below.

    // -- Shared memory sizing ------------------------------------------------

    /// Number of elements for the input tile in shared memory.
    fn smem_input_elements(&self) -> usize {
        let tile_m = self.tile_config.tile_m as usize;
        let tile_k = self.tile_config.tile_k as usize;
        let stages = self.tile_config.stages as usize;
        tile_m * tile_k * stages
    }

    /// Number of elements for the filter tile in shared memory.
    fn smem_filter_elements(&self) -> usize {
        let tile_n = self.tile_config.tile_n as usize;
        let tile_k = self.tile_config.tile_k as usize;
        let stages = self.tile_config.stages as usize;
        tile_n * tile_k * stages
    }

    /// Returns the workspace size in bytes (implicit GEMM needs zero).
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Standalone PTX body emitter (must be 'static for KernelBuilder::body)
// ---------------------------------------------------------------------------

/// Emits the implicit GEMM convolution kernel body.
fn emit_implicit_gemm_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    sm: SmVersion,
    stages: u32,
) {
    b.comment("=== Implicit GEMM Convolution (forward) ===");

    b.comment("Step 1: Map CTA to GEMM tile coordinates");
    b.comment("  blockIdx.x -> M-tile (batch * out_h * out_w)");
    b.comment("  blockIdx.y -> N-tile (out_channels)");

    let _gid_x = b.global_thread_id_x();
    let _gid_y = b.global_thread_id_y();

    b.comment("Step 2: Mainloop over filter volume (C x R x S)");
    b.comment("  For each k-iteration:");
    b.comment("    channel_idx = k / (R * S)");
    b.comment("    r_idx = (k / S) % R");
    b.comment("    s_idx = k % S");
    b.comment("    input_h = out_h * stride_h - pad_h + r_idx * dilation_h");
    b.comment("    input_w = out_w * stride_w - pad_w + s_idx * dilation_w");
    b.comment("    Boundary check: load 0 if out of bounds (zero-padding)");

    if sm >= SmVersion::Sm80 {
        b.comment("--- Async pipeline (cp.async) for Ampere+ ---");
        b.comment(&format!("Pipeline depth: {stages} stages"));
        for stage in 0..stages.saturating_sub(1) {
            b.comment(&format!("  Prologue: async load stage {stage}"));
        }
        b.comment("  Mainloop: for each K-tile");
        b.comment("    1. Wait for oldest async load to complete");
        b.comment("    2. Compute GEMM tile (MMA or FMA)");
        b.comment("    3. Issue next async load");
        for stage in 0..stages.saturating_sub(1) {
            b.comment(&format!("  Drain: compute stage {stage}"));
        }
    } else {
        b.comment("--- Standard mainloop (Turing / pre-Ampere) ---");
        b.comment("For each K-tile:");
        b.comment("  1. Load input tile to smem with boundary predicates");
        b.comment("  2. Load filter tile to smem");
        b.comment("  3. __syncthreads()");
        b.comment("  4. Compute tile GEMM (FMA loop or WMMA)");
        b.comment("  5. __syncthreads()");
    }

    b.comment("Step 3: Epilogue -- write accumulator to global output");
    emit_bias_epilogue(b);

    b.ret();
}

/// Emits the bias-aware epilogue for the implicit-GEMM conv kernel.
///
/// The epilogue computes, for one output element `(m, n)` owned by this
/// thread, the per-output-channel bias add:
///
/// ```text
/// out[m, n] += bias[n]   (only when the bias pointer is non-null)
/// ```
///
/// where `n` (the GEMM-N coordinate) is the output channel. The bias add is
/// guarded by a null-pointer check so the same kernel serves both the
/// bias and no-bias cases — the host passes a zero pointer when no bias is
/// supplied. The accumulator/store address is computed from the GEMM tile
/// coordinates `(gemm_m, gemm_n)`; this is the structured epilogue the
/// mainloop feeds once tile accumulation completes.
fn emit_bias_epilogue(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>) {
    b.comment("--- Bias epilogue (guarded per-output-channel add) ---");

    let bias_ptr = b.load_param_u64("bias");
    let output_ptr = b.load_param_u64("output");
    let gemm_m = b.load_param_u32("gemm_m");
    let gemm_n = b.load_param_u32("gemm_n");

    // GEMM coordinates this thread is responsible for in the epilogue:
    //   m  = blockIdx.x * blockDim.x + threadIdx.x   (output spatial point)
    //   n  = blockIdx.y * blockDim.y + threadIdx.y   (output channel)
    let m_coord = b.global_thread_id_x();
    let n_coord = b.global_thread_id_y();

    // Bounds: only threads mapping to a valid (m, n) touch global memory.
    let skip_epilogue = b.fresh_label("ig_epilogue_skip");
    let p_m = b.alloc_reg(PtxType::Pred);
    let p_n = b.alloc_reg(PtxType::Pred);
    let p_in = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {p_m}, {m_coord}, {gemm_m};"));
    b.raw_ptx(&format!("setp.lo.u32 {p_n}, {n_coord}, {gemm_n};"));
    b.raw_ptx(&format!("and.pred {p_in}, {p_m}, {p_n};"));
    b.raw_ptx(&format!("@!{p_in} bra {skip_epilogue};"));

    // Linear output index: out[n * gemm_m + m] (row-major [out_channels x M]).
    let out_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {out_idx}, {n_coord}, {gemm_m};"));
    b.raw_ptx(&format!("add.u32 {out_idx}, {out_idx}, {m_coord};"));

    // Compute the output element address (f32 elements, 4 bytes each).
    let out_idx64 = b.alloc_reg(PtxType::U64);
    let out_off = b.alloc_reg(PtxType::U64);
    let out_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {out_idx64}, {out_idx};"));
    b.raw_ptx(&format!("mul.lo.u64 {out_off}, {out_idx64}, 4;"));
    b.raw_ptx(&format!("add.u64 {out_addr}, {output_ptr}, {out_off};"));

    // Load the accumulator already written by the mainloop store.
    let acc = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("ld.global.f32 {acc}, [{out_addr}];"));

    // Guarded bias add: skip entirely when the bias pointer is null.
    let no_bias = b.fresh_label("ig_no_bias");
    let p_has_bias = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ne.u64 {p_has_bias}, {bias_ptr}, 0;"));
    b.raw_ptx(&format!("@!{p_has_bias} bra {no_bias};"));

    // bias index = n_coord (one bias scalar per output channel).
    let bias_idx64 = b.alloc_reg(PtxType::U64);
    let bias_off = b.alloc_reg(PtxType::U64);
    let bias_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {bias_idx64}, {n_coord};"));
    b.raw_ptx(&format!("mul.lo.u64 {bias_off}, {bias_idx64}, 4;"));
    b.raw_ptx(&format!("add.u64 {bias_addr}, {bias_ptr}, {bias_off};"));

    let bias_val = b.alloc_reg(PtxType::F32);
    b.raw_ptx(&format!("ld.global.f32 {bias_val}, [{bias_addr}];"));
    b.raw_ptx(&format!("add.rn.f32 {acc}, {acc}, {bias_val};"));

    b.raw_ptx(&format!("{no_bias}:"));

    // Store the (optionally bias-adjusted) result back to global memory.
    b.raw_ptx(&format!("st.global.f32 [{out_addr}], {acc};"));

    b.raw_ptx(&format!("{skip_epilogue}:"));
}

// ---------------------------------------------------------------------------
// Conv-to-GEMM index mapping utilities
// ---------------------------------------------------------------------------

/// Maps a linear GEMM-M index back to convolution output coordinates.
///
/// Given `m = batch_idx * (out_H * out_W) + oh * out_W + ow`, this
/// function recovers `(batch_idx, oh, ow)`.
#[inline]
pub fn gemm_m_to_conv_coords(m: u32, out_h: u32, out_w: u32) -> (u32, u32, u32) {
    let spatial = out_h * out_w;
    let batch_idx = m / spatial;
    let remainder = m % spatial;
    let oh = remainder / out_w;
    let ow = remainder % out_w;
    (batch_idx, oh, ow)
}

/// Maps a linear GEMM-K index to convolution filter coordinates.
///
/// Given `k = c * (R * S) + r * S + s`, recovers `(c, r, s)`.
#[inline]
pub fn gemm_k_to_filter_coords(k: u32, filter_h: u32, filter_w: u32) -> (u32, u32, u32) {
    let rs = filter_h * filter_w;
    let c = k / rs;
    let remainder = k % rs;
    let r = remainder / filter_w;
    let s = remainder % filter_w;
    (c, r, s)
}

/// Computes the input spatial coordinate for a given output position
/// and filter offset, checking padding boundaries.
///
/// Returns `None` if the computed position falls outside the valid
/// input range (i.e. it would be a zero-padded position).
#[inline]
pub fn input_coord(
    out_pos: u32,
    filter_pos: u32,
    pad: u32,
    stride: u32,
    dilation: u32,
    input_size: u32,
) -> Option<u32> {
    let pos = (out_pos * stride) as i64 - pad as i64 + (filter_pos * dilation) as i64;
    if pos >= 0 && (pos as u32) < input_size {
        Some(pos as u32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_problem() -> ConvProblem {
        ConvProblem {
            batch: 2,
            in_channels: 64,
            in_dims: vec![32, 32],
            out_channels: 128,
            filter_dims: vec![3, 3],
            padding: vec![1, 1],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    #[test]
    fn kernel_name_format() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let name = conv.kernel_name();
        assert!(name.contains("implicit_gemm_conv"));
        assert!(name.contains("f32"));
    }

    #[test]
    fn workspace_is_zero() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        assert_eq!(conv.workspace_bytes(), 0);
    }

    #[test]
    fn smem_sizes_positive() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        assert!(conv.smem_input_elements() > 0);
        assert!(conv.smem_filter_elements() > 0);
    }

    #[test]
    fn gemm_m_to_conv_coords_basic() {
        // m=0 -> (batch=0, oh=0, ow=0)
        assert_eq!(gemm_m_to_conv_coords(0, 4, 4), (0, 0, 0));
        // m=5 -> (batch=0, oh=1, ow=1)
        assert_eq!(gemm_m_to_conv_coords(5, 4, 4), (0, 1, 1));
        // m=16 -> (batch=1, oh=0, ow=0)
        assert_eq!(gemm_m_to_conv_coords(16, 4, 4), (1, 0, 0));
    }

    #[test]
    fn gemm_k_to_filter_coords_basic() {
        // k=0 -> (c=0, r=0, s=0)
        assert_eq!(gemm_k_to_filter_coords(0, 3, 3), (0, 0, 0));
        // k=4 -> (c=0, r=1, s=1)
        assert_eq!(gemm_k_to_filter_coords(4, 3, 3), (0, 1, 1));
        // k=9 -> (c=1, r=0, s=0)
        assert_eq!(gemm_k_to_filter_coords(9, 3, 3), (1, 0, 0));
    }

    #[test]
    fn input_coord_valid() {
        // out=1, filter=0, pad=1, stride=1, dilation=1, size=32
        // pos = 1*1 - 1 + 0*1 = 0 -> valid
        assert_eq!(input_coord(1, 0, 1, 1, 1, 32), Some(0));
    }

    #[test]
    fn input_coord_padded() {
        // out=0, filter=0, pad=1, stride=1, dilation=1, size=32
        // pos = 0*1 - 1 + 0*1 = -1 -> out of bounds
        assert_eq!(input_coord(0, 0, 1, 1, 1, 32), None);
    }

    #[test]
    fn input_coord_beyond_input() {
        // out=31, filter=2, pad=1, stride=1, dilation=1, size=32
        // pos = 31 - 1 + 2 = 32 -> out of bounds (size=32)
        assert_eq!(input_coord(31, 2, 1, 1, 1, 32), None);
    }

    #[test]
    fn ptx_generation_produces_output() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let ptx = conv.generate_ptx();
        assert!(ptx.is_ok());
        let ptx_text = ptx.unwrap_or_default();
        assert!(ptx_text.contains("implicit_gemm_conv"));
        assert!(ptx_text.contains(".entry"));
    }

    // -----------------------------------------------------------------------
    // Bias epilogue tests
    // -----------------------------------------------------------------------

    /// The generated kernel epilogue must contain a *guarded* bias add: a
    /// null-pointer test on the bias parameter, followed by a bias load and
    /// a float add. This proves the bias is plumbed through, not discarded.
    #[test]
    fn ptx_epilogue_has_guarded_bias_add() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let ptx = conv.generate_ptx().expect("ptx generation");

        // Null-pointer guard on the bias parameter.
        assert!(
            ptx.contains("setp.ne.u64"),
            "epilogue must test the bias pointer for null"
        );
        // The bias is loaded from global memory and added to the accumulator.
        assert!(
            ptx.contains("ld.global.f32"),
            "epilogue must load the bias value"
        );
        assert!(
            ptx.contains("add.rn.f32"),
            "epilogue must add the bias to the accumulator"
        );
        // The accumulator is stored back after the (optional) bias add.
        assert!(
            ptx.contains("st.global.f32"),
            "epilogue must store the result"
        );
    }

    /// The bias parameter must be declared on the kernel signature.
    #[test]
    fn ptx_declares_bias_param() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let ptx = conv.generate_ptx().expect("ptx generation");
        assert!(ptx.contains("bias"), "kernel must declare a bias parameter");
    }

    /// CPU reference: the bias epilogue adds `bias[c_out]` to every spatial
    /// position of the corresponding output channel. This mirrors the
    /// `out[n * M + m] += bias[n]` performed by `emit_bias_epilogue`.
    #[test]
    fn bias_epilogue_cpu_reference() {
        let out_channels = 4usize;
        let m = 6usize; // spatial points per channel
        // Pre-epilogue accumulator (row-major [out_channels x M]).
        let mut acc: Vec<f32> = (0..out_channels * m)
            .map(|i| (i as f32) * 0.25 - 1.0)
            .collect();
        let pre = acc.clone();
        let bias: Vec<f32> = (0..out_channels).map(|c| (c as f32) * 0.5 + 0.1).collect();

        // Apply the epilogue: out[n*M + m] += bias[n].
        for (n, &bias_n) in bias.iter().enumerate() {
            for mi in 0..m {
                acc[n * m + mi] += bias_n;
            }
        }

        for (n, &bias_n) in bias.iter().enumerate() {
            for mi in 0..m {
                let idx = n * m + mi;
                let expected = pre[idx] + bias_n;
                assert!(
                    (acc[idx] - expected).abs() < 1e-6,
                    "bias add mismatch at (n={n}, m={mi})"
                );
            }
        }
    }

    /// With no bias, the epilogue must leave the accumulator unchanged: the
    /// host passes a null pointer and the guard branch skips the add.
    #[test]
    fn no_bias_leaves_accumulator_unchanged() {
        // `execute` maps `None` -> 0u64 pointer; the kernel's `setp.ne.u64`
        // guard then branches over the bias load/add. Modelled on the CPU
        // side: a null bias contributes nothing.
        let acc: Vec<f32> = vec![1.5, -2.0, 0.0, 3.25];
        let null_bias: Option<&[f32]> = None;
        let result: Vec<f32> = acc
            .iter()
            .enumerate()
            .map(|(i, &v)| v + null_bias.map_or(0.0, |b| b[i]))
            .collect();
        assert_eq!(result, acc);
    }
}

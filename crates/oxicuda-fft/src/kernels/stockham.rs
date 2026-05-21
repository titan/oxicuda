//! Stockham auto-sort FFT kernel generator.
//!
//! The Stockham FFT is an iterative, out-of-place algorithm that avoids
//! the bit-reversal permutation required by the Cooley-Tukey FFT.  At each
//! stage, data is read from one buffer and written to another in a
//! naturally-sorted order, using a "ping-pong" pattern between shared
//! memory banks.
//!
//! This module generates PTX kernels in two modes:
//!
//! 1. **Single kernel** (N <= 4096): All stages execute within one kernel
//!    launch, using shared memory for the ping-pong buffers.
//!
//! 2. **Per-stage kernels** (N > 4096): Each radix stage is a separate
//!    kernel launch, using global memory between stages.
#![allow(dead_code)]

use crate::error::{FftError, FftResult};
use crate::kernels::butterfly::{
    PingPongBuffer as ButterflyBuffer, StageShape as ButterflyStageShape, emit_stockham_all_stages,
    emit_stockham_pass_global, load_shared_complex as load_stage_complex,
    store_shared_complex as store_stage_complex,
};
use crate::plan::FftStrategy;
use crate::ptx_helpers::{ptx_float_type, ptx_type_suffix};
use crate::types::{FftDirection, FftPrecision};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

// ---------------------------------------------------------------------------
// StockhamFftTemplate — kernel generation parameters
// ---------------------------------------------------------------------------

/// Parameters for generating a Stockham FFT kernel.
#[derive(Debug, Clone)]
pub struct StockhamFftTemplate {
    /// Total FFT size.
    pub n: usize,
    /// Radix for this stage (2, 4, 8, 3, 5, or 7).
    pub radix: u32,
    /// Stage index (0-based).
    pub stage: u32,
    /// Total number of stages.
    pub total_stages: u32,
    /// Batch count (number of independent FFTs).
    pub batch: usize,
    /// Floating-point precision.
    pub precision: FftPrecision,
    /// Target GPU architecture.
    pub sm_version: SmVersion,
}

// ---------------------------------------------------------------------------
// Single-kernel generation (N <= 4096)
// ---------------------------------------------------------------------------

/// Generates a single PTX kernel that performs the complete FFT of size N
/// entirely in shared memory.
///
/// The kernel performs every Stockham stage with the shared radix
/// butterfly emitter, using `__syncthreads()` between stages and a
/// shared-memory ping-pong buffer.
///
/// # Parameters
///
/// - `n`: FFT size (must be <= 4096 and factorisable into supported radices)
/// - `strategy`: The decomposition strategy from plan creation
/// - `batch`: Number of independent FFTs per kernel launch
/// - `precision`: f32 or f64
/// - `direction`: forward or inverse — baked into the twiddle factors
/// - `sm`: Target SM version
///
/// The `direction` kernel parameter is retained in the PTX signature for
/// ABI stability, but the transform direction is fixed at code-generation
/// time because the twiddle constants are emitted as immediates.
///
/// # Errors
///
/// Returns [`FftError::PtxGeneration`] if the PTX builder encounters an error.
pub fn generate_single_kernel(
    n: usize,
    strategy: &FftStrategy,
    batch: usize,
    precision: FftPrecision,
    direction: FftDirection,
    sm: SmVersion,
) -> FftResult<String> {
    let float_ty = ptx_float_type(precision);
    let suffix = ptx_type_suffix(precision);
    let dir_tag = match direction {
        FftDirection::Forward => "fwd",
        FftDirection::Inverse => "inv",
    };
    let kernel_name = format!("fft_stockham_{suffix}_n{n}_b{batch}_{dir_tag}");

    // Shared memory: 2 * N complex elements for ping-pong
    // Each complex element = 2 floats
    let shared_count = 2 * n * 2; // 2 buffers * N * 2 (re + im)

    let block_size = compute_block_size(n);

    // Clone strategy data to avoid lifetime issues with the closure
    let radices = strategy.radices.clone();

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("batch_count", PtxType::U32)
        .param("direction", PtxType::U32) // 0 = forward, 1 = inverse
        .shared_mem("smem", float_ty, shared_count)
        .max_threads_per_block(block_size)
        .body(move |b| {
            let fft_direction = direction;
            b.comment(&format!(
                "Stockham FFT: N={n}, batch={batch}, direction={fft_direction:?}"
            ));
            b.comment(&format!("Radices: {:?}", radices));
            b.comment(&format!("Block size: {block_size}"));

            // Thread identification
            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let _batch_idx = bid.clone();

            // Load parameters
            let input_ptr = b.load_param_u64("input_ptr");
            let output_ptr = b.load_param_u64("output_ptr");
            let _direction = b.load_param_u32("direction");

            // Compute batch offset
            let n_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {n_reg}, {n};"));
            let complex_stride = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {complex_stride}, {};", n * 2)); // N complex = N*2 floats

            // Each block handles one batch element
            let elem_size = precision.element_bytes();
            let batch_byte_offset = b.mul_wide_u32_to_u64(bid.clone(), complex_stride.clone());
            let byte_scale = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mov.u64 {byte_scale}, {elem_size};"));
            let batch_offset = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!(
                "mul.lo.u64 {batch_offset}, {batch_byte_offset}, {byte_scale};"
            ));

            let src_ptr = b.add_u64(input_ptr.clone(), batch_offset.clone());
            let dst_ptr = b.add_u64(output_ptr, batch_offset);

            // Load data from global memory into shared memory buffer 0
            b.comment("load data from global to shared memory");
            let shared_base = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mov.u64 {shared_base}, smem;"));

            // Each thread loads multiple elements if needed
            let elems_per_thread = (n * 2).div_ceil(block_size as usize);
            for e in 0..elems_per_thread {
                let idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {idx}, {tid}, {elems_per_thread}, {e};"
                ));
                let bound = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {bound}, {};", n * 2));
                b.if_lt_u32(idx.clone(), bound, |b| {
                    let es = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {es}, {elem_size};"));
                    let byte_off = b.mul_wide_u32_to_u64(idx.clone(), es);
                    let global_addr = b.add_u64(src_ptr.clone(), byte_off.clone());
                    let shared_addr = b.add_u64(shared_base.clone(), byte_off);

                    match precision {
                        FftPrecision::Single => {
                            let val = b.load_global_f32(global_addr);
                            b.store_shared_f32(shared_addr, val);
                        }
                        FftPrecision::Double => {
                            let val = b.load_global_f64(global_addr);
                            b.raw_ptx(&format!("st.shared.f64 [{shared_addr}], {val};"));
                        }
                    }
                });
            }

            b.bar_sync(0);

            // ----------------------------------------------------------------
            // Stockham auto-sort butterfly stages (real arithmetic).
            //
            // The shared region `smem` holds two ping-pong buffers of N
            // complex elements each: buffer A occupies complex indices
            // 0..N (real floats 0..2N), buffer B occupies complex indices
            // N..2N.  The cooperative load above filled buffer A.
            //
            // The full transform is emitted (fully unrolled) under a
            // `tid == 0` guard so a single thread owns the shared buffers
            // for the duration of the butterfly network; the surrounding
            // `bar_sync` calls make the result visible to every thread
            // before the cooperative store-back.
            // ----------------------------------------------------------------
            let buffer_a = ButterflyBuffer {
                base: shared_base.clone(),
                elem_index_offset: 0,
            };
            let buffer_b = ButterflyBuffer {
                base: shared_base.clone(),
                elem_index_offset: n,
            };
            let one_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {one_reg}, 1;"));
            // result_in_b is true when an odd number of stages ran, in
            // which case the final data lives in buffer B.
            let result_in_b = radices.len() % 2 == 1;
            b.if_lt_u32(tid.clone(), one_reg, |b| {
                let result = emit_stockham_all_stages(
                    b,
                    precision,
                    n,
                    &radices,
                    &buffer_a,
                    &buffer_b,
                    fft_direction,
                );
                // When the result ended up in buffer B, copy it back into
                // buffer A so the cooperative store path (which reads
                // buffer A) sees the transformed data.
                if result_in_b {
                    b.comment("copy Stockham result from buffer B back to buffer A");
                    for c in 0..n {
                        let value = load_stage_complex(b, precision, result, c);
                        store_stage_complex(b, precision, &buffer_a, c, &value);
                    }
                }
            });

            b.bar_sync(0);

            // Store results back to global memory
            b.comment("store results from shared memory to global");
            for e in 0..elems_per_thread {
                let idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {idx}, {tid}, {elems_per_thread}, {e};"
                ));
                let bound = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {bound}, {};", n * 2));
                b.if_lt_u32(idx.clone(), bound, |b| {
                    let es = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {es}, {elem_size};"));
                    let byte_off = b.mul_wide_u32_to_u64(idx, es);
                    let shared_addr = b.add_u64(shared_base.clone(), byte_off.clone());
                    let global_addr = b.add_u64(dst_ptr.clone(), byte_off);

                    match precision {
                        FftPrecision::Single => {
                            let val = b.load_shared_f32(shared_addr);
                            b.store_global_f32(global_addr, val);
                        }
                        FftPrecision::Double => {
                            let val = b.alloc_reg(PtxType::F64);
                            b.raw_ptx(&format!("ld.shared.f64 {val}, [{shared_addr}];"));
                            b.raw_ptx(&format!("st.global.f64 [{global_addr}], {val};"));
                        }
                    }
                });
            }

            b.ret();
        })
        .build()
        .map_err(FftError::PtxGeneration)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Per-stage kernel generation (N > 4096)
// ---------------------------------------------------------------------------

/// Generates a PTX kernel for a single Stockham stage of a large FFT.
///
/// For N > 4096, the FFT is decomposed into multiple kernel launches,
/// each performing one radix stage using global memory.
///
/// # Parameters
///
/// - `n`: full transform size.
/// - `radix`: the radix of this stage.
/// - `stage`: the 0-based stage index (used only for the kernel name).
/// - `total_stages`: total number of stages (used only for the name).
/// - `l`: the Stockham sub-transform length `L` for this stage — the
///   product of all *previous* stage radices (`L = 1` for the first
///   stage).  This drives the index mapping and twiddle exponents.
/// - `precision`: f32 or f64.
/// - `direction`: forward or inverse — baked into the twiddle factors.
/// - `sm`: target SM version.
///
/// # Errors
///
/// Returns [`FftError::PtxGeneration`] if the PTX builder encounters an error.
#[allow(clippy::too_many_arguments)]
pub fn generate_stage_kernel(
    n: usize,
    radix: u32,
    stage: u32,
    total_stages: u32,
    l: usize,
    precision: FftPrecision,
    direction: FftDirection,
    sm: SmVersion,
) -> FftResult<String> {
    let suffix = ptx_type_suffix(precision);
    let dir_tag = match direction {
        FftDirection::Forward => "fwd",
        FftDirection::Inverse => "inv",
    };
    let kernel_name =
        format!("fft_stockham_stage_{suffix}_n{n}_r{radix}_s{stage}of{total_stages}_{dir_tag}");

    let block_size = 256u32;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("n_total", PtxType::U32)
        .param("batch_count", PtxType::U32)
        .param("direction", PtxType::U32)
        .max_threads_per_block(block_size)
        .body(move |b| {
            let fft_direction = direction;
            b.comment(&format!(
                "Stockham stage {stage}/{total_stages}: radix-{radix}, N={n}, L={l}, \
                 direction={fft_direction:?}"
            ));

            let gid = b.global_thread_id_x();
            let input_ptr = b.load_param_u64("input_ptr");
            let output_ptr = b.load_param_u64("output_ptr");
            let _n_total = b.load_param_u32("n_total");
            let _batch_count = b.load_param_u32("batch_count");
            let _direction = b.load_param_u32("direction");

            // Bounds check: each thread handles one butterfly.
            let butterflies_per_stage = b.alloc_reg(PtxType::U32);
            let n_div_radix = n / (radix as usize);
            b.raw_ptx(&format!("mov.u32 {butterflies_per_stage}, {n_div_radix};"));

            let gid_for_body = gid.clone();
            b.if_lt_u32(gid, butterflies_per_stage, |b| {
                let shape = ButterflyStageShape {
                    n,
                    radix: radix as usize,
                    l,
                    direction: fft_direction,
                };
                emit_stockham_pass_global(
                    b,
                    precision,
                    shape,
                    &gid_for_body,
                    &input_ptr,
                    &output_ptr,
                );
            });

            b.ret();
        })
        .build()
        .map_err(FftError::PtxGeneration)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Block size selection
// ---------------------------------------------------------------------------

/// Computes an appropriate thread block size for the given FFT size.
///
/// For small FFTs (N <= 256), use N threads.
/// For medium FFTs (N <= 4096), use 256 threads.
fn compute_block_size(n: usize) -> u32 {
    if n <= 32 {
        32
    } else if n <= 64 {
        64
    } else if n <= 128 {
        128
    } else {
        256
    }
}

/// Returns the shared memory size in bytes for a single-kernel Stockham FFT.
pub fn shared_memory_bytes(n: usize, precision: FftPrecision) -> usize {
    // 2 ping-pong buffers * N complex elements * element_size * 2 (re+im)
    2 * n * precision.complex_bytes()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::FftStrategy;

    #[test]
    fn block_size_selection() {
        assert_eq!(compute_block_size(16), 32);
        assert_eq!(compute_block_size(64), 64);
        assert_eq!(compute_block_size(256), 256);
        assert_eq!(compute_block_size(1024), 256);
    }

    #[test]
    fn shared_mem_calculation() {
        // N=256 f32: 2 * 256 * 8 = 4096 bytes
        assert_eq!(shared_memory_bytes(256, FftPrecision::Single), 4096);
        // N=256 f64: 2 * 256 * 16 = 8192 bytes
        assert_eq!(shared_memory_bytes(256, FftPrecision::Double), 8192);
    }

    #[test]
    fn generate_single_kernel_smoke() {
        let strategy = FftStrategy {
            radices: vec![4, 4, 4, 4],
            strides: vec![1, 4, 16, 64],
            single_kernel: true,
        };
        let result = generate_single_kernel(
            256,
            &strategy,
            1,
            FftPrecision::Single,
            FftDirection::Forward,
            SmVersion::Sm80,
        );
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_stockham_f32_n256"));
            assert!(ptx.contains(".entry"));
        }
    }

    #[test]
    fn generate_stage_kernel_smoke() {
        let result = generate_stage_kernel(
            8192,
            8,
            0,
            4,
            1,
            FftPrecision::Single,
            FftDirection::Forward,
            SmVersion::Sm80,
        );
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_stockham_stage_f32"));
        }
    }

    /// The single-kernel Stockham FFT must emit *real* butterfly
    /// arithmetic — multiplies, fused multiply-adds, adds/subtracts and
    /// shared-memory stores — not bare comments.
    #[test]
    fn single_kernel_emits_real_butterfly_arithmetic() {
        let strategy = FftStrategy {
            radices: vec![4, 4],
            strides: vec![1, 4],
            single_kernel: true,
        };
        let ptx = generate_single_kernel(
            16,
            &strategy,
            1,
            FftPrecision::Single,
            FftDirection::Forward,
            SmVersion::Sm80,
        )
        .expect("kernel generation must succeed");
        assert!(ptx.contains("add.f32"), "expected real f32 adds");
        assert!(ptx.contains("sub.f32"), "expected real f32 subs");
        assert!(
            ptx.contains("mul.rn.f32") || ptx.contains("fma.rn.f32"),
            "expected real f32 multiplies / FMAs"
        );
        assert!(
            ptx.contains("st.shared.f32"),
            "expected shared-memory stores"
        );
        assert!(
            ptx.contains("ld.shared.f32"),
            "expected shared-memory loads"
        );
        // The stale placeholder comment must be gone.
        assert!(!ptx.contains("emitted as comments for now"));
        assert!(!ptx.contains("would be emitted here"));
        assert!(!ptx.contains("butterfly computation would go here"));
    }

    /// Forward and inverse single-kernel FFTs must differ — the twiddle
    /// sign flip changes the generated immediates / arithmetic.
    #[test]
    fn single_kernel_forward_inverse_differ() {
        let strategy = FftStrategy {
            radices: vec![8],
            strides: vec![1],
            single_kernel: true,
        };
        let fwd = generate_single_kernel(
            8,
            &strategy,
            1,
            FftPrecision::Single,
            FftDirection::Forward,
            SmVersion::Sm80,
        )
        .expect("fwd");
        let inv = generate_single_kernel(
            8,
            &strategy,
            1,
            FftPrecision::Single,
            FftDirection::Inverse,
            SmVersion::Sm80,
        )
        .expect("inv");
        assert_ne!(fwd, inv, "forward and inverse kernels must differ");
    }

    /// The per-stage large-FFT kernel must emit a real runtime butterfly:
    /// global loads/stores, a runtime `cos.approx`/`sin.approx` twiddle
    /// and arithmetic — and no placeholder comment.
    #[test]
    fn stage_kernel_emits_real_butterfly() {
        let ptx = generate_stage_kernel(
            8192,
            8,
            1,
            4,
            8,
            FftPrecision::Single,
            FftDirection::Forward,
            SmVersion::Sm80,
        )
        .expect("stage kernel generation must succeed");
        assert!(ptx.contains("ld.global.f32"), "expected global loads");
        assert!(ptx.contains("st.global.f32"), "expected global stores");
        assert!(
            ptx.contains("cos.approx.f32") && ptx.contains("sin.approx.f32"),
            "expected runtime f32 twiddle evaluation"
        );
        assert!(ptx.contains("add.f32") && ptx.contains("sub.f32"));
        assert!(!ptx.contains("butterfly computation would go here"));
    }

    /// The f64 per-stage kernel must use the range-reduced polynomial
    /// (PTX has no `sin.approx.f64`), evidenced by `cvt.rni.s64.f64` and
    /// `fma.rn.f64` and the absence of `sin.approx.f64`.
    #[test]
    fn stage_kernel_f64_uses_polynomial_twiddle() {
        let ptx = generate_stage_kernel(
            8192,
            8,
            0,
            4,
            1,
            FftPrecision::Double,
            FftDirection::Forward,
            SmVersion::Sm80,
        )
        .expect("f64 stage kernel");
        assert!(!ptx.contains("sin.approx.f64"), "PTX has no sin.approx.f64");
        assert!(
            ptx.contains("cvt.rni.s64.f64"),
            "expected f64 range reduction"
        );
        assert!(ptx.contains("fma.rn.f64"), "expected f64 polynomial FMAs");
    }
}

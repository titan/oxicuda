//! Batch FFT kernel generator.
//!
//! Optimised for launching many small FFTs (N <= 1024) where each
//! Cooperative Thread Array (CTA / thread block) handles one complete
//! FFT.  This maximises GPU occupancy when the batch count is large.
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{FftError, FftResult};
use crate::kernels::butterfly::{
    PingPongBuffer as ButterflyBuffer, emit_stockham_all_stages, factor_radices,
    load_shared_complex as load_stage_complex, store_shared_complex as store_stage_complex,
};
use crate::ptx_helpers::{ptx_float_type, ptx_type_suffix};
use crate::types::{FftDirection, FftPrecision};

// ---------------------------------------------------------------------------
// Batch FFT kernel generation
// ---------------------------------------------------------------------------

/// Generates a PTX kernel where each thread block computes one complete
/// FFT of size N from a batch.
///
/// The kernel layout:
/// - Grid:  `(batch_count, 1, 1)`
/// - Block: `(block_size, 1, 1)` where `block_size <= N`
///
/// Shared memory holds the Stockham ping-pong buffers for a *single*
/// FFT.  Because each thread block owns its own `smem` allocation and
/// processes exactly one batch row, the butterfly emitter's logical
/// complex indices (`0 .. N` for buffer A, `N .. 2N` for buffer B) can
/// never address another batch row — the shared-memory indexing is
/// confined to one batch row by construction.
///
/// The `direction` kernel parameter is retained in the PTX signature for
/// ABI stability; the transform direction is baked into the twiddle
/// constants at code-generation time.
///
/// # Errors
///
/// Returns [`FftError::PtxGeneration`] if the PTX builder encounters an error.
pub fn generate_batch_fft_kernel(
    n: usize,
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
    let kernel_name = format!("fft_batch_{suffix}_n{n}_b{batch}_{dir_tag}");

    let block_size = select_block_size(n);
    // Shared memory: 2 * N complex elements for ping-pong
    let shared_count = 2 * n * 2; // 2 buffers * N * (re + im)

    let elem_bytes = precision.element_bytes();

    // Stockham radix decomposition for the in-block transform.
    let radices = factor_radices(n);

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("batch_count", PtxType::U32)
        .param("direction", PtxType::U32)
        .shared_mem("smem", float_ty, shared_count)
        .max_threads_per_block(block_size)
        .body(move |b| {
            let fft_direction = direction;
            b.comment(&format!(
                "Batch FFT: N={n}, batch={batch}, block_size={block_size}, \
                 direction={fft_direction:?}"
            ));
            b.comment(&format!("Radices: {radices:?}"));

            // Each block handles one batch element
            let tid = b.thread_id_x();
            let batch_idx = b.block_id_x();

            let _input_ptr = b.load_param_u64("input_ptr");
            let _output_ptr = b.load_param_u64("output_ptr");
            let batch_count = b.load_param_u32("batch_count");
            let _direction = b.load_param_u32("direction");

            // Bounds check: skip if batch_idx >= batch_count
            b.if_lt_u32(batch_idx.clone(), batch_count, |b| {
                b.comment("compute batch offset into input/output arrays");

                // Batch offset = batch_idx * N * complex_size_bytes
                let complex_per_batch = b.alloc_reg(PtxType::U32);
                let total_floats = n * 2;
                b.raw_ptx(&format!("mov.u32 {complex_per_batch}, {total_floats};"));

                let batch_float_offset = b.mul_lo_u32(batch_idx, complex_per_batch);
                let elem_size_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {elem_size_reg}, {elem_bytes};"));
                let batch_byte_offset = b.mul_wide_u32_to_u64(batch_float_offset, elem_size_reg);

                let src = b.add_u64(_input_ptr, batch_byte_offset.clone());
                let dst = b.add_u64(_output_ptr, batch_byte_offset);

                // Load data from global to shared memory
                b.comment("coalesced load from global to shared memory");
                let smem_base = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!("mov.u64 {smem_base}, smem;"));

                let elems_per_thread = total_floats.div_ceil(block_size as usize);
                for e in 0..elems_per_thread {
                    let idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {idx}, {tid}, {elems_per_thread}, {e};"
                    ));
                    let bound = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {bound}, {total_floats};"));
                    b.if_lt_u32(idx.clone(), bound, |b| {
                        let es = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {es}, {elem_bytes};"));
                        let byte_off = b.mul_wide_u32_to_u64(idx, es);
                        let g_addr = b.add_u64(src.clone(), byte_off.clone());
                        let s_addr = b.add_u64(smem_base.clone(), byte_off);

                        match precision {
                            FftPrecision::Single => {
                                let val = b.load_global_f32(g_addr);
                                b.store_shared_f32(s_addr, val);
                            }
                            FftPrecision::Double => {
                                let val = b.alloc_reg(PtxType::F64);
                                b.raw_ptx(&format!("ld.global.f64 {val}, [{g_addr}];"));
                                b.raw_ptx(&format!("st.shared.f64 [{s_addr}], {val};"));
                            }
                        }
                    });
                }

                b.bar_sync(0);

                // ------------------------------------------------------------
                // Stockham auto-sort butterfly stages (real arithmetic).
                //
                // `smem` holds two ping-pong buffers of N complex elements
                // each: buffer A at complex indices 0..N, buffer B at
                // N..2N — all inside *this* block's private allocation, so
                // indexing stays within the batch row.  The cooperative
                // load above filled buffer A.
                //
                // The full transform is emitted (fully unrolled) under a
                // `tid == 0` guard so a single thread owns the shared
                // buffers; the surrounding barriers publish the result to
                // every thread before the cooperative store-back.
                // ------------------------------------------------------------
                b.comment("Stockham FFT stages in shared memory (real butterflies)");
                let buffer_a = ButterflyBuffer {
                    base: smem_base.clone(),
                    elem_index_offset: 0,
                };
                let buffer_b = ButterflyBuffer {
                    base: smem_base.clone(),
                    elem_index_offset: n,
                };
                let one_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {one_reg}, 1;"));
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
                b.comment("coalesced store from shared to global memory");
                for e in 0..elems_per_thread {
                    let idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {idx}, {tid}, {elems_per_thread}, {e};"
                    ));
                    let bound = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {bound}, {total_floats};"));
                    b.if_lt_u32(idx.clone(), bound, |b| {
                        let es = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {es}, {elem_bytes};"));
                        let byte_off = b.mul_wide_u32_to_u64(idx, es);
                        let s_addr = b.add_u64(smem_base.clone(), byte_off.clone());
                        let g_addr = b.add_u64(dst.clone(), byte_off);

                        match precision {
                            FftPrecision::Single => {
                                let val = b.load_shared_f32(s_addr);
                                b.store_global_f32(g_addr, val);
                            }
                            FftPrecision::Double => {
                                let val = b.alloc_reg(PtxType::F64);
                                b.raw_ptx(&format!("ld.shared.f64 {val}, [{s_addr}];"));
                                b.raw_ptx(&format!("st.global.f64 [{g_addr}], {val};"));
                            }
                        }
                    });
                }
            });

            b.ret();
        })
        .build()
        .map_err(FftError::PtxGeneration)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Selects an appropriate block size for a batch FFT kernel.
fn select_block_size(n: usize) -> u32 {
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

/// Returns the shared memory requirement in bytes for a batch FFT kernel.
pub fn batch_fft_shared_bytes(n: usize, precision: FftPrecision) -> usize {
    2 * n * precision.complex_bytes()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_fft_kernel_smoke() {
        let result = generate_batch_fft_kernel(
            64,
            1024,
            FftPrecision::Single,
            FftDirection::Forward,
            SmVersion::Sm80,
        );
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_batch_f32_n64"));
        }
    }

    #[test]
    fn shared_bytes_calculation() {
        assert_eq!(batch_fft_shared_bytes(64, FftPrecision::Single), 1024);
        assert_eq!(batch_fft_shared_bytes(64, FftPrecision::Double), 2048);
    }

    /// The batch FFT kernel must emit real butterfly arithmetic — shared
    /// loads/stores plus multiplies/adds/subs — not bare comments.
    #[test]
    fn batch_fft_emits_real_butterfly_arithmetic() {
        let ptx = generate_batch_fft_kernel(
            64,
            8,
            FftPrecision::Single,
            FftDirection::Forward,
            SmVersion::Sm80,
        )
        .expect("batch kernel generation must succeed");
        assert!(ptx.contains("add.f32") && ptx.contains("sub.f32"));
        assert!(
            ptx.contains("mul.rn.f32") || ptx.contains("fma.rn.f32"),
            "expected real multiplies / FMAs"
        );
        assert!(ptx.contains("st.shared.f32"), "expected shared stores");
        assert!(ptx.contains("ld.shared.f32"), "expected shared loads");
        assert!(!ptx.contains("actual butterfly stages would be emitted here"));
    }

    /// Forward and inverse batch kernels must differ (twiddle sign flip).
    #[test]
    fn batch_fft_forward_inverse_differ() {
        let fwd = generate_batch_fft_kernel(
            32,
            4,
            FftPrecision::Single,
            FftDirection::Forward,
            SmVersion::Sm80,
        )
        .expect("fwd");
        let inv = generate_batch_fft_kernel(
            32,
            4,
            FftPrecision::Single,
            FftDirection::Inverse,
            SmVersion::Sm80,
        )
        .expect("inv");
        assert_ne!(fwd, inv);
    }

    /// The radix factorisation must yield radices whose product is N and
    /// which greedily prefer the larger radices (8, then 4, then 2, ...).
    #[test]
    fn factor_radices_products() {
        for &n in &[2usize, 4, 8, 16, 64, 256, 1024, 24, 48] {
            let radices = factor_radices(n);
            let product: usize = radices.iter().map(|&r| r as usize).product();
            assert_eq!(product, n, "radix product must equal N={n}");
        }
        // Greedy large-radix extraction: 1024 = 8*8*8*2, 64 = 8*8.
        assert_eq!(factor_radices(64), vec![8, 8]);
        assert_eq!(factor_radices(1024), vec![8, 8, 8, 2]);
        assert_eq!(factor_radices(16), vec![8, 2]);
        // Mixed radix: 24 = 8*3.
        assert_eq!(factor_radices(24), vec![8, 3]);
        assert!(factor_radices(0).is_empty());
    }
}

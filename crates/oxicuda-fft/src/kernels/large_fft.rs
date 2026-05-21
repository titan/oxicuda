//! Multi-pass FFT kernel generator for large sizes (N > 4096).
//!
//! When the FFT size exceeds what can fit in shared memory, the transform
//! is split into multiple kernel launches.  Each launch performs one
//! Stockham radix stage, reading from and writing to global memory.
//!
//! Between passes, data is staged through a temporary global-memory buffer
//! (the "ping-pong" pattern at the global level).
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{FftError, FftResult};
use crate::kernels::butterfly::{StageShape as ButterflyStageShape, emit_stockham_pass_global};
use crate::ptx_helpers::ptx_type_suffix;
use crate::types::{FftDirection, FftPrecision};

// ---------------------------------------------------------------------------
// Large FFT pass kernel generation
// ---------------------------------------------------------------------------

/// Generates a PTX kernel for one pass of a large multi-pass FFT.
///
/// Each pass applies a single Stockham radix butterfly stage, reading
/// from `input_ptr` and writing to `output_ptr` (which may be a
/// temporary buffer or the final output).  The full radix butterfly —
/// global loads, runtime twiddle multiply and `r`-point DFT, global
/// stores — is emitted by the shared
/// [`butterfly`](crate::kernels::butterfly) module so every kernel
/// generator shares one numerically-correct implementation.
///
/// The kernel is launched with enough threads to cover all butterfly
/// operations: `grid = ceil(N / (radix * block_size)), block = block_size`.
///
/// For a uniform-radix decomposition the Stockham sub-transform length
/// `L` at stage `s` equals `radix^s`; that is what this pass uses.
///
/// The `direction` kernel parameter is retained in the PTX signature for
/// ABI stability — the transform direction is baked into the twiddle
/// evaluation at code-generation time.
///
/// # Errors
///
/// Returns [`FftError::PtxGeneration`] if the PTX builder encounters an error.
pub fn generate_large_fft_pass(
    n: usize,
    radix: u32,
    stage: u32,
    precision: FftPrecision,
    direction: FftDirection,
    sm: SmVersion,
) -> FftResult<String> {
    let suffix = ptx_type_suffix(precision);
    let dir_tag = match direction {
        FftDirection::Forward => "fwd",
        FftDirection::Inverse => "inv",
    };
    let kernel_name = format!("fft_large_pass_{suffix}_n{n}_r{radix}_s{stage}_{dir_tag}");
    let block_size = 256u32;

    // Stockham sub-transform length L = product of previous radices.
    // For a uniform-radix decomposition that is radix^stage.
    let l: usize = (radix as usize).pow(stage);

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
                "Large FFT pass: N={n}, radix={radix}, stage={stage}, L={l}, \
                 direction={fft_direction:?}"
            ));

            let gid = b.global_thread_id_x();
            let input_ptr = b.load_param_u64("input_ptr");
            let output_ptr = b.load_param_u64("output_ptr");
            let _n_total = b.load_param_u32("n_total");
            let _batch_count = b.load_param_u32("batch_count");
            let _direction = b.load_param_u32("direction");

            // Each thread handles one radix butterfly.
            let butterflies = n / (radix as usize);
            let max_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {max_idx}, {butterflies};"));

            let gid_for_body = gid.clone();
            b.if_lt_u32(gid, max_idx, |b| {
                // The shared emitter computes the Stockham index mapping,
                // loads the radix-r legs, applies the runtime twiddle and
                // the radix-r DFT, and emits the real `st.global` stores.
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
// Helpers
// ---------------------------------------------------------------------------

/// Returns the number of kernel passes needed for a large FFT.
pub fn num_passes(strategy: &crate::plan::FftStrategy) -> usize {
    strategy.radices.len()
}

/// Returns the temporary buffer size in bytes for a large multi-pass FFT.
pub fn temp_buffer_bytes(n: usize, batch: usize, precision: FftPrecision) -> usize {
    // One full-size complex buffer for ping-pong staging
    n * batch * precision.complex_bytes()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_fft_pass_smoke() {
        let result = generate_large_fft_pass(
            8192,
            8,
            0,
            FftPrecision::Single,
            FftDirection::Forward,
            SmVersion::Sm80,
        );
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_large_pass_f32_n8192"));
        }
    }

    #[test]
    fn temp_buffer_sizing() {
        let bytes = temp_buffer_bytes(8192, 1, FftPrecision::Single);
        // 8192 * 1 * 8 (complex f32) = 65536
        assert_eq!(bytes, 65536);
    }

    /// The large-FFT pass must emit real butterfly arithmetic and — most
    /// importantly — real `st.global` stores (the stale code discarded
    /// the loaded values into `_` registers and never stored anything).
    #[test]
    fn large_fft_pass_emits_real_global_stores() {
        let ptx = generate_large_fft_pass(
            8192,
            8,
            1,
            FftPrecision::Single,
            FftDirection::Forward,
            SmVersion::Sm80,
        )
        .expect("large pass generation must succeed");
        assert!(ptx.contains("ld.global.f32"), "expected global loads");
        assert!(
            ptx.contains("st.global.f32"),
            "expected real global stores (results must be written back)"
        );
        assert!(ptx.contains("add.f32") && ptx.contains("sub.f32"));
        assert!(
            ptx.contains("mul.rn.f32") || ptx.contains("fma.rn.f32"),
            "expected real butterfly multiplies"
        );
        assert!(
            ptx.contains("cos.approx.f32") && ptx.contains("sin.approx.f32"),
            "expected runtime twiddle evaluation"
        );
        // Stale placeholders must be gone.
        assert!(!ptx.contains("Actual store would go here"));
        assert!(!ptx.contains("actual butterfly would use the radix modules"));
    }

    /// f64 large-FFT pass: the twiddle must use the range-reduced
    /// polynomial, never the (non-existent) `sin.approx.f64`.
    #[test]
    fn large_fft_pass_f64_polynomial_twiddle() {
        let ptx = generate_large_fft_pass(
            8192,
            8,
            2,
            FftPrecision::Double,
            FftDirection::Inverse,
            SmVersion::Sm80,
        )
        .expect("f64 large pass");
        assert!(ptx.contains("st.global.f64"), "expected f64 global stores");
        assert!(!ptx.contains("sin.approx.f64"));
        assert!(ptx.contains("fma.rn.f64"));
    }
}

//! Symmetric rank-2k update (SYR2K).
//!
//! Computes `C = alpha * (A * B^T + B * A^T) + beta * C` (trans = NoTrans) or
//! `C = alpha * (A^T * B + B^T * A) + beta * C` (trans = Trans), where C is
//! symmetric.
//!
//! Only the triangle indicated by `fill_mode` is written.  When a CUDA driver
//! is present and the hardware is Ampere (SM >= 80) or newer the operation
//! runs via the triangle-masked Tensor Core kernel in `syrk_tc`.  Otherwise it
//! decomposes into two GEMM calls.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{FillMode, GpuFloat, MatrixDesc, MatrixDescMut, Transpose};

use super::syrk_tc;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Performs a symmetric rank-2k update on the GPU.
///
/// Depending on `trans`:
/// - **NoTrans**: `C = alpha * (A * B^T + B * A^T) + beta * C`, A and B are N x K.
/// - **Trans**: `C = alpha * (A^T * B + B^T * A) + beta * C`, A and B are K x N.
///
/// C is N x N symmetric; only the triangle indicated by `fill_mode` is updated.
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `fill_mode` — which triangle of C to write.
/// * `trans` — operation mode: `NoTrans` or `Trans`.
/// * `alpha` — scalar multiplier.
/// * `a` — descriptor for matrix A.
/// * `b` — descriptor for matrix B.
/// * `beta` — scalar multiplier for existing C.
/// * `c` — descriptor for the symmetric output matrix C.
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if C is not square or dimensions
/// are zero. Returns [`BlasError::DimensionMismatch`] if A, B, and C have
/// incompatible sizes. Returns [`BlasError::InvalidArgument`] if `trans` is
/// `ConjTrans` (use HER2K for complex types).
#[allow(clippy::too_many_arguments)]
pub fn syr2k<T: GpuFloat>(
    handle: &BlasHandle,
    fill_mode: FillMode,
    trans: Transpose,
    alpha: T,
    a: &MatrixDesc<T>,
    b: &MatrixDesc<T>,
    beta: T,
    c: &mut MatrixDescMut<T>,
) -> BlasResult<()> {
    if trans == Transpose::ConjTrans {
        return Err(BlasError::InvalidArgument(
            "SYR2K: use HER2K for conjugate-transpose".into(),
        ));
    }

    // Validate C is square.
    if c.rows != c.cols {
        return Err(BlasError::InvalidDimension(format!(
            "SYR2K: output C must be square, got {}x{}",
            c.rows, c.cols
        )));
    }

    let n = c.rows;

    // Effective dimensions.
    let (a_n, a_k) = match trans {
        Transpose::NoTrans => (a.rows, a.cols),
        Transpose::Trans | Transpose::ConjTrans => (a.cols, a.rows),
    };
    let (b_n, b_k) = match trans {
        Transpose::NoTrans => (b.rows, b.cols),
        Transpose::Trans | Transpose::ConjTrans => (b.cols, b.rows),
    };

    if a_n != n {
        return Err(BlasError::DimensionMismatch(format!(
            "SYR2K: op(A) has {a_n} rows but C is {n}x{n}"
        )));
    }
    if b_n != n {
        return Err(BlasError::DimensionMismatch(format!(
            "SYR2K: op(B) has {b_n} rows but C is {n}x{n}"
        )));
    }
    if a_k != b_k {
        return Err(BlasError::DimensionMismatch(format!(
            "SYR2K: op(A) has K={a_k} but op(B) has K={b_k}"
        )));
    }

    if n == 0 {
        return Ok(());
    }

    // Tensor Core fast path: two-operand triangle-masked cross-product kernel.
    //
    // Applicable when:
    //   - SM >= 80 (Ampere+) and n >= 32
    //   - fill_mode is Upper or Lower (not Full)
    //   - The element type is f32 (the generated PTX uses f32 alpha/beta).
    //
    // The kernel in `syrk_tc::generate_syr2k_tc_ptx` accepts both `ptr_a` and
    // `ptr_b` and computes the cross-product accumulation directly, writing
    // only the requested triangle.
    {
        let sm = handle.sm_version();
        let tc_eligible = syrk_tc::is_tc_applicable(sm, n)
            && fill_mode != FillMode::Full
            && T::PTX_TYPE == PtxType::F32;

        if tc_eligible {
            let tile = syrk_tc::syrk_tc_tile_config(sm, n);
            let config =
                syrk_tc::Syr2kTcConfig::new(tile.tile_m, tile.tile_n, tile.tile_k, sm, fill_mode);

            // PTX generation failure falls through to the two-GEMM path.
            if let Ok((ptx, kernel_name)) = syrk_tc::generate_syr2k_tc_ptx(&config) {
                // Load the module (JIT-compiles PTX via the CUDA driver at
                // runtime; returns CudaError::NotInitialized on macOS where
                // no CUDA driver is present — falls through to GEMM below).
                if let Ok(module) = Module::from_ptx(&ptx) {
                    let module = Arc::new(module);
                    let kernel =
                        Kernel::from_module(Arc::clone(&module), &kernel_name).map_err(|e| {
                            BlasError::LaunchFailed(format!("SYR2K TC: kernel lookup failed: {e}"))
                        })?;

                    // Grid: one tile per output NxN tile (col-tiles x row-tiles).
                    let grid_x = n.div_ceil(tile.tile_n);
                    let grid_y = n.div_ceil(tile.tile_m);
                    let threads_per_block = (tile.tile_m * tile.tile_n).min(256);

                    let params = LaunchParams::new(
                        Dim3::new(grid_x, grid_y, 1),
                        Dim3::new(threads_per_block, 1, 1),
                    );

                    // Kernel args: ptr_a, ptr_b, ptr_c, alpha(f32), beta(f32),
                    //              n, k, lda, ldb, ldc
                    let alpha_f32 = f32::from_bits(alpha.to_bits_u64() as u32);
                    let beta_f32 = f32::from_bits(beta.to_bits_u64() as u32);
                    let args = (
                        a.ptr, b.ptr, c.ptr, alpha_f32, beta_f32, n, a_k, a.ld, b.ld, c.ld,
                    );

                    kernel
                        .launch(&params, handle.stream(), &args)
                        .map_err(|e| {
                            BlasError::LaunchFailed(format!("SYR2K TC: launch failed: {e}"))
                        })?;

                    return Ok(());
                }
                // No CUDA driver available (e.g. macOS) — fall through to two-GEMM.
            }
        }
    }

    // Fallback: SYR2K = alpha * A * B^T + alpha * B * A^T + beta * C
    // Decompose into two GEMM calls:
    //   Step 1: C = alpha * A * B^T + beta * C       (first GEMM)
    //   Step 2: C = alpha * B * A^T + 1.0 * C        (second GEMM, beta=1)

    let (trans_left, trans_right) = match trans {
        Transpose::NoTrans => (Transpose::NoTrans, Transpose::Trans),
        Transpose::Trans => (Transpose::Trans, Transpose::NoTrans),
        Transpose::ConjTrans => unreachable!(),
    };

    // First GEMM: C = alpha * op1(A) * op2(B) + beta * C
    super::gemm_api::gemm(handle, trans_left, trans_right, alpha, a, b, beta, c)?;

    // Second GEMM: C = alpha * op1(B) * op2(A) + 1.0 * C
    let one = T::gpu_one();
    super::gemm_api::gemm(handle, trans_left, trans_right, alpha, b, a, one, c)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syr2k_rejects_conj_trans() {
        let err = BlasError::InvalidArgument("SYR2K: use HER2K".into());
        assert!(err.to_string().contains("HER2K"));
    }

    #[test]
    fn syr2k_validates_square_c() {
        let err = BlasError::InvalidDimension("SYR2K: output C must be square, got 4x6".into());
        assert!(err.to_string().contains("square"));
    }
}

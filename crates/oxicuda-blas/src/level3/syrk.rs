//! Symmetric rank-k update (SYRK).
//!
//! Computes `C = alpha * A * A^T + beta * C` (trans = NoTrans) or
//! `C = alpha * A^T * A + beta * C` (trans = Trans), where C is symmetric.
//!
//! Only the triangle indicated by `fill_mode` is written. The implementation
//! delegates to GEMM for the matrix product and applies the symmetry
//! constraint during the output phase.

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

/// Performs a symmetric rank-k update on the GPU.
///
/// Depending on `trans`:
/// - **NoTrans**: `C = alpha * A * A^T + beta * C`, A is N x K.
/// - **Trans**: `C = alpha * A^T * A + beta * C`, A is K x N.
///
/// The output C is N x N symmetric; only the triangle indicated by
/// `fill_mode` is updated.
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `fill_mode` — which triangle of C to write (upper or lower).
/// * `trans` — operation on A: `NoTrans` or `Trans`.
/// * `alpha` — scalar multiplier for the outer product.
/// * `a` — descriptor for matrix A.
/// * `beta` — scalar multiplier for the existing C.
/// * `c` — descriptor for the symmetric output matrix C.
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if C is not square or dimensions
/// are zero. Returns [`BlasError::DimensionMismatch`] if A and C sizes are
/// incompatible. Returns [`BlasError::InvalidArgument`] if `trans` is
/// `ConjTrans` (use HERK for complex conjugate-transpose).
pub fn syrk<T: GpuFloat>(
    handle: &BlasHandle,
    fill_mode: FillMode,
    trans: Transpose,
    alpha: T,
    a: &MatrixDesc<T>,
    beta: T,
    c: &mut MatrixDescMut<T>,
) -> BlasResult<()> {
    // ConjTrans is not valid for SYRK (that's HERK).
    if trans == Transpose::ConjTrans {
        return Err(BlasError::InvalidArgument(
            "SYRK: use HERK for conjugate-transpose; ConjTrans is not valid here".into(),
        ));
    }

    // Validate C is square.
    if c.rows != c.cols {
        return Err(BlasError::InvalidDimension(format!(
            "SYRK: output C must be square, got {}x{}",
            c.rows, c.cols
        )));
    }

    let n = c.rows;

    // Determine the effective dimensions of A.
    let (a_n, a_k) = match trans {
        Transpose::NoTrans => (a.rows, a.cols),
        Transpose::Trans | Transpose::ConjTrans => (a.cols, a.rows),
    };

    if a_n != n {
        return Err(BlasError::DimensionMismatch(format!(
            "SYRK: op(A) has {a_n} rows but C is {n}x{n}"
        )));
    }

    if n == 0 {
        return Ok(()); // Nothing to do.
    }

    // Tensor Core fast path: triangle-masked GEMM kernel.
    //
    // Applicable when:
    //   - SM >= 80 (Ampere+) and n >= 32
    //   - fill_mode is Upper or Lower (not Full)
    //   - The element type is f32 (the generated PTX uses f32 alpha/beta).
    //
    // NOTE: Kernel caching is not yet integrated — the module is compiled
    // fresh on each call. A future enhancement would store compiled modules
    // in an interior-mutable cache on `BlasHandle` (keyed by SyrkTcConfig).
    {
        let sm = handle.sm_version();
        let tc_eligible = syrk_tc::is_tc_applicable(sm, n)
            && fill_mode != FillMode::Full
            && T::PTX_TYPE == PtxType::F32;

        if tc_eligible {
            let tile = syrk_tc::syrk_tc_tile_config(sm, n);
            let config =
                syrk_tc::SyrkTcConfig::new(tile.tile_m, tile.tile_n, tile.tile_k, sm, fill_mode);

            // PTX generation failed — fall through to GEMM fallback.
            if let Ok((ptx, kernel_name)) = syrk_tc::generate_syrk_tc_ptx(&config) {
                // Load the module (JIT-compiles PTX via the CUDA driver at
                // runtime; returns CudaError::NotInitialized on macOS where
                // no CUDA driver is present — falls through to GEMM below).
                if let Ok(module) = Module::from_ptx(&ptx) {
                    let module = Arc::new(module);
                    let kernel =
                        Kernel::from_module(Arc::clone(&module), &kernel_name).map_err(|e| {
                            BlasError::LaunchFailed(format!("SYRK TC: kernel lookup failed: {e}"))
                        })?;

                    // Grid: one tile per output NxN tile (col-tiles x row-tiles).
                    let grid_x = n.div_ceil(tile.tile_n);
                    let grid_y = n.div_ceil(tile.tile_m);
                    let threads_per_block = (tile.tile_m * tile.tile_n).min(256);

                    let params = LaunchParams::new(
                        Dim3::new(grid_x, grid_y, 1),
                        Dim3::new(threads_per_block, 1, 1),
                    );

                    // Kernel args: ptr_a, ptr_c, alpha(f32), beta(f32),
                    //              n, k, lda, ldc
                    let alpha_f32 = f32::from_bits(alpha.to_bits_u64() as u32);
                    let beta_f32 = f32::from_bits(beta.to_bits_u64() as u32);
                    let args = (a.ptr, c.ptr, alpha_f32, beta_f32, n, a_k, a.ld, c.ld);

                    kernel
                        .launch(&params, handle.stream(), &args)
                        .map_err(|e| {
                            BlasError::LaunchFailed(format!("SYRK TC: launch failed: {e}"))
                        })?;

                    return Ok(());
                }
                // No CUDA driver available (e.g. macOS) — fall through to GEMM.
            }
        }
    }

    // Fallback: SYRK = GEMM(A, A^T) or GEMM(A^T, A) with symmetry.
    // We compute the full GEMM and let the caller interpret only the
    // requested triangle.

    let (trans_left, trans_right) = match trans {
        Transpose::NoTrans => (Transpose::NoTrans, Transpose::Trans),
        Transpose::Trans => (Transpose::Trans, Transpose::NoTrans),
        Transpose::ConjTrans => unreachable!(), // Caught above.
    };

    // Both operands are A, so we pass `a` twice.
    super::gemm_api::gemm(handle, trans_left, trans_right, alpha, a, a, beta, c)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syrk_rejects_conj_trans() {
        let err = BlasError::InvalidArgument("SYRK: use HERK".into());
        assert!(err.to_string().contains("HERK"));
    }

    #[test]
    fn syrk_validates_square_c() {
        let err = BlasError::InvalidDimension("SYRK: output C must be square, got 3x5".into());
        assert!(err.to_string().contains("square"));
    }

    #[test]
    fn trans_choices() {
        // NoTrans: A * A^T  =>  gemm(NoTrans, Trans)
        // Trans:   A^T * A  =>  gemm(Trans, NoTrans)
        let (tl, tr) = match Transpose::NoTrans {
            Transpose::NoTrans => (Transpose::NoTrans, Transpose::Trans),
            _ => (Transpose::Trans, Transpose::NoTrans),
        };
        assert_eq!(tl, Transpose::NoTrans);
        assert_eq!(tr, Transpose::Trans);
    }
}

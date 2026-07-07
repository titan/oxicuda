//! Symmetric rank-k update (SYRK).
//!
//! Computes `C = alpha * A * A^T + beta * C` (trans = NoTrans) or
//! `C = alpha * A^T * A + beta * C` (trans = Trans), where C is symmetric.
//!
//! Only the triangle indicated by `fill_mode` is written. The implementation
//! delegates to GEMM for the matrix product and applies the symmetry
//! constraint during the output phase.

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{FillMode, GpuFloat, MatrixDesc, MatrixDescMut, Transpose};

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

    let _ = a_k; // Inner dimension is validated above; GEMM re-derives it.

    // SYRK is computed as a single triangle-masked GEMM:
    //   NoTrans: C = alpha * A * A^T + beta * C   (gemm NoTrans, Trans)
    //   Trans:   C = alpha * A^T * A + beta * C   (gemm Trans, NoTrans)
    //
    // The former `syrk_tc` tensor-core fast path was an incomplete placeholder
    // (it accumulated only the k = 0 term and was f32-only), so it is not used.
    // The GEMM computes the full dot product over K; the triangle mask writes
    // only `fill_mode`, leaving the opposite triangle of C untouched (matching
    // reference-BLAS SYRK semantics, which reference only one triangle).
    //
    // ConjTrans was rejected above, so `trans` is NoTrans or Trans here.
    let (trans_left, trans_right) = if trans == Transpose::NoTrans {
        (Transpose::NoTrans, Transpose::Trans)
    } else {
        (Transpose::Trans, Transpose::NoTrans)
    };

    let mask = match fill_mode {
        FillMode::Upper | FillMode::Lower => Some(fill_mode),
        FillMode::Full => None,
    };

    // Both operands are A, so we pass `a` twice (A and B may alias in GEMM).
    super::gemm_api::gemm_tri(handle, trans_left, trans_right, alpha, a, a, beta, c, mask)
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

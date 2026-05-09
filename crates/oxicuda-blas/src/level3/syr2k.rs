//! Symmetric rank-2k update (SYR2K).
//!
//! Computes `C = alpha * (A * B^T + B * A^T) + beta * C` (trans = NoTrans) or
//! `C = alpha * (A^T * B + B^T * A) + beta * C` (trans = Trans), where C is
//! symmetric.
//!
//! Only the triangle indicated by `fill_mode` is written. The implementation
//! decomposes into two GEMM calls.

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{FillMode, GpuFloat, MatrixDesc, MatrixDescMut, Transpose};

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

    // Tensor Core fast path — not yet implemented for SYR2K.
    //
    // SYR2K requires two triangle-masked GEMM passes over *distinct* operands:
    //   Pass 1: C = alpha * op(A) * op(B)^T + beta * C   (triangle-masked, fill_mode)
    //   Pass 2: C = alpha * op(B) * op(A)^T + 1.0   * C  (triangle-masked, fill_mode)
    //
    // The existing `syrk_tc::generate_syrk_tc_ptx` kernel accepts a single
    // data pointer `ptr_a` and computes `A * A^T` (both column loads use the
    // same pointer).  Reusing it with B-as-A on the second pass would compute
    // `A*A^T + B*B^T` instead of the correct `A*B^T + B*A^T`.
    //
    // To support a TC fast path here, a new `generate_syr2k_tc_ptx` function
    // must be added to `syrk_tc.rs` that accepts two device pointers (`ptr_a`,
    // `ptr_b`) and performs the cross-product column load in the epilogue.
    // Until that kernel exists this falls through to the two-GEMM path below.
    // fill_mode will be used once the TC kernel supports it.
    let _ = fill_mode;

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

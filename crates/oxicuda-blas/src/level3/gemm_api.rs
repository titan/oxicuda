//! Public GEMM API.
//!
//! Provides the [`gemm`] function — the primary entry point for performing
//! general matrix multiplication on the GPU:
//!
//! `C = alpha * op(A) * op(B) + beta * C`
//!
//! The function validates dimensions, constructs a [`GemmProblem`], and
//! delegates to the [`GemmDispatcher`](super::gemm::dispatch::GemmDispatcher) for kernel selection and launch.

use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{FillMode, GpuFloat, Layout, MatrixDesc, MatrixDescMut, Transpose};

use super::gemm::dispatch::GemmProblem;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Performs a general matrix multiplication on the GPU.
///
/// Computes `C = alpha * op(A) * op(B) + beta * C`, where `op(X)` is `X`,
/// `X^T`, or `X^H` depending on the transpose flag.
///
/// # Type parameters
///
/// * `T` — element type (must implement [`GpuFloat`]).
///
/// # Arguments
///
/// * `handle` — the BLAS handle providing context, stream, and SM version.
/// * `trans_a` — transpose mode for matrix A.
/// * `trans_b` — transpose mode for matrix B.
/// * `alpha` — scaling factor for the product `op(A) * op(B)`.
/// * `a` — descriptor for input matrix A.
/// * `b` — descriptor for input matrix B.
/// * `beta` — scaling factor for the existing contents of C.
/// * `c` — descriptor for the output matrix C (read-write).
///
/// # Dimension rules
///
/// After applying the transpose modes:
/// - `op(A)` is M x K
/// - `op(B)` is K x N
/// - `C` must be M x N
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if any dimension is zero.
/// Returns [`BlasError::DimensionMismatch`] if the inner dimensions of
/// `op(A)` and `op(B)` do not agree, or if C does not have the right shape.
/// Returns other [`BlasError`] variants on PTX generation or launch failure.
///
/// # Aliasing
///
/// `C` must not overlap either input `A` or `B` in device memory: the kernel
/// reads `A`/`B` while concurrently writing `C`, so an overlap yields a
/// read/write race and undefined results. Passing a `C` whose storage
/// interval intersects `A`'s or `B`'s returns
/// [`BlasError::InvalidArgument`]. `A` and `B` may safely alias each other
/// (they are read-only), which SYRK-style callers rely on.
///
/// # Example
///
/// ```rust,no_run
/// # use oxicuda_blas::level3::gemm_api::gemm;
/// # use oxicuda_blas::handle::BlasHandle;
/// # use oxicuda_blas::types::*;
/// # fn main() -> Result<(), oxicuda_blas::error::BlasError> {
/// # let handle: BlasHandle = unimplemented!();
/// # let a: MatrixDesc<f32> = unimplemented!();
/// # let b: MatrixDesc<f32> = unimplemented!();
/// # let mut c: MatrixDescMut<f32> = unimplemented!();
/// gemm(&handle, Transpose::NoTrans, Transpose::NoTrans, 1.0f32, &a, &b, 0.0f32, &mut c)?;
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: GpuFloat>(
    handle: &BlasHandle,
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: T,
    a: &MatrixDesc<T>,
    b: &MatrixDesc<T>,
    beta: T,
    c: &mut MatrixDescMut<T>,
) -> BlasResult<()> {
    gemm_impl(handle, trans_a, trans_b, alpha, a, b, beta, c, None)
}

/// Triangle-masked GEMM: identical to [`gemm`] but writes only the requested
/// triangle of `C`, leaving the opposite triangle byte-for-byte unchanged.
///
/// This is the correctness primitive behind SYRK / SYR2K: the symmetric
/// product is computed with a full dot product over `K`, but the store is
/// skipped for elements outside `fill_mode`, so the untouched off-triangle is
/// preserved exactly (matching reference BLAS semantics).
///
/// `fill_mode` of `None` or `Some(FillMode::Full)` behaves exactly like
/// [`gemm`] (a full write).
///
/// # Errors
///
/// Same as [`gemm`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_tri<T: GpuFloat>(
    handle: &BlasHandle,
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: T,
    a: &MatrixDesc<T>,
    b: &MatrixDesc<T>,
    beta: T,
    c: &mut MatrixDescMut<T>,
    fill_mode: Option<FillMode>,
) -> BlasResult<()> {
    gemm_impl(handle, trans_a, trans_b, alpha, a, b, beta, c, fill_mode)
}

/// Shared implementation for [`gemm`] and [`gemm_tri`].
#[allow(clippy::too_many_arguments)]
fn gemm_impl<T: GpuFloat>(
    handle: &BlasHandle,
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: T,
    a: &MatrixDesc<T>,
    b: &MatrixDesc<T>,
    beta: T,
    c: &mut MatrixDescMut<T>,
    fill_mode: Option<FillMode>,
) -> BlasResult<()> {
    // Extract effective dimensions after transpose.
    let (m, k_a) = a.effective_dims(trans_a);
    let (k_b, n) = b.effective_dims(trans_b);

    // Validate non-zero dimensions.
    if m == 0 || n == 0 || k_a == 0 {
        return Err(BlasError::InvalidDimension(
            "GEMM dimensions must be non-zero".into(),
        ));
    }

    // Validate inner dimension agreement.
    if k_a != k_b {
        return Err(BlasError::DimensionMismatch(format!(
            "inner dimensions of op(A) ({k_a}) and op(B) ({k_b}) do not match"
        )));
    }
    let k = k_a;

    // Validate output matrix dimensions.
    if c.rows != m || c.cols != n {
        return Err(BlasError::DimensionMismatch(format!(
            "C is {}x{} but GEMM produces {}x{}",
            c.rows, c.cols, m, n
        )));
    }

    // The dispatcher derives leading dimensions from m/n/k and generates tight
    // row-major kernels — it ignores each descriptor's `layout`/`ld`. A
    // column-major or ld-padded operand would therefore be silently
    // mis-addressed, so reject anything that is not tightly packed row-major.
    for (layout, ld, cols, name) in [
        (a.layout, a.ld, a.cols, "A"),
        (b.layout, b.ld, b.cols, "B"),
        (c.layout, c.ld, c.cols, "C"),
    ] {
        if layout != Layout::RowMajor {
            return Err(BlasError::InvalidArgument(format!(
                "GEMM requires RowMajor operands; {name} is ColMajor"
            )));
        }
        if ld != cols {
            return Err(BlasError::InvalidArgument(format!(
                "GEMM requires tightly packed operands; {name}.ld ({ld}) != {name}.cols ({cols})"
            )));
        }
    }

    // Reject C overlapping either input. The kernel reads A/B while writing C,
    // so an overlap is a device-side data race. A-vs-B overlap is *not*
    // checked: both are read-only and SYRK/SYR2K legitimately pass the same
    // descriptor as A and B.
    if buffers_overlap(c.ptr, c.storage_bytes(), a.ptr, a.storage_bytes())
        || buffers_overlap(c.ptr, c.storage_bytes(), b.ptr, b.storage_bytes())
    {
        return Err(BlasError::InvalidArgument(
            "C must not overlap A/B in GEMM".into(),
        ));
    }

    // Build the problem description.
    let problem = GemmProblem {
        m,
        n,
        k,
        trans_a,
        trans_b,
        input_type: T::PTX_TYPE,
        output_type: accumulator_ptx_type::<T>(),
        math_mode: handle.math_mode(),
    };

    // Reuse the handle-owned dispatcher so its compiled-kernel cache persists
    // across calls (a fresh dispatcher would re-JIT every GEMM).
    let dispatcher = handle.gemm_dispatcher();

    // Convert scalar arguments to bit representation for the kernel.
    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();

    dispatcher.dispatch(
        &problem,
        a.ptr,
        b.ptr,
        c.ptr,
        alpha_bits,
        beta_bits,
        fill_mode,
        handle.stream(),
    )
}

/// Returns the PTX type of the accumulator for type `T`.
///
/// For half-precision types the accumulator is F32; for F32 it's F32;
/// for F64 it's F64.
fn accumulator_ptx_type<T: GpuFloat>() -> PtxType {
    <T::Accumulator as GpuFloat>::PTX_TYPE
}

/// Returns `true` if the two device byte-intervals `[p, p + p_len)` and
/// `[q, q + q_len)` intersect. A zero-length interval never overlaps.
fn buffers_overlap(p: u64, p_len: usize, q: u64, q_len: usize) -> bool {
    if p_len == 0 || q_len == 0 {
        return false;
    }
    let p_end = p.saturating_add(p_len as u64);
    let q_end = q.saturating_add(q_len as u64);
    p < q_end && q < p_end
}

// ---------------------------------------------------------------------------
// Dimension helpers (useful for callers)
// ---------------------------------------------------------------------------

/// Computes the expected output dimensions for a GEMM operation.
///
/// Returns `(M, N)` — the dimensions of the output matrix C after
/// applying the transpose modes to A (rows x K) and B (K x cols).
pub fn gemm_output_dims(
    a_rows: u32,
    a_cols: u32,
    trans_a: Transpose,
    b_rows: u32,
    b_cols: u32,
    trans_b: Transpose,
) -> (u32, u32) {
    let m = match trans_a {
        Transpose::NoTrans => a_rows,
        Transpose::Trans | Transpose::ConjTrans => a_cols,
    };
    let n = match trans_b {
        Transpose::NoTrans => b_cols,
        Transpose::Trans | Transpose::ConjTrans => b_rows,
    };
    (m, n)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_dims_no_trans() {
        let (m, n) = gemm_output_dims(128, 64, Transpose::NoTrans, 64, 256, Transpose::NoTrans);
        assert_eq!((m, n), (128, 256));
    }

    #[test]
    fn output_dims_trans_a() {
        let (m, n) = gemm_output_dims(64, 128, Transpose::Trans, 128, 256, Transpose::NoTrans);
        assert_eq!((m, n), (128, 256));
    }

    #[test]
    fn output_dims_trans_b() {
        let (m, n) = gemm_output_dims(128, 64, Transpose::NoTrans, 256, 64, Transpose::Trans);
        assert_eq!((m, n), (128, 256));
    }

    #[test]
    fn output_dims_both_trans() {
        let (m, n) = gemm_output_dims(64, 128, Transpose::Trans, 256, 128, Transpose::Trans);
        assert_eq!((m, n), (128, 256));
    }

    #[test]
    fn accumulator_ptx_type_f32() {
        assert_eq!(accumulator_ptx_type::<f32>(), PtxType::F32);
    }

    #[test]
    fn accumulator_ptx_type_f64() {
        assert_eq!(accumulator_ptx_type::<f64>(), PtxType::F64);
    }

    #[test]
    fn overlap_detects_intersection() {
        // [1000, 1400) vs [1200, 1600) overlap.
        assert!(buffers_overlap(1000, 400, 1200, 400));
        // Adjacent, non-overlapping: [1000, 1400) vs [1400, 1800).
        assert!(!buffers_overlap(1000, 400, 1400, 400));
        // Disjoint.
        assert!(!buffers_overlap(1000, 100, 5000, 100));
        // Identical intervals overlap (A-vs-B read/read is allowed elsewhere,
        // but C-vs-A with the same pointer must be rejected).
        assert!(buffers_overlap(2048, 256, 2048, 256));
        // Zero-length never overlaps.
        assert!(!buffers_overlap(2048, 0, 2048, 256));
    }
}

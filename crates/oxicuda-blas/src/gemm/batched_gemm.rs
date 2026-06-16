//! Batched single-precision GEMM (CPU reference).
//!
//! Performs a batch of independent general matrix–matrix multiplications in one
//! call — the workhorse of multi-head attention and grouped convolutions, where
//! many small matmuls dominate. For each batch index `i`:
//!
//! ```text
//! C_i = alpha · A_i · B_i + beta · C_i
//! ```
//!
//! with row-major operands `A_i ∈ ℝ^{m×k}`, `B_i ∈ ℝ^{k×n}`, `C_i ∈ ℝ^{m×n}`,
//! all laid out contiguously and back-to-back in the flat slices (stride
//! `m·k`, `k·n`, `m·n` respectively). This is the host reference that the
//! batched Tensor-Core / SIMT GPU kernels in [`crate::batched`] are validated
//! against.

use crate::error::{BlasError, BlasResult};

/// Batched SGEMM: `C_i = alpha · A_i · B_i + beta · C_i` for `i in 0..batch`.
///
/// * `a` — `batch · m · k` elements, each `A_i` row-major `[m × k]`.
/// * `b` — `batch · k · n` elements, each `B_i` row-major `[k × n]`.
/// * `c` — `batch · m · n` elements (in/out), each `C_i` row-major `[m × n]`.
///
/// # Errors
///
/// * [`BlasError::InvalidDimension`] if `batch`, `m`, `n`, or `k` is `0`.
/// * [`BlasError::DimensionMismatch`] if any slice length disagrees with the
///   declared `batch × … ` shape.
// BLAS GEMM signatures inherently carry many scalar/array arguments; this
// mirrors the cuBLAS `cublasSgemmBatched` parameter list and the existing
// `crate::batched::batched_gemm` convention.
#[allow(clippy::too_many_arguments)]
pub fn batched_sgemm(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> BlasResult<()> {
    if batch == 0 {
        return Err(BlasError::InvalidDimension(
            "batched_sgemm: batch is 0".into(),
        ));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err(BlasError::InvalidDimension(format!(
            "batched_sgemm: m, n, k must all be ≥ 1 (got m={m}, n={n}, k={k})"
        )));
    }
    if a.len() != batch * m * k {
        return Err(BlasError::DimensionMismatch(format!(
            "batched_sgemm: A has {} elements, expected batch*m*k = {}",
            a.len(),
            batch * m * k
        )));
    }
    if b.len() != batch * k * n {
        return Err(BlasError::DimensionMismatch(format!(
            "batched_sgemm: B has {} elements, expected batch*k*n = {}",
            b.len(),
            batch * k * n
        )));
    }
    if c.len() != batch * m * n {
        return Err(BlasError::DimensionMismatch(format!(
            "batched_sgemm: C has {} elements, expected batch*m*n = {}",
            c.len(),
            batch * m * n
        )));
    }

    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    for batch_idx in 0..batch {
        let a_i = &a[batch_idx * a_stride..batch_idx * a_stride + a_stride];
        let b_i = &b[batch_idx * b_stride..batch_idx * b_stride + b_stride];
        let c_i = &mut c[batch_idx * c_stride..batch_idx * c_stride + c_stride];

        // C_i = alpha · A_i · B_i + beta · C_i.
        // When beta == 0 the prior contents of C are *not* referenced (BLAS
        // convention) so that uninitialised / NaN buffers are overwritten.
        for row in 0..m {
            let a_row = &a_i[row * k..row * k + k];
            for col in 0..n {
                let mut acc = 0.0_f32;
                for (p, &a_val) in a_row.iter().enumerate() {
                    acc += a_val * b_i[p * n + col];
                }
                let dst = &mut c_i[row * n + col];
                *dst = if beta == 0.0 {
                    alpha * acc
                } else {
                    alpha * acc + beta * *dst
                };
            }
        }
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Single un-batched reference SGEMM for cross-checking.
    fn ref_gemm(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc += a[row * k + p] * b[p * n + col];
                }
                c[row * n + col] = alpha * acc;
            }
        }
        c
    }

    #[test]
    fn batch_1_matches_gemm() {
        let (m, n, k) = (2, 3, 4);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
        let mut c = vec![0.0_f32; m * n];
        batched_sgemm(1, m, n, k, 1.0, &a, &b, 0.0, &mut c).expect("gemm");
        let expected = ref_gemm(m, n, k, 1.0, &a, &b);
        for (got, exp) in c.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn batch_n_independent() {
        // Two batches with different data must not cross-contaminate.
        let (m, n, k) = (2, 2, 2);
        let a0 = [1.0_f32, 0.0, 0.0, 1.0]; // identity
        let a1 = [2.0_f32, 0.0, 0.0, 2.0]; // 2·identity
        let b0 = [5.0_f32, 6.0, 7.0, 8.0];
        let b1 = [1.0_f32, 1.0, 1.0, 1.0];
        let a: Vec<f32> = a0.iter().chain(a1.iter()).copied().collect();
        let b: Vec<f32> = b0.iter().chain(b1.iter()).copied().collect();
        let mut c = vec![0.0_f32; 2 * m * n];
        batched_sgemm(2, m, n, k, 1.0, &a, &b, 0.0, &mut c).expect("gemm");
        // Batch 0: I·B0 = B0.
        assert_eq!(&c[0..4], &[5.0, 6.0, 7.0, 8.0]);
        // Batch 1: 2I·B1 = 2·B1.
        assert_eq!(&c[4..8], &[2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn output_shape() {
        let (batch, m, n, k) = (3, 4, 5, 2);
        let a = vec![1.0_f32; batch * m * k];
        let b = vec![1.0_f32; batch * k * n];
        let mut c = vec![0.0_f32; batch * m * n];
        batched_sgemm(batch, m, n, k, 1.0, &a, &b, 0.0, &mut c).expect("gemm");
        assert_eq!(c.len(), batch * m * n);
        // Each entry is sum of k ones = k.
        for &v in &c {
            assert!((v - k as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn alpha_0() {
        // alpha=0, beta=1 → C unchanged.
        let (m, n, k) = (2, 2, 2);
        let a = vec![3.0_f32; m * k];
        let b = vec![4.0_f32; k * n];
        let mut c = vec![9.0_f32; m * n];
        batched_sgemm(1, m, n, k, 0.0, &a, &b, 1.0, &mut c).expect("gemm");
        for &v in &c {
            assert!((v - 9.0).abs() < 1e-6);
        }
    }

    #[test]
    fn beta_0() {
        // beta=0 → prior C overwritten regardless of its contents (no NaN leak).
        let (m, n, k) = (2, 2, 2);
        let a = vec![1.0_f32; m * k];
        let b = vec![1.0_f32; k * n];
        let mut c = vec![f32::NAN; m * n];
        batched_sgemm(1, m, n, k, 1.0, &a, &b, 0.0, &mut c).expect("gemm");
        for &v in &c {
            assert!(v.is_finite(), "beta=0 must overwrite NaN, got {v}");
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn identity_batch() {
        // A = I, alpha=1, beta=0 → C = B.
        let (m, n, k) = (3, 3, 3);
        let mut a = vec![0.0_f32; m * k];
        for i in 0..m {
            a[i * k + i] = 1.0;
        }
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let mut c = vec![0.0_f32; m * n];
        batched_sgemm(1, m, n, k, 1.0, &a, &b, 0.0, &mut c).expect("gemm");
        for (got, exp) in c.iter().zip(b.iter()) {
            assert!((got - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn dim_mismatch_error() {
        let (batch, m, n, k) = (2, 2, 2, 2);
        let a = vec![1.0_f32; batch * m * k];
        let b = vec![1.0_f32; batch * k * n - 1]; // wrong
        let mut c = vec![0.0_f32; batch * m * n];
        let err = batched_sgemm(batch, m, n, k, 1.0, &a, &b, 0.0, &mut c);
        assert!(matches!(err, Err(BlasError::DimensionMismatch(_))));
    }

    #[test]
    fn batch_0_error() {
        let mut c = vec![0.0_f32; 4];
        let err = batched_sgemm(0, 2, 2, 2, 1.0, &[], &[], 0.0, &mut c);
        assert!(matches!(err, Err(BlasError::InvalidDimension(_))));
    }

    #[test]
    fn output_finite() {
        let (batch, m, n, k) = (4, 3, 3, 5);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32).cos()).collect();
        let mut c = vec![0.5_f32; batch * m * n];
        batched_sgemm(batch, m, n, k, 1.5, &a, &b, -0.5, &mut c).expect("gemm");
        for &v in &c {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn alpha_beta_combination() {
        // C = 2·(A·B) + 3·C_init, single 1×1 case for hand calc.
        // A=[2], B=[5] → A·B=10, alpha·=20, +3·C_init(=1)=3 → 23.
        let mut c = vec![1.0_f32];
        batched_sgemm(1, 1, 1, 1, 2.0, &[2.0], &[5.0], 3.0, &mut c).expect("gemm");
        assert!((c[0] - 23.0).abs() < 1e-5, "got {}", c[0]);
    }

    #[test]
    fn zero_dim_error() {
        let mut c = vec![0.0_f32; 0];
        let err = batched_sgemm(1, 0, 2, 2, 1.0, &[], &[], 0.0, &mut c);
        assert!(matches!(err, Err(BlasError::InvalidDimension(_))));
    }
}

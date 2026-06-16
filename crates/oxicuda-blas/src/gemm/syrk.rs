//! Symmetric rank-k update (SSYRK, CPU reference).
//!
//! Computes the BLAS Level-3 symmetric rank-k update
//!
//! ```text
//! C = alpha · A · Aᵀ + beta · C
//! ```
//!
//! where `A ∈ ℝ^{n×k}` (row-major), `C ∈ ℝ^{n×n}` is symmetric, and only one
//! triangle of `C` (upper or lower) is referenced and written — the other
//! triangle is assumed to mirror it. This is the building block used by
//! Cholesky factorization to form `Aᵀ A` Gram matrices, and the host reference
//! the Tensor-Core SYRK kernels in [`crate::level3`] are validated against.
//!
//! Because `A · Aᵀ` is symmetric positive semidefinite, the updated diagonal
//! (with `beta = 0`, `alpha ≥ 0`) is always non-negative — a property the tests
//! assert explicitly.

use crate::error::{BlasError, BlasResult};

/// Symmetric rank-k update: `C = alpha · A · Aᵀ + beta · C`.
///
/// * `a` — `n · k` elements, row-major `[n × k]`.
/// * `c` — `n · n` elements (in/out), row-major `[n × n]`; only the triangle
///   selected by `upper` is read and written.
/// * `upper` — `true` updates the upper triangle (`j ≥ i`), `false` the lower
///   (`j ≤ i`).
///
/// # Errors
///
/// * [`BlasError::InvalidDimension`] if `n == 0` or `k == 0`.
/// * [`BlasError::DimensionMismatch`] if `a.len() != n*k` or `c.len() != n*n`.
pub fn ssyrk(
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    beta: f32,
    c: &mut [f32],
    upper: bool,
) -> BlasResult<()> {
    if n == 0 || k == 0 {
        return Err(BlasError::InvalidDimension(format!(
            "ssyrk: n and k must be ≥ 1 (got n={n}, k={k})"
        )));
    }
    if a.len() != n * k {
        return Err(BlasError::DimensionMismatch(format!(
            "ssyrk: A has {} elements, expected n*k = {}",
            a.len(),
            n * k
        )));
    }
    if c.len() != n * n {
        return Err(BlasError::DimensionMismatch(format!(
            "ssyrk: C has {} elements, expected n*n = {}",
            c.len(),
            n * n
        )));
    }

    for i in 0..n {
        // Triangle bounds: upper updates columns j in [i, n), lower j in [0, i].
        let (j_lo, j_hi) = if upper { (i, n) } else { (0, i + 1) };
        let a_i = &a[i * k..i * k + k];
        for j in j_lo..j_hi {
            let a_j = &a[j * k..j * k + k];
            // (A·Aᵀ)_{ij} = ⟨row_i(A), row_j(A)⟩.
            let mut acc = 0.0_f32;
            for (&ai, &aj) in a_i.iter().zip(a_j.iter()) {
                acc += ai * aj;
            }
            // beta == 0 must not reference C (so NaN/uninitialised is cleared).
            let dst = &mut c[i * n + j];
            *dst = if beta == 0.0 {
                alpha * acc
            } else {
                alpha * acc + beta * *dst
            };
        }
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the full dense `alpha·A·Aᵀ` for cross-checking (both triangles).
    fn full_aat(n: usize, k: usize, alpha: f32, a: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc += a[i * k + p] * a[j * k + p];
                }
                out[i * n + j] = alpha * acc;
            }
        }
        out
    }

    #[test]
    fn syrk_symmetric_result() {
        // Mirror the computed triangle and confirm symmetry of A·Aᵀ.
        let (n, k) = (3, 2);
        let a: Vec<f32> = (0..n * k).map(|i| (i as f32) + 1.0).collect();
        let mut c = vec![0.0_f32; n * n];
        ssyrk(n, k, 1.0, &a, 0.0, &mut c, true).expect("syrk");
        let full = full_aat(n, k, 1.0, &a);
        for i in 0..n {
            for j in i..n {
                assert!((c[i * n + j] - full[i * n + j]).abs() < 1e-4);
                // Symmetric counterpart in the reference.
                assert!((full[i * n + j] - full[j * n + i]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn alpha_1_beta_0() {
        let (n, k) = (2, 3);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let mut c = vec![0.0_f32; n * n];
        ssyrk(n, k, 1.0, &a, 0.0, &mut c, true).expect("syrk");
        // (A·Aᵀ)_00 = 1+4+9 = 14; _01 = 4+10+18 = 32; _11 = 16+25+36 = 77.
        assert!((c[0] - 14.0).abs() < 1e-4);
        assert!((c[1] - 32.0).abs() < 1e-4);
        assert!((c[3] - 77.0).abs() < 1e-4);
    }

    #[test]
    fn identity_a() {
        // A = I_n (k = n) → A·Aᵀ = I.
        let n = 3;
        let mut a = vec![0.0_f32; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let mut c = vec![0.0_f32; n * n];
        ssyrk(n, n, 1.0, &a, 0.0, &mut c, true).expect("syrk");
        for i in 0..n {
            assert!((c[i * n + i] - 1.0).abs() < 1e-5);
            for j in (i + 1)..n {
                assert!(c[i * n + j].abs() < 1e-5);
            }
        }
    }

    #[test]
    fn upper_vs_lower() {
        // Lower-triangle update leaves the strict upper triangle untouched.
        let (n, k) = (3, 2);
        let a: Vec<f32> = (0..n * k).map(|i| (i as f32) + 1.0).collect();
        let sentinel = -123.0_f32;
        let mut c_lower = vec![sentinel; n * n];
        ssyrk(n, k, 1.0, &a, 0.0, &mut c_lower, false).expect("syrk lower");
        // Strict upper triangle must remain the sentinel.
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(c_lower[i * n + j], sentinel, "upper ({i},{j}) modified");
            }
        }
        // Lower triangle should equal A·Aᵀ.
        let full = full_aat(n, k, 1.0, &a);
        for i in 0..n {
            for j in 0..=i {
                assert!((c_lower[i * n + j] - full[i * n + j]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn k_1_outer_product() {
        // k=1 → A is a column vector v; A·Aᵀ = v vᵀ (rank-1 outer product).
        let n = 3;
        let v = vec![2.0_f32, 3.0, 4.0];
        let mut c = vec![0.0_f32; n * n];
        ssyrk(n, 1, 1.0, &v, 0.0, &mut c, true).expect("syrk");
        for i in 0..n {
            for j in i..n {
                let expected = v[i] * v[j];
                assert!((c[i * n + j] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn dim_mismatch_error() {
        let (n, k) = (2, 2);
        let a = vec![1.0_f32; n * k - 1]; // wrong length
        let mut c = vec![0.0_f32; n * n];
        let err = ssyrk(n, k, 1.0, &a, 0.0, &mut c, true);
        assert!(matches!(err, Err(BlasError::DimensionMismatch(_))));
    }

    #[test]
    fn n_0_error() {
        let mut c = vec![0.0_f32; 0];
        let err = ssyrk(0, 2, 1.0, &[], 0.0, &mut c, true);
        assert!(matches!(err, Err(BlasError::InvalidDimension(_))));
    }

    #[test]
    fn diagonal_nonneg_for_aat() {
        // With alpha ≥ 0, beta = 0, diagonal of A·Aᵀ is ‖row_i‖² ≥ 0.
        let (n, k) = (4, 3);
        let a: Vec<f32> = (0..n * k).map(|i| (i as f32) - 5.0).collect();
        let mut c = vec![0.0_f32; n * n];
        ssyrk(n, k, 2.0, &a, 0.0, &mut c, true).expect("syrk");
        for i in 0..n {
            assert!(c[i * n + i] >= 0.0, "diag {i} negative: {}", c[i * n + i]);
        }
    }

    #[test]
    fn beta_accumulation() {
        // C starts at I, beta=1, alpha=1: result diag = 1 + ‖row‖².
        let (n, k) = (2, 2);
        let a = vec![1.0, 0.0, 0.0, 1.0]; // rows e1, e2 → A·Aᵀ = I
        let mut c = vec![0.0_f32; n * n];
        for i in 0..n {
            c[i * n + i] = 1.0;
        }
        ssyrk(n, k, 1.0, &a, 1.0, &mut c, true).expect("syrk");
        // diag = 1 (beta·C) + 1 (A·Aᵀ) = 2.
        assert!((c[0] - 2.0).abs() < 1e-5);
        assert!((c[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn output_finite() {
        let (n, k) = (5, 4);
        let a: Vec<f32> = (0..n * k).map(|i| (i as f32).sin() * 3.0).collect();
        let mut c = vec![0.1_f32; n * n];
        ssyrk(n, k, 1.5, &a, 0.5, &mut c, false).expect("syrk");
        for &v in &c {
            assert!(v.is_finite());
        }
    }
}

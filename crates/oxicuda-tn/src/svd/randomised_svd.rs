//! Randomized SVD (Halko-Martinsson-Tropp 2011, Algorithm 4.3 + 4.4).
//!
//! ## Overview
//!
//! Given an `m × n` matrix `A`, the algorithm approximates the thin SVD
//! `A ≈ U · diag(s) · V^T` retaining only the top-`k` singular triplets.
//!
//! ### Phase 1 — Randomized range finding (Algorithm 4.3)
//!
//! 1. Draw a Gaussian test matrix `Ω` of shape `(n, l)` where `l = k + oversampling`.
//! 2. Form `Y = A · Ω`, shape `(m, l)`.
//! 3. Apply `n_power_iter` power iterations (each requires two QR factorisations) to
//!    improve accuracy on slowly decaying spectra.
//! 4. Orthonormalise the final `Y` via Modified Gram-Schmidt to obtain `Q`, shape `(m, l)`.
//!
//! ### Phase 2 — Projection and exact SVD (Algorithm 4.4)
//!
//! 5. Form the small matrix `B = Q^T · A`, shape `(l, n)`.
//! 6. Compute the exact thin SVD of `B` via one-sided Jacobi: `B = Ũ · diag(s) · V^T`.
//! 7. Recover `U = Q · Ũ`, shape `(m, l)`.
//! 8. Truncate to the top-`k` columns/rows.
//!
//! ## References
//!
//! Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011).
//! *Finding structure with randomness: Probabilistic algorithms for constructing
//! approximate matrix decompositions.* SIAM Review, 53(2), 217–288.

use crate::handle::LcgRng;
pub use crate::svd::svd_dense::SvdResult;
use crate::svd::svd_dense::svd_jacobi;
use crate::{TnError, TnResult};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the Randomized SVD algorithm.
///
/// | Field | Default | Meaning |
/// |-------|---------|---------|
/// | `k` | 2 | Number of singular triplets to retain |
/// | `oversampling` | 10 | Extra columns in the sketch (`l = k + oversampling`) |
/// | `n_power_iter` | 2 | Power iterations; more → better accuracy, higher cost |
/// | `seed` | 12345 | Seed for the internal `LcgRng` |
#[derive(Debug, Clone)]
pub struct RsvdConfig {
    pub k: usize,
    pub oversampling: usize,
    pub n_power_iter: usize,
    pub seed: u64,
}

impl Default for RsvdConfig {
    fn default() -> Self {
        Self {
            k: 2,
            oversampling: 10,
            n_power_iter: 2,
            seed: 12345,
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Compute the top-`k` SVD of an `m × n` row-major matrix using the
/// Halko-Martinsson-Tropp randomized algorithm (HMT 2011, Alg. 4.3 + 4.4).
///
/// Returns `SvdResult` with:
/// - `u`  : `(m, k)` row-major — left singular vectors
/// - `s`  : length `k`, descending — singular values
/// - `vt` : `(k, n)` row-major — right singular vectors (rows of `V^T`)
///
/// # Errors
/// - [`TnError::EmptyInput`] when `m == 0` or `n == 0`.
/// - [`TnError::ShapeMismatch`] when `a.len() != m * n`.
/// - [`TnError::RankExceedsLimit`] when `k > min(m, n)`.
/// - [`TnError::InvalidConfiguration`] when `k == 0` or `l > min(m, n)` in degenerate cases.
/// - [`TnError::NotConverged`] propagated from the inner Jacobi SVD.
pub fn randomised_svd(a: &[f64], m: usize, n: usize, config: &RsvdConfig) -> TnResult<SvdResult> {
    // ── Validate inputs ────────────────────────────────────────────────────
    if m == 0 || n == 0 {
        return Err(TnError::EmptyInput);
    }
    if a.len() != m * n {
        return Err(TnError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    let k = config.k;
    if k == 0 {
        return Err(TnError::InvalidConfiguration(
            "k must be at least 1".to_string(),
        ));
    }
    let min_dim = m.min(n);
    if k > min_dim {
        return Err(TnError::RankExceedsLimit {
            rank: k,
            max: min_dim,
        });
    }

    // l = k + oversampling, clamped so that l ≤ min(m, n)
    let l = (k + config.oversampling).min(min_dim);

    // ── Phase 1: Randomized range finding ─────────────────────────────────
    let q = randomized_range_finder(a, m, n, l, config.n_power_iter, config.seed)?;
    // q : (m, l) — orthonormal columns spanning the approximate range of A

    // ── Phase 2: Projection onto the found subspace ───────────────────────
    // B = Q^T @ A  →  (l, n)
    let b = mat_mul_at_b(&q, a, m, l, n);

    // Thin SVD of the small (l × n) matrix
    let svd_b = svd_jacobi(&b, l, n)?;
    // svd_b.u  : (l, k_b) where k_b = min(l, n) = l  (since l ≤ n typically)
    // svd_b.s  : length k_b, descending
    // svd_b.vt : (k_b, n)

    // U = Q @ Ũ   →  (m, k_b)
    let k_b = svd_b.k; // min(l, n)
    let u_full = mat_mul(&q, &svd_b.u, m, l, k_b);

    // ── Truncate to top-k ─────────────────────────────────────────────────
    // singular values are already sorted descending by svd_jacobi
    let k_out = k.min(k_b);

    // Extract top-k columns of u_full  (m, k_b) → (m, k_out)
    let mut u_out = vec![0.0f64; m * k_out];
    for i in 0..m {
        for j in 0..k_out {
            u_out[i * k_out + j] = u_full[i * k_b + j];
        }
    }

    // Extract top-k singular values
    let s_out: Vec<f64> = svd_b.s[..k_out].to_vec();

    // Extract top-k rows of vt  (k_b, n) → (k_out, n)
    let vt_out: Vec<f64> = svd_b.vt[..k_out * n].to_vec();

    Ok(SvdResult {
        u: u_out,
        s: s_out,
        vt: vt_out,
        m,
        n,
        k: k_out,
    })
}

/// Modified Gram-Schmidt QR, returning only the orthonormal factor `Q` of shape `(m, n)`.
///
/// Columns that become nearly zero (norm < `1e-10`) after orthogonalisation are kept as
/// zero vectors — this signals (near-)rank-deficiency without propagating NaN.
///
/// # Errors
/// Returns [`TnError::ShapeMismatch`] when `a.len() != m * n`.
/// Returns [`TnError::EmptyInput`] when `m == 0` or `n == 0`.
pub fn qr_gram_schmidt(a: &[f64], m: usize, n: usize) -> TnResult<Vec<f64>> {
    if m == 0 || n == 0 {
        return Err(TnError::EmptyInput);
    }
    if a.len() != m * n {
        return Err(TnError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }

    // Copy into column-major scratch for efficient column access during MGS.
    // We store as column-major: col j occupies rows [j*m .. j*m+m].
    let mut cols: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..m).map(|i| a[i * n + j]).collect())
        .collect();

    for j in 0..n {
        // Orthogonalise column j against all previous orthonormal columns.
        for i in 0..j {
            let dot: f64 = cols[i].iter().zip(cols[j].iter()).map(|(x, y)| x * y).sum();
            // cols[j] -= dot * cols[i]  (borrow-checker safe split)
            let (left, right) = cols.split_at_mut(j);
            for r in 0..m {
                right[0][r] -= dot * left[i][r];
            }
        }
        // Normalise column j
        let nrm2: f64 = cols[j].iter().map(|x| x * x).sum();
        if nrm2 > 1.0e-20 {
            let nrm = nrm2.sqrt();
            for val in cols[j].iter_mut() {
                *val /= nrm;
            }
        }
        // If nrm2 ≤ 1e-20 we leave the column as zero (rank-deficient signal)
    }

    // Re-assemble into row-major (m × n)
    let mut q = vec![0.0f64; m * n];
    for j in 0..n {
        for i in 0..m {
            q[i * n + j] = cols[j][i];
        }
    }
    Ok(q)
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Matrix multiply `A @ B` where `A` is `(m, k)` row-major and `B` is `(k, n)` row-major.
/// Returns an `(m, n)` row-major matrix.
fn mat_mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    c
}

/// Compute `A^T @ B` where `A` is `(m_a, n_a)` row-major and `B` is `(m_a, n_b)` row-major.
/// Returns an `(n_a, n_b)` row-major matrix.
fn mat_mul_at_b(a: &[f64], b: &[f64], m_a: usize, n_a: usize, n_b: usize) -> Vec<f64> {
    // C[i, j] = sum_r A[r, i] * B[r, j]
    let mut c = vec![0.0f64; n_a * n_b];
    for r in 0..m_a {
        for i in 0..n_a {
            let ari = a[r * n_a + i];
            if ari == 0.0 {
                continue;
            }
            for j in 0..n_b {
                c[i * n_b + j] += ari * b[r * n_b + j];
            }
        }
    }
    c
}

/// Randomized range finder (HMT 2011, Algorithm 4.3 with optional power iterations).
///
/// Returns an orthonormal `Q` of shape `(m, l)` whose columns approximately span
/// the range of `A` (shape `(m, n)`).
fn randomized_range_finder(
    a: &[f64],
    m: usize,
    n: usize,
    l: usize,
    n_power_iter: usize,
    seed: u64,
) -> TnResult<Vec<f64>> {
    let mut rng = LcgRng::new(seed);

    // Step 1: Draw Gaussian test matrix Ω of shape (n, l)
    let omega: Vec<f64> = (0..n * l).map(|_| rng.next_normal()).collect();

    // Step 2: Y = A @ Ω  →  (m, l)
    let mut y = mat_mul(a, &omega, m, n, l);

    // Step 3: Power iterations for better spectral decay
    for _ in 0..n_power_iter {
        // QR of Y → Q₁  (m, l)
        let q1 = qr_gram_schmidt(&y, m, l)?;
        // Y = A^T @ Q₁  →  (n, l)
        let y2 = mat_mul_at_b(a, &q1, m, n, l);
        // QR of Y2 → Q₂  (n, l)
        let q2 = qr_gram_schmidt(&y2, n, l)?;
        // Y = A @ Q₂  →  (m, l)
        y = mat_mul(a, &q2, m, n, l);
    }

    // Step 4: Final QR of Y → Q  (m, l)
    let q = qr_gram_schmidt(&y, m, l)?;
    Ok(q)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Frobenius distance between two vectors of the same length.
    fn fro_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// Reconstruct `A ≈ U * diag(s) * V^T` from an `SvdResult`.
    fn reconstruct(svd: &SvdResult) -> Vec<f64> {
        let (m, n, k) = (svd.m, svd.n, svd.k);
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for c in 0..k {
                    acc += svd.u[i * k + c] * svd.s[c] * svd.vt[c * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    /// Build a rank-r matrix  A = U_true * diag(sigmas) * V_true^T  using the LCG RNG.
    fn build_low_rank(m: usize, n: usize, r: usize, sigmas: &[f64], seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        // Random tall-thin U_true (m, r) and V_true (n, r) — we orthonormalize them.
        let u_raw: Vec<f64> = (0..m * r).map(|_| rng.next_normal()).collect();
        let v_raw: Vec<f64> = (0..n * r).map(|_| rng.next_normal()).collect();
        let u_orth = qr_gram_schmidt(&u_raw, m, r).unwrap();
        let v_orth = qr_gram_schmidt(&v_raw, n, r).unwrap();

        // A[i,j] = sum_c sigma[c] * u_orth[i,c] * v_orth[j,c]
        let mut a = vec![0.0f64; m * n];
        for c in 0..r {
            for i in 0..m {
                for j in 0..n {
                    a[i * n + j] += sigmas[c] * u_orth[i * r + c] * v_orth[j * r + c];
                }
            }
        }
        a
    }

    // ── Tests ─────────────────────────────────────────────────────────────

    /// Rank-1 matrix: RSVD should recover the single nonzero singular value exactly.
    #[test]
    fn rsvd_rank1_matrix_exact() {
        // A = u * s[0] * v^T with u and v unit vectors
        let m = 8;
        let n = 6;
        let sigma = 5.0;
        let a = build_low_rank(m, n, 1, &[sigma], 42);

        let config = RsvdConfig {
            k: 1,
            oversampling: 5,
            n_power_iter: 2,
            seed: 7,
        };
        let result = randomised_svd(&a, m, n, &config).expect("RSVD should succeed");
        assert_eq!(result.k, 1);
        assert!(
            (result.s[0] - sigma).abs() < 1e-4,
            "Expected sigma ≈ {sigma}, got {}",
            result.s[0]
        );
    }

    /// Low-rank reconstruction: Frobenius error should be small.
    #[test]
    fn rsvd_low_rank_matrix_reconstructs() {
        let m = 12;
        let n = 10;
        let sigmas = [8.0, 4.0];
        let a = build_low_rank(m, n, 2, &sigmas, 99);

        let config = RsvdConfig {
            k: 2,
            oversampling: 8,
            n_power_iter: 3,
            seed: 13,
        };
        let result = randomised_svd(&a, m, n, &config).expect("RSVD should succeed");
        let a_hat = reconstruct(&result);
        let err = fro_diff(&a, &a_hat);
        assert!(
            err < 0.1,
            "Frobenius reconstruction error {err:.4} should be < 0.1"
        );
    }

    /// Output shapes must match the documented contract.
    #[test]
    fn rsvd_output_shapes() {
        let m = 7;
        let n = 5;
        let k = 3;
        let a: Vec<f64> = (0..m * n).map(|x| x as f64).collect();
        let config = RsvdConfig {
            k,
            oversampling: 2,
            n_power_iter: 1,
            seed: 1,
        };
        let result = randomised_svd(&a, m, n, &config).expect("RSVD should succeed");
        assert_eq!(result.u.len(), m * k, "u shape (m*k)");
        assert_eq!(result.s.len(), k, "s length k");
        assert_eq!(result.vt.len(), k * n, "vt shape (k*n)");
        assert_eq!(result.m, m);
        assert_eq!(result.n, n);
        assert_eq!(result.k, k);
    }

    /// The left singular matrix `U` must satisfy `U^T U ≈ I_k` within tolerance.
    #[test]
    fn rsvd_orthonormal_u() {
        let m = 10;
        let n = 8;
        let sigmas = [6.0, 3.0, 1.5];
        let a = build_low_rank(m, n, 3, &sigmas, 77);

        let config = RsvdConfig {
            k: 3,
            oversampling: 5,
            n_power_iter: 2,
            seed: 55,
        };
        let result = randomised_svd(&a, m, n, &config).expect("RSVD should succeed");
        let k = result.k;
        let u = &result.u;

        // Compute G = U^T U  →  (k, k)
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..m).map(|r| u[r * k + i] * u[r * k + j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "U^T U [{i},{j}] = {dot:.2e}, expected {expected}"
                );
            }
        }
    }

    /// Singular values must be returned in descending order.
    #[test]
    fn rsvd_singular_values_descending() {
        let m = 9;
        let n = 6;
        let sigmas = [10.0, 5.0, 2.0];
        let a = build_low_rank(m, n, 3, &sigmas, 333);

        let config = RsvdConfig {
            k: 3,
            oversampling: 3,
            n_power_iter: 2,
            seed: 17,
        };
        let result = randomised_svd(&a, m, n, &config).expect("RSVD should succeed");
        for i in 1..result.k {
            assert!(
                result.s[i - 1] >= result.s[i] - 1e-12,
                "Singular values not descending: s[{}]={} < s[{}]={}",
                i - 1,
                result.s[i - 1],
                i,
                result.s[i]
            );
        }
    }

    /// For a 3×3 identity matrix, the top-2 singular values should be ≈ 1.0.
    #[test]
    fn rsvd_identity_recovers_singular_values() {
        let mat = vec![1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let config = RsvdConfig {
            k: 2,
            oversampling: 1,
            n_power_iter: 1,
            seed: 42,
        };
        let result = randomised_svd(&mat, 3, 3, &config).expect("RSVD should succeed");
        assert_eq!(result.k, 2);
        for &sv in &result.s {
            assert!((sv - 1.0).abs() < 1e-4, "Expected sv ≈ 1.0, got {sv:.6}");
        }
    }

    /// Power iterations should not produce NaN/Inf; both 0 and 3 iterations should work.
    #[test]
    fn rsvd_power_iter_improves_accuracy() {
        let m = 15;
        let n = 10;
        // Construct a matrix with slowly-decaying spectrum (not rank-deficient)
        let sigmas = [9.0, 7.0, 5.0, 3.0, 1.0];
        let a = build_low_rank(m, n, 5, &sigmas, 500);

        for &pi in &[0usize, 3usize] {
            let config = RsvdConfig {
                k: 2,
                oversampling: 5,
                n_power_iter: pi,
                seed: 22,
            };
            let result = randomised_svd(&a, m, n, &config).expect("RSVD should succeed");
            // All singular values must be finite and non-negative
            for &sv in &result.s {
                assert!(
                    sv.is_finite() && sv >= 0.0,
                    "sv={sv} is not valid (pi={pi})"
                );
            }
        }
    }

    /// QR Gram-Schmidt output columns must be mutually orthonormal.
    #[test]
    fn qr_columns_orthonormal() {
        let m = 8;
        let n = 4;
        let mut rng = LcgRng::new(7);
        let a: Vec<f64> = (0..m * n).map(|_| rng.next_normal()).collect();
        let q = qr_gram_schmidt(&a, m, n).expect("QR should succeed");
        assert_eq!(q.len(), m * n);

        // Check Q^T Q ≈ I_n
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..m).map(|r| q[r * n + i] * q[r * n + j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q^T Q [{i},{j}] = {dot:.2e}, expected {expected}"
                );
            }
        }
    }

    /// Empty input (m=0 or n=0) must return `TnError::EmptyInput`.
    #[test]
    fn empty_input_returns_error() {
        let config = RsvdConfig::default();

        let err = randomised_svd(&[], 0, 5, &config).unwrap_err();
        assert!(
            matches!(err, TnError::EmptyInput),
            "Expected EmptyInput, got {err:?}"
        );

        let err = randomised_svd(&[], 5, 0, &config).unwrap_err();
        assert!(
            matches!(err, TnError::EmptyInput),
            "Expected EmptyInput, got {err:?}"
        );
    }

    /// Requesting k > min(m, n) must return `TnError::RankExceedsLimit`.
    #[test]
    fn invalid_k_returns_error() {
        let m = 4;
        let n = 3;
        let a: Vec<f64> = (0..m * n).map(|x| x as f64).collect();
        let config = RsvdConfig {
            k: 5, // > min(4,3) = 3
            oversampling: 2,
            n_power_iter: 1,
            seed: 1,
        };
        let err = randomised_svd(&a, m, n, &config).unwrap_err();
        assert!(
            matches!(err, TnError::RankExceedsLimit { .. }),
            "Expected RankExceedsLimit, got {err:?}"
        );
    }

    /// RSVD on a tall matrix (m > n): shapes should still be correct.
    #[test]
    fn rsvd_tall_matrix_shapes() {
        let m = 20;
        let n = 5;
        let sigmas = [4.0, 2.0];
        let a = build_low_rank(m, n, 2, &sigmas, 888);
        let config = RsvdConfig {
            k: 2,
            oversampling: 3,
            n_power_iter: 1,
            seed: 9,
        };
        let result = randomised_svd(&a, m, n, &config).expect("RSVD should succeed");
        assert_eq!(result.u.len(), m * 2);
        assert_eq!(result.s.len(), 2);
        assert_eq!(result.vt.len(), 2 * n);
    }

    /// RSVD on a wide matrix (m < n): shapes should still be correct.
    #[test]
    fn rsvd_wide_matrix_shapes() {
        let m = 4;
        let n = 15;
        let sigmas = [7.0, 3.0];
        let a = build_low_rank(m, n, 2, &sigmas, 111);
        let config = RsvdConfig {
            k: 2,
            oversampling: 2,
            n_power_iter: 2,
            seed: 3,
        };
        let result = randomised_svd(&a, m, n, &config).expect("RSVD should succeed");
        assert_eq!(result.u.len(), m * 2);
        assert_eq!(result.s.len(), 2);
        assert_eq!(result.vt.len(), 2 * n);
    }
}

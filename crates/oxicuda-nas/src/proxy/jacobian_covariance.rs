//! Jacobian-covariance ("jacov") zero-cost proxy for predictor-free ranking.
//!
//! Reference: Mellor, Turner, Storkey & Crowley, "Neural Architecture Search
//! without Training", arXiv 2020 (the original *jacov* score that preceded the
//! activation-kernel [`crate::proxy::zero_cost::naswot_score`]).
//!
//! # Idea
//!
//! Feed a minibatch of `N` inputs through an untrained network and collect, for
//! each input `i`, the Jacobian row `J_i = ∂(Σ logits)/∂x_i` (the gradient of the
//! network output with respect to that input, flattened to length `D`). A
//! network that can *distinguish* its inputs produces gradient directions that
//! point different ways for different inputs; a degenerate network produces
//! nearly-parallel gradients.
//!
//! Quantify this with the `N × N` Pearson correlation matrix `C` of the Jacobian
//! rows and its eigenvalues `λ_1 … λ_N`:
//!
//! ```text
//! jacov = − Σ_i [ ln(λ_i + ε) + 1 / (λ_i + ε) ],   ε = 1e-5
//! ```
//!
//! When the rows are mutually uncorrelated `C ≈ I`, every `λ_i ≈ 1`, each bracket
//! `≈ 1`, and the score `≈ −N` — its maximum. When the rows collapse onto one
//! direction `C ≈ 𝟙𝟙ᵀ`, one eigenvalue is `≈ N` and the rest are `≈ 0`; the
//! `1/(0 + ε)` terms blow up and the (negated) score becomes large-negative. So
//! **higher is better**, matching the convention of the sibling proxies.
//!
//! The module is self-contained: it ships a symmetric-matrix Jacobi eigenvalue
//! solver ([`symmetric_eigenvalues`]) and a Pearson correlation builder
//! ([`pearson_correlation_matrix`]); no external linear-algebra dependency and no
//! autodiff (the caller supplies the Jacobian rows).

use crate::error::{NasError, NasResult};

/// Diagonal jitter `ε` added to every eigenvalue inside the jacov score so that
/// `ln(λ + ε)` and `1 / (λ + ε)` stay finite when an eigenvalue is (near) zero.
/// Matches the `1e-5` used by the reference implementation.
pub const JACOV_EPSILON: f32 = 1e-5;

// ─── Pearson correlation ───────────────────────────────────────────────────────

/// Build the `N × N` Pearson correlation matrix of the row vectors `rows`
/// (row-major, symmetric, unit diagonal).
///
/// `C_ij = ⟨c_i, c_j⟩ / (‖c_i‖ ‖c_j‖)` where `c_i` is `rows[i]` centred to zero
/// mean. A row with zero variance (constant entries) has undefined correlation;
/// it is reported as `0` off-diagonal (and `1` on the diagonal), which keeps the
/// matrix a valid positive-semidefinite correlation matrix.
///
/// # Errors
/// - [`NasError::EmptySearchSpace`] if `rows` is empty or each row is empty.
/// - [`NasError::DimensionMismatch`] if the rows have differing lengths.
pub fn pearson_correlation_matrix(rows: &[Vec<f32>]) -> NasResult<Vec<f32>> {
    let n = rows.len();
    if n == 0 {
        return Err(NasError::EmptySearchSpace);
    }
    let d = rows[0].len();
    if d == 0 {
        return Err(NasError::EmptySearchSpace);
    }
    for row in rows {
        if row.len() != d {
            return Err(NasError::DimensionMismatch {
                expected: d,
                got: row.len(),
            });
        }
    }

    // Centre each row and pre-compute its L2 norm (in f64 for stability).
    let mut centered: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut norms: Vec<f64> = Vec::with_capacity(n);
    for row in rows {
        let mean = row.iter().map(|&x| x as f64).sum::<f64>() / d as f64;
        let cen: Vec<f64> = row.iter().map(|&x| x as f64 - mean).collect();
        let norm = cen.iter().map(|v| v * v).sum::<f64>().sqrt();
        centered.push(cen);
        norms.push(norm);
    }

    let mut c = vec![0.0_f32; n * n];
    for i in 0..n {
        c[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let corr = if norms[i] <= 0.0 || norms[j] <= 0.0 {
                0.0_f32
            } else {
                let dot: f64 = centered[i]
                    .iter()
                    .zip(centered[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                // Clamp to [-1, 1] to absorb f64 rounding past the Cauchy-Schwarz bound.
                ((dot / (norms[i] * norms[j])).clamp(-1.0, 1.0)) as f32
            };
            c[i * n + j] = corr;
            c[j * n + i] = corr;
        }
    }
    Ok(c)
}

// ─── Symmetric eigenvalues (Jacobi) ─────────────────────────────────────────────

/// One Jacobi rotation update of the two off-diagonal entries `a[i][j]` and
/// `a[k][l]` (Numerical-Recipes form) given rotation parameters `s`, `tau`.
fn jacobi_rotate(
    a: &mut [f64],
    n: usize,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    s: f64,
    tau: f64,
) {
    let g = a[i * n + j];
    let h = a[k * n + l];
    a[i * n + j] = g - s * (h + g * tau);
    a[k * n + l] = h + s * (g - h * tau);
}

/// Compute the eigenvalues of the symmetric `n × n` matrix `matrix` (row-major)
/// with the cyclic Jacobi rotation method, in f64, returned sorted ascending.
///
/// The input is assumed symmetric; only the upper triangle and diagonal are
/// referenced. Real symmetric matrices have real eigenvalues, so the result is
/// always finite.
fn jacobi_eigenvalues_f64(matrix: &[f32], n: usize) -> NasResult<Vec<f64>> {
    if matrix.len() != n * n {
        return Err(NasError::DimensionMismatch {
            expected: n * n,
            got: matrix.len(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![matrix[0] as f64]);
    }

    let mut a: Vec<f64> = matrix.iter().map(|&v| v as f64).collect();
    let max_sweeps = 100;

    for _sweep in 0..max_sweeps {
        // Off-diagonal magnitude and a diagonal scale for a relative stop test.
        let mut off = 0.0_f64;
        let mut diag_scale = 0.0_f64;
        for p in 0..n {
            diag_scale += a[p * n + p].abs();
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off.sqrt() <= 1e-14 * (diag_scale + 1e-30) {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];

                // Tangent of the rotation angle: the smaller root of
                // t² + 2·θ·t − 1 = 0 with θ = (a_qq − a_pp)/(2 a_pq),
                // which is the numerically-stable choice (|t| ≤ 1).
                let theta = 0.5 * (aqq - app) / apq;
                let t0 = 1.0 / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta < 0.0 { -t0 } else { t0 };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                let dh = t * apq;
                a[p * n + p] = app - dh;
                a[q * n + q] = aqq + dh;
                a[p * n + q] = 0.0;

                // Rotate the remaining off-diagonal entries of rows/cols p, q.
                for r in 0..p {
                    jacobi_rotate(&mut a, n, r, p, r, q, s, tau);
                }
                for r in (p + 1)..q {
                    jacobi_rotate(&mut a, n, p, r, r, q, s, tau);
                }
                for r in (q + 1)..n {
                    jacobi_rotate(&mut a, n, p, r, q, r, s, tau);
                }
            }
        }
    }

    let mut eig: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    eig.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(eig)
}

/// Eigenvalues of a symmetric `n × n` matrix (row-major), sorted ascending.
///
/// Self-contained cyclic-Jacobi solver; the input is assumed symmetric (only the
/// upper triangle is used). Returns the eigenvalues as `f32`.
///
/// # Errors
/// [`NasError::DimensionMismatch`] if `matrix.len() != n * n`.
pub fn symmetric_eigenvalues(matrix: &[f32], n: usize) -> NasResult<Vec<f32>> {
    let eig = jacobi_eigenvalues_f64(matrix, n)?;
    Ok(eig.into_iter().map(|v| v as f32).collect())
}

// ─── jacov score ───────────────────────────────────────────────────────────────

/// Jacobian-covariance score of a minibatch Jacobian (Mellor et al. 2020).
///
/// `jacobian[i]` is the (flattened) gradient of the network output with respect
/// to input sample `i`. The score is
/// `−Σ_i [ ln(λ_i + ε) + 1/(λ_i + ε) ]` over the eigenvalues `λ_i` of the Pearson
/// correlation matrix of the rows, with `ε =` [`JACOV_EPSILON`]. **Higher is
/// better**: architectures whose per-input gradients are mutually uncorrelated
/// (the network separates its inputs) score near `−N`, while architectures whose
/// gradients collapse onto a single direction score large-negative.
///
/// # Errors
/// - [`NasError::EmptySearchSpace`] if `jacobian` is empty or rows are empty.
/// - [`NasError::DimensionMismatch`] if the rows have differing lengths.
pub fn jacobian_covariance_score(jacobian: &[Vec<f32>]) -> NasResult<f32> {
    let n = jacobian.len();
    if n == 0 {
        return Err(NasError::EmptySearchSpace);
    }
    let corr = pearson_correlation_matrix(jacobian)?;
    let eig = jacobi_eigenvalues_f64(&corr, n)?;

    let eps = JACOV_EPSILON as f64;
    let mut acc = 0.0_f64;
    for lam in eig {
        // Correlation matrices are positive-semidefinite; clamp tiny negative
        // round-off before adding the jitter so the log/reciprocal stay finite.
        let v = lam.max(0.0) + eps;
        acc += v.ln() + 1.0 / v;
    }
    Ok(-acc as f32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ── symmetric_eigenvalues ────────────────────────────────────────────────────

    #[test]
    fn eig_diagonal_matrix() {
        // diag(2, 3, 5) → eigenvalues {2, 3, 5}.
        let m = vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0];
        let eig = symmetric_eigenvalues(&m, 3).expect("eig");
        assert!(approx(eig[0], 2.0, 1e-5));
        assert!(approx(eig[1], 3.0, 1e-5));
        assert!(approx(eig[2], 5.0, 1e-5));
    }

    #[test]
    fn eig_two_by_two_known() {
        // [[2,1],[1,2]] → eigenvalues 1 and 3.
        let m = vec![2.0, 1.0, 1.0, 2.0];
        let eig = symmetric_eigenvalues(&m, 2).expect("eig");
        assert!(approx(eig[0], 1.0, 1e-5), "{eig:?}");
        assert!(approx(eig[1], 3.0, 1e-5), "{eig:?}");
    }

    #[test]
    fn eig_rank_one_matrix() {
        // [[1,1],[1,1]] → eigenvalues 0 and 2.
        let m = vec![1.0, 1.0, 1.0, 1.0];
        let eig = symmetric_eigenvalues(&m, 2).expect("eig");
        assert!(approx(eig[0], 0.0, 1e-5), "{eig:?}");
        assert!(approx(eig[1], 2.0, 1e-5), "{eig:?}");
    }

    #[test]
    fn eig_identity() {
        let m = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let eig = symmetric_eigenvalues(&m, 4).expect("eig");
        assert!(eig.iter().all(|&v| approx(v, 1.0, 1e-5)), "{eig:?}");
    }

    #[test]
    fn eig_trace_is_preserved() {
        // Sum of eigenvalues equals the trace for a deterministic symmetric matrix.
        let m = vec![4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0];
        let eig = symmetric_eigenvalues(&m, 3).expect("eig");
        let trace = 4.0 + 5.0 + 6.0;
        let sum: f32 = eig.iter().sum();
        assert!(approx(sum, trace, 1e-3), "sum {sum} vs trace {trace}");
    }

    #[test]
    fn eig_empty_and_singleton() {
        assert!(symmetric_eigenvalues(&[], 0).expect("empty").is_empty());
        let one = symmetric_eigenvalues(&[7.0], 1).expect("one");
        assert!(approx(one[0], 7.0, 1e-6));
    }

    #[test]
    fn eig_dimension_mismatch_errors() {
        assert!(matches!(
            symmetric_eigenvalues(&[1.0, 2.0, 3.0], 2),
            Err(NasError::DimensionMismatch { .. })
        ));
    }

    // ── pearson_correlation_matrix ───────────────────────────────────────────────

    #[test]
    fn correlation_identical_rows_is_all_ones() {
        let rows = vec![vec![1.0, 2.0, 3.0, 4.0]; 3];
        let c = pearson_correlation_matrix(&rows).expect("corr");
        for v in &c {
            assert!(approx(*v, 1.0, 1e-5), "{c:?}");
        }
    }

    #[test]
    fn correlation_orthogonal_rows_is_identity() {
        // Centred rows are pairwise orthogonal ⇒ off-diagonals zero.
        let rows = vec![
            vec![1.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, -1.0],
            vec![1.0, 1.0, -1.0, -1.0],
        ];
        let c = pearson_correlation_matrix(&rows).expect("corr");
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx(c[i * n + j], expected, 1e-5),
                    "C[{i}][{j}]={}",
                    c[i * n + j]
                );
            }
        }
    }

    #[test]
    fn correlation_is_symmetric_with_unit_diagonal() {
        let rows = vec![
            vec![1.0, 2.0, 0.5, -1.0],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![-1.0, 0.0, 1.0, 0.5],
        ];
        let c = pearson_correlation_matrix(&rows).expect("corr");
        let n = 3;
        for i in 0..n {
            assert!(approx(c[i * n + i], 1.0, 1e-6));
            for j in 0..n {
                assert!(approx(c[i * n + j], c[j * n + i], 1e-6));
            }
        }
    }

    #[test]
    fn correlation_zero_variance_row_is_finite() {
        let rows = vec![vec![5.0, 5.0, 5.0], vec![1.0, 2.0, 3.0]];
        let c = pearson_correlation_matrix(&rows).expect("corr");
        assert!(c.iter().all(|v| v.is_finite()));
        // Constant row is treated as uncorrelated with the other.
        assert!(approx(c[1], 0.0, 1e-6));
    }

    #[test]
    fn correlation_empty_and_ragged_errors() {
        let empty: Vec<Vec<f32>> = Vec::new();
        assert_eq!(
            pearson_correlation_matrix(&empty),
            Err(NasError::EmptySearchSpace)
        );
        let ragged = vec![vec![1.0, 2.0], vec![1.0, 2.0, 3.0]];
        assert!(matches!(
            pearson_correlation_matrix(&ragged),
            Err(NasError::DimensionMismatch { .. })
        ));
    }

    // ── jacobian_covariance_score ────────────────────────────────────────────────

    #[test]
    fn jacov_uncorrelated_near_minus_n() {
        // Orthogonal (after centring) Jacobian rows ⇒ C = I ⇒ score ≈ -N.
        let jac = vec![
            vec![1.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, -1.0],
            vec![1.0, 1.0, -1.0, -1.0],
        ];
        let s = jacobian_covariance_score(&jac).expect("jacov");
        let eps = JACOV_EPSILON;
        let per = (1.0_f32 + eps).ln() + 1.0 / (1.0 + eps);
        let expected = -3.0 * per;
        assert!(
            approx(s, expected, 1e-2),
            "score {s} vs expected {expected}"
        );
    }

    #[test]
    fn jacov_uncorrelated_beats_correlated() {
        let uncorrelated = vec![
            vec![1.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, -1.0],
            vec![1.0, 1.0, -1.0, -1.0],
        ];
        // Perfectly correlated: every row proportional to the same base vector.
        let base = [1.0_f32, -1.0, 2.0, -2.0];
        let correlated: Vec<Vec<f32>> = (1..=3)
            .map(|m| base.iter().map(|&x| x * m as f32).collect())
            .collect();

        let s_uncorr = jacobian_covariance_score(&uncorrelated).expect("uncorr");
        let s_corr = jacobian_covariance_score(&correlated).expect("corr");
        assert!(s_uncorr.is_finite() && s_corr.is_finite());
        assert!(
            s_uncorr > s_corr,
            "uncorrelated {s_uncorr} should beat correlated {s_corr}"
        );
    }

    #[test]
    fn jacov_single_sample_finite() {
        // N = 1: C = [[1]], one eigenvalue 1, score = -(ln(1+ε) + 1/(1+ε)).
        let jac = vec![vec![1.0, 2.0, 3.0]];
        let s = jacobian_covariance_score(&jac).expect("jacov");
        let eps = JACOV_EPSILON;
        let expected = -((1.0_f32 + eps).ln() + 1.0 / (1.0 + eps));
        assert!(approx(s, expected, 1e-4), "score {s} vs {expected}");
    }

    #[test]
    fn jacov_is_deterministic() {
        let jac = vec![
            vec![0.5, -0.5, 1.0, 0.0],
            vec![1.0, 0.0, -1.0, 0.5],
            vec![-0.5, 1.0, 0.0, 0.25],
        ];
        let a = jacobian_covariance_score(&jac).expect("a");
        let b = jacobian_covariance_score(&jac).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn jacov_empty_and_ragged_errors() {
        let empty: Vec<Vec<f32>> = Vec::new();
        assert_eq!(
            jacobian_covariance_score(&empty),
            Err(NasError::EmptySearchSpace)
        );
        let ragged = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(matches!(
            jacobian_covariance_score(&ragged),
            Err(NasError::DimensionMismatch { .. })
        ));
    }
}

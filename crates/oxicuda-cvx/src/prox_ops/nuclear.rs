//! Nuclear norm proximal operator (SVD soft-thresholding).
//!
//! For `g(X) = λ ||X||_*` (sum of singular values),
//! `prox_g(Y) = U · diag(soft_threshold(σ_i, λ)) · V^T`.
//!
//! We compute the SVD via one-sided Jacobi rotations on the matrix `M = Y` directly:
//! repeatedly rotate column pairs `(p, q)` to orthogonalise them.  After convergence,
//! column norms are singular values and the columns themselves (normalised) form V.

use crate::error::{CvxError, CvxResult};
use crate::prox_ops::l1::soft_threshold;

/// Compute SVD `M = U Σ V^T` for `m × n` row-major `m` (m ≥ n).
///
/// Returns `(sigmas, u, v)`:
/// - `sigmas` length n (non-sorted but all ≥ 0).
/// - `u` row-major `m × n` orthonormal columns.
/// - `v` row-major `n × n` orthogonal.
pub fn one_sided_jacobi_svd(
    a: &[f64],
    m: usize,
    n: usize,
    max_sweeps: usize,
    tol: f64,
) -> CvxResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if m < n {
        return Err(CvxError::InvalidParameter(format!(
            "Jacobi SVD requires m≥n, got m={m}, n={n}"
        )));
    }
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    let mut u = a.to_vec();
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..max_sweeps {
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                // Compute alpha = U_p · U_p, beta = U_q · U_q, gamma = U_p · U_q.
                let mut alpha = 0.0_f64;
                let mut beta = 0.0_f64;
                let mut gamma = 0.0_f64;
                for i in 0..m {
                    let up = u[i * n + p];
                    let uq = u[i * n + q];
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }
                off += gamma * gamma;
                if gamma.abs() < 1.0e-300 {
                    continue;
                }
                let theta = (beta - alpha) / (2.0 * gamma);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Apply Givens rotation to columns p, q of U.
                for i in 0..m {
                    let up = u[i * n + p];
                    let uq = u[i * n + q];
                    u[i * n + p] = c * up - s * uq;
                    u[i * n + q] = s * up + c * uq;
                }
                // Update V.
                for i in 0..n {
                    let vp = v[i * n + p];
                    let vq = v[i * n + q];
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if off.sqrt() < tol {
            break;
        }
    }
    // Extract sigmas and normalise U columns.
    let mut sigmas = vec![0.0_f64; n];
    for j in 0..n {
        let mut nrm_sq = 0.0_f64;
        for i in 0..m {
            nrm_sq += u[i * n + j] * u[i * n + j];
        }
        let nrm = nrm_sq.sqrt();
        sigmas[j] = nrm;
        if nrm > 1.0e-300 {
            let inv = 1.0 / nrm;
            for i in 0..m {
                u[i * n + j] *= inv;
            }
        }
    }
    Ok((sigmas, u, v))
}

/// Prox of `λ ||·||_*` on row-major `m × n` matrix `y`.
pub fn prox_nuclear(y: &[f64], m: usize, n: usize, lambda: f64) -> CvxResult<Vec<f64>> {
    if y.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "Nuclear prox requires lambda ≥ 0, got {lambda}"
        )));
    }
    // Handle m < n by transposing.
    if m < n {
        // Compute on transpose then transpose back.
        let mut yt = vec![0.0_f64; n * m];
        for i in 0..m {
            for j in 0..n {
                yt[j * m + i] = y[i * n + j];
            }
        }
        let res_t = prox_nuclear(&yt, n, m, lambda)?;
        let mut out = vec![0.0_f64; m * n];
        for i in 0..n {
            for j in 0..m {
                out[j * n + i] = res_t[i * m + j];
            }
        }
        return Ok(out);
    }
    let (sigmas, u, v) = one_sided_jacobi_svd(y, m, n, 200, 1.0e-12)?;
    // Soft-threshold singular values.
    let thresh: Vec<f64> = sigmas
        .into_iter()
        .map(|s| soft_threshold(s, lambda))
        .collect();
    // Reconstruct: out = U diag(thresh) V^T.
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += u[i * n + k] * thresh[k] * v[j * n + k];
            }
            out[i * n + j] = acc;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nuclear_diag_threshold() {
        // Y = diag(3, 0.5).
        let y = vec![3.0, 0.0, 0.0, 0.5];
        let out = prox_nuclear(&y, 2, 2, 1.0).expect("ok");
        // Singular values 3 and 0.5; after threshold: 2 and 0.
        // So out should be ~ diag(2, 0).
        assert!((out[0] - 2.0).abs() < 1.0e-8);
        assert!(out[1].abs() < 1.0e-8);
        assert!(out[2].abs() < 1.0e-8);
        assert!(out[3].abs() < 1.0e-8);
    }

    #[test]
    fn nuclear_zero_lambda_recovers() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let out = prox_nuclear(&y, 2, 2, 0.0).expect("ok");
        for (oi, yi) in out.iter().zip(y.iter()) {
            assert!((oi - yi).abs() < 1.0e-8);
        }
    }

    #[test]
    fn jacobi_svd_recovers_norms() {
        // Matrix A = [[2, 0], [0, 3]] → singular values {2, 3}.
        let a = vec![2.0, 0.0, 0.0, 3.0];
        let (sigmas, _, _) = one_sided_jacobi_svd(&a, 2, 2, 200, 1.0e-14).expect("ok");
        let mut sorted = sigmas.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!((sorted[0] - 2.0).abs() < 1.0e-8);
        assert!((sorted[1] - 3.0).abs() < 1.0e-8);
    }
}

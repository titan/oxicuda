//! Sparse Bayesian Learning (Tipping 2001) for sparse recovery.
//!
//! Bayesian model: y = Φ x + ε, with prior `p(x_j) = N(0, γ_j)` and noise `ε ~ N(0, σ² I)`.
//! Expectation-maximisation alternates:
//!   - Posterior mean μ = Σ Φᵀ y / σ², Σ = (diag(1/γ) + ΦᵀΦ/σ²)⁻¹.
//!   - γ_j ← μ_j² + Σ_{jj}.
//!   - σ² ← ||y − Φμ||² / (m − Σ_j (1 − γ_j⁻¹ Σ_{jj})).
//!
//! Variables with γ_j → 0 are pruned (sparsity emerges naturally).

use crate::error::{CsError, CsResult};
use crate::linalg::cholesky::{cholesky_factor, cholesky_solve};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::sbl::SblResult;

/// Sparse Bayesian Learning EM iteration.
pub fn sparse_bayesian(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    max_iter: usize,
    tol: f64,
) -> CsResult<SblResult> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    let mut gamma = vec![1.0_f64; n];
    let mut sigma2 = (norm2(y) / (m as f64).sqrt()).max(1.0e-6);
    sigma2 *= sigma2;
    let mut mu = vec![0.0_f64; n];
    let mut iter = 0usize;
    for _ in 0..max_iter {
        // Build (diag(1/γ) + ΦᵀΦ/σ²).
        let mut g = vec![0.0_f64; n * n];
        let inv_sigma2 = 1.0 / sigma2;
        for k in 0..m {
            let row = k * n;
            for i in 0..n {
                let pki = phi[row + i] * inv_sigma2;
                for j in 0..n {
                    g[i * n + j] += pki * phi[row + j];
                }
            }
        }
        for j in 0..n {
            let g_inv = if gamma[j] > 1.0e-300 {
                1.0 / gamma[j]
            } else {
                1.0e12
            };
            g[j * n + j] += g_inv;
        }
        let l = cholesky_factor(&g, n)?;
        // μ = Σ Φᵀ y / σ²; rhs = Φᵀ y / σ².
        let phi_t_y = mat_t_vec(phi, m, n, y)?;
        let mut rhs = vec![0.0_f64; n];
        for j in 0..n {
            rhs[j] = phi_t_y[j] * inv_sigma2;
        }
        let mu_new = cholesky_solve(&l, n, &rhs)?;
        // Σ_{jj}: solve L L^T s_j = e_j; we read diag(Σ) = diag(L^{-T} L^{-1}).
        // Cheap approximation: compute the diagonal by solving for each unit vector.
        let mut sigma_diag = vec![0.0_f64; n];
        for j in 0..n {
            let mut ej = vec![0.0_f64; n];
            ej[j] = 1.0;
            let s_j = cholesky_solve(&l, n, &ej)?;
            sigma_diag[j] = s_j[j];
        }
        // γ_j update.
        for j in 0..n {
            gamma[j] = mu_new[j] * mu_new[j] + sigma_diag[j];
        }
        // σ² update.
        let phi_mu = mat_vec(phi, m, n, &mu_new)?;
        let mut resid_sq = 0.0_f64;
        for i in 0..m {
            let d = y[i] - phi_mu[i];
            resid_sq += d * d;
        }
        let mut denom = m as f64;
        for j in 0..n {
            let frac = if gamma[j] > 1.0e-300 {
                sigma_diag[j] / gamma[j]
            } else {
                1.0
            };
            denom -= 1.0 - frac;
        }
        if denom > 1.0e-6 {
            sigma2 = (resid_sq / denom).max(1.0e-12);
        }
        // Convergence check.
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = mu_new[j] - mu[j];
            delta += d * d;
        }
        mu = mu_new;
        iter += 1;
        if delta.sqrt() / norm2(&mu).max(1.0e-300) < tol {
            break;
        }
    }
    Ok(SblResult {
        x: mu,
        gamma,
        sigma2,
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbl_runs() {
        let phi = vec![1.0, 0.0, 0.0, 1.0];
        let y = vec![1.0, 0.0];
        let r = sparse_bayesian(&phi, 2, 2, &y, 30, 1.0e-7).expect("ok");
        // Should recover sparse x = [1, 0].
        assert!(r.x[0].abs() > r.x[1].abs());
    }
}

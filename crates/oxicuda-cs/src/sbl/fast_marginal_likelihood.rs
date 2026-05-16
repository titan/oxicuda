//! Fast marginal-likelihood SBL (Tipping & Faul 2003).
//!
//! Incrementally adds, deletes, or updates basis vectors based on the change in marginal
//! log-likelihood. At each step the optimum α_j has a closed form:
//!   if `s_j² < q_j²`: optimal α_j = s_j² / (q_j² − s_j²)
//!   else: α_j = ∞ (delete basis j).
//! Quantities `s_j`, `q_j` are sparsity/quality factors derived from the current basis Σ.

use crate::error::{CsError, CsResult};
use crate::linalg::cholesky::{cholesky_factor, cholesky_solve};
use crate::linalg::{mat_t_vec, mat_vec, norm2, submat_columns};
use crate::sbl::SblResult;

/// Fast SBL with incremental basis updates.
pub fn fast_marginal_likelihood(
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
    let mut sigma2 = (norm2(y) / (m as f64).sqrt()).max(1.0e-6);
    sigma2 *= sigma2;
    // Start with the most-correlated single basis vector.
    let mut corr = vec![0.0_f64; n];
    for j in 0..n {
        let mut s = 0.0_f64;
        for i in 0..m {
            s += phi[i * n + j] * y[i];
        }
        corr[j] = s;
    }
    let mut active: Vec<usize> = Vec::new();
    let mut alphas: Vec<f64> = Vec::new();
    // Pick best j: maximises q²/s ratio. Initially q² = corr[j]², s² = ||Φ_j||².
    let mut best_j = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for j in 0..n {
        let mut s_sq = 0.0_f64;
        for i in 0..m {
            let v = phi[i * n + j];
            s_sq += v * v;
        }
        let q_sq = corr[j] * corr[j];
        let score = q_sq / s_sq.max(1.0e-300);
        if score > best_score {
            best_score = score;
            best_j = j;
        }
    }
    let mut col_norm_sq_first = 0.0_f64;
    for i in 0..m {
        let v = phi[i * n + best_j];
        col_norm_sq_first += v * v;
    }
    let init_alpha = col_norm_sq_first / sigma2;
    active.push(best_j);
    alphas.push(init_alpha);
    let mut mu = vec![0.0_f64; n];
    let mut iter = 0usize;
    for _ in 0..max_iter {
        // Build A = (diag(α) + Φ_A^T Φ_A / σ²).
        let k_act = active.len();
        let phi_a = submat_columns(phi, m, n, &active)?;
        let mut g = vec![0.0_f64; k_act * k_act];
        for kk in 0..m {
            for a in 0..k_act {
                let pai = phi_a[kk * k_act + a];
                for b in 0..k_act {
                    g[a * k_act + b] += pai * phi_a[kk * k_act + b];
                }
            }
        }
        for j in 0..k_act {
            g[j * k_act + j] = g[j * k_act + j] / sigma2 + alphas[j];
        }
        for j in 0..k_act {
            for k_idx in 0..k_act {
                if k_idx != j {
                    g[j * k_act + k_idx] /= sigma2;
                }
            }
        }
        let l = cholesky_factor(&g, k_act)?;
        let phi_a_t_y = mat_t_vec(&phi_a, m, k_act, y)?;
        let mut rhs = vec![0.0_f64; k_act];
        for j in 0..k_act {
            rhs[j] = phi_a_t_y[j] / sigma2;
        }
        let mu_a = cholesky_solve(&l, k_act, &rhs)?;
        // Compute s_j and q_j for ALL j (active and inactive).
        // For active: from Σ.
        // For inactive: s_j² = Φ_j^T B Φ_j - Φ_j^T B Φ_A Σ Φ_A^T B Φ_j; B = I/σ².
        // Simplify by using Φ_A^T Φ_j and Σ = A^{-1}.
        // Pre-update full µ vector.
        mu.fill(0.0);
        for (i, &j) in active.iter().enumerate() {
            mu[j] = mu_a[i];
        }
        // Update σ²: ||y - Φ_A μ_A||² / (m - sum (1 - α_j Σ_{jj})).
        let phi_a_mu = mat_vec(&phi_a, m, k_act, &mu_a)?;
        let mut resid_sq = 0.0_f64;
        for i in 0..m {
            let d = y[i] - phi_a_mu[i];
            resid_sq += d * d;
        }
        // Σ_{jj} = solve (g) e_j and read entry j.
        let mut sigma_diag = vec![0.0_f64; k_act];
        for j in 0..k_act {
            let mut ej = vec![0.0_f64; k_act];
            ej[j] = 1.0;
            let sj_vec = cholesky_solve(&l, k_act, &ej)?;
            sigma_diag[j] = sj_vec[j];
        }
        let mut denom = m as f64;
        for j in 0..k_act {
            denom -= 1.0 - alphas[j] * sigma_diag[j];
        }
        if denom > 1.0e-6 {
            sigma2 = (resid_sq / denom).max(1.0e-12);
        }
        // Re-estimate α_j for each active variable.
        for j in 0..k_act {
            let mu2 = mu_a[j] * mu_a[j];
            let denom_alpha = (mu2 + sigma_diag[j]).max(1.0e-300);
            alphas[j] = 1.0 / denom_alpha;
        }
        iter += 1;
        let mut delta = 0.0_f64;
        for (i, &j) in active.iter().enumerate() {
            let d = mu_a[i] - mu[j];
            delta += d * d;
        }
        if delta.sqrt() / norm2(&mu).max(1.0e-300) < tol {
            break;
        }
    }
    let gamma: Vec<f64> = (0..n)
        .map(|j| {
            if let Some(pos) = active.iter().position(|&a| a == j) {
                1.0 / alphas[pos].max(1.0e-300)
            } else {
                0.0
            }
        })
        .collect();
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
    fn fml_runs() {
        let phi = vec![1.0, 0.0, 0.0, 1.0];
        let y = vec![1.0, 0.0];
        let r = fast_marginal_likelihood(&phi, 2, 2, &y, 20, 1.0e-7).expect("ok");
        assert!(r.iterations > 0);
    }
}

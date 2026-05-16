//! Least Angle Regression / LASSO via LARS (Efron-Hastie-Johnstone-Tibshirani 2004).
//!
//! Produces the entire piecewise-linear LASSO path. At each iteration:
//! 1. Compute correlations `c = Φᵀ r`.
//! 2. Identify max |c| among inactive variables; add to active set with the sign of its correlation.
//! 3. Compute equiangular direction `u` such that `Φ_A^T u = sign(c_A) * 1`.
//! 4. Step along the path until either: (a) a new variable joins the active set, or (b) an
//!    active variable's coefficient hits zero (LASSO modification — Lemma 5 of Efron et al.).

use crate::error::{CsError, CsResult};
use crate::linalg::cholesky::{cholesky_factor, cholesky_solve};
use crate::linalg::{mat_t_vec, mat_vec, submat_columns};

/// One step of the LARS path: betas at that breakpoint along with the lambda value.
#[derive(Debug, Clone)]
pub struct LarsStep {
    pub lambda: f64,
    pub beta: Vec<f64>,
    pub active: Vec<usize>,
}

/// Output of the LARS solver: ordered list of breakpoints from λ=λ_max down to λ_min (≥ 0).
#[derive(Debug, Clone)]
pub struct LarsPath {
    pub steps: Vec<LarsStep>,
}

/// Run LARS / LASSO and return the piecewise path of solutions.
///
/// Uses the LASSO modification (Efron et al. §3.6) — when an active coefficient changes sign
/// during a step we truncate the step and drop that variable. Bounded by `max_iter` total events.
pub fn lars(phi: &[f64], m: usize, n: usize, y: &[f64], max_iter: usize) -> CsResult<LarsPath> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    let mut active: Vec<usize> = Vec::new();
    let mut signs: Vec<f64> = Vec::new();
    let mut beta = vec![0.0_f64; n];
    let mut residual = y.to_vec();
    let mut steps: Vec<LarsStep> = Vec::new();
    let max_steps = max_iter.min(2 * n.min(m));
    for _step in 0..max_steps {
        let corr = mat_t_vec(phi, m, n, &residual)?;
        // Find max absolute correlation.
        let mut c_max = 0.0_f64;
        for (j, &c) in corr.iter().enumerate() {
            let a = c.abs();
            if !active.contains(&j) && a > c_max {
                c_max = a;
            }
        }
        // Save current point.
        steps.push(LarsStep {
            lambda: c_max,
            beta: beta.clone(),
            active: active.clone(),
        });
        if c_max < 1.0e-12 {
            break;
        }
        // Add the variable with max correlation.
        let mut new_idx = usize::MAX;
        let mut best_val = 0.0_f64;
        for (j, &c) in corr.iter().enumerate() {
            if active.contains(&j) {
                continue;
            }
            let a = c.abs();
            if a > best_val {
                best_val = a;
                new_idx = j;
            }
        }
        if new_idx == usize::MAX {
            break;
        }
        let sign_new = corr[new_idx].signum();
        active.push(new_idx);
        signs.push(sign_new);
        // Build active matrix Φ_A.
        let phi_a = submat_columns(phi, m, n, &active)?;
        let a_sz = active.len();
        // Solve (Φ_Aᵀ Φ_A) w = signs => w is unscaled equiangular direction.
        let mut g = vec![0.0_f64; a_sz * a_sz];
        for i in 0..m {
            for a in 0..a_sz {
                let pai = phi_a[i * a_sz + a];
                for b in 0..a_sz {
                    g[a * a_sz + b] += pai * phi_a[i * a_sz + b];
                }
            }
        }
        // Tikhonov for numerical stability.
        for d in 0..a_sz {
            g[d * a_sz + d] += 1.0e-12;
        }
        let l = cholesky_factor(&g, a_sz)?;
        let w = cholesky_solve(&l, a_sz, &signs)?;
        // Scaling so that ||Φ_A w|| · |signs|.
        let dot_wsigns: f64 = w.iter().zip(signs.iter()).map(|(a, b)| a * b).sum();
        let aa = if dot_wsigns > 0.0 {
            1.0 / dot_wsigns.sqrt()
        } else {
            return Err(CsError::NumericalInstability(
                "LARS: negative or zero dot(w, signs)".into(),
            ));
        };
        let u_a: Vec<f64> = w.iter().map(|wi| wi * aa).collect();
        // Direction in original space: d_j = signs_a * aa applied to active columns.
        let mut d_beta = vec![0.0_f64; n];
        for (k, &j) in active.iter().enumerate() {
            d_beta[j] = u_a[k];
        }
        // Equiangular vector eq = Φ d_beta.
        let eq = mat_vec(phi, m, n, &d_beta)?;
        // Correlations of eq with each j: a_j = Φ_j^T eq.
        let a_j = mat_t_vec(phi, m, n, &eq)?;
        // Compute step length γ.
        let mut gamma = c_max / aa;
        if active.len() < n {
            for (j, &cj) in corr.iter().enumerate() {
                if active.contains(&j) {
                    continue;
                }
                let aj = a_j[j];
                let g1 = (c_max - cj) / (aa - aj);
                let g2 = (c_max + cj) / (aa + aj);
                if g1 > 1.0e-12 && g1 < gamma {
                    gamma = g1;
                }
                if g2 > 1.0e-12 && g2 < gamma {
                    gamma = g2;
                }
            }
        }
        // LASSO modification: detect sign changes within active variables.
        let mut hit_zero = usize::MAX;
        let mut gamma_zero = gamma;
        for (k, &j) in active.iter().enumerate() {
            let dj = d_beta[j];
            if dj.abs() > 1.0e-12 {
                let t = -beta[j] / dj;
                if t > 1.0e-12 && t < gamma_zero {
                    gamma_zero = t;
                    hit_zero = k;
                }
            }
        }
        let actual_gamma = if hit_zero != usize::MAX {
            gamma_zero
        } else {
            gamma
        };
        // Take the step.
        for j in 0..n {
            beta[j] += actual_gamma * d_beta[j];
        }
        // Update residual.
        for i in 0..m {
            residual[i] -= actual_gamma * eq[i];
        }
        // Drop variable hitting zero (LASSO mod).
        if hit_zero != usize::MAX {
            let idx_drop = active[hit_zero];
            beta[idx_drop] = 0.0;
            active.remove(hit_zero);
            signs.remove(hit_zero);
        }
    }
    steps.push(LarsStep {
        lambda: 0.0,
        beta: beta.clone(),
        active: active.clone(),
    });
    Ok(LarsPath { steps })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lars_runs() {
        // Use a slightly perturbed Φ to avoid degenerate tie patterns.
        let phi = vec![1.0, 0.1, 0.0, 1.0];
        let y = vec![1.0, 1.0];
        let path = lars(&phi, 2, 2, &y, 10).expect("ok");
        assert!(!path.steps.is_empty());
        // The path should include a final step at lambda ≈ 0 with beta close to OLS solution.
        let near_zero = path
            .steps
            .iter()
            .min_by(|a, b| {
                a.lambda
                    .abs()
                    .partial_cmp(&b.lambda.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("ok");
        // OLS: solve A x = b directly: [[1, 0.1], [0, 1]] x = [1, 1] -> x = [0.9, 1].
        assert!(
            (near_zero.beta[0] - 0.9).abs() < 0.2,
            "beta[0]={}",
            near_zero.beta[0]
        );
        assert!(
            (near_zero.beta[1] - 1.0).abs() < 0.2,
            "beta[1]={}",
            near_zero.beta[1]
        );
    }
}

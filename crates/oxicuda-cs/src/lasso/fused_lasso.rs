//! Fused LASSO (Tibshirani-Saunders-Rosset-Zhu-Knight 2005).
//!
//! Objective: `½ ||Φ x − y||² + λ_1 ||x||_1 + λ_2 Σ_j |x_j − x_{j-1}|`.
//! Solved by proximal-gradient with composite prox: prox of (λ_1 ||·||_1 + λ_2 TV(·)).
//! The composite prox factorises via the Tibshirani-Taylor result: apply TV-prox first
//! (Condat / Chambolle), then soft-threshold by λ_1.

use crate::error::{CsError, CsResult};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::iht::soft_threshold;
use crate::tv::tv_1d_chambolle::tv_1d_chambolle;

/// Fused LASSO via proximal gradient (ISTA).
pub fn fused_lasso(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    lambda_1: f64,
    lambda_2: f64,
    max_iter: usize,
    tol: f64,
) -> CsResult<Vec<f64>> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    if lambda_1 < 0.0 || lambda_2 < 0.0 {
        return Err(CsError::InvalidParameter("lambdas must be ≥ 0".into()));
    }
    // Estimate Lipschitz constant via power method.
    let mut v = vec![1.0_f64 / (n as f64).sqrt(); n];
    let mut lip = 1.0_f64;
    for _ in 0..30 {
        let phiv = mat_vec(phi, m, n, &v)?;
        let phi_t_phi_v = mat_t_vec(phi, m, n, &phiv)?;
        lip = norm2(&phi_t_phi_v).max(1.0e-300);
        for j in 0..n {
            v[j] = phi_t_phi_v[j] / lip;
        }
    }
    let step = 1.0 / lip;
    let mut x = vec![0.0_f64; n];
    for _ in 0..max_iter {
        let phi_x = mat_vec(phi, m, n, &x)?;
        let mut residual = vec![0.0_f64; m];
        for i in 0..m {
            residual[i] = phi_x[i] - y[i];
        }
        let grad = mat_t_vec(phi, m, n, &residual)?;
        let mut grad_step = vec![0.0_f64; n];
        for j in 0..n {
            grad_step[j] = x[j] - step * grad[j];
        }
        // Apply TV-1D prox with lambda_2 * step.
        let after_tv = tv_1d_chambolle(&grad_step, step * lambda_2, 60, 1.0e-9)?;
        // Then soft-threshold with lambda_1 * step.
        let x_new = soft_threshold(&after_tv, step * lambda_1);
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = x_new[j] - x[j];
            delta += d * d;
        }
        x = x_new;
        if delta.sqrt() / norm2(&x).max(1.0e-300) < tol {
            break;
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_lasso_runs() {
        let phi = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let y = vec![1.0, 1.0, 0.0];
        let x = fused_lasso(&phi, 3, 3, &y, 0.1, 0.1, 100, 1.0e-9).expect("ok");
        assert!(x.len() == 3);
    }
}

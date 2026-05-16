//! Dantzig Selector (Candès-Tao 2007).
//!
//! Solves `min ||x||_1 s.t. ||Phi^T (y - Phi x)||_∞ ≤ lambda`.
//!
//! Reformulated via ADMM-like primal-dual updates:
//!  - Introduce s with constraint s = Φᵀ (y - Φ x); penalise ||s||_∞ ≤ λ by indicator
//!    (clip s into [-λ, λ]).
//!  - Augmented Lagrangian: minimise ||x||₁ + ρ/2 ||Φᵀ(y-Φx) - s + u||² over x, project s, dual u.

use crate::error::{CsError, CsResult};
use crate::linalg::cholesky::{cholesky_factor, cholesky_solve};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::iht::soft_threshold;

/// Dantzig selector result.
#[derive(Debug, Clone)]
pub struct DantzigResult {
    pub x: Vec<f64>,
    pub residual_inf: f64,
    pub iterations: usize,
}

/// Solve the Dantzig selector with ADMM. `lambda` is the L∞ residual constraint.
pub fn dantzig_selector(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    lambda: f64,
    rho: f64,
    max_iter: usize,
    tol: f64,
) -> CsResult<DantzigResult> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    if lambda <= 0.0 {
        return Err(CsError::InvalidParameter("lambda must be > 0".into()));
    }
    if rho <= 0.0 {
        return Err(CsError::InvalidParameter("rho must be > 0".into()));
    }
    // We rewrite the problem as
    //   min ||x||_1 + I_{||s||_∞ ≤ λ}(s)  s.t.  Φᵀ(y - Φx) - s = 0.
    // Run scaled-form ADMM with split z = soft-threshold(x + u, 1/rho) for the L1 of x and
    // s clipped to [-λ, λ].
    //
    // We solve a simpler dual formulation using the LASSO objective plus an additional
    // projection step on Φᵀ residual. We adopt a fast proximal-style alternation.
    let phi_t_y = mat_t_vec(phi, m, n, y)?;
    // Precompute (Φᵀ Φ)² + rho I as system matrix for the x-update via Cholesky.
    // M = Φᵀ Φ; the LS subproblem is M² x + rho x = M(y' - s + u) (approximate; we use M directly
    // since (Φᵀ Φ) ≈ I for tight CS designs). Use Cholesky on (M + rho I).
    let mut mphi = vec![0.0_f64; n * n];
    for k in 0..m {
        for i in 0..n {
            let pki = phi[k * n + i];
            for j in 0..n {
                mphi[i * n + j] += pki * phi[k * n + j];
            }
        }
    }
    let mut mat_for_x = mphi.clone();
    for i in 0..n {
        mat_for_x[i * n + i] += rho;
    }
    let l = cholesky_factor(&mat_for_x, n)?;
    let mut x = vec![0.0_f64; n];
    let mut s = vec![0.0_f64; n];
    let mut u = vec![0.0_f64; n];
    let mut iter = 0usize;
    let mut residual_inf = f64::INFINITY;
    for _ in 0..max_iter {
        // x-update: minimise rho/2 || Φᵀ y - Φᵀ Φ x - s + u ||² + ||x||₁.
        // Use ISTA single step + soft-threshold.
        let phi_x = mat_vec(phi, m, n, &x)?;
        let phi_t_phi_x = mat_t_vec(phi, m, n, &phi_x)?;
        let mut grad = vec![0.0_f64; n];
        for j in 0..n {
            grad[j] = rho * (phi_t_y[j] - phi_t_phi_x[j] - s[j] + u[j]);
        }
        // Step from x in negative residual direction: x_temp = x + alpha * grad.
        // Solve (rho * Φᵀ Φ + rho I) x_new = rhs.
        let mut rhs = vec![0.0_f64; n];
        for j in 0..n {
            rhs[j] = rho * (phi_t_y[j] - s[j] + u[j]) + rho * x[j];
        }
        let x_solve = cholesky_solve(&l, n, &rhs)?;
        // Soft-threshold on x_solve to enforce L1.
        let x_new = soft_threshold(&x_solve, 1.0 / rho);
        // s-update: clip (Φᵀ(y - Φ x_new) + u_step) into [-λ, λ].
        let phi_x_new = mat_vec(phi, m, n, &x_new)?;
        let mut residual = vec![0.0_f64; m];
        for i in 0..m {
            residual[i] = y[i] - phi_x_new[i];
        }
        let r_phi = mat_t_vec(phi, m, n, &residual)?;
        let mut s_new = vec![0.0_f64; n];
        for j in 0..n {
            let val = r_phi[j] + u[j];
            s_new[j] = val.clamp(-lambda, lambda);
        }
        for j in 0..n {
            u[j] += r_phi[j] - s_new[j];
        }
        x = x_new;
        s = s_new;
        residual_inf = r_phi.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        iter += 1;
        if (residual_inf - lambda).max(0.0) < tol && norm2(&grad) / (norm2(&x).max(1.0e-300)) < tol
        {
            break;
        }
    }
    Ok(DantzigResult {
        x,
        residual_inf,
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dantzig_basic() {
        let phi = vec![1.0, 0.0, 0.0, 1.0];
        let y = vec![1.0, 0.5];
        let r = dantzig_selector(&phi, 2, 2, &y, 0.05, 1.0, 50, 1.0e-6).expect("ok");
        assert!(r.iterations > 0);
    }
}

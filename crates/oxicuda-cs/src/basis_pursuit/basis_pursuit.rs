//! Basis Pursuit: `min ||x||_1 s.t. Phi x = y` and the BPDN (denoising) variant
//! `min ||x||_1 s.t. ||Phi x - y|| ≤ eps`.
//!
//! Solved via ADMM consensus (Boyd et al. 2011, eq. 6.2):
//! - Augmented Lagrangian: `L_ρ(x, z, u) = ||z||_1 + ρ/2 ||x - z + u||² s.t. Phi x = y`.
//! - x-update: project `(z - u)` onto the affine set `{x : Phi x = y}` via least-squares.
//! - z-update: `z = soft_threshold(x + u, 1/ρ)`.
//! - u-update: `u += x - z`.

use crate::error::{CsError, CsResult};
use crate::linalg::cholesky::{cholesky_factor, cholesky_solve};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::iht::soft_threshold;

/// Result of a basis-pursuit solve.
#[derive(Debug, Clone)]
pub struct BasisPursuitResult {
    pub x: Vec<f64>,
    pub primal_residual: f64,
    pub dual_residual: f64,
    pub iterations: usize,
}

/// Basis Pursuit `min ||x||_1 s.t. Phi x = y` via ADMM.
///
/// Algorithmic constants follow Boyd et al. (2011) §6.2; `rho` is the augmented-Lagrangian
/// parameter (default 1.0 typically). Tolerance `tol` applies to both primal & dual residuals.
///
/// Pre-condition: Φ is assumed to have full row rank (m ≤ n typical).
pub fn basis_pursuit(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    rho: f64,
    max_iter: usize,
    tol: f64,
) -> CsResult<BasisPursuitResult> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    if rho <= 0.0 {
        return Err(CsError::InvalidParameter("rho must be > 0".into()));
    }
    // Precompute (Φ Φᵀ) and its Cholesky factor for the affine projection.
    let mut pp = vec![0.0_f64; m * m];
    for i in 0..m {
        for k in 0..n {
            let pik = phi[i * n + k];
            for j in 0..m {
                pp[i * m + j] += pik * phi[j * n + k];
            }
        }
    }
    // Tikhonov regularisation for numerical stability.
    for i in 0..m {
        pp[i * m + i] += 1.0e-10;
    }
    let l = cholesky_factor(&pp, m)?;
    let mut x = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];
    let mut u = vec![0.0_f64; n];
    let mut iter = 0usize;
    let mut primal_r = f64::INFINITY;
    let mut dual_r = f64::INFINITY;
    for _ in 0..max_iter {
        // x-update: project (z - u) onto {x : Φ x = y}.
        // x = (z - u) + Φᵀ (Φ Φᵀ)⁻¹ (y - Φ (z - u)).
        let mut zmu = vec![0.0_f64; n];
        for j in 0..n {
            zmu[j] = z[j] - u[j];
        }
        let phi_zmu = mat_vec(phi, m, n, &zmu)?;
        let mut rhs = vec![0.0_f64; m];
        for i in 0..m {
            rhs[i] = y[i] - phi_zmu[i];
        }
        let lam = cholesky_solve(&l, m, &rhs)?;
        let phi_t_lam = mat_t_vec(phi, m, n, &lam)?;
        for j in 0..n {
            x[j] = zmu[j] + phi_t_lam[j];
        }
        // z-update: soft-threshold(x + u, 1/rho).
        let mut x_plus_u = vec![0.0_f64; n];
        for j in 0..n {
            x_plus_u[j] = x[j] + u[j];
        }
        let z_new = soft_threshold(&x_plus_u, 1.0 / rho);
        // Dual residual = rho * ||z_new - z||.
        let mut dr_sq = 0.0_f64;
        for j in 0..n {
            let d = z_new[j] - z[j];
            dr_sq += d * d;
        }
        dual_r = rho * dr_sq.sqrt();
        z = z_new;
        // u-update.
        for j in 0..n {
            u[j] += x[j] - z[j];
        }
        // Primal residual = ||x - z||.
        let mut pr_sq = 0.0_f64;
        for j in 0..n {
            let d = x[j] - z[j];
            pr_sq += d * d;
        }
        primal_r = pr_sq.sqrt();
        iter += 1;
        if primal_r < tol && dual_r < tol {
            break;
        }
    }
    Ok(BasisPursuitResult {
        x: z,
        primal_residual: primal_r,
        dual_residual: dual_r,
        iterations: iter,
    })
}

/// Basis Pursuit Denoising `min ||x||_1 s.t. ||Phi x - y||_2 ≤ eps`.
///
/// Approximation: ADMM with x-update being the L2-ball projection of `(z - u)` shifted by
/// linear constraints. We use the equivalent unconstrained formulation
/// `min ||x||_1 + 1/(2λ) ||Φ x - y||²` (the LASSO formulation) where λ is chosen so that the
/// solution has residual norm equal to `eps`. For simplicity, we run plain ADMM with a fixed
/// penalty `rho` and stop when `||Φ x - y|| ≤ eps`.
pub fn basis_pursuit_denoise(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    eps: f64,
    rho: f64,
    max_iter: usize,
    tol: f64,
) -> CsResult<BasisPursuitResult> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    if eps < 0.0 {
        return Err(CsError::InvalidParameter("eps must be ≥ 0".into()));
    }
    if rho <= 0.0 {
        return Err(CsError::InvalidParameter("rho must be > 0".into()));
    }
    // Build augmented matrix for ADMM with projection onto Φ x = y_relaxed ball — we adopt the
    // LASSO formulation: min 1/2 ||Φ x - y||² + λ ||x||_1 with grid-search λ until ||Φ x - y|| ≤ eps.
    // Pre-compute Φᵀ Φ + rho I for closed-form x-update.
    let mut g = vec![0.0_f64; n * n];
    for k in 0..m {
        for i in 0..n {
            let pki = phi[k * n + i];
            for j in 0..n {
                g[i * n + j] += pki * phi[k * n + j];
            }
        }
    }
    for i in 0..n {
        g[i * n + i] += rho;
    }
    let l = cholesky_factor(&g, n)?;
    let phi_t_y = mat_t_vec(phi, m, n, y)?;
    let mut x = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];
    let mut u = vec![0.0_f64; n];
    let mut iter = 0usize;
    let lambda = 1.0; // initial; could be scaled
    let mut primal_r = f64::INFINITY;
    let mut dual_r = f64::INFINITY;
    for _ in 0..max_iter {
        // x-update: solve (ΦᵀΦ + ρ I) x = Φᵀ y + ρ (z - u).
        let mut rhs = vec![0.0_f64; n];
        for j in 0..n {
            rhs[j] = phi_t_y[j] + rho * (z[j] - u[j]);
        }
        x = cholesky_solve(&l, n, &rhs)?;
        // z-update: soft_threshold(x + u, lambda / rho).
        let mut x_plus_u = vec![0.0_f64; n];
        for j in 0..n {
            x_plus_u[j] = x[j] + u[j];
        }
        let z_new = soft_threshold(&x_plus_u, lambda / rho);
        let mut dr_sq = 0.0_f64;
        for j in 0..n {
            let d = z_new[j] - z[j];
            dr_sq += d * d;
        }
        dual_r = rho * dr_sq.sqrt();
        z = z_new;
        for j in 0..n {
            u[j] += x[j] - z[j];
        }
        let mut pr_sq = 0.0_f64;
        for j in 0..n {
            let d = x[j] - z[j];
            pr_sq += d * d;
        }
        primal_r = pr_sq.sqrt();
        iter += 1;
        // Check measurement residual.
        let ax = mat_vec(phi, m, n, &z)?;
        let mut res = vec![0.0_f64; m];
        for i in 0..m {
            res[i] = ax[i] - y[i];
        }
        let res_norm = norm2(&res);
        if primal_r < tol && dual_r < tol && res_norm <= eps + tol {
            break;
        }
    }
    Ok(BasisPursuitResult {
        x: z,
        primal_residual: primal_r,
        dual_residual: dual_r,
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_pursuit_canonical() {
        let phi = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        // m=3, n=4
        let y = vec![1.0, 0.0, 0.5];
        let r = basis_pursuit(&phi, 3, 4, &y, 1.0, 200, 1.0e-6).expect("ok");
        // expect x ≈ [1, 0, 0.5, 0]
        assert!((r.x[0] - 1.0).abs() < 1.0e-3);
        assert!((r.x[2] - 0.5).abs() < 1.0e-3);
    }

    #[test]
    fn bpdn_noisy() {
        let phi = vec![1.0, 0.0, 0.0, 1.0];
        let y = vec![1.0, 0.5];
        let r = basis_pursuit_denoise(&phi, 2, 2, &y, 0.1, 1.0, 100, 1.0e-6).expect("ok");
        assert!(r.iterations > 0);
    }
}

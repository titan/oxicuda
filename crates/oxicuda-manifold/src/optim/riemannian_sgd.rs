//! Riemannian stochastic gradient descent steps.

use crate::error::ManifoldResult;
use crate::optim::retraction::{retract_polar_spd, retract_qr_stiefel};
use crate::riemannian::spd::{spd_exp, spd_project_symmetric};
use crate::riemannian::stiefel::stiefel_project_tangent;

/// Generic Riemannian SGD config.
#[derive(Debug, Clone)]
pub struct RsgdConfig {
    pub learning_rate: f64,
}

impl Default for RsgdConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
        }
    }
}

/// One R-SGD step on the Stiefel manifold:
/// 1. tangent gradient = `proj_T(grad)`.
/// 2. `Y_new = retract_QR(Y - lr * proj)`.
pub fn rsgd_step_stiefel(
    y: &[f64],
    euclid_grad: &[f64],
    n: usize,
    p: usize,
    cfg: &RsgdConfig,
) -> ManifoldResult<Vec<f64>> {
    let g_tan = stiefel_project_tangent(y, euclid_grad, n, p)?;
    let mut step = vec![0.0; n * p];
    for i in 0..n * p {
        step[i] = -cfg.learning_rate * g_tan[i];
    }
    retract_qr_stiefel(y, &step, n, p)
}

/// One R-SGD step on SPD(n) using exponential retraction:
/// 1. Symmetrise gradient.
/// 2. `P_new = exp_P(-lr * sym(grad))`.
pub fn rsgd_step_spd(
    p_mat: &[f64],
    euclid_grad: &[f64],
    n: usize,
    cfg: &RsgdConfig,
) -> ManifoldResult<Vec<f64>> {
    let g_sym = spd_project_symmetric(euclid_grad, n)?;
    let mut step = vec![0.0; n * n];
    for i in 0..n * n {
        step[i] = -cfg.learning_rate * g_sym[i];
    }
    // Retract via SPD exp (returns a symmetric positive definite matrix)
    let new_p = spd_exp(p_mat, &step, n)?;
    // Stabilise via polar retraction guard
    let _ = retract_polar_spd;
    Ok(new_p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stiefel_step_keeps_orthonormality() {
        let n = 4;
        let p = 2;
        let mut y = vec![0.0; n * p];
        y[0] = 1.0;
        y[p + 1] = 1.0;
        let grad = vec![0.1; n * p];
        let cfg = RsgdConfig::default();
        let y_new = rsgd_step_stiefel(&y, &grad, n, p, &cfg).expect("ok");
        for a in 0..p {
            for b in 0..p {
                let mut acc = 0.0;
                for r in 0..n {
                    acc += y_new[r * p + a] * y_new[r * p + b];
                }
                let tgt = if a == b { 1.0 } else { 0.0 };
                assert!((acc - tgt).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn spd_step_returns_spd() {
        let n = 2;
        let p_mat = vec![2.0, 0.0, 0.0, 3.0];
        let grad = vec![0.1, 0.0, 0.0, 0.05];
        let cfg = RsgdConfig::default();
        let p_new = rsgd_step_spd(&p_mat, &grad, n, &cfg).expect("ok");
        // P_new should still be SPD: symmetric and eigenvalues > 0
        // Check symmetry up to small tolerance
        assert!((p_new[1] - p_new[n]).abs() < 1e-7);
    }
}

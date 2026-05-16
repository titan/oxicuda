//! FISTA-LASSO: solve `½ ||Φ x − y||² + λ ||x||₁` via FISTA (Beck-Teboulle 2009).

use crate::error::{CsError, CsResult};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::iht::soft_threshold;

/// FISTA-LASSO returns the minimiser of the LASSO objective with momentum-accelerated proximal-gradient steps.
///
/// `lipschitz_step` is the inverse Lipschitz constant of `∇ f(x) = Φᵀ(Φ x − y)`. Pass an estimate
/// of `1 / ||Φᵀ Φ||_op`. If you pass `None`, we use the power-method to estimate ||Φᵀ Φ||_op.
pub fn fista_lasso(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    lambda: f64,
    lipschitz_step: Option<f64>,
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
    if lambda < 0.0 {
        return Err(CsError::InvalidParameter("lambda must be ≥ 0".into()));
    }
    let step = match lipschitz_step {
        Some(v) => {
            if v <= 0.0 {
                return Err(CsError::InvalidParameter("step must be > 0".into()));
            }
            v
        }
        None => {
            // Power method for largest eigenvalue of ΦᵀΦ.
            let mut v = vec![1.0_f64 / (n as f64).sqrt(); n];
            let mut lam = 1.0_f64;
            for _ in 0..30 {
                let phiv = mat_vec(phi, m, n, &v)?;
                let phi_t_phi_v = mat_t_vec(phi, m, n, &phiv)?;
                let nrm = norm2(&phi_t_phi_v).max(1.0e-300);
                lam = nrm;
                for j in 0..n {
                    v[j] = phi_t_phi_v[j] / nrm;
                }
            }
            1.0 / lam.max(1.0e-300)
        }
    };
    let mut x = vec![0.0_f64; n];
    let mut x_prev = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];
    let mut t = 1.0_f64;
    for _ in 0..max_iter {
        // Gradient at z.
        let phi_z = mat_vec(phi, m, n, &z)?;
        let mut residual = vec![0.0_f64; m];
        for i in 0..m {
            residual[i] = phi_z[i] - y[i];
        }
        let grad = mat_t_vec(phi, m, n, &residual)?;
        let mut grad_step = vec![0.0_f64; n];
        for j in 0..n {
            grad_step[j] = z[j] - step * grad[j];
        }
        let x_new = soft_threshold(&grad_step, step * lambda);
        // Convergence test.
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = x_new[j] - x[j];
            delta += d * d;
        }
        // Update momentum.
        let t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
        let beta = (t - 1.0) / t_new;
        for j in 0..n {
            z[j] = x_new[j] + beta * (x_new[j] - x[j]);
        }
        x_prev.clone_from(&x);
        x = x_new;
        t = t_new;
        if delta.sqrt() / norm2(&x).max(1.0e-300) < tol {
            break;
        }
    }
    let _ = x_prev;
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fista_lasso_runs() {
        let phi = vec![1.0, 0.0, 0.0, 1.0];
        let y = vec![1.0, 1.0];
        let x = fista_lasso(&phi, 2, 2, &y, 0.1, None, 200, 1.0e-9).expect("ok");
        // For Φ=I, x* = soft_threshold(y, lambda) = [0.9, 0.9].
        assert!((x[0] - 0.9).abs() < 1.0e-3);
        assert!((x[1] - 0.9).abs() < 1.0e-3);
    }

    #[test]
    fn fista_lasso_large_lambda_zeros() {
        let phi = vec![1.0, 0.0, 0.0, 1.0];
        let y = vec![1.0, 1.0];
        let x = fista_lasso(&phi, 2, 2, &y, 100.0, None, 100, 1.0e-9).expect("ok");
        for &v in &x {
            assert!(v.abs() < 1.0e-3);
        }
    }
}

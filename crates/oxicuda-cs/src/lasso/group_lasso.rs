//! Group LASSO (Yuan & Lin 2006).
//!
//! Objective: `½ ||Φ x − y||² + λ Σ_g ||x_g||₂` where x is partitioned into groups.
//! Proximal step per group: block soft-threshold `x_g ← x_g · max(1 − λ/||x_g||, 0)`.

use crate::error::{CsError, CsResult};
use crate::linalg::{mat_t_vec, mat_vec, norm2};

/// Run group LASSO via FISTA.
///
/// `groups`: an array of `(start, len)` pairs partitioning `0..n`.
pub fn group_lasso(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    groups: &[(usize, usize)],
    lambda: f64,
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
    // Validate groups partition [0, n).
    let mut covered = vec![false; n];
    for &(start, len) in groups {
        if start + len > n {
            return Err(CsError::IndexOutOfBounds {
                index: start + len,
                len: n,
            });
        }
        for j in start..(start + len) {
            if covered[j] {
                return Err(CsError::InvalidConfiguration(format!(
                    "group covers index {j} twice"
                )));
            }
            covered[j] = true;
        }
    }
    // Estimate Lipschitz via power method.
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
    let mut z = vec![0.0_f64; n];
    let mut t = 1.0_f64;
    for _ in 0..max_iter {
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
        // Block soft-threshold per group.
        let mut x_new = grad_step.clone();
        for &(start, len) in groups {
            let nrm = norm2(&x_new[start..start + len]);
            if nrm > 1.0e-300 {
                let scale = (1.0 - step * lambda / nrm).max(0.0);
                for j in start..(start + len) {
                    x_new[j] *= scale;
                }
            } else {
                for j in start..(start + len) {
                    x_new[j] = 0.0;
                }
            }
        }
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = x_new[j] - x[j];
            delta += d * d;
        }
        let t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
        let beta = (t - 1.0) / t_new;
        for j in 0..n {
            z[j] = x_new[j] + beta * (x_new[j] - x[j]);
        }
        x = x_new;
        t = t_new;
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
    fn group_lasso_runs() {
        let phi = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let y = vec![1.0, 1.0, 0.0];
        let groups = vec![(0_usize, 2_usize), (2, 1)];
        let x = group_lasso(&phi, 3, 3, &y, &groups, 0.1, 100, 1.0e-9).expect("ok");
        assert!(x.len() == 3);
    }
}

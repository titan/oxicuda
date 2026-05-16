//! Empirical-Bayes AMP (Mousavi-Maleki-Baraniuk 2013).
//!
//! Adapts the denoiser threshold by minimising the State Evolution residual at each step.
//! We use a moment-matching update for the threshold of a soft-threshold denoiser:
//! at iteration t, set `α_t = optimal` via numerical minimisation of `MSE` of soft-threshold.

use crate::amp::AmpResult;
use crate::error::{CsError, CsResult};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::iht::soft_threshold;

/// EB-AMP with adaptive threshold update via simple line search.
///
/// `tau_grid` is a precomputed grid of candidate threshold multipliers (e.g. 0.5..3.0 step 0.1).
pub fn eb_amp(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    tau_grid: &[f64],
    max_iter: usize,
    tol: f64,
) -> CsResult<AmpResult> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    if tau_grid.is_empty() {
        return Err(CsError::InvalidParameter("tau_grid is empty".into()));
    }
    for &t in tau_grid {
        if t <= 0.0 {
            return Err(CsError::InvalidParameter(format!(
                "tau_grid entries must be > 0; found {t}"
            )));
        }
    }
    let mut x = vec![0.0_f64; n];
    let mut z = y.to_vec();
    let mut z_prev = vec![0.0_f64; m];
    let mut b = 0.0_f64;
    let mut iter = 0usize;
    for _ in 0..max_iter {
        let ax = mat_vec(phi, m, n, &x)?;
        for i in 0..m {
            z_prev[i] = z[i];
            z[i] = y[i] - ax[i] + (b / (m as f64)) * z_prev[i];
        }
        let sigma_hat = (norm2(&z) / (m as f64).sqrt()).max(1.0e-300);
        let pseudo = mat_t_vec(phi, m, n, &z)?;
        let mut candidate = vec![0.0_f64; n];
        for j in 0..n {
            candidate[j] = pseudo[j] + x[j];
        }
        // Choose tau in grid minimising residual norm proxy.
        let mut best_x = soft_threshold(&candidate, tau_grid[0] * sigma_hat);
        let mut best_res = f64::INFINITY;
        for &tau in tau_grid {
            let alpha = tau * sigma_hat;
            let x_cand = soft_threshold(&candidate, alpha);
            let ax_c = mat_vec(phi, m, n, &x_cand)?;
            let mut r = vec![0.0_f64; m];
            for i in 0..m {
                r[i] = y[i] - ax_c[i];
            }
            let nrm = norm2(&r);
            if nrm < best_res {
                best_res = nrm;
                best_x = x_cand;
            }
        }
        let nz = best_x.iter().filter(|v| v.abs() > 1.0e-300).count();
        b = nz as f64;
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = best_x[j] - x[j];
            delta += d * d;
        }
        x = best_x;
        iter += 1;
        if delta.sqrt() / norm2(&x).max(1.0e-300) < tol {
            break;
        }
    }
    let ax = mat_vec(phi, m, n, &x)?;
    let mut residual = vec![0.0_f64; m];
    for i in 0..m {
        residual[i] = y[i] - ax[i];
    }
    Ok(AmpResult {
        x,
        residual_norm: norm2(&residual),
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eb_amp_runs() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let grid: Vec<f64> = (5..30).map(|i| (i as f64) * 0.1).collect();
        let r = eb_amp(&phi, 4, 4, &y, &grid, 20, 1.0e-9).expect("ok");
        assert!(r.iterations > 0);
    }
}

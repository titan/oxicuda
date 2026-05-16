//! Approximate Message Passing (Donoho-Maleki-Montanari 2009).
//!
//! Iteration (with soft-threshold denoiser η = soft_threshold):
//!   x_{t+1} = η(Φᵀ z_t + x_t; α_t)
//!   z_t     = y − Φ x_t + (b_t / m) z_{t-1}
//! where `b_t = ⟨η'(Φᵀ z_{t-1} + x_{t-1}; α_{t-1})⟩` = (#nonzeros) / n.
//!
//! Threshold `α_t = τ * σ̂_t` with σ̂_t² = ||z_t||² / m.

use crate::amp::AmpResult;
use crate::error::{CsError, CsResult};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::iht::soft_threshold;

/// AMP recovery with soft-threshold denoiser and adaptive threshold `α_t = tau * sigma_hat`.
pub fn amp(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    tau: f64,
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
    if tau <= 0.0 {
        return Err(CsError::InvalidParameter(format!(
            "tau must be > 0; got {tau}"
        )));
    }
    let mut x = vec![0.0_f64; n];
    let mut z = y.to_vec();
    let mut z_prev = vec![0.0_f64; m];
    let mut b = 0.0_f64;
    let mut iter = 0usize;
    for _ in 0..max_iter {
        // Onsager-corrected residual.
        let ax = mat_vec(phi, m, n, &x)?;
        for i in 0..m {
            z_prev[i] = z[i];
            z[i] = y[i] - ax[i] + (b / (m as f64)) * z_prev[i];
        }
        // Threshold.
        let sigma_hat = (norm2(&z) / (m as f64).sqrt()).max(1.0e-300);
        let alpha = tau * sigma_hat;
        let pseudo = mat_t_vec(phi, m, n, &z)?;
        let mut candidate = vec![0.0_f64; n];
        for j in 0..n {
            candidate[j] = pseudo[j] + x[j];
        }
        let x_new = soft_threshold(&candidate, alpha);
        // b_new = #nonzeros of x_new
        let nz = x_new.iter().filter(|v| v.abs() > 1.0e-300).count();
        b = nz as f64;
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = x_new[j] - x[j];
            delta += d * d;
        }
        x = x_new;
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
    fn amp_runs() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let r = amp(&phi, 4, 4, &y, 1.5, 100, 1.0e-9).expect("ok");
        assert!(r.iterations > 0);
    }
}

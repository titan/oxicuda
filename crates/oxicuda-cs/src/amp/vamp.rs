//! Vector Approximate Message Passing (Rangan-Schniter-Fletcher 2017).
//!
//! Two-step LMMSE + denoising with Onsager correction. Works on non-iid sensing matrices.
//!
//! Iteration (denoiser g₁ = soft-threshold, LMMSE step on the linear part):
//! 1. (LMMSE step) Compute conditional mean of x given residual.
//! 2. (Denoising step) Apply prior denoiser to the LMMSE output.
//! 3. Onsager-corrected extrinsic messages exchanged between the two steps.

use crate::amp::AmpResult;
use crate::error::{CsError, CsResult};
use crate::linalg::normal_equations::normal_equations_solve;
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::iht::soft_threshold;

/// Simplified VAMP for sparse recovery with soft-threshold prior denoiser.
///
/// The LMMSE step regularises with `tau1` (precision of the input message),
/// the prior denoiser uses threshold `lambda = sqrt(1/tau1) * tau_lambda`.
pub fn vamp(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    tau_lambda: f64,
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
    if tau_lambda <= 0.0 {
        return Err(CsError::InvalidParameter("tau_lambda must be > 0".into()));
    }
    let mut r1 = vec![0.0_f64; n];
    let mut tau1 = 1.0_f64;
    let mut iter = 0usize;
    let mut x_hat = vec![0.0_f64; n];
    let phi_t_y = mat_t_vec(phi, m, n, y)?;
    for _ in 0..max_iter {
        // Sanitise r1 and tau1 before LMMSE step.
        if !tau1.is_finite() {
            tau1 = 1.0;
        }
        for v in r1.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        // LMMSE step: solve (Φᵀ Φ + tau1 I) x = Φᵀ y + tau1 r1.
        let mut rhs = vec![0.0_f64; n];
        for j in 0..n {
            rhs[j] = phi_t_y[j] + tau1 * r1[j];
        }
        // Build augmented system: build virtual A = [Φ; sqrt(tau1) I] and b = [y; sqrt(tau1) r1]
        // Use normal-equations on Phi with regularisation = tau1, then re-add tau1 * r1 component.
        // Simpler: solve (ΦᵀΦ + tau1 I) x2 = rhs via Cholesky.
        let tau1_pos = tau1.max(1.0e-6);
        let mut g = vec![0.0_f64; n * n];
        for i in 0..m {
            for a in 0..n {
                let pai = phi[i * n + a];
                for b in 0..n {
                    g[a * n + b] += pai * phi[i * n + b];
                }
            }
        }
        for i in 0..n {
            g[i * n + i] += tau1_pos;
        }
        // Solve via SPD cholesky.
        let l = crate::linalg::cholesky::cholesky_factor(&g, n)?;
        let x2 = crate::linalg::cholesky::cholesky_solve(&l, n, &rhs)?;
        // Variance estimate via trace((ΦᵀΦ+tau1 I)^-1) ≈ n / (m + tau1 * n) — simple approx.
        // Use the exact diagonal-trace via solving for unit-rhs is costly; approximate.
        let alpha1 =
            ((n as f64) / ((m as f64) + tau1_pos * (n as f64))).clamp(1.0e-6, 1.0 - 1.0e-6);
        let beta1 = (1.0 - alpha1).max(1.0e-6);
        // Extrinsic messages.
        let mut r2 = vec![0.0_f64; n];
        for j in 0..n {
            let v = (x2[j] - alpha1 * r1[j]) / beta1;
            r2[j] = if v.is_finite() { v } else { 0.0 };
        }
        let tau2 = (tau1_pos * (1.0 - alpha1) / alpha1).clamp(1.0e-6, 1.0e12);
        // Denoising step: soft-threshold prior.
        let lam = tau_lambda / tau2.sqrt();
        let x_new = soft_threshold(&r2, lam);
        // Divergence estimate: ⟨η'(r2)⟩ = #nonzero / n; clamp away from 0 and 1.
        let nz = x_new.iter().filter(|v| v.abs() > 1.0e-300).count();
        let alpha2 = ((nz as f64) / (n as f64)).clamp(1.0e-6, 1.0 - 1.0e-6);
        let beta2 = (1.0 - alpha2).max(1.0e-6);
        let mut r1_new = vec![0.0_f64; n];
        for j in 0..n {
            r1_new[j] = (x_new[j] - alpha2 * r2[j]) / beta2;
        }
        let tau1_new_raw = tau2 * (1.0 - alpha2) / alpha2;
        let tau1_new = if tau1_new_raw.is_finite() {
            tau1_new_raw.clamp(1.0e-6, 1.0e12)
        } else {
            1.0
        };
        // Convergence: ||x_new - x_hat||.
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = x_new[j] - x_hat[j];
            delta += d * d;
        }
        // Clamp r1 to avoid NaN propagation.
        for v in r1_new.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        let mut x_safe = x_new;
        for v in x_safe.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        x_hat = x_safe;
        r1 = r1_new;
        tau1 = tau1_new;
        iter += 1;
        if delta.sqrt() / norm2(&x_hat).max(1.0e-300) < tol {
            break;
        }
    }
    // Final sanitisation of x_hat in case of late divergence.
    for v in x_hat.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    let ax = mat_vec(phi, m, n, &x_hat)?;
    let mut residual = vec![0.0_f64; m];
    for i in 0..m {
        residual[i] = y[i] - ax[i];
    }
    // touch helper to keep imports useful
    let _ = normal_equations_solve;
    Ok(AmpResult {
        x: x_hat,
        residual_norm: norm2(&residual),
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vamp_runs() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let r = vamp(&phi, 4, 4, &y, 0.1, 50, 1.0e-9).expect("ok");
        assert!(r.iterations > 0);
    }
}

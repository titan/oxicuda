//! Proximal-gradient method (ISTA variant).
//!
//! `x_{k+1} = prox_{α g} (x_k − α ∇f(x_k))`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Proximal gradient with backtracking on Lipschitz estimate.
pub fn proximal_gradient<F, G, P>(
    x0: &[f64],
    f: F,
    grad_f: G,
    prox_g: P,
    mut step: f64,
    max_iter: usize,
    tol: f64,
    backtrack: bool,
) -> CvxResult<Vec<f64>>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    P: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
{
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if step <= 0.0 || !step.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "proximal gradient step must be > 0, got {step}"
        )));
    }
    let n = x0.len();
    let mut x = x0.to_vec();
    for it in 0..max_iter {
        let fx = f(&x)?;
        let gx = grad_f(&x)?;
        if gx.len() != n {
            return Err(CvxError::DimensionMismatch { a: gx.len(), b: n });
        }
        let mut s = step;
        let mut x_new: Vec<f64>;
        loop {
            let y: Vec<f64> = x
                .iter()
                .zip(gx.iter())
                .map(|(xi, gi)| xi - s * gi)
                .collect();
            x_new = prox_g(&y, s)?;
            if x_new.len() != n {
                return Err(CvxError::DimensionMismatch {
                    a: x_new.len(),
                    b: n,
                });
            }
            if !backtrack {
                break;
            }
            // Majorisation test:  f(x_new) ≤ f(x) + g·(x_new−x) + (1/2s) ||x_new−x||².
            let f_new = f(&x_new)?;
            let mut dot_g = 0.0_f64;
            let mut sq = 0.0_f64;
            for i in 0..n {
                let d = x_new[i] - x[i];
                dot_g += gx[i] * d;
                sq += d * d;
            }
            let majorant = fx + dot_g + sq / (2.0 * s);
            if f_new <= majorant + 1.0e-12 {
                step = s;
                break;
            }
            s *= 0.5;
            if s < 1.0e-300 {
                return Err(CvxError::LineSearchFailed(
                    "proximal gradient: step underflowed".into(),
                ));
            }
        }
        let diff: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let d_nrm = norm2(&diff);
        x = x_new;
        if d_nrm < tol {
            return Ok(x);
        }
        let _ = it;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prox_ops::l1::prox_l1;

    #[test]
    fn prox_gradient_lasso_zero_lambda_recovers_unconstrained() {
        // f(x) = ||x - b||^2 / 2, b = [3, -2].  With lambda=0, optimum = b.
        let b = vec![3.0_f64, -2.0];
        let f = |x: &[f64]| -> CvxResult<f64> {
            Ok(x.iter()
                .zip(b.iter())
                .map(|(xi, bi)| 0.5 * (xi - bi).powi(2))
                .sum())
        };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(x.iter().zip(b.iter()).map(|(xi, bi)| xi - bi).collect())
        };
        let p = |y: &[f64], _s: f64| -> CvxResult<Vec<f64>> { prox_l1(y, 0.0) };
        let x = proximal_gradient(&[0.0, 0.0], &f, &g, &p, 1.0, 200, 1.0e-10, true).expect("ok");
        assert!((x[0] - 3.0).abs() < 1.0e-6);
        assert!((x[1] + 2.0).abs() < 1.0e-6);
    }
}

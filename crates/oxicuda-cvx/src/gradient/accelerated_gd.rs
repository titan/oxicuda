//! Nesterov accelerated gradient (smooth convex).
//!
//! Updates:
//!   t_{k+1} = (1 + √(1 + 4 t_k²)) / 2
//!   y_{k+1} = x_{k+1} + (t_k − 1)/t_{k+1} · (x_{k+1} − x_k)
//!   x_{k+1} = y_k − α ∇f(y_k)

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Nesterov accelerated gradient with constant step `step` ≤ 1/L.
pub fn nesterov_accelerated<G>(
    x0: &[f64],
    grad_f: G,
    step: f64,
    max_iter: usize,
    tol: f64,
) -> CvxResult<Vec<f64>>
where
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    if step <= 0.0 || !step.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "nesterov step must be > 0, got {step}"
        )));
    }
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut x_prev = x0.to_vec();
    let mut y = x0.to_vec();
    let mut t = 1.0_f64;
    for it in 0..max_iter {
        let g = grad_f(&y)?;
        if g.len() != n {
            return Err(CvxError::DimensionMismatch { a: g.len(), b: n });
        }
        let x_new: Vec<f64> = y
            .iter()
            .zip(g.iter())
            .map(|(yi, gi)| yi - step * gi)
            .collect();
        let t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
        let beta = (t - 1.0) / t_new;
        let mut y_new = vec![0.0_f64; n];
        for i in 0..n {
            y_new[i] = x_new[i] + beta * (x_new[i] - x_prev[i]);
        }
        let diff: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let d_nrm = norm2(&diff);
        x_prev = x.clone();
        x = x_new;
        y = y_new;
        t = t_new;
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

    #[test]
    fn nesterov_quadratic_to_origin() {
        // min ||x||^2 → grad = 2x; step 0.4 < 1/L=0.5.
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.iter().map(|v| 2.0 * v).collect()) };
        let x = nesterov_accelerated(&[5.0, -3.0], g, 0.4, 1000, 1.0e-10).expect("ok");
        for &xi in &x {
            assert!(xi.abs() < 1.0e-6);
        }
    }
}

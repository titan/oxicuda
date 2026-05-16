//! Polyak heavy-ball gradient method.
//!
//! `x_{k+1} = x_k − α ∇f(x_k) + β (x_k − x_{k-1})`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Heavy-ball update.  Returns the final iterate.
pub fn heavy_ball<G>(
    x0: &[f64],
    grad_f: G,
    alpha: f64,
    beta: f64,
    max_iter: usize,
    tol: f64,
) -> CvxResult<Vec<f64>>
where
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    if alpha <= 0.0 || !alpha.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "heavy-ball alpha must be > 0, got {alpha}"
        )));
    }
    if !beta.is_finite() || !(0.0..1.0).contains(&beta) {
        return Err(CvxError::InvalidParameter(format!(
            "heavy-ball beta must be in [0, 1), got {beta}"
        )));
    }
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut x_prev = x0.to_vec();
    for it in 0..max_iter {
        let g = grad_f(&x)?;
        if g.len() != n {
            return Err(CvxError::DimensionMismatch { a: g.len(), b: n });
        }
        let mut x_new = vec![0.0_f64; n];
        for i in 0..n {
            x_new[i] = x[i] - alpha * g[i] + beta * (x[i] - x_prev[i]);
        }
        let diff: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let d_nrm = norm2(&diff);
        x_prev = x.clone();
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

    #[test]
    fn heavy_ball_quadratic() {
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.iter().map(|v| 2.0 * v).collect()) };
        let x = heavy_ball(&[3.0, 4.0], g, 0.1, 0.5, 2000, 1.0e-9).expect("ok");
        for &xi in &x {
            assert!(xi.abs() < 1.0e-4);
        }
    }
}

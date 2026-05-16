//! Armijo (sufficient decrease) backtracking line search.
//!
//! Find `α > 0` such that `f(x + α d) ≤ f(x) + c1 α ∇f(x)^T d`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::dot;

/// Run Armijo backtracking from initial step `alpha0`, contraction `rho ∈ (0, 1)`,
/// constant `c1 ∈ (0, 1)`.
///
/// `x`, `d`, `grad` are vectors; `f` is a callable `&[f64] → CvxResult<f64>`.
pub fn armijo_search<F>(
    x: &[f64],
    d: &[f64],
    grad: &[f64],
    f: F,
    alpha0: f64,
    rho: f64,
    c1: f64,
    max_iter: usize,
) -> CvxResult<f64>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
{
    if alpha0 <= 0.0 || rho <= 0.0 || rho >= 1.0 || c1 <= 0.0 || c1 >= 1.0 {
        return Err(CvxError::InvalidParameter(format!(
            "armijo parameters invalid: alpha0={alpha0}, rho={rho}, c1={c1}"
        )));
    }
    if x.len() != d.len() || x.len() != grad.len() {
        return Err(CvxError::DimensionMismatch {
            a: x.len(),
            b: d.len(),
        });
    }
    let fx = f(x)?;
    let gd = dot(grad, d)?;
    if gd >= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "armijo requires descent direction (g·d<0), got {gd}"
        )));
    }
    let mut alpha = alpha0;
    for _ in 0..max_iter {
        let x_new: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        let f_new = f(&x_new)?;
        if f_new <= fx + c1 * alpha * gd {
            return Ok(alpha);
        }
        alpha *= rho;
    }
    Err(CvxError::LineSearchFailed(format!(
        "armijo: no sufficient decrease found in {max_iter} iters"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn armijo_finds_step_quadratic() {
        let f = |x: &[f64]| -> CvxResult<f64> { Ok(x.iter().map(|v| v * v).sum::<f64>()) };
        // x = [1, 1], grad = [2, 2], d = -grad = [-2, -2].
        let x = vec![1.0, 1.0];
        let grad = vec![2.0, 2.0];
        let d = vec![-2.0, -2.0];
        let alpha = armijo_search(&x, &d, &grad, f, 1.0, 0.5, 1.0e-4, 50).expect("ok");
        // For pure quadratic, full step alpha=0.5 lands at origin with f=0 — should be accepted.
        assert!(alpha > 0.0 && alpha <= 1.0);
    }
}

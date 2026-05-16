//! Wolfe line search (Armijo + curvature condition).
//!
//! Finds α > 0 satisfying:
//!  - Armijo:   f(x + αd) ≤ f(x) + c1 α g·d
//!  - curvature: g(x + αd)·d ≥ c2 g·d  (g·d is negative for descent direction).

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::dot;

/// Wolfe line search.  `f` and `grad_f` are callable.
pub fn wolfe_search<F, G>(
    x: &[f64],
    d: &[f64],
    grad: &[f64],
    f: F,
    grad_f: G,
    c1: f64,
    c2: f64,
    max_iter: usize,
) -> CvxResult<f64>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    if c1 <= 0.0 || c1 >= c2 || c2 >= 1.0 {
        return Err(CvxError::InvalidParameter(format!(
            "wolfe requires 0 < c1 < c2 < 1, got c1={c1}, c2={c2}"
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
            "wolfe requires descent direction (g·d<0), got {gd}"
        )));
    }
    // Bracketing/bisection in [a_lo, a_hi].
    let mut a_lo = 0.0_f64;
    let mut a_hi = f64::INFINITY;
    let mut alpha = 1.0_f64;
    for _ in 0..max_iter {
        let x_new: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        let f_new = f(&x_new)?;
        if f_new > fx + c1 * alpha * gd {
            a_hi = alpha;
            alpha = 0.5 * (a_lo + a_hi);
            continue;
        }
        let g_new = grad_f(&x_new)?;
        let gd_new = dot(&g_new, d)?;
        if gd_new < c2 * gd {
            a_lo = alpha;
            if a_hi.is_infinite() {
                alpha *= 2.0;
            } else {
                alpha = 0.5 * (a_lo + a_hi);
            }
            continue;
        }
        return Ok(alpha);
    }
    Err(CvxError::LineSearchFailed(format!(
        "wolfe: not satisfied in {max_iter} iters"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wolfe_quadratic() {
        let f = |x: &[f64]| -> CvxResult<f64> { Ok(x.iter().map(|v| v * v).sum::<f64>()) };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.iter().map(|v| 2.0 * v).collect()) };
        let x = vec![1.0];
        let grad = vec![2.0];
        let d = vec![-2.0];
        let alpha = wolfe_search(&x, &d, &grad, &f, &g, 1.0e-4, 0.9, 50).expect("ok");
        assert!(alpha > 0.0);
    }
}

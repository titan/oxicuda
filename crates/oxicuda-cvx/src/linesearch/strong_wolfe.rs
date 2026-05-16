//! Strong Wolfe line search.
//!
//! Adds the absolute-value curvature condition `|g(x + αd)·d| ≤ c2 |g·d|`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::dot;

/// Strong Wolfe line search via standard bracketing + zoom.
pub fn strong_wolfe_search<F, G>(
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
            "strong wolfe requires 0 < c1 < c2 < 1, got c1={c1}, c2={c2}"
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
            "strong wolfe requires descent direction (g·d<0), got {gd}"
        )));
    }
    let mut alpha_prev = 0.0_f64;
    let mut f_prev = fx;
    let mut alpha = 1.0_f64;
    let alpha_max = 1.0e10_f64;
    for it in 0..max_iter {
        let x_new: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        let f_new = f(&x_new)?;
        // Sufficient decrease test or alpha exceeded?
        if f_new > fx + c1 * alpha * gd || (it > 0 && f_new >= f_prev) {
            return zoom(
                x, d, &f, &grad_f, fx, gd, alpha_prev, alpha, c1, c2, max_iter,
            );
        }
        let g_new = grad_f(&x_new)?;
        let gd_new = dot(&g_new, d)?;
        if gd_new.abs() <= -c2 * gd {
            return Ok(alpha);
        }
        if gd_new >= 0.0 {
            return zoom(
                x, d, &f, &grad_f, fx, gd, alpha, alpha_prev, c1, c2, max_iter,
            );
        }
        alpha_prev = alpha;
        f_prev = f_new;
        alpha = (alpha * 2.0).min(alpha_max);
        if alpha >= alpha_max {
            return Err(CvxError::LineSearchFailed(format!(
                "strong wolfe: alpha exceeded {alpha_max}"
            )));
        }
    }
    Err(CvxError::LineSearchFailed(format!(
        "strong wolfe: not satisfied in {max_iter} iters"
    )))
}

#[allow(clippy::too_many_arguments)]
fn zoom<F, G>(
    x: &[f64],
    d: &[f64],
    f: &F,
    grad_f: &G,
    fx: f64,
    gd: f64,
    mut a_lo: f64,
    mut a_hi: f64,
    c1: f64,
    c2: f64,
    max_iter: usize,
) -> CvxResult<f64>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    for _ in 0..max_iter {
        let alpha = 0.5 * (a_lo + a_hi);
        let x_new: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        let f_new = f(&x_new)?;
        let x_lo: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + a_lo * di)
            .collect();
        let f_lo = f(&x_lo)?;
        if f_new > fx + c1 * alpha * gd || f_new >= f_lo {
            a_hi = alpha;
        } else {
            let g_new = grad_f(&x_new)?;
            let gd_new = dot(&g_new, d)?;
            if gd_new.abs() <= -c2 * gd {
                return Ok(alpha);
            }
            if gd_new * (a_hi - a_lo) >= 0.0 {
                a_hi = a_lo;
            }
            a_lo = alpha;
        }
        if (a_hi - a_lo).abs() < 1.0e-14 {
            return Ok(0.5 * (a_lo + a_hi));
        }
    }
    Err(CvxError::LineSearchFailed(
        "strong wolfe zoom: did not converge".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strong_wolfe_quadratic() {
        let f = |x: &[f64]| -> CvxResult<f64> { Ok(x.iter().map(|v| v * v).sum::<f64>()) };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.iter().map(|v| 2.0 * v).collect()) };
        let x = vec![1.0];
        let grad = vec![2.0];
        let d = vec![-2.0];
        let alpha = strong_wolfe_search(&x, &d, &grad, &f, &g, 1.0e-4, 0.9, 50).expect("ok");
        // Verify Armijo and strong curvature.
        let x_new: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        let f_new = f(&x_new).expect("ok");
        let fx = f(&x).expect("ok");
        let gd = -4.0_f64;
        assert!(f_new <= fx + 1.0e-4 * alpha * gd + 1.0e-12);
        let g_new = g(&x_new).expect("ok");
        let gd_new = g_new[0] * d[0];
        assert!(gd_new.abs() <= 0.9 * gd.abs() + 1.0e-9);
    }
}

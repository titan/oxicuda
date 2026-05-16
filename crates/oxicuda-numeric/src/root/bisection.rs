//! Bisection method on a sign-changing bracket.
//!
//! Convergence: linear, halving the interval each step. Guaranteed if `f(a) f(b) < 0`
//! and `f` is continuous.

use crate::error::{NumericError, NumericResult};

/// Find a root of `f` on `[a, b]` such that `f(a) f(b) ≤ 0`.
/// Returns the midpoint of the final interval (width `≤ tol`).
pub fn bisection<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericError::InvalidParameter(format!(
            "non-finite endpoints a={a}, b={b}"
        )));
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    let fa = f(a)?;
    let fb = f(b)?;
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa.signum() == fb.signum() {
        return Err(NumericError::RootNotBracketed { a, b, fa, fb });
    }
    let mut fa_sign = fa.signum();
    for _ in 0..max_iter {
        let mid = 0.5 * (a + b);
        let fmid = f(mid)?;
        if (b - a).abs() < tol || fmid.abs() < tol {
            return Ok(mid);
        }
        if fmid.signum() == fa_sign {
            a = mid;
            fa_sign = fmid.signum();
        } else {
            b = mid;
        }
    }
    Err(NumericError::NotConverged {
        iter: max_iter,
        residual: (b - a).abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn bisection_cos_zero() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.cos()) };
        let r = bisection(f, 0.0, PI, 1.0e-12, 200).expect("ok");
        assert!((r - PI / 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn bisection_quadratic() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x * x - 2.0) };
        let r = bisection(f, 0.0, 2.0, 1.0e-12, 200).expect("ok");
        assert!((r - 2.0_f64.sqrt()).abs() < 1.0e-10);
    }

    #[test]
    fn bisection_not_bracketed_err() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x * x + 1.0) };
        let res = bisection(f, -1.0, 1.0, 1.0e-12, 100);
        assert!(matches!(res, Err(NumericError::RootNotBracketed { .. })));
    }

    #[test]
    fn bisection_exact_endpoint() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x) };
        let r = bisection(f, 0.0, 1.0, 1.0e-12, 100).expect("ok");
        assert!((r - 0.0).abs() < 1.0e-12);
    }
}

//! Secant method.
//!
//! `x_{k+1} = x_k - f(x_k) (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))`.
//! Superlinear convergence with golden ratio ≈ 1.618.

use crate::error::{NumericError, NumericResult};

/// Secant method between two initial points.
pub fn secant<F>(f: F, x0: f64, x1: f64, tol: f64, max_iter: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    if !x0.is_finite() || !x1.is_finite() {
        return Err(NumericError::InvalidParameter(format!(
            "non-finite seed {x0}, {x1}"
        )));
    }
    let mut x_prev = x0;
    let mut x_curr = x1;
    let mut f_prev = f(x_prev)?;
    let mut f_curr = f(x_curr)?;
    for k in 0..max_iter {
        if f_curr.abs() < tol {
            return Ok(x_curr);
        }
        let denom = f_curr - f_prev;
        if denom.abs() < 1.0e-300 {
            return Err(NumericError::NumericalInstability(format!(
                "f(x_k)=f(x_k-1) at iter={k}"
            )));
        }
        let x_next = x_curr - f_curr * (x_curr - x_prev) / denom;
        if !x_next.is_finite() {
            return Err(NumericError::NumericalInstability(format!(
                "iterate diverged at iter={k}"
            )));
        }
        x_prev = x_curr;
        f_prev = f_curr;
        x_curr = x_next;
        f_curr = f(x_curr)?;
        if (x_curr - x_prev).abs() < tol {
            return Ok(x_curr);
        }
    }
    Err(NumericError::NotConverged {
        iter: max_iter,
        residual: f_curr.abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secant_cube_root() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(3) - 27.0) };
        let r = secant(f, 1.0, 4.0, 1.0e-10, 100).expect("ok");
        assert!((r - 3.0).abs() < 1.0e-8);
    }

    #[test]
    fn secant_log_root() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.ln()) };
        let r = secant(f, 0.5, 2.0, 1.0e-10, 100).expect("ok");
        assert!((r - 1.0).abs() < 1.0e-8);
    }

    #[test]
    fn secant_invalid_seed() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x) };
        let res = secant(f, f64::NAN, 1.0, 1.0e-10, 50);
        assert!(res.is_err());
    }
}

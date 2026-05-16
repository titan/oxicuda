//! Newton-Raphson root finder.
//!
//! `x_{k+1} = x_k - f(x_k) / f'(x_k)`. Quadratic convergence when `f'(x*) ≠ 0`.

use crate::error::{NumericError, NumericResult};

/// Newton's method. Requires both `f` and `f'`.
pub fn newton<F, G>(f: F, f_prime: G, x0: f64, tol: f64, max_iter: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
    G: Fn(f64) -> NumericResult<f64>,
{
    if !x0.is_finite() {
        return Err(NumericError::InvalidParameter(format!(
            "non-finite x0={x0}"
        )));
    }
    let mut x = x0;
    for k in 0..max_iter {
        let fx = f(x)?;
        if fx.abs() < tol {
            return Ok(x);
        }
        let dfx = f_prime(x)?;
        if dfx.abs() < 1.0e-300 {
            return Err(NumericError::NumericalInstability(format!(
                "derivative vanished at x={x} (iter={k})"
            )));
        }
        let dx = fx / dfx;
        x -= dx;
        if dx.abs() < tol {
            return Ok(x);
        }
        if !x.is_finite() {
            return Err(NumericError::NumericalInstability(format!(
                "iterate diverged to non-finite at iter={k}"
            )));
        }
    }
    Err(NumericError::NotConverged {
        iter: max_iter,
        residual: f(x).map(|y| y.abs()).unwrap_or(f64::INFINITY),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_cube_root_two() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(3) - 2.0) };
        let g = |x: f64| -> NumericResult<f64> { Ok(3.0 * x * x) };
        let r = newton(f, g, 1.0, 1.0e-12, 50).expect("ok");
        assert!((r - 2.0_f64.powf(1.0 / 3.0)).abs() < 1.0e-10);
    }

    #[test]
    fn newton_quadratic() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x * x - 9.0) };
        let g = |x: f64| -> NumericResult<f64> { Ok(2.0 * x) };
        let r = newton(f, g, 2.0, 1.0e-12, 50).expect("ok");
        assert!((r - 3.0).abs() < 1.0e-10);
    }

    #[test]
    fn newton_zero_derivative_err() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x * x + 1.0) };
        let g = |x: f64| -> NumericResult<f64> { Ok(2.0 * x) };
        let res = newton(f, g, 0.0, 1.0e-12, 20);
        assert!(res.is_err());
    }
}

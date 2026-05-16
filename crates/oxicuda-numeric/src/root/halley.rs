//! Halley's method — cubic convergence using `f`, `f'`, `f''`.
//!
//! `x_{k+1} = x_k - 2·f(x)·f'(x) / (2·f'(x)² - f(x)·f''(x))`.

use crate::error::{NumericError, NumericResult};

/// Halley's method.
pub fn halley<F, G, H>(
    f: F,
    f_prime: G,
    f_double_prime: H,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
    G: Fn(f64) -> NumericResult<f64>,
    H: Fn(f64) -> NumericResult<f64>,
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
        let ddfx = f_double_prime(x)?;
        let denom = 2.0 * dfx * dfx - fx * ddfx;
        if denom.abs() < 1.0e-300 {
            return Err(NumericError::NumericalInstability(format!(
                "Halley denominator vanished at x={x} iter={k}"
            )));
        }
        let step = 2.0 * fx * dfx / denom;
        x -= step;
        if step.abs() < tol {
            return Ok(x);
        }
        if !x.is_finite() {
            return Err(NumericError::NumericalInstability(format!(
                "Halley iterate diverged at iter={k}"
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
    fn halley_cube_root_two() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(3) - 2.0) };
        let g = |x: f64| -> NumericResult<f64> { Ok(3.0 * x * x) };
        let h = |x: f64| -> NumericResult<f64> { Ok(6.0 * x) };
        let r = halley(f, g, h, 1.0, 1.0e-12, 30).expect("ok");
        assert!((r - 2.0_f64.powf(1.0 / 3.0)).abs() < 1.0e-10);
    }

    #[test]
    fn halley_quadratic_fast() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(2) - 7.0) };
        let g = |x: f64| -> NumericResult<f64> { Ok(2.0 * x) };
        let h = |_x: f64| -> NumericResult<f64> { Ok(2.0) };
        let r = halley(f, g, h, 3.0, 1.0e-12, 20).expect("ok");
        assert!((r - 7.0_f64.sqrt()).abs() < 1.0e-10);
    }
}

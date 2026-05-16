//! Richardson extrapolation for numerical derivatives.
//!
//! Combine `D(h)` and `D(h/2)` (central difference) to cancel the leading `O(h²)` term,
//! yielding `O(h⁴)` accuracy:
//! `D_R(h) = (4 · D(h/2) - D(h)) / 3`.
//! Iterating gives a Romberg-like table.

use crate::diff::central_difference::central_difference;
use crate::error::{NumericError, NumericResult};

/// Richardson-extrapolated derivative with `n_levels` halvings of `h_init`.
pub fn richardson_derivative<F>(f: F, x: f64, h_init: f64, n_levels: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    if !h_init.is_finite() || h_init <= 0.0 {
        return Err(NumericError::InvalidStepSize { step: h_init });
    }
    if n_levels == 0 {
        return Err(NumericError::InvalidParameter(
            "n_levels must be ≥ 1".into(),
        ));
    }
    let mut a: Vec<Vec<f64>> = vec![vec![0.0_f64; n_levels]; n_levels];
    let mut h = h_init;
    for i in 0..n_levels {
        a[i][0] = central_difference(&f, x, h)?;
        h *= 0.5;
        let mut pow4 = 4.0_f64;
        for j in 1..=i {
            a[i][j] = (pow4 * a[i][j - 1] - a[i - 1][j - 1]) / (pow4 - 1.0);
            pow4 *= 4.0;
        }
    }
    Ok(a[n_levels - 1][n_levels - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rich_polynomial_exact() {
        // f(x) = x⁴; f'(x) = 4x³; at x=1 → 4
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(4)) };
        let d = richardson_derivative(f, 1.0, 0.1, 5).expect("ok");
        assert!((d - 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn rich_exp() {
        // d/dx e^x = e^x; at x=0 → 1
        let f = |x: f64| -> NumericResult<f64> { Ok(x.exp()) };
        let d = richardson_derivative(f, 0.0, 0.1, 6).expect("ok");
        assert!((d - 1.0).abs() < 1.0e-12);
    }
}

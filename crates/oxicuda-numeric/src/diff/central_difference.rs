//! Central difference numerical derivative `(f(x+h) - f(x-h)) / (2 h)`.

use crate::error::{NumericError, NumericResult};

/// Central finite difference approximation to `f'(x)` with step `h`.
pub fn central_difference<F>(f: F, x: f64, h: f64) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    if !h.is_finite() || h <= 0.0 {
        return Err(NumericError::InvalidStepSize { step: h });
    }
    let fp = f(x + h)?;
    let fm = f(x - h)?;
    Ok((fp - fm) / (2.0 * h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cd_polynomial() {
        // d/dx x³ = 3 x²; at x = 2 → 12
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(3)) };
        let d = central_difference(f, 2.0, 1.0e-4).expect("ok");
        assert!((d - 12.0).abs() < 1.0e-6);
    }

    #[test]
    fn cd_sin() {
        // d/dx sin(x) = cos(x); at π/3 → 0.5
        let f = |x: f64| -> NumericResult<f64> { Ok(x.sin()) };
        let d = central_difference(f, std::f64::consts::FRAC_PI_3, 1.0e-5).expect("ok");
        assert!((d - 0.5).abs() < 1.0e-8);
    }

    #[test]
    fn cd_invalid_step() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x) };
        let r = central_difference(f, 1.0, -0.1);
        assert!(matches!(r, Err(NumericError::InvalidStepSize { .. })));
    }
}

//! Hermite cubic interpolation using values and derivatives at endpoints.

use crate::error::{NumericError, NumericResult};

/// Evaluate the cubic Hermite interpolant on `[x0, x1]` with values `y0, y1` and slopes
/// `dy0, dy1` at the endpoints, at the query point `x`.
pub fn hermite_interpolate(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    dy0: f64,
    dy1: f64,
    x: f64,
) -> NumericResult<f64> {
    let h = x1 - x0;
    if h <= 0.0 {
        return Err(NumericError::InvalidParameter(
            "x1 must be > x0 in hermite_interpolate".into(),
        ));
    }
    let t = (x - x0) / h;
    let h00 = 2.0 * t.powi(3) - 3.0 * t.powi(2) + 1.0;
    let h10 = t.powi(3) - 2.0 * t.powi(2) + t;
    let h01 = -2.0 * t.powi(3) + 3.0 * t.powi(2);
    let h11 = t.powi(3) - t.powi(2);
    Ok(h00 * y0 + h10 * h * dy0 + h01 * y1 + h11 * h * dy1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermite_endpoints_exact() {
        let v0 = hermite_interpolate(0.0, 1.0, 2.0, 5.0, 0.0, 0.0, 0.0).expect("ok");
        let v1 = hermite_interpolate(0.0, 1.0, 2.0, 5.0, 0.0, 0.0, 1.0).expect("ok");
        assert!((v0 - 2.0).abs() < 1.0e-12);
        assert!((v1 - 5.0).abs() < 1.0e-12);
    }

    #[test]
    fn hermite_recovers_cubic() {
        // f(x) = x³ on [0, 1]; f(0)=0, f(1)=1, f'(0)=0, f'(1)=3
        let v = hermite_interpolate(0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.5).expect("ok");
        assert!((v - 0.125).abs() < 1.0e-12);
    }
}

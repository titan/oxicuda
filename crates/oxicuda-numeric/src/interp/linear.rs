//! Piecewise linear interpolation on sorted nodes.

use crate::error::{NumericError, NumericResult};

/// Linearly interpolate `(xs, ys)` at `x`. `xs` must be sorted ascending.
pub fn linear_interpolate(xs: &[f64], ys: &[f64], x: f64) -> NumericResult<f64> {
    if xs.is_empty() || ys.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    if xs.len() != ys.len() {
        return Err(NumericError::DimensionMismatch {
            a: xs.len(),
            b: ys.len(),
        });
    }
    if x <= xs[0] {
        return Ok(ys[0]);
    }
    if x >= xs[xs.len() - 1] {
        return Ok(ys[xs.len() - 1]);
    }
    // binary search for the interval
    let mut lo = 0_usize;
    let mut hi = xs.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    Ok(ys[lo] + t * (ys[hi] - ys[lo]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_midpoint() {
        let xs = vec![0.0_f64, 1.0, 2.0];
        let ys = vec![0.0_f64, 1.0, 4.0];
        let v = linear_interpolate(&xs, &ys, 0.5).expect("ok");
        assert!((v - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn linear_extrapolation_clamp() {
        let xs = vec![0.0_f64, 1.0];
        let ys = vec![1.0_f64, 3.0];
        let v_lo = linear_interpolate(&xs, &ys, -1.0).expect("ok");
        let v_hi = linear_interpolate(&xs, &ys, 5.0).expect("ok");
        assert!((v_lo - 1.0).abs() < 1.0e-12);
        assert!((v_hi - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn linear_size_mismatch() {
        let r = linear_interpolate(&[1.0_f64, 2.0], &[1.0_f64], 1.5);
        assert!(matches!(r, Err(NumericError::DimensionMismatch { .. })));
    }
}

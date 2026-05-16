//! Akima spline interpolation (1970).
//!
//! Uses weighted slopes from 4 surrounding secant slopes:
//! `slope[i] = (|s_{i+1} - s_{i+2}| s_{i-1} + |s_{i-1} - s_i| s_{i+1}) / (|s_{i+1} - s_{i+2}| + |s_{i-1} - s_i|)`.
//! Avoids overshoot near outliers.

use crate::error::{NumericError, NumericResult};

/// Evaluate an Akima interpolant at point `x_eval` given nodes `(xs, ys)`.
pub fn akima_interpolate(xs: &[f64], ys: &[f64], x_eval: f64) -> NumericResult<f64> {
    let n = xs.len();
    if n != ys.len() {
        return Err(NumericError::DimensionMismatch { a: n, b: ys.len() });
    }
    if n < 2 {
        return Err(NumericError::InvalidParameter("need ≥ 2 nodes".into()));
    }
    if n == 2 {
        let t = (x_eval - xs[0]) / (xs[1] - xs[0]);
        return Ok(ys[0] + t * (ys[1] - ys[0]));
    }
    // secant slopes m_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)
    let mut slopes = vec![0.0_f64; n - 1];
    for i in 0..(n - 1) {
        slopes[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    }
    // extend with virtual slopes (Akima's recipe)
    let mut m = vec![0.0_f64; n + 3];
    m[2..(2 + n - 1)].copy_from_slice(&slopes[..(n - 1)]);
    m[1] = 2.0 * m[2] - m[3];
    m[0] = 2.0 * m[1] - m[2];
    m[n + 1] = 2.0 * m[n] - m[n - 1];
    m[n + 2] = 2.0 * m[n + 1] - m[n];
    // tangent at each node
    let mut t = vec![0.0_f64; n];
    for i in 0..n {
        let w1 = (m[i + 3] - m[i + 2]).abs();
        let w2 = (m[i + 1] - m[i]).abs();
        if w1 + w2 < 1.0e-12 {
            t[i] = 0.5 * (m[i + 1] + m[i + 2]);
        } else {
            t[i] = (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2);
        }
    }
    // find interval
    if x_eval <= xs[0] {
        return Ok(ys[0]);
    }
    if x_eval >= xs[n - 1] {
        return Ok(ys[n - 1]);
    }
    let mut lo = 0_usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x_eval {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let h = xs[hi] - xs[lo];
    let s = (x_eval - xs[lo]) / h;
    // Hermite-like with tangents
    let h00 = 2.0 * s.powi(3) - 3.0 * s.powi(2) + 1.0;
    let h10 = s.powi(3) - 2.0 * s.powi(2) + s;
    let h01 = -2.0 * s.powi(3) + 3.0 * s.powi(2);
    let h11 = s.powi(3) - s.powi(2);
    Ok(h00 * ys[lo] + h10 * h * t[lo] + h01 * ys[hi] + h11 * h * t[hi])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn akima_passes_through_nodes() {
        let xs = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0_f64, 1.0, 4.0, 9.0, 16.0];
        for (x, y) in xs.iter().zip(ys.iter()) {
            let v = akima_interpolate(&xs, &ys, *x).expect("ok");
            assert!((v - y).abs() < 1.0e-10);
        }
    }

    #[test]
    fn akima_midpoint_reasonable() {
        let xs = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0_f64, 1.0, 4.0, 9.0, 16.0];
        let v = akima_interpolate(&xs, &ys, 1.5).expect("ok");
        // y = x² → 2.25; Akima should be close
        assert!((v - 2.25).abs() < 0.5);
    }
}

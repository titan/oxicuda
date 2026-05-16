//! PCHIP (Fritsch-Carlson monotone piecewise cubic Hermite) interpolation.
//!
//! Preserves monotonicity of input data via constrained tangents:
//! - if `s_{i-1} s_i ≤ 0`: tangent = 0
//! - else: tangent = harmonic average of neighboring slopes.

use crate::error::{NumericError, NumericResult};

/// Evaluate the PCHIP interpolant at `x_eval` for monotone data `(xs, ys)`.
pub fn pchip_interpolate(xs: &[f64], ys: &[f64], x_eval: f64) -> NumericResult<f64> {
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
    // secant slopes
    let mut h = vec![0.0_f64; n - 1];
    let mut s = vec![0.0_f64; n - 1];
    for i in 0..(n - 1) {
        h[i] = xs[i + 1] - xs[i];
        if h[i] <= 0.0 {
            return Err(NumericError::InvalidParameter(
                "xs must be strictly increasing".into(),
            ));
        }
        s[i] = (ys[i + 1] - ys[i]) / h[i];
    }
    let mut t = vec![0.0_f64; n];
    // interior tangents
    for i in 1..(n - 1) {
        if s[i - 1] * s[i] <= 0.0 {
            t[i] = 0.0;
        } else {
            let w1 = 2.0 * h[i] + h[i - 1];
            let w2 = h[i] + 2.0 * h[i - 1];
            t[i] = (w1 + w2) / (w1 / s[i - 1] + w2 / s[i]);
        }
    }
    // endpoint tangents (Fritsch-Carlson one-sided formula)
    if n >= 3 {
        t[0] = endpoint_tangent(h[0], h[1], s[0], s[1]);
        let last = s.len() - 1;
        t[n - 1] = endpoint_tangent(h[n - 2], h[n - 3], s[last], s[last - 1]);
    } else {
        t[0] = s[0];
        t[n - 1] = s[s.len() - 1];
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
    let h_i = xs[hi] - xs[lo];
    let u = (x_eval - xs[lo]) / h_i;
    let h00 = 2.0 * u.powi(3) - 3.0 * u.powi(2) + 1.0;
    let h10 = u.powi(3) - 2.0 * u.powi(2) + u;
    let h01 = -2.0 * u.powi(3) + 3.0 * u.powi(2);
    let h11 = u.powi(3) - u.powi(2);
    Ok(h00 * ys[lo] + h10 * h_i * t[lo] + h01 * ys[hi] + h11 * h_i * t[hi])
}

fn endpoint_tangent(h0: f64, h1: f64, s0: f64, s1: f64) -> f64 {
    let denom = h0 + h1;
    if denom < 1.0e-15 {
        return s0;
    }
    let val = ((2.0 * h0 + h1) * s0 - h0 * s1) / denom;
    if val * s0 <= 0.0 {
        0.0
    } else if (s0 * s1).is_sign_negative() && val.abs() > 3.0 * s0.abs() {
        3.0 * s0
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pchip_passes_through_nodes() {
        let xs = vec![0.0_f64, 1.0, 2.0, 4.0, 8.0];
        let ys = vec![0.0_f64, 1.0, 2.0, 4.0, 8.0];
        for (x, y) in xs.iter().zip(ys.iter()) {
            let v = pchip_interpolate(&xs, &ys, *x).expect("ok");
            assert!((v - y).abs() < 1.0e-10);
        }
    }

    #[test]
    fn pchip_monotone_preservation() {
        // strictly increasing input
        let xs = vec![0.0_f64, 1.0, 2.0, 4.0, 8.0];
        let ys = vec![0.0_f64, 1.0, 2.0, 4.0, 8.0];
        let mut prev = pchip_interpolate(&xs, &ys, 0.0).expect("ok");
        let mut t = 0.0_f64;
        while t <= 8.0 {
            let v = pchip_interpolate(&xs, &ys, t).expect("ok");
            assert!(v >= prev - 1.0e-12);
            prev = v;
            t += 0.1;
        }
    }
}

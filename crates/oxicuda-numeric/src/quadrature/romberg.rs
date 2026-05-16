//! Romberg integration — Richardson extrapolation on the trapezoidal rule.
//!
//! Build a triangular table `T[i][k]` where
//! - `T[i][0]` is the composite trapezoidal rule with `2^i` subintervals,
//! - `T[i][k] = (4^k T[i][k-1] - T[i-1][k-1]) / (4^k - 1)`.
//!
//! Convergence is `O(h^{2(n+1)})` for smooth integrands.

use crate::error::{NumericError, NumericResult};

/// Romberg integration of `f` over `[a, b]`. The maximum table size is `max_levels`.
pub fn romberg<F>(f: F, a: f64, b: f64, tol: f64, max_levels: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericError::InvalidParameter(format!(
            "non-finite limits a={a}, b={b}"
        )));
    }
    if max_levels == 0 {
        return Err(NumericError::InvalidParameter(
            "max_levels must be ≥ 1".into(),
        ));
    }
    let h0 = b - a;
    let fa = f(a)?;
    let fb = f(b)?;
    let mut t = vec![vec![0.0_f64; max_levels]; max_levels];
    t[0][0] = 0.5 * h0 * (fa + fb);
    for i in 1..max_levels {
        let n = 1_usize << (i - 1);
        let h = h0 / (n as f64);
        let mut s = 0.0_f64;
        for k in 0..n {
            let x = a + h * (k as f64 + 0.5);
            s += f(x)?;
        }
        t[i][0] = 0.5 * t[i - 1][0] + 0.5 * h * s;
        let mut pow4 = 4.0_f64;
        for j in 1..=i {
            t[i][j] = (pow4 * t[i][j - 1] - t[i - 1][j - 1]) / (pow4 - 1.0);
            pow4 *= 4.0;
        }
        if i >= 2 && (t[i][i] - t[i - 1][i - 1]).abs() < tol {
            return Ok(t[i][i]);
        }
    }
    Ok(t[max_levels - 1][max_levels - 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn romberg_arctan_pi_quarter() {
        // ∫_0^1 1/(1+x²) dx = π/4
        let f = |x: f64| -> NumericResult<f64> { Ok(1.0 / (1.0 + x * x)) };
        let r = romberg(f, 0.0, 1.0, 1.0e-12, 12).expect("ok");
        assert!((r - PI / 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn romberg_sin() {
        // ∫_0^π sin(x) dx = 2
        let f = |x: f64| -> NumericResult<f64> { Ok(x.sin()) };
        let r = romberg(f, 0.0, PI, 1.0e-12, 10).expect("ok");
        assert!((r - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn romberg_polynomial() {
        // ∫_0^1 x⁴ dx = 1/5
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(4)) };
        let r = romberg(f, 0.0, 1.0, 1.0e-12, 8).expect("ok");
        assert!((r - 0.2).abs() < 1.0e-10);
    }
}

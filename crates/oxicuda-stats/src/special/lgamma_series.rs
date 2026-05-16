//! Stirling and asymptotic series helpers for lgamma diagnostics and tail behaviour.

/// Stirling series approximation `lgamma(x) ~ (x-0.5) ln(x) - x + 0.5 ln(2pi) + 1/(12x) - ...`.
///
/// Accurate only for large x (>= 8). Used as a cross-check for the main Lanczos implementation.
#[must_use]
pub fn stirling_series(x: f64) -> f64 {
    let term = (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln();
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // Series: 1/(12x) - 1/(360 x^3) + 1/(1260 x^5)
    let corr = inv / 12.0 - inv * inv2 / 360.0 + inv * inv2 * inv2 / 1260.0;
    term + corr
}

/// Logarithm of (n!) via Stirling expansion plus correction.
///
/// `lgamma_series(n) = lgamma(n+1)` for integer arguments.
#[must_use]
pub fn lgamma_series(n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let mut s = 0.0;
    // Sum exactly for small n
    if n <= 20 {
        for k in 2..=n {
            s += (k as f64).ln();
        }
        s
    } else {
        stirling_series((n + 1) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stirling_large_x() {
        // For large x, stirling_series should match lgamma well
        let x = 50.0;
        let s = stirling_series(x);
        // lgamma(50) = ln(49!) ~ 144.5658
        assert!((s - 144.565_743_677_485_07).abs() < 1e-3);
    }

    #[test]
    fn lgamma_series_factorial_match() {
        // ln(5!) = ln(120)
        let v = lgamma_series(5);
        assert!((v - 120f64.ln()).abs() < 1e-10);
        // ln(10!) = ln(3628800)
        let v = lgamma_series(10);
        assert!((v - 3_628_800f64.ln()).abs() < 1e-10);
    }
}

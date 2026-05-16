//! Logarithm of the gamma function via Lanczos approximation (g=7).
//!
//! Reference: Numerical Recipes 3rd ed., section 6.1.

use crate::error::{StatsError, StatsResult};

const LANCZOS_G: f64 = 7.0;
// Standard Numerical Recipes Lanczos coefficients (g=7, n=9)
const LANCZOS_P: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// Logarithm of the Gamma function for `x > 0`.
///
/// Uses Lanczos g=7 approximation with reflection for `x < 0.5`.
#[must_use]
pub fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: lgamma(x) = ln(pi) - ln(|sin(pi*x)|) - lgamma(1 - x)
        let pi = std::f64::consts::PI;
        let sx = (pi * x).sin().abs().max(1e-300);
        pi.ln() - sx.ln() - lgamma(1.0 - x)
    } else {
        let xm = x - 1.0;
        let mut sum = LANCZOS_P[0];
        for (i, &p) in LANCZOS_P.iter().enumerate().skip(1) {
            sum += p / (xm + i as f64);
        }
        let t = xm + LANCZOS_G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (xm + 0.5) * t.ln() - t + sum.ln()
    }
}

/// Logarithm of the Beta function: `ln B(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)`.
pub fn beta_log(a: f64, b: f64) -> StatsResult<f64> {
    if !(a > 0.0 && b > 0.0) {
        return Err(StatsError::InvalidDistributionParameter(format!(
            "beta_log: requires a>0, b>0; got a={a}, b={b}"
        )));
    }
    Ok(lgamma(a) + lgamma(b) - lgamma(a + b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lgamma_integer_values() {
        // lgamma(1) = 0 (Gamma(1) = 1)
        assert!(lgamma(1.0).abs() < 1e-10);
        // lgamma(2) = 0 (Gamma(2) = 1)
        assert!(lgamma(2.0).abs() < 1e-10);
        // lgamma(3) = ln(2)
        assert!((lgamma(3.0) - 2f64.ln()).abs() < 1e-10);
        // lgamma(4) = ln(6)
        assert!((lgamma(4.0) - 6f64.ln()).abs() < 1e-10);
        // lgamma(5) = ln(24)
        assert!((lgamma(5.0) - 24f64.ln()).abs() < 1e-10);
        // lgamma(10) = ln(362880)
        assert!((lgamma(10.0) - 362_880f64.ln()).abs() < 1e-8);
    }

    #[test]
    fn lgamma_half() {
        // lgamma(0.5) = ln(sqrt(pi))
        let expected = std::f64::consts::PI.sqrt().ln();
        assert!((lgamma(0.5) - expected).abs() < 1e-10);
    }

    #[test]
    fn beta_log_simple() {
        // B(1, 1) = 1 => ln = 0
        let v = beta_log(1.0, 1.0).expect("ok");
        assert!(v.abs() < 1e-10);
        // B(2, 3) = 1/12 => ln(1/12)
        let v = beta_log(2.0, 3.0).expect("ok");
        assert!((v - (1.0_f64 / 12.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn beta_log_rejects_nonpos() {
        assert!(beta_log(0.0, 1.0).is_err());
        assert!(beta_log(1.0, -2.0).is_err());
    }
}

//! Polygamma functions ψ⁽ⁿ⁾(x): digamma (n=0) and trigamma (n=1).
//!
//! Implementation: recurrence to shift x into a region where the asymptotic
//! expansion is valid (`x ≥ 6`), then series.

use crate::error::{NumericError, NumericResult};

/// Digamma function `ψ(x) = Γ'(x) / Γ(x)`.
pub fn digamma(x: f64) -> NumericResult<f64> {
    if x <= 0.0 && x == x.floor() {
        return Err(NumericError::OutOfDomain {
            value: x,
            function: "digamma (poles at non-positive integers)".into(),
        });
    }
    // reflection if x < 0.5 for stability
    let mut z = x;
    let mut result = 0.0_f64;
    if z < 0.5 {
        // ψ(1 - x) - ψ(x) = π cot(π x)
        let r = digamma(1.0 - z)?;
        return Ok(r - std::f64::consts::PI / (std::f64::consts::PI * z).tan());
    }
    while z < 6.0 {
        result -= 1.0 / z;
        z += 1.0;
    }
    // ψ(z) ≈ ln(z) - 1/(2z) - Σ B_{2k}/(2k z^{2k})
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    result += z.ln() - 0.5 * inv_z;
    let bernoulli = [
        1.0 / 12.0,
        -1.0 / 120.0,
        1.0 / 252.0,
        -1.0 / 240.0,
        5.0 / 660.0,
    ];
    let mut p = inv_z2;
    for &b in &bernoulli {
        result -= b * p;
        p *= inv_z2;
    }
    Ok(result)
}

/// Trigamma function `ψ'(x)`.
pub fn trigamma(x: f64) -> NumericResult<f64> {
    if x <= 0.0 && x == x.floor() {
        return Err(NumericError::OutOfDomain {
            value: x,
            function: "trigamma (poles at non-positive integers)".into(),
        });
    }
    let mut z = x;
    let mut result = 0.0_f64;
    if z < 0.5 {
        // ψ'(1 - x) + ψ'(x) = π² / sin²(π x)
        let r = trigamma(1.0 - z)?;
        let sin_pi_z = (std::f64::consts::PI * z).sin();
        return Ok(std::f64::consts::PI.powi(2) / (sin_pi_z * sin_pi_z) - r);
    }
    while z < 6.0 {
        result += 1.0 / (z * z);
        z += 1.0;
    }
    // ψ'(z) ≈ 1/z + 1/(2z²) + Σ B_{2k}/z^{2k+1}
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    result += inv_z + 0.5 * inv_z2;
    let bernoulli = [1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0, 5.0 / 66.0];
    let mut p = inv_z2 * inv_z;
    for &b in &bernoulli {
        result += b * p;
        p *= inv_z2;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digamma_one_is_neg_gamma() {
        let r = digamma(1.0).expect("ok");
        assert!((r + 0.577_215_664_901_532_9).abs() < 1.0e-8);
    }

    #[test]
    fn digamma_half() {
        // ψ(1/2) = -γ - 2 ln 2
        let r = digamma(0.5).expect("ok");
        let expected = -0.577_215_664_901_532_9 - 2.0 * 2.0_f64.ln();
        assert!((r - expected).abs() < 1.0e-6);
    }

    #[test]
    fn trigamma_one_is_pi_sq_six() {
        let r = trigamma(1.0).expect("ok");
        assert!((r - std::f64::consts::PI.powi(2) / 6.0).abs() < 1.0e-8);
    }

    #[test]
    fn digamma_pole_err() {
        let r = digamma(0.0);
        assert!(matches!(r, Err(NumericError::OutOfDomain { .. })));
    }
}

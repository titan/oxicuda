//! Airy functions `Ai(x)` and `Bi(x)`.
//!
//! Power series for `|x| < 7`, asymptotic expansion for larger `|x|`.

use crate::error::NumericResult;

const GAMMA_TWO_THIRDS: f64 = 1.354_117_939_426_400_4;
const GAMMA_ONE_THIRD: f64 = 2.678_938_534_707_747_6;

fn coeff_ai() -> f64 {
    1.0 / (3.0_f64.powf(2.0 / 3.0) * GAMMA_TWO_THIRDS)
}

fn coeff_ai_g() -> f64 {
    1.0 / (3.0_f64.powf(1.0 / 3.0) * GAMMA_ONE_THIRD)
}

fn airy_series_f(x: f64) -> f64 {
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    let z = x.powi(3);
    for k in 1..60 {
        let denom = ((3 * k) as f64) * ((3 * k - 1) as f64);
        term *= z / denom;
        sum += term;
        if term.abs() < 1.0e-18 * sum.abs() {
            break;
        }
    }
    sum
}

fn airy_series_g(x: f64) -> f64 {
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    let z = x.powi(3);
    for k in 1..60 {
        let denom = ((3 * k) as f64) * ((3 * k + 1) as f64);
        term *= z / denom;
        sum += term;
        if term.abs() < 1.0e-18 * sum.abs() {
            break;
        }
    }
    x * sum
}

/// Ai(x).
pub fn airy_ai(x: f64) -> NumericResult<f64> {
    if x.abs() < 7.0 {
        Ok(coeff_ai() * airy_series_f(x) - coeff_ai_g() * airy_series_g(x))
    } else if x > 0.0 {
        let z = 2.0 / 3.0 * x.powf(1.5);
        let pre = (-z).exp() / (2.0 * std::f64::consts::PI.sqrt() * x.powf(0.25));
        Ok(pre)
    } else {
        let a = (-x).powf(0.25);
        let z = 2.0 / 3.0 * (-x).powf(1.5);
        let pre = 1.0 / (std::f64::consts::PI.sqrt() * a);
        Ok(pre * (z + std::f64::consts::FRAC_PI_4).sin())
    }
}

/// Bi(x).
pub fn airy_bi(x: f64) -> NumericResult<f64> {
    if x.abs() < 7.0 {
        Ok(3.0_f64.sqrt() * (coeff_ai() * airy_series_f(x) + coeff_ai_g() * airy_series_g(x)))
    } else if x > 0.0 {
        let z = 2.0 / 3.0 * x.powf(1.5);
        Ok(z.exp() / (std::f64::consts::PI.sqrt() * x.powf(0.25)))
    } else {
        let a = (-x).powf(0.25);
        let z = 2.0 / 3.0 * (-x).powf(1.5);
        let pre = 1.0 / (std::f64::consts::PI.sqrt() * a);
        Ok(pre * (z + std::f64::consts::FRAC_PI_4).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ai_at_zero() {
        let r = airy_ai(0.0).expect("ok");
        let expected = 1.0 / (3.0_f64.powf(2.0 / 3.0) * GAMMA_TWO_THIRDS);
        assert!((r - expected).abs() < 1.0e-12);
    }

    #[test]
    fn bi_at_zero() {
        let r = airy_bi(0.0).expect("ok");
        let expected = 1.0 / (3.0_f64.powf(1.0 / 6.0) * GAMMA_TWO_THIRDS);
        assert!((r - expected).abs() < 1.0e-12);
    }

    #[test]
    fn ai_decreasing_positive() {
        let a = airy_ai(0.5).expect("ok");
        let b = airy_ai(1.0).expect("ok");
        assert!(a > b);
        assert!(b > 0.0);
    }
}

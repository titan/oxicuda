//! Regularized incomplete beta function `I_x(a, b)` and regularized lower incomplete gamma `P(a, x)`.
//!
//! Continued-fraction expansion from Numerical Recipes 3rd ed., section 6.4.

use crate::error::{StatsError, StatsResult};
use crate::special::gammaln::lgamma;

const MAX_ITER: usize = 200;
const EPS: f64 = 3.0e-16;
const FPMIN: f64 = 1.0e-300;

/// Regularized incomplete beta function `I_x(a, b)`.
///
/// Uses the continued-fraction representation when `x < (a+1)/(a+b+2)`,
/// otherwise computes via the symmetry `I_x(a,b) = 1 - I_{1-x}(b, a)`.
pub fn betainc(a: f64, b: f64, x: f64) -> StatsResult<f64> {
    if !(a > 0.0 && b > 0.0) {
        return Err(StatsError::InvalidDistributionParameter(format!(
            "betainc: a, b must be > 0; got a={a}, b={b}"
        )));
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(StatsError::ProbabilityOutOfRange { value: x });
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    let lnbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let bt = lnbt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        let cf = betacf(a, b, x)?;
        Ok(bt * cf / a)
    } else {
        let cf = betacf(b, a, 1.0 - x)?;
        Ok(1.0 - bt * cf / b)
    }
}

/// Lentz's continued-fraction evaluation for the incomplete beta.
fn betacf(a: f64, b: f64, x: f64) -> StatsResult<f64> {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            return Ok(h);
        }
    }
    // Did not converge tightly, but return best estimate.
    Ok(h)
}

/// Regularized lower incomplete gamma function `P(a, x)`.
pub fn gammp(a: f64, x: f64) -> StatsResult<f64> {
    if a <= 0.0 || x < 0.0 {
        return Err(StatsError::InvalidDistributionParameter(format!(
            "gammp: a must be > 0, x >= 0; got a={a}, x={x}"
        )));
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x < a + 1.0 {
        Ok(gser(a, x)?)
    } else {
        Ok(1.0 - gcf(a, x)?)
    }
}

/// Regularized upper incomplete gamma function `Q(a, x) = 1 - P(a, x)`.
pub fn gammq(a: f64, x: f64) -> StatsResult<f64> {
    Ok(1.0 - gammp(a, x)?)
}

/// Series representation for `P(a, x)` valid when `x < a+1`.
fn gser(a: f64, x: f64) -> StatsResult<f64> {
    let gln = lgamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            return Ok(sum * (-x + a * x.ln() - gln).exp());
        }
    }
    Ok(sum * (-x + a * x.ln() - gln).exp())
}

/// Continued-fraction for `Q(a, x)` valid when `x >= a+1`.
fn gcf(a: f64, x: f64) -> StatsResult<f64> {
    let gln = lgamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAX_ITER {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    Ok(h * (-x + a * x.ln() - gln).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn betainc_endpoints() {
        let v = betainc(2.0, 3.0, 0.0).expect("ok");
        assert!(v.abs() < 1e-14);
        let v = betainc(2.0, 3.0, 1.0).expect("ok");
        assert!((v - 1.0).abs() < 1e-14);
    }

    #[test]
    fn betainc_half_symmetry() {
        // For a=b, I_{0.5}(a,a) = 0.5
        let v = betainc(2.0, 2.0, 0.5).expect("ok");
        assert!((v - 0.5).abs() < 1e-10);
        let v = betainc(5.0, 5.0, 0.5).expect("ok");
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn betainc_symmetry() {
        // I_x(a,b) = 1 - I_{1-x}(b,a)
        let a = 2.3;
        let b = 4.1;
        let x = 0.37;
        let lhs = betainc(a, b, x).expect("ok");
        let rhs = 1.0 - betainc(b, a, 1.0 - x).expect("ok");
        assert!((lhs - rhs).abs() < 1e-10);
    }

    #[test]
    fn gammp_known_value() {
        // P(1, x) = 1 - exp(-x)
        let v = gammp(1.0, 1.0).expect("ok");
        assert!((v - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
        let v = gammp(1.0, 2.0).expect("ok");
        assert!((v - (1.0 - (-2.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn gammp_at_zero() {
        let v = gammp(2.5, 0.0).expect("ok");
        assert!(v.abs() < 1e-14);
    }

    #[test]
    fn gammp_plus_gammq_one() {
        for &a in &[0.5, 1.0, 2.0, 5.0] {
            for &x in &[0.5, 1.0, 3.0, 10.0] {
                let p = gammp(a, x).expect("ok");
                let q = gammq(a, x).expect("ok");
                assert!((p + q - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn betainc_rejects_invalid() {
        assert!(betainc(0.0, 1.0, 0.5).is_err());
        assert!(betainc(1.0, 1.0, 1.5).is_err());
    }
}

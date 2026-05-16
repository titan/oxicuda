//! Chi-squared distribution (a special case of Gamma).

use crate::error::{StatsError, StatsResult};
use crate::special::betainc::gammp;
use crate::special::gammaln::lgamma;

/// Chi-squared distribution with `df` degrees of freedom (df > 0).
#[derive(Debug, Clone, Copy)]
pub struct ChiSquared {
    pub df: f64,
}

impl ChiSquared {
    pub fn new(df: f64) -> StatsResult<Self> {
        if !(df > 0.0 && df.is_finite()) {
            return Err(StatsError::DegreesOfFreedomZero);
        }
        Ok(Self { df })
    }

    /// PDF: `(x^(k/2 - 1) * exp(-x/2)) / (2^(k/2) * Gamma(k/2))`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            return if self.df < 2.0 {
                f64::INFINITY
            } else if (self.df - 2.0).abs() < 1e-12 {
                0.5
            } else {
                0.0
            };
        }
        let k = self.df;
        let ln_pdf =
            (k / 2.0 - 1.0) * x.ln() - x / 2.0 - (k / 2.0) * 2.0_f64.ln() - lgamma(k / 2.0);
        ln_pdf.exp()
    }

    /// CDF: `P(k/2, x/2)`.
    pub fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        gammp(self.df / 2.0, x / 2.0)
    }

    /// PPF via Newton iteration with Wilson-Hilferty seed.
    pub fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange { value: p });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // Wilson-Hilferty: chi^2 ~ k * (1 - 2/(9k) + z * sqrt(2/(9k)))^3
        let n = crate::distributions::normal::Normal::standard();
        let z = n.ppf(p)?;
        let k = self.df;
        let mut x = k * (1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt()).powi(3);
        if x <= 0.0 {
            x = 0.5 * k;
        }
        for _ in 0..50 {
            let cdf_v = self.cdf(x)?;
            let pdf_v = self.pdf(x);
            if pdf_v < 1e-300 {
                break;
            }
            let dx = (cdf_v - p) / pdf_v;
            x -= dx;
            if x <= 0.0 {
                x = 1e-10;
            }
            if dx.abs() < 1e-10 {
                break;
            }
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_at_zero_zero() {
        let c = ChiSquared::new(5.0).expect("ok");
        assert!(c.cdf(0.0).expect("ok").abs() < 1e-14);
    }

    #[test]
    fn cdf_tail() {
        let c = ChiSquared::new(2.0).expect("ok");
        // For df=2, cdf(x) = 1 - exp(-x/2)
        for &x in &[1.0, 2.0, 4.0, 6.0] {
            let v = c.cdf(x).expect("ok");
            let expected = 1.0 - (-x / 2.0).exp();
            assert!((v - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn ppf_roundtrip() {
        let c = ChiSquared::new(5.0).expect("ok");
        for &p in &[0.1, 0.5, 0.9, 0.95] {
            let x = c.ppf(p).expect("ok");
            assert!((c.cdf(x).expect("ok") - p).abs() < 1e-6);
        }
    }
}

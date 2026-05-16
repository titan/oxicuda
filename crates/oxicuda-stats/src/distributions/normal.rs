//! Normal (Gaussian) distribution.

use crate::error::{StatsError, StatsResult};
use crate::special::erf::{erf, erfinv};

/// Normal distribution N(mu, sigma).
#[derive(Debug, Clone, Copy)]
pub struct Normal {
    pub mean: f64,
    pub std_dev: f64,
}

impl Normal {
    /// Construct a Normal distribution with the given mean and (positive) standard deviation.
    pub fn new(mean: f64, std_dev: f64) -> StatsResult<Self> {
        if !(std_dev > 0.0 && std_dev.is_finite()) {
            return Err(StatsError::InvalidDistributionParameter(format!(
                "Normal: std_dev must be > 0; got {std_dev}"
            )));
        }
        if !mean.is_finite() {
            return Err(StatsError::InvalidDistributionParameter(format!(
                "Normal: mean must be finite; got {mean}"
            )));
        }
        Ok(Self { mean, std_dev })
    }

    /// Standard normal N(0, 1).
    #[must_use]
    pub fn standard() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
        }
    }

    /// Probability density function.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        let coef = 1.0 / (self.std_dev * (2.0 * std::f64::consts::PI).sqrt());
        coef * (-0.5 * z * z).exp()
    }

    /// Logarithm of the PDF for stable computation.
    #[must_use]
    pub fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        -0.5 * (2.0 * std::f64::consts::PI).ln() - self.std_dev.ln() - 0.5 * z * z
    }

    /// Cumulative distribution function.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
    }

    /// Inverse CDF (quantile / ppf).
    pub fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange { value: p });
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let z = std::f64::consts::SQRT_2 * erfinv(2.0 * p - 1.0)?;
        Ok(self.mean + self.std_dev * z)
    }

    /// Survival function `1 - cdf(x)`.
    #[must_use]
    pub fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_at_mean_maximum() {
        let n = Normal::new(0.0, 1.0).expect("ok");
        let v0 = n.pdf(0.0);
        let v1 = n.pdf(1.0);
        let v_neg = n.pdf(-1.0);
        assert!(v0 > v1);
        assert!((v1 - v_neg).abs() < 1e-14);
        assert!((v0 - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cdf_at_mean_half() {
        let n = Normal::new(2.0, 0.5).expect("ok");
        assert!((n.cdf(2.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn cdf_tails() {
        let n = Normal::standard();
        assert!(n.cdf(-5.0) < 1e-6);
        assert!(n.cdf(5.0) > 1.0 - 1e-6);
    }

    #[test]
    fn ppf_roundtrip() {
        let n = Normal::new(0.0, 1.0).expect("ok");
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.975] {
            let x = n.ppf(p).expect("ok");
            assert!((n.cdf(x) - p).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_zero_std_dev() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
    }
}

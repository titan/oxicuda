//! Exponential distribution.

use crate::error::{StatsError, StatsResult};

/// Exponential distribution with rate `rate > 0`.
#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    pub rate: f64,
}

impl Exponential {
    pub fn new(rate: f64) -> StatsResult<Self> {
        if !(rate > 0.0 && rate.is_finite()) {
            return Err(StatsError::InvalidDistributionParameter(format!(
                "Exponential: rate must be > 0; got {rate}"
            )));
        }
        Ok(Self { rate })
    }

    /// PDF: `rate * exp(-rate * x)` for `x >= 0`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * (-self.rate * x).exp()
        }
    }

    /// CDF: `1 - exp(-rate * x)` for `x >= 0`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.rate * x).exp()
        }
    }

    /// PPF: `-ln(1 - p) / rate`.
    pub fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange { value: p });
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(-((1.0 - p).ln()) / self.rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_known_values() {
        let e = Exponential::new(1.0).expect("ok");
        assert!((e.cdf(0.0)).abs() < 1e-14);
        assert!((e.cdf(1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
    }

    #[test]
    fn ppf_roundtrip() {
        let e = Exponential::new(2.0).expect("ok");
        for &p in &[0.1, 0.5, 0.9] {
            let x = e.ppf(p).expect("ok");
            assert!((e.cdf(x) - p).abs() < 1e-10);
        }
    }
}

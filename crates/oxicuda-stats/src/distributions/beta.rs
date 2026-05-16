//! Beta distribution.

use crate::error::{StatsError, StatsResult};
use crate::special::betainc::betainc;
use crate::special::gammaln::lgamma;

/// Beta distribution with parameters `alpha` and `beta` (both > 0).
#[derive(Debug, Clone, Copy)]
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> StatsResult<Self> {
        if !(alpha > 0.0 && beta > 0.0 && alpha.is_finite() && beta.is_finite()) {
            return Err(StatsError::InvalidDistributionParameter(format!(
                "Beta: alpha,beta must be > 0; got alpha={alpha}, beta={beta}"
            )));
        }
        Ok(Self { alpha, beta })
    }

    /// PDF on [0, 1]: `x^(a-1) * (1-x)^(b-1) / B(a, b)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        if x == 0.0 {
            return if self.alpha < 1.0 {
                f64::INFINITY
            } else if self.alpha > 1.0 {
                0.0
            } else {
                self.beta
            };
        }
        if x == 1.0 {
            return if self.beta < 1.0 {
                f64::INFINITY
            } else if self.beta > 1.0 {
                0.0
            } else {
                self.alpha
            };
        }
        let ln_b = lgamma(self.alpha) + lgamma(self.beta) - lgamma(self.alpha + self.beta);
        ((self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln() - ln_b).exp()
    }

    /// CDF: `I_x(a, b)`.
    pub fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        if x >= 1.0 {
            return Ok(1.0);
        }
        betainc(self.alpha, self.beta, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_when_a_b_one() {
        let b = Beta::new(1.0, 1.0).expect("ok");
        assert!((b.pdf(0.3) - 1.0).abs() < 1e-12);
        assert!((b.pdf(0.7) - 1.0).abs() < 1e-12);
        assert!((b.cdf(0.5).expect("ok") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn cdf_at_endpoints() {
        let b = Beta::new(2.0, 3.0).expect("ok");
        assert!(b.cdf(0.0).expect("ok").abs() < 1e-14);
        assert!((b.cdf(1.0).expect("ok") - 1.0).abs() < 1e-14);
    }
}

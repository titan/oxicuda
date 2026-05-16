//! Poisson distribution.

use crate::error::{StatsError, StatsResult};
use crate::special::betainc::gammq;
use crate::special::gammaln::lgamma;

/// Poisson distribution with rate `lambda > 0`.
#[derive(Debug, Clone, Copy)]
pub struct Poisson {
    pub lambda: f64,
}

impl Poisson {
    pub fn new(lambda: f64) -> StatsResult<Self> {
        if !(lambda > 0.0 && lambda.is_finite()) {
            return Err(StatsError::InvalidDistributionParameter(format!(
                "Poisson: lambda must be > 0; got {lambda}"
            )));
        }
        Ok(Self { lambda })
    }

    /// PMF: `lambda^k * exp(-lambda) / k!`.
    #[must_use]
    pub fn pmf(&self, k: usize) -> f64 {
        let kf = k as f64;
        (kf * self.lambda.ln() - self.lambda - lgamma(kf + 1.0)).exp()
    }

    /// CDF: `Pr(X <= k) = Q(k+1, lambda) = 1 - P(k+1, lambda)`.
    pub fn cdf(&self, k: i64) -> StatsResult<f64> {
        if k < 0 {
            return Ok(0.0);
        }
        gammq((k + 1) as f64, self.lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pmf_sums_to_one() {
        let p = Poisson::new(3.5).expect("ok");
        let s: f64 = (0..50).map(|k| p.pmf(k)).sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pmf_known_value() {
        let p = Poisson::new(1.0).expect("ok");
        // P(X=0) = exp(-1)
        assert!((p.pmf(0) - (-1.0_f64).exp()).abs() < 1e-12);
        // P(X=1) = exp(-1)
        assert!((p.pmf(1) - (-1.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn cdf_increasing() {
        let p = Poisson::new(2.0).expect("ok");
        let v0 = p.cdf(0).expect("ok");
        let v1 = p.cdf(1).expect("ok");
        let v2 = p.cdf(2).expect("ok");
        assert!(v0 < v1 && v1 < v2);
    }
}

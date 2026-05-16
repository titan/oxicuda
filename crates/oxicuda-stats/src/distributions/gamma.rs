//! Gamma distribution (shape-rate parameterization).

use crate::error::{StatsError, StatsResult};
use crate::special::betainc::gammp;
use crate::special::gammaln::lgamma;

/// Gamma distribution with shape `k` and rate `theta_inv` (`scale = 1/theta_inv`).
///
/// We use shape-scale parameterization: pdf = x^(k-1) exp(-x/scale) / (Gamma(k) * scale^k).
#[derive(Debug, Clone, Copy)]
pub struct Gamma {
    pub shape: f64,
    pub scale: f64,
}

impl Gamma {
    pub fn new(shape: f64, scale: f64) -> StatsResult<Self> {
        if !(shape > 0.0 && scale > 0.0 && shape.is_finite() && scale.is_finite()) {
            return Err(StatsError::InvalidDistributionParameter(format!(
                "Gamma: shape,scale must be > 0; got shape={shape}, scale={scale}"
            )));
        }
        Ok(Self { shape, scale })
    }

    /// PDF: `x^(k-1) exp(-x/scale) / (Gamma(k) scale^k)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            return if self.shape < 1.0 {
                f64::INFINITY
            } else if (self.shape - 1.0).abs() < 1e-12 {
                1.0 / self.scale
            } else {
                0.0
            };
        }
        let ln_pdf = (self.shape - 1.0) * x.ln()
            - x / self.scale
            - lgamma(self.shape)
            - self.shape * self.scale.ln();
        ln_pdf.exp()
    }

    /// CDF: `P(shape, x/scale)`.
    pub fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        gammp(self.shape, x / self.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_special_case() {
        // Gamma(1, scale) = Exp(1/scale)
        let g = Gamma::new(1.0, 2.0).expect("ok");
        // CDF should equal 1 - exp(-x/2)
        for &x in &[0.5, 1.0, 3.0] {
            let v = g.cdf(x).expect("ok");
            let expected = 1.0 - (-x / 2.0).exp();
            assert!((v - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn rejects_invalid() {
        assert!(Gamma::new(0.0, 1.0).is_err());
        assert!(Gamma::new(1.0, -1.0).is_err());
    }
}

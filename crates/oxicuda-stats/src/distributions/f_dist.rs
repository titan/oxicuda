//! F-distribution.

use crate::error::{StatsError, StatsResult};
use crate::special::betainc::betainc;
use crate::special::gammaln::lgamma;

/// F-distribution with numerator df1 and denominator df2.
#[derive(Debug, Clone, Copy)]
pub struct FDist {
    pub df1: f64,
    pub df2: f64,
}

impl FDist {
    pub fn new(df1: f64, df2: f64) -> StatsResult<Self> {
        if !(df1 > 0.0 && df2 > 0.0 && df1.is_finite() && df2.is_finite()) {
            return Err(StatsError::DegreesOfFreedomZero);
        }
        Ok(Self { df1, df2 })
    }

    /// PDF: `(1 / B(d1/2, d2/2)) * (d1/d2)^(d1/2) * x^(d1/2 - 1) * (1 + d1*x/d2)^(-(d1+d2)/2)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let d1 = self.df1;
        let d2 = self.df2;
        let ln_b = lgamma(d1 / 2.0) + lgamma(d2 / 2.0) - lgamma((d1 + d2) / 2.0);
        let ln_pdf = (d1 / 2.0) * (d1 / d2).ln() + (d1 / 2.0 - 1.0) * x.ln()
            - ((d1 + d2) / 2.0) * (1.0 + d1 * x / d2).ln()
            - ln_b;
        ln_pdf.exp()
    }

    /// CDF: `I_{d1*x / (d1*x + d2)}(d1/2, d2/2)`.
    pub fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        let d1 = self.df1;
        let d2 = self.df2;
        let t = d1 * x / (d1 * x + d2);
        let t = t.clamp(1e-300, 1.0 - 1e-15);
        betainc(d1 / 2.0, d2 / 2.0, t)
    }

    /// PPF via Newton iteration.
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
        // Reasonable seed
        let mut x = 1.0;
        for _ in 0..100 {
            let cdf_v = self.cdf(x)?;
            let pdf_v = self.pdf(x);
            if pdf_v < 1e-300 {
                break;
            }
            let dx = (cdf_v - p) / pdf_v;
            let nx = x - dx;
            x = if nx <= 0.0 { x / 2.0 } else { nx };
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
    fn cdf_increasing() {
        let f = FDist::new(5.0, 10.0).expect("ok");
        let v1 = f.cdf(0.5).expect("ok");
        let v2 = f.cdf(1.0).expect("ok");
        let v3 = f.cdf(3.0).expect("ok");
        assert!(v1 < v2);
        assert!(v2 < v3);
    }

    #[test]
    fn cdf_zero_at_origin() {
        let f = FDist::new(3.0, 7.0).expect("ok");
        assert!(f.cdf(0.0).expect("ok").abs() < 1e-12);
    }

    #[test]
    fn ppf_roundtrip() {
        let f = FDist::new(5.0, 10.0).expect("ok");
        for &p in &[0.25, 0.5, 0.75, 0.95] {
            let x = f.ppf(p).expect("ok");
            assert!((f.cdf(x).expect("ok") - p).abs() < 1e-6);
        }
    }
}

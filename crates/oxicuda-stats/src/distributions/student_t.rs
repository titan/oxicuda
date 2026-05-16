//! Student t-distribution.

use crate::error::{StatsError, StatsResult};
use crate::special::betainc::betainc;
use crate::special::gammaln::lgamma;

/// Student t-distribution with `df` degrees of freedom.
#[derive(Debug, Clone, Copy)]
pub struct StudentT {
    pub df: f64,
}

impl StudentT {
    pub fn new(df: f64) -> StatsResult<Self> {
        if !(df > 0.0 && df.is_finite()) {
            return Err(StatsError::DegreesOfFreedomZero);
        }
        Ok(Self { df })
    }

    /// PDF: `Gamma((df+1)/2) / (sqrt(df * pi) * Gamma(df/2)) * (1 + t^2/df)^(-(df+1)/2)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        let df = self.df;
        let ln_coef =
            lgamma((df + 1.0) / 2.0) - 0.5 * (df * std::f64::consts::PI).ln() - lgamma(df / 2.0);
        let ln_term = -(df + 1.0) / 2.0 * (1.0 + x * x / df).ln();
        (ln_coef + ln_term).exp()
    }

    /// CDF via regularized incomplete beta: for `t >= 0`,
    /// `cdf(t) = 1 - 0.5 * I_{df/(df + t^2)}(df/2, 0.5)`; for negative t, symmetry.
    pub fn cdf(&self, t: f64) -> StatsResult<f64> {
        let df = self.df;
        if t == 0.0 {
            return Ok(0.5);
        }
        let x = df / (df + t * t);
        // Clamp into (0, 1) to avoid boundary issues.
        let x = x.clamp(1e-300, 1.0 - 1e-15);
        let ib = betainc(df / 2.0, 0.5, x)?;
        if t > 0.0 {
            Ok(1.0 - 0.5 * ib)
        } else {
            Ok(0.5 * ib)
        }
    }

    /// PPF via Newton iteration on `cdf(t) - p`.
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
        if (p - 0.5).abs() < 1e-14 {
            return Ok(0.0);
        }
        let df = self.df;
        // Initial guess via Cornish-Fisher / normal approximation refined for small df.
        let n = crate::distributions::normal::Normal::standard();
        let z = n.ppf(p)?;
        let mut t = if df > 30.0 {
            z
        } else {
            // Refine seed via Hill 1970 approximation
            let g1 = (z * z + 1.0) / 4.0;
            let g2 = ((5.0 * z * z + 16.0) * z * z + 3.0) / 96.0;
            z + g1 * z / df + g2 * z / (df * df)
        };
        for _ in 0..30 {
            let cdf_v = self.cdf(t)?;
            let pdf_v = self.pdf(t);
            if pdf_v < 1e-300 {
                break;
            }
            let dt = (cdf_v - p) / pdf_v;
            t -= dt;
            if dt.abs() < 1e-12 {
                break;
            }
        }
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_symmetric() {
        let t = StudentT::new(5.0).expect("ok");
        let p1 = t.pdf(1.0);
        let p2 = t.pdf(-1.0);
        assert!((p1 - p2).abs() < 1e-14);
    }

    #[test]
    fn cdf_at_zero_half() {
        let t = StudentT::new(10.0).expect("ok");
        let v = t.cdf(0.0).expect("ok");
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn cdf_tails() {
        let t = StudentT::new(10.0).expect("ok");
        assert!(t.cdf(-10.0).expect("ok") < 1e-4);
        assert!(t.cdf(10.0).expect("ok") > 1.0 - 1e-4);
    }

    #[test]
    fn ppf_roundtrip() {
        let t = StudentT::new(15.0).expect("ok");
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99] {
            let x = t.ppf(p).expect("ok");
            assert!((t.cdf(x).expect("ok") - p).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_nonpos_df() {
        assert!(StudentT::new(0.0).is_err());
        assert!(StudentT::new(-1.0).is_err());
    }
}

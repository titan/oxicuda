//! Binomial distribution `B(n, p)`.

use crate::error::{StatsError, StatsResult};
use crate::special::betainc::betainc;
use crate::special::gammaln::lgamma;

/// Binomial distribution with `n` trials and success probability `p`.
#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    pub n: usize,
    pub p: f64,
}

impl Binomial {
    pub fn new(n: usize, p: f64) -> StatsResult<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange { value: p });
        }
        Ok(Self { n, p })
    }

    /// PMF: `C(n, k) * p^k * (1-p)^(n-k)`.
    #[must_use]
    pub fn pmf(&self, k: usize) -> f64 {
        if k > self.n {
            return 0.0;
        }
        let n = self.n as f64;
        let kf = k as f64;
        let ln_coef = lgamma(n + 1.0) - lgamma(kf + 1.0) - lgamma(n - kf + 1.0);
        let ln_pmf = if self.p == 0.0 {
            if k == 0 {
                0.0
            } else {
                return 0.0;
            }
        } else if self.p == 1.0 {
            if k == self.n {
                0.0
            } else {
                return 0.0;
            }
        } else {
            ln_coef + kf * self.p.ln() + (n - kf) * (1.0 - self.p).ln()
        };
        ln_pmf.exp()
    }

    /// CDF: `Pr(X <= k)` via the regularized incomplete beta:
    /// `F(k; n, p) = I_{1-p}(n-k, k+1)` for `k < n`.
    pub fn cdf(&self, k: i64) -> StatsResult<f64> {
        if k < 0 {
            return Ok(0.0);
        }
        let k = k as usize;
        if k >= self.n {
            return Ok(1.0);
        }
        let a = (self.n - k) as f64;
        let b = (k + 1) as f64;
        betainc(a, b, 1.0 - self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pmf_sums_to_one() {
        let b = Binomial::new(10, 0.3).expect("ok");
        let s: f64 = (0..=10).map(|k| b.pmf(k)).sum();
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pmf_known_values() {
        let b = Binomial::new(5, 0.5).expect("ok");
        // P(X=2) = 10 * 0.5^5 = 10/32 = 0.3125
        assert!((b.pmf(2) - 0.3125).abs() < 1e-10);
    }

    #[test]
    fn cdf_at_n_one() {
        let b = Binomial::new(8, 0.4).expect("ok");
        let v = b.cdf(8).expect("ok");
        assert!((v - 1.0).abs() < 1e-10);
    }
}

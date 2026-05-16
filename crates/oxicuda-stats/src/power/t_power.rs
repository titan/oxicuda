//! Power and sample-size calculation for two-sample t-tests.

use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};

/// Normal-approximation power for a two-sample, two-sided t-test.
pub fn t_power_two_sample(d: f64, n_per_group: usize, alpha: f64) -> StatsResult<f64> {
    if n_per_group < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: n_per_group,
            need: 2,
        });
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(StatsError::ProbabilityOutOfRange { value: alpha });
    }
    let n = n_per_group as f64;
    let dist = Normal::standard();
    let z_alpha = dist.ppf(1.0 - alpha / 2.0)?;
    let ncp = d * (n / 2.0).sqrt();
    let power = 1.0 - dist.cdf(z_alpha - ncp) + dist.cdf(-z_alpha - ncp);
    Ok(power.clamp(0.0, 1.0))
}

/// Required sample size per group for a two-sample t-test (normal approximation).
pub fn t_sample_size(d: f64, alpha: f64, power: f64) -> StatsResult<usize> {
    if !(0.0..1.0).contains(&alpha) || !(0.0..1.0).contains(&power) {
        return Err(StatsError::ProbabilityOutOfRange {
            value: alpha.max(power),
        });
    }
    if d.abs() < 1e-12 {
        return Err(StatsError::InvalidParameter {
            name: "d".into(),
            reason: "effect size must not be zero".into(),
        });
    }
    let dist = Normal::standard();
    let z_alpha = dist.ppf(1.0 - alpha / 2.0)?;
    let z_beta = dist.ppf(power)?;
    let n = 2.0 * ((z_alpha + z_beta) / d).powi(2);
    Ok(n.ceil() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_power_increases_with_n() {
        let p1 = t_power_two_sample(0.5, 20, 0.05).expect("ok");
        let p2 = t_power_two_sample(0.5, 50, 0.05).expect("ok");
        assert!(p2 > p1);
    }

    #[test]
    fn t_sample_size_basic() {
        let n = t_sample_size(0.5, 0.05, 0.8).expect("ok");
        // Should be ~63 per group; allow some slack
        assert!((50..=80).contains(&n));
    }
}

//! One-sample, two-sample (Student / Welch), and paired t-tests.

use crate::descriptive::summary::{mean, sample_var};
use crate::distributions::student_t::StudentT;
use crate::error::{StatsError, StatsResult};

/// Result of a t-test.
#[derive(Debug, Clone, Copy)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub df: f64,
    pub p_value_two_sided: f64,
    pub p_value_left: f64,
    pub p_value_right: f64,
}

fn p_values(t: f64, df: f64) -> StatsResult<TTestResult> {
    let dist = StudentT::new(df)?;
    let cdf_t = dist.cdf(t)?;
    let two = 2.0 * cdf_t.min(1.0 - cdf_t);
    Ok(TTestResult {
        t_statistic: t,
        df,
        p_value_two_sided: two.clamp(0.0, 1.0),
        p_value_left: cdf_t,
        p_value_right: 1.0 - cdf_t,
    })
}

/// One-sample t-test: H0: mu = mu0.
pub fn one_sample_t(x: &[f64], mu0: f64) -> StatsResult<TTestResult> {
    if x.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: x.len(),
            need: 2,
        });
    }
    let xbar = mean(x)?;
    let s2 = sample_var(x)?;
    if s2 <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "zero variance in one-sample t-test".into(),
        ));
    }
    let n = x.len() as f64;
    let t = (xbar - mu0) / (s2 / n).sqrt();
    p_values(t, n - 1.0)
}

/// Two-sample Student t-test assuming equal variances.
pub fn two_sample_t(x1: &[f64], x2: &[f64]) -> StatsResult<TTestResult> {
    if x1.len() < 2 || x2.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: x1.len().min(x2.len()),
            need: 2,
        });
    }
    let m1 = mean(x1)?;
    let m2 = mean(x2)?;
    let s1 = sample_var(x1)?;
    let s2 = sample_var(x2)?;
    let n1 = x1.len() as f64;
    let n2 = x2.len() as f64;
    let sp2 = ((n1 - 1.0) * s1 + (n2 - 1.0) * s2) / (n1 + n2 - 2.0);
    if sp2 <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "non-positive pooled variance".into(),
        ));
    }
    let t = (m1 - m2) / (sp2 * (1.0 / n1 + 1.0 / n2)).sqrt();
    p_values(t, n1 + n2 - 2.0)
}

/// Welch's t-test for unequal variances.
pub fn welch_t(x1: &[f64], x2: &[f64]) -> StatsResult<TTestResult> {
    if x1.len() < 2 || x2.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: x1.len().min(x2.len()),
            need: 2,
        });
    }
    let m1 = mean(x1)?;
    let m2 = mean(x2)?;
    let s1 = sample_var(x1)?;
    let s2 = sample_var(x2)?;
    let n1 = x1.len() as f64;
    let n2 = x2.len() as f64;
    let v1 = s1 / n1;
    let v2 = s2 / n2;
    let t = (m1 - m2) / (v1 + v2).sqrt();
    let num = (v1 + v2).powi(2);
    let den = v1 * v1 / (n1 - 1.0) + v2 * v2 / (n2 - 1.0);
    let df = num / den;
    p_values(t, df)
}

/// Paired t-test on per-pair differences.
pub fn paired_t(x1: &[f64], x2: &[f64]) -> StatsResult<TTestResult> {
    if x1.len() != x2.len() {
        return Err(StatsError::DimensionMismatch {
            a: x1.len(),
            b: x2.len(),
        });
    }
    let diffs: Vec<f64> = x1.iter().zip(x2).map(|(a, b)| a - b).collect();
    one_sample_t(&diffs, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_sample_t_zero_when_mean_matches() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = one_sample_t(&x, 3.0).expect("ok");
        assert!(r.t_statistic.abs() < 1e-12);
        assert!((r.df - 4.0).abs() < 1e-12);
        assert!((r.p_value_two_sided - 1.0).abs() < 1e-6);
    }

    #[test]
    fn two_sample_t_positive_shift() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = [3.0, 4.0, 5.0, 6.0, 7.0];
        let r = two_sample_t(&x1, &x2).expect("ok");
        assert!(r.t_statistic < 0.0);
    }

    #[test]
    fn welch_t_unequal_sizes() {
        let x1 = [1.0, 2.0, 3.0];
        let x2 = [3.0, 4.0, 5.0, 6.0, 7.0];
        let r = welch_t(&x1, &x2).expect("ok");
        assert!(r.t_statistic.is_finite());
        assert!(r.df.is_finite() && r.df > 0.0);
    }

    #[test]
    fn paired_t_zero_when_paired_equal() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = x1;
        let r = paired_t(&x1, &x2);
        // zero variance triggers error
        assert!(r.is_err());
    }
}

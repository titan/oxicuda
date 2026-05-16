//! Pearson product-moment correlation with a t-test for significance.

use crate::descriptive::summary::mean;
use crate::distributions::student_t::StudentT;
use crate::error::{StatsError, StatsResult};

/// Pearson correlation result.
#[derive(Debug, Clone, Copy)]
pub struct PearsonResult {
    pub r: f64,
    pub t_statistic: f64,
    pub df: f64,
    pub p_value_two_sided: f64,
}

/// Pearson correlation coefficient and its t-statistic.
pub fn pearson_r(x: &[f64], y: &[f64]) -> StatsResult<PearsonResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    if x.len() < 3 {
        return Err(StatsError::InsufficientSampleSize {
            got: x.len(),
            need: 3,
        });
    }
    let mx = mean(x)?;
    let my = mean(y)?;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (xi, yi) in x.iter().zip(y) {
        sxy += (xi - mx) * (yi - my);
        sxx += (xi - mx).powi(2);
        syy += (yi - my).powi(2);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return Err(StatsError::NumericalInstability("zero variance".into()));
    }
    let r = sxy / (sxx * syy).sqrt();
    let r = r.clamp(-1.0, 1.0);
    let n = x.len() as f64;
    let df = n - 2.0;
    let t = r * (df / (1.0 - r * r).max(1e-300)).sqrt();
    let dist = StudentT::new(df)?;
    let cdf_t = dist.cdf(t)?;
    let p = 2.0 * cdf_t.min(1.0 - cdf_t);
    Ok(PearsonResult {
        r,
        t_statistic: t,
        df,
        p_value_two_sided: p.clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_r(&x, &y).expect("ok");
        assert!((r.r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pearson_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [-1.0, -2.0, -3.0, -4.0, -5.0];
        let r = pearson_r(&x, &y).expect("ok");
        assert!((r.r + 1.0).abs() < 1e-9);
    }

    #[test]
    fn pearson_no_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [3.0, 1.0, 4.0, 1.0, 5.0];
        let r = pearson_r(&x, &y).expect("ok");
        assert!(r.r.abs() < 1.0);
    }
}

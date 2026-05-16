//! Robust descriptive statistics: median absolute deviation, IQR, trimmed mean.

use crate::descriptive::quantile::quantile;
use crate::descriptive::summary::mean;
use crate::error::{StatsError, StatsResult};

/// Median absolute deviation (MAD) scaled to be a consistent estimator for the standard
/// deviation under normality: `MAD = 1.4826 * median(|x - median(x)|)`.
pub fn mad(x: &[f64]) -> StatsResult<f64> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let med = quantile(x, 0.5)?;
    let dev: Vec<f64> = x.iter().map(|v| (v - med).abs()).collect();
    let med_dev = quantile(&dev, 0.5)?;
    Ok(1.482_602_218_505_602 * med_dev)
}

/// Interquartile range `Q3 - Q1`.
pub fn iqr(x: &[f64]) -> StatsResult<f64> {
    let q1 = quantile(x, 0.25)?;
    let q3 = quantile(x, 0.75)?;
    Ok(q3 - q1)
}

/// Trimmed mean: drop a fraction `alpha` from each tail.
pub fn trimmed_mean(x: &[f64], alpha: f64) -> StatsResult<f64> {
    if !(0.0..0.5).contains(&alpha) {
        return Err(StatsError::InvalidParameter {
            name: "alpha".into(),
            reason: format!("must be in [0, 0.5), got {alpha}"),
        });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let mut sorted: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let k = (n as f64 * alpha).floor() as usize;
    if 2 * k >= n {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: 2 * k + 1,
        });
    }
    mean(&sorted[k..n - k])
}

/// Winsorized mean: replace tails with the quantile boundary rather than dropping them.
pub fn winsorized_mean(x: &[f64], alpha: f64) -> StatsResult<f64> {
    if !(0.0..0.5).contains(&alpha) {
        return Err(StatsError::InvalidParameter {
            name: "alpha".into(),
            reason: format!("must be in [0, 0.5), got {alpha}"),
        });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let mut sorted: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let k = (n as f64 * alpha).floor() as usize;
    if 2 * k >= n {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: 2 * k + 1,
        });
    }
    let lo = sorted[k];
    let hi = sorted[n - k - 1];
    for v in sorted.iter_mut().take(k) {
        *v = lo;
    }
    for v in sorted.iter_mut().skip(n - k) {
        *v = hi;
    }
    mean(&sorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mad_known_value() {
        // For sample [1,2,3,4,5], median = 3, deviations = [2,1,0,1,2], median dev = 1
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let v = mad(&x).expect("ok");
        assert!((v - 1.482_602_218_505_602).abs() < 1e-12);
    }

    #[test]
    fn iqr_simple() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let v = iqr(&x).expect("ok");
        assert!((v - 2.0).abs() < 1e-12);
    }

    #[test]
    fn trimmed_mean_drops_tails() {
        // [1,2,3,4,5,6,7,8,9,10] alpha=0.1 -> drop 1 each side -> [2..9] mean = 5.5
        let x: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let v = trimmed_mean(&x, 0.1).expect("ok");
        assert!((v - 5.5).abs() < 1e-12);
    }

    #[test]
    fn winsorized_replaces_tails() {
        let x: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        // alpha=0.1 -> k=1 -> replace [1] with 2, [10] with 9 -> mean(2,2,3,4,5,6,7,8,9,9) = 5.5
        let v = winsorized_mean(&x, 0.1).expect("ok");
        assert!((v - 5.5).abs() < 1e-12);
    }
}

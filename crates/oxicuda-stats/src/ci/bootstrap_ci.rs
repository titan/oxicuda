//! Bootstrap confidence intervals: percentile and BCa.

use crate::ci::CiResult;
use crate::descriptive::quantile::quantile;
use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;
use crate::resampling::jackknife::jackknife;

/// Percentile CI from bootstrap replicates.
pub fn percentile_ci(replicates: &[f64], confidence: f64) -> StatsResult<CiResult> {
    if replicates.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(StatsError::ProbabilityOutOfRange { value: confidence });
    }
    let alpha = 1.0 - confidence;
    let lo = quantile(replicates, alpha / 2.0)?;
    let hi = quantile(replicates, 1.0 - alpha / 2.0)?;
    Ok(CiResult {
        lower: lo,
        upper: hi,
    })
}

/// Bias-corrected and accelerated (BCa) CI from bootstrap replicates.
///
/// Requires the original data for the acceleration jackknife step.
pub fn bca_ci(
    data: &[f64],
    replicates: &[f64],
    statistic: impl Fn(&[f64]) -> StatsResult<f64>,
    confidence: f64,
    _rng: &mut LcgRng,
) -> StatsResult<CiResult> {
    if replicates.is_empty() || data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(StatsError::ProbabilityOutOfRange { value: confidence });
    }
    let theta_hat = statistic(data)?;
    let b = replicates.len() as f64;
    // Bias correction z0
    let prop_below: f64 = replicates.iter().filter(|&&r| r < theta_hat).count() as f64 / b;
    let n01 = Normal::standard();
    let z0 = n01.ppf(prop_below.clamp(1e-12, 1.0 - 1e-12))?;
    // Acceleration via jackknife
    let jk = jackknife(data, &statistic)?;
    let mean_pseudo: f64 = jk.pseudovalues.iter().sum::<f64>() / jk.pseudovalues.len() as f64;
    let num: f64 = jk
        .pseudovalues
        .iter()
        .map(|v| (mean_pseudo - v).powi(3))
        .sum();
    let den: f64 = jk
        .pseudovalues
        .iter()
        .map(|v| (mean_pseudo - v).powi(2))
        .sum::<f64>()
        .powf(1.5);
    let a = if den < 1e-300 { 0.0 } else { num / (6.0 * den) };
    let alpha = 1.0 - confidence;
    let z_lo = n01.ppf(alpha / 2.0)?;
    let z_hi = n01.ppf(1.0 - alpha / 2.0)?;
    let g = |z: f64| -> f64 {
        let inner = (z0 + z) / (1.0 - a * (z0 + z));
        n01.cdf(z0 + inner)
    };
    let alpha_lo = g(z_lo).clamp(1e-12, 1.0 - 1e-12);
    let alpha_hi = g(z_hi).clamp(1e-12, 1.0 - 1e-12);
    let lo = quantile(replicates, alpha_lo)?;
    let hi = quantile(replicates, alpha_hi)?;
    Ok(CiResult {
        lower: lo,
        upper: hi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_ci_basic() {
        let r: Vec<f64> = (0..101).map(|v| v as f64).collect();
        let ci = percentile_ci(&r, 0.9).expect("ok");
        assert!((ci.lower - 5.0).abs() < 0.5);
        assert!((ci.upper - 95.0).abs() < 0.5);
    }
}

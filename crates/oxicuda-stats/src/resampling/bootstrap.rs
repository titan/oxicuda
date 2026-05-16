//! Nonparametric bootstrap with a user-supplied statistic.

use crate::descriptive::quantile::quantile;
use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

/// Result of a bootstrap procedure.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    pub theta_hat: f64,
    pub replicates: Vec<f64>,
    pub std_error: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

/// Run `n_boot` bootstrap replicates of a statistic.
///
/// `statistic(sample)` is called with a freshly resampled vector each iteration.
pub fn bootstrap(
    data: &[f64],
    n_boot: usize,
    confidence: f64,
    statistic: impl Fn(&[f64]) -> StatsResult<f64>,
    rng: &mut LcgRng,
) -> StatsResult<BootstrapResult> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if n_boot == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_boot".into(),
            reason: "must be > 0".into(),
        });
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(StatsError::ProbabilityOutOfRange { value: confidence });
    }
    let theta_hat = statistic(data)?;
    let mut sample = vec![0.0; data.len()];
    let mut replicates = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        for v in sample.iter_mut() {
            let idx = rng.next_usize(data.len());
            *v = data[idx];
        }
        replicates.push(statistic(&sample)?);
    }
    let mean: f64 = replicates.iter().sum::<f64>() / n_boot as f64;
    let var: f64 =
        replicates.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_boot as f64 - 1.0).max(1.0);
    let se = var.sqrt();
    let alpha = 1.0 - confidence;
    let lo = quantile(&replicates, alpha / 2.0)?;
    let hi = quantile(&replicates, 1.0 - alpha / 2.0)?;
    Ok(BootstrapResult {
        theta_hat,
        replicates,
        std_error: se,
        ci_lower: lo,
        ci_upper: hi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptive::summary::mean;

    #[test]
    fn bootstrap_mean_ci() {
        let mut rng = LcgRng::new(7);
        let data: Vec<f64> = (1..=30).map(|v| v as f64).collect();
        let r = bootstrap(&data, 500, 0.95, mean, &mut rng).expect("ok");
        // True mean is 15.5; CI should contain it
        assert!(r.ci_lower <= 15.5 && r.ci_upper >= 15.5);
    }
}

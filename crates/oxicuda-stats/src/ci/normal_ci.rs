//! Normal-distribution confidence interval for the mean.

use crate::ci::CiResult;
use crate::descriptive::summary::mean;
use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};

/// Normal CI: `x̄ ± z * sigma / sqrt(n)` with known population sigma.
pub fn normal_ci(x: &[f64], sigma: f64, confidence: f64) -> StatsResult<CiResult> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if sigma <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "sigma".into(),
            reason: "must be > 0".into(),
        });
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(StatsError::ProbabilityOutOfRange { value: confidence });
    }
    let m = mean(x)?;
    let n = x.len() as f64;
    let alpha = 1.0 - confidence;
    let dist = Normal::standard();
    let z = dist.ppf(1.0 - alpha / 2.0)?;
    let half = z * sigma / n.sqrt();
    Ok(CiResult {
        lower: m - half,
        upper: m + half,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_ci_basic() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ci = normal_ci(&x, 1.0, 0.95).expect("ok");
        assert!(ci.lower < 3.0);
        assert!(ci.upper > 3.0);
    }
}

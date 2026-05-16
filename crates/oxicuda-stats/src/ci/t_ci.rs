//! Student t confidence interval for the mean with unknown sigma.

use crate::ci::CiResult;
use crate::descriptive::summary::{mean, sample_std};
use crate::distributions::student_t::StudentT;
use crate::error::{StatsError, StatsResult};

/// t CI: `x̄ ± t(alpha/2, df) * s / sqrt(n)`.
pub fn t_ci(x: &[f64], confidence: f64) -> StatsResult<CiResult> {
    if x.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: x.len(),
            need: 2,
        });
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(StatsError::ProbabilityOutOfRange { value: confidence });
    }
    let m = mean(x)?;
    let s = sample_std(x)?;
    let n = x.len() as f64;
    let alpha = 1.0 - confidence;
    let dist = StudentT::new(n - 1.0)?;
    let t = dist.ppf(1.0 - alpha / 2.0)?;
    let half = t * s / n.sqrt();
    Ok(CiResult {
        lower: m - half,
        upper: m + half,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_ci_contains_mean() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ci = t_ci(&x, 0.95).expect("ok");
        assert!(ci.lower < 3.0 && ci.upper > 3.0);
    }
}

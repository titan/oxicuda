//! Confidence intervals for a binomial proportion.

use crate::ci::CiResult;
use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};
use crate::special::betainc::betainc;

/// Wilson score interval for a single proportion.
pub fn wilson_ci(successes: usize, n: usize, confidence: f64) -> StatsResult<CiResult> {
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if successes > n {
        return Err(StatsError::InvalidParameter {
            name: "successes".into(),
            reason: format!("successes ({successes}) > n ({n})"),
        });
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(StatsError::ProbabilityOutOfRange { value: confidence });
    }
    let p_hat = successes as f64 / n as f64;
    let alpha = 1.0 - confidence;
    let z = Normal::standard().ppf(1.0 - alpha / 2.0)?;
    let z2 = z * z;
    let nf = n as f64;
    let denom = 1.0 + z2 / nf;
    let centre = (p_hat + z2 / (2.0 * nf)) / denom;
    let half = z * ((p_hat * (1.0 - p_hat) / nf + z2 / (4.0 * nf * nf)).sqrt()) / denom;
    Ok(CiResult {
        lower: (centre - half).max(0.0),
        upper: (centre + half).min(1.0),
    })
}

/// Clopper-Pearson exact confidence interval.
pub fn clopper_pearson_ci(successes: usize, n: usize, confidence: f64) -> StatsResult<CiResult> {
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if successes > n {
        return Err(StatsError::InvalidParameter {
            name: "successes".into(),
            reason: format!("successes ({successes}) > n ({n})"),
        });
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(StatsError::ProbabilityOutOfRange { value: confidence });
    }
    let alpha = 1.0 - confidence;
    let nf = n as f64;
    let k = successes as f64;
    // Lower = qbeta(alpha/2, k, n-k+1)
    let lo = if successes == 0 {
        0.0
    } else {
        beta_quantile(alpha / 2.0, k, nf - k + 1.0)?
    };
    // Upper = qbeta(1 - alpha/2, k+1, n-k)
    let hi = if successes == n {
        1.0
    } else {
        beta_quantile(1.0 - alpha / 2.0, k + 1.0, nf - k)?
    };
    Ok(CiResult {
        lower: lo,
        upper: hi,
    })
}

fn beta_quantile(p: f64, a: f64, b: f64) -> StatsResult<f64> {
    // Bisection on cdf for monotone betainc
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let cdf = betainc(a, b, mid)?;
        if cdf < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Agresti-Coull interval ("adjusted Wald").
pub fn agresti_coull_ci(successes: usize, n: usize, confidence: f64) -> StatsResult<CiResult> {
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(StatsError::ProbabilityOutOfRange { value: confidence });
    }
    let alpha = 1.0 - confidence;
    let z = Normal::standard().ppf(1.0 - alpha / 2.0)?;
    let z2 = z * z;
    let n_tilde = n as f64 + z2;
    let p_tilde = (successes as f64 + z2 / 2.0) / n_tilde;
    let half = z * (p_tilde * (1.0 - p_tilde) / n_tilde).sqrt();
    Ok(CiResult {
        lower: (p_tilde - half).max(0.0),
        upper: (p_tilde + half).min(1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_contains_hat() {
        let ci = wilson_ci(30, 100, 0.95).expect("ok");
        assert!(ci.lower < 0.3 && ci.upper > 0.3);
    }

    #[test]
    fn clopper_pearson_runs() {
        let ci = clopper_pearson_ci(5, 50, 0.95).expect("ok");
        assert!(ci.lower < 0.1 && ci.upper > 0.1);
    }

    #[test]
    fn agresti_coull_contains_hat() {
        let ci = agresti_coull_ci(40, 100, 0.95).expect("ok");
        assert!(ci.lower < 0.4 && ci.upper > 0.4);
    }
}

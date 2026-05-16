//! Wilcoxon signed-rank test.

use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};
use crate::nonparametric::{rank_with_ties, tie_correction_sum};

/// Result of a Wilcoxon signed-rank test.
#[derive(Debug, Clone, Copy)]
pub struct WilcoxonResult {
    pub w_statistic: f64,
    pub z: f64,
    pub p_value_two_sided: f64,
}

/// Wilcoxon signed-rank test on paired samples.
pub fn wilcoxon_signed_rank(x1: &[f64], x2: &[f64]) -> StatsResult<WilcoxonResult> {
    if x1.len() != x2.len() {
        return Err(StatsError::DimensionMismatch {
            a: x1.len(),
            b: x2.len(),
        });
    }
    if x1.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    // Differences excluding zeros
    let diffs: Vec<f64> = x1
        .iter()
        .zip(x2)
        .map(|(a, b)| a - b)
        .filter(|d| d.abs() > 1e-15)
        .collect();
    if diffs.is_empty() {
        return Err(StatsError::NumericalInstability(
            "all paired diffs zero".into(),
        ));
    }
    let signs: Vec<f64> = diffs.iter().map(|d| d.signum()).collect();
    let abs_d: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let ranks = rank_with_ties(&abs_d);
    let mut w_pos = 0.0;
    let mut w_neg = 0.0;
    for (i, &r) in ranks.iter().enumerate() {
        if signs[i] > 0.0 {
            w_pos += r;
        } else {
            w_neg += r;
        }
    }
    let w = w_pos.min(w_neg);
    let n = diffs.len() as f64;
    let mu = n * (n + 1.0) / 4.0;
    let t_corr = tie_correction_sum(&ranks);
    let sigma2 = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0 - t_corr / 48.0;
    if sigma2 <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "Wilcoxon: degenerate ranks".into(),
        ));
    }
    let sigma = sigma2.sqrt();
    let z = (w - mu + 0.5 * (w - mu).signum()) / sigma;
    let dist = Normal::standard();
    let p_two = 2.0 * (1.0 - dist.cdf(z.abs()));
    Ok(WilcoxonResult {
        w_statistic: w,
        z,
        p_value_two_sided: p_two.clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilcoxon_systematic_shift() {
        let x1: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let x2: Vec<f64> = (1..=10).map(|v| (v + 2) as f64).collect();
        let r = wilcoxon_signed_rank(&x1, &x2).expect("ok");
        assert!(r.p_value_two_sided < 0.05);
    }

    #[test]
    fn wilcoxon_zero_diff_errors() {
        let x = [1.0, 2.0, 3.0];
        let r = wilcoxon_signed_rank(&x, &x);
        assert!(r.is_err());
    }
}

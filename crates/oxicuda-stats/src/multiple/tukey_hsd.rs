//! Tukey's HSD (honestly significant difference) post-hoc test using the studentized range.

use crate::descriptive::summary::mean;
use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};

/// Pairwise Tukey HSD comparison.
#[derive(Debug, Clone, Copy)]
pub struct TukeyComparison {
    pub i: usize,
    pub j: usize,
    pub mean_diff: f64,
    pub q_statistic: f64,
    pub p_value_approx: f64,
}

/// Aggregated Tukey HSD results.
#[derive(Debug, Clone)]
pub struct TukeyResult {
    pub comparisons: Vec<TukeyComparison>,
    pub mse: f64,
    pub df_error: f64,
}

/// Run Tukey HSD on `k` balanced or unbalanced groups using a normal approximation for q.
///
/// The asymptotic approximation `q ~ sqrt(2) * z` is used; for production use, a studentized
/// range table is preferred, but this gives a usable approximation.
pub fn tukey_hsd(groups: &[&[f64]]) -> StatsResult<TukeyResult> {
    let k = groups.len();
    if k < 2 {
        return Err(StatsError::InsufficientSampleSize { got: k, need: 2 });
    }
    // Compute MSE (within-group variance) like ANOVA
    let mut means = Vec::with_capacity(k);
    let mut ns = Vec::with_capacity(k);
    let mut ss_within = 0.0;
    let mut n_total = 0usize;
    for g in groups {
        if g.len() < 2 {
            return Err(StatsError::InsufficientSampleSize {
                got: g.len(),
                need: 2,
            });
        }
        let m = mean(g)?;
        means.push(m);
        ns.push(g.len());
        n_total += g.len();
        for &v in *g {
            ss_within += (v - m).powi(2);
        }
    }
    let df_error = (n_total - k) as f64;
    let mse = ss_within / df_error;
    if mse <= 0.0 {
        return Err(StatsError::NumericalInstability("MSE <= 0".into()));
    }
    let dist = Normal::standard();
    let mut comparisons = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in (i + 1)..k {
            let n_harm = 2.0 / (1.0 / ns[i] as f64 + 1.0 / ns[j] as f64);
            let se = (mse / n_harm).sqrt();
            let diff = means[i] - means[j];
            let q = diff / se;
            // Approximate two-sided p-value via |q|/sqrt(2) ~ z
            let z = q.abs() / std::f64::consts::SQRT_2;
            let p = 2.0 * (1.0 - dist.cdf(z));
            comparisons.push(TukeyComparison {
                i,
                j,
                mean_diff: diff,
                q_statistic: q,
                p_value_approx: p.clamp(0.0, 1.0),
            });
        }
    }
    Ok(TukeyResult {
        comparisons,
        mse,
        df_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tukey_hsd_three_groups() {
        let g1: &[f64] = &[1.0, 2.0, 3.0];
        let g2: &[f64] = &[3.0, 4.0, 5.0];
        let g3: &[f64] = &[5.0, 6.0, 7.0];
        let r = tukey_hsd(&[g1, g2, g3]).expect("ok");
        assert_eq!(r.comparisons.len(), 3);
        for cmp in &r.comparisons {
            assert!(cmp.q_statistic.is_finite());
        }
    }
}

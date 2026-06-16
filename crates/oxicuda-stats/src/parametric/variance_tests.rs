//! Tests for homogeneity of variance across `k ≥ 2` groups.
//!
//! Three classic tests are provided:
//!
//! - **Levene's test** transforms each observation to its absolute deviation
//!   from the group **mean**, `z_ij = |x_ij − x̄_i|`, then runs a one-way
//!   ANOVA on the `z_ij`. The resulting `W` statistic is `F`-distributed with
//!   `(k−1, N−k)` degrees of freedom under the null of equal variances. Levene's
//!   test is moderately robust to non-normality.
//!
//! - **Brown–Forsythe** is the more robust variant that centres on the group
//!   **median** instead of the mean; it is the recommended default when the
//!   data may be skewed or heavy-tailed.
//!
//! - **Bartlett's test** is the likelihood-ratio test under normality. It is
//!   the most powerful when the groups really are normal, but it is highly
//!   sensitive to departures from normality. The statistic is
//!   chi-squared-distributed with `k−1` degrees of freedom.
//!
//! ## References
//! - Levene, H. (1960). "Robust tests for equality of variances." In
//!   *Contributions to Probability and Statistics*.
//! - Brown, M. B. & Forsythe, A. B. (1974). "Robust tests for the equality of
//!   variances." JASA 69(346).
//! - Bartlett, M. S. (1937). "Properties of sufficiency and statistical tests."
//!   Proc. R. Soc. Lond. A 160.

use crate::distributions::chi_squared::ChiSquared;
use crate::distributions::f_dist::FDist;
use crate::error::{StatsError, StatsResult};

/// How Levene-type tests centre each group before measuring spread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeveneCenter {
    /// Centre on the group **mean** (original Levene test).
    Mean,
    /// Centre on the group **median** (Brown–Forsythe; more robust).
    Median,
}

/// Result of a Levene / Brown–Forsythe test.
#[derive(Debug, Clone, Copy)]
pub struct LeveneResult {
    /// The `W` test statistic.
    pub statistic: f64,
    /// Numerator degrees of freedom (`k − 1`).
    pub df_between: f64,
    /// Denominator degrees of freedom (`N − k`).
    pub df_within: f64,
    /// Upper-tail p-value from the `F` distribution.
    pub p_value: f64,
}

/// Result of Bartlett's test.
#[derive(Debug, Clone, Copy)]
pub struct BartlettResult {
    /// The (corrected) chi-squared test statistic.
    pub statistic: f64,
    /// Degrees of freedom (`k − 1`).
    pub df: f64,
    /// Upper-tail p-value from the chi-squared distribution.
    pub p_value: f64,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn group_mean(g: &[f64]) -> f64 {
    g.iter().sum::<f64>() / g.len() as f64
}

fn group_median(g: &[f64]) -> f64 {
    let mut sorted = g.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Bessel-corrected sample variance of a group.
fn group_variance(g: &[f64]) -> f64 {
    let n = g.len() as f64;
    let m = group_mean(g);
    let ss: f64 = g.iter().map(|&x| (x - m).powi(2)).sum();
    ss / (n - 1.0)
}

fn validate_groups(groups: &[&[f64]], min_per_group: usize) -> StatsResult<()> {
    if groups.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: groups.len(),
            need: 2,
        });
    }
    for g in groups {
        if g.len() < min_per_group {
            return Err(StatsError::InsufficientSampleSize {
                got: g.len(),
                need: min_per_group,
            });
        }
        for (i, &v) in g.iter().enumerate() {
            if !v.is_finite() {
                return Err(StatsError::NonFiniteValue(i));
            }
        }
    }
    Ok(())
}

// ─── Levene / Brown–Forsythe ──────────────────────────────────────────────────

/// Levene's test (or Brown–Forsythe when `center == Median`) for homogeneity of
/// variance across `k ≥ 2` groups.
///
/// Each group must contain at least two observations.
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] with fewer than two groups or any
///   group of size `< 2`.
/// - [`StatsError::NonFiniteValue`] on non-finite data.
/// - [`StatsError::NumericalInstability`] if the within-group spread of the
///   transformed values is exactly zero.
pub fn levene_test(groups: &[&[f64]], center: LeveneCenter) -> StatsResult<LeveneResult> {
    validate_groups(groups, 2)?;
    let k = groups.len();

    // Transform to absolute deviations from the chosen centre.
    let z: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            let c = match center {
                LeveneCenter::Mean => group_mean(g),
                LeveneCenter::Median => group_median(g),
            };
            g.iter().map(|&x| (x - c).abs()).collect()
        })
        .collect();

    let n_total: usize = z.iter().map(|zi| zi.len()).sum();
    let grand_mean = z.iter().flatten().sum::<f64>() / n_total as f64;

    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for zi in &z {
        let n_i = zi.len() as f64;
        let mean_i = zi.iter().sum::<f64>() / n_i;
        ss_between += n_i * (mean_i - grand_mean).powi(2);
        for &v in zi {
            ss_within += (v - mean_i).powi(2);
        }
    }

    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;
    if ss_within <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "levene: zero within-group spread of absolute deviations".into(),
        ));
    }
    let w = (ss_between / df_between) / (ss_within / df_within);
    let p = 1.0 - FDist::new(df_between, df_within)?.cdf(w)?;
    Ok(LeveneResult {
        statistic: w,
        df_between,
        df_within,
        p_value: p.clamp(0.0, 1.0),
    })
}

// ─── Bartlett ─────────────────────────────────────────────────────────────────

/// Bartlett's test for homogeneity of variance across `k ≥ 2` normal groups.
///
/// Each group must contain at least two observations.
///
/// The statistic is
/// `χ² = ((N−k) ln(s_p²) − Σ (n_i−1) ln(s_i²)) / C`,
/// where `s_p²` is the pooled variance and `C` is the Bartlett bias correction.
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] with fewer than two groups or any
///   group of size `< 2`.
/// - [`StatsError::NonFiniteValue`] on non-finite data.
/// - [`StatsError::NumericalInstability`] if any group variance is zero (the
///   log is undefined).
pub fn bartlett_test(groups: &[&[f64]]) -> StatsResult<BartlettResult> {
    validate_groups(groups, 2)?;
    let k = groups.len();

    let mut variances = Vec::with_capacity(k);
    let mut dfs = Vec::with_capacity(k);
    for g in groups {
        let v = group_variance(g);
        if v <= 0.0 {
            return Err(StatsError::NumericalInstability(
                "bartlett: a group has zero variance".into(),
            ));
        }
        variances.push(v);
        dfs.push((g.len() - 1) as f64);
    }
    let df_total: f64 = dfs.iter().sum();

    // Pooled variance s_p² = Σ (n_i−1) s_i² / (N−k).
    let pooled: f64 = dfs
        .iter()
        .zip(variances.iter())
        .map(|(&d, &v)| d * v)
        .sum::<f64>()
        / df_total;

    let sum_df_ln: f64 = dfs
        .iter()
        .zip(variances.iter())
        .map(|(&d, &v)| d * v.ln())
        .sum();
    let numerator = df_total * pooled.ln() - sum_df_ln;

    // Bartlett correction C = 1 + (1/(3(k−1))) (Σ 1/(n_i−1) − 1/(N−k)).
    let sum_inv: f64 = dfs.iter().map(|&d| 1.0 / d).sum();
    let correction = 1.0 + (sum_inv - 1.0 / df_total) / (3.0 * (k - 1) as f64);

    let statistic = (numerator / correction).max(0.0);
    let df = (k - 1) as f64;
    let p = 1.0 - ChiSquared::new(df)?.cdf(statistic)?;
    Ok(BartlettResult {
        statistic,
        df,
        p_value: p.clamp(0.0, 1.0),
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn shifted(base: &[f64], scale: f64) -> Vec<f64> {
        base.iter().map(|&x| x * scale).collect()
    }

    #[test]
    fn levene_equal_variance_high_p() {
        // Three groups with identical spread but different means → fail to reject.
        let g1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2: Vec<f64> = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        let g3: Vec<f64> = vec![21.0, 22.0, 23.0, 24.0, 25.0];
        let r = levene_test(&[&g1, &g2, &g3], LeveneCenter::Mean).expect("ok");
        assert!(r.p_value > 0.5, "p={}", r.p_value);
    }

    #[test]
    fn levene_unequal_variance_low_p() {
        // One tight group, one very dispersed group → reject equal variances.
        let tight: Vec<f64> = (0..20).map(|i| 10.0 + (i % 2) as f64 * 0.1).collect();
        let wide: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64 - 10.0) * 3.0).collect();
        let r = levene_test(&[&tight, &wide], LeveneCenter::Mean).expect("ok");
        assert!(r.p_value < 0.05, "p={}", r.p_value);
        assert!(r.statistic > 0.0);
    }

    #[test]
    fn levene_df_correct() {
        let g1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let g2: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0];
        let g3: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0];
        let r = levene_test(&[&g1, &g2, &g3], LeveneCenter::Mean).expect("ok");
        assert_eq!(r.df_between, 2.0);
        assert_eq!(r.df_within, 9.0); // N=12, k=3
    }

    #[test]
    fn brown_forsythe_median_centering() {
        // With an outlier the median-centred test is less inflated than the mean.
        let g1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        let g2: Vec<f64> = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mean_r = levene_test(&[&g1, &g2], LeveneCenter::Mean).expect("ok");
        let med_r = levene_test(&[&g1, &g2], LeveneCenter::Median).expect("ok");
        assert!(mean_r.statistic.is_finite() && med_r.statistic.is_finite());
        // Both produce valid p-values.
        assert!((0.0..=1.0).contains(&mean_r.p_value));
        assert!((0.0..=1.0).contains(&med_r.p_value));
    }

    #[test]
    fn levene_statistic_nonneg() {
        let g1: Vec<f64> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        let g2: Vec<f64> = vec![2.0, 7.0, 1.0, 8.0, 2.0, 8.0];
        let r = levene_test(&[&g1, &g2], LeveneCenter::Median).expect("ok");
        assert!(r.statistic >= 0.0 && r.statistic.is_finite());
    }

    #[test]
    fn levene_too_few_groups_error() {
        let g1: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            levene_test(&[&g1], LeveneCenter::Mean).unwrap_err(),
            StatsError::InsufficientSampleSize { .. }
        ));
    }

    #[test]
    fn levene_singleton_group_error() {
        let g1: Vec<f64> = vec![1.0];
        let g2: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            levene_test(&[&g1, &g2], LeveneCenter::Mean).unwrap_err(),
            StatsError::InsufficientSampleSize { .. }
        ));
    }

    #[test]
    fn levene_non_finite_error() {
        let g1: Vec<f64> = vec![1.0, f64::NAN, 3.0];
        let g2: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            levene_test(&[&g1, &g2], LeveneCenter::Mean).unwrap_err(),
            StatsError::NonFiniteValue(_)
        ));
    }

    #[test]
    fn bartlett_equal_variance_high_p() {
        let g1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let g2 = shifted(&g1, 1.0); // same spread, shifted by +0 here
        let g2: Vec<f64> = g2.iter().map(|&x| x + 10.0).collect();
        let g3: Vec<f64> = g1.iter().map(|&x| x + 20.0).collect();
        let r = bartlett_test(&[&g1, &g2, &g3]).expect("ok");
        assert!(r.p_value > 0.5, "p={}", r.p_value);
    }

    #[test]
    fn bartlett_unequal_variance_low_p() {
        let tight: Vec<f64> = (0..30).map(|i| 5.0 + (i % 2) as f64 * 0.05).collect();
        let wide: Vec<f64> = (0..30).map(|i| 5.0 + (i as f64 - 15.0) * 2.0).collect();
        let r = bartlett_test(&[&tight, &wide]).expect("ok");
        assert!(r.p_value < 0.01, "p={}", r.p_value);
    }

    #[test]
    fn bartlett_df_and_correction() {
        let g1: Vec<f64> = vec![1.0, 2.0, 3.0, 5.0];
        let g2: Vec<f64> = vec![2.0, 4.0, 6.0, 9.0];
        let g3: Vec<f64> = vec![1.0, 3.0, 6.0, 10.0];
        let r = bartlett_test(&[&g1, &g2, &g3]).expect("ok");
        assert_eq!(r.df, 2.0);
        assert!(r.statistic >= 0.0 && r.statistic.is_finite());
    }

    #[test]
    fn bartlett_zero_variance_error() {
        let g1: Vec<f64> = vec![3.0, 3.0, 3.0, 3.0]; // zero variance
        let g2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        assert!(matches!(
            bartlett_test(&[&g1, &g2]).unwrap_err(),
            StatsError::NumericalInstability(_)
        ));
    }

    #[test]
    fn bartlett_too_few_groups_error() {
        let g1: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            bartlett_test(&[&g1]).unwrap_err(),
            StatsError::InsufficientSampleSize { .. }
        ));
    }

    #[test]
    fn bartlett_non_finite_error() {
        let g1: Vec<f64> = vec![1.0, 2.0, f64::INFINITY];
        let g2: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            bartlett_test(&[&g1, &g2]).unwrap_err(),
            StatsError::NonFiniteValue(_)
        ));
    }

    #[test]
    fn levene_p_value_in_range() {
        let g1: Vec<f64> = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let g2: Vec<f64> = vec![4.0, 4.5, 5.0, 5.5, 6.0];
        let r = levene_test(&[&g1, &g2], LeveneCenter::Median).expect("ok");
        assert!((0.0..=1.0).contains(&r.p_value));
    }

    #[test]
    fn bartlett_deterministic() {
        let g1: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0];
        let g2: Vec<f64> = vec![2.0, 3.0, 5.0, 7.0];
        let a = bartlett_test(&[&g1, &g2]).expect("ok");
        let b = bartlett_test(&[&g1, &g2]).expect("ok");
        assert_eq!(a.statistic, b.statistic);
        assert_eq!(a.p_value, b.p_value);
    }
}

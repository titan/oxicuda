//! Classical outlier detection tests: Grubbs's test and Dixon's Q-test.

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Internal statistical approximation utilities
// ---------------------------------------------------------------------------

/// Rational approximation for the standard normal inverse CDF.
///
/// Uses the Abramowitz & Stegun algorithm. Returns z such that Φ(z) = p.
fn normal_inv_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    // We compute for the upper tail (p > 0.5) and negate for lower
    let (q, negate) = if p >= 0.5 {
        (p, false)
    } else {
        (1.0 - p, true)
    };
    let t = (-2.0 * (1.0 - q).ln()).sqrt();
    let c0 = 2.515_517_f64;
    let c1 = 0.802_853_f64;
    let c2 = 0.010_328_f64;
    let d1 = 1.432_788_f64;
    let d2 = 0.189_269_f64;
    let d3 = 0.001_308_f64;
    let num = c0 + c1 * t + c2 * t * t;
    let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    let z = t - num / den;
    if negate { -z } else { z }
}

/// Approximate two-tailed t critical value using Cornish-Fisher expansion.
///
/// Returns t* such that P(|T| > t*) = alpha for T ~ t(df).
fn t_inv_cdf_two_tailed(alpha: f64, df: f64) -> f64 {
    let z = normal_inv_cdf(1.0 - alpha / 2.0);
    let g1 = (z.powi(3) + z) / (4.0 * df);
    let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * df.powi(2));
    z + g1 + g2
}

// ---------------------------------------------------------------------------
// Grubbs's Test
// ---------------------------------------------------------------------------

/// Result of Grubbs's outlier test.
#[derive(Debug, Clone)]
#[must_use]
pub struct GrubbsResult {
    /// Test statistic G = max_i |x_i - x̄| / s.
    pub statistic: f64,
    /// Critical value G_crit at the requested significance level.
    pub critical_value: f64,
    /// Significance level alpha.
    pub alpha: f64,
    /// Whether the null hypothesis (no outlier) is rejected.
    pub is_outlier: bool,
    /// Index of the candidate outlier in the original data slice.
    pub outlier_index: usize,
    /// Value of the candidate outlier.
    pub outlier_value: f64,
}

/// Grubbs's test for a single outlier in approximately-normal data (Grubbs 1969).
///
/// Detects the most extreme observation and tests whether it is a statistically
/// significant outlier at significance level `alpha` using a Bonferroni-corrected
/// t-distribution critical value.
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `n < 6`
/// - [`StatsError::InvalidParameter`] if `alpha` is not in `(0.0, 0.5)`
/// - [`StatsError::NonFiniteValue`] if any datum is non-finite
pub fn grubbs_test(data: &[f64], alpha: f64) -> StatsResult<GrubbsResult> {
    let n = data.len();

    if n < 6 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 6 });
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 0.5 {
        return Err(StatsError::InvalidParameter {
            name: "alpha".into(),
            reason: "must be in the open interval (0, 0.5)".into(),
        });
    }
    for (i, &v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    // Compute mean
    let mean = data.iter().sum::<f64>() / n as f64;

    // Compute sample standard deviation (Bessel-corrected)
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_dev = variance.sqrt();

    if std_dev < f64::EPSILON {
        // All identical: G = 0, no outlier possible
        return Ok(GrubbsResult {
            statistic: 0.0,
            critical_value: 0.0,
            alpha,
            is_outlier: false,
            outlier_index: 0,
            outlier_value: data[0],
        });
    }

    // Find the most extreme point
    let (outlier_index, outlier_value, max_deviation) = data
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x, (x - mean).abs()))
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        .expect("data is non-empty (checked above)");

    let g_stat = max_deviation / std_dev;

    // Bonferroni-corrected critical value via t-distribution approximation
    // p = alpha / (2*n) for the one-sided tail (two-sided Bonferroni)
    let p_bonf = alpha / (2.0 * n as f64);
    let t_crit = t_inv_cdf_two_tailed(p_bonf * 2.0, (n - 2) as f64);
    let t_sq = t_crit * t_crit;
    let n_f = n as f64;
    let g_crit = ((n_f - 1.0) / n_f.sqrt()) * (t_sq / (n_f - 2.0 + t_sq)).sqrt();

    Ok(GrubbsResult {
        statistic: g_stat,
        critical_value: g_crit,
        alpha,
        is_outlier: g_stat > g_crit,
        outlier_index,
        outlier_value,
    })
}

// ---------------------------------------------------------------------------
// Dixon's Q-Test
// ---------------------------------------------------------------------------

/// Critical Q values for α = 0.05 (n, Q_crit).
const Q_CRIT_005: &[(usize, f64)] = &[
    (3, 0.970),
    (4, 0.829),
    (5, 0.710),
    (6, 0.625),
    (7, 0.568),
    (8, 0.526),
    (9, 0.493),
    (10, 0.466),
    (11, 0.444),
    (12, 0.426),
    (13, 0.410),
    (14, 0.396),
    (15, 0.384),
    (16, 0.374),
    (17, 0.365),
    (18, 0.356),
    (19, 0.349),
    (20, 0.342),
    (25, 0.317),
    (30, 0.290),
];

/// Critical Q values for α = 0.01 (n, Q_crit).
const Q_CRIT_001: &[(usize, f64)] = &[
    (3, 0.994),
    (4, 0.926),
    (5, 0.821),
    (6, 0.740),
    (7, 0.680),
    (8, 0.634),
    (9, 0.598),
    (10, 0.568),
    (11, 0.542),
    (12, 0.522),
    (13, 0.503),
    (14, 0.488),
    (15, 0.475),
    (16, 0.463),
    (17, 0.452),
    (18, 0.442),
    (19, 0.433),
    (20, 0.425),
    (25, 0.393),
    (30, 0.372),
];

/// Interpolate (or extrapolate to nearest boundary) a Q critical value from a table.
fn interpolate_q_crit(table: &[(usize, f64)], n: usize) -> f64 {
    // Exact match
    if let Some(&(_, q)) = table.iter().find(|&&(k, _)| k == n) {
        return q;
    }
    // Find surrounding bracket
    let lower = table.iter().rev().find(|&&(k, _)| k < n);
    let upper = table.iter().find(|&&(k, _)| k > n);
    match (lower, upper) {
        (Some(&(n0, q0)), Some(&(n1, q1))) => {
            // Linear interpolation in n
            let frac = (n - n0) as f64 / (n1 - n0) as f64;
            q0 + frac * (q1 - q0)
        }
        // n is beyond upper boundary (n > max table n); return last entry
        (Some(&(_, q)), None) => q,
        // n is below lower boundary; return first entry
        (None, Some(&(_, q))) => q,
        (None, None) => 0.0,
    }
}

/// Result of Dixon's Q outlier test.
#[derive(Debug, Clone)]
#[must_use]
pub struct DixonQResult {
    /// Q statistic = gap / range.
    pub statistic: f64,
    /// Critical Q value at the requested significance level.
    pub critical_value: f64,
    /// Significance level alpha.
    pub alpha: f64,
    /// Whether the null hypothesis (no outlier) is rejected.
    pub is_outlier: bool,
    /// Index of the candidate outlier in the original data slice.
    pub outlier_index: usize,
    /// Value of the candidate outlier.
    pub outlier_value: f64,
}

/// Dixon's Q-test for small samples (3 ≤ n ≤ 30).
///
/// Tests whether the most extreme value (minimum or maximum) is a statistically
/// significant outlier. The Q statistic is the ratio of the gap between the
/// suspected outlier and its nearest neighbor to the overall range.
///
/// Only `alpha = 0.05` and `alpha = 0.01` are supported (standard lookup tables).
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `n < 3`
/// - [`StatsError::InvalidParameter`] if `alpha` is not `0.05` or `0.01`
/// - [`StatsError::InvalidParameter`] if `n > 30`
/// - [`StatsError::NonFiniteValue`] if any datum is non-finite
/// - [`StatsError::NumericalInstability`] if range ≈ 0 (all values nearly identical)
pub fn dixon_q_test(data: &[f64], alpha: f64) -> StatsResult<DixonQResult> {
    let n = data.len();

    if n < 3 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 3 });
    }
    if n > 30 {
        return Err(StatsError::InvalidParameter {
            name: "n".into(),
            reason: "Dixon's Q-test is only valid for n ≤ 30".into(),
        });
    }

    // Only 0.05 and 0.01 are supported
    let use_005 = (alpha - 0.05).abs() < 1e-9;
    let use_001 = (alpha - 0.01).abs() < 1e-9;
    if !use_005 && !use_001 {
        return Err(StatsError::InvalidParameter {
            name: "alpha".into(),
            reason: "Dixon's Q-test only supports alpha = 0.05 or alpha = 0.01".into(),
        });
    }

    for (i, &v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    // Build sorted index to preserve original indices
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| {
        data[a]
            .partial_cmp(&data[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let x_min = data[sorted_idx[0]];
    let x_max = data[sorted_idx[n - 1]];
    let range = x_max - x_min;

    if range < f64::EPSILON * x_max.abs().max(1.0) {
        return Err(StatsError::NumericalInstability(
            "Dixon's Q-test: range is (near) zero; all values are identical".into(),
        ));
    }

    // Q for the minimum suspect: gap = x[1] - x[0]
    let q_min = (data[sorted_idx[1]] - x_min) / range;
    // Q for the maximum suspect: gap = x[n-1] - x[n-2]
    let q_max = (x_max - data[sorted_idx[n - 2]]) / range;

    // The candidate is whichever gives the larger Q
    let (q_stat, outlier_sorted_pos) = if q_max >= q_min {
        (q_max, n - 1)
    } else {
        (q_min, 0)
    };

    let outlier_index = sorted_idx[outlier_sorted_pos];
    let outlier_value = data[outlier_index];

    let table = if use_005 { Q_CRIT_005 } else { Q_CRIT_001 };
    let q_crit = interpolate_q_crit(table, n);

    Ok(DixonQResult {
        statistic: q_stat,
        critical_value: q_crit,
        alpha,
        is_outlier: q_stat > q_crit,
        outlier_index,
        outlier_value,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Grubbs detects obvious outlier
    #[test]
    fn grubbs_obvious_outlier_detected() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let r = grubbs_test(&data, 0.05).expect("valid input");
        assert!(r.is_outlier, "100.0 should be flagged as outlier");
        assert!(
            (r.outlier_value - 100.0).abs() < 1e-9,
            "outlier value should be 100.0"
        );
        assert_eq!(r.outlier_index, 5);
    }

    // 2. Grubbs on uniform data — no outlier
    #[test]
    fn grubbs_uniform_data_no_outlier() {
        let data = [10.0, 10.1, 10.2, 10.15, 10.05, 10.08, 10.12, 10.07];
        let r = grubbs_test(&data, 0.05).expect("valid input");
        assert!(
            !r.is_outlier,
            "tightly clustered data should not yield outlier"
        );
    }

    // 3. Grubbs with n < 6 → InsufficientSampleSize
    #[test]
    fn grubbs_too_few_samples_error() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let err = grubbs_test(&data, 0.05).expect_err("should error on n=5");
        assert!(
            matches!(err, StatsError::InsufficientSampleSize { got: 5, need: 6 }),
            "unexpected error: {err}"
        );
    }

    // 4. Grubbs with invalid alpha → InvalidParameter
    #[test]
    fn grubbs_invalid_alpha_error() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = grubbs_test(&data, 0.9).expect_err("alpha=0.9 is out of range");
        assert!(
            matches!(err, StatsError::InvalidParameter { ref name, .. } if name == "alpha"),
            "unexpected error: {err}"
        );
        let err2 = grubbs_test(&data, -0.01).expect_err("negative alpha");
        assert!(
            matches!(err2, StatsError::InvalidParameter { ref name, .. } if name == "alpha"),
            "unexpected error: {err2}"
        );
    }

    // 5. Dixon detects outlier in [1,2,3,4,50]
    #[test]
    fn dixon_obvious_outlier_detected() {
        let data = [1.0, 2.0, 3.0, 4.0, 50.0];
        let r = dixon_q_test(&data, 0.05).expect("valid input");
        assert!(r.is_outlier, "50.0 should be flagged as outlier");
        assert!(
            (r.outlier_value - 50.0).abs() < 1e-9,
            "outlier value should be 50.0"
        );
    }

    // 6. Dixon on [1,2,3,4,5] — no outlier
    #[test]
    fn dixon_uniform_data_no_outlier() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = dixon_q_test(&data, 0.05).expect("valid input");
        assert!(!r.is_outlier, "uniform sequence should not yield outlier");
    }

    // 7. Dixon with n < 3 → InsufficientSampleSize
    #[test]
    fn dixon_too_few_samples_error() {
        let data = [1.0, 2.0];
        let err = dixon_q_test(&data, 0.05).expect_err("should error on n=2");
        assert!(
            matches!(err, StatsError::InsufficientSampleSize { got: 2, need: 3 }),
            "unexpected error: {err}"
        );
    }

    // 8. Dixon with unsupported alpha → InvalidParameter
    #[test]
    fn dixon_invalid_alpha_error() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let err = dixon_q_test(&data, 0.10).expect_err("alpha=0.10 not supported");
        assert!(
            matches!(err, StatsError::InvalidParameter { ref name, .. } if name == "alpha"),
            "unexpected error: {err}"
        );
    }

    // 9. Grubbs statistic equals max|x - mean| / std
    #[test]
    fn grubbs_statistic_formula_correctness() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let std_dev = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let expected_g = data
            .iter()
            .map(|&x| (x - mean).abs() / std_dev)
            .fold(0.0_f64, f64::max);

        let r = grubbs_test(&data, 0.05).expect("valid input");
        assert!(
            (r.statistic - expected_g).abs() < 1e-10,
            "G statistic mismatch: got {}, expected {expected_g}",
            r.statistic
        );
    }

    // 10. Dixon with all-identical values → NumericalInstability
    #[test]
    fn dixon_all_identical_numerical_instability() {
        let data = [5.0, 5.0, 5.0, 5.0, 5.0];
        let err = dixon_q_test(&data, 0.05).expect_err("zero range should fail");
        assert!(
            matches!(err, StatsError::NumericalInstability(_)),
            "unexpected error: {err}"
        );
    }
}

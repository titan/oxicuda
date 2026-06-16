//! Mann-Kendall trend test, Sen's slope estimator, and the seasonal Mann-Kendall test.
//!
//! # Mann-Kendall statistic
//!
//! For a series `x_1, …, x_n`,
//!
//! ```text
//! S = Σ_{i<j} sign(x_j − x_i),
//! ```
//!
//! counting the net number of increasing versus decreasing pairs.  Under the null
//! of no trend `E[S] = 0` and, with a correction for tied groups,
//!
//! ```text
//! Var(S) = [ n(n−1)(2n+5) − Σ_g t_g(t_g−1)(2t_g+5) ] / 18,
//! ```
//!
//! where `t_g` is the size of the `g`-th tied group.  The continuity-corrected
//! standard normal score is
//!
//! ```text
//!         ⎧ (S − 1) / √Var(S),  S > 0
//!  Z =    ⎨  0,                 S = 0
//!         ⎩ (S + 1) / √Var(S),  S < 0,
//! ```
//!
//! and the two-sided p-value is `2·(1 − Φ(|Z|))`.
//!
//! # Sen's slope
//!
//! ```text
//! β = median_{i<j} (x_j − x_i) / (j − i),
//! ```
//!
//! the robust trend magnitude (median of pairwise slopes against the time index).
//!
//! # Seasonal Mann-Kendall (Hirsch 1982)
//!
//! Compute `S_m` and `Var(S_m)` within each season `m`, then aggregate
//! `S = Σ_m S_m`, `Var(S) = Σ_m Var(S_m)` and form the same continuity-corrected
//! `Z`.  This removes within-year cyclical structure that would otherwise inflate
//! the apparent trend.
//!
//! # References
//! - Mann, H. B. (1945) "Nonparametric tests against trend". *Econometrica*
//!   13(3):245-259.
//! - Kendall, M. G. (1975) *Rank Correlation Methods*, 4th ed. Charles Griffin.
//! - Sen, P. K. (1968) "Estimates of the regression coefficient based on Kendall's
//!   tau". *JASA* 63(324):1379-1389.
//! - Hirsch, R. M., Slack, J. R. & Smith, R. A. (1982) "Techniques of trend
//!   analysis for monthly water quality data". *Water Resources Research*
//!   18(1):107-121.

use crate::error::{StatsError, StatsResult};

/// Direction of the detected trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// `Z` significantly positive at the chosen level.
    Increasing,
    /// `Z` significantly negative at the chosen level.
    Decreasing,
    /// Null of no trend not rejected.
    NoTrend,
}

/// Result of a Mann-Kendall trend test.
#[derive(Debug, Clone, PartialEq)]
pub struct MannKendallResult {
    /// Mann-Kendall statistic `S`.
    pub s: f64,
    /// Variance of `S` (tie-corrected).
    pub var_s: f64,
    /// Continuity-corrected standard normal score `Z`.
    pub z: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Kendall's `τ = S / (n(n−1)/2)`.
    pub tau: f64,
    /// Sen's slope estimate (median pairwise slope).
    pub sen_slope: f64,
    /// Classified trend direction at the supplied significance level.
    pub trend: TrendDirection,
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// `sign(x)` returning −1, 0, or +1.
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Standard normal CDF `Φ(z)` using the crate's `erf`.
fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + crate::special::erf::erf(z / std::f64::consts::SQRT_2))
}

/// Raw Mann-Kendall `S` statistic of a series.
fn mk_s(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            s += sign(x[j] - x[i]);
        }
    }
    s
}

/// Tie-corrected variance of `S` for a single series.
///
/// `Var(S) = [n(n−1)(2n+5) − Σ_g t_g(t_g−1)(2t_g+5)] / 18`.
fn mk_var_s(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let base = n * (n - 1.0) * (2.0 * n + 5.0);
    let tie = tie_term(x);
    (base - tie) / 18.0
}

/// `Σ_g t_g(t_g−1)(2t_g+5)` over groups of equal values.
fn tie_term(x: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut term = 0.0;
    let mut run = 1.0_f64;
    for w in sorted.windows(2) {
        if w[0] == w[1] {
            run += 1.0;
        } else {
            if run > 1.0 {
                term += run * (run - 1.0) * (2.0 * run + 5.0);
            }
            run = 1.0;
        }
    }
    if run > 1.0 {
        term += run * (run - 1.0) * (2.0 * run + 5.0);
    }
    term
}

/// Continuity-corrected `Z` from `S` and `Var(S)`.
fn mk_z(s: f64, var_s: f64) -> f64 {
    if var_s <= 0.0 {
        return 0.0;
    }
    let sd = var_s.sqrt();
    if s > 0.0 {
        (s - 1.0) / sd
    } else if s < 0.0 {
        (s + 1.0) / sd
    } else {
        0.0
    }
}

/// Two-sided normal p-value from `Z`.
fn two_sided_p(z: f64) -> f64 {
    (2.0 * (1.0 - standard_normal_cdf(z.abs()))).clamp(0.0, 1.0)
}

/// Sen's slope: median of `(x_j − x_i)/(j − i)` over `i < j`.
fn sen_slope(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut slopes: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            slopes.push((x[j] - x[i]) / (j - i) as f64);
        }
    }
    median(&mut slopes)
}

/// Median of a slice (via partial selection).  Returns 0 for empty input.
fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    if n % 2 == 1 {
        let (_, m, _) = values.select_nth_unstable_by(n / 2, cmp);
        *m
    } else {
        let hi = n / 2;
        let (_, upper, _) = values.select_nth_unstable_by(hi, cmp);
        let upper = *upper;
        let lo = hi - 1;
        values.select_nth_unstable_by(lo, cmp);
        0.5 * (values[lo] + upper)
    }
}

fn classify(z: f64, p: f64, alpha: f64) -> TrendDirection {
    if p < alpha {
        if z > 0.0 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        }
    } else {
        TrendDirection::NoTrend
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Mann-Kendall trend test with Sen's slope.
///
/// `alpha` is the two-sided significance level used to classify the trend (the
/// numeric `z`/`p_value`/`sen_slope` outputs are independent of it).
///
/// # Errors
/// Returns an error for fewer than three observations or non-finite values.
pub fn mann_kendall(x: &[f64], alpha: f64) -> StatsResult<MannKendallResult> {
    let n = x.len();
    if n < 3 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 3 });
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "alpha".to_string(),
            reason: format!("significance level must be in (0, 1); got {alpha}"),
        });
    }
    for (i, v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    let s = mk_s(x);
    let var_s = mk_var_s(x);
    let z = mk_z(s, var_s);
    let p_value = two_sided_p(z);
    let n_f = n as f64;
    let tau = s / (n_f * (n_f - 1.0) / 2.0);
    let slope = sen_slope(x);
    let trend = classify(z, p_value, alpha);

    Ok(MannKendallResult {
        s,
        var_s,
        z,
        p_value,
        tau,
        sen_slope: slope,
        trend,
    })
}

/// Standalone Sen's slope estimator (median pairwise slope against the time index).
///
/// # Errors
/// Returns an error for fewer than two observations or non-finite values.
pub fn sens_slope(x: &[f64]) -> StatsResult<f64> {
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    for (i, v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    Ok(sen_slope(x))
}

/// Seasonal Mann-Kendall test (Hirsch 1982).
///
/// `data` is row-major `n_years × n_seasons` (e.g. years × months).  `S` and
/// `Var(S)` are accumulated within each season's column, then aggregated across
/// seasons before forming the continuity-corrected `Z`.
///
/// # Errors
/// Returns an error for shape mismatch, fewer than two seasons or two years, or
/// non-finite values.
pub fn seasonal_mann_kendall(
    data: &[f64],
    n_years: usize,
    n_seasons: usize,
    alpha: f64,
) -> StatsResult<MannKendallResult> {
    if data.len() != n_years * n_seasons {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_years, n_seasons],
            got: vec![data.len()],
        });
    }
    if n_seasons == 0 || n_years < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: n_years,
            need: 2,
        });
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "alpha".to_string(),
            reason: format!("significance level must be in (0, 1); got {alpha}"),
        });
    }
    for (i, v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    let mut s_total = 0.0;
    let mut var_total = 0.0;
    let mut tau_num = 0.0;
    let mut tau_den = 0.0;
    let mut all_slopes: Vec<f64> = Vec::new();

    for m in 0..n_seasons {
        // Extract season-m column (one value per year).
        let column: Vec<f64> = (0..n_years).map(|yr| data[yr * n_seasons + m]).collect();
        s_total += mk_s(&column);
        var_total += mk_var_s(&column);
        let ny = n_years as f64;
        tau_num += mk_s(&column);
        tau_den += ny * (ny - 1.0) / 2.0;
        // Sen's slope across seasons pools all within-season pairwise slopes
        // (against the year index), per Hirsch et al.
        for i in 0..n_years {
            for j in (i + 1)..n_years {
                all_slopes.push((column[j] - column[i]) / (j - i) as f64);
            }
        }
    }

    let z = mk_z(s_total, var_total);
    let p_value = two_sided_p(z);
    let tau = if tau_den > 0.0 {
        tau_num / tau_den
    } else {
        0.0
    };
    let slope = median(&mut all_slopes);
    let trend = classify(z, p_value, alpha);

    Ok(MannKendallResult {
        s: s_total,
        var_s: var_total,
        z,
        p_value,
        tau,
        sen_slope: slope,
        trend,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── (a) Strictly increasing series ────────────────────────────────────────

    #[test]
    fn strictly_increasing_is_max_s() {
        let n = 12;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let res = mann_kendall(&x, 0.05).expect("mk");
        let max_s = (n * (n - 1) / 2) as f64;
        assert!((res.s - max_s).abs() < 1e-12, "S={} max={}", res.s, max_s);
        assert!(res.z > 3.0, "Z={} should be large positive", res.z);
        assert!(res.p_value < 1e-3, "p={}", res.p_value);
        assert_eq!(res.trend, TrendDirection::Increasing);
        assert!((res.tau - 1.0).abs() < 1e-12);
    }

    // ── (b) Strictly decreasing series ────────────────────────────────────────

    #[test]
    fn strictly_decreasing_is_min_s() {
        let n = 12;
        let x: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
        let res = mann_kendall(&x, 0.05).expect("mk");
        let min_s = -((n * (n - 1) / 2) as f64);
        assert!((res.s - min_s).abs() < 1e-12, "S={} min={}", res.s, min_s);
        assert!(res.z < -3.0, "Z={} should be large negative", res.z);
        assert!(res.p_value < 1e-3);
        assert_eq!(res.trend, TrendDirection::Decreasing);
        assert!((res.tau + 1.0).abs() < 1e-12);
    }

    // ── (c) Constant / no trend ───────────────────────────────────────────────

    #[test]
    fn constant_series_has_zero_s() {
        let x = vec![5.0; 10];
        let res = mann_kendall(&x, 0.05).expect("mk");
        assert!((res.s - 0.0).abs() < 1e-12);
        assert!((res.z - 0.0).abs() < 1e-12);
        assert!((res.p_value - 1.0).abs() < 1e-9, "p={}", res.p_value);
        assert_eq!(res.trend, TrendDirection::NoTrend);
        assert!((res.sen_slope - 0.0).abs() < 1e-12);
    }

    #[test]
    fn symmetric_tent_no_trend() {
        // A symmetric tent rises then falls by the same amount ⇒ S = 0.
        // [1,2,3,2,1]: +4 increasing pairs, −4 decreasing pairs, 2 ties.
        let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let res = mann_kendall(&x, 0.05).expect("mk");
        assert!((res.s - 0.0).abs() < 1e-12, "S={}", res.s);
        assert_eq!(res.trend, TrendDirection::NoTrend);
    }

    // ── (d) Tie correction reduces Var(S) ─────────────────────────────────────

    #[test]
    fn ties_reduce_variance() {
        // Series with repeats vs the no-tie formula.
        let x = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        let n = x.len() as f64;
        let no_tie_var = n * (n - 1.0) * (2.0 * n + 5.0) / 18.0;
        let with_tie_var = mk_var_s(&x);
        assert!(
            with_tie_var < no_tie_var,
            "tie var {with_tie_var} should be < no-tie var {no_tie_var}"
        );
        // Explicit check of the correction: groups of size 2 and 3.
        // tie = 2·1·9 + 3·2·11 = 18 + 66 = 84 ; Var = (base − 84)/18.
        let base = n * (n - 1.0) * (2.0 * n + 5.0);
        let expected = (base - 84.0) / 18.0;
        assert!((with_tie_var - expected).abs() < 1e-9);
    }

    // ── (e) Sen's slope equals median pairwise slope & recovers a trend ───────

    #[test]
    fn sens_slope_matches_median_and_trend() {
        // Linear trend slope 0.5 with a small deterministic zig-zag.
        let n = 21;
        let x: Vec<f64> = (0..n)
            .map(|i| 0.5 * i as f64 + if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let slope = sens_slope(&x).expect("slope");
        // Recovers the underlying slope.
        assert!((slope - 0.5).abs() < 0.05, "Sen slope {slope}");

        // Matches an independent naive median of all pairwise slopes.
        let mut naive: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                naive.push((x[j] - x[i]) / (j - i) as f64);
            }
        }
        naive.sort_by(|a, b| a.partial_cmp(b).expect("partial_cmp should succeed"));
        let nn = naive.len();
        let med = if nn % 2 == 1 {
            naive[nn / 2]
        } else {
            0.5 * (naive[nn / 2 - 1] + naive[nn / 2])
        };
        assert!((slope - med).abs() < 1e-12, "slope {slope} median {med}");

        let res = mann_kendall(&x, 0.05).expect("mk");
        assert!((res.sen_slope - slope).abs() < 1e-12);
    }

    #[test]
    fn sens_slope_hand_value() {
        // x = [1, 4, 9]; pairwise slopes: (4−1)/1=3, (9−1)/2=4, (9−4)/1=5.
        // median = 4.
        let x = [1.0, 4.0, 9.0];
        let slope = sens_slope(&x).expect("slope");
        assert!((slope - 4.0).abs() < 1e-12, "slope={slope}");
    }

    // ── (f) Seasonal MK aggregates per-season S and variance ──────────────────

    #[test]
    fn seasonal_aggregates_match_manual_sum() {
        // 5 years × 2 seasons. Season 0 increasing, season 1 increasing.
        // data row-major: year-major, season-minor.
        let season0 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let season1 = [10.0, 12.0, 11.0, 14.0, 16.0];
        let mut data = vec![0.0; 5 * 2];
        for yr in 0..5 {
            data[yr * 2] = season0[yr];
            data[yr * 2 + 1] = season1[yr];
        }
        let res = seasonal_mann_kendall(&data, 5, 2, 0.05).expect("smk");

        // Manual per-season aggregation.
        let s0 = mk_s(&season0);
        let s1 = mk_s(&season1);
        let v0 = mk_var_s(&season0);
        let v1 = mk_var_s(&season1);
        assert!(
            (res.s - (s0 + s1)).abs() < 1e-12,
            "S={} vs {}",
            res.s,
            s0 + s1
        );
        assert!(
            (res.var_s - (v0 + v1)).abs() < 1e-12,
            "Var={} vs {}",
            res.var_s,
            v0 + v1
        );
        // Both seasons trend up ⇒ aggregate increasing.
        assert_eq!(res.trend, TrendDirection::Increasing);
        // The aggregated Z matches the continuity-corrected combination.
        let expected_z = mk_z(s0 + s1, v0 + v1);
        assert!((res.z - expected_z).abs() < 1e-12);
    }

    #[test]
    fn seasonal_removes_within_year_cycle() {
        // Strong sawtooth season means but a clear long-run rise within each season.
        let n_years = 6;
        let n_seasons = 4;
        let mut data = vec![0.0; n_years * n_seasons];
        let season_offset = [0.0, 50.0, -50.0, 20.0];
        for yr in 0..n_years {
            for s in 0..n_seasons {
                data[yr * n_seasons + s] = season_offset[s] + 2.0 * yr as f64;
            }
        }
        let res = seasonal_mann_kendall(&data, n_years, n_seasons, 0.05).expect("smk");
        assert_eq!(res.trend, TrendDirection::Increasing);
        assert!(res.sen_slope > 0.0);
    }

    // ── (g) p ∈ [0,1] and symmetric under sign flip ───────────────────────────

    #[test]
    fn p_value_symmetric_under_sign_flip() {
        let x = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0];
        let neg: Vec<f64> = x.iter().map(|&v| -v).collect();
        let a = mann_kendall(&x, 0.05).expect("a");
        let b = mann_kendall(&neg, 0.05).expect("b");
        assert!((0.0..=1.0).contains(&a.p_value));
        assert!((0.0..=1.0).contains(&b.p_value));
        // S flips sign, |Z| and p unchanged.
        assert!((a.s + b.s).abs() < 1e-9, "S not antisymmetric");
        assert!((a.p_value - b.p_value).abs() < 1e-12, "p not symmetric");
        assert!((a.z + b.z).abs() < 1e-9, "Z not antisymmetric");
    }

    // ── Error paths ───────────────────────────────────────────────────────────

    #[test]
    fn too_short_errors() {
        assert!(mann_kendall(&[1.0, 2.0], 0.05).is_err());
        assert!(sens_slope(&[1.0]).is_err());
    }

    #[test]
    fn bad_alpha_errors() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert!(mann_kendall(&x, 0.0).is_err());
        assert!(mann_kendall(&x, 1.0).is_err());
    }

    #[test]
    fn seasonal_shape_mismatch_errors() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(seasonal_mann_kendall(&data, 2, 2, 0.05).is_err());
    }

    #[test]
    fn non_finite_errors() {
        let x = vec![1.0, f64::NAN, 3.0, 4.0];
        assert!(mann_kendall(&x, 0.05).is_err());
    }
}

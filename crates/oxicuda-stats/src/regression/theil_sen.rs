//! Theil-Sen and Siegel robust (median-of-slopes) linear regression.
//!
//! # Algorithm Overview
//!
//! These estimators replace the least-squares slope (which is fully corrupted by a
//! single outlier — 0 % breakdown) with a *median* of pairwise slopes, giving a
//! robust fit that tolerates a large fraction of contaminated data.
//!
//! ## Theil-Sen estimator (Theil 1950, Sen 1968)
//!
//! ```text
//! slope     = median_{i<j, x_i ≠ x_j}  (y_j − y_i) / (x_j − x_i)
//! intercept = median_i (y_i − slope · x_i)
//! ```
//!
//! The breakdown point of the Theil-Sen slope is ≈ 29.3 % (`1 − 1/√2`): the median
//! of the `C(n,2)` pairwise slopes resists corruption until roughly that fraction of
//! the *points* are bad.
//!
//! ## Siegel repeated-median estimator (Siegel 1982)
//!
//! ```text
//! slope = median_i ( median_{j ≠ i} (y_j − y_i) / (x_j − x_i) )
//! ```
//!
//! The nested ("repeated") median pushes the breakdown point up to ≈ 50 %, the
//! highest attainable for an affine-equivariant regression estimator.
//!
//! ## Confidence interval (Kendall-score / rank method)
//!
//! The distribution-free CI for the Theil-Sen slope inverts the Kendall *S* statistic.
//! With `N` valid pairwise slopes sorted ascending, the rank offsets are
//!
//! ```text
//! C_α = z_{1−α/2} · sqrt(Var(S)),   Var(S) = n(n−1)(2n+5)/18  (tie-corrected),
//! M_lower = (N − C_α) / 2,   M_upper = (N + C_α) / 2,
//! ```
//!
//! and the CI endpoints are the `M_lower`-th and `(M_upper + 1)`-th order statistics
//! of the sorted slopes (Hollander & Wolfe 1999, §9.3).
//!
//! # References
//! - Theil, H. (1950) "A rank-invariant method of linear and polynomial regression
//!   analysis". *Proc. Kon. Ned. Akad. Wetensch.* A 53:386-392, 521-525, 1397-1412.
//! - Sen, P. K. (1968) "Estimates of the regression coefficient based on Kendall's
//!   tau". *JASA* 63(324):1379-1389.
//! - Siegel, A. F. (1982) "Robust regression using repeated medians".
//!   *Biometrika* 69(1):242-244.
//! - Hollander, M. & Wolfe, D. A. (1999) *Nonparametric Statistical Methods*, 2nd ed.

use crate::error::{StatsError, StatsResult};

/// Result of a robust median-of-slopes regression fit.
#[derive(Debug, Clone, PartialEq)]
pub struct TheilSenFit {
    /// Estimated slope (median of pairwise / repeated-median slopes).
    pub slope: f64,
    /// Estimated intercept (median of `y_i − slope · x_i`).
    pub intercept: f64,
    /// Number of valid pairwise slopes that contributed (pairs with `x_i ≠ x_j`).
    pub n_slopes: usize,
    /// Number of input observations.
    pub n_obs: usize,
}

impl TheilSenFit {
    /// Predict the response at a single covariate value `x`.
    #[must_use]
    pub fn predict(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }
}

/// Distribution-free confidence interval for the Theil-Sen slope (Kendall method).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlopeConfidenceInterval {
    /// Lower confidence limit for the slope.
    pub lower: f64,
    /// Upper confidence limit for the slope.
    pub upper: f64,
    /// Nominal two-sided confidence level (e.g. 0.95).
    pub level: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Median helper (consumes a scratch buffer; no allocation policy beyond caller's)
// ─────────────────────────────────────────────────────────────────────────────

/// Median of a slice via in-place partial sort.  Returns `None` for empty input.
///
/// Uses `select_nth_unstable_by` so the cost is `O(n)` on average rather than the
/// `O(n log n)` of a full sort.
fn median_in_place(values: &mut [f64]) -> Option<f64> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(values[0]);
    }
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    if n % 2 == 1 {
        let mid = n / 2;
        let (_, m, _) = values.select_nth_unstable_by(mid, cmp);
        Some(*m)
    } else {
        let hi = n / 2;
        let (_, upper, _) = values.select_nth_unstable_by(hi, cmp);
        let upper = *upper;
        // The lower-middle element is the maximum of the left partition.
        let lo = hi - 1;
        let (left, _, _) = values.select_nth_unstable_by(lo, cmp);
        let _ = left;
        let lower = values[lo];
        Some(0.5 * (lower + upper))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Input validation
// ─────────────────────────────────────────────────────────────────────────────

fn validate(x: &[f64], y: &[f64]) -> StatsResult<()> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    if x.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: x.len(),
            need: 2,
        });
    }
    for (i, v) in x.iter().chain(y.iter()).enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    Ok(())
}

/// Compute the intercept as `median_i (y_i − slope · x_i)`.
fn intercept_from_slope(x: &[f64], y: &[f64], slope: f64) -> StatsResult<f64> {
    let mut residual_levels: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| yi - slope * xi)
        .collect();
    median_in_place(&mut residual_levels).ok_or(StatsError::EmptyInput)
}

// ─────────────────────────────────────────────────────────────────────────────
// Theil-Sen
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a line by the **Theil-Sen** estimator.
///
/// The slope is the median over all `i < j` pairs (with `x_i ≠ x_j`) of the
/// pairwise slope `(y_j − y_i)/(x_j − x_i)`; the intercept is the median of
/// `y_i − slope · x_i`.
///
/// # Errors
/// Returns an error when the inputs differ in length, contain fewer than two
/// observations, hold a non-finite value, or share a single distinct `x` value
/// (no admissible pairs).
pub fn theil_sen_fit(x: &[f64], y: &[f64]) -> StatsResult<TheilSenFit> {
    validate(x, y)?;
    let n = x.len();
    let mut slopes: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            if dx != 0.0 {
                slopes.push((y[j] - y[i]) / dx);
            }
        }
    }
    if slopes.is_empty() {
        return Err(StatsError::InvalidParameter {
            name: "x".to_string(),
            reason: "all x-values are identical; Theil-Sen slope is undefined".to_string(),
        });
    }
    let n_slopes = slopes.len();
    let slope = median_in_place(&mut slopes).ok_or(StatsError::EmptyInput)?;
    let intercept = intercept_from_slope(x, y, slope)?;
    Ok(TheilSenFit {
        slope,
        intercept,
        n_slopes,
        n_obs: n,
    })
}

/// Collect every admissible pairwise slope `(y_j − y_i)/(x_j − x_i)` for `i < j`.
fn pairwise_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            if dx != 0.0 {
                slopes.push((y[j] - y[i]) / dx);
            }
        }
    }
    slopes
}

// ─────────────────────────────────────────────────────────────────────────────
// Siegel repeated-median
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a line by the **Siegel repeated-median** estimator.
///
/// For each point `i`, the inner median over `j ≠ i` of `(y_j − y_i)/(x_j − x_i)`
/// is computed (skipping `x_j == x_i`); the slope is the median of those `n` inner
/// medians.  This nested median attains the maximal ≈ 50 % breakdown point.
///
/// # Errors
/// Returns an error under the same conditions as [`theil_sen_fit`], and additionally
/// when no point has at least one admissible partner.
pub fn siegel_fit(x: &[f64], y: &[f64]) -> StatsResult<TheilSenFit> {
    validate(x, y)?;
    let n = x.len();
    let mut inner_medians: Vec<f64> = Vec::with_capacity(n);
    let mut scratch: Vec<f64> = Vec::with_capacity(n);
    let mut n_slopes = 0usize;
    for i in 0..n {
        scratch.clear();
        for j in 0..n {
            if j == i {
                continue;
            }
            let dx = x[j] - x[i];
            if dx != 0.0 {
                scratch.push((y[j] - y[i]) / dx);
            }
        }
        if let Some(med) = median_in_place(&mut scratch) {
            n_slopes += scratch.len();
            inner_medians.push(med);
        }
    }
    if inner_medians.is_empty() {
        return Err(StatsError::InvalidParameter {
            name: "x".to_string(),
            reason: "no point has an admissible partner; Siegel slope is undefined".to_string(),
        });
    }
    let slope = median_in_place(&mut inner_medians).ok_or(StatsError::EmptyInput)?;
    let intercept = intercept_from_slope(x, y, slope)?;
    Ok(TheilSenFit {
        slope,
        intercept,
        n_slopes,
        n_obs: n,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Confidence interval (Kendall rank method)
// ─────────────────────────────────────────────────────────────────────────────

/// Distribution-free confidence interval for the Theil-Sen slope.
///
/// The CI is obtained by inverting the Kendall *S* statistic: with `N` valid
/// pairwise slopes sorted ascending and `C_α = z_{1−α/2}·√Var(S)`, the endpoints
/// are the order statistics at ranks `⌊(N − C_α)/2⌋` and `⌈(N + C_α)/2⌉ + 1`.
/// `Var(S) = n(n−1)(2n+5)/18` is corrected for ties in the `x`-values via the
/// `Σ t(t−1)(2t+5)` term.
///
/// `level` is the two-sided confidence level (e.g. `0.95`).
///
/// # Errors
/// Returns an error when `level ∉ (0, 1)`, the inputs are invalid, or no admissible
/// pairwise slopes exist.
pub fn theil_sen_confidence_interval(
    x: &[f64],
    y: &[f64],
    level: f64,
) -> StatsResult<SlopeConfidenceInterval> {
    if !(0.0 < level && level < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "level".to_string(),
            reason: format!("confidence level must be in (0, 1); got {level}"),
        });
    }
    validate(x, y)?;
    let n = x.len();

    let mut slopes = pairwise_slopes(x, y);
    if slopes.is_empty() {
        return Err(StatsError::InvalidParameter {
            name: "x".to_string(),
            reason: "all x-values are identical; slope CI is undefined".to_string(),
        });
    }
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_slopes = slopes.len();

    // Var(S) with the Kendall tie correction on the x-values.
    let n_f = n as f64;
    let mut var_s = n_f * (n_f - 1.0) * (2.0 * n_f + 5.0) / 18.0;
    var_s -= tie_correction(x) / 18.0;
    let var_s = var_s.max(0.0);

    // z_{1 − α/2}
    let alpha = 1.0 - level;
    let z = standard_normal_quantile(1.0 - alpha / 2.0)?;
    let c_alpha = z * var_s.sqrt();

    // Rank offsets (Hollander & Wolfe 1999, eq. 9.5).
    let n_slopes_f = n_slopes as f64;
    let m_lower = ((n_slopes_f - c_alpha) / 2.0).floor();
    let m_upper = ((n_slopes_f + c_alpha) / 2.0).ceil();

    // Clamp to valid 0-based order-statistic indices.
    let lower_idx = (m_lower.max(0.0) as usize).min(n_slopes - 1);
    // Upper endpoint is the (M_upper + 1)-th order statistic (1-based) → index M_upper.
    let upper_idx = ((m_upper as usize) + 1).min(n_slopes).saturating_sub(1);

    Ok(SlopeConfidenceInterval {
        lower: slopes[lower_idx],
        upper: slopes[upper_idx],
        level,
    })
}

/// Kendall tie-correction term `Σ_groups t(t−1)(2t+5)` over tied `x`-values.
fn tie_correction(x: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut correction = 0.0;
    let mut run = 1.0_f64;
    for w in sorted.windows(2) {
        if w[0] == w[1] {
            run += 1.0;
        } else {
            if run > 1.0 {
                correction += run * (run - 1.0) * (2.0 * run + 5.0);
            }
            run = 1.0;
        }
    }
    if run > 1.0 {
        correction += run * (run - 1.0) * (2.0 * run + 5.0);
    }
    correction
}

/// Standard-normal quantile `Φ⁻¹(p)` via the crate's `erfinv`.
fn standard_normal_quantile(p: f64) -> StatsResult<f64> {
    let z = std::f64::consts::SQRT_2 * crate::special::erf::erfinv(2.0 * p - 1.0)?;
    Ok(z)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Plain ordinary-least-squares slope, computed independently in the test so
    /// the robustness comparison does not depend on the production OLS path.
    fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        let mut sxy = 0.0;
        let mut sxx = 0.0;
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            sxy += (xi - mean_x) * (yi - mean_y);
            sxx += (xi - mean_x) * (xi - mean_x);
        }
        sxy / sxx
    }

    fn median_naive(values: &[f64]) -> f64 {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).expect("partial_cmp should succeed"));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    }

    // ── (a) Exactly-linear data y = 3x + 2 ────────────────────────────────────

    #[test]
    fn theil_sen_recovers_exact_line() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 2.0).collect();
        let fit = theil_sen_fit(&x, &y).expect("fit");
        assert!((fit.slope - 3.0).abs() < 1e-9, "slope={}", fit.slope);
        assert!(
            (fit.intercept - 2.0).abs() < 1e-9,
            "intercept={}",
            fit.intercept
        );
    }

    #[test]
    fn siegel_recovers_exact_line() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 2.0).collect();
        let fit = siegel_fit(&x, &y).expect("fit");
        assert!((fit.slope - 3.0).abs() < 1e-9, "slope={}", fit.slope);
        assert!(
            (fit.intercept - 2.0).abs() < 1e-9,
            "intercept={}",
            fit.intercept
        );
    }

    // ── (b) Robustness: 20-25 % gross outliers ────────────────────────────────

    #[test]
    fn theil_sen_resists_gross_outliers() {
        let true_slope = 2.0;
        let true_intercept = 1.0;
        // 40 clean points.
        let mut x: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let mut y: Vec<f64> = x
            .iter()
            .map(|&xi| true_slope * xi + true_intercept)
            .collect();
        // Add 10 gross-outlier points (20 % contamination), all yanked far off.
        for i in 0..10 {
            x.push(i as f64);
            y.push(1000.0 + 50.0 * i as f64);
        }
        let ts = theil_sen_fit(&x, &y).expect("fit");
        let ols = ols_slope(&x, &y);
        let ts_err = (ts.slope - true_slope).abs();
        let ols_err = (ols - true_slope).abs();
        // Theil-Sen barely moves; OLS is badly corrupted.
        assert!(ts_err < 0.25, "Theil-Sen slope error {ts_err} too large");
        assert!(
            ts_err < 0.25 * ols_err,
            "expected TS err ({ts_err}) << OLS err ({ols_err})"
        );
    }

    // ── (c) Median-of-slopes on a hand-computable set ─────────────────────────

    #[test]
    fn median_of_slopes_matches_by_hand() {
        // Points: (0,0), (1,1), (2,1), (3,10).
        // Pairwise slopes (i<j):
        //   (0,1):1  (0,2):0.5  (0,3):10/3
        //   (1,2):0  (1,3):4.5
        //   (2,3):9
        // Sorted: 0, 0.5, 1, 10/3, 4.5, 9  → median = (1 + 10/3)/2 = 13/6.
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 1.0, 1.0, 10.0];
        let fit = theil_sen_fit(&x, &y).expect("fit");
        let expected = (1.0 + 10.0 / 3.0) / 2.0;
        assert!(
            (fit.slope - expected).abs() < 1e-12,
            "slope={} expected={}",
            fit.slope,
            expected
        );
        assert_eq!(fit.n_slopes, 6);
        // Cross-check against the naive median of the explicit slope list.
        let slopes = [1.0, 0.5, 10.0 / 3.0, 0.0, 4.5, 9.0];
        assert!((fit.slope - median_naive(&slopes)).abs() < 1e-12);
    }

    // ── (d) Siegel survives ~40 % contamination where Theil-Sen degrades ──────

    #[test]
    fn siegel_survives_where_theil_sen_degrades() {
        let true_slope = 1.0;
        // 30 clean points on y = x.
        let mut x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let mut y: Vec<f64> = x.iter().map(|&xi| true_slope * xi).collect();
        // Inject 20 collinear outliers on a competing steep line y = 8x + 400
        // at large x (40 % of the 50 total points form a coherent bad cluster).
        for i in 0..20 {
            let xi = 60.0 + i as f64;
            x.push(xi);
            y.push(8.0 * xi + 400.0);
        }
        let ts = theil_sen_fit(&x, &y).expect("ts");
        let siegel = siegel_fit(&x, &y).expect("siegel");
        let ts_err = (ts.slope - true_slope).abs();
        let siegel_err = (siegel.slope - true_slope).abs();
        // Siegel's repeated median stays on the clean line; Theil-Sen is pulled away.
        assert!(
            siegel_err < 0.05,
            "Siegel slope error {siegel_err} (slope={})",
            siegel.slope
        );
        assert!(
            siegel_err < ts_err,
            "expected Siegel err ({siegel_err}) < Theil-Sen err ({ts_err})"
        );
        assert!(
            ts_err > 0.5,
            "expected Theil-Sen to degrade here; err was only {ts_err}"
        );
    }

    // ── (e) n = 2 gives the exact connecting line ─────────────────────────────

    #[test]
    fn two_points_exact_line() {
        let x = [1.0, 4.0];
        let y = [3.0, 12.0]; // slope 3, intercept 0
        let fit = theil_sen_fit(&x, &y).expect("fit");
        assert!((fit.slope - 3.0).abs() < 1e-12);
        assert!((fit.intercept - 0.0).abs() < 1e-12);
        assert_eq!(fit.n_slopes, 1);
        let siegel = siegel_fit(&x, &y).expect("fit");
        assert!((siegel.slope - 3.0).abs() < 1e-12);
    }

    // ── (f) Duplicate x-values skipped without NaN / panic ────────────────────

    #[test]
    fn duplicate_x_values_are_skipped() {
        // Two points share x = 1; that pair must be skipped, not produce inf/NaN.
        let x = [1.0, 1.0, 2.0, 3.0];
        let y = [5.0, 9.0, 4.0, 7.0];
        let fit = theil_sen_fit(&x, &y).expect("fit");
        assert!(fit.slope.is_finite(), "slope was {}", fit.slope);
        assert!(fit.intercept.is_finite());
        // C(4,2)=6 pairs but the (0,1) pair is skipped → 5 valid slopes.
        assert_eq!(fit.n_slopes, 5);

        let siegel = siegel_fit(&x, &y).expect("fit");
        assert!(siegel.slope.is_finite());
        assert!(siegel.intercept.is_finite());
    }

    #[test]
    fn all_identical_x_is_error() {
        let x = [2.0, 2.0, 2.0, 2.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        assert!(theil_sen_fit(&x, &y).is_err());
        assert!(siegel_fit(&x, &y).is_err());
    }

    // ── Confidence interval ───────────────────────────────────────────────────

    #[test]
    fn confidence_interval_brackets_slope() {
        // Linear data with continuously-distributed deterministic noise so the
        // pairwise-slope order statistics are genuinely distinct.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut noise = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5 // uniform on (−0.5, 0.5)
        };
        let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0 + noise()).collect();
        let fit = theil_sen_fit(&x, &y).expect("fit");
        let ci = theil_sen_confidence_interval(&x, &y, 0.95).expect("ci");
        assert!(ci.lower <= fit.slope && fit.slope <= ci.upper);
        assert!(ci.lower <= 2.0 && 2.0 <= ci.upper, "CI {:?} misses 2.0", ci);
        assert!(ci.lower < ci.upper, "CI {:?} collapsed", ci);
        assert!((ci.level - 0.95).abs() < 1e-12);
    }

    #[test]
    fn confidence_interval_rejects_bad_level() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 1.0, 2.0, 3.0];
        assert!(theil_sen_confidence_interval(&x, &y, 0.0).is_err());
        assert!(theil_sen_confidence_interval(&x, &y, 1.0).is_err());
    }

    // ── Median helper sanity (even/odd) ───────────────────────────────────────

    #[test]
    fn median_helper_even_and_odd() {
        let mut a = [3.0, 1.0, 2.0];
        assert!(
            (median_in_place(&mut a).expect("median_in_place should succeed") - 2.0).abs() < 1e-12
        );
        let mut b = [4.0, 1.0, 3.0, 2.0];
        assert!(
            (median_in_place(&mut b).expect("median_in_place should succeed") - 2.5).abs() < 1e-12
        );
        let mut empty: [f64; 0] = [];
        assert!(median_in_place(&mut empty).is_none());
    }

    #[test]
    fn predict_uses_line() {
        let fit = TheilSenFit {
            slope: 2.0,
            intercept: -1.0,
            n_slopes: 1,
            n_obs: 2,
        };
        assert!((fit.predict(3.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn mismatched_lengths_error() {
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 2.0];
        assert!(theil_sen_fit(&x, &y).is_err());
    }
}

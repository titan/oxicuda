//! L-moments and heavy-tail robust statistics.
//!
//! L-moments (Hosking 1990) are linear combinations of order statistics that
//! characterise location, scale, skewness and kurtosis of a distribution.  They
//! are more robust to outliers and heavy tails than conventional central moments.
//!
//! # Reference
//! Hosking, J. R. M. (1990). L-moments: Analysis and Estimation of Distributions
//! using Linear Combinations of Order Statistics. *J. R. Statist. Soc. B*, 52(1),
//! 105-124.

use crate::error::{StatsError, StatsResult};

// ─── public types ─────────────────────────────────────────────────────────────

/// The four sample L-moments and their ratio statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct LMoments {
    /// L-mean (= ordinary arithmetic mean).
    pub l1: f64,
    /// L-scale (≥ 0; analogous to standard deviation but linear in order statistics).
    pub l2: f64,
    /// L-skewness numerator L₃.
    pub l3: f64,
    /// L-kurtosis numerator L₄.
    pub l4: f64,
    /// L-skewness ratio τ₃ = L₃ / L₂ ∈ (−1, 1).
    pub tau3: f64,
    /// L-kurtosis ratio τ₄ = L₄ / L₂.
    pub tau4: f64,
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Compute the binomial coefficient C(n, k) as f64 without overflow for moderate n.
#[inline]
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    // Use the multiplicative formula to stay in f64 range for the sizes we encounter.
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

/// Sorted copy of `data`, filtering NaN/Inf.
fn sorted_finite(data: &[f64]) -> StatsResult<Vec<f64>> {
    let mut v: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(v)
}

// ─── probability-weighted moments ────────────────────────────────────────────

/// Probability-weighted moment β_r (order `r`).
///
/// β_r = (1/n) Σᵢ x_(i) C(i−1, r) / C(n−1, r)  (1-indexed, sorted ascending)
///
/// These are the building blocks for L-moments:
///
/// L₁ = β₀,  L₂ = 2β₁ − β₀,  L₃ = 6β₂ − 6β₁ + β₀,  L₄ = 20β₃ − 30β₂ + 12β₁ − β₀
pub fn pwm(data: &[f64], r: usize) -> StatsResult<f64> {
    let sorted = sorted_finite(data)?;
    let n = sorted.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    let cn1r = binom(n - 1, r);
    if cn1r == 0.0 {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: r + 1,
        });
    }
    let mut sum = 0.0_f64;
    for (idx, &x) in sorted.iter().enumerate() {
        // idx is 0-based; i (1-based) = idx + 1; C(i-1, r) = C(idx, r)
        sum += x * binom(idx, r) / cn1r;
    }
    Ok(sum / n as f64)
}

// ─── individual L-moments ────────────────────────────────────────────────────

/// L-mean (first L-moment): equals the ordinary arithmetic mean.
pub fn l_moment_1(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let s: f64 = data.iter().filter(|x| x.is_finite()).sum();
    let n = data.iter().filter(|x| x.is_finite()).count();
    if n == 0 { f64::NAN } else { s / n as f64 }
}

/// L-scale (second L-moment L₂ ≥ 0).
///
/// L₂ = 2β₁ − β₀.
///
/// Requires n ≥ 2.
pub fn l_moment_2(data: &[f64]) -> StatsResult<f64> {
    let sorted = sorted_finite(data)?;
    let n = sorted.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    // Direct formula via order statistics:
    // L₂ = (1/C(n,2)) * Σᵢ x_(i) * (2i-1-n) / 2   (1-based i)
    // which is equivalent to 2β₁ - β₀.
    let beta0 = pwm(&sorted, 0)?;
    let beta1 = pwm(&sorted, 1)?;
    Ok(2.0 * beta1 - beta0)
}

/// L-skewness numerator (third L-moment L₃).
///
/// L₃ = 6β₂ − 6β₁ + β₀.
///
/// Requires n ≥ 3.
pub fn l_moment_3(data: &[f64]) -> StatsResult<f64> {
    let sorted = sorted_finite(data)?;
    let n = sorted.len();
    if n < 3 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 3 });
    }
    let beta0 = pwm(&sorted, 0)?;
    let beta1 = pwm(&sorted, 1)?;
    let beta2 = pwm(&sorted, 2)?;
    Ok(6.0 * beta2 - 6.0 * beta1 + beta0)
}

/// L-kurtosis numerator (fourth L-moment L₄).
///
/// L₄ = 20β₃ − 30β₂ + 12β₁ − β₀.
///
/// Requires n ≥ 4.
pub fn l_moment_4(data: &[f64]) -> StatsResult<f64> {
    let sorted = sorted_finite(data)?;
    let n = sorted.len();
    if n < 4 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 4 });
    }
    let beta0 = pwm(&sorted, 0)?;
    let beta1 = pwm(&sorted, 1)?;
    let beta2 = pwm(&sorted, 2)?;
    let beta3 = pwm(&sorted, 3)?;
    Ok(20.0 * beta3 - 30.0 * beta2 + 12.0 * beta1 - beta0)
}

/// Compute all four L-moments and their ratio statistics in a single pass.
///
/// Returns [`LMoments`] with `l1..l4` and the ratios `tau3 = l3/l2`, `tau4 = l4/l2`.
///
/// Requires n ≥ 4.
pub fn l_moments(data: &[f64]) -> StatsResult<LMoments> {
    let sorted = sorted_finite(data)?;
    let n = sorted.len();
    if n < 4 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 4 });
    }
    let beta0 = pwm(&sorted, 0)?;
    let beta1 = pwm(&sorted, 1)?;
    let beta2 = pwm(&sorted, 2)?;
    let beta3 = pwm(&sorted, 3)?;

    let l1 = beta0;
    let l2 = 2.0 * beta1 - beta0;
    let l3 = 6.0 * beta2 - 6.0 * beta1 + beta0;
    let l4 = 20.0 * beta3 - 30.0 * beta2 + 12.0 * beta1 - beta0;

    let (tau3, tau4) = if l2.abs() < 1e-300 {
        (0.0, 0.0)
    } else {
        (l3 / l2, l4 / l2)
    };

    Ok(LMoments {
        l1,
        l2,
        l3,
        l4,
        tau3,
        tau4,
    })
}

// ─── heavy-tail robust statistics ────────────────────────────────────────────

/// Trimmed mean: remove the bottom and top `trim_frac` fraction of observations
/// and return the mean of the retained sample.
///
/// `trim_frac` must be in `[0, 0.5)`.
///
/// Note: this shadows the simpler version in `robust.rs`; here we handle the
/// variance and standard-error companions and expose them from `lmoments`.
pub fn trimmed_mean_lm(data: &[f64], trim_frac: f64) -> StatsResult<f64> {
    validate_trim_frac(trim_frac)?;
    let sorted = sorted_finite(data)?;
    let n = sorted.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    let k = (n as f64 * trim_frac).floor() as usize;
    let retained = &sorted[k..n - k];
    if retained.is_empty() {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: 2 * k + 1,
        });
    }
    Ok(retained.iter().sum::<f64>() / retained.len() as f64)
}

/// Winsorised mean: replace the bottom and top `trim_frac` fraction of observations
/// with the boundary values (rather than dropping them) and return the mean of the
/// Winsorised sample.
///
/// `trim_frac` must be in `[0, 0.5)`.
pub fn winsorised_mean_lm(data: &[f64], trim_frac: f64) -> StatsResult<f64> {
    let win = winsorise(data, trim_frac)?;
    Ok(win.iter().sum::<f64>() / win.len() as f64)
}

/// Trimmed variance (Bessel-corrected for the retained sub-sample).
///
/// The denominator is `n_trimmed − 1` where `n_trimmed = n − 2k`.
/// Requires `n_trimmed ≥ 2`.
pub fn trimmed_variance(data: &[f64], trim_frac: f64) -> StatsResult<f64> {
    validate_trim_frac(trim_frac)?;
    let sorted = sorted_finite(data)?;
    let n = sorted.len();
    let k = (n as f64 * trim_frac).floor() as usize;
    let retained = &sorted[k..n - k];
    let nt = retained.len();
    if nt < 2 {
        return Err(StatsError::InsufficientSampleSize { got: nt, need: 2 });
    }
    let mean_t = retained.iter().sum::<f64>() / nt as f64;
    let var_t = retained.iter().map(|&x| (x - mean_t).powi(2)).sum::<f64>() / (nt - 1) as f64;
    Ok(var_t)
}

/// Trimmed standard error of the trimmed mean (studentized via the Winsorised
/// variance).
///
/// The Winsorised estimator: replace the outer `k = floor(n * trim_frac)` values
/// with the Winsorised boundary, compute the Winsorised sample variance `s_w²`,
/// then the effective standard error of the trimmed mean is:
///
/// SE = sqrt(s_w² / (n_w * h²))
///
/// where `n_w = n − 2k + 2k = n` (Winsorised sample size is always n) and
/// `h = (n − 2k) / n` is the trimming proportion (fraction retained).
///
/// This is the Dixon-Tukey formula for the studentized trimmed mean.
pub fn trimmed_std_error(data: &[f64], trim_frac: f64) -> StatsResult<f64> {
    validate_trim_frac(trim_frac)?;
    let sorted = sorted_finite(data)?;
    let n = sorted.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    let k = (n as f64 * trim_frac).floor() as usize;
    let h = (n - 2 * k) as f64 / n as f64;
    if h <= 0.0 {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: 2 * k + 1,
        });
    }
    // Winsorise and compute variance
    let win = winsorise_sorted(&sorted, k);
    let n_f = n as f64;
    let mean_w = win.iter().sum::<f64>() / n_f;
    let var_w = win.iter().map(|&x| (x - mean_w).powi(2)).sum::<f64>() / (n_f - 1.0);
    // SE of trimmed mean
    let se = (var_w / (n_f * h * h)).sqrt();
    Ok(se)
}

// ─── internal helpers ─────────────────────────────────────────────────────────

fn validate_trim_frac(trim_frac: f64) -> StatsResult<()> {
    if !(0.0..0.5).contains(&trim_frac) {
        return Err(StatsError::InvalidParameter {
            name: "trim_frac".into(),
            reason: format!("must be in [0, 0.5), got {trim_frac}"),
        });
    }
    Ok(())
}

/// Return a Winsorised version of `data` (sorted ascending, `k` values clamped on each side).
fn winsorise(data: &[f64], trim_frac: f64) -> StatsResult<Vec<f64>> {
    validate_trim_frac(trim_frac)?;
    let sorted = sorted_finite(data)?;
    let n = sorted.len();
    let k = (n as f64 * trim_frac).floor() as usize;
    Ok(winsorise_sorted(&sorted, k))
}

fn winsorise_sorted(sorted: &[f64], k: usize) -> Vec<f64> {
    let n = sorted.len();
    let lo = if k == 0 { sorted[0] } else { sorted[k] };
    let hi = if k == 0 {
        sorted[n - 1]
    } else {
        sorted[n - k - 1]
    };
    sorted.iter().copied().map(|x| x.max(lo).min(hi)).collect()
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // — helper datasets ——————————————————————————————————————————————————————

    fn uniform_data() -> Vec<f64> {
        (1..=20).map(|i| i as f64).collect()
    }

    fn symmetric_data() -> Vec<f64> {
        vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    }

    fn right_skewed_data() -> Vec<f64> {
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 5.0, 10.0, 100.0]
    }

    // ── 1. L-mean equals arithmetic mean ────────────────────────────────────
    #[test]
    fn l_moment_1_equals_mean() {
        let data = uniform_data();
        let l1 = l_moment_1(&data);
        let m: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!((l1 - m).abs() < 1e-12, "L₁={l1} should equal mean={m}");
    }

    // ── 2. L-scale is non-negative ───────────────────────────────────────────
    #[test]
    fn l_moment_2_non_negative() {
        for data in [uniform_data(), right_skewed_data(), symmetric_data()] {
            let l2 = l_moment_2(&data).expect("ok");
            assert!(l2 >= 0.0, "L₂={l2} should be ≥ 0");
        }
    }

    // ── 3. L-scale is zero for constant data ────────────────────────────────
    #[test]
    fn l_moment_2_zero_for_constant() {
        let data = vec![5.0_f64; 10];
        let l2 = l_moment_2(&data).expect("ok");
        assert!(l2.abs() < 1e-12, "L₂={l2} should be 0 for constant data");
    }

    // ── 4. L-skewness τ₃ ∈ [−1, 1] ─────────────────────────────────────────
    #[test]
    fn l_skewness_ratio_in_range() {
        for data in [uniform_data(), right_skewed_data(), symmetric_data()] {
            let lm = l_moments(&data).expect("ok");
            assert!(
                lm.tau3 >= -1.0 && lm.tau3 <= 1.0,
                "τ₃={} should be in [-1,1]",
                lm.tau3
            );
        }
    }

    // ── 5. τ₃ ≈ 0 for symmetric data ────────────────────────────────────────
    #[test]
    fn l_skewness_symmetric_near_zero() {
        let data = symmetric_data();
        let lm = l_moments(&data).expect("ok");
        assert!(
            lm.tau3.abs() < 1e-10,
            "τ₃={} should be ~0 for symmetric data",
            lm.tau3
        );
    }

    // ── 6. PWM β₀ = mean ────────────────────────────────────────────────────
    #[test]
    fn pwm_beta0_equals_mean() {
        let data = uniform_data();
        let b0 = pwm(&data, 0).expect("ok");
        let m: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!((b0 - m).abs() < 1e-12, "β₀={b0} should equal mean={m}");
    }

    // ── 7. PWM requires sufficient sample size ───────────────────────────────
    #[test]
    fn pwm_insufficient_sample_size() {
        let data = vec![1.0, 2.0]; // n=2; β₂ needs C(1,2)=0 → error
        let result = pwm(&data, 2);
        assert!(result.is_err(), "pwm with r=2 and n=2 should error");
    }

    // ── 8. Trimmed mean lies between min and max ─────────────────────────────
    #[test]
    fn trimmed_mean_between_min_max() {
        let data = right_skewed_data();
        let min = *data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).expect("partial_cmp should succeed"))
            .expect("value should be present");
        let max = *data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).expect("partial_cmp should succeed"))
            .expect("value should be present");
        let tm = trimmed_mean_lm(&data, 0.1).expect("ok");
        assert!(
            tm >= min && tm <= max,
            "trimmed mean {tm} not in [{min},{max}]"
        );
    }

    // ── 9. Trimmed mean closer to median for heavy right tail ────────────────
    #[test]
    fn trimmed_mean_robust_to_outlier() {
        let mut data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let full_mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        data.push(1000.0); // heavy outlier
        let trim = trimmed_mean_lm(&data, 0.1).expect("ok");
        // trimmed should be much closer to the original mean
        assert!(
            (trim - full_mean).abs()
                < (data.iter().sum::<f64>() / data.len() as f64 - full_mean).abs(),
            "trimmed mean should be more robust to outlier"
        );
    }

    // ── 10. Winsorised mean ≈ trimmed mean for uniform/light-tailed data ─────
    #[test]
    fn winsorised_approx_trimmed_light_tail() {
        let data = uniform_data();
        let tm = trimmed_mean_lm(&data, 0.1).expect("ok");
        let wm = winsorised_mean_lm(&data, 0.1).expect("ok");
        // For uniform data the two should agree closely (they differ by the boundary mass)
        assert!(
            (tm - wm).abs() < 1.0,
            "trimmed={tm} winsorised={wm} should be close"
        );
    }

    // ── 11. Trimmed variance ≤ full variance for fat-tailed data ────────────
    #[test]
    fn trimmed_variance_le_full_variance_fat_tail() {
        let data = right_skewed_data();
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let full_var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let trim_var = trimmed_variance(&data, 0.1).expect("ok");
        assert!(
            trim_var <= full_var,
            "trimmed var={trim_var} should be ≤ full var={full_var}"
        );
    }

    // ── 12. Trimmed std-error is finite and positive ─────────────────────────
    #[test]
    fn trimmed_std_error_finite_positive() {
        let data = uniform_data();
        let se = trimmed_std_error(&data, 0.1).expect("ok");
        assert!(
            se.is_finite() && se > 0.0,
            "SE={se} should be finite and positive"
        );
    }

    // ── 13. l_moments error on empty input ───────────────────────────────────
    #[test]
    fn l_moments_empty_error() {
        let result = l_moments(&[]);
        assert!(result.is_err());
    }

    // ── 14. l_moment_2 consistency with pwm formula ──────────────────────────
    #[test]
    fn l_moment_2_consistent_with_pwm() {
        let data = uniform_data();
        let l2_direct = l_moment_2(&data).expect("ok");
        let b0 = pwm(&data, 0).expect("ok");
        let b1 = pwm(&data, 1).expect("ok");
        let l2_pwm = 2.0 * b1 - b0;
        assert!(
            (l2_direct - l2_pwm).abs() < 1e-12,
            "L₂ direct={l2_direct} vs pwm={l2_pwm}"
        );
    }

    // ── 15. l4 consistency with pwm formula ──────────────────────────────────
    #[test]
    fn l_moment_4_consistent_with_pwm() {
        let data = uniform_data();
        let l4_direct = l_moment_4(&data).expect("ok");
        let b0 = pwm(&data, 0).expect("ok");
        let b1 = pwm(&data, 1).expect("ok");
        let b2 = pwm(&data, 2).expect("ok");
        let b3 = pwm(&data, 3).expect("ok");
        let l4_pwm = 20.0 * b3 - 30.0 * b2 + 12.0 * b1 - b0;
        assert!(
            (l4_direct - l4_pwm).abs() < 1e-12,
            "L₄ direct={l4_direct} vs pwm={l4_pwm}"
        );
    }
}

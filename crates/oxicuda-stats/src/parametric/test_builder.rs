//! Builder-pattern configurators for parametric hypothesis tests.
//!
//! Wraps the underlying `t_test`, `anova`, and `resampling::bootstrap` functions
//! with a fluent builder API that tracks configuration in one place and derives
//! extended result fields (confidence intervals, significance flags, effect sizes)
//! from the raw test outputs.

use crate::descriptive::quantile::quantile;
use crate::descriptive::summary::mean;
use crate::distributions::student_t::StudentT;
use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;
use crate::parametric::anova::{AnovaResult as RawAnovaResult, one_way_anova};
use crate::parametric::t_test::{one_sample_t, paired_t, two_sample_t, welch_t};

// ─────────────────────────────────────────────────────────────────────────────
// TTestBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Direction of the alternative hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailDirection {
    /// H₁: μ ≠ μ₀ (default).
    Two,
    /// H₁: μ < μ₀.
    Left,
    /// H₁: μ > μ₀.
    Right,
}

/// Result returned by all `TTestBuilder` methods.
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// t-statistic.
    pub statistic: f64,
    /// p-value adjusted for the configured tail direction.
    pub p_value: f64,
    /// Degrees of freedom.
    pub df: f64,
    /// Lower bound of the confidence interval for the mean difference.
    pub ci_lower: f64,
    /// Upper bound of the confidence interval for the mean difference.
    pub ci_upper: f64,
    /// `true` when `p_value < alpha`.
    pub significant: bool,
}

/// Fluent builder for one-sample, two-sample, and paired t-tests.
///
/// ```
/// use oxicuda_stats::parametric::test_builder::{TTestBuilder, TailDirection};
/// let r = TTestBuilder::new()
///     .two_tailed()
///     .alpha(0.05)
///     .null_mean(5.0)
///     .one_sample(&[4.5, 5.0, 5.5, 6.0, 5.5])
///     .expect("ok");
/// assert!(r.statistic.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct TTestBuilder {
    tail: TailDirection,
    /// `true` → pooled (Student), `false` → Welch.
    equal_var: bool,
    /// Significance level.
    alpha: f64,
    /// Null-hypothesis mean for one-sample tests.
    mu0: f64,
}

impl Default for TTestBuilder {
    fn default() -> Self {
        Self {
            tail: TailDirection::Two,
            equal_var: true,
            alpha: 0.05,
            mu0: 0.0,
        }
    }
}

impl TTestBuilder {
    /// Construct a builder with defaults: two-tailed, Student (equal-var), α=0.05, μ₀=0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set two-sided alternative hypothesis (default).
    pub fn two_tailed(mut self) -> Self {
        self.tail = TailDirection::Two;
        self
    }

    /// Set left-sided alternative hypothesis H₁: μ < μ₀.
    pub fn left_tailed(mut self) -> Self {
        self.tail = TailDirection::Left;
        self
    }

    /// Set right-sided alternative hypothesis H₁: μ > μ₀.
    pub fn right_tailed(mut self) -> Self {
        self.tail = TailDirection::Right;
        self
    }

    /// Use the equal-variance (Student) two-sample variant.
    pub fn equal_variance(mut self) -> Self {
        self.equal_var = true;
        self
    }

    /// Use Welch's unequal-variance t-test for two-sample tests.
    pub fn welch(mut self) -> Self {
        self.equal_var = false;
        self
    }

    /// Set the significance level α.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `a` is not in `(0, 1)`.
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Set the null-hypothesis mean for one-sample tests.
    pub fn null_mean(mut self, mu0: f64) -> Self {
        self.mu0 = mu0;
        self
    }

    // ── internal helpers ────────────────────────────────────────────────────

    /// Pick the appropriate p-value from the raw result based on tail direction.
    fn select_p(&self, raw: &crate::parametric::t_test::TTestResult) -> f64 {
        match self.tail {
            TailDirection::Two => raw.p_value_two_sided,
            TailDirection::Left => raw.p_value_left,
            TailDirection::Right => raw.p_value_right,
        }
    }

    /// Compute symmetric confidence interval for a mean difference with given
    /// standard error and degrees of freedom.
    fn ci_from_se(&self, stat: f64, se: f64, df: f64) -> StatsResult<(f64, f64)> {
        if !(self.alpha > 0.0 && self.alpha < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        let dist = StudentT::new(df)?;
        // Two-tailed critical value; for one-tailed CIs the same formula still
        // gives the symmetric bound (conservative).
        let t_crit = dist.ppf(1.0 - self.alpha / 2.0)?;
        let margin = t_crit * se;
        Ok((stat - margin, stat + margin))
    }

    // ── public test methods ─────────────────────────────────────────────────

    /// Run a one-sample t-test: H₀: μ = μ₀.
    pub fn one_sample(&self, data: &[f64]) -> StatsResult<TTestResult> {
        if !(self.alpha > 0.0 && self.alpha < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        let raw = one_sample_t(data, self.mu0)?;
        let p = self.select_p(&raw);

        // SE = s/sqrt(n), mean difference = xbar - mu0
        let n = data.len() as f64;
        let xbar = mean(data)?;
        let diff = xbar - self.mu0;
        // From the t-statistic: t = diff / se → se = diff / t (if t != 0)
        let se = if raw.t_statistic.abs() > 1e-15 {
            diff.abs() / raw.t_statistic.abs()
        } else {
            0.0
        };
        let (ci_lower, ci_upper) = if n >= 2.0 {
            self.ci_from_se(diff, se, raw.df)?
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };

        Ok(TTestResult {
            statistic: raw.t_statistic,
            p_value: p,
            df: raw.df,
            ci_lower,
            ci_upper,
            significant: p < self.alpha,
        })
    }

    /// Run a two-sample t-test.
    ///
    /// Uses Student (equal-variance) or Welch (unequal-variance) depending on
    /// the builder configuration.
    pub fn two_sample(&self, data_a: &[f64], data_b: &[f64]) -> StatsResult<TTestResult> {
        if !(self.alpha > 0.0 && self.alpha < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        let raw = if self.equal_var {
            two_sample_t(data_a, data_b)?
        } else {
            welch_t(data_a, data_b)?
        };
        let p = self.select_p(&raw);

        // mean difference for CI
        let ma = mean(data_a)?;
        let mb = mean(data_b)?;
        let diff = ma - mb;
        let se = if raw.t_statistic.abs() > 1e-15 {
            diff.abs() / raw.t_statistic.abs()
        } else {
            0.0
        };
        let (ci_lower, ci_upper) = self.ci_from_se(diff, se, raw.df)?;

        Ok(TTestResult {
            statistic: raw.t_statistic,
            p_value: p,
            df: raw.df,
            ci_lower,
            ci_upper,
            significant: p < self.alpha,
        })
    }

    /// Run a paired t-test.
    ///
    /// H₀: mean(before − after) = 0.
    pub fn paired(&self, before: &[f64], after: &[f64]) -> StatsResult<TTestResult> {
        if !(self.alpha > 0.0 && self.alpha < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        if before.len() != after.len() {
            return Err(StatsError::DimensionMismatch {
                a: before.len(),
                b: after.len(),
            });
        }
        let diffs: Vec<f64> = before.iter().zip(after).map(|(b, a)| b - a).collect();
        let raw = paired_t(before, after)?;
        let p = self.select_p(&raw);

        let diff_mean = mean(&diffs)?;
        let se = if raw.t_statistic.abs() > 1e-15 {
            diff_mean.abs() / raw.t_statistic.abs()
        } else {
            0.0
        };
        let (ci_lower, ci_upper) = self.ci_from_se(diff_mean, se, raw.df)?;

        Ok(TTestResult {
            statistic: raw.t_statistic,
            p_value: p,
            df: raw.df,
            ci_lower,
            ci_upper,
            significant: p < self.alpha,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnovaBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Result returned by `AnovaBuilder`.
#[derive(Debug, Clone)]
pub struct AnovaBuilderResult {
    /// F-statistic.
    pub f_stat: f64,
    /// p-value for the F-test.
    pub p_value: f64,
    /// Between-group degrees of freedom.
    pub df_between: f64,
    /// Within-group degrees of freedom.
    pub df_within: f64,
    /// `true` when `p_value < alpha`.
    pub significant: bool,
    /// Eta-squared effect size (`SS_between / SS_total`); `None` unless
    /// `with_effect_size()` was called.
    pub eta_squared: Option<f64>,
    /// Raw ANOVA result from the underlying solver.
    pub raw: RawAnovaResult,
}

/// Fluent builder for one-way ANOVA.
///
/// ```
/// use oxicuda_stats::parametric::test_builder::AnovaBuilder;
/// let r = AnovaBuilder::new()
///     .alpha(0.05)
///     .with_effect_size()
///     .one_way(&[
///         vec![1.0, 2.0, 3.0],
///         vec![3.0, 4.0, 5.0],
///         vec![5.0, 6.0, 7.0],
///     ])
///     .expect("ok");
/// assert!((r.f_stat - 12.0).abs() < 1e-9);
/// assert!(r.eta_squared.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct AnovaBuilder {
    alpha: f64,
    compute_effect_size: bool,
}

impl Default for AnovaBuilder {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            compute_effect_size: false,
        }
    }
}

impl AnovaBuilder {
    /// Create a builder with defaults: α=0.05, no effect size.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the significance level α.
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Request computation of η² (eta-squared) effect size.
    pub fn with_effect_size(mut self) -> Self {
        self.compute_effect_size = true;
        self
    }

    /// Run a one-way ANOVA on `groups` (each group is a `Vec<f64>`).
    pub fn one_way(&self, groups: &[Vec<f64>]) -> StatsResult<AnovaBuilderResult> {
        if !(self.alpha > 0.0 && self.alpha < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        // Collect references for the underlying function.
        let refs: Vec<&[f64]> = groups.iter().map(|v| v.as_slice()).collect();
        let raw = one_way_anova(&refs)?;

        let eta_squared = if self.compute_effect_size {
            let ss_total = raw.ss_between + raw.ss_within;
            if ss_total > 0.0 {
                Some(raw.ss_between / ss_total)
            } else {
                Some(0.0)
            }
        } else {
            None
        };

        Ok(AnovaBuilderResult {
            f_stat: raw.f_statistic,
            p_value: raw.p_value,
            df_between: raw.df_between,
            df_within: raw.df_within,
            significant: raw.p_value < self.alpha,
            eta_squared,
            raw,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BootstrapBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Bootstrap confidence interval estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapCiMethod {
    /// Simple percentile method.
    Percentile,
    /// Bias-corrected and accelerated (BCa) method.
    Bca,
}

/// Fluent builder for bootstrap confidence intervals.
///
/// Uses the workspace [`LcgRng`] seeded at construction time so results are
/// reproducible.
#[derive(Debug, Clone)]
pub struct BootstrapBuilder {
    pub n_resamples: usize,
    pub confidence: f64,
    pub seed: u64,
    pub ci_method: BootstrapCiMethod,
}

impl Default for BootstrapBuilder {
    fn default() -> Self {
        Self {
            n_resamples: 2000,
            confidence: 0.95,
            seed: 42,
            ci_method: BootstrapCiMethod::Percentile,
        }
    }
}

impl BootstrapBuilder {
    /// Create a builder with defaults: n_resamples=2000, confidence=0.95,
    /// seed=42, Percentile CI.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of bootstrap resamples.
    pub fn n_resamples(mut self, n: usize) -> Self {
        self.n_resamples = n;
        self
    }

    /// Set the confidence level (e.g. 0.95 for 95 %).
    pub fn confidence(mut self, c: f64) -> Self {
        self.confidence = c;
        self
    }

    /// Set the RNG seed for reproducibility.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Use the percentile CI method (default).
    pub fn percentile(mut self) -> Self {
        self.ci_method = BootstrapCiMethod::Percentile;
        self
    }

    /// Use the bias-corrected and accelerated (BCa) CI method.
    pub fn bca(mut self) -> Self {
        self.ci_method = BootstrapCiMethod::Bca;
        self
    }

    // ── internal ────────────────────────────────────────────────────────────

    fn validate(&self) -> StatsResult<()> {
        if self.n_resamples == 0 {
            return Err(StatsError::InvalidParameter {
                name: "n_resamples".into(),
                reason: "must be > 0".into(),
            });
        }
        if !(self.confidence > 0.0 && self.confidence < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "confidence".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        Ok(())
    }

    /// Run `n_resamples` bootstrap resamples applying `stat_fn` each time and
    /// return the two-sided confidence interval using the configured method.
    fn run_bootstrap<F>(&self, data: &[f64], stat_fn: F) -> StatsResult<(f64, f64)>
    where
        F: Fn(&[f64]) -> f64,
    {
        self.validate()?;
        if data.is_empty() {
            return Err(StatsError::EmptyInput);
        }
        let n = data.len();
        let mut rng = LcgRng::new(self.seed);
        let mut sample = vec![0.0_f64; n];
        let mut replicates = Vec::with_capacity(self.n_resamples);

        for _ in 0..self.n_resamples {
            for slot in sample.iter_mut() {
                *slot = data[rng.next_usize(n)];
            }
            replicates.push(stat_fn(&sample));
        }

        let alpha = 1.0 - self.confidence;
        match self.ci_method {
            BootstrapCiMethod::Percentile => {
                let lo = quantile(&replicates, alpha / 2.0)?;
                let hi = quantile(&replicates, 1.0 - alpha / 2.0)?;
                Ok((lo, hi))
            }
            BootstrapCiMethod::Bca => {
                let theta_hat = stat_fn(data);
                self.bca_ci(data, theta_hat, &replicates, &stat_fn, alpha)
            }
        }
    }

    /// BCa confidence interval computation.
    ///
    /// Uses the Efron (1987) bias-correction `z0` and acceleration `a` factors.
    fn bca_ci<F>(
        &self,
        data: &[f64],
        theta_hat: f64,
        replicates: &[f64],
        stat_fn: &F,
        alpha: f64,
    ) -> StatsResult<(f64, f64)>
    where
        F: Fn(&[f64]) -> f64,
    {
        let b = replicates.len() as f64;
        // Bias correction z0: proportion of boot stats < theta_hat.
        let below = replicates.iter().filter(|&&v| v < theta_hat).count() as f64;
        let p0 = (below / b).clamp(1e-10, 1.0 - 1e-10);
        let z0 = probit(p0);

        // Acceleration a: jackknife skewness of influence function.
        let n = data.len();
        let jk_stats: Vec<f64> = (0..n)
            .map(|i| {
                let jk: Vec<f64> = data
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &v)| v)
                    .collect();
                stat_fn(&jk)
            })
            .collect();
        let jk_mean = jk_stats.iter().sum::<f64>() / n as f64;
        let num: f64 = jk_stats.iter().map(|&v| (jk_mean - v).powi(3)).sum();
        let den_sq: f64 = jk_stats.iter().map(|&v| (jk_mean - v).powi(2)).sum();
        let a = if den_sq > 1e-30 {
            num / (6.0 * den_sq.powf(1.5))
        } else {
            0.0
        };

        // Adjusted quantiles.
        let z_lo = standard_normal_ppf(alpha / 2.0);
        let z_hi = standard_normal_ppf(1.0 - alpha / 2.0);

        let adj_lo = norm_cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)));
        let adj_hi = norm_cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)));

        let adj_lo = adj_lo.clamp(1e-6, 1.0 - 1e-6);
        let adj_hi = adj_hi.clamp(1e-6, 1.0 - 1e-6);

        let lo = quantile(replicates, adj_lo)?;
        let hi = quantile(replicates, adj_hi)?;
        Ok((lo, hi))
    }

    // ── public API ──────────────────────────────────────────────────────────

    /// Bootstrap confidence interval for the **sample mean**.
    pub fn mean_ci(&self, data: &[f64]) -> StatsResult<(f64, f64)> {
        self.run_bootstrap(data, |s| s.iter().sum::<f64>() / s.len() as f64)
    }

    /// Bootstrap confidence interval for the **sample median**.
    pub fn median_ci(&self, data: &[f64]) -> StatsResult<(f64, f64)> {
        self.run_bootstrap(data, |s| {
            let mut tmp = s.to_vec();
            tmp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = tmp.len();
            if n % 2 == 0 {
                (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0
            } else {
                tmp[n / 2]
            }
        })
    }

    /// Bootstrap confidence interval for an **arbitrary statistic** `stat_fn`.
    pub fn statistic_ci<F>(&self, data: &[f64], stat_fn: F) -> StatsResult<(f64, f64)>
    where
        F: Fn(&[f64]) -> f64,
    {
        self.run_bootstrap(data, stat_fn)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private utilities (no external deps)
// ─────────────────────────────────────────────────────────────────────────────

/// Standard normal CDF via error function approximation.
#[inline]
fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Probit function (inverse of standard normal CDF) via rational approximation.
/// Adapted from Peter Acklam's algorithm (2002).
#[inline]
fn probit(p: f64) -> f64 {
    standard_normal_ppf(p)
}

/// Rational approximation for the standard normal PPF (Acklam 2002).
fn standard_normal_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Coefficients.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Approximation of erf(x) via Horner's method (Abramowitz & Stegun 7.1.26).
#[inline]
fn erf_approx(x: f64) -> f64 {
    let neg = x < 0.0;
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (0.254_829_592
            + (-0.284_496_736 + (1.421_413_741 + (-1.453_152_027 + 1.061_405_429 * t) * t) * t)
                * t)
            * t
            * (-x * x).exp();
    if neg { -y } else { y }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TTestBuilder ────────────────────────────────────────────────────────

    #[test]
    fn one_sample_default_builder_runs() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = TTestBuilder::new().one_sample(&data).expect("ok");
        assert!(r.statistic.is_finite());
        assert!(r.df > 0.0);
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn one_sample_null_mean_shift_changes_t() {
        let data = [10.0, 11.0, 12.0, 13.0, 14.0];
        let r_zero = TTestBuilder::new()
            .null_mean(0.0)
            .one_sample(&data)
            .expect("ok");
        let r_twelve = TTestBuilder::new()
            .null_mean(12.0)
            .one_sample(&data)
            .expect("ok");
        // With null_mean = 12 (the actual mean), t should be very small.
        assert!(r_zero.statistic.abs() > r_twelve.statistic.abs());
        assert!(r_twelve.statistic.abs() < 1e-12);
    }

    #[test]
    fn two_sample_student_same_as_direct_call() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [3.0, 4.0, 5.0, 6.0, 7.0];
        let builder_result = TTestBuilder::new()
            .equal_variance()
            .two_sample(&a, &b)
            .expect("ok");
        let direct = two_sample_t(&a, &b).expect("ok");
        assert!((builder_result.statistic - direct.t_statistic).abs() < 1e-12);
        assert!((builder_result.df - direct.df).abs() < 1e-12);
    }

    #[test]
    fn welch_vs_student_differ_on_heteroscedastic_data() {
        // Group with high variance vs group with low variance.
        let a = [1.0, 1.0, 1.0, 1.0, 50.0]; // high variance
        let b = [2.0, 2.1, 1.9, 2.0, 2.0]; // low variance
        let student = TTestBuilder::new()
            .equal_variance()
            .two_sample(&a, &b)
            .expect("ok");
        let welch = TTestBuilder::new().welch().two_sample(&a, &b).expect("ok");
        // Different DFs should produce different results.
        assert!((student.df - welch.df).abs() > 0.5);
    }

    #[test]
    fn significance_reflects_alpha() {
        // Mean clearly different from 100.
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r_strict = TTestBuilder::new()
            .alpha(0.001)
            .null_mean(100.0)
            .one_sample(&data)
            .expect("ok");
        let r_loose = TTestBuilder::new()
            .alpha(0.999)
            .null_mean(100.0)
            .one_sample(&data)
            .expect("ok");
        // With alpha=0.999 the test is almost always significant.
        assert!(r_loose.significant);
        // significance flag equals p < alpha.
        assert_eq!(r_strict.significant, r_strict.p_value < 0.001);
    }

    #[test]
    fn alpha_zero_point_zero_one_changes_threshold() {
        // Verify that the significance flag tracks the configured alpha.
        // Use a large dataset so p is very small and significant at both α levels.
        let data: Vec<f64> = (1..=30).map(|v| v as f64).collect(); // mean=15.5, far from 0
        let r_05 = TTestBuilder::new()
            .alpha(0.05)
            .null_mean(0.0)
            .one_sample(&data)
            .expect("ok");
        let r_01 = TTestBuilder::new()
            .alpha(0.01)
            .null_mean(0.0)
            .one_sample(&data)
            .expect("ok");
        // Both should be significant with n=30.
        assert!(r_05.significant, "α=0.05 should be significant");
        assert!(r_01.significant, "α=0.01 should be significant");
        // CIs with alpha=0.01 (99 % CI) should be wider than 95 % CI.
        let width_05 = r_05.ci_upper - r_05.ci_lower;
        let width_01 = r_01.ci_upper - r_01.ci_lower;
        assert!(
            width_01 > width_05,
            "99% CI ({width_01}) should be wider than 95% CI ({width_05})"
        );
        // Changing alpha changes the significance threshold.
        assert_eq!(r_05.significant, r_05.p_value < 0.05);
        assert_eq!(r_01.significant, r_01.p_value < 0.01);
    }

    #[test]
    fn ci_contains_true_mean() {
        let data = [10.0, 11.0, 12.0, 13.0, 14.0]; // true mean = 12
        let r = TTestBuilder::new()
            .null_mean(0.0)
            .one_sample(&data)
            .expect("ok");
        // CI is for the mean difference (xbar - mu0) = xbar - 0 = xbar.
        // True difference is 12.
        assert!(r.ci_lower < 12.0 && r.ci_upper > 12.0);
    }

    #[test]
    fn tailed_p_values_differ() {
        let a = [1.0, 2.0, 3.0];
        let b = [5.0, 6.0, 7.0];
        let two = TTestBuilder::new()
            .two_tailed()
            .two_sample(&a, &b)
            .expect("ok");
        let left = TTestBuilder::new()
            .left_tailed()
            .two_sample(&a, &b)
            .expect("ok");
        let right = TTestBuilder::new()
            .right_tailed()
            .two_sample(&a, &b)
            .expect("ok");
        // p_left + p_right should sum to 1.0 for continuous distributions.
        assert!((left.p_value + right.p_value - 1.0).abs() < 1e-10);
        // two-sided p ≈ 2 * min(left, right).
        assert!((two.p_value - 2.0 * left.p_value.min(right.p_value)).abs() < 1e-6);
    }

    #[test]
    fn paired_builder_symmetric() {
        let before = [5.0, 6.0, 7.0, 8.0, 9.0];
        let after = [5.0, 6.0, 7.0, 8.0, 9.0]; // identical → zero variance error
        let r = TTestBuilder::new().paired(&before, &after);
        // Zero variance in differences → error (as expected).
        assert!(r.is_err());
    }

    #[test]
    fn paired_builder_with_shift() {
        // Differences must have non-zero variance so the t-test is valid.
        // Differences: 3, 1, 4, 1, 5, 2 — mean ≈ 2.67, variance > 0.
        let before = [10.0, 8.0, 12.0, 7.0, 15.0, 9.0];
        let after = [7.0, 7.0, 8.0, 6.0, 10.0, 7.0];
        let r = TTestBuilder::new().paired(&before, &after).expect("ok");
        // Mean difference (≈2.67) should be significantly different from 0.
        assert!(
            r.significant,
            "expected significant paired result, p={}",
            r.p_value
        );
    }

    #[test]
    fn builder_chaining_compiles_and_runs() {
        let data = [3.0, 4.0, 5.0, 6.0, 7.0];
        let r = TTestBuilder::new()
            .welch()
            .left_tailed()
            .alpha(0.1)
            .null_mean(5.0)
            .one_sample(&data);
        assert!(r.is_ok());
    }

    // ── AnovaBuilder ────────────────────────────────────────────────────────

    #[test]
    fn anova_canonical_case_f12() {
        let groups = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 4.0, 5.0],
            vec![5.0, 6.0, 7.0],
        ];
        let r = AnovaBuilder::new().one_way(&groups).expect("ok");
        assert!(
            (r.f_stat - 12.0).abs() < 1e-9,
            "expected F=12, got {}",
            r.f_stat
        );
        assert!(r.p_value < 0.01);
        assert!(r.significant);
    }

    #[test]
    fn anova_with_effect_size_returns_eta_squared() {
        let groups = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 4.0, 5.0],
            vec![5.0, 6.0, 7.0],
        ];
        let r = AnovaBuilder::new()
            .with_effect_size()
            .one_way(&groups)
            .expect("ok");
        let eta = r.eta_squared.expect("eta_squared should be present");
        // Grand mean=4, group means 2,4,6.
        // SS_between = 3*(2-4)²+3*(4-4)²+3*(6-4)² = 12+0+12 = 24
        // SS_within  = (1-2)²+(2-2)²+(3-2)²+(3-4)²+(4-4)²+(5-4)²+(5-6)²+(6-6)²+(7-6)² = 2+2+2 = 6
        // SS_total   = 30 → eta² = 24/30 = 0.8
        assert!((eta - 24.0 / 30.0).abs() < 1e-9, "eta²={eta}");
    }

    #[test]
    fn anova_without_effect_size_is_none() {
        let groups = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let r = AnovaBuilder::new().one_way(&groups).expect("ok");
        assert!(r.eta_squared.is_none());
    }

    #[test]
    fn anova_identical_groups_not_significant() {
        let groups = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        let r = AnovaBuilder::new()
            .alpha(0.05)
            .one_way(&groups)
            .expect("ok");
        assert!(!r.significant);
        assert!(r.p_value > 0.99);
    }

    // ── BootstrapBuilder ────────────────────────────────────────────────────

    #[test]
    fn bootstrap_mean_ci_covers_true_mean() {
        // Data: 1..=30, true mean = 15.5.
        let data: Vec<f64> = (1..=30).map(|v| v as f64).collect();
        let (lo, hi) = BootstrapBuilder::new()
            .n_resamples(2000)
            .confidence(0.95)
            .seed(7)
            .mean_ci(&data)
            .expect("ok");
        assert!(lo < 15.5 && hi > 15.5, "CI [{lo},{hi}] does not cover 15.5");
    }

    #[test]
    fn bootstrap_median_ci_covers_true_median() {
        let data: Vec<f64> = (1..=31).map(|v| v as f64).collect(); // true median = 16
        let (lo, hi) = BootstrapBuilder::new()
            .n_resamples(2000)
            .seed(99)
            .median_ci(&data)
            .expect("ok");
        assert!(lo < 16.0 && hi > 16.0, "CI [{lo},{hi}] does not cover 16");
    }

    #[test]
    fn bootstrap_statistic_ci_custom_fn() {
        let data: Vec<f64> = (1..=20).map(|v| v as f64).collect();
        // Max statistic — should be near 20.
        let (lo, hi) = BootstrapBuilder::new()
            .n_resamples(1000)
            .seed(13)
            .statistic_ci(&data, |s| {
                s.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            })
            .expect("ok");
        assert!(hi >= 19.0 && lo > 0.0, "max CI [{lo},{hi}]");
    }

    #[test]
    fn bootstrap_bca_method_runs() {
        let data: Vec<f64> = (1..=20).map(|v| v as f64).collect(); // mean = 10.5
        let (lo, hi) = BootstrapBuilder::new()
            .n_resamples(1000)
            .seed(17)
            .bca()
            .mean_ci(&data)
            .expect("ok");
        // BCa CI should still bracket the true mean.
        assert!(
            lo < 10.5 && hi > 10.5,
            "BCa CI [{lo},{hi}] does not cover 10.5"
        );
    }

    #[test]
    fn bootstrap_percentile_vs_bca_differ() {
        let data: Vec<f64> = (1..=30).map(|v| v as f64).collect();
        let (lo_p, hi_p) = BootstrapBuilder::new()
            .n_resamples(500)
            .seed(1)
            .percentile()
            .mean_ci(&data)
            .expect("ok");
        let (lo_b, hi_b) = BootstrapBuilder::new()
            .n_resamples(500)
            .seed(1)
            .bca()
            .mean_ci(&data)
            .expect("ok");
        // Both should be finite; they need not be identical.
        assert!(lo_p.is_finite() && hi_p.is_finite());
        assert!(lo_b.is_finite() && hi_b.is_finite());
    }

    #[test]
    fn bootstrap_empty_data_errors() {
        let r = BootstrapBuilder::new().mean_ci(&[]);
        assert!(r.is_err());
    }

    #[test]
    fn bootstrap_n_resamples_zero_errors() {
        let data = [1.0, 2.0, 3.0];
        let r = BootstrapBuilder::new().n_resamples(0).mean_ci(&data);
        assert!(r.is_err());
    }

    #[test]
    fn bootstrap_confidence_boundary_errors() {
        let data = [1.0, 2.0, 3.0];
        let r = BootstrapBuilder::new().confidence(1.5).mean_ci(&data);
        assert!(r.is_err());
    }
}

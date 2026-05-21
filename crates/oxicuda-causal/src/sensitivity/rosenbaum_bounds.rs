//! Rosenbaum bounds for sensitivity analysis on matched-pair data.
//!
//! Reference: Rosenbaum, P. R. (1987). "Sensitivity analysis for certain
//! permutation inferences in matched observational studies."
//! *Biometrika*, 74(1), 13-26.  See also Rosenbaum, P. R. (2002).
//! *Observational Studies* (2nd ed.), Springer, Chapter 4.
//!
//! # Algorithm — Wilcoxon signed-rank under bias Γ
//!
//! Consider `n` matched pairs of treated / control observations with the
//! within-pair outcome differences `dᵢ = Y_i^T − Y_i^C`.  Under the sharp
//! null `H₀: Y_i^T = Y_i^C` and *random* pair assignment, the sign of each
//! `dᵢ` is symmetric Bernoulli(1/2).  Rosenbaum (1987) introduced a
//! sensitivity parameter Γ ≥ 1 that bounds the unmeasured-confounder
//! odds ratio between two paired units: under bias Γ the sign of pair `i`
//! is Bernoulli(p) with `p` somewhere in `[1/(1+Γ), Γ/(1+Γ)]`.
//!
//! For the Wilcoxon signed-rank statistic `T⁺ = Σᵢ rankᵢ · 1[dᵢ > 0]`,
//! the *worst-case upper p-value* over the null distribution under bias Γ
//! is obtained by setting every Bernoulli probability to its upper bound
//! `p_high = Γ/(1+Γ)`; the *best-case lower p-value* uses the lower bound
//! `p_low = 1/(1+Γ)`.  For both, the asymptotic mean and variance of `T⁺`
//! over the independent sum of weighted Bernoullis are
//!
//! ```text
//!   μ(p) = p · Σᵢ rankᵢ,
//!   σ²(p) = p · (1 − p) · Σᵢ rankᵢ².
//! ```
//!
//! The normal-approximation p-value for the *observed* `T⁺_obs` is
//!
//! ```text
//!   P(p) = 1 − Φ( (T⁺_obs − μ(p) − 0.5) / σ(p) ),
//! ```
//!
//! with the standard `−0.5` continuity correction.  For pair counts
//! `n < 20` we instead enumerate the `2^n` sign assignments exactly and
//! sum the Bernoulli-weighted indicator that the realised `T⁺ ≥ T⁺_obs`.
//!
//! Zero-magnitude differences are dropped (Pratt's recommended rule);
//! tied `|dᵢ|` values share the average of the tied integer ranks.
//!
//! # Critical Γ
//!
//! The *critical* Γ at level α is the smallest Γ ≥ 1 for which the
//! worst-case upper p-value crosses α.  We locate it by bisection on
//! `[1, 20]` (large enough for nearly all practical observational
//! studies — see Rosenbaum 2002 §4).  If the upper p-value already
//! exceeds α at Γ = 1 the critical value is reported as `1.0`.

use crate::error::{CausalError, CausalResult};

/// Configuration for [`RosenbaumBounds::wilcoxon_signed_rank`].
#[derive(Debug, Clone)]
pub struct RosenbaumConfig {
    /// Sensitivity-parameter grid; every entry must satisfy Γ ≥ 1.
    pub gamma_grid: Vec<f64>,
    /// Two-sided significance level on `(0, 1)`; only used as a reference
    /// in downstream reporting (the bounds themselves are independent of
    /// α, but the documented sensitivity-aware decision rule rejects when
    /// the *upper* p-value falls below α).
    pub alpha: f64,
}

impl Default for RosenbaumConfig {
    fn default() -> Self {
        Self {
            gamma_grid: vec![1.0, 1.25, 1.5, 1.75, 2.0],
            alpha: 0.05,
        }
    }
}

/// A single (Γ, p-bound) result row produced by
/// [`RosenbaumBounds::wilcoxon_signed_rank`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RosenbaumResult {
    /// Sensitivity parameter for this row.
    pub gamma: f64,
    /// Best-case (lower) one-sided p-value over Bernoulli sign
    /// probabilities `1/(1+Γ)`.
    pub p_lower: f64,
    /// Worst-case (upper) one-sided p-value over Bernoulli sign
    /// probabilities `Γ/(1+Γ)`.
    pub p_upper: f64,
}

/// Zero-sized handle for the Rosenbaum-bounds entry points.
pub struct RosenbaumBounds;

impl RosenbaumBounds {
    /// Compute lower / upper one-sided p-value bounds for the Wilcoxon
    /// signed-rank test under bias Γ for each Γ in `cfg.gamma_grid`.
    ///
    /// # Errors
    /// - [`CausalError::EmptyInput`] if `differences` is empty or every
    ///   pair has a zero magnitude difference.
    /// - [`CausalError::IncompatibleData`] if `cfg.alpha ∉ (0, 1)`,
    ///   `cfg.gamma_grid` is empty, or any entry of `cfg.gamma_grid` is
    ///   `< 1.0` or non-finite, or any `differences[i]` is non-finite.
    pub fn wilcoxon_signed_rank(
        differences: &[f64],
        cfg: &RosenbaumConfig,
    ) -> CausalResult<Vec<RosenbaumResult>> {
        validate_inputs(differences, cfg)?;

        let (ranks, signs) = signed_rank(differences)?;
        let n_eff = ranks.len();
        if n_eff == 0 {
            return Err(CausalError::EmptyInput);
        }

        let t_obs = observed_t_plus(&ranks, &signs);
        let sum_rank: f64 = ranks.iter().sum();
        let sum_rank_sq: f64 = ranks.iter().map(|r| r * r).sum();

        let mut out = Vec::with_capacity(cfg.gamma_grid.len());
        for &gamma in &cfg.gamma_grid {
            let p_high = gamma / (1.0 + gamma);
            let p_low = 1.0 / (1.0 + gamma);
            let p_upper = p_value_for_bernoulli(t_obs, &ranks, p_high, sum_rank, sum_rank_sq);
            let p_lower = p_value_for_bernoulli(t_obs, &ranks, p_low, sum_rank, sum_rank_sq);
            out.push(RosenbaumResult {
                gamma,
                p_lower,
                p_upper,
            });
        }
        Ok(out)
    }

    /// Smallest Γ ≥ 1 at which the worst-case (upper) Wilcoxon signed-rank
    /// p-value first crosses `alpha`.  Located by bisection on `[1, 20]`.
    ///
    /// Returns `1.0` when the upper bound at Γ = 1 already lies above α
    /// (i.e. the result is not even significant under no bias).  Returns
    /// the upper search endpoint when the test remains significant up to
    /// Γ = 20.
    ///
    /// # Errors
    /// As for [`RosenbaumBounds::wilcoxon_signed_rank`] with a single-entry
    /// grid.
    pub fn critical_gamma(differences: &[f64], alpha: f64) -> CausalResult<f64> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(CausalError::IncompatibleData);
        }
        let cfg_check = RosenbaumConfig {
            gamma_grid: vec![1.0],
            alpha,
        };
        let baseline = Self::wilcoxon_signed_rank(differences, &cfg_check)?;
        // At Γ=1 the upper bound *is* the unadjusted p-value.
        if baseline[0].p_upper > alpha {
            return Ok(1.0);
        }

        let (ranks, signs) = signed_rank(differences)?;
        let t_obs = observed_t_plus(&ranks, &signs);
        let sum_rank: f64 = ranks.iter().sum();
        let sum_rank_sq: f64 = ranks.iter().map(|r| r * r).sum();

        let upper_at = |gamma: f64| -> f64 {
            let p = gamma / (1.0 + gamma);
            p_value_for_bernoulli(t_obs, &ranks, p, sum_rank, sum_rank_sq)
        };

        // If even Γ = 20 keeps the upper bound below α, return 20.
        let g_hi_start = 20.0_f64;
        if upper_at(g_hi_start) < alpha {
            return Ok(g_hi_start);
        }

        // Bisection: ensure lower < alpha < upper.
        let mut lo = 1.0_f64;
        let mut hi = g_hi_start;
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let p_mid = upper_at(mid);
            if p_mid < alpha {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo) < 1e-6 {
                break;
            }
        }
        Ok(0.5 * (lo + hi))
    }
}

// =====================================================================
// Input validation
// =====================================================================

fn validate_inputs(differences: &[f64], cfg: &RosenbaumConfig) -> CausalResult<()> {
    if differences.is_empty() {
        return Err(CausalError::EmptyInput);
    }
    for &d in differences {
        if !d.is_finite() {
            return Err(CausalError::IncompatibleData);
        }
    }
    if !cfg.alpha.is_finite() || cfg.alpha <= 0.0 || cfg.alpha >= 1.0 {
        return Err(CausalError::IncompatibleData);
    }
    if cfg.gamma_grid.is_empty() {
        return Err(CausalError::IncompatibleData);
    }
    for &g in &cfg.gamma_grid {
        if !g.is_finite() || g < 1.0 {
            return Err(CausalError::IncompatibleData);
        }
    }
    Ok(())
}

// =====================================================================
// Wilcoxon signed-rank machinery
// =====================================================================

/// Compute average ranks for `|dᵢ|` (dropping zeros) and the sign of each
/// non-zero `dᵢ` (`+1.0` if positive, `0.0` if negative).
fn signed_rank(differences: &[f64]) -> CausalResult<(Vec<f64>, Vec<f64>)> {
    // Keep only non-zero pairs (Pratt's rule).
    let mut absolutes: Vec<(f64, f64)> = differences
        .iter()
        .filter(|&&d| d != 0.0)
        .map(|&d| (d.abs(), if d > 0.0 { 1.0 } else { 0.0 }))
        .collect();
    if absolutes.is_empty() {
        return Err(CausalError::EmptyInput);
    }

    // Sort ascending by absolute value, tracking original sign.
    absolutes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = absolutes.len();
    let mut ranks = vec![0.0_f64; n];
    let signs: Vec<f64> = absolutes.iter().map(|(_, s)| *s).collect();

    // Assign average ranks for tied |d| groups.
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n
            && (absolutes[j].0 - absolutes[i].0).abs() <= f64::EPSILON * absolutes[i].0.max(1.0)
        {
            j += 1;
        }
        // Indices in tied group: [i, j); integer ranks (1-based) [i+1, j+1).
        let lo = (i + 1) as f64;
        let hi = j as f64;
        let avg = 0.5 * (lo + hi);
        for slot in ranks.iter_mut().take(j).skip(i) {
            *slot = avg;
        }
        i = j;
    }
    Ok((ranks, signs))
}

fn observed_t_plus(ranks: &[f64], signs: &[f64]) -> f64 {
    ranks
        .iter()
        .zip(signs.iter())
        .map(|(r, s)| r * s)
        .sum::<f64>()
}

/// Right-tail p-value under the independent-Bernoulli model where each
/// `signs[i]` is independently `1` with probability `p`.  Uses exact
/// enumeration for `n < 20` and a continuity-corrected normal
/// approximation otherwise.
fn p_value_for_bernoulli(
    t_obs: f64,
    ranks: &[f64],
    p: f64,
    sum_rank: f64,
    sum_rank_sq: f64,
) -> f64 {
    let n = ranks.len();
    if n == 0 {
        return 1.0;
    }
    if n < 20 {
        return p_value_exact(t_obs, ranks, p);
    }
    let mean = p * sum_rank;
    let var = p * (1.0 - p) * sum_rank_sq;
    if var <= 0.0 {
        // Degenerate case: every sign is fixed (p ∈ {0, 1}).
        return if t_obs <= mean { 1.0 } else { 0.0 };
    }
    let sigma = var.sqrt();
    let z = (t_obs - mean - 0.5) / sigma;
    1.0 - normal_cdf(z)
}

/// Exact right-tail p-value `P(T⁺ ≥ T⁺_obs)` under independent
/// `Bernoulli(p)` signs, enumerating all `2^n` sign assignments.  Only
/// invoked for `n < 20`.
fn p_value_exact(t_obs: f64, ranks: &[f64], p: f64) -> f64 {
    let n = ranks.len();
    // n ≤ 19 ⇒ 2^n ≤ 524288.
    let total: u64 = 1u64 << n;
    let mut acc = 0.0_f64;
    for mask in 0..total {
        // Bernoulli weight: ∏ᵢ p^{bᵢ} · (1−p)^{1−bᵢ}.
        let mut weight = 1.0_f64;
        let mut t_plus = 0.0_f64;
        for (i, &rank) in ranks.iter().enumerate() {
            let bit = (mask >> i) & 1;
            if bit == 1 {
                weight *= p;
                t_plus += rank;
            } else {
                weight *= 1.0 - p;
            }
        }
        if t_plus >= t_obs {
            acc += weight;
        }
    }
    acc.clamp(0.0, 1.0)
}

/// Standard normal CDF Φ(z) via Abramowitz & Stegun 7.1.26 polynomial
/// approximation of `erf`; max relative error ≈ 1.5·10⁻⁷.
fn normal_cdf(z: f64) -> f64 {
    if !z.is_finite() {
        if z > 0.0 {
            return 1.0;
        }
        return 0.0;
    }
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// `erf(x)` via the rational approximation in Abramowitz & Stegun 7.1.26.
fn erf(x: f64) -> f64 {
    if !x.is_finite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let p = 0.327_591_1_f64;
    let a1 = 0.254_829_592_f64;
    let a2 = -0.284_496_736_f64;
    let a3 = 1.421_413_741_f64;
    let a4 = -1.453_152_027_f64;
    let a5 = 1.061_405_429_f64;
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

// =====================================================================
// Helpers re-exported for sibling test module.
// =====================================================================

#[cfg(test)]
#[inline]
pub(super) fn normal_cdf_for_tests(z: f64) -> f64 {
    normal_cdf(z)
}

#[cfg(test)]
#[inline]
pub(super) fn signed_rank_for_tests(differences: &[f64]) -> CausalResult<(Vec<f64>, Vec<f64>)> {
    signed_rank(differences)
}

// tests live in `rosenbaum_bounds_tests.rs` (registered from `sensitivity/mod.rs`).

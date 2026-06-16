//! Tweedie Generalised Linear Model (compound Poisson–Gamma EDM) fitted by IRLS.
//!
//! The Tweedie family is the subclass of exponential-dispersion models (EDMs)
//! whose variance is a power of the mean:
//!
//! ```text
//! Var(Y) = φ · V(μ),   V(μ) = μ^p
//! ```
//!
//! The **power index** `p` (the "Tweedie index") selects the member:
//!
//! | range / value | distribution                                    |
//! |---------------|-------------------------------------------------|
//! | `p = 0`       | Gaussian (Normal)                               |
//! | `0 < p < 1`   | (does not exist as an EDM)                       |
//! | `p = 1`       | Poisson (φ = 1)                                  |
//! | `1 < p < 2`   | **compound Poisson–Gamma** (mass at 0, cont. >0)|
//! | `p = 2`       | Gamma                                           |
//! | `p = 3`       | inverse-Gaussian                                |
//!
//! This module focuses on the **`1 < p < 2`** regime — the only EDM that mixes
//! an exact point mass at `y = 0` with a continuous positive density, making it
//! the canonical model for insurance pure-premium, rainfall and other
//! zero-inflated non-negative data. A `log` link is used by default so that the
//! mean `μ = exp(η)` is strictly positive.
//!
//! # Algorithm — IRLS (Fisher scoring)
//!
//! With link `g`, linear predictor `η = Xβ`, mean `μ = g⁻¹(η)`:
//!
//! 1. working response `z = η + (y − μ) · g'(μ)`   where `g'(μ) = ∂η/∂μ`;
//! 2. working weight  `w = (∂μ/∂η)² / V(μ)`   (the Fisher weight, since `φ`
//!    is a nuisance scale that cancels in the score equations);
//! 3. solve the weighted least-squares system `(Xᵀ W X) β = Xᵀ W z`;
//! 4. iterate until `‖Δβ‖₂ < tol`.
//!
//! For the `log` link `g(μ)=ln μ`, `∂μ/∂η = μ` so `z = η + (y−μ)/μ` and
//! `w = μ² / μ^p = μ^{2−p}`.
//!
//! # Unit deviance
//!
//! The Tweedie unit deviance (Jørgensen, 1987) for `p ∉ {0,1,2}` is
//!
//! ```text
//! d(y,μ) = 2 [ y · (y^{1−p} − μ^{1−p})/(1−p) − (y^{2−p} − μ^{2−p})/(2−p) ].
//! ```
//!
//! For `1 < p < 2` it is **finite at `y = 0`**, where it collapses to
//! `d(0,μ) = 2 μ^{2−p}/(2−p)`, because `y·y^{1−p} = y^{2−p} → 0` as `y → 0⁺`.
//! As `p → 1⁺` it approaches the Poisson deviance and as `p → 2⁻` the Gamma
//! deviance; both limiting forms are used directly when `p` is within a small
//! tolerance of `1` or `2` to avoid catastrophic cancellation.
//!
//! # References
//! - Jørgensen, B. (1987). "Exponential Dispersion Models". *JRSS-B* 49(2):127–162.
//! - Jørgensen, B. (1997). *The Theory of Dispersion Models*. Chapman & Hall.
//! - Dunn, P.K. & Smyth, G.K. (2005). "Series evaluation of Tweedie exponential
//!   dispersion model densities". *Statistics and Computing* 15:267–280.
//! - Smyth, G.K. (1996). "Regression analysis of quantity data with exact zeros".

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::matrix_inverse_lu;

// ─────────────────────────────────────────────────────────────────────────────
// Link
// ─────────────────────────────────────────────────────────────────────────────

/// Link function for the Tweedie GLM. `g(μ) = η`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TweedieLink {
    /// `g(μ) = ln μ`  (default; guarantees `μ > 0`).
    Log,
    /// `g(μ) = μ`  (identity; only sensible when `μ` is guaranteed positive).
    Identity,
}

impl TweedieLink {
    /// `μ = g⁻¹(η)`.
    #[inline]
    fn inverse(self, eta: f64) -> f64 {
        match self {
            TweedieLink::Log => eta.exp(),
            TweedieLink::Identity => eta,
        }
    }

    /// `g(μ) = η`.
    #[inline]
    fn forward(self, mu: f64) -> f64 {
        match self {
            TweedieLink::Log => mu.max(f64::MIN_POSITIVE).ln(),
            TweedieLink::Identity => mu,
        }
    }

    /// `∂μ/∂η` expressed in terms of `μ`.
    #[inline]
    fn dmu_deta(self, mu: f64) -> f64 {
        match self {
            TweedieLink::Log => mu,
            TweedieLink::Identity => 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`tweedie_fit`].
#[derive(Debug, Clone)]
pub struct TweedieConfig {
    /// Tweedie power index `p`. Must satisfy `1 < p < 2` for the compound
    /// Poisson–Gamma regime supported here (the deviance is then finite at 0).
    pub power: f64,
    /// Link function (default [`TweedieLink::Log`]).
    pub link: TweedieLink,
    /// Maximum number of IRLS iterations (default 100).
    pub max_iter: usize,
    /// Convergence tolerance on `‖Δβ‖₂` (default 1e-10).
    pub tol: f64,
    /// Prepend an intercept column of ones (default `true`).
    pub intercept: bool,
}

impl Default for TweedieConfig {
    fn default() -> Self {
        Self {
            power: 1.5,
            link: TweedieLink::Log,
            max_iter: 100,
            tol: 1e-10,
            intercept: true,
        }
    }
}

impl TweedieConfig {
    /// Convenience constructor fixing the power index, log link, defaults elsewhere.
    #[must_use]
    pub fn with_power(power: f64) -> Self {
        Self {
            power,
            ..Default::default()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted model
// ─────────────────────────────────────────────────────────────────────────────

/// A fitted Tweedie GLM.
#[derive(Debug, Clone)]
pub struct TweedieFit {
    /// Estimated coefficients (intercept first when `cfg.intercept == true`).
    pub coefficients: Vec<f64>,
    /// Fitted means `μ̂ = g⁻¹(Xβ̂)`, length `n_samples` (all strictly positive).
    pub fitted_values: Vec<f64>,
    /// Total residual deviance `Σ d(yᵢ, μ̂ᵢ)` (≥ 0).
    pub deviance: f64,
    /// Deviance recorded at the *start* of each executed IRLS iteration, in
    /// order. Monotonically non-increasing for a well-behaved fit; the final
    /// entry equals (within `tol`) [`TweedieFit::deviance`].
    pub deviance_history: Vec<f64>,
    /// Tweedie power index used.
    pub power: f64,
    /// Dispersion estimate `φ̂ = (1/(n−p)) Σ (yᵢ−μ̂ᵢ)² / V(μ̂ᵢ)` (Pearson).
    pub dispersion: f64,
    /// Number of IRLS iterations executed.
    pub n_iter: usize,
    /// Whether IRLS converged within `tol`.
    pub converged: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public family functions
// ─────────────────────────────────────────────────────────────────────────────

/// Tweedie variance function `V(μ) = μ^p`.
///
/// For `1 < p < 2` and the compound Poisson–Gamma model the mean must be
/// positive; a tiny floor keeps the evaluation finite for `μ → 0⁺`.
#[must_use]
pub fn tweedie_variance(mu: f64, power: f64) -> f64 {
    let mu_safe = mu.max(f64::MIN_POSITIVE);
    mu_safe.powf(power)
}

/// Tweedie **unit deviance** `d(y, μ)` for power `p`.
///
/// Implements Jørgensen's closed form for general `p ∉ {0, 1, 2}` and switches
/// to the exact Poisson / Gamma limiting deviance when `p` is within `1e-6` of
/// `1` or `2`. Finite (and non-negative) at `y = 0` whenever `1 < p < 2`.
#[must_use]
pub fn tweedie_unit_deviance(y: f64, mu: f64, power: f64) -> f64 {
    let mu_s = mu.max(f64::MIN_POSITIVE);
    let y_nn = y.max(0.0);

    // Poisson limit p → 1.
    if (power - 1.0).abs() < 1e-6 {
        return if y_nn <= 0.0 {
            2.0 * mu_s
        } else {
            2.0 * (y_nn * (y_nn / mu_s).ln() - (y_nn - mu_s))
        };
    }
    // Gamma limit p → 2.
    if (power - 2.0).abs() < 1e-6 {
        let y_s = y_nn.max(f64::MIN_POSITIVE);
        return 2.0 * (-(y_s / mu_s).ln() + (y_s - mu_s) / mu_s);
    }

    // General Jørgensen form.
    let one_m_p = 1.0 - power;
    let two_m_p = 2.0 - power;
    // y · (y^{1-p} − μ^{1-p}) / (1−p).  When y = 0 and 1<p<2 the whole first
    // bracket vanishes (y·y^{1-p} = y^{2-p} → 0, and y·μ^{1-p} = 0).
    let term1 = if y_nn <= 0.0 {
        0.0
    } else {
        y_nn * (y_nn.powf(one_m_p) - mu_s.powf(one_m_p)) / one_m_p
    };
    // (y^{2-p} − μ^{2-p}) / (2−p);  y^{2-p} → 0 as y → 0 since 2-p > 0.
    let y_pow_2mp = if y_nn <= 0.0 { 0.0 } else { y_nn.powf(two_m_p) };
    let term2 = (y_pow_2mp - mu_s.powf(two_m_p)) / two_m_p;
    let d = 2.0 * (term1 - term2);
    // Clamp tiny negative round-off to zero; the deviance is non-negative.
    d.max(0.0)
}

/// Total Tweedie deviance `Σ d(yᵢ, μᵢ)`.
///
/// `y` and `mu` must have equal length.
pub fn tweedie_deviance(y: &[f64], mu: &[f64], power: f64) -> StatsResult<f64> {
    if y.len() != mu.len() {
        return Err(StatsError::DimensionMismatch {
            a: y.len(),
            b: mu.len(),
        });
    }
    Ok(y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| tweedie_unit_deviance(yi, mui, power))
        .sum())
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build the (possibly intercept-augmented) design matrix.
fn build_design(x: &[f64], n: usize, p: usize, intercept: bool) -> (Vec<f64>, usize) {
    if !intercept {
        return (x.to_vec(), p);
    }
    let p_aug = p + 1;
    let mut xa = vec![0.0_f64; n * p_aug];
    for i in 0..n {
        xa[i * p_aug] = 1.0;
        for j in 0..p {
            xa[i * p_aug + 1 + j] = x[i * p + j];
        }
    }
    (xa, p_aug)
}

/// Solve `(Xᵀ W X) β = Xᵀ W z` with a small adaptive ridge for stability.
fn wls_solve(xa: &[f64], z: &[f64], w: &[f64], n: usize, p: usize) -> StatsResult<Vec<f64>> {
    let mut xtwx = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in i..p {
            let mut acc = 0.0;
            for k in 0..n {
                acc += xa[k * p + i] * w[k] * xa[k * p + j];
            }
            xtwx[i * p + j] = acc;
            xtwx[j * p + i] = acc;
        }
    }
    let diag_max = (0..p)
        .map(|j| xtwx[j * p + j].abs())
        .fold(0.0_f64, f64::max);
    let ridge = (diag_max * 1e-12).max(1e-14);
    for j in 0..p {
        xtwx[j * p + j] += ridge;
    }
    let mut xtwz = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = 0.0;
        for k in 0..n {
            acc += xa[k * p + i] * w[k] * z[k];
        }
        xtwz[i] = acc;
    }
    let inv = matrix_inverse_lu(&xtwx, p)?;
    let mut beta = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = 0.0;
        for j in 0..p {
            acc += inv[i * p + j] * xtwz[j];
        }
        beta[i] = acc;
    }
    Ok(beta)
}

/// Linear predictor `η = Xβ` for every row.
fn linear_predictor(xa: &[f64], beta: &[f64], n: usize, p: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..p).map(|j| xa[i * p + j] * beta[j]).sum())
        .collect()
}

/// `‖a − b‖₂`.
fn l2_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Fit
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a Tweedie GLM by IRLS.
///
/// # Parameters
/// - `x` — row-major design matrix `(n_samples, n_features)` **without** the
///   intercept column (it is prepended when `cfg.intercept` is `true`).
/// - `y` — non-negative response vector of length `n_samples` (exact zeros are
///   allowed for `1 < p < 2`).
///
/// # Errors
/// Returns an error for shape/length mismatches, non-finite or negative inputs,
/// an out-of-range power index, insufficient sample size, or a singular design.
pub fn tweedie_fit(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &TweedieConfig,
) -> StatsResult<TweedieFit> {
    if n_samples == 0 {
        return Err(StatsError::EmptyInput);
    }
    if x.len() != n_samples * n_features {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_samples, n_features],
            got: vec![x.len()],
        });
    }
    if y.len() != n_samples {
        return Err(StatsError::DimensionMismatch {
            a: y.len(),
            b: n_samples,
        });
    }
    if !(cfg.power > 1.0 && cfg.power < 2.0) {
        return Err(StatsError::InvalidParameter {
            name: "power".to_string(),
            reason: format!(
                "must be in (1, 2) for compound Poisson–Gamma, got {}",
                cfg.power
            ),
        });
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
        if v < 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "y".to_string(),
                reason: format!("Tweedie response must be non-negative, got {v} at index {i}"),
            });
        }
    }
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    let power = cfg.power;
    let link = cfg.link;
    let (xa, p) = build_design(x, n_samples, n_features, cfg.intercept);
    if n_samples < p {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: p,
        });
    }

    // ── Initialisation ──────────────────────────────────────────────────────
    // Start μ slightly inside the positive orthant; for the log link this also
    // gives a sane intercept seed g(ȳ⁺).
    let y_bar = y.iter().sum::<f64>() / n_samples as f64;
    let mu_seed = (y_bar).max(1e-3);
    let mut beta = vec![0.0_f64; p];
    if cfg.intercept && p > 0 {
        beta[0] = link.forward(mu_seed);
    } else {
        // Without an intercept, seed every coefficient at zero except fall back
        // to a constant predictor implied by the first column if present.
        beta.iter_mut().for_each(|b| *b = 0.0);
    }

    let mut deviance_history: Vec<f64> = Vec::with_capacity(cfg.max_iter + 1);
    let mut converged = false;
    let mut n_iter = 0_usize;

    for iter in 0..cfg.max_iter {
        n_iter = iter + 1;

        // Current η, μ.
        let eta = linear_predictor(&xa, &beta, n_samples, p);
        let mu: Vec<f64> = eta.iter().map(|&e| link.inverse(e).max(1e-12)).collect();

        // Record deviance at the start of this iteration (before the update).
        let dev_now: f64 = y
            .iter()
            .zip(mu.iter())
            .map(|(&yi, &mui)| tweedie_unit_deviance(yi, mui, power))
            .sum();
        deviance_history.push(dev_now);

        // Working weights w = (∂μ/∂η)² / V(μ).
        let mut w = vec![0.0_f64; n_samples];
        let mut z = vec![0.0_f64; n_samples];
        for k in 0..n_samples {
            let dmu = link.dmu_deta(mu[k]);
            let v = tweedie_variance(mu[k], power);
            let wk = (dmu * dmu) / v.max(f64::MIN_POSITIVE);
            w[k] = if wk.is_finite() && wk > 0.0 { wk } else { 0.0 };
            // Working response z = η + (y − μ) · (∂η/∂μ) = η + (y − μ)/(∂μ/∂η).
            let deta_dmu = if dmu.abs() < 1e-12 { 0.0 } else { 1.0 / dmu };
            let zk = eta[k] + (y[k] - mu[k]) * deta_dmu;
            z[k] = if zk.is_finite() { zk } else { eta[k] };
        }

        let beta_new = wls_solve(&xa, &z, &w, n_samples, p)?;
        let delta = l2_diff(&beta_new, &beta);
        beta = beta_new;

        if delta < cfg.tol {
            converged = true;
            break;
        }
    }

    // ── Final fitted values and diagnostics ─────────────────────────────────
    let eta_final = linear_predictor(&xa, &beta, n_samples, p);
    let fitted_values: Vec<f64> = eta_final
        .iter()
        .map(|&e| link.inverse(e).max(1e-12))
        .collect();

    let deviance: f64 = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(&yi, &mui)| tweedie_unit_deviance(yi, mui, power))
        .sum();
    // Final deviance closes the history.
    deviance_history.push(deviance);

    let df = n_samples.saturating_sub(p);
    let pearson_chi2: f64 = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(&yi, &mui)| {
            let v = tweedie_variance(mui, power);
            (yi - mui) * (yi - mui) / v.max(f64::MIN_POSITIVE)
        })
        .sum();
    let dispersion = if df > 0 {
        pearson_chi2 / df as f64
    } else {
        1.0
    };

    Ok(TweedieFit {
        coefficients: beta,
        fitted_values,
        deviance,
        deviance_history,
        power,
        dispersion,
        n_iter,
        converged,
    })
}

/// Predict means (or the linear predictor) for new observations.
///
/// `on_link_scale = true` returns `η = Xβ`; otherwise returns `μ = g⁻¹(η)`.
pub fn tweedie_predict(
    fit: &TweedieFit,
    x_new: &[f64],
    n_new: usize,
    cfg: &TweedieConfig,
    on_link_scale: bool,
) -> StatsResult<Vec<f64>> {
    if n_new == 0 {
        return Ok(Vec::new());
    }
    let p = fit.coefficients.len();
    let n_features = if cfg.intercept { p - 1 } else { p };
    if x_new.len() != n_new * n_features {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_new, n_features],
            got: vec![x_new.len()],
        });
    }
    let (xa, p_aug) = build_design(x_new, n_new, n_features, cfg.intercept);
    if p_aug != p {
        return Err(StatsError::DimensionMismatch { a: p_aug, b: p });
    }
    let eta = linear_predictor(&xa, &fit.coefficients, n_new, p);
    Ok(eta
        .iter()
        .map(|&e| {
            if on_link_scale {
                e
            } else {
                cfg.link.inverse(e).max(1e-12)
            }
        })
        .collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build synthetic Tweedie-ish data: log μ = β₀ + β₁ x, response = μ·(noise),
    /// with multiplicative Gamma-like positive noise and occasional exact zeros.
    fn synthetic(n: usize, b0: f64, b1: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| {
                let mu = (b0 + b1 * x).exp();
                // Poisson-count of Gamma jumps → compound Poisson–Gamma flavour:
                // with small probability produce an exact zero, else μ scaled by
                // a positive multiplicative factor around 1.
                if rng.next_f64() < 0.15 {
                    0.0
                } else {
                    let g = (-rng.next_f64().max(1e-9).ln() - rng.next_f64().max(1e-9).ln()) / 2.0;
                    mu * g.max(1e-6)
                }
            })
            .collect();
        (xs, ys)
    }

    // (a) IRLS converges and the deviance is monotonically non-increasing.
    #[test]
    fn deviance_decreases_monotonically() {
        let (xs, ys) = synthetic(120, 0.4, 1.2, 7);
        let cfg = TweedieConfig::with_power(1.5);
        let fit = tweedie_fit(&xs, &ys, 120, 1, &cfg).expect("fit");
        assert!(fit.deviance_history.len() >= 2);
        for w in fit.deviance_history.windows(2) {
            // Allow a tiny positive tolerance for round-off near convergence.
            assert!(
                w[1] <= w[0] + 1e-6,
                "deviance increased: {} -> {}",
                w[0],
                w[1]
            );
        }
        assert!(fit.deviance >= 0.0);
    }

    // (b) Recovers coefficients on data generated from the fitted (log-link) model.
    #[test]
    fn recovers_coefficients() {
        // Deterministic μ exactly on the model: y = μ = exp(0.3 + 0.9 x).
        let n = 60;
        let b0 = 0.3;
        let b1 = 0.9;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| (b0 + b1 * x).exp()).collect();
        let cfg = TweedieConfig::with_power(1.5);
        let fit = tweedie_fit(&xs, &ys, n, 1, &cfg).expect("fit");
        assert!(fit.converged, "should converge on exact data");
        assert!(
            (fit.coefficients[0] - b0).abs() < 1e-3,
            "intercept {} vs {}",
            fit.coefficients[0],
            b0
        );
        assert!(
            (fit.coefficients[1] - b1).abs() < 1e-3,
            "slope {} vs {}",
            fit.coefficients[1],
            b1
        );
    }

    // (c) Deviance ≥ 0 always and ≈ 0 for a (near-)perfect fit.
    #[test]
    fn deviance_nonneg_and_zero_at_perfect_fit() {
        let n = 40;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| (0.5 + 0.7 * x).exp()).collect();
        let cfg = TweedieConfig::with_power(1.6);
        let fit = tweedie_fit(&xs, &ys, n, 1, &cfg).expect("fit");
        assert!(fit.deviance >= 0.0);
        assert!(
            fit.deviance < 1e-6,
            "perfect-fit deviance should be ~0, got {}",
            fit.deviance
        );
    }

    // (d) Variance function V(μ) = μ^p.
    #[test]
    fn variance_function_correct() {
        for &p in &[1.1_f64, 1.5, 1.9] {
            for &mu in &[0.5_f64, 1.0, 2.0, 5.0] {
                let v = tweedie_variance(mu, p);
                assert!((v - mu.powf(p)).abs() < 1e-12, "V({mu},{p}) wrong");
            }
        }
    }

    // (d cont.) p → 1⁺ deviance ≈ Poisson deviance; p → 2⁻ ≈ Gamma deviance.
    #[test]
    fn deviance_limits_poisson_and_gamma() {
        let y = 3.0_f64;
        let mu = 2.0_f64;
        // Poisson deviance.
        let pois = 2.0 * (y * (y / mu).ln() - (y - mu));
        let d_near1 = tweedie_unit_deviance(y, mu, 1.001);
        assert!(
            (d_near1 - pois).abs() < 5e-2,
            "p=1.001 dev {d_near1} vs Poisson {pois}"
        );
        // Gamma deviance.
        let gamma = 2.0 * (-(y / mu).ln() + (y - mu) / mu);
        let d_near2 = tweedie_unit_deviance(y, mu, 1.999);
        assert!(
            (d_near2 - gamma).abs() < 5e-2,
            "p=1.999 dev {d_near2} vs Gamma {gamma}"
        );
    }

    // (e) Predictions μ = exp(η) strictly positive.
    #[test]
    fn predictions_strictly_positive() {
        let (xs, ys) = synthetic(50, -0.2, 0.6, 11);
        let cfg = TweedieConfig::with_power(1.4);
        let fit = tweedie_fit(&xs, &ys, 50, 1, &cfg).expect("fit");
        assert!(fit.fitted_values.iter().all(|&m| m > 0.0));
        let x_new = vec![0.1, 0.5, 0.9, -1.0, 3.0];
        let preds = tweedie_predict(&fit, &x_new, 5, &cfg, false).expect("predict");
        assert!(preds.iter().all(|&m| m > 0.0), "all means positive");
    }

    // (f) Exact-zero responses are handled without NaN (deviance finite at y=0).
    #[test]
    fn exact_zeros_no_nan() {
        // Many exact zeros mixed with positive values.
        let xs: Vec<f64> = (0..30).map(|i| (i as f64) / 30.0).collect();
        let ys: Vec<f64> = (0..30)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    (0.3 + 0.5 * (i as f64 / 30.0)).exp()
                }
            })
            .collect();
        let cfg = TweedieConfig::with_power(1.5);
        let fit = tweedie_fit(&xs, &ys, 30, 1, &cfg).expect("fit");
        assert!(fit.deviance.is_finite() && fit.deviance >= 0.0);
        assert!(fit.coefficients.iter().all(|c| c.is_finite()));
        assert!(fit.fitted_values.iter().all(|m| m.is_finite() && *m > 0.0));
        // Direct check: d(0, μ) is finite and equals 2 μ^{2-p}/(2-p).
        let p = 1.5;
        let mu = 2.0;
        let d0 = tweedie_unit_deviance(0.0, mu, p);
        let expected = 2.0 * mu.powf(2.0 - p) / (2.0 - p);
        assert!(d0.is_finite(), "d(0,μ) must be finite");
        assert!((d0 - expected).abs() < 1e-9, "d(0,μ)={d0} vs {expected}");
    }

    // Total deviance equals the sum of unit deviances.
    #[test]
    fn total_deviance_matches_units() {
        let y = vec![0.0, 1.0, 2.5, 4.0];
        let mu = vec![1.0, 1.2, 2.0, 3.5];
        let p = 1.5;
        let total = tweedie_deviance(&y, &mu, p).expect("ok");
        let manual: f64 = y
            .iter()
            .zip(mu.iter())
            .map(|(&yi, &mui)| tweedie_unit_deviance(yi, mui, p))
            .sum();
        assert!((total - manual).abs() < 1e-12);
    }

    // Invalid power → error.
    #[test]
    fn invalid_power_errors() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![1.0, 2.0, 3.0];
        for &bad in &[0.5_f64, 1.0, 2.0, 2.5] {
            let cfg = TweedieConfig::with_power(bad);
            assert!(tweedie_fit(&xs, &ys, 3, 1, &cfg).is_err(), "power {bad}");
        }
    }

    // Negative response → error.
    #[test]
    fn negative_response_errors() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![1.0, -2.0, 3.0];
        let cfg = TweedieConfig::with_power(1.5);
        assert!(tweedie_fit(&xs, &ys, 3, 1, &cfg).is_err());
    }

    // Empty input → error.
    #[test]
    fn empty_input_errors() {
        let cfg = TweedieConfig::with_power(1.5);
        assert!(tweedie_fit(&[], &[], 0, 1, &cfg).is_err());
    }

    // Dispersion is finite and non-negative.
    #[test]
    fn dispersion_finite() {
        let (xs, ys) = synthetic(80, 0.1, 0.8, 3);
        let cfg = TweedieConfig::with_power(1.5);
        let fit = tweedie_fit(&xs, &ys, 80, 1, &cfg).expect("fit");
        assert!(fit.dispersion.is_finite() && fit.dispersion >= 0.0);
    }
}

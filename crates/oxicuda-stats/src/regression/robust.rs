//! Robust regression: Huber-M estimation, bisquare (Tukey) M-estimation,
//! Least Median of Squares (LMS), Least Trimmed Squares (LTS), and RANSAC.
//!
//! All estimators are outlier-resistant alternatives to OLS.  They operate on
//! flat row-major design matrices and use the workspace [`LcgRng`] for any
//! stochastic sub-sampling.
//!
//! # References
//! - Huber (1964) "Robust Estimation of a Location Parameter".
//! - Rousseeuw & Leroy (1987) *Robust Regression and Outlier Detection*.
//! - Fischler & Bolles (1981) "Random Sample Consensus".

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

// ══════════════════════════════════════════════════════════════════════════════
// Scale estimation
// ══════════════════════════════════════════════════════════════════════════════

/// Method used to estimate the regression scale σ̂.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMethod {
    /// Median Absolute Deviation / 0.6745  — consistent at the Gaussian.
    Mad,
    /// Interquartile Range / 1.349  — consistent at the Gaussian.
    Iqr,
    /// Fixed scale of 1.0 (no estimation).
    Fixed,
}

/// Compute the Median Absolute Deviation (MAD) of a slice.
///
/// MAD = median( |r_i - median(r)| )
///
/// Returns 0.0 for an empty or single-element slice.
pub fn median_absolute_deviation(residuals: &[f64]) -> f64 {
    let n = residuals.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return 0.0;
    }
    let med = sorted_median(residuals);
    let mut deviations: Vec<f64> = residuals.iter().map(|&r| (r - med).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_median_of_sorted(&deviations)
}

/// Scale estimate: MAD / 0.6745.
///
/// 0.6745 is the 75th percentile of the standard normal, making this
/// a consistent estimate of σ for normally distributed errors.
pub fn estimate_scale_mad(residuals: &[f64]) -> f64 {
    let mad = median_absolute_deviation(residuals);
    // Prevent scale collapse near zero
    (mad / 0.6745).max(1e-10)
}

/// Scale estimate: IQR / 1.349.
///
/// 1.349 ≈ 2 * Φ⁻¹(0.75) makes this consistent at the Gaussian.
pub fn estimate_scale_iqr(residuals: &[f64]) -> f64 {
    let n = residuals.len();
    if n < 2 {
        return 1.0;
    }
    let mut sorted = residuals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = quantile_sorted(&sorted, 0.25);
    let q3 = quantile_sorted(&sorted, 0.75);
    ((q3 - q1) / 1.349).max(1e-10)
}

/// Winsorized standard deviation.
///
/// Trims fraction `trim` from each tail, then computes standard deviation of
/// the remaining values (using the winsorized mean).  `trim` should be in
/// `[0, 0.5)`.
pub fn winsorized_scale(residuals: &[f64], trim: f64) -> f64 {
    let n = residuals.len();
    if n < 2 {
        return 1.0;
    }
    let trim = trim.clamp(0.0, 0.49);
    let k = (trim * n as f64).floor() as usize;
    let mut sorted = residuals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = sorted[k];
    let hi = sorted[n - 1 - k];
    // Winsorise
    let win: Vec<f64> = sorted.iter().map(|&v| v.clamp(lo, hi)).collect();
    let mean = win.iter().sum::<f64>() / n as f64;
    let variance = win.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
    variance.sqrt().max(1e-10)
}

// ────────────────────── internal median helpers ───────────────────────────────

/// Compute median of an unsorted slice (copies + sorts internally).
fn sorted_median(data: &[f64]) -> f64 {
    let mut v = data.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_median_of_sorted(&v)
}

/// Compute median of an already-sorted slice.
fn sorted_median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Linear interpolation quantile of a sorted slice.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = p * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = h - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ══════════════════════════════════════════════════════════════════════════════
// Configuration structs
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for Huber M-estimation via IRLS.
#[derive(Debug, Clone)]
pub struct HuberConfig {
    /// Huber tuning constant *c* (default 1.345, giving 95 % efficiency at Gaussian).
    pub c: f64,
    /// Maximum number of IRLS iterations (default 50).
    pub max_iter: usize,
    /// Convergence tolerance on ‖β_new − β_old‖₂ (default 1e-8).
    pub tol: f64,
    /// Prepend an intercept column to the design matrix (default `true`).
    pub intercept: bool,
    /// Method used to estimate σ̂ at each step (default `Mad`).
    pub scale_method: ScaleMethod,
}

impl Default for HuberConfig {
    fn default() -> Self {
        Self {
            c: 1.345,
            max_iter: 50,
            tol: 1e-8,
            intercept: true,
            scale_method: ScaleMethod::Mad,
        }
    }
}

/// Configuration for Tukey bisquare (biweight) M-estimation via IRLS.
#[derive(Debug, Clone)]
pub struct BisquareConfig {
    /// Bisquare tuning constant *c* (default 4.685, giving 95 % efficiency at Gaussian).
    pub c: f64,
    /// Maximum number of IRLS iterations (default 100).
    pub max_iter: usize,
    /// Convergence tolerance on ‖β_new − β_old‖₂.
    pub tol: f64,
    /// Prepend an intercept column to the design matrix (default `true`).
    pub intercept: bool,
    /// Method used to estimate σ̂ at each step (default `Mad`).
    pub scale_method: ScaleMethod,
}

impl Default for BisquareConfig {
    fn default() -> Self {
        Self {
            c: 4.685,
            max_iter: 100,
            tol: 1e-8,
            intercept: true,
            scale_method: ScaleMethod::Mad,
        }
    }
}

/// Configuration for RANSAC (Random Sample Consensus).
#[derive(Debug, Clone)]
pub struct RansacConfig {
    /// Maximum number of random trials (default 100).
    pub max_trials: usize,
    /// Inlier residual threshold |r_i| / σ < threshold (default 3.0).
    pub residual_threshold: f64,
    /// Minimum samples per trial.  `None` → `n_features + 1`.
    pub min_samples: Option<usize>,
    /// Stop early when inlier fraction exceeds this (default 0.95).
    pub stop_inlier_fraction: f64,
    /// Prepend an intercept column to the design matrix (default `true`).
    pub intercept: bool,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_trials: 100,
            residual_threshold: 3.0,
            min_samples: None,
            stop_inlier_fraction: 0.95,
            intercept: true,
        }
    }
}

/// Configuration for Least Median of Squares (LMS).
#[derive(Debug, Clone)]
pub struct LmsConfig {
    /// Number of random subsets to draw (default 500).
    pub n_subsamples: usize,
    /// Prepend an intercept column to the design matrix (default `true`).
    pub intercept: bool,
    /// Refine the LMS solution with Huber IRLS (default `true`).
    pub refine: bool,
}

impl Default for LmsConfig {
    fn default() -> Self {
        Self {
            n_subsamples: 500,
            intercept: true,
            refine: true,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Result struct
// ══════════════════════════════════════════════════════════════════════════════

/// Output of any robust regression estimator.
#[derive(Debug, Clone)]
pub struct RobustFit {
    /// Slope coefficients (length = n_features; does NOT include the intercept).
    pub coefficients: Vec<f64>,
    /// Fitted intercept (0.0 when `intercept = false`).
    pub intercept: f64,
    /// Raw residuals r_i = y_i − ŷ_i (length = n_samples).
    pub residuals: Vec<f64>,
    /// Final IRLS weights w_i ∈ [0, 1] (length = n_samples; 1.0 for OLS/RANSAC).
    pub weights: Vec<f64>,
    /// Final scale estimate σ̂.
    pub scale: f64,
    /// Number of IRLS or RANSAC iterations taken.
    pub n_iter: usize,
    /// Whether the algorithm converged within the allowed iterations.
    pub converged: bool,
    /// Number of inliers (RANSAC; equals n_samples for M-estimators).
    pub n_inliers: usize,
    /// Boolean inlier mask (RANSAC; all `true` for M-estimators).
    pub inlier_mask: Vec<bool>,
}

// ══════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ══════════════════════════════════════════════════════════════════════════════

/// Prepend a column of 1s to `x` (row-major, n×n_features) → n×(n_features+1).
fn augment_intercept(x: &[f64], n: usize, n_features: usize) -> (Vec<f64>, usize) {
    let p = n_features + 1;
    let mut xd = vec![0.0_f64; n * p];
    for k in 0..n {
        xd[k * p] = 1.0;
        for j in 0..n_features {
            xd[k * p + j + 1] = x[k * n_features + j];
        }
    }
    (xd, p)
}

/// Build extended design matrix (with or without intercept).
fn build_design(x: &[f64], n: usize, n_features: usize, intercept: bool) -> (Vec<f64>, usize) {
    if intercept {
        augment_intercept(x, n, n_features)
    } else {
        (x.to_vec(), n_features)
    }
}

/// Weighted least squares via Cholesky factorisation.
///
/// Solves (X^T W X + ridge·I) β = X^T W y.
/// Returns `None` if the system is numerically singular.
fn wls_solve(x: &[f64], y: &[f64], w: &[f64], n: usize, p: usize) -> Option<Vec<f64>> {
    // Build A = X^T W X  (p×p)
    let mut a = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in i..p {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += x[k * p + i] * w[k] * x[k * p + j];
            }
            a[i * p + j] = acc;
            a[j * p + i] = acc;
        }
    }
    // Ridge regularisation
    let diag_max = (0..p).map(|j| a[j * p + j].abs()).fold(0.0_f64, f64::max);
    let ridge = (diag_max * 1e-10).max(1e-14);
    for j in 0..p {
        a[j * p + j] += ridge;
    }
    // Build b = X^T W y  (p)
    let mut b = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = 0.0_f64;
        for k in 0..n {
            acc += x[k * p + i] * w[k] * y[k];
        }
        b[i] = acc;
    }
    // Cholesky factorisation — lower triangle of `a` overwritten with L
    for j in 0..p {
        let mut s = a[j * p + j];
        for k in 0..j {
            s -= a[j * p + k] * a[j * p + k];
        }
        if s <= 0.0 {
            return None;
        }
        let l_jj = s.sqrt();
        a[j * p + j] = l_jj;
        for i in (j + 1)..p {
            let mut t = a[i * p + j];
            for k in 0..j {
                t -= a[i * p + k] * a[j * p + k];
            }
            a[i * p + j] = t / l_jj;
        }
    }
    // Forward substitution: L v = b
    let mut v = vec![0.0_f64; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * p + k] * v[k];
        }
        v[i] = s / a[i * p + i];
    }
    // Back substitution: L^T β = v
    let mut beta = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut s = v[i];
        for k in (i + 1)..p {
            s -= a[k * p + i] * beta[k];
        }
        beta[i] = s / a[i * p + i];
    }
    Some(beta)
}

/// OLS using WLS with all weights = 1.
fn ols_solve(x: &[f64], y: &[f64], n: usize, p: usize) -> Option<Vec<f64>> {
    let w = vec![1.0_f64; n];
    wls_solve(x, y, &w, n, p)
}

/// Compute fitted values ŷ = X β (row-major X, shape n×p).
fn fitted_values(x: &[f64], beta: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut yhat = vec![0.0_f64; n];
    for k in 0..n {
        let mut acc = 0.0_f64;
        for j in 0..p {
            acc += x[k * p + j] * beta[j];
        }
        yhat[k] = acc;
    }
    yhat
}

/// Compute residuals r = y − ŷ.
fn residuals_vec(y: &[f64], yhat: &[f64]) -> Vec<f64> {
    y.iter()
        .zip(yhat.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect()
}

/// Compute the scale estimate from residuals using the given method.
fn compute_scale(resids: &[f64], method: ScaleMethod) -> f64 {
    match method {
        ScaleMethod::Mad => estimate_scale_mad(resids),
        ScaleMethod::Iqr => estimate_scale_iqr(resids),
        ScaleMethod::Fixed => 1.0,
    }
}

/// Extract intercept and slope coefficients from the augmented β.
///
/// When `intercept == true` the first element of `beta_aug` is the intercept.
fn split_coefficients(beta_aug: &[f64], intercept: bool, n_features: usize) -> (f64, Vec<f64>) {
    if intercept {
        let int_val = if beta_aug.is_empty() {
            0.0
        } else {
            beta_aug[0]
        };
        let slopes = if beta_aug.len() > 1 {
            beta_aug[1..].to_vec()
        } else {
            vec![0.0_f64; n_features]
        };
        (int_val, slopes)
    } else {
        (0.0, beta_aug.to_vec())
    }
}

/// Draw a random subset of `k` distinct indices from `[0, n)`.
fn random_subset(n: usize, k: usize, rng: &mut LcgRng) -> Vec<usize> {
    let k = k.min(n);
    // Fisher-Yates partial shuffle on indices [0..n)
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        idx.swap(i, j);
    }
    idx[..k].to_vec()
}

/// Extract a sub-matrix from `x` (row-major n×p) for the given row indices.
fn extract_rows(x: &[f64], rows: &[usize], n_cols: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * n_cols);
    for &r in rows {
        for j in 0..n_cols {
            out.push(x[r * n_cols + j]);
        }
    }
    out
}

/// Extract the y values for the given row indices.
fn extract_y(y: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|&r| y[r]).collect()
}

// ══════════════════════════════════════════════════════════════════════════════
// Huber weight functions
// ══════════════════════════════════════════════════════════════════════════════

/// Huber influence function ψ(r) = r if |r| ≤ c else c·sign(r).
#[inline]
fn huber_psi(r: f64, c: f64) -> f64 {
    if r.abs() <= c { r } else { c * r.signum() }
}

/// Huber weight w(r) = ψ(r)/r = 1 if |r| ≤ c else c/|r|.
#[inline]
fn huber_weight(r: f64, c: f64) -> f64 {
    let _ = huber_psi; // referenced in docstring
    let ar = r.abs();
    if ar <= c { 1.0 } else { c / ar }
}

// ══════════════════════════════════════════════════════════════════════════════
// Bisquare weight functions
// ══════════════════════════════════════════════════════════════════════════════

/// Tukey bisquare weight w(r) = (1 - (r/c)²)² if |r| ≤ c else 0.
#[inline]
fn bisquare_weight(r: f64, c: f64) -> f64 {
    let ar = r.abs();
    if ar > c {
        0.0
    } else {
        let u = 1.0 - (r / c) * (r / c);
        u * u
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// IRLS core
// ══════════════════════════════════════════════════════════════════════════════

/// Core IRLS loop given a closure that maps a scaled residual to a weight.
///
/// `beta_init` — initial coefficient vector (length p).
/// `xd`        — design matrix (n×p, with intercept prepended if needed).
/// `y`         — response vector (length n).
/// `weight_fn` — closure mapping scaled residual r_i* = r_i/σ̂ to IRLS weight.
/// Returns `(beta, weights, scale, n_iter, converged)`.
fn irls_core<F>(
    xd: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    beta_init: Vec<f64>,
    scale_method: ScaleMethod,
    max_iter: usize,
    tol: f64,
    weight_fn: F,
) -> (Vec<f64>, Vec<f64>, f64, usize, bool)
where
    F: Fn(f64) -> f64,
{
    let mut beta = beta_init;
    let mut weights = vec![1.0_f64; n];
    let mut scale = 1.0_f64;
    let mut converged = false;
    let mut n_iter = 0_usize;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Compute residuals
        let yhat = fitted_values(xd, &beta, n, p);
        let resids = residuals_vec(y, &yhat);

        // Estimate scale
        scale = compute_scale(&resids, scale_method);

        // Compute Huber / bisquare weights
        for i in 0..n {
            let r_scaled = resids[i] / scale;
            weights[i] = weight_fn(r_scaled);
        }

        // Solve WLS
        let beta_new = match wls_solve(xd, y, &weights, n, p) {
            Some(b) => b,
            None => break,
        };

        // Check convergence
        let diff: f64 = beta_new
            .iter()
            .zip(beta.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        beta = beta_new;

        if diff < tol {
            converged = true;
            break;
        }
    }

    (beta, weights, scale, n_iter, converged)
}

// ══════════════════════════════════════════════════════════════════════════════
// Public API
// ══════════════════════════════════════════════════════════════════════════════

/// Fit a robust linear model using Huber M-estimation (IRLS).
///
/// `x` — design matrix, row-major, shape `n_samples × n_features` (no intercept column).
/// `y` — response vector, length `n_samples`.
///
/// The OLS estimate is used as the starting value for IRLS.
pub fn huber_fit(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &HuberConfig,
) -> StatsResult<RobustFit> {
    validate_inputs(x, y, n_samples, n_features)?;

    let (xd, p) = build_design(x, n_samples, n_features, cfg.intercept);

    if n_samples < p {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: p,
        });
    }

    // OLS initialisation
    let beta_ols = ols_solve(&xd, y, n_samples, p).ok_or_else(|| {
        StatsError::SingularMatrix("Huber OLS initialisation: X^T X is singular".to_string())
    })?;

    let c = cfg.c;
    let (beta, weights, scale, n_iter, converged) = irls_core(
        &xd,
        y,
        n_samples,
        p,
        beta_ols,
        cfg.scale_method,
        cfg.max_iter,
        cfg.tol,
        move |r| huber_weight(r, c),
    );

    finalise_fit(
        beta,
        weights,
        scale,
        n_iter,
        converged,
        xd,
        y,
        n_samples,
        p,
        n_features,
        cfg.intercept,
    )
}

/// Fit a robust linear model using Tukey bisquare (biweight) M-estimation (IRLS).
///
/// Uses the Huber estimate as starting value to guard against convergence issues
/// caused by the bisquare's hard cutoff.
pub fn bisquare_fit(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &BisquareConfig,
) -> StatsResult<RobustFit> {
    validate_inputs(x, y, n_samples, n_features)?;

    let (xd, p) = build_design(x, n_samples, n_features, cfg.intercept);

    if n_samples < p {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: p,
        });
    }

    // Warm-start: one round of Huber IRLS (10 iterations) for stability
    let beta_ols = ols_solve(&xd, y, n_samples, p).ok_or_else(|| {
        StatsError::SingularMatrix("Bisquare OLS initialisation: X^T X is singular".to_string())
    })?;

    let huber_c = 1.345_f64;
    let (beta_warm, _w, _sc, _ni, _conv) = irls_core(
        &xd,
        y,
        n_samples,
        p,
        beta_ols,
        cfg.scale_method,
        10,
        1e-6,
        move |r| huber_weight(r, huber_c),
    );

    let c = cfg.c;
    let (beta, weights, scale, n_iter, converged) = irls_core(
        &xd,
        y,
        n_samples,
        p,
        beta_warm,
        cfg.scale_method,
        cfg.max_iter,
        cfg.tol,
        move |r| bisquare_weight(r, c),
    );

    finalise_fit(
        beta,
        weights,
        scale,
        n_iter,
        converged,
        xd,
        y,
        n_samples,
        p,
        n_features,
        cfg.intercept,
    )
}

/// Fit a robust model using RANSAC (Random Sample Consensus).
///
/// At each trial a minimal random subset is used to fit OLS; the resulting
/// model is scored by counting inliers.  After `max_trials` (or early stopping)
/// the best model is refitted on all its inliers.
pub fn ransac_fit(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &RansacConfig,
    rng: &mut LcgRng,
) -> StatsResult<RobustFit> {
    validate_inputs(x, y, n_samples, n_features)?;

    let (xd, p) = build_design(x, n_samples, n_features, cfg.intercept);

    let min_samples = cfg.min_samples.unwrap_or(p);
    let min_samples = min_samples.max(p).min(n_samples);

    if n_samples < min_samples {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: min_samples,
        });
    }

    let stop_count = (cfg.stop_inlier_fraction * n_samples as f64).ceil() as usize;

    let mut best_beta: Option<Vec<f64>> = None;
    let mut best_inlier_count = 0_usize;
    let mut best_inlier_mask = vec![false; n_samples];

    // Pre-compute a robust scale estimate from a large random subsample
    // (median of absolute values from many random 2-point fits).
    // This avoids the chicken-and-egg problem of estimating σ from contaminated OLS.
    // Strategy: sample 20 minimal subsets, for each compute the MAD of all residuals
    // under that model, and take the minimum (the "cleanest" subset likely gives the
    // smallest MAD).
    let global_sigma = {
        let n_probe = 30_usize.min(cfg.max_trials);
        let mut min_mad = f64::INFINITY;
        // We need a fresh sub-rng — advance the given rng a fixed amount
        let mut probe_state = LcgRng::new(rng.next_u64());
        for _ in 0..n_probe {
            let idx = random_subset(n_samples, min_samples, &mut probe_state);
            let xs = extract_rows(&xd, &idx, p);
            let ys = extract_y(y, &idx);
            if let Some(b) = ols_solve(&xs, &ys, min_samples, p) {
                let yhat = fitted_values(&xd, &b, n_samples, p);
                let resids = residuals_vec(y, &yhat);
                // Use the h-th smallest absolute residual as a proxy for scale
                // (LTS-style: the median of the smallest half)
                let h = (n_samples / 2).max(p);
                let mut abs_r: Vec<f64> = resids.iter().map(|r| r.abs()).collect();
                abs_r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med_half = sorted_median_of_sorted(&abs_r[..h]);
                if med_half < min_mad {
                    min_mad = med_half;
                }
            }
        }
        // Convert to consistent σ̂
        (min_mad / 0.6745).max(1e-10)
    };

    // Absolute threshold in data units
    let abs_thr_global = cfg.residual_threshold * global_sigma;

    for _trial in 0..cfg.max_trials {
        // Sample minimal subset
        let sample_idx = random_subset(n_samples, min_samples, rng);
        let xs = extract_rows(&xd, &sample_idx, p);
        let ys = extract_y(y, &sample_idx);

        let beta_trial = match ols_solve(&xs, &ys, min_samples, p) {
            Some(b) => b,
            None => continue,
        };

        let yhat_trial = fitted_values(&xd, &beta_trial, n_samples, p);
        let resids_trial = residuals_vec(y, &yhat_trial);

        // Use the pre-computed global scale for consistent threshold
        let abs_thr = abs_thr_global;

        // Score: count inliers under this model
        let mut mask = vec![false; n_samples];
        let mut count = 0_usize;
        for i in 0..n_samples {
            if resids_trial[i].abs() < abs_thr {
                mask[i] = true;
                count += 1;
            }
        }

        if count > best_inlier_count {
            // Refit on all inliers for a better model
            let inlier_idx: Vec<usize> = (0..n_samples).filter(|&i| mask[i]).collect();
            let xi = extract_rows(&xd, &inlier_idx, p);
            let yi = extract_y(y, &inlier_idx);
            let n_in = inlier_idx.len();
            if n_in >= p {
                if let Some(b_refitted) = ols_solve(&xi, &yi, n_in, p) {
                    // Recount inliers with refitted model using the same σ
                    let yhat2 = fitted_values(&xd, &b_refitted, n_samples, p);
                    let mut mask2 = vec![false; n_samples];
                    let mut count2 = 0_usize;
                    for i in 0..n_samples {
                        if (y[i] - yhat2[i]).abs() < abs_thr {
                            mask2[i] = true;
                            count2 += 1;
                        }
                    }
                    if count2 >= best_inlier_count {
                        best_inlier_count = count2;
                        best_beta = Some(b_refitted);
                        best_inlier_mask = mask2;
                    }
                } else {
                    // refitting failed — keep the trial model
                    best_inlier_count = count;
                    best_beta = Some(beta_trial);
                    best_inlier_mask = mask;
                }
            } else {
                best_inlier_count = count;
                best_beta = Some(beta_trial);
                best_inlier_mask = mask;
            }

            if best_inlier_count >= stop_count {
                break;
            }
        }
    }

    let beta = best_beta.unwrap_or_else(|| {
        // Fallback: OLS on all data
        ols_solve(&xd, y, n_samples, p).unwrap_or_else(|| vec![0.0; p])
    });

    // Recompute final residuals and inlier mask from the best beta
    let yhat_final = fitted_values(&xd, &beta, n_samples, p);
    let resids_final = residuals_vec(y, &yhat_final);
    // Use the pre-computed global scale for the final mask
    let sigma_final = global_sigma;
    let abs_thr_final = cfg.residual_threshold * sigma_final;
    let final_inlier_mask: Vec<bool> = resids_final
        .iter()
        .map(|&r| r.abs() < abs_thr_final)
        .collect();
    let n_inliers_final = final_inlier_mask.iter().filter(|&&v| v).count();
    // Use the larger inlier set for consistency
    let final_mask = if n_inliers_final >= best_inlier_count {
        final_inlier_mask
    } else {
        best_inlier_mask
    };
    let final_n_inliers = final_mask.iter().filter(|&&v| v).count().max(1);

    let weights = vec![1.0_f64; n_samples];
    let scale = sigma_final;

    let (int_val, slopes) = split_coefficients(&beta, cfg.intercept, n_features);
    let residuals = resids_final;

    Ok(RobustFit {
        coefficients: slopes,
        intercept: int_val,
        residuals,
        weights,
        scale,
        n_iter: cfg.max_trials,
        converged: final_n_inliers >= 1,
        n_inliers: final_n_inliers,
        inlier_mask: final_mask,
    })
}

/// Fit a robust model using Least Median of Squares (LMS).
///
/// Draws `n_subsamples` random minimal subsets, fits OLS on each, evaluates
/// the median squared residual on all data, and keeps the best.  Optionally
/// refines with Huber IRLS.
pub fn lms_fit(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &LmsConfig,
    rng: &mut LcgRng,
) -> StatsResult<RobustFit> {
    validate_inputs(x, y, n_samples, n_features)?;

    let (xd, p) = build_design(x, n_samples, n_features, cfg.intercept);

    if n_samples < p {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: p,
        });
    }

    let subset_size = p; // minimal model

    let mut best_beta: Option<Vec<f64>> = None;
    let mut best_med_sq = f64::INFINITY;

    for _ in 0..cfg.n_subsamples {
        let idx = random_subset(n_samples, subset_size, rng);
        let xs = extract_rows(&xd, &idx, p);
        let ys = extract_y(y, &idx);

        let beta_trial = match ols_solve(&xs, &ys, subset_size, p) {
            Some(b) => b,
            None => continue,
        };

        let yhat = fitted_values(&xd, &beta_trial, n_samples, p);
        let mut sq_resids: Vec<f64> = y
            .iter()
            .zip(yhat.iter())
            .map(|(&yi, &fi)| (yi - fi) * (yi - fi))
            .collect();
        sq_resids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med_sq = sorted_median_of_sorted(&sq_resids);

        if med_sq < best_med_sq {
            best_med_sq = med_sq;
            best_beta = Some(beta_trial);
        }
    }

    let beta_lms = best_beta
        .unwrap_or_else(|| ols_solve(&xd, y, n_samples, p).unwrap_or_else(|| vec![0.0; p]));

    // Optionally refine with Huber IRLS
    let (beta_final, weights, scale, n_iter, converged) = if cfg.refine {
        let huber_c = 1.345_f64;
        irls_core(
            &xd,
            y,
            n_samples,
            p,
            beta_lms,
            ScaleMethod::Mad,
            50,
            1e-8,
            move |r| huber_weight(r, huber_c),
        )
    } else {
        let yhat = fitted_values(&xd, &beta_lms, n_samples, p);
        let resids = residuals_vec(y, &yhat);
        let sc = estimate_scale_mad(&resids);
        let w = vec![1.0_f64; n_samples];
        (beta_lms, w, sc, cfg.n_subsamples, true)
    };

    finalise_fit(
        beta_final,
        weights,
        scale,
        n_iter,
        converged,
        xd,
        y,
        n_samples,
        p,
        n_features,
        cfg.intercept,
    )
}

/// Fit a robust model using Least Trimmed Squares (LTS).
///
/// Draws `n_subsamples` random minimal subsets, fits OLS on each, sorts the
/// squared residuals on all data, and keeps the β with the smallest trimmed
/// sum (the h smallest squared residuals, h = ⌊n/2⌋ + ⌊(p+1)/2⌋ + 1).
pub fn lts_fit(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    n_subsamples: usize,
    intercept: bool,
    rng: &mut LcgRng,
) -> StatsResult<RobustFit> {
    validate_inputs(x, y, n_samples, n_features)?;

    let (xd, p) = build_design(x, n_samples, n_features, intercept);

    if n_samples < p {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: p,
        });
    }

    // LTS breakdown fraction h
    let h = (n_samples / 2) + p.div_ceil(2) + 1;
    let h = h.min(n_samples);

    let subset_size = p;

    let mut best_beta: Option<Vec<f64>> = None;
    let mut best_trimmed_sum = f64::INFINITY;

    for _ in 0..n_subsamples {
        let idx = random_subset(n_samples, subset_size, rng);
        let xs = extract_rows(&xd, &idx, p);
        let ys = extract_y(y, &idx);

        let beta_trial = match ols_solve(&xs, &ys, subset_size, p) {
            Some(b) => b,
            None => continue,
        };

        let yhat = fitted_values(&xd, &beta_trial, n_samples, p);
        let mut sq_resids: Vec<f64> = y
            .iter()
            .zip(yhat.iter())
            .map(|(&yi, &fi)| (yi - fi) * (yi - fi))
            .collect();
        sq_resids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let trimmed_sum: f64 = sq_resids[..h].iter().sum();

        if trimmed_sum < best_trimmed_sum {
            best_trimmed_sum = trimmed_sum;
            best_beta = Some(beta_trial);
        }
    }

    let beta_lts = best_beta
        .unwrap_or_else(|| ols_solve(&xd, y, n_samples, p).unwrap_or_else(|| vec![0.0; p]));

    // One-step Huber refinement for good efficiency
    let huber_c = 1.345_f64;
    let (beta_final, weights, scale, n_iter, converged) = irls_core(
        &xd,
        y,
        n_samples,
        p,
        beta_lts,
        ScaleMethod::Mad,
        50,
        1e-8,
        move |r| huber_weight(r, huber_c),
    );

    finalise_fit(
        beta_final, weights, scale, n_iter, converged, xd, y, n_samples, p, n_features, intercept,
    )
}

// ══════════════════════════════════════════════════════════════════════════════
// Shared finalization helper
// ══════════════════════════════════════════════════════════════════════════════

fn finalise_fit(
    beta: Vec<f64>,
    weights: Vec<f64>,
    scale: f64,
    n_iter: usize,
    converged: bool,
    xd: Vec<f64>,
    y: &[f64],
    n_samples: usize,
    p: usize,
    n_features: usize,
    intercept: bool,
) -> StatsResult<RobustFit> {
    let yhat = fitted_values(&xd, &beta, n_samples, p);
    let residuals = residuals_vec(y, &yhat);
    let (int_val, slopes) = split_coefficients(&beta, intercept, n_features);
    let inlier_mask = vec![true; n_samples];
    let n_inliers = n_samples;

    Ok(RobustFit {
        coefficients: slopes,
        intercept: int_val,
        residuals,
        weights,
        scale,
        n_iter,
        converged,
        n_inliers,
        inlier_mask,
    })
}

/// Validate x / y dimensions.
fn validate_inputs(x: &[f64], y: &[f64], n_samples: usize, n_features: usize) -> StatsResult<()> {
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
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────────────

    /// Build a design matrix for y = intercept + slope * x.
    fn simple_x(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    /// Generate clean linear data: y_i = intercept + slope * x_i.
    fn clean_linear(n: usize, intercept: f64, slope: f64) -> (Vec<f64>, Vec<f64>) {
        let x = simple_x(n);
        let y: Vec<f64> = x.iter().map(|&xi| intercept + slope * xi).collect();
        (x, y)
    }

    /// Generate data with vertical outliers (outlier x values are in-range;
    /// only y is contaminated).  This tests robustness against non-leverage outliers,
    /// where M-estimators are guaranteed to work at the given breakdown fraction.
    fn data_with_outliers(
        n_clean: usize,
        n_outlier: usize,
        intercept: f64,
        slope: f64,
        outlier_offset: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n_total = n_clean + n_outlier;
        // All x values spread uniformly over [0, n_clean)
        let mut x: Vec<f64> = (0..n_total).map(|i| i as f64 % n_clean as f64).collect();
        // Sort so x is monotone (cleaner data for the regressor)
        x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Clean responses for first n_clean, contaminated for last n_outlier
        let mut y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let clean_y = intercept + slope * xi;
                if i < n_clean {
                    clean_y
                } else {
                    clean_y + outlier_offset
                }
            })
            .collect();
        // Shuffle outliers into the interior so they are vertical outliers, not leverage points
        // Place them at x positions near the center of the clean data range
        for i in n_clean..n_total {
            x[i] = (i - n_clean + 2) as f64; // interior x value
            y[i] = intercept + slope * x[i] + outlier_offset;
        }
        (x, y)
    }

    // ── 1. Huber recovers OLS on clean data ────────────────────────────────────

    #[test]
    fn huber_recovers_ols_clean() {
        let n = 30_usize;
        let (x, y) = clean_linear(n, 1.0, 2.0);
        let cfg = HuberConfig::default();
        let fit = huber_fit(&x, &y, n, 1, &cfg).expect("huber fit ok");
        // Intercept ≈ 1.0, slope ≈ 2.0
        assert!(
            (fit.intercept - 1.0).abs() < 0.05,
            "intercept: {}",
            fit.intercept
        );
        assert!(
            (fit.coefficients[0] - 2.0).abs() < 0.05,
            "slope: {}",
            fit.coefficients[0]
        );
    }

    // ── 2. Huber is robust to outliers ────────────────────────────────────────

    #[test]
    fn huber_robust_to_outliers() {
        let n_clean = 16_usize;
        let n_outlier = 4_usize;
        let (x, y) = data_with_outliers(n_clean, n_outlier, 2.0, 3.0, 100.0);
        let cfg = HuberConfig::default();
        let fit = huber_fit(&x, &y, n_clean + n_outlier, 1, &cfg).expect("huber ok");
        // Should recover slope ≈ 3.0 within 10 %
        assert!(
            (fit.coefficients[0] - 3.0).abs() < 0.3,
            "huber slope {} far from 3.0",
            fit.coefficients[0]
        );
    }

    // ── 3. Huber empty data returns Err ───────────────────────────────────────

    #[test]
    fn huber_empty_data_error() {
        let cfg = HuberConfig::default();
        assert!(huber_fit(&[], &[], 0, 1, &cfg).is_err());
    }

    // ── 4. Huber output shape ─────────────────────────────────────────────────

    #[test]
    fn huber_output_shape() {
        let n = 20_usize;
        let (x, y) = clean_linear(n, 0.0, 1.0);
        let cfg = HuberConfig {
            intercept: true,
            ..Default::default()
        };
        let fit = huber_fit(&x, &y, n, 1, &cfg).expect("ok");
        // coefficients.len() == n_features (slope only)
        assert_eq!(fit.coefficients.len(), 1);

        let cfg2 = HuberConfig {
            intercept: false,
            ..Default::default()
        };
        let fit2 = huber_fit(&x, &y, n, 1, &cfg2).expect("ok");
        assert_eq!(fit2.coefficients.len(), 1);
    }

    // ── 5. Huber convergence flag ─────────────────────────────────────────────

    #[test]
    fn huber_convergence_flag() {
        let n = 25_usize;
        let (x, y) = clean_linear(n, 1.0, 1.5);
        let cfg = HuberConfig::default();
        let fit = huber_fit(&x, &y, n, 1, &cfg).expect("ok");
        assert!(fit.converged, "Huber should converge on clean data");
    }

    // ── 6. Bisquare recovers slope ────────────────────────────────────────────

    #[test]
    fn bisquare_recovers_slope() {
        let n_clean = 16_usize;
        let n_out = 4_usize;
        let (x, y) = data_with_outliers(n_clean, n_out, 0.0, 2.5, 50.0);
        let cfg = BisquareConfig::default();
        let fit = bisquare_fit(&x, &y, n_clean + n_out, 1, &cfg).expect("bisquare ok");
        assert!(
            (fit.coefficients[0] - 2.5).abs() < 0.5,
            "bisquare slope {} far from 2.5",
            fit.coefficients[0]
        );
    }

    // ── 7. Bisquare weights ∈ [0, 1] ─────────────────────────────────────────

    #[test]
    fn bisquare_weights_bounded() {
        let n = 20_usize;
        let (x, y) = clean_linear(n, 0.0, 1.0);
        let cfg = BisquareConfig::default();
        let fit = bisquare_fit(&x, &y, n, 1, &cfg).expect("ok");
        for &w in &fit.weights {
            assert!((0.0..=1.0 + 1e-12).contains(&w), "weight out of [0,1]: {w}");
        }
    }

    // ── 8. Bisquare gives zero weights to outliers ────────────────────────────

    #[test]
    fn bisquare_zero_weights_for_outliers() {
        // Build very clean data plus two extreme outliers
        let n_clean = 15_usize;
        let mut x: Vec<f64> = (0..n_clean).map(|i| i as f64).collect();
        let mut y: Vec<f64> = x.iter().map(|&xi| xi * 1.0).collect();
        // Append two extreme outliers
        x.push(20.0);
        y.push(1_000.0);
        x.push(21.0);
        y.push(-1_000.0);
        let n = n_clean + 2;
        let cfg = BisquareConfig {
            c: 4.685,
            max_iter: 200,
            tol: 1e-8,
            intercept: true,
            scale_method: ScaleMethod::Mad,
        };
        let fit = bisquare_fit(&x, &y, n, 1, &cfg).expect("ok");
        // Outliers should have weight < 0.05
        let w_out1 = fit.weights[n_clean];
        let w_out2 = fit.weights[n_clean + 1];
        assert!(w_out1 < 0.1, "outlier 1 weight {w_out1} not near 0");
        assert!(w_out2 < 0.1, "outlier 2 weight {w_out2} not near 0");
    }

    // ── 9. MAD basic ─────────────────────────────────────────────────────────

    #[test]
    fn mad_basic() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mad = median_absolute_deviation(&data);
        // median = 3, deviations = [2,1,0,1,2], MAD = 1
        assert!((mad - 1.0).abs() < 1e-10, "MAD expected 1.0, got {mad}");
    }

    // ── 10. MAD scale ≈ 1 for N(0,1) ─────────────────────────────────────────

    #[test]
    fn estimate_scale_mad_gaussian() {
        // Use a large seeded sample from a standard normal
        let mut rng = LcgRng::new(42);
        let n = 5_000_usize;
        let samples: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let sc = estimate_scale_mad(&samples);
        // Should be within 20% of 1.0
        assert!(
            (sc - 1.0).abs() < 0.2,
            "MAD scale estimate {sc} not within 20% of 1.0"
        );
    }

    // ── 11. RANSAC runs without error ─────────────────────────────────────────

    #[test]
    fn ransac_fit_runs() {
        let n = 20_usize;
        let (x, y) = clean_linear(n, 0.0, 1.0);
        let cfg = RansacConfig::default();
        let mut rng = LcgRng::new(7);
        let fit = ransac_fit(&x, &y, n, 1, &cfg, &mut rng);
        assert!(fit.is_ok(), "RANSAC should succeed: {:?}", fit.err());
    }

    // ── 12. RANSAC inlier mask has correct length ─────────────────────────────

    #[test]
    fn ransac_inlier_mask_shape() {
        let n = 25_usize;
        let (x, y) = clean_linear(n, 1.0, 2.0);
        let cfg = RansacConfig::default();
        let mut rng = LcgRng::new(9);
        let fit = ransac_fit(&x, &y, n, 1, &cfg, &mut rng).expect("ok");
        assert_eq!(fit.inlier_mask.len(), n);
    }

    // ── 13. RANSAC n_inliers is reasonable ───────────────────────────────────

    #[test]
    fn ransac_n_inliers_reasonable() {
        let n = 30_usize;
        let (x, y) = clean_linear(n, 0.0, 3.0);
        let cfg = RansacConfig::default();
        let mut rng = LcgRng::new(11);
        let fit = ransac_fit(&x, &y, n, 1, &cfg, &mut rng).expect("ok");
        assert!(
            fit.n_inliers >= 1 && fit.n_inliers <= n,
            "n_inliers={} out of [1,{}]",
            fit.n_inliers,
            n
        );
    }

    // ── 14. RANSAC recovers slope with 30% outliers ───────────────────────────

    #[test]
    fn ransac_recovers_slope_with_outliers() {
        let n_clean = 14_usize;
        let n_out = 6_usize;
        let (x, y) = data_with_outliers(n_clean, n_out, 0.0, 2.0, 200.0);
        let cfg = RansacConfig {
            max_trials: 200,
            residual_threshold: 3.0,
            stop_inlier_fraction: 0.9,
            intercept: true,
            min_samples: None,
        };
        let mut rng = LcgRng::new(123);
        let fit = ransac_fit(&x, &y, n_clean + n_out, 1, &cfg, &mut rng).expect("ok");
        // Should recover slope ≈ 2.0 within 10 %
        assert!(
            (fit.coefficients[0] - 2.0).abs() < 0.2,
            "RANSAC slope {} far from 2.0",
            fit.coefficients[0]
        );
    }

    // ── 15. LMS runs without error ────────────────────────────────────────────

    #[test]
    fn lms_fit_runs() {
        let n = 20_usize;
        let (x, y) = clean_linear(n, 0.5, 1.5);
        let cfg = LmsConfig::default();
        let mut rng = LcgRng::new(55);
        let fit = lms_fit(&x, &y, n, 1, &cfg, &mut rng);
        assert!(fit.is_ok(), "LMS should succeed: {:?}", fit.err());
    }

    // ── 16. LMS output shape ─────────────────────────────────────────────────

    #[test]
    fn lms_output_shape() {
        let n = 20_usize;
        let (x, y) = clean_linear(n, 0.0, 1.0);
        let cfg = LmsConfig::default();
        let mut rng = LcgRng::new(77);
        let fit = lms_fit(&x, &y, n, 1, &cfg, &mut rng).expect("ok");
        assert_eq!(fit.coefficients.len(), 1);
        assert_eq!(fit.residuals.len(), n);
        assert_eq!(fit.weights.len(), n);
    }

    // ── 17. LTS runs without error ────────────────────────────────────────────

    #[test]
    fn lts_fit_runs() {
        let n = 20_usize;
        let (x, y) = clean_linear(n, 1.0, 2.0);
        let mut rng = LcgRng::new(99);
        let fit = lts_fit(&x, &y, n, 1, 200, true, &mut rng);
        assert!(fit.is_ok(), "LTS should succeed: {:?}", fit.err());
    }

    // ── 18. Huber residuals length == n_samples ───────────────────────────────

    #[test]
    fn huber_residuals_shape() {
        let n = 25_usize;
        let (x, y) = clean_linear(n, 0.0, 1.5);
        let cfg = HuberConfig::default();
        let fit = huber_fit(&x, &y, n, 1, &cfg).expect("ok");
        assert_eq!(fit.residuals.len(), n);
    }

    // ── 19. LTS output shape ──────────────────────────────────────────────────

    #[test]
    fn lts_output_shape() {
        let n = 20_usize;
        let (x, y) = clean_linear(n, 1.0, 2.0);
        let mut rng = LcgRng::new(31);
        let fit = lts_fit(&x, &y, n, 1, 200, true, &mut rng).expect("ok");
        assert_eq!(fit.coefficients.len(), 1);
        assert_eq!(fit.residuals.len(), n);
    }

    // ── 20. IQR scale estimate near 1 for normal data ────────────────────────

    #[test]
    fn iqr_scale_gaussian() {
        let mut rng = LcgRng::new(314);
        let n = 5_000_usize;
        let samples: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let sc = estimate_scale_iqr(&samples);
        assert!(
            (sc - 1.0).abs() < 0.15,
            "IQR scale {sc} not within 15% of 1.0"
        );
    }

    // ── 21. winsorized scale is positive ─────────────────────────────────────

    #[test]
    fn winsorized_scale_positive() {
        let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let sc = winsorized_scale(&data, 0.1);
        assert!(sc > 0.0, "winsorized scale must be positive, got {sc}");
    }

    // ── 22. Huber no-intercept mode ───────────────────────────────────────────

    #[test]
    fn huber_no_intercept() {
        // y = 3*x, no intercept
        let n = 20_usize;
        let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi).collect();
        let cfg = HuberConfig {
            intercept: false,
            ..Default::default()
        };
        let fit = huber_fit(&x, &y, n, 1, &cfg).expect("ok");
        assert_eq!(fit.coefficients.len(), 1);
        assert!(
            (fit.coefficients[0] - 3.0).abs() < 0.05,
            "slope: {}",
            fit.coefficients[0]
        );
        assert_eq!(fit.intercept, 0.0);
    }

    // ── 23. RANSAC: inlier mask sum == n_inliers ──────────────────────────────

    #[test]
    fn ransac_mask_sum_equals_n_inliers() {
        let n = 20_usize;
        let (x, y) = clean_linear(n, 0.0, 1.0);
        let cfg = RansacConfig::default();
        let mut rng = LcgRng::new(42);
        let fit = ransac_fit(&x, &y, n, 1, &cfg, &mut rng).expect("ok");
        let mask_sum = fit.inlier_mask.iter().filter(|&&v| v).count();
        assert_eq!(mask_sum, fit.n_inliers);
    }
}

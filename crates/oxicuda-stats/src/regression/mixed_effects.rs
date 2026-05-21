//! Linear Mixed-Effects Models (LMM) via EM algorithm (Henderson's equations).
//!
//! Model: `y = Xβ + Zb + ε` where
//!   - X ∈ ℝ^{n×p}  fixed-effects design matrix (intercept prepended when `intercept=true`)
//!   - β ∈ ℝ^p       fixed effects
//!   - Z ∈ ℝ^{n×q}  group-indicator random-effects design matrix  (Z_{ig}=1 iff obs i ∈ group g)
//!   - b ~ N(0, σ²_b I)  random effects (one per group)
//!   - ε ~ N(0, σ²ε I)   residual errors
//!
//! Estimation via EM (Bates et al. 2015 / lme4 approach):
//!   E-step: b̂ = (Z^T Z + λI)^{-1} Z^T r,   λ = σ²ε / σ²_b,  r = y − Xβ
//!   M-step β:  OLS of (y − Zb̂) on X
//!   M-step σ²_b, σ²ε: ML or REML moment estimates
//!
//! Reference: Bates, Mächler, Bolker, Walker (2015), *Fitting Linear Mixed-Effects
//! Models Using lme4*, J. Stat. Software 67(1).

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::matrix_inverse_lu;

// ─────────────────────────────── Configuration ───────────────────────────────

/// Configuration for LMM fitting via EM.
#[derive(Debug, Clone)]
pub struct LmmConfig {
    /// Maximum EM iterations (default 100).
    pub max_iter: usize,
    /// Parameter convergence tolerance on Euclidean norm of Δ(β, σ²_b, σ²ε) (default 1e-6).
    pub tol: f64,
    /// Use Restricted Maximum Likelihood (REML) variance estimation (default true).
    pub reml: bool,
    /// Prepend an intercept column to X (default true).
    pub intercept: bool,
}

impl Default for LmmConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-6,
            reml: true,
            intercept: true,
        }
    }
}

// ───────────────────────────────── Input data ─────────────────────────────────

/// Input dataset for LMM.
///
/// `x` is row-major `[n_samples × n_features]` (raw features, **without** intercept column;
/// the intercept is added automatically when `LmmConfig::intercept = true`).
#[derive(Debug, Clone)]
pub struct LmmData {
    /// Fixed-effects raw design [n_samples × n_features] row-major (no intercept column).
    pub x: Vec<f64>,
    /// Response vector `[n_samples]`.
    pub y: Vec<f64>,
    /// Group membership `[n_samples]`, values in `0..n_groups`.
    pub groups: Vec<usize>,
    pub n_samples: usize,
    pub n_features: usize,
    pub n_groups: usize,
}

// ──────────────────────────────── Fitted model ────────────────────────────────

/// Fitted LMM.
#[derive(Debug, Clone)]
pub struct LmmFit {
    /// Fixed effects β, length = p (includes intercept column when `intercept=true`).
    pub beta: Vec<f64>,
    /// Posterior mode random effects b̂, length = n_groups.
    pub b_hat: Vec<f64>,
    /// Random-effect variance σ²_b.
    pub sigma_sq_b: f64,
    /// Residual variance σ²ε.
    pub sigma_sq_e: f64,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Number of EM iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
    /// Residuals `y − ŷ` `[n_samples]`.
    pub residuals: Vec<f64>,
    /// Fitted values ŷ `[n_samples]`.
    pub fitted: Vec<f64>,
    /// Number of fixed-effect columns p (including intercept if applicable).
    pub(crate) n_fixed: usize,
    /// Whether intercept was included in the model.
    pub(crate) has_intercept: bool,
    /// Number of groups q.
    pub(crate) n_groups: usize,
}

// ─────────────────────── Internal linear algebra helpers ─────────────────────

/// Compute `A^T A` for A [m × n] → result [n × n].
fn ata(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let mut acc = 0.0;
            for k in 0..m {
                acc += a[k * n + i] * a[k * n + j];
            }
            out[i * n + j] = acc;
            out[j * n + i] = acc;
        }
    }
    out
}

/// Compute `A^T v` for A [m × n], v [m] → result [n].
fn at_vec(a: &[f64], v: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for j in 0..n {
        let mut acc = 0.0;
        for i in 0..m {
            acc += a[i * n + j] * v[i];
        }
        out[j] = acc;
    }
    out
}

/// Compute `A v` for A [m × n], v [n] → result [m].
fn mat_vec(a: &[f64], v: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m];
    for i in 0..m {
        let mut acc = 0.0;
        for j in 0..n {
            acc += a[i * n + j] * v[j];
        }
        out[i] = acc;
    }
    out
}

/// Add λ to diagonal of a square matrix in-place.
fn add_diagonal(m: &mut [f64], n: usize, lambda: f64) {
    for i in 0..n {
        m[i * n + i] += lambda;
    }
}

/// Euclidean norm of a slice.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ──────────────────────────── Design matrix builder ──────────────────────────

/// Build the full fixed-effects design matrix X_full [n × p] (row-major), possibly
/// prepending an intercept column.
fn build_x_full(data: &LmmData, intercept: bool) -> Vec<f64> {
    let n = data.n_samples;
    let p_raw = data.n_features;
    let p = if intercept { p_raw + 1 } else { p_raw };
    let mut xf = vec![0.0; n * p];
    for i in 0..n {
        let mut col = 0usize;
        if intercept {
            xf[i * p + col] = 1.0;
            col += 1;
        }
        for j in 0..p_raw {
            xf[i * p + col + j] = data.x[i * p_raw + j];
        }
    }
    xf
}

/// Build the group-indicator Z [n × q] (row-major).
///
/// Useful for downstream diagnostics or custom predictions.
pub fn build_z(data: &LmmData) -> Vec<f64> {
    let n = data.n_samples;
    let q = data.n_groups;
    let mut z = vec![0.0; n * q];
    for i in 0..n {
        let g = data.groups[i];
        z[i * q + g] = 1.0;
    }
    z
}

// ─────────────────── Log-likelihood / information criteria ───────────────────

/// Compute the ML log-likelihood of the fitted model.
///
/// Under model N(Xβ + Zb̂, σ²ε I) the conditional log-likelihood is:
/// ℓ = −n/2 · log(2π σ²ε) − ||y − Xβ − Zb̂||² / (2 σ²ε)
/// minus the KL divergence term from the random effects prior:
/// − q/2 · log(2π σ²_b) − b̂^T b̂ / (2 σ²_b)
fn compute_log_likelihood(
    resid: &[f64],
    b_hat: &[f64],
    sigma_sq_e: f64,
    sigma_sq_b: f64,
    n: usize,
    q: usize,
) -> f64 {
    use std::f64::consts::PI;
    let rss: f64 = resid.iter().map(|r| r * r).sum();
    let b_sq: f64 = b_hat.iter().map(|b| b * b).sum();
    let ll_eps = -(n as f64) / 2.0 * (2.0 * PI * sigma_sq_e).ln() - rss / (2.0 * sigma_sq_e);
    let ll_b = -(q as f64) / 2.0 * (2.0 * PI * sigma_sq_b).ln() - b_sq / (2.0 * sigma_sq_b);
    ll_eps + ll_b
}

// ──────────────────────────────── Main fitter ─────────────────────────────────

/// Fit a Linear Mixed-Effects Model via the EM algorithm.
///
/// # Errors
/// Returns `Err` on empty input, mismatched dimensions, or singular systems.
pub fn lmm_fit(data: &LmmData, cfg: &LmmConfig) -> StatsResult<LmmFit> {
    let n = data.n_samples;
    let q = data.n_groups;

    // ── Input validation ──────────────────────────────────────────────────────
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if data.y.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: data.y.len(),
            b: n,
        });
    }
    if data.groups.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: data.groups.len(),
            b: n,
        });
    }
    if data.x.len() != n * data.n_features {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n, data.n_features],
            got: vec![data.x.len()],
        });
    }
    if q == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_groups".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    for (i, &g) in data.groups.iter().enumerate() {
        if g >= q {
            return Err(StatsError::IndexOutOfBounds { index: g, len: q });
        }
        let _ = i;
    }
    for (i, &v) in data.y.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    // ── Build design matrices ─────────────────────────────────────────────────
    let x_full = build_x_full(data, cfg.intercept);
    let p = if cfg.intercept {
        data.n_features + 1
    } else {
        data.n_features
    };
    // Z (group indicator) is used implicitly via group membership indexing;
    // build_z is available for callers but the EM uses the diagonal Z^T Z directly.

    // Pre-compute Z^T Z (diagonal for indicator Z: Z_g^T Z_g = n_g)
    let mut n_g = vec![0usize; q]; // group counts
    for &g in &data.groups {
        n_g[g] += 1;
    }
    // Z^T Z is diagonal with diagonal = n_g (counts per group)
    let ztg_diag: Vec<f64> = n_g.iter().map(|&c| c as f64).collect();

    // X^T X (p × p)
    let xtx = ata(&x_full, n, p);

    // ── Initialise parameters ─────────────────────────────────────────────────
    let y_mean = data.y.iter().sum::<f64>() / n as f64;
    let y_var = data.y.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>() / n as f64;
    let mut sigma_sq_e = (y_var * 0.5).max(1e-6);
    let mut sigma_sq_b = (y_var * 0.5).max(1e-6);

    let mut beta = vec![0.0; p];
    // Initialise β[intercept] to y_mean
    if cfg.intercept && p > 0 {
        beta[0] = y_mean;
    }
    let mut b_hat = vec![0.0; q];
    let mut n_iter = 0usize;
    let mut converged = false;

    for _em_iter in 0..cfg.max_iter {
        n_iter += 1;

        // ── E-step: posterior mean of b ───────────────────────────────────────
        // r = y − Xβ  [n]
        let mut r = data.y.clone();
        let xb = mat_vec(&x_full, &beta, n, p);
        for i in 0..n {
            r[i] -= xb[i];
        }

        // λ = σ²ε / σ²_b
        let lambda = sigma_sq_e / sigma_sq_b;

        // D = Z^T Z + λ I  (diagonal, d_g = n_g + λ)
        // D^{-1} is also diagonal: 1 / (n_g + λ)
        let d_diag: Vec<f64> = ztg_diag.iter().map(|&ng| ng + lambda).collect();

        // b̂ = D^{-1} Z^T r
        // Z^T r [q]: (Z^T r)_g = Σ_{i: g_i=g} r_i
        let mut zt_r = vec![0.0; q];
        for i in 0..n {
            zt_r[data.groups[i]] += r[i];
        }
        for g in 0..q {
            b_hat[g] = zt_r[g] / d_diag[g];
        }

        // ── M-step: update β via OLS on (y − Zb̂) ────────────────────────────
        // z_b_hat = Z b̂ [n]
        let mut zb = vec![0.0; n];
        for i in 0..n {
            zb[i] = b_hat[data.groups[i]];
        }

        // r_fixed = y − Zb̂  [n]
        let r_fixed: Vec<f64> = data.y.iter().zip(&zb).map(|(y, z)| y - z).collect();

        // β = (X^T X)^{-1} X^T r_fixed
        let xty: Vec<f64> = at_vec(&x_full, &r_fixed, n, p);

        // Add tiny ridge to stabilise
        let mut xtx_reg = xtx.clone();
        add_diagonal(&mut xtx_reg, p, 1e-10);
        let xtx_inv = matrix_inverse_lu(&xtx_reg, p)?;

        let beta_new: Vec<f64> = {
            let mut b = vec![0.0; p];
            for i in 0..p {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += xtx_inv[i * p + j] * xty[j];
                }
                b[i] = acc;
            }
            b
        };

        // ── M-step: update variance components ───────────────────────────────
        // Var_post(b) = σ²ε · D^{-1}  (diagonal: σ²ε / (n_g + λ))
        // tr(Var_post(b)) = Σ_g σ²ε / (n_g + λ)
        let tr_var_b: f64 = d_diag.iter().map(|&d| sigma_sq_e / d).sum();

        // σ²_b new: (b̂^T b̂ + tr(Var_post(b))) / q
        let b_sq: f64 = b_hat.iter().map(|b| b * b).sum();
        let sigma_sq_b_new = (b_sq + tr_var_b) / q as f64;

        // Full residual r2 = y − Xβ_new − Zb̂  [n]
        let xb_new = mat_vec(&x_full, &beta_new, n, p);
        let resid_full: Vec<f64> = data
            .y
            .iter()
            .enumerate()
            .map(|(i, &y)| y - xb_new[i] - b_hat[data.groups[i]])
            .collect();
        let rss_full: f64 = resid_full.iter().map(|r| r * r).sum();

        // tr(Z Var_post(b) Z^T) = Σ_g σ²ε · n_g / (n_g + λ)
        let tr_z_var_zt: f64 = (0..q).map(|g| sigma_sq_e * ztg_diag[g] / d_diag[g]).sum();

        let denom_e = if cfg.reml {
            // REML: degrees of freedom correction
            ((n - p) as f64).max(1.0)
        } else {
            n as f64
        };
        let sigma_sq_e_new = (rss_full + tr_z_var_zt) / denom_e;

        // Clamp to avoid collapse
        let sigma_sq_b_new = sigma_sq_b_new.max(1e-10);
        let sigma_sq_e_new = sigma_sq_e_new.max(1e-10);

        // ── Convergence check ─────────────────────────────────────────────────
        let mut delta = vec![0.0; p + 2];
        for j in 0..p {
            delta[j] = beta_new[j] - beta[j];
        }
        delta[p] = sigma_sq_b_new - sigma_sq_b;
        delta[p + 1] = sigma_sq_e_new - sigma_sq_e;

        beta = beta_new;
        sigma_sq_b = sigma_sq_b_new;
        sigma_sq_e = sigma_sq_e_new;

        if l2_norm(&delta) < cfg.tol {
            converged = true;
            break;
        }
    }

    // ── Compute final diagnostics ─────────────────────────────────────────────
    let fitted: Vec<f64> = {
        let xb = mat_vec(&x_full, &beta, n, p);
        (0..n).map(|i| xb[i] + b_hat[data.groups[i]]).collect()
    };
    let residuals: Vec<f64> = data.y.iter().zip(&fitted).map(|(y, f)| y - f).collect();

    let log_likelihood = compute_log_likelihood(&residuals, &b_hat, sigma_sq_e, sigma_sq_b, n, q);

    // Number of parameters: p fixed effects + σ²_b + σ²ε = p + 2
    let n_params = (p + 2) as f64;
    let aic = -2.0 * log_likelihood + 2.0 * n_params;
    let bic = -2.0 * log_likelihood + n_params * (n as f64).ln();

    Ok(LmmFit {
        beta,
        b_hat,
        sigma_sq_b,
        sigma_sq_e,
        log_likelihood,
        aic,
        bic,
        n_iter,
        converged,
        residuals,
        fitted,
        n_fixed: p,
        has_intercept: cfg.intercept,
        n_groups: q,
    })
}

// ─────────────────────────────── Prediction ───────────────────────────────────

/// Predict on new data, optionally including group random effects.
///
/// `x_new` is row-major `[n_new × n_features]` (raw features, **no intercept column**).
/// `groups_new[i]` must be a group index seen during training (`0..n_groups`).
///
/// The prediction is `ŷ_i = X_new_i β + b̂_{g_i}`.
pub fn lmm_predict(
    fit: &LmmFit,
    x_new: &[f64],
    groups_new: &[usize],
    n_new: usize,
) -> StatsResult<Vec<f64>> {
    if n_new == 0 {
        return Ok(Vec::new());
    }
    // Number of raw features (excluding intercept)
    let n_raw = if fit.has_intercept {
        fit.n_fixed - 1
    } else {
        fit.n_fixed
    };
    if x_new.len() != n_new * n_raw {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_new, n_raw],
            got: vec![x_new.len()],
        });
    }
    if groups_new.len() != n_new {
        return Err(StatsError::DimensionMismatch {
            a: groups_new.len(),
            b: n_new,
        });
    }
    for (i, &g) in groups_new.iter().enumerate() {
        if g >= fit.n_groups {
            return Err(StatsError::IndexOutOfBounds {
                index: g,
                len: fit.n_groups,
            });
        }
        let _ = i;
    }

    let p = fit.n_fixed;
    let mut preds = vec![0.0; n_new];
    for i in 0..n_new {
        let mut eta = 0.0;
        let mut col = 0usize;
        if fit.has_intercept {
            eta += fit.beta[0];
            col = 1;
        }
        for j in 0..n_raw {
            eta += fit.beta[col + j] * x_new[i * n_raw + j];
        }
        eta += fit.b_hat[groups_new[i]];
        let _ = p;
        preds[i] = eta;
    }
    Ok(preds)
}

// ─────────────────────────── Derived quantities ───────────────────────────────

/// Intraclass Correlation Coefficient (ICC):
/// ICC = σ²_b / (σ²_b + σ²ε).
///
/// Measures the fraction of total variance attributable to between-group differences.
#[must_use]
pub fn lmm_icc(fit: &LmmFit) -> f64 {
    let total = fit.sigma_sq_b + fit.sigma_sq_e;
    if total < 1e-300 {
        return 0.0;
    }
    (fit.sigma_sq_b / total).clamp(0.0, 1.0)
}

/// Return per-group residuals: `residuals_by_group[g]` is the vector of residuals
/// for all observations belonging to group `g` (in observation order).
#[must_use]
pub fn lmm_residuals_by_group(fit: &LmmFit, data: &LmmData) -> Vec<Vec<f64>> {
    let q = fit.n_groups;
    let mut out: Vec<Vec<f64>> = vec![Vec::new(); q];
    for i in 0..data.n_samples {
        let g = data.groups[i];
        if g < q {
            out[g].push(fit.residuals[i]);
        }
    }
    out
}

// ──────────────────────────────────── Tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple balanced dataset: `n_groups` groups × `obs_per_group` observations,
    /// with `y = intercept + x1 * b1 + b_group + noise_free`.
    fn make_dataset(
        n_groups: usize,
        obs_per_group: usize,
        fixed_beta: &[f64], // [intercept, slope]
        group_effects: &[f64],
    ) -> LmmData {
        let n = n_groups * obs_per_group;
        let n_features = fixed_beta.len() - 1; // raw features (no intercept)
        let mut x = Vec::with_capacity(n * n_features);
        let mut y = Vec::with_capacity(n);
        let mut groups = Vec::with_capacity(n);

        for (g, &group_effect) in group_effects.iter().enumerate().take(n_groups) {
            for obs in 0..obs_per_group {
                let x_val = (obs as f64 + 1.0) * 0.5;
                // raw X (without intercept)
                x.push(x_val);
                // y = intercept + slope * x + group_effect (noise-free)
                let yval = fixed_beta[0] + fixed_beta[1] * x_val + group_effect;
                y.push(yval);
                groups.push(g);
            }
        }
        LmmData {
            x,
            y,
            groups,
            n_samples: n,
            n_features,
            n_groups,
        }
    }

    // 1. Basic fitting runs without panic/error
    #[test]
    fn lmm_fit_runs() {
        let group_effects = [1.0, -1.0, 0.5];
        let data = make_dataset(3, 5, &[2.0, 0.8], &group_effects);
        let cfg = LmmConfig::default();
        let fit = lmm_fit(&data, &cfg);
        assert!(fit.is_ok(), "lmm_fit should return Ok: {:?}", fit);
    }

    // 2. Converged flag is set on simple data
    #[test]
    fn lmm_converges() {
        let group_effects = [0.5, -0.5, 1.0, -1.0];
        let data = make_dataset(4, 6, &[1.0, 0.5], &group_effects);
        let cfg = LmmConfig {
            max_iter: 200,
            tol: 1e-8,
            ..LmmConfig::default()
        };
        let fit = lmm_fit(&data, &cfg).expect("fit ok");
        assert!(
            fit.converged,
            "should converge on clean data (n_iter={})",
            fit.n_iter
        );
    }

    // 3. beta has correct length (n_features + 1 for intercept)
    #[test]
    fn lmm_fixed_effects_shape() {
        let data = make_dataset(3, 5, &[2.0, 0.8], &[1.0, -1.0, 0.5]);
        let cfg = LmmConfig {
            intercept: true,
            ..LmmConfig::default()
        };
        let fit = lmm_fit(&data, &cfg).expect("ok");
        // n_features = 1 raw + 1 intercept = 2
        assert_eq!(fit.beta.len(), data.n_features + 1);
    }

    // 4. ICC in [0, 1]
    #[test]
    fn lmm_icc_range() {
        let data = make_dataset(4, 5, &[1.0, 1.0], &[2.0, 0.0, -2.0, 1.0]);
        let cfg = LmmConfig::default();
        let fit = lmm_fit(&data, &cfg).expect("ok");
        let icc = lmm_icc(&fit);
        assert!((0.0..=1.0).contains(&icc), "ICC={icc} out of range");
    }

    // 5. Prediction returns n_new values
    #[test]
    fn lmm_predict_shape() {
        let data = make_dataset(3, 5, &[1.0, 0.5], &[0.3, -0.3, 0.1]);
        let cfg = LmmConfig::default();
        let fit = lmm_fit(&data, &cfg).expect("ok");
        let x_new = vec![1.0, 1.5, 2.0]; // 3 new observations, 1 raw feature each
        let groups_new = vec![0usize, 1, 2];
        let preds = lmm_predict(&fit, &x_new, &groups_new, 3).expect("ok");
        assert_eq!(preds.len(), 3);
    }

    // 6. On noise-free data fitted values should be very close to y
    #[test]
    fn lmm_fitted_close_to_y() {
        let group_effects = [3.0, -3.0, 1.5, -1.5];
        let data = make_dataset(4, 8, &[0.0, 2.0], &group_effects);
        let cfg = LmmConfig {
            max_iter: 300,
            tol: 1e-10,
            reml: false,
            ..LmmConfig::default()
        };
        let fit = lmm_fit(&data, &cfg).expect("ok");
        let max_resid = fit
            .residuals
            .iter()
            .map(|r| r.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_resid < 0.5,
            "max residual {max_resid:.4} too large on noise-free data"
        );
    }

    // 7. Both variance components are strictly positive
    #[test]
    fn lmm_sigma_sq_positive() {
        let data = make_dataset(3, 5, &[1.0, 1.0], &[1.0, -1.0, 0.0]);
        let cfg = LmmConfig::default();
        let fit = lmm_fit(&data, &cfg).expect("ok");
        assert!(fit.sigma_sq_b > 0.0, "σ²_b should be > 0");
        assert!(fit.sigma_sq_e > 0.0, "σ²ε should be > 0");
    }

    // 8. residuals_by_group returns one Vec per group
    #[test]
    fn lmm_residuals_by_group_shape() {
        let n_groups = 3;
        let obs_per = 5;
        let data = make_dataset(n_groups, obs_per, &[1.0, 0.5], &[0.5, -0.5, 0.0]);
        let cfg = LmmConfig::default();
        let fit = lmm_fit(&data, &cfg).expect("ok");
        let by_group = lmm_residuals_by_group(&fit, &data);
        assert_eq!(by_group.len(), n_groups);
        for (g, group_residuals) in by_group.iter().enumerate().take(n_groups) {
            assert_eq!(
                group_residuals.len(),
                obs_per,
                "group {g} has wrong obs count"
            );
        }
    }

    // 9. Random effects b̂ should have correct signs on structured data
    //    Groups with positive shifts → positive b̂; negative shifts → negative b̂
    #[test]
    fn lmm_group_effect_recovered() {
        // Clear group effects: +5, 0, -5 with many observations per group
        let group_effects = [5.0, 0.0, -5.0];
        let data = make_dataset(3, 20, &[0.0, 0.0], &group_effects);
        let cfg = LmmConfig {
            max_iter: 500,
            tol: 1e-10,
            reml: false,
            ..LmmConfig::default()
        };
        let fit = lmm_fit(&data, &cfg).expect("ok");
        assert!(
            fit.b_hat[0] > 0.0,
            "group 0 b̂ should be > 0 (got {})",
            fit.b_hat[0]
        );
        assert!(
            fit.b_hat[2] < 0.0,
            "group 2 b̂ should be < 0 (got {})",
            fit.b_hat[2]
        );
    }

    // 10. Empty data returns Err
    #[test]
    fn lmm_empty_data_error() {
        let data = LmmData {
            x: vec![],
            y: vec![],
            groups: vec![],
            n_samples: 0,
            n_features: 2,
            n_groups: 3,
        };
        let cfg = LmmConfig::default();
        let result = lmm_fit(&data, &cfg);
        assert!(result.is_err(), "empty data should return Err");
    }

    // 11. ICC near 1 when group variance >> residual
    #[test]
    fn lmm_icc_high_when_group_dominant() {
        // Very large group effects, zero within-group variation
        let group_effects = [100.0, -100.0, 50.0, -50.0];
        let data = make_dataset(4, 10, &[0.0, 0.0], &group_effects);
        let cfg = LmmConfig {
            reml: false,
            max_iter: 300,
            tol: 1e-10,
            ..LmmConfig::default()
        };
        let fit = lmm_fit(&data, &cfg).expect("ok");
        let icc = lmm_icc(&fit);
        // ICC should be > 0.5 given dominant group effects
        assert!(
            icc > 0.5,
            "ICC={icc:.4} should be high when group effects dominate"
        );
    }

    // 12. Predict with out-of-range group index returns error
    #[test]
    fn lmm_predict_bad_group_returns_error() {
        let data = make_dataset(3, 5, &[1.0, 0.5], &[1.0, 0.0, -1.0]);
        let fit = lmm_fit(&data, &LmmConfig::default()).expect("ok");
        // group index 99 is out of range
        let x_new = vec![1.0];
        let groups_new = vec![99usize];
        let result = lmm_predict(&fit, &x_new, &groups_new, 1);
        assert!(result.is_err(), "out-of-range group should return Err");
    }

    // 13. AIC, BIC are finite and AIC < BIC for large n
    #[test]
    fn lmm_information_criteria_finite() {
        let data = make_dataset(4, 20, &[1.0, 2.0], &[1.0, -1.0, 2.0, -2.0]);
        let cfg = LmmConfig::default();
        let fit = lmm_fit(&data, &cfg).expect("ok");
        assert!(fit.aic.is_finite(), "AIC should be finite");
        assert!(fit.bic.is_finite(), "BIC should be finite");
        // For large n, BIC penalizes more than AIC
        assert!(fit.bic >= fit.aic, "BIC >= AIC for n ≥ e^2 ≈ 7");
    }

    // 14. No-intercept model produces beta of correct length
    #[test]
    fn lmm_no_intercept_shape() {
        let data = make_dataset(2, 5, &[0.0, 1.0], &[0.5, -0.5]);
        let cfg = LmmConfig {
            intercept: false,
            ..LmmConfig::default()
        };
        let fit = lmm_fit(&data, &cfg).expect("ok");
        assert_eq!(
            fit.beta.len(),
            data.n_features,
            "no-intercept: beta.len() = n_features"
        );
    }
}

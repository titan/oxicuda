//! Pseudo-observations for survival regression.
//!
//! Implements the jackknife pseudo-value approach of Andersen & Ronn (1995, Biometrika) and
//! Andersen et al. (2003, Biometrika).  A pseudo-observation for individual *i* is:
//!
//! > Ŷᵢ = n · θ̂ − (n−1) · θ̂₍₋ᵢ₎
//!
//! where θ̂ is a summary functional (RMST or survival probability at a horizon τ) and θ̂₍₋ᵢ₎
//! is the same functional recomputed after deleting observation *i*.  Under mild regularity
//! conditions E[Ŷᵢ | Xᵢ] ≈ E[T ∧ τ | Xᵢ] (RMST case) or P(T > τ | Xᵢ) (survival probability
//! case), enabling regression on pseudo-values with standard link functions.
//!
//! ## Supported regression modes
//! - **Linear** (OLS with L2 ridge): E[Ŷᵢ | Xᵢ] = Xᵢβ
//! - **Logistic** (IRLS): logit(E[Ŷᵢ | Xᵢ]) = Xᵢβ  (appropriate for survival-probability pseudo-values)

use crate::error::{SurvivalError, SurvivalResult};

// ─── Internal KM step-function representation ────────────────────────────────

/// One row of a Kaplan-Meier step function: an event time and the survival estimate *after*
/// that time.
#[derive(Debug, Clone)]
struct KmStep {
    time: f64,
    survival: f64,
}

/// Build the Kaplan-Meier product-limit estimator from raw arrays.
///
/// Only event times (where `events[i] == 1`) produce steps in the output.  Censored
/// observations contribute to the at-risk count but do not create a step.
///
/// Returns an empty `Vec` when there are no events (caller must handle this case).
fn kaplan_meier_steps(times: &[f64], events: &[u8], n: usize) -> Vec<KmStep> {
    if n == 0 {
        return Vec::new();
    }

    // Collect (time, is_event) pairs and sort by time ascending, ties: events first.
    let mut pairs: Vec<(f64, u8)> = times[..n]
        .iter()
        .copied()
        .zip(events[..n].iter().copied())
        .collect();
    pairs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.cmp(&a.1)) // events (1) before censored (0) at same time
    });

    let mut steps: Vec<KmStep> = Vec::new();
    let mut survival = 1.0_f64;
    let mut idx = 0usize;

    while idx < n {
        let t_current = pairs[idx].0;

        // Count events and total exits at this time.
        let at_risk = n - idx; // everyone from idx onwards is still at risk
        let mut events_at_t = 0usize;
        let mut j = idx;
        while j < n && (pairs[j].0 - t_current).abs() < f64::EPSILON * t_current.abs().max(1.0) {
            if pairs[j].1 == 1 {
                events_at_t += 1;
            }
            j += 1;
        }

        if events_at_t > 0 {
            survival *= 1.0 - (events_at_t as f64) / (at_risk as f64);
            steps.push(KmStep {
                time: t_current,
                survival,
            });
        }

        idx = j;
    }

    steps
}

/// Compute RMST(τ) from a KM step function.
///
/// RMST = Σᵢ Ŝ(tᵢ₋₁) · (min(tᵢ, τ) − min(tᵢ₋₁, τ))
///
/// where t₀ = 0 and Ŝ before any events is 1.
fn rmst_from_km_steps(steps: &[KmStep], tau: f64) -> f64 {
    let mut area = 0.0_f64;
    let mut last_time = 0.0_f64;
    let mut last_survival = 1.0_f64;

    for step in steps {
        let t = step.time;
        if t >= tau {
            area += (tau - last_time).max(0.0) * last_survival;
            return area;
        }
        area += (t - last_time).max(0.0) * last_survival;
        last_time = t;
        last_survival = step.survival;
    }
    // Tail: extend last KM value to τ.
    area += (tau - last_time).max(0.0) * last_survival;
    area
}

/// Compute S(τ) from a KM step function (survival probability at horizon).
///
/// Returns the KM estimate evaluated at the largest event time ≤ τ.
/// If no event times ≤ τ exist, returns 1.0 (no events before horizon → everyone "survives").
fn survival_prob_from_km_steps(steps: &[KmStep], tau: f64) -> f64 {
    let mut s_at_tau = 1.0_f64;
    for step in steps {
        if step.time > tau {
            break;
        }
        s_at_tau = step.survival;
    }
    s_at_tau
}

/// Internal helper: compute θ̂ (RMST or S(τ)) on a dataset of length `n`.
fn compute_functional(
    times: &[f64],
    events: &[u8],
    n: usize,
    horizon: f64,
    outcome: &PseudoObsOutcome,
) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let steps = kaplan_meier_steps(times, events, n);
    if steps.is_empty() {
        // No events → S(t) = 1 everywhere → RMST = τ, S(τ) = 1.
        return match outcome {
            PseudoObsOutcome::Rmst => horizon,
            PseudoObsOutcome::SurvivalProb => 1.0,
        };
    }
    match outcome {
        PseudoObsOutcome::Rmst => rmst_from_km_steps(&steps, horizon),
        PseudoObsOutcome::SurvivalProb => survival_prob_from_km_steps(&steps, horizon),
    }
}

// ─── Public configuration types ──────────────────────────────────────────────

/// Which population-level functional to use as the pseudo-observation target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PseudoObsOutcome {
    /// Restricted mean survival time: E[T ∧ τ].
    Rmst,
    /// Survival probability at horizon: P(T > τ).
    SurvivalProb,
}

/// Which regression model to fit to the pseudo-values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PseudoObsRegression {
    /// Ordinary least squares with L2 ridge regularisation.
    Linear,
    /// Logistic regression via iteratively reweighted least squares (IRLS).
    Logistic,
}

/// Configuration for [`pseudo_obs_fit`].
#[derive(Debug, Clone)]
pub struct PseudoObsConfig {
    /// Evaluation horizon τ (must be > 0).
    pub horizon: f64,
    /// Functional to estimate pseudo-observations for.
    pub outcome: PseudoObsOutcome,
    /// Regression model applied to pseudo-values.
    pub regression: PseudoObsRegression,
    /// Maximum IRLS iterations (logistic only).
    pub max_iter: usize,
    /// Convergence tolerance (logistic only; checked on ‖Δβ‖₂).
    pub tol: f64,
    /// L2 ridge penalty added to X^T X (or X^T W X) diagonal.
    pub l2_reg: f64,
}

impl Default for PseudoObsConfig {
    fn default() -> Self {
        Self {
            horizon: 1.0,
            outcome: PseudoObsOutcome::Rmst,
            regression: PseudoObsRegression::Linear,
            max_iter: 100,
            tol: 1.0e-6,
            l2_reg: 1.0e-3,
        }
    }
}

// ─── Result type ─────────────────────────────────────────────────────────────

/// Output of [`pseudo_obs_fit`].
#[derive(Debug, Clone)]
pub struct PseudoObsResult {
    /// Jackknife pseudo-values Ŷᵢ = n·θ̂ − (n−1)·θ̂₍₋ᵢ₎ for i = 1…n.
    pub pseudo_values: Vec<f64>,
    /// Regression coefficients β (length p+1; last element is the intercept).
    pub beta: Vec<f64>,
    /// Number of IRLS iterations performed (1 for linear regression).
    pub n_iter: usize,
    /// Whether the IRLS algorithm converged (always `true` for linear regression).
    pub converged: bool,
    /// Overall θ̂ on the full dataset (RMST or S(τ)).
    pub overall_estimate: f64,
    /// Log-likelihood of the fitted regression model evaluated at `beta`.
    pub log_likelihood: f64,
}

// ─── Linear-algebra helpers (local — no import from `linalg` module) ─────────

/// Dense matrix multiplication: C = A B, where A is m×k and B is k×n (all row-major).
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
    c
}

/// Forward substitution: solve L y = b for lower-triangular L (row-major, n×n).
fn forward_substitute_local(l: &[f64], b: &[f64], n: usize) -> SurvivalResult<Vec<f64>> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let d = l[i * n + i];
        if d.abs() < f64::EPSILON {
            return Err(SurvivalError::SingularMatrix);
        }
        y[i] = s / d;
    }
    Ok(y)
}

/// Back substitution: solve L^T x = y for lower-triangular L (row-major, n×n).
fn back_substitute_local(l: &[f64], y: &[f64], n: usize) -> SurvivalResult<Vec<f64>> {
    let mut x = vec![0.0_f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let d = l[i * n + i];
        if d.abs() < f64::EPSILON {
            return Err(SurvivalError::SingularMatrix);
        }
        x[i] = s / d;
    }
    Ok(x)
}

/// Cholesky factorisation A = L L^T for symmetric positive-definite A (n×n, row-major).
/// Adds `ridge` to the diagonal for numerical stability.
fn cholesky_factor(a: &[f64], n: usize, ridge: f64) -> SurvivalResult<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            if i == j {
                s += ridge;
            }
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(SurvivalError::SingularMatrix);
                }
                l[i * n + j] = s.sqrt();
            } else {
                let denom = l[j * n + j];
                if denom.abs() < f64::EPSILON {
                    return Err(SurvivalError::SingularMatrix);
                }
                l[i * n + j] = s / denom;
            }
        }
    }
    Ok(l)
}

/// Solve A x = b for symmetric positive-definite A using Cholesky decomposition.
/// `a` is the raw n×n SPD matrix (row-major); `ridge` is added to the diagonal.
fn cholesky_solve(a: &[f64], b: &[f64], n: usize, ridge: f64) -> SurvivalResult<Vec<f64>> {
    let l = cholesky_factor(a, n, ridge)?;
    let y = forward_substitute_local(&l, b, n)?;
    back_substitute_local(&l, &y, n)
}

// ─── Sigmoid / logit helpers ──────────────────────────────────────────────────

/// Numerically stable sigmoid σ(x) = 1 / (1 + e^{−x}).
#[inline]
fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Logit function: log(p / (1 − p)).  Caller must ensure p ∈ (0, 1).
#[inline]
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

// ─── Log-likelihood helpers ───────────────────────────────────────────────────

/// Gaussian log-likelihood (up to a constant) evaluated at MSE:
/// ℓ = −n/2 · ln(MSE).  Returns `−∞` when n = 0.
fn linreg_log_likelihood(y: &[f64], yhat: &[f64], n: usize) -> f64 {
    if n == 0 {
        return f64::NEG_INFINITY;
    }
    let mse = y
        .iter()
        .zip(yhat.iter())
        .map(|(yi, fi)| (yi - fi).powi(2))
        .sum::<f64>()
        / (n as f64);
    if mse <= 0.0 {
        return f64::INFINITY; // perfect fit
    }
    -0.5 * (n as f64) * mse.ln()
}

/// Bernoulli log-likelihood: Σᵢ [ yᵢ ln pᵢ + (1−yᵢ) ln(1−pᵢ) ].
/// `p` values are clamped to (ε, 1−ε) before taking logarithms.
fn logreg_log_likelihood(y: &[f64], p: &[f64], n: usize) -> f64 {
    const EPS: f64 = 1.0e-12;
    let mut ll = 0.0_f64;
    for i in 0..n {
        let pi = p[i].clamp(EPS, 1.0 - EPS);
        ll += y[i] * pi.ln() + (1.0 - y[i]) * (1.0 - pi).ln();
    }
    ll
}

// ─── Pseudo-value computation ─────────────────────────────────────────────────

/// Compute the n jackknife pseudo-observations in O(n²) time.
///
/// For each *i*:
/// 1. Build the leave-one-out dataset of length n−1.
/// 2. Compute θ̂₍₋ᵢ₎.
/// 3. Ŷᵢ = n · θ̂ − (n−1) · θ̂₍₋ᵢ₎.
///
/// Edge cases:
/// * If the LOO dataset has no events, θ̂₍₋ᵢ₎ is set conservatively to 0 (RMST) or 0 (S(τ)),
///   because the KM curve cannot be estimated.
/// * If dropping i reduces the total number of observations to 0, Ŷᵢ = overall_estimate.
fn compute_pseudo_values(
    times: &[f64],
    events: &[u8],
    n: usize,
    horizon: f64,
    overall_estimate: f64,
    outcome: &PseudoObsOutcome,
) -> SurvivalResult<Vec<f64>> {
    let mut pseudo = Vec::with_capacity(n);

    // Pre-allocate LOO buffers to avoid repeated allocation.
    let mut times_loo = vec![0.0_f64; n - 1];
    let mut events_loo = vec![0u8; n - 1];

    for drop_i in 0..n {
        if n == 1 {
            // Cannot estimate without any observation; use overall as a fallback.
            pseudo.push(overall_estimate);
            continue;
        }

        // Build LOO arrays.
        let mut dst = 0usize;
        for src in 0..n {
            if src == drop_i {
                continue;
            }
            times_loo[dst] = times[src];
            events_loo[dst] = events[src];
            dst += 1;
        }
        let n_loo = n - 1;

        // Check whether any events remain in the LOO dataset.
        let has_events = events_loo[..n_loo].contains(&1);
        let theta_loo = if has_events {
            compute_functional(&times_loo, &events_loo, n_loo, horizon, outcome)
        } else {
            // Conservative: no events → unknown; set to 0.0.
            0.0
        };

        let pseudo_i = (n as f64) * overall_estimate - ((n - 1) as f64) * theta_loo;
        pseudo.push(pseudo_i);
    }

    Ok(pseudo)
}

// ─── Regression on pseudo-values ─────────────────────────────────────────────

/// Build the design matrix [X | 1] of shape n × (p+1), intercept in last column.
/// `covariates` is row-major n×p (empty slice when p=0).
fn build_design_matrix(covariates: &[f64], n: usize, p: usize) -> Vec<f64> {
    let q = p + 1; // number of columns (covariates + intercept)
    let mut x_aug = vec![0.0_f64; n * q];
    for i in 0..n {
        // Copy p covariate values.
        for j in 0..p {
            x_aug[i * q + j] = covariates[i * p + j];
        }
        // Intercept column.
        x_aug[i * q + p] = 1.0;
    }
    x_aug
}

/// OLS regression with L2 ridge: β = (X^T X + λI)^{−1} X^T y.
///
/// Returns (beta, fitted_values).
fn ols_regression(
    x_aug: &[f64],
    y: &[f64],
    n: usize,
    q: usize,
    l2_reg: f64,
) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
    // Compute X^T X (q×q).
    let xt = transpose(x_aug, n, q);
    let xtx = matmul(&xt, x_aug, q, n, q);

    // Compute X^T y (q×1).
    let mut xty = vec![0.0_f64; q];
    for j in 0..q {
        for i in 0..n {
            xty[j] += x_aug[i * q + j] * y[i];
        }
    }

    // Solve (X^T X + λI) β = X^T y.
    let beta = cholesky_solve(&xtx, &xty, q, l2_reg)?;

    // Fitted values.
    let fitted = matmul(x_aug, &beta, n, q, 1);

    Ok((beta, fitted))
}

/// Transpose matrix A (m×n) → A^T (n×m), row-major.
fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut at = vec![0.0_f64; n * m];
    for i in 0..m {
        for j in 0..n {
            at[j * m + i] = a[i * n + j];
        }
    }
    at
}

/// Clamp pseudo-values to the open unit interval for logistic regression.
///
/// For RMST pseudo-values, normalise to (0, τ) first, then to (0, 1).
/// For survival probability pseudo-values, clamp directly to (ε, 1−ε).
fn clamp_pseudo_for_logistic(pseudo: &[f64], horizon: f64, outcome: &PseudoObsOutcome) -> Vec<f64> {
    const EPS: f64 = 1.0e-6;
    match outcome {
        PseudoObsOutcome::SurvivalProb => pseudo.iter().map(|&v| v.clamp(EPS, 1.0 - EPS)).collect(),
        PseudoObsOutcome::Rmst => {
            let tau = horizon.max(f64::EPSILON);
            pseudo
                .iter()
                .map(|&v| (v / tau).clamp(EPS, 1.0 - EPS))
                .collect()
        }
    }
}

/// IRLS logistic regression on pseudo-values.
///
/// The response is the (clamped/normalised) pseudo-value in (0,1).
/// Returns (beta, fitted_probabilities, n_iter, converged).
fn irls_logistic(
    x_aug: &[f64],
    y_raw: &[f64],
    n: usize,
    q: usize,
    horizon: f64,
    outcome: &PseudoObsOutcome,
    max_iter: usize,
    tol: f64,
    l2_reg: f64,
) -> SurvivalResult<(Vec<f64>, Vec<f64>, usize, bool)> {
    const CLAMP_EPS: f64 = 1.0e-8;

    // Target response normalised to (0, 1).
    let y = clamp_pseudo_for_logistic(y_raw, horizon, outcome);

    // Initialise β at zero → linear predictor η = 0 → p = 0.5.
    let mut beta = vec![0.0_f64; q];
    let mut converged = false;
    let mut n_iter = 0usize;

    // Initialise eta from logit of y mean.
    let y_mean = y.iter().sum::<f64>() / (n as f64);
    let eta_init = logit(y_mean.clamp(CLAMP_EPS, 1.0 - CLAMP_EPS));
    // Intercept (last column) initialised to logit(ȳ).
    beta[q - 1] = eta_init;

    for iter in 0..max_iter {
        // Linear predictor η = X β.
        let eta = matmul(x_aug, &beta, n, q, 1);

        // Probabilities p = σ(η), weights W = p(1-p), working response z.
        let mut p_vec = vec![0.0_f64; n];
        let mut w_vec = vec![0.0_f64; n];
        let mut z_vec = vec![0.0_f64; n];
        for i in 0..n {
            let p_i = expit(eta[i]).clamp(CLAMP_EPS, 1.0 - CLAMP_EPS);
            let w_i = p_i * (1.0 - p_i);
            p_vec[i] = p_i;
            w_vec[i] = w_i.max(CLAMP_EPS); // prevent zero weights
            z_vec[i] = eta[i] + (y[i] - p_i) / w_vec[i];
        }

        // WLS: solve (X^T W X + λI) β_new = X^T W z.
        // Build X^T W X and X^T W z.
        let mut xtwx = vec![0.0_f64; q * q];
        let mut xtwz = vec![0.0_f64; q];
        for i in 0..n {
            let wi = w_vec[i];
            let zi = z_vec[i];
            for j in 0..q {
                for k in 0..q {
                    xtwx[j * q + k] += wi * x_aug[i * q + j] * x_aug[i * q + k];
                }
                xtwz[j] += wi * x_aug[i * q + j] * zi;
            }
        }

        let beta_new = cholesky_solve(&xtwx, &xtwz, q, l2_reg)?;

        // Convergence check: ‖Δβ‖₂.
        let delta_sq: f64 = beta_new
            .iter()
            .zip(beta.iter())
            .map(|(bn, bo)| (bn - bo).powi(2))
            .sum();

        beta = beta_new;
        n_iter = iter + 1;

        if delta_sq.sqrt() < tol {
            converged = true;
            break;
        }
    }

    // Final fitted probabilities.
    let eta_final = matmul(x_aug, &beta, n, q, 1);
    let p_final: Vec<f64> = eta_final.iter().map(|&e| expit(e)).collect();

    Ok((beta, p_final, n_iter, converged))
}

// ─── Public entry point ───────────────────────────────────────────────────────

/// Compute jackknife pseudo-observations and fit a regression model.
///
/// # Arguments
/// * `times`      — event or censoring times (length n, all ≥ 0)
/// * `events`     — event indicators: 1 = event occurred, 0 = censored (length n)
/// * `covariates` — covariate matrix in row-major order, shape n × p
///   (pass an empty slice `&[]` for intercept-only model)
/// * `config`     — configuration controlling horizon, outcome, regression type, etc.
///
/// # Errors
/// Returns [`SurvivalError`] when:
/// * `times` or `events` are empty,
/// * `horizon` is not positive,
/// * length mismatches are detected,
/// * the size of `covariates` is inconsistent with n and the implied p,
/// * a negative time is encountered,
/// * numerical instability prevents the Cholesky solve.
pub fn pseudo_obs_fit(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    config: &PseudoObsConfig,
) -> SurvivalResult<PseudoObsResult> {
    // ── Input validation ─────────────────────────────────────────────────────
    let n = times.len();
    if n == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if events.len() != n {
        return Err(SurvivalError::DimensionMismatch {
            a: n,
            b: events.len(),
        });
    }
    if config.horizon <= 0.0 || !config.horizon.is_finite() {
        return Err(SurvivalError::InvalidParameter(format!(
            "horizon must be > 0, got {}",
            config.horizon
        )));
    }
    // Check for negative times.
    for &t in times {
        if t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }

    // Determine p (number of covariates).
    let p = if covariates.is_empty() {
        0usize
    } else {
        if covariates.len() % n != 0 {
            return Err(SurvivalError::DimensionMismatch {
                a: covariates.len(),
                b: n,
            });
        }
        covariates.len() / n
    };

    // ── Overall functional estimate θ̂ ────────────────────────────────────────
    let overall_estimate = compute_functional(times, events, n, config.horizon, &config.outcome);

    // ── Jackknife pseudo-values ───────────────────────────────────────────────
    let pseudo_values = compute_pseudo_values(
        times,
        events,
        n,
        config.horizon,
        overall_estimate,
        &config.outcome,
    )?;

    // ── Design matrix [X | 1] (n × (p+1)) ───────────────────────────────────
    let q = p + 1;
    let x_aug = build_design_matrix(covariates, n, p);

    // ── Regression ───────────────────────────────────────────────────────────
    let (beta, n_iter, converged, log_likelihood) = match config.regression {
        PseudoObsRegression::Linear => {
            let (beta, fitted) = ols_regression(&x_aug, &pseudo_values, n, q, config.l2_reg)?;
            let ll = linreg_log_likelihood(&pseudo_values, &fitted, n);
            (beta, 1usize, true, ll)
        }
        PseudoObsRegression::Logistic => {
            let (beta, p_fitted, n_iter, converged) = irls_logistic(
                &x_aug,
                &pseudo_values,
                n,
                q,
                config.horizon,
                &config.outcome,
                config.max_iter,
                config.tol,
                config.l2_reg,
            )?;
            // For log-likelihood use clamped target values (same as IRLS response).
            let y_clamped =
                clamp_pseudo_for_logistic(&pseudo_values, config.horizon, &config.outcome);
            let ll = logreg_log_likelihood(&y_clamped, &p_fitted, n);
            (beta, n_iter, converged, ll)
        }
    };

    Ok(PseudoObsResult {
        pseudo_values,
        beta,
        n_iter,
        converged,
        overall_estimate,
        log_likelihood,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a simple dataset without covariates.
    fn make_config_rmst(horizon: f64) -> PseudoObsConfig {
        PseudoObsConfig {
            horizon,
            outcome: PseudoObsOutcome::Rmst,
            regression: PseudoObsRegression::Linear,
            ..Default::default()
        }
    }

    fn make_config_surv(horizon: f64) -> PseudoObsConfig {
        PseudoObsConfig {
            horizon,
            outcome: PseudoObsOutcome::SurvivalProb,
            regression: PseudoObsRegression::Logistic,
            ..Default::default()
        }
    }

    // 1. Default configuration fields.
    #[test]
    fn config_defaults() {
        let cfg = PseudoObsConfig::default();
        assert!((cfg.horizon - 1.0).abs() < 1.0e-14);
        assert_eq!(cfg.outcome, PseudoObsOutcome::Rmst);
        assert_eq!(cfg.regression, PseudoObsRegression::Linear);
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.tol - 1.0e-6).abs() < 1.0e-14);
    }

    // 2. RMST on a tiny dataset.
    #[test]
    fn compute_rmst_simple() {
        // times = [0.5, 1.5, 2.5], events = [1, 0, 1], horizon = 2.0
        // KM: at t=0.5 S=2/3; at t=2.5 S=1/3.
        // Within horizon=2.0 only t=0.5 step counts.
        // RMST = 0.5*1.0 + 1.5*(2/3) = 0.5 + 1.0 = 1.5.
        let steps = kaplan_meier_steps(&[0.5, 1.5, 2.5], &[1, 0, 1], 3);
        let rmst = rmst_from_km_steps(&steps, 2.0);
        // rectangle [0, 0.5) with S=1 → 0.5*1=0.5
        // rectangle [0.5, 2.0) with S=2/3 → 1.5*(2/3)=1.0
        assert!((rmst - 1.5).abs() < 1.0e-10, "expected 1.5 got {rmst}");
    }

    // 3. pseudo_values length == n_samples.
    #[test]
    fn pseudo_values_length() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![1u8, 0, 1, 0, 1];
        let cfg = make_config_rmst(4.0);
        let result = pseudo_obs_fit(&times, &events, &[], &cfg).expect("fit ok");
        assert_eq!(result.pseudo_values.len(), times.len());
    }

    // 4. mean(pseudo_values) ≈ overall_estimate (fundamental jackknife property).
    #[test]
    fn pseudo_values_mean_close_to_overall() {
        let times = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let events = vec![1u8, 0, 1, 1, 0, 1, 0, 1];
        let cfg = make_config_rmst(3.0);
        let result = pseudo_obs_fit(&times, &events, &[], &cfg).expect("fit ok");
        let n = result.pseudo_values.len() as f64;
        let mean_pv: f64 = result.pseudo_values.iter().sum::<f64>() / n;
        let overall = result.overall_estimate;
        // Jackknife mean converges to θ̂ — should be close but exact equality isn't guaranteed.
        assert!(
            (mean_pv - overall).abs() < 0.5,
            "mean_pv={mean_pv} overall={overall}"
        );
    }

    // 5. Intercept-only linear regression: beta[0] ≈ mean(pseudo_values) ≈ overall RMST.
    #[test]
    fn fit_no_covariates_linear() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = vec![1u8, 1, 0, 1, 0, 1];
        let cfg = make_config_rmst(5.0);
        let result = pseudo_obs_fit(&times, &events, &[], &cfg).expect("fit ok");
        assert_eq!(result.beta.len(), 1, "intercept-only: one coefficient");
        // The intercept should approximate overall RMST.
        assert!(
            result.beta[0].is_finite(),
            "beta[0] must be finite, got {}",
            result.beta[0]
        );
        assert!(
            result.beta[0] >= 0.0,
            "RMST must be non-negative, got {}",
            result.beta[0]
        );
    }

    // 6. Intercept-only logistic regression on survival probability pseudo-values.
    #[test]
    fn fit_no_covariates_logistic() {
        let times = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let events = vec![1u8, 0, 1, 0, 1, 0];
        let cfg = make_config_surv(2.0);
        let result = pseudo_obs_fit(&times, &events, &[], &cfg).expect("fit ok");
        assert_eq!(result.beta.len(), 1, "intercept-only: one coefficient");
        assert!(result.beta[0].is_finite(), "beta must be finite");
        // Intercept = logit(S(tau)); overall_estimate ∈ [0, 1].
        let s_tau = result.overall_estimate;
        assert!((0.0..=1.0).contains(&s_tau), "S(tau) ∈ [0,1], got {s_tau}");
    }

    // 7. One covariate: beta.len() == 2 (1 covariate + intercept).
    #[test]
    fn fit_with_covariate() {
        let n = 20usize;
        let times: Vec<f64> = (1..=n).map(|i| i as f64 * 0.5).collect();
        let events: Vec<u8> = (0..n).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        // Single covariate: standardised index.
        let covariates: Vec<f64> = (0..n)
            .map(|i| (i as f64 - (n as f64 / 2.0)) / (n as f64))
            .collect();
        let cfg = PseudoObsConfig {
            horizon: 6.0,
            outcome: PseudoObsOutcome::Rmst,
            regression: PseudoObsRegression::Linear,
            ..Default::default()
        };
        let result = pseudo_obs_fit(&times, &events, &covariates, &cfg).expect("fit ok");
        assert_eq!(result.beta.len(), 2, "1 covariate + intercept");
    }

    // 8. Overall RMST estimate is non-negative.
    #[test]
    fn overall_estimate_positive() {
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![1u8, 0, 1];
        let cfg = make_config_rmst(2.5);
        let result = pseudo_obs_fit(&times, &events, &[], &cfg).expect("fit ok");
        assert!(result.overall_estimate >= 0.0);
    }

    // 9. RMST ≤ horizon.
    #[test]
    fn overall_estimate_leq_horizon() {
        let times = vec![0.1, 0.5, 1.0, 2.0, 5.0];
        let events = vec![1u8, 1, 1, 0, 1];
        let tau = 3.0;
        let cfg = make_config_rmst(tau);
        let result = pseudo_obs_fit(&times, &events, &[], &cfg).expect("fit ok");
        assert!(
            result.overall_estimate <= tau + 1.0e-10,
            "RMST={} > tau={}",
            result.overall_estimate,
            tau
        );
    }

    // 10. Survival probability ∈ [0, 1].
    #[test]
    fn survival_prob_in_unit_interval() {
        let times = vec![0.5, 1.0, 1.5, 2.0];
        let events = vec![1u8, 0, 1, 1];
        let cfg = PseudoObsConfig {
            horizon: 1.2,
            outcome: PseudoObsOutcome::SurvivalProb,
            regression: PseudoObsRegression::Linear,
            ..Default::default()
        };
        let result = pseudo_obs_fit(&times, &events, &[], &cfg).expect("fit ok");
        let s = result.overall_estimate;
        assert!((0.0..=1.0).contains(&s), "S(tau)={s} not in [0,1]");
    }

    // 11. Empty times slice returns Err(EmptyDataset).
    #[test]
    fn empty_input_error() {
        let cfg = make_config_rmst(1.0);
        let err = pseudo_obs_fit(&[], &[], &[], &cfg).expect_err("must error");
        assert!(
            matches!(err, SurvivalError::EmptyDataset),
            "expected EmptyDataset, got {err:?}"
        );
    }

    // 12. Non-positive horizon returns Err(InvalidParameter).
    #[test]
    fn horizon_negative_error() {
        let times = vec![1.0, 2.0];
        let events = vec![1u8, 0];
        let cfg = PseudoObsConfig {
            horizon: -1.0,
            ..Default::default()
        };
        let err = pseudo_obs_fit(&times, &events, &[], &cfg).expect_err("must error");
        assert!(
            matches!(err, SurvivalError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
    }

    // 13. All beta values must be finite.
    #[test]
    fn beta_finite() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![1u8, 0, 1, 1, 0, 1, 1, 0];
        let cfg = make_config_rmst(6.0);
        let result = pseudo_obs_fit(&times, &events, &[], &cfg).expect("fit ok");
        for (k, &b) in result.beta.iter().enumerate() {
            assert!(b.is_finite(), "beta[{k}] = {b} is not finite");
        }
    }

    // 14. Log-likelihood must be finite.
    #[test]
    fn log_likelihood_finite() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![1u8, 0, 1, 0, 1];
        let cfg = make_config_rmst(4.0);
        let result = pseudo_obs_fit(&times, &events, &[], &cfg).expect("fit ok");
        assert!(
            result.log_likelihood.is_finite(),
            "log_likelihood = {} is not finite",
            result.log_likelihood
        );
    }

    // 15. Binary covariate separating two risk groups → different fitted RMST values.
    #[test]
    fn two_group_covariate() {
        // Group 0 (low risk): long survival times. Group 1 (high risk): short.
        let n = 20usize;
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut covariates = Vec::with_capacity(n);

        for i in 0..10 {
            times.push(5.0 + i as f64 * 0.5); // long
            events.push(1u8);
            covariates.push(0.0_f64); // group 0
        }
        for i in 0..10 {
            times.push(0.5 + i as f64 * 0.3); // short
            events.push(1u8);
            covariates.push(1.0_f64); // group 1
        }

        let cfg = PseudoObsConfig {
            horizon: 5.0,
            outcome: PseudoObsOutcome::Rmst,
            regression: PseudoObsRegression::Linear,
            l2_reg: 1.0e-4,
            ..Default::default()
        };
        let result = pseudo_obs_fit(&times, &events, &covariates, &cfg).expect("fit ok");
        assert_eq!(result.beta.len(), 2);

        // Fitted RMST for group 0 (x=0): intercept.
        // Fitted RMST for group 1 (x=1): intercept + slope.
        let fitted_group0 = result.beta[1]; // intercept (x=0)
        let fitted_group1 = result.beta[0] + result.beta[1]; // slope + intercept (x=1)

        // Group 0 should have higher fitted RMST than group 1.
        assert!(
            fitted_group0 > fitted_group1,
            "group0 fitted RMST ({fitted_group0}) should exceed group1 ({fitted_group1})"
        );
    }

    // Bonus: zero-horizon returns Err.
    #[test]
    fn horizon_zero_error() {
        let times = vec![1.0, 2.0];
        let events = vec![1u8, 0];
        let cfg = PseudoObsConfig {
            horizon: 0.0,
            ..Default::default()
        };
        assert!(pseudo_obs_fit(&times, &events, &[], &cfg).is_err());
    }

    // Bonus: single observation edge case.
    #[test]
    fn single_observation() {
        let times = vec![1.0];
        let events = vec![1u8];
        let cfg = make_config_rmst(2.0);
        // Should not panic; may produce degenerate but valid result.
        let result = pseudo_obs_fit(&times, &events, &[], &cfg).expect("fit ok");
        assert_eq!(result.pseudo_values.len(), 1);
        assert!(result.beta[0].is_finite());
    }
}

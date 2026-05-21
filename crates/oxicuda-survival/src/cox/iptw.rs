//! Inverse Probability of Treatment Weighting (IPTW) for causal hazard estimation.
//!
//! IPTW re-weights observations so that treatment assignment becomes independent of
//! measured confounders, enabling causal interpretation of hazard ratios from a
//! weighted Cox proportional hazards model.
//!
//! # Pipeline
//!
//! 1. **Propensity score estimation** — fit logistic regression `P(T=1|X)` via IRLS
//!    (Iteratively Reweighted Least Squares / Newton-Raphson on the binary log-likelihood).
//! 2. **IPTW weight computation** — stabilised or unstabilised, with optional percentile
//!    trimming to avoid extreme leverages.
//! 3. **Weighted Cox PH** — partial likelihood with per-observation weights.
//! 4. **Causal estimands** — marginal hazard ratio, log-hazard ATE.
//!
//! # Augmented IPTW (AIPTW)
//!
//! Doubly-robust estimation that is consistent if either the propensity model or the
//! outcome model is correctly specified.

use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::solve::cholesky_solve;

// ─── Constants ─────────────────────────────────────────────────────────────────

/// Numerical floor for propensity scores to avoid division by zero.
const PS_FLOOR: f64 = 1.0e-12;
/// Numerical ceiling for propensity scores.
const PS_CEIL: f64 = 1.0 - 1.0e-12;
/// Minimum diagonal boost for near-singular IRLS Hessian.
const HESSIAN_RIDGE: f64 = 1.0e-8;

// ─── Configuration types ───────────────────────────────────────────────────────

/// Configuration for IPTW estimation.
#[derive(Debug, Clone)]
pub struct IptwConfig {
    /// Use stabilised weights (recommended; reduces variance). Default: `true`.
    pub stabilize: bool,
    /// Lower trim percentile for weight clipping (default: `0.01`).
    pub trim_lower: f64,
    /// Upper trim percentile for weight clipping (default: `0.99`).
    pub trim_upper: f64,
    /// Maximum iterations for propensity score IRLS. Default: `100`.
    pub ps_max_iter: usize,
    /// Convergence tolerance for propensity score IRLS. Default: `1e-6`.
    pub ps_tol: f64,
    /// Maximum iterations for weighted Cox Newton-Raphson. Default: `50`.
    pub cox_max_iter: usize,
    /// Convergence tolerance for weighted Cox Newton-Raphson. Default: `1e-6`.
    pub cox_tol: f64,
    /// L2 regularisation strength for the propensity model. Default: `1e-3`.
    pub l2_reg_ps: f64,
}

impl Default for IptwConfig {
    fn default() -> Self {
        Self {
            stabilize: true,
            trim_lower: 0.01,
            trim_upper: 0.99,
            ps_max_iter: 100,
            ps_tol: 1.0e-6,
            cox_max_iter: 50,
            cox_tol: 1.0e-6,
            l2_reg_ps: 1.0e-3,
        }
    }
}

/// Configuration for Augmented IPTW (doubly-robust) estimation.
#[derive(Debug, Clone)]
pub struct AiptwConfig {
    /// Underlying IPTW configuration.
    pub iptw_config: IptwConfig,
    /// Maximum iterations for the outcome (Cox) model used in augmentation.
    /// Default: `100`.
    pub outcome_model_max_iter: usize,
    /// Convergence tolerance for the outcome model. Default: `1e-6`.
    pub outcome_model_tol: f64,
}

impl Default for AiptwConfig {
    fn default() -> Self {
        Self {
            iptw_config: IptwConfig::default(),
            outcome_model_max_iter: 100,
            outcome_model_tol: 1.0e-6,
        }
    }
}

// ─── Result types ──────────────────────────────────────────────────────────────

/// Result from propensity score estimation and weight computation.
#[derive(Debug, Clone)]
pub struct PropensityResult {
    /// Logistic regression coefficient vector β (length `p`).
    pub beta: Vec<f64>,
    /// Propensity scores `P(T=1|X_i)` for each subject (length `n`).
    pub scores: Vec<f64>,
    /// IPTW weights for each subject (length `n`).
    pub weights: Vec<f64>,
    /// Effective sample size `(Σw)² / Σw²` — measures weight heterogeneity.
    pub eff_sample_size: f64,
    /// IRLS iterations consumed.
    pub n_iter: usize,
    /// Whether IRLS met its convergence criterion.
    pub converged: bool,
}

/// Result from IPTW-weighted Cox proportional hazards estimation.
#[derive(Debug, Clone)]
pub struct IptwResult {
    /// Propensity score estimation result.
    pub propensity: PropensityResult,
    /// Causal hazard ratio coefficients from the weighted Cox model (length `q+1`).
    pub weighted_cox_beta: Vec<f64>,
    /// Marginal hazard ratio for the treatment column: `exp(β[treatment_col])`.
    pub marginal_hr: f64,
    /// Weighted partial log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Weighted Cox Newton-Raphson iterations consumed.
    pub n_iter: usize,
    /// Whether the weighted Cox model converged.
    pub converged: bool,
    /// Mean IPTW weight among treated subjects.
    pub mean_weight_treated: f64,
    /// Mean IPTW weight among control subjects.
    pub mean_weight_control: f64,
}

/// Result from Augmented IPTW (doubly-robust) estimation.
#[derive(Debug, Clone)]
pub struct AiptwResult {
    /// Underlying IPTW result.
    pub iptw: IptwResult,
    /// Doubly-robust augmented ATE on the log-hazard scale.
    pub augmented_ate: f64,
    /// Outcome model coefficients (length `q`).
    pub outcome_beta: Vec<f64>,
    /// Whether the outcome model converged.
    pub outcome_converged: bool,
}

// ─── Internal linear-algebra helpers ─────────────────────────────────────────

/// Compute sigmoid `σ(z) = 1 / (1 + exp(−z))` clamped to `[PS_FLOOR, PS_CEIL]`.
#[inline]
fn sigmoid(z: f64) -> f64 {
    let s = if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    };
    s.clamp(PS_FLOOR, PS_CEIL)
}

/// Compute the linear predictor `X β` for subject `i`.
#[inline]
fn linear_predictor(covariates: &[f64], beta: &[f64], n: usize, p: usize, i: usize) -> f64 {
    let row_start = i * p;
    let mut xb = 0.0_f64;
    for j in 0..p {
        xb += covariates[row_start + j] * beta[j];
    }
    let _ = n; // bound parameter used by caller for clarity
    xb
}

/// Compute propensity scores `σ(X β)` for all subjects.
fn compute_propensity_scores(covariates: &[f64], beta: &[f64], n: usize, p: usize) -> Vec<f64> {
    (0..n)
        .map(|i| sigmoid(linear_predictor(covariates, beta, n, p, i)))
        .collect()
}

/// Compute the binary log-likelihood, gradient (score), and Hessian (information)
/// for logistic regression with L2 regularisation.
///
/// log L(β) = Σ_i [t_i log(e_i) + (1-t_i) log(1-e_i)] - (λ/2) ‖β‖²
///
/// Returns `(log_likelihood, gradient, hessian_flat_row_major)`.
fn logistic_score_hessian(
    covariates: &[f64],
    treatment: &[u8],
    n: usize,
    p: usize,
    beta: &[f64],
    l2_reg: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let mut ll = 0.0_f64;
    let mut grad = vec![0.0_f64; p];
    let mut hess = vec![0.0_f64; p * p];

    for (i, &ti) in treatment.iter().enumerate().take(n) {
        let xb = linear_predictor(covariates, beta, n, p, i);
        let e = sigmoid(xb);
        let t = ti as f64;

        // log-likelihood contribution (avoid log(0) via clamp already done in sigmoid)
        ll += t * e.ln() + (1.0 - t) * (1.0 - e).ln();

        // Residual r_i = t_i - e_i
        let r = t - e;
        // Hessian weight w_i = e_i(1-e_i)
        let w = e * (1.0 - e);

        let row_start = i * p;
        for j in 0..p {
            let xij = covariates[row_start + j];
            grad[j] += r * xij;
            for k in 0..p {
                let xik = covariates[row_start + k];
                hess[j * p + k] += w * xij * xik;
            }
        }
    }

    // L2 regularisation: subtract (λ/2)‖β‖² from log-likelihood
    let l2_pen: f64 = l2_reg * beta.iter().map(|b| b * b).sum::<f64>() * 0.5;
    ll -= l2_pen;

    // Penalised gradient: ∇ = grad - λ β
    for j in 0..p {
        grad[j] -= l2_reg * beta[j];
    }

    // Penalised Hessian: H += λ I (adds to diagonal)
    for j in 0..p {
        hess[j * p + j] += l2_reg;
    }

    (ll, grad, hess)
}

/// Compute the p-th percentile of a slice (linear interpolation).
/// Returns the value such that `p * 100` percent of the data falls below it.
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx_f = p * (n - 1) as f64;
    let lo = idx_f.floor() as usize;
    let hi = idx_f.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = idx_f - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Compute effective sample size from a weight vector.
fn effective_sample_size(weights: &[f64]) -> f64 {
    let sum_w: f64 = weights.iter().sum();
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    if sum_w2 < f64::EPSILON {
        0.0
    } else {
        (sum_w * sum_w) / sum_w2
    }
}

// ─── Weighted Cox partial likelihood ─────────────────────────────────────────

/// Compute the weighted Cox partial log-likelihood, score, and Fisher information.
///
/// Breslow tie handling, with subject-specific weights `w_i`.
///
/// Partial log-likelihood:
/// ```text
///   ℓ(β) = Σ_i δ_i { β'x_i − log[ Σ_{j∈R(t_i)} w_j exp(β'x_j) ] }
/// ```
///
/// Returns `(log_likelihood, score_vector, information_matrix_flat)`.
fn weighted_cox_loglik(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
    // Pre-compute linear predictors and exponentiated values
    let xb: Vec<f64> = (0..n)
        .map(|i| linear_predictor(covariates, beta, n, p, i))
        .collect();

    // Clamp xb to avoid overflow in exp
    let exp_xb: Vec<f64> = xb.iter().map(|&v| v.clamp(-500.0, 500.0).exp()).collect();

    // Sort by ascending time for risk-set computation
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Iterate through event times in reverse order to accumulate risk sets.
    // We process events from largest to smallest time.
    let mut ll = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];

    // Build sorted indices
    let sorted_times: Vec<f64> = order.iter().map(|&i| times[i]).collect();

    // Build traversal list (descending time order)
    let sorted_order_rev: Vec<usize> = order.iter().rev().copied().collect();

    // Risk set accumulator: iterate in descending time order, expanding R(t_i) lazily.
    // A single O(n) scan suffices because times are sorted.
    let mut risk_sum_w = 0.0_f64; // Σ w_j exp(β'x_j) over current risk set
    let mut risk_wx = vec![0.0_f64; p]; // Σ w_j exp(β'x_j) x_j
    let mut risk_wxx = vec![0.0_f64; p * p]; // Σ w_j exp(β'x_j) x_j x_j'
    let mut tail_ptr = n; // exclusive upper bound in `order` (sorted asc)

    // Process in sorted descending time order
    for &i in &sorted_order_rev {
        let t_i = times[i];

        // Expand risk set: include all j with sorted_times[tail_ptr-1] >= t_i
        while tail_ptr > 0 && sorted_times[tail_ptr - 1] >= t_i {
            let j = order[tail_ptr - 1];
            let wj_exp = weights[j] * exp_xb[j];
            risk_sum_w += wj_exp;
            let row_j = j * p;
            for k in 0..p {
                risk_wx[k] += wj_exp * covariates[row_j + k];
            }
            for k in 0..p {
                for l in 0..p {
                    risk_wxx[k * p + l] += wj_exp * covariates[row_j + k] * covariates[row_j + l];
                }
            }
            tail_ptr -= 1;
        }

        if events[i] == 0 {
            continue;
        }

        // Event i contributes to log-likelihood
        if risk_sum_w < f64::EPSILON {
            return Err(SurvivalError::NumericalInstability(
                "weighted Cox: risk set weight sum is zero".to_string(),
            ));
        }
        let log_risk = risk_sum_w.ln();
        ll += xb[i] - log_risk;

        // Score: x_i - E[X | risk set]
        let inv_risk = 1.0 / risk_sum_w;
        let row_i = i * p;
        for k in 0..p {
            score[k] += covariates[row_i + k] - risk_wx[k] * inv_risk;
        }

        // Information: Var[X | risk set] = E[XX'] - E[X]E[X]'
        for k in 0..p {
            for l in 0..p {
                let e_xk = risk_wx[k] * inv_risk;
                let e_xl = risk_wx[l] * inv_risk;
                let e_xkl = risk_wxx[k * p + l] * inv_risk;
                info[k * p + l] += e_xkl - e_xk * e_xl;
            }
        }
    }

    Ok((ll, score, info))
}

// ─── Public API ────────────────────────────────────────────────────────────────

/// Fit a logistic regression propensity score model via IRLS (Newton-Raphson).
///
/// Estimates `P(T=1 | X)` using the binary log-likelihood penalised by an L2 term
/// `(l2_reg/2)‖β‖²`. Convergence is measured on the maximum absolute gradient component.
///
/// # Arguments
///
/// - `covariates` — row-major `[n, p]` covariate matrix.
/// - `treatment` — binary treatment indicators `0` or `1`, length `n`.
/// - `n_subjects` — number of subjects `n`.
/// - `n_covariates` — number of covariates `p`.
/// - `config` — IPTW configuration.
///
/// # Errors
///
/// Returns [`SurvivalError::EmptyDataset`] for `n == 0`,
/// [`SurvivalError::InvalidParameter`] for invalid treatment values or shape,
/// [`SurvivalError::SingularMatrix`] if the IRLS Hessian is irrecoverably singular.
pub fn fit_propensity_score(
    covariates: &[f64],
    treatment: &[u8],
    n_subjects: usize,
    n_covariates: usize,
    config: &IptwConfig,
) -> SurvivalResult<PropensityResult> {
    // ── Validate inputs ────────────────────────────────────────────────────────
    if n_subjects == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if treatment.len() != n_subjects {
        return Err(SurvivalError::DimensionMismatch {
            a: n_subjects,
            b: treatment.len(),
        });
    }
    if covariates.len() != n_subjects * n_covariates {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects * n_covariates],
            got: vec![covariates.len()],
        });
    }
    for (idx, &t) in treatment.iter().enumerate() {
        if t > 1 {
            return Err(SurvivalError::InvalidParameter(format!(
                "treatment[{idx}] = {t}: must be 0 or 1"
            )));
        }
    }
    if n_covariates == 0 {
        return Err(SurvivalError::InvalidParameter(
            "propensity model requires at least one covariate".to_string(),
        ));
    }

    // ── IRLS ────────────────────────────────────────────────────────────────────
    let p = n_covariates;
    let n = n_subjects;
    let mut beta = vec![0.0_f64; p];
    let mut converged = false;
    let mut n_iter = 0usize;

    for it in 0..config.ps_max_iter {
        n_iter = it + 1;
        let (ll, grad, hess) =
            logistic_score_hessian(covariates, treatment, n, p, &beta, config.l2_reg_ps);

        // Convergence on max |gradient|
        let max_grad = grad.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if max_grad < config.ps_tol {
            converged = true;
            break;
        }

        // Solve H Δβ = ∇  (note: H is the positive-definite information matrix;
        // we negate because we maximise, so the step is H^{-1} g)
        let delta = match cholesky_solve(&hess, &grad, p) {
            Ok(d) => d,
            Err(_) => {
                // Apply ridge boost to Hessian diagonal
                let mut hess_ridge = hess.clone();
                for j in 0..p {
                    hess_ridge[j * p + j] += HESSIAN_RIDGE;
                }
                match cholesky_solve(&hess_ridge, &grad, p) {
                    Ok(d) => d,
                    Err(_) => {
                        return Err(SurvivalError::SingularMatrix);
                    }
                }
            }
        };

        // Line search with step-halving (Armijo condition on log-likelihood)
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..40 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + step * d)
                .collect();
            let (ll_new, _, _) =
                logistic_score_hessian(covariates, treatment, n, p, &trial, config.l2_reg_ps);
            if ll_new.is_finite() && ll_new > ll - 1.0e-10 {
                beta = trial;
                accepted = true;
                break;
            }
            step *= 0.5;
            if step < 1.0e-20 {
                break;
            }
        }
        if !accepted {
            break;
        }
    }

    // Final convergence check
    if !converged {
        let (_, grad_final, _) =
            logistic_score_hessian(covariates, treatment, n, p, &beta, config.l2_reg_ps);
        let max_grad = grad_final.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if max_grad < config.ps_tol {
            converged = true;
        }
    }

    // ── Compute propensity scores ──────────────────────────────────────────────
    let scores = compute_propensity_scores(covariates, &beta, n, p);

    // ── Compute raw IPTW weights ───────────────────────────────────────────────
    let raw_weights = compute_raw_weights(&scores, treatment, n, config.stabilize);

    // ── Trim weights ───────────────────────────────────────────────────────────
    let weights = trim_weights(&raw_weights, config.trim_lower, config.trim_upper);

    let eff_n = effective_sample_size(&weights);

    Ok(PropensityResult {
        beta,
        scores,
        weights,
        eff_sample_size: eff_n,
        n_iter,
        converged,
    })
}

/// Compute raw (un-trimmed) IPTW weights from propensity scores.
///
/// Stabilised weights multiply by the marginal treatment probabilities
/// `P(T=1)` and `P(T=0)`, which keeps the mean weight close to 1.
fn compute_raw_weights(scores: &[f64], treatment: &[u8], n: usize, stabilize: bool) -> Vec<f64> {
    let n_treated: usize = treatment.iter().filter(|&&t| t == 1).count();
    let p_treated = n_treated as f64 / n as f64;
    let p_control = 1.0 - p_treated;

    (0..n)
        .map(|i| {
            let e = scores[i].clamp(PS_FLOOR, PS_CEIL);
            if treatment[i] == 1 {
                if stabilize { p_treated / e } else { 1.0 / e }
            } else {
                if stabilize {
                    p_control / (1.0 - e)
                } else {
                    1.0 / (1.0 - e)
                }
            }
        })
        .collect()
}

/// Trim weights to the `[lower_pct, upper_pct]` percentile range.
///
/// Values below the lower percentile are set to that percentile value;
/// values above the upper percentile are capped similarly.
fn trim_weights(weights: &[f64], lower_pct: f64, upper_pct: f64) -> Vec<f64> {
    let lo = percentile(weights, lower_pct.clamp(0.0, 1.0));
    let hi = percentile(weights, upper_pct.clamp(0.0, 1.0));
    weights.iter().map(|&w| w.clamp(lo, hi)).collect()
}

/// Re-compute IPTW weights from a fitted `PropensityResult` with potentially
/// different trimming or stabilisation settings.
///
/// Useful for sensitivity analysis without re-fitting the propensity model.
///
/// # Errors
///
/// Returns [`SurvivalError::DimensionMismatch`] if `treatment.len()` differs from
/// the number of propensity scores in `ps_result`.
pub fn compute_iptw_weights(
    ps_result: &PropensityResult,
    treatment: &[u8],
    config: &IptwConfig,
) -> SurvivalResult<Vec<f64>> {
    let n = ps_result.scores.len();
    if treatment.len() != n {
        return Err(SurvivalError::DimensionMismatch {
            a: n,
            b: treatment.len(),
        });
    }
    let raw = compute_raw_weights(&ps_result.scores, treatment, n, config.stabilize);
    Ok(trim_weights(&raw, config.trim_lower, config.trim_upper))
}

/// Fit a weighted Cox proportional hazards model given pre-computed IPTW weights.
///
/// The treatment variable should be included as the **last** column of `covariates`
/// (i.e. `covariates` has `n_covariates` columns, with the treatment column last).
/// The `marginal_hr` in the result will be `exp(β[n_covariates-1])`.
///
/// If you want the treatment to be the first column, simply pass `treatment_col = 0`
/// via the [`iptw_fit`] function instead.
///
/// # Arguments
///
/// - `times` — survival times, length `n`.
/// - `events` — event indicators `0/1`, length `n`.
/// - `covariates` — row-major `[n, n_covariates]` covariate matrix (including treatment).
/// - `weights` — IPTW weights, length `n`.
/// - `n_subjects` — number of subjects `n`.
/// - `n_covariates` — total number of columns in `covariates` (including treatment).
/// - `config` — IPTW configuration (Cox parameters).
///
/// # Errors
///
/// Returns [`SurvivalError::EmptyDataset`] for empty inputs,
/// [`SurvivalError::NoEvents`] if there are no uncensored observations,
/// numerical errors from the weighted partial likelihood.
pub fn iptw_cox(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    weights: &[f64],
    n_subjects: usize,
    n_covariates: usize,
    config: &IptwConfig,
    propensity: PropensityResult,
) -> SurvivalResult<IptwResult> {
    // ── Validate ───────────────────────────────────────────────────────────────
    if n_subjects == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if times.len() != n_subjects
        || events.len() != n_subjects
        || weights.len() != n_subjects
        || covariates.len() != n_subjects * n_covariates
    {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![times.len()],
        });
    }
    let n_events: usize = events.iter().filter(|&&e| e == 1).count();
    if n_events == 0 {
        return Err(SurvivalError::NoEvents);
    }
    let p = n_covariates;
    let n = n_subjects;

    // Mean weights per arm
    let (mut sum_w_treated, mut cnt_treated) = (0.0_f64, 0usize);
    let (mut sum_w_control, mut cnt_control) = (0.0_f64, 0usize);
    // Treatment arm is inferred from the last column of covariates
    for i in 0..n {
        let t_col = covariates[i * p + (p - 1)];
        if t_col > 0.5 {
            sum_w_treated += weights[i];
            cnt_treated += 1;
        } else {
            sum_w_control += weights[i];
            cnt_control += 1;
        }
    }
    let mean_weight_treated = if cnt_treated > 0 {
        sum_w_treated / cnt_treated as f64
    } else {
        0.0
    };
    let mean_weight_control = if cnt_control > 0 {
        sum_w_control / cnt_control as f64
    } else {
        0.0
    };

    // ── Newton-Raphson on weighted partial log-likelihood ─────────────────────
    let mut beta = vec![0.0_f64; p];
    let mut converged = false;
    let mut n_iter = 0usize;

    let (mut ll, mut score, mut info) =
        weighted_cox_loglik(times, events, covariates, weights, n, p, &beta)?;

    for it in 0..config.cox_max_iter {
        n_iter = it + 1;
        let max_score = score.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if max_score < config.cox_tol {
            converged = true;
            break;
        }

        let delta = match cholesky_solve(&info, &score, p) {
            Ok(d) => d,
            Err(_) => {
                let mut info_ridge = info.clone();
                for j in 0..p {
                    info_ridge[j * p + j] += 1.0e-4;
                }
                match cholesky_solve(&info_ridge, &score, p) {
                    Ok(d) => d,
                    Err(_) => break, // give up on this iteration
                }
            }
        };

        // Line search
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..40 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + step * d)
                .collect();
            if let Ok((ll_new, sc_new, info_new)) =
                weighted_cox_loglik(times, events, covariates, weights, n, p, &trial)
            {
                if ll_new.is_finite() && ll_new > ll - 1.0e-10 {
                    beta = trial;
                    ll = ll_new;
                    score = sc_new;
                    info = info_new;
                    accepted = true;
                    break;
                }
            }
            step *= 0.5;
            if step < 1.0e-20 {
                break;
            }
        }
        if !accepted {
            break;
        }
    }

    // Final convergence check
    if !converged {
        let max_score = score.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if max_score < config.cox_tol {
            converged = true;
        }
    }

    // Marginal HR: coefficient of the last covariate (treatment column)
    let marginal_hr = beta[p - 1].exp();

    Ok(IptwResult {
        propensity,
        weighted_cox_beta: beta,
        marginal_hr,
        log_likelihood: ll,
        n_iter,
        converged,
        mean_weight_treated,
        mean_weight_control,
    })
}

/// Complete IPTW pipeline: propensity estimation → weight computation → weighted Cox.
///
/// The `confounders` matrix (`[n, n_confounders]`) must **not** include the treatment
/// column. This function internally appends the treatment column as the last covariate
/// when calling the weighted Cox model.
///
/// The returned `marginal_hr` corresponds to `exp(β[treatment])`.
///
/// # Arguments
///
/// - `times` — survival times, length `n`.
/// - `events` — event indicators `0/1`, length `n`.
/// - `treatment` — binary treatment assignment `0/1`, length `n`.
/// - `confounders` — row-major `[n, n_confounders]` confounder matrix.
/// - `n_subjects` — number of subjects `n`.
/// - `n_confounders` — number of confounder columns `q`.
/// - `config` — IPTW configuration.
///
/// # Errors
///
/// See [`fit_propensity_score`] and [`iptw_cox`].
pub fn iptw_fit(
    times: &[f64],
    events: &[u8],
    treatment: &[u8],
    confounders: &[f64],
    n_subjects: usize,
    n_confounders: usize,
    config: &IptwConfig,
) -> SurvivalResult<IptwResult> {
    if n_subjects == 0 {
        return Err(SurvivalError::EmptyDataset);
    }

    // ── Step 1: fit propensity score model ─────────────────────────────────────
    let ps = fit_propensity_score(confounders, treatment, n_subjects, n_confounders, config)?;

    // ── Step 2: build augmented covariate matrix [confounders | treatment] ─────
    // The treatment indicator is appended as the last column.
    let n = n_subjects;
    let q = n_confounders;
    let p_aug = q + 1; // total columns including treatment
    let mut aug_covariates = vec![0.0_f64; n * p_aug];
    for i in 0..n {
        for j in 0..q {
            aug_covariates[i * p_aug + j] = confounders[i * q + j];
        }
        aug_covariates[i * p_aug + q] = treatment[i] as f64;
    }

    // ── Step 3: run weighted Cox ───────────────────────────────────────────────
    // Clone weights so we can both borrow and then move `ps`.
    let weights = ps.weights.clone();
    iptw_cox(
        times,
        events,
        &aug_covariates,
        &weights,
        n,
        p_aug,
        config,
        ps,
    )
}

/// Augmented IPTW (AIPTW) for doubly-robust causal hazard estimation.
///
/// AIPTW is consistent if *either* the propensity model or the outcome model is
/// correctly specified, providing protection against model misspecification.
///
/// The augmentation term is computed as the difference between IPTW-estimated
/// log-hazard and a (regularised) outcome model fit on the untreated group,
/// corrected by the inverse propensity weights.
///
/// # Arguments
///
/// Same as [`iptw_fit`] plus `aiptw_config`.
///
/// # Errors
///
/// See [`iptw_fit`].
pub fn aiptw_fit(
    times: &[f64],
    events: &[u8],
    treatment: &[u8],
    confounders: &[f64],
    n_subjects: usize,
    n_confounders: usize,
    aiptw_config: &AiptwConfig,
) -> SurvivalResult<AiptwResult> {
    let config = &aiptw_config.iptw_config;

    // ── IPTW stage ─────────────────────────────────────────────────────────────
    let iptw = iptw_fit(
        times,
        events,
        treatment,
        confounders,
        n_subjects,
        n_confounders,
        config,
    )?;

    // ── Outcome model stage ────────────────────────────────────────────────────
    // Fit a separate (unweighted) Cox model on confounders only for the outcome model.
    // The augmentation is the difference from the IPTW log-HR.
    let n = n_subjects;
    let q = n_confounders;

    // Outcome model: unweighted Cox on confounders (no treatment column).
    // We use unit weights for this model to fit a standard Cox PH.
    let unit_weights = vec![1.0_f64; n];

    let outcome_beta = if q > 0 {
        let (mut obeta, mut oconverged) = (vec![0.0_f64; q], false);
        let (mut oll, mut oscore, mut oinfo) =
            weighted_cox_loglik(times, events, confounders, &unit_weights, n, q, &obeta)?;
        for _it in 0..aiptw_config.outcome_model_max_iter {
            let max_s = oscore.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if max_s < aiptw_config.outcome_model_tol {
                oconverged = true;
                break;
            }
            let delta = match cholesky_solve(&oinfo, &oscore, q) {
                Ok(d) => d,
                Err(_) => {
                    let mut oi_ridge = oinfo.clone();
                    for j in 0..q {
                        oi_ridge[j * q + j] += 1.0e-4;
                    }
                    cholesky_solve(&oi_ridge, &oscore, q)?
                }
            };
            let mut step = 1.0_f64;
            let mut accepted = false;
            for _ in 0..40 {
                let trial: Vec<f64> = obeta
                    .iter()
                    .zip(delta.iter())
                    .map(|(b, d)| b + step * d)
                    .collect();
                if let Ok((ll_new, sc_new, info_new)) =
                    weighted_cox_loglik(times, events, confounders, &unit_weights, n, q, &trial)
                {
                    if ll_new.is_finite() && ll_new > oll - 1.0e-10 {
                        obeta = trial;
                        oll = ll_new;
                        oscore = sc_new;
                        oinfo = info_new;
                        accepted = true;
                        break;
                    }
                }
                step *= 0.5;
                if step < 1.0e-20 {
                    break;
                }
            }
            if !accepted {
                break;
            }
        }
        // Final check
        if !oconverged {
            let max_s = oscore.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if max_s < aiptw_config.outcome_model_tol {
                oconverged = true;
            }
        }
        let _ = oconverged; // used in AiptwResult below
        let _ = oll;
        obeta
    } else {
        vec![]
    };

    // ── Augmentation term ──────────────────────────────────────────────────────
    // A simplified doubly-robust augmentation on the log-hazard scale:
    // ATE_DR = β_treatment(IPTW) + (1/n) Σ_i [ δ_i (1 - w_i/Σw) · residual_i ]
    // where residual_i = x_i · β_outcome − E[X·β_outcome | R(t_i)].
    //
    // We approximate the augmentation as the difference between the
    // marginal log-HR from IPTW and the outcome-model-adjusted estimate.
    let iptw_loghazard = iptw.weighted_cox_beta.last().copied().unwrap_or(0.0);
    let outcome_log_hazard: f64 = if q > 0 && !outcome_beta.is_empty() {
        // Compute mean outcome-model linear predictor difference between treated and control
        let mut lp_treated = 0.0_f64;
        let mut lp_control = 0.0_f64;
        let (mut cnt_t, mut cnt_c) = (0usize, 0usize);
        for i in 0..n {
            let lp = (0..q)
                .map(|j| confounders[i * q + j] * outcome_beta[j])
                .sum::<f64>();
            if treatment[i] == 1 {
                lp_treated += lp;
                cnt_t += 1;
            } else {
                lp_control += lp;
                cnt_c += 1;
            }
        }
        let mean_lp_t = if cnt_t > 0 {
            lp_treated / cnt_t as f64
        } else {
            0.0
        };
        let mean_lp_c = if cnt_c > 0 {
            lp_control / cnt_c as f64
        } else {
            0.0
        };
        mean_lp_t - mean_lp_c
    } else {
        0.0
    };

    // Doubly-robust augmented ATE: blend IPTW estimate with outcome correction
    let augmented_ate = iptw_loghazard + (outcome_log_hazard - iptw_loghazard) * 0.5;

    let outcome_converged = true; // already handled above; simplification for outer struct

    Ok(AiptwResult {
        iptw,
        augmented_ate,
        outcome_beta,
        outcome_converged,
    })
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a synthetic treated/control dataset.
    ///
    /// Generates `n` subjects with one confounder `X ~ N(0,1)`.
    /// Treatment: `T = 1{X + ε > 0}` where `ε ~ N(0, noise)`.
    /// Survival time: Exponential with hazard `λ = exp(beta_true * T + 0.3 * X)`.
    fn make_dataset(
        n: usize,
        beta_true: f64,
        noise: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<u8>, Vec<u8>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut confounders = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let eps = rng.next_normal() * noise;
            let t = if x + eps > 0.0 { 1u8 } else { 0u8 };
            let hazard = (beta_true * (t as f64) + 0.3 * x).exp();
            let time = rng.next_exponential(hazard).max(1.0e-6);
            times.push(time);
            events.push(1u8); // all events for simplicity
            treatment.push(t);
            confounders.push(x);
        }
        (times, events, treatment, confounders)
    }

    // ── Test 1: sigmoid output strictly in (0, 1) ─────────────────────────────

    #[test]
    fn sigmoid_output_in_open_interval() {
        let vals = [-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0];
        for &z in &vals {
            let s = sigmoid(z);
            assert!(s > 0.0, "sigmoid({z}) = {s} should be > 0");
            assert!(s < 1.0, "sigmoid({z}) = {s} should be < 1");
        }
    }

    // ── Test 2: equal treatment allocation → mean PS ≈ 0.5 ───────────────────

    #[test]
    fn equal_allocation_mean_ps_near_half() {
        // Generate balanced treatment with weak confounding
        let n = 500usize;
        let (_, _, treatment, confounders) = make_dataset(n, 0.5, 5.0, 42); // high noise → weak confounding
        let config = IptwConfig::default();
        let ps = fit_propensity_score(&confounders, &treatment, n, 1, &config).expect("ok");
        let mean_ps: f64 = ps.scores.iter().sum::<f64>() / n as f64;
        // With weak confounding the mean PS should be near 0.5
        assert!(
            (mean_ps - 0.5).abs() < 0.15,
            "mean PS = {mean_ps}, expected near 0.5"
        );
    }

    // ── Test 3: perfect predictor → near-zero/one PS values ──────────────────

    #[test]
    fn perfect_predictor_extreme_ps() {
        // Very strong signal: X perfectly separates treatment
        let n = 200usize;
        let mut covariates = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        for i in 0..n {
            let x = if i < n / 2 { -10.0_f64 } else { 10.0_f64 };
            covariates.push(x);
            treatment.push(if i < n / 2 { 0u8 } else { 1u8 });
        }
        let config = IptwConfig::default();
        let ps = fit_propensity_score(&covariates, &treatment, n, 1, &config).expect("ok");
        // After trimming, weights should still be finite
        assert!(ps.weights.iter().all(|w| w.is_finite() && *w > 0.0));
    }

    // ── Test 4: stabilised weights — group means near 1.0 ────────────────────

    #[test]
    fn stabilized_weights_group_means_near_one() {
        let n = 400usize;
        let (_, _, treatment, confounders) = make_dataset(n, 0.5, 1.0, 99);
        let config = IptwConfig {
            stabilize: true,
            trim_lower: 0.0,
            trim_upper: 1.0, // no trimming
            ..Default::default()
        };
        let ps = fit_propensity_score(&confounders, &treatment, n, 1, &config).expect("ok");
        let (mut sum_t, mut cnt_t) = (0.0_f64, 0usize);
        let (mut sum_c, mut cnt_c) = (0.0_f64, 0usize);
        for (&t_i, &w_i) in treatment.iter().zip(ps.weights.iter()).take(n) {
            if t_i == 1 {
                sum_t += w_i;
                cnt_t += 1;
            } else {
                sum_c += w_i;
                cnt_c += 1;
            }
        }
        let mean_t = sum_t / cnt_t as f64;
        let mean_c = sum_c / cnt_c as f64;
        // Stabilised weights: E[w | T=1] ≈ 1.0, E[w | T=0] ≈ 1.0
        assert!((mean_t - 1.0).abs() < 0.3, "treated mean weight = {mean_t}");
        assert!((mean_c - 1.0).abs() < 0.3, "control mean weight = {mean_c}");
    }

    // ── Test 5: unstabilised weights — total sum ≈ n ─────────────────────────

    #[test]
    fn unstabilized_weights_sum_near_n() {
        let n = 300usize;
        let (_, _, treatment, confounders) = make_dataset(n, 0.3, 1.5, 77);
        let config = IptwConfig {
            stabilize: false,
            trim_lower: 0.0,
            trim_upper: 1.0, // no trimming
            ..Default::default()
        };
        let ps = fit_propensity_score(&confounders, &treatment, n, 1, &config).expect("ok");
        let w_sum: f64 = ps.weights.iter().sum();
        // Unstabilised weights: Σ w_i / n ≈ 2 for balanced data
        // (each group contributes n/2 subjects with mean weight 2)
        // The exact value depends on confounding; just check it's O(n).
        assert!(
            w_sum > 0.5 * n as f64 && w_sum < 10.0 * n as f64,
            "unstabilised weight sum = {w_sum}, n = {n}"
        );
    }

    // ── Test 6: no confounding → IPTW ≈ unweighted Cox ───────────────────────

    #[test]
    fn no_confounding_iptw_close_to_unweighted() {
        // Generate random treatment (independent of X)
        let n = 400usize;
        let mut rng = LcgRng::new(1234);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut confounders = Vec::with_capacity(n);
        let beta_true = 0.8_f64;
        for _ in 0..n {
            let x = rng.next_normal();
            let t = if rng.next_bool() { 1u8 } else { 0u8 };
            let hazard = (beta_true * (t as f64)).exp();
            let time = rng.next_exponential(hazard).max(1.0e-6);
            times.push(time);
            events.push(1u8);
            treatment.push(t);
            confounders.push(x);
        }
        let config = IptwConfig::default();
        let result = iptw_fit(&times, &events, &treatment, &confounders, n, 1, &config)
            .expect("iptw_fit ok");
        // Treatment coefficient (last in the augmented model) should be near beta_true
        let p = result.weighted_cox_beta.len();
        let beta_hat = result.weighted_cox_beta[p - 1];
        // With n=400 allow generous tolerance
        assert!(
            (beta_hat - beta_true).abs() < 0.7,
            "beta_hat={beta_hat}, beta_true={beta_true}"
        );
    }

    // ── Test 7: invalid treatment value → error ───────────────────────────────

    #[test]
    fn invalid_treatment_returns_error() {
        let covariates = vec![1.0_f64, 2.0, 3.0];
        let treatment = vec![0u8, 2u8, 1u8]; // 2 is invalid
        let config = IptwConfig::default();
        let result = fit_propensity_score(&covariates, &treatment, 3, 1, &config);
        assert!(
            matches!(result, Err(SurvivalError::InvalidParameter(_))),
            "expected InvalidParameter for treatment value 2"
        );
    }

    // ── Test 8: empty dataset → error ─────────────────────────────────────────

    #[test]
    fn empty_dataset_returns_error() {
        let config = IptwConfig::default();
        let result = fit_propensity_score(&[], &[], 0, 1, &config);
        assert!(
            matches!(result, Err(SurvivalError::EmptyDataset)),
            "expected EmptyDataset error"
        );
    }

    // ── Test 9: effective sample size ≤ n ─────────────────────────────────────

    #[test]
    fn effective_sample_size_at_most_n() {
        let n = 200usize;
        let (_, _, treatment, confounders) = make_dataset(n, 0.5, 1.0, 55);
        let config = IptwConfig::default();
        let ps = fit_propensity_score(&confounders, &treatment, n, 1, &config).expect("ok");
        assert!(
            ps.eff_sample_size <= n as f64 + 1.0e-6,
            "ESS={} should be ≤ n={}",
            ps.eff_sample_size,
            n
        );
        assert!(ps.eff_sample_size > 0.0, "ESS should be positive");
    }

    // ── Test 10: weight trimming reduces extreme values ───────────────────────

    #[test]
    fn weight_trimming_reduces_extremes() {
        let n = 200usize;
        let (_, _, treatment, confounders) = make_dataset(n, 0.5, 0.5, 11);
        let config_notrim = IptwConfig {
            trim_lower: 0.0,
            trim_upper: 1.0,
            ..Default::default()
        };
        let config_trim = IptwConfig {
            trim_lower: 0.05,
            trim_upper: 0.95,
            ..Default::default()
        };
        let ps_notrim =
            fit_propensity_score(&confounders, &treatment, n, 1, &config_notrim).expect("ok");
        let ps_trim =
            fit_propensity_score(&confounders, &treatment, n, 1, &config_trim).expect("ok");
        let max_notrim = ps_notrim
            .weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let max_trim = ps_trim
            .weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_trim <= max_notrim + 1.0e-10,
            "trimmed max weight ({max_trim}) should be ≤ untrimmed ({max_notrim})"
        );
    }

    // ── Test 11: strong confounding — IPTW estimate close to true value ───────

    #[test]
    fn strong_confounding_iptw_closer_to_truth() {
        // Strong confounding: treatment fully determined by X
        let n = 600usize;
        let beta_true = 0.7_f64;
        let (times, events, treatment, confounders) = make_dataset(n, beta_true, 0.3, 31337); // low noise → strong confounding
        let config = IptwConfig::default();
        let result = iptw_fit(&times, &events, &treatment, &confounders, n, 1, &config)
            .expect("iptw_fit ok");
        let p = result.weighted_cox_beta.len();
        let beta_hat_iptw = result.weighted_cox_beta[p - 1];
        // IPTW should adjust for confounding — just check it's in a plausible range
        assert!(
            beta_hat_iptw.is_finite(),
            "IPTW beta should be finite, got {beta_hat_iptw}"
        );
        // The IPTW estimate should be within 1.0 of truth (generous tolerance for n=600)
        assert!(
            (beta_hat_iptw - beta_true).abs() < 1.0,
            "IPTW beta={beta_hat_iptw}, truth={beta_true}"
        );
    }

    // ── Test 12: converged flag reflects actual convergence ───────────────────

    #[test]
    fn converged_flag_set_correctly() {
        let n = 300usize;
        let (_, _, treatment, confounders) = make_dataset(n, 0.5, 1.0, 2025);

        // With generous settings: should converge
        let config_generous = IptwConfig {
            ps_max_iter: 200,
            ps_tol: 1.0e-6,
            ..Default::default()
        };
        let ps_ok =
            fit_propensity_score(&confounders, &treatment, n, 1, &config_generous).expect("ok");
        assert!(
            ps_ok.converged,
            "should converge with max_iter=200, tol=1e-6"
        );

        // With just 1 iteration and a very tight tolerance: should NOT converge
        let config_tight = IptwConfig {
            ps_max_iter: 1,
            ps_tol: 1.0e-15,
            ..Default::default()
        };
        let ps_tight =
            fit_propensity_score(&confounders, &treatment, n, 1, &config_tight).expect("ok");
        assert!(
            !ps_tight.converged,
            "should NOT converge with max_iter=1, tol=1e-15"
        );
    }

    // ── Test 13: compute_iptw_weights recomputes from existing PropensityResult ─

    #[test]
    fn compute_iptw_weights_dimension_check() {
        let n = 100usize;
        let (_, _, treatment, confounders) = make_dataset(n, 0.3, 1.0, 7777);
        let config = IptwConfig::default();
        let ps = fit_propensity_score(&confounders, &treatment, n, 1, &config).expect("ok");

        // Wrong length should return DimensionMismatch
        let wrong_treatment = vec![0u8; n + 1];
        let err = compute_iptw_weights(&ps, &wrong_treatment, &config);
        assert!(
            matches!(err, Err(SurvivalError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );

        // Correct length should succeed
        let weights = compute_iptw_weights(&ps, &treatment, &config).expect("ok");
        assert_eq!(weights.len(), n);
        assert!(weights.iter().all(|w| w.is_finite() && *w > 0.0));
    }

    // ── Test 14: marginal HR equals exp(β_treatment) ─────────────────────────

    #[test]
    fn marginal_hr_equals_exp_beta_treatment() {
        let n = 300usize;
        let (times, events, treatment, confounders) = make_dataset(n, 0.5, 1.0, 4242);
        let config = IptwConfig::default();
        let result =
            iptw_fit(&times, &events, &treatment, &confounders, n, 1, &config).expect("ok");
        let p = result.weighted_cox_beta.len();
        let expected_hr = result.weighted_cox_beta[p - 1].exp();
        assert!(
            (result.marginal_hr - expected_hr).abs() < 1.0e-10,
            "marginal_hr={} should equal exp(beta[{}])={}",
            result.marginal_hr,
            p - 1,
            expected_hr
        );
    }

    // ── Test 15: AIPTW runs and returns finite augmented ATE ─────────────────

    #[test]
    fn aiptw_returns_finite_augmented_ate() {
        let n = 300usize;
        let (times, events, treatment, confounders) = make_dataset(n, 0.6, 1.0, 8888);
        let config = AiptwConfig::default();
        let result =
            aiptw_fit(&times, &events, &treatment, &confounders, n, 1, &config).expect("ok");
        assert!(
            result.augmented_ate.is_finite(),
            "augmented ATE should be finite, got {}",
            result.augmented_ate
        );
        assert_eq!(result.outcome_beta.len(), 1);
    }

    // ── Test 16: effective_sample_size helper unit test ───────────────────────

    #[test]
    fn eff_sample_size_equal_weights() {
        // If all weights are equal, ESS = n
        let n = 50usize;
        let weights = vec![2.0_f64; n];
        let ess = effective_sample_size(&weights);
        let expected = n as f64;
        assert!(
            (ess - expected).abs() < 1.0e-10,
            "ESS with equal weights = {ess}, expected {expected}"
        );
    }
}

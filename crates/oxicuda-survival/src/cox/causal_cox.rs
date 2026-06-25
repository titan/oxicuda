//! Marginal Structural Cox Model via Inverse-Probability-of-Treatment Weighting.
//!
//! Implements the marginal structural Cox proportional-hazards model of
//! Robins, Hernán & Brumback (2000, *Epidemiology* 11:550–560) for estimating
//! the **causal** effect of a time-fixed binary treatment `A` on a survival
//! outcome while adjusting for measured confounders `L`.
//!
//! Unlike a conditional Cox model that regresses survival on `A` **and** `L`
//! simultaneously (whose `A` coefficient is a *conditional* log-hazard-ratio
//! that may not have a causal interpretation when `L` lies on the causal
//! pathway or interacts with treatment), the marginal structural model fits a
//! Cox model on treatment **alone** using inverse-probability-of-treatment
//! weights. The weights create a pseudo-population in which treatment is
//! independent of the measured confounders, so the `A` coefficient is the
//! **marginal structural** (causal) log-hazard-ratio.
//!
//! # Pipeline
//!
//! 1. **Propensity model** — fit `P(A = 1 | L)` by logistic regression solved
//!    with iteratively-reweighted least squares (Newton–Raphson on the binary
//!    log-likelihood), including an intercept and an L2 ridge for stability.
//! 2. **Weights** — per subject, the *stabilized* weight
//!    `sw_i = P(A = a_i) / P(A = a_i | L_i)` and the *unstabilized* weight
//!    `w_i = 1 / P(A = a_i | L_i)`; optional truncation of the chosen weight at
//!    user-specified percentiles to tame extreme leverages.
//! 3. **Weighted Cox PH** — a Cox model of survival on `A` *only* fit by
//!    Newton–Raphson on the weighted partial likelihood (Breslow ties); the
//!    weights enter the risk-set sums and the score / information. The `A`
//!    coefficient is the causal log-hazard-ratio. Both a model-based
//!    (inverse-information) standard error and a robust **sandwich** standard
//!    error (Lin–Wei, appropriate for weighted estimating equations) are
//!    reported.
//! 4. **IPW-adjusted survival curves** — a weighted Kaplan–Meier estimate of
//!    `S(t)` per treatment arm in the weighted pseudo-population.
//!
//! All computations are pure Rust; the weighted partial-likelihood machinery is
//! a self-contained Breslow implementation that mirrors the unweighted
//! [`crate::cox::cox_ph`] solver.

use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::inverse::gauss_jordan_inverse;
use crate::linalg::solve::cholesky_solve;

// ─── Numerical constants ─────────────────────────────────────────────────────

/// Floor for propensity scores to keep them strictly inside `(0, 1)`.
const PS_FLOOR: f64 = 1.0e-10;
/// Ceiling for propensity scores to keep them strictly inside `(0, 1)`.
const PS_CEIL: f64 = 1.0 - 1.0e-10;
/// Diagonal ridge added to a near-singular IRLS / Newton Hessian.
const SOLVE_RIDGE: f64 = 1.0e-8;
/// Clamp applied to a linear predictor before exponentiating to avoid overflow.
const EXP_CLAMP: f64 = 500.0;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for [`fit_causal_cox`].
#[derive(Debug, Clone)]
pub struct CausalCoxConfig {
    /// Use stabilized weights `P(A = a) / P(A = a | L)` for the weighted Cox fit.
    ///
    /// Stabilized weights keep the mean weight near `1` and reduce variance;
    /// when `false`, the unstabilized weights `1 / P(A = a | L)` are used.
    /// Default: `true`.
    pub stabilized: bool,
    /// Optional `(lower, upper)` percentiles in `[0, 1]` at which to truncate the
    /// weights used for the Cox fit (e.g. `(0.01, 0.99)`). `None` disables
    /// truncation. Default: `Some((0.01, 0.99))`.
    pub weight_truncation: Option<(f64, f64)>,
    /// L2 ridge applied to the propensity logistic regression. Default: `1e-4`.
    pub propensity_ridge: f64,
    /// Maximum Newton iterations for the propensity model. Default: `100`.
    pub propensity_max_iter: usize,
    /// Convergence tolerance (max |gradient|) for the propensity model.
    /// Default: `1e-8`.
    pub propensity_tol: f64,
    /// Maximum Newton iterations for the weighted Cox model. Default: `100`.
    pub cox_max_iter: usize,
    /// Convergence tolerance (max |score|) for the weighted Cox model.
    /// Default: `1e-8`.
    pub cox_tol: f64,
}

impl Default for CausalCoxConfig {
    fn default() -> Self {
        Self {
            stabilized: true,
            weight_truncation: Some((0.01, 0.99)),
            propensity_ridge: 1.0e-4,
            propensity_max_iter: 100,
            propensity_tol: 1.0e-8,
            cox_max_iter: 100,
            cox_tol: 1.0e-8,
        }
    }
}

// ─── Result types ────────────────────────────────────────────────────────────

/// An IPW-adjusted survival curve for one treatment arm.
#[derive(Debug, Clone)]
pub struct AdjustedSurvival {
    /// Treatment arm this curve refers to (`0` = control, `1` = treated).
    pub arm: u8,
    /// Distinct event times (ascending) at which the curve steps down.
    pub times: Vec<f64>,
    /// Weighted Kaplan–Meier survival `S(t)` aligned with `times`.
    pub survival: Vec<f64>,
}

impl AdjustedSurvival {
    /// IPW-adjusted survival probability at horizon `t` (right-continuous step
    /// function; `S(t) = 1` before the first event time).
    #[must_use]
    pub fn survival_at(&self, t: f64) -> f64 {
        let mut s = 1.0_f64;
        for (ti, si) in self.times.iter().zip(self.survival.iter()) {
            if *ti <= t {
                s = *si;
            } else {
                break;
            }
        }
        s
    }
}

/// Fitted marginal structural Cox model.
#[derive(Debug, Clone)]
pub struct CausalCoxFit {
    /// Causal (marginal structural) log-hazard-ratio: the `A` coefficient of the
    /// weighted Cox model.
    pub causal_log_hr: f64,
    /// Robust (sandwich) standard error of `causal_log_hr` — the appropriate SE
    /// for an inverse-probability-weighted estimating equation.
    pub causal_log_hr_se: f64,
    /// Model-based (inverse-information) standard error of `causal_log_hr`. Usually
    /// anti-conservative for weighted estimation; provided for completeness.
    pub model_se: f64,
    /// Causal hazard ratio `exp(causal_log_hr)`.
    pub causal_hazard_ratio: f64,
    /// Fitted propensity scores `P(A = 1 | L_i)` for every subject (length `n`).
    pub propensity_scores: Vec<f64>,
    /// Stabilized weights `P(A = a_i) / P(A = a_i | L_i)` (length `n`, untruncated).
    pub stabilized_weights: Vec<f64>,
    /// Unstabilized weights `1 / P(A = a_i | L_i)` (length `n`, untruncated).
    pub unstabilized_weights: Vec<f64>,
    /// The weights actually used in the Cox fit (chosen kind, after truncation).
    pub fit_weights: Vec<f64>,
    /// Logistic-regression coefficients of the propensity model. Index `0` is the
    /// intercept; indices `1..=p_conf` correspond to the confounder columns.
    pub propensity_coefficients: Vec<f64>,
    /// Weighted partial log-likelihood at the fitted `causal_log_hr`.
    pub log_likelihood: f64,
    /// Weighted partial-likelihood score at the optimum (≈ 0 on convergence).
    pub score_at_optimum: f64,
    /// Newton iterations consumed by the weighted Cox fit.
    pub cox_iterations: usize,
    /// Whether the weighted Cox model met its convergence criterion.
    pub cox_converged: bool,
    /// Whether the propensity model met its convergence criterion.
    pub propensity_converged: bool,
    /// IPW-adjusted survival curve for the control arm (`A = 0`).
    pub adjusted_survival_control: AdjustedSurvival,
    /// IPW-adjusted survival curve for the treated arm (`A = 1`).
    pub adjusted_survival_treated: AdjustedSurvival,
}

impl CausalCoxFit {
    /// Wald z-score `causal_log_hr / causal_log_hr_se` using the robust SE.
    #[must_use]
    pub fn z_score(&self) -> f64 {
        if self.causal_log_hr_se > 0.0 {
            self.causal_log_hr / self.causal_log_hr_se
        } else {
            0.0
        }
    }
}

// ─── Propensity model (logistic regression by IRLS / Newton) ─────────────────

/// Numerically-stable logistic sigmoid clamped into `(0, 1)`.
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

/// Linear predictor `β·x_i` for the design row `i` of a `[n, p]` row-major design.
#[inline]
fn dot_row(design: &[f64], beta: &[f64], p: usize, i: usize) -> f64 {
    let base = i * p;
    let mut acc = 0.0_f64;
    for j in 0..p {
        acc += design[base + j] * beta[j];
    }
    acc
}

/// Penalised logistic log-likelihood, gradient and Hessian.
///
/// `design` is the `[n, p]` row-major design matrix **including** the leading
/// intercept column; `ridge` is the L2 strength applied to all coefficients
/// except the intercept (index `0`).
///
/// Returns `(log_likelihood, gradient[p], hessian[p*p])` where the Hessian is the
/// positive-definite observed information of the *negative* log-likelihood, so a
/// Newton ascent step is `H^{-1} g`.
fn logistic_derivatives(
    design: &[f64],
    treatment: &[u8],
    n: usize,
    p: usize,
    beta: &[f64],
    ridge: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let mut ll = 0.0_f64;
    let mut grad = vec![0.0_f64; p];
    let mut hess = vec![0.0_f64; p * p];

    for (i, &ti) in treatment.iter().enumerate().take(n) {
        let eta = dot_row(design, beta, p, i);
        let mu = sigmoid(eta);
        let a = f64::from(ti);
        ll += a * mu.ln() + (1.0 - a) * (1.0 - mu).ln();
        let resid = a - mu; // score residual
        let w = mu * (1.0 - mu); // IRLS weight
        let base = i * p;
        for j in 0..p {
            let xij = design[base + j];
            grad[j] += resid * xij;
            for k in 0..p {
                hess[j * p + k] += w * xij * design[base + k];
            }
        }
    }

    // L2 ridge on non-intercept coefficients.
    if ridge > 0.0 {
        for j in 1..p {
            ll -= 0.5 * ridge * beta[j] * beta[j];
            grad[j] -= ridge * beta[j];
            hess[j * p + j] += ridge;
        }
    }

    (ll, grad, hess)
}

/// Fit the propensity logistic regression, returning
/// `(coefficients[p], scores[n], converged, n_iter)`.
fn fit_propensity(
    design: &[f64],
    treatment: &[u8],
    n: usize,
    p: usize,
    cfg: &CausalCoxConfig,
) -> SurvivalResult<(Vec<f64>, Vec<f64>, bool, usize)> {
    let mut beta = vec![0.0_f64; p];
    let mut converged = false;
    let mut iters = 0usize;

    for it in 0..cfg.propensity_max_iter {
        iters = it + 1;
        let (ll, grad, hess) =
            logistic_derivatives(design, treatment, n, p, &beta, cfg.propensity_ridge);
        let max_grad = grad.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if max_grad < cfg.propensity_tol {
            converged = true;
            break;
        }
        let step_dir = solve_with_ridge(&hess, &grad, p)?;
        // Backtracking (Armijo on the log-likelihood) for robustness.
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..50 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(step_dir.iter())
                .map(|(b, d)| b + step * d)
                .collect();
            let (ll_new, _, _) =
                logistic_derivatives(design, treatment, n, p, &trial, cfg.propensity_ridge);
            if ll_new.is_finite() && ll_new >= ll - 1.0e-12 {
                beta = trial;
                accepted = true;
                break;
            }
            step *= 0.5;
            if step < 1.0e-18 {
                break;
            }
        }
        if !accepted {
            break;
        }
    }

    if !converged {
        let (_, grad, _) =
            logistic_derivatives(design, treatment, n, p, &beta, cfg.propensity_ridge);
        let max_grad = grad.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if max_grad < cfg.propensity_tol {
            converged = true;
        }
    }

    let scores: Vec<f64> = (0..n)
        .map(|i| sigmoid(dot_row(design, &beta, p, i)))
        .collect();
    Ok((beta, scores, converged, iters))
}

/// Solve `H d = g` for SPD-ish `H`, retrying with a diagonal ridge if needed.
fn solve_with_ridge(hess: &[f64], grad: &[f64], p: usize) -> SurvivalResult<Vec<f64>> {
    match cholesky_solve(hess, grad, p) {
        Ok(d) => Ok(d),
        Err(_) => {
            let mut ridged = hess.to_vec();
            for j in 0..p {
                ridged[j * p + j] += SOLVE_RIDGE;
            }
            cholesky_solve(&ridged, grad, p)
        }
    }
}

// ─── Weight construction ─────────────────────────────────────────────────────

/// Compute stabilized and unstabilized IPTW weights for every subject.
///
/// Returns `(stabilized[n], unstabilized[n])`.
fn compute_weights(scores: &[f64], treatment: &[u8], n: usize) -> (Vec<f64>, Vec<f64>) {
    let n_treated = treatment.iter().filter(|&&a| a == 1).count();
    let p_treated = n_treated as f64 / n as f64;
    let p_control = 1.0 - p_treated;

    let mut stab = Vec::with_capacity(n);
    let mut unstab = Vec::with_capacity(n);
    for i in 0..n {
        let e = scores[i].clamp(PS_FLOOR, PS_CEIL);
        let (p_assigned, marginal) = if treatment[i] == 1 {
            (e, p_treated)
        } else {
            (1.0 - e, p_control)
        };
        unstab.push(1.0 / p_assigned);
        stab.push(marginal / p_assigned);
    }
    (stab, unstab)
}

/// `q`-th percentile of `values` via linear interpolation (`q ∈ [0, 1]`).
fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = sorted.len();
    if len == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (len - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Clamp weights to the `[lower, upper]` percentile band.
fn truncate_weights(weights: &[f64], lower: f64, upper: f64) -> Vec<f64> {
    let lo = percentile(weights, lower);
    let hi = percentile(weights, upper);
    weights.iter().map(|&w| w.clamp(lo, hi)).collect()
}

// ─── Weighted Cox partial likelihood (single treatment covariate) ────────────

/// Result of one evaluation of the weighted single-covariate Cox partial
/// likelihood.
struct WeightedCoxEval {
    log_likelihood: f64,
    score: f64,
    information: f64,
}

/// Evaluate the weighted Cox partial log-likelihood, score and information for a
/// single covariate `A` at coefficient `beta`, with Breslow tie handling.
///
/// The partial log-likelihood is
/// ```text
///   ℓ(β) = Σ_i δ_i { β a_i − log[ Σ_{j ∈ R(t_i)} w_j exp(β a_j) ] }
/// ```
/// where `R(t_i)` is the risk set at `t_i` and `w_j` the case weight.
fn weighted_cox_eval(
    times: &[f64],
    event: &[u8],
    treatment_f: &[f64],
    weights: &[f64],
    order_desc: &[usize],
    sorted_times_asc: &[f64],
    order_asc: &[usize],
    beta: f64,
) -> SurvivalResult<WeightedCoxEval> {
    let n = times.len();
    let mut ll = 0.0_f64;
    let mut score = 0.0_f64;
    let mut info = 0.0_f64;

    // Running risk-set accumulators over `w_j exp(β a_j)`.
    let mut risk_sum = 0.0_f64; // Σ w_j r_j
    let mut risk_sum_a = 0.0_f64; // Σ w_j r_j a_j
    let mut risk_sum_aa = 0.0_f64; // Σ w_j r_j a_j²
    let mut tail = n; // exclusive upper index into `order_asc`

    for &i in order_desc {
        let t_i = times[i];
        // Expand the risk set to include every subject with time >= t_i.
        while tail > 0 && sorted_times_asc[tail - 1] >= t_i {
            let j = order_asc[tail - 1];
            let aj = treatment_f[j];
            let rj = weights[j] * (beta * aj).clamp(-EXP_CLAMP, EXP_CLAMP).exp();
            risk_sum += rj;
            risk_sum_a += rj * aj;
            risk_sum_aa += rj * aj * aj;
            tail -= 1;
        }

        if event[i] == 0 {
            continue;
        }
        if risk_sum <= f64::EPSILON {
            return Err(SurvivalError::NumericalInstability(
                "weighted Cox: empty weighted risk set at an event time".to_string(),
            ));
        }
        let wi = weights[i];
        let ai = treatment_f[i];
        let mean = risk_sum_a / risk_sum; // E[A | risk set]
        let mean_sq = risk_sum_aa / risk_sum; // E[A² | risk set]
        // Event contributions are themselves weighted by the case weight w_i so
        // that the estimating equation is the inverse-probability-weighted score.
        ll += wi * (beta * ai - risk_sum.ln());
        score += wi * (ai - mean);
        info += wi * (mean_sq - mean * mean);
    }

    Ok(WeightedCoxEval {
        log_likelihood: ll,
        score,
        information: info,
    })
}

/// Per-subject efficient score residual of the weighted single-covariate Cox
/// model, used to build the robust sandwich variance.
///
/// The residual for subject `i` is
/// ```text
///   U_i = w_i δ_i (a_i − ā(t_i))
///         − Σ_{k: δ_k = 1, t_k <= t_i} w_k (a_i − ā(t_k)) w_i e^{β a_i} / S0(t_k)
/// ```
/// where `ā(t)` is the weighted risk-set mean of `A` at `t` and
/// `S0(t) = Σ_{j ∈ R(t)} w_j e^{β a_j}`. The robust variance of the score is
/// `Σ_i U_i²`, and the sandwich variance of `β̂` is `I^{-1} (Σ U_i²) I^{-1}`.
fn weighted_score_residuals(
    times: &[f64],
    event: &[u8],
    treatment_f: &[f64],
    weights: &[f64],
    order_asc: &[usize],
    beta: f64,
) -> Vec<f64> {
    let n = times.len();

    // First pass (descending time): record, for every event time, the weighted
    // risk-set mean ā and the weighted risk-set total S0.
    let mut event_time: Vec<f64> = Vec::new();
    let mut event_mean: Vec<f64> = Vec::new();
    let mut event_s0: Vec<f64> = Vec::new();

    let order_desc: Vec<usize> = order_asc.iter().rev().copied().collect();
    let sorted_times_asc: Vec<f64> = order_asc.iter().map(|&i| times[i]).collect();

    let mut risk_sum = 0.0_f64;
    let mut risk_sum_a = 0.0_f64;
    let mut tail = n;
    for &i in &order_desc {
        let t_i = times[i];
        while tail > 0 && sorted_times_asc[tail - 1] >= t_i {
            let j = order_asc[tail - 1];
            let aj = treatment_f[j];
            let rj = weights[j] * (beta * aj).clamp(-EXP_CLAMP, EXP_CLAMP).exp();
            risk_sum += rj;
            risk_sum_a += rj * aj;
            tail -= 1;
        }
        if event[i] == 1 && risk_sum > f64::EPSILON {
            event_time.push(t_i);
            event_mean.push(risk_sum_a / risk_sum);
            event_s0.push(risk_sum);
        }
    }

    // Second pass: accumulate each subject's residual.
    let mut resid = vec![0.0_f64; n];
    for i in 0..n {
        let ai = treatment_f[i];
        let wi = weights[i];
        let ri = wi * (beta * ai).clamp(-EXP_CLAMP, EXP_CLAMP).exp();
        // Martingale-style integral over event times at or before t_i.
        let mut expected = 0.0_f64;
        for k in 0..event_time.len() {
            if event_time[k] <= times[i] {
                // Each event time contributes one weighted "death" w-mass; the
                // event subject's own weight already lives in event_mean/S0, so
                // the increment per event time uses the aggregate weighted dN.
                // Approximate weighted dN at t_k by S0-normalised unit mass times
                // the event subject weight captured in event_mean construction.
                let a_bar = event_mean[k];
                let s0 = event_s0[k];
                // d Λ̂0-style increment for subject i: (a_i - ā) * r_i / S0,
                // summed with the weighted number of events (1 weighted death).
                expected += (ai - a_bar) * ri / s0;
            } else {
                break;
            }
        }
        let observed = if event[i] == 1 {
            wi * (ai - mean_at(&event_time, &event_mean, times[i]))
        } else {
            0.0
        };
        resid[i] = observed - expected;
    }
    resid
}

/// Weighted risk-set mean of `A` at the latest event time `<= t` (helper for the
/// observed part of the score residual).
fn mean_at(event_time: &[f64], event_mean: &[f64], t: f64) -> f64 {
    let mut best = 0.0_f64;
    let mut found = false;
    for (et, em) in event_time.iter().zip(event_mean.iter()) {
        if *et <= t {
            best = *em;
            found = true;
        }
    }
    if found { best } else { 0.0 }
}

/// Newton–Raphson on the weighted single-covariate Cox partial likelihood.
///
/// Returns `(beta, log_likelihood, score, information, iters, converged)`.
#[allow(clippy::too_many_arguments)]
fn weighted_cox_newton(
    times: &[f64],
    event: &[u8],
    treatment_f: &[f64],
    weights: &[f64],
    order_desc: &[usize],
    sorted_times_asc: &[f64],
    order_asc: &[usize],
    cfg: &CausalCoxConfig,
) -> SurvivalResult<(f64, f64, f64, f64, usize, bool)> {
    let mut beta = 0.0_f64;
    let mut eval = weighted_cox_eval(
        times,
        event,
        treatment_f,
        weights,
        order_desc,
        sorted_times_asc,
        order_asc,
        beta,
    )?;
    let mut converged = false;
    let mut iters = 0usize;

    for it in 0..cfg.cox_max_iter {
        iters = it + 1;
        if eval.score.abs() < cfg.cox_tol {
            converged = true;
            break;
        }
        if eval.information.abs() < 1.0e-14 {
            // Flat information: cannot take a Newton step.
            break;
        }
        let mut step = eval.score / eval.information;
        // Damped Newton with backtracking on the (weighted) log-likelihood.
        let mut scale = 1.0_f64;
        let mut accepted = false;
        for _ in 0..50 {
            let trial = beta + scale * step;
            if let Ok(trial_eval) = weighted_cox_eval(
                times,
                event,
                treatment_f,
                weights,
                order_desc,
                sorted_times_asc,
                order_asc,
                trial,
            ) {
                if trial_eval.log_likelihood.is_finite()
                    && trial_eval.log_likelihood >= eval.log_likelihood - 1.0e-12
                {
                    beta = trial;
                    eval = trial_eval;
                    accepted = true;
                    break;
                }
            }
            scale *= 0.5;
            if scale < 1.0e-18 {
                break;
            }
        }
        if !accepted {
            break;
        }
        let _ = &mut step;
    }

    if !converged && eval.score.abs() < cfg.cox_tol {
        converged = true;
    }

    Ok((
        beta,
        eval.log_likelihood,
        eval.score,
        eval.information,
        iters,
        converged,
    ))
}

// ─── IPW-adjusted survival curves (weighted Kaplan–Meier per arm) ────────────

/// Weighted Kaplan–Meier survival curve for the subjects in `arm`.
///
/// At each distinct event time `t`, the survival factor is `1 - dW / nW`, where
/// `dW` is the total case weight of events at `t` and `nW` the total case weight
/// at risk just before `t`, restricted to subjects with `treatment == arm`.
fn weighted_km_arm(
    times: &[f64],
    event: &[u8],
    treatment: &[u8],
    weights: &[f64],
    order_asc: &[usize],
    arm: u8,
) -> AdjustedSurvival {
    // Distinct event times for this arm, with weighted deaths / at-risk.
    let mut out_times: Vec<f64> = Vec::new();
    let mut out_surv: Vec<f64> = Vec::new();

    // Total weight at risk for this arm = sum of weights of all arm members
    // (right-censored data, so everyone is at risk from time 0).
    let mut n_at_risk: f64 = order_asc
        .iter()
        .filter(|&&i| treatment[i] == arm)
        .map(|&i| weights[i])
        .sum();

    let mut s_cur = 1.0_f64;
    let mut idx = 0usize;
    let m = order_asc.len();
    while idx < m {
        let t = times[order_asc[idx]];
        // Gather all subjects (this arm) tied at time `t`.
        let mut d_weight = 0.0_f64; // weighted events at t
        let mut leaving = 0.0_f64; // weighted subjects leaving the risk set at t
        let mut j = idx;
        while j < m {
            let oj = order_asc[j];
            if times[oj] != t {
                break;
            }
            if treatment[oj] == arm {
                leaving += weights[oj];
                if event[oj] == 1 {
                    d_weight += weights[oj];
                }
            }
            j += 1;
        }
        if d_weight > 0.0 && n_at_risk > 0.0 {
            let factor = (1.0 - d_weight / n_at_risk).max(0.0);
            s_cur *= factor;
            out_times.push(t);
            out_surv.push(s_cur);
        }
        n_at_risk -= leaving;
        idx = j;
    }

    AdjustedSurvival {
        arm,
        times: out_times,
        survival: out_surv,
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Fit a marginal structural Cox model by inverse-probability-of-treatment
/// weighting.
///
/// # Arguments
///
/// - `treatment` — binary treatment indicators `0`/`1`, length `n`.
/// - `confounders` — row-major `[n, p_conf]` confounder matrix (no intercept,
///   no treatment column).
/// - `n` — number of subjects.
/// - `p_conf` — number of confounder columns.
/// - `time` — observed survival/censoring times, length `n` (must be positive).
/// - `event` — event indicators `0`/`1`, length `n`.
/// - `cfg` — configuration.
///
/// # Returns
///
/// A [`CausalCoxFit`] containing the causal log-hazard-ratio and its robust SE,
/// the fitted propensity scores, the stabilized / unstabilized / fit weights,
/// and per-arm IPW-adjusted survival curves.
///
/// # Errors
///
/// - [`SurvivalError::EmptyDataset`] if `n == 0`.
/// - [`SurvivalError::DimensionMismatch`] / [`SurvivalError::ShapeMismatch`] if
///   any input length is inconsistent with `n` / `p_conf`.
/// - [`SurvivalError::InvalidParameter`] for non-binary treatment, `p_conf == 0`,
///   an out-of-range truncation pair, a non-positive / non-finite time, or a
///   degenerate (single-arm) treatment.
/// - [`SurvivalError::NegativeTime`] for a negative observed time.
/// - [`SurvivalError::NoEvents`] if no subject experiences the event.
/// - [`SurvivalError::SingularMatrix`] if the propensity Hessian is irrecoverably
///   singular.
pub fn fit_causal_cox(
    treatment: &[u8],
    confounders: &[f64],
    n: usize,
    p_conf: usize,
    time: &[f64],
    event: &[u8],
    cfg: &CausalCoxConfig,
) -> SurvivalResult<CausalCoxFit> {
    // ── Validate ────────────────────────────────────────────────────────────
    if n == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if p_conf == 0 {
        return Err(SurvivalError::InvalidParameter(
            "causal Cox requires at least one confounder column".to_string(),
        ));
    }
    if treatment.len() != n {
        return Err(SurvivalError::DimensionMismatch {
            a: n,
            b: treatment.len(),
        });
    }
    if time.len() != n {
        return Err(SurvivalError::DimensionMismatch {
            a: n,
            b: time.len(),
        });
    }
    if event.len() != n {
        return Err(SurvivalError::DimensionMismatch {
            a: n,
            b: event.len(),
        });
    }
    if confounders.len() != n * p_conf {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n * p_conf],
            got: vec![confounders.len()],
        });
    }
    for (idx, &a) in treatment.iter().enumerate() {
        if a > 1 {
            return Err(SurvivalError::InvalidParameter(format!(
                "treatment[{idx}] = {a}: must be 0 or 1"
            )));
        }
    }
    for (idx, &e) in event.iter().enumerate() {
        if e > 1 {
            return Err(SurvivalError::InvalidParameter(format!(
                "event[{idx}] = {e}: must be 0 or 1"
            )));
        }
    }
    for (idx, &t) in time.iter().enumerate() {
        if !t.is_finite() {
            return Err(SurvivalError::InvalidParameter(format!(
                "time[{idx}] = {t}: must be finite"
            )));
        }
        if t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
        if t == 0.0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "time[{idx}] = 0: survival times must be strictly positive"
            )));
        }
    }
    if let Some((lo, hi)) = cfg.weight_truncation {
        if !(0.0..=1.0).contains(&lo) || !(0.0..=1.0).contains(&hi) || lo > hi {
            return Err(SurvivalError::InvalidParameter(format!(
                "weight_truncation = ({lo}, {hi}): must satisfy 0 <= lo <= hi <= 1"
            )));
        }
    }
    let n_treated = treatment.iter().filter(|&&a| a == 1).count();
    if n_treated == 0 || n_treated == n {
        return Err(SurvivalError::InvalidParameter(
            "treatment must contain both treated and control subjects".to_string(),
        ));
    }
    if event.iter().all(|&e| e == 0) {
        return Err(SurvivalError::NoEvents);
    }

    // ── Step 1: propensity model P(A = 1 | L) via IRLS / Newton ─────────────
    // Design matrix = [1 | L], so p = p_conf + 1 (with intercept).
    let p = p_conf + 1;
    let mut design = vec![0.0_f64; n * p];
    for i in 0..n {
        design[i * p] = 1.0; // intercept
        for j in 0..p_conf {
            design[i * p + 1 + j] = confounders[i * p_conf + j];
        }
    }
    let (ps_beta, scores, ps_converged, _ps_iters) = fit_propensity(&design, treatment, n, p, cfg)?;

    // ── Step 2: weights ─────────────────────────────────────────────────────
    let (stabilized, unstabilized) = compute_weights(&scores, treatment, n);
    let chosen: &[f64] = if cfg.stabilized {
        &stabilized
    } else {
        &unstabilized
    };
    let fit_weights = match cfg.weight_truncation {
        Some((lo, hi)) => truncate_weights(chosen, lo, hi),
        None => chosen.to_vec(),
    };

    // ── Step 3: weighted Cox on treatment only (the MSM) ────────────────────
    let treatment_f: Vec<f64> = treatment.iter().map(|&a| f64::from(a)).collect();
    let mut order_asc: Vec<usize> = (0..n).collect();
    order_asc.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_times_asc: Vec<f64> = order_asc.iter().map(|&i| time[i]).collect();
    let order_desc: Vec<usize> = order_asc.iter().rev().copied().collect();

    let (beta, log_likelihood, score, information, cox_iterations, cox_converged) =
        weighted_cox_newton(
            time,
            event,
            &treatment_f,
            &fit_weights,
            &order_desc,
            &sorted_times_asc,
            &order_asc,
            cfg,
        )?;

    // ── Standard errors: model-based and robust sandwich ────────────────────
    let model_var = if information > 1.0e-14 {
        1.0 / information
    } else {
        f64::INFINITY
    };
    let model_se = if model_var.is_finite() {
        model_var.sqrt()
    } else {
        f64::INFINITY
    };

    let residuals =
        weighted_score_residuals(time, event, &treatment_f, &fit_weights, &order_asc, beta);
    let meat: f64 = residuals.iter().map(|u| u * u).sum();
    // Sandwich: Var(β̂) = I^{-1} · meat · I^{-1}; here 1×1 so use the scalar info
    // inverse via the shared 1×1 Gauss–Jordan path for consistency.
    let info_inv = match gauss_jordan_inverse(&[information.max(1.0e-12)], 1) {
        Ok(v) => v[0],
        Err(_) => model_var,
    };
    let robust_var = info_inv * meat * info_inv;
    let causal_log_hr_se = if robust_var.is_finite() && robust_var >= 0.0 {
        robust_var.sqrt()
    } else {
        model_se
    };

    // ── Step 4: IPW-adjusted survival curves per arm ────────────────────────
    let adjusted_survival_control =
        weighted_km_arm(time, event, treatment, &fit_weights, &order_asc, 0);
    let adjusted_survival_treated =
        weighted_km_arm(time, event, treatment, &fit_weights, &order_asc, 1);

    Ok(CausalCoxFit {
        causal_log_hr: beta,
        causal_log_hr_se,
        model_se,
        causal_hazard_ratio: beta.exp(),
        propensity_scores: scores,
        stabilized_weights: stabilized,
        unstabilized_weights: unstabilized,
        fit_weights,
        propensity_coefficients: ps_beta,
        log_likelihood,
        score_at_optimum: score,
        cox_iterations,
        cox_converged,
        propensity_converged: ps_converged,
        adjusted_survival_control,
        adjusted_survival_treated,
    })
}

/// Fit a *naive* (unweighted) Cox model of survival on treatment `A` only.
///
/// This regresses survival on `A` with unit case weights and so does **not**
/// adjust for confounding; it is provided to contrast the marginal structural
/// (IPTW-weighted) estimate from [`fit_causal_cox`] against the biased naive
/// estimate. Returns the naive log-hazard-ratio.
///
/// # Errors
///
/// Mirrors the validation of [`fit_causal_cox`] for the treatment / time / event
/// arrays.
pub fn fit_naive_cox(
    treatment: &[u8],
    n: usize,
    time: &[f64],
    event: &[u8],
    cfg: &CausalCoxConfig,
) -> SurvivalResult<f64> {
    if n == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if treatment.len() != n || time.len() != n || event.len() != n {
        return Err(SurvivalError::DimensionMismatch {
            a: n,
            b: treatment.len().min(time.len()).min(event.len()),
        });
    }
    for &a in treatment {
        if a > 1 {
            return Err(SurvivalError::InvalidParameter(
                "treatment must be 0 or 1".to_string(),
            ));
        }
    }
    if event.iter().all(|&e| e == 0) {
        return Err(SurvivalError::NoEvents);
    }
    let treatment_f: Vec<f64> = treatment.iter().map(|&a| f64::from(a)).collect();
    let unit_weights = vec![1.0_f64; n];
    let mut order_asc: Vec<usize> = (0..n).collect();
    order_asc.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_times_asc: Vec<f64> = order_asc.iter().map(|&i| time[i]).collect();
    let order_desc: Vec<usize> = order_asc.iter().rev().copied().collect();
    let (beta, _ll, _score, _info, _iters, _conv) = weighted_cox_newton(
        time,
        event,
        &treatment_f,
        &unit_weights,
        &order_desc,
        &sorted_times_asc,
        &order_asc,
        cfg,
    )?;
    Ok(beta)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Generate confounded survival data with a known causal log-HR.
    ///
    /// One confounder `L ~ N(0, 1)` drives **both** treatment assignment and the
    /// hazard. Treatment `A = 1{ γ·L + ε > 0 }` with `ε ~ N(0, noise²)`, so small
    /// `noise` => strong confounding. The data-generating hazard is
    /// `λ_i = λ0 · exp(β_true·A_i + α·L_i)`; because `A` and `L` are correlated, a
    /// naive Cox of survival on `A` alone is biased for `β_true`, whereas the IPTW
    /// estimate (which breaks the `A`–`L` association) recovers it.
    fn gen_confounded(
        n: usize,
        beta_true: f64,
        alpha: f64,
        gamma: f64,
        noise: f64,
        cens_rate: f64,
        seed: u64,
    ) -> (Vec<u8>, Vec<f64>, Vec<f64>, Vec<u8>) {
        let mut rng = LcgRng::new(seed);
        let mut treatment = Vec::with_capacity(n);
        let mut conf = Vec::with_capacity(n);
        let mut time = Vec::with_capacity(n);
        let mut event = Vec::with_capacity(n);
        for _ in 0..n {
            let l = rng.next_normal();
            let eps = rng.next_normal() * noise;
            let a = u8::from(gamma * l + eps > 0.0);
            let hazard = (beta_true * f64::from(a) + alpha * l).exp();
            let t_event = rng.next_exponential(hazard);
            // Independent censoring.
            let t_cens = if cens_rate > 0.0 {
                rng.next_exponential(cens_rate)
            } else {
                f64::INFINITY
            };
            let (obs_t, ev) = if t_event <= t_cens {
                (t_event, 1u8)
            } else {
                (t_cens, 0u8)
            };
            treatment.push(a);
            conf.push(l);
            time.push(obs_t.max(1.0e-6));
            event.push(ev);
        }
        (treatment, conf, time, event)
    }

    /// Randomised-treatment data: `A` is a fair coin, independent of `L`.
    fn gen_randomised(
        n: usize,
        beta_true: f64,
        seed: u64,
    ) -> (Vec<u8>, Vec<f64>, Vec<f64>, Vec<u8>) {
        let mut rng = LcgRng::new(seed);
        let mut treatment = Vec::with_capacity(n);
        let mut conf = Vec::with_capacity(n);
        let mut time = Vec::with_capacity(n);
        let mut event = Vec::with_capacity(n);
        for _ in 0..n {
            let l = rng.next_normal();
            let a = u8::from(rng.next_bool());
            let hazard = (beta_true * f64::from(a)).exp();
            let t = rng.next_exponential(hazard).max(1.0e-6);
            treatment.push(a);
            conf.push(l);
            time.push(t);
            event.push(1u8);
        }
        (treatment, conf, time, event)
    }

    #[test]
    fn weighted_beats_naive_on_confounded_data() {
        // A confounder L drives both treatment (gamma) and the hazard (alpha),
        // so a naive Cox of survival on A is biased upward; IPTW with stabilized
        // weights breaks the A–L association and recovers the planted causal
        // log-HR. Moderate alpha keeps the (non-collapsible) marginal HR close to
        // the conditional one, and untruncated stabilized weights give the
        // standard consistent IPTW estimator.
        let n = 1500usize;
        let beta_true = 0.8_f64;
        let (treatment, conf, time, event) =
            gen_confounded(n, beta_true, 0.7, 1.2, 0.7, 0.0, 0xC0FFEE);
        let cfg = CausalCoxConfig {
            weight_truncation: None,
            ..Default::default()
        };
        let fit = fit_causal_cox(&treatment, &conf, n, 1, &time, &event, &cfg).expect("fit ok");
        let naive = fit_naive_cox(&treatment, n, &time, &event, &cfg).expect("naive ok");

        let err_iptw = (fit.causal_log_hr - beta_true).abs();
        let err_naive = (naive - beta_true).abs();
        // The naive estimate must be meaningfully biased and the IPTW estimate
        // must be closer to the planted causal log-HR.
        assert!(
            err_naive > 0.15,
            "naive should be biased: naive={naive}, truth={beta_true}, err={err_naive}"
        );
        assert!(
            err_iptw < err_naive,
            "IPTW (err={err_iptw}, est={}) should beat naive (err={err_naive}, est={naive}); truth={beta_true}",
            fit.causal_log_hr
        );
        // And the IPTW estimate should land near the truth.
        assert!(
            err_iptw < 0.25,
            "IPTW estimate {} should be near truth {beta_true}",
            fit.causal_log_hr
        );
        assert!(fit.causal_log_hr_se.is_finite() && fit.causal_log_hr_se > 0.0);
    }

    #[test]
    fn propensity_scores_in_open_unit_interval() {
        let n = 400usize;
        let (treatment, conf, time, event) = gen_confounded(n, 0.5, 0.8, 1.5, 0.6, 0.0, 7);
        let cfg = CausalCoxConfig::default();
        let fit = fit_causal_cox(&treatment, &conf, n, 1, &time, &event, &cfg).expect("ok");
        assert_eq!(fit.propensity_scores.len(), n);
        for &p in &fit.propensity_scores {
            assert!(p > 0.0 && p < 1.0, "propensity {p} must be in (0,1)");
        }
    }

    #[test]
    fn stabilized_weights_have_mean_near_one() {
        let n = 1000usize;
        let (treatment, conf, time, event) = gen_confounded(n, 0.5, 0.7, 1.2, 0.7, 0.0, 2024);
        let cfg = CausalCoxConfig {
            weight_truncation: None, // mean-1 property holds on untruncated weights
            ..Default::default()
        };
        let fit = fit_causal_cox(&treatment, &conf, n, 1, &time, &event, &cfg).expect("ok");
        let mean: f64 = fit.stabilized_weights.iter().sum::<f64>() / n as f64;
        assert!(
            (mean - 1.0).abs() < 0.1,
            "stabilized weight mean = {mean}, expected ≈ 1"
        );
        // Unstabilized weights are all strictly positive.
        assert!(fit.unstabilized_weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn truncation_caps_extreme_weights() {
        let n = 800usize;
        // Low noise + strong gamma => some near-deterministic propensities =>
        // extreme weights that truncation should cap.
        let (treatment, conf, time, event) = gen_confounded(n, 0.5, 0.5, 3.0, 0.25, 0.0, 31337);
        let cfg_trim = CausalCoxConfig {
            weight_truncation: Some((0.02, 0.98)),
            ..Default::default()
        };
        let cfg_none = CausalCoxConfig {
            weight_truncation: None,
            ..Default::default()
        };
        let fit_trim =
            fit_causal_cox(&treatment, &conf, n, 1, &time, &event, &cfg_trim).expect("ok");
        let fit_none =
            fit_causal_cox(&treatment, &conf, n, 1, &time, &event, &cfg_none).expect("ok");
        let max_trim = fit_trim
            .fit_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let max_none = fit_none
            .fit_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_trim <= max_none + 1.0e-9,
            "trimmed max weight {max_trim} should be <= untrimmed {max_none}"
        );
        assert!(
            max_trim < max_none,
            "truncation should strictly cap the largest weight (trim={max_trim}, none={max_none})"
        );
    }

    #[test]
    fn randomised_treatment_weighted_matches_unweighted() {
        let n = 1200usize;
        let beta_true = 0.6_f64;
        let (treatment, conf, time, event) = gen_randomised(n, beta_true, 555);
        let cfg = CausalCoxConfig::default();
        let fit = fit_causal_cox(&treatment, &conf, n, 1, &time, &event, &cfg).expect("ok");
        let naive = fit_naive_cox(&treatment, n, &time, &event, &cfg).expect("ok");
        // Under randomisation, confounder adjustment is unnecessary so the two
        // estimates should agree closely, and both sit near the truth.
        assert!(
            (fit.causal_log_hr - naive).abs() < 0.1,
            "weighted={} vs unweighted={} should agree under randomisation",
            fit.causal_log_hr,
            naive
        );
        assert!(
            (fit.causal_log_hr - beta_true).abs() < 0.2,
            "weighted estimate {} should be near truth {beta_true}",
            fit.causal_log_hr
        );
    }

    #[test]
    fn score_near_zero_at_optimum_and_se_finite() {
        let n = 700usize;
        let (treatment, conf, time, event) = gen_confounded(n, 0.7, 0.8, 1.5, 0.5, 0.0, 909);
        let cfg = CausalCoxConfig::default();
        let fit = fit_causal_cox(&treatment, &conf, n, 1, &time, &event, &cfg).expect("ok");
        assert!(fit.cox_converged, "weighted Cox should converge");
        assert!(
            fit.score_at_optimum.abs() < 1.0e-5,
            "weighted score at optimum = {} should be ≈ 0",
            fit.score_at_optimum
        );
        assert!(
            fit.causal_log_hr_se.is_finite() && fit.causal_log_hr_se > 0.0,
            "robust SE should be finite and positive, got {}",
            fit.causal_log_hr_se
        );
        assert!(fit.model_se.is_finite() && fit.model_se > 0.0);
    }

    #[test]
    fn adjusted_survival_curves_are_monotone_and_in_unit_range() {
        let n = 600usize;
        let (treatment, conf, time, event) = gen_confounded(n, 0.6, 0.7, 1.2, 0.6, 0.3, 4242);
        let cfg = CausalCoxConfig::default();
        let fit = fit_causal_cox(&treatment, &conf, n, 1, &time, &event, &cfg).expect("ok");
        for curve in [
            &fit.adjusted_survival_control,
            &fit.adjusted_survival_treated,
        ] {
            let mut prev = 1.0_f64;
            for &s in &curve.survival {
                assert!((0.0..=1.0).contains(&s), "survival {s} out of [0,1]");
                assert!(s <= prev + 1.0e-12, "survival not non-increasing");
                prev = s;
            }
            assert!((curve.survival_at(0.0) - 1.0).abs() < 1.0e-12);
        }
        // With a protective-looking large positive beta the treated arm should
        // generally not sit above control everywhere; we only assert validity of
        // the survival_at lookup here.
        let big_t = time.iter().cloned().fold(0.0_f64, f64::max) + 1.0;
        let s_ctrl = fit.adjusted_survival_control.survival_at(big_t);
        let s_trt = fit.adjusted_survival_treated.survival_at(big_t);
        assert!((0.0..=1.0).contains(&s_ctrl));
        assert!((0.0..=1.0).contains(&s_trt));
    }

    #[test]
    fn hazard_ratio_matches_exp_of_log_hr() {
        let n = 300usize;
        let (treatment, conf, time, event) = gen_confounded(n, 0.5, 0.6, 1.0, 0.7, 0.0, 11);
        let cfg = CausalCoxConfig::default();
        let fit = fit_causal_cox(&treatment, &conf, n, 1, &time, &event, &cfg).expect("ok");
        assert!(
            (fit.causal_hazard_ratio - fit.causal_log_hr.exp()).abs() < 1.0e-12,
            "HR should equal exp(log-HR)"
        );
        let z = fit.z_score();
        assert!(z.is_finite());
    }

    #[test]
    fn rejects_non_binary_treatment() {
        let treatment = vec![0u8, 1u8, 2u8, 1u8];
        let conf = vec![0.1_f64, -0.2, 0.3, 0.0];
        let time = vec![1.0_f64, 2.0, 1.5, 0.8];
        let event = vec![1u8, 1u8, 0u8, 1u8];
        let cfg = CausalCoxConfig::default();
        let r = fit_causal_cox(&treatment, &conf, 4, 1, &time, &event, &cfg);
        assert!(matches!(r, Err(SurvivalError::InvalidParameter(_))));
    }

    #[test]
    fn rejects_length_mismatch_and_empty() {
        let cfg = CausalCoxConfig::default();
        // Empty.
        let r0 = fit_causal_cox(&[], &[], 0, 1, &[], &[], &cfg);
        assert!(matches!(r0, Err(SurvivalError::EmptyDataset)));
        // Time length mismatch.
        let treatment = vec![0u8, 1u8, 1u8];
        let conf = vec![0.0_f64, 1.0, -1.0];
        let time = vec![1.0_f64, 2.0]; // wrong length
        let event = vec![1u8, 1u8, 1u8];
        let r1 = fit_causal_cox(&treatment, &conf, 3, 1, &time, &event, &cfg);
        assert!(matches!(r1, Err(SurvivalError::DimensionMismatch { .. })));
        // Confounder shape mismatch.
        let time_ok = vec![1.0_f64, 2.0, 3.0];
        let conf_bad = vec![0.0_f64, 1.0]; // should be length 3
        let r2 = fit_causal_cox(&treatment, &conf_bad, 3, 1, &time_ok, &event, &cfg);
        assert!(matches!(r2, Err(SurvivalError::ShapeMismatch { .. })));
    }

    #[test]
    fn rejects_no_events_and_single_arm_and_bad_time() {
        let cfg = CausalCoxConfig::default();
        // No events.
        let treatment = vec![0u8, 1u8, 1u8, 0u8];
        let conf = vec![0.1_f64, 0.2, -0.1, 0.0];
        let time = vec![1.0_f64, 2.0, 3.0, 1.5];
        let event_none = vec![0u8, 0u8, 0u8, 0u8];
        let r0 = fit_causal_cox(&treatment, &conf, 4, 1, &time, &event_none, &cfg);
        assert!(matches!(r0, Err(SurvivalError::NoEvents)));
        // Single-arm treatment (all treated).
        let all_treated = vec![1u8, 1u8, 1u8, 1u8];
        let event_ok = vec![1u8, 1u8, 0u8, 1u8];
        let r1 = fit_causal_cox(&all_treated, &conf, 4, 1, &time, &event_ok, &cfg);
        assert!(matches!(r1, Err(SurvivalError::InvalidParameter(_))));
        // Negative time.
        let bad_time = vec![1.0_f64, -2.0, 3.0, 1.5];
        let r2 = fit_causal_cox(&treatment, &conf, 4, 1, &bad_time, &event_ok, &cfg);
        assert!(matches!(r2, Err(SurvivalError::NegativeTime(_))));
        // Zero time (must be strictly positive).
        let zero_time = vec![1.0_f64, 0.0, 3.0, 1.5];
        let r3 = fit_causal_cox(&treatment, &conf, 4, 1, &zero_time, &event_ok, &cfg);
        assert!(matches!(r3, Err(SurvivalError::InvalidParameter(_))));
    }

    #[test]
    fn rejects_zero_confounders_and_bad_truncation() {
        let treatment = vec![0u8, 1u8, 1u8, 0u8];
        let conf = vec![0.1_f64, 0.2, -0.1, 0.0];
        let time = vec![1.0_f64, 2.0, 3.0, 1.5];
        let event = vec![1u8, 1u8, 0u8, 1u8];
        // Zero confounders.
        let cfg = CausalCoxConfig::default();
        let r0 = fit_causal_cox(&treatment, &[], 4, 0, &time, &event, &cfg);
        assert!(matches!(r0, Err(SurvivalError::InvalidParameter(_))));
        // Bad truncation pair (lo > hi).
        let cfg_bad = CausalCoxConfig {
            weight_truncation: Some((0.9, 0.1)),
            ..Default::default()
        };
        let r1 = fit_causal_cox(&treatment, &conf, 4, 1, &time, &event, &cfg_bad);
        assert!(matches!(r1, Err(SurvivalError::InvalidParameter(_))));
    }

    #[test]
    fn unstabilized_option_runs_and_differs_in_weights() {
        let n = 500usize;
        let (treatment, conf, time, event) = gen_confounded(n, 0.6, 0.8, 1.5, 0.5, 0.0, 246);
        let cfg_unstab = CausalCoxConfig {
            stabilized: false,
            weight_truncation: None,
            ..Default::default()
        };
        let fit = fit_causal_cox(&treatment, &conf, n, 1, &time, &event, &cfg_unstab).expect("ok");
        // With unstabilized weights chosen, fit_weights equals the (untruncated)
        // unstabilized vector, whose mean exceeds 1 (≈ 2 for balanced arms).
        assert_eq!(fit.fit_weights, fit.unstabilized_weights);
        let mean_unstab: f64 = fit.unstabilized_weights.iter().sum::<f64>() / n as f64;
        assert!(
            mean_unstab > 1.2,
            "unstabilized mean weight {mean_unstab} should exceed stabilized ≈ 1"
        );
        assert!(fit.causal_log_hr.is_finite());
    }
}

//! Cox proportional hazards regression (Cox 1972) by partial-likelihood
//! maximisation.
//!
//! The Cox model is the workhorse of survival analysis. For a subject `i` with
//! covariate vector `x_i` the hazard at time `t` is
//!
//! ```text
//! λ_i(t) = λ_0(t) · exp(x_iᵀ β),
//! ```
//!
//! where `λ_0(t)` is an unspecified baseline hazard. Because the baseline
//! cancels out of the conditional probability that subject `i` is the one who
//! fails at an observed event time, `β` is estimated by maximising the
//! **partial likelihood** rather than the full likelihood — no parametric form
//! for `λ_0` is required (a *semiparametric* model).
//!
//! # Partial likelihood
//! Let the distinct event times be `τ_1 < τ_2 < …`. At event time `τ_k` let
//! `D_k` be the set of subjects that fail (size `d_k`) and `R_k = {i : t_i ≥ τ_k}`
//! the *risk set* (all subjects still under observation just before `τ_k`).
//! Writing `θ_i = exp(x_iᵀ β)`:
//!
//! - **Breslow** (default) approximation for ties:
//!   ```text
//!   L(β) = ∏_k  (∏_{i∈D_k} θ_i) / (Σ_{j∈R_k} θ_j)^{d_k}.
//!   ```
//! - **Efron** approximation (more accurate with ties, matches R `survival`):
//!   ```text
//!   L(β) = ∏_k (∏_{i∈D_k} θ_i)
//!          / ∏_{r=0}^{d_k−1} ( Σ_{j∈R_k} θ_j − (r/d_k) Σ_{i∈D_k} θ_i ).
//!   ```
//!
//! The log partial likelihood is concave, so Newton–Raphson on the analytic
//! score `U(β) = ∂ℓ/∂β` and observed information `I(β) = −∂²ℓ/∂β∂βᵀ`
//! converges quadratically:
//!
//! ```text
//! β ← β + I(β)⁻¹ U(β).
//! ```
//!
//! # What this module provides
//! - [`cox_ph_fit`] — Newton–Raphson MLE with Breslow / Efron tie handling.
//! - Coefficient covariance `I(β̂)⁻¹`, standard errors, Wald z-statistics and
//!   two-sided p-values, hazard ratios `exp(β̂)`.
//! - Likelihood-ratio test against the null model (`β = 0`).
//! - [`CoxFit::baseline_cumulative_hazard`] — the Breslow estimator
//!   `Ĥ_0(t)`, and [`CoxFit::survival_function`] for an arbitrary covariate
//!   pattern.
//! - [`concordance_index`] — Harrell's C, the survival-analysis analogue of the
//!   area under the ROC curve.
//!
//! # References
//! - Cox, D. R. (1972). *Regression Models and Life-Tables*. JRSS-B 34(2): 187-220.
//! - Efron, B. (1977). *The Efficiency of Cox's Likelihood Function for
//!   Censored Data*. JASA 72(359): 557-565.
//! - Breslow, N. (1974). *Covariance Analysis of Censored Survival Data*.
//!   Biometrics 30(1): 89-99.
//! - Harrell, F. E. et al. (1982). *Evaluating the Yield of Medical Tests*.
//!   JAMA 247(18): 2543-2546.
//! - Therneau & Grambsch (2000). *Modeling Survival Data*. Springer.

use crate::distributions::chi_squared::ChiSquared;
use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};
use crate::regression::linear::matrix_inverse_lu;
use std::cmp::Ordering;

/// Tie-handling approximation for the partial likelihood.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TieMethod {
    /// Breslow (1974) approximation — fastest, the classic default.
    Breslow,
    /// Efron (1977) approximation — more accurate under ties; matches R's
    /// `survival::coxph` default. This is the crate default.
    #[default]
    Efron,
}

/// Configuration for [`cox_ph_fit`].
#[derive(Debug, Clone)]
pub struct CoxConfig {
    /// Maximum Newton–Raphson iterations (default 100).
    pub max_iter: usize,
    /// Convergence tolerance on the relative change of the log partial
    /// likelihood (default `1e-9`).
    pub tol: f64,
    /// Tie-handling approximation (default [`TieMethod::Efron`]).
    pub tie_method: TieMethod,
    /// Step-halving guard: maximum back-tracking halvings per Newton step when
    /// the proposed step fails to increase the log-likelihood (default 20).
    pub max_halvings: usize,
}

impl Default for CoxConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-9,
            tie_method: TieMethod::Efron,
            max_halvings: 20,
        }
    }
}

impl CoxConfig {
    /// Construct and validate a configuration.
    ///
    /// # Errors
    /// [`StatsError::InvalidParameter`] if `max_iter == 0` or `tol <= 0`.
    pub fn new(max_iter: usize, tol: f64, tie_method: TieMethod) -> StatsResult<Self> {
        if max_iter == 0 {
            return Err(StatsError::InvalidParameter {
                name: "max_iter".to_owned(),
                reason: "must be ≥ 1".to_owned(),
            });
        }
        if !(tol > 0.0 && tol.is_finite()) {
            return Err(StatsError::InvalidParameter {
                name: "tol".to_owned(),
                reason: "must be a positive finite number".to_owned(),
            });
        }
        Ok(Self {
            max_iter,
            tol,
            tie_method,
            max_halvings: 20,
        })
    }
}

/// A fitted Cox proportional hazards model.
#[derive(Debug, Clone)]
pub struct CoxFit {
    /// Estimated coefficients β̂ (one per covariate; **no intercept** — the
    /// baseline hazard absorbs it).
    pub coef: Vec<f64>,
    /// Standard errors `sqrt(diag(I(β̂)⁻¹))`.
    pub std_err: Vec<f64>,
    /// Wald z-statistics `β̂_j / se_j`.
    pub z: Vec<f64>,
    /// Two-sided Wald p-values `2(1 − Φ(|z_j|))`.
    pub p_value: Vec<f64>,
    /// Hazard ratios `exp(β̂_j)`.
    pub hazard_ratio: Vec<f64>,
    /// Flattened row-major `p × p` coefficient covariance matrix `I(β̂)⁻¹`.
    pub covariance: Vec<f64>,
    /// Maximised log partial likelihood `ℓ(β̂)`.
    pub log_likelihood: f64,
    /// Null log partial likelihood `ℓ(0)`.
    pub null_log_likelihood: f64,
    /// Number of Newton–Raphson iterations performed.
    pub n_iter: usize,
    /// Number of covariates `p`.
    pub n_features: usize,
    /// Number of subjects `n`.
    pub n_samples: usize,
    /// Number of observed events (`status == true`).
    pub n_events: usize,
    /// Tie-handling approximation used for the fit.
    pub tie_method: TieMethod,
    // Internals retained for baseline-hazard / survival queries.
    /// Sorted distinct event times, ascending.
    event_times: Vec<f64>,
    /// Breslow baseline cumulative hazard `Ĥ_0` evaluated at `event_times`.
    baseline_cum_hazard: Vec<f64>,
}

impl CoxFit {
    /// Likelihood-ratio statistic against the null model:
    /// `LR = 2(ℓ(β̂) − ℓ(0)) ~ χ²(p)` under `H₀: β = 0`.
    #[must_use]
    pub fn lr_statistic(&self) -> f64 {
        2.0 * (self.log_likelihood - self.null_log_likelihood)
    }

    /// p-value of the global likelihood-ratio test (`χ²` with `p` degrees of
    /// freedom).
    ///
    /// # Errors
    /// [`StatsError::DegreesOfFreedomZero`] if the model has no covariates.
    pub fn lr_p_value(&self) -> StatsResult<f64> {
        if self.n_features == 0 {
            return Err(StatsError::DegreesOfFreedomZero);
        }
        let chi = ChiSquared::new(self.n_features as f64)?;
        Ok((1.0 - chi.cdf(self.lr_statistic())?).clamp(0.0, 1.0))
    }

    /// Linear predictor (risk score) `x_iᵀ β̂` for a single covariate vector.
    ///
    /// # Errors
    /// [`StatsError::DimensionMismatch`] if `x.len() != n_features`.
    pub fn linear_predictor(&self, x: &[f64]) -> StatsResult<f64> {
        if x.len() != self.n_features {
            return Err(StatsError::DimensionMismatch {
                a: x.len(),
                b: self.n_features,
            });
        }
        Ok(x.iter().zip(self.coef.iter()).map(|(a, b)| a * b).sum())
    }

    /// Breslow baseline cumulative hazard `Ĥ_0(t)`, evaluated as a right-
    /// continuous step function: returns the cumulative hazard accumulated up to
    /// and including all event times `≤ t`.
    #[must_use]
    pub fn baseline_cumulative_hazard(&self, t: f64) -> f64 {
        let mut h = 0.0;
        for (et, dh) in self.event_times.iter().zip(self.baseline_cum_hazard.iter()) {
            if *et <= t {
                h = *dh;
            } else {
                break;
            }
        }
        h
    }

    /// The distinct event times and their accumulated baseline hazard, as
    /// `(time, Ĥ_0(time))` pairs (ascending in time).
    #[must_use]
    pub fn baseline_hazard_table(&self) -> Vec<(f64, f64)> {
        self.event_times
            .iter()
            .zip(self.baseline_cum_hazard.iter())
            .map(|(&t, &h)| (t, h))
            .collect()
    }

    /// Estimated survival probability `Ŝ(t | x) = exp(−Ĥ_0(t) · exp(xᵀβ̂))`
    /// for a covariate pattern `x`.
    ///
    /// # Errors
    /// [`StatsError::DimensionMismatch`] if `x.len() != n_features`.
    pub fn survival_function(&self, t: f64, x: &[f64]) -> StatsResult<f64> {
        let lp = self.linear_predictor(x)?;
        let h = self.baseline_cumulative_hazard(t) * lp.exp();
        Ok((-h).exp())
    }
}

// ---------------------------------------------------------------------------
// Internal: ordered observation bookkeeping
// ---------------------------------------------------------------------------

/// Observation indices grouped so that they are sorted by ascending time, and
/// within identical times the *events* precede the *censorings* (so that an
/// event and a censoring at the same time put the censored subject correctly
/// inside the risk set of that event — the standard convention).
struct Ordering0 {
    /// Permutation of original indices, sorted by (time asc, event-before-censor).
    order: Vec<usize>,
}

fn build_ordering(time: &[f64], status: &[bool]) -> Ordering0 {
    let n = time.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| match time[a].partial_cmp(&time[b]) {
        Some(Ordering::Equal) | None => {
            // Events (true) before censorings (false) at equal times.
            status[b].cmp(&status[a])
        }
        Some(ord) => ord,
    });
    Ordering0 { order }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_inputs(
    x: &[f64],
    time: &[f64],
    status: &[bool],
    n: usize,
    p: usize,
) -> StatsResult<()> {
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if p == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_features".to_owned(),
            reason: "Cox model needs at least one covariate".to_owned(),
        });
    }
    if x.len() != n * p {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n * p],
            got: vec![x.len()],
        });
    }
    if time.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: time.len(),
            b: n,
        });
    }
    if status.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: status.len(),
            b: n,
        });
    }
    for (i, &t) in time.iter().enumerate() {
        if !t.is_finite() || t < 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "time".to_owned(),
                reason: format!("survival time at index {i} must be finite and ≥ 0, got {t}"),
            });
        }
    }
    for (i, &xv) in x.iter().enumerate() {
        if !xv.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    if !status.iter().any(|&s| s) {
        return Err(StatsError::InvalidParameter {
            name: "status".to_owned(),
            reason: "at least one observed event (status == true) is required".to_owned(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Partial-likelihood, score and information
// ---------------------------------------------------------------------------

/// Accumulated value, gradient (score) and negative Hessian (observed
/// information) of the log partial likelihood at the current `beta`.
struct PlComponents {
    loglik: f64,
    score: Vec<f64>,
    /// Observed information `I(β) = −∂²ℓ/∂β∂βᵀ` (flattened `p × p`).
    info: Vec<f64>,
}

/// Compute the log partial likelihood, its score and observed information.
///
/// Risk sets are accumulated by sweeping the subjects in *descending* time
/// order, so each subject is added to the running risk-set totals exactly once
/// as the threshold time falls below it. Within each block of tied times, the
/// failing subjects' contributions are handled per `tie_method`.
fn pl_components(
    x: &[f64],
    time: &[f64],
    status: &[bool],
    n: usize,
    p: usize,
    beta: &[f64],
    ord: &Ordering0,
    tie_method: TieMethod,
) -> PlComponents {
    // Running risk-set accumulators (subjects with time ≥ current threshold):
    //   risk_sum   = Σ θ_j
    //   risk_xsum  = Σ θ_j x_j        (length p)
    //   risk_xxsum = Σ θ_j x_j x_jᵀ   (length p*p)
    let mut risk_sum = 0.0_f64;
    let mut risk_xsum = vec![0.0_f64; p];
    let mut risk_xxsum = vec![0.0_f64; p * p];

    let mut loglik = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];

    // θ_i and the linear predictor for each subject.
    let theta = |i: usize| -> f64 {
        let mut eta = 0.0;
        for k in 0..p {
            eta += beta[k] * x[i * p + k];
        }
        eta.exp()
    };

    // Sweep blocks of equal time from largest to smallest. `order` is ascending,
    // so iterate it in reverse and gather tied blocks.
    let mut idx = n; // one-past-end into ord.order
    while idx > 0 {
        // Current block has time == t_block; gather [lo, idx) with that time.
        let hi = idx;
        let t_block = time[ord.order[hi - 1]];
        let mut lo = hi;
        while lo > 0 && time[ord.order[lo - 1]] == t_block {
            lo -= 1;
        }

        // Add every subject in this block to the running risk set (they all have
        // time == t_block ≥ threshold) BEFORE scoring the block's events, so the
        // tied subjects belong to their own risk set.
        for &i in &ord.order[lo..hi] {
            let th = theta(i);
            risk_sum += th;
            for a in 0..p {
                let xa = x[i * p + a];
                risk_xsum[a] += th * xa;
                for b in 0..p {
                    risk_xxsum[a * p + b] += th * xa * x[i * p + b];
                }
            }
        }

        // Collect the failing subjects (events) in this tied block.
        let deaths: Vec<usize> = ord.order[lo..hi]
            .iter()
            .copied()
            .filter(|&i| status[i])
            .collect();
        let d = deaths.len();
        if d == 0 {
            idx = lo;
            continue;
        }

        // Sum over the deaths: Σ_{i∈D} x_i and Σ_{i∈D} θ_i (and θ_i x_i x_iᵀ).
        let mut death_xsum = vec![0.0_f64; p];
        let mut death_theta = 0.0_f64;
        let mut death_txsum = vec![0.0_f64; p];
        let mut death_txxsum = vec![0.0_f64; p * p];
        for &i in &deaths {
            let th = theta(i);
            death_theta += th;
            for a in 0..p {
                let xa = x[i * p + a];
                death_xsum[a] += xa;
                death_txsum[a] += th * xa;
                for b in 0..p {
                    death_txxsum[a * p + b] += th * xa * x[i * p + b];
                }
            }
            // Numerator log term: Σ_{i∈D} x_iᵀβ.
            for k in 0..p {
                loglik += beta[k] * x[i * p + k];
            }
            // Numerator score term: + Σ_{i∈D} x_i.
            for a in 0..p {
                score[a] += x[i * p + a];
            }
        }

        match tie_method {
            TieMethod::Breslow => {
                // Denominator uses the full risk set raised to the d-th power.
                loglik -= d as f64 * risk_sum.ln();
                // E[x] under the risk-set weighting.
                for a in 0..p {
                    let ea = risk_xsum[a] / risk_sum;
                    score[a] -= d as f64 * ea;
                    for b in 0..p {
                        let eb = risk_xsum[b] / risk_sum;
                        let cov_ab = risk_xxsum[a * p + b] / risk_sum - ea * eb;
                        info[a * p + b] += d as f64 * cov_ab;
                    }
                }
            }
            TieMethod::Efron => {
                // Efron: subtract a fraction r/d of the death totals at step r.
                for r in 0..d {
                    let frac = r as f64 / d as f64;
                    let denom = risk_sum - frac * death_theta;
                    loglik -= denom.ln();
                    for a in 0..p {
                        let num_a = risk_xsum[a] - frac * death_txsum[a];
                        let ea = num_a / denom;
                        score[a] -= ea;
                        for b in 0..p {
                            let num_b = risk_xsum[b] - frac * death_txsum[b];
                            let eb = num_b / denom;
                            let xx = risk_xxsum[a * p + b] - frac * death_txxsum[a * p + b];
                            let cov_ab = xx / denom - ea * eb;
                            info[a * p + b] += cov_ab;
                        }
                    }
                }
            }
        }

        idx = lo;
    }

    PlComponents {
        loglik,
        score,
        info,
    }
}

// ---------------------------------------------------------------------------
// Breslow baseline cumulative hazard
// ---------------------------------------------------------------------------

/// Compute the Breslow baseline cumulative hazard at every distinct event time.
///
/// `Ĥ_0(τ_k) = Σ_{l ≤ k} d_l / (Σ_{j∈R_l} exp(x_jᵀ β̂))`, returned together
/// with the ascending list of distinct event times.
fn breslow_baseline(
    x: &[f64],
    time: &[f64],
    status: &[bool],
    n: usize,
    p: usize,
    beta: &[f64],
    ord: &Ordering0,
) -> (Vec<f64>, Vec<f64>) {
    let theta = |i: usize| -> f64 {
        let mut eta = 0.0;
        for k in 0..p {
            eta += beta[k] * x[i * p + k];
        }
        eta.exp()
    };

    // Sweep ascending event times but accumulate risk set descending; easiest is
    // to first compute, for each event time, the risk-set θ-sum.
    let mut event_times: Vec<f64> = Vec::new();
    let mut increments: Vec<f64> = Vec::new();

    let mut risk_sum = 0.0_f64;
    let mut idx = n;
    while idx > 0 {
        let hi = idx;
        let t_block = time[ord.order[hi - 1]];
        let mut lo = hi;
        while lo > 0 && time[ord.order[lo - 1]] == t_block {
            lo -= 1;
        }
        for &i in &ord.order[lo..hi] {
            risk_sum += theta(i);
        }
        let d = ord.order[lo..hi].iter().filter(|&&i| status[i]).count();
        if d > 0 && risk_sum > 0.0 {
            event_times.push(t_block);
            increments.push(d as f64 / risk_sum);
        }
        idx = lo;
    }

    // We discovered times in descending order; reverse to ascending and form the
    // running cumulative sum.
    event_times.reverse();
    increments.reverse();
    let mut cum = 0.0_f64;
    let mut cumulative = Vec::with_capacity(increments.len());
    for inc in &increments {
        cum += inc;
        cumulative.push(cum);
    }
    (event_times, cumulative)
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

/// Fit a Cox proportional hazards model by Newton–Raphson on the partial
/// likelihood.
///
/// # Arguments
/// - `x` — row-major `n × p` covariate matrix (**no intercept column**).
/// - `time` — survival/censoring times (`≥ 0`), length `n`.
/// - `status` — event indicator, length `n` (`true` = event observed,
///   `false` = right-censored).
/// - `cfg` — solver configuration.
///
/// # Errors
/// - [`StatsError::EmptyInput`] / [`StatsError::ShapeMismatch`] /
///   [`StatsError::DimensionMismatch`] on inconsistent inputs.
/// - [`StatsError::InvalidParameter`] if `p == 0`, a time is negative/non-finite,
///   or there are no observed events.
/// - [`StatsError::NonFiniteValue`] if any covariate is non-finite.
/// - [`StatsError::SingularMatrix`] if the information matrix is singular
///   (e.g. perfectly collinear covariates or complete separation).
/// - [`StatsError::NotConverged`] if Newton–Raphson fails to converge within
///   `cfg.max_iter` iterations.
pub fn cox_ph_fit(
    x: &[f64],
    time: &[f64],
    status: &[bool],
    n: usize,
    p: usize,
    cfg: &CoxConfig,
) -> StatsResult<CoxFit> {
    validate_inputs(x, time, status, n, p)?;
    let ord = build_ordering(time, status);
    let n_events = status.iter().filter(|&&s| s).count();

    // Null log-likelihood at β = 0.
    let zero = vec![0.0_f64; p];
    let null_ll = pl_components(x, time, status, n, p, &zero, &ord, cfg.tie_method).loglik;

    let mut beta = vec![0.0_f64; p];
    let mut comp = pl_components(x, time, status, n, p, &beta, &ord, cfg.tie_method);
    let mut prev_ll = comp.loglik;
    let mut n_iter = 0;
    let mut converged = false;

    for it in 1..=cfg.max_iter {
        n_iter = it;
        // Solve I(β) Δ = U(β); step is β ← β + Δ.
        let inv = matrix_inverse_lu(&comp.info, p).map_err(|_| {
            StatsError::SingularMatrix(
                "Cox information matrix is singular (collinear covariates or separation)"
                    .to_owned(),
            )
        })?;
        let mut step = vec![0.0_f64; p];
        for a in 0..p {
            let mut acc = 0.0;
            for b in 0..p {
                acc += inv[a * p + b] * comp.score[b];
            }
            step[a] = acc;
        }

        // Step-halving line search to guarantee monotone ascent of ℓ.
        let mut scale = 1.0_f64;
        let mut accepted = false;
        let mut trial = beta.clone();
        let mut trial_comp = comp.loglik; // placeholder, replaced below
        let mut new_comp_opt: Option<PlComponents> = None;
        for _ in 0..=cfg.max_halvings {
            for a in 0..p {
                trial[a] = beta[a] + scale * step[a];
            }
            let c = pl_components(x, time, status, n, p, &trial, &ord, cfg.tie_method);
            if c.loglik.is_finite() && c.loglik >= prev_ll - 1e-12 {
                trial_comp = c.loglik;
                new_comp_opt = Some(c);
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        let new_comp = match new_comp_opt {
            Some(c) => c,
            None => {
                // No ascent direction found; treat as converged at current point.
                converged = true;
                break;
            }
        };

        beta = trial;
        comp = new_comp;
        let _ = accepted;

        let rel = if prev_ll.abs() > 1e-12 {
            (trial_comp - prev_ll).abs() / prev_ll.abs()
        } else {
            (trial_comp - prev_ll).abs()
        };
        prev_ll = trial_comp;
        if rel < cfg.tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(StatsError::NotConverged {
            iter: n_iter,
            residual: comp.score.iter().map(|v| v.abs()).fold(0.0, f64::max),
        });
    }

    // Covariance = I(β̂)⁻¹.
    let covariance = matrix_inverse_lu(&comp.info, p).map_err(|_| {
        StatsError::SingularMatrix("Cox information matrix is singular at the optimum".to_owned())
    })?;

    let normal = Normal::standard();
    let mut std_err = vec![0.0_f64; p];
    let mut z = vec![0.0_f64; p];
    let mut p_value = vec![0.0_f64; p];
    let mut hazard_ratio = vec![0.0_f64; p];
    for j in 0..p {
        let var = covariance[j * p + j];
        let se = if var > 0.0 { var.sqrt() } else { f64::NAN };
        std_err[j] = se;
        z[j] = beta[j] / se;
        let zabs = z[j].abs();
        p_value[j] = (2.0 * (1.0 - normal.cdf(zabs))).clamp(0.0, 1.0);
        hazard_ratio[j] = beta[j].exp();
    }

    let (event_times, baseline_cum_hazard) = breslow_baseline(x, time, status, n, p, &beta, &ord);

    Ok(CoxFit {
        coef: beta,
        std_err,
        z,
        p_value,
        hazard_ratio,
        covariance,
        log_likelihood: comp.loglik,
        null_log_likelihood: null_ll,
        n_iter,
        n_features: p,
        n_samples: n,
        n_events,
        tie_method: cfg.tie_method,
        event_times,
        baseline_cum_hazard,
    })
}

// ---------------------------------------------------------------------------
// Harrell's concordance index
// ---------------------------------------------------------------------------

/// Harrell's concordance index (C-index) for survival predictions.
///
/// For every *comparable* pair of subjects — where the one with the shorter
/// observed time experienced an event (so the ordering of their true survival
/// is known) — the pair is **concordant** if the subject who failed earlier has
/// the higher risk score. Ties in risk score contribute ½.
///
/// `risk` is a higher-is-riskier score for each subject (e.g.
/// [`CoxFit::linear_predictor`] applied to each covariate row, since
/// `exp(·)` is monotone). The returned value lies in `[0, 1]`; `0.5` is random,
/// `1.0` is perfect discrimination.
///
/// # Errors
/// - [`StatsError::DimensionMismatch`] if the three slices differ in length.
/// - [`StatsError::EmptyInput`] if there are no subjects.
/// - [`StatsError::InvalidParameter`] if no comparable pair exists (e.g. all
///   observations censored).
pub fn concordance_index(time: &[f64], status: &[bool], risk: &[f64]) -> StatsResult<f64> {
    let n = time.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if status.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: status.len(),
            b: n,
        });
    }
    if risk.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: risk.len(),
            b: n,
        });
    }
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            // Determine whether the pair is comparable and who is the earlier
            // (definitely-shorter-survival) subject.
            let (early, late, ok) = match time[i].partial_cmp(&time[j]) {
                Some(Ordering::Less) => (i, j, status[i]),
                Some(Ordering::Greater) => (j, i, status[j]),
                _ => {
                    // Equal times: comparable only if exactly one is an event,
                    // and such pairs count as a half-concordance (tie in time).
                    if status[i] != status[j] {
                        comparable += 1.0;
                        concordant += 0.5;
                    }
                    continue;
                }
            };
            if !ok {
                // The earlier subject is censored ⇒ ordering unknown ⇒ skip.
                continue;
            }
            comparable += 1.0;
            // Earlier failure should carry the *higher* risk score.
            match risk[early].partial_cmp(&risk[late]) {
                Some(Ordering::Greater) => concordant += 1.0,
                Some(Ordering::Equal) | None => concordant += 0.5,
                Some(Ordering::Less) => {}
            }
        }
    }
    if comparable == 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "status".to_owned(),
            reason: "no comparable (uncensored-earlier) pairs for the concordance index".to_owned(),
        });
    }
    Ok(concordant / comparable)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Numeric central-difference check that the analytic score equals the
    /// gradient of the log partial likelihood.
    fn numeric_score(
        x: &[f64],
        time: &[f64],
        status: &[bool],
        n: usize,
        p: usize,
        beta: &[f64],
        tie: TieMethod,
    ) -> Vec<f64> {
        let ord = build_ordering(time, status);
        let h = 1e-6;
        let mut g = vec![0.0; p];
        for k in 0..p {
            let mut bp = beta.to_vec();
            let mut bm = beta.to_vec();
            bp[k] += h;
            bm[k] -= h;
            let lp = pl_components(x, time, status, n, p, &bp, &ord, tie).loglik;
            let lm = pl_components(x, time, status, n, p, &bm, &ord, tie).loglik;
            g[k] = (lp - lm) / (2.0 * h);
        }
        g
    }

    #[test]
    fn analytic_score_matches_finite_difference_breslow() {
        // n = 6, p = 2, mixed events/censoring, no ties.
        let x = vec![
            0.5, 1.0, // s0
            -0.3, 0.2, // s1
            1.1, -0.5, // s2
            0.0, 0.8, // s3
            -1.0, -1.0, // s4
            0.7, 0.3, // s5
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![true, false, true, true, false, true];
        let beta = vec![0.4, -0.2];
        let ord = build_ordering(&time, &status);
        let comp = pl_components(&x, &time, &status, 6, 2, &beta, &ord, TieMethod::Breslow);
        let num = numeric_score(&x, &time, &status, 6, 2, &beta, TieMethod::Breslow);
        for (k, &ng) in num.iter().enumerate() {
            assert!(
                (comp.score[k] - ng).abs() < 1e-4,
                "Breslow score[{k}]: analytic {} vs numeric {ng}",
                comp.score[k],
            );
        }
    }

    #[test]
    fn analytic_score_matches_finite_difference_efron_with_ties() {
        // Deliberate ties at time 2.0 (two events) and 4.0 (event + censor).
        let x = vec![
            0.5, -0.2, 0.9, 0.1, -0.4, 0.6, 0.3, -0.8, 1.2, 0.4, -0.1, 0.2,
        ];
        let time = vec![1.0, 2.0, 2.0, 4.0, 4.0, 5.0];
        let status = vec![true, true, true, true, false, true];
        let beta = vec![0.3, 0.5];
        let ord = build_ordering(&time, &status);
        let comp = pl_components(&x, &time, &status, 6, 2, &beta, &ord, TieMethod::Efron);
        let num = numeric_score(&x, &time, &status, 6, 2, &beta, TieMethod::Efron);
        for (k, &ng) in num.iter().enumerate() {
            assert!(
                (comp.score[k] - ng).abs() < 1e-4,
                "Efron score[{k}]: analytic {} vs numeric {ng}",
                comp.score[k],
            );
        }
    }

    #[test]
    fn information_is_symmetric_positive_definite() {
        let x = vec![0.5, 1.0, -0.3, 0.2, 1.1, -0.5, 0.0, 0.8, -1.0, -1.0];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![true, true, false, true, true];
        let beta = vec![0.1, -0.1];
        let ord = build_ordering(&time, &status);
        let comp = pl_components(&x, &time, &status, 5, 2, &beta, &ord, TieMethod::Efron);
        // Symmetry.
        assert!((comp.info[1] - comp.info[2]).abs() < 1e-12);
        // Positive definiteness: both leading minors positive (concave PL).
        let det = comp.info[0] * comp.info[3] - comp.info[1] * comp.info[2];
        assert!(
            comp.info[0] > 0.0 && det > 0.0,
            "information not SPD: {:?}",
            comp.info
        );
    }

    /// Closed-form check: a single covariate that is 1 for events and 0 for
    /// censored subjects with no ties pushes the coefficient strongly positive.
    #[test]
    fn coefficient_sign_tracks_risk() {
        // Subjects failing early have x=1; surviving long have x=0.
        let x = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let time = vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];
        let status = vec![true, true, true, true, true, true];
        let fit = cox_ph_fit(&x, &time, &status, 6, 1, &CoxConfig::default()).expect("fit");
        assert!(
            fit.coef[0] > 0.0,
            "early-failing high-covariate group should give β > 0, got {}",
            fit.coef[0]
        );
        assert!(fit.hazard_ratio[0] > 1.0, "hazard ratio should exceed 1");
        // Log-likelihood must exceed the null.
        assert!(fit.log_likelihood >= fit.null_log_likelihood - 1e-9);
    }

    /// Two-subject hand-computable partial likelihood (Cox 1972 textbook case).
    ///
    /// With one event at the first time (covariate `x₀`) and one censored
    /// subject (covariate `x₁`) at a later time, the only contributing term is
    /// `θ₀ / (θ₀ + θ₁)` and the MLE diverges, so we instead verify the score
    /// vanishes at the analytic stationary value for a *balanced* two-event set.
    #[test]
    fn two_event_partial_likelihood_value() {
        // Two events, single covariate; PL = [θ0/(θ0+θ1)] · [θ1/θ1].
        // ℓ(β) = βx0 − ln(e^{βx0}+e^{βx1}).
        let x = vec![1.0, 0.0];
        let time = vec![1.0, 2.0];
        let status = vec![true, true];
        let beta = vec![0.5];
        let ord = build_ordering(&time, &status);
        let comp = pl_components(&x, &time, &status, 2, 1, &beta, &ord, TieMethod::Breslow);
        let expected = 0.5 * 1.0 - ((0.5_f64).exp() + 1.0).ln();
        assert!(
            (comp.loglik - expected).abs() < 1e-12,
            "PL value {} vs expected {}",
            comp.loglik,
            expected
        );
    }

    #[test]
    fn recover_coefficient_from_simulated_exponential_data() {
        // Simulate Cox/exponential survival: λ_i = λ0·exp(β x_i), x_i ∈ {0,1}.
        // For exponential T, survival time = −ln(U) / (λ0·exp(β x)).
        let mut rng = LcgRng::new(0xC0FFEE);
        let n = 400;
        let p = 1;
        let beta_true = 0.8_f64;
        let lambda0 = 0.1_f64;
        let mut x = vec![0.0; n];
        let mut time = vec![0.0; n];
        let mut status = vec![true; n];
        for i in 0..n {
            let xi = if rng.next_bool() { 1.0 } else { 0.0 };
            x[i] = xi;
            let u = rng.next_f64().max(1e-12);
            let rate = lambda0 * (beta_true * xi).exp();
            let t = -u.ln() / rate;
            // Administrative censoring at t = 60.
            if t > 60.0 {
                time[i] = 60.0;
                status[i] = false;
            } else {
                time[i] = t;
                status[i] = true;
            }
        }
        let fit = cox_ph_fit(&x, &time, &status, n, p, &CoxConfig::default()).expect("fit");
        assert!(
            (fit.coef[0] - beta_true).abs() < 0.25,
            "estimated β {} should be near true {beta_true}",
            fit.coef[0]
        );
        // The standard error should be well-defined and the effect significant.
        assert!(fit.std_err[0].is_finite() && fit.std_err[0] > 0.0);
        assert!(fit.p_value[0] < 0.05, "strong effect should be significant");
        // LR test should reject the null.
        assert!(fit.lr_statistic() > 0.0);
        assert!(fit.lr_p_value().expect("lr p") < 0.05);
    }

    #[test]
    fn breslow_and_efron_agree_without_ties() {
        // With no tied event times the two approximations coincide exactly.
        let x = vec![
            0.5, 1.0, -0.3, 0.2, 1.1, -0.5, 0.0, 0.8, -1.0, -0.4, 0.7, 0.3,
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![true, true, true, true, true, true];
        let cfg_breslow = CoxConfig::new(200, 1e-10, TieMethod::Breslow).expect("breslow cfg");
        let cfg_efron = CoxConfig::new(200, 1e-10, TieMethod::Efron).expect("efron cfg");
        let breslow = cox_ph_fit(&x, &time, &status, 6, 2, &cfg_breslow).expect("breslow");
        let efron = cox_ph_fit(&x, &time, &status, 6, 2, &cfg_efron).expect("efron");
        for k in 0..2 {
            assert!(
                (breslow.coef[k] - efron.coef[k]).abs() < 1e-6,
                "no-tie Breslow {} vs Efron {} differ at {k}",
                breslow.coef[k],
                efron.coef[k]
            );
        }
    }

    #[test]
    fn baseline_hazard_is_monotone_nondecreasing() {
        let x = vec![0.3, -0.4, 0.9, 0.1, -1.0, 0.6];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![true, false, true, true, false, true];
        let fit = cox_ph_fit(&x, &time, &status, 6, 1, &CoxConfig::default()).expect("fit");
        let table = fit.baseline_hazard_table();
        assert!(!table.is_empty());
        for w in table.windows(2) {
            assert!(
                w[1].1 >= w[0].1 - 1e-12,
                "cumulative hazard must be non-decreasing: {:?}",
                w
            );
            assert!(w[1].0 > w[0].0, "event times strictly increasing");
        }
        // Survival is monotone non-increasing in t and within (0, 1].
        let xq = [0.0];
        let s1 = fit.survival_function(1.5, &xq).expect("s");
        let s2 = fit.survival_function(5.5, &xq).expect("s");
        assert!(s1 <= 1.0 + 1e-12 && s2 > 0.0 && s2 <= s1 + 1e-12);
    }

    #[test]
    fn concordance_perfect_and_random() {
        // Risk score perfectly inversely orders survival ⇒ C = 1.
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![true, true, true, true, true];
        let risk_perfect = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let c = concordance_index(&time, &status, &risk_perfect).expect("c");
        assert!((c - 1.0).abs() < 1e-12, "perfect concordance, got {c}");
        // Reversed risk ⇒ C = 0.
        let risk_rev = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let c0 = concordance_index(&time, &status, &risk_rev).expect("c");
        assert!(c0.abs() < 1e-12, "anti-concordant, got {c0}");
        // Constant risk ⇒ all ties ⇒ C = 0.5.
        let risk_const = vec![2.0; 5];
        let chalf = concordance_index(&time, &status, &risk_const).expect("c");
        assert!((chalf - 0.5).abs() < 1e-12, "all ties, got {chalf}");
    }

    #[test]
    fn concordance_matches_fitted_risk_score() {
        // The fitted model's own risk scores should be concordant with the data.
        let x = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let time = vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];
        let status = vec![true, true, true, true, true, true];
        let fit = cox_ph_fit(&x, &time, &status, 6, 1, &CoxConfig::default()).expect("fit");
        let risk: Vec<f64> = (0..6)
            .map(|i| fit.linear_predictor(&x[i..i + 1]).expect("lp"))
            .collect();
        let c = concordance_index(&time, &status, &risk).expect("c");
        // A single binary covariate produces only two distinct risk scores, so
        // every within-group pair ties (½ each); the cross-group pairs are all
        // concordant. C therefore lands well above the 0.5 random baseline.
        assert!(
            c > 0.75,
            "fitted risk should be strongly concordant, got {c}"
        );
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let cfg = CoxConfig::default();
        // No events.
        assert!(matches!(
            cox_ph_fit(&[1.0, 2.0], &[1.0, 2.0], &[false, false], 2, 1, &cfg),
            Err(StatsError::InvalidParameter { .. })
        ));
        // Shape mismatch.
        assert!(matches!(
            cox_ph_fit(&[1.0, 2.0, 3.0], &[1.0, 2.0], &[true, false], 2, 1, &cfg),
            Err(StatsError::ShapeMismatch { .. })
        ));
        // p = 0.
        assert!(matches!(
            cox_ph_fit(&[], &[1.0], &[true], 1, 0, &cfg),
            Err(StatsError::InvalidParameter { .. })
        ));
        // Negative time.
        assert!(matches!(
            cox_ph_fit(&[1.0, 2.0], &[-1.0, 2.0], &[true, true], 2, 1, &cfg),
            Err(StatsError::InvalidParameter { .. })
        ));
        // Non-finite covariate.
        assert!(matches!(
            cox_ph_fit(&[1.0, f64::NAN], &[1.0, 2.0], &[true, true], 2, 1, &cfg),
            Err(StatsError::NonFiniteValue(_))
        ));
        // Bad config.
        assert!(CoxConfig::new(0, 1e-9, TieMethod::Efron).is_err());
        assert!(CoxConfig::new(50, -1.0, TieMethod::Efron).is_err());
    }

    #[test]
    fn linear_predictor_dimension_error() {
        // A non-degenerate 5×2 design so the information matrix is invertible.
        let x = vec![0.5, 1.0, -0.3, 0.2, 1.1, -0.5, 0.0, 0.8, -1.0, -0.4];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![true, true, true, true, false];
        let fit = cox_ph_fit(&x, &time, &status, 5, 2, &CoxConfig::default()).expect("fit");
        assert!(matches!(
            fit.linear_predictor(&[1.0]),
            Err(StatsError::DimensionMismatch { .. })
        ));
        // A correctly-sized vector succeeds.
        assert!(fit.linear_predictor(&[1.0, 0.0]).is_ok());
    }
}

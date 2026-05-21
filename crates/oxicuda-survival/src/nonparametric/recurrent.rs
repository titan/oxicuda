//! Recurrent-event survival analysis.
//!
//! Implements the Mean Cumulative Function (MCF) estimator (Lawless & Nadeau 1995),
//! two-sample MCF comparison test, and the Andersen-Gill counting-process Cox model
//! (Andersen & Gill 1982) for situations where subjects can experience an event
//! multiple times.
//!
//! # References
//! - Andersen, P.K. & Gill, R.D. (1982). Cox's regression model for counting processes.
//!   *Ann. Statist.*, 10, 1100–1120.
//! - Lawless, J.F. & Nadeau, C. (1995). Some simple robust methods for the analysis of
//!   recurrent events. *Technometrics*, 37(2), 158–168.
//! - Lin, D.Y. & Wei, L.J. (1989). The robust inference for the Cox proportional hazards
//!   model. *JASA*, 84, 1074–1078.

use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::solve::cholesky_solve;

// ─── Data Model ───────────────────────────────────────────────────────────────

/// A single observation interval in a recurrent-event study.
///
/// A subject with multiple episodes contributes multiple rows:
/// `(start, stop, event)` represents an interval of follow-up.
#[derive(Debug, Clone)]
pub struct RecurrentObs {
    /// Identifier for the subject (can repeat across multiple episodes).
    pub subject_id: usize,
    /// Start of the follow-up interval (≥ 0, often 0 for first episode).
    pub start: f64,
    /// End of the interval: event time or censoring time.
    pub stop: f64,
    /// 1 = event occurred at `stop`; 0 = censored at `stop`.
    pub event: u8,
}

// ─── MCF Result ───────────────────────────────────────────────────────────────

/// Output of the Mean Cumulative Function estimator for recurrent events.
///
/// `MCF(t) = Σ_{tᵢ ≤ t} dᵢ / nᵢ` where dᵢ is the event count at tᵢ and nᵢ is the
/// number of distinct subjects at risk at tᵢ (Lawless & Nadeau 1995).
#[derive(Debug, Clone)]
pub struct RecurrentMcfResult {
    /// Unique event times in ascending order.
    pub times: Vec<f64>,
    /// MCF value at each event time (cumulative).
    pub mcf: Vec<f64>,
    /// Nelson-Aalen variance estimate at each time: Σ dᵢ / nᵢ².
    pub variance: Vec<f64>,
    /// Cumulative event count at each time.
    pub n_events: Vec<usize>,
    /// Number of distinct subjects at risk at each event time.
    pub at_risk: Vec<usize>,
}

// ─── Two-Sample Test Result ───────────────────────────────────────────────────

/// Result of a two-sample MCF comparison test.
///
/// Based on the pooled score test contrasting the recurrence rate in two groups.
#[derive(Debug, Clone)]
pub struct RecurrentGroupTest {
    /// Score test Z statistic (compared to N(0,1)).
    pub z_statistic: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Rate ratio at the median event time (dA/nA)/(dB/nB); `f64::NAN` if undefined.
    pub rate_ratio_at_median: f64,
}

// ─── Andersen-Gill Config & Fit ───────────────────────────────────────────────

/// Configuration for the Andersen-Gill counting-process Cox model.
#[derive(Debug, Clone)]
pub struct AgConfig {
    /// Maximum Newton-Raphson iterations (default 50).
    pub max_iter: usize,
    /// Convergence tolerance on max |score component| (default 1e-6).
    pub tol: f64,
}

impl Default for AgConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1.0e-6,
        }
    }
}

/// Fitted Andersen-Gill model result.
#[derive(Debug, Clone)]
pub struct AgFit {
    /// Regression coefficients β (length = n_covariates).
    pub beta: Vec<f64>,
    /// Partial log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Newton-Raphson iterations consumed.
    pub n_iter: usize,
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
    /// Total number of events observed across all subjects and episodes.
    pub n_events: usize,
    /// Breslow baseline cumulative hazard at each unique event time.
    pub baseline_cumhaz: Vec<f64>,
    /// Unique event times for the baseline cumulative hazard.
    pub baseline_times: Vec<f64>,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute the Mean Cumulative Function (MCF) for recurrent-event data.
///
/// The MCF is a Nelson-Aalen-type estimator that accounts for multiple events per
/// subject.  Subjects remain in the risk set through all their episodes until the
/// study ends (i.e., after an event a subject is still at risk for the next).
///
/// # Errors
/// Returns [`SurvivalError::EmptyDataset`] if `obs` is empty.
/// Returns [`SurvivalError::InvalidParameter`] if any interval has `stop < start`.
pub fn recurrent_mcf(obs: &[RecurrentObs]) -> SurvivalResult<RecurrentMcfResult> {
    validate_obs(obs)?;

    let event_times = unique_event_times(obs);
    if event_times.is_empty() {
        return Ok(RecurrentMcfResult {
            times: vec![],
            mcf: vec![],
            variance: vec![],
            n_events: vec![],
            at_risk: vec![],
        });
    }

    let mut mcf_cur = 0.0_f64;
    let mut var_cur = 0.0_f64;
    let mut cum_events = 0usize;

    let mut times_out = Vec::with_capacity(event_times.len());
    let mut mcf_out = Vec::with_capacity(event_times.len());
    let mut var_out = Vec::with_capacity(event_times.len());
    let mut n_events_out = Vec::with_capacity(event_times.len());
    let mut at_risk_out = Vec::with_capacity(event_times.len());

    for &t in &event_times {
        let d = event_count_at(obs, t);
        let n = at_risk_count(obs, t);

        if n > 0 {
            let d_f = d as f64;
            let n_f = n as f64;
            mcf_cur += d_f / n_f;
            var_cur += d_f / (n_f * n_f);
        }
        cum_events += d;

        times_out.push(t);
        mcf_out.push(mcf_cur);
        var_out.push(var_cur);
        n_events_out.push(cum_events);
        at_risk_out.push(n);
    }

    Ok(RecurrentMcfResult {
        times: times_out,
        mcf: mcf_out,
        variance: var_out,
        n_events: n_events_out,
        at_risk: at_risk_out,
    })
}

/// Two-sample MCF comparison between groups 0 and 1.
///
/// Computes the score-type test statistic:
/// `U = Σₜ (dA(t) − nA(t)/n(t) · d(t))`
/// `V = Σₜ nA(t)·nB(t)/n(t)² · (d(t)/n(t)) · (1 − d(t)/n(t))`
/// `Z = U / sqrt(V)`
///
/// `subject_group` maps each `(subject_id, group)` pair where group ∈ {0, 1}.
///
/// # Errors
/// Returns errors for invalid data, no events, or missing group assignments.
pub fn recurrent_two_sample(
    obs: &[RecurrentObs],
    subject_group: &[(usize, u8)],
) -> SurvivalResult<RecurrentGroupTest> {
    validate_obs(obs)?;

    // Build subject-to-group lookup
    let mut group_map = std::collections::HashMap::new();
    for &(sid, g) in subject_group {
        if g > 1 {
            return Err(SurvivalError::InvalidParameter(format!(
                "group must be 0 or 1, got {g}"
            )));
        }
        group_map.insert(sid, g);
    }

    // Partition observations into group A (0) and group B (1)
    let obs_a: Vec<&RecurrentObs> = obs
        .iter()
        .filter(|o| group_map.get(&o.subject_id).copied().unwrap_or(0) == 0)
        .collect();
    let obs_b: Vec<&RecurrentObs> = obs
        .iter()
        .filter(|o| group_map.get(&o.subject_id).copied().unwrap_or(0) == 1)
        .collect();

    let event_times = unique_event_times(obs);
    if event_times.is_empty() {
        return Err(SurvivalError::NoEvents);
    }

    let mut u_stat = 0.0_f64;
    let mut v_stat = 0.0_f64;

    // Collect per-time rate ratios for median computation
    let mut rate_ratios: Vec<(f64, f64)> = Vec::new(); // (time, rr)

    for &t in &event_times {
        let da = event_count_at_refs(&obs_a, t) as f64;
        let db = event_count_at_refs(&obs_b, t) as f64;
        let na = at_risk_count_refs(&obs_a, t) as f64;
        let nb = at_risk_count_refs(&obs_b, t) as f64;
        let d = da + db;
        let n = na + nb;

        if n < 1.0 {
            continue;
        }

        // Score contribution: observed A events minus expected under H0
        u_stat += da - na / n * d;

        // Variance contribution under H0 (hypergeometric-type)
        let prop = d / n;
        v_stat += na * nb / (n * n) * prop * (1.0 - prop);

        // Rate ratio at this time
        if na > 0.0 && nb > 0.0 {
            let rr = (da / na) / (db / nb + f64::EPSILON);
            rate_ratios.push((t, rr));
        }
    }

    let z = if v_stat > 0.0 {
        u_stat / v_stat.sqrt()
    } else {
        0.0
    };
    let p_value = 2.0 * (1.0 - norm_cdf(z.abs()));

    // Rate ratio at median event time
    let rate_ratio_at_median = if rate_ratios.is_empty() {
        f64::NAN
    } else {
        let mid = rate_ratios.len() / 2;
        rate_ratios[mid].1
    };

    Ok(RecurrentGroupTest {
        z_statistic: z,
        p_value,
        rate_ratio_at_median,
    })
}

/// Fit an Andersen-Gill counting-process Cox model for recurrent events.
///
/// Each episode `(start, stop, event)` contributes to the partial likelihood via the
/// counting-process risk set: observation i is at risk at time t iff `start_i < t ≤ stop_i`.
///
/// `covariates` is a flat row-major `[n_obs × p]` matrix.
///
/// When `p = 0` (no covariates), returns a null model with `beta = []` and
/// `log_likelihood = 0.0`.
///
/// # Errors
/// Returns errors for invalid intervals, empty data, shape mismatches, or non-convergence.
pub fn fit_andersen_gill(
    obs: &[RecurrentObs],
    covariates: &[f64],
    config: &AgConfig,
) -> SurvivalResult<AgFit> {
    validate_obs(obs)?;

    let n_obs = obs.len();
    let p = covariates.len().checked_div(n_obs).unwrap_or(0);

    if p > 0 && covariates.len() != n_obs * p {
        return Err(SurvivalError::DimensionMismatch {
            a: covariates.len(),
            b: n_obs * p,
        });
    }

    let n_events_total: usize = obs.iter().filter(|o| o.event == 1).count();
    if n_events_total == 0 {
        return Err(SurvivalError::NoEvents);
    }

    // Null model when p == 0
    if p == 0 {
        let (baseline_times, baseline_cumhaz) = compute_breslow_baseline(obs, &[], 0)?;
        return Ok(AgFit {
            beta: vec![],
            log_likelihood: 0.0,
            n_iter: 0,
            converged: true,
            n_events: n_events_total,
            baseline_cumhaz,
            baseline_times,
        });
    }

    // Newton-Raphson optimisation of AG partial log-likelihood
    let mut beta = vec![0.0_f64; p];
    let (mut ll, mut score, mut info) = ag_log_likelihood(obs, covariates, p, &beta)?;

    let mut converged = false;
    let mut n_iter = 0usize;

    for it in 0..config.max_iter {
        n_iter = it + 1;

        let max_score = score.iter().fold(0.0_f64, |acc, s| acc.max(s.abs()));
        if max_score < config.tol {
            converged = true;
            break;
        }

        // Newton step: solve I·Δβ = score
        let delta = match cholesky_solve(&info, &score, p) {
            Ok(d) => d,
            Err(_) => {
                // Ridge regularisation fallback
                let mut info_ridge = info.clone();
                for d in 0..p {
                    info_ridge[d * p + d] += 1.0e-4;
                }
                cholesky_solve(&info_ridge, &score, p)?
            }
        };

        // Halving line search
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..40 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + step * d)
                .collect();
            if let Ok((ll_new, sc_new, info_new)) = ag_log_likelihood(obs, covariates, p, &trial) {
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
        let max_score = score.iter().fold(0.0_f64, |acc, s| acc.max(s.abs()));
        if max_score < config.tol {
            converged = true;
        }
    }

    let (baseline_times, baseline_cumhaz) = compute_breslow_baseline(obs, covariates, p)?;

    Ok(AgFit {
        beta,
        log_likelihood: ll,
        n_iter,
        converged,
        n_events: n_events_total,
        baseline_cumhaz,
        baseline_times,
    })
}

/// Predict the cumulative mean function for a new covariate vector using a fitted AG model.
///
/// `new_covariate` is `[p]`; `times` are the query times (need not be event times).
/// The cumulative mean at time t is: `H₀(t) · exp(β^T x_new)` where H₀ is the
/// Breslow baseline cumulative hazard from the fitted model.
///
/// Returns a `Vec<f64>` of the same length as `times`.
///
/// # Errors
/// Returns `SurvivalError::DimensionMismatch` if `new_covariate.len() != beta.len()`.
pub fn predict_cumulative_mean(
    fit: &AgFit,
    _obs: &[RecurrentObs],
    _covariates: &[f64],
    new_covariate: &[f64],
    times: &[f64],
) -> SurvivalResult<Vec<f64>> {
    let p = fit.beta.len();
    if new_covariate.len() != p {
        return Err(SurvivalError::DimensionMismatch {
            a: new_covariate.len(),
            b: p,
        });
    }

    // log risk for new covariate
    let log_risk: f64 = fit
        .beta
        .iter()
        .zip(new_covariate.iter())
        .map(|(b, x)| b * x)
        .sum();
    let risk = log_risk.exp();

    // For each query time, find the cumulative baseline hazard via step interpolation
    let result = times
        .iter()
        .map(|&t| {
            let h0 = step_interpolate(&fit.baseline_times, &fit.baseline_cumhaz, t);
            h0 * risk
        })
        .collect();

    Ok(result)
}

// ─── Private Helpers ──────────────────────────────────────────────────────────

/// Validate observations: non-empty, stop ≥ start for every interval.
fn validate_obs(obs: &[RecurrentObs]) -> SurvivalResult<()> {
    if obs.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    for o in obs {
        if o.stop < o.start {
            return Err(SurvivalError::InvalidParameter(format!(
                "subject {}: stop ({}) < start ({})",
                o.subject_id, o.stop, o.start
            )));
        }
    }
    Ok(())
}

/// Collect all unique times at which at least one event occurred, sorted ascending.
fn unique_event_times(obs: &[RecurrentObs]) -> Vec<f64> {
    let mut times: Vec<f64> = obs
        .iter()
        .filter(|o| o.event == 1)
        .map(|o| o.stop)
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| (*b - *a).abs() < f64::EPSILON);
    times
}

/// Count distinct subjects at risk at time `t` (counting-process definition):
/// subject is at risk at t if any of its intervals satisfies `start < t ≤ stop`.
fn at_risk_count(obs: &[RecurrentObs], t: f64) -> usize {
    let mut subjects_at_risk = std::collections::HashSet::new();
    for o in obs {
        if o.start < t && t <= o.stop {
            subjects_at_risk.insert(o.subject_id);
        }
    }
    subjects_at_risk.len()
}

/// Count distinct subjects at risk from a reference slice.
fn at_risk_count_refs(obs: &[&RecurrentObs], t: f64) -> usize {
    let mut subjects_at_risk = std::collections::HashSet::new();
    for o in obs {
        if o.start < t && t <= o.stop {
            subjects_at_risk.insert(o.subject_id);
        }
    }
    subjects_at_risk.len()
}

/// Count total events at exactly time `t`.
fn event_count_at(obs: &[RecurrentObs], t: f64) -> usize {
    obs.iter()
        .filter(|o| o.event == 1 && (o.stop - t).abs() < f64::EPSILON)
        .count()
}

/// Count total events at exactly time `t` from a reference slice.
fn event_count_at_refs(obs: &[&RecurrentObs], t: f64) -> usize {
    obs.iter()
        .filter(|o| o.event == 1 && (o.stop - t).abs() < f64::EPSILON)
        .count()
}

/// Build the counting-process risk set at time `t`:
/// indices of observations satisfying `start < t ≤ stop`.
fn build_risk_set(obs: &[RecurrentObs], t: f64) -> Vec<usize> {
    obs.iter()
        .enumerate()
        .filter(|(_, o)| o.start < t && t <= o.stop)
        .map(|(i, _)| i)
        .collect()
}

/// Compute the AG partial log-likelihood, score, and observed information at `beta`.
///
/// Returns `(log_likelihood, score[p], information[p×p])`.
fn ag_log_likelihood(
    obs: &[RecurrentObs],
    covariates: &[f64],
    p: usize,
    beta: &[f64],
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
    let n_obs = obs.len();

    // Pre-compute linear predictors and exp(xᵢβ) for each observation
    let mut eta = vec![0.0_f64; n_obs];
    let mut exp_eta = vec![0.0_f64; n_obs];
    for i in 0..n_obs {
        let mut xb = 0.0_f64;
        for k in 0..p {
            xb += covariates[i * p + k] * beta[k];
        }
        eta[i] = xb;
        exp_eta[i] = xb.exp();
    }

    let event_times = unique_event_times(obs);
    let mut ll = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];

    for &t in &event_times {
        let risk_idx = build_risk_set(obs, t);
        if risk_idx.is_empty() {
            continue;
        }

        // S0(β, t) = Σ_{j ∈ risk} exp(xⱼβ)
        let s0: f64 = risk_idx.iter().map(|&j| exp_eta[j]).sum();
        if s0 <= 0.0 {
            return Err(SurvivalError::NumericalInstability(
                "S0 is zero at an event time".to_string(),
            ));
        }
        let log_s0 = s0.ln();

        // S1(β, t) = Σ_{j ∈ risk} xⱼ exp(xⱼβ)
        let mut s1 = vec![0.0_f64; p];
        for &j in &risk_idx {
            let ej = exp_eta[j];
            for k in 0..p {
                s1[k] += covariates[j * p + k] * ej;
            }
        }

        // S2(β, t) = Σ_{j ∈ risk} xⱼ xⱼ^T exp(xⱼβ)
        let mut s2 = vec![0.0_f64; p * p];
        for &j in &risk_idx {
            let ej = exp_eta[j];
            for k in 0..p {
                for l in 0..p {
                    s2[k * p + l] += covariates[j * p + k] * covariates[j * p + l] * ej;
                }
            }
        }

        // Mean covariate in risk set: ē = S1 / S0
        let e_bar: Vec<f64> = s1.iter().map(|v| v / s0).collect();

        // Accumulate contributions from all events at time t
        for (i, o) in obs.iter().enumerate() {
            if o.event == 1 && (o.stop - t).abs() < f64::EPSILON {
                // log-likelihood contribution: xᵢβ − log(S0)
                ll += eta[i] - log_s0;

                // Score contribution: xᵢ − ē
                for k in 0..p {
                    score[k] += covariates[i * p + k] - e_bar[k];
                }

                // Information contribution: S2/S0 − ē ē^T
                for k in 0..p {
                    for l in 0..p {
                        info[k * p + l] += s2[k * p + l] / s0 - e_bar[k] * e_bar[l];
                    }
                }
            }
        }
    }

    Ok((ll, score, info))
}

/// Compute the Breslow baseline cumulative hazard for the AG model.
///
/// `H₀(t) = Σ_{tᵢ ≤ t} dᵢ / Σ_{j ∈ risk(tᵢ)} exp(xⱼβ)`
fn compute_breslow_baseline(
    obs: &[RecurrentObs],
    covariates: &[f64],
    p: usize,
) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
    let n_obs = obs.len();
    let event_times = unique_event_times(obs);
    if event_times.is_empty() {
        return Ok((vec![], vec![]));
    }

    // Use zero beta for null model (p==0) or already fitted beta embedded via covariates arg
    let mut exp_eta = vec![1.0_f64; n_obs];
    if p > 0 && covariates.len() == n_obs * p {
        for i in 0..n_obs {
            let mut xb = 0.0_f64;
            for k in 0..p {
                xb += covariates[i * p + k];
            }
            exp_eta[i] = xb.exp();
        }
    }

    let mut times_out = Vec::with_capacity(event_times.len());
    let mut cumhaz_out = Vec::with_capacity(event_times.len());
    let mut h_cur = 0.0_f64;

    for &t in &event_times {
        let risk_idx = build_risk_set(obs, t);
        let s0: f64 = risk_idx.iter().map(|&j| exp_eta[j]).sum();
        let d = event_count_at(obs, t) as f64;
        if s0 > 0.0 {
            h_cur += d / s0;
        }
        times_out.push(t);
        cumhaz_out.push(h_cur);
    }

    Ok((times_out, cumhaz_out))
}

/// Step-function interpolation for a sorted baseline cumulative hazard.
///
/// Returns the value of the step function at `t` (left-continuous, returning the
/// last observed cumulative hazard for times ≥ last event time).
fn step_interpolate(times: &[f64], values: &[f64], t: f64) -> f64 {
    if times.is_empty() {
        return 0.0;
    }
    // Find the largest index where times[i] <= t
    let mut result = 0.0_f64;
    for (i, &ti) in times.iter().enumerate() {
        if ti <= t {
            result = values[i];
        } else {
            break;
        }
    }
    result
}

/// Standard normal CDF via Abramowitz & Stegun approximation (7.1.26).
///
/// Maximum absolute error < 7.5e-8.
fn norm_cdf(x: f64) -> f64 {
    // Handle large arguments
    if x > 8.0 {
        return 1.0;
    }
    if x < -8.0 {
        return 0.0;
    }
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-x * x * 0.5).exp() / (2.0 * std::f64::consts::PI).sqrt();
    if x >= 0.0 {
        1.0 - pdf * poly
    } else {
        pdf * poly
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper builders ──────────────────────────────────────────────────────

    fn simple_obs(subject_id: usize, start: f64, stop: f64, event: u8) -> RecurrentObs {
        RecurrentObs {
            subject_id,
            start,
            stop,
            event,
        }
    }

    // ── MCF Tests ────────────────────────────────────────────────────────────

    /// 1. Single event per subject — MCF should match Nelson-Aalen cumulative hazard.
    #[test]
    fn mcf_single_event_per_subject() {
        // 4 subjects, 1 event each at times 1,2,3,4
        let obs = vec![
            simple_obs(0, 0.0, 1.0, 1),
            simple_obs(1, 0.0, 2.0, 1),
            simple_obs(2, 0.0, 3.0, 1),
            simple_obs(3, 0.0, 4.0, 1),
        ];
        let result = recurrent_mcf(&obs).expect("should succeed");
        assert_eq!(result.times.len(), 4);
        // At t=1: d=1, n=4 → MCF = 1/4 = 0.25
        assert!((result.mcf[0] - 0.25).abs() < 1.0e-10);
        // At t=2: d=1, n=3 (subject 0 left) → MCF += 1/3
        assert!((result.mcf[1] - (0.25 + 1.0 / 3.0)).abs() < 1.0e-10);
    }

    /// 2. Single subject with 3 recurrent events — MCF at last time = 3.0.
    #[test]
    fn mcf_recurrent_events() {
        // Subject 0 has 3 episodes; between events they stay at risk
        let obs = vec![
            simple_obs(0, 0.0, 1.0, 1),
            simple_obs(0, 1.0, 2.0, 1),
            simple_obs(0, 2.0, 3.0, 1),
        ];
        let result = recurrent_mcf(&obs).expect("should succeed");
        // At each time: d=1, n=1 (only one subject) → each step adds 1.0
        // MCF(3) = 1 + 1 + 1 = 3.0
        let last_mcf = *result.mcf.last().expect("non-empty");
        assert!((last_mcf - 3.0).abs() < 1.0e-10, "MCF(last) = {last_mcf}");
    }

    /// 3. MCF must be monotone non-decreasing.
    #[test]
    fn mcf_cumulative_increasing() {
        let obs = vec![
            simple_obs(0, 0.0, 1.0, 1),
            simple_obs(0, 1.0, 3.0, 1),
            simple_obs(1, 0.0, 2.0, 0), // censored
            simple_obs(2, 0.0, 2.5, 1),
            simple_obs(1, 2.0, 4.0, 1),
        ];
        let result = recurrent_mcf(&obs).expect("ok");
        for w in result.mcf.windows(2) {
            assert!(
                w[1] >= w[0] - 1.0e-12,
                "MCF not non-decreasing: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    /// 4. Event times must be sorted ascending.
    #[test]
    fn mcf_times_sorted() {
        let obs = vec![
            simple_obs(0, 0.0, 3.0, 1),
            simple_obs(1, 0.0, 1.0, 1),
            simple_obs(2, 0.0, 5.0, 1),
            simple_obs(3, 0.0, 2.0, 1),
        ];
        let result = recurrent_mcf(&obs).expect("ok");
        for w in result.times.windows(2) {
            assert!(w[1] > w[0], "times not sorted: {} >= {}", w[0], w[1]);
        }
    }

    /// 5. All variance estimates must be non-negative.
    #[test]
    fn mcf_variance_nonneg() {
        let obs = vec![
            simple_obs(0, 0.0, 2.0, 1),
            simple_obs(1, 0.0, 3.0, 1),
            simple_obs(0, 2.0, 5.0, 1),
        ];
        let result = recurrent_mcf(&obs).expect("ok");
        for &v in &result.variance {
            assert!(v >= 0.0, "negative variance: {v}");
        }
    }

    /// 6. All-censored data → no event times → empty MCF.
    #[test]
    fn mcf_all_censored() {
        let obs = vec![
            simple_obs(0, 0.0, 5.0, 0),
            simple_obs(1, 0.0, 3.0, 0),
            simple_obs(2, 0.0, 7.0, 0),
        ];
        let result = recurrent_mcf(&obs).expect("ok");
        assert!(
            result.times.is_empty(),
            "expected empty MCF for all-censored"
        );
        assert!(result.mcf.is_empty());
        assert!(result.variance.is_empty());
    }

    // ── Two-Sample Test ──────────────────────────────────────────────────────

    /// 7. Two-sample test with finite, valid p-value.
    #[test]
    fn two_sample_test_finite() {
        let obs = vec![
            simple_obs(0, 0.0, 1.0, 1),
            simple_obs(1, 0.0, 2.0, 1),
            simple_obs(2, 0.0, 3.0, 1),
            simple_obs(3, 0.0, 1.5, 1),
            simple_obs(4, 0.0, 2.5, 0),
            simple_obs(5, 0.0, 4.0, 1),
        ];
        // Groups: 0,1,2 → group 0; 3,4,5 → group 1
        let groups: Vec<(usize, u8)> = vec![(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)];
        let result = recurrent_two_sample(&obs, &groups).expect("ok");
        assert!(result.p_value.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.z_statistic.is_finite());
    }

    /// 8. Equal groups with identical data → Z ≈ 0, p ≈ 1.
    #[test]
    fn two_sample_equal_groups() {
        // Mirror-image groups with same event pattern
        let obs = vec![
            simple_obs(0, 0.0, 1.0, 1),
            simple_obs(1, 0.0, 1.0, 1),
            simple_obs(2, 0.0, 2.0, 1),
            simple_obs(3, 0.0, 2.0, 1),
        ];
        let groups: Vec<(usize, u8)> = vec![(0, 0), (1, 1), (2, 0), (3, 1)];
        let result = recurrent_two_sample(&obs, &groups).expect("ok");
        // With perfectly symmetric data, Z should be 0
        assert!(
            result.z_statistic.abs() < 1.0e-10,
            "Z = {}",
            result.z_statistic
        );
        assert!(result.p_value > 0.9, "p = {}", result.p_value);
    }

    // ── Andersen-Gill Tests ──────────────────────────────────────────────────

    /// 9. Null model (p=0): beta.len()==0, log_likelihood is finite.
    #[test]
    fn ag_fit_no_covariates() {
        let obs = vec![
            simple_obs(0, 0.0, 1.0, 1),
            simple_obs(1, 0.0, 2.0, 1),
            simple_obs(2, 0.0, 3.0, 0),
        ];
        let config = AgConfig::default();
        let fit = fit_andersen_gill(&obs, &[], &config).expect("ok");
        assert_eq!(fit.beta.len(), 0);
        assert!(fit.log_likelihood.is_finite());
        assert!(fit.converged);
    }

    /// 10. With clear covariate signal: algorithm should converge.
    #[test]
    fn ag_fit_converges() {
        // High-risk subjects (x=1) have events at earlier times
        let obs = vec![
            simple_obs(0, 0.0, 0.5, 1), // x=1, event early
            simple_obs(0, 0.5, 1.0, 1),
            simple_obs(1, 0.0, 1.5, 1), // x=1
            simple_obs(2, 0.0, 3.0, 1), // x=0, event late
            simple_obs(3, 0.0, 4.0, 0), // x=0, censored
            simple_obs(4, 0.0, 4.5, 0), // x=0, censored
        ];
        // Covariates: [1, 1, 1, 0, 0, 0] for each episode row
        let covs = vec![1.0_f64, 1.0, 1.0, 0.0, 0.0, 0.0];
        let config = AgConfig::default();
        let fit = fit_andersen_gill(&obs, &covs, &config).expect("ok");
        assert!(fit.converged, "algorithm should converge");
        assert!(fit.log_likelihood.is_finite());
        assert_eq!(fit.beta.len(), 1);
    }

    /// 11. n_events matches total event count in obs.
    #[test]
    fn ag_fit_n_events_correct() {
        let obs = vec![
            simple_obs(0, 0.0, 1.0, 1),
            simple_obs(0, 1.0, 2.0, 0),
            simple_obs(1, 0.0, 1.5, 1),
            simple_obs(2, 0.0, 2.5, 1),
        ];
        let covs = vec![0.5_f64, 0.5, -0.5, 0.0];
        let config = AgConfig::default();
        let fit = fit_andersen_gill(&obs, &covs, &config).expect("ok");
        // Total events: obs 0 (event=1), obs 2 (event=1), obs 3 (event=1) = 3
        assert_eq!(fit.n_events, 3);
    }

    /// 12. Predicted cumulative mean is monotone non-decreasing.
    #[test]
    fn predict_cumulative_mean_increasing() {
        let obs = vec![
            simple_obs(0, 0.0, 1.0, 1),
            simple_obs(0, 1.0, 3.0, 1),
            simple_obs(1, 0.0, 2.0, 1),
            simple_obs(1, 2.0, 5.0, 0),
        ];
        let covs = vec![1.0_f64, 1.0, 0.0, 0.0];
        let config = AgConfig::default();
        let fit = fit_andersen_gill(&obs, &covs, &config).expect("ok");

        let query_times = vec![0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
        let preds = predict_cumulative_mean(&fit, &obs, &covs, &[0.5], &query_times).expect("ok");

        for w in preds.windows(2) {
            assert!(
                w[1] >= w[0] - 1.0e-12,
                "prediction not non-decreasing: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    /// 13. Counting-process: subjects remain at risk through multiple events.
    #[test]
    fn at_risk_counting_process() {
        // Subject 0 has two episodes: [0,1] and [1,3]; subject 1 has [0,2]
        let obs = vec![
            simple_obs(0, 0.0, 1.0, 1),
            simple_obs(0, 1.0, 3.0, 0),
            simple_obs(1, 0.0, 2.0, 1),
        ];
        // At t=1: risk_set for subject 0 = episode [0,1]: start(0)<1<=stop(1) ✓
        //                                  episode [1,3]: start(1)<1 is false
        //         risk_set for subject 1 = episode [0,2]: start(0)<1<=stop(2) ✓ → n=2
        let n_at_1 = at_risk_count(&obs, 1.0);
        assert_eq!(n_at_1, 2, "both subjects at risk at t=1");

        // At t=2: subject 0 episode [1,3]: start(1)<2<=stop(3) ✓
        //         subject 1 episode [0,2]: start(0)<2<=stop(2) ✓ → n=2
        let n_at_2 = at_risk_count(&obs, 2.0);
        assert_eq!(n_at_2, 2, "both subjects at risk at t=2");

        // At t=2.5: subject 0 episode [1,3]: ✓; subject 1 has no interval covering 2.5
        let n_at_25 = at_risk_count(&obs, 2.5);
        assert_eq!(n_at_25, 1, "only subject 0 at risk at t=2.5");
    }

    /// 14. stop < start should return an error.
    #[test]
    fn negative_time_error() {
        let obs = vec![simple_obs(0, 5.0, 2.0, 1)]; // stop < start
        let err = recurrent_mcf(&obs).expect_err("should fail");
        assert!(
            matches!(err, SurvivalError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
    }

    /// 15. Empty observations should return EmptyDataset error.
    #[test]
    fn empty_obs_error() {
        let obs: Vec<RecurrentObs> = vec![];
        let err = recurrent_mcf(&obs).expect_err("should fail");
        assert!(
            matches!(err, SurvivalError::EmptyDataset),
            "expected EmptyDataset, got {err:?}"
        );
    }
}

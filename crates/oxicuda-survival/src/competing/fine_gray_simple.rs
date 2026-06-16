//! Simplified Fine-Gray sub-distribution hazard model operating on raw slices.
//!
//! Unlike `fine_gray.rs`, this module does NOT depend on the `Dataset` struct.
//! It implements the Fine-Gray (1999) competing-risks regression via IPCW-weighted
//! gradient descent on the partial log-likelihood.
//!
//! ## Model
//!
//! The sub-distribution hazard for cause `k` is:
//!
//! ```text
//! λ_k(t|x) = λ₀_k(t) · exp(βᵀx)
//! ```
//!
//! Subjects who experienced a competing event remain in the risk set indefinitely,
//! but are reweighted by `G(t) / G(t_i)` where `G` is the Kaplan-Meier estimator
//! of the censoring distribution.
//!
//! ## Algorithm
//!
//! 1. Estimate G(t) (censoring KM) from all subjects.
//! 2. Compute IPCW weights for competing-event subjects.
//! 3. Run gradient ascent on the weighted partial log-likelihood.
//! 4. Estimate the CIF via the Aalen-Johansen estimator.

use crate::error::{SurvivalError, SurvivalResult};

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for the simplified Fine-Gray sub-distribution hazard model.
#[derive(Debug, Clone)]
pub struct FineGrayConfig {
    /// Number of covariates.
    pub n_covariates: usize,
    /// Which event type is "of interest" (1 or 2).
    pub event_type: u8,
    /// Learning rate for gradient ascent.
    pub lr: f64,
    /// Maximum number of gradient-ascent iterations.
    pub n_iter: usize,
}

impl Default for FineGrayConfig {
    fn default() -> Self {
        Self {
            n_covariates: 1,
            event_type: 1,
            lr: 0.01,
            n_iter: 200,
        }
    }
}

/// Result of a Fine-Gray model fit on raw slices.
#[derive(Debug, Clone)]
pub struct FineGraySimpleFit {
    /// Regression coefficients β (length = `n_covariates`).
    pub coefficients: Vec<f64>,
    /// Cumulative Incidence Function (CIF) at each unique event time.
    pub cif: Vec<f64>,
    /// Unique sorted times at which the CIF is evaluated.
    pub cif_times: Vec<f64>,
    /// Observed partial log-likelihood at the final iterate.
    pub log_likelihood: f64,
    /// Whether any gradient update was non-degenerate.
    pub converged: bool,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Censoring Kaplan-Meier estimator.
///
/// Treats censoring (event == 0) as "events" and observed events as "censored".
/// Returns `(sorted_times, G_values)` where `G_values[i]` is G(sorted_times[i]).
fn censoring_km(times: &[f64], events: &[u8], n: usize) -> (Vec<f64>, Vec<f64>) {
    // Sort indices by time
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut g_times: Vec<f64> = Vec::new();
    let mut g_vals: Vec<f64> = Vec::new();
    let mut g_cur = 1.0_f64;
    let mut at_risk = n as f64;
    let mut k = 0usize;

    while k < n {
        let t = times[idx[k]];
        let mut m = k;
        let mut n_censor = 0u64; // "censor events" for G(t) — actual censorings

        while m < n && times[idx[m]] == t {
            if events[idx[m]] == 0 {
                n_censor += 1;
            }
            m += 1;
        }
        if n_censor > 0 && at_risk > 0.0 {
            g_cur *= 1.0 - n_censor as f64 / at_risk;
        }
        g_times.push(t);
        g_vals.push(g_cur);
        at_risk -= (m - k) as f64;
        k = m;
    }

    (g_times, g_vals)
}

/// Evaluate G(t) by step-function lookup: return G at the largest recorded time ≤ t.
#[inline]
fn eval_g(g_times: &[f64], g_vals: &[f64], t: f64) -> f64 {
    let mut v = 1.0_f64;
    for (i, &gt) in g_times.iter().enumerate() {
        if gt <= t {
            v = g_vals[i];
        } else {
            break;
        }
    }
    v.max(1.0e-300)
}

/// Overall Kaplan-Meier estimator treating both event types as failures.
///
/// Returns `(sorted_unique_times, S_values)`.
fn overall_km(times: &[f64], events: &[u8], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out_times: Vec<f64> = Vec::new();
    let mut out_surv: Vec<f64> = Vec::new();
    let mut s = 1.0_f64;
    let mut at_risk = n as f64;
    let mut k = 0usize;

    while k < n {
        let t = times[idx[k]];
        let mut m = k;
        let mut d_total = 0u64;

        while m < n && times[idx[m]] == t {
            if events[idx[m]] != 0 {
                d_total += 1;
            }
            m += 1;
        }
        if d_total > 0 && at_risk > 0.0 {
            s *= 1.0 - d_total as f64 / at_risk;
            out_times.push(t);
            out_surv.push(s);
        }
        at_risk -= (m - k) as f64;
        k = m;
    }

    (out_times, out_surv)
}

/// Evaluate S(t^-) = S just before t (left-limit).
/// For Aalen-Johansen we need S(t^-) which equals S at the previous jump.
#[inline]
fn eval_s_minus(km_times: &[f64], km_surv: &[f64], t: f64) -> f64 {
    // S(t^-) is the KM value strictly before t
    let mut s_before = 1.0_f64;
    for (i, &kt) in km_times.iter().enumerate() {
        if kt < t {
            s_before = km_surv[i];
        } else {
            break;
        }
    }
    s_before
}

/// Aalen-Johansen non-parametric CIF estimator.
///
/// CIF_k(t) = Σ_{t_j ≤ t} S(t_j^-) · d_k(t_j) / n_risk(t_j)
///
/// Returns `(unique_event_times_for_cause_k, CIF_values)`.
fn aalen_johansen_cif(times: &[f64], events: &[u8], n: usize, cause: u8) -> (Vec<f64>, Vec<f64>) {
    // Compute overall KM survival
    let (km_times, km_surv) = overall_km(times, events, n);

    // Sort all subjects by time
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out_times: Vec<f64> = Vec::new();
    let mut out_cif: Vec<f64> = Vec::new();
    let mut cif_cum = 0.0_f64;
    let mut at_risk = n as f64;
    let mut k = 0usize;

    while k < n {
        let t = times[idx[k]];
        let mut m = k;
        let mut d_cause = 0u64;

        while m < n && times[idx[m]] == t {
            if events[idx[m]] == cause {
                d_cause += 1;
            }
            m += 1;
        }
        if d_cause > 0 && at_risk > 0.0 {
            let s_minus = eval_s_minus(&km_times, &km_surv, t);
            cif_cum += s_minus * d_cause as f64 / at_risk;
            out_times.push(t);
            out_cif.push(cif_cum);
        }
        at_risk -= (m - k) as f64;
        k = m;
    }

    (out_times, out_cif)
}

/// Compute the IPCW-weighted partial log-likelihood score (gradient) for one pass.
///
/// Uses the Schoenfeld-style score:
/// `score[k] = Σ_{events of interest i} { x_ik - weighted_mean_x_k(t_i) }`
///
/// where the weighted mean uses:
/// - subjects still at risk (t_j >= t_i): weight = 1
/// - subjects with competing event at t_j < t_i: weight = G(t_i) / G(t_j)
/// - censored subjects with t_j < t_i: weight = 0 (removed from risk set)
fn compute_score_and_ll(
    covariates: &[f64],
    times: &[f64],
    events: &[u8],
    n: usize,
    p: usize,
    beta: &[f64],
    event_type: u8,
    g_times: &[f64],
    g_vals: &[f64],
) -> (f64, Vec<f64>) {
    // Collect unique event-of-interest times
    let mut ev_times: Vec<f64> = (0..n)
        .filter(|&i| events[i] == event_type)
        .map(|i| times[i])
        .collect();
    ev_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ev_times.dedup_by(|a, b| (*a - *b).abs() < 1.0e-12);

    let mut loglik = 0.0_f64;
    let mut score = vec![0.0_f64; p];

    for &t in &ev_times {
        let g_t = eval_g(g_times, g_vals, t);
        let mut s0 = 0.0_f64;
        let mut s1 = vec![0.0_f64; p];
        let mut x_ev_sum = vec![0.0_f64; p];
        let mut eta_ev_sum = 0.0_f64;
        let mut d_count = 0.0_f64;

        for i in 0..n {
            let ti = times[i];
            // Determine IPCW weight for subject i in risk set at time t
            let w_i = if events[i] == 0 && ti < t {
                // censored before t: excluded
                0.0
            } else if events[i] != event_type && events[i] != 0 && ti < t {
                // competing event before t: reweighted
                let g_ti = eval_g(g_times, g_vals, ti);
                g_t / g_ti
            } else if ti >= t || (events[i] == event_type && (ti - t).abs() < 1.0e-12) {
                // still at risk or current event subject
                1.0
            } else {
                0.0
            };

            if w_i <= 0.0 {
                continue;
            }

            let dot: f64 = (0..p).map(|k| beta[k] * covariates[i * p + k]).sum();
            let exp_dot = dot.exp();
            let w = w_i * exp_dot;
            s0 += w;

            for k in 0..p {
                s1[k] += w * covariates[i * p + k];
            }

            // Accumulate event contributions at exactly t
            if events[i] == event_type && (ti - t).abs() < 1.0e-12 {
                d_count += 1.0;
                eta_ev_sum += dot;
                for k in 0..p {
                    x_ev_sum[k] += covariates[i * p + k];
                }
            }
        }

        if d_count == 0.0 || s0 <= 0.0 {
            continue;
        }

        loglik += eta_ev_sum - d_count * s0.ln();
        let x_bar: Vec<f64> = s1.iter().map(|x| x / s0).collect();
        for k in 0..p {
            score[k] += x_ev_sum[k] - d_count * x_bar[k];
        }
    }

    (loglik, score)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Fit a Fine-Gray sub-distribution hazard model from raw slices.
///
/// # Parameters
/// - `covariates`: row-major `[n × n_covariates]` covariate matrix.
/// - `times`: observed times, length `n`.
/// - `events`: event indicators, length `n`. 0 = censored, 1 = event of interest, 2 = competing.
/// - `n`: number of subjects.
/// - `config`: algorithm configuration.
///
/// # Errors
/// - [`SurvivalError::EmptyDataset`] when `n == 0`.
/// - [`SurvivalError::InvalidParameter`] for invalid config fields or array length mismatches.
/// - [`SurvivalError::NoEvents`] when no subject has the event of interest.
pub fn fine_gray_fit(
    covariates: &[f64],
    times: &[f64],
    events: &[u8],
    n: usize,
    config: &FineGrayConfig,
) -> SurvivalResult<FineGraySimpleFit> {
    // ── Validation ────────────────────────────────────────────────────────────
    if n == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    let p = config.n_covariates;
    if p == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_covariates must be >= 1".to_string(),
        ));
    }
    if covariates.len() != n * p {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n * p],
            got: vec![covariates.len()],
        });
    }
    if times.len() != n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n],
            got: vec![times.len()],
        });
    }
    if events.len() != n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n],
            got: vec![events.len()],
        });
    }
    if config.lr <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "lr (learning rate) must be > 0".to_string(),
        ));
    }
    if config.n_iter == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_iter must be >= 1".to_string(),
        ));
    }
    for &t in times {
        if t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }

    let event_type = config.event_type;
    let n_events_of_interest = events.iter().filter(|&&e| e == event_type).count();
    if n_events_of_interest == 0 {
        return Err(SurvivalError::NoEvents);
    }

    // ── Censoring KM (G estimator) ────────────────────────────────────────────
    let (g_times, g_vals) = censoring_km(times, events, n);

    // ── Gradient ascent on partial log-likelihood ─────────────────────────────
    let mut beta = vec![0.0_f64; p];
    let mut log_likelihood = 0.0_f64;
    let mut any_update = false;

    for _iter in 0..config.n_iter {
        let (ll, score) = compute_score_and_ll(
            covariates, times, events, n, p, &beta, event_type, &g_times, &g_vals,
        );
        log_likelihood = ll;

        // Check for non-trivial gradient
        let max_score = score.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if !max_score.is_finite() {
            break;
        }
        if max_score < 1.0e-10 {
            any_update = true;
            break;
        }

        // SGD update: beta += lr * score / n
        let scale = config.lr / n as f64;
        for k in 0..p {
            beta[k] += scale * score[k];
        }
        any_update = true;
    }

    // ── Aalen-Johansen CIF ────────────────────────────────────────────────────
    let (cif_times, cif) = aalen_johansen_cif(times, events, n, event_type);

    Ok(FineGraySimpleFit {
        coefficients: beta,
        cif,
        cif_times,
        log_likelihood,
        converged: any_update,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_competing_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<u8>) {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(seed);
        let mut cov = Vec::with_capacity(n);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        for i in 0..n {
            let x = rng.next_normal();
            let t = rng.next_exponential(1.0).max(1.0e-6);
            cov.push(x);
            times.push(t);
            // Cycle: 0=censored, 1=event of interest, 2=competing
            events.push((i % 3) as u8);
        }
        (cov, times, events)
    }

    #[test]
    fn coefficients_len() {
        let (cov, times, events) = make_competing_data(60, 1001);
        let config = FineGrayConfig::default();
        let fit = fine_gray_fit(&cov, &times, &events, 60, &config)
            .expect("fine_gray_fit should succeed");
        assert_eq!(fit.coefficients.len(), config.n_covariates);
    }

    #[test]
    fn cif_nonneg() {
        let (cov, times, events) = make_competing_data(60, 1002);
        let config = FineGrayConfig::default();
        let fit = fine_gray_fit(&cov, &times, &events, 60, &config)
            .expect("fine_gray_fit should succeed");
        for &v in &fit.cif {
            assert!(v >= 0.0, "CIF value {v} < 0");
        }
    }

    #[test]
    fn cif_monotone() {
        let (cov, times, events) = make_competing_data(90, 1003);
        let config = FineGrayConfig::default();
        let fit = fine_gray_fit(&cov, &times, &events, 90, &config)
            .expect("fine_gray_fit should succeed");
        for w in fit.cif.windows(2) {
            assert!(
                w[1] >= w[0] - 1.0e-12,
                "CIF not non-decreasing: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn fit_finite() {
        let (cov, times, events) = make_competing_data(60, 1004);
        let config = FineGrayConfig::default();
        let fit = fine_gray_fit(&cov, &times, &events, 60, &config)
            .expect("fine_gray_fit should succeed");
        for &c in &fit.coefficients {
            assert!(c.is_finite(), "coefficient {c} not finite");
        }
    }

    #[test]
    fn n_covariates_0_error() {
        let config = FineGrayConfig {
            n_covariates: 0,
            ..Default::default()
        };
        let result = fine_gray_fit(&[], &[1.0], &[1u8], 1, &config);
        assert!(
            matches!(result, Err(SurvivalError::InvalidParameter(_))),
            "expected InvalidParameter, got: {result:?}"
        );
    }

    #[test]
    fn all_censored_returns_no_events() {
        let cov = vec![0.5_f64, 1.0, -0.5];
        let times = vec![1.0_f64, 2.0, 3.0];
        let events = vec![0u8, 0, 0];
        let config = FineGrayConfig::default();
        let result = fine_gray_fit(&cov, &times, &events, 3, &config);
        assert!(
            matches!(result, Err(SurvivalError::NoEvents)),
            "expected NoEvents, got: {result:?}"
        );
    }

    #[test]
    fn competing_event_type_2() {
        // Flip: event_type = 2, so events coded as 2 are "of interest"
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(2002);
        let n = 60usize;
        let mut cov = Vec::with_capacity(n);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        for i in 0..n {
            cov.push(rng.next_normal());
            times.push(rng.next_exponential(1.0).max(1.0e-6));
            events.push((i % 3 + 1) as u8); // 1, 2, 3 — but event_type=2
        }
        // Replace 3→0 so we have censored subjects
        for e in events.iter_mut() {
            if *e == 3 {
                *e = 0;
            }
        }
        let config = FineGrayConfig {
            event_type: 2,
            ..Default::default()
        };
        let fit = fine_gray_fit(&cov, &times, &events, n, &config)
            .expect("fine_gray_fit with event_type=2 should succeed");
        assert_eq!(fit.coefficients.len(), 1);
        assert!(fit.cif.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn convergence_stable() {
        let (cov, times, events) = make_competing_data(60, 3003);
        let config = FineGrayConfig {
            n_iter: 300,
            ..Default::default()
        };
        let fit1 =
            fine_gray_fit(&cov, &times, &events, 60, &config).expect("first fit should succeed");
        let fit2 =
            fine_gray_fit(&cov, &times, &events, 60, &config).expect("second fit should succeed");
        // Deterministic — must give identical results
        assert_eq!(fit1.coefficients.len(), fit2.coefficients.len());
        for (a, b) in fit1.coefficients.iter().zip(fit2.coefficients.iter()) {
            assert!((a - b).abs() < 1.0e-15, "coefficients differ: {a} vs {b}");
        }
    }

    #[test]
    fn empty_dataset_error() {
        let config = FineGrayConfig::default();
        let result = fine_gray_fit(&[], &[], &[], 0, &config);
        assert!(
            matches!(result, Err(SurvivalError::EmptyDataset)),
            "expected EmptyDataset, got: {result:?}"
        );
    }

    #[test]
    fn shape_mismatch_error() {
        // covariates length doesn't match n * n_covariates
        let config = FineGrayConfig {
            n_covariates: 2,
            ..Default::default()
        };
        // n=3, p=2 → need 6 elements, give 5
        let result = fine_gray_fit(
            &[1.0, 0.5, -1.0, 0.3, 0.7],
            &[1.0, 2.0, 3.0],
            &[1u8, 0, 2],
            3,
            &config,
        );
        assert!(
            matches!(result, Err(SurvivalError::ShapeMismatch { .. })),
            "expected ShapeMismatch, got: {result:?}"
        );
    }
}

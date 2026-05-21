//! Time-dependent covariates Cox model (Andersen-Gill counting process formulation).
//!
//! Covariates can change over follow-up via a start–stop interval representation.  Each
//! subject contributes one or more rows of the form `(id, t_start, t_stop, event, x)`.
//! The risk set at any event time `t` consists of all intervals `(t_start, t_stop]` that
//! contain `t`, i.e., `t_start < t ≤ t_stop`.
//!
//! # Partial log-likelihood (Andersen-Gill)
//! ```text
//! L(β) = Σ_{r: event_r}  [x_r^T β − log(Σ_{r': at risk at t_stop_r} exp(x_{r'}^T β))]
//! ```
//! where "at risk at t" means `t_start_{r'} < t ≤ t_stop_{r'}`.
//!
//! Ties in event times are handled by the Breslow approximation.
//!
//! # Baseline hazard (Breslow)
//! ```text
//! Λ₀(t) = Σ_{t_i ≤ t, event}  1 / (Σ_{r': at risk at t_i} exp(x_{r'}^T β))
//! ```

use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::inverse::gauss_jordan_inverse;
use crate::linalg::solve::cholesky_solve;

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// A single interval record for the time-dependent Cox model.
#[derive(Debug, Clone)]
pub struct TimeDepRecord {
    /// Subject identifier.
    pub id: usize,
    /// Left endpoint of the interval (exclusive): subject enters risk at `t_start`.
    pub t_start: f64,
    /// Right endpoint of the interval (inclusive): subject exits risk at `t_stop`.
    pub t_stop: f64,
    /// Whether an event occurred at `t_stop`.
    pub event: bool,
    /// Covariate values active during `(t_start, t_stop]`.
    pub x: Vec<f64>,
}

/// Configuration for [`time_dep_cox_fit`].
#[derive(Debug, Clone, Copy)]
pub struct TimeDepCoxConfig {
    /// Maximum Newton-Raphson iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the max absolute score component.
    pub tol: f64,
}

impl Default for TimeDepCoxConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1.0e-6,
        }
    }
}

/// Fitted time-dependent Cox model.
#[derive(Debug, Clone)]
pub struct TimeDepCoxFit {
    /// Estimated regression coefficients β (length p).
    pub coef: Vec<f64>,
    /// Standard errors of β.
    pub se: Vec<f64>,
    /// Wald z-scores β / se.
    pub z_scores: Vec<f64>,
    /// Final partial log-likelihood.
    pub log_likelihood: f64,
    /// Number of unique subjects in the dataset.
    pub n_subjects: usize,
    /// Total number of interval records.
    pub n_intervals: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot(x: &[f64], beta: &[f64]) -> f64 {
    x.iter().zip(beta.iter()).map(|(xi, bi)| xi * bi).sum()
}

/// Validate all records and return the number of unique subjects and
/// the sorted unique event times.
fn validate_records(records: &[TimeDepRecord], p: usize) -> SurvivalResult<(usize, Vec<f64>)> {
    if records.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    // Validate each record.
    for (idx, r) in records.iter().enumerate() {
        if r.t_start >= r.t_stop {
            return Err(SurvivalError::InvalidParameter(format!(
                "record {idx}: t_start ({}) >= t_stop ({})",
                r.t_start, r.t_stop
            )));
        }
        if r.x.len() != p {
            return Err(SurvivalError::DimensionMismatch { a: p, b: r.x.len() });
        }
    }
    // Unique subjects.
    let mut ids: Vec<usize> = records.iter().map(|r| r.id).collect();
    ids.sort_unstable();
    ids.dedup();
    let n_subjects = ids.len();

    // Collect sorted unique event times.
    let mut event_times: Vec<f64> = records
        .iter()
        .filter(|r| r.event)
        .map(|r| r.t_stop)
        .collect();
    if event_times.is_empty() {
        return Err(SurvivalError::NoEvents);
    }
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
    Ok((n_subjects, event_times))
}

/// Compute (log-likelihood, score, Fisher information) for the counting-process
/// partial likelihood at `beta`.
fn time_dep_loglik(
    records: &[TimeDepRecord],
    p: usize,
    beta: &[f64],
    event_times: &[f64],
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
    // Precompute exp(x^T β) for every record.
    let exp_xb: Vec<f64> = records.iter().map(|r| dot(&r.x, beta).exp()).collect();

    let mut loglik = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];

    for &t in event_times {
        // Build risk set: all records r' with t_start < t ≤ t_stop.
        let mut s0 = 0.0_f64;
        let mut s1 = vec![0.0_f64; p];
        let mut s2 = vec![0.0_f64; p * p];

        for (ri, r) in records.iter().enumerate() {
            if r.t_start < t && t <= r.t_stop {
                let w = exp_xb[ri];
                s0 += w;
                for a in 0..p {
                    s1[a] += w * r.x[a];
                    for b in 0..p {
                        s2[a * p + b] += w * r.x[a] * r.x[b];
                    }
                }
            }
        }

        if s0 <= 0.0 {
            return Err(SurvivalError::NumericalInstability(
                "non-positive risk-set sum at event time in time-dependent Cox".to_string(),
            ));
        }

        // Sum over all events at time t (Breslow tie handling).
        let mut d_count = 0.0_f64;
        let mut etabsum = 0.0_f64;
        let mut x_events = vec![0.0_f64; p];

        for r in records
            .iter()
            .filter(|r| r.event && (r.t_stop - t).abs() < f64::EPSILON)
        {
            d_count += 1.0;
            etabsum += dot(&r.x, beta);
            for (xe, &rx) in x_events.iter_mut().zip(r.x.iter()) {
                *xe += rx;
            }
        }

        if d_count > 0.0 {
            loglik += etabsum - d_count * s0.ln();
            let x_bar: Vec<f64> = s1.iter().map(|si| si / s0).collect();
            for a in 0..p {
                score[a] += x_events[a] - d_count * x_bar[a];
            }
            for a in 0..p {
                for b in 0..p {
                    let cov = s2[a * p + b] / s0 - x_bar[a] * x_bar[b];
                    info[a * p + b] += d_count * cov;
                }
            }
        }
    }

    Ok((loglik, score, info))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fit a time-dependent Cox model using Newton-Raphson on the Andersen-Gill partial likelihood.
///
/// # Parameters
/// - `records`: one or more interval records per subject.
/// - `p`: number of covariates (must equal `record.x.len()` for every record).
/// - `cfg`: algorithm configuration.
pub fn time_dep_cox_fit(
    records: &[TimeDepRecord],
    p: usize,
    cfg: &TimeDepCoxConfig,
) -> SurvivalResult<TimeDepCoxFit> {
    let (n_subjects, event_times) = validate_records(records, p)?;
    let n_intervals = records.len();

    let mut beta = vec![0.0_f64; p];
    let (mut ll, mut score, mut info) = time_dep_loglik(records, p, &beta, &event_times)?;

    let mut converged = false;
    for _iter in 0..cfg.max_iter {
        let max_score = score.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        if max_score < cfg.tol {
            converged = true;
            break;
        }
        if p == 0 {
            converged = true;
            break;
        }

        let delta = match cholesky_solve(&info, &score, p) {
            Ok(d) => d,
            Err(_) => {
                let mut info_ridge = info.clone();
                for d in 0..p {
                    info_ridge[d * p + d] += 1.0e-4;
                }
                match cholesky_solve(&info_ridge, &score, p) {
                    Ok(d) => d,
                    Err(_) => break,
                }
            }
        };

        // Armijo line search.
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..40 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + step * d)
                .collect();
            if let Ok((ll_new, sc_new, info_new)) =
                time_dep_loglik(records, p, &trial, &event_times)
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
    if !converged {
        let max_score = score.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        if max_score < cfg.tol {
            converged = true;
        }
    }
    let _ = converged;

    // ---------- variance-covariance ----------
    let variance = if p > 0 {
        gauss_jordan_inverse(&info, p).unwrap_or_else(|_| vec![0.0_f64; p * p])
    } else {
        vec![]
    };

    let se: Vec<f64> = (0..p)
        .map(|i| variance[i * p + i].max(0.0).sqrt())
        .collect();
    let z_scores: Vec<f64> = beta
        .iter()
        .zip(se.iter())
        .map(|(b, s)| if *s > 0.0 { b / s } else { 0.0 })
        .collect();

    Ok(TimeDepCoxFit {
        coef: beta,
        se,
        z_scores,
        log_likelihood: ll,
        n_subjects,
        n_intervals,
    })
}

/// Compute the Breslow cumulative baseline hazard from a fitted time-dependent Cox model.
///
/// Returns `Vec<(time, cumulative_hazard)>` sorted by time.
pub fn time_dep_cox_baseline_hazard(
    fit: &TimeDepCoxFit,
    records: &[TimeDepRecord],
    _p: usize,
) -> Vec<(f64, f64)> {
    // Collect sorted unique event times.
    let mut event_times: Vec<f64> = records
        .iter()
        .filter(|r| r.event)
        .map(|r| r.t_stop)
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

    let exp_xb: Vec<f64> = records.iter().map(|r| dot(&r.x, &fit.coef).exp()).collect();

    let mut result = Vec::with_capacity(event_times.len());
    let mut cumulative_h = 0.0_f64;

    for t in event_times {
        // Risk-set denominator.
        let s0: f64 = records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.t_start < t && t <= r.t_stop)
            .map(|(ri, _)| exp_xb[ri])
            .sum();

        // Count events at t.
        let d: f64 = records
            .iter()
            .filter(|r| r.event && (r.t_stop - t).abs() < f64::EPSILON)
            .count() as f64;

        if d > 0.0 && s0 > 0.0 {
            cumulative_h += d / s0;
        }
        result.push((t, cumulative_h));
    }
    result
}

/// Score test statistic at β = 0 (chi-square with `p` d.f.).
pub fn time_dep_cox_score_test(records: &[TimeDepRecord], p: usize) -> SurvivalResult<f64> {
    let (_, event_times) = validate_records(records, p)?;
    if p == 0 {
        return Ok(0.0);
    }
    let beta0 = vec![0.0_f64; p];
    let (_ll, score, info) = time_dep_loglik(records, p, &beta0, &event_times)?;
    let info_inv = gauss_jordan_inverse(&info, p).unwrap_or_else(|_| vec![0.0_f64; p * p]);
    let mut stat = 0.0_f64;
    for a in 0..p {
        for b in 0..p {
            stat += score[a] * info_inv[a * p + b] * score[b];
        }
    }
    Ok(stat)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build simple one-interval-per-subject data (equivalent to standard Cox).
    fn make_simple_records(n: usize, beta_true: f64, seed: u64) -> Vec<TimeDepRecord> {
        let mut rng = LcgRng::new(seed);
        let mut recs = Vec::with_capacity(n);
        for id in 0..n {
            let xi = rng.next_normal();
            let lam = (beta_true * xi).exp();
            let t = rng.next_exponential(lam).max(1.0e-4);
            let ev = rng.next_f64() < 0.75;
            recs.push(TimeDepRecord {
                id,
                t_start: 0.0,
                t_stop: t,
                event: ev,
                x: vec![xi],
            });
        }
        // Guarantee at least one event.
        recs[0].event = true;
        recs
    }

    /// Build multi-interval data where covariate changes at `t_change`.
    fn make_multi_interval_records(n: usize, beta_true: f64, seed: u64) -> Vec<TimeDepRecord> {
        let mut rng = LcgRng::new(seed);
        let mut recs = Vec::new();
        for id in 0..n {
            let x1 = rng.next_normal();
            let x2 = rng.next_normal();
            let t_change = rng.next_range(0.5, 1.5);
            let lam = (beta_true * x2).exp();
            let t_event = t_change + rng.next_exponential(lam).max(1.0e-4);
            let ev = rng.next_f64() < 0.7;
            // Interval 1: (0, t_change], x = x1, no event.
            recs.push(TimeDepRecord {
                id,
                t_start: 0.0,
                t_stop: t_change,
                event: false,
                x: vec![x1],
            });
            // Interval 2: (t_change, t_event], x = x2, possible event.
            recs.push(TimeDepRecord {
                id,
                t_start: t_change,
                t_stop: t_event,
                event: ev,
                x: vec![x2],
            });
        }
        recs[1].event = true; // Guarantee at least one event.
        recs
    }

    #[test]
    fn simple_case_coef_is_finite() {
        let recs = make_simple_records(80, 0.7, 42);
        let cfg = TimeDepCoxConfig::default();
        let fit = time_dep_cox_fit(&recs, 1, &cfg).expect("fit ok");
        assert!(fit.coef[0].is_finite(), "coef must be finite");
    }

    #[test]
    fn simple_case_se_positive() {
        let recs = make_simple_records(80, 0.7, 43);
        let cfg = TimeDepCoxConfig::default();
        let fit = time_dep_cox_fit(&recs, 1, &cfg).expect("fit ok");
        assert!(fit.se[0] > 0.0, "SE must be positive");
    }

    #[test]
    fn log_likelihood_is_negative() {
        let recs = make_simple_records(80, 0.5, 44);
        let cfg = TimeDepCoxConfig::default();
        let fit = time_dep_cox_fit(&recs, 1, &cfg).expect("fit ok");
        assert!(
            fit.log_likelihood < 0.0,
            "log-likelihood of a probability must be negative, got {}",
            fit.log_likelihood
        );
    }

    #[test]
    fn n_intervals_matches_record_count() {
        let recs = make_simple_records(50, 0.5, 45);
        let n_recs = recs.len();
        let cfg = TimeDepCoxConfig::default();
        let fit = time_dep_cox_fit(&recs, 1, &cfg).expect("fit ok");
        assert_eq!(
            fit.n_intervals, n_recs,
            "n_intervals must equal number of records"
        );
    }

    #[test]
    fn n_subjects_counted_correctly() {
        let recs = make_multi_interval_records(30, 0.5, 46);
        // Each subject contributes 2 intervals -> 30 subjects total.
        let cfg = TimeDepCoxConfig::default();
        let fit = time_dep_cox_fit(&recs, 1, &cfg).expect("fit ok");
        assert_eq!(fit.n_subjects, 30, "should be 30 subjects");
    }

    #[test]
    fn multi_interval_coef_finite() {
        let recs = make_multi_interval_records(40, 0.8, 47);
        let cfg = TimeDepCoxConfig::default();
        let fit = time_dep_cox_fit(&recs, 1, &cfg).expect("fit ok");
        assert!(fit.coef[0].is_finite());
    }

    #[test]
    fn baseline_hazard_monotone() {
        let recs = make_simple_records(80, 0.5, 48);
        let cfg = TimeDepCoxConfig::default();
        let fit = time_dep_cox_fit(&recs, 1, &cfg).expect("fit ok");
        let bh = time_dep_cox_baseline_hazard(&fit, &recs, 1);
        for w in bh.windows(2) {
            assert!(
                w[1].1 >= w[0].1 - 1.0e-12,
                "baseline hazard must be monotone non-decreasing"
            );
        }
    }

    #[test]
    fn baseline_hazard_non_negative() {
        let recs = make_multi_interval_records(40, 0.5, 49);
        let cfg = TimeDepCoxConfig::default();
        let fit = time_dep_cox_fit(&recs, 1, &cfg).expect("fit ok");
        let bh = time_dep_cox_baseline_hazard(&fit, &recs, 1);
        for &(_, h) in &bh {
            assert!(h >= 0.0, "cumulative hazard must be non-negative");
        }
    }

    #[test]
    fn score_test_finite_and_nonneg() {
        let recs = make_simple_records(60, 0.5, 50);
        let stat = time_dep_cox_score_test(&recs, 1).expect("score test ok");
        assert!(stat.is_finite(), "score test statistic must be finite");
        assert!(stat >= 0.0, "chi-square statistic must be non-negative");
    }

    #[test]
    fn error_on_empty_records() {
        let result = time_dep_cox_fit(&[], 1, &TimeDepCoxConfig::default());
        assert!(
            matches!(result, Err(SurvivalError::EmptyDataset)),
            "expected EmptyDataset error"
        );
    }

    #[test]
    fn error_on_invalid_interval() {
        let records = vec![TimeDepRecord {
            id: 0,
            t_start: 5.0, // t_start >= t_stop: invalid
            t_stop: 2.0,
            event: true,
            x: vec![1.0],
        }];
        let result = time_dep_cox_fit(&records, 1, &TimeDepCoxConfig::default());
        assert!(
            matches!(result, Err(SurvivalError::InvalidParameter(_))),
            "expected InvalidParameter error"
        );
    }

    #[test]
    fn ties_at_same_event_time_no_panic() {
        // Multiple records with the same t_stop and event=true.
        let mut recs = make_simple_records(40, 0.5, 51);
        // Force a tie by setting 3 records to the same event time.
        let tie_time = 1.0_f64;
        for r in recs.iter_mut().take(3) {
            r.t_stop = tie_time;
            r.t_start = 0.0;
            r.event = true;
        }
        let cfg = TimeDepCoxConfig::default();
        let fit = time_dep_cox_fit(&recs, 1, &cfg).expect("ties should not panic");
        assert!(fit.coef[0].is_finite());
    }

    #[test]
    fn z_scores_computed() {
        let recs = make_simple_records(80, 0.7, 52);
        let cfg = TimeDepCoxConfig::default();
        let fit = time_dep_cox_fit(&recs, 1, &cfg).expect("fit ok");
        let expected_z = fit.coef[0] / fit.se[0];
        assert!((fit.z_scores[0] - expected_z).abs() < 1.0e-10);
    }
}

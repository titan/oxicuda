//! Landmarking model for dynamic survival prediction (Van Houwelingen 2007).
//!
//! At each landmark time `s`, subjects still at risk (`time > s`) are selected,
//! a Cox partial-likelihood model is fit on the window `[s, horizon]`, and the
//! resulting slice can be used to produce a predicted survival probability at
//! `horizon` conditional on surviving to `s`.

use crate::error::{SurvivalError, SurvivalResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for a landmark super-model.
#[derive(Debug, Clone)]
pub struct LandmarkConfig {
    /// Ordered landmark times at which a separate Cox slice is fit.
    pub landmark_times: Vec<f64>,
    /// Prediction horizon (must exceed every landmark time).
    pub horizon: f64,
    /// Maximum Newton-Raphson iterations per slice.
    pub max_iter: usize,
    /// Convergence tolerance (gradient sup-norm).
    pub tol: f64,
}

/// Fitted slice at a single landmark time `s`.
#[derive(Debug, Clone)]
pub struct LandmarkSlice {
    /// The landmark time.
    pub s: f64,
    /// Fitted regression coefficients β̂.
    pub beta: Vec<f64>,
    /// Breslow baseline cumulative hazard pairs `(t, H₀(t))` for `t ≤ horizon`.
    pub baseline_hazard: Vec<(f64, f64)>,
    /// Number of subjects remaining at risk after filtering to `time > s`.
    pub n_subjects: usize,
}

/// Collection of fitted landmark slices.
#[derive(Debug, Clone)]
pub struct LandmarkModel {
    pub config: LandmarkConfig,
    pub slices: Vec<LandmarkSlice>,
    pub n_covariates: usize,
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

/// Fit a landmark model.
///
/// # Arguments
/// * `times`        – observed times (length `n_obs`)
/// * `events`       – event indicator (length `n_obs`)
/// * `covariates`   – row-major covariate matrix, shape `n_obs × n_covariates`
/// * `n_obs`        – number of observations
/// * `n_covariates` – number of covariates
/// * `cfg`          – landmark configuration
pub fn landmark_fit(
    times: &[f64],
    events: &[bool],
    covariates: &[f64],
    n_obs: usize,
    n_covariates: usize,
    cfg: &LandmarkConfig,
) -> SurvivalResult<LandmarkModel> {
    // ---- basic validation ------------------------------------------------
    if times.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if times.len() != n_obs {
        return Err(SurvivalError::DimensionMismatch {
            a: times.len(),
            b: n_obs,
        });
    }
    if events.len() != n_obs {
        return Err(SurvivalError::DimensionMismatch {
            a: events.len(),
            b: n_obs,
        });
    }
    if covariates.len() != n_obs * n_covariates {
        return Err(SurvivalError::DimensionMismatch {
            a: covariates.len(),
            b: n_obs * n_covariates,
        });
    }
    if cfg.landmark_times.is_empty() {
        return Err(SurvivalError::InvalidConfiguration(
            "landmark_times must not be empty".to_string(),
        ));
    }
    if cfg.horizon <= 0.0 || !cfg.horizon.is_finite() {
        return Err(SurvivalError::InvalidParameter(
            "horizon must be a positive finite number".to_string(),
        ));
    }

    // Check all times are non-negative and finite
    for &t in times {
        if !t.is_finite() || t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }

    // Validate that horizon exceeds every landmark time
    for &s in &cfg.landmark_times {
        if !s.is_finite() || s < 0.0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "landmark time {s} is not a valid non-negative finite number"
            )));
        }
        if cfg.horizon <= s {
            return Err(SurvivalError::InvalidConfiguration(format!(
                "horizon ({}) must be strictly greater than landmark time ({})",
                cfg.horizon, s
            )));
        }
    }

    // ---- fit each slice --------------------------------------------------
    let mut slices = Vec::with_capacity(cfg.landmark_times.len());

    for &s in &cfg.landmark_times {
        // Filter: keep subjects with time > s (still at risk at s)
        let mut sub_times: Vec<f64> = Vec::new();
        let mut sub_events: Vec<bool> = Vec::new();
        let mut sub_cov: Vec<f64> = Vec::new();

        for i in 0..n_obs {
            if times[i] > s {
                // Clamp time to horizon for the landmark window
                let t_win = times[i].min(cfg.horizon);
                // If time was already at or beyond horizon, treat as censored at horizon
                let ev_win = if times[i] > cfg.horizon {
                    false
                } else {
                    events[i]
                };
                sub_times.push(t_win);
                sub_events.push(ev_win);
                // Copy covariate row
                let row_start = i * n_covariates;
                sub_cov.extend_from_slice(&covariates[row_start..row_start + n_covariates]);
            }
        }

        let n_sub = sub_times.len();

        if n_sub == 0 {
            return Err(SurvivalError::NoEvents);
        }

        // Check events exist in this slice
        let n_events_slice = sub_events.iter().filter(|&&e| e).count();
        if n_events_slice == 0 {
            // No events in this slice — skip with zero coefficients
            // Baseline hazard is empty; survival prediction will yield 1.0
            slices.push(LandmarkSlice {
                s,
                beta: vec![0.0; n_covariates],
                baseline_hazard: Vec::new(),
                n_subjects: n_sub,
            });
            continue;
        }

        // Fit Cox slice
        let beta = cox_nr_fit(
            &sub_times,
            &sub_events,
            &sub_cov,
            n_sub,
            n_covariates,
            cfg.max_iter,
            cfg.tol,
        )?;

        // Compute Breslow baseline cumulative hazard up to horizon
        let baseline_hazard = breslow_cumulative_hazard(
            &sub_times,
            &sub_events,
            &sub_cov,
            n_sub,
            n_covariates,
            &beta,
            cfg.horizon,
        )?;

        slices.push(LandmarkSlice {
            s,
            beta,
            baseline_hazard,
            n_subjects: n_sub,
        });
    }

    Ok(LandmarkModel {
        config: cfg.clone(),
        slices,
        n_covariates,
    })
}

// ---------------------------------------------------------------------------
// Predict
// ---------------------------------------------------------------------------

/// Predict survival probability at `model.config.horizon` for a subject with
/// covariate vector `z`, using the slice whose landmark time is closest to `s`.
///
/// Returns `S(horizon | s, z) = exp(−H₀(horizon) · exp(z · β̂))`.
pub fn landmark_predict(model: &LandmarkModel, s: f64, z: &[f64]) -> SurvivalResult<f64> {
    if model.slices.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if z.len() != model.n_covariates {
        return Err(SurvivalError::DimensionMismatch {
            a: z.len(),
            b: model.n_covariates,
        });
    }

    // Find closest slice by |slice.s - s|
    let best_slice = model
        .slices
        .iter()
        .min_by(|a, b| {
            let da = (a.s - s).abs();
            let db = (b.s - s).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or(SurvivalError::EmptyDataset)?;

    // Linear predictor
    let eta: f64 = z
        .iter()
        .zip(best_slice.beta.iter())
        .map(|(zi, bi)| zi * bi)
        .sum();

    if !eta.is_finite() {
        return Err(SurvivalError::NumericalInstability(
            "linear predictor is not finite".to_string(),
        ));
    }

    // Cumulative baseline hazard at horizon: last entry where t <= horizon
    let h0_horizon = best_slice
        .baseline_hazard
        .iter()
        .rev()
        .find(|(t, _)| *t <= model.config.horizon)
        .map(|(_, h)| *h)
        .unwrap_or(0.0);

    // S = exp(-H0 * exp(eta))
    let survival = (-h0_horizon * eta.exp()).exp();

    if !survival.is_finite() {
        return Err(SurvivalError::NumericalInstability(
            "predicted survival is not finite".to_string(),
        ));
    }

    Ok(survival.clamp(0.0, 1.0))
}

// ---------------------------------------------------------------------------
// Self-contained Cox Newton-Raphson fitting
// ---------------------------------------------------------------------------

/// Fit Cox partial likelihood via Newton-Raphson with Armijo line search.
/// Returns the converged β vector.
fn cox_nr_fit(
    times: &[f64],
    events: &[bool],
    covariates: &[f64],
    n: usize,
    p: usize,
    max_iter: usize,
    tol: f64,
) -> SurvivalResult<Vec<f64>> {
    if p == 0 {
        return Ok(Vec::new());
    }

    let mut beta = vec![0.0_f64; p];

    for iter in 0..max_iter {
        let (ll, score, info) = cox_pll(times, events, covariates, n, p, &beta)?;

        // Solve info * delta = score  (Newton direction)
        let delta = solve_linear_system(&info, &score, p)?;

        // Check convergence: sup-norm of score
        let score_norm = score.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if score_norm < tol {
            return Ok(beta);
        }

        // Armijo line search
        let step = armijo_step(times, events, covariates, n, p, &beta, &delta, ll, &score)?;

        // Update beta
        for j in 0..p {
            beta[j] += step * delta[j];
        }

        let _ = iter; // silence unused warning inside the loop body
    }

    // Check final score norm
    let (_, score_final, _) = cox_pll(times, events, covariates, n, p, &beta)?;
    let score_norm = score_final.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if score_norm < tol {
        Ok(beta)
    } else {
        Err(SurvivalError::NotConverged { iter: max_iter })
    }
}

/// Compute Cox Breslow partial log-likelihood, score, and observed information.
///
/// Returns `(log_lik, score: Vec<f64> length p, info: Vec<f64> length p*p row-major)`.
fn cox_pll(
    times: &[f64],
    events: &[bool],
    covariates: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
    // Pre-compute exp(beta · x_i) for all i
    let mut exp_xb = vec![0.0_f64; n];
    for i in 0..n {
        let xb: f64 = (0..p).map(|j| beta[j] * covariates[i * p + j]).sum();
        if !xb.is_finite() {
            return Err(SurvivalError::NumericalInstability(
                "overflow in linear predictor during partial likelihood".to_string(),
            ));
        }
        exp_xb[i] = xb.exp();
    }

    // Sort indices by ascending time
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut log_lik = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];

    // Collect unique event times
    let mut event_times: Vec<f64> = events
        .iter()
        .enumerate()
        .filter(|&(_, ev)| *ev)
        .map(|(i, _)| times[i])
        .collect();
    event_times.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    for &t_ev in &event_times {
        // Risk set: all subjects with time >= t_ev
        // Event set: subjects with time == t_ev and event == true
        let mut denom = 0.0_f64;
        let mut s1 = vec![0.0_f64; p]; // sum exp(xb) * x_j
        let mut s2 = vec![0.0_f64; p * p]; // sum exp(xb) * x_j * x_k

        for &i in &order {
            if times[i] >= t_ev {
                let exi = exp_xb[i];
                denom += exi;
                for j in 0..p {
                    s1[j] += exi * covariates[i * p + j];
                    for k in 0..p {
                        s2[j * p + k] += exi * covariates[i * p + j] * covariates[i * p + k];
                    }
                }
            }
        }

        if denom <= 0.0 || !denom.is_finite() {
            return Err(SurvivalError::NumericalInstability(
                "zero or non-finite risk set denominator".to_string(),
            ));
        }

        // Iterate over all events at t_ev (Breslow tie handling: sum log(exp(xb_i)) - d*log(denom))
        for &i in order.iter().filter(|&&i| times[i] == t_ev && events[i]) {
            // log-likelihood contribution
            let xb_i: f64 = (0..p).map(|j| beta[j] * covariates[i * p + j]).sum();
            log_lik += xb_i - denom.ln();

            // Score: x_i - s1/denom
            for j in 0..p {
                score[j] += covariates[i * p + j] - s1[j] / denom;
            }
        }

        // Count events at t_ev for information matrix (d events)
        let d_ev = order
            .iter()
            .filter(|&&i| times[i] == t_ev && events[i])
            .count() as f64;

        // Information matrix: d * (s2/denom - (s1/denom) ⊗ (s1/denom))
        for j in 0..p {
            for k in 0..p {
                let a2 = s2[j * p + k] / denom;
                let a1j = s1[j] / denom;
                let a1k = s1[k] / denom;
                info[j * p + k] += d_ev * (a2 - a1j * a1k);
            }
        }
    }

    Ok((log_lik, score, info))
}

/// Armijo backtracking line search.
/// Returns a step length satisfying the sufficient-decrease condition.
fn armijo_step(
    times: &[f64],
    events: &[bool],
    covariates: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
    delta: &[f64],
    ll0: f64,
    score: &[f64],
) -> SurvivalResult<f64> {
    let c1 = 1e-4_f64;
    let rho = 0.5_f64;
    let mut step = 1.0_f64;

    // Directional derivative  score · delta
    let dir_deriv: f64 = score.iter().zip(delta.iter()).map(|(s, d)| s * d).sum();

    for _ in 0..40 {
        let beta_new: Vec<f64> = beta
            .iter()
            .zip(delta.iter())
            .map(|(b, d)| b + step * d)
            .collect();
        if let Ok((ll_new, _, _)) = cox_pll(times, events, covariates, n, p, &beta_new) {
            if ll_new >= ll0 + c1 * step * dir_deriv {
                return Ok(step);
            }
        }
        step *= rho;
    }

    // Return smallest tried step; convergence check will catch failures
    Ok(step)
}

// ---------------------------------------------------------------------------
// Self-contained Breslow baseline cumulative hazard
// ---------------------------------------------------------------------------

/// Compute the Breslow estimator of the cumulative baseline hazard `H₀(t)`
/// for `t ≤ horizon`, at each distinct event time.
fn breslow_cumulative_hazard(
    times: &[f64],
    events: &[bool],
    covariates: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
    horizon: f64,
) -> SurvivalResult<Vec<(f64, f64)>> {
    // Pre-compute exp(beta · x_i)
    let mut exp_xb = vec![0.0_f64; n];
    for i in 0..n {
        let xb: f64 = (0..p).map(|j| beta[j] * covariates[i * p + j]).sum();
        exp_xb[i] = xb.exp();
    }

    // Collect distinct event times ≤ horizon, sorted
    let mut event_times: Vec<f64> = events
        .iter()
        .enumerate()
        .filter(|&(i, ev)| *ev && times[i] <= horizon)
        .map(|(i, _)| times[i])
        .collect();
    event_times.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    let mut result = Vec::with_capacity(event_times.len());
    let mut cumulative = 0.0_f64;

    for t_ev in event_times {
        // Risk set denominator
        let denom: f64 = (0..n)
            .filter(|&i| times[i] >= t_ev)
            .map(|i| exp_xb[i])
            .sum();

        if denom <= 0.0 || !denom.is_finite() {
            continue; // skip degenerate times
        }

        // Number of events at t_ev (Breslow: d / denom)
        let d_ev = (0..n).filter(|&i| times[i] == t_ev && events[i]).count() as f64;

        cumulative += d_ev / denom;
        result.push((t_ev, cumulative));
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Linear system solver (Gaussian elimination with ridge fallback)
// ---------------------------------------------------------------------------

/// Solve `A * x = b` where `A` is `p × p` row-major.
/// Adds a ridge `1e-8` to the diagonal before factorisation to handle
/// near-singular matrices.
fn solve_linear_system(a: &[f64], b: &[f64], p: usize) -> SurvivalResult<Vec<f64>> {
    if p == 0 {
        return Ok(Vec::new());
    }

    // Build augmented matrix [A | b] with ridge
    let mut aug = vec![0.0_f64; p * (p + 1)];
    for i in 0..p {
        for j in 0..p {
            aug[i * (p + 1) + j] = a[i * p + j];
        }
        aug[i * (p + 1) + i] += 1e-8; // ridge
        aug[i * (p + 1) + p] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..p {
        // Find pivot
        let pivot_row = (col..p)
            .max_by(|&r1, &r2| {
                aug[r1 * (p + 1) + col]
                    .abs()
                    .partial_cmp(&aug[r2 * (p + 1) + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(SurvivalError::SingularMatrix)?;

        if aug[pivot_row * (p + 1) + col].abs() < 1e-15 {
            return Err(SurvivalError::SingularMatrix);
        }

        // Swap rows
        if pivot_row != col {
            for j in 0..=p {
                aug.swap(col * (p + 1) + j, pivot_row * (p + 1) + j);
            }
        }

        let pivot_val = aug[col * (p + 1) + col];

        // Eliminate below
        for row in (col + 1)..p {
            let factor = aug[row * (p + 1) + col] / pivot_val;
            for j in col..=p {
                let sub = factor * aug[col * (p + 1) + j];
                aug[row * (p + 1) + j] -= sub;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut sum = aug[i * (p + 1) + p];
        for j in (i + 1)..p {
            sum -= aug[i * (p + 1) + j] * x[j];
        }
        let diag = aug[i * (p + 1) + i];
        if diag.abs() < 1e-15 {
            return Err(SurvivalError::SingularMatrix);
        }
        x[i] = sum / diag;
    }

    for &xi in &x {
        if !xi.is_finite() {
            return Err(SurvivalError::NumericalInstability(
                "non-finite value in linear system solution".to_string(),
            ));
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple, well-separated dataset: 10 subjects, 1 covariate,
    /// half experience events, times spread over [1, 10].
    fn simple_dataset() -> (Vec<f64>, Vec<bool>, Vec<f64>, usize, usize) {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let events = vec![
            true, false, true, false, true, false, true, false, true, false,
        ];
        let covariates = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let n_obs = 10;
        let n_cov = 1;
        (times, events, covariates, n_obs, n_cov)
    }

    fn simple_config() -> LandmarkConfig {
        LandmarkConfig {
            landmark_times: vec![0.5, 2.5],
            horizon: 11.0,
            max_iter: 100,
            tol: 1e-6,
        }
    }

    // 1 ── basic fit does not panic / return an error
    #[test]
    fn fit_doesnt_crash() {
        let (times, events, covariates, n_obs, n_cov) = simple_dataset();
        let cfg = simple_config();
        let result = landmark_fit(&times, &events, &covariates, n_obs, n_cov, &cfg);
        assert!(result.is_ok(), "landmark_fit failed: {:?}", result.err());
    }

    // 2 ── number of slices matches number of landmark times
    #[test]
    fn slices_count_matches_landmark_times() {
        let (times, events, covariates, n_obs, n_cov) = simple_dataset();
        let cfg = simple_config();
        let model = landmark_fit(&times, &events, &covariates, n_obs, n_cov, &cfg)
            .expect("landmark_fit should succeed");
        assert_eq!(model.slices.len(), cfg.landmark_times.len());
    }

    // 3 ── n_subjects is non-increasing as landmark times increase
    #[test]
    fn n_subjects_decreasing() {
        let (times, events, covariates, n_obs, n_cov) = simple_dataset();
        let cfg = LandmarkConfig {
            landmark_times: vec![0.5, 3.0, 6.0],
            horizon: 12.0,
            max_iter: 100,
            tol: 1e-6,
        };
        let model = landmark_fit(&times, &events, &covariates, n_obs, n_cov, &cfg)
            .expect("landmark_fit should succeed");
        let counts: Vec<usize> = model.slices.iter().map(|sl| sl.n_subjects).collect();
        for w in counts.windows(2) {
            assert!(
                w[0] >= w[1],
                "n_subjects should be non-increasing: {} < {}",
                w[0],
                w[1]
            );
        }
    }

    // 4 ── predicted survival is in [0, 1]
    #[test]
    fn predict_returns_in_0_1() {
        let (times, events, covariates, n_obs, n_cov) = simple_dataset();
        let cfg = simple_config();
        let model = landmark_fit(&times, &events, &covariates, n_obs, n_cov, &cfg)
            .expect("landmark_fit should succeed");
        let z = vec![0.5_f64];
        let surv = landmark_predict(&model, 0.5, &z).expect("landmark_predict should succeed");
        assert!((0.0..=1.0).contains(&surv), "survival {surv} not in [0, 1]");
    }

    // 5 ── horizon <= landmark time returns an error
    #[test]
    fn horizon_too_small_errors() {
        let (times, events, covariates, n_obs, n_cov) = simple_dataset();
        // horizon = 2.0, landmark_times contains 3.0 which is > horizon
        let cfg = LandmarkConfig {
            landmark_times: vec![1.0, 3.0],
            horizon: 2.0,
            max_iter: 50,
            tol: 1e-6,
        };
        let result = landmark_fit(&times, &events, &covariates, n_obs, n_cov, &cfg);
        assert!(
            result.is_err(),
            "expected error when horizon <= a landmark time"
        );
        match result {
            Err(SurvivalError::InvalidConfiguration(_)) => {}
            other => panic!("expected InvalidConfiguration, got {other:?}"),
        }
    }

    // 6 ── negative times in data return NegativeTime error
    #[test]
    fn negative_time_errors() {
        let times = vec![-1.0, 2.0, 3.0];
        let events = vec![true, false, true];
        let covariates = vec![0.1, 0.2, 0.3];
        let cfg = LandmarkConfig {
            landmark_times: vec![0.5],
            horizon: 5.0,
            max_iter: 50,
            tol: 1e-6,
        };
        let result = landmark_fit(&times, &events, &covariates, 3, 1, &cfg);
        assert!(result.is_err());
        match result {
            Err(SurvivalError::NegativeTime(_)) => {}
            other => panic!("expected NegativeTime, got {other:?}"),
        }
    }

    // 7 ── landmark slice where all remaining subjects are censored is handled gracefully
    #[test]
    fn no_events_at_landmark_handled() {
        // All events happen at t=1.0; landmark at s=1.5 means no events in [1.5, horizon]
        let times = vec![1.0, 1.0, 1.0, 5.0, 6.0, 7.0];
        let events = vec![true, true, true, false, false, false];
        let covariates = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let cfg = LandmarkConfig {
            landmark_times: vec![1.5],
            horizon: 10.0,
            max_iter: 50,
            tol: 1e-6,
        };
        // Should succeed (gracefully handles no-event slice)
        let result = landmark_fit(&times, &events, &covariates, 6, 1, &cfg);
        assert!(
            result.is_ok(),
            "should handle all-censored slice gracefully: {:?}",
            result.err()
        );
        let model = result.expect("result should be present");
        // Predict should still return a valid probability
        let z = vec![0.3_f64];
        let surv = landmark_predict(&model, 1.5, &z).expect("landmark_predict should succeed");
        assert!((0.0..=1.0).contains(&surv), "survival {surv} not in [0,1]");
    }

    // 8 ── predict with each landmark slice returns a valid result
    #[test]
    fn predict_with_each_slice() {
        let (times, events, covariates, n_obs, n_cov) = simple_dataset();
        let cfg = LandmarkConfig {
            landmark_times: vec![0.5, 2.5, 5.5],
            horizon: 12.0,
            max_iter: 100,
            tol: 1e-6,
        };
        let model = landmark_fit(&times, &events, &covariates, n_obs, n_cov, &cfg)
            .expect("landmark_fit should succeed");
        let z = vec![0.5_f64];

        for sl in &model.slices {
            let surv = landmark_predict(&model, sl.s, &z);
            assert!(
                surv.is_ok(),
                "predict failed for slice s={}: {:?}",
                sl.s,
                surv.err()
            );
            let p = surv.expect("surv should be present");
            assert!(
                (0.0..=1.0).contains(&p),
                "survival {} not in [0,1] for slice s={}",
                p,
                sl.s
            );
        }
    }
}

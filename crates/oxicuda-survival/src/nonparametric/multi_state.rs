//! Aalen-Johansen estimator for multi-state survival models.
//!
//! The Aalen-Johansen (AJ) estimator (Aalen & Johansen 1978, Scandinavian J Statistics)
//! generalises the Kaplan-Meier estimator to settings with multiple states and multiple
//! possible transitions between them.
//!
//! # Setting
//! - States: `0, 1, …, S-1` (state `initial_state` is the starting state; absorbing
//!   states are those with no outgoing transitions in the data).
//! - At each event time `tₖ`, some subjects transition between states.
//! - For transition `h → j` (h ≠ j): `dN_hj(tₖ)` = number of `h→j` transitions at `tₖ`;
//!   `n_h(tₖ)` = number at risk in state `h` just before `tₖ`.
//!
//! # Transition Intensity Matrix
//! At each event time `tₖ`, the incremental intensity matrix `dΛ(tₖ)` is S×S:
//! - Off-diagonal: `dΛ_hj(tₖ) = dN_hj(tₖ) / n_h(tₖ)` for h ≠ j.
//! - Diagonal: `dΛ_hh(tₖ) = -Σ_{j≠h} dΛ_hj(tₖ)` (sum of outgoing rates negated).
//!
//! # Aalen-Johansen Product-Integral
//! `P(s, t) = Π_{s < tₖ ≤ t} [I + dΛ(tₖ)]`
//!
//! `P_hj(s, t)` = probability of being in state `j` at time `t`, given in state `h` at time `s`.
//! Occupation probability for initial state `h₀`: `π_j(t) = P_{h₀,j}(0, t)`.

use crate::error::{SurvivalError, SurvivalResult};

// ─── Public data structures ──────────────────────────────────────────────────

/// A single observed transition or censoring event in a multi-state model.
///
/// Uses counting-process convention: the subject is at risk in `from_state`
/// over the half-open interval `(start_time, end_time]`.
#[derive(Debug, Clone)]
pub struct MultiStateObs {
    /// Entry time (often 0.0 for all subjects).
    pub start_time: f64,
    /// Event or censoring time (must be strictly greater than `start_time`).
    pub end_time: f64,
    /// State occupied at `start_time`.
    pub from_state: usize,
    /// Destination state.  `None` = right-censored; `Some(j)` = transition to state `j`.
    pub to_state: Option<usize>,
}

/// Configuration for the Aalen-Johansen multi-state estimator.
#[derive(Debug, Clone)]
pub struct MultiStateConfig {
    /// Total number of states S (must be ≥ 2).
    pub n_states: usize,
    /// State occupied at time 0 by the reference cohort.
    /// Used to select the row of the transition matrix for occupation probabilities.
    pub initial_state: usize,
}

impl Default for MultiStateConfig {
    fn default() -> Self {
        Self {
            n_states: 2,
            initial_state: 0,
        }
    }
}

/// Output of the Aalen-Johansen multi-state estimator.
///
/// All matrices are stored in **row-major** order with dimension `n_states × n_states`.
#[derive(Debug, Clone)]
pub struct MultiStateFit {
    /// Unique event times at which at least one transition was observed, sorted ascending.
    pub event_times: Vec<f64>,
    /// State transition probability matrices `P(0, tₖ)` at each event time.
    ///
    /// `transition_probs[k]` is a flattened `n_states × n_states` matrix in row-major
    /// order, where entry `[h * n_states + j]` gives `P_{h→j}(0, tₖ)`.
    pub transition_probs: Vec<Vec<f64>>,
    /// State occupation probability vector at each event time.
    ///
    /// `occupation_probs[k][j]` = probability of being in state `j` at `event_times[k]`
    /// for a subject who started in `config.initial_state` at time 0.
    /// Equals row `initial_state` of the corresponding transition matrix.
    pub occupation_probs: Vec<Vec<f64>>,
    /// Number of states S.
    pub n_states: usize,
    /// Number of observations supplied.
    pub n_obs: usize,
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Create the n×n identity matrix in row-major order.
fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0_f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Multiply two n×n row-major matrices: result = a × b.
fn mat_mul_sq(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            if a_ik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

/// Count subjects at risk in `state` at time `t`.
///
/// Counting-process convention: a subject contributes to the risk set at `t` when
/// `start_time ≤ t AND end_time ≥ t AND from_state == state`.
/// Subjects who transition *out* of `state` at exactly `t` are still included
/// in the risk set (they are at risk just before `t`).
fn at_risk_count(obs: &[MultiStateObs], state: usize, t: f64) -> usize {
    obs.iter()
        .filter(|o| o.from_state == state && o.start_time <= t && o.end_time >= t)
        .count()
}

/// Count observed `from → to` transitions at exactly time `t`.
fn transition_count(obs: &[MultiStateObs], from: usize, to: usize, t: f64) -> usize {
    obs.iter()
        .filter(|o| {
            o.from_state == from
                && o.to_state == Some(to)
                && (o.end_time - t).abs() < f64::EPSILON * t.abs().max(1.0)
        })
        .count()
}

// ─── Main fitting function ────────────────────────────────────────────────────

/// Fit the Aalen-Johansen multi-state estimator.
///
/// # Algorithm
/// 1. Validate inputs.
/// 2. Collect unique event times (times where at least one transition occurred).
/// 3. For each event time `tₖ`:
///    a. Count at-risk subjects per state (`n_h`).
///    b. Count `h→j` transitions (`dN_hj`).
///    c. Build incremental intensity matrix `dΛ`.
///    d. Form `I + dΛ` and left-multiply the running product: `P ← P · (I + dΛ)`.
/// 4. Record transition and occupation probabilities at each step.
///
/// # Returns
/// A [`MultiStateFit`] containing the full transition probability process.
/// If no transitions were observed (all censored), the returned fit has empty
/// `event_times`, `transition_probs`, and `occupation_probs` vectors.
///
/// # Errors
/// - [`SurvivalError::InvalidParameter`] if `n_states < 2`, `initial_state >= n_states`,
///   any state index exceeds `n_states - 1`, or any `end_time ≤ start_time`.
/// - [`SurvivalError::EmptyDataset`] if `observations` is empty.
pub fn fit_multi_state(
    observations: &[MultiStateObs],
    config: &MultiStateConfig,
) -> SurvivalResult<MultiStateFit> {
    // ── (1) Validate configuration ────────────────────────────────────────────
    if config.n_states < 2 {
        return Err(SurvivalError::InvalidParameter(
            "n_states must be ≥ 2 for a multi-state model".to_string(),
        ));
    }
    if config.initial_state >= config.n_states {
        return Err(SurvivalError::InvalidParameter(format!(
            "initial_state {} is out of range for n_states {}",
            config.initial_state, config.n_states
        )));
    }
    if observations.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }

    let s = config.n_states;
    let n_obs = observations.len();

    // ── (2) Validate each observation ────────────────────────────────────────
    for (idx, obs) in observations.iter().enumerate() {
        if obs.end_time <= obs.start_time {
            return Err(SurvivalError::InvalidParameter(format!(
                "observation {idx}: end_time ({}) must be strictly greater than start_time ({})",
                obs.end_time, obs.start_time
            )));
        }
        if obs.from_state >= s {
            return Err(SurvivalError::InvalidParameter(format!(
                "observation {idx}: from_state {} >= n_states {}",
                obs.from_state, s
            )));
        }
        if let Some(to) = obs.to_state {
            if to >= s {
                return Err(SurvivalError::InvalidParameter(format!(
                    "observation {idx}: to_state {to} >= n_states {s}",
                )));
            }
        }
    }

    // ── (3) Collect and sort unique event times ───────────────────────────────
    let mut event_times: Vec<f64> = observations
        .iter()
        .filter_map(|o| {
            if o.to_state.is_some() {
                Some(o.end_time)
            } else {
                None
            }
        })
        .collect();

    // Deduplicate — collect into a sorted, deduplicated vec using ordered comparison.
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * b.abs().max(1.0));

    // If no events at all, return an empty fit (not an error per spec).
    if event_times.is_empty() {
        return Ok(MultiStateFit {
            event_times: vec![],
            transition_probs: vec![],
            occupation_probs: vec![],
            n_states: s,
            n_obs,
        });
    }

    // ── (4) Product-integral ──────────────────────────────────────────────────
    // Running product P(0, t) starts as identity.
    let mut p_current = identity(s);

    let mut transition_probs: Vec<Vec<f64>> = Vec::with_capacity(event_times.len());
    let mut occupation_probs: Vec<Vec<f64>> = Vec::with_capacity(event_times.len());

    for &t_k in &event_times {
        // 4a. At-risk counts per state.
        let at_risk: Vec<usize> = (0..s)
            .map(|h| at_risk_count(observations, h, t_k))
            .collect();

        // 4b & 4c. Build incremental intensity matrix dΛ (S×S, row-major).
        let mut d_lambda = vec![0.0_f64; s * s];

        for h in 0..s {
            let n_h = at_risk[h] as f64;
            if n_h <= 0.0 {
                continue;
            }
            let mut outgoing_rate = 0.0_f64;
            for j in 0..s {
                if j == h {
                    continue;
                }
                let d_hj = transition_count(observations, h, j, t_k) as f64;
                let rate = d_hj / n_h;
                d_lambda[h * s + j] = rate;
                outgoing_rate += rate;
            }
            // Diagonal entry: negative sum of outgoing rates.
            d_lambda[h * s + h] = -outgoing_rate;
        }

        // 4d. Form increment matrix I + dΛ.
        let mut increment = identity(s);
        for idx in 0..(s * s) {
            increment[idx] += d_lambda[idx];
        }

        // 4e. Update running product: P_new = P_old · (I + dΛ).
        p_current = mat_mul_sq(&p_current, &increment, s);

        // Clamp small negatives to zero to prevent numerical drift.
        for v in &mut p_current {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        // 4f. Renormalise rows to ensure row sums remain exactly 1.
        for h in 0..s {
            let row_sum: f64 = (0..s).map(|j| p_current[h * s + j]).sum();
            if row_sum > 0.0 && (row_sum - 1.0).abs() > 1.0e-12 {
                for j in 0..s {
                    p_current[h * s + j] /= row_sum;
                }
            }
        }

        transition_probs.push(p_current.clone());

        // Occupation probability = row `initial_state` of P.
        let h0 = config.initial_state;
        let occ: Vec<f64> = (0..s).map(|j| p_current[h0 * s + j]).collect();
        occupation_probs.push(occ);
    }

    Ok(MultiStateFit {
        event_times,
        transition_probs,
        occupation_probs,
        n_states: s,
        n_obs,
    })
}

// ─── Prediction ──────────────────────────────────────────────────────────────

/// Return the transition probability matrix P(0, t) for arbitrary query time `t`.
///
/// Uses the step-function convention: returns the matrix at the largest event time `≤ t`.
/// - If `t < event_times[0]`, returns the S×S identity matrix.
/// - If `t ≥ event_times.last()`, returns the matrix at the last event time.
/// - If `event_times` is empty, returns the identity matrix.
#[must_use]
pub fn predict_transition_probs(fit: &MultiStateFit, t: f64) -> Vec<f64> {
    if fit.event_times.is_empty() {
        return identity(fit.n_states);
    }
    // Binary search: find rightmost event time ≤ t.
    match fit
        .event_times
        .binary_search_by(|et| et.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal))
    {
        Ok(idx) => fit.transition_probs[idx].clone(),
        Err(0) => identity(fit.n_states),
        Err(ins) => fit.transition_probs[ins - 1].clone(),
    }
}

/// Return the state occupation probability vector at arbitrary query time `t`.
///
/// `initial_state` selects which row of P(0, t) to extract as the occupation vector.
///
/// # Errors
/// Returns [`SurvivalError::InvalidParameter`] if `initial_state >= fit.n_states`.
pub fn predict_occupation(
    fit: &MultiStateFit,
    t: f64,
    initial_state: usize,
) -> SurvivalResult<Vec<f64>> {
    let s = fit.n_states;
    if initial_state >= s {
        return Err(SurvivalError::InvalidParameter(format!(
            "initial_state {initial_state} >= n_states {s}"
        )));
    }
    let p = predict_transition_probs(fit, t);
    let occ: Vec<f64> = (0..s).map(|j| p[initial_state * s + j]).collect();
    Ok(occ)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple illness-death model with 3 states.
    // States: 0=healthy, 1=ill, 2=dead
    // Obs: [
    //   (0→1 at t=1.0), (0→2 at t=2.0), (0→2 at t=3.0),
    //   (1→2 at t=2.5), (0→None censored at t=5.0)
    // ]
    fn illness_death_obs() -> Vec<MultiStateObs> {
        vec![
            MultiStateObs {
                start_time: 0.0,
                end_time: 1.0,
                from_state: 0,
                to_state: Some(1),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 2.0,
                from_state: 0,
                to_state: Some(2),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 3.0,
                from_state: 0,
                to_state: Some(2),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 2.5,
                from_state: 1,
                to_state: Some(2),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 5.0,
                from_state: 0,
                to_state: None,
            },
        ]
    }

    fn illness_death_config() -> MultiStateConfig {
        MultiStateConfig {
            n_states: 3,
            initial_state: 0,
        }
    }

    // ── Test 1: illness-death model basic check ───────────────────────────────

    #[test]
    fn illness_death_model_basic() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");

        // Event times should be sorted ascending.
        let et = &fit.event_times;
        for i in 1..et.len() {
            assert!(et[i] > et[i - 1], "event_times not sorted at index {i}");
        }

        // At each event time, occupation probabilities must sum to ~1.
        for (k, occ) in fit.occupation_probs.iter().enumerate() {
            let sum: f64 = occ.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1.0e-10,
                "occupation_probs[{k}] sum = {sum}, expected ~1"
            );
        }
    }

    // ── Test 2: single event time ─────────────────────────────────────────────

    #[test]
    fn single_event_time() {
        let obs = vec![
            MultiStateObs {
                start_time: 0.0,
                end_time: 1.0,
                from_state: 0,
                to_state: Some(1),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 3.0,
                from_state: 0,
                to_state: None,
            },
        ];
        let config = MultiStateConfig {
            n_states: 2,
            initial_state: 0,
        };
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");
        assert_eq!(fit.event_times.len(), 1, "expected exactly 1 event time");
        assert_eq!(
            fit.transition_probs.len(),
            1,
            "expected 1 transition matrix"
        );
    }

    // ── Test 3: absorbing state conservation (row sums = 1) ──────────────────

    #[test]
    fn absorbing_state_conservation() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");
        let s = fit.n_states;
        for (k, p) in fit.transition_probs.iter().enumerate() {
            for h in 0..s {
                let row_sum: f64 = (0..s).map(|j| p[h * s + j]).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1.0e-10,
                    "P[{k}] row {h} sum = {row_sum}, expected ~1"
                );
            }
        }
    }

    // ── Test 4: identity at time zero (before first event) ───────────────────

    #[test]
    fn identity_at_time_zero() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");
        let s = fit.n_states;

        // Predict at t=0 (before any event) → identity.
        let p = predict_transition_probs(&fit, 0.0);
        let expected = identity(s);
        for (i, (a, b)) in p.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1.0e-12,
                "P[{i}] = {a}, expected {b} (identity)"
            );
        }
    }

    // ── Test 5: occupation_probs inner length == n_states ────────────────────

    #[test]
    fn occupation_prob_length() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");
        for (k, occ) in fit.occupation_probs.iter().enumerate() {
            assert_eq!(
                occ.len(),
                fit.n_states,
                "occupation_probs[{k}].len() = {}, expected {}",
                occ.len(),
                fit.n_states
            );
        }
    }

    // ── Test 6: transition_probs.len() == event_times.len() ──────────────────

    #[test]
    fn transition_probs_size() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");
        assert_eq!(
            fit.transition_probs.len(),
            fit.event_times.len(),
            "transition_probs and event_times must have the same length"
        );
    }

    // ── Test 7: every row of P sums to ~1 ────────────────────────────────────

    #[test]
    fn transition_matrix_row_sums_one() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");
        let s = fit.n_states;
        for (k, p) in fit.transition_probs.iter().enumerate() {
            assert_eq!(
                p.len(),
                s * s,
                "P[{k}] has {} elements, expected {}",
                p.len(),
                s * s
            );
            for h in 0..s {
                let row_sum: f64 = (0..s).map(|j| p[h * s + j]).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1.0e-10,
                    "P[{k}] row {h} sum = {row_sum}"
                );
            }
        }
    }

    // ── Test 8: occupation prob sums to 1 ────────────────────────────────────

    #[test]
    fn occupation_prob_sums_one() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");
        for (k, occ) in fit.occupation_probs.iter().enumerate() {
            let sum: f64 = occ.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1.0e-10,
                "occupation_probs[{k}] sum = {sum}"
            );
        }
    }

    // ── Test 9: all censored → empty event_times (not an error) ──────────────

    #[test]
    fn no_events_returns_empty() {
        let obs = vec![
            MultiStateObs {
                start_time: 0.0,
                end_time: 5.0,
                from_state: 0,
                to_state: None,
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 3.0,
                from_state: 0,
                to_state: None,
            },
        ];
        let config = MultiStateConfig {
            n_states: 2,
            initial_state: 0,
        };
        let fit = fit_multi_state(&obs, &config).expect("all-censored should succeed");
        assert!(
            fit.event_times.is_empty(),
            "expected empty event_times when all censored"
        );
        assert!(fit.transition_probs.is_empty());
        assert!(fit.occupation_probs.is_empty());
    }

    // ── Test 10: invalid state index → Err ───────────────────────────────────

    #[test]
    fn invalid_state_error() {
        let obs = vec![MultiStateObs {
            start_time: 0.0,
            end_time: 1.0,
            from_state: 0,
            to_state: Some(5), // out of range for n_states=2
        }];
        let config = MultiStateConfig {
            n_states: 2,
            initial_state: 0,
        };
        let result = fit_multi_state(&obs, &config);
        assert!(result.is_err(), "expected error for out-of-range to_state");
    }

    // ── Test 11: end_time ≤ start_time → Err ─────────────────────────────────

    #[test]
    fn negative_time_error() {
        let obs = vec![MultiStateObs {
            start_time: 2.0,
            end_time: 1.0, // end < start
            from_state: 0,
            to_state: Some(1),
        }];
        let config = MultiStateConfig {
            n_states: 2,
            initial_state: 0,
        };
        let result = fit_multi_state(&obs, &config);
        assert!(
            result.is_err(),
            "expected error when end_time <= start_time"
        );
    }

    // ── Test 12: predict interpolates to earlier matrix ──────────────────────

    #[test]
    fn predict_interpolate() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");

        // event_times contains [1.0, 2.0, 2.5, 3.0] (illness-death obs)
        // predict at t=1.8 → should give matrix at t=1.0
        let p_at_1 = predict_transition_probs(&fit, 1.0);
        let p_interpolated = predict_transition_probs(&fit, 1.8);
        assert_eq!(
            p_at_1, p_interpolated,
            "interpolation should return matrix at last event_time ≤ t"
        );
    }

    // ── Test 13: predict beyond last event returns last matrix ────────────────

    #[test]
    fn predict_beyond_last_event() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");

        let last_idx = fit.event_times.len() - 1;
        let p_last = fit.transition_probs[last_idx].clone();
        let p_beyond = predict_transition_probs(&fit, 100.0);
        assert_eq!(
            p_last, p_beyond,
            "predict beyond last event should return last matrix"
        );
    }

    // ── Test 14: two-state model reduces to Kaplan-Meier ─────────────────────
    //
    // In a 2-state (alive=0, dead=1) model where all subjects start alive,
    // P_{00}(0, t) = Kaplan-Meier S(t).
    //
    // Manual KM for 4 obs: deaths at t=1,2,3 with 1 censored at t=5:
    //   At t=1: n=4, d=1 → S(1) = (1 - 1/4) = 0.75
    //   At t=2: n=3, d=1 → S(2) = 0.75 * (1 - 1/3) = 0.50
    //   At t=3: n=2, d=1 → S(3) = 0.50 * (1 - 1/2) = 0.25
    //
    // AJ P_{00}(0, t) must match these values.

    #[test]
    fn two_state_reduces_to_km() {
        let obs = vec![
            MultiStateObs {
                start_time: 0.0,
                end_time: 1.0,
                from_state: 0,
                to_state: Some(1),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 2.0,
                from_state: 0,
                to_state: Some(1),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 3.0,
                from_state: 0,
                to_state: Some(1),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 5.0,
                from_state: 0,
                to_state: None,
            },
        ];
        let config = MultiStateConfig {
            n_states: 2,
            initial_state: 0,
        };
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");

        // Expected KM survival values.
        let expected_s: [(f64, f64); 3] = [(1.0, 0.75), (2.0, 0.50), (3.0, 0.25)];

        for (t_expected, s_expected) in &expected_s {
            let p = predict_transition_probs(&fit, *t_expected);
            // P_{00} = p[0*2 + 0] = p[0]
            let p00 = p[0];
            assert!(
                (p00 - s_expected).abs() < 1.0e-10,
                "at t={t_expected}: P_{{00}} = {p00}, expected KM S(t) = {s_expected}"
            );
        }
    }

    // ── Test 15: n_states < 2 returns Err ────────────────────────────────────

    #[test]
    fn single_state_returns_error() {
        let obs = vec![MultiStateObs {
            start_time: 0.0,
            end_time: 1.0,
            from_state: 0,
            to_state: None,
        }];
        let config = MultiStateConfig {
            n_states: 1,
            initial_state: 0,
        };
        let result = fit_multi_state(&obs, &config);
        assert!(result.is_err(), "n_states=1 should return an error");
    }

    // ── Test 16: n_obs is recorded correctly ─────────────────────────────────

    #[test]
    fn n_obs_correct() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");
        assert_eq!(
            fit.n_obs, 5,
            "n_obs should equal the number of observations"
        );
    }

    // ── Test 17: predict_occupation invalid initial_state → Err ──────────────

    #[test]
    fn predict_occupation_invalid_state() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");
        let result = predict_occupation(&fit, 1.0, 99);
        assert!(
            result.is_err(),
            "predict_occupation with invalid initial_state should error"
        );
    }

    // ── Test 18: predict_occupation returns correct length ───────────────────

    #[test]
    fn predict_occupation_length() {
        let obs = illness_death_obs();
        let config = illness_death_config();
        let fit = fit_multi_state(&obs, &config).expect("fit should succeed");
        let occ = predict_occupation(&fit, 2.0, 0).expect("should succeed");
        assert_eq!(
            occ.len(),
            fit.n_states,
            "occupation vector length should equal n_states"
        );
    }
}

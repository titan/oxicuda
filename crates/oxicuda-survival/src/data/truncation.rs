//! Truncation support for survival data.
//!
//! Provides:
//! - Left-truncated (delayed-entry) Kaplan-Meier estimator
//! - Right-truncated observations with inverse-probability weighting scaffold
//! - Turnbull EM algorithm for interval-censored data
//! - Counting-process conversion for left-truncated Cox regression
//! - Conditional survival probability under left-truncation
//!
//! # Concepts
//!
//! **Left-truncation**: Subject i enters the risk set at time `entry_time_i > 0`.
//! They contribute to the risk set only for t ≥ entry_time_i.
//! The modified risk set at time t_k is:
//! ```text
//! R(t_k) = { j : entry_j ≤ t_k ≤ time_j }
//! ```
//!
//! **Interval censoring**: The event time is known only to lie in [L_i, R_i].
//! The Turnbull EM algorithm finds the non-parametric MLE for S(t).

use crate::error::{SurvivalError, SurvivalResult};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A survival observation with left-truncation (delayed entry).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruncatedObs {
    /// Event or censoring time (must be > entry_time).
    pub time: f64,
    /// Event indicator: 1 = event occurred, 0 = right-censored.
    pub event: u8,
    /// Left-truncation (delayed entry) time; 0.0 means no truncation.
    pub entry_time: f64,
}

impl TruncatedObs {
    /// Construct a `TruncatedObs` with validation.
    pub fn new(time: f64, event: u8, entry_time: f64) -> SurvivalResult<Self> {
        if !time.is_finite() || time < 0.0 {
            return Err(SurvivalError::NegativeTime(time));
        }
        if !entry_time.is_finite() || entry_time < 0.0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "entry_time must be finite and >= 0: {entry_time}"
            )));
        }
        if entry_time >= time {
            return Err(SurvivalError::InvalidParameter(format!(
                "entry_time ({entry_time}) must be strictly less than time ({time})"
            )));
        }
        if event > 1 {
            return Err(SurvivalError::InvalidParameter(format!(
                "event indicator must be 0 or 1, got {event}"
            )));
        }
        Ok(Self {
            time,
            event,
            entry_time,
        })
    }
}

/// A survival observation with interval censoring.
///
/// The event time is known to lie in `[lower, upper]`.
/// Special cases:
/// - `lower == upper`: exactly observed event (point mass)
/// - `lower == 0.0`: left-censored (event occurred before `upper`)
/// - `upper == f64::INFINITY`: right-censored (event hasn't occurred by `lower`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalObs {
    /// Lower bound of the event time interval (inclusive).
    pub lower: f64,
    /// Upper bound of the event time interval (inclusive; `f64::INFINITY` = right-censored).
    pub upper: f64,
}

impl IntervalObs {
    /// Construct an `IntervalObs` with validation.
    pub fn new(lower: f64, upper: f64) -> SurvivalResult<Self> {
        if lower < 0.0 || !lower.is_finite() {
            return Err(SurvivalError::InvalidParameter(format!(
                "lower bound must be finite and >= 0: {lower}"
            )));
        }
        if upper < lower {
            return Err(SurvivalError::InvalidParameter(format!(
                "upper bound ({upper}) must be >= lower bound ({lower})"
            )));
        }
        Ok(Self { lower, upper })
    }

    /// Returns `true` if this represents an exact (point) observation.
    #[must_use]
    pub fn is_exact(&self) -> bool {
        (self.lower - self.upper).abs() < 1.0e-14
    }

    /// Returns `true` if this represents a right-censored observation.
    #[must_use]
    pub fn is_right_censored(&self) -> bool {
        self.upper == f64::INFINITY
    }

    /// Returns `true` if this represents a left-censored observation.
    #[must_use]
    pub fn is_left_censored(&self) -> bool {
        self.lower == 0.0 && self.upper.is_finite()
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Output of the truncated Kaplan-Meier estimator.
#[derive(Debug, Clone)]
pub struct TruncatedKmResult {
    /// Unique event times (ascending).
    pub times: Vec<f64>,
    /// Survival estimate Ŝ(t) at each event time.
    pub survival: Vec<f64>,
    /// Greenwood variance estimate of Ŝ(t) at each event time.
    pub greenwood_var: Vec<f64>,
    /// Number at risk n_k at each event time (subjects in modified risk set).
    pub n_risk: Vec<usize>,
    /// Number of events d_k at each event time.
    pub n_events: Vec<usize>,
    /// `true` if any subject had entry_time > 0 (left-truncation was active).
    pub entry_times_used: bool,
}

impl TruncatedKmResult {
    /// Standard error √Var(Ŝ) at each event time.
    #[must_use]
    pub fn standard_error(&self) -> Vec<f64> {
        self.greenwood_var
            .iter()
            .map(|v| v.max(0.0).sqrt())
            .collect()
    }

    /// Evaluate Ŝ(t) at an arbitrary query time `t` by step-function interpolation.
    ///
    /// Returns 1.0 for t before the first event time, and the last survival
    /// value for t beyond the last event time.
    #[must_use]
    pub fn eval(&self, t: f64) -> f64 {
        if self.times.is_empty() {
            return 1.0;
        }
        // Find the largest event time <= t
        let pos = self.times.partition_point(|&tk| tk <= t);
        if pos == 0 {
            1.0
        } else {
            self.survival[pos - 1]
        }
    }
}

/// Output of the Turnbull EM algorithm for interval-censored data.
#[derive(Debug, Clone)]
pub struct TurnbullResult {
    /// Candidate times where positive probability mass may reside.
    pub mass_points: Vec<f64>,
    /// Non-parametric MLE probability mass at each mass point.
    pub prob_mass: Vec<f64>,
    /// Survival function Ŝ(t) = Pr(T > t) evaluated at each mass point.
    pub survival: Vec<f64>,
    /// Number of EM iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
}

impl TurnbullResult {
    /// Total probability mass assigned (≤ 1.0; any remainder is on the infinite tail).
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        self.prob_mass.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate left-truncated observations: entry_time < time, no negatives.
///
/// Checks the first `n` entries of `obs` (or all of `obs` if `obs.len() <= n`).
pub fn validate_truncated(obs: &[TruncatedObs], n: usize) -> SurvivalResult<()> {
    let m = n.min(obs.len());
    if m == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    for (idx, o) in obs[..m].iter().enumerate() {
        if !o.time.is_finite() || o.time < 0.0 {
            return Err(SurvivalError::NegativeTime(o.time));
        }
        if !o.entry_time.is_finite() || o.entry_time < 0.0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "observation[{idx}]: entry_time must be finite and >= 0: {}",
                o.entry_time
            )));
        }
        if o.entry_time >= o.time {
            return Err(SurvivalError::InvalidParameter(format!(
                "observation[{idx}]: entry_time ({}) must be strictly less than time ({})",
                o.entry_time, o.time
            )));
        }
        if o.event > 1 {
            return Err(SurvivalError::InvalidParameter(format!(
                "observation[{idx}]: event must be 0 or 1, got {}",
                o.event
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Effective sample size
// ---------------------------------------------------------------------------

/// Effective sample size under left-truncation.
///
/// Computes the harmonic-mean-based effective n that accounts for delayed
/// entry: subjects who enter the risk set later reduce the effective sample
/// size because they are at risk for a shorter proportion of follow-up time.
///
/// Formally this is defined as:
/// ```text
/// n_eff = n * Σ_i (time_i - entry_i) / Σ_i time_i
/// ```
/// which equals n when all entry_times are 0.  The result is ≤ n always.
///
/// Only the first `n` observations in `obs` are used.
pub fn effective_sample_size(obs: &[TruncatedObs], n: usize) -> f64 {
    let m = n.min(obs.len());
    if m == 0 {
        return 0.0;
    }
    let total_follow_up: f64 = obs[..m]
        .iter()
        .map(|o| (o.time - o.entry_time).max(0.0))
        .sum();
    let total_time: f64 = obs[..m].iter().map(|o| o.time).sum();
    if total_time <= 0.0 {
        return m as f64;
    }
    (m as f64) * total_follow_up / total_time
}

// ---------------------------------------------------------------------------
// Left-truncated Kaplan-Meier
// ---------------------------------------------------------------------------

/// Kaplan-Meier estimator with left-truncation (delayed entry).
///
/// The risk set at each event time t_k is:
/// ```text
/// R(t_k) = { j : entry_j ≤ t_k  AND  t_k ≤ time_j }
/// ```
/// This differs from the standard KM only when some subjects have `entry_time > 0`.
///
/// Uses the first `n` entries in `obs` (or all of `obs` if `obs.len() <= n`).
///
/// # Errors
/// Returns [`SurvivalError::EmptyDataset`] if `n == 0` or `obs` is empty.
/// Returns [`SurvivalError::NoEvents`] if no events are present.
/// Returns [`SurvivalError::NumericalInstability`] if any risk set is empty at an event time.
pub fn truncated_km(obs: &[TruncatedObs], n: usize) -> SurvivalResult<TruncatedKmResult> {
    let m = n.min(obs.len());
    if m == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    validate_truncated(obs, m)?;

    let entry_times_used = obs[..m].iter().any(|o| o.entry_time > 0.0);

    // Collect unique event times (where event == 1)
    let mut event_times: Vec<f64> = obs[..m]
        .iter()
        .filter(|o| o.event == 1)
        .map(|o| o.time)
        .collect();
    if event_times.is_empty() {
        return Err(SurvivalError::NoEvents);
    }
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < 1.0e-14);

    let k = event_times.len();
    let mut times = Vec::with_capacity(k);
    let mut n_risk = Vec::with_capacity(k);
    let mut n_events = Vec::with_capacity(k);
    let mut survival = Vec::with_capacity(k);
    let mut greenwood_var = Vec::with_capacity(k);

    let mut s_cur = 1.0_f64;
    let mut var_log_acc = 0.0_f64; // Σ d_k / (n_k * (n_k - d_k))

    for &tk in &event_times {
        // Modified risk set: subjects with entry_time <= tk AND time >= tk
        let nk: usize = obs[..m]
            .iter()
            .filter(|o| o.entry_time <= tk && o.time >= tk)
            .count();
        if nk == 0 {
            return Err(SurvivalError::NumericalInstability(format!(
                "empty risk set at event time {tk}"
            )));
        }

        // Number of events at exactly tk
        let dk: usize = obs[..m]
            .iter()
            .filter(|o| o.event == 1 && (o.time - tk).abs() < 1.0e-14)
            .count();

        let nk_f = nk as f64;
        let dk_f = dk as f64;
        let factor = (1.0 - dk_f / nk_f).max(0.0);
        s_cur *= factor;

        // Greenwood term: d_k / (n_k * (n_k - d_k)), only when n_k > d_k
        if dk > 0 && nk > dk {
            var_log_acc += dk_f / (nk_f * (nk_f - dk_f));
        }

        times.push(tk);
        n_risk.push(nk);
        n_events.push(dk);
        survival.push(s_cur);
        greenwood_var.push(s_cur * s_cur * var_log_acc);
    }

    Ok(TruncatedKmResult {
        times,
        survival,
        greenwood_var,
        n_risk,
        n_events,
        entry_times_used,
    })
}

// ---------------------------------------------------------------------------
// Counting-process conversion
// ---------------------------------------------------------------------------

/// Convert left-truncated observations to counting-process (start, stop, event) format.
///
/// This is the standard format used by time-varying Cox regression:
/// - `start` = `entry_time` (subject enters risk set just after `start`)
/// - `stop` = `time` (event or censoring time)
/// - `event` = 1 if event occurred, 0 if censored
///
/// The result can be passed directly to `CountingProcessDataset` for Cox regression
/// with left-truncated data.
///
/// Uses the first `n` entries in `obs`.
///
/// # Errors
/// Returns [`SurvivalError::EmptyDataset`] if `n == 0` or `obs` is empty.
pub fn to_counting_process(obs: &[TruncatedObs], n: usize) -> SurvivalResult<Vec<(f64, f64, u8)>> {
    let m = n.min(obs.len());
    if m == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    validate_truncated(obs, m)?;

    let result = obs[..m]
        .iter()
        .map(|o| (o.entry_time, o.time, o.event))
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Conditional survival probability
// ---------------------------------------------------------------------------

/// Compute the conditional survival probability P(T > t | T > s) for t >= s.
///
/// Uses the left-truncated KM curve. By the conditional probability law:
/// ```text
/// P(T > t | T > s) = P(T > t) / P(T > s) = Ŝ(t) / Ŝ(s)
/// ```
/// When t == s this equals 1.0. When t < s this is not well-defined as
/// a conditional probability (event is impossible), so an error is returned.
///
/// # Errors
/// - [`SurvivalError::InvalidParameter`] if `t < s`
/// - [`SurvivalError::NumericalInstability`] if `Ŝ(s) == 0`
pub fn conditional_survival(km: &TruncatedKmResult, t: f64, s: f64) -> SurvivalResult<f64> {
    if t < s - 1.0e-14 {
        return Err(SurvivalError::InvalidParameter(format!(
            "t ({t}) must be >= s ({s}) for conditional survival"
        )));
    }
    let s_t = km.eval(t);
    let s_s = km.eval(s);
    if s_s <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "S(s) = 0; conditional survival is undefined".to_string(),
        ));
    }
    Ok((s_t / s_s).min(1.0))
}

// ---------------------------------------------------------------------------
// Turnbull EM algorithm
// ---------------------------------------------------------------------------

/// Estimate the survival function from interval-censored data using the
/// Turnbull EM (non-parametric MLE) algorithm.
///
/// # Algorithm
///
/// 1. **Initialize**: candidate mass points = all distinct finite `lower` and `upper`
///    values from `obs` (excluding 0 from left-censored obs and ∞ from right-censored).
///    Uniform mass across candidate points.
///
/// 2. **E-step**: For each subject i, compute the set of mass points that fall
///    in `[lower_i, upper_i]`.  Allocate the subject's probability mass
///    proportionally to the current mass at those points.
///
/// 3. **M-step**: The new mass at each point t_k is:
///    ```text
///    p_k = (1/n) Σ_i  [ allocated mass from i to t_k ]
///    ```
///
/// 4. **Convergence**: Stop when `max_k |p_k_new - p_k_old| < tol`.
///
/// Uses the first `n` entries in `obs`.
///
/// # Errors
/// - [`SurvivalError::EmptyDataset`] if `n == 0` or `obs` is empty.
/// - [`SurvivalError::InvalidParameter`] if `tol` or `max_iter` are invalid.
pub fn turnbull_em(
    obs: &[IntervalObs],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> SurvivalResult<TurnbullResult> {
    let m = n.min(obs.len());
    if m == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if tol < 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "tol must be >= 0, got {tol}"
        )));
    }
    if max_iter == 0 {
        return Err(SurvivalError::InvalidParameter(
            "max_iter must be >= 1".to_string(),
        ));
    }

    // -----------------------------------------------------------------------
    // Build the set of candidate mass points.
    //
    // Per Turnbull (1976): the NPMLE places mass only at "innermost intervals"
    // formed by the data.  A practical approximation that works well is to
    // use all distinct finite lower and upper bound values as candidates.
    // -----------------------------------------------------------------------
    let mut candidate_set: Vec<f64> = Vec::new();
    for o in &obs[..m] {
        // Lower bound: include if > 0 (skip 0 for left-censored obs as a mass
        // point at 0 would be degenerate for survival models)
        if o.lower > 0.0 {
            candidate_set.push(o.lower);
        }
        // Upper bound: include if finite
        if o.upper.is_finite() {
            candidate_set.push(o.upper);
        }
    }
    candidate_set.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidate_set.dedup_by(|a, b| (*a - *b).abs() < 1.0e-14);

    if candidate_set.is_empty() {
        // All observations are purely right-censored with lower==0 and upper==∞;
        // no mass points can be identified — return trivial result.
        return Ok(TurnbullResult {
            mass_points: Vec::new(),
            prob_mass: Vec::new(),
            survival: Vec::new(),
            n_iter: 0,
            converged: true,
        });
    }

    let kp = candidate_set.len();

    // Initialize uniform mass
    let init_mass = 1.0 / kp as f64;
    let mut mass: Vec<f64> = vec![init_mass; kp];

    // Precompute which mass points fall in [lower_i, upper_i] for each subject.
    // A mass point t_k is "eligible" for subject i if lower_i <= t_k <= upper_i.
    let eligible: Vec<Vec<usize>> = obs[..m]
        .iter()
        .map(|o| {
            candidate_set
                .iter()
                .enumerate()
                .filter(|&(_, &tk)| tk >= o.lower && (o.upper == f64::INFINITY || tk <= o.upper))
                .map(|(k, _)| k)
                .collect()
        })
        .collect();

    let mut n_iter = 0usize;
    let mut converged = false;

    for _iter in 0..max_iter {
        n_iter += 1;
        let mut new_mass = vec![0.0_f64; kp];

        // E-step + M-step (combined in one pass per Turnbull's ISCF approach)
        for (i, elig) in eligible.iter().enumerate() {
            let _ = i; // subject index (informational)
            if elig.is_empty() {
                // Subject's interval contains no candidate mass point;
                // they contribute no information.
                continue;
            }
            // Total current mass in subject's eligibility set
            let w_total: f64 = elig.iter().map(|&k| mass[k]).sum();
            if w_total <= 0.0 {
                // Distribute equally to avoid degenerate allocation
                let share = 1.0 / elig.len() as f64;
                for &k in elig {
                    new_mass[k] += share;
                }
            } else {
                // Allocate proportionally to current mass (E-step)
                for &k in elig {
                    new_mass[k] += mass[k] / w_total;
                }
            }
        }

        // Normalize to get probability mass (M-step: divide by n)
        let n_f = m as f64;
        for val in &mut new_mass {
            *val /= n_f;
        }

        // Convergence check
        let max_delta = mass
            .iter()
            .zip(new_mass.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        mass = new_mass;

        if max_delta < tol {
            converged = true;
            break;
        }
    }

    // Compute survival function: S(t_k) = Pr(T > t_k) = Σ_{j > k} p_j
    // (probability of event time strictly after t_k)
    let total: f64 = mass.iter().sum();
    let mut survival = Vec::with_capacity(kp);
    let mut cumulative = 0.0_f64;
    for &mk in &mass {
        cumulative += mk;
        // S(t_k) = 1 - F(t_k) where F(t_k) = Pr(T <= t_k) = sum of mass up to k
        let s_val = (total - cumulative).max(0.0);
        survival.push(s_val);
    }

    Ok(TurnbullResult {
        mass_points: candidate_set,
        prob_mass: mass,
        survival,
        n_iter,
        converged,
    })
}

// ---------------------------------------------------------------------------
// Right-truncation scaffold
// ---------------------------------------------------------------------------

/// Observation with right-truncation.
///
/// Subject is observed only if their event time T ≤ `trunc_time` (right-truncation time).
/// Inverse probability weighting is used to account for the selection bias.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RightTruncatedObs {
    /// Observed event or censoring time.
    pub time: f64,
    /// Event indicator: 1 = event, 0 = censored.
    pub event: u8,
    /// Right-truncation time: subject is only observed if time ≤ trunc_time.
    pub trunc_time: f64,
}

impl RightTruncatedObs {
    /// Construct a `RightTruncatedObs` with validation.
    pub fn new(time: f64, event: u8, trunc_time: f64) -> SurvivalResult<Self> {
        if !time.is_finite() || time < 0.0 {
            return Err(SurvivalError::NegativeTime(time));
        }
        if !trunc_time.is_finite() || trunc_time <= 0.0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "trunc_time must be finite and > 0: {trunc_time}"
            )));
        }
        if time > trunc_time {
            return Err(SurvivalError::InvalidParameter(format!(
                "time ({time}) must be <= trunc_time ({trunc_time}) for right-truncated observation"
            )));
        }
        if event > 1 {
            return Err(SurvivalError::InvalidParameter(format!(
                "event must be 0 or 1, got {event}"
            )));
        }
        Ok(Self {
            time,
            event,
            trunc_time,
        })
    }

    /// Inverse probability weight for this observation.
    ///
    /// Under right-truncation, each observation has weight ∝ 1 / Pr(T ≤ trunc_time).
    /// Since the marginal truncation probability depends on the unknown distribution,
    /// a natural plug-in estimate is 1 / (time / trunc_time) for uniform truncation.
    /// This method returns the raw ratio for use in weighted analyses.
    #[must_use]
    pub fn ipw_weight(&self) -> f64 {
        if self.trunc_time <= 0.0 {
            return 1.0;
        }
        self.trunc_time / self.time.max(1.0e-14)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test 1: truncated_km with entry_time=0 matches standard KM
    // -----------------------------------------------------------------------
    #[test]
    fn truncated_km_no_truncation_matches_standard_km() {
        // 4 subjects, all enter at 0, all have events.
        // Standard KM: S(1)=0.75, S(2)=0.5, S(3)=0.25, S(4)=0.0
        let obs = vec![
            TruncatedObs {
                time: 1.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 2.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 3.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 4.0,
                event: 1,
                entry_time: 0.0,
            },
        ];
        let km = truncated_km(&obs, obs.len()).expect("ok");
        assert!(!km.entry_times_used, "no truncation should be detected");
        assert!(
            (km.survival[0] - 0.75).abs() < 1.0e-12,
            "S(1) should be 0.75"
        );
        assert!((km.survival[1] - 0.5).abs() < 1.0e-12, "S(2) should be 0.5");
        assert!(
            (km.survival[2] - 0.25).abs() < 1.0e-12,
            "S(3) should be 0.25"
        );
        assert!((km.survival[3] - 0.0).abs() < 1.0e-12, "S(4) should be 0.0");
    }

    // -----------------------------------------------------------------------
    // Test 2: left-truncation changes the survival estimate
    // -----------------------------------------------------------------------
    #[test]
    fn truncated_km_with_delayed_entry_differs_from_standard() {
        // Subject 1: enters at 0, event at 3
        // Subject 2: enters at 2, event at 4  (NOT in risk set at t=3)
        // At t=3: risk set = {subj 1} (entry<=3 AND time>=3), n=1, d=1 → S(3)=0
        // At t=4: risk set = {subj 2} (entry<=4 AND time>=4), n=1, d=1 → S(4)=0
        // (S drops to 0 earlier than without truncation)
        let obs = vec![
            TruncatedObs {
                time: 3.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 4.0,
                event: 1,
                entry_time: 2.0,
            },
        ];
        let km = truncated_km(&obs, obs.len()).expect("ok");
        assert!(km.entry_times_used, "entry_time=2 should be detected");
        // At t=3: both subjects are in risk set? entry_2=2 <= 3 AND time_2=4 >= 3 → yes, n=2
        // At t=3: d=1, n=2 → S(3) = 0.5
        // At t=4: entry_1=0 <= 4, time_1=3 < 4 → NOT in risk set; entry_2=2<=4, time_2=4>=4 → yes; n=1
        // S(4) = 0.5 * (1 - 1/1) = 0.0
        assert_eq!(km.n_risk[0], 2, "both subjects at risk at t=3");
        assert!((km.survival[0] - 0.5).abs() < 1.0e-12, "S(3) = 0.5");
        assert_eq!(km.n_risk[1], 1, "only subject 2 at risk at t=4");
        assert!((km.survival[1] - 0.0).abs() < 1.0e-12, "S(4) = 0.0");
    }

    // -----------------------------------------------------------------------
    // Test 3: negative entry_time triggers validation error
    // -----------------------------------------------------------------------
    #[test]
    fn validate_truncated_rejects_negative_entry() {
        let obs = vec![TruncatedObs {
            time: 2.0,
            event: 1,
            entry_time: -1.0,
        }];
        assert!(validate_truncated(&obs, obs.len()).is_err());
    }

    // -----------------------------------------------------------------------
    // Test 4: entry_time >= time triggers validation error
    // -----------------------------------------------------------------------
    #[test]
    fn validate_truncated_rejects_entry_ge_time() {
        let obs = vec![TruncatedObs {
            time: 2.0,
            event: 1,
            entry_time: 2.0,
        }];
        assert!(validate_truncated(&obs, obs.len()).is_err());

        let obs2 = vec![TruncatedObs {
            time: 2.0,
            event: 1,
            entry_time: 3.0,
        }];
        assert!(validate_truncated(&obs2, obs2.len()).is_err());
    }

    // -----------------------------------------------------------------------
    // Test 5: all subjects have same entry time → effectively standard KM
    // -----------------------------------------------------------------------
    #[test]
    fn truncated_km_same_entry_matches_standard() {
        // All enter at 0.5; S(t) should be same as standard KM (entry_time=0.5 < all event times)
        let obs = vec![
            TruncatedObs {
                time: 1.0,
                event: 1,
                entry_time: 0.5,
            },
            TruncatedObs {
                time: 2.0,
                event: 1,
                entry_time: 0.5,
            },
            TruncatedObs {
                time: 3.0,
                event: 0,
                entry_time: 0.5,
            },
        ];
        let km = truncated_km(&obs, obs.len()).expect("ok");
        // At t=1: n=3 (all have entry<=1 and time>=1), d=1 → S=2/3
        // At t=2: n=2 (subj 1 has time=1<2 → not in risk set; others: entry<=2 and time>=2), d=1 → S=2/3*(1/2)=1/3
        // (subj 3 is censored at 3, still at risk at t=2)
        assert_eq!(km.n_risk[0], 3);
        assert!((km.survival[0] - 2.0 / 3.0).abs() < 1.0e-12);
        assert_eq!(km.n_risk[1], 2);
        assert!((km.survival[1] - 1.0 / 3.0).abs() < 1.0e-12);
    }

    // -----------------------------------------------------------------------
    // Test 6: Turnbull EM with right-censored only approximates standard KM
    // -----------------------------------------------------------------------
    #[test]
    fn turnbull_em_right_censored_only() {
        // When all observations are right-censored [L, ∞), mass concentrates
        // at the lower bounds (which are also event times).
        // 3 exact events: [1,1], [2,2], [3,3] plus one right-censored [2,∞)
        let obs = vec![
            IntervalObs {
                lower: 1.0,
                upper: 1.0,
            },
            IntervalObs {
                lower: 2.0,
                upper: 2.0,
            },
            IntervalObs {
                lower: 3.0,
                upper: 3.0,
            },
            IntervalObs {
                lower: 2.0,
                upper: f64::INFINITY,
            },
        ];
        let result = turnbull_em(&obs, obs.len(), 200, 1.0e-8).expect("ok");
        // Should place mass at 1.0, 2.0, 3.0
        assert!(result.mass_points.contains(&1.0));
        assert!(result.mass_points.contains(&2.0));
        assert!(result.mass_points.contains(&3.0));
        // Total mass should be <= 1.0
        assert!(result.total_mass() <= 1.0 + 1.0e-10);
        // All mass values should be non-negative
        for &p in &result.prob_mass {
            assert!(p >= -1.0e-12, "mass must be non-negative, got {p}");
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: Turnbull EM with exact events places mass at event times
    // -----------------------------------------------------------------------
    #[test]
    fn turnbull_em_exact_events_mass_at_event_times() {
        // Three exact events: all mass must go to exactly those three times
        let obs = vec![
            IntervalObs {
                lower: 1.0,
                upper: 1.0,
            },
            IntervalObs {
                lower: 2.0,
                upper: 2.0,
            },
            IntervalObs {
                lower: 2.0,
                upper: 2.0,
            },
            IntervalObs {
                lower: 5.0,
                upper: 5.0,
            },
        ];
        let result = turnbull_em(&obs, obs.len(), 200, 1.0e-8).expect("ok");
        // Convergence should happen quickly for exact observations
        assert!(result.converged, "should converge for exact events");
        // mass at t=1: 1/4, t=2: 2/4=0.5, t=5: 1/4
        let idx1 = result
            .mass_points
            .iter()
            .position(|&t| (t - 1.0).abs() < 1.0e-12);
        let idx2 = result
            .mass_points
            .iter()
            .position(|&t| (t - 2.0).abs() < 1.0e-12);
        let idx5 = result
            .mass_points
            .iter()
            .position(|&t| (t - 5.0).abs() < 1.0e-12);
        assert!(idx1.is_some() && idx2.is_some() && idx5.is_some());
        assert!((result.prob_mass[idx1.unwrap()] - 0.25).abs() < 1.0e-6);
        assert!((result.prob_mass[idx2.unwrap()] - 0.50).abs() < 1.0e-6);
        assert!((result.prob_mass[idx5.unwrap()] - 0.25).abs() < 1.0e-6);
    }

    // -----------------------------------------------------------------------
    // Test 8: left-censored subjects (lower=0) contribute mass below first event
    // -----------------------------------------------------------------------
    #[test]
    fn turnbull_em_left_censored_subjects() {
        // Left-censored [0, 3.0]: event happened before t=3
        // Exact [5.0, 5.0]: event at t=5
        let obs = vec![
            IntervalObs {
                lower: 0.0,
                upper: 3.0,
            },
            IntervalObs {
                lower: 0.0,
                upper: 3.0,
            },
            IntervalObs {
                lower: 5.0,
                upper: 5.0,
            },
        ];
        let result = turnbull_em(&obs, obs.len(), 200, 1.0e-8).expect("ok");
        // Mass points should include 3.0 and 5.0
        // (lower=0 is excluded from candidate set per the algorithm)
        let has_3 = result
            .mass_points
            .iter()
            .any(|&t| (t - 3.0).abs() < 1.0e-12);
        let has_5 = result
            .mass_points
            .iter()
            .any(|&t| (t - 5.0).abs() < 1.0e-12);
        assert!(has_3, "should have mass point at 3.0");
        assert!(has_5, "should have mass point at 5.0");
        // Total mass <= 1.0
        assert!(result.total_mass() <= 1.0 + 1.0e-10);
    }

    // -----------------------------------------------------------------------
    // Test 9: to_counting_process correctness
    // -----------------------------------------------------------------------
    #[test]
    fn to_counting_process_correct() {
        let obs = vec![
            TruncatedObs {
                time: 5.0,
                event: 1,
                entry_time: 1.0,
            },
            TruncatedObs {
                time: 8.0,
                event: 0,
                entry_time: 3.0,
            },
            TruncatedObs {
                time: 10.0,
                event: 1,
                entry_time: 0.0,
            },
        ];
        let cp = to_counting_process(&obs, obs.len()).expect("ok");
        assert_eq!(cp.len(), 3);
        assert_eq!(cp[0], (1.0, 5.0, 1));
        assert_eq!(cp[1], (3.0, 8.0, 0));
        assert_eq!(cp[2], (0.0, 10.0, 1));
    }

    // -----------------------------------------------------------------------
    // Test 10: conditional_survival(t=s) == 1.0
    // -----------------------------------------------------------------------
    #[test]
    fn conditional_survival_at_s_equals_one() {
        let obs = vec![
            TruncatedObs {
                time: 1.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 2.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 3.0,
                event: 0,
                entry_time: 0.0,
            },
        ];
        let km = truncated_km(&obs, obs.len()).expect("ok");
        // P(T > 1.5 | T > 1.5) should equal 1.0
        let cond = conditional_survival(&km, 1.5, 1.5).expect("ok");
        assert!((cond - 1.0).abs() < 1.0e-12, "P(T>s|T>s)=1, got {cond}");
    }

    // -----------------------------------------------------------------------
    // Test 11: conditional_survival(t < s) returns error
    // -----------------------------------------------------------------------
    #[test]
    fn conditional_survival_t_less_than_s_errors() {
        let obs = vec![
            TruncatedObs {
                time: 1.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 2.0,
                event: 1,
                entry_time: 0.0,
            },
        ];
        let km = truncated_km(&obs, obs.len()).expect("ok");
        assert!(conditional_survival(&km, 0.5, 1.5).is_err());
    }

    // -----------------------------------------------------------------------
    // Test 12: effective_sample_size <= n always
    // -----------------------------------------------------------------------
    #[test]
    fn effective_sample_size_le_n() {
        let obs = vec![
            TruncatedObs {
                time: 5.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 5.0,
                event: 0,
                entry_time: 2.0,
            },
            TruncatedObs {
                time: 8.0,
                event: 1,
                entry_time: 4.0,
            },
        ];
        let n = obs.len();
        let ess = effective_sample_size(&obs, n);
        assert!(ess <= n as f64 + 1.0e-12, "ESS ({ess}) must be <= n ({n})");
        assert!(ess > 0.0, "ESS must be positive");

        // No truncation → ESS == n
        let obs2 = vec![
            TruncatedObs {
                time: 5.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 6.0,
                event: 1,
                entry_time: 0.0,
            },
        ];
        // ESS = 2 * (5+6) / (5+6) = 2.0
        let ess2 = effective_sample_size(&obs2, obs2.len());
        assert!(
            (ess2 - 2.0).abs() < 1.0e-12,
            "No truncation: ESS should be n=2, got {ess2}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 13: empty observations return error
    // -----------------------------------------------------------------------
    #[test]
    fn truncated_km_empty_returns_error() {
        let obs: Vec<TruncatedObs> = Vec::new();
        assert!(truncated_km(&obs, 0).is_err());
        assert!(to_counting_process(&obs, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Test 14: TurnbullResult prob_mass sums to <= 1.0
    // -----------------------------------------------------------------------
    #[test]
    fn turnbull_prob_mass_sums_le_one() {
        // Mix of right-censored and exact observations
        let obs = vec![
            IntervalObs {
                lower: 1.0,
                upper: 1.0,
            },
            IntervalObs {
                lower: 3.0,
                upper: f64::INFINITY,
            },
            IntervalObs {
                lower: 2.0,
                upper: 4.0,
            },
            IntervalObs {
                lower: 4.0,
                upper: 4.0,
            },
            IntervalObs {
                lower: 5.0,
                upper: f64::INFINITY,
            },
        ];
        let result = turnbull_em(&obs, obs.len(), 200, 1.0e-8).expect("ok");
        let total = result.total_mass();
        assert!(
            total <= 1.0 + 1.0e-10,
            "prob_mass total ({total}) must be <= 1.0"
        );
        assert!(total >= 0.0, "prob_mass total must be >= 0.0");
    }

    // -----------------------------------------------------------------------
    // Test 15: TruncatedKmResult.eval interpolates correctly
    // -----------------------------------------------------------------------
    #[test]
    fn truncated_km_eval_interpolation() {
        let obs = vec![
            TruncatedObs {
                time: 2.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 4.0,
                event: 1,
                entry_time: 0.0,
            },
            TruncatedObs {
                time: 6.0,
                event: 0,
                entry_time: 0.0,
            },
        ];
        let km = truncated_km(&obs, obs.len()).expect("ok");
        // Before first event: S=1.0
        assert!((km.eval(0.5) - 1.0).abs() < 1.0e-12);
        // At first event t=2: S=0.5 (2 at risk, 1 event)
        assert!((km.eval(2.0) - km.survival[0]).abs() < 1.0e-12);
        // Between t=2 and t=4: same as at t=2 (step function)
        assert!((km.eval(3.0) - km.survival[0]).abs() < 1.0e-12);
        // At t=4: S(4)
        assert!((km.eval(4.0) - km.survival[1]).abs() < 1.0e-12);
        // After last event: same as last survival value
        assert!((km.eval(100.0) - km.survival[1]).abs() < 1.0e-12);
    }

    // -----------------------------------------------------------------------
    // Test 16: RightTruncatedObs construction and validation
    // -----------------------------------------------------------------------
    #[test]
    fn right_truncated_obs_validation() {
        // Valid
        let o = RightTruncatedObs::new(3.0, 1, 5.0).expect("ok");
        assert_eq!(o.time, 3.0);
        assert_eq!(o.trunc_time, 5.0);

        // time > trunc_time should fail
        assert!(RightTruncatedObs::new(6.0, 1, 5.0).is_err());
        // negative trunc_time should fail
        assert!(RightTruncatedObs::new(1.0, 1, -1.0).is_err());
        // invalid event
        assert!(RightTruncatedObs::new(1.0, 2, 5.0).is_err());
    }

    // -----------------------------------------------------------------------
    // Test 17: Turnbull convergence flag when tol is large
    // -----------------------------------------------------------------------
    #[test]
    fn turnbull_converged_flag_large_tol() {
        let obs = vec![
            IntervalObs {
                lower: 1.0,
                upper: 2.0,
            },
            IntervalObs {
                lower: 3.0,
                upper: 4.0,
            },
        ];
        // Very large tolerance → should converge in 1 iteration
        let result = turnbull_em(&obs, obs.len(), 100, 1.0).expect("ok");
        assert!(result.converged, "should converge immediately with tol=1.0");
    }

    // -----------------------------------------------------------------------
    // Test 18: IntervalObs helper predicates
    // -----------------------------------------------------------------------
    #[test]
    fn interval_obs_predicates() {
        let exact = IntervalObs {
            lower: 3.0,
            upper: 3.0,
        };
        assert!(exact.is_exact());
        assert!(!exact.is_right_censored());
        assert!(!exact.is_left_censored());

        let right_cens = IntervalObs {
            lower: 5.0,
            upper: f64::INFINITY,
        };
        assert!(!right_cens.is_exact());
        assert!(right_cens.is_right_censored());

        let left_cens = IntervalObs {
            lower: 0.0,
            upper: 4.0,
        };
        assert!(!left_cens.is_exact());
        assert!(left_cens.is_left_censored());
    }

    // -----------------------------------------------------------------------
    // Test 19: validate_truncated rejects empty slice
    // -----------------------------------------------------------------------
    #[test]
    fn validate_truncated_empty_returns_error() {
        let obs: Vec<TruncatedObs> = Vec::new();
        assert!(validate_truncated(&obs, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Test 20: Turnbull empty input returns error
    // -----------------------------------------------------------------------
    #[test]
    fn turnbull_empty_input_returns_error() {
        let obs: Vec<IntervalObs> = Vec::new();
        assert!(turnbull_em(&obs, 0, 100, 1.0e-6).is_err());
    }
}

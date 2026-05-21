//! Hidden Semi-Markov Model (HSMM) with explicit duration distributions.
//!
//! Reference: Yu 2010, "Hidden semi-Markov models", Artificial Intelligence
//! 174:215–243, §2.2.
//!
//! Unlike a standard HMM (which forces geometric sojourn times), an HSMM
//! models an explicit duration distribution d_j(τ) = P(stay in state j for
//! exactly τ consecutive steps).  The transition matrix A has zeros on the
//! diagonal (self-transitions are absorbed into the duration model); each row
//! sums to 1 over the off-diagonal entries.
//!
//! The implementation follows the *state-completion* forward-backward
//! formulation of Yu 2010:
//!
//!   e_j(t)  = P(o_1..o_t, a segment in state j completes at time t)
//!   f_j(t)  = P(o_{t+1}..o_T | current-segment in state j ended at t)

use crate::error::{SeqError, SeqResult};

// ─── logsumexp helper ─────────────────────────────────────────────────────────

#[inline]
fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let s: f64 = xs.iter().map(|&x| (x - m).exp()).sum();
    m + s.ln()
}

#[inline]
fn log_safe(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        f64::NEG_INFINITY
    } else {
        x.ln()
    }
}

// ─── Duration distributions ───────────────────────────────────────────────────

/// Duration distribution for a single state.
///
/// All distributions are truncated to `max_dur` in the forward-backward
/// algorithm; `max_dur` is passed as an argument to `prob` / `log_prob` to
/// facilitate normalisation where needed.
#[derive(Debug, Clone)]
pub enum DurationDistrib {
    /// Poisson duration: P(dur = τ) ∝ λ^τ exp(−λ) / τ!, τ ≥ 1.
    /// Truncated and renormalised over τ = 1..max_dur.
    Poisson { lambda: f64 },
    /// Geometric: P(dur = τ) = p · (1−p)^{τ−1}, τ ≥ 1.
    Geometric { p: f64 },
    /// Histogram: `probs[τ−1]` = P(dur = τ), τ = 1..len.
    Histogram { probs: Vec<f64> },
}

impl DurationDistrib {
    /// Probability of duration `tau` (≥ 1).
    ///
    /// For `Poisson`, the distribution is truncated and renormalised to the
    /// range τ = 1..`max_dur` so that probabilities sum to 1 within the
    /// algorithm's horizon.
    pub fn prob(&self, tau: usize, max_dur: usize) -> f64 {
        if tau == 0 {
            return 0.0;
        }
        match self {
            DurationDistrib::Poisson { lambda } => {
                if *lambda <= 0.0 {
                    return if tau == 1 { 1.0 } else { 0.0 };
                }
                // Compute unnormalised Poisson pmf at tau; normalise over 1..max_dur.
                let raw = poisson_pmf(*lambda, tau);
                let total: f64 = (1..=max_dur).map(|d| poisson_pmf(*lambda, d)).sum();
                if total > 0.0 { raw / total } else { 0.0 }
            }
            DurationDistrib::Geometric { p } => {
                let p = p.clamp(0.0, 1.0);
                p * (1.0 - p).powi((tau - 1) as i32)
            }
            DurationDistrib::Histogram { probs } => {
                if tau <= probs.len() {
                    probs[tau - 1]
                } else {
                    0.0
                }
            }
        }
    }

    /// Log probability of duration `tau`; returns −∞ if `prob` = 0.
    pub fn log_prob(&self, tau: usize, max_dur: usize) -> f64 {
        log_safe(self.prob(tau, max_dur))
    }
}

/// Unnormalised Poisson pmf at non-negative integer k.
fn poisson_pmf(lambda: f64, k: usize) -> f64 {
    if lambda <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    // Compute in log space then exponentiate to avoid overflow for large k.
    let log_p = (k as f64) * lambda.ln() - lambda - log_factorial(k);
    log_p.exp()
}

/// ln(k!) computed via Stirling / iterative sum for k ≤ ~20, else lgamma.
fn log_factorial(k: usize) -> f64 {
    if k <= 1 {
        return 0.0;
    }
    (1..=k).map(|i| (i as f64).ln()).sum()
}

// ─── Model struct ─────────────────────────────────────────────────────────────

/// Hidden Semi-Markov Model with explicit per-state duration distributions.
#[derive(Debug, Clone)]
pub struct Hsmm {
    /// Number of hidden states.
    pub n_states: usize,
    /// Number of distinct observation symbols.
    pub n_obs: usize,
    /// Maximum segment duration (D_max).
    pub max_dur: usize,
    /// Initial state probability vector (n_states,).
    pub pi: Vec<f64>,
    /// Transition matrix without self-transitions, row-major (n_states × n_states).
    /// Diagonal entries must be 0; each row sums to 1.
    pub a: Vec<f64>,
    /// Emission matrix, row-major (n_states × n_obs).  Each row sums to 1.
    pub b: Vec<f64>,
    /// Duration distribution for each state.
    pub dur: Vec<DurationDistrib>,
}

impl Hsmm {
    /// Construct and validate an HSMM.
    pub fn new(
        n_states: usize,
        n_obs: usize,
        max_dur: usize,
        pi: Vec<f64>,
        a: Vec<f64>,
        b: Vec<f64>,
        dur: Vec<DurationDistrib>,
    ) -> SeqResult<Self> {
        if n_states == 0 || n_obs == 0 || max_dur == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_states, n_obs, and max_dur must all be > 0".to_string(),
            ));
        }
        if pi.len() != n_states {
            return Err(SeqError::ShapeMismatch {
                expected: n_states,
                got: pi.len(),
            });
        }
        if a.len() != n_states * n_states {
            return Err(SeqError::ShapeMismatch {
                expected: n_states * n_states,
                got: a.len(),
            });
        }
        if b.len() != n_states * n_obs {
            return Err(SeqError::ShapeMismatch {
                expected: n_states * n_obs,
                got: b.len(),
            });
        }
        if dur.len() != n_states {
            return Err(SeqError::ShapeMismatch {
                expected: n_states,
                got: dur.len(),
            });
        }

        // Validate diagonal = 0
        for i in 0..n_states {
            let diag = a[i * n_states + i];
            if diag.abs() > 1e-9 {
                return Err(SeqError::InvalidConfiguration(format!(
                    "transition matrix diagonal A[{i},{i}] = {diag} must be 0"
                )));
            }
        }

        // Validate π sums to ~1.
        let pi_sum: f64 = pi.iter().sum();
        if (pi_sum - 1.0).abs() > 1e-5 {
            return Err(SeqError::InvalidConfiguration(format!(
                "pi sums to {pi_sum}, expected 1"
            )));
        }
        // Validate A rows (only for n_states > 1; single-state has all-zero row).
        if n_states > 1 {
            for i in 0..n_states {
                let s: f64 = a[i * n_states..(i + 1) * n_states].iter().sum();
                if (s - 1.0).abs() > 1e-5 {
                    return Err(SeqError::InvalidConfiguration(format!(
                        "A row {i} sums to {s}, expected 1"
                    )));
                }
            }
        }
        // Validate B rows.
        for j in 0..n_states {
            let s: f64 = b[j * n_obs..(j + 1) * n_obs].iter().sum();
            if (s - 1.0).abs() > 1e-5 {
                return Err(SeqError::InvalidConfiguration(format!(
                    "B row {j} sums to {s}, expected 1"
                )));
            }
        }

        Ok(Self {
            n_states,
            n_obs,
            max_dur,
            pi,
            a,
            b,
            dur,
        })
    }

    /// Compute log P(o₁..o_T | model) using the HSMM forward algorithm.
    pub fn log_likelihood(&self, obs: &[usize]) -> SeqResult<f64> {
        if obs.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        for &o in obs {
            if o >= self.n_obs {
                return Err(SeqError::InvalidObservation(format!(
                    "observation {o} >= n_obs {}",
                    self.n_obs
                )));
            }
        }
        let (_, log_z) = hsmm_forward(self, obs);
        Ok(log_z)
    }

    /// Viterbi decoding: return the most likely state sequence.
    pub fn decode(&self, obs: &[usize]) -> SeqResult<Vec<usize>> {
        if obs.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        for &o in obs {
            if o >= self.n_obs {
                return Err(SeqError::InvalidObservation(format!(
                    "observation {o} >= n_obs {}",
                    self.n_obs
                )));
            }
        }
        hsmm_viterbi(self, obs)
    }
}

// ─── Pre-compute cumulative log-emission sums ──────────────────────────────────

/// Build cumulative log-emission sums for each state and each time range.
///
/// `cum_log_b[j * (t_max+1) + t]` = Σ_{u=0}^{t-1} log B_j(o_u)
/// so that the sum over t1..=t2 (0-indexed) = cum[t2+1] − cum[t1].
fn build_cum_log_b(model: &Hsmm, obs: &[usize]) -> Vec<f64> {
    let t_max = obs.len();
    let n = model.n_states;
    // Shape: n × (t_max+1)
    let mut cum = vec![0.0f64; n * (t_max + 1)];
    for j in 0..n {
        for t in 0..t_max {
            let le = log_safe(model.b[j * model.n_obs + obs[t]]);
            cum[j * (t_max + 1) + t + 1] = cum[j * (t_max + 1) + t] + le;
        }
    }
    cum
}

/// Log emission of state j for the segment obs[t1..=t2] (0-indexed, inclusive).
#[inline]
fn seg_log_em(cum: &[f64], j: usize, t1: usize, t2: usize, t_max: usize) -> f64 {
    cum[j * (t_max + 1) + t2 + 1] - cum[j * (t_max + 1) + t1]
}

// ─── HSMM forward algorithm ───────────────────────────────────────────────────

/// Compute the HSMM forward variables using the state-completion form.
///
/// Returns `(log_e, log_z)` where:
///   `log_e[j * t_max + t]` = log e_j(t+1)  (1-indexed time, 0-indexed array)
///   `log_z` = log p(o_1..o_T | model)
fn hsmm_forward(model: &Hsmm, obs: &[usize]) -> (Vec<f64>, f64) {
    let t_max = obs.len();
    let n = model.n_states;
    let d_max = model.max_dur.min(t_max);
    let cum = build_cum_log_b(model, obs);

    // log_e[j, t] = log e_j(t+1) where t is 0-based (t+1 is the 1-based time
    // at which a segment in state j completes).
    let mut log_e = vec![f64::NEG_INFINITY; n * t_max];

    // log_pi[j] = log π_j
    let log_pi: Vec<f64> = model.pi.iter().map(|&p| log_safe(p)).collect();

    // log_a[i, j] = log A_{ij} (i→j off-diagonal)
    let log_a: Vec<f64> = model.a.iter().map(|&v| log_safe(v)).collect();

    // We buffer logsumexp terms.
    let mut terms = Vec::with_capacity(d_max * n);

    for t in 0..t_max {
        // t is the 0-based end-time index; the 1-based version is t+1.
        for j in 0..n {
            terms.clear();
            let d_end = (t + 1).min(d_max); // max segment duration ending at t
            for d in 1..=d_end {
                // Segment spans obs[t-d+1 .. t] (0-based, inclusive), length d.
                let t_start = t + 1 - d; // 0-based start index
                let log_dur = model.dur[j].log_prob(d, model.max_dur);
                let log_em_seg = seg_log_em(&cum, j, t_start, t, t_max);

                // Compute init term: log π_j (if d = t+1, i.e. segment starts at 0)
                //                    + logsumexp_{i≠j}(log A_{ij} + log e_i(t-d))
                let log_init = if d == t + 1 {
                    // Segment starts at the very beginning of the sequence.
                    log_pi[j]
                } else {
                    // t_start > 0 → must have come from a previous segment ending at t_start-1.
                    let prev_t = t_start - 1; // 0-based
                    let mut trans_terms: Vec<f64> = Vec::with_capacity(n);
                    for i in 0..n {
                        if i == j {
                            continue;
                        }
                        let log_prev = log_e[i * t_max + prev_t];
                        if log_prev == f64::NEG_INFINITY {
                            continue;
                        }
                        trans_terms.push(log_a[i * n + j] + log_prev);
                    }
                    if trans_terms.is_empty() {
                        f64::NEG_INFINITY
                    } else {
                        logsumexp(&trans_terms)
                    }
                };

                if log_init == f64::NEG_INFINITY || log_dur == f64::NEG_INFINITY {
                    continue;
                }
                terms.push(log_dur + log_em_seg + log_init);
            }
            log_e[j * t_max + t] = if terms.is_empty() {
                f64::NEG_INFINITY
            } else {
                logsumexp(&terms)
            };
        }
    }

    // log Z = logsumexp_j log_e_j(T)  (last time step, 0-based index t_max-1)
    let last_terms: Vec<f64> = (0..n).map(|j| log_e[j * t_max + t_max - 1]).collect();
    let log_z = logsumexp(&last_terms);

    (log_e, log_z)
}

// ─── HSMM backward algorithm ─────────────────────────────────────────────────

/// Compute HSMM backward variables.
///
/// `log_f[j, t]` = log f_j(t+1) (probability of future observations given a
/// segment in state j completed at 1-based time t+1).
///
/// Boundary: log_f[j, T-1] = 0 for all j (1-based T = t_max).
fn hsmm_backward(model: &Hsmm, obs: &[usize], cum: &[f64]) -> Vec<f64> {
    let t_max = obs.len();
    let n = model.n_states;
    let d_max = model.max_dur;

    let log_a: Vec<f64> = model.a.iter().map(|&v| log_safe(v)).collect();

    // log_f[j * t_max + t] = log f_j(t+1)
    let mut log_f = vec![f64::NEG_INFINITY; n * t_max];

    // Boundary: f_j(T) = 1 → log = 0.
    for j in 0..n {
        log_f[j * t_max + t_max - 1] = 0.0;
    }

    // Recursion: for t = T-2 down to 0 (0-based).
    // f_j(t) = Σ_{k≠j} Σ_{d=1}^{min(D, T-(t+1))} A_{jk} * dur_k(d) * Π_em_k(t+1..t+d) * f_k(t+d)
    // Note: t+1 to t+d are 0-based indices t+1-1..t+d-1 = t..t+d-1.
    let mut terms: Vec<f64> = Vec::with_capacity(d_max * n);

    for t in (0..t_max.saturating_sub(1)).rev() {
        for j in 0..n {
            terms.clear();
            // Remaining time steps after t: from t+1 (0-based) to t_max-1.
            let remaining = t_max - t - 1; // number of steps after t
            let d_end = d_max.min(remaining);

            for k in 0..n {
                if k == j {
                    continue;
                }
                let log_ajk = log_a[j * n + k];
                if log_ajk == f64::NEG_INFINITY {
                    continue;
                }
                for d in 1..=d_end {
                    // Segment in k spans obs[t+1 .. t+d] (0-based), i.e. t_start=t+1, t_end=t+d.
                    let t_start_new = t + 1;
                    let t_end_new = t + d; // 0-based inclusive
                    let log_dur = model.dur[k].log_prob(d, model.max_dur);
                    let log_em_seg = seg_log_em(cum, k, t_start_new, t_end_new, t_max);
                    let log_f_next = log_f[k * t_max + t_end_new];
                    if log_dur == f64::NEG_INFINITY || log_f_next == f64::NEG_INFINITY {
                        continue;
                    }
                    terms.push(log_ajk + log_dur + log_em_seg + log_f_next);
                }
            }

            log_f[j * t_max + t] = if terms.is_empty() {
                f64::NEG_INFINITY
            } else {
                logsumexp(&terms)
            };
        }
    }

    log_f
}

// ─── HSMM Viterbi ─────────────────────────────────────────────────────────────

/// HSMM Viterbi: max-product version of the forward algorithm.
/// Returns the most-likely state sequence of length T.
fn hsmm_viterbi(model: &Hsmm, obs: &[usize]) -> SeqResult<Vec<usize>> {
    let t_max = obs.len();
    let n = model.n_states;
    let d_max = model.max_dur.min(t_max);
    let cum = build_cum_log_b(model, obs);

    let log_pi: Vec<f64> = model.pi.iter().map(|&p| log_safe(p)).collect();
    let log_a: Vec<f64> = model.a.iter().map(|&v| log_safe(v)).collect();

    // v[j, t] = max log-prob of a path in which the segment containing time t
    //           ends at t in state j.
    let mut log_v = vec![f64::NEG_INFINITY; n * t_max];

    // Backpointer: for each (j, t) the best (d, prev_state) that achieves log_v[j,t].
    // We store (d, prev_j) where d = segment duration, prev_j = previous state (n if init).
    let mut bp_d = vec![0usize; n * t_max];
    let mut bp_prev = vec![n; n * t_max]; // n means "start of sequence"

    for t in 0..t_max {
        for j in 0..n {
            let d_end = (t + 1).min(d_max);
            let mut best_val = f64::NEG_INFINITY;
            let mut best_d = 1;
            let mut best_prev = n; // sentinel = sequence start

            for d in 1..=d_end {
                let t_start = t + 1 - d;
                let log_dur = model.dur[j].log_prob(d, model.max_dur);
                let log_em_seg = seg_log_em(&cum, j, t_start, t, t_max);

                if log_dur == f64::NEG_INFINITY {
                    continue;
                }

                let log_seg_cost = log_dur + log_em_seg;

                if d == t + 1 {
                    // Segment starts at beginning.
                    let v = log_seg_cost + log_pi[j];
                    if v > best_val {
                        best_val = v;
                        best_d = d;
                        best_prev = n; // sentinel
                    }
                } else {
                    let prev_t = t_start - 1;
                    for i in 0..n {
                        if i == j {
                            continue;
                        }
                        let log_prev_v = log_v[i * t_max + prev_t];
                        if log_prev_v == f64::NEG_INFINITY {
                            continue;
                        }
                        let v = log_seg_cost + log_a[i * n + j] + log_prev_v;
                        if v > best_val {
                            best_val = v;
                            best_d = d;
                            best_prev = i;
                        }
                    }
                }
            }

            log_v[j * t_max + t] = best_val;
            bp_d[j * t_max + t] = best_d;
            bp_prev[j * t_max + t] = best_prev;
        }
    }

    // Termination: find best final state.
    let last_t = t_max - 1;
    let mut best_final = f64::NEG_INFINITY;
    let mut best_j = 0;
    for j in 0..n {
        let v = log_v[j * t_max + last_t];
        if v > best_final {
            best_final = v;
            best_j = j;
        }
    }

    if best_final == f64::NEG_INFINITY {
        // All paths have zero probability; fall back to uniform state 0.
        return Ok(vec![0usize; t_max]);
    }

    // Backtrack to fill the state path.
    let mut path = vec![0usize; t_max];
    let mut cur_t = last_t as isize;
    let mut cur_j = best_j;

    while cur_t >= 0 {
        let t = cur_t as usize;
        let d = bp_d[cur_j * t_max + t];
        let t_start = t + 1 - d;

        // Fill states t_start..=t with cur_j.
        for u in t_start..=t {
            path[u] = cur_j;
        }

        if t_start == 0 {
            break;
        }
        let prev_state = bp_prev[cur_j * t_max + t];
        if prev_state == n {
            // Sentinel: segment starts at sequence start.
            break;
        }
        cur_t = (t_start as isize) - 1;
        cur_j = prev_state;
    }

    Ok(path)
}

// ─── Configuration & result types ─────────────────────────────────────────────

/// Configuration for HSMM EM training.
#[derive(Debug, Clone)]
pub struct HsmConfig {
    /// Number of hidden states.
    pub n_states: usize,
    /// Number of distinct observation symbols.
    pub n_obs: usize,
    /// Maximum segment duration D_max (default 10).
    pub max_dur: usize,
    /// Maximum EM iterations (default 100).
    pub max_iter: usize,
    /// Convergence tolerance on log-likelihood change (default 1e-5).
    pub tol: f64,
}

impl Default for HsmConfig {
    fn default() -> Self {
        Self {
            n_states: 2,
            n_obs: 2,
            max_dur: 10,
            max_iter: 100,
            tol: 1e-5,
        }
    }
}

/// Result of HSMM EM fitting.
#[derive(Debug, Clone)]
pub struct HsmResult {
    /// Fitted HSMM model.
    pub model: Hsmm,
    /// Log-likelihood at each iteration.
    pub log_likelihood_history: Vec<f64>,
    /// Number of EM iterations executed.
    pub n_iter: usize,
    /// Whether the algorithm converged within the tolerance.
    pub converged: bool,
}

// ─── EM helper: build initial model ──────────────────────────────────────────

fn build_initial_model(cfg: &HsmConfig) -> SeqResult<Hsmm> {
    let n = cfg.n_states;
    let k = cfg.n_obs;
    let d_max = cfg.max_dur;

    // π: uniform.
    let pi: Vec<f64> = vec![1.0 / n as f64; n];

    // A: uniform off-diagonal (diagonal = 0).
    let mut a = vec![0.0f64; n * n];
    if n > 1 {
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j { 0.0 } else { 1.0 / (n as f64 - 1.0) };
            }
        }
    }

    // B: state-indexed perturbation so states are distinguishable.
    let mut b = vec![0.0f64; n * k];
    for j in 0..n {
        let mut row_sum = 0.0f64;
        for sym in 0..k {
            // Assign slightly higher weight to symbol = (j % k) for state j.
            let base = 1.0 / k as f64;
            let bump = if sym == j % k { 0.2 / k as f64 } else { 0.0 };
            b[j * k + sym] = base + bump;
            row_sum += b[j * k + sym];
        }
        // Normalise row.
        for sym in 0..k {
            b[j * k + sym] /= row_sum;
        }
    }

    // Duration: Geometric with p = 1 / max_dur for each state.
    let p = 1.0 / d_max.max(1) as f64;
    let dur: Vec<DurationDistrib> = (0..n).map(|_| DurationDistrib::Geometric { p }).collect();

    Hsmm::new(n, k, d_max, pi, a, b, dur)
}

// ─── Main EM entry point ───────────────────────────────────────────────────────

/// Fit an HSMM by EM on one or more observation sequences.
pub fn hsm_fit(observations: &[&[usize]], cfg: &HsmConfig) -> SeqResult<HsmResult> {
    if observations.is_empty() || observations.iter().all(|s| s.is_empty()) {
        return Err(SeqError::EmptyInput);
    }
    if cfg.n_states == 0 || cfg.n_obs == 0 || cfg.max_dur == 0 {
        return Err(SeqError::InvalidConfiguration(
            "n_states, n_obs, and max_dur must all be > 0".to_string(),
        ));
    }
    for seq in observations.iter() {
        for &o in *seq {
            if o >= cfg.n_obs {
                return Err(SeqError::InvalidObservation(format!(
                    "observation {o} >= n_obs {}",
                    cfg.n_obs
                )));
            }
        }
    }

    let n = cfg.n_states;
    let k = cfg.n_obs;
    let d_max = cfg.max_dur;

    let mut model = build_initial_model(cfg)?;
    let mut history: Vec<f64> = Vec::with_capacity(cfg.max_iter + 1);
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0usize;

    for iter in 0..cfg.max_iter {
        n_iter = iter + 1;

        // ── E-step: collect sufficient statistics ──
        // For π: Σ γ_j(segment starts at t=0 and ends at t=d-1)
        let mut ss_pi = vec![0.0f64; n];
        // For A: Σ_{t,d} γ_i→j (i≠j) transition at time t.
        let mut ss_a = vec![0.0f64; n * n];
        // For B: Σ_{t: o_t=sym} occupancy of state j at time t.
        let mut ss_b = vec![0.0f64; n * k];
        // For duration: ss_dur[j * (d_max+1) + d] = Σ expected # of segs of length d in state j.
        let mut ss_dur = vec![0.0f64; n * (d_max + 1)];

        let mut total_ll = 0.0f64;

        for seq in observations.iter() {
            if seq.is_empty() {
                continue;
            }
            let t_max = seq.len();
            let cum = build_cum_log_b(&model, seq);
            let (log_e, log_z) = hsmm_forward(&model, seq);
            let log_f = hsmm_backward(&model, seq, &cum);

            if !log_z.is_finite() {
                // Sequence had probability 0 under current model; skip accumulation.
                continue;
            }

            total_ll += log_z;

            // Compute segment posteriors γ_{j,t_start,d}.
            // P(segment j, t_start, d | obs) ∝ init_or_trans * dur_j(d) * em_j * f_j(t_end)
            //
            // We iterate over all possible segments (j, t_start, d).
            let log_pi_v: Vec<f64> = model.pi.iter().map(|&p| log_safe(p)).collect();
            let log_a_v: Vec<f64> = model.a.iter().map(|&v| log_safe(v)).collect();

            for j in 0..n {
                for t_end in 0..t_max {
                    for d in 1..=(t_end + 1).min(d_max) {
                        let t_start = t_end + 1 - d;
                        let log_dur = model.dur[j].log_prob(d, d_max);
                        if log_dur == f64::NEG_INFINITY {
                            continue;
                        }
                        let log_em_seg = seg_log_em(&cum, j, t_start, t_end, t_max);

                        // Init term: from log_e perspective, e_j(t_end) = sum over d of terms;
                        // here we need the "slice" contribution.
                        let log_init = if t_start == 0 {
                            log_pi_v[j]
                        } else {
                            let prev_t = t_start - 1;
                            let mut terms: Vec<f64> = Vec::with_capacity(n);
                            for i in 0..n {
                                if i == j {
                                    continue;
                                }
                                let lv = log_e[i * t_max + prev_t];
                                if lv == f64::NEG_INFINITY {
                                    continue;
                                }
                                terms.push(log_a_v[i * n + j] + lv);
                            }
                            if terms.is_empty() {
                                f64::NEG_INFINITY
                            } else {
                                logsumexp(&terms)
                            }
                        };

                        if log_init == f64::NEG_INFINITY {
                            continue;
                        }

                        let log_f_val = log_f[j * t_max + t_end];
                        if log_f_val == f64::NEG_INFINITY {
                            continue;
                        }

                        let log_gamma_seg = log_init + log_dur + log_em_seg + log_f_val - log_z;
                        let gamma_seg = log_gamma_seg.exp();

                        if !gamma_seg.is_finite() || gamma_seg <= 0.0 {
                            continue;
                        }

                        // Accumulate π.
                        if t_start == 0 {
                            ss_pi[j] += gamma_seg;
                        }

                        // Accumulate duration.
                        ss_dur[j * (d_max + 1) + d] += gamma_seg;

                        // Accumulate emission (each time step in the segment).
                        for u in t_start..=t_end {
                            ss_b[j * k + seq[u]] += gamma_seg;
                        }

                        // Accumulate transitions from i to j (for t_start > 0).
                        if t_start > 0 {
                            let prev_t = t_start - 1;
                            for i in 0..n {
                                if i == j {
                                    continue;
                                }
                                let lv = log_e[i * t_max + prev_t];
                                if lv == f64::NEG_INFINITY {
                                    continue;
                                }
                                let log_xi =
                                    log_a_v[i * n + j] + lv + log_dur + log_em_seg + log_f_val
                                        - log_z;
                                let xi_val = log_xi.exp();
                                if xi_val.is_finite() && xi_val > 0.0 {
                                    ss_a[i * n + j] += xi_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        history.push(total_ll);

        // Convergence check.
        if iter > 0 && (total_ll - prev_ll).abs() < cfg.tol {
            converged = true;
            break;
        }
        prev_ll = total_ll;

        // ── M-step ──

        // Update π.
        let pi_sum: f64 = ss_pi.iter().sum();
        let new_pi: Vec<f64> = if pi_sum > 0.0 {
            ss_pi.iter().map(|&v| v / pi_sum).collect()
        } else {
            vec![1.0 / n as f64; n]
        };

        // Update A (normalise each row, zero diagonal).
        let mut new_a = vec![0.0f64; n * n];
        if n > 1 {
            for i in 0..n {
                let row_sum: f64 = ss_a[i * n..(i + 1) * n].iter().sum();
                for j in 0..n {
                    if i == j {
                        new_a[i * n + j] = 0.0;
                    } else {
                        new_a[i * n + j] = if row_sum > 0.0 {
                            ss_a[i * n + j] / row_sum
                        } else {
                            1.0 / (n as f64 - 1.0)
                        };
                    }
                }
            }
        }

        // Update B (normalise each row).
        let mut new_b = vec![0.0f64; n * k];
        for j in 0..n {
            let row_sum: f64 = ss_b[j * k..(j + 1) * k].iter().sum();
            for sym in 0..k {
                new_b[j * k + sym] = if row_sum > 0.0 {
                    ss_b[j * k + sym] / row_sum
                } else {
                    1.0 / k as f64
                };
            }
        }

        // Update duration distributions (histogram form).
        let mut new_dur: Vec<DurationDistrib> = Vec::with_capacity(n);
        for j in 0..n {
            let total: f64 = ss_dur[j * (d_max + 1) + 1..=(j * (d_max + 1) + d_max)]
                .iter()
                .sum();
            let probs: Vec<f64> = if total > 0.0 {
                (1..=d_max)
                    .map(|d| ss_dur[j * (d_max + 1) + d] / total)
                    .collect()
            } else {
                // Fall back to geometric if no data.
                let p = 1.0 / d_max as f64;
                (1..=d_max)
                    .map(|d| {
                        let geo = DurationDistrib::Geometric { p };
                        geo.prob(d, d_max)
                    })
                    .collect()
            };
            new_dur.push(DurationDistrib::Histogram { probs });
        }

        // Install updated model (using unchecked path for diagonal since we built it correctly).
        model = Hsmm {
            n_states: n,
            n_obs: k,
            max_dur: d_max,
            pi: new_pi,
            a: new_a,
            b: new_b,
            dur: new_dur,
        };
    }

    Ok(HsmResult {
        model,
        log_likelihood_history: history,
        n_iter,
        converged,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DurationDistrib tests ──────────────────────────────────────────────────

    #[test]
    fn poisson_probs_sum_to_one() {
        // With truncation to max_dur=20.
        let d = DurationDistrib::Poisson { lambda: 3.0 };
        let s: f64 = (1..=20).map(|t| d.prob(t, 20)).sum();
        assert!((s - 1.0).abs() < 1e-9, "Poisson prob sum = {s}");
    }

    #[test]
    fn geometric_probs_approx_one() {
        let d = DurationDistrib::Geometric { p: 0.3 };
        // Over d=1..1000 the geometric CDF should be essentially 1.
        let s: f64 = (1..=1000).map(|t| d.prob(t, 1000)).sum();
        assert!((s - 1.0).abs() < 1e-9, "Geometric prob sum = {s}");
    }

    #[test]
    fn histogram_probs_sum_to_one() {
        let probs = vec![0.2, 0.5, 0.3];
        let d = DurationDistrib::Histogram { probs };
        let s: f64 = (1..=3).map(|t| d.prob(t, 3)).sum();
        assert!((s - 1.0).abs() < 1e-9, "Histogram prob sum = {s}");
    }

    #[test]
    fn poisson_log_prob_finite_for_positive_lambda() {
        let d = DurationDistrib::Poisson { lambda: 2.0 };
        let lp = d.log_prob(1, 10);
        assert!(lp.is_finite(), "Poisson log_prob(1, 10) = {lp}");
    }

    #[test]
    fn geometric_prob_decreasing() {
        let d = DurationDistrib::Geometric { p: 0.5 };
        for t in 1..=5 {
            assert!(
                d.prob(t, 20) > d.prob(t + 1, 20),
                "Geometric should be decreasing"
            );
        }
    }

    // ── Hsmm construction tests ────────────────────────────────────────────────

    fn two_state_model() -> Hsmm {
        Hsmm::new(
            2,
            2,
            5,
            vec![0.5, 0.5],
            vec![0.0, 1.0, 1.0, 0.0],
            vec![0.9, 0.1, 0.1, 0.9],
            vec![
                DurationDistrib::Geometric { p: 0.3 },
                DurationDistrib::Geometric { p: 0.3 },
            ],
        )
        .expect("valid model")
    }

    #[test]
    fn hsmm_new_validates_shapes() {
        // Wrong pi length.
        assert!(
            Hsmm::new(
                2,
                2,
                5,
                vec![1.0],
                vec![0.0, 1.0, 1.0, 0.0],
                vec![0.5, 0.5, 0.5, 0.5],
                vec![DurationDistrib::Geometric { p: 0.5 }; 2],
            )
            .is_err()
        );
    }

    #[test]
    fn hsmm_new_rejects_nonzero_diagonal() {
        // Non-zero diagonal.
        assert!(
            Hsmm::new(
                2,
                2,
                5,
                vec![0.5, 0.5],
                vec![0.5, 0.5, 0.5, 0.5], // diagonal = 0.5 (invalid)
                vec![0.5, 0.5, 0.5, 0.5],
                vec![DurationDistrib::Geometric { p: 0.5 }; 2],
            )
            .is_err()
        );
    }

    #[test]
    fn log_likelihood_finite_for_valid_obs() {
        let m = two_state_model();
        let ll = m.log_likelihood(&[0, 1, 0, 1]).expect("should succeed");
        assert!(ll.is_finite(), "ll = {ll}");
    }

    #[test]
    fn log_likelihood_err_for_empty_obs() {
        let m = two_state_model();
        assert!(m.log_likelihood(&[]).is_err());
    }

    #[test]
    fn log_likelihood_err_for_obs_out_of_range() {
        let m = two_state_model();
        assert!(m.log_likelihood(&[0, 5]).is_err());
    }

    #[test]
    fn decode_returns_sequence_of_correct_length() {
        let m = two_state_model();
        let obs = vec![0usize, 0, 1, 1, 0];
        let path = m.decode(&obs).expect("ok");
        assert_eq!(path.len(), obs.len());
    }

    #[test]
    fn decode_all_same_when_one_state_dominates() {
        // In an HSMM, A diagonal must be 0 (no self-transitions), so a 2-state
        // model alternates.  Use a single-state model to verify that observing
        // a symbol repeatedly maps only to that one state.
        let m = Hsmm::new(
            1, // single state
            2,
            3,
            vec![1.0],          // pi
            vec![0.0],          // A (1×1, diagonal=0)
            vec![0.999, 0.001], // B: state 0 strongly emits symbol 0
            vec![DurationDistrib::Geometric { p: 0.5 }],
        )
        .expect("ok");
        let obs = vec![0usize; 4];
        let path = m.decode(&obs).expect("ok");
        assert!(
            path.iter().all(|&s| s == 0),
            "expected all state 0, got {:?}",
            path
        );
    }

    // ── hsm_fit tests ──────────────────────────────────────────────────────────

    #[test]
    fn hsm_fit_runs_without_error() {
        let obs = vec![0usize, 0, 1, 1, 0, 1, 0, 0, 1, 1];
        let cfg = HsmConfig::default();
        assert!(hsm_fit(&[&obs], &cfg).is_ok());
    }

    #[test]
    fn hsm_fit_ll_non_decreasing() {
        let obs: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let cfg = HsmConfig {
            max_iter: 30,
            ..Default::default()
        };
        let r = hsm_fit(&[&obs], &cfg).expect("ok");
        for w in r.log_likelihood_history.windows(2) {
            assert!(w[1] >= w[0] - 1e-4, "LL decreased: {} → {}", w[0], w[1]);
        }
    }

    #[test]
    fn hsm_fit_converged_flag() {
        let obs: Vec<usize> = (0..50).map(|i| i % 2).collect();
        let cfg = HsmConfig {
            max_iter: 500,
            tol: 1e-3,
            ..Default::default()
        };
        let r = hsm_fit(&[&obs], &cfg).expect("ok");
        assert!(r.converged, "expected convergence");
    }

    #[test]
    fn hsm_fit_result_pi_sums_to_one() {
        let obs = vec![0usize, 1, 0, 1, 0, 0];
        let cfg = HsmConfig::default();
        let r = hsm_fit(&[&obs], &cfg).expect("ok");
        let s: f64 = r.model.pi.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "pi sums to {s}");
    }

    #[test]
    fn hsm_fit_result_b_rows_sum_to_one() {
        let obs = vec![0usize, 1, 0, 1, 0, 0];
        let cfg = HsmConfig::default();
        let r = hsm_fit(&[&obs], &cfg).expect("ok");
        let n = cfg.n_states;
        let k = cfg.n_obs;
        for j in 0..n {
            let s: f64 = r.model.b[j * k..(j + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "B row {j} sums to {s}");
        }
    }

    #[test]
    fn hsm_fit_n_iter_within_max_iter() {
        let obs = vec![0usize, 1, 0, 1, 0, 0];
        let cfg = HsmConfig {
            max_iter: 10,
            ..Default::default()
        };
        let r = hsm_fit(&[&obs], &cfg).expect("ok");
        assert!(r.n_iter <= 10);
    }

    #[test]
    fn hsm_fit_multiple_sequences() {
        let s1 = vec![0usize, 0, 1, 1];
        let s2 = vec![1usize, 0, 1, 0, 0];
        let s3 = vec![0usize, 1, 1, 0, 1, 0];
        let cfg = HsmConfig::default();
        assert!(hsm_fit(&[&s1, &s2, &s3], &cfg).is_ok());
    }

    #[test]
    fn hsm_fit_short_sequence_length_one() {
        let obs = vec![0usize];
        let cfg = HsmConfig::default();
        let r = hsm_fit(&[&obs], &cfg).expect("length-1 sequence should work");
        assert!(!r.log_likelihood_history.is_empty());
    }

    #[test]
    fn hsm_fit_max_dur_one() {
        // max_dur=1 means every segment has exactly 1 step → equivalent to standard HMM.
        let obs: Vec<usize> = (0..10).map(|i| i % 2).collect();
        let cfg = HsmConfig {
            max_dur: 1,
            ..Default::default()
        };
        let r = hsm_fit(&[&obs], &cfg).expect("max_dur=1 should work");
        assert!(!r.log_likelihood_history.is_empty());
    }

    #[test]
    fn hsmm_a_rows_zero_diagonal() {
        // After fit, diagonal should remain 0.
        let obs: Vec<usize> = (0..10).map(|i| i % 2).collect();
        let cfg = HsmConfig::default();
        let r = hsm_fit(&[&obs], &cfg).expect("ok");
        let n = cfg.n_states;
        for i in 0..n {
            let diag = r.model.a[i * n + i];
            assert!(diag.abs() < 1e-9, "diagonal A[{i},{i}] = {diag}");
        }
    }

    #[test]
    fn hsm_fit_empty_input_err() {
        let cfg = HsmConfig::default();
        assert!(hsm_fit(&[], &cfg).is_err());
    }
}

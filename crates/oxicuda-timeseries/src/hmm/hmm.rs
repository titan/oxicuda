//! Hidden Markov Models — Baum-Welch EM training and Viterbi decoding.
//!
//! Reference: Rabiner 1989 Proc. IEEE "A Tutorial on Hidden Markov Models and
//! Selected Applications in Speech Recognition"; Baum et al. 1970.
//!
//! Supports two observation models:
//! - **Discrete**: categorical emission over `M` symbols (classic HMM).
//! - **Gaussian**: univariate Gaussian emission per state (continuous HMM).
//!
//! All probability computations are performed in log-space for numerical
//! stability. The forward-backward and Viterbi algorithms use logsumexp
//! to avoid underflow on long sequences.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Types ────────────────────────────────────────────────────────────────────

/// Observation model selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HmmObsType {
    /// Categorical emission: `n_obs` symbols in `{0, …, n_obs-1}`.
    Discrete { n_obs: usize },
    /// Univariate Gaussian emission per state (mean + variance).
    Gaussian,
}

/// Training / inference configuration.
#[derive(Debug, Clone)]
pub struct HmmConfig {
    /// Number of hidden states `K`.
    pub n_states: usize,
    /// Observation model.
    pub obs_type: HmmObsType,
    /// Maximum Baum-Welch EM iterations.
    pub max_iter: usize,
    /// Log-likelihood convergence tolerance.
    pub tol: f64,
}

impl Default for HmmConfig {
    fn default() -> Self {
        Self {
            n_states: 2,
            obs_type: HmmObsType::Discrete { n_obs: 4 },
            max_iter: 100,
            tol: 1e-4,
        }
    }
}

/// A trained HMM.
///
/// Layout conventions:
/// - `pi`: length `K`.
/// - `a`: row-major `K×K`; `a[i*K+j]` = P(state j | state i).
/// - `b` (discrete): row-major `K×M`; `b[i*M+m]` = P(obs m | state i).
/// - `b` (Gaussian): flat pairs `[mu_0, sigma2_0, mu_1, sigma2_1, …]`; length `2*K`.
#[derive(Debug, Clone)]
pub struct HmmModel {
    /// Initial state distribution.
    pub pi: Vec<f64>,
    /// Transition matrix (row-major K×K).
    pub a: Vec<f64>,
    /// Emission parameters.
    pub b: Vec<f64>,
    /// Final log-likelihood.
    pub log_likelihood: f64,
    /// Number of EM iterations performed.
    pub n_iter: usize,
    /// Whether convergence was declared.
    pub converged: bool,
    /// Config snapshot used for fitting.
    pub config: HmmConfig,
}

/// Viterbi decoding output.
#[derive(Debug, Clone)]
pub struct HmmDecodeResult {
    /// MAP state sequence.
    pub states: Vec<usize>,
    /// Log-probability of the observation sequence given the model.
    pub log_prob: f64,
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Fit a discrete HMM to an integer observation sequence using Baum-Welch EM.
///
/// # Errors
/// - `InvalidNumVariates` if `n_states == 0`.
/// - `InvalidSequenceLength` if `obs.len() < 2`.
/// - `InvalidInput` if any symbol ≥ `n_obs`.
pub fn hmm_fit(obs: &[usize], config: &HmmConfig) -> TsResult<HmmModel> {
    let k = config.n_states;
    let m = match config.obs_type {
        HmmObsType::Discrete { n_obs } => n_obs,
        HmmObsType::Gaussian => {
            return Err(TsError::ShapeMismatch {
                msg: "use hmm_fit_gaussian for Gaussian HMM".into(),
            });
        }
    };
    validate_discrete(obs, k, m)?;

    let (mut pi, mut a, mut b) = init_discrete(k, m);

    let mut log_lik = f64::NEG_INFINITY;
    let mut n_iter = 0usize;
    let mut converged = false;

    for _ in 0..config.max_iter {
        n_iter += 1;
        let (new_pi, new_a, new_b, new_ll) = bw_step_discrete(obs, &pi, &a, &b, k, m);
        let delta = (new_ll - log_lik).abs();
        log_lik = new_ll;
        pi = new_pi;
        a = new_a;
        b = new_b;
        if delta < config.tol {
            converged = true;
            break;
        }
    }

    Ok(HmmModel {
        pi,
        a,
        b,
        log_likelihood: log_lik,
        n_iter,
        converged,
        config: config.clone(),
    })
}

/// Fit a continuous Gaussian HMM to a float observation sequence using Baum-Welch.
///
/// # Errors
/// - `InvalidNumVariates` if `n_states == 0`.
/// - `InvalidSequenceLength` if `obs.len() < 2`.
pub fn hmm_fit_gaussian(obs: &[f64], config: &HmmConfig) -> TsResult<HmmModel> {
    let k = config.n_states;
    if k == 0 {
        return Err(TsError::InvalidNumVariates(0));
    }
    if obs.len() < 2 {
        return Err(TsError::InvalidSequenceLength(obs.len()));
    }

    let (mut pi, mut a, mut b_gauss) = init_gaussian(obs, k);

    let mut log_lik = f64::NEG_INFINITY;
    let mut n_iter = 0usize;
    let mut converged = false;

    for _ in 0..config.max_iter {
        n_iter += 1;
        let (new_pi, new_a, new_b, new_ll) = bw_step_gaussian(obs, &pi, &a, &b_gauss, k);
        let delta = (new_ll - log_lik).abs();
        log_lik = new_ll;
        pi = new_pi;
        a = new_a;
        b_gauss = new_b;
        if delta < config.tol {
            converged = true;
            break;
        }
    }

    Ok(HmmModel {
        pi,
        a,
        b: b_gauss,
        log_likelihood: log_lik,
        n_iter,
        converged,
        config: config.clone(),
    })
}

/// Viterbi decoding for a discrete HMM.
///
/// # Errors
/// - `InvalidSequenceLength` if `obs.len() < 2`.
/// - `InvalidInput` if any symbol is out of range.
pub fn hmm_decode(model: &HmmModel, obs: &[usize]) -> TsResult<HmmDecodeResult> {
    let k = model.config.n_states;
    let m = match model.config.obs_type {
        HmmObsType::Discrete { n_obs } => n_obs,
        HmmObsType::Gaussian => {
            return Err(TsError::ShapeMismatch {
                msg: "use hmm_decode_gaussian for Gaussian HMM".into(),
            });
        }
    };
    if obs.len() < 2 {
        return Err(TsError::InvalidSequenceLength(obs.len()));
    }
    for &o in obs {
        if o >= m {
            return Err(TsError::DimensionMismatch {
                expected: m,
                got: o + 1,
            });
        }
    }
    let (states, log_prob) = viterbi_discrete(obs, &model.pi, &model.a, &model.b, k, m);
    Ok(HmmDecodeResult { states, log_prob })
}

/// Viterbi decoding for a Gaussian HMM.
///
/// # Errors
/// - `InvalidSequenceLength` if `obs.len() < 2`.
pub fn hmm_decode_gaussian(model: &HmmModel, obs: &[f64]) -> TsResult<HmmDecodeResult> {
    let k = model.config.n_states;
    if obs.len() < 2 {
        return Err(TsError::InvalidSequenceLength(obs.len()));
    }
    let (states, log_prob) = viterbi_gaussian(obs, &model.pi, &model.a, &model.b, k);
    Ok(HmmDecodeResult { states, log_prob })
}

/// Compute the log-likelihood of a discrete observation sequence under the model.
///
/// # Errors
/// - `InvalidSequenceLength` if `obs.len() < 2`.
/// - `InvalidInput` if any symbol is out of range.
pub fn hmm_log_likelihood(model: &HmmModel, obs: &[usize]) -> TsResult<f64> {
    let k = model.config.n_states;
    let m = match model.config.obs_type {
        HmmObsType::Discrete { n_obs } => n_obs,
        HmmObsType::Gaussian => {
            return Err(TsError::ShapeMismatch {
                msg: "use hmm_fit_gaussian for Gaussian HMM".into(),
            });
        }
    };
    if obs.len() < 2 {
        return Err(TsError::InvalidSequenceLength(obs.len()));
    }
    for &o in obs {
        if o >= m {
            return Err(TsError::DimensionMismatch {
                expected: m,
                got: o + 1,
            });
        }
    }
    let log_alpha = forward_discrete(obs, &model.pi, &model.a, &model.b, k, m);
    let t = obs.len();
    let ll = logsumexp(&log_alpha[(t - 1) * k..t * k]);
    Ok(ll)
}

/// Sample a sequence of length `n` from the discrete HMM using `rng`.
///
/// # Errors
/// - `InvalidSequenceLength` if `n == 0`.
pub fn hmm_generate(model: &HmmModel, n: usize, rng: &mut LcgRng) -> TsResult<Vec<usize>> {
    if n == 0 {
        return Err(TsError::InvalidSequenceLength(0));
    }
    let k = model.config.n_states;
    let m = match model.config.obs_type {
        HmmObsType::Discrete { n_obs } => n_obs,
        HmmObsType::Gaussian => {
            return Err(TsError::ShapeMismatch {
                msg: "use hmm_fit_gaussian for Gaussian HMM".into(),
            });
        }
    };

    let mut state = categorical_sample(&model.pi, rng);
    let mut seq = Vec::with_capacity(n);

    let row_b = |s: usize| &model.b[s * m..(s + 1) * m];
    let row_a = |s: usize| &model.a[s * k..(s + 1) * k];

    for _ in 0..n {
        let obs = categorical_sample(row_b(state), rng);
        seq.push(obs);
        state = categorical_sample(row_a(state), rng);
    }
    Ok(seq)
}

/// Compute the stationary distribution of the transition matrix via power iteration.
///
/// Returns a length-K probability vector `π*` such that `π* A = π*`.
#[must_use]
pub fn hmm_stationary(model: &HmmModel) -> Vec<f64> {
    let k = model.config.n_states;
    if k == 0 {
        return vec![];
    }
    let mut dist = vec![1.0f64 / k as f64; k];
    // Power iteration: multiply dist (row-vector) by A repeatedly
    for _ in 0..1000 {
        let mut next = vec![0.0f64; k];
        for (i, &d_i) in dist.iter().enumerate() {
            for (j, n_j) in next.iter_mut().enumerate() {
                *n_j += d_i * model.a[i * k + j];
            }
        }
        let s: f64 = next.iter().sum();
        if s > 1e-15 {
            for v in &mut next {
                *v /= s;
            }
        }
        let max_diff = dist
            .iter()
            .zip(next.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        dist = next;
        if max_diff < 1e-12 {
            break;
        }
    }
    dist
}

// ─── Initialisation ──────────────────────────────────────────────────────────

/// Deterministic initialisation for discrete HMM parameters.
fn init_discrete(k: usize, m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // π uniform
    let pi = vec![1.0f64 / k as f64; k];

    // A: diagonal dominant
    let mut a = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            if i == j {
                a[i * k + j] = 0.6;
            } else {
                a[i * k + j] = if k > 1 { 0.4 / (k as f64 - 1.0) } else { 0.0 };
            }
        }
    }

    // B: slightly non-uniform, row-normalised
    let mut b = vec![0.0f64; k * m];
    for i in 0..k {
        for obs in 0..m {
            // 1/M + small perturbation proportional to (i+1)*(obs+1)
            b[i * m + obs] = 1.0 + 0.1 * ((i + 1) * (obs + 1)) as f64 / (k * m) as f64;
        }
        let row_sum: f64 = (0..m).map(|obs| b[i * m + obs]).sum();
        for obs in 0..m {
            b[i * m + obs] /= row_sum;
        }
    }
    (pi, a, b)
}

/// Deterministic initialisation for Gaussian HMM parameters.
/// Splits the sorted observations into K equal segments.
fn init_gaussian(obs: &[f64], k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let pi = vec![1.0f64 / k as f64; k];

    let mut a = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            if i == j {
                a[i * k + j] = 0.6;
            } else {
                a[i * k + j] = if k > 1 { 0.4 / (k as f64 - 1.0) } else { 0.0 };
            }
        }
    }

    // Sort a copy to find segment statistics
    let mut sorted = obs.to_vec();
    sorted.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let seg = (n / k).max(1);

    // b: [mu_0, sigma2_0, mu_1, sigma2_1, ...]
    let mut b = vec![0.0f64; 2 * k];
    for i in 0..k {
        let lo = (i * seg).min(n);
        let hi = if i + 1 < k { ((i + 1) * seg).min(n) } else { n };
        let slice = if lo < hi {
            &sorted[lo..hi]
        } else {
            &sorted[lo.min(n - 1)..]
        };
        let len = slice.len() as f64;
        let mu = slice.iter().sum::<f64>() / len;
        let var = slice.iter().map(|&v| (v - mu) * (v - mu)).sum::<f64>() / len;
        b[2 * i] = mu;
        b[2 * i + 1] = var.max(1e-6);
    }
    (pi, a, b)
}

// ─── Baum-Welch ──────────────────────────────────────────────────────────────

/// One Baum-Welch EM step for a discrete HMM.
/// Returns updated (pi, A, B, log_likelihood).
fn bw_step_discrete(
    obs: &[usize],
    pi: &[f64],
    a: &[f64],
    b: &[f64],
    k: usize,
    m: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let big_t = obs.len();

    // --- Forward ---
    let log_alpha = forward_discrete(obs, pi, a, b, k, m);
    // --- Backward ---
    let log_beta = backward_discrete(obs, a, b, k, m);

    // Log-likelihood
    let log_p = logsumexp(&log_alpha[(big_t - 1) * k..big_t * k]);

    // --- E-step: gamma and xi ---
    // log_gamma[t*k+i]
    let mut log_gamma = vec![f64::NEG_INFINITY; big_t * k];
    for t in 0..big_t {
        let log_z = logsumexp_indexed(k, |i| log_alpha[t * k + i] + log_beta[t * k + i]);
        for i in 0..k {
            log_gamma[t * k + i] = log_alpha[t * k + i] + log_beta[t * k + i] - log_z;
        }
    }

    // log_xi[t*k*k + i*k + j] for t=0..T-2
    let big_t_minus_1 = big_t - 1;
    let mut log_xi = vec![f64::NEG_INFINITY; big_t_minus_1 * k * k];
    for t in 0..big_t_minus_1 {
        let ot1 = obs[t + 1];
        let log_z = logsumexp_indexed(k * k, |ij| {
            let i = ij / k;
            let j = ij % k;
            log_alpha[t * k + i]
                + log_a(a, i, j, k)
                + log_b_discrete(b, j, ot1, m)
                + log_beta[(t + 1) * k + j]
        });
        for i in 0..k {
            for j in 0..k {
                let ot1 = obs[t + 1];
                log_xi[t * k * k + i * k + j] = log_alpha[t * k + i]
                    + log_a(a, i, j, k)
                    + log_b_discrete(b, j, ot1, m)
                    + log_beta[(t + 1) * k + j]
                    - log_z;
            }
        }
    }

    // --- M-step ---
    // pi
    let new_pi: Vec<f64> = (0..k).map(|i| log_gamma[i].exp()).collect();
    let pi_sum: f64 = new_pi.iter().sum();
    let new_pi: Vec<f64> = new_pi.iter().map(|&v| v / pi_sum.max(1e-30)).collect();

    // A
    let mut new_a = vec![0.0f64; k * k];
    for i in 0..k {
        let denom_log = logsumexp_indexed(big_t_minus_1, |t| log_gamma[t * k + i]);
        for j in 0..k {
            let numer_log = logsumexp_indexed(big_t_minus_1, |t| log_xi[t * k * k + i * k + j]);
            new_a[i * k + j] = (numer_log - denom_log).exp();
        }
        let row_sum: f64 = (0..k).map(|j| new_a[i * k + j]).sum();
        if row_sum > 1e-30 {
            for j in 0..k {
                new_a[i * k + j] /= row_sum;
            }
        } else {
            // Fallback: uniform
            for j in 0..k {
                new_a[i * k + j] = 1.0 / k as f64;
            }
        }
    }

    // B
    let mut new_b = vec![0.0f64; k * m];
    for i in 0..k {
        let denom_log = logsumexp_indexed(big_t, |t| log_gamma[t * k + i]);
        for sym in 0..m {
            let numer_log = logsumexp_indices(
                obs.iter()
                    .enumerate()
                    .filter(|&(_, &o)| o == sym)
                    .map(|(t, _)| log_gamma[t * k + i]),
            );
            new_b[i * m + sym] = (numer_log - denom_log).exp();
        }
        let row_sum: f64 = (0..m).map(|sym| new_b[i * m + sym]).sum();
        if row_sum > 1e-30 {
            for sym in 0..m {
                new_b[i * m + sym] /= row_sum;
            }
        } else {
            for sym in 0..m {
                new_b[i * m + sym] = 1.0 / m as f64;
            }
        }
    }

    (new_pi, new_a, new_b, log_p)
}

/// One Baum-Welch EM step for a Gaussian HMM.
fn bw_step_gaussian(
    obs: &[f64],
    pi: &[f64],
    a: &[f64],
    b: &[f64],
    k: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let big_t = obs.len();

    let log_alpha = forward_gaussian(obs, pi, a, b, k);
    let log_beta = backward_gaussian(obs, a, b, k);

    let log_p = logsumexp(&log_alpha[(big_t - 1) * k..big_t * k]);

    // gamma
    let mut log_gamma = vec![f64::NEG_INFINITY; big_t * k];
    for t in 0..big_t {
        let log_z = logsumexp_indexed(k, |i| log_alpha[t * k + i] + log_beta[t * k + i]);
        for i in 0..k {
            log_gamma[t * k + i] = log_alpha[t * k + i] + log_beta[t * k + i] - log_z;
        }
    }

    // xi
    let big_t_minus_1 = big_t - 1;
    let mut log_xi = vec![f64::NEG_INFINITY; big_t_minus_1 * k * k];
    for t in 0..big_t_minus_1 {
        let ot1 = obs[t + 1];
        let log_z = logsumexp_indexed(k * k, |ij| {
            let i = ij / k;
            let j = ij % k;
            log_alpha[t * k + i]
                + log_a(a, i, j, k)
                + log_gaussian(ot1, b[2 * j], b[2 * j + 1])
                + log_beta[(t + 1) * k + j]
        });
        for i in 0..k {
            for j in 0..k {
                log_xi[t * k * k + i * k + j] = log_alpha[t * k + i]
                    + log_a(a, i, j, k)
                    + log_gaussian(ot1, b[2 * j], b[2 * j + 1])
                    + log_beta[(t + 1) * k + j]
                    - log_z;
            }
        }
    }

    // M-step pi
    let new_pi: Vec<f64> = (0..k).map(|i| log_gamma[i].exp()).collect();
    let pi_sum: f64 = new_pi.iter().sum();
    let new_pi: Vec<f64> = new_pi.iter().map(|&v| v / pi_sum.max(1e-30)).collect();

    // M-step A
    let mut new_a = vec![0.0f64; k * k];
    for i in 0..k {
        let denom_log = logsumexp_indexed(big_t_minus_1, |t| log_gamma[t * k + i]);
        for j in 0..k {
            let numer_log = logsumexp_indexed(big_t_minus_1, |t| log_xi[t * k * k + i * k + j]);
            new_a[i * k + j] = (numer_log - denom_log).exp();
        }
        let row_sum: f64 = (0..k).map(|j| new_a[i * k + j]).sum();
        if row_sum > 1e-30 {
            for j in 0..k {
                new_a[i * k + j] /= row_sum;
            }
        } else {
            for j in 0..k {
                new_a[i * k + j] = 1.0 / k as f64;
            }
        }
    }

    // M-step B (Gaussian)
    let mut new_b = vec![0.0f64; 2 * k];
    for i in 0..k {
        let denom_log = logsumexp_indexed(big_t, |t| log_gamma[t * k + i]);
        // mu
        let gamma_sum = denom_log.exp();
        let weighted_sum: f64 = (0..big_t)
            .map(|t| log_gamma[t * k + i].exp() * obs[t])
            .sum();
        let mu_i = if gamma_sum > 1e-30 {
            weighted_sum / gamma_sum
        } else {
            0.0
        };
        // sigma2
        let var_i: f64 = (0..big_t)
            .map(|t| log_gamma[t * k + i].exp() * (obs[t] - mu_i) * (obs[t] - mu_i))
            .sum::<f64>()
            / gamma_sum.max(1e-30);
        new_b[2 * i] = mu_i;
        new_b[2 * i + 1] = var_i.max(1e-6);
    }

    (new_pi, new_a, new_b, log_p)
}

// ─── Forward / Backward ──────────────────────────────────────────────────────

fn forward_discrete(
    obs: &[usize],
    pi: &[f64],
    a: &[f64],
    b: &[f64],
    k: usize,
    m: usize,
) -> Vec<f64> {
    let big_t = obs.len();
    let mut log_alpha = vec![f64::NEG_INFINITY; big_t * k];
    let o0 = obs[0];
    for i in 0..k {
        log_alpha[i] = log_safe(pi[i]) + log_b_discrete(b, i, o0, m);
    }
    for t in 1..big_t {
        let ot = obs[t];
        for j in 0..k {
            let prev_log = logsumexp_indexed(k, |i| log_alpha[(t - 1) * k + i] + log_a(a, i, j, k));
            log_alpha[t * k + j] = prev_log + log_b_discrete(b, j, ot, m);
        }
    }
    log_alpha
}

fn backward_discrete(obs: &[usize], a: &[f64], b: &[f64], k: usize, m: usize) -> Vec<f64> {
    let big_t = obs.len();
    let mut log_beta = vec![f64::NEG_INFINITY; big_t * k];
    for i in 0..k {
        log_beta[(big_t - 1) * k + i] = 0.0;
    }
    for t in (0..big_t - 1).rev() {
        let ot1 = obs[t + 1];
        for i in 0..k {
            log_beta[t * k + i] = logsumexp_indexed(k, |j| {
                log_a(a, i, j, k) + log_b_discrete(b, j, ot1, m) + log_beta[(t + 1) * k + j]
            });
        }
    }
    log_beta
}

fn forward_gaussian(obs: &[f64], pi: &[f64], a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let big_t = obs.len();
    let mut log_alpha = vec![f64::NEG_INFINITY; big_t * k];
    for i in 0..k {
        log_alpha[i] = log_safe(pi[i]) + log_gaussian(obs[0], b[2 * i], b[2 * i + 1]);
    }
    for t in 1..big_t {
        for j in 0..k {
            let prev_log = logsumexp_indexed(k, |i| log_alpha[(t - 1) * k + i] + log_a(a, i, j, k));
            log_alpha[t * k + j] = prev_log + log_gaussian(obs[t], b[2 * j], b[2 * j + 1]);
        }
    }
    log_alpha
}

fn backward_gaussian(obs: &[f64], a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let big_t = obs.len();
    let mut log_beta = vec![f64::NEG_INFINITY; big_t * k];
    for i in 0..k {
        log_beta[(big_t - 1) * k + i] = 0.0;
    }
    for t in (0..big_t - 1).rev() {
        for i in 0..k {
            log_beta[t * k + i] = logsumexp_indexed(k, |j| {
                log_a(a, i, j, k)
                    + log_gaussian(obs[t + 1], b[2 * j], b[2 * j + 1])
                    + log_beta[(t + 1) * k + j]
            });
        }
    }
    log_beta
}

// ─── Viterbi ─────────────────────────────────────────────────────────────────

fn viterbi_discrete(
    obs: &[usize],
    pi: &[f64],
    a: &[f64],
    b: &[f64],
    k: usize,
    m: usize,
) -> (Vec<usize>, f64) {
    let big_t = obs.len();
    let mut log_delta = vec![f64::NEG_INFINITY; big_t * k];
    let mut psi = vec![0usize; big_t * k];

    let o0 = obs[0];
    for i in 0..k {
        log_delta[i] = log_safe(pi[i]) + log_b_discrete(b, i, o0, m);
    }

    for t in 1..big_t {
        let ot = obs[t];
        for j in 0..k {
            let (best_val, best_i) = (0..k)
                .map(|i| (log_delta[(t - 1) * k + i] + log_a(a, i, j, k), i))
                .max_by(|(v1, _), (v2, _)| v1.partial_cmp(v2).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((f64::NEG_INFINITY, 0));
            log_delta[t * k + j] = best_val + log_b_discrete(b, j, ot, m);
            psi[t * k + j] = best_i;
        }
    }

    // Backtrack
    let mut states = vec![0usize; big_t];
    states[big_t - 1] = (0..k)
        .max_by(|&a, &b| {
            log_delta[(big_t - 1) * k + a]
                .partial_cmp(&log_delta[(big_t - 1) * k + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let log_prob = log_delta[(big_t - 1) * k + states[big_t - 1]];
    for t in (0..big_t - 1).rev() {
        states[t] = psi[(t + 1) * k + states[t + 1]];
    }
    (states, log_prob)
}

fn viterbi_gaussian(obs: &[f64], pi: &[f64], a: &[f64], b: &[f64], k: usize) -> (Vec<usize>, f64) {
    let big_t = obs.len();
    let mut log_delta = vec![f64::NEG_INFINITY; big_t * k];
    let mut psi = vec![0usize; big_t * k];

    for i in 0..k {
        log_delta[i] = log_safe(pi[i]) + log_gaussian(obs[0], b[2 * i], b[2 * i + 1]);
    }

    for t in 1..big_t {
        for j in 0..k {
            let (best_val, best_i) = (0..k)
                .map(|i| (log_delta[(t - 1) * k + i] + log_a(a, i, j, k), i))
                .max_by(|(v1, _), (v2, _)| v1.partial_cmp(v2).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((f64::NEG_INFINITY, 0));
            log_delta[t * k + j] = best_val + log_gaussian(obs[t], b[2 * j], b[2 * j + 1]);
            psi[t * k + j] = best_i;
        }
    }

    let mut states = vec![0usize; big_t];
    states[big_t - 1] = (0..k)
        .max_by(|&a, &b| {
            log_delta[(big_t - 1) * k + a]
                .partial_cmp(&log_delta[(big_t - 1) * k + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let log_prob = log_delta[(big_t - 1) * k + states[big_t - 1]];
    for t in (0..big_t - 1).rev() {
        states[t] = psi[(t + 1) * k + states[t + 1]];
    }
    (states, log_prob)
}

// ─── Numerical utilities ──────────────────────────────────────────────────────

/// Numerically stable logsumexp: m + log(Σ exp(v - m)), m = max(v).
fn logsumexp(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m.is_infinite() {
        return f64::NEG_INFINITY;
    }
    m + v.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

/// logsumexp over a generated iterator.
#[inline]
fn logsumexp_indexed<F: Fn(usize) -> f64>(n: usize, f: F) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for i in 0..n {
        let v = f(i);
        if v > m {
            m = v;
        }
    }
    if m.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let s: f64 = (0..n).map(|i| (f(i) - m).exp()).sum();
    m + s.ln()
}

/// logsumexp over an arbitrary iterator of log-values.
fn logsumexp_indices<I: Iterator<Item = f64>>(it: I) -> f64 {
    let vals: Vec<f64> = it.collect();
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    logsumexp(&vals)
}

/// log(N(x; mu, sigma2)).
fn log_gaussian(x: f64, mu: f64, sigma2: f64) -> f64 {
    let s2 = sigma2.max(1e-10);
    -0.5 * (2.0 * std::f64::consts::PI * s2).ln() - (x - mu) * (x - mu) / (2.0 * s2)
}

/// log(p) with floor at NEG_INFINITY for p ≤ 0.
#[inline]
fn log_safe(p: f64) -> f64 {
    if p <= 0.0 { f64::NEG_INFINITY } else { p.ln() }
}

/// log(A[i,j]).
#[inline]
fn log_a(a: &[f64], i: usize, j: usize, k: usize) -> f64 {
    log_safe(a[i * k + j])
}

/// log(B[i, obs]).
#[inline]
fn log_b_discrete(b: &[f64], i: usize, obs: usize, m: usize) -> f64 {
    log_safe(b[i * m + obs])
}

// ─── Sampling helper ─────────────────────────────────────────────────────────

/// Sample from a categorical distribution with given weights using uniform RNG.
/// Safe uniform [0,1): `rng.next_u32() as f64 / 2^31`.
fn categorical_sample(weights: &[f64], rng: &mut LcgRng) -> usize {
    let u = rng.next_u32() as f64 / 4_294_967_296.0; // [0, 1)
    let mut cumsum = 0.0;
    for (k, &w) in weights.iter().enumerate() {
        cumsum += w;
        if u < cumsum {
            return k;
        }
    }
    weights.len().saturating_sub(1) // fallback
}

// ─── Validation ──────────────────────────────────────────────────────────────

fn validate_discrete(obs: &[usize], k: usize, m: usize) -> TsResult<()> {
    if k == 0 {
        return Err(TsError::InvalidNumVariates(0));
    }
    if obs.len() < 2 {
        return Err(TsError::InvalidSequenceLength(obs.len()));
    }
    for &o in obs {
        if o >= m {
            return Err(TsError::DimensionMismatch {
                expected: m,
                got: o + 1,
            });
        }
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn simple_obs(n: usize, n_obs: usize) -> Vec<usize> {
        // Alternating pattern
        (0..n).map(|i| i % n_obs).collect()
    }

    #[test]
    fn test_pi_sums_to_one() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let s: f64 = model.pi.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "pi sum = {s}");
    }

    #[test]
    fn test_a_rows_sum_to_one() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let k = model.config.n_states;
        for i in 0..k {
            let s: f64 = (0..k).map(|j| model.a[i * k + j]).sum();
            assert!((s - 1.0).abs() < 1e-6, "A row {i} sum = {s}");
        }
    }

    #[test]
    fn test_b_rows_sum_to_one() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let k = model.config.n_states;
        let m = match model.config.obs_type {
            HmmObsType::Discrete { n_obs } => n_obs,
            _ => panic!("not discrete"),
        };
        for i in 0..k {
            let s: f64 = (0..m).map(|sym| model.b[i * m + sym]).sum();
            assert!((s - 1.0).abs() < 1e-6, "B row {i} sum = {s}");
        }
    }

    #[test]
    fn test_log_likelihood_finite_negative() {
        let obs = simple_obs(40, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        assert!(
            model.log_likelihood.is_finite(),
            "log_likelihood not finite"
        );
        assert!(
            model.log_likelihood < 0.0,
            "log_likelihood should be negative"
        );
    }

    #[test]
    fn test_converged_or_max_iter() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig {
            max_iter: 5,
            tol: 1e-20,
            ..HmmConfig::default()
        };
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        assert!(model.n_iter <= 5);
    }

    #[test]
    fn test_decode_states_len() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let result = hmm_decode(&model, &obs).expect("hmm_decode should succeed");
        assert_eq!(result.states.len(), obs.len());
    }

    #[test]
    fn test_decode_states_valid() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let result = hmm_decode(&model, &obs).expect("hmm_decode should succeed");
        let k = model.config.n_states;
        for &s in &result.states {
            assert!(s < k, "state {s} >= k={k}");
        }
    }

    #[test]
    fn test_viterbi_log_prob_finite() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let result = hmm_decode(&model, &obs).expect("hmm_decode should succeed");
        assert!(result.log_prob.is_finite(), "Viterbi log_prob not finite");
    }

    #[test]
    fn test_generate_output_len() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let mut rng = make_rng();
        let seq = hmm_generate(&model, 20, &mut rng).expect("hmm_generate should succeed");
        assert_eq!(seq.len(), 20);
    }

    #[test]
    fn test_generate_symbols_valid() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let mut rng = make_rng();
        let seq = hmm_generate(&model, 50, &mut rng).expect("hmm_generate should succeed");
        let n_obs = match model.config.obs_type {
            HmmObsType::Discrete { n_obs } => n_obs,
            _ => panic!("not discrete"),
        };
        for &o in &seq {
            assert!(o < n_obs, "generated obs {o} >= n_obs={n_obs}");
        }
    }

    #[test]
    fn test_stationary_sums_to_one() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let stat = hmm_stationary(&model);
        let s: f64 = stat.iter().sum();
        assert!((s - 1.0).abs() < 1e-8, "stationary sum = {s}");
    }

    #[test]
    fn test_stationary_is_fixed_point() {
        let obs = simple_obs(50, 4);
        let cfg = HmmConfig {
            max_iter: 200,
            tol: 1e-6,
            ..HmmConfig::default()
        };
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let k = model.config.n_states;
        let stat = hmm_stationary(&model);
        // Compute A @ stat (note: stat is a row vector, so stat @ A)
        let mut next = vec![0.0f64; k];
        for (i, &s_i) in stat.iter().enumerate() {
            for (j, n_j) in next.iter_mut().enumerate() {
                *n_j += s_i * model.a[i * k + j];
            }
        }
        for (s, n) in stat.iter().zip(next.iter()) {
            assert!(
                (s - n).abs() < 1e-4,
                "stationary not fixed point: {s} vs {n}"
            );
        }
    }

    #[test]
    fn test_small_hmm_k2_m2() {
        let obs: Vec<usize> = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let cfg = HmmConfig {
            n_states: 2,
            obs_type: HmmObsType::Discrete { n_obs: 2 },
            max_iter: 50,
            tol: 1e-4,
        };
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        assert_eq!(model.pi.len(), 2);
        let s: f64 = model.pi.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_hmm_fit() {
        let obs: Vec<f64> = (0..40)
            .map(|i| (i as f64 * 0.2).sin() * 2.0 + 1.0)
            .collect();
        let cfg = HmmConfig {
            n_states: 2,
            obs_type: HmmObsType::Gaussian,
            max_iter: 50,
            tol: 1e-4,
        };
        let model = hmm_fit_gaussian(&obs, &cfg).expect("hmm_fit_gaussian should succeed");
        assert_eq!(model.pi.len(), 2);
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_gaussian_b_length() {
        let obs: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        let cfg = HmmConfig {
            n_states: 3,
            obs_type: HmmObsType::Gaussian,
            max_iter: 20,
            tol: 1e-4,
        };
        let model = hmm_fit_gaussian(&obs, &cfg).expect("hmm_fit_gaussian should succeed");
        // b should have 2*n_states entries (mu, sigma2 pairs)
        assert_eq!(model.b.len(), 2 * cfg.n_states);
    }

    #[test]
    fn test_gaussian_sigma2_positive() {
        let obs: Vec<f64> = (0..40).map(|i| (i as f64 * 0.3).sin()).collect();
        let cfg = HmmConfig {
            n_states: 2,
            obs_type: HmmObsType::Gaussian,
            max_iter: 50,
            tol: 1e-4,
        };
        let model = hmm_fit_gaussian(&obs, &cfg).expect("hmm_fit_gaussian should succeed");
        let k = model.config.n_states;
        for i in 0..k {
            let sigma2 = model.b[2 * i + 1];
            assert!(sigma2 > 0.0, "sigma2[{i}] = {sigma2} not positive");
        }
    }

    #[test]
    fn test_gaussian_decode_valid_states() {
        let obs: Vec<f64> = (0..20).map(|i| (i as f64 * 0.25).sin()).collect();
        let cfg = HmmConfig {
            n_states: 2,
            obs_type: HmmObsType::Gaussian,
            max_iter: 30,
            tol: 1e-4,
        };
        let model = hmm_fit_gaussian(&obs, &cfg).expect("hmm_fit_gaussian should succeed");
        let result = hmm_decode_gaussian(&model, &obs).expect("hmm_decode_gaussian should succeed");
        let k = model.config.n_states;
        assert_eq!(result.states.len(), obs.len());
        for &s in &result.states {
            assert!(s < k, "state {s} >= k={k}");
        }
    }

    #[test]
    fn test_long_sequence_t100() {
        let obs: Vec<usize> = (0..100).map(|i| i % 4).collect();
        let cfg = HmmConfig {
            max_iter: 30,
            tol: 1e-3,
            ..HmmConfig::default()
        };
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        assert!(model.log_likelihood.is_finite());
        let result = hmm_decode(&model, &obs).expect("hmm_decode should succeed");
        assert_eq!(result.states.len(), 100);
    }

    #[test]
    fn test_k3_m4_larger_hmm() {
        let obs: Vec<usize> = (0..60).map(|i| i % 4).collect();
        let cfg = HmmConfig {
            n_states: 3,
            obs_type: HmmObsType::Discrete { n_obs: 4 },
            max_iter: 50,
            tol: 1e-4,
        };
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        assert_eq!(model.pi.len(), 3);
        let k = model.config.n_states;
        for i in 0..k {
            let row_sum: f64 = (0..k).map(|j| model.a[i * k + j]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "A row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_short_obs_errors() {
        let obs = vec![0usize]; // length 1 < 2
        let cfg = HmmConfig::default();
        let result = hmm_fit(&obs, &cfg);
        assert!(result.is_err(), "expected error for obs.len() < 2");
    }

    #[test]
    fn test_zero_states_errors() {
        let obs = simple_obs(20, 4);
        let cfg = HmmConfig {
            n_states: 0,
            obs_type: HmmObsType::Discrete { n_obs: 4 },
            max_iter: 10,
            tol: 1e-4,
        };
        let result = hmm_fit(&obs, &cfg);
        assert!(result.is_err(), "expected error for n_states=0");
    }

    #[test]
    fn test_log_likelihood_le_zero() {
        let obs = simple_obs(30, 4);
        let cfg = HmmConfig::default();
        let model = hmm_fit(&obs, &cfg).expect("hmm_fit should succeed");
        let ll = hmm_log_likelihood(&model, &obs).expect("hmm_log_likelihood should succeed");
        assert!(ll <= 0.0, "log_likelihood = {ll} should be ≤ 0");
    }
}

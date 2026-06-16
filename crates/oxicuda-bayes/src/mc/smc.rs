//! Sequential Monte Carlo (SMC) / Bootstrap Particle Filter.
//!
//! Implements the Bootstrap Particle Filter of Gordon, Salmond & Smith (1993)
//! and the systematic resampling scheme described in Doucet & Johansen (2009).
//!
//! # Algorithm
//!
//! At each observation `y_t`:
//! 1. **Propagate** each particle through the transition kernel.
//! 2. **Weight** by the log-likelihood and normalise.
//! 3. Compute effective sample size (ESS = 1/Σw²).
//! 4. Update the log-evidence.
//! 5. **Resample** (systematic) if ESS < N·threshold.

use crate::error::{BayesError, BayesResult};
pub use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the Bootstrap Particle Filter.
#[derive(Debug, Clone)]
pub struct SmcConfig {
    /// Number of particles. Must be > 0.
    pub n_particles: usize,
    /// ESS / N threshold that triggers resampling. Must be in (0, 1].
    pub resample_threshold: f64,
    /// Dimensionality of the latent state. Must be > 0.
    pub state_dim: usize,
    /// RNG seed for particle initialisation and resampling.
    pub seed: u64,
}

impl Default for SmcConfig {
    fn default() -> Self {
        Self {
            n_particles: 500,
            resample_threshold: 0.5,
            state_dim: 1,
            seed: 0,
        }
    }
}

// ─── State ────────────────────────────────────────────────────────────────────

/// Running state of the Bootstrap Particle Filter.
#[derive(Debug, Clone)]
pub struct SmcState {
    /// Particle matrix, row-major `N × state_dim`.
    pub particles: Vec<f64>,
    /// Normalised importance weights summing to 1.
    pub weights: Vec<f64>,
    /// Cumulative log p(y_{1:t}) (marginal likelihood).
    pub log_evidence: f64,
    /// Current effective sample size.
    pub ess: f64,
    /// Number of systematic resampling steps performed so far.
    pub n_resamples: usize,
    /// Current time index (number of observations assimilated).
    pub t: usize,
}

// ─── Validation ───────────────────────────────────────────────────────────────

fn validate_config(config: &SmcConfig) -> BayesResult<()> {
    if config.n_particles == 0 {
        return Err(BayesError::InvalidConfig("n_particles must be > 0".into()));
    }
    if config.state_dim == 0 {
        return Err(BayesError::InvalidConfig("state_dim must be > 0".into()));
    }
    if config.resample_threshold <= 0.0 || config.resample_threshold > 1.0 {
        return Err(BayesError::InvalidConfig(
            "resample_threshold must be in (0, 1]".into(),
        ));
    }
    Ok(())
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Initialise the particle filter: sample particles from the Gaussian prior
/// and set uniform weights.
///
/// # Errors
/// - `InvalidConfig` when `config.n_particles == 0`, `state_dim == 0`, or
///   `resample_threshold` outside (0, 1].
/// - `DimensionMismatch` when `prior_mean.len() != state_dim`.
pub fn smc_init(config: &SmcConfig, prior_mean: &[f64], prior_std: f64) -> BayesResult<SmcState> {
    validate_config(config)?;
    if prior_mean.len() != config.state_dim {
        return Err(BayesError::DimensionMismatch {
            expected: config.state_dim,
            got: prior_mean.len(),
        });
    }

    let n = config.n_particles;
    let d = config.state_dim;
    let mut rng = LcgRng::new(config.seed);
    let mut particles = vec![0.0_f64; n * d];

    for i in 0..n {
        for j in 0..d {
            // SAFE: use only the first value of next_normal_pair()
            let z = rng.next_normal_pair().0 as f64;
            particles[i * d + j] = prior_mean[j] + prior_std * z;
        }
    }

    let uniform_w = 1.0 / n as f64;
    let weights = vec![uniform_w; n];
    let ess = n as f64;

    Ok(SmcState {
        particles,
        weights,
        log_evidence: 0.0,
        ess,
        n_resamples: 0,
        t: 0,
    })
}

/// Apply one observation step to an existing `SmcState`.
///
/// This function mutates `state` in-place and advances `state.t`.
/// The embedded resample threshold is fixed at 0.5 for the step API;
/// use [`smc_filter`] to control the threshold via [`SmcConfig`].
///
/// # Errors
/// This function is infallible (returns `Ok(())`).
pub fn smc_step(
    state: &mut SmcState,
    observation: &[f64],
    transition_fn: &dyn Fn(&[f64], &mut LcgRng) -> Vec<f64>,
    log_likelihood_fn: &dyn Fn(&[f64], &[f64]) -> f64,
    rng: &mut LcgRng,
) -> BayesResult<()> {
    let n = state.weights.len();
    let d = state.particles.len().checked_div(n).unwrap_or(0);

    // ── 1. Propagate ──────────────────────────────────────────────────────────
    let mut new_particles = Vec::with_capacity(n * d);
    for i in 0..n {
        let x_i = &state.particles[i * d..(i + 1) * d];
        let x_new = transition_fn(x_i, rng);
        new_particles.extend_from_slice(&x_new);
    }
    state.particles = new_particles;

    // ── 2. Update log-weights and normalise ───────────────────────────────────
    let mut log_w: Vec<f64> = state
        .weights
        .iter()
        .map(|&w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY })
        .collect();

    // Add log-likelihood to each particle's log-weight
    for (lw, i) in log_w.iter_mut().zip(0..n) {
        let x_i = &state.particles[i * d..(i + 1) * d];
        *lw += log_likelihood_fn(x_i, observation);
    }

    // Subtract max for numerical stability
    let max_log_w = log_w
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    let w_unnorm: Vec<f64> = log_w
        .iter()
        .map(|&lw| {
            if lw.is_finite() {
                (lw - max_log_w).exp()
            } else {
                0.0
            }
        })
        .collect();

    let sum_w: f64 = w_unnorm.iter().sum();

    // ── 4. Log-evidence update ────────────────────────────────────────────────
    if sum_w > 0.0 {
        state.log_evidence += sum_w.ln() + max_log_w;
    }

    // Normalise
    if sum_w > 0.0 {
        for (w, &uw) in state.weights.iter_mut().zip(w_unnorm.iter()) {
            *w = uw / sum_w;
        }
    } else {
        let uniform_w = 1.0 / n as f64;
        for w in state.weights.iter_mut() {
            *w = uniform_w;
        }
    }

    // ── 3. Effective sample size ───────────────────────────────────────────────
    state.ess = effective_sample_size(&state.weights);

    // ── 5. Systematic resampling if needed (threshold = 0.5 for step API) ────
    let ess_fraction = state.ess / n as f64;
    if ess_fraction < 0.5 {
        let new_particles = systematic_resample(&state.particles, &state.weights, n, d, rng);
        state.particles = new_particles;
        let uniform_w = 1.0 / n as f64;
        for w in state.weights.iter_mut() {
            *w = uniform_w;
        }
        state.ess = n as f64;
        state.n_resamples += 1;
    }

    state.t += 1;
    Ok(())
}

/// Run the full Bootstrap Particle Filter over a sequence of observations.
///
/// `observations` is a flat row-major array of shape `n_obs × obs_dim`.
///
/// Returns a `Vec<SmcState>` of length `n_obs`, one per observation.
///
/// # Errors
/// - `InvalidConfig` / `DimensionMismatch` (propagated from `smc_init`).
pub fn smc_filter(
    observations: &[f64],
    n_obs: usize,
    obs_dim: usize,
    config: &SmcConfig,
    prior_mean: &[f64],
    prior_std: f64,
    transition_fn: &dyn Fn(&[f64], &mut LcgRng) -> Vec<f64>,
    log_likelihood_fn: &dyn Fn(&[f64], &[f64]) -> f64,
) -> BayesResult<Vec<SmcState>> {
    validate_config(config)?;

    let mut state = smc_init(config, prior_mean, prior_std)?;
    let n = config.n_particles;
    let d = config.state_dim;
    let resample_threshold = config.resample_threshold;

    // Separate RNG for the propagation / resampling phase (different seed
    // offset from smc_init for reproducibility).
    let mut rng = LcgRng::new(config.seed.wrapping_add(0xDEAD_BEEF_1234_5678));

    let mut history = Vec::with_capacity(n_obs);

    for t in 0..n_obs {
        let obs = &observations[t * obs_dim..(t + 1) * obs_dim];

        // ── Propagate ──────────────────────────────────────────────────────
        let mut new_particles = Vec::with_capacity(n * d);
        for i in 0..n {
            let x_i = &state.particles[i * d..(i + 1) * d];
            let x_new = transition_fn(x_i, &mut rng);
            new_particles.extend_from_slice(&x_new);
        }
        state.particles = new_particles;

        // ── Weight update ─────────────────────────────────────────────────
        let mut log_w: Vec<f64> = state
            .weights
            .iter()
            .map(|&w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY })
            .collect();

        for (lw, i) in log_w.iter_mut().zip(0..n) {
            let x_i = &state.particles[i * d..(i + 1) * d];
            *lw += log_likelihood_fn(x_i, obs);
        }

        let max_log_w = log_w
            .iter()
            .cloned()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        let w_unnorm: Vec<f64> = log_w
            .iter()
            .map(|&lw| {
                if lw.is_finite() {
                    (lw - max_log_w).exp()
                } else {
                    0.0
                }
            })
            .collect();

        let sum_w: f64 = w_unnorm.iter().sum();

        if sum_w > 0.0 {
            state.log_evidence += sum_w.ln() + max_log_w;
        }

        if sum_w > 0.0 {
            for (w, &uw) in state.weights.iter_mut().zip(w_unnorm.iter()) {
                *w = uw / sum_w;
            }
        } else {
            let uniform_w = 1.0 / n as f64;
            for w in state.weights.iter_mut() {
                *w = uniform_w;
            }
        }

        // ── ESS ────────────────────────────────────────────────────────────
        state.ess = effective_sample_size(&state.weights);

        // ── Systematic resample ────────────────────────────────────────────
        if state.ess < n as f64 * resample_threshold {
            let new_particles =
                systematic_resample(&state.particles, &state.weights, n, d, &mut rng);
            state.particles = new_particles;
            let uniform_w = 1.0 / n as f64;
            for w in state.weights.iter_mut() {
                *w = uniform_w;
            }
            state.ess = n as f64;
            state.n_resamples += 1;
        }

        state.t = t + 1;
        history.push(state.clone());
    }

    Ok(history)
}

/// Compute the weighted mean of particles.
///
/// Returns a vector of length `state_dim`.
#[must_use]
pub fn smc_mean(state: &SmcState) -> Vec<f64> {
    let n = state.weights.len();
    if n == 0 {
        return Vec::new();
    }
    let d = state.particles.len() / n;
    let mut mean = vec![0.0_f64; d];
    for (i, &wi) in state.weights.iter().enumerate() {
        for (mj, &pij) in mean
            .iter_mut()
            .zip(state.particles[i * d..(i + 1) * d].iter())
        {
            *mj += wi * pij;
        }
    }
    mean
}

/// Compute the weighted variance of particles.
///
/// Returns a vector of length `state_dim` with non-negative entries.
#[must_use]
pub fn smc_variance(state: &SmcState) -> Vec<f64> {
    let n = state.weights.len();
    if n == 0 {
        return Vec::new();
    }
    let d = state.particles.len() / n;
    let mean = smc_mean(state);
    let mut var = vec![0.0_f64; d];
    for (i, &wi) in state.weights.iter().enumerate() {
        for (vj, (&pij, &mj)) in var
            .iter_mut()
            .zip(state.particles[i * d..(i + 1) * d].iter().zip(mean.iter()))
        {
            let diff = pij - mj;
            *vj += wi * diff * diff;
        }
    }
    // Clamp to zero to guard against floating-point negatives
    for v in var.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    var
}

/// Systematic resampling (O(N)).
///
/// Returns a new particle array of shape `n × d` (row-major).
pub fn systematic_resample(
    particles: &[f64],
    weights: &[f64],
    n: usize,
    d: usize,
    rng: &mut LcgRng,
) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    // u_1 ~ Uniform[0, 1/N) using the safe high-31-bit recipe
    let u1 = rng.next_u32() as f64 / (4_294_967_296.0 * n as f64);

    // Build cumulative weights
    let mut cum_w = vec![0.0_f64; n];
    cum_w[0] = weights[0];
    for i in 1..n {
        cum_w[i] = cum_w[i - 1] + weights[i];
    }

    let mut new_particles = vec![0.0_f64; n * d];
    let mut j = 0usize;

    for i in 0..n {
        let u_i = u1 + i as f64 / n as f64;
        // Advance j until cum_w[j] >= u_i (or we hit the end)
        while j < n - 1 && cum_w[j] < u_i {
            j += 1;
        }
        let src = j * d;
        let dst = i * d;
        new_particles[dst..dst + d].copy_from_slice(&particles[src..src + d]);
    }

    new_particles
}

/// Compute the effective sample size: ESS = 1 / Σᵢ `w[i]`².
#[must_use]
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let sum_sq: f64 = weights.iter().map(|&w| w * w).sum();
    if sum_sq <= 0.0 {
        return 0.0;
    }
    1.0 / sum_sq
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_config_1d() -> SmcConfig {
        SmcConfig {
            n_particles: 500,
            resample_threshold: 0.5,
            state_dim: 1,
            seed: 42,
        }
    }

    /// AR(1) transition: x_{t+1} = 0.9 x_t + 0.1 * N(0,1)
    fn transition_ar1(x: &[f64], rng: &mut LcgRng) -> Vec<f64> {
        let noise = rng.next_normal_pair().0 as f64;
        vec![0.9 * x[0] + 0.1 * noise]
    }

    /// Log-likelihood: y | x ~ N(x, 0.1²)
    fn log_lik_gaussian(x: &[f64], y: &[f64]) -> f64 {
        let sigma = 0.1_f64;
        let diff = (y[0] - x[0]) / sigma;
        -0.5 * diff * diff - sigma.ln() - 0.5 * (2.0 * PI).ln()
    }

    // ── Test 1: mean ≈ prior_mean after init ──────────────────────────────────
    #[test]
    fn smc_init_mean_near_prior() {
        let config = default_config_1d();
        let prior_mean = vec![2.0_f64];
        let prior_std = 1.0_f64;
        let state = smc_init(&config, &prior_mean, prior_std).expect("smc_init should succeed");
        let mean = smc_mean(&state);
        // Within 3 * prior_std / sqrt(N)
        let tol = 3.0 * prior_std / (config.n_particles as f64).sqrt();
        assert!(
            (mean[0] - prior_mean[0]).abs() < tol,
            "mean={} prior_mean={} tol={}",
            mean[0],
            prior_mean[0],
            tol
        );
    }

    // ── Test 2: uniform weights after init ────────────────────────────────────
    #[test]
    fn smc_init_uniform_weights() {
        let config = default_config_1d();
        let state = smc_init(&config, &[0.0], 1.0).expect("smc_init should succeed");
        let expected = 1.0 / config.n_particles as f64;
        for &w in &state.weights {
            assert!((w - expected).abs() < 1e-10, "w={w} expected={expected}");
        }
    }

    // ── Test 3: ESS == N after init ───────────────────────────────────────────
    #[test]
    fn smc_init_ess_equals_n() {
        let config = default_config_1d();
        let state = smc_init(&config, &[0.0], 1.0).expect("smc_init should succeed");
        assert!(
            (state.ess - config.n_particles as f64).abs() < 1e-6,
            "ess={} expected={}",
            state.ess,
            config.n_particles
        );
    }

    // ── Test 4: smc_mean shape == state_dim ──────────────────────────────────
    #[test]
    fn smc_mean_shape() {
        let config = SmcConfig {
            state_dim: 3,
            ..default_config_1d()
        };
        let state = smc_init(&config, &[0.0, 1.0, -1.0], 1.0).expect("smc_init should succeed");
        assert_eq!(smc_mean(&state).len(), 3);
    }

    // ── Test 5: variance all non-negative ────────────────────────────────────
    #[test]
    fn smc_variance_nonneg() {
        let config = default_config_1d();
        let state = smc_init(&config, &[1.0], 2.0).expect("smc_init should succeed");
        for v in smc_variance(&state) {
            assert!(v >= 0.0, "variance negative: {v}");
        }
    }

    // ── Test 6: smc_filter returns n_obs states ───────────────────────────────
    #[test]
    fn smc_filter_returns_n_obs_states() {
        let config = default_config_1d();
        let obs = vec![0.5_f64, 0.3, 0.7, 0.2, 0.6];
        let states = smc_filter(
            &obs,
            5,
            1,
            &config,
            &[0.0],
            1.0,
            &transition_ar1,
            &log_lik_gaussian,
        )
        .expect("value should be present");
        assert_eq!(states.len(), 5);
    }

    // ── Test 7: smc_filter time indices correct ───────────────────────────────
    #[test]
    fn smc_filter_time_indices() {
        let config = default_config_1d();
        let obs = vec![0.0_f64; 4];
        let states = smc_filter(
            &obs,
            4,
            1,
            &config,
            &[0.0],
            1.0,
            &transition_ar1,
            &log_lik_gaussian,
        )
        .expect("value should be present");
        for (i, s) in states.iter().enumerate() {
            assert_eq!(s.t, i + 1, "state {i} has t={}", s.t);
        }
    }

    // ── Test 8: weights sum to 1 after each step ─────────────────────────────
    #[test]
    fn weights_sum_to_one_after_each_step() {
        let config = default_config_1d();
        let obs = vec![0.0_f64; 5];
        let states = smc_filter(
            &obs,
            5,
            1,
            &config,
            &[0.0],
            1.0,
            &transition_ar1,
            &log_lik_gaussian,
        )
        .expect("value should be present");
        for (i, s) in states.iter().enumerate() {
            let sum: f64 = s.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "step {i}: weights sum={sum}");
        }
    }

    // ── Test 9: log_evidence finite after 5 observations ─────────────────────
    #[test]
    fn log_evidence_finite() {
        let config = default_config_1d();
        let obs = vec![0.1_f64, -0.1, 0.2, 0.0, 0.3];
        let states = smc_filter(
            &obs,
            5,
            1,
            &config,
            &[0.0],
            1.0,
            &transition_ar1,
            &log_lik_gaussian,
        )
        .expect("value should be present");
        assert!(
            states
                .last()
                .expect("last should succeed")
                .log_evidence
                .is_finite()
        );
    }

    // ── Test 10: n_resamples > 0 after 20 steps with strong obs ──────────────
    #[test]
    fn n_resamples_increases() {
        let config = SmcConfig {
            n_particles: 200,
            resample_threshold: 0.5,
            state_dim: 1,
            seed: 7,
        };
        // Strong observations (very tight likelihood) force particles to
        // collapse and trigger resampling.
        let obs: Vec<f64> = (0..20).map(|i| i as f64 * 0.01).collect();

        // Tight likelihood: sigma=0.01 to force weight collapse
        let tight_lik = |x: &[f64], y: &[f64]| -> f64 {
            let sigma = 0.01_f64;
            let diff = (y[0] - x[0]) / sigma;
            -0.5 * diff * diff - sigma.ln() - 0.5 * (2.0 * PI).ln()
        };

        let states = smc_filter(
            &obs,
            20,
            1,
            &config,
            &[0.0],
            1.0,
            &transition_ar1,
            &tight_lik,
        )
        .expect("value should be present");
        let final_resamples = states.last().expect("last should succeed").n_resamples;
        assert!(
            final_resamples > 0,
            "expected n_resamples > 0, got {final_resamples}"
        );
    }

    // ── Test 11: ESS([1/N, ...]) == N ────────────────────────────────────────
    #[test]
    fn ess_uniform_weights() {
        let n = 100usize;
        let weights = vec![1.0 / n as f64; n];
        let ess = effective_sample_size(&weights);
        assert!(
            (ess - n as f64).abs() < 1e-6,
            "ESS uniform should be N={n}, got {ess}"
        );
    }

    // ── Test 12: ESS([1.0, 0.0, ...]) ≈ 1 ───────────────────────────────────
    #[test]
    fn ess_degenerate_weights() {
        let n = 100usize;
        let mut weights = vec![0.0_f64; n];
        weights[0] = 1.0;
        let ess = effective_sample_size(&weights);
        assert!(
            (ess - 1.0).abs() < 0.01,
            "ESS degenerate should ≈ 1, got {ess}"
        );
    }

    // ── Test 13: systematic_resample output shape ─────────────────────────────
    #[test]
    fn systematic_resample_output_shape() {
        let n = 10usize;
        let d = 3usize;
        let particles: Vec<f64> = (0..n * d).map(|i| i as f64).collect();
        let weights = vec![1.0 / n as f64; n];
        let mut rng = LcgRng::new(1);
        let new_p = systematic_resample(&particles, &weights, n, d, &mut rng);
        assert_eq!(new_p.len(), n * d);
    }

    // ── Test 14: high-weight particle replicated ──────────────────────────────
    #[test]
    fn systematic_resample_high_weight_replicated() {
        let n = 10usize;
        let d = 1usize;
        // Particle 0 has value 999.0; rest have distinct values 1..9
        let mut particles = vec![0.0_f64; n];
        particles[0] = 999.0;
        for (p, idx) in particles[1..].iter_mut().zip(1..n) {
            *p = idx as f64;
        }
        // Give particle 0 weight 0.9, rest share the remaining 0.1
        let mut weights = vec![0.1 / (n - 1) as f64; n];
        weights[0] = 0.9;

        let mut rng = LcgRng::new(5);
        let new_p = systematic_resample(&particles, &weights, n, d, &mut rng);

        let count_999 = new_p.iter().filter(|&&v| (v - 999.0).abs() < 1e-10).count();
        assert!(
            count_999 > 1,
            "particle 0 (weight 0.9) should appear >1 times, got {count_999}"
        );
    }

    // ── Test 15: state_dim=1 works ───────────────────────────────────────────
    #[test]
    fn smc_state_dim_1() {
        let config = SmcConfig {
            state_dim: 1,
            ..default_config_1d()
        };
        let state = smc_init(&config, &[0.0], 1.0).expect("smc_init should succeed");
        assert_eq!(smc_mean(&state).len(), 1);
    }

    // ── Test 16: state_dim=3 works ───────────────────────────────────────────
    #[test]
    fn smc_state_dim_3() {
        let config = SmcConfig {
            state_dim: 3,
            n_particles: 100,
            ..default_config_1d()
        };
        let prior_mean = vec![1.0, 2.0, 3.0];
        let state = smc_init(&config, &prior_mean, 0.5).expect("smc_init should succeed");
        let m = smc_mean(&state);
        assert_eq!(m.len(), 3);
        for j in 0..3 {
            assert!(
                (m[j] - prior_mean[j]).abs() < 1.0,
                "dim {j}: mean={} prior={}",
                m[j],
                prior_mean[j]
            );
        }
    }

    // ── Test 17: n_particles=1 (degenerate, no panic) ────────────────────────
    #[test]
    fn smc_n_particles_1_no_panic() {
        let config = SmcConfig {
            n_particles: 1,
            ..default_config_1d()
        };
        let state = smc_init(&config, &[0.0], 1.0).expect("smc_init should succeed");
        assert_eq!(state.weights.len(), 1);
        assert_eq!(state.particles.len(), 1);
    }

    // ── Test 18: n_particles=0 → InvalidConfig ────────────────────────────────
    #[test]
    fn smc_n_particles_0_error() {
        let config = SmcConfig {
            n_particles: 0,
            ..default_config_1d()
        };
        let result = smc_init(&config, &[0.0], 1.0);
        assert!(
            matches!(result, Err(BayesError::InvalidConfig(_))),
            "expected InvalidConfig, got {result:?}"
        );
    }

    // ── Test 19: state_dim=0 → InvalidConfig ─────────────────────────────────
    #[test]
    fn smc_state_dim_0_error() {
        let config = SmcConfig {
            state_dim: 0,
            ..default_config_1d()
        };
        let result = smc_init(&config, &[], 1.0);
        assert!(
            matches!(result, Err(BayesError::InvalidConfig(_))),
            "expected InvalidConfig, got {result:?}"
        );
    }

    // ── Test 20: resample_threshold=0.0 → InvalidConfig ──────────────────────
    #[test]
    fn smc_resample_threshold_zero_error() {
        let config = SmcConfig {
            resample_threshold: 0.0,
            ..default_config_1d()
        };
        let result = smc_init(&config, &[0.0], 1.0);
        assert!(
            matches!(result, Err(BayesError::InvalidConfig(_))),
            "expected InvalidConfig, got {result:?}"
        );
    }

    // ── Test 21: resample_threshold=1.5 → InvalidConfig ──────────────────────
    #[test]
    fn smc_resample_threshold_too_large_error() {
        let config = SmcConfig {
            resample_threshold: 1.5,
            ..default_config_1d()
        };
        let result = smc_init(&config, &[0.0], 1.0);
        assert!(
            matches!(result, Err(BayesError::InvalidConfig(_))),
            "expected InvalidConfig, got {result:?}"
        );
    }

    // ── Test 22: seed determinism ─────────────────────────────────────────────
    #[test]
    fn smc_seed_determinism() {
        let config_a = SmcConfig {
            seed: 123,
            ..default_config_1d()
        };
        let config_b = SmcConfig {
            seed: 123,
            ..default_config_1d()
        };
        let obs = vec![0.1_f64, 0.2, -0.1, 0.3, 0.0];

        let states_a = smc_filter(
            &obs,
            5,
            1,
            &config_a,
            &[0.0],
            1.0,
            &transition_ar1,
            &log_lik_gaussian,
        )
        .expect("value should be present");
        let states_b = smc_filter(
            &obs,
            5,
            1,
            &config_b,
            &[0.0],
            1.0,
            &transition_ar1,
            &log_lik_gaussian,
        )
        .expect("value should be present");

        for (sa, sb) in states_a.iter().zip(states_b.iter()) {
            assert_eq!(sa.particles, sb.particles, "particles differ at t={}", sa.t);
        }
    }
}

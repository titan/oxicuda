//! # C51 — Categorical DQN Loss
//!
//! Bellemare, Dabney, Munos (2017) "A Distributional Perspective on Reinforcement Learning".
//! ICML 2017.
//!
//! Represents the Q-value return distribution as a categorical distribution over N atoms
//! {z_1, …, z_N} evenly spaced on [v_min, v_max]. The distributional Bellman operator
//! projects the bootstrapped distribution back onto the fixed support via linear interpolation,
//! then minimises the cross-entropy between the projected target and the online network output.
//!
//! ## Support
//!
//! ```text
//! z_i = v_min + i * (v_max - v_min) / (N - 1),  i = 0..N-1
//! ```
//!
//! ## Distributional Bellman projection (per-atom j)
//!
//! ```text
//! Tz_j  = clip(r + γ * (1 - done) * z_j,  v_min, v_max)
//! l     = floor((Tz_j - v_min) / Δz)
//! u     = l + 1  (clamped to N-1)
//! m[l] += p_j * (u - (Tz_j - v_min) / Δz)
//! m[u] += p_j * ((Tz_j - v_min) / Δz - l)
//! ```
//!
//! ## Loss
//!
//! Cross-entropy: `H(m, softmax(logits))` averaged over the batch, optionally
//! weighted by PER importance-sampling weights.

use crate::error::{RlError, RlResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// C51 / Categorical DQN configuration.
#[derive(Debug, Clone, Copy)]
pub struct C51Config {
    /// Number of support atoms N (must be ≥ 2; default 51).
    pub n_atoms: usize,
    /// Minimum of the support interval (default −10.0).
    pub v_min: f32,
    /// Maximum of the support interval (default +10.0).
    pub v_max: f32,
    /// Discount factor γ (must be in (0, 1]; default 0.99).
    pub gamma: f32,
}

impl Default for C51Config {
    fn default() -> Self {
        Self {
            n_atoms: 51,
            v_min: -10.0,
            v_max: 10.0,
            gamma: 0.99,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────────

/// C51 loss output.
#[derive(Debug, Clone)]
pub struct C51Loss {
    /// Mean cross-entropy loss over the batch (scalar to minimise).
    pub loss: f32,
    /// Per-sample cross-entropy (proxy for TD-error for PER priority updates).
    /// Length == batch size B.
    pub kl_errors: Vec<f32>,
}

// ─── Validation helpers ───────────────────────────────────────────────────────

/// Validate that the C51 config has consistent hyperparameter values.
fn validate_cfg(cfg: &C51Config) -> RlResult<()> {
    if cfg.n_atoms < 2 {
        return Err(RlError::InvalidHyperparameter {
            name: "n_atoms".into(),
            msg: "must be >= 2".into(),
        });
    }
    if cfg.v_min >= cfg.v_max {
        return Err(RlError::InvalidHyperparameter {
            name: "v_min / v_max".into(),
            msg: "v_min must be strictly less than v_max".into(),
        });
    }
    if cfg.gamma <= 0.0 || cfg.gamma > 1.0 {
        return Err(RlError::InvalidHyperparameter {
            name: "gamma".into(),
            msg: "must be in (0, 1]".into(),
        });
    }
    Ok(())
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute the atom support z_i = v_min + i · Δz for i = 0..N−1.
///
/// Δz = (v_max − v_min) / (N − 1).
///
/// # Panics
///
/// Never panics — only the `n_atoms >= 2` invariant is checked via [`C51Config`]
/// validation elsewhere.
#[must_use]
pub fn c51_support(cfg: &C51Config) -> Vec<f32> {
    let n = cfg.n_atoms;
    if n < 2 {
        // Degenerate: return a single-element vec so the caller can validate.
        return vec![cfg.v_min];
    }
    let dz = (cfg.v_max - cfg.v_min) / (n as f32 - 1.0);
    (0..n).map(|i| cfg.v_min + i as f32 * dz).collect()
}

/// Project the distributional Bellman target onto the fixed support.
///
/// For each sample b and each atom j:
/// ```text
/// Tz_j  = clip(r_b + γ * (1 - done_b) * z_j,  v_min, v_max)
/// l     = floor((Tz_j - v_min) / Δz)       ∈ [0, N-1]
/// u     = min(l + 1, N-1)
/// m[l] += p_j * (u − (Tz_j - v_min) / Δz)
/// m[u] += p_j * ((Tz_j - v_min) / Δz − l)
/// ```
///
/// # Arguments
///
/// * `rewards`    — `[B]` rewards r_b.
/// * `dones`      — `[B]` done flags (1.0 = terminal).
/// * `next_probs` — `[B × N]` probability distributions over the support for the
///   best next action (already selected from the target network).
/// * `cfg`        — C51 configuration.
///
/// # Errors
///
/// * [`RlError::InvalidHyperparameter`] — invalid config values.
/// * [`RlError::DimensionMismatch`]     — slice lengths are inconsistent.
///
/// # Returns
///
/// `[B × N]` target probability distributions (each row sums to 1).
pub fn c51_project(
    rewards: &[f32],
    dones: &[f32],
    next_probs: &[f32],
    cfg: &C51Config,
) -> RlResult<Vec<f32>> {
    validate_cfg(cfg)?;

    let b = rewards.len();
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if dones.len() != b {
        return Err(RlError::DimensionMismatch {
            expected: b,
            got: dones.len(),
        });
    }
    let n = cfg.n_atoms;
    if next_probs.len() != b * n {
        return Err(RlError::DimensionMismatch {
            expected: b * n,
            got: next_probs.len(),
        });
    }

    let dz = (cfg.v_max - cfg.v_min) / (n as f32 - 1.0);
    let mut target = vec![0.0_f32; b * n];

    for b_idx in 0..b {
        let r = rewards[b_idx];
        let done_flag = dones[b_idx];
        let gamma_factor = cfg.gamma * (1.0 - done_flag);

        for j in 0..n {
            let z_j = cfg.v_min + j as f32 * dz;
            let tz_j = (r + gamma_factor * z_j).clamp(cfg.v_min, cfg.v_max);

            let lower_f = (tz_j - cfg.v_min) / dz;
            // l clamped to [0, n-1]
            let l = (lower_f.floor() as usize).min(n - 1);
            let u = (l + 1).min(n - 1);

            let p = next_probs[b_idx * n + j];

            if l == u {
                // tz_j landed exactly on the top atom (or n=2 edge case):
                // all probability mass goes to that single atom.
                target[b_idx * n + l] += p;
            } else {
                // Split probability linearly between adjacent atoms.
                // weight_l = u - lower_f, weight_u = lower_f - l  (sum to 1)
                let weight_l = (u as f32) - lower_f;
                let weight_u = lower_f - (l as f32);
                target[b_idx * n + l] += p * weight_l;
                target[b_idx * n + u] += p * weight_u;
            }
        }
    }

    Ok(target)
}

/// Compute C51 categorical cross-entropy loss.
///
/// The online network outputs raw logits for the chosen action. We apply a numerically
/// stable softmax (subtract max before exp) and compute cross-entropy against the
/// projected target distribution from [`c51_project`].
///
/// # Arguments
///
/// * `logits`       — `[B × N]` raw logits from the online network for the chosen actions.
/// * `target_probs` — `[B × N]` projected target distributions (from [`c51_project`]).
/// * `is_weights`   — `[B]` importance-sampling weights (all 1.0 when not using PER).
/// * `cfg`          — C51 configuration.
///
/// # Errors
///
/// * [`RlError::InvalidHyperparameter`] — invalid config values.
/// * [`RlError::DimensionMismatch`]     — slice lengths inconsistent.
/// * [`RlError::Internal`]              — NaN loss encountered (e.g. due to infinite logits).
pub fn c51_loss(
    logits: &[f32],
    target_probs: &[f32],
    is_weights: &[f32],
    cfg: &C51Config,
) -> RlResult<C51Loss> {
    validate_cfg(cfg)?;

    let b = is_weights.len();
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let n = cfg.n_atoms;
    if logits.len() != b * n {
        return Err(RlError::DimensionMismatch {
            expected: b * n,
            got: logits.len(),
        });
    }
    if target_probs.len() != b * n {
        return Err(RlError::DimensionMismatch {
            expected: b * n,
            got: target_probs.len(),
        });
    }

    let mut kl_errors = Vec::with_capacity(b);
    let mut weighted_loss = 0.0_f32;

    for b_idx in 0..b {
        let row_logits = &logits[b_idx * n..(b_idx + 1) * n];
        let row_target = &target_probs[b_idx * n..(b_idx + 1) * n];

        // Numerically stable softmax: subtract max before exp.
        let max_logit = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut exp_vals = Vec::with_capacity(n);
        let mut sum_exp = 0.0_f32;
        for &l in row_logits {
            let e = (l - max_logit).exp();
            exp_vals.push(e);
            sum_exp += e;
        }

        // Cross-entropy H = -Σ_i target_i * log(softmax_i + ε)
        let mut cross_entropy = 0.0_f32;
        for i in 0..n {
            let prob_i = exp_vals[i] / sum_exp;
            // Add small epsilon for numerical stability in log
            cross_entropy -= row_target[i] * (prob_i + 1e-8_f32).ln();
        }

        kl_errors.push(cross_entropy);
        weighted_loss += is_weights[b_idx] * cross_entropy;
    }

    let loss = weighted_loss / b as f32;

    if loss.is_nan() {
        return Err(RlError::Internal("NaN loss encountered in c51_loss".into()));
    }

    Ok(C51Loss { loss, kl_errors })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a uniform next_probs of shape [B * N]
    fn uniform_probs(b: usize, n: usize) -> Vec<f32> {
        vec![1.0 / n as f32; b * n]
    }

    #[test]
    fn default_config_ok() {
        let cfg = C51Config::default();
        assert_eq!(cfg.n_atoms, 51);
        assert!((cfg.v_min - (-10.0)).abs() < 1e-6);
        assert!((cfg.v_max - 10.0).abs() < 1e-6);
        assert!((cfg.gamma - 0.99).abs() < 1e-6);
    }

    #[test]
    fn support_endpoints() {
        let cfg = C51Config::default();
        let z = c51_support(&cfg);
        assert!(
            (z[0] - cfg.v_min).abs() < 1e-5,
            "first atom != v_min: {}",
            z[0]
        );
        assert!(
            (z[cfg.n_atoms - 1] - cfg.v_max).abs() < 1e-5,
            "last atom != v_max: {}",
            z[cfg.n_atoms - 1]
        );
    }

    #[test]
    fn support_length() {
        let cfg = C51Config::default();
        let z = c51_support(&cfg);
        assert_eq!(z.len(), cfg.n_atoms);
    }

    #[test]
    fn project_sum_to_one() {
        let cfg = C51Config {
            n_atoms: 11,
            v_min: -5.0,
            v_max: 5.0,
            gamma: 0.99,
        };
        let b = 4;
        let n = cfg.n_atoms;
        let rewards = vec![0.5_f32; b];
        let dones = vec![0.0_f32; b];
        let next_probs = uniform_probs(b, n);
        let target = c51_project(&rewards, &dones, &next_probs, &cfg)
            .expect("project should succeed with valid inputs");
        for b_idx in 0..b {
            let row_sum: f32 = target[b_idx * n..(b_idx + 1) * n].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "row {b_idx} sum = {row_sum}");
        }
    }

    #[test]
    fn project_deterministic_terminal() {
        // done=1 → bootstrap ignored; full probability concentrates near r.
        let cfg = C51Config {
            n_atoms: 11,
            v_min: -5.0,
            v_max: 5.0,
            gamma: 0.99,
        };
        let reward = 2.0_f32;
        let rewards = vec![reward];
        let dones = vec![1.0_f32];
        let next_probs = uniform_probs(1, cfg.n_atoms);
        let target = c51_project(&rewards, &dones, &next_probs, &cfg)
            .expect("project should succeed for terminal state");
        // All mass must be near atom closest to reward=2.0
        let z = c51_support(&cfg);
        let total_near_reward: f32 = z
            .iter()
            .zip(&target)
            .filter(|(zi, _)| (*zi - reward).abs() < 1.5)
            .map(|(_, p)| p)
            .sum();
        assert!(
            total_near_reward > 0.9,
            "mass near reward={reward}: {total_near_reward}"
        );
    }

    #[test]
    fn project_uniform_next_stays_bounded() {
        let cfg = C51Config::default();
        let b = 8;
        let next_probs = uniform_probs(b, cfg.n_atoms);
        let rewards = vec![0.0_f32; b];
        let dones = vec![0.0_f32; b];
        let target = c51_project(&rewards, &dones, &next_probs, &cfg)
            .expect("project should succeed with uniform next_probs");
        for &p in &target {
            assert!(p >= 0.0, "negative probability: {p}");
            assert!(p <= 1.0 + 1e-5, "probability > 1: {p}");
        }
    }

    #[test]
    fn project_zero_reward_no_done() {
        // reward=0, done=0, uniform next → target distribution should be roughly uniform
        let cfg = C51Config {
            n_atoms: 11,
            v_min: -5.0,
            v_max: 5.0,
            gamma: 0.99,
        };
        let b = 2;
        let n = cfg.n_atoms;
        let rewards = vec![0.0_f32; b];
        let dones = vec![0.0_f32; b];
        let next_probs = uniform_probs(b, n);
        let target = c51_project(&rewards, &dones, &next_probs, &cfg)
            .expect("project should succeed with zero reward, no done");
        // Each row must sum to 1
        for b_idx in 0..b {
            let row_sum: f32 = target[b_idx * n..(b_idx + 1) * n].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "row {b_idx} sum = {row_sum}");
        }
    }

    #[test]
    fn loss_identical_target_and_pred() {
        // When logits strongly encode the same distribution as target, CE should be small.
        let cfg = C51Config {
            n_atoms: 5,
            v_min: -2.0,
            v_max: 2.0,
            gamma: 0.99,
        };
        let n = cfg.n_atoms;
        let b = 3;
        // Very strong logits pointing to atom 2 (middle)
        let mut logits = vec![-100.0_f32; b * n];
        let mut target_probs = vec![0.0_f32; b * n];
        for b_idx in 0..b {
            logits[b_idx * n + 2] = 100.0;
            target_probs[b_idx * n + 2] = 1.0;
        }
        let is_weights = vec![1.0_f32; b];
        let result = c51_loss(&logits, &target_probs, &is_weights, &cfg)
            .expect("loss should succeed with compatible inputs");
        // CE = -1 * log(1 + 1e-8) ≈ very small
        assert!(
            result.loss < 1.0,
            "loss should be small when pred ≈ target: {}",
            result.loss
        );
    }

    #[test]
    fn loss_uniform_is_weights() {
        let cfg = C51Config {
            n_atoms: 7,
            v_min: -3.0,
            v_max: 3.0,
            gamma: 0.99,
        };
        let b = 4;
        let n = cfg.n_atoms;
        let logits = vec![0.0_f32; b * n]; // uniform softmax
        let target_probs = uniform_probs(b, n);
        let is_weights = vec![1.0_f32; b];
        let result = c51_loss(&logits, &target_probs, &is_weights, &cfg)
            .expect("loss should succeed with uniform inputs");
        // Cross-entropy of uniform over itself = log(N)
        let expected_ce = (n as f32).ln();
        assert!(
            (result.loss - expected_ce).abs() < 0.1,
            "loss={}, expected≈{expected_ce}",
            result.loss
        );
    }

    #[test]
    fn loss_nan_check() {
        let cfg = C51Config {
            n_atoms: 5,
            v_min: -1.0,
            v_max: 1.0,
            gamma: 0.99,
        };
        let n = cfg.n_atoms;
        let b = 2;
        // Infinite logits can produce NaN via inf - inf = NaN in softmax
        let logits = vec![f32::INFINITY; b * n];
        let target_probs = uniform_probs(b, n);
        let is_weights = vec![1.0_f32; b];
        // Should either succeed with finite result or return NanLoss error
        let result = c51_loss(&logits, &target_probs, &is_weights, &cfg);
        if let Ok(c51) = result {
            assert!(!c51.loss.is_nan(), "loss must not be NaN");
        }
        // Err(NanLoss) is also acceptable
    }

    #[test]
    fn loss_batch_size_1() {
        let cfg = C51Config {
            n_atoms: 3,
            v_min: 0.0,
            v_max: 2.0,
            gamma: 0.99,
        };
        let logits = vec![1.0_f32, 2.0, 1.0];
        let target_probs = vec![0.25_f32, 0.5, 0.25];
        let is_weights = vec![1.0_f32];
        let result =
            c51_loss(&logits, &target_probs, &is_weights, &cfg).expect("B=1 loss should succeed");
        assert!(result.loss.is_finite(), "loss must be finite");
    }

    #[test]
    fn loss_kl_errors_length() {
        let cfg = C51Config {
            n_atoms: 5,
            v_min: -2.0,
            v_max: 2.0,
            gamma: 0.99,
        };
        let b = 6;
        let n = cfg.n_atoms;
        let logits = vec![0.0_f32; b * n];
        let target_probs = uniform_probs(b, n);
        let is_weights = vec![1.0_f32; b];
        let result =
            c51_loss(&logits, &target_probs, &is_weights, &cfg).expect("loss should succeed");
        assert_eq!(result.kl_errors.len(), b);
    }

    #[test]
    fn err_empty_rewards() {
        let cfg = C51Config::default();
        let result = c51_project(&[], &[], &[], &cfg);
        assert!(result.is_err(), "empty rewards should return Err");
    }

    #[test]
    fn err_n_atoms_lt_2() {
        let cfg = C51Config {
            n_atoms: 1,
            v_min: -1.0,
            v_max: 1.0,
            gamma: 0.99,
        };
        let result = c51_project(&[0.0], &[0.0], &[1.0], &cfg);
        assert!(result.is_err(), "n_atoms=1 should return Err");
    }

    #[test]
    fn err_vmin_geq_vmax() {
        let cfg = C51Config {
            n_atoms: 10,
            v_min: 5.0,
            v_max: 5.0,
            gamma: 0.99,
        };
        let n = cfg.n_atoms;
        let result = c51_project(&[0.0], &[0.0], &vec![0.1_f32; n], &cfg);
        assert!(result.is_err(), "v_min >= v_max should return Err");
    }

    #[test]
    fn err_dimension_mismatch() {
        let cfg = C51Config {
            n_atoms: 5,
            v_min: -2.0,
            v_max: 2.0,
            gamma: 0.99,
        };
        // B=2, N=5 → expected next_probs len = 10, but pass 7
        let result = c51_project(&[0.0, 0.0], &[0.0, 0.0], &[0.2_f32; 7], &cfg);
        assert!(result.is_err(), "dimension mismatch should return Err");
    }

    #[test]
    fn project_output_nonneg() {
        let cfg = C51Config {
            n_atoms: 11,
            v_min: -5.0,
            v_max: 5.0,
            gamma: 0.99,
        };
        let b = 10;
        let n = cfg.n_atoms;
        let rewards: Vec<f32> = (0..b).map(|i| (i as f32) - 4.5).collect();
        let dones = vec![0.0_f32; b];
        let next_probs = uniform_probs(b, n);
        let target =
            c51_project(&rewards, &dones, &next_probs, &cfg).expect("project should succeed");
        for &p in &target {
            assert!(p >= -1e-8, "negative probability: {p}");
        }
    }

    #[test]
    fn full_roundtrip() {
        // project then loss: no error, loss is finite
        let cfg = C51Config {
            n_atoms: 11,
            v_min: -5.0,
            v_max: 5.0,
            gamma: 0.99,
        };
        let b = 4;
        let n = cfg.n_atoms;
        let rewards = vec![1.0_f32, -1.0, 0.0, 2.0];
        let dones = vec![0.0_f32, 0.0, 1.0, 0.0];
        let next_probs = uniform_probs(b, n);

        let target_probs =
            c51_project(&rewards, &dones, &next_probs, &cfg).expect("project should succeed");

        // Use logits that produce a near-uniform distribution
        let logits = vec![0.0_f32; b * n];
        let is_weights = vec![1.0_f32; b];

        let result = c51_loss(&logits, &target_probs, &is_weights, &cfg)
            .expect("loss after project should succeed");
        assert!(
            result.loss.is_finite(),
            "roundtrip loss must be finite: {}",
            result.loss
        );
    }
}

//! # IQN — Implicit Quantile Networks Loss
//!
//! Dabney, Ostrovski, Silver, Munos (2018)
//! "Implicit Quantile Networks for Distributional Reinforcement Learning". ICML 2018.
//!
//! Unlike QR-DQN which fixes quantile fractions to deterministic midpoints, IQN
//! samples quantile fractions τ ~ U(0,1) at each training step and embeds them
//! via a cosine basis before passing through a shared network. This gives an
//! implicit representation of the full quantile function Z^π(s,a) rather than a
//! finite N-atom discretisation.
//!
//! ## Cosine embedding
//!
//! ```text
//! φ_l(τ) = cos(π · l · τ)   for l = 0 .. embedding_dim − 1
//! ```
//!
//! (The caller then multiplies by a learnable weight matrix and applies ReLU.)
//!
//! ## Quantile Huber loss
//!
//! Given N online fractions τ_i and N' target fractions τ'_j:
//!
//! ```text
//! δ_{ij} = target_j − pred_i
//!
//! L^κ(u) = 0.5 u²               if |u| ≤ κ
//!           κ (|u| − 0.5 κ)     otherwise     [Huber]
//!
//! ρ^κ_{τ_i}(δ) = |τ_i − 1(δ < 0)| · L^κ(δ)
//!
//! per-sample loss_b = (1/N) Σ_i (1/N') Σ_j ρ^κ_{τ_i}(δ_{ij})
//!
//! batch loss = (1/B) Σ_b  is_weight_b · loss_b
//! ```

use std::f32::consts::PI;

use crate::error::{RlError, RlResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// IQN configuration.
#[derive(Debug, Clone, Copy)]
pub struct IqnConfig {
    /// Number of online quantile fractions N sampled from U(0,1) (default 8).
    pub n_tau: usize,
    /// Number of target quantile fractions N' (default 8).
    pub n_tau_prime: usize,
    /// Cosine embedding dimension (number of cosine features; default 64).
    pub embedding_dim: usize,
    /// Huber loss threshold κ (default 1.0).
    pub kappa: f32,
    /// Discount factor γ (must be in (0, 1]; default 0.99).
    pub gamma: f32,
}

impl Default for IqnConfig {
    fn default() -> Self {
        Self {
            n_tau: 8,
            n_tau_prime: 8,
            embedding_dim: 64,
            kappa: 1.0,
            gamma: 0.99,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────────

/// IQN loss output.
#[derive(Debug, Clone)]
pub struct IqnLoss {
    /// Mean quantile Huber loss over the batch (scalar to minimise).
    pub loss: f32,
    /// Per-sample mean absolute TD error over all (i,j) pairs.
    /// Length == batch size B.  Used for PER priority updates.
    pub td_errors: Vec<f32>,
}

// ─── Validation helpers ───────────────────────────────────────────────────────

/// Validate IQN config.
fn validate_cfg(cfg: &IqnConfig) -> RlResult<()> {
    if cfg.n_tau == 0 {
        return Err(RlError::InvalidHyperparameter {
            name: "n_tau".into(),
            msg: "must be >= 1".into(),
        });
    }
    if cfg.n_tau_prime == 0 {
        return Err(RlError::InvalidHyperparameter {
            name: "n_tau_prime".into(),
            msg: "must be >= 1".into(),
        });
    }
    if cfg.embedding_dim == 0 {
        return Err(RlError::InvalidHyperparameter {
            name: "embedding_dim".into(),
            msg: "must be >= 1".into(),
        });
    }
    if cfg.kappa <= 0.0 {
        return Err(RlError::InvalidHyperparameter {
            name: "kappa".into(),
            msg: "must be > 0".into(),
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

// ─── Huber loss (private) ─────────────────────────────────────────────────────

/// Scalar Huber (smooth L1) loss: L_κ(u).
#[inline]
fn huber(u: f32, kappa: f32) -> f32 {
    if u.abs() <= kappa {
        0.5 * u * u
    } else {
        kappa * (u.abs() - 0.5 * kappa)
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Sample `n` quantile fractions uniformly from [0, 1).
///
/// **CRITICAL**: this function divides by 2³¹ (= 2\_147\_483\_648), NOT by
/// `u32::MAX + 1`. The reason is that `LcgRng::next_u32` returns values in
/// `[0, 2³¹)` (only the high 31 bits of the state), so dividing by 2³² would
/// confine every sample to `[0, 0.5)` — which would systematically under-sample
/// the upper half of the quantile space and break the IQN loss.
///
/// Using `next_u32() as f32 / 4_294_967_296.0` gives the correct `[0, 1)` range.
#[must_use]
pub fn sample_taus(n: usize, rng: &mut LcgRng) -> Vec<f32> {
    (0..n)
        .map(|_| rng.next_u32() as f32 / 4_294_967_296.0_f32)
        .collect()
}

/// Compute the cosine embedding features for a single quantile fraction τ.
///
/// Returns a vector of length `embedding_dim` where:
/// ```text
/// φ[l] = cos(π · l · τ)   for l = 0 .. embedding_dim − 1
/// ```
///
/// Note: `φ[0]` = cos(0) = 1 always, regardless of τ.
/// The caller is responsible for multiplying the output by a learnable weight
/// matrix and applying ReLU to obtain the final embedding.
#[must_use]
pub fn iqn_cosine_embedding(tau: f32, embedding_dim: usize) -> Vec<f32> {
    (0..embedding_dim)
        .map(|l| (PI * l as f32 * tau).cos())
        .collect()
}

/// Build IQN Bellman targets from next-state quantile values.
///
/// Applies the standard distributional Bellman backup:
/// `y[b·N' + j] = rewards[b] + γ · (1 − dones[b]) · next_quantiles[b·N' + j]`
///
/// # Arguments
///
/// * `rewards`        — `[B]` rewards r_b.
/// * `dones`          — `[B]` done flags (1.0 = terminal).
/// * `next_quantiles` — `[B × N']` quantile values from target network.
/// * `n_tau_prime`    — N', number of target quantile fractions.
/// * `gamma`          — discount factor γ.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] — inconsistent slice lengths or N'=0.
pub fn iqn_targets(
    rewards: &[f32],
    dones: &[f32],
    next_quantiles: &[f32],
    n_tau_prime: usize,
    gamma: f32,
) -> RlResult<Vec<f32>> {
    if n_tau_prime == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
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
    if next_quantiles.len() != b * n_tau_prime {
        return Err(RlError::DimensionMismatch {
            expected: b * n_tau_prime,
            got: next_quantiles.len(),
        });
    }

    let mut targets = Vec::with_capacity(b * n_tau_prime);
    for b_idx in 0..b {
        let r = rewards[b_idx];
        let gamma_factor = gamma * (1.0 - dones[b_idx]);
        for j in 0..n_tau_prime {
            targets.push(r + gamma_factor * next_quantiles[b_idx * n_tau_prime + j]);
        }
    }

    Ok(targets)
}

/// Compute the IQN quantile Huber loss.
///
/// For each sample b and each (i, j) prediction-target pair:
/// ```text
/// δ_{ij}  = target[b·N' + j] − pred[b·N + i]
/// τ_i     = taus[b·N + i]
/// ρ       = |τ_i − 1(δ < 0)| · L^κ(δ)
///
/// loss_b  = (1/N)(1/N') Σ_i Σ_j ρ
/// batch   = (1/B) Σ_b is_weight_b · loss_b
/// ```
///
/// # Arguments
///
/// * `pred`       — `[B × N]` online quantile values at the sampled τ_i.
/// * `target`     — `[B × N']` target quantile atoms (from [`iqn_targets`]).
/// * `taus`       — `[B × N]` the τ_i fractions used to compute `pred`.
/// * `is_weights` — `[B]` PER importance-sampling weights (1.0 when uniform).
/// * `cfg`        — IQN configuration.
///
/// # Errors
///
/// * [`RlError::InvalidHyperparameter`] — invalid config fields.
/// * [`RlError::DimensionMismatch`]     — slice lengths inconsistent.
/// * [`RlError::Internal`]              — NaN loss encountered.
pub fn iqn_loss(
    pred: &[f32],
    target: &[f32],
    taus: &[f32],
    is_weights: &[f32],
    cfg: IqnConfig,
) -> RlResult<IqnLoss> {
    validate_cfg(&cfg)?;

    let b = is_weights.len();
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let n = cfg.n_tau;
    let n_prime = cfg.n_tau_prime;

    if pred.len() != b * n {
        return Err(RlError::DimensionMismatch {
            expected: b * n,
            got: pred.len(),
        });
    }
    if target.len() != b * n_prime {
        return Err(RlError::DimensionMismatch {
            expected: b * n_prime,
            got: target.len(),
        });
    }
    if taus.len() != b * n {
        return Err(RlError::DimensionMismatch {
            expected: b * n,
            got: taus.len(),
        });
    }

    let mut td_errors = Vec::with_capacity(b);
    let mut weighted_loss = 0.0_f32;

    for b_idx in 0..b {
        let pred_row = &pred[b_idx * n..(b_idx + 1) * n];
        let target_row = &target[b_idx * n_prime..(b_idx + 1) * n_prime];
        let tau_row = &taus[b_idx * n..(b_idx + 1) * n];

        let mut loss_b = 0.0_f32;
        let mut td_err_b = 0.0_f32;

        for (i, (&pred_i, &tau_i)) in pred_row.iter().zip(tau_row.iter()).enumerate() {
            let _ = i; // i is implicitly used via zip offset
            for &target_j in target_row {
                let delta = target_j - pred_i;
                let huber_val = huber(delta, cfg.kappa);
                // Asymmetric quantile weight: |τ_i − 1(δ < 0)|
                let indicator = if delta < 0.0 { 1.0_f32 } else { 0.0_f32 };
                let rho = (tau_i - indicator).abs() * huber_val;

                loss_b += rho;
                td_err_b += delta.abs();
            }
        }

        // Normalise: (1/N)(1/N') averaged over all (i,j) pairs.
        let n_pairs = (n * n_prime) as f32;
        loss_b /= n_pairs;
        td_err_b /= n_pairs;

        td_errors.push(td_err_b);
        weighted_loss += is_weights[b_idx] * loss_b;
    }

    let loss = weighted_loss / b as f32;

    if loss.is_nan() {
        return Err(RlError::Internal("NaN loss encountered in iqn_loss".into()));
    }

    Ok(IqnLoss { loss, td_errors })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── sample_taus ───────────────────────────────────────────────────────────

    /// THE CRITICAL RNG GUARD: verify that sample_taus produces values above 0.5.
    /// If the broken next_f32 recipe (divides by 2³²) were used, all 1000 samples
    /// would fall in [0, 0.5) and this test would fail.
    #[test]
    fn sample_taus_spans_above_half() {
        let mut rng = LcgRng::new(42);
        let taus = sample_taus(1000, &mut rng);
        assert!(
            taus.iter().any(|&t| t > 0.5),
            "sample_taus must produce values > 0.5; got none — likely broken ÷2³² recipe"
        );
    }

    #[test]
    fn sample_taus_all_in_unit_interval() {
        let mut rng = LcgRng::new(17);
        let taus = sample_taus(1000, &mut rng);
        for &t in &taus {
            assert!((0.0..1.0).contains(&t), "tau={t} is outside [0,1)");
        }
    }

    // ── iqn_cosine_embedding ─────────────────────────────────────────────────

    #[test]
    fn cosine_embedding_tau_zero_all_ones() {
        // cos(π·l·0) = cos(0) = 1 for all l.
        let phi = iqn_cosine_embedding(0.0, 8);
        for (l, &v) in phi.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "phi[{l}]={v} != 1 for tau=0");
        }
    }

    #[test]
    fn cosine_embedding_tau_one_alternating() {
        // cos(π·l·1): l=0 → 1, l=1 → cos(π)=−1, l=2 → cos(2π)=1, l=3 → −1, …
        let phi = iqn_cosine_embedding(1.0, 6);
        let expected = [1.0_f32, -1.0, 1.0, -1.0, 1.0, -1.0];
        for (l, (&v, &e)) in phi.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-5, "phi[{l}]={v}, expected={e} for tau=1");
        }
    }

    #[test]
    fn cosine_embedding_length() {
        let dim = 64;
        let phi = iqn_cosine_embedding(0.3, dim);
        assert_eq!(phi.len(), dim);
    }

    // ── iqn_targets ──────────────────────────────────────────────────────────

    #[test]
    fn iqn_targets_no_discount_at_terminal() {
        // done=1 → target = reward only (γ term drops out).
        let rewards = vec![3.0_f32];
        let dones = vec![1.0_f32];
        let n_prime = 4;
        let next_q = vec![999.0_f32; n_prime]; // should be ignored
        let gamma = 0.99;
        let targets = iqn_targets(&rewards, &dones, &next_q, n_prime, gamma)
            .expect("terminal iqn_targets should succeed");
        for &t in &targets {
            assert!((t - 3.0).abs() < 1e-5, "terminal target={t}, expected 3.0");
        }
    }

    #[test]
    fn iqn_targets_discount_applied() {
        // done=0, r=1, γ=0.9, q'=2 → target = 1 + 0.9*2 = 2.8
        let rewards = vec![1.0_f32];
        let dones = vec![0.0_f32];
        let next_q = vec![2.0_f32];
        let targets =
            iqn_targets(&rewards, &dones, &next_q, 1, 0.9).expect("iqn_targets should succeed");
        assert!(
            (targets[0] - 2.8).abs() < 1e-5,
            "target={}, expected 2.8",
            targets[0]
        );
    }

    #[test]
    fn iqn_targets_shape() {
        let b = 3;
        let n_prime = 5;
        let rewards = vec![0.5_f32; b];
        let dones = vec![0.0_f32; b];
        let next_q = vec![1.0_f32; b * n_prime];
        let targets = iqn_targets(&rewards, &dones, &next_q, n_prime, 0.99)
            .expect("iqn_targets shape should succeed");
        assert_eq!(targets.len(), b * n_prime);
    }

    // ── iqn_loss ─────────────────────────────────────────────────────────────

    #[test]
    fn iqn_loss_zero_when_pred_equals_target() {
        let cfg = IqnConfig {
            n_tau: 4,
            n_tau_prime: 4,
            embedding_dim: 16,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 2;
        let n = cfg.n_tau;
        let n_prime = cfg.n_tau_prime;
        let val = 1.5_f32;
        let pred = vec![val; b * n];
        let target = vec![val; b * n_prime];
        let taus = vec![0.5_f32; b * n];
        let is_weights = vec![1.0_f32; b];
        let result = iqn_loss(&pred, &target, &taus, &is_weights, cfg)
            .expect("zero-error loss should succeed");
        assert!(
            result.loss.abs() < 1e-6,
            "loss should be 0 when pred==target: {}",
            result.loss
        );
    }

    #[test]
    fn iqn_loss_positive_mismatched() {
        let cfg = IqnConfig::default();
        let b = 2;
        let n = cfg.n_tau;
        let n_prime = cfg.n_tau_prime;
        let pred = vec![0.0_f32; b * n];
        let target = vec![1.0_f32; b * n_prime];
        let taus = vec![0.5_f32; b * n];
        let is_weights = vec![1.0_f32; b];
        let result = iqn_loss(&pred, &target, &taus, &is_weights, cfg)
            .expect("positive mismatch loss should succeed");
        assert!(result.loss > 0.0, "loss must be > 0 when pred != target");
    }

    #[test]
    fn iqn_loss_asymmetric_huber() {
        // With τ=0.9 and κ=1, the quantile weight is higher when target > pred
        // (δ > 0) → |τ − 0| = 0.9, vs δ < 0 → |τ − 1| = 0.1.
        // So loss with positive δ=+d should be higher than loss with negative δ=−d.
        let cfg = IqnConfig {
            n_tau: 1,
            n_tau_prime: 1,
            embedding_dim: 64,
            kappa: 1.0,
            gamma: 0.99,
        };
        let d = 0.5_f32;
        // positive delta: target - pred = +d
        let pred_pos = vec![0.0_f32];
        let target_pos = vec![d];
        let taus_pos = vec![0.9_f32];
        let is_w = vec![1.0_f32];
        let loss_pos = iqn_loss(&pred_pos, &target_pos, &taus_pos, &is_w, cfg)
            .expect("asymmetric pos loss should succeed")
            .loss;

        // negative delta: target - pred = -d
        let pred_neg = vec![d];
        let target_neg = vec![0.0_f32];
        let taus_neg = vec![0.9_f32];
        let loss_neg = iqn_loss(&pred_neg, &target_neg, &taus_neg, &is_w, cfg)
            .expect("asymmetric neg loss should succeed")
            .loss;

        assert!(
            loss_pos > loss_neg,
            "with τ=0.9: loss(+δ)={loss_pos} should > loss(-δ)={loss_neg}"
        );
    }

    #[test]
    fn iqn_loss_shape_b1_n8() {
        let cfg = IqnConfig {
            n_tau: 8,
            n_tau_prime: 8,
            embedding_dim: 64,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 1;
        let n = cfg.n_tau;
        let n_prime = cfg.n_tau_prime;
        let pred = vec![0.5_f32; b * n];
        let target = vec![1.0_f32; b * n_prime];
        let taus = vec![0.5_f32; b * n];
        let is_weights = vec![1.0_f32; b];
        let result = iqn_loss(&pred, &target, &taus, &is_weights, cfg)
            .expect("B=1,N=8 iqn_loss should succeed");
        assert!(result.loss.is_finite(), "loss must be finite");
        assert_eq!(result.td_errors.len(), 1);
    }

    #[test]
    fn iqn_loss_shape_b4_n8() {
        let cfg = IqnConfig {
            n_tau: 8,
            n_tau_prime: 8,
            embedding_dim: 64,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 4;
        let n = cfg.n_tau;
        let n_prime = cfg.n_tau_prime;
        let pred = vec![0.5_f32; b * n];
        let target = vec![1.0_f32; b * n_prime];
        let taus = vec![0.5_f32; b * n];
        let is_weights = vec![1.0_f32; b];
        let result = iqn_loss(&pred, &target, &taus, &is_weights, cfg)
            .expect("B=4,N=8 iqn_loss should succeed");
        assert!(result.loss.is_finite(), "loss must be finite");
        assert_eq!(result.td_errors.len(), b);
    }

    // ── DimensionMismatch guards ──────────────────────────────────────────────

    #[test]
    fn iqn_loss_dim_mismatch_pred() {
        let cfg = IqnConfig {
            n_tau: 4,
            n_tau_prime: 4,
            ..IqnConfig::default()
        };
        let b = 2;
        let n_prime = cfg.n_tau_prime;
        // pred should be b*n_tau = 8, provide 7 instead
        let pred = vec![0.0_f32; b * cfg.n_tau - 1];
        let target = vec![1.0_f32; b * n_prime];
        let taus = vec![0.5_f32; b * cfg.n_tau];
        let is_weights = vec![1.0_f32; b];
        let result = iqn_loss(&pred, &target, &taus, &is_weights, cfg);
        assert!(
            result.is_err(),
            "wrong pred len should return DimensionMismatch"
        );
    }

    #[test]
    fn iqn_loss_dim_mismatch_target() {
        let cfg = IqnConfig {
            n_tau: 4,
            n_tau_prime: 4,
            ..IqnConfig::default()
        };
        let b = 2;
        // target should be b*n_prime = 8, provide 5 instead
        let pred = vec![0.0_f32; b * cfg.n_tau];
        let target = vec![1.0_f32; 5];
        let taus = vec![0.5_f32; b * cfg.n_tau];
        let is_weights = vec![1.0_f32; b];
        let result = iqn_loss(&pred, &target, &taus, &is_weights, cfg);
        assert!(
            result.is_err(),
            "wrong target len should return DimensionMismatch"
        );
    }

    #[test]
    fn iqn_loss_dim_mismatch_taus() {
        let cfg = IqnConfig {
            n_tau: 4,
            n_tau_prime: 4,
            ..IqnConfig::default()
        };
        let b = 2;
        let n_prime = cfg.n_tau_prime;
        let pred = vec![0.0_f32; b * cfg.n_tau];
        let target = vec![1.0_f32; b * n_prime];
        // taus should be b*n_tau = 8, provide 3 instead
        let taus = vec![0.5_f32; 3];
        let is_weights = vec![1.0_f32; b];
        let result = iqn_loss(&pred, &target, &taus, &is_weights, cfg);
        assert!(
            result.is_err(),
            "wrong taus len should return DimensionMismatch"
        );
    }

    #[test]
    fn iqn_loss_dim_mismatch_is_weights() {
        let cfg = IqnConfig {
            n_tau: 4,
            n_tau_prime: 4,
            ..IqnConfig::default()
        };
        let b = 2;
        let n_prime = cfg.n_tau_prime;
        let pred = vec![0.0_f32; b * cfg.n_tau];
        let target = vec![1.0_f32; b * n_prime];
        let taus = vec![0.5_f32; b * cfg.n_tau];
        // is_weights should be b=2, provide 3 instead → pred len mismatch detected first
        // Actually is_weights len is b, so 3 is_weights means b=3, and pred len = 8 != 3*4=12
        let is_weights = vec![1.0_f32; 3];
        let result = iqn_loss(&pred, &target, &taus, &is_weights, cfg);
        assert!(
            result.is_err(),
            "wrong is_weights len should return DimensionMismatch"
        );
    }

    // ── NaN guard ─────────────────────────────────────────────────────────────

    #[test]
    fn iqn_loss_nan_guard() {
        let cfg = IqnConfig {
            n_tau: 2,
            n_tau_prime: 2,
            embedding_dim: 16,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 1;
        let n = cfg.n_tau;
        let n_prime = cfg.n_tau_prime;
        let pred = vec![f32::NAN; b * n];
        let target = vec![1.0_f32; b * n_prime];
        let taus = vec![0.5_f32; b * n];
        let is_weights = vec![1.0_f32; b];
        let result = iqn_loss(&pred, &target, &taus, &is_weights, cfg);
        // NaN in pred should produce NaN loss → Internal error
        assert!(result.is_err(), "NaN input should return Internal error");
    }

    // ── validate_cfg guards ───────────────────────────────────────────────────

    #[test]
    fn validate_cfg_bad_ntau() {
        let cfg = IqnConfig {
            n_tau: 0,
            ..IqnConfig::default()
        };
        assert!(
            validate_cfg(&cfg).is_err(),
            "n_tau=0 should return InvalidHyperparameter"
        );
    }

    #[test]
    fn validate_cfg_bad_gamma() {
        let cfg = IqnConfig {
            gamma: 1.01,
            ..IqnConfig::default()
        };
        assert!(
            validate_cfg(&cfg).is_err(),
            "gamma>1 should return InvalidHyperparameter"
        );
    }

    #[test]
    fn validate_cfg_bad_kappa() {
        let cfg = IqnConfig {
            kappa: 0.0,
            ..IqnConfig::default()
        };
        assert!(
            validate_cfg(&cfg).is_err(),
            "kappa=0 should return InvalidHyperparameter"
        );
    }

    // ── IS weight upweighting ─────────────────────────────────────────────────

    #[test]
    fn iqn_loss_is_weights_upweight_error() {
        // With a 2-sample batch where both samples have identical loss,
        // doubling the IS weight of the first sample should double its
        // relative contribution, raising the total loss.
        let cfg = IqnConfig {
            n_tau: 2,
            n_tau_prime: 2,
            embedding_dim: 16,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 2;
        let n = cfg.n_tau;
        let n_prime = cfg.n_tau_prime;
        let pred = vec![0.0_f32; b * n];
        let target = vec![1.0_f32; b * n_prime];
        let taus = vec![0.5_f32; b * n];

        let is_uniform = vec![1.0_f32; b];
        let loss_uniform = iqn_loss(&pred, &target, &taus, &is_uniform, cfg)
            .expect("uniform IS iqn_loss should succeed")
            .loss;

        // Weight the first sample by 2x, second by 1x
        let is_upweighted = vec![2.0_f32, 1.0];
        let loss_upweighted = iqn_loss(&pred, &target, &taus, &is_upweighted, cfg)
            .expect("upweighted IS iqn_loss should succeed")
            .loss;

        // With uniform samples, loss_upweighted = (2*loss_b + 1*loss_b)/2 = 1.5*loss_b
        // vs loss_uniform = (1*loss_b + 1*loss_b)/2 = loss_b
        assert!(
            loss_upweighted > loss_uniform,
            "2x IS weight should increase loss: uniform={loss_uniform}, upweighted={loss_upweighted}"
        );
    }
}

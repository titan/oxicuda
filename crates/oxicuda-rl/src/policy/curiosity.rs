//! # Intrinsic-curiosity modules — ICM and RND.
//!
//! Exploration bonuses computed from prediction error in a learned feature
//! space. Both return a non-negative scalar *intrinsic reward* per transition
//! that augments the environment (extrinsic) reward.
//!
//! * [`IcmReward`] — **Intrinsic Curiosity Module** (Pathak et al. 2017,
//!   <https://arxiv.org/abs/1705.05363>): a forward model predicts the next
//!   feature `φ̂(s_{t+1})` from `(φ(s_t), a_t)`; the intrinsic reward is the
//!   prediction error `r^i_t = (η/2)·‖φ̂(s_{t+1}) − φ(s_{t+1})‖²`. An inverse
//!   model loss (predicting `a_t` from `φ(s_t), φ(s_{t+1})`) is provided for
//!   training the encoder to ignore uncontrollable noise.
//!
//! * [`RndReward`] — **Random Network Distillation** (Burda et al. 2018,
//!   <https://arxiv.org/abs/1810.12894>): a fixed random *target* network maps
//!   the next state to an embedding; a trained *predictor* network regresses
//!   toward it. The intrinsic reward is the squared embedding error
//!   `r^i_t = ‖predictor(s_{t+1}) − target(s_{t+1})‖²`, normalised by a running
//!   std for scale invariance.
//!
//! All functions operate on flat `&[f32]` feature/embedding slices; the caller
//! owns the networks and supplies their outputs.

use crate::error::{RlError, RlResult};
use crate::normalize::RunningStats;

// ─── ICM ────────────────────────────────────────────────────────────────────────

/// Configuration for the ICM intrinsic reward.
#[derive(Debug, Clone, Copy)]
pub struct IcmConfig {
    /// Forward-model reward scaling η (must be `> 0`; default 0.5).
    pub eta: f32,
    /// Weight β balancing forward vs inverse model loss in `[0, 1]`
    /// (default 0.2: 0.2·inverse + 0.8·forward, per Pathak et al. 2017).
    pub beta: f32,
}

impl Default for IcmConfig {
    fn default() -> Self {
        Self {
            eta: 0.5,
            beta: 0.2,
        }
    }
}

/// Output of an ICM evaluation.
#[derive(Debug, Clone)]
pub struct IcmReward {
    /// Per-transition intrinsic reward `(η/2)·‖φ̂' − φ'‖²` (length `B`).
    pub intrinsic_rewards: Vec<f32>,
    /// Mean forward-model loss `½·‖φ̂' − φ'‖²` over the batch.
    pub forward_loss: f32,
}

/// Compute the ICM forward-model intrinsic reward and forward loss.
///
/// # Arguments
/// * `predicted_next_feat` — `[B × F]` forward-model predictions `φ̂(s_{t+1})`.
/// * `actual_next_feat`    — `[B × F]` encoder features `φ(s_{t+1})`.
/// * `feat_dim`            — feature dimensionality `F`.
/// * `cfg`                 — ICM configuration.
///
/// # Errors
/// * [`RlError::InvalidHyperparameter`] — invalid config / `feat_dim == 0`.
/// * [`RlError::DimensionMismatch`]     — slice shapes inconsistent.
pub fn icm_intrinsic_reward(
    predicted_next_feat: &[f32],
    actual_next_feat: &[f32],
    feat_dim: usize,
    cfg: IcmConfig,
) -> RlResult<IcmReward> {
    if cfg.eta <= 0.0 {
        return Err(RlError::InvalidHyperparameter {
            name: "eta".into(),
            msg: "must be > 0".into(),
        });
    }
    if !(0.0..=1.0).contains(&cfg.beta) {
        return Err(RlError::InvalidHyperparameter {
            name: "beta".into(),
            msg: "must be in [0, 1]".into(),
        });
    }
    if feat_dim == 0 {
        return Err(RlError::InvalidHyperparameter {
            name: "feat_dim".into(),
            msg: "must be > 0".into(),
        });
    }
    if predicted_next_feat.len() != actual_next_feat.len() {
        return Err(RlError::DimensionMismatch {
            expected: predicted_next_feat.len(),
            got: actual_next_feat.len(),
        });
    }
    if predicted_next_feat.len() % feat_dim != 0 {
        return Err(RlError::DimensionMismatch {
            expected: feat_dim,
            got: predicted_next_feat.len(),
        });
    }

    let b = predicted_next_feat.len() / feat_dim;
    let mut intrinsic_rewards = Vec::with_capacity(b);
    let mut total_forward = 0.0_f32;
    for i in 0..b {
        let p = &predicted_next_feat[i * feat_dim..(i + 1) * feat_dim];
        let a = &actual_next_feat[i * feat_dim..(i + 1) * feat_dim];
        let sq: f32 = p.iter().zip(a).map(|(&pi, &ai)| (pi - ai).powi(2)).sum();
        // Forward loss is ½‖·‖²; intrinsic reward scales it by η.
        let half_sq = 0.5 * sq;
        total_forward += half_sq;
        intrinsic_rewards.push(cfg.eta * half_sq);
    }
    let forward_loss = total_forward / b as f32;
    Ok(IcmReward {
        intrinsic_rewards,
        forward_loss,
    })
}

/// Compute the ICM inverse-model loss: cross-entropy between the predicted
/// action distribution `â_t` (from `φ(s_t), φ(s_{t+1})`) and the true action.
///
/// `L_I = − mean_t  log p(â_t = a_t)`.
///
/// # Arguments
/// * `predicted_action_probs` — `[B × A]` softmax probabilities over actions.
/// * `true_actions`           — `[B]` discrete action indices.
/// * `n_actions`              — number of actions `A`.
///
/// # Errors
/// * [`RlError::InvalidHyperparameter`] — `n_actions == 0`.
/// * [`RlError::DimensionMismatch`]     — shape mismatch or out-of-range action.
pub fn icm_inverse_loss(
    predicted_action_probs: &[f32],
    true_actions: &[usize],
    n_actions: usize,
) -> RlResult<f32> {
    if n_actions == 0 {
        return Err(RlError::InvalidHyperparameter {
            name: "n_actions".into(),
            msg: "must be > 0".into(),
        });
    }
    let b = true_actions.len();
    if predicted_action_probs.len() != b * n_actions {
        return Err(RlError::DimensionMismatch {
            expected: b * n_actions,
            got: predicted_action_probs.len(),
        });
    }
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let mut total = 0.0_f32;
    for (i, &a) in true_actions.iter().enumerate() {
        if a >= n_actions {
            return Err(RlError::DimensionMismatch {
                expected: n_actions,
                got: a,
            });
        }
        let p = predicted_action_probs[i * n_actions + a].clamp(1e-10, 1.0);
        total += -p.ln();
    }
    Ok(total / b as f32)
}

// ─── RND ────────────────────────────────────────────────────────────────────────

/// Random Network Distillation intrinsic-reward estimator.
///
/// Maintains a [`RunningStats`] tracker over the raw squared prediction error
/// so that the returned bonus is scale-normalised across training, as
/// recommended by Burda et al. 2018.
#[derive(Debug, Clone)]
pub struct RndReward {
    /// Embedding dimensionality `E`.
    embed_dim: usize,
    /// Running stats over the *scalar* squared error for reward normalisation.
    error_stats: RunningStats,
}

impl RndReward {
    /// Create a new RND estimator for embeddings of dimension `embed_dim`.
    ///
    /// # Errors
    /// * [`RlError::InvalidHyperparameter`] if `embed_dim == 0`.
    pub fn new(embed_dim: usize) -> RlResult<Self> {
        if embed_dim == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "embed_dim".into(),
                msg: "must be > 0".into(),
            });
        }
        Ok(Self {
            embed_dim,
            error_stats: RunningStats::new(1),
        })
    }

    /// Number of running-error samples observed so far.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.error_stats.count()
    }

    /// Compute normalised intrinsic rewards and update the running error stats.
    ///
    /// `r^i = ‖predictor − target‖² / (std(error) + ε)`.
    ///
    /// On the first call (no variance yet) the raw error is returned (no scale).
    ///
    /// # Arguments
    /// * `predictor_embed` — `[B × E]` predictor-network embeddings.
    /// * `target_embed`    — `[B × E]` fixed-random target embeddings.
    ///
    /// # Errors
    /// * [`RlError::DimensionMismatch`] — shape mismatch.
    pub fn intrinsic_reward(
        &mut self,
        predictor_embed: &[f32],
        target_embed: &[f32],
    ) -> RlResult<Vec<f32>> {
        if predictor_embed.len() != target_embed.len() {
            return Err(RlError::DimensionMismatch {
                expected: predictor_embed.len(),
                got: target_embed.len(),
            });
        }
        if predictor_embed.len() % self.embed_dim != 0 {
            return Err(RlError::DimensionMismatch {
                expected: self.embed_dim,
                got: predictor_embed.len(),
            });
        }
        let b = predictor_embed.len() / self.embed_dim;

        // First compute raw squared errors and update running stats.
        let mut raw = Vec::with_capacity(b);
        for i in 0..b {
            let p = &predictor_embed[i * self.embed_dim..(i + 1) * self.embed_dim];
            let t = &target_embed[i * self.embed_dim..(i + 1) * self.embed_dim];
            let sq: f32 = p.iter().zip(t).map(|(&pi, &ti)| (pi - ti).powi(2)).sum();
            raw.push(sq);
            self.error_stats.update(&[sq])?;
        }

        // Normalise by running std (Welford). std is 0 before 2 samples.
        let std = self.error_stats.std_f32();
        let scale = std[0];
        let out = raw
            .iter()
            .map(|&r| if scale > 1e-8 { r / scale } else { r })
            .collect();
        Ok(out)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icm_default_config() {
        let c = IcmConfig::default();
        assert!((c.eta - 0.5).abs() < 1e-6);
        assert!((c.beta - 0.2).abs() < 1e-6);
    }

    #[test]
    fn icm_zero_error_zero_reward() {
        // Perfect forward prediction ⇒ zero intrinsic reward and zero loss.
        let feat = vec![1.0_f32, 2.0, 3.0, 4.0]; // B=2, F=2
        let out = icm_intrinsic_reward(&feat, &feat, 2, IcmConfig::default()).expect("ok");
        for &r in &out.intrinsic_rewards {
            assert!(r.abs() < 1e-6, "reward should be 0, got {r}");
        }
        assert!(out.forward_loss.abs() < 1e-6);
    }

    #[test]
    fn icm_reward_scales_with_eta() {
        let pred = vec![0.0_f32, 0.0];
        let actual = vec![1.0_f32, 0.0]; // sq error = 1
        let cfg1 = IcmConfig {
            eta: 1.0,
            beta: 0.2,
        };
        let cfg2 = IcmConfig {
            eta: 2.0,
            beta: 0.2,
        };
        let r1 = icm_intrinsic_reward(&pred, &actual, 2, cfg1).expect("ok");
        let r2 = icm_intrinsic_reward(&pred, &actual, 2, cfg2).expect("ok");
        // r = eta * 0.5 * sq. eta=1 ⇒ 0.5; eta=2 ⇒ 1.0
        assert!((r1.intrinsic_rewards[0] - 0.5).abs() < 1e-6);
        assert!((r2.intrinsic_rewards[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn icm_reward_nonneg() {
        let pred = vec![-1.0_f32, 2.0, 0.5, -3.0];
        let actual = vec![1.0_f32, -2.0, 0.0, 3.0];
        let out = icm_intrinsic_reward(&pred, &actual, 2, IcmConfig::default()).expect("ok");
        for &r in &out.intrinsic_rewards {
            assert!(r >= 0.0, "intrinsic reward must be non-negative: {r}");
        }
    }

    #[test]
    fn icm_inverse_loss_perfect_prediction() {
        // p(true action) = 1 ⇒ loss ~ 0.
        let probs = vec![1.0_f32, 0.0, 0.0, 1.0]; // B=2, A=2; actions [0,1]
        let actions = vec![0_usize, 1];
        let l = icm_inverse_loss(&probs, &actions, 2).expect("ok");
        assert!(l < 1e-4, "perfect inverse loss ~0, got {l}");
    }

    #[test]
    fn icm_inverse_loss_uniform() {
        // p(true)=0.5 ⇒ loss = -ln(0.5) = 0.6931.
        let probs = vec![0.5_f32, 0.5];
        let actions = vec![0_usize];
        let l = icm_inverse_loss(&probs, &actions, 2).expect("ok");
        assert!((l - std::f32::consts::LN_2).abs() < 1e-3, "loss={l}");
    }

    #[test]
    fn rnd_construct_and_count() {
        let rnd = RndReward::new(8).expect("ok");
        assert_eq!(rnd.count(), 0);
    }

    #[test]
    fn rnd_zero_error_zero_reward() {
        let mut rnd = RndReward::new(4).expect("ok");
        let emb = vec![1.0_f32, 2.0, 3.0, 4.0];
        let r = rnd.intrinsic_reward(&emb, &emb).expect("ok");
        assert!(r[0].abs() < 1e-6, "zero error ⇒ zero reward, got {}", r[0]);
    }

    #[test]
    fn rnd_reward_nonneg_and_updates_count() {
        let mut rnd = RndReward::new(3).expect("ok");
        let pred = vec![0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // B=2
        let target = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r = rnd.intrinsic_reward(&pred, &target).expect("ok");
        assert_eq!(r.len(), 2);
        for &ri in &r {
            assert!(ri >= 0.0, "reward must be non-negative: {ri}");
        }
        assert_eq!(rnd.count(), 2, "should record 2 error samples");
    }

    #[test]
    fn rnd_larger_error_larger_reward() {
        // Within one batch all errors share the same normalising scale, so a
        // larger embedding distance ⇒ proportionally larger reward.
        let mut rnd = RndReward::new(2).expect("ok");
        let pred = vec![0.0_f32, 0.0, 0.0, 0.0]; // B=2
        let target = vec![1.0_f32, 0.0, 5.0, 0.0]; // errors: 1 and 25
        let r = rnd.intrinsic_reward(&pred, &target).expect("ok");
        assert!(r[1] > r[0], "bigger error should give bigger reward");
    }

    #[test]
    fn rnd_normalizes_over_time() {
        // After many samples the reward magnitudes are scale-normalised; verify
        // they stay finite and non-negative.
        let mut rnd = RndReward::new(2).expect("ok");
        for k in 0..50 {
            let pred = vec![0.0_f32, 0.0];
            let target = vec![(k as f32) * 0.1, 0.0];
            let r = rnd.intrinsic_reward(&pred, &target).expect("ok");
            assert!(r[0].is_finite() && r[0] >= 0.0);
        }
        assert_eq!(rnd.count(), 50);
    }

    #[test]
    fn err_icm_bad_eta() {
        let cfg = IcmConfig {
            eta: 0.0,
            beta: 0.2,
        };
        assert!(matches!(
            icm_intrinsic_reward(&[0.0], &[0.0], 1, cfg),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn err_icm_dim_mismatch() {
        assert!(matches!(
            icm_intrinsic_reward(&[0.0, 0.0], &[0.0], 2, IcmConfig::default()),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_icm_inverse_action_out_of_range() {
        let probs = vec![0.5_f32, 0.5];
        let actions = vec![5_usize]; // >= n_actions
        assert!(matches!(
            icm_inverse_loss(&probs, &actions, 2),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_rnd_zero_embed_dim() {
        assert!(matches!(
            RndReward::new(0),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn err_rnd_dim_mismatch() {
        let mut rnd = RndReward::new(4).expect("ok");
        assert!(matches!(
            rnd.intrinsic_reward(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0]),
            Err(RlError::DimensionMismatch { .. })
        ));
    }
}

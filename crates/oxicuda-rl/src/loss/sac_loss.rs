//! SAC (Soft Actor-Critic) structured loss object.
//!
//! Haarnoja et al. (2018). Off-policy maximum entropy RL with twin critics.
//! This module provides a struct-based API complementing the free-function API
//! in [`crate::loss::sac`].
//!
//! The structured API groups hyperparameters into [`SacOffPolicyConfig`] and
//! exposes the full SAC update suite — critic, actor, alpha, and soft target
//! parameter update — through [`SacOffPolicyLoss`].

use crate::error::{RlError, RlResult};

// ─── SacOffPolicyConfig ───────────────────────────────────────────────────────

/// Hyperparameters for off-policy SAC training.
#[derive(Debug, Clone, Copy)]
pub struct SacOffPolicyConfig {
    /// Discount factor γ ∈ (0, 1].
    pub gamma: f32,
    /// Entropy temperature α ≥ 0.
    pub alpha: f32,
    /// Soft target update coefficient τ ∈ (0, 1].
    pub tau: f32,
}

impl Default for SacOffPolicyConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            alpha: 0.2,
            tau: 0.005,
        }
    }
}

// ─── SacOffPolicyLoss ─────────────────────────────────────────────────────────

/// Struct-based SAC loss object providing critic, actor, and alpha losses.
pub struct SacOffPolicyLoss {
    config: SacOffPolicyConfig,
}

impl SacOffPolicyLoss {
    /// Create a new [`SacOffPolicyLoss`] with the given hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::InvalidHyperparameter`] if:
    /// - `gamma` is not in `(0, 1]`
    /// - `alpha` is negative
    /// - `tau` is not in `(0, 1]`
    pub fn new(config: SacOffPolicyConfig) -> RlResult<Self> {
        if config.gamma <= 0.0 || config.gamma > 1.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "gamma".into(),
                msg: "must be in (0, 1]".into(),
            });
        }
        if config.alpha < 0.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "alpha".into(),
                msg: "must be >= 0".into(),
            });
        }
        if config.tau <= 0.0 || config.tau > 1.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "tau".into(),
                msg: "must be in (0, 1]".into(),
            });
        }
        Ok(Self { config })
    }

    /// Compute twin-critic Q-function MSE losses.
    ///
    /// # Arguments
    ///
    /// * `q1`              — `[B]` Q1(s_t, a_t) predictions.
    /// * `q2`              — `[B]` Q2(s_t, a_t) predictions.
    /// * `next_q_target`   — `[B]` pre-computed `min(Q1_t, Q2_t) - α * log_π`
    ///   at `(s_{t+1}, ã_{t+1})`.
    /// * `rewards`         — `[B]` rewards `r_t`.
    /// * `dones`           — `[B]` done flags (1.0 = terminal).
    ///
    /// Returns `(loss_q1, loss_q2)` where each is the mean squared Bellman error
    /// for the respective critic.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if any slice lengths differ.
    pub fn critic_loss(
        &self,
        q1: &[f32],
        q2: &[f32],
        next_q_target: &[f32],
        rewards: &[f32],
        dones: &[f32],
    ) -> RlResult<(f32, f32)> {
        let b = q1.len();
        if q2.len() != b || next_q_target.len() != b || rewards.len() != b || dones.len() != b {
            return Err(RlError::DimensionMismatch {
                expected: b,
                got: if q2.len() != b {
                    q2.len()
                } else if next_q_target.len() != b {
                    next_q_target.len()
                } else if rewards.len() != b {
                    rewards.len()
                } else {
                    dones.len()
                },
            });
        }
        let mut sum_q1 = 0.0_f32;
        let mut sum_q2 = 0.0_f32;
        for i in 0..b {
            let mask = 1.0 - dones[i];
            let target = rewards[i] + self.config.gamma * mask * next_q_target[i];
            let d1 = q1[i] - target;
            let d2 = q2[i] - target;
            sum_q1 += d1 * d1;
            sum_q2 += d2 * d2;
        }
        let b_f = b as f32;
        Ok((sum_q1 / b_f, sum_q2 / b_f))
    }

    /// Compute the policy (actor) loss.
    ///
    /// ```text
    /// L_π = mean(α * log_π(ã|s) - Q(s, ã))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `q_values`  — `[B]` min(Q1, Q2)(s_t, ã_t) for re-sampled actions.
    /// * `log_probs` — `[B]` log π(ã_t|s_t).
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if slice lengths differ or are empty.
    pub fn actor_loss(&self, q_values: &[f32], log_probs: &[f32]) -> RlResult<f32> {
        let b = q_values.len();
        if log_probs.len() != b {
            return Err(RlError::DimensionMismatch {
                expected: b,
                got: log_probs.len(),
            });
        }
        if b == 0 {
            return Err(RlError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let loss: f32 = q_values
            .iter()
            .zip(log_probs.iter())
            .map(|(&q, &lp)| self.config.alpha * lp - q)
            .sum::<f32>()
            / b as f32;
        Ok(loss)
    }

    /// Compute the entropy temperature (α) loss.
    ///
    /// ```text
    /// L_α = mean(-α * (log_π + H̄))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `log_probs`       — `[B]` log π(ã_t|s_t).
    /// * `target_entropy`  — target entropy `H̄` (typically `-action_dim`).
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if `log_probs` is empty.
    pub fn alpha_loss(&self, log_probs: &[f32], target_entropy: f32) -> RlResult<f32> {
        if log_probs.is_empty() {
            return Err(RlError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let mean_lp = log_probs.iter().sum::<f32>() / log_probs.len() as f32;
        Ok(-self.config.alpha * (mean_lp + target_entropy))
    }

    /// Polyak soft-update: `target = (1 - τ) * target + τ * online`.
    ///
    /// # Arguments
    ///
    /// * `target` — mutable target network parameters.
    /// * `online` — online network parameters.
    /// * `tau`    — mixing coefficient τ ∈ (0, 1].
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if slice lengths differ.
    /// Returns [`RlError::InvalidHyperparameter`] if `tau` is not in `(0, 1]`.
    pub fn soft_update_params(target: &mut [f32], online: &[f32], tau: f32) -> RlResult<()> {
        if tau <= 0.0 || tau > 1.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "tau".into(),
                msg: "must be in (0, 1]".into(),
            });
        }
        if target.len() != online.len() {
            return Err(RlError::DimensionMismatch {
                expected: target.len(),
                got: online.len(),
            });
        }
        for (t, &o) in target.iter_mut().zip(online.iter()) {
            *t = (1.0 - tau) * *t + tau * o;
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_loss() -> SacOffPolicyLoss {
        SacOffPolicyLoss::new(SacOffPolicyConfig::default())
            .expect("default SacOffPolicyConfig should be valid")
    }

    #[test]
    fn critic_loss_finite() {
        let loss = default_loss();
        let b = 8;
        let q1 = vec![1.0_f32; b];
        let q2 = vec![1.1_f32; b];
        let nq = vec![0.9_f32; b];
        let r = vec![0.5_f32; b];
        let d = vec![0.0_f32; b];
        let (l1, l2) = loss
            .critic_loss(&q1, &q2, &nq, &r, &d)
            .expect("critic_loss should succeed on equal-length slices");
        assert!(l1.is_finite(), "q1_loss={l1}");
        assert!(l2.is_finite(), "q2_loss={l2}");
    }

    #[test]
    fn actor_loss_finite() {
        let loss = default_loss();
        let b = 8;
        let q = vec![1.0_f32; b];
        let lp = vec![-0.5_f32; b];
        let l = loss
            .actor_loss(&q, &lp)
            .expect("actor_loss should succeed on equal-length slices");
        assert!(l.is_finite(), "actor_loss={l}");
    }

    #[test]
    fn alpha_loss_finite() {
        let loss = default_loss();
        let lp = vec![-0.5_f32; 8];
        let l = loss
            .alpha_loss(&lp, -1.0)
            .expect("alpha_loss should succeed on non-empty slice");
        assert!(l.is_finite(), "alpha_loss={l}");
    }

    #[test]
    fn critic_done_ignores_future() {
        // When done=1, target = reward (future Q is masked out)
        let loss = SacOffPolicyLoss::new(SacOffPolicyConfig {
            gamma: 1.0,
            alpha: 0.2,
            tau: 0.005,
        })
        .expect("config valid");
        let q1 = vec![2.0_f32; 1];
        let q2 = vec![2.0_f32; 1];
        let nq = vec![999.0_f32; 1]; // large future Q should be masked
        let r = vec![2.0_f32; 1];
        let d = vec![1.0_f32; 1]; // done
        let (l1, l2) = loss
            .critic_loss(&q1, &q2, &nq, &r, &d)
            .expect("critic_loss should succeed");
        // target = 2.0, Q1=Q2=2.0 → loss = 0
        assert!(l1.abs() < 1e-5, "q1_loss={l1} should be 0 with done=1");
        assert!(l2.abs() < 1e-5, "q2_loss={l2} should be 0 with done=1");
    }

    #[test]
    fn critic_loss_nonneg() {
        let loss = default_loss();
        let b = 16;
        let q1: Vec<f32> = (0..b).map(|i| i as f32 * 0.1).collect();
        let q2: Vec<f32> = (0..b).map(|i| i as f32 * 0.2).collect();
        let nq = vec![0.5_f32; b];
        let r = vec![1.0_f32; b];
        let d = vec![0.0_f32; b];
        let (l1, l2) = loss
            .critic_loss(&q1, &q2, &nq, &r, &d)
            .expect("critic_loss should succeed");
        assert!(l1 >= 0.0, "MSE q1 loss must be non-negative: {l1}");
        assert!(l2 >= 0.0, "MSE q2 loss must be non-negative: {l2}");
    }

    #[test]
    fn actor_entropy_bonus() {
        // With alpha>0 and high log_pi, actor_loss = alpha*log_pi - q increases
        let config = SacOffPolicyConfig {
            alpha: 1.0,
            ..SacOffPolicyConfig::default()
        };
        let loss = SacOffPolicyLoss::new(config).expect("config valid");
        let q = vec![1.0_f32; 4];
        let lp_low = vec![-5.0_f32; 4]; // low entropy
        let lp_high = vec![0.0_f32; 4]; // high entropy (log_prob near 0)
        let l_low = loss
            .actor_loss(&q, &lp_low)
            .expect("actor_loss low entropy");
        let l_high = loss
            .actor_loss(&q, &lp_high)
            .expect("actor_loss high entropy");
        assert!(
            l_high > l_low,
            "higher log_pi should increase actor_loss: l_low={l_low} l_high={l_high}"
        );
    }

    #[test]
    fn soft_update_tau_bounds() {
        let tau = 0.005_f32;
        let mut target = vec![1.0_f32; 4];
        let online = vec![0.0_f32; 4];
        SacOffPolicyLoss::soft_update_params(&mut target, &online, tau)
            .expect("soft_update_params should succeed with tau=0.005");
        // target should be close to 1.0 (barely updated)
        for &v in &target {
            assert!(
                v > 0.99,
                "target={v} should remain close to 1.0 with tau=0.005"
            );
        }
    }

    #[test]
    fn batch_mismatch_error() {
        let loss = default_loss();
        let q1 = vec![1.0_f32; 4];
        let q2 = vec![1.0_f32; 5]; // wrong length
        let nq = vec![0.5_f32; 4];
        let r = vec![1.0_f32; 4];
        let d = vec![0.0_f32; 4];
        let result = loss.critic_loss(&q1, &q2, &nq, &r, &d);
        assert!(result.is_err(), "mismatched lengths should return Err");
    }

    #[test]
    fn target_entropy_negative_is_ok() {
        let loss = default_loss();
        let lp = vec![-1.0_f32; 8];
        let result = loss.alpha_loss(&lp, -1.0);
        assert!(result.is_ok(), "negative target_entropy should be allowed");
        assert!(result.expect("result should be present").is_finite());
    }

    #[test]
    fn alpha_0_no_entropy() {
        let config = SacOffPolicyConfig {
            alpha: 0.0,
            ..SacOffPolicyConfig::default()
        };
        let loss = SacOffPolicyLoss::new(config).expect("alpha=0 is valid");
        let lp = vec![-2.0_f32; 8];
        let l = loss
            .alpha_loss(&lp, -1.0)
            .expect("alpha_loss should succeed");
        // loss = -0.0 * (...) = 0.0
        assert!(
            l.abs() < 1e-6,
            "alpha=0 should give zero alpha_loss, got {l}"
        );
    }

    #[test]
    fn soft_update_invalid_tau_error() {
        let mut target = vec![1.0_f32; 4];
        let online = vec![0.5_f32; 4];
        assert!(
            SacOffPolicyLoss::soft_update_params(&mut target, &online, 0.0).is_err(),
            "tau=0 should return Err"
        );
        assert!(
            SacOffPolicyLoss::soft_update_params(&mut target, &online, 1.5).is_err(),
            "tau=1.5 should return Err"
        );
    }
}

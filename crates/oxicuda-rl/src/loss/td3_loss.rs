//! TD3 (Twin Delayed DDPG) structured loss object.
//!
//! Fujimoto et al. (2018). Target policy smoothing + twin-Q critics.
//! This module provides a struct-based API complementing the free-function API
//! in [`crate::loss::td3`].
//!
//! Key TD3 innovations captured here:
//! - Twin critics with `min(Q1, Q2)` target to reduce overestimation bias.
//! - Delayed policy updates via `policy_delay`.
//! - Target policy smoothing: Gaussian noise clipped to `[-noise_clip, noise_clip]`
//!   added to target actions before evaluating target Q values.

use crate::error::{RlError, RlResult};
use crate::handle::LcgRng;

/// Convenience alias for the local RNG type.
pub type RlRng = LcgRng;

// ─── Box-Muller helper ───────────────────────────────────────────────────────

/// Sample one standard normal variate via the Box-Muller transform.
fn sample_normal(rng: &mut RlRng) -> f32 {
    let u1 = (rng.next_f32() + 1e-10_f32).min(1.0 - 1e-10_f32);
    let u2 = rng.next_f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

// ─── Td3PolicyConfig ─────────────────────────────────────────────────────────

/// Hyperparameters for TD3 training.
#[derive(Debug, Clone, Copy)]
pub struct Td3PolicyConfig {
    /// Discount factor γ ∈ (0, 1].
    pub gamma: f32,
    /// Soft target update coefficient τ ∈ (0, 1].
    pub tau: f32,
    /// Standard deviation of target policy smoothing noise.
    pub policy_noise: f32,
    /// Clip bound for target policy noise: noise is clamped to `[-noise_clip, noise_clip]`.
    pub noise_clip: f32,
    /// Frequency of policy (actor) updates: actor updates every `policy_delay`
    /// critic updates.
    pub policy_delay: usize,
}

impl Default for Td3PolicyConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            tau: 0.005,
            policy_noise: 0.2,
            noise_clip: 0.5,
            policy_delay: 2,
        }
    }
}

// ─── Td3PolicyLoss ────────────────────────────────────────────────────────────

/// Struct-based TD3 loss object providing critic, actor, and noise generation.
pub struct Td3PolicyLoss {
    config: Td3PolicyConfig,
}

impl Td3PolicyLoss {
    /// Create a new [`Td3PolicyLoss`] with the given hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::InvalidHyperparameter`] if:
    /// - `gamma` is not in `(0, 1]`
    /// - `tau` is not in `(0, 1]`
    /// - `policy_noise` is negative
    /// - `noise_clip` is negative
    /// - `policy_delay` is zero
    pub fn new(config: Td3PolicyConfig) -> RlResult<Self> {
        if config.gamma <= 0.0 || config.gamma > 1.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "gamma".into(),
                msg: "must be in (0, 1]".into(),
            });
        }
        if config.tau <= 0.0 || config.tau > 1.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "tau".into(),
                msg: "must be in (0, 1]".into(),
            });
        }
        if config.policy_noise < 0.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "policy_noise".into(),
                msg: "must be >= 0".into(),
            });
        }
        if config.noise_clip < 0.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "noise_clip".into(),
                msg: "must be >= 0".into(),
            });
        }
        if config.policy_delay == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "policy_delay".into(),
                msg: "must be >= 1".into(),
            });
        }
        Ok(Self { config })
    }

    /// Compute twin-critic Q-function MSE losses.
    ///
    /// Target Bellman backup:
    /// ```text
    /// y_t = r_t + γ (1-done_t) min(Q1'(s_{t+1}, ã), Q2'(s_{t+1}, ã))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `q1`, `q2` — `[B]` online Q1/Q2 at `(s_t, a_t)`.
    /// * `next_q1`, `next_q2` — `[B]` target-network Q values at `(s_{t+1}, ã_{t+1})`.
    /// * `rewards` — `[B]` rewards `r_t`.
    /// * `dones` — `[B]` done flags.
    /// * `_noisy_next_action` — `[B*action_dim]` smoothed target actions
    ///   (reserved; callers pre-compute and forward to the target Q-networks).
    ///
    /// Returns `(loss_q1, loss_q2)`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if any primary slice lengths differ.
    #[allow(clippy::too_many_arguments)]
    pub fn critic_loss(
        &self,
        q1: &[f32],
        q2: &[f32],
        next_q1: &[f32],
        next_q2: &[f32],
        rewards: &[f32],
        dones: &[f32],
        _noisy_next_action: &[f32],
    ) -> RlResult<(f32, f32)> {
        let b = q1.len();
        if q2.len() != b
            || next_q1.len() != b
            || next_q2.len() != b
            || rewards.len() != b
            || dones.len() != b
        {
            return Err(RlError::DimensionMismatch {
                expected: b,
                got: if q2.len() != b {
                    q2.len()
                } else if next_q1.len() != b {
                    next_q1.len()
                } else if next_q2.len() != b {
                    next_q2.len()
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
            let min_q_next = next_q1[i].min(next_q2[i]);
            let target = rewards[i] + self.config.gamma * mask * min_q_next;
            let d1 = q1[i] - target;
            let d2 = q2[i] - target;
            sum_q1 += d1 * d1;
            sum_q2 += d2 * d2;
        }
        let b_f = b as f32;
        Ok((sum_q1 / b_f, sum_q2 / b_f))
    }

    /// Compute the actor (policy) loss.
    ///
    /// ```text
    /// L_π = -mean(Q1(s_t, μ_θ(s_t)))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `q1_for_actor` — `[B]` Q1(s_t, μ_θ(s_t)) evaluated at the current
    ///   policy's actions (without target-policy noise).
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if `q1_for_actor` is empty.
    pub fn actor_loss(&self, q1_for_actor: &[f32]) -> RlResult<f32> {
        if q1_for_actor.is_empty() {
            return Err(RlError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let mean_q = q1_for_actor.iter().sum::<f32>() / q1_for_actor.len() as f32;
        Ok(-mean_q)
    }

    /// Generate target policy smoothing noise.
    ///
    /// Draws `action_dim * batch_size` samples from `N(0, policy_noise²)` and
    /// clamps each to `[-noise_clip, noise_clip]`.
    ///
    /// Returns a flat `Vec<f32>` of shape `[batch_size × action_dim]`.
    pub fn target_noise(&self, action_dim: usize, batch_size: usize, rng: &mut RlRng) -> Vec<f32> {
        let n = action_dim * batch_size;
        (0..n)
            .map(|_| {
                let z = sample_normal(rng) * self.config.policy_noise;
                z.clamp(-self.config.noise_clip, self.config.noise_clip)
            })
            .collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_loss() -> Td3PolicyLoss {
        Td3PolicyLoss::new(Td3PolicyConfig::default())
            .expect("default Td3PolicyConfig should be valid")
    }

    fn make_rng(seed: u64) -> RlRng {
        RlRng::new(seed)
    }

    #[test]
    fn critic_loss_finite() {
        let loss = default_loss();
        let b = 8;
        let q1 = vec![1.0_f32; b];
        let q2 = vec![1.1_f32; b];
        let nq1 = vec![0.9_f32; b];
        let nq2 = vec![0.8_f32; b];
        let r = vec![0.5_f32; b];
        let d = vec![0.0_f32; b];
        let na = vec![0.0_f32; b];
        let (l1, l2) = loss
            .critic_loss(&q1, &q2, &nq1, &nq2, &r, &d, &na)
            .expect("critic_loss should succeed on equal-length slices");
        assert!(l1.is_finite(), "q1_loss={l1}");
        assert!(l2.is_finite(), "q2_loss={l2}");
    }

    #[test]
    fn actor_loss_finite() {
        let loss = default_loss();
        let q = vec![1.5_f32; 8];
        let l = loss
            .actor_loss(&q)
            .expect("actor_loss should succeed on non-empty slice");
        assert!(l.is_finite(), "actor_loss={l}");
    }

    #[test]
    fn target_noise_clipped() {
        let loss = default_loss();
        let mut rng = make_rng(42);
        let noise = loss.target_noise(4, 16, &mut rng);
        for &v in &noise {
            assert!(
                (-0.5..=0.5).contains(&v),
                "noise={v} outside [-noise_clip=0.5, noise_clip=0.5]"
            );
        }
    }

    #[test]
    fn target_noise_shape() {
        let loss = default_loss();
        let mut rng = make_rng(1);
        let action_dim = 3;
        let batch_size = 32;
        let noise = loss.target_noise(action_dim, batch_size, &mut rng);
        assert_eq!(noise.len(), action_dim * batch_size, "noise shape mismatch");
    }

    #[test]
    fn done_flag_masks_future() {
        let loss = Td3PolicyLoss::new(Td3PolicyConfig {
            gamma: 1.0,
            ..Td3PolicyConfig::default()
        })
        .expect("config valid");
        let q1 = vec![2.0_f32; 1];
        let q2 = vec![2.0_f32; 1];
        let nq1 = vec![999.0_f32; 1]; // would dominate if not masked
        let nq2 = vec![999.0_f32; 1];
        let r = vec![2.0_f32; 1];
        let d = vec![1.0_f32; 1]; // done
        let na = vec![0.0_f32; 1];
        let (l1, l2) = loss
            .critic_loss(&q1, &q2, &nq1, &nq2, &r, &d, &na)
            .expect("critic_loss should succeed");
        assert!(l1.abs() < 1e-5, "q1_loss={l1} should be 0 with done=1");
        assert!(l2.abs() < 1e-5, "q2_loss={l2} should be 0 with done=1");
    }

    #[test]
    fn min_q_critic_target() {
        // next_q1=10, next_q2=1 → target uses min=1, not max
        let loss = Td3PolicyLoss::new(Td3PolicyConfig {
            gamma: 1.0,
            ..Td3PolicyConfig::default()
        })
        .expect("config valid");
        let q1 = vec![0.0_f32; 1]; // set to 0 to measure target directly
        let q2 = vec![0.0_f32; 1];
        let nq1 = vec![10.0_f32; 1];
        let nq2 = vec![1.0_f32; 1];
        let r = vec![0.0_f32; 1];
        let d = vec![0.0_f32; 1];
        let na = vec![0.0_f32; 1];
        let (l1, _) = loss
            .critic_loss(&q1, &q2, &nq1, &nq2, &r, &d, &na)
            .expect("critic_loss should succeed");
        // target = 0 + 1.0 * 1.0 * min(10,1) = 1.0; Q1=0 → (0-1)^2 = 1.0
        let expected = 1.0_f32; // MSE of (0 - 1)^2
        assert!(
            (l1 - expected).abs() < 1e-5,
            "expected q1_loss={expected}, got {l1}"
        );
    }

    #[test]
    fn tau_soft_update() {
        let tau = 0.01_f32;
        let mut target = vec![1.0_f32; 4];
        let online = [0.0_f32; 4];
        // Manual soft update
        for (t, &o) in target.iter_mut().zip(online.iter()) {
            *t = (1.0 - tau) * *t + tau * o;
        }
        for &v in &target {
            assert!(
                (v - (1.0 - tau)).abs() < 1e-6,
                "expected {}, got {v}",
                1.0 - tau
            );
        }
    }

    #[test]
    fn noise_clip_respected() {
        let config = Td3PolicyConfig {
            policy_noise: 100.0, // very large noise
            noise_clip: 0.5,
            ..Td3PolicyConfig::default()
        };
        let loss = Td3PolicyLoss::new(config).expect("config valid");
        let mut rng = make_rng(99);
        let noise = loss.target_noise(2, 100, &mut rng);
        for &v in &noise {
            assert!(
                (-0.5..=0.5).contains(&v),
                "noise={v} outside [-0.5, 0.5] despite large policy_noise"
            );
        }
    }

    #[test]
    fn policy_noise_0_deterministic() {
        let config = Td3PolicyConfig {
            policy_noise: 0.0,
            noise_clip: 0.5,
            ..Td3PolicyConfig::default()
        };
        let loss = Td3PolicyLoss::new(config).expect("config valid");
        let mut rng = make_rng(7);
        let noise = loss.target_noise(3, 10, &mut rng);
        for &v in &noise {
            assert!(
                v.abs() < 1e-7,
                "policy_noise=0 should give all-zero noise, got {v}"
            );
        }
    }
}

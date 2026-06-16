//! # Discrete Soft Actor-Critic (SAC) Loss Functions
//!
//! Christodoulou (2019), "Soft Actor-Critic for Discrete Action Settings".
//!
//! Extends the continuous-action SAC framework to finite discrete action
//! spaces.  Because the policy is a categorical distribution over actions,
//! expectations over actions can be computed exactly:
//!
//! ```text
//! V(s) = Σ_a π(a|s) * (Q(s, a) − α * log π(a|s))
//! ```
//!
//! ## Critic Loss
//!
//! ```text
//! y_i = r_i + γ * (1 − done_i) * V(s'_i)
//! L_Q = E[ (Q(s, a) − y)² ]
//! ```
//!
//! ## Policy Loss
//!
//! ```text
//! L_π = E_s [ Σ_a π(a|s) * (α * log π(a|s) − Q(s, a)) ]
//! ```
//!
//! The policy is improved by minimising this loss (which is the negative of the
//! expected entropy-regularised Q-value).

use crate::error::{RlError, RlResult};

// ─── DiscreteSacConfig ───────────────────────────────────────────────────────

/// Hyperparameters for [`DiscreteSacLoss`].
#[derive(Debug, Clone, Copy)]
pub struct DiscreteSacConfig {
    /// Number of discrete actions.
    pub n_actions: usize,
    /// Entropy temperature coefficient α.
    pub entropy_coeff: f32,
    /// Discount factor γ ∈ `[0, 1]`.
    pub gamma: f32,
    /// Soft update coefficient τ ∈ `[0, 1]`.
    pub tau: f32,
}

// ─── DiscreteSacLoss ─────────────────────────────────────────────────────────

/// Discrete SAC loss computation bundle.
///
/// Construct via [`DiscreteSacLoss::new`] after validating the configuration.
#[derive(Debug, Clone)]
pub struct DiscreteSacLoss {
    /// Validated configuration.
    config: DiscreteSacConfig,
}

impl DiscreteSacLoss {
    /// Create a new [`DiscreteSacLoss`] instance after validating `config`.
    ///
    /// # Errors
    ///
    /// * [`RlError::InvalidHyperparameter`] when:
    ///   - `n_actions == 0`
    ///   - `entropy_coeff < 0`
    ///   - `gamma` is outside `[0, 1]`
    ///   - `tau` is outside `[0, 1]`
    pub fn new(config: DiscreteSacConfig) -> RlResult<Self> {
        if config.n_actions == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "n_actions".into(),
                msg: "must be > 0".into(),
            });
        }
        if config.entropy_coeff < 0.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "entropy_coeff".into(),
                msg: "must be >= 0".into(),
            });
        }
        if !(0.0..=1.0).contains(&config.gamma) {
            return Err(RlError::InvalidHyperparameter {
                name: "gamma".into(),
                msg: "must be in [0, 1]".into(),
            });
        }
        if !(0.0..=1.0).contains(&config.tau) {
            return Err(RlError::InvalidHyperparameter {
                name: "tau".into(),
                msg: "must be in [0, 1]".into(),
            });
        }
        Ok(Self { config })
    }

    /// Return a reference to the configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &DiscreteSacConfig {
        &self.config
    }

    // ── Critic loss ──────────────────────────────────────────────────────────

    /// Compute the discrete SAC critic (Q-network) MSE loss.
    ///
    /// # Arguments
    ///
    /// * `q_values`     — `[B × A]` online Q-values for all actions, flat
    ///   row-major.
    /// * `target_q`     — `[B × A]` target-network Q-values, flat row-major.
    /// * `action_probs` — `[B × A]` action probabilities π(a|s), flat
    ///   row-major.
    /// * `rewards`      — `[B]` rewards.
    /// * `actions`      — `[B]` taken action indices; each must be
    ///   `< n_actions`.
    /// * `dones`        — `[B]` episode-termination flags (1.0 = terminal).
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] when any slice length is inconsistent.
    /// * [`RlError::InvalidHyperparameter`] when an action index is out of
    ///   range.
    pub fn q_loss(
        &self,
        q_values: &[f32],
        target_q: &[f32],
        action_probs: &[f32],
        rewards: &[f32],
        actions: &[usize],
        dones: &[f32],
    ) -> RlResult<f32> {
        let n = self.config.n_actions;
        let batch_size = rewards.len();

        // ── Length validation ────────────────────────────────────────────────
        if dones.len() != batch_size {
            return Err(RlError::DimensionMismatch {
                expected: batch_size,
                got: dones.len(),
            });
        }
        if actions.len() != batch_size {
            return Err(RlError::DimensionMismatch {
                expected: batch_size,
                got: actions.len(),
            });
        }
        if q_values.len() != batch_size * n {
            return Err(RlError::DimensionMismatch {
                expected: batch_size * n,
                got: q_values.len(),
            });
        }
        if target_q.len() != batch_size * n {
            return Err(RlError::DimensionMismatch {
                expected: batch_size * n,
                got: target_q.len(),
            });
        }
        if action_probs.len() != batch_size * n {
            return Err(RlError::DimensionMismatch {
                expected: batch_size * n,
                got: action_probs.len(),
            });
        }

        // ── Action index range check ─────────────────────────────────────────
        for (i, &act) in actions.iter().enumerate() {
            if act >= n {
                return Err(RlError::InvalidHyperparameter {
                    name: "actions".into(),
                    msg: format!("actions[{i}]={act} is out of range (n_actions={n})"),
                });
            }
        }

        // ── Per-sample Bellman error ─────────────────────────────────────────
        let mut total_loss = 0.0_f32;
        for i in 0..batch_size {
            let probs_i = &action_probs[i * n..(i + 1) * n];
            let tgt_q_i = &target_q[i * n..(i + 1) * n];

            // Soft value: V(s') = Σ_a π(a|s') * (Q_target(s',a) - α log π(a|s'))
            let v_next: f32 = probs_i
                .iter()
                .zip(tgt_q_i.iter())
                .map(|(&p, &q)| p * (q - self.config.entropy_coeff * (p + 1e-8_f32).ln()))
                .sum();

            let target_i = rewards[i] + self.config.gamma * (1.0 - dones[i]) * v_next;
            let q_sa = q_values[i * n + actions[i]];
            let td_error = q_sa - target_i;
            total_loss += td_error * td_error;
        }

        Ok(total_loss / batch_size as f32)
    }

    // ── Policy loss ──────────────────────────────────────────────────────────

    /// Compute the discrete SAC policy loss.
    ///
    /// ```text
    /// L_π = E_s [ Σ_a π(a|s) * (α * log π(a|s) − Q(s, a)) ]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `q_values`     — `[B × A]` online Q-values for all actions (flat).
    /// * `action_probs` — `[B × A]` action probabilities π(a|s) (flat).
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] when slice lengths are inconsistent.
    pub fn policy_loss(&self, q_values: &[f32], action_probs: &[f32]) -> RlResult<f32> {
        let n = self.config.n_actions;
        if q_values.len() % n != 0 {
            return Err(RlError::DimensionMismatch {
                expected: q_values.len() / n * n,
                got: q_values.len(),
            });
        }
        let batch_size = q_values.len() / n;
        if batch_size == 0 {
            return Err(RlError::DimensionMismatch {
                expected: n,
                got: 0,
            });
        }
        if action_probs.len() != batch_size * n {
            return Err(RlError::DimensionMismatch {
                expected: batch_size * n,
                got: action_probs.len(),
            });
        }

        let mut total = 0.0_f32;
        for i in 0..batch_size {
            let probs_i = &action_probs[i * n..(i + 1) * n];
            let q_i = &q_values[i * n..(i + 1) * n];
            let pg_i: f32 = probs_i
                .iter()
                .zip(q_i.iter())
                .map(|(&p, &q)| p * (self.config.entropy_coeff * (p + 1e-8_f32).ln() - q))
                .sum();
            total += pg_i;
        }

        Ok(total / batch_size as f32)
    }

    // ── Entropy ──────────────────────────────────────────────────────────────

    /// Compute the Shannon entropy of a discrete probability distribution.
    ///
    /// ```text
    /// H(π) = -Σ_a π(a) * log(π(a) + ε)
    /// ```
    ///
    /// Returns `0.0` for an empty distribution.
    #[must_use]
    pub fn entropy(probs: &[f32]) -> f32 {
        if probs.is_empty() {
            return 0.0;
        }
        -probs.iter().map(|&p| p * (p + 1e-8_f32).ln()).sum::<f32>()
    }

    // ── Soft update ──────────────────────────────────────────────────────────

    /// Perform a Polyak (soft) update of `target` towards `online`:
    ///
    /// ```text
    /// target[i] ← τ * online[i] + (1 − τ) * target[i]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `target` — mutable target-network parameters.
    /// * `online` — online-network parameters.
    /// * `tau`    — interpolation weight ∈ `[0, 1]`.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] when `target.len() != online.len()`.
    pub fn soft_update(target: &mut [f32], online: &[f32], tau: f32) -> RlResult<()> {
        if target.len() != online.len() {
            return Err(RlError::DimensionMismatch {
                expected: target.len(),
                got: online.len(),
            });
        }
        let one_minus_tau = 1.0 - tau;
        for (t, &o) in target.iter_mut().zip(online.iter()) {
            *t = tau * o + one_minus_tau * *t;
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_loss(n_actions: usize) -> DiscreteSacLoss {
        DiscreteSacLoss::new(DiscreteSacConfig {
            n_actions,
            entropy_coeff: 0.1,
            gamma: 0.99,
            tau: 0.005,
        })
        .expect("valid config should construct DiscreteSacLoss")
    }

    /// Uniform action probabilities helper.
    fn uniform_probs(batch: usize, n: usize) -> Vec<f32> {
        vec![1.0 / n as f32; batch * n]
    }

    #[test]
    fn q_loss_nonneg() {
        let loss = make_loss(3);
        let batch = 4;
        let n = 3;
        let q = vec![0.5_f32; batch * n];
        let tgt = vec![1.0_f32; batch * n];
        let probs = uniform_probs(batch, n);
        let rewards = vec![0.0_f32; batch];
        let actions = vec![0_usize; batch];
        let dones = vec![0.0_f32; batch];
        let l = loss
            .q_loss(&q, &tgt, &probs, &rewards, &actions, &dones)
            .expect("valid inputs should compute q_loss");
        assert!(l >= 0.0, "q_loss must be >= 0, got {l}");
    }

    #[test]
    fn policy_loss_finite() {
        let loss = make_loss(4);
        let batch = 8;
        let n = 4;
        let q = vec![1.0_f32; batch * n];
        let probs = uniform_probs(batch, n);
        let l = loss
            .policy_loss(&q, &probs)
            .expect("valid inputs should compute policy_loss");
        assert!(l.is_finite(), "policy_loss must be finite, got {l}");
    }

    #[test]
    fn entropy_uniform_is_log_n() {
        let n = 4;
        let probs = vec![1.0_f32 / n as f32; n];
        let h = DiscreteSacLoss::entropy(&probs);
        let expected = (n as f32).ln();
        // H(uniform) ≈ log(n) (the 1e-8 correction is negligible for p=0.25)
        assert!(
            (h - expected).abs() < 0.01,
            "entropy of uniform({n})={h}, expected ≈ {expected}"
        );
    }

    #[test]
    fn entropy_deterministic_is_zero() {
        let n = 5;
        let mut probs = vec![0.0_f32; n];
        probs[2] = 1.0;
        let h = DiscreteSacLoss::entropy(&probs);
        // H = -(1.0 * log(1.0+1e-8) + 0*log(0+1e-8)*4)
        // 0*log(0+1e-8) ≈ 0, and log(1.0+1e-8) ≈ 1e-8, so H ≈ -1e-8 ≈ 0
        assert!(
            h.abs() < 0.001,
            "entropy of deterministic dist should be ≈ 0, got {h}"
        );
    }

    #[test]
    fn soft_update_tau_0_no_change() {
        let mut target = vec![1.0_f32, 2.0, 3.0];
        let online = vec![10.0_f32, 20.0, 30.0];
        let original = target.clone();
        DiscreteSacLoss::soft_update(&mut target, &online, 0.0)
            .expect("equal-length soft_update should succeed");
        for (i, (&t, &o)) in target.iter().zip(original.iter()).enumerate() {
            assert!(
                (t - o).abs() < 1e-6,
                "tau=0: target[{i}]={t} should equal original={o}"
            );
        }
    }

    #[test]
    fn soft_update_tau_1_copies() {
        let mut target = vec![1.0_f32, 2.0, 3.0];
        let online = vec![10.0_f32, 20.0, 30.0];
        DiscreteSacLoss::soft_update(&mut target, &online, 1.0)
            .expect("equal-length soft_update should succeed");
        for (i, (&t, &o)) in target.iter().zip(online.iter()).enumerate() {
            assert!(
                (t - o).abs() < 1e-6,
                "tau=1: target[{i}]={t} should equal online={o}"
            );
        }
    }

    #[test]
    fn q_loss_terminal_ignores_future() {
        // With done=1: target = reward (no future term)
        let loss = make_loss(2);
        let n = 2;
        // Set Q(s,a=0) = 5.0; reward = 5.0; done=1 → target = 5.0 → td_error = 0
        let q = vec![5.0_f32, 0.0]; // batch=1, n=2
        let tgt = vec![100.0_f32, 100.0]; // would give huge V_next, but masked by done
        let probs = uniform_probs(1, n);
        let rewards = vec![5.0_f32];
        let actions = vec![0_usize];
        let dones = vec![1.0_f32]; // terminal
        let l = loss
            .q_loss(&q, &tgt, &probs, &rewards, &actions, &dones)
            .expect("terminal q_loss should compute");
        // target = 5.0 + 0.99 * (1-1) * V_next = 5.0; td_error = 5.0 - 5.0 = 0
        assert!(
            l.abs() < 1e-4,
            "done=1 should make loss ≈ 0 when Q(s,a)=reward, got {l}"
        );
    }

    #[test]
    fn action_probs_n_actions_check() {
        let loss = make_loss(3);
        let batch = 4;
        let n = 3;
        let q = vec![0.5_f32; batch * n];
        let tgt = vec![1.0_f32; batch * n];
        // Wrong: action_probs has batch * (n+1) instead of batch * n
        let wrong_probs = vec![0.25_f32; batch * (n + 1)];
        let rewards = vec![0.0_f32; batch];
        let actions = vec![0_usize; batch];
        let dones = vec![0.0_f32; batch];
        assert!(
            loss.q_loss(&q, &tgt, &wrong_probs, &rewards, &actions, &dones)
                .is_err(),
            "wrong action_probs length should return error"
        );
    }

    #[test]
    fn batch_size_mismatch_error() {
        // q_values.len() is not divisible by n_actions
        let loss = make_loss(3);
        let n = 3;
        // q_values has 10 elements, not divisible by 3 → DimensionMismatch
        let q = vec![0.5_f32; 10]; // 10 % 3 != 0
        let probs = vec![1.0_f32 / n as f32; 10];
        assert!(
            loss.policy_loss(&q, &probs).is_err(),
            "q_values not divisible by n_actions should error in policy_loss"
        );
    }

    #[test]
    fn entropy_nonneg() {
        // Test various distributions — entropy should always be >= 0
        let cases: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![0.5, 0.5],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.9, 0.05, 0.05],
            vec![0.0, 0.0, 1.0],
        ];
        for (idx, probs) in cases.iter().enumerate() {
            let h = DiscreteSacLoss::entropy(probs);
            assert!(h >= -1e-6, "entropy of case {idx} should be >= 0, got {h}");
        }
    }
}

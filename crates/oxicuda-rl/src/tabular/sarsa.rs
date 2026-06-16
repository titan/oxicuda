//! # Tabular SARSA (State-Action-Reward-State-Action)
//!
//! Rummery & Niranjan (1994), "On-line Q-learning using connectionist
//! systems"; Sutton & Barto (2018), "Reinforcement Learning: An
//! Introduction", §6.4 (SARSA) and §6.6 (Expected SARSA).
//!
//! ## Algorithm
//!
//! SARSA is the canonical **on-policy** temporal-difference control method.
//! Unlike Q-learning, the bootstrap uses the action `a'` actually selected by
//! the behaviour policy in the next state:
//!
//! ```text
//! target = r + γ · (1 − done) · Q[s', a']
//! Q[s, a] ← Q[s, a] + α · (target − Q[s, a])
//! ```
//!
//! **Expected SARSA** replaces the sampled `Q[s', a']` with the expectation
//! under the policy `π(·|s')`, reducing variance:
//!
//! ```text
//! target = r + γ · (1 − done) · Σ_{a'} π(a'|s') · Q[s', a']
//! ```

use crate::error::{RlError, RlResult};

/// Tabular SARSA agent backed by a dense `n_states × n_actions` table.
///
/// The table is stored row-major: `Q[s, a]` lives at index `s * n_actions + a`.
#[derive(Debug, Clone)]
pub struct Sarsa {
    q: Vec<f32>,
    n_states: usize,
    n_actions: usize,
    /// Learning rate α ∈ (0, 1].
    alpha: f32,
    /// Discount factor γ ∈ [0, 1].
    gamma: f32,
}

impl Sarsa {
    /// Create a SARSA agent with a zero-initialised table.
    ///
    /// # Errors
    ///
    /// * [`RlError::InvalidHyperparameter`] if `n_states` or `n_actions` is 0,
    ///   if `alpha ∉ (0, 1]`, or if `gamma ∉ [0, 1]`.
    pub fn new(n_states: usize, n_actions: usize, alpha: f32, gamma: f32) -> RlResult<Self> {
        if n_states == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "n_states".into(),
                msg: "must be > 0".into(),
            });
        }
        if n_actions == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "n_actions".into(),
                msg: "must be > 0".into(),
            });
        }
        if !(alpha > 0.0 && alpha <= 1.0) {
            return Err(RlError::InvalidHyperparameter {
                name: "alpha".into(),
                msg: "must be in (0, 1]".into(),
            });
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(RlError::InvalidHyperparameter {
                name: "gamma".into(),
                msg: "must be in [0, 1]".into(),
            });
        }
        Ok(Self {
            q: vec![0.0; n_states * n_actions],
            n_states,
            n_actions,
            alpha,
            gamma,
        })
    }

    /// Number of states.
    #[must_use]
    #[inline]
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// Number of actions.
    #[must_use]
    #[inline]
    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    /// Read `Q[state, action]`.
    ///
    /// # Errors
    ///
    /// * [`RlError::InvalidHyperparameter`] if `state` or `action` is out of
    ///   range.
    pub fn q_value(&self, state: usize, action: usize) -> RlResult<f32> {
        let idx = self.index(state, action)?;
        Ok(self.q[idx])
    }

    /// Greedy action `argmax_a Q[state, a]` (ties broken toward lowest index).
    ///
    /// # Errors
    ///
    /// * [`RlError::InvalidHyperparameter`] if `state` is out of range.
    pub fn greedy_action(&self, state: usize) -> RlResult<usize> {
        if state >= self.n_states {
            return Err(RlError::InvalidHyperparameter {
                name: "state".into(),
                msg: format!("{state} out of range [0, {})", self.n_states),
            });
        }
        let row = &self.q[state * self.n_actions..(state + 1) * self.n_actions];
        let mut best = 0_usize;
        let mut best_val = row[0];
        for (a, &v) in row.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best = a;
            }
        }
        Ok(best)
    }

    /// Perform a single on-policy SARSA update from a full transition tuple
    /// `(s, a, r, s', a')`.
    ///
    /// Returns the TD error `target − Q[s, a]`.
    ///
    /// # Arguments
    ///
    /// * `state`, `action` — the visited state-action pair.
    /// * `reward` — observed reward.
    /// * `next_state`, `next_action` — the *next* state and the action the
    ///   behaviour policy chose there (ignored when `done`).
    /// * `done` — whether the episode terminated at the transition.
    ///
    /// # Errors
    ///
    /// * [`RlError::InvalidHyperparameter`] if any index is out of range.
    pub fn update(
        &mut self,
        state: usize,
        action: usize,
        reward: f32,
        next_state: usize,
        next_action: usize,
        done: bool,
    ) -> RlResult<f32> {
        let idx = self.index(state, action)?;
        let bootstrap = if done {
            0.0
        } else {
            self.q_value(next_state, next_action)?
        };
        let target = reward + self.gamma * bootstrap;
        let td_error = target - self.q[idx];
        self.q[idx] += self.alpha * td_error;
        Ok(td_error)
    }

    /// Perform an **Expected SARSA** update, bootstrapping with the expected
    /// next value `Σ_{a'} π(a'|s') · Q[s', a']`.
    ///
    /// `next_policy` must be a probability distribution of length `n_actions`
    /// over the actions in `next_state`.  It need not sum exactly to 1, but
    /// each entry must be non-negative and finite.
    ///
    /// Returns the TD error.
    ///
    /// # Errors
    ///
    /// * [`RlError::InvalidHyperparameter`] if any index is out of range.
    /// * [`RlError::DimensionMismatch`] if `next_policy.len() != n_actions`.
    /// * [`RlError::InvalidDistribution`] if any probability is negative/NaN.
    pub fn update_expected(
        &mut self,
        state: usize,
        action: usize,
        reward: f32,
        next_state: usize,
        next_policy: &[f32],
        done: bool,
    ) -> RlResult<f32> {
        let idx = self.index(state, action)?;
        if next_policy.len() != self.n_actions {
            return Err(RlError::DimensionMismatch {
                expected: self.n_actions,
                got: next_policy.len(),
            });
        }
        let bootstrap = if done {
            0.0
        } else {
            if next_state >= self.n_states {
                return Err(RlError::InvalidHyperparameter {
                    name: "next_state".into(),
                    msg: format!("{next_state} out of range [0, {})", self.n_states),
                });
            }
            let row = &self.q[next_state * self.n_actions..(next_state + 1) * self.n_actions];
            let mut acc = 0.0_f32;
            for (&p, &qv) in next_policy.iter().zip(row.iter()) {
                if p < 0.0 || !p.is_finite() {
                    return Err(RlError::InvalidDistribution { sum: p, tol: 0.0 });
                }
                acc += p * qv;
            }
            acc
        };
        let target = reward + self.gamma * bootstrap;
        let td_error = target - self.q[idx];
        self.q[idx] += self.alpha * td_error;
        Ok(td_error)
    }

    /// Immutable view of the underlying flat Q-table.
    #[must_use]
    #[inline]
    pub fn table(&self) -> &[f32] {
        &self.q
    }

    fn index(&self, state: usize, action: usize) -> RlResult<usize> {
        if state >= self.n_states {
            return Err(RlError::InvalidHyperparameter {
                name: "state".into(),
                msg: format!("{state} out of range [0, {})", self.n_states),
            });
        }
        if action >= self.n_actions {
            return Err(RlError::InvalidHyperparameter {
                name: "action".into(),
                msg: format!("{action} out of range [0, {})", self.n_actions),
            });
        }
        Ok(state * self.n_actions + action)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn agent() -> Sarsa {
        Sarsa::new(5, 3, 0.1, 0.99).expect("valid hyperparameters")
    }

    #[test]
    fn new_zero_initialised() {
        let s = agent();
        assert_eq!(s.table().len(), 5 * 3);
        assert!(s.table().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn new_invalid_alpha_error() {
        assert!(Sarsa::new(5, 3, 0.0, 0.99).is_err());
        assert!(Sarsa::new(5, 3, 2.0, 0.99).is_err());
    }

    #[test]
    fn new_invalid_dims_error() {
        assert!(Sarsa::new(0, 3, 0.1, 0.99).is_err());
        assert!(Sarsa::new(5, 0, 0.1, 0.99).is_err());
    }

    #[test]
    fn update_moves_toward_target() {
        let mut s = agent();
        let td = s.update(0, 1, 1.0, 0, 0, true).expect("valid");
        assert!((td - 1.0).abs() < 1e-6);
        let v = s.q_value(0, 1).expect("valid");
        assert!((v - 0.1).abs() < 1e-6, "Q should be 0.1, got {v}");
    }

    #[test]
    fn update_uses_next_action_value() {
        let mut s = agent();
        // Make Q[1,2] large; Q[1,0] stays 0.
        for _ in 0..200 {
            s.update(1, 2, 1.0, 1, 2, true).expect("valid");
        }
        // Bootstrapping through next_action=2 (high) should give positive td.
        let td_high = s.update(0, 0, 0.0, 1, 2, false).expect("valid");
        // Bootstrapping through next_action=0 (zero) gives ~0 td.
        let td_low = s.update(4, 0, 0.0, 1, 0, false).expect("valid");
        assert!(
            td_high > td_low,
            "on-policy bootstrap must follow next_action: {td_high} vs {td_low}"
        );
    }

    #[test]
    fn done_ignores_bootstrap() {
        let mut s = agent();
        for _ in 0..100 {
            s.update(3, 0, 5.0, 3, 0, true).expect("valid");
        }
        let td = s.update(0, 0, 1.0, 3, 0, true).expect("valid");
        assert!((td - 1.0).abs() < 1e-6, "done target = reward, got {td}");
    }

    #[test]
    fn expected_sarsa_uniform_policy() {
        let mut s = agent();
        // Seed Q[1, :] = [0, 0, 0.9] approximately.
        for _ in 0..400 {
            s.update(1, 2, 1.0, 1, 2, true).expect("valid");
        }
        let q12 = s.q_value(1, 2).expect("valid");
        // Uniform policy expectation = mean of row = q12 / 3.
        let uniform = vec![1.0 / 3.0; 3];
        let td = s
            .update_expected(0, 0, 0.0, 1, &uniform, false)
            .expect("valid");
        let expected = 0.99 * (q12 / 3.0);
        assert!(
            (td - expected).abs() < 1e-3,
            "expected sarsa td={td}, want {expected}"
        );
    }

    #[test]
    fn expected_sarsa_greedy_equals_qlearning_bound() {
        let mut s = agent();
        for _ in 0..400 {
            s.update(1, 2, 1.0, 1, 2, true).expect("valid");
        }
        let q_max = s.q_value(1, 2).expect("valid");
        // Greedy (one-hot on best action 2) → expectation = max.
        let greedy = vec![0.0, 0.0, 1.0];
        let td = s
            .update_expected(0, 0, 0.0, 1, &greedy, false)
            .expect("valid");
        assert!((td - 0.99 * q_max).abs() < 1e-3);
    }

    #[test]
    fn expected_sarsa_dim_mismatch_error() {
        let mut s = agent();
        let bad = vec![0.5, 0.5]; // length 2, need 3
        assert!(matches!(
            s.update_expected(0, 0, 0.0, 1, &bad, false),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn expected_sarsa_negative_prob_error() {
        let mut s = agent();
        let bad = vec![-0.1, 0.6, 0.5];
        assert!(matches!(
            s.update_expected(0, 0, 0.0, 1, &bad, false),
            Err(RlError::InvalidDistribution { .. })
        ));
    }

    #[test]
    fn out_of_range_errors() {
        let mut s = agent();
        assert!(s.q_value(99, 0).is_err());
        assert!(s.update(99, 0, 1.0, 0, 0, false).is_err());
        assert!(s.greedy_action(99).is_err());
    }

    #[test]
    fn converges_two_state_chain() {
        let mut s = Sarsa::new(2, 1, 0.5, 0.9).expect("valid");
        for _ in 0..100 {
            s.update(0, 0, 1.0, 1, 0, true).expect("valid");
        }
        let v = s.q_value(0, 0).expect("valid");
        assert!(
            (v - 1.0).abs() < 1e-3,
            "Q[0,0] should converge to 1, got {v}"
        );
    }
}

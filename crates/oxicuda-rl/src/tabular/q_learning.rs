//! # Tabular Q-Learning
//!
//! Watkins (1989), "Learning from Delayed Rewards"; Watkins & Dayan (1992),
//! "Q-Learning", Machine Learning 8(3-4):279-292.
//!
//! ## Algorithm
//!
//! Q-learning is the canonical **off-policy** temporal-difference control
//! method.  It maintains an action-value table `Q[s, a]` and updates it from
//! single transitions `(s, a, r, s')` toward the bootstrapped target that
//! uses the *greedy* next action (regardless of the behaviour policy):
//!
//! ```text
//! target = r + γ · (1 − done) · max_{a'} Q[s', a']
//! Q[s, a] ← Q[s, a] + α · (target − Q[s, a])
//! ```
//!
//! Because the target maximises over `a'` rather than following the
//! behaviour policy, Q-learning converges to the optimal action-value
//! function `Q*` under standard Robbins-Monro step-size conditions, even
//! when actions are taken ε-greedily.

use crate::error::{RlError, RlResult};

/// Tabular Q-learning agent backed by a dense `n_states × n_actions` table.
///
/// The table is stored row-major: `Q[s, a]` lives at index `s * n_actions + a`.
#[derive(Debug, Clone)]
pub struct QLearning {
    q: Vec<f32>,
    n_states: usize,
    n_actions: usize,
    /// Learning rate α ∈ (0, 1].
    alpha: f32,
    /// Discount factor γ ∈ [0, 1].
    gamma: f32,
}

impl QLearning {
    /// Create a Q-learning agent with a zero-initialised table.
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

    /// Maximum action value `max_a Q[state, a]`.
    ///
    /// # Errors
    ///
    /// * [`RlError::InvalidHyperparameter`] if `state` is out of range.
    pub fn max_q(&self, state: usize) -> RlResult<f32> {
        if state >= self.n_states {
            return Err(RlError::InvalidHyperparameter {
                name: "state".into(),
                msg: format!("{state} out of range [0, {})", self.n_states),
            });
        }
        let row = &self.q[state * self.n_actions..(state + 1) * self.n_actions];
        Ok(row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
    }

    /// Perform a single off-policy Q-learning update from a transition.
    ///
    /// Returns the TD error `target − Q[state, action]` for diagnostics /
    /// prioritised replay priorities.
    ///
    /// # Arguments
    ///
    /// * `state`, `action` — the visited state-action pair.
    /// * `reward` — observed reward.
    /// * `next_state` — successor state (ignored when `done`).
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
        done: bool,
    ) -> RlResult<f32> {
        let idx = self.index(state, action)?;
        let bootstrap = if done { 0.0 } else { self.max_q(next_state)? };
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

    fn agent() -> QLearning {
        QLearning::new(5, 3, 0.1, 0.99).expect("valid hyperparameters")
    }

    #[test]
    fn new_zero_initialised() {
        let q = agent();
        assert_eq!(q.table().len(), 5 * 3);
        assert!(q.table().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn new_invalid_alpha_error() {
        assert!(QLearning::new(5, 3, 0.0, 0.99).is_err());
        assert!(QLearning::new(5, 3, 1.5, 0.99).is_err());
    }

    #[test]
    fn new_invalid_gamma_error() {
        assert!(QLearning::new(5, 3, 0.1, 1.5).is_err());
        assert!(QLearning::new(5, 3, 0.1, -0.1).is_err());
    }

    #[test]
    fn new_zero_dims_error() {
        assert!(QLearning::new(0, 3, 0.1, 0.99).is_err());
        assert!(QLearning::new(5, 0, 0.1, 0.99).is_err());
    }

    #[test]
    fn update_moves_toward_target() {
        let mut q = agent();
        // Terminal reward of 1: target = 1, Q goes from 0 toward 1 by alpha.
        let td = q.update(0, 1, 1.0, 0, true).expect("valid indices");
        assert!(
            (td - 1.0).abs() < 1e-6,
            "td error should be reward, got {td}"
        );
        let v = q.q_value(0, 1).expect("valid indices");
        assert!(
            (v - 0.1).abs() < 1e-6,
            "Q should be alpha*target=0.1, got {v}"
        );
    }

    #[test]
    fn update_uses_max_next() {
        let mut q = agent();
        // Seed a high value at next state's action 2.
        for _ in 0..200 {
            q.update(1, 2, 1.0, 1, true).expect("valid");
        }
        let max_next = q.max_q(1).expect("valid state");
        assert!(max_next > 0.5, "max_q should be large, got {max_next}");
        // Now an update into state 0 from a transition s'->1 should bootstrap.
        let td = q.update(0, 0, 0.0, 1, false).expect("valid");
        assert!(td > 0.0, "bootstrapped target should be positive, got {td}");
    }

    #[test]
    fn done_ignores_bootstrap() {
        let mut q = agent();
        // Put junk in next state to ensure it is NOT used when done.
        for _ in 0..100 {
            q.update(3, 0, 5.0, 3, true).expect("valid");
        }
        let td = q.update(0, 0, 1.0, 3, true).expect("valid");
        assert!(
            (td - 1.0).abs() < 1e-6,
            "done update target must equal reward, got {td}"
        );
    }

    #[test]
    fn greedy_action_picks_max() {
        let mut q = agent();
        q.update(2, 1, 10.0, 2, true).expect("valid");
        assert_eq!(q.greedy_action(2).expect("valid state"), 1);
    }

    #[test]
    fn greedy_action_ties_lowest_index() {
        let q = agent(); // all zero → tie → index 0
        assert_eq!(q.greedy_action(0).expect("valid state"), 0);
    }

    #[test]
    fn out_of_range_errors() {
        let mut q = agent();
        assert!(q.q_value(5, 0).is_err());
        assert!(q.q_value(0, 3).is_err());
        assert!(q.greedy_action(99).is_err());
        assert!(q.update(99, 0, 1.0, 0, false).is_err());
        assert!(q.max_q(99).is_err());
    }

    #[test]
    fn converges_two_state_chain() {
        // Deterministic chain: state 0 --action 0--> state 1 (terminal, reward 1).
        // Optimal Q[0,0] = gamma * 0 + 1 ... actually target = r + gamma*max(terminal)=1.
        let mut q = QLearning::new(2, 1, 0.5, 0.9).expect("valid");
        for _ in 0..100 {
            q.update(0, 0, 1.0, 1, true).expect("valid");
        }
        let v = q.q_value(0, 0).expect("valid");
        assert!(
            (v - 1.0).abs() < 1e-3,
            "Q[0,0] should converge to 1, got {v}"
        );
    }

    #[test]
    fn table_values_finite() {
        let mut q = agent();
        for s in 0..5 {
            for a in 0..3 {
                q.update(s, a, (s + a) as f32, (s + 1) % 5, false)
                    .expect("valid");
            }
        }
        assert!(q.table().iter().all(|v| v.is_finite()));
    }
}

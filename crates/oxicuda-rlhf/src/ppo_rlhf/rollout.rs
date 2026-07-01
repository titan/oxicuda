use crate::error::{RlhfError, RlhfResult};

pub struct RlhfRollout {
    pub log_probs: Vec<f32>,
    pub ref_log_probs: Vec<f32>,
    pub rewards: Vec<f32>,
    pub values: Vec<f32>,
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
}

impl RlhfRollout {
    pub fn new(capacity: usize) -> Self {
        Self {
            log_probs: Vec::with_capacity(capacity),
            ref_log_probs: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            advantages: Vec::with_capacity(capacity),
            returns: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.log_probs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.log_probs.is_empty()
    }

    pub fn compute_advantages(&mut self, gamma: f32, lam: f32) -> RlhfResult<()> {
        let t = self.rewards.len();
        if t == 0 {
            return Err(RlhfError::EmptyInput);
        }
        if self.values.len() != t {
            return Err(RlhfError::DimensionMismatch {
                expected: t,
                got: self.values.len(),
            });
        }
        self.advantages = vec![0.0_f32; t];
        self.returns = vec![0.0_f32; t];
        let mut gae = 0.0_f32;
        for step in (0..t).rev() {
            let next_value = if step + 1 < t {
                self.values[step + 1]
            } else {
                0.0
            };
            let delta = self.rewards[step] + gamma * next_value - self.values[step];
            gae = delta + gamma * lam * gae;
            self.advantages[step] = gae;
            self.returns[step] = gae + self.values[step];
        }
        Ok(())
    }

    pub fn apply_kl_penalty(&mut self, beta: f32) -> RlhfResult<()> {
        if self.log_probs.len() != self.ref_log_probs.len() {
            return Err(RlhfError::DimensionMismatch {
                expected: self.log_probs.len(),
                got: self.ref_log_probs.len(),
            });
        }
        if self.rewards.len() != self.log_probs.len() {
            return Err(RlhfError::DimensionMismatch {
                expected: self.log_probs.len(),
                got: self.rewards.len(),
            });
        }
        self.rewards
            .iter_mut()
            .zip(self.log_probs.iter())
            .zip(self.ref_log_probs.iter())
            .for_each(|((r, &lp), &ref_lp)| {
                let kl = lp - ref_lp;
                *r -= beta * kl;
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a rollout with only rewards and values pre-loaded (the common setup
    /// for `compute_advantages` tests).
    fn make_rollout(rewards: Vec<f32>, values: Vec<f32>) -> RlhfRollout {
        let n = rewards.len();
        let mut r = RlhfRollout::new(n);
        r.rewards = rewards;
        r.values = values;
        r
    }

    // ── GAE exact recursion ───────────────────────────────────────────────────

    #[test]
    fn gae_exact_three_step() {
        // γ=0.5, λ=0.5, rewards=[1,1,1], values=[0.5,0.5,0.5].
        // All quantities are exact binary fractions — no rounding error.
        //
        // Backward pass (δ_t = r_t + γ·V_{t+1} − V_t, A_t = δ_t + γλ·A_{t+1}):
        //   t=2: δ=1.0+0−0.5=0.5,   gae=0.5+0=0.5,      ret=0.5+0.5=1.0
        //   t=1: δ=1.0+0.25−0.5=0.75, gae=0.75+0.25*0.5=0.875, ret=1.375
        //   t=0: δ=0.75,             gae=0.75+0.25*0.875=0.96875, ret=1.46875
        let mut roll = make_rollout(vec![1.0, 1.0, 1.0], vec![0.5, 0.5, 0.5]);
        roll.compute_advantages(0.5, 0.5).expect("valid rollout");

        assert!(
            (roll.advantages[2] - 0.5).abs() < 1e-6,
            "adv[2] expected 0.5, got {}",
            roll.advantages[2]
        );
        assert!(
            (roll.advantages[1] - 0.875).abs() < 1e-6,
            "adv[1] expected 0.875, got {}",
            roll.advantages[1]
        );
        assert!(
            (roll.advantages[0] - 0.96875).abs() < 1e-6,
            "adv[0] expected 0.96875, got {}",
            roll.advantages[0]
        );
        assert!(
            (roll.returns[2] - 1.0).abs() < 1e-6,
            "ret[2] expected 1.0, got {}",
            roll.returns[2]
        );
        assert!(
            (roll.returns[1] - 1.375).abs() < 1e-6,
            "ret[1] expected 1.375, got {}",
            roll.returns[1]
        );
        assert!(
            (roll.returns[0] - 1.46875).abs() < 1e-6,
            "ret[0] expected 1.46875, got {}",
            roll.returns[0]
        );
    }

    // ── Limiting case: λ=0 → one-step TD residual ────────────────────────────

    #[test]
    fn gae_lambda_zero_equals_td_residual() {
        // When λ=0, the γλ term vanishes and A_t = δ_t = r_t + γ·V_{t+1} − V_t.
        // γ=0.5, rewards=[1.0, 2.0], values=[0.5, 1.0]:
        //   t=1: δ = 2.0 + 0.5·0 − 1.0 = 1.0
        //   t=0: δ = 1.0 + 0.5·1.0 − 0.5 = 1.0
        let mut roll = make_rollout(vec![1.0, 2.0], vec![0.5, 1.0]);
        roll.compute_advantages(0.5, 0.0).expect("valid rollout");

        assert!(
            (roll.advantages[1] - 1.0).abs() < 1e-6,
            "λ=0 adv[1] expected 1.0 (TD δ), got {}",
            roll.advantages[1]
        );
        assert!(
            (roll.advantages[0] - 1.0).abs() < 1e-6,
            "λ=0 adv[0] expected 1.0 (TD δ), got {}",
            roll.advantages[0]
        );
        // returns[t] = advantages[t] + values[t] always.
        for t in 0..2 {
            let expected_ret = roll.advantages[t] + roll.values[t];
            assert!(
                (roll.returns[t] - expected_ret).abs() < 1e-6,
                "λ=0 ret[{t}] must equal adv+val ({expected_ret}), got {}",
                roll.returns[t]
            );
        }
    }

    // ── Limiting case: λ=1 → Monte-Carlo return minus baseline ───────────────

    #[test]
    fn gae_lambda_one_equals_mc_minus_baseline() {
        // When λ=1, GAE collapses to A_t = G_t − V_t where G_t is the discounted
        // Monte-Carlo return from step t onwards (with V_{T}=0 at terminal).
        // γ=0.5, rewards=[1,1,1], values=[0.5,0.5,0.5]:
        //   G_2 = 1.0             → A_2 = 0.5
        //   G_1 = 1+0.5·1 = 1.5  → A_1 = 1.0
        //   G_0 = 1+0.5·1.5 = 1.75 → A_0 = 1.25
        let mut roll = make_rollout(vec![1.0, 1.0, 1.0], vec![0.5, 0.5, 0.5]);
        roll.compute_advantages(0.5, 1.0).expect("valid rollout");

        assert!(
            (roll.advantages[2] - 0.5).abs() < 1e-6,
            "λ=1 adv[2] expected 0.5, got {}",
            roll.advantages[2]
        );
        assert!(
            (roll.advantages[1] - 1.0).abs() < 1e-6,
            "λ=1 adv[1] expected 1.0, got {}",
            roll.advantages[1]
        );
        assert!(
            (roll.advantages[0] - 1.25).abs() < 1e-6,
            "λ=1 adv[0] expected 1.25, got {}",
            roll.advantages[0]
        );
    }

    // ── Limiting case: γ=0 → immediate reward minus baseline ─────────────────

    #[test]
    fn gae_gamma_zero_equals_immediate_residual() {
        // When γ=0, all future terms vanish: A_t = r_t − V_t.
        // returns[t] = A_t + V_t = r_t (immediate reward only).
        let rewards = vec![1.0_f32, 2.0, 3.0];
        let values = vec![0.5_f32, 1.0, 1.5];
        let mut roll = make_rollout(rewards.clone(), values.clone());
        roll.compute_advantages(0.0, 0.9).expect("valid rollout");

        for t in 0..3 {
            let expected_adv = rewards[t] - values[t];
            assert!(
                (roll.advantages[t] - expected_adv).abs() < 1e-6,
                "γ=0 adv[{t}] expected {expected_adv}, got {}",
                roll.advantages[t]
            );
            // γ=0 → returns[t] = r_t (no future bootstrapping).
            assert!(
                (roll.returns[t] - rewards[t]).abs() < 1e-6,
                "γ=0 ret[{t}] expected r_t={}, got {}",
                rewards[t],
                roll.returns[t]
            );
        }
    }

    // ── Structural identity: returns = advantages + values ────────────────────

    #[test]
    fn returns_equals_advantages_plus_values() {
        // This must hold for any valid (γ, λ) by construction of compute_advantages.
        let mut roll = make_rollout(vec![0.5, 1.0, 1.5], vec![0.25, 0.75, 1.25]);
        roll.compute_advantages(0.99, 0.95).expect("valid rollout");

        for t in 0..3 {
            let expected = roll.advantages[t] + roll.values[t];
            assert!(
                (roll.returns[t] - expected).abs() < 1e-5,
                "returns[{t}] = adv+val identity violated: got {}, expected {expected}",
                roll.returns[t]
            );
        }
    }

    // ── Error path: empty rewards ─────────────────────────────────────────────

    #[test]
    fn empty_rollout_compute_advantages_errors() {
        let mut roll = RlhfRollout::new(0);
        let err = roll
            .compute_advantages(0.99, 0.95)
            .expect_err("empty rollout must error");
        assert!(
            matches!(err, RlhfError::EmptyInput),
            "expected EmptyInput for zero-step rollout, got {err:?}"
        );
    }

    // ── apply_kl_penalty analytic ─────────────────────────────────────────────

    #[test]
    fn apply_kl_penalty_analytic() {
        // log_probs=[0.0, 0.0], ref_log_probs=[−1.0, −1.0]:
        //   per-token KL = lp − ref_lp = 0 − (−1) = 1.0
        //   new reward   = reward − β·KL = {2.0, 3.0} − 0.5·1.0 = {1.5, 2.5}
        let mut roll = RlhfRollout::new(2);
        roll.log_probs = vec![0.0, 0.0];
        roll.ref_log_probs = vec![-1.0, -1.0];
        roll.rewards = vec![2.0, 3.0];
        roll.apply_kl_penalty(0.5).expect("valid inputs");

        assert!(
            (roll.rewards[0] - 1.5).abs() < 1e-6,
            "after KL penalty rewards[0] expected 1.5, got {}",
            roll.rewards[0]
        );
        assert!(
            (roll.rewards[1] - 2.5).abs() < 1e-6,
            "after KL penalty rewards[1] expected 2.5, got {}",
            roll.rewards[1]
        );
    }
}

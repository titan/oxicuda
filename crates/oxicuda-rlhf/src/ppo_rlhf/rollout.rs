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

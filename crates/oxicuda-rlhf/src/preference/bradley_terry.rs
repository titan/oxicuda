use crate::error::{RlhfError, RlhfResult};
use crate::handle::LcgRng;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn bt_reward_loss(chosen_rewards: &[f32], rejected_rewards: &[f32]) -> RlhfResult<f32> {
    if chosen_rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if chosen_rewards.len() != rejected_rewards.len() {
        return Err(RlhfError::MismatchedPairLength {
            chosen: chosen_rewards.len(),
            rejected: rejected_rewards.len(),
        });
    }
    let loss: f32 = chosen_rewards
        .iter()
        .zip(rejected_rewards.iter())
        .map(|(&rw, &rl)| -sigmoid(rw - rl).ln())
        .sum::<f32>()
        / chosen_rewards.len() as f32;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

pub struct RewardHead {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl RewardHead {
    pub fn new(dim: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0_f32 / dim as f32).sqrt();
        let weights = (0..dim)
            .map(|_| {
                let (a, _) = rng.next_normal_pair();
                a * scale
            })
            .collect();
        Self { weights, bias: 0.0 }
    }

    pub fn forward(&self, hidden: &[f32]) -> RlhfResult<f32> {
        if hidden.len() != self.weights.len() {
            return Err(RlhfError::DimensionMismatch {
                expected: self.weights.len(),
                got: hidden.len(),
            });
        }
        let dot: f32 = hidden
            .iter()
            .zip(self.weights.iter())
            .map(|(&h, &w)| h * w)
            .sum();
        let out = dot + self.bias;
        if out.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(out)
    }
}

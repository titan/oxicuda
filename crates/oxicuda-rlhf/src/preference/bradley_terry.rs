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

/// Gradient of the Bradley-Terry reward loss w.r.t. the chosen / rejected rewards.
///
/// Finite-difference verified against [`bt_reward_loss`].
#[derive(Debug, Clone)]
pub struct BtRewardGrad {
    /// `∂L/∂r_w` for each chosen reward (same length / order as the input).
    pub d_chosen_rewards: Vec<f32>,
    /// `∂L/∂r_l` for each rejected reward (same length / order as the input).
    pub d_rejected_rewards: Vec<f32>,
}

/// Analytic gradient of [`bt_reward_loss`].
///
/// Per pair `L = −log σ(r_w − r_l)`, so with `d = r_w − r_l`,
/// `∂L/∂r_w = −σ(−d)` and `∂L/∂r_l = +σ(−d)` (raising the chosen reward, or
/// lowering the rejected reward, decreases the loss). The mean reduction scales
/// every partial by `1 / n`.
///
/// # Errors
///
/// - [`RlhfError::EmptyInput`] if `chosen_rewards` is empty.
/// - [`RlhfError::MismatchedPairLength`] if the two slices differ in length.
/// - [`RlhfError::NanEncountered`] on a non-finite gradient.
pub fn bt_reward_grad(
    chosen_rewards: &[f32],
    rejected_rewards: &[f32],
) -> RlhfResult<BtRewardGrad> {
    if chosen_rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if chosen_rewards.len() != rejected_rewards.len() {
        return Err(RlhfError::MismatchedPairLength {
            chosen: chosen_rewards.len(),
            rejected: rejected_rewards.len(),
        });
    }
    let inv_n = 1.0 / chosen_rewards.len() as f32;
    let mut d_chosen_rewards = Vec::with_capacity(chosen_rewards.len());
    let mut d_rejected_rewards = Vec::with_capacity(chosen_rewards.len());
    for (&rw, &rl) in chosen_rewards.iter().zip(rejected_rewards.iter()) {
        // dL/dd = −σ(−d); chain through d = r_w − r_l.
        let g = -sigmoid(-(rw - rl)) * inv_n;
        if !g.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        d_chosen_rewards.push(g);
        d_rejected_rewards.push(-g);
    }
    Ok(BtRewardGrad {
        d_chosen_rewards,
        d_rejected_rewards,
    })
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

#[cfg(test)]
mod grad_tests {
    use super::*;

    fn central_diff(f: impl Fn(f32) -> f32, x: f32, h: f32) -> f32 {
        ((f(x + h) as f64 - f(x - h) as f64) / (2.0 * h as f64)) as f32
    }

    fn assert_close(analytic: f32, fd: f32, label: &str) {
        let denom = analytic.abs().max(1e-3);
        let rel = (analytic - fd).abs() / denom;
        assert!(
            rel <= 1e-3,
            "{label}: analytic={analytic}, fd={fd}, rel_err={rel}"
        );
    }

    #[test]
    fn bt_reward_grad_matches_fd() {
        let chosen = [1.2_f32, 0.3, -0.5];
        let rejected = [0.4_f32, 0.9, -1.0];
        let g = bt_reward_grad(&chosen, &rejected).expect("grad");
        let h = 1e-2;
        for i in 0..chosen.len() {
            let fd_w = central_diff(
                |v| {
                    let mut c = chosen.to_vec();
                    c[i] = v;
                    bt_reward_loss(&c, &rejected).expect("loss")
                },
                chosen[i],
                h,
            );
            let fd_l = central_diff(
                |v| {
                    let mut r = rejected.to_vec();
                    r[i] = v;
                    bt_reward_loss(&chosen, &r).expect("loss")
                },
                rejected[i],
                h,
            );
            assert_close(g.d_chosen_rewards[i], fd_w, "d_chosen");
            assert_close(g.d_rejected_rewards[i], fd_l, "d_rejected");
        }
    }

    #[test]
    fn bt_reward_grad_signs_push_margin_up() {
        // Raising chosen reward / lowering rejected reward lowers the loss.
        let g = bt_reward_grad(&[0.0], &[0.0]).expect("grad");
        assert!(g.d_chosen_rewards[0] < 0.0, "{}", g.d_chosen_rewards[0]);
        assert!(g.d_rejected_rewards[0] > 0.0, "{}", g.d_rejected_rewards[0]);
        // At equal rewards σ(0) = 0.5 → ∂/∂r_w = −0.5 (n = 1).
        assert!((g.d_chosen_rewards[0] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn bt_reward_grad_mismatch_errors() {
        assert!(matches!(
            bt_reward_grad(&[1.0, 2.0], &[0.0]),
            Err(RlhfError::MismatchedPairLength { .. })
        ));
    }

    #[test]
    fn bt_reward_grad_empty_errors() {
        assert!(matches!(
            bt_reward_grad(&[], &[]),
            Err(RlhfError::EmptyInput)
        ));
    }
}

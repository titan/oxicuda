//! RLOO — REINFORCE Leave-One-Out (Ahmadian et al. 2024).
//!
//! Reference: Ahmadian, A., Cremer, C., Gallé, M., Fadaee, M., Kreutzer, J.,
//! Pietquin, O., Üstün, A., & Hooker, S. (2024). *Back to Basics: Revisiting
//! REINFORCE-Style Optimization for Learning from Human Feedback in LLMs*.
//! arXiv:2402.14740. <https://arxiv.org/abs/2402.14740>
//!
//! RLOO is a low-variance REINFORCE estimator for RLHF. For a single prompt it
//! samples a group of `k` complete responses, scores each with a reward model,
//! and forms a **leave-one-out baseline**: every sample is centred against the
//! mean reward of the *other* `k − 1` samples,
//!
//! ```text
//!   Âᵢ = rᵢ − (1 / (k − 1)) · Σ_{j ≠ i} rⱼ .
//! ```
//!
//! Because the baseline excludes sample `i`, it is an **unbiased** control
//! variate (it does not depend on the action whose log-prob is being scaled),
//! which is exactly what makes the estimator low-variance. Two algebraic
//! identities follow directly and are checked in the tests:
//!
//! ```text
//!   Âᵢ = (k / (k − 1)) · (rᵢ − mean(r)) ,        Σᵢ Âᵢ = 0 .
//! ```
//!
//! The policy is updated by the REINFORCE objective `maxₚ Σᵢ Âᵢ · log p(yᵢ)`,
//! where `log p(yᵢ)` is the sequence log-probability (the sum of token
//! log-probs) of response `i`. The functions here return the **mean negated**
//! objective, so a lower value corresponds to a better policy.
//!
//! Unlike GRPO (which divides the group-centred reward by the group standard
//! deviation), RLOO keeps the *unnormalised* leave-one-out advantage; this is
//! the defining difference between the two group-baseline estimators.

use crate::error::{RlhfError, RlhfResult};

/// Hyper-parameters for the KL-augmented RLOO objective.
#[derive(Debug, Clone)]
pub struct RlooConfig {
    /// Coefficient `β ≥ 0` of the per-sample KL penalty folded into the reward.
    pub kl_coeff: f32,
}

impl Default for RlooConfig {
    fn default() -> Self {
        Self { kl_coeff: 0.0 }
    }
}

impl RlooConfig {
    fn validate(&self) -> RlhfResult<()> {
        if !self.kl_coeff.is_finite() || self.kl_coeff < 0.0 {
            return Err(RlhfError::InvalidBeta {
                beta: self.kl_coeff,
            });
        }
        Ok(())
    }
}

/// Leave-one-out advantages `Âᵢ = rᵢ − mean_{j≠i}(rⱼ)` for a group of rewards.
///
/// Requires at least two samples (the baseline averages the *other* members).
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if `rewards` is empty.
/// - [`RlhfError::Internal`] if fewer than two rewards are supplied.
/// - [`RlhfError::NanEncountered`] if any reward is non-finite.
pub fn rloo_advantages(rewards: &[f32]) -> RlhfResult<Vec<f32>> {
    let k = rewards.len();
    if k == 0 {
        return Err(RlhfError::EmptyInput);
    }
    if k < 2 {
        return Err(RlhfError::Internal {
            msg: "RLOO requires at least 2 samples per group".to_string(),
        });
    }
    for &r in rewards {
        if !r.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
    }
    let sum: f32 = rewards.iter().sum();
    let denom = (k as f32) - 1.0;
    Ok(rewards
        .iter()
        .map(|&r| {
            let baseline = (sum - r) / denom;
            r - baseline
        })
        .collect())
}

/// Mean negated REINFORCE-LOO objective `−(1/k) Σᵢ Âᵢ · log p(yᵢ)`.
///
/// `rewards[i]` and `logps[i]` are the reward and the sequence log-probability
/// of response `i`. Identical rewards give zero advantages and therefore zero
/// loss (no learning signal).
///
/// # Errors
/// - [`RlhfError::DimensionMismatch`] if `rewards` and `logps` differ in length.
/// - Propagates errors from [`rloo_advantages`].
/// - [`RlhfError::NanEncountered`] if any log-prob or the result is non-finite.
pub fn rloo_loss(rewards: &[f32], logps: &[f32]) -> RlhfResult<f32> {
    if rewards.len() != logps.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: rewards.len(),
            got: logps.len(),
        });
    }
    let advantages = rloo_advantages(rewards)?;
    let k = rewards.len() as f32;
    let mut total = 0.0_f32;
    for (&adv, &lp) in advantages.iter().zip(logps.iter()) {
        if !lp.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        total += adv * lp;
    }
    let loss = -(total / k);
    if !loss.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// KL-augmented RLOO loss.
///
/// Each reward is first penalised by a single-sample sequence-KL estimate
/// `log p(yᵢ) − log p_ref(yᵢ)`,
///
/// ```text
///   r'ᵢ = rᵢ − β · ( log p(yᵢ) − log p_ref(yᵢ) ) ,
/// ```
///
/// then the standard leave-one-out REINFORCE loss is computed from `r'`. With
/// `kl_coeff = 0` this is identical to [`rloo_loss`].
///
/// # Errors
/// - [`RlhfError::InvalidBeta`] if `kl_coeff` is non-finite or negative.
/// - [`RlhfError::DimensionMismatch`] if the three slices differ in length.
/// - [`RlhfError::NanEncountered`] if any input or the result is non-finite.
pub fn rloo_loss_with_kl(
    rewards: &[f32],
    logps: &[f32],
    ref_logps: &[f32],
    cfg: &RlooConfig,
) -> RlhfResult<f32> {
    cfg.validate()?;
    let k = rewards.len();
    if logps.len() != k || ref_logps.len() != k {
        return Err(RlhfError::DimensionMismatch {
            expected: k,
            got: logps.len().min(ref_logps.len()),
        });
    }
    let mut augmented = Vec::with_capacity(k);
    for ((&r, &lp), &rlp) in rewards.iter().zip(logps.iter()).zip(ref_logps.iter()) {
        if !r.is_finite() || !lp.is_finite() || !rlp.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        augmented.push(r - cfg.kl_coeff * (lp - rlp));
    }
    rloo_loss(&augmented, logps)
}

/// Analytic gradient of [`rloo_loss`] w.r.t. the per-sample sequence log-probs.
///
/// The leave-one-out advantages `Âᵢ` depend only on the rewards (held as inputs,
/// not on the policy log-probs), so the REINFORCE objective `−(1/k) Σᵢ Âᵢ·logp_i`
/// has the deterministic gradient `∂L/∂logp_i = −Âᵢ / k`. Identical rewards give
/// zero advantages and therefore zero gradient. Finite-difference verified
/// against [`rloo_loss`].
///
/// # Errors
/// - [`RlhfError::DimensionMismatch`] if `rewards` and `logps` differ in length.
/// - Propagates errors from [`rloo_advantages`].
/// - [`RlhfError::NanEncountered`] if any log-prob or gradient is non-finite.
pub fn rloo_grad(rewards: &[f32], logps: &[f32]) -> RlhfResult<Vec<f32>> {
    if rewards.len() != logps.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: rewards.len(),
            got: logps.len(),
        });
    }
    let advantages = rloo_advantages(rewards)?;
    let k = rewards.len() as f32;
    let mut grads = Vec::with_capacity(advantages.len());
    for (&adv, &lp) in advantages.iter().zip(logps.iter()) {
        if !lp.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        let g = -adv / k;
        if !g.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        grads.push(g);
    }
    Ok(grads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advantages_sum_to_zero() {
        let adv = rloo_advantages(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("adv");
        let sum: f32 = adv.iter().sum();
        assert!(sum.abs() < 1e-5, "advantages must sum to zero, got {sum}");
    }

    #[test]
    fn advantages_match_centred_scaling_identity() {
        let rewards = [0.5_f32, 1.5, -2.0, 4.0];
        let k = rewards.len() as f32;
        let mean = rewards.iter().sum::<f32>() / k;
        let adv = rloo_advantages(&rewards).expect("adv");
        for (&r, &a) in rewards.iter().zip(adv.iter()) {
            let expected = (k / (k - 1.0)) * (r - mean);
            assert!(
                (a - expected).abs() < 1e-4,
                "Âᵢ must equal k/(k-1)·(rᵢ-mean): got {a}, expected {expected}"
            );
        }
    }

    #[test]
    fn two_sample_baseline_is_the_other_reward() {
        let adv = rloo_advantages(&[3.0, 1.0]).expect("adv");
        // k = 2 ⇒ baseline for i is just the other reward.
        assert!(
            (adv[0] - 2.0).abs() < 1e-6,
            "A0 = 3 - 1 = 2, got {}",
            adv[0]
        );
        assert!(
            (adv[1] + 2.0).abs() < 1e-6,
            "A1 = 1 - 3 = -2, got {}",
            adv[1]
        );
    }

    #[test]
    fn best_reward_positive_worst_negative() {
        let adv = rloo_advantages(&[0.0, 0.0, 0.0, 5.0]).expect("adv");
        assert!(adv[3] > 0.0, "highest reward must get positive advantage");
        for a in &adv[..3] {
            assert!(*a < 0.0, "below-average rewards must be negative");
        }
    }

    #[test]
    fn identical_rewards_give_zero_loss() {
        let loss = rloo_loss(&[2.0, 2.0, 2.0], &[-1.0, -0.5, -2.0]).expect("loss");
        assert!(
            loss.abs() < 1e-6,
            "no reward spread → no signal, got {loss}"
        );
    }

    #[test]
    fn loss_is_finite() {
        let loss = rloo_loss(&[1.0, 3.0, 2.0], &[-1.0, -0.8, -1.2]).expect("loss");
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
    }

    #[test]
    fn increasing_logp_of_best_sample_decreases_loss() {
        // dL/d(logp_i) = -(Aᵢ / k); the max-advantage sample has Aᵢ > 0, so
        // raising its log-prob must lower the loss.
        let rewards = [1.0_f32, 5.0, 2.0];
        let logps = [-1.0_f32, -1.0, -1.0];
        let adv = rloo_advantages(&rewards).expect("adv");
        let best = adv
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("partial_cmp should succeed"))
            .map(|(i, _)| i)
            .expect("argmax");
        let base = rloo_loss(&rewards, &logps).expect("base");
        let mut bumped = logps;
        bumped[best] += 0.1;
        let after = rloo_loss(&rewards, &bumped).expect("after");
        assert!(
            after < base,
            "raising best-advantage log-prob should lower loss: base={base}, after={after}"
        );
    }

    #[test]
    fn kl_coeff_zero_matches_plain_loss() {
        let rewards = [1.0_f32, 3.0, 2.0];
        let logps = [-1.0_f32, -0.8, -1.2];
        let ref_logps = [-1.1_f32, -0.7, -1.5];
        let cfg = RlooConfig { kl_coeff: 0.0 };
        let with_kl = rloo_loss_with_kl(&rewards, &logps, &ref_logps, &cfg).expect("kl");
        let plain = rloo_loss(&rewards, &logps).expect("plain");
        assert!(
            (with_kl - plain).abs() < 1e-6,
            "kl_coeff=0 must reduce to plain loss: {with_kl} vs {plain}"
        );
    }

    #[test]
    fn kl_penalty_changes_loss_and_stays_finite() {
        let rewards = [1.0_f32, 3.0, 2.0];
        let logps = [-1.0_f32, -0.2, -1.2];
        let ref_logps = [-1.1_f32, -2.0, -1.5];
        let cfg = RlooConfig { kl_coeff: 0.5 };
        let with_kl = rloo_loss_with_kl(&rewards, &logps, &ref_logps, &cfg).expect("kl");
        let plain = rloo_loss(&rewards, &logps).expect("plain");
        assert!(with_kl.is_finite(), "KL loss must be finite");
        assert!(
            (with_kl - plain).abs() > 1e-6,
            "a positive KL penalty should move the loss"
        );
    }

    #[test]
    fn single_sample_and_empty_error() {
        assert!(matches!(
            rloo_advantages(&[1.0]),
            Err(RlhfError::Internal { .. })
        ));
        assert!(matches!(rloo_advantages(&[]), Err(RlhfError::EmptyInput)));
    }

    #[test]
    fn advantages_nonfinite_error() {
        assert!(matches!(
            rloo_advantages(&[1.0, f32::INFINITY]),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn loss_dimension_mismatch_error() {
        assert!(matches!(
            rloo_loss(&[1.0, 2.0, 3.0], &[-1.0, -2.0]),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn kl_invalid_coeff_error() {
        let cfg = RlooConfig { kl_coeff: -0.1 };
        assert!(matches!(
            rloo_loss_with_kl(&[1.0, 2.0], &[-1.0, -1.0], &[-1.0, -1.0], &cfg),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }

    #[test]
    fn kl_dimension_mismatch_error() {
        let cfg = RlooConfig::default();
        assert!(matches!(
            rloo_loss_with_kl(&[1.0, 2.0], &[-1.0, -1.0], &[-1.0], &cfg),
            Err(RlhfError::DimensionMismatch { .. })
        ));
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
    fn rloo_grad_matches_fd() {
        let rewards = [1.0_f32, 5.0, 2.0, 0.5];
        let logps = [-1.0_f32, -0.8, -1.2, -1.5];
        let g = rloo_grad(&rewards, &logps).expect("grad");
        let h = 1e-2;
        for i in 0..logps.len() {
            let fd = central_diff(
                |v| {
                    let mut l = logps;
                    l[i] = v;
                    rloo_loss(&rewards, &l).expect("loss")
                },
                logps[i],
                h,
            );
            assert_close(g[i], fd, "rloo_grad");
            // Closed form: −Aᵢ/k.
            let adv = rloo_advantages(&rewards).expect("adv");
            assert_close(g[i], -adv[i] / rewards.len() as f32, "closed form");
        }
    }

    #[test]
    fn rloo_grad_best_sample_negative() {
        // The max-advantage sample has Aᵢ > 0, so its gradient is negative
        // (descent raises its log-prob).
        let rewards = [1.0_f32, 5.0, 2.0];
        let logps = [-1.0_f32, -1.0, -1.0];
        let g = rloo_grad(&rewards, &logps).expect("grad");
        assert!(
            g[1] < 0.0,
            "best sample gradient should be negative: {}",
            g[1]
        );
    }

    #[test]
    fn rloo_grad_zero_for_equal_rewards() {
        let g = rloo_grad(&[2.0, 2.0, 2.0], &[-1.0, -0.5, -2.0]).expect("grad");
        for &gi in &g {
            assert!(gi.abs() < 1e-6, "equal rewards → zero gradient, got {gi}");
        }
    }

    #[test]
    fn rloo_grad_dimension_mismatch_errors() {
        assert!(matches!(
            rloo_grad(&[1.0, 2.0, 3.0], &[-1.0, -2.0]),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }
}

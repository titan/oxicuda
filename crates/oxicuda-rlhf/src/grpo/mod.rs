//! GRPO — Group Relative Policy Optimization (Shao et al. 2024).
//!
//! Reference: Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H.,
//! Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). *DeepSeekMath: Pushing the
//! Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
//! <https://arxiv.org/abs/2402.03300>
//!
//! GRPO removes the separate value network used by PPO. For each prompt it samples
//! a **group** of `G` outputs, scores them with a reward model, and turns the group
//! rewards into baselines by *normalising within the group*:
//!
//! ```text
//!   Âᵢ = (rᵢ − mean(r)) / (std(r) + ε) ,        i = 1 … G .
//! ```
//!
//! Every token of output `i` receives the same scalar advantage `Âᵢ`. The policy is
//! then updated with the usual clipped surrogate plus a per-token KL penalty against
//! a frozen reference policy:
//!
//! ```text
//!   L = −mean_i mean_t [ min(ρ_{i,t} Âᵢ, clip(ρ_{i,t}, 1−ε, 1+ε) Âᵢ) − β · KL_{i,t} ]
//! ```
//!
//! where `ρ_{i,t} = exp(logπ − logπ_old)` is the importance ratio and `KL_{i,t}` is
//! the **k3 unbiased estimator** `exp(logπ_ref − logπ) − (logπ_ref − logπ) − 1 ≥ 0`
//! used in the DeepSeekMath paper (it is always non-negative and low-variance).
//!
//! All functions here operate on log-probabilities and rewards in pure CPU code; a
//! matching PTX kernel for the surrogate lives in [`crate::ptx_kernels`].

use crate::error::{RlhfError, RlhfResult};

/// Hyper-parameters for GRPO.
#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// PPO clip range `ε` (e.g. `0.2`).
    pub clip_eps: f32,
    /// KL-penalty coefficient `β` against the reference policy.
    pub kl_coeff: f32,
    /// Numerical-stability constant added to the group standard deviation.
    pub adv_eps: f32,
}

impl GrpoConfig {
    fn validate(&self) -> RlhfResult<()> {
        if !self.clip_eps.is_finite() || self.clip_eps <= 0.0 {
            return Err(RlhfError::Internal {
                msg: format!("clip_eps must be > 0, got {}", self.clip_eps),
            });
        }
        if !self.kl_coeff.is_finite() || self.kl_coeff < 0.0 {
            return Err(RlhfError::InvalidBeta {
                beta: self.kl_coeff,
            });
        }
        if !self.adv_eps.is_finite() || self.adv_eps <= 0.0 {
            return Err(RlhfError::Internal {
                msg: format!("adv_eps must be > 0, got {}", self.adv_eps),
            });
        }
        Ok(())
    }
}

/// Compute the group-relative advantages `Âᵢ = (rᵢ − μ) / (σ + ε)`.
///
/// Returns one advantage per group member. With a single member the advantage is
/// `0` (no relative signal).
///
/// # Errors
///
/// - [`RlhfError::EmptyInput`] if `rewards` is empty.
/// - [`RlhfError::NanEncountered`] if any reward is non-finite.
pub fn group_advantages(rewards: &[f32], adv_eps: f32) -> RlhfResult<Vec<f32>> {
    if rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if rewards.iter().any(|r| !r.is_finite()) {
        return Err(RlhfError::NanEncountered);
    }
    let g = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / g;
    let var = rewards
        .iter()
        .map(|&r| (r - mean) * (r - mean))
        .sum::<f32>()
        / g;
    let std = var.sqrt();
    Ok(rewards
        .iter()
        .map(|&r| (r - mean) / (std + adv_eps))
        .collect())
}

/// k3 unbiased KL estimator `exp(Δ) − Δ − 1` with `Δ = logπ_ref − logπ`.
///
/// This is the per-token KL surrogate used by GRPO; it is always non-negative.
#[inline]
#[must_use]
pub fn kl_k3(logp: f32, ref_logp: f32) -> f32 {
    let delta = ref_logp - logp;
    delta.exp() - delta - 1.0
}

/// Per-output GRPO surrogate for a single group member.
///
/// `token_logps`, `token_old_logps`, and `token_ref_logps` are the current, behaviour
/// (old-policy), and reference log-probs for each generated token of this output;
/// `advantage` is the scalar group-relative advantage broadcast to every token.
/// Returns the **mean per-token loss** (already negated, so lower is better).
///
/// # Errors
///
/// - [`RlhfError::EmptyInput`] if `token_logps` is empty.
/// - [`RlhfError::DimensionMismatch`] if the three token slices disagree in length.
/// - [`RlhfError::NanEncountered`] if the result is non-finite.
pub fn output_surrogate(
    token_logps: &[f32],
    token_old_logps: &[f32],
    token_ref_logps: &[f32],
    advantage: f32,
    cfg: &GrpoConfig,
) -> RlhfResult<f32> {
    cfg.validate()?;
    let t = token_logps.len();
    if t == 0 {
        return Err(RlhfError::EmptyInput);
    }
    if token_old_logps.len() != t || token_ref_logps.len() != t {
        return Err(RlhfError::DimensionMismatch {
            expected: t,
            got: token_old_logps.len().min(token_ref_logps.len()),
        });
    }
    let lo = 1.0 - cfg.clip_eps;
    let hi = 1.0 + cfg.clip_eps;
    let mut total = 0.0_f32;
    for ((&lp, &old), &rlp) in token_logps
        .iter()
        .zip(token_old_logps.iter())
        .zip(token_ref_logps.iter())
    {
        let ratio = (lp - old).exp();
        let unclipped = ratio * advantage;
        let clipped = ratio.clamp(lo, hi) * advantage;
        let surrogate = unclipped.min(clipped);
        let kl = kl_k3(lp, rlp);
        total += surrogate - cfg.kl_coeff * kl;
    }
    let loss = -(total / t as f32);
    if !loss.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// One generated output in a GRPO group: its reward and per-token log-probs.
#[derive(Debug, Clone)]
pub struct GrpoOutput {
    /// Scalar reward assigned to this output by the reward model.
    pub reward: f32,
    /// Current-policy per-token log-probabilities.
    pub logps: Vec<f32>,
    /// Behaviour-policy (old) per-token log-probabilities.
    pub old_logps: Vec<f32>,
    /// Reference-policy per-token log-probabilities.
    pub ref_logps: Vec<f32>,
}

/// Full GRPO loss over one group of sampled outputs.
///
/// Computes group-relative advantages from the rewards, then averages the per-output
/// clipped-surrogate-plus-KL loss across the group.
///
/// # Errors
///
/// - [`RlhfError::EmptyInput`] if `group` is empty.
/// - Propagates errors from [`group_advantages`] and [`output_surrogate`].
pub fn grpo_loss(group: &[GrpoOutput], cfg: &GrpoConfig) -> RlhfResult<f32> {
    cfg.validate()?;
    if group.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let rewards: Vec<f32> = group.iter().map(|o| o.reward).collect();
    let advantages = group_advantages(&rewards, cfg.adv_eps)?;
    let mut total = 0.0_f32;
    for (o, &adv) in group.iter().zip(advantages.iter()) {
        total += output_surrogate(&o.logps, &o.old_logps, &o.ref_logps, adv, cfg)?;
    }
    let loss = total / group.len() as f32;
    if !loss.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GrpoConfig {
        GrpoConfig {
            clip_eps: 0.2,
            kl_coeff: 0.04,
            adv_eps: 1e-4,
        }
    }

    #[test]
    fn advantages_zero_mean() {
        let adv =
            group_advantages(&[1.0, 2.0, 3.0, 4.0], 1e-8).expect("group_advantages should succeed");
        let mean: f32 = adv.iter().sum::<f32>() / adv.len() as f32;
        assert!(
            mean.abs() < 1e-5,
            "group advantages must be ~zero-mean, got {mean}"
        );
    }

    #[test]
    fn advantages_unit_std() {
        let adv = group_advantages(&[10.0, 20.0, 30.0, 40.0, 50.0], 1e-9)
            .expect("group_advantages should succeed");
        let mean: f32 = adv.iter().sum::<f32>() / adv.len() as f32;
        let var: f32 = adv.iter().map(|&a| (a - mean) * (a - mean)).sum::<f32>() / adv.len() as f32;
        assert!((var - 1.0).abs() < 1e-3, "std should be ~1, var={var}");
    }

    #[test]
    fn advantages_best_output_positive() {
        let adv =
            group_advantages(&[0.0, 0.0, 0.0, 5.0], 1e-6).expect("group_advantages should succeed");
        // The high-reward member must get a positive advantage; the rest negative.
        assert!(adv[3] > 0.0, "best reward must have positive advantage");
        for a in &adv[..3] {
            assert!(
                *a < 0.0,
                "below-average rewards must have negative advantage"
            );
        }
    }

    #[test]
    fn advantages_empty_errors() {
        assert!(matches!(
            group_advantages(&[], 1e-8),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn advantages_nonfinite_errors() {
        assert!(matches!(
            group_advantages(&[1.0, f32::NAN], 1e-8),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn kl_k3_zero_at_equal_logps() {
        let kl = kl_k3(-1.3, -1.3);
        assert!(kl.abs() < 1e-6, "KL must be 0 when logps match, got {kl}");
    }

    #[test]
    fn kl_k3_nonnegative() {
        for (lp, rlp) in [(-0.5, -2.0), (-3.0, -0.1), (0.0, -1.0), (-1.0, 0.0)] {
            let kl = kl_k3(lp, rlp);
            assert!(kl >= -1e-6, "k3 KL must be >= 0, got {kl} for ({lp},{rlp})");
        }
    }

    #[test]
    fn surrogate_no_update_when_ratio_one_and_no_kl() {
        // logp == old_logp ⇒ ratio = 1; logp == ref ⇒ KL = 0.
        // loss = -(1 * adv) = -adv.
        let mut c = cfg();
        c.kl_coeff = 0.0;
        let lp = vec![-1.0_f32, -1.0, -1.0];
        let loss =
            output_surrogate(&lp, &lp, &lp, 0.5, &c).expect("output_surrogate should succeed");
        assert!(
            (loss - (-0.5)).abs() < 1e-5,
            "loss should be -advantage, got {loss}"
        );
    }

    #[test]
    fn surrogate_clip_caps_positive_advantage() {
        // Large positive ratio with positive advantage must be clipped to (1+eps)*adv.
        let c = cfg();
        let adv = 1.0_f32;
        let lp = vec![0.0_f32]; // logp
        let old = vec![-2.0_f32]; // ratio = exp(2) ≈ 7.39, clipped to 1.2
        let ref_lp = vec![0.0_f32]; // KL = 0
        let loss =
            output_surrogate(&lp, &old, &ref_lp, adv, &c).expect("output_surrogate should succeed");
        // min(7.39*1, 1.2*1) = 1.2 ⇒ loss = -1.2
        assert!(
            (loss - (-1.2)).abs() < 1e-4,
            "clip must cap surrogate, got {loss}"
        );
    }

    #[test]
    fn surrogate_dim_mismatch_errors() {
        let c = cfg();
        let r = output_surrogate(&[0.0, 0.0], &[0.0], &[0.0, 0.0], 1.0, &c);
        assert!(matches!(r, Err(RlhfError::DimensionMismatch { .. })));
    }

    #[test]
    fn surrogate_empty_errors() {
        let c = cfg();
        assert!(matches!(
            output_surrogate(&[], &[], &[], 1.0, &c),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn grpo_loss_finite_and_empty_error() {
        let c = cfg();
        let group = vec![
            GrpoOutput {
                reward: 1.0,
                logps: vec![-1.0, -1.2],
                old_logps: vec![-1.1, -1.3],
                ref_logps: vec![-1.0, -1.1],
            },
            GrpoOutput {
                reward: 3.0,
                logps: vec![-0.5, -0.6],
                old_logps: vec![-0.7, -0.8],
                ref_logps: vec![-0.6, -0.7],
            },
        ];
        let loss = grpo_loss(&group, &c).expect("grpo_loss should succeed");
        assert!(loss.is_finite(), "GRPO loss must be finite, got {loss}");
        assert!(matches!(grpo_loss(&[], &c), Err(RlhfError::EmptyInput)));
    }

    #[test]
    fn invalid_config_rejected() {
        let mut c = cfg();
        c.clip_eps = 0.0;
        assert!(output_surrogate(&[0.0], &[0.0], &[0.0], 1.0, &c).is_err());
        let mut c2 = cfg();
        c2.kl_coeff = -1.0;
        assert!(matches!(
            output_surrogate(&[0.0], &[0.0], &[0.0], 1.0, &c2),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }
}

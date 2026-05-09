use crate::error::{RlhfError, RlhfResult};
use crate::ppo_rlhf::rollout::RlhfRollout;

pub struct RlhfPpoConfig {
    pub clip_ratio: f32,
    pub vf_coef: f32,
    pub ent_coef: f32,
    pub max_grad_norm: f32,
}

pub fn rlhf_ppo_loss(
    rollout: &RlhfRollout,
    old_log_probs: &[f32],
    cfg: &RlhfPpoConfig,
) -> RlhfResult<(f32, f32, f32, f32)> {
    let t = rollout.len();
    if t == 0 {
        return Err(RlhfError::EmptyInput);
    }
    if old_log_probs.len() != t {
        return Err(RlhfError::DimensionMismatch {
            expected: t,
            got: old_log_probs.len(),
        });
    }
    if rollout.advantages.len() != t || rollout.returns.len() != t || rollout.values.len() != t {
        return Err(RlhfError::Internal {
            msg: "advantages/returns/values not computed — call compute_advantages first".into(),
        });
    }

    let mut policy_loss_sum = 0.0_f32;
    let mut value_loss_sum = 0.0_f32;
    let mut entropy_sum = 0.0_f32;
    let mut approx_kl_sum = 0.0_f32;

    for ((&lp, &old_lp), (&adv, &ret)) in rollout
        .log_probs
        .iter()
        .zip(old_log_probs.iter())
        .zip(rollout.advantages.iter().zip(rollout.returns.iter()))
    {
        let log_ratio = lp - old_lp;
        let ratio = log_ratio.exp();
        let clipped = ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio);
        let pg_loss = -(ratio * adv).min(clipped * adv);
        policy_loss_sum += pg_loss;

        let value = rollout.values[0];
        let vf_loss = 0.5 * (value - ret) * (value - ret);
        value_loss_sum += vf_loss;

        entropy_sum += -lp;
        approx_kl_sum += log_ratio;
    }

    let n = t as f32;
    let policy_loss = policy_loss_sum / n;
    let value_loss = (cfg.vf_coef * value_loss_sum) / n;
    let entropy_bonus = (cfg.ent_coef * entropy_sum) / n;
    let approx_kl = approx_kl_sum / n;

    if policy_loss.is_nan() || value_loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok((policy_loss, value_loss, entropy_bonus, approx_kl))
}

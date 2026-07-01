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

    for (((&lp, &old_lp), (&adv, &ret)), &value) in rollout
        .log_probs
        .iter()
        .zip(old_log_probs.iter())
        .zip(rollout.advantages.iter().zip(rollout.returns.iter()))
        .zip(rollout.values.iter())
    {
        let log_ratio = lp - old_lp;
        let ratio = log_ratio.exp();
        let clipped = ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio);
        let pg_loss = -(ratio * adv).min(clipped * adv);
        policy_loss_sum += pg_loss;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RlhfError;
    use crate::ppo_rlhf::rollout::RlhfRollout;

    /// Build a fully-populated rollout with pre-computed advantages/returns/values.
    fn make_ppo_rollout(
        log_probs: Vec<f32>,
        advantages: Vec<f32>,
        returns: Vec<f32>,
        values: Vec<f32>,
    ) -> RlhfRollout {
        let n = log_probs.len();
        let mut r = RlhfRollout::new(n);
        r.log_probs = log_probs;
        r.ref_log_probs = vec![0.0_f32; n];
        r.rewards = vec![0.0_f32; n];
        r.values = values;
        r.advantages = advantages;
        r.returns = returns;
        r
    }

    fn default_cfg() -> RlhfPpoConfig {
        RlhfPpoConfig {
            clip_ratio: 0.2,
            vf_coef: 1.0,
            ent_coef: 0.01,
            max_grad_norm: 1.0,
        }
    }

    // ── ratio=1 (old==new log_prob): unclipped == clipped → policy_loss = −A ─

    #[test]
    fn ratio_one_unclipped_equals_clipped() {
        // lp = old_lp = −1.0 → log_ratio = 0, ratio = 1, clipped = 1.
        // A = 2.0: min(1*2, 1*2) = 2.0 → pg_loss = −2.0 → policy_loss = −2.0.
        // approx_kl = log_ratio / 1 = 0.0.
        let roll = make_ppo_rollout(vec![-1.0_f32], vec![2.0_f32], vec![3.0_f32], vec![1.0_f32]);
        let cfg = default_cfg();
        let (pl, _vl, _ent, kl) =
            rlhf_ppo_loss(&roll, &[-1.0_f32], &cfg).expect("ratio=1 must succeed");
        assert!(
            (pl - (-2.0)).abs() < 1e-5,
            "ratio=1 policy_loss expected -2.0, got {pl}"
        );
        assert!(kl.abs() < 1e-6, "ratio=1 approx_kl expected 0.0, got {kl}");
    }

    // ── Clip activates: positive advantage, ratio > 1+ε ──────────────────────

    #[test]
    fn positive_advantage_clip_activates() {
        // lp=0.5, old_lp=0.0 → ratio=exp(0.5)≈1.6487, ε=0.2 → clipped=1.2.
        // A=1.0: ratio*A=1.6487 > clipped*A=1.2 → min=1.2 → pg_loss=−1.2.
        let roll = make_ppo_rollout(vec![0.5_f32], vec![1.0_f32], vec![0.0_f32], vec![0.0_f32]);
        let cfg = default_cfg();
        let (pl, _vl, _ent, _kl) =
            rlhf_ppo_loss(&roll, &[0.0_f32], &cfg).expect("positive-adv clip must succeed");
        // Clipped surrogate: −(1+ε)*A = −1.2
        assert!(
            (pl - (-1.2)).abs() < 1e-5,
            "positive A + ratio>1+ε: policy_loss expected −1.2, got {pl}"
        );
    }

    // ── Clip activates: negative advantage, ratio < 1−ε ──────────────────────

    #[test]
    fn negative_advantage_clip_activates() {
        // lp=−0.5, old_lp=0.0 → ratio=exp(−0.5)≈0.6065, ε=0.2 → clipped=0.8.
        // A=−1.0: ratio*A=−0.6065, clipped*A=−0.8 → min=−0.8 → pg_loss=0.8.
        let roll = make_ppo_rollout(vec![-0.5_f32], vec![-1.0_f32], vec![0.0_f32], vec![0.0_f32]);
        let cfg = default_cfg();
        let (pl, _vl, _ent, _kl) =
            rlhf_ppo_loss(&roll, &[0.0_f32], &cfg).expect("negative-adv clip must succeed");
        // Clipped surrogate: −(1−ε)*A = −0.8*(−1) = 0.8
        assert!(
            (pl - 0.8).abs() < 1e-5,
            "negative A + ratio<1−ε: policy_loss expected 0.8, got {pl}"
        );
    }

    // ── Per-step value loss (bug-fix: values[step] not values[0]) ────────────
    //
    // Before the fix, `let value = rollout.values[0]` was used inside the loop,
    // meaning step 1 computed 0.5*(values[0]−ret[1])² = 0.5*(1.0−3.5)² = 3.125
    // instead of 0.5*(values[1]−ret[1])² = 0.5*(3.0−3.5)² = 0.125.

    #[test]
    fn value_loss_uses_per_step_values() {
        // values=[1.0, 3.0], returns=[1.5, 3.5].
        // Per-step:  vf0 = 0.5*(1.0−1.5)² = 0.125
        //            vf1 = 0.5*(3.0−3.5)² = 0.125
        // average = 0.25/2 = 0.125 → value_loss = vf_coef * 0.125 = 0.125.
        let roll = make_ppo_rollout(
            vec![0.0_f32, 0.0],
            vec![0.0_f32, 0.0],
            vec![1.5_f32, 3.5],
            vec![1.0_f32, 3.0],
        );
        let cfg = default_cfg(); // vf_coef = 1.0
        let (_pl, vl, _ent, _kl) =
            rlhf_ppo_loss(&roll, &[0.0_f32, 0.0], &cfg).expect("per-step vf must succeed");
        assert!(
            (vl - 0.125).abs() < 1e-5,
            "per-step value_loss expected 0.125, got {vl} (buggy code would yield 1.625)"
        );
    }

    // ── Entropy bonus analytic ────────────────────────────────────────────────

    #[test]
    fn entropy_bonus_analytic() {
        // lp=−2.0: entropy_sum = −(−2.0) = 2.0 → entropy_bonus = ent_coef * 2.0 / 1.
        // With ent_coef=0.5: entropy_bonus = 0.5 * 2.0 = 1.0.
        let roll = make_ppo_rollout(vec![-2.0_f32], vec![0.0_f32], vec![0.0_f32], vec![0.0_f32]);
        let cfg = RlhfPpoConfig {
            ent_coef: 0.5,
            ..default_cfg()
        };
        let (_pl, _vl, ent, _kl) =
            rlhf_ppo_loss(&roll, &[-2.0_f32], &cfg).expect("entropy analytic must succeed");
        assert!(
            (ent - 1.0).abs() < 1e-5,
            "entropy_bonus expected 1.0, got {ent}"
        );
    }

    // ── Error: empty rollout ──────────────────────────────────────────────────

    #[test]
    fn empty_rollout_errors() {
        let roll = RlhfRollout::new(0);
        let err = rlhf_ppo_loss(&roll, &[], &default_cfg()).expect_err("empty rollout must error");
        assert!(
            matches!(err, RlhfError::EmptyInput),
            "expected EmptyInput for zero-step rollout, got {err:?}"
        );
    }

    // ── Error: old_log_probs length mismatch → DimensionMismatch ─────────────

    #[test]
    fn old_log_probs_length_mismatch_errors() {
        let roll = make_ppo_rollout(
            vec![0.0_f32, 0.0],
            vec![1.0_f32, 1.0],
            vec![1.0_f32, 1.0],
            vec![0.5_f32, 0.5],
        );
        let err = rlhf_ppo_loss(&roll, &[0.0_f32], &default_cfg())
            .expect_err("old_log_probs length mismatch must error");
        assert!(
            matches!(err, RlhfError::DimensionMismatch { .. }),
            "expected DimensionMismatch, got {err:?}"
        );
    }
}

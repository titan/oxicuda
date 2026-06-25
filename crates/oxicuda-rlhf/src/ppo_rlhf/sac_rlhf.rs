//! Soft-Actor-Critic-style entropy-regularised RLHF.
//!
//! References:
//! * Haarnoja et al. 2018, "Soft Actor-Critic", arXiv:1801.01290 — the
//!   maximum-entropy actor-critic objective with a soft value baseline.
//! * Haarnoja et al. 2018, "Soft Actor-Critic Algorithms and Applications",
//!   arXiv:1812.05905 — automatic temperature tuning toward a target entropy.
//!
//! Standard PPO-RLHF maximises expected reward with a *fixed* entropy bonus.
//! Max-entropy RL instead folds entropy directly into the objective,
//!
//! ```text
//! J(π) = E[ Σ_t r_t + α · H(π(·|s_t)) ] ,
//! ```
//!
//! where the temperature `α ≥ 0` is *learned* so the policy's entropy tracks a
//! target `H̄` (typically a fraction of `log|vocab|`). This module provides the
//! three CPU pieces of a soft actor-critic step specialised to the single-step
//! RLHF bandit (one scalar reward per response, optionally KL-shaped):
//!
//! 1. the **soft target** `y = r − α · log π(a)` (entropy-augmented return — the
//!    `−α log π` term rewards uncertainty; note `−log π ≥ 0` is the per-sample
//!    entropy contribution),
//! 2. the **soft policy loss** `α · log π(a) − Q̂(a)` whose gradient pushes the
//!    policy toward high-value, high-entropy actions, and
//! 3. the **temperature loss** `−α · (log π(a) + H̄)` whose minimiser drives the
//!    realised entropy `−log π` toward `H̄`.
//!
//! Everything is forward-value only (the crate is gradient-free) and validates
//! its inputs.

use crate::error::{RlhfError, RlhfResult};

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the soft actor-critic RLHF objective.
#[derive(Debug, Clone)]
pub struct SacRlhfConfig {
    /// Entropy temperature α ≥ 0. Weighs the entropy bonus against reward. When
    /// [`SacRlhfConfig::auto_tune`] is set this is the *current* temperature.
    pub alpha: f32,
    /// Target entropy H̄ (in nats), e.g. a fraction of `log|vocab|`. Used by the
    /// temperature loss / update. Must be finite.
    pub target_entropy: f32,
    /// Learning rate for the multiplicative temperature update (≥ 0). `0.0`
    /// freezes α.
    pub alpha_lr: f32,
    /// Whether [`sac_update_temperature`] adjusts α toward the target entropy.
    pub auto_tune: bool,
}

impl SacRlhfConfig {
    fn validate(&self) -> RlhfResult<()> {
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(RlhfError::InvalidLambda { lambda: self.alpha });
        }
        if !self.target_entropy.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        if !self.alpha_lr.is_finite() || self.alpha_lr < 0.0 {
            return Err(RlhfError::InvalidLambda {
                lambda: self.alpha_lr,
            });
        }
        Ok(())
    }
}

// ── Soft target ─────────────────────────────────────────────────────────────

/// Entropy-augmented soft target `y_i = reward_i − α · log π(a_i)` per sample.
///
/// Since `log π ≤ 0`, the `−α log π` term is non-negative and bonuses uncertain
/// (high-entropy) actions. `rewards` and `log_probs` must have equal length.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`], [`RlhfError::DimensionMismatch`],
/// config errors, and [`RlhfError::NanEncountered`] for NaN inputs.
pub fn sac_soft_target(
    rewards: &[f32],
    log_probs: &[f32],
    cfg: &SacRlhfConfig,
) -> RlhfResult<Vec<f32>> {
    cfg.validate()?;
    if rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if log_probs.len() != rewards.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: rewards.len(),
            got: log_probs.len(),
        });
    }
    let mut out = Vec::with_capacity(rewards.len());
    for (&r, &lp) in rewards.iter().zip(log_probs.iter()) {
        if r.is_nan() || lp.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        out.push(r - cfg.alpha * lp);
    }
    Ok(out)
}

// ── Soft critic (value) loss ────────────────────────────────────────────────

/// Mean soft-critic MSE: `mean_i 0.5 · (value_i − soft_target_i)²`.
///
/// The soft target is computed internally from `rewards` / `log_probs` via
/// [`sac_soft_target`]; `values` is the critic's current estimate per sample.
///
/// # Errors
///
/// Same validation as [`sac_soft_target`], plus a length check on `values`.
pub fn sac_value_loss(
    values: &[f32],
    rewards: &[f32],
    log_probs: &[f32],
    cfg: &SacRlhfConfig,
) -> RlhfResult<f32> {
    let targets = sac_soft_target(rewards, log_probs, cfg)?;
    if values.len() != targets.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: targets.len(),
            got: values.len(),
        });
    }
    let mut acc = 0.0_f32;
    for (&v, &y) in values.iter().zip(targets.iter()) {
        if v.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        let d = v - y;
        acc += 0.5 * d * d;
    }
    Ok(acc / values.len() as f32)
}

// ── Soft policy loss ────────────────────────────────────────────────────────

/// Mean soft-policy loss `mean_i (α · log π(a_i) − q_i)`.
///
/// `q_values` is the critic's action-value estimate `Q̂(a_i)` per sample. A
/// lower loss corresponds to a policy that places mass on high-value,
/// high-entropy actions.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`], [`RlhfError::DimensionMismatch`],
/// config errors, and [`RlhfError::NanEncountered`].
pub fn sac_policy_loss(
    log_probs: &[f32],
    q_values: &[f32],
    cfg: &SacRlhfConfig,
) -> RlhfResult<f32> {
    cfg.validate()?;
    if log_probs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if q_values.len() != log_probs.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: log_probs.len(),
            got: q_values.len(),
        });
    }
    let mut acc = 0.0_f32;
    for (&lp, &q) in log_probs.iter().zip(q_values.iter()) {
        if lp.is_nan() || q.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        acc += cfg.alpha * lp - q;
    }
    let loss = acc / log_probs.len() as f32;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

// ── Temperature loss + update ───────────────────────────────────────────────

/// Mean temperature loss `mean_i (−α · (log π(a_i) + H̄))`.
///
/// Minimising over α drives the realised per-sample entropy `−log π` toward the
/// target `H̄`: where entropy exceeds the target the gradient lowers α, and
/// where it falls short the gradient raises α.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`], config errors, and
/// [`RlhfError::NanEncountered`].
pub fn sac_temperature_loss(log_probs: &[f32], cfg: &SacRlhfConfig) -> RlhfResult<f32> {
    cfg.validate()?;
    if log_probs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let mut acc = 0.0_f32;
    for &lp in log_probs {
        if lp.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        acc += -cfg.alpha * (lp + cfg.target_entropy);
    }
    Ok(acc / log_probs.len() as f32)
}

/// Update the temperature α toward the target entropy and return the new value.
///
/// Uses a multiplicative gradient step on the temperature loss in log-space:
/// `α' = α · exp(alpha_lr · (H̄ − Ĥ))` where `Ĥ = mean_i (−log π(a_i))` is the
/// realised mean entropy. When `Ĥ > H̄` (too random) α decreases; when
/// `Ĥ < H̄` (too greedy) α increases. With `auto_tune = false` or
/// `alpha_lr = 0` the temperature is returned unchanged. The result is clamped
/// to be non-negative.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`], config errors, and
/// [`RlhfError::NanEncountered`].
pub fn sac_update_temperature(log_probs: &[f32], cfg: &SacRlhfConfig) -> RlhfResult<f32> {
    cfg.validate()?;
    if log_probs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.auto_tune || cfg.alpha_lr == 0.0 {
        return Ok(cfg.alpha);
    }
    let mut entropy = 0.0_f32;
    for &lp in log_probs {
        if lp.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        entropy += -lp;
    }
    entropy /= log_probs.len() as f32;
    let new_alpha = cfg.alpha * (cfg.alpha_lr * (cfg.target_entropy - entropy)).exp();
    if new_alpha.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(new_alpha.max(0.0))
}

// ── Gradients ───────────────────────────────────────────────────────────────

/// Gradient of [`sac_policy_loss`] w.r.t. its inputs.
///
/// Finite-difference verified against [`sac_policy_loss`].
#[derive(Debug, Clone)]
pub struct SacPolicyGrad {
    /// `∂L/∂log π(a_i)` = `α / N` for each sample.
    pub d_log_probs: Vec<f32>,
    /// `∂L/∂Q̂(a_i)` = `−1 / N` for each sample.
    pub d_q_values: Vec<f32>,
}

/// Analytic gradient of [`sac_policy_loss`] (`mean_i (α·log π(a_i) − q_i)`).
///
/// The loss is linear in its inputs, giving `∂L/∂log π = α / N` and
/// `∂L/∂q = −1 / N`. The critic estimate `q` is held as an input.
///
/// # Errors
///
/// Mirrors [`sac_policy_loss`].
pub fn sac_policy_grad(
    log_probs: &[f32],
    q_values: &[f32],
    cfg: &SacRlhfConfig,
) -> RlhfResult<SacPolicyGrad> {
    cfg.validate()?;
    if log_probs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if q_values.len() != log_probs.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: log_probs.len(),
            got: q_values.len(),
        });
    }
    let inv_n = 1.0 / log_probs.len() as f32;
    let mut d_log_probs = Vec::with_capacity(log_probs.len());
    let mut d_q_values = Vec::with_capacity(log_probs.len());
    for (&lp, &q) in log_probs.iter().zip(q_values.iter()) {
        if lp.is_nan() || q.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        d_log_probs.push(cfg.alpha * inv_n);
        d_q_values.push(-inv_n);
    }
    Ok(SacPolicyGrad {
        d_log_probs,
        d_q_values,
    })
}

/// Gradient of [`sac_value_loss`] w.r.t. its inputs.
///
/// Finite-difference verified against [`sac_value_loss`].
#[derive(Debug, Clone)]
pub struct SacValueGrad {
    /// `∂L/∂value_i` = `(v_i − y_i) / N`.
    pub d_values: Vec<f32>,
    /// `∂L/∂log π(a_i)` = `α·(v_i − y_i) / N` (through the soft target `y`).
    pub d_log_probs: Vec<f32>,
}

/// Analytic gradient of [`sac_value_loss`] (`mean_i 0.5·(v_i − y_i)²`,
/// `y_i = r_i − α·log π(a_i)`).
///
/// `∂L/∂v_i = (v_i − y_i) / N`; the soft target depends on the log-prob through
/// `∂y_i/∂log π = −α`, giving `∂L/∂log π_i = α·(v_i − y_i) / N`. The rewards are
/// held as inputs.
///
/// # Errors
///
/// Same validation as [`sac_soft_target`], plus a length check on `values`.
pub fn sac_value_grad(
    values: &[f32],
    rewards: &[f32],
    log_probs: &[f32],
    cfg: &SacRlhfConfig,
) -> RlhfResult<SacValueGrad> {
    let targets = sac_soft_target(rewards, log_probs, cfg)?;
    if values.len() != targets.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: targets.len(),
            got: values.len(),
        });
    }
    let inv_n = 1.0 / values.len() as f32;
    let mut d_values = Vec::with_capacity(values.len());
    let mut d_log_probs = Vec::with_capacity(values.len());
    for (&v, &y) in values.iter().zip(targets.iter()) {
        if v.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        let diff = v - y;
        d_values.push(diff * inv_n);
        d_log_probs.push(cfg.alpha * diff * inv_n);
    }
    Ok(SacValueGrad {
        d_values,
        d_log_probs,
    })
}

/// Analytic gradient of [`sac_temperature_loss`] w.r.t. the per-sample log-probs.
///
/// The loss `mean_i (−α·(log π(a_i) + H̄))` is linear in the log-probs, so
/// `∂L/∂log π_i = −α / N`. (α is the optimisation variable of the temperature
/// objective; this gradient is w.r.t. the log-probs that the entropy term sees.)
/// Finite-difference verified against [`sac_temperature_loss`].
///
/// # Errors
///
/// Mirrors [`sac_temperature_loss`].
pub fn sac_temperature_grad(log_probs: &[f32], cfg: &SacRlhfConfig) -> RlhfResult<Vec<f32>> {
    cfg.validate()?;
    if log_probs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let inv_n = 1.0 / log_probs.len() as f32;
    let mut grads = Vec::with_capacity(log_probs.len());
    for &lp in log_probs {
        if lp.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        grads.push(-cfg.alpha * inv_n);
    }
    Ok(grads)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(alpha: f32) -> SacRlhfConfig {
        SacRlhfConfig {
            alpha,
            target_entropy: 1.0,
            alpha_lr: 0.1,
            auto_tune: true,
        }
    }

    // 1. Soft target adds the entropy bonus (−α log π ≥ 0).
    #[test]
    fn soft_target_adds_entropy_bonus() {
        let rewards = [1.0_f32, 2.0];
        let log_probs = [-2.0_f32, -0.5];
        let y = sac_soft_target(&rewards, &log_probs, &cfg(0.5)).expect("target");
        // y0 = 1 - 0.5*(-2) = 2.0 ; y1 = 2 - 0.5*(-0.5) = 2.25
        assert!((y[0] - 2.0).abs() < 1e-6);
        assert!((y[1] - 2.25).abs() < 1e-6);
        // Bonus always non-negative: y >= reward.
        assert!(y[0] >= rewards[0] && y[1] >= rewards[1]);
    }

    // 2. alpha = 0 → soft target equals raw reward.
    #[test]
    fn alpha_zero_target_is_reward() {
        let rewards = [1.0_f32, -3.0];
        let log_probs = [-2.0_f32, -0.5];
        let y = sac_soft_target(&rewards, &log_probs, &cfg(0.0)).expect("target");
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[1] - (-3.0)).abs() < 1e-6);
    }

    // 3. Value loss is zero when the critic matches the soft target.
    #[test]
    fn value_loss_zero_when_matched() {
        let rewards = [1.0_f32, 2.0];
        let log_probs = [-2.0_f32, -0.5];
        let c = cfg(0.5);
        let targets = sac_soft_target(&rewards, &log_probs, &c).expect("targets");
        let loss = sac_value_loss(&targets, &rewards, &log_probs, &c).expect("vloss");
        assert!(loss.abs() < 1e-6, "matched critic → 0 loss, got {loss}");
    }

    // 4. Value loss grows with critic error.
    #[test]
    fn value_loss_grows_with_error() {
        let rewards = [1.0_f32];
        let log_probs = [-1.0_f32];
        let c = cfg(0.5);
        let targets = sac_soft_target(&rewards, &log_probs, &c).expect("targets");
        let near = sac_value_loss(&[targets[0] + 0.1], &rewards, &log_probs, &c).expect("near");
        let far = sac_value_loss(&[targets[0] + 1.0], &rewards, &log_probs, &c).expect("far");
        assert!(far > near, "larger error → larger loss");
        // 0.5 * 0.1^2 = 0.005
        assert!((near - 0.005).abs() < 1e-6);
    }

    // 5. Policy loss = α·logπ − Q, mean over samples.
    #[test]
    fn policy_loss_formula() {
        let log_probs = [-2.0_f32, -1.0];
        let q = [0.5_f32, 1.5];
        let loss = sac_policy_loss(&log_probs, &q, &cfg(0.5)).expect("ploss");
        // term0 = 0.5*(-2)-0.5 = -1.5 ; term1 = 0.5*(-1)-1.5 = -2.0 ; mean = -1.75
        assert!((loss - (-1.75)).abs() < 1e-6, "loss {loss}");
    }

    // 6. Higher Q lowers the policy loss (policy prefers high value).
    #[test]
    fn higher_q_lowers_policy_loss() {
        let log_probs = [-1.0_f32];
        let lo = sac_policy_loss(&log_probs, &[0.0], &cfg(0.5)).expect("lo");
        let hi = sac_policy_loss(&log_probs, &[5.0], &cfg(0.5)).expect("hi");
        assert!(hi < lo, "higher Q → lower loss");
    }

    // 7. Temperature update: too-random policy (entropy > target) lowers α.
    #[test]
    fn high_entropy_lowers_alpha() {
        // mean entropy = 3.0 > target 1.0 → α should decrease.
        let log_probs = [-3.0_f32, -3.0];
        let new_alpha = sac_update_temperature(&log_probs, &cfg(1.0)).expect("update");
        assert!(
            new_alpha < 1.0,
            "entropy>target should lower α, got {new_alpha}"
        );
    }

    // 8. Temperature update: too-greedy policy (entropy < target) raises α.
    #[test]
    fn low_entropy_raises_alpha() {
        // mean entropy = 0.2 < target 1.0 → α should increase.
        let log_probs = [-0.2_f32, -0.2];
        let new_alpha = sac_update_temperature(&log_probs, &cfg(1.0)).expect("update");
        assert!(
            new_alpha > 1.0,
            "entropy<target should raise α, got {new_alpha}"
        );
    }

    // 9. Update is a fixed point when realised entropy equals the target.
    #[test]
    fn alpha_stationary_at_target_entropy() {
        // mean entropy = 1.0 = target → α unchanged.
        let log_probs = [-1.0_f32, -1.0];
        let new_alpha = sac_update_temperature(&log_probs, &cfg(0.7)).expect("update");
        assert!(
            (new_alpha - 0.7).abs() < 1e-6,
            "α should be stationary, got {new_alpha}"
        );
    }

    // 10. auto_tune = false freezes α.
    #[test]
    fn auto_tune_off_freezes_alpha() {
        let mut c = cfg(0.9);
        c.auto_tune = false;
        let log_probs = [-3.0_f32]; // would otherwise change α
        let new_alpha = sac_update_temperature(&log_probs, &c).expect("update");
        assert!((new_alpha - 0.9).abs() < 1e-9);
    }

    // 11. Temperature loss minimised in α only when entropy matches target.
    #[test]
    fn temperature_loss_sign() {
        // entropy 3 > target 1: loss = -α(logπ + H̄) = -1*(-3+1)=2 > 0 (raising α hurts → lowers it).
        let high = sac_temperature_loss(&[-3.0], &cfg(1.0)).expect("high");
        assert!(high > 0.0, "loss {high}");
        // entropy 0.2 < target 1: -1*(-0.2+1) = -0.8 < 0 (raising α helps).
        let low = sac_temperature_loss(&[-0.2], &cfg(1.0)).expect("low");
        assert!(low < 0.0, "loss {low}");
    }

    // 12. Negative alpha rejected.
    #[test]
    fn negative_alpha_errors() {
        let log_probs = [-1.0_f32];
        assert!(matches!(
            sac_policy_loss(&log_probs, &[1.0], &cfg(-0.1)),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }

    // 13. Length mismatches rejected.
    #[test]
    fn length_mismatch_errors() {
        assert!(matches!(
            sac_soft_target(&[1.0, 2.0], &[-1.0], &cfg(0.5)),
            Err(RlhfError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            sac_policy_loss(&[-1.0, -2.0], &[1.0], &cfg(0.5)),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    // 14. NaN inputs rejected.
    #[test]
    fn nan_inputs_error() {
        assert!(matches!(
            sac_soft_target(&[f32::NAN], &[-1.0], &cfg(0.5)),
            Err(RlhfError::NanEncountered)
        ));
        assert!(matches!(
            sac_policy_loss(&[f32::NAN], &[1.0], &cfg(0.5)),
            Err(RlhfError::NanEncountered)
        ));
    }

    // 15. Empty inputs rejected.
    #[test]
    fn empty_inputs_error() {
        assert!(matches!(
            sac_soft_target(&[], &[], &cfg(0.5)),
            Err(RlhfError::EmptyInput)
        ));
        assert!(matches!(
            sac_temperature_loss(&[], &cfg(0.5)),
            Err(RlhfError::EmptyInput)
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

    fn cfg(alpha: f32) -> SacRlhfConfig {
        SacRlhfConfig {
            alpha,
            target_entropy: 1.0,
            alpha_lr: 0.1,
            auto_tune: true,
        }
    }

    #[test]
    fn sac_policy_grad_matches_fd() {
        let lp = [-2.0_f32, -1.0, -0.5];
        let q = [0.5_f32, 1.5, -0.3];
        let c = cfg(0.4);
        let g = sac_policy_grad(&lp, &q, &c).expect("grad");
        let h = 1e-2;
        for i in 0..lp.len() {
            let fd_lp = central_diff(
                |v| {
                    let mut l = lp;
                    l[i] = v;
                    sac_policy_loss(&l, &q, &c).expect("loss")
                },
                lp[i],
                h,
            );
            let fd_q = central_diff(
                |v| {
                    let mut qq = q;
                    qq[i] = v;
                    sac_policy_loss(&lp, &qq, &c).expect("loss")
                },
                q[i],
                h,
            );
            assert_close(g.d_log_probs[i], fd_lp, "policy d_logp");
            assert_close(g.d_q_values[i], fd_q, "policy d_q");
        }
    }

    #[test]
    fn sac_value_grad_matches_fd() {
        let values = [1.0_f32, 2.0];
        let rewards = [1.0_f32, 2.0];
        let lp = [-2.0_f32, -0.5];
        let c = cfg(0.5);
        let g = sac_value_grad(&values, &rewards, &lp, &c).expect("grad");
        let h = 1e-2;
        for i in 0..values.len() {
            let fd_v = central_diff(
                |v| {
                    let mut vv = values;
                    vv[i] = v;
                    sac_value_loss(&vv, &rewards, &lp, &c).expect("loss")
                },
                values[i],
                h,
            );
            let fd_lp = central_diff(
                |v| {
                    let mut l = lp;
                    l[i] = v;
                    sac_value_loss(&values, &rewards, &l, &c).expect("loss")
                },
                lp[i],
                h,
            );
            assert_close(g.d_values[i], fd_v, "value d_value");
            assert_close(g.d_log_probs[i], fd_lp, "value d_logp");
        }
    }

    #[test]
    fn sac_temperature_grad_matches_fd() {
        let lp = [-2.0_f32, -0.3];
        let c = cfg(0.7);
        let g = sac_temperature_grad(&lp, &c).expect("grad");
        let h = 1e-2;
        for i in 0..lp.len() {
            let fd = central_diff(
                |v| {
                    let mut l = lp;
                    l[i] = v;
                    sac_temperature_loss(&l, &c).expect("loss")
                },
                lp[i],
                h,
            );
            assert_close(g[i], fd, "temperature d_logp");
            // Closed form: −α / N.
            assert_close(g[i], -c.alpha / lp.len() as f32, "temp closed form");
        }
    }

    #[test]
    fn sac_grad_length_mismatch_errors() {
        assert!(matches!(
            sac_policy_grad(&[-1.0, -2.0], &[1.0], &cfg(0.5)),
            Err(RlhfError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            sac_value_grad(&[1.0], &[1.0, 2.0], &[-1.0, -2.0], &cfg(0.5)),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn sac_grad_negative_alpha_errors() {
        assert!(matches!(
            sac_policy_grad(&[-1.0], &[1.0], &cfg(-0.1)),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }
}

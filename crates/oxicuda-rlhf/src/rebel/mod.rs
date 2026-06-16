//! REBEL — REgression to RElative REward Based RL (Gao et al. 2024).
//!
//! Reference: Gao, Z., Chang, J. D., Zhan, W., Oertell, O., Swamy, G., Brantley, K.,
//! Joachims, T., Bagnell, J. A., Lee, J. D., & Sun, W. (2024). *REBEL: Reinforcement
//! Learning via Regressing Relative Rewards*. NeurIPS 2024.
//! <https://arxiv.org/abs/2404.16767>
//!
//! REBEL replaces the policy-gradient / clipped-surrogate machinery of PPO with a
//! single **least-squares regression**. Given a pair of responses `(y, y′)` to the
//! same prompt, it regresses the *predicted* relative reward — a scaled difference of
//! log-likelihood ratios against the behaviour policy `π_old` — onto the *observed*
//! relative reward:
//!
//! ```text
//!   predicted Δ(y, y′) = (1/η) · [ (logπ_θ(y) − logπ_old(y))
//!                                − (logπ_θ(y′) − logπ_old(y′)) ]
//!   target    Δ(y, y′) = r(y) − r(y′)
//!   L = mean_pairs ( predicted Δ − target Δ )²
//! ```
//!
//! The step-size-like temperature `η > 0` controls how aggressively the policy moves;
//! large `η` ⇒ conservative updates. Because the objective is a plain MSE there is no
//! importance-sampling variance and no clipping heuristic, and at the minimiser the
//! induced policy provably matches a mirror-descent step on the reward.
//!
//! All routines operate on log-probabilities and rewards in pure CPU code.

use crate::error::{RlhfError, RlhfResult};

/// Hyper-parameters for REBEL.
#[derive(Debug, Clone)]
pub struct RebelConfig {
    /// Inverse step size / temperature `η > 0` scaling the predicted reward diff.
    pub eta: f32,
}

impl RebelConfig {
    fn validate(&self) -> RlhfResult<()> {
        if !self.eta.is_finite() || self.eta <= 0.0 {
            return Err(RlhfError::InvalidBeta { beta: self.eta });
        }
        Ok(())
    }
}

/// Predicted relative reward `(1/η)·[(logp_a − old_a) − (logp_b − old_b)]` for one pair.
#[inline]
#[must_use]
pub fn predicted_relative_reward(
    logp_a: f32,
    old_logp_a: f32,
    logp_b: f32,
    old_logp_b: f32,
    eta: f32,
) -> f32 {
    let ratio_a = logp_a - old_logp_a;
    let ratio_b = logp_b - old_logp_b;
    (ratio_a - ratio_b) / eta
}

/// A REBEL training pair: two responses to one prompt with their rewards and the
/// current / behaviour-policy sequence log-probs.
#[derive(Debug, Clone)]
pub struct RebelPair {
    /// Reward of the first response `y`.
    pub reward_a: f32,
    /// Reward of the second response `y′`.
    pub reward_b: f32,
    /// Current-policy log-prob of `y` (sequence sum).
    pub logp_a: f32,
    /// Current-policy log-prob of `y′`.
    pub logp_b: f32,
    /// Behaviour-policy log-prob of `y`.
    pub old_logp_a: f32,
    /// Behaviour-policy log-prob of `y′`.
    pub old_logp_b: f32,
}

/// Per-pair REBEL squared error.
///
/// # Errors
///
/// - [`RlhfError::InvalidBeta`] if `eta ≤ 0` or non-finite.
/// - [`RlhfError::NanEncountered`] if the result is non-finite.
pub fn rebel_pair_loss(pair: &RebelPair, cfg: &RebelConfig) -> RlhfResult<f32> {
    cfg.validate()?;
    let predicted = predicted_relative_reward(
        pair.logp_a,
        pair.old_logp_a,
        pair.logp_b,
        pair.old_logp_b,
        cfg.eta,
    );
    let target = pair.reward_a - pair.reward_b;
    let diff = predicted - target;
    let loss = diff * diff;
    if !loss.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// Mean REBEL loss over a batch of pairs.
///
/// # Errors
///
/// - [`RlhfError::EmptyInput`] if `pairs` is empty.
/// - Propagates errors from [`rebel_pair_loss`].
pub fn rebel_loss(pairs: &[RebelPair], cfg: &RebelConfig) -> RlhfResult<f32> {
    cfg.validate()?;
    if pairs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let mut total = 0.0_f32;
    for p in pairs {
        total += rebel_pair_loss(p, cfg)?;
    }
    let loss = total / pairs.len() as f32;
    if !loss.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// Batched REBEL loss from flat slices (one entry per pair). All slices must have the
/// same length `n`.
///
/// # Errors
///
/// - [`RlhfError::EmptyInput`] if `reward_a` is empty.
/// - [`RlhfError::DimensionMismatch`] if any slice length differs from `reward_a`.
/// - Propagates errors from [`rebel_pair_loss`].
#[allow(clippy::too_many_arguments)]
pub fn rebel_loss_slices(
    reward_a: &[f32],
    reward_b: &[f32],
    logp_a: &[f32],
    logp_b: &[f32],
    old_logp_a: &[f32],
    old_logp_b: &[f32],
    cfg: &RebelConfig,
) -> RlhfResult<f32> {
    cfg.validate()?;
    let n = reward_a.len();
    if n == 0 {
        return Err(RlhfError::EmptyInput);
    }
    for got in [
        reward_b.len(),
        logp_a.len(),
        logp_b.len(),
        old_logp_a.len(),
        old_logp_b.len(),
    ] {
        if got != n {
            return Err(RlhfError::DimensionMismatch { expected: n, got });
        }
    }
    let mut total = 0.0_f32;
    for i in 0..n {
        let pair = RebelPair {
            reward_a: reward_a[i],
            reward_b: reward_b[i],
            logp_a: logp_a[i],
            logp_b: logp_b[i],
            old_logp_a: old_logp_a[i],
            old_logp_b: old_logp_b[i],
        };
        total += rebel_pair_loss(&pair, cfg)?;
    }
    let loss = total / n as f32;
    if !loss.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(eta: f32) -> RebelConfig {
        RebelConfig { eta }
    }

    #[test]
    fn predicted_zero_when_no_policy_change() {
        // logp == old_logp for both ⇒ predicted relative reward = 0.
        let p = predicted_relative_reward(-1.0, -1.0, -2.0, -2.0, 0.5);
        assert!(p.abs() < 1e-6, "predicted should be 0, got {p}");
    }

    #[test]
    fn predicted_scales_inversely_with_eta() {
        let p1 = predicted_relative_reward(-0.5, -1.0, -2.0, -1.5, 1.0);
        let p2 = predicted_relative_reward(-0.5, -1.0, -2.0, -1.5, 2.0);
        assert!(
            (p1 - 2.0 * p2).abs() < 1e-5,
            "doubling η must halve prediction"
        );
    }

    #[test]
    fn pair_loss_zero_when_perfectly_regressed() {
        // Construct a pair whose predicted reward diff exactly equals the target.
        // ratio_a - ratio_b = (logp_a-old_a)-(logp_b-old_b).
        // Pick logps so (1/η)*(ratio_diff) == reward_a-reward_b.
        let eta = 0.5_f32;
        // ratio_a = 0.0, ratio_b = -1.0 ⇒ ratio_diff = 1.0 ⇒ predicted = 1/0.5 = 2.0
        let pair = RebelPair {
            reward_a: 3.0,
            reward_b: 1.0, // target diff = 2.0
            logp_a: -1.0,
            old_logp_a: -1.0, // ratio_a = 0
            logp_b: -2.0,
            old_logp_b: -1.0, // ratio_b = -1
        };
        let loss = rebel_pair_loss(&pair, &cfg(eta)).expect("value should be present");
        assert!(
            loss < 1e-6,
            "loss must vanish at perfect regression, got {loss}"
        );
    }

    #[test]
    fn pair_loss_positive_on_mismatch() {
        let pair = RebelPair {
            reward_a: 5.0,
            reward_b: 0.0,
            logp_a: -1.0,
            old_logp_a: -1.0,
            logp_b: -1.0,
            old_logp_b: -1.0, // predicted = 0, target = 5 ⇒ loss = 25
        };
        let loss = rebel_pair_loss(&pair, &cfg(1.0)).expect("value should be present");
        assert!((loss - 25.0).abs() < 1e-4, "loss should be 25, got {loss}");
    }

    #[test]
    fn pair_loss_is_squared() {
        // Doubling the residual should quadruple the loss.
        let make = |target: f32| RebelPair {
            reward_a: target,
            reward_b: 0.0,
            logp_a: -1.0,
            old_logp_a: -1.0,
            logp_b: -1.0,
            old_logp_b: -1.0, // predicted = 0
        };
        let l1 = rebel_pair_loss(&make(1.0), &cfg(1.0)).expect("value should be present");
        let l2 = rebel_pair_loss(&make(2.0), &cfg(1.0)).expect("value should be present");
        assert!(
            (l2 - 4.0 * l1).abs() < 1e-4,
            "MSE must be quadratic: {l1} {l2}"
        );
    }

    #[test]
    fn batch_loss_finite() {
        let pairs = vec![
            RebelPair {
                reward_a: 1.0,
                reward_b: 0.5,
                logp_a: -0.5,
                logp_b: -1.0,
                old_logp_a: -0.7,
                old_logp_b: -1.1,
            },
            RebelPair {
                reward_a: 2.0,
                reward_b: 1.0,
                logp_a: -0.3,
                logp_b: -0.9,
                old_logp_a: -0.4,
                old_logp_b: -1.0,
            },
        ];
        let loss = rebel_loss(&pairs, &cfg(0.1)).expect("value should be present");
        assert!(loss.is_finite() && loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn batch_loss_is_mean_of_pairs() {
        let p1 = RebelPair {
            reward_a: 3.0,
            reward_b: 0.0,
            logp_a: -1.0,
            old_logp_a: -1.0,
            logp_b: -1.0,
            old_logp_b: -1.0,
        }; // loss 9 at eta=1
        let p2 = RebelPair {
            reward_a: 1.0,
            reward_b: 0.0,
            logp_a: -1.0,
            old_logp_a: -1.0,
            logp_b: -1.0,
            old_logp_b: -1.0,
        }; // loss 1
        let mean = rebel_loss(&[p1, p2], &cfg(1.0)).expect("value should be present");
        assert!(
            (mean - 5.0).abs() < 1e-4,
            "mean of {{9,1}} must be 5, got {mean}"
        );
    }

    #[test]
    fn empty_batch_errors() {
        assert!(matches!(
            rebel_loss(&[], &cfg(1.0)),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn invalid_eta_errors() {
        let pair = RebelPair {
            reward_a: 1.0,
            reward_b: 0.0,
            logp_a: -1.0,
            logp_b: -1.0,
            old_logp_a: -1.0,
            old_logp_b: -1.0,
        };
        assert!(matches!(
            rebel_pair_loss(&pair, &cfg(0.0)),
            Err(RlhfError::InvalidBeta { .. })
        ));
        assert!(matches!(
            rebel_pair_loss(&pair, &cfg(-1.0)),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }

    #[test]
    fn slices_loss_matches_pair_loss() {
        let loss_slices = rebel_loss_slices(
            &[3.0, 1.0],
            &[0.0, 0.0],
            &[-1.0, -1.0],
            &[-1.0, -1.0],
            &[-1.0, -1.0],
            &[-1.0, -1.0],
            &cfg(1.0),
        )
        .expect("value should be present");
        assert!(
            (loss_slices - 5.0).abs() < 1e-4,
            "slice loss must equal pair mean"
        );
    }

    #[test]
    fn slices_dim_mismatch_errors() {
        let r = rebel_loss_slices(
            &[1.0, 2.0],
            &[0.0],
            &[-1.0, -1.0],
            &[-1.0, -1.0],
            &[-1.0, -1.0],
            &[-1.0, -1.0],
            &cfg(1.0),
        );
        assert!(matches!(r, Err(RlhfError::DimensionMismatch { .. })));
    }

    #[test]
    fn slices_empty_errors() {
        let r = rebel_loss_slices(&[], &[], &[], &[], &[], &[], &cfg(1.0));
        assert!(matches!(r, Err(RlhfError::EmptyInput)));
    }
}

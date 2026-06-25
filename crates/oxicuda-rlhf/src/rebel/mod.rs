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

/// Gradient of the per-pair REBEL squared error w.r.t. the current log-probs.
///
/// Finite-difference verified against [`rebel_pair_loss`].
#[derive(Debug, Clone, Copy)]
pub struct RebelGrad {
    /// `∂L/∂(logp_a)` (current-policy log-prob of `y`).
    pub d_logp_a: f32,
    /// `∂L/∂(logp_b)` (current-policy log-prob of `y′`).
    pub d_logp_b: f32,
}

#[inline]
fn rebel_pair_grad_inner(pair: &RebelPair, eta: f32) -> RebelGrad {
    let predicted = predicted_relative_reward(
        pair.logp_a,
        pair.old_logp_a,
        pair.logp_b,
        pair.old_logp_b,
        eta,
    );
    let target = pair.reward_a - pair.reward_b;
    let residual = predicted - target;
    // L = residual²; ∂predicted/∂logp_a = 1/η, ∂predicted/∂logp_b = −1/η.
    let common = 2.0 * residual / eta;
    RebelGrad {
        d_logp_a: common,
        d_logp_b: -common,
    }
}

/// Analytic gradient of [`rebel_pair_loss`].
///
/// With `L = (predicted − target)²`,
/// `predicted = (1/η)·[(logp_a − old_a) − (logp_b − old_b)]`, and `target` held
/// constant (the observed reward difference), the chain rule gives
/// `∂L/∂logp_a = 2·residual/η` and `∂L/∂logp_b = −2·residual/η`, where
/// `residual = predicted − target`. The rewards and behaviour log-probs are
/// held as inputs.
///
/// # Errors
///
/// - [`RlhfError::InvalidBeta`] if `eta ≤ 0` or non-finite.
/// - [`RlhfError::NanEncountered`] if a gradient is non-finite.
pub fn rebel_pair_grad(pair: &RebelPair, cfg: &RebelConfig) -> RlhfResult<RebelGrad> {
    cfg.validate()?;
    let grad = rebel_pair_grad_inner(pair, cfg.eta);
    if !grad.d_logp_a.is_finite() || !grad.d_logp_b.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(grad)
}

/// Analytic gradient of the mean-reduced [`rebel_loss`].
///
/// Returns one [`RebelGrad`] per pair, each scaled by `1 / pairs.len()`.
///
/// # Errors
///
/// - [`RlhfError::EmptyInput`] if `pairs` is empty.
/// - Propagates errors from [`rebel_pair_grad`].
pub fn rebel_grad(pairs: &[RebelPair], cfg: &RebelConfig) -> RlhfResult<Vec<RebelGrad>> {
    cfg.validate()?;
    if pairs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let inv_n = 1.0 / pairs.len() as f32;
    let mut grads = Vec::with_capacity(pairs.len());
    for p in pairs {
        let g = rebel_pair_grad_inner(p, cfg.eta);
        let scaled = RebelGrad {
            d_logp_a: g.d_logp_a * inv_n,
            d_logp_b: g.d_logp_b * inv_n,
        };
        if !scaled.d_logp_a.is_finite() || !scaled.d_logp_b.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        grads.push(scaled);
    }
    Ok(grads)
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

    fn cfg(eta: f32) -> RebelConfig {
        RebelConfig { eta }
    }

    fn mk(ra: f32, rb: f32, la: f32, lb: f32, oa: f32, ob: f32) -> RebelPair {
        RebelPair {
            reward_a: ra,
            reward_b: rb,
            logp_a: la,
            logp_b: lb,
            old_logp_a: oa,
            old_logp_b: ob,
        }
    }

    #[test]
    fn rebel_pair_grad_matches_fd() {
        let eta = 0.5_f32;
        let p = mk(3.0, 1.0, -0.5, -1.2, -0.7, -1.0);
        let g = rebel_pair_grad(&p, &cfg(eta)).expect("grad");
        let h = 1e-2;
        let fd_a = central_diff(
            |v| {
                let mut q = p.clone();
                q.logp_a = v;
                rebel_pair_loss(&q, &cfg(eta)).expect("l")
            },
            p.logp_a,
            h,
        );
        let fd_b = central_diff(
            |v| {
                let mut q = p.clone();
                q.logp_b = v;
                rebel_pair_loss(&q, &cfg(eta)).expect("l")
            },
            p.logp_b,
            h,
        );
        assert_close(g.d_logp_a, fd_a, "d_logp_a");
        assert_close(g.d_logp_b, fd_b, "d_logp_b");
    }

    #[test]
    fn rebel_grad_zero_at_perfect_regression() {
        // predicted exactly equals target → residual 0 → zero gradient.
        let eta = 0.5_f32;
        let p = mk(3.0, 1.0, -1.0, -2.0, -1.0, -1.0); // predicted = 2 = target
        let g = rebel_pair_grad(&p, &cfg(eta)).expect("grad");
        assert!(g.d_logp_a.abs() < 1e-5, "{}", g.d_logp_a);
        assert!(g.d_logp_b.abs() < 1e-5, "{}", g.d_logp_b);
    }

    #[test]
    fn rebel_grad_batch_matches_fd() {
        let eta = 0.3_f32;
        let pairs = vec![
            mk(1.0, 0.5, -0.5, -1.0, -0.7, -1.1),
            mk(2.0, 1.0, -0.3, -0.9, -0.4, -1.0),
        ];
        let grads = rebel_grad(&pairs, &cfg(eta)).expect("grads");
        let h = 1e-2;
        let fd = central_diff(
            |v| {
                let mut ps = pairs.clone();
                ps[1].logp_a = v;
                rebel_loss(&ps, &cfg(eta)).expect("loss")
            },
            pairs[1].logp_a,
            h,
        );
        assert_close(grads[1].d_logp_a, fd, "batch d_logp_a[1]");
    }

    #[test]
    fn rebel_grad_invalid_eta_errors() {
        let p = mk(1.0, 0.0, -1.0, -1.0, -1.0, -1.0);
        assert!(matches!(
            rebel_pair_grad(&p, &cfg(0.0)),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }
}

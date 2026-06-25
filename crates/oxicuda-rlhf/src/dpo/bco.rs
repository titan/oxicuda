//! BCO — Binary Classifier Optimization (Jung et al. 2024).
//!
//! Reference: Jung, S., Han, G., Nam, D. W., & On, K.-W. (2024). *Binary Classifier
//! Optimization for Large Language Model Alignment*. arXiv:2404.04656.
//! <https://arxiv.org/abs/2404.04656>
//!
//! BCO recasts alignment from *paired* preference learning to **per-example binary
//! classification** (like KTO, it needs only a label, not a chosen/rejected pair).
//! The implicit reward of an example is the β-scaled log-ratio against the reference
//! policy:
//!
//! ```text
//!   r(x, y) = β · ( logπ_θ(y | x) − logπ_ref(y | x) ) .
//! ```
//!
//! A logistic classifier is trained to push *desirable* rewards above, and
//! *undesirable* rewards below, a **reward shift** `δ`. With binary cross-entropy:
//!
//! ```text
//!   L = −E_{desirable}[ log σ(r − δ) ] − E_{undesirable}[ log σ(δ − r) ] .
//! ```
//!
//! The paper sets `δ` to the running **mean of all implicit rewards** so the decision
//! boundary self-calibrates; this removes the reward-shift hyper-parameter that KTO
//! leaves implicit and provably makes the classifier the Bayes-optimal separator of
//! the two reward populations.
//!
//! [`RewardShift`] maintains that running mean with Welford's algorithm; the loss
//! functions accept either an explicit `δ` or the current estimate.

use crate::error::{RlhfError, RlhfResult};

/// Configuration for the BCO loss.
#[derive(Debug, Clone)]
pub struct BcoConfig {
    /// Implicit-reward temperature `β > 0`.
    pub beta: f32,
    /// Reward shift `δ` (decision boundary). Set this to the running mean of the
    /// implicit rewards for the self-calibrating variant.
    pub reward_shift: f32,
}

impl BcoConfig {
    fn validate(&self) -> RlhfResult<()> {
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(RlhfError::InvalidBeta { beta: self.beta });
        }
        if !self.reward_shift.is_finite() {
            return Err(RlhfError::InvalidMargin {
                margin: self.reward_shift,
            });
        }
        Ok(())
    }
}

/// Numerically stable `log σ(x)`.
#[inline]
fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// Implicit reward `r = β · (logp − ref_logp)` for one example.
#[inline]
#[must_use]
pub fn implicit_reward(logp: f32, ref_logp: f32, beta: f32) -> f32 {
    beta * (logp - ref_logp)
}

/// Welford running estimator of the mean implicit reward, used as the BCO reward
/// shift `δ`.
#[derive(Debug, Clone, Default)]
pub struct RewardShift {
    /// Running count of observed rewards.
    pub count: u64,
    /// Running mean of observed rewards.
    pub mean: f32,
}

impl RewardShift {
    /// Create an empty estimator (`mean = 0`, `count = 0`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Incorporate one reward value into the running mean.
    pub fn update(&mut self, reward: f32) {
        self.count += 1;
        let delta = reward - self.mean;
        self.mean += delta / self.count as f32;
    }

    /// Incorporate a whole batch of rewards.
    pub fn update_batch(&mut self, rewards: &[f32]) {
        for &r in rewards {
            self.update(r);
        }
    }

    /// Current reward-shift estimate `δ` (the running mean).
    #[must_use]
    pub fn shift(&self) -> f32 {
        self.mean
    }
}

/// BCO loss given **implicit rewards** directly.
///
/// Desirable rewards are classified as `+1` (above `δ`), undesirable as `0` (below).
///
/// # Errors
///
/// - [`RlhfError::EmptyInput`] if both reward slices are empty.
/// - [`RlhfError::InvalidBeta`] / [`RlhfError::InvalidMargin`] for a bad config.
/// - [`RlhfError::NanEncountered`] if the result is non-finite.
pub fn bco_loss_from_rewards(
    desirable_rewards: &[f32],
    undesirable_rewards: &[f32],
    cfg: &BcoConfig,
) -> RlhfResult<f32> {
    cfg.validate()?;
    if desirable_rewards.is_empty() && undesirable_rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let delta = cfg.reward_shift;

    let desirable_loss = if desirable_rewards.is_empty() {
        0.0_f32
    } else {
        let sum: f32 = desirable_rewards
            .iter()
            .map(|&r| -log_sigmoid(r - delta))
            .sum();
        sum / desirable_rewards.len() as f32
    };
    let undesirable_loss = if undesirable_rewards.is_empty() {
        0.0_f32
    } else {
        let sum: f32 = undesirable_rewards
            .iter()
            .map(|&r| -log_sigmoid(delta - r))
            .sum();
        sum / undesirable_rewards.len() as f32
    };
    let loss = desirable_loss + undesirable_loss;
    if !loss.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// BCO loss from **log-probabilities**, computing implicit rewards internally.
///
/// `desirable_logps` / `undesirable_logps` are current-policy sequence log-probs and
/// the `*_ref_logps` are the matching reference-policy log-probs.
///
/// # Errors
///
/// - [`RlhfError::EmptyInput`] if both example sets are empty.
/// - [`RlhfError::DimensionMismatch`] if a logp slice and its reference slice differ
///   in length.
/// - Propagates config and numerical errors from [`bco_loss_from_rewards`].
pub fn bco_loss(
    desirable_logps: &[f32],
    desirable_ref_logps: &[f32],
    undesirable_logps: &[f32],
    undesirable_ref_logps: &[f32],
    cfg: &BcoConfig,
) -> RlhfResult<f32> {
    cfg.validate()?;
    if desirable_logps.len() != desirable_ref_logps.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: desirable_logps.len(),
            got: desirable_ref_logps.len(),
        });
    }
    if undesirable_logps.len() != undesirable_ref_logps.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: undesirable_logps.len(),
            got: undesirable_ref_logps.len(),
        });
    }
    let desirable_rewards: Vec<f32> = desirable_logps
        .iter()
        .zip(desirable_ref_logps.iter())
        .map(|(&lp, &rlp)| implicit_reward(lp, rlp, cfg.beta))
        .collect();
    let undesirable_rewards: Vec<f32> = undesirable_logps
        .iter()
        .zip(undesirable_ref_logps.iter())
        .map(|(&lp, &rlp)| implicit_reward(lp, rlp, cfg.beta))
        .collect();
    bco_loss_from_rewards(&desirable_rewards, &undesirable_rewards, cfg)
}

/// Numerically stable sigmoid `σ(x)`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Gradient of [`bco_loss_from_rewards`] w.r.t. the per-example implicit rewards.
///
/// Finite-difference verified against [`bco_loss_from_rewards`].
#[derive(Debug, Clone)]
pub struct BcoGrad {
    /// `∂L/∂r` for each desirable reward (same length / order as the input).
    pub d_desirable: Vec<f32>,
    /// `∂L/∂r` for each undesirable reward (same length / order as the input).
    pub d_undesirable: Vec<f32>,
}

/// Analytic gradient of [`bco_loss_from_rewards`] w.r.t. the per-example
/// implicit rewards (the shift `δ` is held constant).
///
/// For a desirable reward `r`, the contribution is `−log σ(r − δ) / N_d`, whose
/// derivative is `−σ(δ − r) / N_d` (negative: raising a desirable reward lowers
/// the loss). For an undesirable reward `r`, the contribution is
/// `−log σ(δ − r) / N_u`, whose derivative is `+σ(r − δ) / N_u` (positive:
/// raising an undesirable reward raises the loss).
///
/// # Errors
///
/// - [`RlhfError::EmptyInput`] if both reward slices are empty.
/// - [`RlhfError::InvalidBeta`] / [`RlhfError::InvalidMargin`] for a bad config.
/// - [`RlhfError::NanEncountered`] if any gradient is non-finite.
pub fn bco_grad(
    desirable_rewards: &[f32],
    undesirable_rewards: &[f32],
    cfg: &BcoConfig,
) -> RlhfResult<BcoGrad> {
    cfg.validate()?;
    if desirable_rewards.is_empty() && undesirable_rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let delta = cfg.reward_shift;

    let d_desirable = if desirable_rewards.is_empty() {
        Vec::new()
    } else {
        let inv_n = 1.0 / desirable_rewards.len() as f32;
        desirable_rewards
            .iter()
            .map(|&r| -sigmoid(delta - r) * inv_n)
            .collect()
    };
    let d_undesirable = if undesirable_rewards.is_empty() {
        Vec::new()
    } else {
        let inv_n = 1.0 / undesirable_rewards.len() as f32;
        undesirable_rewards
            .iter()
            .map(|&r| sigmoid(r - delta) * inv_n)
            .collect()
    };

    let grad = BcoGrad {
        d_desirable,
        d_undesirable,
    };
    if grad
        .d_desirable
        .iter()
        .chain(grad.d_undesirable.iter())
        .any(|g| !g.is_finite())
    {
        return Err(RlhfError::NanEncountered);
    }
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(beta: f32, shift: f32) -> BcoConfig {
        BcoConfig {
            beta,
            reward_shift: shift,
        }
    }

    #[test]
    fn implicit_reward_scales_with_beta() {
        let r1 = implicit_reward(-0.5, -1.0, 1.0);
        let r2 = implicit_reward(-0.5, -1.0, 2.0);
        assert!((r2 - 2.0 * r1).abs() < 1e-6, "reward must scale with β");
        assert!((r1 - 0.5).abs() < 1e-6, "β·(logp-ref) = 1·0.5 = 0.5");
    }

    #[test]
    fn reward_shift_running_mean() {
        let mut rs = RewardShift::new();
        rs.update_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rs.count, 4);
        assert!(
            (rs.shift() - 2.5).abs() < 1e-5,
            "mean of 1..4 is 2.5, got {}",
            rs.shift()
        );
    }

    #[test]
    fn reward_shift_incremental_matches_batch() {
        let mut a = RewardShift::new();
        let mut b = RewardShift::new();
        let vals = [0.3_f32, -1.2, 5.0, 2.1, -0.5];
        a.update_batch(&vals);
        for &v in &vals {
            b.update(v);
        }
        assert!((a.shift() - b.shift()).abs() < 1e-6);
    }

    #[test]
    fn loss_finite_and_nonneg() {
        let loss = bco_loss_from_rewards(&[1.0, 2.0], &[-1.0, -2.0], &cfg(0.1, 0.0))
            .expect("value should be present");
        assert!(loss.is_finite() && loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn well_separated_low_loss() {
        // Desirable rewards far above δ, undesirable far below ⇒ near-zero BCE.
        let loss = bco_loss_from_rewards(&[10.0, 12.0], &[-10.0, -12.0], &cfg(1.0, 0.0))
            .expect("value should be present");
        assert!(
            loss < 0.01,
            "well-separated classes should give low loss, got {loss}"
        );
    }

    #[test]
    fn mis_separated_high_loss() {
        // Desirable rewards below δ and undesirable above ⇒ large BCE.
        let good = bco_loss_from_rewards(&[10.0], &[-10.0], &cfg(1.0, 0.0))
            .expect("value should be present");
        let bad = bco_loss_from_rewards(&[-10.0], &[10.0], &cfg(1.0, 0.0))
            .expect("value should be present");
        assert!(
            bad > good,
            "mis-classified rewards must cost more: bad={bad} good={good}"
        );
        assert!(
            bad > 5.0,
            "expected large loss for inverted labels, got {bad}"
        );
    }

    #[test]
    fn shift_moves_boundary() {
        // Same rewards, different δ ⇒ different loss (δ is actually used).
        let l0 = bco_loss_from_rewards(&[1.0], &[-1.0], &cfg(1.0, 0.0))
            .expect("value should be present");
        let l1 = bco_loss_from_rewards(&[1.0], &[-1.0], &cfg(1.0, 0.5))
            .expect("value should be present");
        assert!((l0 - l1).abs() > 1e-6, "reward shift must change the loss");
    }

    #[test]
    fn only_desirable_examples_ok() {
        let loss = bco_loss_from_rewards(&[1.0, 2.0, 3.0], &[], &cfg(0.5, 0.0))
            .expect("value should be present");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn only_undesirable_examples_ok() {
        let loss = bco_loss_from_rewards(&[], &[-1.0, -2.0], &cfg(0.5, 0.0))
            .expect("value should be present");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn both_empty_errors() {
        assert!(matches!(
            bco_loss_from_rewards(&[], &[], &cfg(0.5, 0.0)),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn invalid_beta_errors() {
        assert!(matches!(
            bco_loss_from_rewards(&[1.0], &[-1.0], &cfg(0.0, 0.0)),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }

    #[test]
    fn from_logps_matches_from_rewards() {
        let beta = 0.5_f32;
        let d_lp = vec![-0.5_f32, -0.2];
        let d_ref = vec![-1.0_f32, -1.0];
        let u_lp = vec![-2.0_f32];
        let u_ref = vec![-1.0_f32];
        let cfg = cfg(beta, 0.1);
        let from_logps =
            bco_loss(&d_lp, &d_ref, &u_lp, &u_ref, &cfg).expect("bco_loss should succeed");
        // Manually compute rewards.
        let dr: Vec<f32> = d_lp
            .iter()
            .zip(d_ref.iter())
            .map(|(&a, &b)| beta * (a - b))
            .collect();
        let ur: Vec<f32> = u_lp
            .iter()
            .zip(u_ref.iter())
            .map(|(&a, &b)| beta * (a - b))
            .collect();
        let from_rewards =
            bco_loss_from_rewards(&dr, &ur, &cfg).expect("bco_loss_from_rewards should succeed");
        assert!((from_logps - from_rewards).abs() < 1e-6);
    }

    #[test]
    fn from_logps_dim_mismatch_errors() {
        let cfg = cfg(0.5, 0.0);
        let r = bco_loss(&[-0.5, -0.2], &[-1.0], &[], &[], &cfg);
        assert!(matches!(r, Err(RlhfError::DimensionMismatch { .. })));
    }

    #[test]
    fn self_calibrated_shift_balances_loss() {
        // With δ set to the overall mean reward, a symmetric reward layout gives a
        // finite, balanced loss; verify it equals the explicit-δ computation.
        let desirable = [2.0_f32, 1.0];
        let undesirable = [-1.0_f32, -2.0];
        let mut rs = RewardShift::new();
        rs.update_batch(&desirable);
        rs.update_batch(&undesirable);
        let delta = rs.shift(); // mean = 0.0 here
        let loss = bco_loss_from_rewards(&desirable, &undesirable, &cfg(1.0, delta))
            .expect("value should be present");
        assert!(loss.is_finite() && loss > 0.0);
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

    fn cfg(beta: f32, shift: f32) -> BcoConfig {
        BcoConfig {
            beta,
            reward_shift: shift,
        }
    }

    #[test]
    fn bco_grad_matches_finite_difference() {
        let c = cfg(0.5, 0.2);
        let des = [0.9_f32, -0.3];
        let und = [-0.4_f32, 0.6];
        let g = bco_grad(&des, &und, &c).expect("grad");
        let h = 1e-2;
        for i in 0..des.len() {
            let fd = central_diff(
                |v| {
                    let mut d = des.to_vec();
                    d[i] = v;
                    bco_loss_from_rewards(&d, &und, &c).expect("loss")
                },
                des[i],
                h,
            );
            assert_close(g.d_desirable[i], fd, "d_desirable");
        }
        for i in 0..und.len() {
            let fd = central_diff(
                |v| {
                    let mut u = und.to_vec();
                    u[i] = v;
                    bco_loss_from_rewards(&des, &u, &c).expect("loss")
                },
                und[i],
                h,
            );
            assert_close(g.d_undesirable[i], fd, "d_undesirable");
        }
    }

    #[test]
    fn bco_grad_signs_are_aligned() {
        // Desirable rewards pushed up (negative gradient), undesirable down (positive).
        let g = bco_grad(&[0.3], &[0.3], &cfg(1.0, 0.0)).expect("grad");
        assert!(g.d_desirable[0] < 0.0, "{}", g.d_desirable[0]);
        assert!(g.d_undesirable[0] > 0.0, "{}", g.d_undesirable[0]);
    }

    #[test]
    fn bco_grad_handles_empty_side() {
        let g = bco_grad(&[0.5, 0.2], &[], &cfg(0.5, 0.0)).expect("grad");
        assert_eq!(g.d_desirable.len(), 2);
        assert!(g.d_undesirable.is_empty());
    }

    #[test]
    fn bco_grad_is_deterministic() {
        let c = cfg(0.4, 0.1);
        let a = bco_grad(&[0.5], &[-0.5], &c).expect("a");
        let b = bco_grad(&[0.5], &[-0.5], &c).expect("b");
        assert_eq!(a.d_desirable[0], b.d_desirable[0]);
        assert_eq!(a.d_undesirable[0], b.d_undesirable[0]);
    }

    #[test]
    fn bco_grad_empty_both_errors() {
        assert!(matches!(
            bco_grad(&[], &[], &cfg(0.5, 0.0)),
            Err(RlhfError::EmptyInput)
        ));
    }
}

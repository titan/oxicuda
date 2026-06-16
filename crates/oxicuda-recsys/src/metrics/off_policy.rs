//! Off-policy / counterfactual evaluation estimators for recommendation.
//!
//! When logged interaction data is collected under a *logging* policy `π_0` but
//! we wish to estimate the value of a new *target* policy `π_e`, naive averaging
//! is biased. These estimators correct for the policy mismatch using importance
//! weights `w_i = π_e(a_i | x_i) / π_0(a_i | x_i)`:
//!
//! - **IPS** (Inverse Propensity Scoring): `V̂ = (1/n) Σ w_i r_i`.
//! - **Capped/Clipped IPS**: clip `w_i ≤ τ` to bound variance.
//! - **SNIPS** (Self-Normalised IPS): `V̂ = (Σ w_i r_i) / (Σ w_i)` — lower
//!   variance, removes the multiplicative control-variate bias of IPS.
//! - **Direct Method (DM)**: average a reward model `r̂(x_i, a_i)`.
//! - **Doubly Robust (DR)**: `V̂ = (1/n) Σ [ r̂_i + w_i (r_i − r̂_i) ]` — unbiased
//!   if *either* the propensities *or* the reward model are correct.
//!
//! Each logged sample is represented by a [`LoggedSample`]. All estimators are
//! pure functions returning `Result` over [`OffPolicyError`].

/// A single logged interaction used for off-policy evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoggedSample {
    /// Observed reward `r_i` (e.g. click = 1.0, no-click = 0.0).
    pub reward: f32,
    /// Logging-policy propensity `π_0(a_i | x_i) ∈ (0, 1]`.
    pub logging_prop: f32,
    /// Target-policy probability `π_e(a_i | x_i) ∈ [0, 1]`.
    pub target_prop: f32,
    /// Reward-model estimate `r̂(x_i, a_i)` for the logged action (used by DM/DR).
    pub reward_hat: f32,
}

/// Errors raised by off-policy estimators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffPolicyError {
    /// No logged samples were supplied.
    EmptyLog,
    /// A logging propensity was non-positive (`≤ 0`), making the weight undefined.
    NonPositivePropensity,
    /// A clipping threshold `τ` was non-positive.
    InvalidClip,
}

impl std::fmt::Display for OffPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLog => write!(f, "off-policy log is empty"),
            Self::NonPositivePropensity => {
                write!(f, "logging propensity must be > 0")
            }
            Self::InvalidClip => write!(f, "clip threshold must be > 0"),
        }
    }
}

impl std::error::Error for OffPolicyError {}

/// Convenience result alias.
pub type OffPolicyResult<T> = Result<T, OffPolicyError>;

#[inline]
fn weight(s: &LoggedSample) -> OffPolicyResult<f32> {
    if s.logging_prop <= 0.0 {
        return Err(OffPolicyError::NonPositivePropensity);
    }
    Ok(s.target_prop / s.logging_prop)
}

/// Inverse Propensity Scoring estimate: `V̂ = (1/n) Σ w_i r_i`.
///
/// # Errors
///
/// [`OffPolicyError::EmptyLog`] if `samples` is empty, or
/// [`OffPolicyError::NonPositivePropensity`] if any logging propensity `≤ 0`.
pub fn ips(samples: &[LoggedSample]) -> OffPolicyResult<f32> {
    if samples.is_empty() {
        return Err(OffPolicyError::EmptyLog);
    }
    let mut acc = 0.0_f32;
    for s in samples {
        acc += weight(s)? * s.reward;
    }
    Ok(acc / samples.len() as f32)
}

/// Clipped (capped) IPS: identical to [`ips`] but each weight is capped at
/// `clip` to reduce variance at the cost of a small downward bias.
///
/// # Errors
///
/// [`OffPolicyError::EmptyLog`], [`OffPolicyError::NonPositivePropensity`], or
/// [`OffPolicyError::InvalidClip`] when `clip <= 0`.
pub fn ips_clipped(samples: &[LoggedSample], clip: f32) -> OffPolicyResult<f32> {
    if samples.is_empty() {
        return Err(OffPolicyError::EmptyLog);
    }
    if clip <= 0.0 {
        return Err(OffPolicyError::InvalidClip);
    }
    let mut acc = 0.0_f32;
    for s in samples {
        let w = weight(s)?.min(clip);
        acc += w * s.reward;
    }
    Ok(acc / samples.len() as f32)
}

/// Self-Normalised IPS: `V̂ = (Σ w_i r_i) / (Σ w_i)`.
///
/// If the total weight is zero (e.g. the target policy never selects any logged
/// action) the estimate is defined as `0.0`.
///
/// # Errors
///
/// [`OffPolicyError::EmptyLog`] or [`OffPolicyError::NonPositivePropensity`].
pub fn snips(samples: &[LoggedSample]) -> OffPolicyResult<f32> {
    if samples.is_empty() {
        return Err(OffPolicyError::EmptyLog);
    }
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for s in samples {
        let w = weight(s)?;
        num += w * s.reward;
        den += w;
    }
    if den.abs() < 1e-12 {
        return Ok(0.0);
    }
    Ok(num / den)
}

/// Direct Method: `V̂ = (1/n) Σ r̂_i`, the plain average of the reward-model
/// predictions for the logged actions (ignores observed rewards).
///
/// # Errors
///
/// [`OffPolicyError::EmptyLog`] if `samples` is empty.
pub fn direct_method(samples: &[LoggedSample]) -> OffPolicyResult<f32> {
    if samples.is_empty() {
        return Err(OffPolicyError::EmptyLog);
    }
    let sum: f32 = samples.iter().map(|s| s.reward_hat).sum();
    Ok(sum / samples.len() as f32)
}

/// Doubly Robust: `V̂ = (1/n) Σ [ r̂_i + w_i (r_i − r̂_i) ]`.
///
/// Unbiased when either the propensities or the reward model are correct, and
/// typically lower-variance than IPS.
///
/// # Errors
///
/// [`OffPolicyError::EmptyLog`] or [`OffPolicyError::NonPositivePropensity`].
pub fn doubly_robust(samples: &[LoggedSample]) -> OffPolicyResult<f32> {
    if samples.is_empty() {
        return Err(OffPolicyError::EmptyLog);
    }
    let mut acc = 0.0_f32;
    for s in samples {
        let w = weight(s)?;
        acc += s.reward_hat + w * (s.reward - s.reward_hat);
    }
    Ok(acc / samples.len() as f32)
}

/// Clipped Doubly Robust: as [`doubly_robust`] but caps the importance weight at
/// `clip` to bound variance.
///
/// # Errors
///
/// [`OffPolicyError::EmptyLog`], [`OffPolicyError::NonPositivePropensity`], or
/// [`OffPolicyError::InvalidClip`].
pub fn doubly_robust_clipped(samples: &[LoggedSample], clip: f32) -> OffPolicyResult<f32> {
    if samples.is_empty() {
        return Err(OffPolicyError::EmptyLog);
    }
    if clip <= 0.0 {
        return Err(OffPolicyError::InvalidClip);
    }
    let mut acc = 0.0_f32;
    for s in samples {
        let w = weight(s)?.min(clip);
        acc += s.reward_hat + w * (s.reward - s.reward_hat);
    }
    Ok(acc / samples.len() as f32)
}

/// Effective sample size of the importance weights:
/// `ESS = (Σ w_i)^2 / Σ w_i^2`, a variance-diagnostic in `(0, n]`. Low ESS warns
/// that the estimate is dominated by a few high-weight samples.
///
/// # Errors
///
/// [`OffPolicyError::EmptyLog`] or [`OffPolicyError::NonPositivePropensity`].
pub fn effective_sample_size(samples: &[LoggedSample]) -> OffPolicyResult<f32> {
    if samples.is_empty() {
        return Err(OffPolicyError::EmptyLog);
    }
    let mut sum = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    for s in samples {
        let w = weight(s)?;
        sum += w;
        sum_sq += w * w;
    }
    if sum_sq.abs() < 1e-12 {
        return Ok(0.0);
    }
    Ok(sum * sum / sum_sq)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(reward: f32, log_p: f32, tgt_p: f32, rhat: f32) -> LoggedSample {
        LoggedSample {
            reward,
            logging_prop: log_p,
            target_prop: tgt_p,
            reward_hat: rhat,
        }
    }

    #[test]
    fn ips_on_policy_recovers_mean_reward() {
        // π_e == π_0 → all weights 1 → IPS == mean reward.
        let data = vec![
            sample(1.0, 0.5, 0.5, 0.0),
            sample(0.0, 0.5, 0.5, 0.0),
            sample(1.0, 0.5, 0.5, 0.0),
        ];
        let v = ips(&data).expect("ips");
        assert!((v - 2.0 / 3.0).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn ips_reweights_correctly() {
        // single sample: w = 0.8/0.4 = 2, reward 1 → V = 2*1 / 1 = 2
        let data = vec![sample(1.0, 0.4, 0.8, 0.0)];
        let v = ips(&data).expect("ips");
        assert!((v - 2.0).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn ips_empty_errors() {
        assert_eq!(ips(&[]), Err(OffPolicyError::EmptyLog));
    }

    #[test]
    fn ips_zero_propensity_errors() {
        let data = vec![sample(1.0, 0.0, 0.5, 0.0)];
        assert_eq!(ips(&data), Err(OffPolicyError::NonPositivePropensity));
    }

    #[test]
    fn clipped_ips_caps_weight() {
        // w would be 0.9/0.1 = 9, clip at 3 → V = 3*1 = 3.
        let data = vec![sample(1.0, 0.1, 0.9, 0.0)];
        let v = ips_clipped(&data, 3.0).expect("clipped");
        assert!((v - 3.0).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn clipped_ips_invalid_clip_errors() {
        let data = vec![sample(1.0, 0.5, 0.5, 0.0)];
        assert_eq!(ips_clipped(&data, 0.0), Err(OffPolicyError::InvalidClip));
        assert_eq!(ips_clipped(&[], 1.0), Err(OffPolicyError::EmptyLog));
    }

    #[test]
    fn snips_normalises() {
        // weights: 2 and 0.5; rewards 1 and 0.
        // SNIPS = (2*1 + 0.5*0) / (2 + 0.5) = 2 / 2.5 = 0.8
        let data = vec![sample(1.0, 0.4, 0.8, 0.0), sample(0.0, 0.8, 0.4, 0.0)];
        let v = snips(&data).expect("snips");
        assert!((v - 0.8).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn snips_zero_weight_returns_zero() {
        // target policy never picks the logged actions → weights 0.
        let data = vec![sample(1.0, 0.5, 0.0, 0.0), sample(0.0, 0.5, 0.0, 0.0)];
        let v = snips(&data).expect("snips");
        assert!(v.abs() < 1e-7, "got {v}");
    }

    #[test]
    fn snips_empty_errors() {
        assert_eq!(snips(&[]), Err(OffPolicyError::EmptyLog));
    }

    #[test]
    fn direct_method_averages_rhat() {
        let data = vec![
            sample(0.0, 0.5, 0.5, 0.2),
            sample(0.0, 0.5, 0.5, 0.4),
            sample(0.0, 0.5, 0.5, 0.6),
        ];
        let v = direct_method(&data).expect("dm");
        assert!((v - 0.4).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn direct_method_empty_errors() {
        assert_eq!(direct_method(&[]), Err(OffPolicyError::EmptyLog));
    }

    #[test]
    fn dr_perfect_reward_model_equals_dm() {
        // If r̂ == r exactly, the correction term vanishes regardless of weights.
        let data = vec![sample(1.0, 0.3, 0.9, 1.0), sample(0.0, 0.7, 0.2, 0.0)];
        let dr = doubly_robust(&data).expect("dr");
        let dm = direct_method(&data).expect("dm");
        assert!((dr - dm).abs() < 1e-5, "dr {dr} should equal dm {dm}");
    }

    #[test]
    fn dr_zero_reward_model_equals_ips() {
        // If r̂ == 0 everywhere, DR reduces to IPS.
        let data = vec![
            sample(1.0, 0.4, 0.8, 0.0),
            sample(0.0, 0.5, 0.5, 0.0),
            sample(1.0, 0.25, 0.5, 0.0),
        ];
        let dr = doubly_robust(&data).expect("dr");
        let ips_v = ips(&data).expect("ips");
        assert!(
            (dr - ips_v).abs() < 1e-5,
            "dr {dr} should equal ips {ips_v}"
        );
    }

    #[test]
    fn dr_empty_and_bad_propensity_errors() {
        assert_eq!(doubly_robust(&[]), Err(OffPolicyError::EmptyLog));
        let data = vec![sample(1.0, -0.1, 0.5, 0.0)];
        assert_eq!(
            doubly_robust(&data),
            Err(OffPolicyError::NonPositivePropensity)
        );
    }

    #[test]
    fn dr_clipped_matches_dr_when_below_threshold() {
        let data = vec![sample(1.0, 0.5, 0.5, 0.3), sample(0.0, 0.5, 0.5, 0.1)];
        let dr = doubly_robust(&data).expect("dr");
        let drc = doubly_robust_clipped(&data, 10.0).expect("drc");
        assert!((dr - drc).abs() < 1e-5, "dr {dr} vs clipped {drc}");
        assert_eq!(
            doubly_robust_clipped(&data, 0.0),
            Err(OffPolicyError::InvalidClip)
        );
    }

    #[test]
    fn ess_uniform_weights_equals_n() {
        // all weights equal → ESS = n.
        let data = vec![
            sample(1.0, 0.5, 0.5, 0.0),
            sample(0.0, 0.5, 0.5, 0.0),
            sample(1.0, 0.5, 0.5, 0.0),
            sample(0.0, 0.5, 0.5, 0.0),
        ];
        let ess = effective_sample_size(&data).expect("ess");
        assert!((ess - 4.0).abs() < 1e-4, "got {ess}");
    }

    #[test]
    fn ess_concentrated_weights_low() {
        // one huge weight dominates → ESS near 1.
        let data = vec![
            sample(1.0, 0.01, 1.0, 0.0), // w = 100
            sample(0.0, 1.0, 0.0, 0.0),  // w = 0
            sample(0.0, 1.0, 0.0, 0.0),  // w = 0
        ];
        let ess = effective_sample_size(&data).expect("ess");
        assert!(ess < 1.5, "expected low ESS, got {ess}");
    }

    #[test]
    fn error_display_messages() {
        assert!(OffPolicyError::EmptyLog.to_string().contains("empty"));
        assert!(
            OffPolicyError::NonPositivePropensity
                .to_string()
                .contains("propensity")
        );
        assert!(OffPolicyError::InvalidClip.to_string().contains("clip"));
    }
}

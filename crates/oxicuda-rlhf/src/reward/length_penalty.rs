//! Length-bias correction in reward modelling.
//!
//! A well-known failure mode of RLHF reward models is **length bias**: the
//! reward becomes spuriously correlated with response length, so the policy
//! learns to "hack" the reward by producing longer (not better) responses. This
//! module removes the length-correlated component of the reward signal, in the
//! spirit of length-controlled / length-debiased preference optimisation
//! (e.g. Park et al. 2024, *Disentangling Length from Quality in Direct
//! Preference Optimization*; Dubois et al. 2024, *Length-Controlled
//! AlpacaEval*).
//!
//! The estimator fits an ordinary least-squares line of reward on length,
//!
//! ```text
//!   r ≈ a + b · ℓ ,
//! ```
//!
//! where the slope `b = Cov(ℓ, r) / Var(ℓ)` captures how much the reward moves
//! with a unit change in length. The **length-debiased reward** subtracts the
//! length-driven deviation while preserving the overall reward level:
//!
//! ```text
//!   r_debiased = r − b · (ℓ − mean(ℓ)) .
//! ```
//!
//! Two limiting cases motivate the construction:
//!
//! * If the reward is *purely* length-driven (`r = a + b · ℓ` exactly), then
//!   `r_debiased = a + b · mean(ℓ)` is **constant** — the length signal is fully
//!   removed.
//! * If the reward is *uncorrelated* with length (`b = 0`), the reward passes
//!   through **unchanged**.
//!
//! By the OLS normal equations the residual `r_debiased` is exactly
//! uncorrelated with `ℓ`, so the reward–length correlation collapses to zero
//! after debiasing.

use crate::error::{RlhfError, RlhfResult};

/// Linear length-debiasing model for reward scores.
///
/// Fit it once on a corpus of `(reward, length)` observations, then apply
/// [`LengthDebiasedReward::debias`] to individual rewards to strip the
/// length-correlated component.
#[derive(Debug, Clone)]
pub struct LengthDebiasedReward {
    slope: f32,
    intercept: f32,
    mean_length: f32,
    mean_reward: f32,
    fitted: bool,
}

impl Default for LengthDebiasedReward {
    fn default() -> Self {
        Self::new()
    }
}

impl LengthDebiasedReward {
    /// Construct an unfitted model (identity debiasing until [`Self::fit`]).
    #[must_use]
    pub fn new() -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
            mean_length: 0.0,
            mean_reward: 0.0,
            fitted: false,
        }
    }

    /// The fitted length slope `b = Cov(ℓ, r) / Var(ℓ)` (`0` if unfitted or if
    /// every length is identical).
    #[must_use]
    pub fn slope(&self) -> f32 {
        self.slope
    }

    /// The fitted intercept `a = mean(r) − b · mean(ℓ)`.
    #[must_use]
    pub fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Whether [`Self::fit`] has been called successfully.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Fit the OLS length slope and mean length from `(reward, length)` pairs.
    ///
    /// If every length is identical the slope is left at `0` (no length signal
    /// can be estimated), so debiasing becomes a no-op.
    ///
    /// # Errors
    /// - [`RlhfError::EmptyInput`] if either slice is empty.
    /// - [`RlhfError::DimensionMismatch`] if the slices differ in length.
    /// - [`RlhfError::NanEncountered`] if any value is non-finite.
    pub fn fit(&mut self, rewards: &[f32], lengths: &[f32]) -> RlhfResult<()> {
        if rewards.is_empty() || lengths.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        if rewards.len() != lengths.len() {
            return Err(RlhfError::DimensionMismatch {
                expected: rewards.len(),
                got: lengths.len(),
            });
        }
        for (&r, &l) in rewards.iter().zip(lengths.iter()) {
            if !r.is_finite() || !l.is_finite() {
                return Err(RlhfError::NanEncountered);
            }
        }

        let n = rewards.len() as f32;
        let mean_reward = rewards.iter().sum::<f32>() / n;
        let mean_length = lengths.iter().sum::<f32>() / n;

        let mut cov = 0.0_f32;
        let mut var = 0.0_f32;
        for (&r, &l) in rewards.iter().zip(lengths.iter()) {
            let dl = l - mean_length;
            cov += dl * (r - mean_reward);
            var += dl * dl;
        }
        let slope = if var > 0.0 { cov / var } else { 0.0 };

        self.slope = slope;
        self.mean_length = mean_length;
        self.mean_reward = mean_reward;
        self.intercept = mean_reward - slope * mean_length;
        self.fitted = true;
        Ok(())
    }

    /// Remove the length-correlated component of a single reward:
    /// `r − b · (ℓ − mean(ℓ))`.
    #[must_use]
    pub fn debias(&self, reward: f32, length: f32) -> f32 {
        reward - self.slope * (length - self.mean_length)
    }

    /// Debias a batch of rewards.
    ///
    /// # Errors
    /// - [`RlhfError::DimensionMismatch`] if the slices differ in length.
    pub fn debias_batch(&self, rewards: &[f32], lengths: &[f32]) -> RlhfResult<Vec<f32>> {
        if rewards.len() != lengths.len() {
            return Err(RlhfError::DimensionMismatch {
                expected: rewards.len(),
                got: lengths.len(),
            });
        }
        Ok(rewards
            .iter()
            .zip(lengths.iter())
            .map(|(&r, &l)| self.debias(r, l))
            .collect())
    }
}

/// Pearson correlation coefficient between two equal-length samples.
///
/// Returns `0.0` when either sample has zero variance (correlation is undefined
/// — there is no linear relationship to report). The result is clamped to
/// `[-1, 1]` to absorb floating-point error.
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if either slice is empty.
/// - [`RlhfError::DimensionMismatch`] if the slices differ in length.
/// - [`RlhfError::NanEncountered`] if any value is non-finite.
pub fn pearson_correlation(a: &[f32], b: &[f32]) -> RlhfResult<f32> {
    if a.is_empty() || b.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
    }

    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut cov = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_b = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        return Ok(0.0);
    }
    Ok((cov / (var_a.sqrt() * var_b.sqrt())).clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variance(values: &[f32]) -> f32 {
        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n
    }

    #[test]
    fn purely_length_driven_reward_becomes_constant() {
        let lengths = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        // reward is an exact linear function of length.
        let rewards: Vec<f32> = lengths.iter().map(|&l| 2.0 + 0.5 * l).collect();
        let mut model = LengthDebiasedReward::new();
        model.fit(&rewards, &lengths).expect("fit");
        let debiased = model.debias_batch(&rewards, &lengths).expect("debias");
        let max = debiased.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = debiased.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            (max - min).abs() < 1e-3,
            "purely length-driven reward must debias to a constant: {debiased:?}"
        );
    }

    #[test]
    fn slope_recovers_true_coefficient() {
        let lengths = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let rewards: Vec<f32> = lengths.iter().map(|&l| 2.0 + 0.5 * l).collect();
        let mut model = LengthDebiasedReward::new();
        model.fit(&rewards, &lengths).expect("fit");
        assert!(
            (model.slope() - 0.5).abs() < 1e-4,
            "slope should recover 0.5, got {}",
            model.slope()
        );
    }

    #[test]
    fn uncorrelated_reward_is_unchanged() {
        // Cov(reward, length) = 0 by construction.
        let lengths = [1.0_f32, 2.0, 3.0, 4.0];
        let rewards = [1.0_f32, -1.0, -1.0, 1.0];
        let mut model = LengthDebiasedReward::new();
        model.fit(&rewards, &lengths).expect("fit");
        assert!(model.slope().abs() < 1e-5, "slope should be ~0");
        for (&r, &l) in rewards.iter().zip(lengths.iter()) {
            assert!(
                (model.debias(r, l) - r).abs() < 1e-5,
                "uncorrelated reward must be unchanged"
            );
        }
    }

    #[test]
    fn correlation_magnitude_decreases_after_debiasing() {
        let lengths = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let rewards = [1.0_f32, 2.0, 2.0, 4.0, 5.0];
        let corr_before = pearson_correlation(&rewards, &lengths).expect("corr before");
        assert!(
            corr_before.abs() > 0.5,
            "test data should be length-correlated, got {corr_before}"
        );

        let mut model = LengthDebiasedReward::new();
        model.fit(&rewards, &lengths).expect("fit");
        let debiased = model.debias_batch(&rewards, &lengths).expect("debias");
        let corr_after = pearson_correlation(&debiased, &lengths).expect("corr after");

        assert!(
            corr_after.abs() < corr_before.abs(),
            "debiasing must reduce |correlation|: before={corr_before}, after={corr_after}"
        );
        assert!(
            corr_after.abs() < 1e-4,
            "OLS residual is uncorrelated with length, got {corr_after}"
        );
    }

    #[test]
    fn constant_length_disables_debiasing() {
        // Var(length) = 0 → slope stays 0, debias is a no-op.
        let lengths = [3.0_f32, 3.0, 3.0];
        let rewards = [1.0_f32, 5.0, 9.0];
        let mut model = LengthDebiasedReward::new();
        model.fit(&rewards, &lengths).expect("fit");
        assert!(model.slope().abs() < 1e-6);
        for (&r, &l) in rewards.iter().zip(lengths.iter()) {
            assert!((model.debias(r, l) - r).abs() < 1e-6);
        }
    }

    #[test]
    fn debiased_reward_has_near_zero_variance_when_length_explains_all() {
        let lengths = [10.0_f32, 20.0, 30.0, 40.0];
        let rewards: Vec<f32> = lengths.iter().map(|&l| -1.0 + 0.25 * l).collect();
        let mut model = LengthDebiasedReward::new();
        model.fit(&rewards, &lengths).expect("fit");
        let debiased = model.debias_batch(&rewards, &lengths).expect("debias");
        assert!(
            variance(&debiased) < 1e-4,
            "variance should collapse, got {}",
            variance(&debiased)
        );
    }

    #[test]
    fn pearson_perfect_positive_and_negative() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let y_pos: Vec<f32> = x.iter().map(|&v| 2.0 * v + 1.0).collect();
        let y_neg: Vec<f32> = x.iter().map(|&v| -3.0 * v).collect();
        assert!(
            (pearson_correlation(&x, &y_pos).expect("pearson_correlation should succeed") - 1.0)
                .abs()
                < 1e-5
        );
        assert!(
            (pearson_correlation(&x, &y_neg).expect("pearson_correlation should succeed") + 1.0)
                .abs()
                < 1e-5
        );
    }

    #[test]
    fn pearson_constant_is_zero() {
        let x = [1.0_f32, 2.0, 3.0];
        let c = [7.0_f32, 7.0, 7.0];
        assert!(
            pearson_correlation(&x, &c)
                .expect("pearson_correlation should succeed")
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn fit_empty_and_mismatch_error() {
        let mut model = LengthDebiasedReward::new();
        assert!(matches!(model.fit(&[], &[]), Err(RlhfError::EmptyInput)));
        assert!(matches!(
            model.fit(&[1.0, 2.0], &[1.0]),
            Err(RlhfError::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn fit_nan_errors() {
        let mut model = LengthDebiasedReward::new();
        assert!(matches!(
            model.fit(&[1.0, f32::NAN], &[1.0, 2.0]),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn pearson_mismatch_and_empty_error() {
        assert!(matches!(
            pearson_correlation(&[1.0, 2.0], &[1.0]),
            Err(RlhfError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            pearson_correlation(&[], &[]),
            Err(RlhfError::EmptyInput)
        ));
    }
}

use crate::error::{RlhfError, RlhfResult};

pub struct KlController {
    pub beta: f32,
    pub target_kl: f32,
    pub k_beta: f32,
}

impl KlController {
    pub fn new(init_beta: f32, target_kl: f32) -> Self {
        Self {
            beta: init_beta,
            target_kl,
            k_beta: 0.2,
        }
    }

    pub fn update_beta(&mut self, current_kl: f32) {
        let proportional_error = (current_kl - self.target_kl) / self.target_kl;
        self.beta *= 1.0 + self.k_beta * proportional_error;
        self.beta = self.beta.max(1e-6);
    }
}

pub fn kl_divergence_from_logps(log_probs: &[f32], ref_log_probs: &[f32]) -> RlhfResult<f32> {
    if log_probs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if log_probs.len() != ref_log_probs.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: log_probs.len(),
            got: ref_log_probs.len(),
        });
    }
    let kl: f32 = log_probs
        .iter()
        .zip(ref_log_probs.iter())
        .map(|(&lp, &rlp)| lp - rlp)
        .sum::<f32>()
        / log_probs.len() as f32;
    if kl.is_nan() {
        return Err(RlhfError::KlDivergence {
            msg: "NaN in KL computation".into(),
        });
    }
    Ok(kl)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // kl_divergence_from_logps
    // -----------------------------------------------------------------------

    /// Identical log-prob vectors must give KL = 0.0 exactly (all differences
    /// are 0 before any floating-point accumulation, so the sum is exactly 0).
    #[test]
    fn kl_identical_vectors_gives_exact_zero() {
        let lps: &[f32] = &[-1.0, -2.0, -3.5];
        let result = kl_divergence_from_logps(lps, lps).expect("identical vectors must succeed");
        assert_eq!(
            result, 0.0,
            "KL of identical log-prob vectors must be exactly 0.0, got {result}"
        );
    }

    /// Hand-chosen vectors where every difference is 0.5 → mean = 0.5.
    #[test]
    fn kl_hand_chosen_closed_form_positive() {
        // diffs: 0.5, 0.5, 0.5  →  mean = 0.5
        let log_probs = [-1.0_f32, -2.0, -3.0];
        let ref_lps = [-1.5_f32, -2.5, -3.5];
        let result =
            kl_divergence_from_logps(&log_probs, &ref_lps).expect("valid input must succeed");
        assert!(
            (result - 0.5_f32).abs() < 1e-6,
            "expected KL ≈ 0.5, got {result}"
        );
    }

    /// When the reference distribution is more likely the estimator is negative.
    /// log_probs < ref_log_probs  →  mean(lp - rlp) = −0.5.
    #[test]
    fn kl_negative_estimator_when_ref_higher() {
        let log_probs = [-1.5_f32, -2.5];
        let ref_lps = [-1.0_f32, -2.0];
        let result =
            kl_divergence_from_logps(&log_probs, &ref_lps).expect("valid input must succeed");
        assert!(
            (result - (-0.5_f32)).abs() < 1e-6,
            "expected KL ≈ -0.5, got {result}"
        );
    }

    /// Empty slice must return RlhfError::EmptyInput.
    #[test]
    fn kl_empty_input_returns_error() {
        let err = kl_divergence_from_logps(&[], &[]).expect_err("empty input must error");
        assert!(
            matches!(err, RlhfError::EmptyInput),
            "expected EmptyInput, got {err:?}"
        );
    }

    /// Mismatched lengths must return RlhfError::DimensionMismatch with the
    /// correct expected/got counts.
    #[test]
    fn kl_dimension_mismatch_returns_error() {
        let log_probs = [-1.0_f32, -2.0];
        let ref_lps = [-1.0_f32];
        let err = kl_divergence_from_logps(&log_probs, &ref_lps).expect_err("mismatch must error");
        assert!(
            matches!(
                err,
                RlhfError::DimensionMismatch {
                    expected: 2,
                    got: 1
                }
            ),
            "expected DimensionMismatch{{expected:2, got:1}}, got {err:?}"
        );
    }

    /// A NaN element propagates through the sum and triggers the KlDivergence
    /// guard inside the function.
    #[test]
    fn kl_nan_in_log_probs_returns_kl_divergence_error() {
        let log_probs = [-1.0_f32, f32::NAN];
        let ref_lps = [-1.0_f32, -1.0];
        let err = kl_divergence_from_logps(&log_probs, &ref_lps).expect_err("NaN must error");
        assert!(
            matches!(err, RlhfError::KlDivergence { .. }),
            "expected KlDivergence error, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // KlController::update_beta
    // -----------------------------------------------------------------------

    /// current_kl ABOVE target_kl → beta must increase.
    ///
    /// Setup: beta=0.1, target=0.01, k_beta=0.2, current_kl=0.02
    ///   proportional_error = (0.02 − 0.01) / 0.01 = 1.0
    ///   new_beta = 0.1 × (1 + 0.2 × 1.0) = 0.12
    #[test]
    fn update_beta_above_target_raises_beta() {
        let mut ctrl = KlController::new(0.1, 0.01);
        let before = ctrl.beta;
        ctrl.update_beta(0.02);
        assert!(
            ctrl.beta > before,
            "beta must increase when current_kl > target_kl: before={before}, after={}",
            ctrl.beta
        );
    }

    /// current_kl BELOW target_kl → beta must decrease.
    ///
    /// Setup: beta=0.1, target=0.1, k_beta=0.2, current_kl=0.05
    ///   proportional_error = −0.5  →  new_beta = 0.1 × 0.9 = 0.09
    #[test]
    fn update_beta_below_target_lowers_beta() {
        let mut ctrl = KlController::new(0.1, 0.1);
        let before = ctrl.beta;
        ctrl.update_beta(0.05);
        assert!(
            ctrl.beta < before,
            "beta must decrease when current_kl < target_kl: before={before}, after={}",
            ctrl.beta
        );
    }

    /// current_kl == target_kl → proportional_error = 0 → beta unchanged.
    #[test]
    fn update_beta_at_target_leaves_beta_unchanged() {
        let mut ctrl = KlController::new(0.3, 0.1);
        let before = ctrl.beta;
        let target = ctrl.target_kl;
        ctrl.update_beta(target);
        assert!(
            (ctrl.beta - before).abs() < 1e-7,
            "beta must be unchanged when current_kl == target_kl: before={before}, after={}",
            ctrl.beta
        );
    }

    /// beta is clamped to the floor 1e-6 even when the proportional update
    /// would push it below that value.
    ///
    /// Setup: beta=1e-6, target=0.1, k_beta=0.2, current_kl=0.0
    ///   proportional_error = −1.0  →  raw_beta = 1e-6 × 0.8 = 8e-7 < 1e-6
    ///   After clamp: 1e-6
    #[test]
    fn update_beta_floor_clamp_at_1e6() {
        let mut ctrl = KlController::new(1e-6, 0.1);
        ctrl.update_beta(0.0);
        assert!(
            ctrl.beta >= 1e-6,
            "beta must not go below 1e-6 after clamp, got {}",
            ctrl.beta
        );
        assert!(
            (ctrl.beta - 1e-6_f32).abs() < 1e-11,
            "beta must be exactly the floor 1e-6 after clamp, got {}",
            ctrl.beta
        );
    }

    /// Exact pinned value for one specific (beta, target_kl, k_beta, current_kl) tuple.
    ///
    /// beta=0.5, target=0.1, k_beta=0.2 (default), current_kl=0.15
    ///   proportional_error = (0.15 − 0.1) / 0.1 = 0.5
    ///   new_beta = 0.5 × (1 + 0.2 × 0.5) = 0.5 × 1.1 = 0.55
    #[test]
    fn update_beta_pinned_exact_value() {
        let mut ctrl = KlController::new(0.5, 0.1);
        ctrl.update_beta(0.15);
        assert!(
            (ctrl.beta - 0.55_f32).abs() < 1e-6,
            "expected beta ≈ 0.55, got {}",
            ctrl.beta
        );
    }
}

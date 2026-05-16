//! Report-Noisy-Max mechanism.
//!
//! Adds independent Laplace noise Lap(Δq/ε) to each score and returns the
//! argmax.  This is a clean (ε, 0)-DP mechanism that exploits the fact that
//! only the *identity* of the maximum matters, not its value.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for the Report-Noisy-Max mechanism.
#[derive(Debug, Clone)]
pub struct RnmConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
    /// Global sensitivity Δq > 0.
    pub sensitivity: f64,
}

impl RnmConfig {
    /// Construct and validate an `RnmConfig`.
    ///
    /// # Errors
    /// Returns `NonPositiveEpsilon` or `NonPositiveSensitivity` on invalid params.
    pub fn new(epsilon: f64, sensitivity: f64) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        Ok(Self {
            epsilon,
            sensitivity,
        })
    }
}

/// Sample Laplace(0, scale) via the inverse CDF method.
///
/// `u` must be drawn from Uniform(0, 1) exclusive of exact 0 or 1
/// to keep the log well-defined; the caller should use `rng.next_f64()`
/// which gives values in `[0, 1)`.
fn laplace_sample(scale: f64, rng: &mut LcgRng) -> f64 {
    // u ∈ [0, 1); map to (-0.5, 0.5) then apply inverse CDF.
    let u = rng.next_f64() - 0.5;
    // Avoid ln(0): |u| is in [0, 0.5), so 1 - 2|u| ∈ (0, 1].
    let abs_u = u.abs();
    let log_term = (1.0 - 2.0 * abs_u).ln();
    -scale * u.signum() * log_term
}

/// Report-Noisy-Max: add Lap(Δq/ε) to each score and return the argmax index.
///
/// Provides (ε, 0)-differential privacy.  The noise scale is `sensitivity / epsilon`.
///
/// # Errors
/// - `EmptyInput` if `scores` is empty.
/// - `NonPositiveEpsilon` or `NonPositiveSensitivity` if config values are invalid.
pub fn report_noisy_max(scores: &[f64], cfg: &RnmConfig, rng: &mut LcgRng) -> PrivacyResult<usize> {
    if scores.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if cfg.epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon));
    }
    if cfg.sensitivity <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(cfg.sensitivity));
    }

    let scale = cfg.sensitivity / cfg.epsilon;

    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;

    for (i, &s) in scores.iter().enumerate() {
        let noisy = s + laplace_sample(scale, rng);
        if noisy > best_val {
            best_val = noisy;
            best_idx = i;
        }
    }

    Ok(best_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnm_returns_valid_index() {
        let scores = vec![0.1, 5.0, 2.0, 1.0];
        let cfg = RnmConfig::new(1.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(99);
        for _ in 0..200 {
            let idx = report_noisy_max(&scores, &cfg, &mut rng).expect("ok");
            assert!(idx < scores.len());
        }
    }

    #[test]
    fn test_rnm_empty_error() {
        let cfg = RnmConfig::new(1.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(0);
        assert!(report_noisy_max(&[], &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_rnm_high_epsilon_favors_max() {
        // With very high epsilon (very little noise) the true argmax should dominate.
        let scores = vec![0.0, 0.0, 100.0, 0.0];
        let cfg = RnmConfig::new(1000.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(7);
        let mut count_correct = 0usize;
        for _ in 0..100 {
            if report_noisy_max(&scores, &cfg, &mut rng).expect("ok") == 2 {
                count_correct += 1;
            }
        }
        // With epsilon=1000 and score gap of 100, should almost always pick idx=2.
        assert!(
            count_correct >= 95,
            "expected >=95 correct, got {count_correct}"
        );
    }
}

//! Propose-Test-Release (PTR) mechanism (Dwork & Lei, 2009).
//!
//! PTR allows release of a sensitive statistic by first testing whether the
//! local sensitivity is small enough (with high probability) before adding
//! calibrated noise.  If the test fails, the mechanism returns `None`.
//!
//! # Protocol
//! 1. Compute `c = (1/ε) · ln(1 / (2δ))` — the noise threshold.
//! 2. Draw `ξ ~ Lap(1/ε)`.
//! 3. If `local_sens + ξ ≤ c`: release `output + Lap(sensitivity_bound / ε)`.
//! 4. Otherwise: return `None`.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for the Propose-Test-Release mechanism.
#[derive(Debug, Clone)]
pub struct PtrConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
    /// Failure probability δ ∈ (0, 1).
    pub delta: f64,
    /// An upper bound on the local sensitivity to use when adding output noise.
    pub sensitivity_bound: f64,
}

impl PtrConfig {
    /// Construct and validate a `PtrConfig`.
    ///
    /// # Errors
    /// Returns `NonPositiveEpsilon`, `InvalidDelta`, or `NonPositiveSensitivity`
    /// when arguments are out of range.
    pub fn new(epsilon: f64, delta: f64, sensitivity_bound: f64) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        if sensitivity_bound <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity_bound));
        }
        Ok(Self {
            epsilon,
            delta,
            sensitivity_bound,
        })
    }
}

/// Sample a single Laplace(0, scale) deviate via inverse-CDF.
fn laplace_sample(scale: f64, rng: &mut LcgRng) -> f64 {
    let u = rng.next_f64() - 0.5;
    let abs_u = u.abs().min(0.5 - f64::EPSILON);
    let log_term = (1.0 - 2.0 * abs_u).ln();
    -scale * u.signum() * log_term
}

/// Run the Propose-Test-Release protocol.
///
/// # Arguments
/// - `local_sens`: the caller-computed local sensitivity LS_f(x) at the current dataset.
/// - `output`: the proposed (noiseless) output value f(x).
/// - `cfg`: PTR parameters (ε, δ, sensitivity_bound).
/// - `rng`: deterministic LCG for reproducibility.
///
/// # Returns
/// - `Ok(Some(y))` — release with `y = output + Lap(sensitivity_bound / ε)`.
/// - `Ok(None)` — test failed; mechanism abstains.
///
/// # Errors
/// Returns `NonPositiveEpsilon`, `InvalidDelta`, or `NonPositiveSensitivity`
/// if config values are invalid.
pub fn propose_test_release(
    local_sens: f64,
    output: f64,
    cfg: &PtrConfig,
    rng: &mut LcgRng,
) -> PrivacyResult<Option<f64>> {
    if cfg.epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon));
    }
    if !(cfg.delta > 0.0 && cfg.delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(cfg.delta));
    }
    if cfg.sensitivity_bound <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(cfg.sensitivity_bound));
    }

    // Step 1: compute the threshold c = (1/ε) · ln(1/(2δ)).
    let c = (1.0 / cfg.epsilon) * (1.0 / (2.0 * cfg.delta)).ln();

    // Step 2: draw ξ ~ Lap(1/ε).
    let xi = laplace_sample(1.0 / cfg.epsilon, rng);

    // Step 3: test.
    if local_sens + xi <= c {
        // Test passed — release noisy output.
        let noise = laplace_sample(cfg.sensitivity_bound / cfg.epsilon, rng);
        Ok(Some(output + noise))
    } else {
        // Test failed — abstain.
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptr_valid_output_type() {
        let cfg = PtrConfig::new(1.0, 1e-5, 1.0).expect("ok");
        let mut rng = LcgRng::new(77);
        // local_sens = 0.0 should almost always pass the test.
        let result = propose_test_release(0.0, 42.0, &cfg, &mut rng).expect("ok");
        // Either None or Some(f64); just verify it doesn't panic.
        if let Some(v) = result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ptr_bad_epsilon() {
        let result = PtrConfig::new(0.0, 1e-5, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ptr_bad_delta() {
        let result = PtrConfig::new(1.0, 1.5, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ptr_very_small_local_sens_often_releases() {
        let cfg = PtrConfig::new(10.0, 1e-10, 1.0).expect("ok");
        let mut rng = LcgRng::new(33);
        let mut released = 0usize;
        for _ in 0..100 {
            if propose_test_release(0.0, 1.0, &cfg, &mut rng)
                .expect("ok")
                .is_some()
            {
                released += 1;
            }
        }
        // With ε=10 and δ=1e-10, c is large; local_sens=0 almost always passes.
        assert!(released >= 90, "expected >=90 releases, got {released}");
    }
}

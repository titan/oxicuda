//! Generic AboveThreshold mechanism.
//!
//! Given a stream of query values and a threshold, returns the *indices* of
//! queries that exceed the noisy threshold (up to `max_above` answers), while
//! providing (ε, 0)-DP via Laplace noise on both the threshold and each query.
//!
//! Unlike `SvtState` (which is stateful and streaming), this function takes
//! the entire query array at once for convenience.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for the AboveThreshold primitive.
#[derive(Debug, Clone)]
pub struct AboveThresholdConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
    /// Threshold T (comparisons are noisy T̃ = T + Lap(2·Δ·max_above / ε)).
    pub threshold: f64,
    /// Global sensitivity of each query function Δ.
    pub sensitivity: f64,
    /// Maximum number of above-threshold indices to return.
    pub max_above: usize,
}

impl AboveThresholdConfig {
    /// Construct and validate an `AboveThresholdConfig`.
    ///
    /// # Errors
    /// Returns appropriate `PrivacyError` variants on invalid parameters.
    pub fn new(
        epsilon: f64,
        threshold: f64,
        sensitivity: f64,
        max_above: usize,
    ) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        if max_above == 0 {
            return Err(PrivacyError::InvalidParameter(
                "max_above must be ≥ 1".into(),
            ));
        }
        Ok(Self {
            epsilon,
            threshold,
            sensitivity,
            max_above,
        })
    }
}

/// Compute the indices of above-threshold queries from a query array.
///
/// The mechanism:
/// 1. Computes a noisy threshold: T̃ = threshold + Lap(2·k·Δ / ε).
/// 2. For each query qᵢ, adds noise νᵢ ~ Lap(4·k·Δ / ε).
/// 3. Collects indices where qᵢ + νᵢ ≥ T̃, stopping at `max_above`.
///
/// # Errors
/// - `EmptyInput` if `queries` is empty.
/// - Validation errors from `AboveThresholdConfig`.
pub fn above_threshold(
    queries: &[f64],
    cfg: &AboveThresholdConfig,
    rng: &mut LcgRng,
) -> PrivacyResult<Vec<usize>> {
    if queries.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if cfg.epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon));
    }
    if cfg.sensitivity <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(cfg.sensitivity));
    }
    if cfg.max_above == 0 {
        return Err(PrivacyError::InvalidParameter(
            "max_above must be ≥ 1".into(),
        ));
    }

    let k = cfg.max_above as f64;

    // Noisy threshold.
    let thresh_scale = 2.0 * k * cfg.sensitivity / cfg.epsilon;
    let noisy_thresh = cfg.threshold + laplace_sample(thresh_scale, rng);

    // Query noise scale.
    let query_scale = 4.0 * k * cfg.sensitivity / cfg.epsilon;

    let mut results = Vec::new();
    for (i, &q) in queries.iter().enumerate() {
        if results.len() >= cfg.max_above {
            break;
        }
        let noisy_q = q + laplace_sample(query_scale, rng);
        if noisy_q >= noisy_thresh {
            results.push(i);
        }
    }

    Ok(results)
}

/// Sample a single Laplace(0, scale) deviate via the inverse CDF.
fn laplace_sample(scale: f64, rng: &mut LcgRng) -> f64 {
    let u = rng.next_f64() - 0.5;
    let abs_u = u.abs().min(0.5 - f64::EPSILON);
    -scale * u.signum() * (1.0 - 2.0 * abs_u).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_above_threshold_basic() {
        // Queries well above threshold should mostly be selected.
        let queries = vec![100.0; 10];
        let cfg = AboveThresholdConfig::new(2.0, 0.0, 1.0, 5).expect("ok");
        let mut rng = LcgRng::new(42);
        let selected = above_threshold(&queries, &cfg, &mut rng).expect("ok");
        assert!(!selected.is_empty());
        assert!(selected.len() <= 5);
        for &idx in &selected {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_above_threshold_respects_max() {
        let queries = vec![1_000.0; 100];
        let cfg = AboveThresholdConfig::new(1.0, -10_000.0, 1.0, 3).expect("ok");
        let mut rng = LcgRng::new(7);
        let selected = above_threshold(&queries, &cfg, &mut rng).expect("ok");
        assert!(selected.len() <= 3);
    }

    #[test]
    fn test_above_threshold_empty_error() {
        let cfg = AboveThresholdConfig::new(1.0, 0.0, 1.0, 1).expect("ok");
        let mut rng = LcgRng::new(0);
        assert!(above_threshold(&[], &cfg, &mut rng).is_err());
    }
}

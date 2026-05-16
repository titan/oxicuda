//! Exponential mechanism (McSherry & Talwar, 2007).
//!
//! The exponential mechanism over a finite set of outcomes selects outcome `i`
//! with probability proportional to `exp(ε · q(x, rᵢ) / (2 · Δq))`, where
//! `Δq` is the global sensitivity of the quality function `q`.
//!
//! The implementation uses a numerically stable shifted softmax followed by
//! a linear scan (alias method is overkill for typical output set sizes).

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for the exponential mechanism.
#[derive(Debug, Clone)]
pub struct ExponentialConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
    /// Global sensitivity Δq > 0: max |q(x,r) - q(x',r)| over neighboring x,x'.
    pub sensitivity: f64,
}

impl ExponentialConfig {
    /// Construct and validate an `ExponentialConfig`.
    ///
    /// # Errors
    /// Returns `NonPositiveEpsilon` or `NonPositiveSensitivity` on bad params.
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

/// Sample from the exponential mechanism over a discrete set of `scores`.
///
/// Selects outcome `i` with probability proportional to
/// `exp(ε · scores[i] / (2 · Δq))`.
///
/// Uses a numerically stable shifted log-sum-exp approach:
/// - Shift scores by `max(scores)` before exponentiation to prevent overflow.
/// - Perform a linear scan to select the output index.
///
/// # Errors
/// - `EmptyInput` if `scores` is empty.
/// - `NonPositiveEpsilon` / `NonPositiveSensitivity` if config is invalid.
pub fn exponential_sample(
    scores: &[f64],
    cfg: &ExponentialConfig,
    rng: &mut LcgRng,
) -> PrivacyResult<usize> {
    if scores.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if cfg.epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon));
    }
    if cfg.sensitivity <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(cfg.sensitivity));
    }

    let scale = cfg.epsilon / (2.0 * cfg.sensitivity);

    // Numerically stable: subtract max before exponentiation.
    let shift = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute unnormalized weights.
    let weights: Vec<f64> = scores
        .iter()
        .map(|&s| ((s - shift) * scale).exp())
        .collect();

    let total: f64 = weights.iter().sum();

    // Draw u ~ Uniform(0, total) and walk cumulative sum.
    let u = rng.next_f64() * total;
    let mut cumsum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumsum += w;
        if cumsum >= u {
            return Ok(i);
        }
    }

    // Numerical fallback: return the last index.
    Ok(weights.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_sample_valid_index() {
        let scores = vec![1.0, 2.0, 3.0, 0.0];
        let cfg = ExponentialConfig::new(1.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(42);
        for _ in 0..100 {
            let idx = exponential_sample(&scores, &cfg, &mut rng).expect("ok");
            assert!(idx < scores.len());
        }
    }

    #[test]
    fn test_exponential_sample_empty_error() {
        let cfg = ExponentialConfig::new(1.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(0);
        assert!(exponential_sample(&[], &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_exponential_sample_single() {
        let scores = vec![5.0];
        let cfg = ExponentialConfig::new(2.0, 0.5).expect("ok");
        let mut rng = LcgRng::new(1);
        let idx = exponential_sample(&scores, &cfg, &mut rng).expect("ok");
        assert_eq!(idx, 0);
    }
}

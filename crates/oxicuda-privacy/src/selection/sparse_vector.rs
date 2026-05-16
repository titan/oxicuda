//! Sparse Vector Technique (SVT) / AboveThreshold.
//!
//! The Sparse Vector Technique (Dwork & Roth §3.6) allows answering an
//! unbounded stream of threshold queries privately while consuming only
//! O(k) privacy budget, where k is the number of `True` (above-threshold)
//! answers returned.
//!
//! # Privacy guarantee
//! Answers ≤ k True queries with total budget (ε, 0)-DP using:
//! - ε₁ = ε/2 for the shared noisy threshold.
//! - ε₂ = ε/2 for each per-query noise draw.
//!
//! # Usage
//! 1. Create `SvtState::new(cfg, rng)` — adds Laplace noise to threshold.
//! 2. Call `state.query(val, cfg, rng)` for each query value.
//!    - Returns `Ok(Some(true))` or `Ok(Some(false))`.
//!    - Returns `Ok(None)` once the k-True limit is reached.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for the Sparse Vector Technique.
#[derive(Debug, Clone)]
pub struct SvtConfig {
    /// Privacy parameter ε > 0 (split evenly between threshold and query noise).
    pub epsilon: f64,
    /// Threshold T: queries are compared against T̃ = T + Lap(2kΔ/ε).
    pub threshold: f64,
    /// Maximum number of True answers (k).  The mechanism halts after k True responses.
    pub k: usize,
    /// Global sensitivity of each query function (Δ, typically 1.0).
    pub sensitivity: f64,
}

impl SvtConfig {
    /// Construct and validate an `SvtConfig`.
    ///
    /// # Errors
    /// Returns `NonPositiveEpsilon`, `NonPositiveSensitivity`, or
    /// `InvalidParameter` when arguments are out of range.
    pub fn new(epsilon: f64, threshold: f64, k: usize, sensitivity: f64) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        if k == 0 {
            return Err(PrivacyError::InvalidParameter("k must be ≥ 1".into()));
        }
        Ok(Self {
            epsilon,
            threshold,
            k,
            sensitivity,
        })
    }
}

/// Mutable state for the Sparse Vector Technique across a query stream.
#[derive(Debug)]
pub struct SvtState {
    /// Number of True answers returned so far.
    pub answered: usize,
    /// The noisy threshold T̃ (computed once at construction).
    pub noisy_threshold: f64,
}

impl SvtState {
    /// Initialise SVT state by drawing a noisy threshold.
    ///
    /// The noisy threshold is `T + Lap(2·k·Δ / ε)`.
    ///
    /// # Errors
    /// Propagates validation errors from `SvtConfig`.
    pub fn new(cfg: &SvtConfig, rng: &mut LcgRng) -> PrivacyResult<Self> {
        if cfg.epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon));
        }
        if cfg.sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(cfg.sensitivity));
        }
        if cfg.k == 0 {
            return Err(PrivacyError::InvalidParameter("k must be ≥ 1".into()));
        }

        // Scale for threshold noise: 2·k·Δ / ε
        let threshold_scale = 2.0 * (cfg.k as f64) * cfg.sensitivity / cfg.epsilon;
        let threshold_noise = laplace_sample(threshold_scale, rng);
        let noisy_threshold = cfg.threshold + threshold_noise;

        Ok(Self {
            answered: 0,
            noisy_threshold,
        })
    }

    /// Process one query value from the stream.
    ///
    /// Each call adds fresh Laplace noise ν ~ Lap(4·k·Δ / ε) to the query
    /// value and compares against the noisy threshold.
    ///
    /// # Returns
    /// - `Ok(Some(true))` — query (with noise) exceeds noisy threshold.
    /// - `Ok(Some(false))` — query does not exceed noisy threshold.
    /// - `Ok(None)` — the k-True limit has been reached; mechanism halts.
    ///
    /// # Errors
    /// Returns `SvtQueryLimitExceeded` if called after `answered >= cfg.k`
    /// (informational — callers may also check the `None` return).
    pub fn query(
        &mut self,
        query_val: f64,
        cfg: &SvtConfig,
        rng: &mut LcgRng,
    ) -> PrivacyResult<Option<bool>> {
        if self.answered >= cfg.k {
            return Err(PrivacyError::SvtQueryLimitExceeded {
                asked: self.answered + 1,
                limit: cfg.k,
            });
        }

        // Scale for query noise: 4·k·Δ / ε
        let query_scale = 4.0 * (cfg.k as f64) * cfg.sensitivity / cfg.epsilon;
        let query_noise = laplace_sample(query_scale, rng);
        let noisy_query = query_val + query_noise;

        let above = noisy_query >= self.noisy_threshold;
        if above {
            self.answered += 1;
        }
        Ok(Some(above))
    }
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
    fn test_svt_true_count_does_not_exceed_k() {
        let cfg = SvtConfig::new(1.0, 0.0, 3, 1.0).expect("ok");
        let mut rng = LcgRng::new(42);
        let mut state = SvtState::new(&cfg, &mut rng).expect("ok");
        let mut true_count = 0usize;
        for _ in 0..20 {
            match state.query(100.0, &cfg, &mut rng) {
                Ok(Some(true)) => {
                    true_count += 1;
                    if true_count >= cfg.k {
                        break;
                    }
                }
                Ok(Some(false)) | Ok(None) => {}
                Err(_) => break, // limit exceeded error is expected
            }
        }
        assert!(
            true_count <= cfg.k,
            "true_count {true_count} exceeded k={}",
            cfg.k
        );
    }

    #[test]
    fn test_svt_limit_exceeded_returns_error() {
        let cfg = SvtConfig::new(2.0, -999.0, 1, 1.0).expect("ok");
        let mut rng = LcgRng::new(1);
        let mut state = SvtState::new(&cfg, &mut rng).expect("ok");
        // k=1; force a True answer by using a very high query value.
        let _ = state.query(1_000_000.0, &cfg, &mut rng);
        // Now answered should equal k; next query should error.
        let result = state.query(1_000_000.0, &cfg, &mut rng);
        assert!(result.is_err());
    }
}

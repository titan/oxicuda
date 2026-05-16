//! Generalized Randomized Response (GRR) for categorical local DP.
//!
//! Reference: Kairouz, Oh & Viswanath (2016), "Extremal Mechanisms for
//! Local Differential Privacy", JMLR.
//!
//! # Protocol
//! For k-ary categorical input v ∈ {0, …, k-1}:
//!
//! - P(output = v | input = v) = e^ε / (e^ε + k − 1)      (report truth)
//! - P(output = v' ≠ v | input = v) = 1 / (e^ε + k − 1)   (report false uniform)
//!
//! This achieves (ε, 0)-LDP.
//!
//! # Frequency estimation
//! Unbiased estimator for frequency of value v:
//!
//! `f̂_v = (count_v/n − q) / (p − q)`
//!
//! where p = e^ε/(e^ε+k−1), q = 1/(e^ε+k−1).

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for GRR local DP.
#[derive(Debug, Clone)]
pub struct GrrConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
    /// Domain size k ≥ 2.
    pub k: usize,
}

impl GrrConfig {
    /// Construct and validate a `GrrConfig`.
    ///
    /// # Errors
    /// Returns `NonPositiveEpsilon` if `epsilon ≤ 0` or `InvalidParameter` if `k < 2`.
    pub fn new(epsilon: f64, k: usize) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if k < 2 {
            return Err(PrivacyError::InvalidParameter(
                "domain size k must be ≥ 2".into(),
            ));
        }
        Ok(Self { epsilon, k })
    }

    /// Probability of reporting the truth: p = e^ε / (e^ε + k − 1).
    #[must_use]
    fn p(&self) -> f64 {
        let exp_eps = self.epsilon.exp();
        exp_eps / (exp_eps + (self.k - 1) as f64)
    }

    /// Probability of reporting any false value: q = 1 / (e^ε + k − 1).
    #[must_use]
    fn q(&self) -> f64 {
        let exp_eps = self.epsilon.exp();
        1.0 / (exp_eps + (self.k - 1) as f64)
    }
}

/// Encode a categorical input via GRR.
///
/// With probability p reports the true value; otherwise reports a uniform
/// random value from {0, …, k-1} \ {input}.
///
/// # Errors
/// - `IndexOutOfRange` if `input ≥ k`.
/// - `NonPositiveEpsilon` / `InvalidParameter` from config validation.
pub fn grr_encode(input: usize, cfg: &GrrConfig, rng: &mut LcgRng) -> PrivacyResult<usize> {
    if input >= cfg.k {
        return Err(PrivacyError::IndexOutOfRange(input, cfg.k));
    }
    let p = cfg.p();
    let u = rng.next_f64();
    if u < p {
        // Report truth.
        return Ok(input);
    }
    // Report a uniform random false value (uniformly from k-1 remaining).
    let false_idx = rng.next_u64() as usize % (cfg.k - 1);
    // Map: if false_idx >= input, shift by 1 to avoid the true value.
    let output = if false_idx < input {
        false_idx
    } else {
        false_idx + 1
    };
    Ok(output)
}

/// Estimate value frequencies from GRR reports.
///
/// Returns a vector of length k with unbiased frequency estimates.
/// Each estimate may be slightly negative due to noise; clamp as needed by
/// the caller for downstream use.
///
/// `f̂_v = (count_v/n − q) / (p − q)`
///
/// # Errors
/// - `EmptyInput` if `reports` is empty.
/// - `IndexOutOfRange` if any report value ≥ k.
pub fn grr_estimate_frequency(reports: &[usize], cfg: &GrrConfig) -> PrivacyResult<Vec<f64>> {
    if reports.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }

    let n = reports.len() as f64;
    let p = cfg.p();
    let q = cfg.q();
    let denom = p - q;

    let mut counts = vec![0usize; cfg.k];
    for &r in reports {
        if r >= cfg.k {
            return Err(PrivacyError::IndexOutOfRange(r, cfg.k));
        }
        counts[r] += 1;
    }

    let freqs = counts.iter().map(|&c| (c as f64 / n - q) / denom).collect();

    Ok(freqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grr_encode_in_range() {
        let cfg = GrrConfig::new(2.0, 4).expect("ok");
        let mut rng = LcgRng::new(42);
        for input in 0..4 {
            for _ in 0..50 {
                let out = grr_encode(input, &cfg, &mut rng).expect("ok");
                assert!(out < 4);
            }
        }
    }

    #[test]
    fn test_grr_frequency_estimate_unbiased() {
        // All reports are 0; with n=10000 and high ε the estimate of f(0) should be near 1.
        let cfg = GrrConfig::new(5.0, 3).expect("ok");
        let mut rng = LcgRng::new(1);
        let n = 10_000;
        let reports: Vec<usize> = (0..n)
            .map(|_| grr_encode(0, &cfg, &mut rng).expect("ok"))
            .collect();
        let freqs = grr_estimate_frequency(&reports, &cfg).expect("ok");
        // f̂(0) should be close to 1.0.
        assert!(
            (freqs[0] - 1.0).abs() < 0.1,
            "f̂(0) = {}, expected ≈ 1.0",
            freqs[0]
        );
    }

    #[test]
    fn test_grr_out_of_range_error() {
        let cfg = GrrConfig::new(1.0, 3).expect("ok");
        let mut rng = LcgRng::new(0);
        assert!(grr_encode(3, &cfg, &mut rng).is_err());
    }
}

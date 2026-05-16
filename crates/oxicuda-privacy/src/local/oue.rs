//! Optimized Unary Encoding (OUE) for local DP frequency estimation.
//!
//! Reference: Wang, Blocki, Li & Jha (2017), "Locally Differentially Private
//! Protocols for Frequency Estimation", USENIX Security 2017.
//!
//! # Protocol
//! For input v ∈ {0, …, k-1}:
//! 1. Form a one-hot bit vector B* of length k: B*`[v]` = 1, all others 0.
//! 2. For each bit i:
//!    - If B*`[i]` = 1 (i.e., i == v): output 1 with prob ½, else 0.
//!    - If B*`[i]` = 0 (i.e., i ≠ v): output 1 with prob p = 1/(e^ε + 1).
//!
//! The resulting bit vector B̃ of length k is reported.
//!
//! # Frequency estimation
//! Unbiased estimator:
//!
//! `f̂_v = (Σᵢ B̃_v(i) / n − p) / (½ − p)`
//!
//! where p = 1/(e^ε + 1), the sum is over all n reports, and B̃_v(i) is
//! bit v of the i-th report.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for OUE local DP.
#[derive(Debug, Clone)]
pub struct OueConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
    /// Domain size k ≥ 2.
    pub k: usize,
}

impl OueConfig {
    /// Construct and validate an `OueConfig`.
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

    /// False-bit flip probability p = 1 / (e^ε + 1).
    #[must_use]
    fn p_flip(&self) -> f64 {
        1.0 / (self.epsilon.exp() + 1.0)
    }
}

/// Encode an input value via OUE.
///
/// Returns a bit vector of length k (values 0 or 1).
///
/// # Errors
/// - `IndexOutOfRange` if `input ≥ k`.
pub fn oue_encode(input: usize, cfg: &OueConfig, rng: &mut LcgRng) -> PrivacyResult<Vec<u8>> {
    if input >= cfg.k {
        return Err(PrivacyError::IndexOutOfRange(input, cfg.k));
    }

    let p_flip = cfg.p_flip();
    let mut bits = Vec::with_capacity(cfg.k);

    for i in 0..cfg.k {
        let u = rng.next_f64();
        let bit = if i == input {
            // True bit: output 1 with prob 1/2.
            u < 0.5
        } else {
            // False bit: output 1 with prob p_flip.
            u < p_flip
        };
        bits.push(bit as u8);
    }

    Ok(bits)
}

/// Estimate value frequencies from OUE reports.
///
/// Returns a vector of length k with unbiased frequency estimates.
///
/// `f̂_v = (mean_v − p) / (½ − p)` where `mean_v = Σ B̃_v(i) / n`.
///
/// # Errors
/// - `EmptyInput` if `reports` is empty.
/// - `DimensionMismatch` if any report has length ≠ k.
pub fn oue_estimate_frequency(reports: &[Vec<u8>], cfg: &OueConfig) -> PrivacyResult<Vec<f64>> {
    if reports.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    let n = reports.len() as f64;
    let p = cfg.p_flip();
    let denom = 0.5 - p;

    // Sum bits per position.
    let mut sums = vec![0u64; cfg.k];
    for report in reports {
        if report.len() != cfg.k {
            return Err(PrivacyError::DimensionMismatch {
                expected: cfg.k,
                got: report.len(),
            });
        }
        for (j, &b) in report.iter().enumerate() {
            sums[j] += b as u64;
        }
    }

    let freqs = sums.iter().map(|&s| (s as f64 / n - p) / denom).collect();

    Ok(freqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oue_encode_length() {
        let cfg = OueConfig::new(2.0, 5).expect("ok");
        let mut rng = LcgRng::new(42);
        let bits = oue_encode(2, &cfg, &mut rng).expect("ok");
        assert_eq!(bits.len(), 5);
        for &b in &bits {
            assert!(b == 0 || b == 1);
        }
    }

    #[test]
    fn test_oue_estimate_unbiased() {
        let cfg = OueConfig::new(3.0, 4).expect("ok");
        let mut rng = LcgRng::new(99);
        let n = 10_000;
        let reports: Vec<Vec<u8>> = (0..n)
            .map(|_| oue_encode(1, &cfg, &mut rng).expect("ok"))
            .collect();
        let freqs = oue_estimate_frequency(&reports, &cfg).expect("ok");
        // f̂(1) should be close to 1.0.
        assert!(
            (freqs[1] - 1.0).abs() < 0.1,
            "f̂(1) = {}, expected ≈ 1.0",
            freqs[1]
        );
        // f̂(j≠1) should be close to 0.
        for (j, &f) in freqs.iter().enumerate() {
            if j != 1 {
                assert!(f.abs() < 0.15, "f̂({j}) = {f}, expected ≈ 0");
            }
        }
    }

    #[test]
    fn test_oue_out_of_range_error() {
        let cfg = OueConfig::new(1.0, 3).expect("ok");
        let mut rng = LcgRng::new(0);
        assert!(oue_encode(3, &cfg, &mut rng).is_err());
    }
}

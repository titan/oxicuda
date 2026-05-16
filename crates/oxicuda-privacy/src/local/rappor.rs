//! RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response.
//!
//! Reference: Erlingsson, Pihur & Korolova (2014), "RAPPOR: Randomized
//! Aggregatable Privacy-Preserving Ordinal Response", CCS 2014.
//!
//! # Simplified single-level RAPPOR
//! Full RAPPOR uses two stages of randomized response and a Bloom filter for
//! cohort-based frequency estimation.  This implementation provides a
//! single-level RAPPOR that:
//! 1. Hashes the input to `num_hashes` bit positions in a bitvector of length k.
//! 2. Sets those bits to 1 (the "Bloom filter" encoding).
//! 3. Flips each bit independently with probability `f/2 = 1/(e^(ε/k) + 1)`
//!    (per-bit randomized response).
//!
//! # Hash function
//! Uses a deterministic linear hash: `h_j(v) = (v * primes[j]) % k` where
//! `primes` are small fixed primes for each hash index.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// First few primes used as hash multipliers.
const HASH_PRIMES: [usize; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

/// Configuration for single-level RAPPOR.
#[derive(Debug, Clone)]
pub struct RapporConfig {
    /// Total privacy budget ε for the single-stage encoding.
    pub epsilon: f64,
    /// Bitvector / domain width k (number of hash buckets).
    pub k: usize,
    /// Number of hash functions (determines Bloom filter density).
    pub num_hashes: usize,
}

impl RapporConfig {
    /// Construct and validate a `RapporConfig`.
    ///
    /// # Errors
    /// Returns appropriate `PrivacyError` for invalid parameters.
    pub fn new(epsilon: f64, k: usize, num_hashes: usize) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if k < 2 {
            return Err(PrivacyError::InvalidParameter("k must be ≥ 2".into()));
        }
        if num_hashes == 0 {
            return Err(PrivacyError::InvalidParameter(
                "num_hashes must be ≥ 1".into(),
            ));
        }
        if num_hashes > HASH_PRIMES.len() {
            return Err(PrivacyError::InvalidParameter(format!(
                "num_hashes must be ≤ {}, got {num_hashes}",
                HASH_PRIMES.len()
            )));
        }
        Ok(Self {
            epsilon,
            k,
            num_hashes,
        })
    }

    /// Per-bit flip probability: p_flip = 1 / (e^(ε/k) + 1).
    #[must_use]
    fn p_flip(&self) -> f64 {
        let eps_per_bit = self.epsilon / self.k as f64;
        1.0 / (eps_per_bit.exp() + 1.0)
    }

    /// Hash the input to bit position j: h_j(v) = (v * prime_j + j) % k.
    fn hash(&self, input: usize, hash_idx: usize) -> usize {
        let prime = HASH_PRIMES[hash_idx];
        (input.wrapping_mul(prime).wrapping_add(hash_idx)) % self.k
    }
}

/// RAPPOR encode: Bloom-filter hash + per-bit randomized response.
///
/// # Arguments
/// - `input`: integer value in `[0, domain_size)`.  The domain is unrestricted
///   as long as the hashing distributes values across `[0, k)`.
/// - `cfg`: RAPPOR configuration.
/// - `rng`: LCG for randomization.
///
/// # Returns
/// A bit vector of length k (values 0 or 1).
///
/// # Errors
/// Propagates config validation errors.
pub fn rappor_encode(input: usize, cfg: &RapporConfig, rng: &mut LcgRng) -> PrivacyResult<Vec<u8>> {
    let p_flip = cfg.p_flip();

    // Step 1: Bloom-filter encoding — set bits at hash positions.
    let mut bloom = vec![0u8; cfg.k];
    for h in 0..cfg.num_hashes {
        let pos = cfg.hash(input, h);
        bloom[pos] = 1;
    }

    // Step 2: Per-bit randomized response.
    // For each bit b:
    //   - Report b with prob (1 − p_flip)
    //   - Report 1−b with prob p_flip
    let mut result = Vec::with_capacity(cfg.k);
    for &b in &bloom {
        let u = rng.next_f64();
        let flipped = if u < p_flip { 1 - b } else { b };
        result.push(flipped);
    }

    Ok(result)
}

/// Decode RAPPOR reports to estimate frequency of each hash bucket.
///
/// Returns an unbiased frequency estimate for each of the k hash buckets.
/// Note: mapping back from bucket frequencies to individual value frequencies
/// requires a least-squares or LASSO solver (not implemented here; use
/// the raw bucket-frequency vector for downstream estimation).
///
/// The estimator for bucket v is:
/// `f̂_v = (count_v/n − p_flip) / (1 − 2·p_flip)`
///
/// # Errors
/// - `EmptyInput` if `reports` is empty.
/// - `DimensionMismatch` if any report has length ≠ k.
pub fn rappor_decode_frequency(reports: &[Vec<u8>], cfg: &RapporConfig) -> PrivacyResult<Vec<f64>> {
    if reports.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }

    let n = reports.len() as f64;
    let p_flip = cfg.p_flip();
    let denom = 1.0 - 2.0 * p_flip;

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

    let freqs = sums
        .iter()
        .map(|&s| (s as f64 / n - p_flip) / denom)
        .collect();

    Ok(freqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rappor_encode_length_and_values() {
        let cfg = RapporConfig::new(2.0, 8, 2).expect("ok");
        let mut rng = LcgRng::new(42);
        let bits = rappor_encode(3, &cfg, &mut rng).expect("ok");
        assert_eq!(bits.len(), 8);
        for &b in &bits {
            assert!(b == 0 || b == 1, "bit must be 0 or 1, got {b}");
        }
    }

    #[test]
    fn test_rappor_decode_non_empty() {
        let cfg = RapporConfig::new(1.0, 4, 1).expect("ok");
        let mut rng = LcgRng::new(7);
        let n = 1000;
        let reports: Vec<Vec<u8>> = (0..n)
            .map(|_| rappor_encode(0, &cfg, &mut rng).expect("ok"))
            .collect();
        let freqs = rappor_decode_frequency(&reports, &cfg).expect("ok");
        assert_eq!(freqs.len(), 4);
        // The hash bucket for input=0 should have elevated frequency.
        // Just verify values are finite.
        for &f in &freqs {
            assert!(f.is_finite(), "frequency must be finite");
        }
    }

    #[test]
    fn test_rappor_decode_empty_error() {
        let cfg = RapporConfig::new(1.0, 4, 1).expect("ok");
        assert!(rappor_decode_frequency(&[], &cfg).is_err());
    }

    #[test]
    fn test_rappor_num_hashes_zero_error() {
        assert!(RapporConfig::new(1.0, 4, 0).is_err());
    }

    #[test]
    fn test_rappor_decode_unbiased_for_single_hash() {
        // With 1 hash, high epsilon, all inputs=0: the hash bucket h(0,0)=0 should
        // have high frequency.
        let cfg = RapporConfig::new(10.0, 8, 1).expect("ok");
        let mut rng = LcgRng::new(55);
        let n = 5_000;
        let reports: Vec<Vec<u8>> = (0..n)
            .map(|_| rappor_encode(0, &cfg, &mut rng).expect("ok"))
            .collect();
        let freqs = rappor_decode_frequency(&reports, &cfg).expect("ok");
        let hash_pos = cfg.hash(0, 0);
        // The bucket for input=0 should have frequency close to 1.
        assert!(
            freqs[hash_pos] > 0.7,
            "bucket {} frequency = {}, expected > 0.7",
            hash_pos,
            freqs[hash_pos]
        );
    }
}

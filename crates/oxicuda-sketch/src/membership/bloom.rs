//! Bloom filter (Bloom 1970).
//!
//! Bit array of size `m` with `k` independent hash functions.
//! Insert: set the `k` bit positions.
//! Contains: return true iff all `k` bit positions are set.
//! False-positive rate after `n` insertions: `(1 - exp(-k*n/m))^k`.
//! Optimal `k = (m/n) * ln(2)`.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

/// Classic Bloom filter.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    pub m: usize,
    pub k: usize,
    pub bits: Vec<u64>,
    pub seed_base: u64,
}

impl BloomFilter {
    /// Construct a Bloom filter with `m` bits and `k` hash functions.
    pub fn new(m: usize, k: usize, seed_base: u64) -> SketchResult<Self> {
        if m == 0 || k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(m,k)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let words = m.div_ceil(64);
        Ok(Self {
            m,
            k,
            bits: vec![0u64; words],
            seed_base,
        })
    }

    /// Choose Bloom filter parameters to hold `n_expected` items with target false-positive rate `p`.
    /// `m = -n * ln(p) / (ln 2)^2`, `k = (m/n) * ln 2`.
    pub fn with_expected_fp(n_expected: usize, p: f64, seed_base: u64) -> SketchResult<Self> {
        if !(0.0 < p && p < 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "p".to_string(),
                reason: "must be in (0,1)".to_string(),
            });
        }
        if n_expected == 0 {
            return Err(SketchError::InvalidParameter {
                name: "n_expected".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let ln2 = std::f64::consts::LN_2;
        let m = (-(n_expected as f64) * p.ln() / (ln2 * ln2)).ceil() as usize;
        let k = ((m as f64 / n_expected as f64) * ln2).ceil() as usize;
        Self::new(m.max(1), k.max(1), seed_base)
    }

    /// Compute the `k` bit positions for an item.
    fn positions(&self, x: u64) -> Vec<usize> {
        // Double hashing: h1 + i * h2 mod m for i = 0..k.
        let h1 = xxh3_64_u64(x, self.seed_base);
        let h2 = xxh3_64_u64(x, self.seed_base.wrapping_add(0x9E37_79B9_7F4A_7C15));
        (0..self.k)
            .map(|i| ((h1.wrapping_add((i as u64).wrapping_mul(h2))) as usize) % self.m)
            .collect()
    }

    /// Insert an item.
    pub fn insert(&mut self, x: u64) {
        for p in self.positions(x) {
            let w = p / 64;
            let b = p % 64;
            self.bits[w] |= 1u64 << b;
        }
    }

    /// Test membership.
    #[must_use]
    pub fn contains(&self, x: u64) -> bool {
        for p in self.positions(x) {
            let w = p / 64;
            let b = p % 64;
            if (self.bits[w] >> b) & 1 == 0 {
                return false;
            }
        }
        true
    }

    /// Number of set bits.
    #[must_use]
    pub fn popcount(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Approximate false-positive rate after `n` insertions: `(1 - exp(-k*n/m))^k`.
    #[must_use]
    pub fn expected_fp_rate(&self, n: usize) -> f64 {
        let exponent = -(self.k as f64) * (n as f64) / (self.m as f64);
        (1.0 - exponent.exp()).powi(self.k as i32)
    }

    /// Estimate the current load `n` from the popcount.
    /// `n_est = -(m/k) * ln(1 - popcount/m)`.
    #[must_use]
    pub fn estimate_load(&self) -> f64 {
        let pc = self.popcount() as f64;
        let fraction = pc / self.m as f64;
        if fraction >= 1.0 {
            return f64::INFINITY;
        }
        -(self.m as f64) / (self.k as f64) * (1.0 - fraction).ln()
    }

    /// Reset to empty.
    pub fn clear(&mut self) {
        for w in self.bits.iter_mut() {
            *w = 0;
        }
    }

    /// Bitwise-OR merge with another Bloom (must have same dimensions).
    pub fn merge(&mut self, other: &BloomFilter) -> SketchResult<()> {
        if self.m != other.m || self.k != other.k {
            return Err(SketchError::DimensionMismatch {
                a: self.m,
                b: other.m,
            });
        }
        for i in 0..self.bits.len() {
            self.bits[i] |= other.bits[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_constructs() {
        let bf = BloomFilter::new(1024, 4, 0).expect("ok");
        assert_eq!(bf.m, 1024);
        assert_eq!(bf.k, 4);
    }

    #[test]
    fn bloom_no_false_negatives() {
        let mut bf = BloomFilter::new(4096, 5, 0).expect("ok");
        for i in 0..500u64 {
            bf.insert(i);
        }
        for i in 0..500u64 {
            assert!(bf.contains(i), "missing inserted item {i}");
        }
    }

    #[test]
    fn bloom_fp_rate_low_with_room() {
        let mut bf = BloomFilter::with_expected_fp(1000, 0.01, 0).expect("ok");
        for i in 0..1000u64 {
            bf.insert(i);
        }
        // Test 5000 unseen items.
        let mut fp = 0usize;
        for i in 10_000..15_000u64 {
            if bf.contains(i) {
                fp += 1;
            }
        }
        let rate = fp as f64 / 5_000.0;
        assert!(rate < 0.05, "FP rate {rate} above 5%");
    }

    #[test]
    fn bloom_estimate_load() {
        let mut bf = BloomFilter::new(8192, 5, 0).expect("ok");
        for i in 0..500u64 {
            bf.insert(i);
        }
        let est = bf.estimate_load();
        let rel = (est - 500.0).abs() / 500.0;
        assert!(rel < 0.2, "load estimate rel-err {rel}");
    }

    #[test]
    fn bloom_merge() {
        let mut bf1 = BloomFilter::new(1024, 4, 11).expect("ok");
        let mut bf2 = BloomFilter::new(1024, 4, 11).expect("ok");
        bf1.insert(1);
        bf2.insert(2);
        bf1.merge(&bf2).expect("ok");
        assert!(bf1.contains(1));
        assert!(bf1.contains(2));
    }
}

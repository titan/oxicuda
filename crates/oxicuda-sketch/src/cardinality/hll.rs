//! HyperLogLog cardinality estimator (Flajolet, Fusy, Gandouet, Meunier 2007).
//!
//! Uses `m = 2^p` registers, each storing the maximum leading-zero count + 1
//! of the low (64 - p) bits of the hashed values that map to that register's bucket.
//! Estimate:
//!     E = alpha_m * m^2 / sum_j 2^{-M_j}
//! With small-range correction via linear counting and large-range correction
//! for full 32-bit hash overflow (not relevant here since we use 64-bit hashes).

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

/// HyperLogLog sketch with `m = 2^p` 8-bit registers (compact: each register fits in u8
/// because the max leading-zero-count + 1 of a 64-bit value is at most 64).
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    pub p: u32,
    pub m: usize,
    pub seed: u64,
    pub registers: Vec<u8>,
}

impl HyperLogLog {
    /// Create a new HLL with precision `p` (so `m = 2^p`). Must have `4 <= p <= 16`.
    pub fn new(p: u32, seed: u64) -> SketchResult<Self> {
        if !(4..=16).contains(&p) {
            return Err(SketchError::InvalidPrecision(p));
        }
        let m = 1usize << p;
        Ok(Self {
            p,
            m,
            seed,
            registers: vec![0u8; m],
        })
    }

    /// Insert a `u64` value into the sketch.
    pub fn add_u64(&mut self, x: u64) {
        let h = xxh3_64_u64(x, self.seed);
        self.add_hash(h);
    }

    /// Insert raw 64-bit hash directly (useful when caller already hashed).
    pub fn add_hash(&mut self, h: u64) {
        // Use the top `p` bits as the register index.
        let idx = (h >> (64 - self.p)) as usize;
        // Use the remaining (64 - p) bits to compute leading-zero count + 1.
        let w = (h << self.p) | (1u64 << (self.p.saturating_sub(1)));
        let leading_zeros = (w.leading_zeros() as u8) + 1;
        // Clamp to fit in u8 (always <= 64).
        let new_val = leading_zeros.min(64);
        if new_val > self.registers[idx] {
            self.registers[idx] = new_val;
        }
    }

    /// Estimate the number of distinct elements.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        let m = self.m as f64;
        let alpha = Self::alpha(self.m);
        // Raw harmonic-mean estimate.
        let mut sum = 0.0;
        let mut zero_count = 0usize;
        for &reg in &self.registers {
            sum += 2.0_f64.powi(-(reg as i32));
            if reg == 0 {
                zero_count += 1;
            }
        }
        let raw = alpha * m * m / sum;
        // Small-range correction: linear counting when raw is small enough.
        if raw <= 2.5 * m && zero_count > 0 {
            return m * (m / zero_count as f64).ln();
        }
        raw
    }

    /// Merge another HLL sketch into this one (must have same precision and seed).
    pub fn merge(&mut self, other: &HyperLogLog) -> SketchResult<()> {
        if self.p != other.p {
            return Err(SketchError::DimensionMismatch {
                a: self.p as usize,
                b: other.p as usize,
            });
        }
        for i in 0..self.m {
            if other.registers[i] > self.registers[i] {
                self.registers[i] = other.registers[i];
            }
        }
        Ok(())
    }

    /// Reset the sketch to empty.
    pub fn clear(&mut self) {
        for r in self.registers.iter_mut() {
            *r = 0;
        }
    }

    /// HyperLogLog alpha constant for given `m`. Reference values from Flajolet 2007.
    fn alpha(m: usize) -> f64 {
        match m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m as f64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hll_constructs_correctly() {
        let h = HyperLogLog::new(10, 0).expect("ok");
        assert_eq!(h.m, 1024);
        assert_eq!(h.registers.len(), 1024);
    }

    #[test]
    fn hll_invalid_precision() {
        assert!(HyperLogLog::new(2, 0).is_err());
        assert!(HyperLogLog::new(20, 0).is_err());
    }

    #[test]
    fn hll_empty_estimate_is_small() {
        let h = HyperLogLog::new(10, 0).expect("ok");
        let e = h.estimate();
        assert!(e < 1.0, "empty HLL estimate = {e}");
    }

    #[test]
    fn hll_accuracy_distinct() {
        let mut h = HyperLogLog::new(14, 0).expect("ok");
        let n: u64 = 10_000;
        for i in 0..n {
            h.add_u64(i);
        }
        let e = h.estimate();
        let rel = (e - n as f64).abs() / n as f64;
        assert!(rel < 0.05, "expected within 5% of {n}, got {e}");
    }

    #[test]
    fn hll_merge_works() {
        let mut h1 = HyperLogLog::new(10, 0).expect("ok");
        let mut h2 = HyperLogLog::new(10, 0).expect("ok");
        for i in 0..500u64 {
            h1.add_u64(i);
        }
        for i in 500..1000u64 {
            h2.add_u64(i);
        }
        h1.merge(&h2).expect("ok");
        let e = h1.estimate();
        let rel = (e - 1000.0).abs() / 1000.0;
        assert!(rel < 0.15, "merged HLL relative error {rel}");
    }

    #[test]
    fn hll_duplicates_dont_inflate() {
        let mut h = HyperLogLog::new(12, 0).expect("ok");
        for _ in 0..1000 {
            h.add_u64(42);
        }
        let e = h.estimate();
        assert!(e < 5.0, "duplicate-only HLL estimate = {e}");
    }
}

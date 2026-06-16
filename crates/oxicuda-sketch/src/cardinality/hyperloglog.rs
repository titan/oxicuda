//! HyperLogLog cardinality estimator for byte-slice items (Flajolet 2007).
//!
//! Unlike [`crate::cardinality::hll::HyperLogLog`] which accepts `u64` integers,
//! this implementation accepts arbitrary byte slices (`&[u8]`) using `xxh3_64`.
//!
//! ## Algorithm
//!
//! Given `m = 2^b` registers:
//! 1. Hash item: `h = xxh3_64(item, seed)`
//! 2. Register index: `idx = h >> (64 - b)`  (top `b` bits)
//! 3. rho: leading-zero count of the remaining `(64 - b)` bits + 1
//!    `rho = leading_zeros(h << b | guard_bit) + 1`, clamped to 64
//! 4. Update: `registers[idx] = max(registers[idx], rho)`
//! 5. Estimate: `alpha_m * m^2 / sum_j 2^{-M_j}` with small-range correction
//!    via linear counting when `raw < 2.5 * m` and there are zero registers.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for a HyperLogLog sketch.
#[derive(Debug, Clone)]
pub struct HllConfig {
    /// Number of bits used for register indexing.  Uses `2^b` registers.
    /// Valid range: [4, 16].
    pub b: usize,
}

// ── Core struct ───────────────────────────────────────────────────────────────

/// HyperLogLog cardinality estimator (Flajolet, Fusy, Gandouet, Meunier 2007).
///
/// Accepts arbitrary byte-slice items hashed with `xxh3_64`.
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    /// `m = 2^b` 8-bit registers.  Each register holds the maximum rho seen
    /// for items that hash to that register index.
    registers: Vec<u8>,
    config: HllConfig,
    /// Seed used for xxh3_64; fixed at construction time.
    seed: u64,
}

// ── Helper constants / functions ──────────────────────────────────────────────

/// Alpha bias-correction constant for `m` registers (Flajolet 2007, §4).
#[inline]
fn alpha_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

// ── Implementation ────────────────────────────────────────────────────────────

impl HyperLogLog {
    /// Create a new HyperLogLog with `2^b` registers.
    ///
    /// The number of bits `b` must be in `[4, 16]`.
    ///
    /// # Errors
    /// Returns [`SketchError::InvalidPrecision`] if `b` is outside `[4, 16]`.
    pub fn new(config: HllConfig) -> SketchResult<Self> {
        if !(4..=16).contains(&config.b) {
            return Err(SketchError::InvalidPrecision(config.b as u32));
        }
        let m = 1usize << config.b;
        Ok(Self {
            registers: vec![0u8; m],
            config,
            seed: 0,
        })
    }

    /// Create a HyperLogLog with an explicit hash seed.
    ///
    /// # Errors
    /// Returns [`SketchError::InvalidPrecision`] if `b` is outside `[4, 16]`.
    pub fn with_seed(config: HllConfig, seed: u64) -> SketchResult<Self> {
        let mut hll = Self::new(config)?;
        hll.seed = seed;
        Ok(hll)
    }

    /// Number of registers `m = 2^b`.
    #[must_use]
    pub fn n_registers(&self) -> usize {
        1 << self.config.b
    }

    // ── Core operations ────────────────────────────────────────────────────────

    /// Add a byte-slice item to the sketch.
    pub fn add(&mut self, item: &[u8]) {
        let b = self.config.b;
        let h = xxh3_64(item, self.seed);

        // Top b bits → register index.
        let idx = (h >> (64 - b)) as usize;

        // Remaining (64 - b) bits with a guard bit to handle the all-zeros edge case.
        // We shift h left by b bits; the low b bits vacated are irrelevant.
        // Setting bit (b-1) of the low half ensures leading_zeros terminates even if
        // all remaining bits are 0 (otherwise rho would be 64 - b + 1 which is fine,
        // but the guard bit gives a deterministic upper bound of 64 - b + 1 ≤ 64).
        let shifted = (h << b) | (1u64 << b.saturating_sub(1));
        let rho = (shifted.leading_zeros() + 1).min(64) as u8;

        if rho > self.registers[idx] {
            self.registers[idx] = rho;
        }
    }

    /// Estimate the number of distinct items with bias correction.
    ///
    /// Uses small-range linear-counting correction when `raw < 2.5 * m` and
    /// there are zero registers.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        let m = self.n_registers();
        let m_f = m as f64;
        let alpha = alpha_m(m);

        let mut harmonic_sum = 0.0_f64;
        let mut zero_count = 0usize;

        for &reg in &self.registers {
            harmonic_sum += 2.0_f64.powi(-(reg as i32));
            if reg == 0 {
                zero_count += 1;
            }
        }

        let raw = alpha * m_f * m_f / harmonic_sum;

        // Small-range correction (linear counting).
        if raw <= 2.5 * m_f && zero_count > 0 {
            return m_f * (m_f / zero_count as f64).ln();
        }

        // Large-range correction is not required for 64-bit hashes
        // (would only apply for 32-bit hashes near 2^32 / 30).
        raw
    }

    /// Merge another HyperLogLog sketch into this one (component-wise max).
    ///
    /// Both sketches must have the same `b`; seeds need not match (the merged
    /// sketch inherits `self`'s seed, which only matters for future `add` calls).
    ///
    /// # Errors
    /// Returns [`SketchError::DimensionMismatch`] if `b` values differ.
    pub fn merge(&mut self, other: &HyperLogLog) -> SketchResult<()> {
        if self.config.b != other.config.b {
            return Err(SketchError::DimensionMismatch {
                a: self.config.b,
                b: other.config.b,
            });
        }
        let m = self.n_registers();
        for i in 0..m {
            if other.registers[i] > self.registers[i] {
                self.registers[i] = other.registers[i];
            }
        }
        Ok(())
    }

    /// Reset all registers to zero.
    pub fn clear(&mut self) {
        for r in self.registers.iter_mut() {
            *r = 0;
        }
    }

    /// Return the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &HllConfig {
        &self.config
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn item(i: u64) -> Vec<u8> {
        i.to_le_bytes().to_vec()
    }

    // 1. Estimate is within 20% of 1000 after inserting 1000 distinct items.
    #[test]
    fn estimate_approx_correct_1000_items() {
        let mut hll = HyperLogLog::new(HllConfig { b: 12 }).expect("constructs");
        for i in 0u64..1000 {
            hll.add(&item(i));
        }
        let est = hll.estimate();
        let rel_err = (est - 1000.0).abs() / 1000.0;
        assert!(
            rel_err < 0.20,
            "estimate {est} outside 20% of 1000 (rel_err={rel_err})"
        );
    }

    // 2. Estimate is always non-negative.
    #[test]
    fn estimate_never_negative() {
        let mut hll = HyperLogLog::new(HllConfig { b: 8 }).expect("constructs");
        for i in 0u64..200 {
            hll.add(&item(i));
        }
        assert!(hll.estimate() >= 0.0, "estimate should never be negative");
    }

    // 3. Minimum precision b=4 constructs and produces a non-negative estimate.
    #[test]
    fn b_4_constructs_and_works() {
        let mut hll = HyperLogLog::new(HllConfig { b: 4 }).expect("constructs");
        assert_eq!(hll.n_registers(), 16);
        for i in 0u64..100 {
            hll.add(&item(i));
        }
        assert!(hll.estimate() >= 0.0);
    }

    // 4. Estimate after 1000 distinct adds is strictly greater than after 10 distinct adds.
    #[test]
    fn add_increases_estimate_monotonically() {
        let mut hll = HyperLogLog::new(HllConfig { b: 10 }).expect("constructs");
        for i in 0u64..10 {
            hll.add(&item(i));
        }
        let est_10 = hll.estimate();
        for i in 10u64..1000 {
            hll.add(&item(i));
        }
        let est_1000 = hll.estimate();
        assert!(
            est_1000 > est_10,
            "estimate should grow: est_10={est_10}, est_1000={est_1000}"
        );
    }

    // 5. Merge union property: merged estimate >= individual estimates.
    #[test]
    fn merge_satisfies_union_lower_bound() {
        let mut a = HyperLogLog::new(HllConfig { b: 10 }).expect("constructs");
        let mut b = HyperLogLog::new(HllConfig { b: 10 }).expect("constructs");
        for i in 0u64..500 {
            a.add(&item(i));
        }
        for i in 500u64..1000 {
            b.add(&item(i));
        }
        let est_a = a.estimate();
        let est_b = b.estimate();
        let mut merged = a.clone();
        merged.merge(&b).expect("same b");
        let est_merged = merged.estimate();
        // Union estimate must be >= both individual estimates (with 5% tolerance for noise).
        let lower = est_a.max(est_b) * 0.95;
        assert!(
            est_merged >= lower,
            "merged estimate {est_merged} < max(A={est_a}, B={est_b}) * 0.95"
        );
    }

    // 6. Fresh (empty) HLL estimate is 0 or near 0.
    #[test]
    fn empty_estimate_is_near_zero() {
        let hll = HyperLogLog::new(HllConfig { b: 10 }).expect("constructs");
        let est = hll.estimate();
        assert!(est < 1.0, "empty HLL estimate should be near 0, got {est}");
    }

    // 7. Insert 5000 distinct items; estimate within 30%.
    #[test]
    fn large_cardinality_within_30_percent() {
        let mut hll = HyperLogLog::new(HllConfig { b: 12 }).expect("constructs");
        for i in 0u64..5000 {
            hll.add(&item(i));
        }
        let est = hll.estimate();
        let rel_err = (est - 5000.0).abs() / 5000.0;
        assert!(
            rel_err < 0.30,
            "estimate {est} outside 30% of 5000 (rel_err={rel_err})"
        );
    }

    // 8. Merging with an exact copy of self leaves the estimate approximately the same.
    #[test]
    fn merge_with_self_copy_is_idempotent() {
        let mut hll = HyperLogLog::new(HllConfig { b: 10 }).expect("constructs");
        for i in 0u64..500 {
            hll.add(&item(i));
        }
        let est_before = hll.estimate();
        let copy = hll.clone();
        hll.merge(&copy).expect("same b");
        let est_after = hll.estimate();
        // Merging with itself is idempotent: max-of-same = same.
        let rel_diff = (est_after - est_before).abs() / est_before.max(1.0);
        assert!(
            rel_diff < 0.01,
            "merge with self changed estimate: before={est_before}, after={est_after}"
        );
    }

    // 9. b=0 (and b=2) must return an error.
    #[test]
    fn invalid_b_returns_error() {
        assert!(
            HyperLogLog::new(HllConfig { b: 0 }).is_err(),
            "b=0 should fail"
        );
        assert!(
            HyperLogLog::new(HllConfig { b: 2 }).is_err(),
            "b=2 should fail"
        );
        assert!(
            HyperLogLog::new(HllConfig { b: 17 }).is_err(),
            "b=17 should fail"
        );
    }
}

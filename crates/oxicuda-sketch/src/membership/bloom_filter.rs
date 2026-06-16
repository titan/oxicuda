//! Bloom filter for byte-slice items (Bloom 1970).
//!
//! Unlike [`crate::membership::bloom::BloomFilter`] which accepts `u64` integers,
//! this implementation accepts arbitrary byte slices (`&[u8]`) and uses double hashing
//! with the xxh3_64 byte-slice hasher:
//!
//!   position_i = (h1 + i * h2) mod m
//!
//! where `h1 = xxh3_64(item, 0)` and `h2 = xxh3_64(item, 1)`.
//!
//! **False-positive rate** after `n` insertions: `(1 - exp(-k*n/m))^k`.
//! **Optimal** bit count: `m = -n*ln(fpr) / ln(2)^2`,  hash count: `k = m/n * ln(2)`.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for a byte-slice Bloom filter.
#[derive(Debug, Clone)]
pub struct BloomConfig {
    /// Number of bits in the filter.
    pub m: usize,
    /// Number of independent hash functions (simulated by double hashing).
    pub k: usize,
}

// ── Core struct ───────────────────────────────────────────────────────────────

/// Bloom filter for arbitrary byte-slice items (Bloom 1970).
///
/// Items are hashed with `xxh3_64` using two seeds (0 and 1) to simulate `k`
/// independent hash functions via the Kirsch-Mitzenmacher double-hashing scheme.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Bit array stored as packed 64-bit words (word i covers bits [64*i, 64*i+63]).
    bits: Vec<u64>,
    config: BloomConfig,
    /// Count of items inserted (used for theoretical FPR calculation).
    n_inserted: usize,
}

// ── Implementation ────────────────────────────────────────────────────────────

impl BloomFilter {
    /// Construct with explicit `m` bits and `k` hash functions.
    ///
    /// # Errors
    /// Returns [`SketchError::InvalidParameter`] if `m == 0` or `k == 0`.
    pub fn new(config: BloomConfig) -> SketchResult<Self> {
        if config.m == 0 {
            return Err(SketchError::InvalidParameter {
                name: "m".to_string(),
                reason: "number of bits must be positive".to_string(),
            });
        }
        if config.k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "number of hash functions must be positive".to_string(),
            });
        }
        let words = config.m.div_ceil(64);
        Ok(Self {
            bits: vec![0u64; words],
            config,
            n_inserted: 0,
        })
    }

    /// Construct with a target false-positive rate for `n_elements` items.
    ///
    /// Applies the optimal formulas:
    /// - `m = ceil(-n * ln(fpr) / ln(2)^2)`
    /// - `k = round((m / n) * ln(2))`
    ///
    /// # Errors
    /// Returns [`SketchError::InvalidParameter`] if `n_elements == 0` or `target_fpr` is
    /// not in `(0.0, 1.0)`.
    pub fn from_target_fpr(n_elements: usize, target_fpr: f64) -> SketchResult<Self> {
        if n_elements == 0 {
            return Err(SketchError::InvalidParameter {
                name: "n_elements".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if !(target_fpr > 0.0 && target_fpr < 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "target_fpr".to_string(),
                reason: "must be strictly in (0, 1)".to_string(),
            });
        }
        let ln2 = std::f64::consts::LN_2;
        let m = (-(n_elements as f64) * target_fpr.ln() / (ln2 * ln2)).ceil() as usize;
        let m = m.max(1);
        let k = ((m as f64 / n_elements as f64) * ln2).round() as usize;
        let k = k.max(1);
        Self::new(BloomConfig { m, k })
    }

    // ── Bit-level helpers ──────────────────────────────────────────────────────

    /// Compute the `k` bit-array positions for `item` using double hashing.
    ///
    /// `pos_i = (h1 + i * h2) mod m`
    #[inline]
    fn positions(&self, item: &[u8]) -> impl Iterator<Item = usize> + '_ {
        let h1 = xxh3_64(item, 0);
        let h2 = xxh3_64(item, 1);
        let m = self.config.m;
        let k = self.config.k;
        (0..k).map(move |i| (h1.wrapping_add((i as u64).wrapping_mul(h2)) as usize) % m)
    }

    #[inline]
    fn set_bit(&mut self, pos: usize) {
        let word = pos / 64;
        let bit = pos % 64;
        self.bits[word] |= 1u64 << bit;
    }

    #[inline]
    fn test_bit(&self, pos: usize) -> bool {
        let word = pos / 64;
        let bit = pos % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /// Insert a byte-slice item into the filter.
    pub fn insert(&mut self, item: &[u8]) {
        let positions: Vec<usize> = self.positions(item).collect();
        for pos in positions {
            self.set_bit(pos);
        }
        self.n_inserted += 1;
    }

    /// Test membership.
    ///
    /// False negatives are **impossible**. False positives may occur with
    /// probability bounded by [`Self::false_positive_rate`].
    #[must_use]
    pub fn contains(&self, item: &[u8]) -> bool {
        self.positions(item).all(|pos| self.test_bit(pos))
    }

    /// Theoretical false-positive rate given the current number of inserted items:
    ///
    /// `fpr = (1 - exp(-k * n / m))^k`
    #[must_use]
    pub fn false_positive_rate(&self) -> f64 {
        let cfg = &self.config;
        let exp_arg = -(cfg.k as f64) * (self.n_inserted as f64) / (cfg.m as f64);
        (1.0_f64 - exp_arg.exp()).powi(cfg.k as i32)
    }

    /// Count the number of bits currently set to 1.
    #[must_use]
    pub fn n_bits_set(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Clear all bits and reset the inserted-item counter.
    pub fn clear(&mut self) {
        for w in self.bits.iter_mut() {
            *w = 0;
        }
        self.n_inserted = 0;
    }

    /// Return the filter configuration.
    #[must_use]
    pub fn config(&self) -> &BloomConfig {
        &self.config
    }

    /// Return the total number of items inserted (not deduplicated).
    #[must_use]
    pub fn n_inserted(&self) -> usize {
        self.n_inserted
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a deterministic byte-slice from an integer.
    fn item(i: u64) -> Vec<u8> {
        i.to_le_bytes().to_vec()
    }

    // 1. Insert then contains returns true.
    #[test]
    fn contains_after_insert() {
        let mut bf = BloomFilter::new(BloomConfig { m: 1024, k: 4 }).expect("constructs");
        bf.insert(b"hello world");
        assert!(bf.contains(b"hello world"));
    }

    // 2. False negatives are impossible: 100 inserted items are always found.
    #[test]
    fn false_negatives_impossible() {
        let mut bf = BloomFilter::new(BloomConfig { m: 8192, k: 5 }).expect("constructs");
        let items: Vec<Vec<u8>> = (0u64..100).map(item).collect();
        for it in &items {
            bf.insert(it);
        }
        for it in &items {
            assert!(bf.contains(it), "false negative for item {:?}", it);
        }
    }

    // 3. Theoretical FPR is bounded by the target after n_elements inserts.
    #[test]
    fn fpr_bounded() {
        let target = 0.01;
        let n = 1000usize;
        let mut bf = BloomFilter::from_target_fpr(n, target).expect("constructs");
        for i in 0..n as u64 {
            bf.insert(&item(i));
        }
        let fpr = bf.false_positive_rate();
        // Theoretical bound may slightly exceed target due to ceiling rounding; allow 2×.
        assert!(
            fpr <= target * 2.0,
            "theoretical FPR {fpr} far exceeds target {target}"
        );
    }

    // 4. n_bits_set increases as items are inserted.
    #[test]
    fn n_bits_set_increases_with_inserts() {
        let mut bf = BloomFilter::new(BloomConfig { m: 4096, k: 4 }).expect("constructs");
        let before = bf.n_bits_set();
        bf.insert(b"alpha");
        let after = bf.n_bits_set();
        assert!(after >= before, "bits should not decrease after insert");
    }

    // 5. from_target_fpr constructs successfully.
    #[test]
    fn from_target_fpr_constructs() {
        let result = BloomFilter::from_target_fpr(100, 0.01);
        assert!(result.is_ok(), "from_target_fpr(100, 0.01) should succeed");
        let bf = result.expect("ok");
        assert!(bf.config().m > 0);
        assert!(bf.config().k > 0);
    }

    // 6. n_bits_set strictly increases after inserting a genuinely new item.
    #[test]
    fn insert_grows_bit_count() {
        let mut bf = BloomFilter::new(BloomConfig { m: 2048, k: 3 }).expect("constructs");
        let start = bf.n_bits_set();
        // Insert enough distinct items that at least one new bit must be set.
        for i in 0u64..20 {
            bf.insert(&item(i));
        }
        assert!(
            bf.n_bits_set() > start,
            "n_bits_set should grow after 20 distinct inserts"
        );
    }

    // 7. Insert 1000 items; all are contained.
    #[test]
    fn large_set_no_false_negatives() {
        let mut bf = BloomFilter::from_target_fpr(1000, 0.01).expect("constructs");
        let items: Vec<Vec<u8>> = (0u64..1000).map(item).collect();
        for it in &items {
            bf.insert(it);
        }
        for (idx, it) in items.iter().enumerate() {
            assert!(bf.contains(it), "false negative at index {idx}");
        }
    }

    // 8. n_bits_set never exceeds the total number of bits m.
    #[test]
    fn bit_count_never_exceeds_m() {
        let m = 512usize;
        let mut bf = BloomFilter::new(BloomConfig { m, k: 4 }).expect("constructs");
        for i in 0u64..200 {
            bf.insert(&item(i));
        }
        assert!(
            bf.n_bits_set() <= m,
            "n_bits_set {} exceeds m {}",
            bf.n_bits_set(),
            m
        );
    }

    // 9. from_target_fpr: theoretical FPR after n_elements inserts is near target.
    #[test]
    fn from_target_fpr_fpr_bounded_after_full_load() {
        let target = 0.05;
        let n = 500usize;
        let mut bf = BloomFilter::from_target_fpr(n, target).expect("constructs");
        for i in 0..n as u64 {
            bf.insert(&item(i));
        }
        let fpr = bf.false_positive_rate();
        assert!(
            fpr <= target * 2.0,
            "FPR {fpr} exceeds 2× target {target} after n_elements inserts"
        );
    }

    // 10. After clear(), previously inserted items are no longer found.
    #[test]
    fn clear_resets_filter() {
        let mut bf = BloomFilter::new(BloomConfig { m: 4096, k: 5 }).expect("constructs");
        let items: Vec<Vec<u8>> = (0u64..50).map(item).collect();
        for it in &items {
            bf.insert(it);
        }
        // Sanity: all present before clear.
        for it in &items {
            assert!(bf.contains(it));
        }
        bf.clear();
        assert_eq!(bf.n_bits_set(), 0, "all bits should be zero after clear");
        assert_eq!(bf.n_inserted(), 0, "n_inserted should be zero after clear");
        // After clear, the inserted items should not be found (no false negatives is
        // only a guarantee while items remain in the filter).
        let still_present = items.iter().filter(|it| bf.contains(it.as_slice())).count();
        // An empty filter cannot possibly return true (all bits are 0), so still_present must be 0.
        assert_eq!(
            still_present, 0,
            "cleared filter should not contain any items"
        );
    }
}

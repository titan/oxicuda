//! Theta Sketch for set cardinality estimation and set operations.
//!
//! Broder 1997, Apache DataSketches variant.
//!
//! Maintains the bottom-`k` hash values from a stream as a summary.
//! The threshold θ = max_retained_hash / u64::MAX determines the sampling probability.
//! Supports union, intersection estimate, and difference estimate without decompression.
//!
//! Relative error ≈ 1 / √k.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64;

/// Theta sketch for cardinality estimation and set operations.
///
/// Maintains the bottom-`k` hash values from a stream as a summary.
/// The threshold θ = max_retained_hash / u64::MAX determines the sampling probability.
/// When the sketch has < k elements, it operates in exact mode (θ = 1.0).
/// When full (≥ k elements seen), θ < 1.0 and the estimate is k / θ.
#[derive(Debug, Clone)]
pub struct ThetaSketch {
    /// Target sketch size (precision parameter). Relative error ≈ 1 / √k.
    pub k: usize,
    /// Sorted ascending list of retained hashes (len ≤ k).
    pub hashes: Vec<u64>,
    /// Current threshold: accept hash iff hash < theta_u64.
    /// In sparse mode (not yet full): u64::MAX.
    pub theta_u64: u64,
    /// Total distinct items processed (exact count in sparse mode).
    pub n: u64,
    /// Hash seed for xxh3_64.
    pub seed: u64,
}

impl ThetaSketch {
    /// Create a new empty Theta sketch with `k` retained hashes.
    ///
    /// `k` must be ≥ 2. Larger k yields lower relative error (RE ≈ 1/√k).
    pub fn new(k: usize, seed: u64) -> SketchResult<Self> {
        if k < 2 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            k,
            hashes: Vec::with_capacity(k + 1),
            theta_u64: u64::MAX,
            n: 0,
            seed,
        })
    }

    /// Hash a u64 item to the [0, u64::MAX) universe via xxh3_64.
    #[inline]
    fn hash_item(x: u64, seed: u64) -> u64 {
        xxh3_64(&x.to_le_bytes(), seed)
    }

    /// Hash bytes directly via xxh3_64.
    #[inline]
    fn hash_bytes_item(bytes: &[u8], seed: u64) -> u64 {
        xxh3_64(bytes, seed)
    }

    /// Insert a computed hash value into the sketch (core insertion logic).
    ///
    /// Steps:
    /// 1. Reject if h ≥ theta_u64 (above threshold).
    /// 2. Insert h in sorted position.
    /// 3. If over capacity, evict the maximum and update theta_u64.
    fn insert_hash(&mut self, h: u64) {
        if h >= self.theta_u64 {
            return;
        }
        // Binary-search insert to keep hashes sorted ascending.
        let pos = self.hashes.partition_point(|&v| v < h);
        // Deduplicate: skip if h already present.
        if pos < self.hashes.len() && self.hashes[pos] == h {
            return;
        }
        self.hashes.insert(pos, h);
        self.n += 1;
        // If over capacity, evict maximum (last element) and update threshold.
        if self.hashes.len() > self.k {
            // The evicted element becomes the new theta.
            self.theta_u64 = self.hashes.pop().unwrap_or(u64::MAX);
            // n tracks distinct inserts; actual retained count is k.
        }
    }

    /// Add a raw 64-bit item.
    pub fn add_u64(&mut self, x: u64) {
        let h = Self::hash_item(x, self.seed);
        self.insert_hash(h);
    }

    /// Add an item by hashing its bytes.
    pub fn add_bytes(&mut self, bytes: &[u8]) {
        let h = Self::hash_bytes_item(bytes, self.seed);
        self.insert_hash(h);
    }

    /// Current θ as a fraction in [0, 1]: theta_u64 / u64::MAX as f64.
    #[must_use]
    pub fn theta_fraction(&self) -> f64 {
        self.theta_u64 as f64 / u64::MAX as f64
    }

    /// Check whether the sketch is in "full" (sampling) mode.
    ///
    /// In full mode, hashes.len() == k and theta_u64 < u64::MAX.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.hashes.len() >= self.k
    }

    /// Estimate distinct cardinality.
    ///
    /// - If sketch is not yet full (hashes.len() < k): return exact count (n).
    /// - Else: return k / theta_fraction().
    #[must_use]
    pub fn estimate_cardinality(&self) -> f64 {
        if !self.is_full() {
            // Sparse mode: exact count.
            self.n as f64
        } else {
            let theta = self.theta_fraction();
            if theta <= 0.0 {
                // Degenerate case: theta collapsed to zero; return sketch size.
                self.k as f64
            } else {
                self.hashes.len() as f64 / theta
            }
        }
    }

    /// Merge two theta sketches into a union sketch.
    ///
    /// Union algorithm:
    /// 1. Use min(self.theta_u64, other.theta_u64) as the combined threshold.
    /// 2. Collect all hashes from both below that threshold (already sorted; use merge).
    /// 3. Deduplicate.
    /// 4. If combined len > k, evict extras and tighten theta.
    ///
    /// The two sketches may have different seeds; the union sketch uses self.seed.
    /// Seeds should ideally match for valid set semantics, but this is not enforced.
    pub fn union(&self, other: &ThetaSketch) -> SketchResult<ThetaSketch> {
        let k = self.k.min(other.k);
        let min_theta = self.theta_u64.min(other.theta_u64);

        // Merge two sorted slices, keeping only those < min_theta.
        let mut merged: Vec<u64> = Vec::with_capacity(self.hashes.len() + other.hashes.len());
        let (mut ai, mut bi) = (0usize, 0usize);
        while ai < self.hashes.len() && bi < other.hashes.len() {
            let a = self.hashes[ai];
            let b = other.hashes[bi];
            if a >= min_theta && b >= min_theta {
                break;
            }
            if a <= b {
                if a < min_theta {
                    merged.push(a);
                }
                ai += 1;
                if a == b {
                    bi += 1; // deduplicate equal elements
                }
            } else {
                if b < min_theta {
                    merged.push(b);
                }
                bi += 1;
            }
        }
        // Drain remaining from each slice.
        while ai < self.hashes.len() {
            let a = self.hashes[ai];
            if a >= min_theta {
                break;
            }
            merged.push(a);
            ai += 1;
        }
        while bi < other.hashes.len() {
            let b = other.hashes[bi];
            if b >= min_theta {
                break;
            }
            merged.push(b);
            bi += 1;
        }
        // merged is sorted ascending; no duplicates remain from the merge.
        // Trim to k and update theta if over capacity.
        let new_theta = if merged.len() > k {
            let evicted = merged[k];
            merged.truncate(k);
            evicted
        } else {
            min_theta
        };

        Ok(ThetaSketch {
            k,
            hashes: merged,
            theta_u64: new_theta,
            n: self.n + other.n,
            seed: self.seed,
        })
    }

    /// Estimate cardinality of intersection |A ∩ B| using theta sketch union.
    ///
    /// Algorithm: count hashes present in both sets below min(theta_a, theta_b).
    /// intersection_estimate = count_common / theta_common_fraction
    /// where theta_common_fraction = min(self.theta_u64, other.theta_u64) / u64::MAX.
    pub fn intersection_estimate(&self, other: &ThetaSketch) -> SketchResult<f64> {
        let min_theta = self.theta_u64.min(other.theta_u64);
        let theta_f = min_theta as f64 / u64::MAX as f64;
        if theta_f <= 0.0 {
            return Err(SketchError::NumericalInstability(
                "theta is zero; intersection estimate undefined".to_string(),
            ));
        }
        let mut count_common = 0u64;
        for &h in &self.hashes {
            if h >= min_theta {
                break; // hashes are sorted; no further entries can match
            }
            if other.hashes.binary_search(&h).is_ok() {
                count_common += 1;
            }
        }
        Ok(count_common as f64 / theta_f)
    }

    /// Estimate cardinality of difference |A \ B| ≈ |A| - |A ∩ B|.
    pub fn difference_estimate(&self, other: &ThetaSketch) -> SketchResult<f64> {
        let a_card = self.estimate_cardinality();
        let inter = self.intersection_estimate(other)?;
        Ok((a_card - inter).max(0.0))
    }

    /// Number of retained hash values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Returns true if no hashes have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a ThetaSketch with `n` distinct u64 items starting at `offset`.
    fn build(k: usize, seed: u64, offset: u64, count: u64) -> ThetaSketch {
        let mut ts = ThetaSketch::new(k, seed).expect("new ok");
        for i in offset..offset + count {
            ts.add_u64(i);
        }
        ts
    }

    #[test]
    fn theta_sketch_new_empty() {
        let ts = ThetaSketch::new(16, 0).expect("ok");
        assert!(ts.is_empty());
        assert_eq!(ts.estimate_cardinality(), 0.0);
        assert_eq!(ts.len(), 0);
    }

    #[test]
    fn theta_sketch_add_single() {
        let mut ts = ThetaSketch::new(16, 0).expect("ok");
        ts.add_u64(42);
        // One item in exact mode.
        assert_eq!(ts.estimate_cardinality(), 1.0);
    }

    #[test]
    fn theta_sketch_k1_err() {
        assert!(ThetaSketch::new(1, 0).is_err());
        assert!(ThetaSketch::new(0, 0).is_err());
    }

    #[test]
    fn theta_sketch_exact_mode_small() {
        // k=100 >> 5 items → sparse mode, exact count.
        let mut ts = ThetaSketch::new(100, 7).expect("ok");
        for i in 0u64..5 {
            ts.add_u64(i);
        }
        assert!(!ts.is_full());
        assert_eq!(ts.estimate_cardinality(), 5.0);
    }

    #[test]
    fn theta_sketch_full_mode_large() {
        // k=512, 10000 distinct items → estimate within 10%.
        let ts = build(512, 1, 0, 10_000);
        assert!(ts.is_full());
        let est = ts.estimate_cardinality();
        let rel = (est - 10_000.0).abs() / 10_000.0;
        assert!(
            rel < 0.10,
            "expected within 10%, got est={est:.1}, rel_err={rel:.3}"
        );
    }

    #[test]
    fn theta_sketch_duplicate_items() {
        // Same item added many times → deduplicated to ~1.
        let mut ts = ThetaSketch::new(64, 3).expect("ok");
        for _ in 0..500 {
            ts.add_u64(999);
        }
        let est = ts.estimate_cardinality();
        assert!(est < 5.0, "duplicate-only estimate should be ~1, got {est}");
    }

    #[test]
    fn theta_sketch_theta_decreases_as_more_added() {
        let mut ts = ThetaSketch::new(16, 0).expect("ok");
        let mut prev_theta = 1.0f64;
        // Add items one by one and observe theta monotonically non-increasing.
        for i in 0u64..100 {
            ts.add_u64(i);
            let theta = ts.theta_fraction();
            assert!(
                theta <= prev_theta + f64::EPSILON,
                "theta increased at i={i}: prev={prev_theta}, now={theta}"
            );
            prev_theta = theta;
        }
    }

    #[test]
    fn theta_sketch_is_full_after_k_items() {
        let k = 32usize;
        let ts = build(k, 5, 0, (k as u64) + 10);
        assert!(ts.is_full(), "should be full after k+10 unique items");
        assert!(ts.len() <= k);
    }

    #[test]
    fn union_two_disjoint_sets() {
        // A = [0, 5000), B = [5000, 10000) → union estimate ≈ 10000.
        let k = 512;
        let a = build(k, 42, 0, 5_000);
        let b = build(k, 42, 5_000, 5_000);
        let u = a.union(&b).expect("union ok");
        let est = u.estimate_cardinality();
        let rel = (est - 10_000.0).abs() / 10_000.0;
        assert!(
            rel < 0.12,
            "disjoint union estimate off: est={est:.1}, rel={rel:.3}"
        );
    }

    #[test]
    fn union_identical_sets() {
        // A ∪ A ≈ |A|.
        let k = 256;
        let a = build(k, 7, 0, 2_000);
        let u = a.union(&a).expect("union ok");
        let est = u.estimate_cardinality();
        let rel = (est - 2_000.0).abs() / 2_000.0;
        assert!(
            rel < 0.12,
            "identical union estimate off: est={est:.1}, rel={rel:.3}"
        );
    }

    #[test]
    fn union_partial_overlap() {
        // A = [0, 1000), B = [500, 1500) → |A ∪ B| = 1500.
        let k = 512;
        let a = build(k, 11, 0, 1_000);
        let b = build(k, 11, 500, 1_000);
        let u = a.union(&b).expect("union ok");
        let est = u.estimate_cardinality();
        let rel = (est - 1_500.0).abs() / 1_500.0;
        assert!(
            rel < 0.15,
            "partial overlap union estimate off: est={est:.1}, rel={rel:.3}"
        );
    }

    #[test]
    fn intersection_disjoint_sets_zero() {
        // |A ∩ B| ≈ 0 for disjoint sets.
        let k = 256;
        let a = build(k, 13, 0, 1_000);
        let b = build(k, 13, 10_000, 1_000);
        let inter = a.intersection_estimate(&b).expect("ok");
        // Allow a small tolerance due to hash collisions and floating-point.
        assert!(
            inter < 50.0,
            "disjoint intersection should be near 0, got {inter:.2}"
        );
    }

    #[test]
    fn intersection_identical_sets() {
        // |A ∩ A| ≈ |A|.
        let k = 512;
        let a = build(k, 17, 0, 3_000);
        let inter = a.intersection_estimate(&a).expect("ok");
        let rel = (inter - 3_000.0).abs() / 3_000.0;
        assert!(
            rel < 0.12,
            "identical intersection estimate off: inter={inter:.1}, rel={rel:.3}"
        );
    }

    #[test]
    fn difference_estimate_non_negative() {
        // Difference must always be ≥ 0.
        let k = 128;
        let a = build(k, 19, 0, 500);
        let b = build(k, 19, 0, 600);
        let diff = a.difference_estimate(&b).expect("ok");
        assert!(diff >= 0.0, "difference should be non-negative, got {diff}");
    }

    #[test]
    fn difference_estimate_disjoint() {
        // |A \ B| ≈ |A| when B is disjoint from A.
        let k = 512;
        let a = build(k, 23, 0, 2_000);
        let b = build(k, 23, 50_000, 2_000);
        let diff = a.difference_estimate(&b).expect("ok");
        let rel = (diff - 2_000.0).abs() / 2_000.0;
        assert!(
            rel < 0.15,
            "disjoint difference estimate off: diff={diff:.1}, rel={rel:.3}"
        );
    }

    #[test]
    fn add_bytes_works() {
        let mut ts = ThetaSketch::new(64, 0).expect("ok");
        ts.add_bytes(b"hello");
        ts.add_bytes(b"world");
        ts.add_bytes(b"foo");
        let est = ts.estimate_cardinality();
        assert!(est > 0.0, "add_bytes should yield non-zero estimate");
    }

    #[test]
    fn union_same_seed_k() {
        // Merged union has k == min(k_a, k_b).
        let a = build(64, 31, 0, 500);
        let b = build(128, 31, 500, 500);
        let u = a.union(&b).expect("ok");
        assert_eq!(u.k, 64);
        assert!(u.len() <= 64);
    }

    #[test]
    fn len_at_most_k() {
        // hashes.len() ≤ k at all times.
        let k = 32usize;
        let mut ts = ThetaSketch::new(k, 0).expect("ok");
        for i in 0u64..10_000 {
            ts.add_u64(i);
            assert!(
                ts.len() <= k,
                "len {} exceeded k={} after {} items",
                ts.len(),
                k,
                i + 1
            );
        }
    }

    #[test]
    fn theta_sketch_hashes_sorted() {
        // Internal invariant: hashes must remain sorted ascending.
        let mut ts = ThetaSketch::new(32, 99).expect("ok");
        for i in 0u64..500 {
            ts.add_u64(i);
        }
        let sorted = ts.hashes.windows(2).all(|w| w[0] <= w[1]);
        assert!(sorted, "hashes must be sorted ascending");
    }

    #[test]
    fn theta_sketch_add_bytes_matches_different_inputs() {
        // Different byte slices should generally yield different estimates.
        let mut ts1 = ThetaSketch::new(64, 0).expect("ok");
        let mut ts2 = ThetaSketch::new(64, 0).expect("ok");
        for i in 0u64..100 {
            ts1.add_bytes(&i.to_le_bytes());
        }
        for i in 200u64..300 {
            ts2.add_bytes(&i.to_le_bytes());
        }
        // Both should have non-zero estimate.
        assert!(ts1.estimate_cardinality() > 0.0);
        assert!(ts2.estimate_cardinality() > 0.0);
    }
}

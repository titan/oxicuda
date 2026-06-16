//! KMV (K-Minimum Values) sketch — Bar-Yossef et al. 2002, Beyer et al. 2007.
//!
//! Maintains the k smallest distinct hash values from a stream using a single
//! hash function.  This simultaneously supports cardinality estimation
//! (|S| ≈ (k-1) / kbar_normalized) and Jaccard similarity estimation between
//! two sketches built with the same seed (bottom-k MinHash estimator).

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

/// KMV sketch keeping the `k` smallest distinct hash values.
#[derive(Debug, Clone)]
pub struct KmvSketch {
    pub k: usize,
    /// Hash values stored in ascending sorted order.  `values[k-1]` is the current
    /// threshold ("kbar") once the buffer is full.
    pub values: Vec<u64>,
    /// Total number of `add` calls (before deduplication).
    pub n_updates: usize,
    seed: u64,
}

impl KmvSketch {
    /// Create a new KMV sketch keeping the `k` minimum distinct hash values.
    pub fn new(k: usize, seed: u64) -> SketchResult<Self> {
        if k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            k,
            values: Vec::with_capacity(k),
            n_updates: 0,
            seed,
        })
    }

    /// Insert a `u64` element, hashed internally via xxh3-64.
    pub fn add(&mut self, x: u64) {
        self.n_updates += 1;
        let h = xxh3_64_u64(x, self.seed);
        self.insert_hash(h);
    }

    /// Insert a string element, hashed via xxh3-64 over the UTF-8 bytes.
    pub fn add_str(&mut self, s: &str) {
        self.n_updates += 1;
        let h = crate::hash::xxh3_min::xxh3_64(s.as_bytes(), self.seed);
        self.insert_hash(h);
    }

    fn insert_hash(&mut self, h: u64) {
        match self.values.binary_search(&h) {
            Ok(_) => {}
            Err(pos) => {
                if self.values.len() < self.k {
                    self.values.insert(pos, h);
                } else if h < self.kbar() {
                    self.values.pop();
                    self.values.insert(pos, h);
                }
            }
        }
    }

    /// The k-th minimum hash value (the current threshold).  Returns `u64::MAX`
    /// when fewer than k values have been stored.
    #[must_use]
    pub fn kbar(&self) -> u64 {
        if self.values.len() < self.k {
            u64::MAX
        } else {
            *self.values.last().unwrap_or(&u64::MAX)
        }
    }

    /// Estimate the number of distinct elements seen.
    ///
    /// When fewer than k values are buffered the count is exact (no hash
    /// collision is assumed for typical small-n usage).  Otherwise the
    /// order-statistics estimator (k-1)/kbar_normalized is applied where
    /// kbar_normalized = kbar / u64::MAX.
    #[must_use]
    pub fn estimate_distinct(&self) -> f64 {
        let n = self.values.len();
        if n < self.k {
            return n as f64;
        }
        let kbar = self.kbar();
        if kbar == 0 {
            return f64::INFINITY;
        }
        let kbar_norm = kbar as f64 / u64::MAX as f64;
        (self.k - 1) as f64 / kbar_norm
    }

    /// Merge two KMV sketches that share the same `k` and the same hash seed.
    /// The merged sketch holds the k smallest values from the union of both
    /// bottom-k sets.
    pub fn merge(&self, other: &Self) -> SketchResult<Self> {
        if self.k != other.k {
            return Err(SketchError::DimensionMismatch {
                a: self.k,
                b: other.k,
            });
        }
        if self.seed != other.seed {
            return Err(SketchError::InvalidParameter {
                name: "seed".to_string(),
                reason: "both sketches must use the same hash seed".to_string(),
            });
        }

        let mut merged = Self {
            k: self.k,
            values: Vec::with_capacity(self.k),
            n_updates: self.n_updates + other.n_updates,
            seed: self.seed,
        };

        let (mut i, mut j) = (0usize, 0usize);
        while merged.values.len() < self.k && (i < self.values.len() || j < other.values.len()) {
            let take_left = match (i < self.values.len(), j < other.values.len()) {
                (true, false) => true,
                (false, true) => false,
                (true, true) => self.values[i] <= other.values[j],
                (false, false) => break,
            };
            let v = if take_left {
                let v = self.values[i];
                i += 1;
                v
            } else {
                let v = other.values[j];
                j += 1;
                v
            };
            if merged.values.last().copied() == Some(v) {
                continue;
            }
            merged.values.push(v);
        }

        Ok(merged)
    }

    /// Estimate the Jaccard similarity between two sketches via the bottom-k
    /// MinHash estimator.
    ///
    /// Jaccard ≈ |bottom_k(A) ∩ bottom_k(B)| / |bottom_k(A ∪ B)|_k
    ///
    /// The denominator is the size of the merged k-minimum set (at most k),
    /// and the numerator counts values in that set that appear in both A and B.
    pub fn jaccard_similarity(&self, other: &Self) -> SketchResult<f64> {
        let union_kmv = self.merge(other)?;
        let union_len = union_kmv.values.len();
        if union_len == 0 {
            return Ok(0.0);
        }
        let intersection_count = union_kmv
            .values
            .iter()
            .filter(|&&v| {
                self.values.binary_search(&v).is_ok() && other.values.binary_search(&v).is_ok()
            })
            .count();
        Ok(intersection_count as f64 / union_len as f64)
    }

    /// Estimate the union cardinality of two sketches.
    pub fn estimate_union(&self, other: &Self) -> SketchResult<f64> {
        let union_kmv = self.merge(other)?;
        Ok(union_kmv.estimate_distinct())
    }

    /// Estimate the intersection size: |A ∩ B| ≈ Jaccard(A, B) * |A ∪ B|.
    pub fn estimate_intersection(&self, other: &Self) -> SketchResult<f64> {
        let j = self.jaccard_similarity(other)?;
        let u = self.estimate_union(other)?;
        Ok((j * u).max(0.0))
    }

    /// Containment of `self` in `other`: |A ∩ B| / |A|.
    ///
    /// Returns 0 if `self` is empty.
    pub fn containment(&self, other: &Self) -> SketchResult<f64> {
        let self_card = self.estimate_distinct();
        if self_card <= 0.0 {
            return Ok(0.0);
        }
        let inter = self.estimate_intersection(other)?;
        Ok((inter / self_card).clamp(0.0, 1.0))
    }

    /// Number of hash values currently stored (≤ k).
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when no elements have been inserted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmv_new_empty() {
        let sketch = KmvSketch::new(5, 0).expect("new should succeed");
        assert_eq!(sketch.len(), 0);
        assert!(sketch.is_empty());
    }

    #[test]
    fn kmv_add_k_elements() {
        let mut sketch = KmvSketch::new(5, 1).expect("new should succeed");
        for i in 0..5u64 {
            sketch.add(i);
        }
        assert_eq!(sketch.len(), 5);
    }

    #[test]
    fn kmv_add_deduplication() {
        let mut sketch = KmvSketch::new(5, 2).expect("new should succeed");
        sketch.add(42);
        sketch.add(42);
        sketch.add(42);
        assert_eq!(sketch.len(), 1);
    }

    #[test]
    fn kmv_add_beyond_k() {
        let mut sketch = KmvSketch::new(5, 3).expect("new should succeed");
        for i in 0..100u64 {
            sketch.add(i);
        }
        assert_eq!(sketch.len(), 5);
    }

    #[test]
    fn kmv_kbar_inf_when_less_than_k() {
        let sketch = KmvSketch::new(5, 4).expect("new should succeed");
        assert_eq!(sketch.kbar(), u64::MAX);
        let mut sketch = KmvSketch::new(5, 4).expect("new should succeed");
        sketch.add(1);
        assert_eq!(sketch.kbar(), u64::MAX);
    }

    #[test]
    fn kmv_kbar_decreases_with_inserts() {
        let mut sketch = KmvSketch::new(4, 5).expect("new should succeed");
        for i in 0..4u64 {
            sketch.add(i * 1000);
        }
        let kbar_after_k = sketch.kbar();
        assert_ne!(kbar_after_k, u64::MAX);
        let kbar_prev = kbar_after_k;
        for i in 4..1000u64 {
            sketch.add(i);
        }
        assert!(sketch.kbar() <= kbar_prev);
    }

    #[test]
    fn kmv_estimate_distinct_few() {
        let mut sketch = KmvSketch::new(256, 6).expect("new should succeed");
        for i in 0..10u64 {
            sketch.add(i);
        }
        let est = sketch.estimate_distinct();
        assert!((est - 10.0).abs() < 1e-9, "expected 10, got {est}");
    }

    #[test]
    fn kmv_estimate_distinct_large() {
        let k = 256;
        let mut sketch = KmvSketch::new(k, 7).expect("new should succeed");
        let n = 10_000u64;
        for i in 0..n {
            sketch.add(i);
        }
        let est = sketch.estimate_distinct();
        let lo = n as f64 * 0.80;
        let hi = n as f64 * 1.20;
        assert!(
            est >= lo && est <= hi,
            "estimate {est} out of [{lo}, {hi}] for n={n}, k={k}"
        );
    }

    #[test]
    fn kmv_merge_combined_values() {
        let mut a = KmvSketch::new(8, 8).expect("new should succeed");
        let mut b = KmvSketch::new(8, 8).expect("new should succeed");
        for i in 0..20u64 {
            a.add(i);
        }
        for i in 10..30u64 {
            b.add(i);
        }
        let merged = a.merge(&b).expect("merge should succeed");
        for &v in &merged.values {
            let in_a = a.values.binary_search(&v).is_ok();
            let in_b = b.values.binary_search(&v).is_ok();
            assert!(in_a || in_b, "merged value {v} not from either sketch");
        }
    }

    #[test]
    fn kmv_merge_len_at_most_k() {
        let mut a = KmvSketch::new(16, 9).expect("new should succeed");
        let mut b = KmvSketch::new(16, 9).expect("new should succeed");
        for i in 0..100u64 {
            a.add(i);
            b.add(i + 50);
        }
        let merged = a.merge(&b).expect("merge should succeed");
        assert!(merged.len() <= 16);
    }

    #[test]
    fn kmv_jaccard_identical() {
        let mut sketch = KmvSketch::new(64, 10).expect("new should succeed");
        for i in 0..200u64 {
            sketch.add(i);
        }
        let j = sketch
            .jaccard_similarity(&sketch.clone())
            .expect("value should be present");
        assert!((j - 1.0).abs() < 1e-9, "identical sketch Jaccard = {j}");
    }

    #[test]
    fn kmv_jaccard_disjoint() {
        let mut a = KmvSketch::new(128, 11).expect("new should succeed");
        let mut b = KmvSketch::new(128, 11).expect("new should succeed");
        for i in 0..500u64 {
            a.add(i);
        }
        for i in 1_000_000..1_000_500u64 {
            b.add(i);
        }
        let j = a
            .jaccard_similarity(&b)
            .expect("jaccard_similarity should succeed");
        assert!(j < 0.05, "disjoint Jaccard = {j}");
    }

    #[test]
    fn kmv_jaccard_partial_overlap() {
        let k = 512;
        let mut a = KmvSketch::new(k, 12).expect("new should succeed");
        let mut b = KmvSketch::new(k, 12).expect("new should succeed");
        for i in 0..500u64 {
            a.add(i);
        }
        for i in 250..750u64 {
            b.add(i);
        }
        let j = a
            .jaccard_similarity(&b)
            .expect("jaccard_similarity should succeed");
        let true_j = 250.0 / 750.0;
        assert!((j - true_j).abs() < 0.20, "estimated {j} true {true_j}");
    }

    #[test]
    fn kmv_estimate_union_at_least_max() {
        let mut a = KmvSketch::new(128, 13).expect("new should succeed");
        let mut b = KmvSketch::new(128, 13).expect("new should succeed");
        for i in 0..300u64 {
            a.add(i);
        }
        for i in 200..500u64 {
            b.add(i);
        }
        let est_a = a.estimate_distinct();
        let est_b = b.estimate_distinct();
        let est_u = a.estimate_union(&b).expect("estimate_union should succeed");
        assert!(
            est_u >= est_a.max(est_b) * 0.80,
            "union {est_u} should be at least ~max({est_a},{est_b})"
        );
    }

    #[test]
    fn kmv_containment_subset() {
        let k = 256;
        let mut a = KmvSketch::new(k, 14).expect("new should succeed");
        let mut b = KmvSketch::new(k, 14).expect("new should succeed");
        for i in 0..100u64 {
            a.add(i);
            b.add(i);
        }
        for i in 100..400u64 {
            b.add(i);
        }
        let cont = a.containment(&b).expect("containment should succeed");
        assert!(cont > 0.70, "containment of subset A in B = {cont}");
    }

    #[test]
    fn kmv_err_k_zero() {
        assert!(KmvSketch::new(0, 15).is_err());
    }

    #[test]
    fn kmv_add_str_works() {
        let mut sketch = KmvSketch::new(8, 16).expect("new should succeed");
        sketch.add_str("hello");
        assert_eq!(sketch.len(), 1);
        sketch.add_str("world");
        assert_eq!(sketch.len(), 2);
        sketch.add_str("hello");
        assert_eq!(sketch.len(), 2);
    }

    #[test]
    fn kmv_estimate_intersection_nonneg() {
        let mut a = KmvSketch::new(64, 17).expect("new should succeed");
        let mut b = KmvSketch::new(64, 17).expect("new should succeed");
        for i in 0..100u64 {
            a.add(i);
        }
        for i in 50..150u64 {
            b.add(i);
        }
        let inter = a
            .estimate_intersection(&b)
            .expect("estimate_intersection should succeed");
        assert!(inter >= 0.0, "intersection estimate {inter} < 0");
    }

    #[test]
    fn kmv_values_sorted_invariant() {
        let mut sketch = KmvSketch::new(16, 99).expect("new should succeed");
        for i in (0u64..200).rev() {
            sketch.add(i);
        }
        for w in sketch.values.windows(2) {
            assert!(
                w[0] < w[1],
                "values not strictly ascending: {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn kmv_merge_seed_mismatch_err() {
        let mut a = KmvSketch::new(8, 1).expect("new should succeed");
        let mut b = KmvSketch::new(8, 2).expect("new should succeed");
        a.add(0);
        b.add(0);
        assert!(a.merge(&b).is_err());
    }
}

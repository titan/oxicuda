//! HeavyKeeper (Huang et al. 2018 NSDI) top-k frequency sketch.
//!
//! A decay-based top-k frequency estimator using fingerprint-matched buckets
//! and probabilistic decay. Outperforms Misra-Gries and Space-Saving in both
//! speed and accuracy for top-k heavy-hitter detection.
//!
//! Reference: Yang et al. "HeavyKeeper: An Accurate Algorithm for Finding
//! Top-k Elephant Flows", NSDI 2018.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::murmur3::murmur3_32;
use crate::hash::twouniv::TwoUniversal;

/// HeavyKeeper configuration.
#[derive(Debug, Clone, Copy)]
pub struct HeavyKeeperConfig {
    /// Number of top items to track (k ≥ 1).
    pub k: usize,
    /// Number of columns per hash row (≥ 1).
    pub width: usize,
    /// Number of independent hash rows (≥ 1; default 3).
    pub depth: usize,
    /// Decay base b > 1.0 (default 1.08 from paper).
    pub b: f64,
}

impl Default for HeavyKeeperConfig {
    fn default() -> Self {
        Self {
            k: 10,
            width: 512,
            depth: 3,
            b: 1.08,
        }
    }
}

/// A single bucket: (fingerprint, count).
#[derive(Debug, Clone, Copy, Default)]
pub struct HkBucket {
    pub fingerprint: u32,
    pub count: u64,
}

/// HeavyKeeper top-k frequency sketch.
///
/// Maintains a `depth × width` array of fingerprint-count buckets plus a
/// min-heap of the current top-k items.
#[derive(Debug, Clone)]
pub struct HeavyKeeper {
    /// Configuration.
    pub cfg: HeavyKeeperConfig,
    /// depth × width array of buckets, row-major.
    pub buckets: Vec<HkBucket>,
    /// Current top-k items: (key, estimated_count), kept as a min-heap by count.
    pub min_heap: Vec<(u64, u64)>,
    /// Total items inserted.
    pub n: u64,
    /// Per-row 2-universal hash functions (depth hashes, each mapping u64 → [0, width)).
    row_hashes: Vec<TwoUniversal>,
}

impl HeavyKeeper {
    /// Create a new HeavyKeeper with the given configuration.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if any parameter is out of range.
    pub fn new(cfg: HeavyKeeperConfig) -> SketchResult<Self> {
        if cfg.k < 1 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if cfg.width < 1 {
            return Err(SketchError::InvalidParameter {
                name: "width".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if cfg.depth < 1 {
            return Err(SketchError::InvalidParameter {
                name: "depth".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if cfg.b <= 1.0 {
            return Err(SketchError::InvalidParameter {
                name: "b".to_string(),
                reason: "decay base must be > 1.0".to_string(),
            });
        }
        // Use a fixed seed for the internal row hashes so that HeavyKeeper::new
        // without an external rng is still deterministic from the config alone.
        // Callers that want seeded randomness should use a seeded TwoUniversal::many.
        let mut internal_rng = LcgRng::new(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(cfg.k as u64));
        let row_hashes = TwoUniversal::many(&mut internal_rng, cfg.depth, cfg.width as u64);
        let buckets = vec![HkBucket::default(); cfg.depth * cfg.width];
        Ok(Self {
            cfg,
            buckets,
            min_heap: Vec::with_capacity(cfg.k + 1),
            n: 0,
            row_hashes,
        })
    }

    /// Create a new HeavyKeeper with the given configuration and an explicit RNG
    /// for reproducibility / deterministic tests.
    pub fn new_with_rng(cfg: HeavyKeeperConfig, rng: &mut LcgRng) -> SketchResult<Self> {
        if cfg.k < 1 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if cfg.width < 1 {
            return Err(SketchError::InvalidParameter {
                name: "width".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if cfg.depth < 1 {
            return Err(SketchError::InvalidParameter {
                name: "depth".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if cfg.b <= 1.0 {
            return Err(SketchError::InvalidParameter {
                name: "b".to_string(),
                reason: "decay base must be > 1.0".to_string(),
            });
        }
        let row_hashes = TwoUniversal::many(rng, cfg.depth, cfg.width as u64);
        let buckets = vec![HkBucket::default(); cfg.depth * cfg.width];
        Ok(Self {
            cfg,
            buckets,
            min_heap: Vec::with_capacity(cfg.k + 1),
            n: 0,
            row_hashes,
        })
    }

    /// Insert item `x`. Returns the estimated count of `x` after the update.
    pub fn add(&mut self, x: u64, rng: &mut LcgRng) -> u64 {
        self.n += 1;
        // Compute fingerprint using murmur3 with seed 0.
        let f = murmur3_32(x, 0);
        let b_ln = self.cfg.b.ln();

        for d in 0..self.cfg.depth {
            let j = self.row_hashes[d].hash(x) as usize;
            let bucket = &mut self.buckets[d * self.cfg.width + j];

            if bucket.count == 0 {
                // Case 1: empty bucket — claim it.
                bucket.fingerprint = f;
                bucket.count = 1;
            } else if bucket.fingerprint == f {
                // Case 2: matching fingerprint — increment.
                bucket.count += 1;
            } else {
                // Case 3: collision — probabilistic decay.
                // prob = b^{-count} = exp(-count * ln(b))
                let prob = (-(bucket.count as f64) * b_ln).exp();
                let u = rng.next_f64();
                if u < prob {
                    // Decay: replace bucket with new item at count 1.
                    bucket.fingerprint = f;
                    bucket.count = 1;
                }
                // else: bucket unchanged (paper's "not decrement" variant).
            }
        }

        let estimated = self.estimate(x);
        self.heap_update(x, estimated);
        estimated
    }

    /// Estimated frequency of `x` (min over rows; 0 if fingerprint doesn't match).
    #[must_use]
    pub fn estimate(&self, x: u64) -> u64 {
        let f = murmur3_32(x, 0);
        let mut min_count: Option<u64> = None;
        for d in 0..self.cfg.depth {
            let j = self.row_hashes[d].hash(x) as usize;
            let bucket = &self.buckets[d * self.cfg.width + j];
            if bucket.fingerprint == f && bucket.count > 0 {
                let c = bucket.count;
                min_count = Some(min_count.map_or(c, |m: u64| m.min(c)));
            }
        }
        min_count.unwrap_or(0)
    }

    /// Top-k items sorted by count descending.
    ///
    /// May return fewer than k if fewer distinct items have been seen.
    #[must_use]
    pub fn top_k(&self) -> Vec<(u64, u64)> {
        let mut result = self.min_heap.clone();
        result.sort_unstable_by_key(|b| std::cmp::Reverse(b.1));
        result
    }

    /// Number of items tracked in the top-k heap.
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.min_heap.len()
    }

    // ---- Heap helpers -------------------------------------------------------

    /// Update the heap after computing the estimated count of `x`.
    fn heap_update(&mut self, x: u64, estimated: u64) {
        // Search for x in heap.
        if let Some(pos) = self.min_heap.iter().position(|(key, _)| *key == x) {
            // Update count if estimated is larger.
            if estimated > self.min_heap[pos].1 {
                self.min_heap[pos].1 = estimated;
                // The count increased, so sift-down to restore min-heap.
                self.sift_down(pos);
                // Also sift-up in case this is now smaller than parent (shouldn't
                // happen when increasing, but keep it safe).
                self.sift_up(pos);
            }
        } else if self.min_heap.len() < self.cfg.k {
            // Room in heap: push new item.
            self.min_heap.push((x, estimated));
            let pos = self.min_heap.len() - 1;
            self.sift_up(pos);
        } else if !self.min_heap.is_empty() && estimated > self.min_heap[0].1 {
            // Replace heap minimum.
            self.min_heap[0] = (x, estimated);
            self.sift_down(0);
        }
    }

    /// Sift the element at `pos` up toward the root (min-heap by count).
    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if self.min_heap[pos].1 < self.min_heap[parent].1 {
                self.min_heap.swap(pos, parent);
                pos = parent;
            } else {
                break;
            }
        }
    }

    /// Sift the element at `pos` down (min-heap by count).
    fn sift_down(&mut self, mut pos: usize) {
        let len = self.min_heap.len();
        loop {
            let left = 2 * pos + 1;
            let right = 2 * pos + 2;
            let mut smallest = pos;
            if left < len && self.min_heap[left].1 < self.min_heap[smallest].1 {
                smallest = left;
            }
            if right < len && self.min_heap[right].1 < self.min_heap[smallest].1 {
                smallest = right;
            }
            if smallest == pos {
                break;
            }
            self.min_heap.swap(pos, smallest);
            pos = smallest;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // 1. Default config creates without error.
    #[test]
    fn new_default_ok() {
        let hk = HeavyKeeper::new(HeavyKeeperConfig::default());
        assert!(hk.is_ok());
    }

    // 2. k=0 → Err.
    #[test]
    fn new_invalid_k_zero() {
        let cfg = HeavyKeeperConfig {
            k: 0,
            ..Default::default()
        };
        assert!(HeavyKeeper::new(cfg).is_err());
    }

    // 3. b ≤ 1.0 → Err.
    #[test]
    fn new_invalid_b_not_gt_1() {
        let cfg1 = HeavyKeeperConfig {
            b: 1.0,
            ..Default::default()
        };
        assert!(HeavyKeeper::new(cfg1).is_err());
        let cfg2 = HeavyKeeperConfig {
            b: 0.5,
            ..Default::default()
        };
        assert!(HeavyKeeper::new(cfg2).is_err());
    }

    // 4. width=0 → Err.
    #[test]
    fn new_invalid_width_zero() {
        let cfg = HeavyKeeperConfig {
            width: 0,
            ..Default::default()
        };
        assert!(HeavyKeeper::new(cfg).is_err());
    }

    // 5. depth=0 → Err.
    #[test]
    fn new_invalid_depth_zero() {
        let cfg = HeavyKeeperConfig {
            depth: 0,
            ..Default::default()
        };
        assert!(HeavyKeeper::new(cfg).is_err());
    }

    // 6. Never inserted x; estimate(x)==0.
    #[test]
    fn estimate_unseen_is_zero() {
        let hk = HeavyKeeper::new(HeavyKeeperConfig::default()).unwrap();
        assert_eq!(hk.estimate(99999), 0);
    }

    // 7. add same x 10 times; estimate(x) ≥ 1.
    #[test]
    fn add_single_item_estimate() {
        let mut hk = HeavyKeeper::new(HeavyKeeperConfig::default()).unwrap();
        let mut rng = default_rng();
        for _ in 0..10 {
            hk.add(42, &mut rng);
        }
        assert!(hk.estimate(42) >= 1);
    }

    // 8. No inserts; top_k().is_empty().
    #[test]
    fn top_k_empty() {
        let hk = HeavyKeeper::new(HeavyKeeperConfig::default()).unwrap();
        assert!(hk.top_k().is_empty());
    }

    // 9. Insert same x 1000 times, others 1 time each; top_k()[0].0 == x.
    #[test]
    fn top_k_single_heavy_hitter() {
        let cfg = HeavyKeeperConfig {
            k: 5,
            width: 512,
            depth: 3,
            b: 1.08,
        };
        let mut hk = HeavyKeeper::new(cfg).unwrap();
        let mut rng = default_rng();
        let heavy: u64 = 777;
        for _ in 0..1000 {
            hk.add(heavy, &mut rng);
        }
        for i in 1u64..=20 {
            hk.add(i * 1000, &mut rng);
        }
        let top = hk.top_k();
        assert!(!top.is_empty());
        assert_eq!(top[0].0, heavy, "heavy hitter should be top: {:?}", top);
    }

    // 10. Result is sorted by count descending.
    #[test]
    fn top_k_sorted_desc() {
        let cfg = HeavyKeeperConfig {
            k: 5,
            width: 256,
            depth: 3,
            b: 1.08,
        };
        let mut hk = HeavyKeeper::new(cfg).unwrap();
        let mut rng = default_rng();
        for i in 0u64..5 {
            let count = (i + 1) * 100;
            for _ in 0..count {
                hk.add(i, &mut rng);
            }
        }
        let top = hk.top_k();
        for w in top.windows(2) {
            assert!(w[0].1 >= w[1].1, "not sorted desc: {:?}", top);
        }
    }

    // 11. top_k().len() ≤ k.
    #[test]
    fn top_k_returns_at_most_k() {
        let cfg = HeavyKeeperConfig {
            k: 3,
            width: 256,
            depth: 3,
            b: 1.08,
        };
        let mut hk = HeavyKeeper::new(cfg).unwrap();
        let mut rng = default_rng();
        for i in 0u64..100 {
            hk.add(i, &mut rng);
        }
        assert!(hk.top_k().len() <= 3);
    }

    // 12. One key inserted far more; it appears in top_k.
    #[test]
    fn heavy_hitter_dominates() {
        let cfg = HeavyKeeperConfig {
            k: 5,
            width: 512,
            depth: 3,
            b: 1.08,
        };
        let mut hk = HeavyKeeper::new(cfg).unwrap();
        let mut rng = default_rng();
        let dominant: u64 = 42;
        for _ in 0..2000 {
            hk.add(dominant, &mut rng);
        }
        for i in 1u64..=50 {
            hk.add(i * 100_000, &mut rng);
        }
        let top = hk.top_k();
        let keys: Vec<u64> = top.iter().map(|(k, _)| *k).collect();
        assert!(
            keys.contains(&dominant),
            "dominant key not in top_k: {:?}",
            top
        );
    }

    // 13. estimate(x) after 100 inserts ≥ after 10 inserts.
    #[test]
    fn estimate_monotone_after_more_inserts() {
        let mut hk = HeavyKeeper::new(HeavyKeeperConfig::default()).unwrap();
        let mut rng = default_rng();
        for _ in 0..10 {
            hk.add(1, &mut rng);
        }
        let est10 = hk.estimate(1);
        for _ in 0..90 {
            hk.add(1, &mut rng);
        }
        let est100 = hk.estimate(1);
        assert!(
            est100 >= est10,
            "estimate not monotone: {est10} -> {est100}"
        );
    }

    // 14. n equals total add() calls.
    #[test]
    fn n_counts_total_inserts() {
        let mut hk = HeavyKeeper::new(HeavyKeeperConfig::default()).unwrap();
        let mut rng = default_rng();
        for i in 0u64..50 {
            hk.add(i, &mut rng);
        }
        assert_eq!(hk.n, 50);
    }

    // 15. k=3, insert 5 distinct keys with different freqs; top_k has most frequent.
    #[test]
    fn multiple_keys_tracked() {
        let cfg = HeavyKeeperConfig {
            k: 3,
            width: 512,
            depth: 3,
            b: 1.08,
        };
        let mut hk = HeavyKeeper::new(cfg).unwrap();
        let mut rng = default_rng();
        // Insert with frequencies 500, 200, 100, 20, 5.
        let freqs: &[(u64, u64)] = &[(1, 500), (2, 200), (3, 100), (4, 20), (5, 5)];
        for &(key, freq) in freqs {
            for _ in 0..freq {
                hk.add(key, &mut rng);
            }
        }
        let top = hk.top_k();
        assert!(top.len() <= 3);
        // The top key should be key 1 (highest frequency).
        assert_eq!(top[0].0, 1, "expected key 1 at top: {:?}", top);
    }

    // 16. heap_size() ≤ k at all times.
    #[test]
    fn heap_size_bounded_by_k() {
        let cfg = HeavyKeeperConfig {
            k: 5,
            width: 256,
            depth: 3,
            b: 1.08,
        };
        let mut hk = HeavyKeeper::new(cfg).unwrap();
        let mut rng = default_rng();
        for i in 0u64..200 {
            hk.add(i, &mut rng);
            assert!(hk.heap_size() <= 5, "heap_size exceeded k at i={i}");
        }
    }

    // 17. Insert two keys that hash to same bucket; one survives (no panic/error).
    #[test]
    fn fingerprint_collision_handled() {
        let cfg = HeavyKeeperConfig {
            k: 2,
            width: 1, // force all keys to same column → frequent collisions
            depth: 1,
            b: 1.08,
        };
        let mut hk = HeavyKeeper::new(cfg).unwrap();
        let mut rng = default_rng();
        // Two distinct keys — one will survive via decay.
        for _ in 0..50 {
            hk.add(10, &mut rng);
        }
        for _ in 0..50 {
            hk.add(20, &mut rng);
        }
        // No panic and estimate is sensible.
        let e10 = hk.estimate(10);
        let e20 = hk.estimate(20);
        // At least one should have a non-zero estimate.
        assert!(e10 + e20 > 0, "both estimates are 0, unexpected");
    }

    // 18. Two HeavyKeeper with same seed produce same top_k after same inserts.
    #[test]
    fn deterministic_same_seed() {
        let cfg = HeavyKeeperConfig {
            k: 5,
            width: 256,
            depth: 3,
            b: 1.08,
        };
        let mut rng1 = LcgRng::new(12345);
        let mut rng2 = LcgRng::new(12345);
        let mut hk1 = HeavyKeeper::new_with_rng(cfg, &mut rng1).unwrap();
        let mut hk2 = HeavyKeeper::new_with_rng(cfg, &mut rng2).unwrap();

        let mut insert_rng1 = LcgRng::new(999);
        let mut insert_rng2 = LcgRng::new(999);
        for i in 0u64..100 {
            let key = i % 10;
            hk1.add(key, &mut insert_rng1);
            hk2.add(key, &mut insert_rng2);
        }
        let t1 = hk1.top_k();
        let t2 = hk2.top_k();
        assert_eq!(t1, t2, "two identically seeded HK produced different top_k");
    }
}

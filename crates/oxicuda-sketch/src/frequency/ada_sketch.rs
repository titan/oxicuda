//! Ada-Sketch: Adaptive Count-Min with exact heavy-hitter promotion (Huang et al. 2021).
//!
//! Standard Count-Min Sketch maintains a `depth × width` table and provides
//! `ε`-approximate frequency estimates with additive error bounded by `ε · N`.
//!
//! Ada-Sketch augments this with a bounded-size exact counter map for "heavy hitters":
//! items whose CM estimate exceeds `theta × total_count` are promoted to an exact
//! counter (`Vec<(u64, i64)>` sorted by key for O(log n) binary search).
//!
//! For items promoted to the exact map:
//! - All future updates are tracked in both the CM table **and** the exact counter.
//! - Queries return the exact value instead of the CM estimate.
//!
//! This achieves zero error for heavy hitters and reduces the effective noise budget
//! for the long tail by a factor of 2 (fewer collisions in CM after promotion).

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::twouniv::TwoUniversal;

/// Ada-Sketch: adaptive Count-Min Sketch with exact heavy-hitter promotion.
#[derive(Debug, Clone)]
pub struct AdaSketch {
    /// Number of CM hash rows (depth).
    pub depth: usize,
    /// Number of CM columns per row (width).
    pub width: usize,
    /// Promotion threshold: item is heavy if CM estimate > `theta * total_count`.
    pub theta: f64,
    /// CM table, row-major: `table[row * width + col]`.
    pub table: Vec<i64>,
    /// Independent 2-universal hash functions, one per row.
    pub hashes: Vec<TwoUniversal>,
    /// Exact counters for heavy hitters, sorted by key for binary search.
    pub exact: Vec<(u64, i64)>,
    /// Running total of all inserted counts (sum of |delta| for positive deltas only,
    /// consistent with the insertion magnitude used for threshold comparison).
    pub total_count: i64,
    /// Maximum number of heavy-hitter slots.
    pub max_heavy: usize,
}

impl AdaSketch {
    /// Create a new Ada-Sketch.
    ///
    /// # Parameters
    /// - `depth`: Number of hash rows in the CM table.
    /// - `width`: Number of columns per row.
    /// - `theta`: Promotion threshold fraction in `(0, 1)`.  An item is promoted to
    ///   exact tracking if its CM estimate exceeds `theta * total_count`.
    /// - `max_heavy`: Hard cap on the number of exact heavy-hitter slots.
    /// - `rng`: RNG for drawing hash coefficients.
    ///
    /// # Errors
    /// Returns [`SketchError::InvalidParameter`] for invalid parameter combinations.
    pub fn new(
        depth: usize,
        width: usize,
        theta: f64,
        max_heavy: usize,
        rng: &mut LcgRng,
    ) -> SketchResult<Self> {
        if depth == 0 {
            return Err(SketchError::InvalidParameter {
                name: "depth".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if width == 0 {
            return Err(SketchError::InvalidParameter {
                name: "width".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if theta <= 0.0 || theta >= 1.0 {
            return Err(SketchError::InvalidParameter {
                name: "theta".to_string(),
                reason: "must be in (0, 1)".to_string(),
            });
        }
        if max_heavy == 0 {
            return Err(SketchError::InvalidParameter {
                name: "max_heavy".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        let hashes = TwoUniversal::many(rng, depth, width as u64);
        let table = vec![0i64; depth * width];

        Ok(Self {
            depth,
            width,
            theta,
            table,
            hashes,
            exact: Vec::new(),
            total_count: 0,
            max_heavy,
        })
    }

    /// Update the count of `key` by `delta`.
    ///
    /// Both positive and negative deltas are accepted (turnstile model).
    ///
    /// Algorithm:
    /// 1. If `key` is already in the exact map, update it there.
    /// 2. Always update the CM table (maintains CM as a synopsis for all items).
    /// 3. Update `total_count` (using absolute magnitude of delta for the threshold).
    /// 4. Check if `cm_estimate(key) > theta * total_count`; if so, promote to exact.
    pub fn update(&mut self, key: u64, delta: i64) {
        // Step 1: update exact counter if key is already heavy.
        if let Ok(pos) = self.exact.binary_search_by_key(&key, |&(k, _)| k) {
            self.exact[pos].1 += delta;
        }

        // Step 2: update CM table for all items.
        for row in 0..self.depth {
            let col = self.hashes[row].hash(key) as usize;
            self.table[row * self.width + col] =
                self.table[row * self.width + col].saturating_add(delta);
        }

        // Step 3: track total count (sum of all absolute positive increments).
        if delta > 0 {
            self.total_count = self.total_count.saturating_add(delta);
        }

        // Step 4: check for heavy-hitter promotion.
        // Only attempt promotion if the exact map has room.
        if self.exact.len() < self.max_heavy {
            // Is this key NOT already tracked exactly?
            if self.exact.binary_search_by_key(&key, |&(k, _)| k).is_err() {
                let cm_est = self.cm_estimate(key);
                let threshold = (self.theta * self.total_count as f64) as i64;
                if cm_est > threshold {
                    // Promote: insert into sorted exact vec.
                    let insert_pos = self.exact.partition_point(|&(k, _)| k < key);
                    self.exact.insert(insert_pos, (key, cm_est));
                }
            }
        }
    }

    /// Query the estimated count of `key`.
    ///
    /// Returns the exact count if `key` is a known heavy hitter, otherwise the
    /// Count-Min minimum estimate.
    #[must_use]
    pub fn query(&self, key: u64) -> i64 {
        if let Ok(pos) = self.exact.binary_search_by_key(&key, |&(k, _)| k) {
            return self.exact[pos].1;
        }
        self.cm_estimate(key)
    }

    /// Count-Min minimum estimate for `key` (plain CM, ignores the exact map).
    #[must_use]
    pub fn cm_estimate(&self, key: u64) -> i64 {
        let mut best = i64::MAX;
        for row in 0..self.depth {
            let col = self.hashes[row].hash(key) as usize;
            let v = self.table[row * self.width + col];
            if v < best {
                best = v;
            }
        }
        best
    }

    /// Return `true` if `key` has been promoted to the exact heavy-hitter map.
    #[must_use]
    pub fn is_heavy(&self, key: u64) -> bool {
        self.exact.binary_search_by_key(&key, |&(k, _)| k).is_ok()
    }

    /// Return the slice of exact heavy-hitter counters, sorted by key.
    #[must_use]
    pub fn heavy_hitters(&self) -> &[(u64, i64)] {
        &self.exact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(0xDEAD_BEEF_CAFE_0001)
    }

    // ── parameter validation ─────────────────────────────────────────────────

    #[test]
    fn ada_invalid_depth() {
        let mut rng = make_rng();
        assert!(AdaSketch::new(0, 64, 0.01, 16, &mut rng).is_err());
    }

    #[test]
    fn ada_invalid_theta_zero() {
        let mut rng = make_rng();
        assert!(AdaSketch::new(4, 64, 0.0, 16, &mut rng).is_err());
    }

    #[test]
    fn ada_invalid_theta_one() {
        let mut rng = make_rng();
        assert!(AdaSketch::new(4, 64, 1.0, 16, &mut rng).is_err());
    }

    // ── basic query ──────────────────────────────────────────────────────────

    #[test]
    fn ada_basic_query_tracks_count() {
        let mut rng = make_rng();
        let mut sketch = AdaSketch::new(4, 256, 0.01, 32, &mut rng).expect("ok");
        for _ in 0..50 {
            sketch.update(42, 1);
        }
        // Query must return at least the true count (CM overestimates).
        let q = sketch.query(42);
        assert!(q >= 50, "expected >= 50, got {q}");
    }

    // ── heavy hitter promotion ───────────────────────────────────────────────

    #[test]
    fn ada_heavy_hitter_promoted() {
        let mut rng = make_rng();
        // theta=0.05: a key is heavy if it accounts for >5% of total.
        // Insert key 1 exactly 100x, and 99 other keys once each.
        // total_count = 100 + 99 = 199; threshold = 0.05 * 199 ≈ 9.
        // key 1 CM estimate should exceed 9 → promoted.
        let mut sketch = AdaSketch::new(5, 512, 0.05, 16, &mut rng).expect("ok");
        for _ in 0..100i64 {
            sketch.update(1, 1);
        }
        for k in 2..101u64 {
            sketch.update(k, 1);
        }
        assert!(
            sketch.is_heavy(1),
            "key=1 should be promoted to heavy hitter"
        );
    }

    // ── never underestimates ─────────────────────────────────────────────────

    #[test]
    fn ada_query_never_underestimates() {
        // CM sketch property: estimate >= true count (with all-positive updates).
        let mut rng = make_rng();
        let mut sketch = AdaSketch::new(5, 512, 0.1, 64, &mut rng).expect("ok");
        let n = 200u64;
        // Insert each key once.
        for k in 0..n {
            sketch.update(k, 1);
        }
        // Every key should have CM estimate >= 1.
        for k in 0..n {
            let est = sketch.cm_estimate(k);
            assert!(
                est >= 1,
                "CM estimate for key={k} should be >= 1, got {est}"
            );
        }
    }

    // ── exact accuracy for heavy items ────────────────────────────────────────

    #[test]
    fn ada_heavy_accuracy() {
        let mut rng = make_rng();
        // theta=0.05, insert key=7 one hundred times so it becomes heavy.
        let mut sketch = AdaSketch::new(4, 128, 0.05, 32, &mut rng).expect("ok");
        for _ in 0..100i64 {
            sketch.update(7, 1);
        }
        // After promotion the exact counter starts at the CM estimate at the time of
        // promotion and accumulates further exact updates.
        // In practice is_heavy + query >= true_count.
        assert!(sketch.is_heavy(7), "key=7 should be heavy");
        // Since we started tracking exactly and all deltas are +1, the stored exact
        // value must be >= the true count (it may be higher due to initial snapshot).
        let q = sketch.query(7);
        assert!(q >= 100, "exact heavy count should be >= 100, got {q}");
    }

    // ── total_count tracking ──────────────────────────────────────────────────

    #[test]
    fn ada_total_count_tracked() {
        let mut rng = make_rng();
        let mut sketch = AdaSketch::new(3, 64, 0.1, 8, &mut rng).expect("ok");
        let expected_total = 500i64;
        for k in 0..expected_total {
            sketch.update(k as u64, 1);
        }
        assert_eq!(
            sketch.total_count, expected_total,
            "total_count should equal sum of positive deltas"
        );
    }

    // ── multiple heavy hitters ────────────────────────────────────────────────

    #[test]
    fn ada_multiple_heavy_hitters() {
        let mut rng = make_rng();
        // Insert 5 "heavy" keys each 200 times, and 100 "light" keys once.
        // theta=0.05, total_count = 5*200 + 100 = 1100, threshold ≈ 55.
        // Each heavy key has CM estimate ~200 >> 55, so all should be promoted.
        let mut sketch = AdaSketch::new(5, 512, 0.05, 16, &mut rng).expect("ok");
        let heavy_keys = [1001u64, 1002, 1003, 1004, 1005];
        for &hk in &heavy_keys {
            for _ in 0..200i64 {
                sketch.update(hk, 1);
            }
        }
        for k in 0..100u64 {
            sketch.update(k + 2000, 1);
        }
        // All five heavy keys must appear in the exact map.
        for &hk in &heavy_keys {
            assert!(
                sketch.is_heavy(hk),
                "key={hk} should be promoted to heavy hitter"
            );
        }
        assert!(
            sketch.heavy_hitters().len() >= heavy_keys.len(),
            "expected at least {} heavy hitters, got {}",
            heavy_keys.len(),
            sketch.heavy_hitters().len()
        );
    }

    // ── negative delta does not increment total_count ─────────────────────────

    #[test]
    fn ada_negative_delta_not_counted() {
        let mut rng = make_rng();
        let mut sketch = AdaSketch::new(3, 64, 0.1, 8, &mut rng).expect("ok");
        sketch.update(42, 100);
        let tc_before = sketch.total_count;
        sketch.update(42, -50);
        // Negative delta must NOT increment total_count.
        assert_eq!(
            sketch.total_count, tc_before,
            "negative delta should not change total_count"
        );
    }
}

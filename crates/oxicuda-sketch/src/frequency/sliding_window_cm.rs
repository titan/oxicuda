//! Sliding-window Count-Min Sketch with time-decaying buckets.
//!
//! Estimates per-key frequencies over only the **most recent `W` time units**
//! of a `(key, count, timestamp)` stream, in sublinear space. The construction
//! is a ring of `B` sub-sketches ("time buckets"), each a full Count-Min table
//! sharing the *same* hash family, where bucket `b` accumulates updates whose
//! timestamp falls in epoch `⌊t / bucket_span⌋ ≡ b (mod B)`.
//!
//! ## Window semantics
//!
//! With `bucket_span = ⌈W / B⌉`, the ring holds `B` epochs ≈ one window.
//! Advancing time to a new epoch **clears** the bucket about to be reused (its
//! data is now older than `W`), giving a *jumping-window* approximation of the
//! sliding window: the answer covers between `W` and `W + bucket_span` units of
//! history, so `B` controls the temporal resolution (larger `B` ⇒ tighter
//! window, more memory).
//!
//! A query for `key` sums that key's Count-Min estimate across every live
//! bucket. Because every bucket is an over-estimate and we *add* them, the
//! windowed estimate retains the Count-Min one-sided guarantee:
//! `est(key) ≥ true windowed frequency`, with additive error bounded by
//! `ε · (windowed total)` per bucket with high probability.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::twouniv::TwoUniversal;

/// Sliding-window Count-Min sketch.
#[derive(Debug, Clone)]
pub struct SlidingWindowCm {
    /// Count-Min depth (rows).
    pub d: usize,
    /// Count-Min width (columns).
    pub w: usize,
    /// Number of time buckets in the ring.
    pub n_buckets: usize,
    /// Time span (in timestamp units) covered by one bucket.
    pub bucket_span: u64,
    /// Shared hash family across all buckets (length `d`).
    hashes: Vec<TwoUniversal>,
    /// `n_buckets` Count-Min tables, each `d · w` cells, flattened.
    tables: Vec<u64>,
    /// Epoch index currently stored in each bucket (`u64::MAX` ⇒ empty).
    epoch_of_bucket: Vec<u64>,
    /// Highest epoch observed so far.
    current_epoch: u64,
}

impl SlidingWindowCm {
    /// Construct a sliding-window Count-Min.
    ///
    /// * `d`, `w` — Count-Min depth/width (`> 0`).
    /// * `window` — window length `W` in timestamp units (`> 0`).
    /// * `n_buckets` — temporal resolution `B` (`> 0`); `bucket_span = ⌈W/B⌉`.
    pub fn new(
        d: usize,
        w: usize,
        window: u64,
        n_buckets: usize,
        rng: &mut LcgRng,
    ) -> SketchResult<Self> {
        if d == 0 || w == 0 || n_buckets == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(d, w, n_buckets)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if window == 0 {
            return Err(SketchError::InvalidParameter {
                name: "window".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let bucket_span = window.div_ceil(n_buckets as u64).max(1);
        let hashes = TwoUniversal::many(rng, d, w as u64);
        Ok(Self {
            d,
            w,
            n_buckets,
            bucket_span,
            hashes,
            tables: vec![0u64; n_buckets * d * w],
            epoch_of_bucket: vec![u64::MAX; n_buckets],
            current_epoch: 0,
        })
    }

    /// Epoch index for a timestamp.
    fn epoch(&self, timestamp: u64) -> u64 {
        timestamp / self.bucket_span
    }

    /// Ensure the bucket for `epoch` is live (clearing it if it currently holds a
    /// stale epoch), returning its bucket index. Also advances `current_epoch`
    /// and evicts any buckets that have fallen out of the `[epoch - B + 1, epoch]`
    /// window.
    fn activate_bucket(&mut self, epoch: u64) -> usize {
        if epoch > self.current_epoch {
            self.current_epoch = epoch;
            // Evict every bucket whose stored epoch is now older than the window.
            let cutoff = epoch.saturating_sub(self.n_buckets as u64 - 1);
            for b in 0..self.n_buckets {
                let stored = self.epoch_of_bucket[b];
                if stored != u64::MAX && stored < cutoff {
                    self.clear_bucket(b);
                }
            }
        }
        let idx = (epoch % self.n_buckets as u64) as usize;
        if self.epoch_of_bucket[idx] != epoch {
            // Re-purpose this slot for the new epoch.
            self.clear_bucket(idx);
            self.epoch_of_bucket[idx] = epoch;
        }
        idx
    }

    fn clear_bucket(&mut self, b: usize) {
        let base = b * self.d * self.w;
        for cell in self.tables[base..base + self.d * self.w].iter_mut() {
            *cell = 0;
        }
        self.epoch_of_bucket[b] = u64::MAX;
    }

    /// Stream `f[key] += count` at time `timestamp`.
    ///
    /// Timestamps must be **non-decreasing**; an out-of-order timestamp older
    /// than the current window is dropped (it cannot be represented).
    pub fn update(&mut self, key: u64, count: u64, timestamp: u64) {
        let epoch = self.epoch(timestamp);
        // Reject updates older than the current window entirely.
        if self.current_epoch >= self.n_buckets as u64
            && epoch <= self.current_epoch - self.n_buckets as u64
        {
            return;
        }
        let bucket = self.activate_bucket(epoch);
        let base = bucket * self.d * self.w;
        for row in 0..self.d {
            let col = self.hashes[row].hash(key) as usize;
            let cell = base + row * self.w + col;
            self.tables[cell] = self.tables[cell].saturating_add(count);
        }
    }

    /// Insert one occurrence of `key` at `timestamp`.
    pub fn add(&mut self, key: u64, timestamp: u64) {
        self.update(key, 1, timestamp);
    }

    /// Query the windowed frequency of `key` as of the latest observed epoch.
    ///
    /// Sums the per-bucket Count-Min minimum across all live buckets. The result
    /// never under-estimates the true windowed frequency.
    #[must_use]
    pub fn query(&self, key: u64) -> u64 {
        let cutoff = self.current_epoch.saturating_sub(self.n_buckets as u64 - 1);
        let mut total = 0u64;
        for b in 0..self.n_buckets {
            let stored = self.epoch_of_bucket[b];
            if stored == u64::MAX || stored < cutoff {
                continue;
            }
            let base = b * self.d * self.w;
            let mut best = u64::MAX;
            for row in 0..self.d {
                let col = self.hashes[row].hash(key) as usize;
                let v = self.tables[base + row * self.w + col];
                if v < best {
                    best = v;
                }
            }
            if best != u64::MAX {
                total = total.saturating_add(best);
            }
        }
        total
    }

    /// Number of live (non-evicted) buckets currently in the window.
    #[must_use]
    pub fn live_buckets(&self) -> usize {
        let cutoff = self.current_epoch.saturating_sub(self.n_buckets as u64 - 1);
        self.epoch_of_bucket
            .iter()
            .filter(|&&e| e != u64::MAX && e >= cutoff)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swcm_constructs() {
        let mut rng = LcgRng::new(1);
        let s = SlidingWindowCm::new(4, 256, 100, 10, &mut rng).expect("ok");
        assert_eq!(s.n_buckets, 10);
        assert_eq!(s.bucket_span, 10);
    }

    #[test]
    fn swcm_invalid_params() {
        let mut rng = LcgRng::new(0);
        assert!(SlidingWindowCm::new(0, 16, 100, 8, &mut rng).is_err());
        assert!(SlidingWindowCm::new(4, 0, 100, 8, &mut rng).is_err());
        assert!(SlidingWindowCm::new(4, 16, 0, 8, &mut rng).is_err());
        assert!(SlidingWindowCm::new(4, 16, 100, 0, &mut rng).is_err());
    }

    #[test]
    fn swcm_recent_counts_overestimate() {
        let mut rng = LcgRng::new(11);
        let mut s = SlidingWindowCm::new(5, 2048, 100, 10, &mut rng).expect("ok");
        // All within one window at increasing timestamps.
        for t in 0..100u64 {
            s.add(7, t);
        }
        let q = s.query(7);
        assert!(q >= 100, "windowed estimate {q} under-counted 100");
        assert!(q < 130, "windowed estimate {q} grossly over-counted");
    }

    #[test]
    fn swcm_old_data_expires() {
        let mut rng = LcgRng::new(13);
        let mut s = SlidingWindowCm::new(5, 2048, 100, 10, &mut rng).expect("ok");
        // 50 hits at the very start of time.
        for t in 0..50u64 {
            s.add(9, t);
        }
        assert!(s.query(9) >= 50);
        // Advance well beyond the window; the old hits must expire.
        for t in 1000..1010u64 {
            s.add(123, t);
        }
        let q = s.query(9);
        assert_eq!(q, 0, "expired key 9 should read 0, got {q}");
    }

    #[test]
    fn swcm_window_boundary_partial_expiry() {
        let mut rng = LcgRng::new(21);
        // window 100, 10 buckets ⇒ span 10.
        let mut s = SlidingWindowCm::new(6, 4096, 100, 10, &mut rng).expect("ok");
        // key A at t = 0..10 (epoch 0), key B at t = 95 (epoch 9).
        for t in 0..10u64 {
            s.add(1, t);
        }
        s.add(2, 95);
        // Advance to t = 105 (epoch 10): epoch 0 falls out, epoch 9 stays.
        s.add(3, 105);
        assert_eq!(s.query(1), 0, "epoch-0 key should have expired");
        assert!(s.query(2) >= 1, "epoch-9 key should still be live");
        assert!(s.query(3) >= 1);
    }

    #[test]
    fn swcm_out_of_window_old_update_dropped() {
        let mut rng = LcgRng::new(31);
        let mut s = SlidingWindowCm::new(4, 1024, 50, 5, &mut rng).expect("ok");
        // Move current epoch forward first.
        for t in 200..210u64 {
            s.add(5, t);
        }
        let before = s.query(5);
        // A stale update at t = 0 is far older than the window ⇒ ignored.
        s.update(5, 1000, 0);
        assert_eq!(s.query(5), before, "stale update must be dropped");
    }

    #[test]
    fn swcm_distinct_keys_independent() {
        let mut rng = LcgRng::new(41);
        let mut s = SlidingWindowCm::new(6, 4096, 100, 10, &mut rng).expect("ok");
        for t in 0..100u64 {
            s.add(t % 5, t); // 5 keys, 20 each within the window
        }
        for k in 0..5u64 {
            let q = s.query(k);
            assert!(q >= 20, "key {k} windowed count {q} under 20");
            assert!(q < 45, "key {k} windowed count {q} too high");
        }
    }

    #[test]
    fn swcm_live_buckets_tracks_window() {
        let mut rng = LcgRng::new(51);
        let mut s = SlidingWindowCm::new(4, 512, 100, 10, &mut rng).expect("ok");
        for t in 0..100u64 {
            s.add(1, t);
        }
        // Updates spanning epochs 0..9 ⇒ all 10 buckets live.
        assert_eq!(s.live_buckets(), 10);
    }
}

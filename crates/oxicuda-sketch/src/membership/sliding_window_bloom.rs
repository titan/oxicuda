//! Sliding-window Bloom filter with time-decaying buckets.
//!
//! Answers "was `x` inserted within the last `W` time units?" in Bloom-filter
//! space, with no false negatives for items currently inside the window and a
//! tunable false-positive rate. The structure is a ring of `B` sub-filters
//! ("time buckets"), each a classic Bloom filter sharing the same `(m, k)`
//! geometry and hash seed; insertions land in the bucket selected by the epoch
//! `⌊t / bucket_span⌋ mod B`.
//!
//! ## Window semantics
//!
//! With `bucket_span = ⌈W / B⌉`, the ring spans ≈ one window. When time advances
//! into a new epoch, the bucket about to be reused is **cleared** (its contents
//! are now older than `W`), giving a jumping-window approximation: a positive
//! `contains` means the item was seen within the last `W` to `W + bucket_span`
//! units. Larger `B` ⇒ tighter window resolution at higher memory cost.
//!
//! A membership test ORs the per-bucket results across all live buckets, so it
//! never reports a false negative for an item inserted inside the window. The
//! effective false-positive rate is bounded by `B ×` (single-filter FP rate)
//! in the worst case, though in practice each item touches only the buckets in
//! which it was actually inserted.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

/// Sliding-window Bloom filter.
#[derive(Debug, Clone)]
pub struct SlidingWindowBloom {
    /// Bits per sub-filter.
    pub m: usize,
    /// Hash functions per item.
    pub k: usize,
    /// Number of time buckets in the ring.
    pub n_buckets: usize,
    /// Timestamp span covered by one bucket.
    pub bucket_span: u64,
    /// Shared base seed for the (double-hashing) hash family.
    seed_base: u64,
    /// Words per sub-filter (`⌈m/64⌉`).
    words_per_bucket: usize,
    /// `n_buckets · words_per_bucket` bit words, bucket-major.
    bits: Vec<u64>,
    /// Epoch stored in each bucket (`u64::MAX` ⇒ empty).
    epoch_of_bucket: Vec<u64>,
    /// Highest epoch observed so far.
    current_epoch: u64,
}

impl SlidingWindowBloom {
    /// Construct a sliding-window Bloom filter.
    ///
    /// * `m`, `k` — per-bucket Bloom geometry (`> 0`).
    /// * `window` — window length `W` in timestamp units (`> 0`).
    /// * `n_buckets` — temporal resolution `B` (`> 0`); `bucket_span = ⌈W/B⌉`.
    pub fn new(
        m: usize,
        k: usize,
        window: u64,
        n_buckets: usize,
        seed_base: u64,
    ) -> SketchResult<Self> {
        if m == 0 || k == 0 || n_buckets == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(m, k, n_buckets)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if window == 0 {
            return Err(SketchError::InvalidParameter {
                name: "window".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let words_per_bucket = m.div_ceil(64);
        let bucket_span = window.div_ceil(n_buckets as u64).max(1);
        Ok(Self {
            m,
            k,
            n_buckets,
            bucket_span,
            seed_base,
            words_per_bucket,
            bits: vec![0u64; n_buckets * words_per_bucket],
            epoch_of_bucket: vec![u64::MAX; n_buckets],
            current_epoch: 0,
        })
    }

    /// Choose `(m, k)` to hold `n_expected` items **per bucket** at target FP
    /// rate `p`, then build the windowed filter with `n_buckets` buckets.
    pub fn with_expected_fp(
        n_expected: usize,
        p: f64,
        window: u64,
        n_buckets: usize,
        seed_base: u64,
    ) -> SketchResult<Self> {
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
        let k = ((m as f64 / n_expected as f64) * ln2).round() as usize;
        Self::new(m.max(1), k.max(1), window, n_buckets, seed_base)
    }

    fn epoch(&self, timestamp: u64) -> u64 {
        timestamp / self.bucket_span
    }

    fn positions(&self, x: u64) -> impl Iterator<Item = usize> + '_ {
        let h1 = xxh3_64_u64(x, self.seed_base);
        let h2 = xxh3_64_u64(x, self.seed_base.wrapping_add(0x9E37_79B9_7F4A_7C15));
        let m = self.m;
        (0..self.k).map(move |i| (h1.wrapping_add((i as u64).wrapping_mul(h2)) as usize) % m)
    }

    /// Activate (and if necessary clear) the bucket for `epoch`, advancing the
    /// window and evicting stale buckets. Returns the bucket index.
    fn activate_bucket(&mut self, epoch: u64) -> usize {
        if epoch > self.current_epoch {
            self.current_epoch = epoch;
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
            self.clear_bucket(idx);
            self.epoch_of_bucket[idx] = epoch;
        }
        idx
    }

    fn clear_bucket(&mut self, b: usize) {
        let base = b * self.words_per_bucket;
        for w in self.bits[base..base + self.words_per_bucket].iter_mut() {
            *w = 0;
        }
        self.epoch_of_bucket[b] = u64::MAX;
    }

    /// Insert `x` at time `timestamp` (timestamps must be non-decreasing). A
    /// stale timestamp older than the current window is dropped.
    pub fn insert(&mut self, x: u64, timestamp: u64) {
        let epoch = self.epoch(timestamp);
        if self.current_epoch >= self.n_buckets as u64
            && epoch <= self.current_epoch - self.n_buckets as u64
        {
            return;
        }
        let positions: Vec<usize> = self.positions(x).collect();
        let bucket = self.activate_bucket(epoch);
        let base = bucket * self.words_per_bucket;
        for pos in positions {
            self.bits[base + pos / 64] |= 1u64 << (pos % 64);
        }
    }

    /// Test whether `x` was inserted within the current window. Never a false
    /// negative for items still inside the window.
    #[must_use]
    pub fn contains(&self, x: u64) -> bool {
        let cutoff = self.current_epoch.saturating_sub(self.n_buckets as u64 - 1);
        let positions: Vec<usize> = self.positions(x).collect();
        for b in 0..self.n_buckets {
            let stored = self.epoch_of_bucket[b];
            if stored == u64::MAX || stored < cutoff {
                continue;
            }
            let base = b * self.words_per_bucket;
            let all_set = positions
                .iter()
                .all(|&pos| (self.bits[base + pos / 64] >> (pos % 64)) & 1 == 1);
            if all_set {
                return true;
            }
        }
        false
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
    fn swb_constructs() {
        let s = SlidingWindowBloom::new(1024, 5, 100, 10, 0).expect("ok");
        assert_eq!(s.n_buckets, 10);
        assert_eq!(s.bucket_span, 10);
    }

    #[test]
    fn swb_invalid_params() {
        assert!(SlidingWindowBloom::new(0, 5, 100, 10, 0).is_err());
        assert!(SlidingWindowBloom::new(16, 0, 100, 10, 0).is_err());
        assert!(SlidingWindowBloom::new(16, 5, 0, 10, 0).is_err());
        assert!(SlidingWindowBloom::new(16, 5, 100, 0, 0).is_err());
        assert!(SlidingWindowBloom::with_expected_fp(100, 0.0, 50, 8, 0).is_err());
        assert!(SlidingWindowBloom::with_expected_fp(0, 0.01, 50, 8, 0).is_err());
    }

    #[test]
    fn swb_no_false_negative_in_window() {
        let mut s = SlidingWindowBloom::new(8192, 6, 100, 10, 7).expect("ok");
        for t in 0..100u64 {
            s.insert(t, t);
        }
        for t in 0..100u64 {
            assert!(s.contains(t), "item {t} should be present within window");
        }
    }

    #[test]
    fn swb_old_items_expire() {
        let mut s = SlidingWindowBloom::new(8192, 6, 100, 10, 1).expect("ok");
        s.insert(42, 0);
        assert!(s.contains(42));
        // Advance far past the window.
        for t in 1000..1010u64 {
            s.insert(7, t);
        }
        assert!(!s.contains(42), "expired item 42 should read absent");
        assert!(s.contains(7));
    }

    #[test]
    fn swb_window_boundary() {
        let mut s = SlidingWindowBloom::new(16384, 7, 100, 10, 3).expect("ok");
        s.insert(1, 5); // epoch 0
        s.insert(2, 95); // epoch 9
        s.insert(3, 105); // epoch 10 ⇒ epoch 0 evicted
        assert!(!s.contains(1), "epoch-0 item should have expired");
        assert!(s.contains(2), "epoch-9 item should remain");
        assert!(s.contains(3));
    }

    #[test]
    fn swb_stale_insert_dropped() {
        let mut s = SlidingWindowBloom::new(4096, 5, 50, 5, 9).expect("ok");
        for t in 200..210u64 {
            s.insert(5, t);
        }
        // Stale insert far older than the window is ignored, so a fresh key at
        // t=0 must NOT become a member.
        s.insert(999, 0);
        assert!(!s.contains(999), "stale insert must be dropped");
    }

    #[test]
    fn swb_fp_rate_bounded() {
        let mut s = SlidingWindowBloom::with_expected_fp(500, 0.01, 100, 10, 11).expect("ok");
        // Fill the window with 500 items spread across epochs.
        for i in 0..500u64 {
            let t = i % 100;
            s.insert(i, t);
        }
        let mut fp = 0usize;
        let trials = 10_000u64;
        for i in 1_000_000..1_000_000 + trials {
            if s.contains(i) {
                fp += 1;
            }
        }
        let rate = fp as f64 / trials as f64;
        // Windowed OR inflates FP vs a single filter but must stay modest.
        assert!(rate < 0.10, "windowed Bloom FP rate {rate} too high");
    }

    #[test]
    fn swb_live_buckets() {
        let mut s = SlidingWindowBloom::new(1024, 5, 100, 10, 0).expect("ok");
        for t in 0..100u64 {
            s.insert(1, t);
        }
        assert_eq!(s.live_buckets(), 10);
    }
}

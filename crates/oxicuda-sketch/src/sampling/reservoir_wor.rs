//! Reservoir Sampling Without Replacement — Algorithm L.
//!
//! Li 1994 / Kim-Hung Li: "Reservoir-sampling algorithms of time complexity O(n(1+log(N/n)))".
//! Based on Vitter 1985 insights, Algorithm L achieves O(n/k) random number generations
//! instead of the O(n) required by Algorithm R.
//!
//! ## Algorithm L synopsis
//!
//! After filling the reservoir with the first k items:
//!
//! 1. W ← exp(ln(U₁) / k), U₁ ~ Uniform(0, 1)  (initial skip weight)
//! 2. skip ← ⌊ln(U₂) / ln(1 − W)⌋, U₂ ~ Uniform(0, 1)  (geometric skip distance)
//! 3. Skip the next `skip` stream items without processing.
//! 4. Replace a uniformly random slot in the reservoir with the arriving item.
//! 5. Update W ← W · exp(ln(U₃) / k)
//! 6. Compute new skip ← ⌊ln(U₄) / ln(1 − W)⌋
//! 7. Goto 3.
//!
//! This produces a uniform random sample of exactly k items (or all items if n < k).

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Compute a new skip weight W = exp(ln(U) / k).
///
/// W ∈ (0, 1) for k ≥ 1 and U ∈ (0, 1).
/// Clamped to ensure numerical stability.
#[inline]
fn compute_w(rng: &mut LcgRng, k: usize) -> f64 {
    let u = rng.next_f64().max(1e-300); // avoid ln(0)
    (u.ln() / k as f64).exp()
}

/// Compute the geometric skip distance = ⌊ln(U) / ln(1 − W)⌋.
///
/// Returns u64::MAX when W ≥ 1 (degenerate), or 0 when the skip collapses.
#[inline]
fn compute_skip(rng: &mut LcgRng, w: f64) -> u64 {
    if w >= 1.0 {
        // W ≥ 1 means every item would be accepted; treat as zero skip.
        return 0;
    }
    let one_minus_w = 1.0 - w;
    if one_minus_w <= 0.0 {
        return 0;
    }
    let log_denom = one_minus_w.ln(); // ln(1 - W) < 0
    if log_denom >= 0.0 {
        // Numerically degenerate: (1-W) >= 1 would mean W ≤ 0, handled above.
        return 0;
    }
    let u = rng.next_f64().max(1e-300);
    let skip_f = u.ln() / log_denom; // both numerator and denominator are negative → positive
    if !skip_f.is_finite() || skip_f < 0.0 {
        return 0;
    }
    skip_f.floor() as u64
}

/// Reservoir sampler using Algorithm L for efficient sampling without replacement.
///
/// More efficient than Algorithm R (Vitter 1985): generates O(n/k) random numbers
/// instead of O(n), using geometric skip distances between reservoir updates.
///
/// The reservoir holds exactly `min(n_seen, k)` uniformly random elements from the stream.
#[derive(Debug, Clone)]
pub struct ReservoirWor {
    /// Reservoir capacity.
    pub k: usize,
    /// Current reservoir contents (len = min(n_seen, k)).
    pub reservoir: Vec<u64>,
    /// Total items processed so far.
    pub n_seen: u64,
    /// Internal RNG (LCG MMIX variant).
    rng: LcgRng,
    /// Current skip weight W ∈ (0, 1). Valid only once the reservoir is full.
    w: f64,
    /// Items to skip before the next replacement. Decremented on each `add()` call.
    skip: u64,
}

impl ReservoirWor {
    /// Create a new reservoir of capacity `k` using Algorithm L.
    ///
    /// `k` must be ≥ 1. The reservoir starts empty and fills until k items are seen.
    pub fn new(k: usize, seed: u64) -> SketchResult<Self> {
        if k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let mut rng = LcgRng::new(seed);
        // Warm up the LCG to escape correlated startup transients.
        for _ in 0..8 {
            let _ = rng.next_u64();
        }
        Ok(Self {
            k,
            reservoir: Vec::with_capacity(k),
            n_seen: 0,
            rng,
            w: 0.0,
            skip: 0,
        })
    }

    /// Process one item from the stream (Algorithm L).
    ///
    /// During the filling phase (n_seen < k): append directly.
    /// After the reservoir becomes full for the first time: initialise W and skip.
    /// Thereafter: decrement skip or perform a replacement when skip reaches 0.
    pub fn add(&mut self, x: u64) {
        self.n_seen += 1;

        if self.reservoir.len() < self.k {
            // Filling phase: collect the first k items verbatim.
            self.reservoir.push(x);
            // Transition: when we just reached capacity, bootstrap W and skip.
            if self.reservoir.len() == self.k {
                self.w = compute_w(&mut self.rng, self.k);
                self.skip = compute_skip(&mut self.rng, self.w);
            }
            return;
        }

        // Steady state: reservoir is full.
        if self.skip == 0 {
            // Replace a uniformly random slot.
            let idx = self.rng.next_usize(self.k);
            self.reservoir[idx] = x;
            // Update W: multiply by a new geometric factor.
            let w_factor = compute_w(&mut self.rng, self.k);
            self.w *= w_factor;
            // Recompute skip distance.
            self.skip = compute_skip(&mut self.rng, self.w);
        } else {
            // Skip this item; no random number consumed.
            self.skip -= 1;
        }
    }

    /// Process a batch of items efficiently.
    ///
    /// Items are processed in order, identical to calling `add()` for each.
    pub fn add_batch(&mut self, items: &[u64]) {
        for &item in items {
            self.add(item);
        }
    }

    /// View the current reservoir contents (may be smaller than k if fewer items seen).
    #[must_use]
    pub fn sample(&self) -> &[u64] {
        &self.reservoir
    }

    /// Whether the reservoir has been filled to capacity k.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.reservoir.len() >= self.k
    }

    /// Total number of items processed so far.
    #[must_use]
    pub fn n_seen(&self) -> u64 {
        self.n_seen
    }

    /// Fraction of items retained: min(k, n_seen) / n_seen, or 1.0 if reservoir not yet full.
    #[must_use]
    pub fn retention_rate(&self) -> f64 {
        if self.n_seen == 0 {
            return 1.0;
        }
        (self.reservoir.len() as f64) / (self.n_seen as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn reservoir_wor_new_invalid_k() {
        assert!(ReservoirWor::new(0, 0).is_err());
    }

    #[test]
    fn reservoir_wor_empty_sample() {
        let r = ReservoirWor::new(10, 0).expect("ok");
        assert_eq!(r.sample().len(), 0);
        assert!(!r.is_full());
    }

    #[test]
    fn reservoir_wor_fill_first_k() {
        // Adding exactly k items → all retained, order may differ.
        let k = 8usize;
        let mut r = ReservoirWor::new(k, 1).expect("ok");
        for i in 0u64..k as u64 {
            r.add(i);
        }
        assert_eq!(r.sample().len(), k);
        assert!(r.is_full());
        let set: HashSet<u64> = r.sample().iter().copied().collect();
        let expected: HashSet<u64> = (0u64..k as u64).collect();
        assert_eq!(set, expected);
    }

    #[test]
    fn reservoir_wor_k1_retains_one() {
        let mut r = ReservoirWor::new(1, 2).expect("ok");
        for i in 0u64..100 {
            r.add(i);
        }
        assert_eq!(r.sample().len(), 1);
    }

    #[test]
    fn reservoir_wor_sample_size_bounded() {
        let k = 50usize;
        let mut r = ReservoirWor::new(k, 3).expect("ok");
        for i in 0u64..200 {
            r.add(i);
            assert!(
                r.sample().len() <= k,
                "sample len {} exceeded k={} at i={}",
                r.sample().len(),
                k,
                i
            );
        }
    }

    #[test]
    fn reservoir_wor_n_seen_tracked() {
        let mut r = ReservoirWor::new(10, 4).expect("ok");
        for i in 0u64..37 {
            r.add(i);
        }
        assert_eq!(r.n_seen(), 37);
    }

    #[test]
    fn reservoir_wor_is_full() {
        let k = 5usize;
        let mut r = ReservoirWor::new(k, 5).expect("ok");
        for i in 0u64..(k as u64 - 1) {
            r.add(i);
            assert!(!r.is_full(), "should not be full yet at i={i}");
        }
        r.add(99);
        assert!(r.is_full(), "should be full after k items");
    }

    #[test]
    fn reservoir_wor_retention_rate() {
        let k = 100usize;
        let n = 1000u64;
        let mut r = ReservoirWor::new(k, 6).expect("ok");
        for i in 0..n {
            r.add(i);
        }
        let rate = r.retention_rate();
        let expected = k as f64 / n as f64;
        assert!(
            (rate - expected).abs() < 1e-12,
            "retention rate {rate} != expected {expected}"
        );
    }

    #[test]
    fn reservoir_wor_uniformity_100k() {
        // 100k items from 10 equally-sized buckets; each bucket should have ~10% share ± 30%.
        let n = 100_000u64;
        let k = 1_000usize;
        let n_buckets = 10u64;
        let mut r = ReservoirWor::new(k, 7).expect("ok");
        for i in 0..n {
            r.add(i);
        }
        let mut counts = vec![0usize; n_buckets as usize];
        for &v in r.sample() {
            let bucket = (v * n_buckets / n) as usize;
            let bucket = bucket.min(n_buckets as usize - 1);
            counts[bucket] += 1;
        }
        let expected = k / n_buckets as usize; // 100
        for (b, &c) in counts.iter().enumerate() {
            let rel = (c as f64 - expected as f64).abs() / expected as f64;
            assert!(
                rel < 0.30,
                "bucket {b} non-uniform: count={c}, expected~{expected}, rel={rel:.3}"
            );
        }
    }

    #[test]
    fn reservoir_wor_all_distinct_included_when_small() {
        // 5 items with k=10 → all 5 must appear.
        let k = 10usize;
        let n = 5u64;
        let mut r = ReservoirWor::new(k, 8).expect("ok");
        for i in 0..n {
            r.add(i);
        }
        let set: HashSet<u64> = r.sample().iter().copied().collect();
        for i in 0..n {
            assert!(set.contains(&i), "item {i} missing from small stream");
        }
    }

    #[test]
    fn reservoir_wor_batch_equals_individual() {
        // add_batch([1, 2, 3]) must produce same reservoir as three add() calls with same seed.
        let seed = 9u64;
        let mut r1 = ReservoirWor::new(5, seed).expect("ok");
        r1.add(1);
        r1.add(2);
        r1.add(3);

        let mut r2 = ReservoirWor::new(5, seed).expect("ok");
        r2.add_batch(&[1, 2, 3]);

        assert_eq!(r1.sample(), r2.sample());
        assert_eq!(r1.n_seen(), r2.n_seen());
    }

    #[test]
    fn reservoir_wor_deterministic() {
        // Same seed + same items → same reservoir.
        let seed = 42u64;
        let items: Vec<u64> = (0..200).collect();

        let mut r1 = ReservoirWor::new(20, seed).expect("ok");
        r1.add_batch(&items);

        let mut r2 = ReservoirWor::new(20, seed).expect("ok");
        r2.add_batch(&items);

        assert_eq!(r1.sample(), r2.sample());
    }

    #[test]
    fn reservoir_wor_large_stream() {
        // n = 1_000_000, k = 100 → exactly k items in reservoir.
        let k = 100usize;
        let mut r = ReservoirWor::new(k, 11).expect("ok");
        for i in 0u64..1_000_000 {
            r.add(i);
        }
        assert_eq!(r.sample().len(), k);
        assert_eq!(r.n_seen(), 1_000_000);
    }

    #[test]
    fn reservoir_wor_w_stays_valid() {
        // No panics for various seeds and stream sizes.
        for seed in [0u64, 1, 13, 99, u64::MAX / 2, u64::MAX] {
            let mut r = ReservoirWor::new(16, seed).expect("ok");
            for i in 0u64..500 {
                r.add(i);
            }
            let rate = r.retention_rate();
            assert!(rate.is_finite() && rate > 0.0 && rate <= 1.0);
        }
    }

    #[test]
    fn reservoir_wor_sample_all_from_input() {
        // All reservoir items must have come from the input.
        let k = 30usize;
        let n = 200u64;
        let items: Vec<u64> = (0..n).map(|i| i * 7 + 3).collect();
        let item_set: HashSet<u64> = items.iter().copied().collect();
        let mut r = ReservoirWor::new(k, 12).expect("ok");
        r.add_batch(&items);
        for &v in r.sample() {
            assert!(
                item_set.contains(&v),
                "reservoir item {v} was not in the input stream"
            );
        }
    }

    #[test]
    fn reservoir_wor_k_equals_n() {
        // When k == n, all items must be retained.
        let k = 50usize;
        let items: Vec<u64> = (0u64..k as u64).collect();
        let mut r = ReservoirWor::new(k, 13).expect("ok");
        r.add_batch(&items);
        let set: HashSet<u64> = r.sample().iter().copied().collect();
        let expected: HashSet<u64> = items.iter().copied().collect();
        assert_eq!(set, expected, "k==n: all items must be in reservoir");
    }

    #[test]
    fn reservoir_wor_two_items_k1() {
        // k=1, 2 items → exactly 1 in reservoir.
        let mut r = ReservoirWor::new(1, 14).expect("ok");
        r.add(10);
        r.add(20);
        assert_eq!(r.sample().len(), 1);
        let v = r.sample()[0];
        assert!(v == 10 || v == 20, "unexpected reservoir value {v}");
    }

    #[test]
    fn reservoir_wor_retention_rate_approx_k_over_n() {
        // After many items, retention rate ≈ k / n_seen.
        let k = 200usize;
        let n = 10_000u64;
        let mut r = ReservoirWor::new(k, 15).expect("ok");
        for i in 0..n {
            r.add(i);
        }
        let rate = r.retention_rate();
        let expected = k as f64 / n as f64;
        let rel = (rate - expected).abs() / expected;
        assert!(rel < 1e-10, "retention rate {rate:.6} != k/n {expected:.6}");
    }
}

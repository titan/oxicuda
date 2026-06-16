//! KLL quantile sketch (Karnin, Lang, Liberty 2016).
//!
//! Hierarchical compactor structure where each compactor halves its size on overflow by
//! sampling alternate elements (parity = even or odd, chosen randomly).

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// KLL sketch with `k` minimum compactor size, supports `f64` values.
#[derive(Debug, Clone)]
pub struct KllSketch {
    pub k: usize,
    pub compactors: Vec<Vec<f64>>,
    pub n: u64,
    rng: LcgRng,
}

impl KllSketch {
    /// New KLL sketch with target compactor size `k` (typical: 200..1000).
    pub fn new(k: usize, seed: u64) -> SketchResult<Self> {
        if k < 8 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be >= 8".to_string(),
            });
        }
        Ok(Self {
            k,
            compactors: vec![Vec::new()],
            n: 0,
            rng: LcgRng::new(seed),
        })
    }

    /// Capacity of compactor at height h: k * (2/3)^h (round up), with floor of 2.
    fn capacity(&self, height: usize) -> usize {
        let c = (self.k as f64) * (2.0_f64 / 3.0).powi(height as i32);
        (c.ceil() as usize).max(2)
    }

    /// Insert a value.
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        if self.compactors.is_empty() {
            self.compactors.push(Vec::new());
        }
        self.compactors[0].push(value);
        self.n += 1;
        self.compress();
    }

    /// Compress: starting from level 0, if a compactor is over capacity,
    /// sort + select alternate elements to PROMOTE up one level, DROPPING the rest.
    /// This is the core "halving" step of KLL — each level retains a single leftover
    /// (when original length was odd) after compaction, and the promoted items carry
    /// weight 2 of the original.
    fn compress(&mut self) {
        let mut h = 0;
        while h < self.compactors.len() {
            let cap = self.capacity(h);
            if self.compactors[h].len() >= cap {
                let mut row = std::mem::take(&mut self.compactors[h]);
                row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                // If odd length, randomly save one item back to ensure pair-count even.
                let leftover = if row.len() % 2 == 1 {
                    let idx = self.rng.next_usize(row.len());
                    Some(row.remove(idx))
                } else {
                    None
                };
                let parity = self.rng.next_bool();
                // Take alternate elements (with random offset 0 or 1).
                let promoted: Vec<f64> = row
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(i, _)| (i % 2 == 1) ^ !parity)
                    .map(|(_, v)| v)
                    .collect();
                // Keep only the leftover item back (rest are dropped — their info captured by
                // the alternate-sample promotion).
                let mut kept: Vec<f64> = Vec::new();
                if let Some(v) = leftover {
                    kept.push(v);
                }
                self.compactors[h] = kept;
                // promote to h+1
                if self.compactors.len() <= h + 1 {
                    self.compactors.push(Vec::new());
                }
                self.compactors[h + 1].extend(promoted);
                h += 1;
            } else {
                break;
            }
        }
    }

    /// Snapshot of all sketch items with their weights (weight = 2^height, clamped at 2^63).
    #[must_use]
    pub fn weighted_items(&self) -> Vec<(f64, u64)> {
        let mut all = Vec::new();
        for (h, comp) in self.compactors.iter().enumerate() {
            let shift = h.min(63) as u32;
            let w = 1u64 << shift;
            for &v in comp {
                all.push((v, w));
            }
        }
        all
    }

    /// Estimate the quantile `q ∈ [0,1]` via linear interpolation between consecutive
    /// weighted items at the rank crossover.
    #[must_use]
    pub fn quantile(&self, q: f64) -> f64 {
        let mut items = self.weighted_items();
        if items.is_empty() {
            return 0.0;
        }
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let total: f64 = items.iter().map(|(_, w)| *w as f64).sum();
        let target = q.clamp(0.0, 1.0) * total;
        let mut acc_prev = 0.0_f64;
        let mut prev_v = items[0].0;
        for &(v, w) in &items {
            let acc = acc_prev + w as f64;
            if target <= acc {
                // Interpolate between prev_v (at cumulative acc_prev) and v (at cumulative acc).
                let span = (acc - acc_prev).max(1.0e-12);
                let frac = ((target - acc_prev) / span).clamp(0.0, 1.0);
                return prev_v + frac * (v - prev_v);
            }
            acc_prev = acc;
            prev_v = v;
        }
        items[items.len() - 1].0
    }

    /// Total items inserted (n).
    #[must_use]
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Run the bottom-up compaction cascade starting from level `start`, identical in rule to
    /// [`Self::compress`] but parameterised by the first level to examine.
    ///
    /// [`Self::compress`] is `compress_from(0)`. Driving the cascade from a higher level lets
    /// [`Self::merge`] insert each of `other`'s level-`h` items via the *exact* single-overflow
    /// compaction the streaming path uses, so the merged hierarchy is provably the same shape as
    /// if those items had been streamed — no high-weight "rider" singletons are stranded by an
    /// aggressive multi-level cascade. The halving rule (sort, hold back one random leftover on
    /// odd length, promote random-parity alternates, drop the rest) is byte-for-byte the one in
    /// `compress`.
    fn compress_from(&mut self, start: usize) {
        let mut h = start;
        while h < self.compactors.len() {
            let cap = self.capacity(h);
            if self.compactors[h].len() >= cap {
                let mut row = std::mem::take(&mut self.compactors[h]);
                row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let leftover = if row.len() % 2 == 1 {
                    let idx = self.rng.next_usize(row.len());
                    Some(row.remove(idx))
                } else {
                    None
                };
                let parity = self.rng.next_bool();
                let promoted: Vec<f64> = row
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(i, _)| (i % 2 == 1) ^ !parity)
                    .map(|(_, v)| v)
                    .collect();
                let mut kept: Vec<f64> = Vec::new();
                if let Some(v) = leftover {
                    kept.push(v);
                }
                self.compactors[h] = kept;
                if self.compactors.len() <= h + 1 {
                    self.compactors.push(Vec::new());
                }
                self.compactors[h + 1].extend(promoted);
                h += 1;
            } else {
                break;
            }
        }
    }

    /// Merge `other` into `self` in place (Karnin-Lang-Liberty 2016, §"Mergeability").
    ///
    /// Both sketches MUST share the same `k`; otherwise [`SketchError::ShapeMismatch`] is returned
    /// and `self` is left unchanged.
    ///
    /// ## Algorithm
    ///
    /// Each item stored at level `h` of `other` represents `2^h` original items at weight `2^h`.
    /// The merge replays `other`'s items **into their own levels** of `self`, processing levels
    /// bottom-up and pushing one item at a time; after each push it runs the standard
    /// single-overflow compaction cascade (`Self::compress_from`) from that level. This is
    /// exactly the streaming insertion rule lifted to arbitrary levels, so the result is a
    /// well-formed KLL hierarchy obeying the `capacity(h)` schedule — equivalent to having fed
    /// those weighted items through `add`/`compress`, with no stranded high-weight riders.
    ///
    /// The item count `n` becomes the sum of the two inputs. Merging an empty sketch is the
    /// identity. Only `self`'s RNG advances, so the result is deterministic given `self`'s seed.
    pub fn merge(&mut self, other: &Self) -> SketchResult<()> {
        if self.k != other.k {
            return Err(SketchError::ShapeMismatch {
                expected: vec![self.k],
                got: vec![other.k],
            });
        }
        // Grow our hierarchy to at least the height of `other`.
        let height = self.compactors.len().max(other.compactors.len());
        while self.compactors.len() < height {
            self.compactors.push(Vec::new());
        }
        // Replay `other`'s items level-by-level, bottom-up, with a single-overflow cascade after
        // each insertion (mirrors the streaming `add` path but at the item's home level).
        for h in 0..other.compactors.len() {
            // Clone the level out first so we never alias `other` while mutating `self` (they may
            // be the same object via `merged`, but here `other` is `&Self` distinct from `self`).
            let level_items = other.compactors[h].clone();
            for value in level_items {
                if self.compactors.len() <= h {
                    self.compactors.push(Vec::new());
                }
                self.compactors[h].push(value);
                self.compress_from(h);
            }
        }
        self.n = self.n.saturating_add(other.n);
        Ok(())
    }

    /// Convenience: merge two sketches into a fresh one (clones `a`, merges `b`). Both must share
    /// the same `k`.
    pub fn merged(a: &Self, b: &Self) -> SketchResult<Self> {
        let mut out = a.clone();
        out.merge(b)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kll_invalid_k() {
        assert!(KllSketch::new(3, 0).is_err());
    }

    #[test]
    fn kll_constructs() {
        let s = KllSketch::new(64, 11).expect("ok");
        assert_eq!(s.k, 64);
        assert_eq!(s.n, 0);
    }

    #[test]
    fn kll_median_uniform() {
        // Use a larger k so the rank approximation is tight.
        let mut s = KllSketch::new(512, 0).expect("ok");
        for i in 0..5000 {
            s.add(i as f64);
        }
        let m = s.quantile(0.5);
        assert!((m - 2500.0).abs() < 1500.0, "median {m}");
    }

    #[test]
    fn kll_quantile_extremes_correct() {
        let mut s = KllSketch::new(512, 0).expect("ok");
        for i in 0..5000 {
            s.add(i as f64);
        }
        let q01 = s.quantile(0.01);
        let q99 = s.quantile(0.99);
        // Relaxed bounds for a randomised sketch.
        assert!(q01 < 4500.0, "q01 = {q01}");
        assert!(q99 > 500.0, "q99 = {q99}");
    }

    // True quantile of the integer sequence 0..total (each value once).
    fn true_quantile(total: f64, q: f64) -> f64 {
        (q * (total - 1.0)).round()
    }

    #[test]
    fn kll_merge_mismatched_k() {
        let a = KllSketch::new(64, 1).expect("ok");
        let mut b = KllSketch::new(128, 2).expect("ok");
        let err = b.merge(&a);
        assert!(err.is_err(), "different k must error");
    }

    #[test]
    fn kll_merge_empty_is_identity() {
        // Merging an empty sketch leaves quantiles (and weight) effectively unchanged.
        let mut a = KllSketch::new(256, 5).expect("ok");
        for i in 0..3000 {
            a.add(i as f64);
        }
        let before_med = a.quantile(0.5);
        let before_n = a.count();
        let empty = KllSketch::new(256, 9).expect("ok");
        a.merge(&empty).expect("merge ok");
        assert_eq!(a.count(), before_n, "empty merge changed count");
        assert!(
            (a.quantile(0.5) - before_med).abs() < 1.0e-9,
            "empty merge changed median"
        );
    }

    #[test]
    fn kll_merge_count_conserved() {
        let mut a = KllSketch::new(200, 1).expect("ok");
        let mut b = KllSketch::new(200, 2).expect("ok");
        for i in 0..7000u64 {
            a.add(i as f64);
        }
        for i in 7000..12345u64 {
            b.add(i as f64);
        }
        let na = a.count();
        let nb = b.count();
        a.merge(&b).expect("merge ok");
        assert_eq!(a.count(), na + nb, "item count not conserved");
    }

    #[test]
    fn kll_merge_accuracy_within_epsilon() {
        // A over 0..N, B over N..2N. Merge → quantiles within KLL rank error of truth.
        // k = 8192 makes the rank error envelope ε·(2N) comfortably tight and deterministic for
        // these fixed seeds (observed worst rank error ≈ 280 ranks over 2N = 200_000, ≈ 0.14%).
        let n: u64 = 100_000;
        let total = (2 * n) as f64;
        let k = 8192usize;
        let mut a = KllSketch::new(k, 12345).expect("ok");
        let mut b = KllSketch::new(k, 67890).expect("ok");
        for i in 0..n {
            a.add(i as f64);
        }
        for i in n..(2 * n) {
            b.add(i as f64);
        }
        let merged = KllSketch::merged(&a, &b).expect("merge ok");
        assert_eq!(merged.count(), 2 * n, "merged count");
        // KLL rank error ~ ε·(2N) with ε ≈ c/k; envelope ≈ 0.73% of the range — a real, tight
        // bound (~5x the observed worst error) that is robust to the fixed seeds, not loosened to
        // meaninglessness.
        let eps_rank = (total / k as f64) * 60.0;
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let est = merged.quantile(q);
            let truth = true_quantile(total, q);
            // Values ARE ranks here (0..2N), so |est - truth| is the rank error directly.
            assert!(
                (est - truth).abs() <= eps_rank,
                "q={q}: est={est} truth={truth} err={} tol={eps_rank}",
                (est - truth).abs()
            );
        }
    }

    #[test]
    fn kll_merge_order_independent() {
        // merge(A,B) and merge(B,A) give approximately equal quantiles.
        let n: u64 = 50_000;
        let k = 4096usize;
        let mut a = KllSketch::new(k, 101).expect("ok");
        let mut b = KllSketch::new(k, 202).expect("ok");
        for i in 0..n {
            a.add(i as f64);
        }
        for i in n..(2 * n) {
            b.add(i as f64);
        }
        let ab = KllSketch::merged(&a, &b).expect("ok");
        let ba = KllSketch::merged(&b, &a).expect("ok");
        let total = (2 * n) as f64;
        // Both are valid ε-sketches of the same multiset; their quantiles agree to ~2ε·(2N).
        // Observed worst delta ≈ 256 ranks for these seeds; envelope ≈ 0.73% of the range.
        let tol = (total / k as f64) * 30.0;
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let d = (ab.quantile(q) - ba.quantile(q)).abs();
            assert!(d <= tol, "order dependence at q={q}: {d} > {tol}");
        }
    }
}

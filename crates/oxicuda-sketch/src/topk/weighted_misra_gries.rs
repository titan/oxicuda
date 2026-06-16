//! Weighted Misra-Gries heavy-hitter sketch.
//!
//! The classic Misra-Gries (1982) summary maintains `k − 1` integer counters over an unweighted
//! stream. This is its generalization to **weighted** streams, in which each arrival contributes
//! a real weight `w > 0` to its key, following Berinde, Cormode, Indyk & Strauss,
//! *"Space-optimal Heavy Hitters with Strong Error Bounds"* (PODS 2010 / ACM TODS 2010) — the
//! same `(k − 1)`-counter "frequent / generalized Misra-Gries" family that gives the
//! `f_x − W/k ≤ est(x) ≤ f_x` guarantee on weighted streams.
//!
//! ## The weighted decrement (the subtle part)
//!
//! Maintain at most `k − 1` monitored `(key → weight)` counters and the running total weight `W`.
//! On `(key, w)` with `w > 0`:
//!
//! 1. **Hit** — `key` already monitored: add `w` to its counter.
//! 2. **Free slot** — fewer than `k − 1` counters: insert `(key, w)`.
//! 3. **Full, key absent** — perform the *weighted decrement*. Let `m` be the smallest monitored
//!    counter and set `δ = min(w, m)`. Subtract `δ` from **every** monitored counter (removing any
//!    that reach `0`) and from the incoming weight (`w ← w − δ`). If `w` is still positive
//!    afterwards — which happens exactly when `δ = m < w`, so the minimum counter(s) dropped to
//!    zero and freed at least one slot — insert the surviving remainder `(key, w)`.
//!
//! ### Why this is correct (one-directional error bound)
//!
//! A decrement event of size `δ` destroys `δ` from each of the `≤ k − 1` counters **and** `δ` from
//! the incoming weight: at most `δ · k` units of mass total. Since the total mass ever presented is
//! `W`, the sum of all `δ` over the whole stream is `≤ W / k`. Every key's counter is decremented
//! by at most that sum, so for every monitored key `x`:
//!
//! ```text
//! f_x − W/k  ≤  est(x)  ≤  f_x.
//! ```
//!
//! Hence estimates **never overcount** and undercount by at most `W/k`; in particular every key
//! with true weight `f_x > W/k` is retained. The guarantee is one-directional: a key with weight
//! `≤ W/k` may be evicted, and no claim is made about recovering it.
//!
//! When **all weights equal `1.0`** the rule degenerates to the textbook unweighted Misra-Gries
//! (`δ = min(1, m) = 1`, decrement everyone by one, never insert on a full-table event), so the
//! surviving counts match [`crate::topk::misra_gries::MisraGries`] exactly.

use crate::error::{SketchError, SketchResult};

/// Weighted Misra-Gries heavy-hitter sketch with at most `k − 1` monitored counters.
#[derive(Debug, Clone)]
pub struct WeightedMisraGries {
    /// Capacity parameter; at most `k − 1` counters are kept.
    pub k: usize,
    /// Monitored `(key, weight)` counters; length `≤ k − 1`.
    counters: Vec<(u64, f64)>,
    /// Running total weight `W = Σ w` presented to the sketch.
    total_weight: f64,
}

impl WeightedMisraGries {
    /// New weighted Misra-Gries sketch keeping at most `k − 1` counters. Requires `k ≥ 2`.
    pub fn new(k: usize) -> SketchResult<Self> {
        if k < 2 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be >= 2".to_string(),
            });
        }
        Ok(Self {
            k,
            counters: Vec::with_capacity(k - 1),
            total_weight: 0.0,
        })
    }

    /// Maximum number of monitored counters (`k − 1`).
    #[inline]
    fn capacity(&self) -> usize {
        self.k - 1
    }

    /// Index of the monitored counter with the smallest weight, if any.
    fn argmin(&self) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (i, &(_, w)) in self.counters.iter().enumerate() {
            match best {
                Some((_, bw)) if w >= bw => {}
                _ => best = Some((i, w)),
            }
        }
        best.map(|(i, _)| i)
    }

    /// Update the sketch with `(key, w)`. Non-positive or non-finite `w` is ignored.
    pub fn update(&mut self, key: u64, w: f64) {
        // Reject non-finite (NaN/±inf) and non-positive weights; only w > 0 contributes.
        if !w.is_finite() || w <= 0.0 {
            return;
        }
        self.total_weight += w;

        // Case 1: key already monitored — add its weight.
        for slot in self.counters.iter_mut() {
            if slot.0 == key {
                slot.1 += w;
                return;
            }
        }

        // Case 2: a free slot exists — insert directly.
        if self.counters.len() < self.capacity() {
            self.counters.push((key, w));
            return;
        }

        // Case 3: table full, key absent — weighted decrement.
        // δ = min(incoming weight, smallest counter). Subtract δ from all counters and from the
        // incoming weight; drop counters that hit zero; insert the remainder if it survives.
        let min_w = match self.argmin() {
            Some(idx) => self.counters[idx].1,
            None => {
                // capacity() == 0 (k == 2 keeps exactly 1 counter ⇒ this branch only when k-1==0,
                // which cannot happen since k >= 2). Defensive: nothing to decrement, drop key.
                return;
            }
        };
        let delta = w.min(min_w);
        let mut survivors: Vec<(u64, f64)> = Vec::with_capacity(self.counters.len());
        for &(slot_key, slot_w) in &self.counters {
            let reduced = slot_w - delta;
            // Keep only strictly-positive counters; those driven to zero free their slots.
            if reduced > 0.0 {
                survivors.push((slot_key, reduced));
            }
        }
        self.counters = survivors;
        let remainder = w - delta;
        if remainder > 0.0 && self.counters.len() < self.capacity() {
            self.counters.push((key, remainder));
        }
    }

    /// Estimated weight of `key` (the monitored counter, or `0.0` if not monitored).
    ///
    /// Always an **underestimate**: `est(key) ≤ f_key`, with `f_key − est(key) ≤ W/k`.
    #[must_use]
    pub fn estimate(&self, key: u64) -> f64 {
        self.counters
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, w)| *w)
            .unwrap_or(0.0)
    }

    /// Heavy hitters: monitored keys with estimated weight strictly greater than `phi · W`.
    #[must_use]
    pub fn heavy_hitters(&self, phi: f64) -> Vec<(u64, f64)> {
        let threshold = phi * self.total_weight;
        self.counters
            .iter()
            .filter(|(_, w)| *w > threshold)
            .copied()
            .collect()
    }

    /// All monitored `(key, weight)` counters.
    #[must_use]
    pub fn counters(&self) -> &[(u64, f64)] {
        &self.counters
    }

    /// Total weight `W = Σ w` presented to the sketch (exact bookkeeping).
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Merge `other` into `self` (Agarwal et al. 2013, *"Mergeable Summaries"*, weighted MG).
    ///
    /// Both sketches must share the same `k` (else [`SketchError::ShapeMismatch`]). The combined
    /// counter set is formed by summing weights of shared keys; if more than `k − 1` counters
    /// remain, the `k`-th largest counter value is subtracted from every counter and non-positive
    /// counters are dropped, leaving at most `k − 1`. Total weights add. This preserves the
    /// additive `W/k` error guarantee across the union.
    pub fn merge(&mut self, other: &Self) -> SketchResult<()> {
        if self.k != other.k {
            return Err(SketchError::ShapeMismatch {
                expected: vec![self.k],
                got: vec![other.k],
            });
        }
        // Combine counters by key (sum shared keys).
        for &(other_key, other_w) in &other.counters {
            match self.counters.iter_mut().find(|(k, _)| *k == other_key) {
                Some(slot) => slot.1 += other_w,
                None => self.counters.push((other_key, other_w)),
            }
        }
        self.total_weight += other.total_weight;
        self.prune_to_capacity();
        Ok(())
    }

    /// Reduce the counter set to at most `k − 1` entries via the canonical MG merge decrement:
    /// subtract the `k`-th largest weight from every counter and drop non-positive counters.
    fn prune_to_capacity(&mut self) {
        let cap = self.capacity();
        if self.counters.len() <= cap {
            return;
        }
        // Find the (k)-th largest weight = weight at sorted-descending index `cap` (0-based),
        // i.e. the largest weight that must be driven to ≤ 0 to leave at most `cap` survivors.
        let mut weights: Vec<f64> = self.counters.iter().map(|(_, w)| *w).collect();
        weights.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = weights[cap];
        let mut survivors: Vec<(u64, f64)> = Vec::with_capacity(cap);
        for &(k, w) in &self.counters {
            let reduced = w - threshold;
            if reduced > 0.0 {
                survivors.push((k, reduced));
            }
        }
        self.counters = survivors;
    }

    /// Convenience: merge two sketches into a fresh one. Both must share the same `k`.
    pub fn merged(a: &Self, b: &Self) -> SketchResult<Self> {
        let mut out = a.clone();
        out.merge(b)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topk::misra_gries::MisraGries;

    #[test]
    fn wmg_constructs() {
        let m = WeightedMisraGries::new(8).expect("ok");
        assert_eq!(m.k, 8);
        assert_eq!(m.total_weight(), 0.0);
    }

    #[test]
    fn wmg_invalid_k() {
        assert!(WeightedMisraGries::new(1).is_err());
    }

    #[test]
    fn wmg_total_weight_exact() {
        let mut m = WeightedMisraGries::new(4).expect("ok");
        let mut expected = 0.0;
        let weights = [1.5, 3.25, 0.75, 10.0, 2.0, 4.5, 0.1, 7.7];
        for (i, &w) in weights.iter().enumerate() {
            m.update(i as u64, w);
            expected += w;
        }
        assert!((m.total_weight() - expected).abs() < 1.0e-9);
    }

    #[test]
    fn wmg_heavy_keys_retained_and_underestimate_bounded() {
        // A few HEAVY keys (weight ≫ W/k) hidden among many light keys.
        let k = 16usize;
        let mut m = WeightedMisraGries::new(k).expect("ok");
        let heavy: [(u64, f64); 3] = [(1001, 500.0), (1002, 400.0), (1003, 350.0)];
        // Track true weights to check the bound.
        let mut truth: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
        for &(key, w) in &heavy {
            m.update(key, w);
            *truth.entry(key).or_insert(0.0) += w;
        }
        // Many light keys, small weights.
        for i in 0..2000u64 {
            let w = 0.5;
            m.update(i, w);
            *truth.entry(i).or_insert(0.0) += w;
        }
        let total_w = m.total_weight();
        let bound = total_w / k as f64;
        // (a) all heavy keys retained.
        for &(key, _) in &heavy {
            assert!(m.estimate(key) > 0.0, "heavy key {key} dropped");
        }
        // est ≤ true and (true − est) ≤ W/k for EVERY monitored key.
        for &(key, w) in m.counters() {
            let true_w = *truth.get(&key).unwrap_or(&0.0);
            assert!(w <= true_w + 1.0e-9, "overcount: est {w} > true {true_w}");
            assert!(
                true_w - w <= bound + 1.0e-9,
                "undercount {} exceeds W/k = {bound} for key {key}",
                true_w - w
            );
        }
    }

    #[test]
    fn wmg_matches_unweighted_when_all_weights_one() {
        // With every weight = 1.0 the weighted sketch reproduces classic Misra-Gries counts.
        let k = 6usize;
        // A deterministic mixed stream: 70x key 7, 30x key 9, 120 light keys cycling 100..140.
        let mut stream: Vec<u64> = Vec::new();
        stream.extend(std::iter::repeat_n(7u64, 50));
        stream.extend(std::iter::repeat_n(9u64, 30));
        stream.extend((0..120u64).map(|i| 100 + (i % 40)));
        stream.extend(std::iter::repeat_n(7u64, 20));
        let mut w = WeightedMisraGries::new(k).expect("ok");
        let mut u = MisraGries::new(k).expect("ok");
        for &x in &stream {
            w.update(x, 1.0);
            u.add(x);
        }
        // Compare as sorted (key, count) sets; weighted counts are exact integers here.
        let mut wc: Vec<(u64, i64)> = w
            .counters()
            .iter()
            .map(|&(k, v)| (k, v.round() as i64))
            .collect();
        let mut uc: Vec<(u64, i64)> = u.candidates().iter().map(|&(k, c)| (k, c as i64)).collect();
        wc.sort();
        uc.sort();
        assert_eq!(wc, uc, "weighted (w=1) must match unweighted Misra-Gries");
    }

    #[test]
    fn wmg_light_key_may_be_dropped() {
        // Guarantee is one-directional: a key with weight ≤ W/k can be evicted.
        let k = 4usize; // keeps 3 counters.
        let mut m = WeightedMisraGries::new(k).expect("ok");
        // Three heavy keys dominate the 3 slots; a light key should be squeezed out.
        m.update(1, 100.0);
        m.update(2, 100.0);
        m.update(3, 100.0);
        m.update(999, 1.0); // light, W/k = 301/4 ≈ 75.25 ≫ 1.0
        // The guarantee: heavy keys survive (assert that), light key has no recovery promise.
        for key in [1u64, 2, 3] {
            assert!(m.estimate(key) > 0.0, "heavy key {key} dropped");
        }
        assert_eq!(m.estimate(999), 0.0, "light key unexpectedly retained");
    }

    #[test]
    fn wmg_merge_preserves_heavy_and_weight() {
        let k = 12usize;
        let mut a = WeightedMisraGries::new(k).expect("ok");
        let mut b = WeightedMisraGries::new(k).expect("ok");
        // Heavy key split across both halves.
        a.update(42, 300.0);
        b.update(42, 250.0);
        for i in 0..500u64 {
            a.update(i, 0.3);
        }
        for i in 500..1000u64 {
            b.update(i, 0.3);
        }
        let wa = a.total_weight();
        let wb = b.total_weight();
        let merged = WeightedMisraGries::merged(&a, &b).expect("merge ok");
        assert!(
            (merged.total_weight() - (wa + wb)).abs() < 1.0e-6,
            "merged W bookkeeping"
        );
        // Heavy key (true 550) must survive and be an underestimate within W/k.
        let est = merged.estimate(42);
        assert!(est > 0.0, "heavy key lost on merge");
        let bound = merged.total_weight() / k as f64;
        assert!(est <= 550.0 + 1.0e-6, "merge overcount");
        assert!(550.0 - est <= bound + 1.0e-6, "merge undercount too large");
        assert!(merged.counters().len() < k, "merge exceeded capacity");
    }

    #[test]
    fn wmg_merge_rejects_mismatched_k() {
        let a = WeightedMisraGries::new(4).expect("ok");
        let mut b = WeightedMisraGries::new(8).expect("ok");
        assert!(b.merge(&a).is_err());
    }
}

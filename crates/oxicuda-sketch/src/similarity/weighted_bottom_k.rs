//! Weighted Bottom-`k` MinHash via exponential consistent weighted sampling.
//!
//! Goal: a single bottom-`k` sketch that, unlike plain KMV (uniform over the
//! support), samples each distinct element with probability **proportional to
//! its weight**, and from which the *weighted* Jaccard similarity
//!
//! ```text
//!     J_W(A, B) = Σ_e min(w_A(e), w_B(e)) / Σ_e max(w_A(e), w_B(e))
//! ```
//!
//! can be estimated from the agreement rate of two sketches built with the same
//! seed.
//!
//! ## Exponential rank trick (consistent weighted sampling)
//!
//! For element `e` with non-negative weight `w(e)`, draw a uniform
//! `u_e ∈ (0, 1)` deterministically from `hash(e, seed)` and form the
//! *exponential rank*
//!
//! ```text
//!     r(e) = −ln(u_e) / w(e) .
//! ```
//!
//! Then `r(e)` is distributed `Exponential(w(e))`. Across a weighted set the
//! smallest rank belongs to element `e` with probability `w(e) / Σ w`, and the
//! same `u_e` is reused for the same `e` regardless of which set it appears in —
//! i.e. the sampling is *consistent*. Keeping the `k` smallest ranks gives a
//! bottom-`k` weighted MinHash signature. Two such signatures (same seed,
//! same `k`) agree at rank `j` with probability equal to the generalised
//! (weighted) Jaccard similarity, so the agreement fraction is an unbiased
//! estimator of `J_W` with standard error `≈ 1/√k`.
//!
//! Each retained slot stores `(rank, element)` so that two sketches can be
//! compared by checking which `(rank, element)` survivors coincide — exactly
//! the consistent-weighted-sampling matching condition.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

/// One retained bottom-`k` entry: the exponential rank and the element that
/// produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedSlot {
    /// Exponential rank `−ln(u_e)/w(e)` (smaller ⇒ more likely sampled).
    pub rank: f64,
    /// The element id responsible for this rank.
    pub element: u64,
}

/// Weighted bottom-`k` MinHash sketch.
#[derive(Debug, Clone)]
pub struct WeightedBottomK {
    /// Number of retained minima.
    pub k: usize,
    /// Retained `(rank, element)` slots, kept sorted ascending by `rank`.
    pub slots: Vec<WeightedSlot>,
    /// Sum of weights of all *distinct* elements presented (best effort: the
    /// caller is expected to fold duplicate weights before insertion).
    pub total_weight: f64,
    seed: u64,
}

impl WeightedBottomK {
    /// Create a weighted bottom-`k` sketch.
    pub fn new(k: usize, seed: u64) -> SketchResult<Self> {
        if k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            k,
            slots: Vec::with_capacity(k),
            total_weight: 0.0,
            seed,
        })
    }

    /// Map an element to its uniform `u_e ∈ (0, 1)` via xxh3-min, using the full
    /// 32-bit-wide range divided by `2³²` (full-range, never `2³¹`).
    fn uniform(&self, element: u64) -> f64 {
        let h = xxh3_64_u64(element, self.seed);
        // Take the top 32 bits and normalise over the full 2³² range.
        let top = (h >> 32) as u32;
        // Map into the open interval (0, 1) to keep −ln(u) finite.
        ((top as f64) + 0.5) / (u32::MAX as f64 + 1.0)
    }

    /// Insert `element` carrying weight `weight ≥ 0`. Zero/negative weights are
    /// ignored (they can never be sampled).
    pub fn add(&mut self, element: u64, weight: f64) {
        if !(weight.is_finite() && weight > 0.0) {
            return;
        }
        self.total_weight += weight;
        let u = self.uniform(element);
        let rank = -u.ln() / weight;
        self.insert_rank(WeightedSlot { rank, element });
    }

    fn insert_rank(&mut self, slot: WeightedSlot) {
        // If this element already occupies a slot, keep the smaller rank.
        if let Some(existing) = self.slots.iter_mut().find(|s| s.element == slot.element) {
            if slot.rank < existing.rank {
                existing.rank = slot.rank;
                self.slots.sort_by(|a, b| {
                    a.rank
                        .partial_cmp(&b.rank)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            return;
        }
        if self.slots.len() < self.k {
            self.slots.push(slot);
            self.slots.sort_by(|a, b| {
                a.rank
                    .partial_cmp(&b.rank)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else if slot.rank < self.slots[self.k - 1].rank {
            self.slots[self.k - 1] = slot;
            self.slots.sort_by(|a, b| {
                a.rank
                    .partial_cmp(&b.rank)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    /// Current threshold: the largest retained rank, or `+∞` until full.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        if self.slots.len() < self.k {
            f64::INFINITY
        } else {
            self.slots.last().map_or(f64::INFINITY, |s| s.rank)
        }
    }

    /// Estimate the weighted Jaccard similarity `J_W(A, B)` against another
    /// sketch built with the same `k` and seed.
    ///
    /// Forms the union bottom-`k` rank set, then counts how many of those
    /// minima are produced by the *same element with the same rank* in both
    /// sketches — that is the consistent-weighted-sampling collision condition.
    pub fn weighted_jaccard(&self, other: &Self) -> SketchResult<f64> {
        if self.k != other.k {
            return Err(SketchError::DimensionMismatch {
                a: self.k,
                b: other.k,
            });
        }
        if self.seed != other.seed {
            return Err(SketchError::InvalidParameter {
                name: "seed".to_string(),
                reason: "both sketches must share the same hash seed".to_string(),
            });
        }
        // Merge the two slot lists by rank to get the union bottom-k.
        let mut merged: Vec<WeightedSlot> =
            Vec::with_capacity(self.slots.len() + other.slots.len());
        merged.extend_from_slice(&self.slots);
        merged.extend_from_slice(&other.slots);
        merged.sort_by(|a, b| {
            a.rank
                .partial_cmp(&b.rank)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut union_set: Vec<WeightedSlot> = Vec::with_capacity(self.k);
        for s in merged {
            if union_set.iter().any(|u| u.element == s.element) {
                continue;
            }
            union_set.push(s);
            if union_set.len() == self.k {
                break;
            }
        }
        if union_set.is_empty() {
            return Ok(0.0);
        }
        let matches = union_set
            .iter()
            .filter(|s| {
                let in_a = self
                    .slots
                    .iter()
                    .any(|x| x.element == s.element && (x.rank - s.rank).abs() < 1e-12);
                let in_b = other
                    .slots
                    .iter()
                    .any(|x| x.element == s.element && (x.rank - s.rank).abs() < 1e-12);
                in_a && in_b
            })
            .count();
        Ok(matches as f64 / union_set.len() as f64)
    }

    /// Number of retained slots (`≤ k`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// True if no positive-weight element has been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference weighted Jaccard over explicit weight maps.
    fn true_weighted_jaccard(a: &[(u64, f64)], b: &[(u64, f64)]) -> f64 {
        let mut keys: Vec<u64> = a.iter().map(|(k, _)| *k).collect();
        keys.extend(b.iter().map(|(k, _)| *k));
        keys.sort_unstable();
        keys.dedup();
        let wa = |e: u64| a.iter().find(|(k, _)| *k == e).map_or(0.0, |(_, w)| *w);
        let wb = |e: u64| b.iter().find(|(k, _)| *k == e).map_or(0.0, |(_, w)| *w);
        let mut num = 0.0;
        let mut den = 0.0;
        for e in keys {
            num += wa(e).min(wb(e));
            den += wa(e).max(wb(e));
        }
        if den == 0.0 { 0.0 } else { num / den }
    }

    #[test]
    fn wbk_constructs() {
        let s = WeightedBottomK::new(8, 0).expect("ok");
        assert_eq!(s.k, 8);
        assert!(s.is_empty());
    }

    #[test]
    fn wbk_invalid_k() {
        assert!(WeightedBottomK::new(0, 0).is_err());
    }

    #[test]
    fn wbk_zero_weight_ignored() {
        let mut s = WeightedBottomK::new(4, 1).expect("ok");
        s.add(7, 0.0);
        s.add(8, -1.0);
        s.add(9, f64::NAN);
        assert!(s.is_empty());
    }

    #[test]
    fn wbk_dedup_keeps_min_rank() {
        let mut s = WeightedBottomK::new(4, 2).expect("ok");
        s.add(42, 1.0);
        let r1 = s.slots[0].rank;
        // A larger weight ⇒ smaller exponential rank for the same uniform.
        s.add(42, 10.0);
        assert_eq!(s.len(), 1, "duplicate element must not create a new slot");
        assert!(s.slots[0].rank <= r1);
    }

    #[test]
    fn wbk_keeps_only_k_smallest() {
        let mut s = WeightedBottomK::new(5, 3).expect("ok");
        for e in 0..100u64 {
            s.add(e, 1.0);
        }
        assert_eq!(s.len(), 5);
        // Slots stay sorted ascending by rank.
        for w in s.slots.windows(2) {
            assert!(w[0].rank <= w[1].rank);
        }
    }

    #[test]
    fn wbk_identical_sets_similarity_one() {
        let mut a = WeightedBottomK::new(64, 10).expect("ok");
        let items: Vec<(u64, f64)> = (0..200u64).map(|i| (i, 1.0 + (i % 7) as f64)).collect();
        for &(e, w) in &items {
            a.add(e, w);
        }
        let b = a.clone();
        let j = a.weighted_jaccard(&b).expect("ok");
        assert!((j - 1.0).abs() < 1e-9, "identical weighted Jaccard = {j}");
    }

    #[test]
    fn wbk_disjoint_sets_similarity_zero() {
        let mut a = WeightedBottomK::new(128, 11).expect("ok");
        let mut b = WeightedBottomK::new(128, 11).expect("ok");
        for e in 0..500u64 {
            a.add(e, 1.0 + (e % 5) as f64);
        }
        for e in 1_000_000..1_000_500u64 {
            b.add(e, 1.0 + (e % 5) as f64);
        }
        let j = a.weighted_jaccard(&b).expect("ok");
        assert!(j < 0.03, "disjoint weighted Jaccard = {j}");
    }

    #[test]
    fn wbk_estimate_close_to_true_weighted_jaccard() {
        // Build two overlapping weighted multisets and verify the bottom-k
        // estimate is within ~1/sqrt(k) of the exact weighted Jaccard.
        let k = 1024;
        let seed = 2024;
        let mut a_items: Vec<(u64, f64)> = Vec::new();
        let mut b_items: Vec<(u64, f64)> = Vec::new();
        for e in 0..600u64 {
            a_items.push((e, 1.0 + (e % 9) as f64));
        }
        for e in 300..900u64 {
            b_items.push((e, 1.0 + (e % 9) as f64));
        }
        let mut a = WeightedBottomK::new(k, seed).expect("ok");
        let mut b = WeightedBottomK::new(k, seed).expect("ok");
        for &(e, w) in &a_items {
            a.add(e, w);
        }
        for &(e, w) in &b_items {
            b.add(e, w);
        }
        let est = a.weighted_jaccard(&b).expect("ok");
        let truth = true_weighted_jaccard(&a_items, &b_items);
        let tol = 0.12; // generous multiple of the 1/sqrt(k) ≈ 0.031 std error
        assert!(
            (est - truth).abs() < tol,
            "weighted Jaccard est {est} vs true {truth}"
        );
    }

    #[test]
    fn wbk_threshold_infinite_until_full() {
        let mut s = WeightedBottomK::new(4, 7).expect("ok");
        assert!(s.threshold().is_infinite());
        s.add(1, 1.0);
        assert!(s.threshold().is_infinite());
        for e in 2..=4u64 {
            s.add(e, 1.0);
        }
        assert!(s.threshold().is_finite());
    }

    #[test]
    fn wbk_seed_or_k_mismatch_errors() {
        let mut a = WeightedBottomK::new(8, 1).expect("ok");
        let mut b = WeightedBottomK::new(8, 2).expect("ok");
        a.add(0, 1.0);
        b.add(0, 1.0);
        assert!(a.weighted_jaccard(&b).is_err());
        let c = WeightedBottomK::new(16, 1).expect("ok");
        assert!(a.weighted_jaccard(&c).is_err());
    }

    #[test]
    fn wbk_heavier_weight_more_likely_sampled() {
        // One very heavy element among many light ones should almost always be
        // present in a small sketch (its sampling probability ≈ w/Σw is high).
        let k = 4;
        let mut present = 0usize;
        let trials = 50u64;
        for t in 0..trials {
            let mut s = WeightedBottomK::new(k, 100 + t).expect("ok");
            s.add(999_999, 500.0); // heavy
            for e in 0..200u64 {
                s.add(e, 1.0); // light
            }
            if s.slots.iter().any(|sl| sl.element == 999_999) {
                present += 1;
            }
        }
        // With weight 500 vs total ~700 the heavy element dominates the minima.
        assert!(
            present as f64 / trials as f64 > 0.8,
            "heavy element retained in only {present}/{trials} trials"
        );
    }
}

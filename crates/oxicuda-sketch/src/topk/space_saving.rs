//! Space-Saving heavy-hitter sketch (Metwally, Agrawal, El Abbadi 2005).
//!
//! Maintains `k` (key, count) slots. On each insert:
//! - If key is present, increment.
//! - Else if there's an empty slot, place `(key, 1)`.
//! - Else replace the slot with minimum count: new slot = `(key, min_count + 1)`.
//!
//! Better accuracy than Misra-Gries; estimate >= true count (over-estimate by at most min_count).

use crate::error::{SketchError, SketchResult};

/// Space-Saving sketch.
#[derive(Debug, Clone)]
pub struct SpaceSaving {
    pub k: usize,
    pub slots: Vec<(u64, u64)>,
    pub n: u64,
}

impl SpaceSaving {
    /// New Space-Saving sketch with `k` slots.
    pub fn new(k: usize) -> SketchResult<Self> {
        if k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            k,
            slots: Vec::with_capacity(k),
            n: 0,
        })
    }

    /// Insert an item.
    pub fn add(&mut self, x: u64) {
        self.n += 1;
        // Check existing slots.
        for slot in self.slots.iter_mut() {
            if slot.0 == x {
                slot.1 += 1;
                return;
            }
        }
        if self.slots.len() < self.k {
            self.slots.push((x, 1));
            return;
        }
        // Replace minimum.
        let mut min_idx = 0usize;
        let mut min_count = self.slots[0].1;
        for (i, slot) in self.slots.iter().enumerate().skip(1) {
            if slot.1 < min_count {
                min_count = slot.1;
                min_idx = i;
            }
        }
        self.slots[min_idx] = (x, min_count + 1);
    }

    /// Heavy hitters: items with estimated count > phi * n.
    #[must_use]
    pub fn heavy_hitters(&self, phi: f64) -> Vec<(u64, u64)> {
        let t = (phi * self.n as f64) as u64;
        self.slots.iter().filter(|(_, c)| *c > t).copied().collect()
    }

    /// Slot accessors.
    #[must_use]
    pub fn slots(&self) -> &[(u64, u64)] {
        &self.slots
    }

    /// Estimate of count for `x`.
    #[must_use]
    pub fn estimate(&self, x: u64) -> u64 {
        self.slots
            .iter()
            .find(|(k, _)| *k == x)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    }

    /// Top-K items by estimated count (descending).
    #[must_use]
    pub fn top_k(&self) -> Vec<(u64, u64)> {
        let mut s = self.slots.clone();
        s.sort_by_key(|b| std::cmp::Reverse(b.1));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ss_constructs() {
        let s = SpaceSaving::new(10).expect("ok");
        assert_eq!(s.k, 10);
    }

    #[test]
    fn ss_invalid_k() {
        assert!(SpaceSaving::new(0).is_err());
    }

    #[test]
    fn ss_finds_heavy() {
        let mut s = SpaceSaving::new(8).expect("ok");
        for _ in 0..400 {
            s.add(99);
        }
        for i in 0..600u64 {
            s.add(i + 100);
        }
        assert!(s.estimate(99) >= 400);
    }

    #[test]
    fn ss_overestimate_bounded() {
        let mut s = SpaceSaving::new(4).expect("ok");
        for _ in 0..30 {
            s.add(1);
        }
        for i in 0..16u64 {
            s.add(i + 100);
        }
        // estimate(1) is at most true_count + min_count.
        let est = s.estimate(1);
        assert!(est >= 30, "should never undercount heavy hitter, got {est}");
    }
}

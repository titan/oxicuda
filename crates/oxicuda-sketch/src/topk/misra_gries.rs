//! Misra-Gries (1982) heavy-hitter sketch.
//!
//! Maintains `k - 1` candidate slots: on each insert, if `x` matches a slot, increment;
//! else, find an empty slot and place `(x, 1)`; else, decrement all slots by 1.
//! After processing `n` items, any element with true frequency > n/k is in the table.

use crate::error::{SketchError, SketchResult};

/// Misra-Gries heavy-hitter sketch.
#[derive(Debug, Clone)]
pub struct MisraGries {
    pub k: usize,
    pub slots: Vec<(u64, u64)>, // (key, count)
    pub n: u64,
}

impl MisraGries {
    /// New MG sketch with `k - 1` candidate slots.
    pub fn new(k: usize) -> SketchResult<Self> {
        if k < 2 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be >= 2".to_string(),
            });
        }
        Ok(Self {
            k,
            slots: Vec::with_capacity(k - 1),
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
        if self.slots.len() < self.k - 1 {
            self.slots.push((x, 1));
            return;
        }
        // Decrement all and remove empty.
        let mut new_slots = Vec::with_capacity(self.slots.len());
        for &(key, c) in &self.slots {
            if c > 1 {
                new_slots.push((key, c - 1));
            }
        }
        self.slots = new_slots;
    }

    /// Get candidate (key, count) pairs.
    #[must_use]
    pub fn candidates(&self) -> &[(u64, u64)] {
        &self.slots
    }

    /// Return ε-heavy hitters: items with frequency > φ * n. Note that Misra-Gries
    /// undercounts by at most n/k, so any item with TRUE freq > n/k IS in this list.
    #[must_use]
    pub fn heavy_hitters(&self, phi: f64) -> Vec<(u64, u64)> {
        let threshold = (phi * self.n as f64) as u64;
        self.slots
            .iter()
            .filter(|(_, c)| *c > threshold)
            .copied()
            .collect()
    }

    /// Estimate the true count of `x`. May undercount by at most n/k.
    #[must_use]
    pub fn estimate(&self, x: u64) -> u64 {
        self.slots
            .iter()
            .find(|(k, _)| *k == x)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mg_constructs() {
        let m = MisraGries::new(8).expect("ok");
        assert_eq!(m.k, 8);
    }

    #[test]
    fn mg_invalid_k() {
        assert!(MisraGries::new(1).is_err());
    }

    #[test]
    fn mg_finds_heavy_hitters() {
        let mut m = MisraGries::new(8).expect("ok");
        // Insert 1000 items: item 7 appears 200 times (frequency > n/k = 125).
        for _ in 0..200 {
            m.add(7);
        }
        for i in 0..800u64 {
            m.add(i + 100);
        }
        let candidates: Vec<u64> = m.candidates().iter().map(|(k, _)| *k).collect();
        assert!(candidates.contains(&7), "missing heavy hitter 7");
    }

    #[test]
    fn mg_undercount_bounded() {
        let mut m = MisraGries::new(4).expect("ok");
        for _ in 0..30 {
            m.add(1);
        }
        for i in 0..6u64 {
            m.add(i + 100);
        }
        // For item 1: true count = 30, estimate may undercount by up to n/k = 36/4 = 9.
        let est = m.estimate(1);
        assert!(est >= 21, "estimate too low for heavy hitter: {est}");
    }
}

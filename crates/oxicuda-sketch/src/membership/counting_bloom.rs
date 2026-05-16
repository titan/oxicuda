//! Counting Bloom filter (Fan, Cao, Almeida, Broder 2000).
//!
//! Uses 4-bit counters (stored as `u8`, two slots per byte conceptually but we use one byte
//! per counter for simplicity). Supports deletion.
//!
//! Insert: increment each of k counters by 1 (saturating at 15).
//! Delete: decrement each of k counters by 1 (saturating at 0).
//! Contains: return true iff all k counters > 0.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

const COUNTER_MAX: u8 = 15;

/// Counting Bloom filter with 4-bit-wide counters (saturating at 15).
#[derive(Debug, Clone)]
pub struct CountingBloomFilter {
    pub m: usize,
    pub k: usize,
    pub counters: Vec<u8>,
    pub seed_base: u64,
}

impl CountingBloomFilter {
    /// Create a new counting Bloom filter.
    pub fn new(m: usize, k: usize, seed_base: u64) -> SketchResult<Self> {
        if m == 0 || k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(m,k)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            m,
            k,
            counters: vec![0u8; m],
            seed_base,
        })
    }

    fn positions(&self, x: u64) -> Vec<usize> {
        let h1 = xxh3_64_u64(x, self.seed_base);
        let h2 = xxh3_64_u64(x, self.seed_base.wrapping_add(0x1234_5678_9ABC_DEF1));
        (0..self.k)
            .map(|i| ((h1.wrapping_add((i as u64).wrapping_mul(h2))) as usize) % self.m)
            .collect()
    }

    /// Insert an item (increment all k counters).
    pub fn insert(&mut self, x: u64) {
        for p in self.positions(x) {
            if self.counters[p] < COUNTER_MAX {
                self.counters[p] += 1;
            }
        }
    }

    /// Delete an item. Note: deletion is safe only if the item was previously inserted
    /// (otherwise we may erroneously decrement counters for false-positive matches,
    /// which can introduce false negatives for other items).
    pub fn delete(&mut self, x: u64) {
        for p in self.positions(x) {
            if self.counters[p] > 0 {
                self.counters[p] -= 1;
            }
        }
    }

    /// Test membership.
    #[must_use]
    pub fn contains(&self, x: u64) -> bool {
        self.positions(x).iter().all(|&p| self.counters[p] > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbf_insert_contains() {
        let mut bf = CountingBloomFilter::new(2048, 4, 0).expect("ok");
        for i in 0..100u64 {
            bf.insert(i);
        }
        for i in 0..100u64 {
            assert!(bf.contains(i));
        }
    }

    #[test]
    fn cbf_delete_clears_membership() {
        let mut bf = CountingBloomFilter::new(2048, 4, 0).expect("ok");
        bf.insert(42);
        assert!(bf.contains(42));
        bf.delete(42);
        // After single delete the counters for item 42's k slots are zeroed unless other items
        // collided. With 2048 slots and only one item present, this is essentially certain.
        assert!(!bf.contains(42));
    }

    #[test]
    fn cbf_saturation_does_not_overflow() {
        let mut bf = CountingBloomFilter::new(64, 2, 0).expect("ok");
        for _ in 0..1000 {
            bf.insert(7);
        }
        // No panic; counters are clamped at 15.
        for p in bf.positions(7) {
            assert!(bf.counters[p] <= 15);
        }
    }
}

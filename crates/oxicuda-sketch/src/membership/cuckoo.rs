//! Cuckoo filter (Fan, Andersen, Kaminsky, Mitzenmacher 2014).
//!
//! Each bucket holds `b` fingerprints (small hashes of the item). An item is mapped to
//! two candidate buckets `i1 = h(x) mod n_buckets` and `i2 = i1 XOR h(fingerprint) mod n_buckets`.
//! Insert: try i1 then i2; if both full, kick out a random fingerprint and re-insert.
//! Lookup: scan both buckets.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::xxh3_min::xxh3_64_u64;

const MAX_KICKS: usize = 500;

/// Cuckoo filter.
#[derive(Debug, Clone)]
pub struct CuckooFilter {
    pub n_buckets: usize,
    pub bucket_size: usize,
    pub fingerprint_bits: u32,
    pub buckets: Vec<u32>, // flat row-major: buckets[bucket][slot] = fingerprint (0 = empty)
    pub n_items: usize,
    pub seed_base: u64,
    rng: LcgRng,
}

impl CuckooFilter {
    /// Construct a cuckoo filter with `n_buckets` buckets of `bucket_size` slots each,
    /// using `fingerprint_bits` bits per fingerprint (1..=16 typical).
    pub fn new(
        n_buckets: usize,
        bucket_size: usize,
        fingerprint_bits: u32,
        seed_base: u64,
    ) -> SketchResult<Self> {
        if n_buckets == 0 || bucket_size == 0 {
            return Err(SketchError::InvalidParameter {
                name: "n_buckets/bucket_size".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if !(1..=31).contains(&fingerprint_bits) {
            return Err(SketchError::InvalidParameter {
                name: "fingerprint_bits".to_string(),
                reason: "must be in [1,31]".to_string(),
            });
        }
        Ok(Self {
            n_buckets,
            bucket_size,
            fingerprint_bits,
            buckets: vec![0u32; n_buckets * bucket_size],
            n_items: 0,
            seed_base,
            rng: LcgRng::new(seed_base.wrapping_add(0xDEAD_BEEF)),
        })
    }

    fn fingerprint(&self, x: u64) -> u32 {
        let mask = if self.fingerprint_bits >= 32 {
            u32::MAX
        } else {
            (1u32 << self.fingerprint_bits) - 1
        };
        let fp = (xxh3_64_u64(x, self.seed_base.wrapping_add(1)) as u32) & mask;
        // 0 reserved for "empty".
        if fp == 0 { 1 } else { fp }
    }

    fn primary_index(&self, x: u64) -> usize {
        (xxh3_64_u64(x, self.seed_base) as usize) % self.n_buckets
    }

    fn alt_index(&self, i: usize, fp: u32) -> usize {
        let h = xxh3_64_u64(fp as u64, self.seed_base.wrapping_add(2)) as usize;
        (i ^ (h % self.n_buckets)) % self.n_buckets
    }

    fn try_insert_in_bucket(&mut self, bucket: usize, fp: u32) -> bool {
        let base = bucket * self.bucket_size;
        for s in 0..self.bucket_size {
            if self.buckets[base + s] == 0 {
                self.buckets[base + s] = fp;
                return true;
            }
        }
        false
    }

    /// Insert an item.
    pub fn insert(&mut self, x: u64) -> SketchResult<()> {
        let fp = self.fingerprint(x);
        let i1 = self.primary_index(x);
        if self.try_insert_in_bucket(i1, fp) {
            self.n_items += 1;
            return Ok(());
        }
        let i2 = self.alt_index(i1, fp);
        if self.try_insert_in_bucket(i2, fp) {
            self.n_items += 1;
            return Ok(());
        }
        // Kick out logic.
        let mut current_idx = if self.rng.next_bool() { i1 } else { i2 };
        let mut current_fp = fp;
        for _ in 0..MAX_KICKS {
            // Choose random slot in current_idx.
            let slot = self.rng.next_usize(self.bucket_size);
            let base = current_idx * self.bucket_size;
            std::mem::swap(&mut current_fp, &mut self.buckets[base + slot]);
            current_idx = self.alt_index(current_idx, current_fp);
            if self.try_insert_in_bucket(current_idx, current_fp) {
                self.n_items += 1;
                return Ok(());
            }
        }
        Err(SketchError::HashTableFull { tries: MAX_KICKS })
    }

    /// Test membership.
    #[must_use]
    pub fn contains(&self, x: u64) -> bool {
        let fp = self.fingerprint(x);
        let i1 = self.primary_index(x);
        let i2 = self.alt_index(i1, fp);
        for &b in &[i1, i2] {
            let base = b * self.bucket_size;
            for s in 0..self.bucket_size {
                if self.buckets[base + s] == fp {
                    return true;
                }
            }
        }
        false
    }

    /// Delete an item (returns true if a matching fingerprint was found and removed).
    pub fn delete(&mut self, x: u64) -> bool {
        let fp = self.fingerprint(x);
        let i1 = self.primary_index(x);
        let i2 = self.alt_index(i1, fp);
        for &b in &[i1, i2] {
            let base = b * self.bucket_size;
            for s in 0..self.bucket_size {
                if self.buckets[base + s] == fp {
                    self.buckets[base + s] = 0;
                    self.n_items = self.n_items.saturating_sub(1);
                    return true;
                }
            }
        }
        false
    }

    /// Current load factor.
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        let cap = (self.n_buckets * self.bucket_size) as f64;
        (self.n_items as f64) / cap.max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuckoo_insert_contains() {
        let mut cf = CuckooFilter::new(512, 4, 12, 0).expect("ok");
        for i in 0..200u64 {
            cf.insert(i).expect("ok");
        }
        for i in 0..200u64 {
            assert!(cf.contains(i), "missing inserted item {i}");
        }
    }

    #[test]
    fn cuckoo_delete_removes() {
        let mut cf = CuckooFilter::new(512, 4, 12, 0).expect("ok");
        cf.insert(42).expect("ok");
        assert!(cf.contains(42));
        assert!(cf.delete(42));
        // After delete, may still be probabilistically present if collisions exist,
        // but with empty filter, no collisions.
    }

    #[test]
    fn cuckoo_load_factor_progresses() {
        let mut cf = CuckooFilter::new(128, 4, 12, 0).expect("ok");
        for i in 0..200u64 {
            let _ = cf.insert(i);
        }
        assert!(cf.load_factor() > 0.1);
    }

    #[test]
    fn cuckoo_invalid_params() {
        assert!(CuckooFilter::new(0, 4, 12, 0).is_err());
        assert!(CuckooFilter::new(64, 0, 12, 0).is_err());
        assert!(CuckooFilter::new(64, 4, 0, 0).is_err());
        assert!(CuckooFilter::new(64, 4, 32, 0).is_err());
    }
}

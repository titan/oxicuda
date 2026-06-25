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

/// Cuckoo filter with full **32-bit (4-byte) fingerprints** for very low false-positive rates.
///
/// [`CuckooFilter`] reserves the value `0` as an empty-slot sentinel and silently remaps a
/// computed fingerprint of `0` to `1`. That makes two distinct fingerprints collide and caps
/// the usable fingerprint width at 31 bits, so its false-positive rate floors out around
/// `2 * bucket_size / 2^31`. This variant instead tracks slot occupancy in a **separate
/// bitmap**, so all `2^32` fingerprint values — `0` included — are first-class, distinctly
/// stored fingerprints. The achievable false-positive rate is therefore roughly
/// `2 * bucket_size / 2^32` (≈ `1.9e-9` at `bucket_size = 4`), orders of magnitude below the
/// narrow filter, while never producing a false negative.
///
/// The partial-key cuckoo displacement of Fan, Andersen, Kaminsky & Mitzenmacher (2014) is
/// preserved: the alternate bucket is derived solely from the fingerprint via
/// `i2 = i1 XOR (hash(fp) mod n_buckets)`. For that fold to be an exact involution
/// (`alt(alt(i, fp), fp) == i`, which is what guarantees no false negatives), `n_buckets` is
/// rounded up to the next power of two on construction.
#[derive(Debug, Clone)]
pub struct CuckooFilter32 {
    /// Number of buckets (rounded up to a power of two at construction).
    pub n_buckets: usize,
    /// Slots per bucket.
    pub bucket_size: usize,
    /// Flat row-major fingerprint store; a slot's value is meaningful only when its
    /// corresponding occupancy bit is set.
    pub buckets: Vec<u32>,
    /// Occupancy bitmap: one bit per slot, packed into `u64` words. This is what replaces the
    /// `0`-sentinel scheme and frees up every fingerprint value (including `0`).
    occupied: Vec<u64>,
    /// Number of stored items (equal to the population count of `occupied`).
    pub n_items: usize,
    /// Base seed for the fingerprint / index hash family.
    pub seed_base: u64,
    rng: LcgRng,
}

impl CuckooFilter32 {
    /// Construct a 32-bit-fingerprint cuckoo filter with `n_buckets` buckets (rounded up to a
    /// power of two) of `bucket_size` slots each.
    pub fn new(n_buckets: usize, bucket_size: usize, seed_base: u64) -> SketchResult<Self> {
        if n_buckets == 0 || bucket_size == 0 {
            return Err(SketchError::InvalidParameter {
                name: "n_buckets/bucket_size".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let n_buckets = n_buckets.next_power_of_two();
        let n_slots = n_buckets * bucket_size;
        Ok(Self {
            n_buckets,
            bucket_size,
            buckets: vec![0u32; n_slots],
            occupied: vec![0u64; n_slots.div_ceil(64)],
            n_items: 0,
            seed_base,
            rng: LcgRng::new(seed_base.wrapping_add(0xDEAD_BEEF)),
        })
    }

    #[inline]
    fn index_mask(&self) -> usize {
        self.n_buckets - 1
    }

    /// Full 32-bit fingerprint. No value is reserved — occupancy is tracked separately — so a
    /// fingerprint of `0` is stored and matched like any other value.
    fn fingerprint(&self, x: u64) -> u32 {
        xxh3_64_u64(x, self.seed_base.wrapping_add(1)) as u32
    }

    fn primary_index(&self, x: u64) -> usize {
        (xxh3_64_u64(x, self.seed_base) as usize) & self.index_mask()
    }

    fn alt_index(&self, i: usize, fp: u32) -> usize {
        let h = xxh3_64_u64(fp as u64, self.seed_base.wrapping_add(2)) as usize;
        // `n_buckets` is a power of two, so `& mask` keeps the XOR fold an exact involution.
        (i ^ (h & self.index_mask())) & self.index_mask()
    }

    #[inline]
    fn is_occupied(&self, slot: usize) -> bool {
        (self.occupied[slot >> 6] >> (slot & 63)) & 1 == 1
    }

    #[inline]
    fn set_occupied(&mut self, slot: usize) {
        self.occupied[slot >> 6] |= 1u64 << (slot & 63);
    }

    #[inline]
    fn clear_occupied(&mut self, slot: usize) {
        self.occupied[slot >> 6] &= !(1u64 << (slot & 63));
    }

    fn try_insert_in_bucket(&mut self, bucket: usize, fp: u32) -> bool {
        let base = bucket * self.bucket_size;
        for s in 0..self.bucket_size {
            let slot = base + s;
            if !self.is_occupied(slot) {
                self.buckets[slot] = fp;
                self.set_occupied(slot);
                return true;
            }
        }
        false
    }

    /// Core insertion of an already-computed fingerprint with primary bucket `i1`.
    fn insert_fp(&mut self, fp: u32, i1: usize) -> SketchResult<()> {
        if self.try_insert_in_bucket(i1, fp) {
            self.n_items += 1;
            return Ok(());
        }
        let i2 = self.alt_index(i1, fp);
        if self.try_insert_in_bucket(i2, fp) {
            self.n_items += 1;
            return Ok(());
        }
        // Both candidate buckets full: kick out a random resident and relocate it.
        let mut current_idx = if self.rng.next_bool() { i1 } else { i2 };
        let mut current_fp = fp;
        for _ in 0..MAX_KICKS {
            let slot = current_idx * self.bucket_size + self.rng.next_usize(self.bucket_size);
            // The slot was full (occupancy bit already set) and stays full after the swap,
            // so the occupancy bitmap needs no update here.
            std::mem::swap(&mut current_fp, &mut self.buckets[slot]);
            current_idx = self.alt_index(current_idx, current_fp);
            if self.try_insert_in_bucket(current_idx, current_fp) {
                self.n_items += 1;
                return Ok(());
            }
        }
        Err(SketchError::HashTableFull { tries: MAX_KICKS })
    }

    /// Insert an item.
    pub fn insert(&mut self, x: u64) -> SketchResult<()> {
        let fp = self.fingerprint(x);
        let i1 = self.primary_index(x);
        self.insert_fp(fp, i1)
    }

    /// Core membership test for an already-computed fingerprint with primary bucket `i1`.
    fn contains_fp(&self, fp: u32, i1: usize) -> bool {
        let i2 = self.alt_index(i1, fp);
        for &b in &[i1, i2] {
            let base = b * self.bucket_size;
            for s in 0..self.bucket_size {
                let slot = base + s;
                // The occupancy check is essential: every slot's backing store starts at `0`,
                // so without it a fingerprint of `0` would match every empty slot.
                if self.is_occupied(slot) && self.buckets[slot] == fp {
                    return true;
                }
            }
        }
        false
    }

    /// Test membership.
    #[must_use]
    pub fn contains(&self, x: u64) -> bool {
        let fp = self.fingerprint(x);
        let i1 = self.primary_index(x);
        self.contains_fp(fp, i1)
    }

    /// Core deletion for an already-computed fingerprint with primary bucket `i1`.
    fn delete_fp(&mut self, fp: u32, i1: usize) -> bool {
        let i2 = self.alt_index(i1, fp);
        for &b in &[i1, i2] {
            let base = b * self.bucket_size;
            for s in 0..self.bucket_size {
                let slot = base + s;
                if self.is_occupied(slot) && self.buckets[slot] == fp {
                    self.clear_occupied(slot);
                    self.n_items = self.n_items.saturating_sub(1);
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
        self.delete_fp(fp, i1)
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

    // --- 32-bit (wide) cuckoo filter ---

    #[test]
    fn cuckoo32_invalid_params_and_rounding() {
        assert!(CuckooFilter32::new(0, 4, 0).is_err());
        assert!(CuckooFilter32::new(64, 0, 0).is_err());
        // n_buckets is rounded up to the next power of two so the XOR fold stays involutive.
        let cf = CuckooFilter32::new(3000, 4, 0).expect("ok");
        assert_eq!(cf.n_buckets, 4096);
        let cf2 = CuckooFilter32::new(4096, 4, 0).expect("ok");
        assert_eq!(cf2.n_buckets, 4096);
    }

    #[test]
    fn cuckoo32_zero_fingerprint_insertable() {
        // The bug the wide path fixes: the narrow filter reserves fp=0 as "empty" (remapping a
        // computed 0 to 1). With a separate occupancy bitmap, fp=0 is a first-class fingerprint.
        let mut cf = CuckooFilter32::new(64, 4, 7).expect("ok");
        // All slots are 0-initialised, yet an empty filter must report fp=0 ABSENT — proving the
        // occupancy bitmap, not a reserved value, decides emptiness.
        assert!(
            !cf.contains_fp(0, 5),
            "0-initialised slots must not match fp=0"
        );
        cf.insert_fp(0, 5).expect("insert fp=0");
        assert!(cf.contains_fp(0, 5), "fp=0 must be present after insert");
        assert_eq!(cf.n_items, 1);
        assert!(cf.delete_fp(0, 5), "delete fp=0");
        assert!(!cf.contains_fp(0, 5), "fp=0 absent after delete");
        assert_eq!(cf.n_items, 0);
    }

    #[test]
    fn cuckoo32_no_false_negatives() {
        // Every successfully inserted key must always report present (no false negatives).
        let mut cf = CuckooFilter32::new(2048, 4, 12345).expect("ok");
        let mut rng = LcgRng::new(0xABCD);
        let mut inserted = Vec::new();
        for _ in 0..6000u32 {
            let k = rng.next_u64();
            if cf.insert(k).is_ok() {
                inserted.push(k);
            }
        }
        assert!(
            inserted.len() > 5000,
            "too many insert failures: {}",
            inserted.len()
        );
        for &k in &inserted {
            assert!(cf.contains(k), "false negative for inserted key {k}");
        }
    }

    #[test]
    fn cuckoo32_insert_delete_roundtrip() {
        let mut cf = CuckooFilter32::new(256, 4, 1).expect("ok");
        let mut rng = LcgRng::new(2024);
        let keys: Vec<u64> = (0..400u32).map(|_| rng.next_u64()).collect();
        for &k in &keys {
            cf.insert(k).expect("insert");
        }
        for &k in &keys {
            assert!(cf.contains(k), "missing key {k}");
        }
        for &k in &keys {
            assert!(cf.delete(k), "delete failed for {k}");
        }
        assert_eq!(cf.n_items, 0);
        // Deleted keys (32-bit fingerprints make collisions vanishingly unlikely) are now absent.
        for &k in &keys {
            assert!(!cf.contains(k), "key {k} still present after delete");
        }
    }

    #[test]
    fn cuckoo32_lower_fpr_than_narrow() {
        use std::collections::HashSet;
        let n_buckets = 4096;
        let bucket_size = 4;
        let seed = 555;
        // Same buckets / bucket_size / index hashing; only the fingerprint width differs.
        let mut narrow = CuckooFilter::new(n_buckets, bucket_size, 8, seed).expect("ok");
        let mut wide = CuckooFilter32::new(n_buckets, bucket_size, seed).expect("ok");

        let mut rng = LcgRng::new(0x1234_5678);
        let mut inserted: HashSet<u64> = HashSet::new();
        while inserted.len() < 8000 {
            let k = rng.next_u64();
            if inserted.insert(k) {
                narrow.insert(k).expect("narrow insert");
                wide.insert(k).expect("wide insert");
            }
        }

        // Neither filter may have false negatives.
        for &k in &inserted {
            assert!(narrow.contains(k), "narrow false negative {k}");
            assert!(wide.contains(k), "wide false negative {k}");
        }

        // Measure the false-positive rate over a large disjoint query set.
        let n_queries = 200_000usize;
        let mut narrow_fp = 0usize;
        let mut wide_fp = 0usize;
        let mut q = 0usize;
        while q < n_queries {
            let k = rng.next_u64();
            if inserted.contains(&k) {
                continue;
            }
            if narrow.contains(k) {
                narrow_fp += 1;
            }
            if wide.contains(k) {
                wide_fp += 1;
            }
            q += 1;
        }

        let narrow_fpr = narrow_fp as f64 / n_queries as f64;
        let wide_fpr = wide_fp as f64 / n_queries as f64;

        // The 8-bit filter has a clearly measurable FP rate (~2*b*load/2^8).
        assert!(
            narrow_fpr > 0.005,
            "narrow (8-bit) FPR unexpectedly low: {narrow_fpr} ({narrow_fp}/{n_queries})"
        );
        // The 32-bit fingerprint drives it down by orders of magnitude.
        assert!(
            wide_fpr * 50.0 < narrow_fpr,
            "wide FPR {wide_fpr} not far below narrow FPR {narrow_fpr}"
        );
        // Expected wide FP count over 200k queries is ~2*b/2^32 * 2e5 ≈ 4e-4, i.e. essentially 0.
        assert!(
            wide_fp <= 1,
            "wide (32-bit) false positives unexpectedly high: {wide_fp}/{n_queries}"
        );
    }
}

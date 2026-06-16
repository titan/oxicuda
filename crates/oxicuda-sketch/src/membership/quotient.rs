//! Quotient filter (Bender et al. 2011 PODS / 2012 SIGMOD Journal).
//!
//! Each fingerprint is split into:
//!   q (quotient, upper q_bits) – canonical slot index.
//!   r (remainder, lower r_bits) – value stored in the slot.
//!
//! Per-slot metadata bits:
//!   is_occupied    – canonical slot q has ≥1 stored element.
//!   is_continuation – this physical slot is not the first of its run.
//!   is_shifted      – this slot's remainder was displaced from its home.
//!
//! A *cluster* is a maximal block of consecutive non-empty physical slots.
//! A *run* is the group of remainders inside a cluster that belong to one
//! canonical quotient.  Within a run, remainders are sorted ascending.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

const IS_OCCUPIED_BIT: u64 = 1 << 63;
const IS_CONTINUATION_BIT: u64 = 1 << 62;
const IS_SHIFTED_BIT: u64 = 1 << 61;

const MAX_LOAD_FACTOR: f64 = 0.95;

/// Quotient filter with `2^q_bits` slots and `r_bits` remainder bits per slot.
#[derive(Debug, Clone)]
pub struct QuotientFilter {
    pub n_slots: usize,
    pub r_bits: u32,
    remainder_mask: u64,
    slots: Vec<u64>,
    pub n_items: usize,
    pub q_bits: u32,
    seed: u64,
}

impl QuotientFilter {
    pub fn new(q_bits: u32, r_bits: u32, seed: u64) -> SketchResult<Self> {
        if q_bits == 0 {
            return Err(SketchError::InvalidParameter {
                name: "q_bits".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if r_bits == 0 {
            return Err(SketchError::InvalidParameter {
                name: "r_bits".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if q_bits + r_bits > 61 {
            return Err(SketchError::InvalidParameter {
                name: "q_bits+r_bits".to_string(),
                reason: "combined fingerprint must be ≤ 61 bits (3 bits reserved for metadata)"
                    .to_string(),
            });
        }
        let n_slots = 1usize << q_bits;
        let remainder_mask = (1u64 << r_bits) - 1;
        Ok(Self {
            n_slots,
            r_bits,
            remainder_mask,
            slots: vec![0u64; n_slots],
            n_items: 0,
            q_bits,
            seed,
        })
    }

    #[must_use]
    pub fn fingerprint(&self, x: u64) -> u64 {
        let raw = xxh3_64_u64(x, self.seed);
        let total = self.q_bits + self.r_bits;
        if total >= 64 {
            raw
        } else {
            raw & ((1u64 << total) - 1)
        }
    }

    fn fp_quotient(&self, fp: u64) -> usize {
        (fp >> self.r_bits) as usize
    }

    fn fp_remainder(&self, fp: u64) -> u64 {
        fp & self.remainder_mask
    }

    fn get_is_occupied(&self, s: usize) -> bool {
        self.slots[s] & IS_OCCUPIED_BIT != 0
    }

    fn get_is_continuation(&self, s: usize) -> bool {
        self.slots[s] & IS_CONTINUATION_BIT != 0
    }

    fn get_is_shifted(&self, s: usize) -> bool {
        self.slots[s] & IS_SHIFTED_BIT != 0
    }

    fn get_remainder(&self, s: usize) -> u64 {
        self.slots[s] & self.remainder_mask
    }

    fn set_slot(&mut self, s: usize, is_occ: bool, is_cont: bool, is_shift: bool, rem: u64) {
        let mut v = rem & self.remainder_mask;
        if is_occ {
            v |= IS_OCCUPIED_BIT;
        }
        if is_cont {
            v |= IS_CONTINUATION_BIT;
        }
        if is_shift {
            v |= IS_SHIFTED_BIT;
        }
        self.slots[s] = v;
    }

    fn is_slot_empty(&self, s: usize) -> bool {
        self.slots[s] == 0
    }

    /// Walk backwards to the start of the cluster containing physical slot `s`.
    fn find_cluster_start(&self, s: usize) -> usize {
        let mut b = s;
        while b > 0 && self.get_is_shifted(b) {
            b -= 1;
        }
        b
    }

    /// Locate the physical slot where the run for canonical slot `q` starts.
    ///
    /// Called with the filter in its current (pre-insert) state where q may or
    /// may not be occupied.  Returns (slot, true) if occupied, (0, false) if not.
    fn find_run_index(&self, q: usize) -> (usize, bool) {
        if !self.get_is_occupied(q) {
            return (0, false);
        }
        let b = self.find_cluster_start(q);
        let rank = (b..=q).filter(|&i| self.get_is_occupied(i)).count();
        let mut j = b;
        let mut runs_seen = 0usize;
        loop {
            if j >= self.n_slots {
                return (0, false);
            }
            if !self.get_is_continuation(j) {
                runs_seen += 1;
                if runs_seen == rank {
                    return (j, true);
                }
            }
            j += 1;
        }
    }

    fn find_first_empty(&self, start: usize) -> Option<usize> {
        (start..self.n_slots).find(|&s| self.is_slot_empty(s))
    }

    /// Insert, shifting slots from `from` to `to` (inclusive) one slot to the right.
    fn shift_right_range(&mut self, from: usize, to: usize) {
        let mut s = to + 1;
        while s > from {
            let src = s - 1;
            let cont = self.get_is_continuation(src);
            let rem = self.get_remainder(src);
            let dst_occ = self.get_is_occupied(s);
            self.set_slot(s, dst_occ, cont, true, rem);
            s -= 1;
        }
    }

    pub fn insert(&mut self, x: u64) -> SketchResult<()> {
        if self.load_factor() >= MAX_LOAD_FACTOR {
            return Err(SketchError::CapacityExceeded {
                cap: self.n_slots,
                attempted: self.n_items + 1,
            });
        }

        let fp = self.fingerprint(x);
        let q = self.fp_quotient(fp);
        let r = self.fp_remainder(fp);

        let was_occupied = self.get_is_occupied(q);

        if !was_occupied && self.is_slot_empty(q) {
            self.set_slot(q, true, false, false, r);
            self.n_items += 1;
            return Ok(());
        }

        {
            let cont = self.get_is_continuation(q);
            let shift = self.get_is_shifted(q);
            let rem = self.get_remainder(q);
            self.set_slot(q, true, cont, shift, rem);
        }

        let run_start_opt: Option<usize> = if was_occupied {
            let (rs, _) = self.find_run_index(q);
            Some(rs)
        } else {
            None
        };

        let ins: usize;
        match run_start_opt {
            None => {
                let b = self.find_cluster_start(q);
                let rank_before_q = (b..q).filter(|&i| self.get_is_occupied(i)).count();
                let mut j = b;
                let mut runs_seen = 0usize;
                if rank_before_q == 0 {
                    while j < self.n_slots && !self.is_slot_empty(j) {
                        j += 1;
                    }
                    ins = j;
                } else {
                    loop {
                        if j >= self.n_slots || self.is_slot_empty(j) {
                            ins = j;
                            break;
                        }
                        if !self.get_is_continuation(j) {
                            runs_seen += 1;
                            if runs_seen == rank_before_q {
                                j += 1;
                                while j < self.n_slots && self.get_is_continuation(j) {
                                    j += 1;
                                }
                                ins = j;
                                break;
                            }
                        }
                        j += 1;
                    }
                }
            }
            Some(run_start) => {
                let mut s = run_start;
                loop {
                    let rem_s = self.get_remainder(s);
                    if rem_s >= r {
                        break;
                    }
                    let next = s + 1;
                    if next >= self.n_slots || !self.get_is_continuation(next) {
                        s = next;
                        break;
                    }
                    s = next;
                }
                ins = s;
            }
        }

        if ins >= self.n_slots {
            return Err(SketchError::CapacityExceeded {
                cap: self.n_slots,
                attempted: self.n_items + 1,
            });
        }

        let empty = match self.find_first_empty(ins) {
            Some(e) => e,
            None => {
                return Err(SketchError::CapacityExceeded {
                    cap: self.n_slots,
                    attempted: self.n_items + 1,
                });
            }
        };

        if empty > ins {
            self.shift_right_range(ins, empty - 1);
        }

        let is_cont = match run_start_opt {
            None => false,
            Some(rs) => ins != rs,
        };
        let is_shift = ins != q;
        let ins_occ = self.get_is_occupied(ins);
        self.set_slot(ins, ins_occ, is_cont, is_shift, r);

        if let Some(rs) = run_start_opt {
            if ins == rs && ins + 1 < self.n_slots && ins < empty {
                let next_occ = self.get_is_occupied(ins + 1);
                let next_rem = self.get_remainder(ins + 1);
                let next_shift = self.get_is_shifted(ins + 1);
                self.set_slot(ins + 1, next_occ, true, next_shift, next_rem);
            }
        }

        self.n_items += 1;
        Ok(())
    }

    #[must_use]
    pub fn contains(&self, x: u64) -> bool {
        let fp = self.fingerprint(x);
        let q = self.fp_quotient(fp);
        let r = self.fp_remainder(fp);

        if !self.get_is_occupied(q) {
            return false;
        }

        let (run_start, run_exists) = self.find_run_index(q);
        if !run_exists {
            return false;
        }

        let mut s = run_start;
        loop {
            let rem = self.get_remainder(s);
            if rem == r {
                return true;
            }
            if rem > r {
                return false;
            }
            s += 1;
            if s >= self.n_slots || !self.get_is_continuation(s) {
                return false;
            }
        }
    }

    pub fn delete(&mut self, x: u64) -> bool {
        let fp = self.fingerprint(x);
        let q = self.fp_quotient(fp);
        let r = self.fp_remainder(fp);

        if !self.get_is_occupied(q) {
            return false;
        }

        let (run_start, run_exists) = self.find_run_index(q);
        if !run_exists {
            return false;
        }

        let mut del = run_start;
        loop {
            let rem = self.get_remainder(del);
            if rem == r {
                break;
            }
            if rem > r {
                return false;
            }
            del += 1;
            if del >= self.n_slots || !self.get_is_continuation(del) {
                return false;
            }
        }

        let is_sole_in_run =
            del == run_start && (del + 1 >= self.n_slots || !self.get_is_continuation(del + 1));

        let mut s = del + 1;
        while s < self.n_slots && self.get_is_shifted(s) {
            let cont = self.get_is_continuation(s);
            let shift = self.get_is_shifted(s);
            let rem = self.get_remainder(s);
            let prev_occ = self.get_is_occupied(s - 1);
            self.set_slot(s - 1, prev_occ, cont, shift, rem);
            s += 1;
        }
        if s > 0 && s <= self.n_slots {
            let prev_occ = self.get_is_occupied(s - 1);
            self.set_slot(s - 1, prev_occ, false, false, 0);
        }

        if is_sole_in_run {
            let cont = self.get_is_continuation(q);
            let shift = self.get_is_shifted(q);
            let rem = self.get_remainder(q);
            self.set_slot(q, false, cont, shift, rem);
        }

        self.n_items = self.n_items.saturating_sub(1);
        true
    }

    #[must_use]
    pub fn load_factor(&self) -> f64 {
        self.n_items as f64 / self.n_slots as f64
    }

    #[must_use]
    pub fn estimated_fp_rate(&self) -> f64 {
        let p = 1.0 / (1u64 << self.r_bits.min(62)) as f64;
        1.0 - (1.0 - p).powi(self.n_items as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qf_new_empty() {
        let qf = QuotientFilter::new(8, 8, 0).expect("new should succeed");
        assert_eq!(qf.load_factor(), 0.0);
        assert_eq!(qf.n_items, 0);
    }

    #[test]
    fn qf_insert_lookup_basic() {
        let mut qf = QuotientFilter::new(8, 8, 1).expect("new should succeed");
        qf.insert(42).expect("insert should succeed");
        assert!(qf.contains(42));
    }

    #[test]
    fn qf_false_negative_free() {
        let mut qf = QuotientFilter::new(9, 8, 2).expect("new should succeed");
        for i in 0..100u64 {
            qf.insert(i).expect("insert should succeed");
        }
        for i in 0..100u64 {
            assert!(qf.contains(i), "false negative for item {i}");
        }
    }

    #[test]
    fn qf_false_positive_rate() {
        let r_bits = 8u32;
        let mut qf = QuotientFilter::new(8, r_bits, 3).expect("new should succeed");
        for i in 0..50u64 {
            qf.insert(i).expect("insert should succeed");
        }
        let trials = 5000u64;
        let mut fp_count = 0u64;
        for i in 1_000_000..1_000_000 + trials {
            if qf.contains(i) {
                fp_count += 1;
            }
        }
        let fp_rate = fp_count as f64 / trials as f64;
        let expected = 1.0 / (1u64 << r_bits) as f64;
        assert!(
            fp_rate <= expected * 4.0 + 0.02,
            "fp_rate={fp_rate} too high vs expected≈{expected}"
        );
    }

    #[test]
    fn qf_delete_found_item() {
        let mut qf = QuotientFilter::new(8, 8, 4).expect("new should succeed");
        qf.insert(99).expect("insert should succeed");
        assert!(qf.contains(99));
        let deleted = qf.delete(99);
        assert!(deleted);
    }

    #[test]
    fn qf_delete_not_inserted() {
        let mut qf = QuotientFilter::new(8, 8, 5).expect("new should succeed");
        assert!(!qf.delete(12345));
    }

    #[test]
    fn qf_load_factor_increases() {
        let mut qf = QuotientFilter::new(8, 8, 6).expect("new should succeed");
        let lf0 = qf.load_factor();
        for i in 0..10u64 {
            qf.insert(i).expect("insert should succeed");
        }
        assert!(qf.load_factor() > lf0);
    }

    #[test]
    fn qf_estimated_fp_rate_positive() {
        let mut qf = QuotientFilter::new(8, 8, 7).expect("new should succeed");
        qf.insert(1).expect("insert should succeed");
        assert!(qf.estimated_fp_rate() > 0.0);
    }

    #[test]
    fn qf_err_zero_q_bits() {
        assert!(QuotientFilter::new(0, 8, 0).is_err());
    }

    #[test]
    fn qf_err_zero_r_bits() {
        assert!(QuotientFilter::new(8, 0, 0).is_err());
    }

    #[test]
    fn qf_multiple_inserts_same_quotient() {
        let mut qf = QuotientFilter::new(2, 16, 8).expect("new should succeed");
        let mut inserted = Vec::new();
        for i in 0u64..10 {
            if qf.insert(i).is_ok() {
                inserted.push(i);
            }
        }
        for &item in &inserted {
            assert!(qf.contains(item), "missing item {item}");
        }
    }

    #[test]
    fn qf_full_returns_err() {
        let mut qf = QuotientFilter::new(2, 8, 9).expect("new should succeed");
        let mut inserted = 0;
        for i in 0u64..1000 {
            if qf.insert(i).is_err() {
                break;
            }
            inserted += 1;
        }
        let target = (qf.n_slots as f64 * MAX_LOAD_FACTOR).ceil() as usize;
        assert!(inserted >= target - 2, "inserted {inserted} < {target}-2");
    }

    #[test]
    fn qf_fingerprint_q_r_split() {
        let qf = QuotientFilter::new(4, 8, 10).expect("new should succeed");
        let fp = qf.fingerprint(42);
        let q = qf.fp_quotient(fp);
        let r = qf.fp_remainder(fp);
        assert!(q < qf.n_slots, "quotient {q} out of range");
        assert!(r <= qf.remainder_mask, "remainder {r} out of range");
    }

    #[test]
    fn qf_many_inserts_no_panic() {
        let mut qf = QuotientFilter::new(8, 8, 11).expect("new should succeed");
        let mut count = 0;
        for i in 0u64..100 {
            if qf.insert(i).is_ok() {
                count += 1;
            }
        }
        assert!(count >= 50, "only inserted {count} items");
    }

    #[test]
    fn qf_get_set_slot_metadata() {
        let mut qf = QuotientFilter::new(4, 8, 12).expect("new should succeed");
        qf.set_slot(0, true, false, true, 0xAB);
        assert!(qf.get_is_occupied(0));
        assert!(!qf.get_is_continuation(0));
        assert!(qf.get_is_shifted(0));
        assert_eq!(qf.get_remainder(0), 0xAB);

        qf.set_slot(1, false, true, false, 0x55);
        assert!(!qf.get_is_occupied(1));
        assert!(qf.get_is_continuation(1));
        assert!(!qf.get_is_shifted(1));
        assert_eq!(qf.get_remainder(1), 0x55);
    }

    #[test]
    fn qf_collision_handling() {
        let mut qf = QuotientFilter::new(3, 16, 13).expect("new should succeed");
        let mut inserted = Vec::new();
        for i in 0u64..20 {
            if qf.insert(i).is_ok() {
                inserted.push(i);
            }
        }
        for &item in &inserted {
            assert!(
                qf.contains(item),
                "collision: item {item} not found after insert"
            );
        }
    }

    #[test]
    fn qf_different_seeds_different_fp() {
        let qf1 = QuotientFilter::new(8, 8, 14).expect("new should succeed");
        let qf2 = QuotientFilter::new(8, 8, 99).expect("new should succeed");
        let fp1 = qf1.fingerprint(12345);
        let fp2 = qf2.fingerprint(12345);
        assert_ne!(
            fp1, fp2,
            "different seeds should produce different fingerprints"
        );
    }

    #[test]
    fn qf_large_r_bits_low_fp() {
        let r_bits = 20u32;
        let mut qf = QuotientFilter::new(4, r_bits, 15).expect("new should succeed");
        for i in 0u64..5 {
            qf.insert(i).expect("insert should succeed");
        }
        let actual = qf.estimated_fp_rate();
        assert!(
            actual < 0.001,
            "fp rate {actual} should be very low for r_bits={r_bits}"
        );
    }
}

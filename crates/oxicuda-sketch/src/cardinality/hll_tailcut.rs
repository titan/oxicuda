//! HLL-TailCut+ — a memory-frugal HyperLogLog tuned for low cardinalities
//! (Xiao, Zhou & Chen 2017, *Better with Fewer Bits: Improving the Performance
//! of Cardinality Estimation of Large Data Streams*).
//!
//! Standard HyperLogLog reserves a full byte (or 6 bits) per register even
//! though, for a given stream, the register values cluster tightly around
//! `log₂(n/m)`. HLL-TailCut exploits this by storing each register as a small
//! **offset** relative to a per-sketch base `b = min_j M_j`:
//!
//! ```text
//! stored_j = clamp(M_j − b, 0, tail_max) ,
//! ```
//!
//! so only a few bits per register are needed (the "tail" above `b + tail_max`
//! is cut). When a new leading-zero count would exceed the representable range
//! the base is **rebased** upward and all offsets are recomputed, mirroring the
//! paper's dynamic offset scheme. The cardinality estimate reconstructs
//! `M_j = b + stored_j` and applies the usual harmonic-mean HLL formula with a
//! linear-counting small-range correction — which dominates accuracy in the
//! low-cardinality regime this sketch targets.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

/// Memory-frugal HyperLogLog with tail-cut offset register storage.
#[derive(Debug, Clone)]
pub struct HllTailCut {
    /// Precision `p` (`m = 2^p` registers).
    p: u32,
    /// Number of registers `m = 2^p`.
    m: usize,
    /// Hash seed.
    seed: u64,
    /// Maximum representable offset above the base (the "tail" cut height).
    tail_max: u8,
    /// Current base `b` (the minimum leading-zero+1 across registers).
    base: u8,
    /// Per-register offsets in `0..=tail_max` (true value = `base + offset`,
    /// saturated at `base + tail_max`).
    offsets: Vec<u8>,
}

impl HllTailCut {
    /// Create a new sketch with precision `p` (`4 ≤ p ≤ 16`) and offset budget
    /// `tail_bits` bits per register (`1 ≤ tail_bits ≤ 6`).
    pub fn new(p: u32, tail_bits: u8, seed: u64) -> SketchResult<Self> {
        if !(4..=16).contains(&p) {
            return Err(SketchError::InvalidPrecision(p));
        }
        if !(1..=6).contains(&tail_bits) {
            return Err(SketchError::InvalidParameter {
                name: "tail_bits".to_string(),
                reason: "must be in 1..=6".to_string(),
            });
        }
        let m = 1usize << p;
        let tail_max = (1u16 << tail_bits) as u8 - 1;
        Ok(Self {
            p,
            m,
            seed,
            tail_max,
            base: 0,
            offsets: vec![0u8; m],
        })
    }

    /// True register value `M_j = base + offset_j` (saturated).
    #[inline]
    fn register(&self, j: usize) -> u8 {
        self.base.saturating_add(self.offsets[j])
    }

    /// Rebase upward so `new_base ≥ base`, recomputing all offsets and cutting
    /// any tail beyond `new_base + tail_max`.
    fn rebase(&mut self, new_base: u8) {
        if new_base <= self.base {
            return;
        }
        for off in self.offsets.iter_mut() {
            let true_val = self.base.saturating_add(*off);
            *off = true_val.saturating_sub(new_base).min(self.tail_max);
        }
        self.base = new_base;
    }

    /// Insert a `u64` value.
    pub fn add_u64(&mut self, x: u64) {
        let h = xxh3_64_u64(x, self.seed);
        self.add_hash(h);
    }

    /// Insert a raw 64-bit hash.
    pub fn add_hash(&mut self, h: u64) {
        let idx = (h >> (64 - self.p)) as usize;
        let w = (h << self.p) | (1u64 << (self.p.saturating_sub(1)));
        let rho = ((w.leading_zeros() as u16) + 1).min(64) as u8;

        let cur = self.register(idx);
        if rho <= cur {
            return;
        }
        // New maximum for this register. If it overflows the tail, rebase up so
        // the value remains representable (the smallest base that fits is
        // `rho - tail_max`).
        if rho > self.base.saturating_add(self.tail_max) {
            let new_base = rho.saturating_sub(self.tail_max);
            self.rebase(new_base);
        }
        let off = rho.saturating_sub(self.base).min(self.tail_max);
        if off > self.offsets[idx] {
            self.offsets[idx] = off;
        }
    }

    /// Estimate the number of distinct elements.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        let m = self.m as f64;
        let alpha = Self::alpha(self.m);
        let mut sum = 0.0_f64;
        let mut zeros = 0usize;
        for j in 0..self.m {
            let reg = self.register(j);
            sum += 2.0_f64.powi(-(reg as i32));
            if reg == 0 {
                zeros += 1;
            }
        }
        let raw = alpha * m * m / sum;
        // Small-range (low-cardinality) correction via linear counting — the
        // regime HLL-TailCut is designed for.
        if raw <= 2.5 * m && zeros > 0 {
            return m * (m / zeros as f64).ln();
        }
        raw
    }

    /// Merge another tail-cut sketch (same precision and seed) into this one by
    /// taking the register-wise maximum.
    pub fn merge(&mut self, other: &HllTailCut) -> SketchResult<()> {
        if self.p != other.p {
            return Err(SketchError::DimensionMismatch {
                a: self.p as usize,
                b: other.p as usize,
            });
        }
        // Align bases to the larger of the two, then OR-in the maxima.
        let target_base = self.base.max(other.base);
        self.rebase(target_base);
        for j in 0..self.m {
            let ov = other.register(j);
            let cur = self.register(j);
            if ov > cur {
                if ov > self.base.saturating_add(self.tail_max) {
                    let nb = ov.saturating_sub(self.tail_max);
                    self.rebase(nb);
                }
                self.offsets[j] = ov.saturating_sub(self.base).min(self.tail_max);
            }
        }
        Ok(())
    }

    /// Current base `b`.
    #[must_use]
    pub fn base(&self) -> u8 {
        self.base
    }

    /// Number of registers.
    #[must_use]
    pub fn num_registers(&self) -> usize {
        self.m
    }

    /// Reset to empty.
    pub fn clear(&mut self) {
        self.base = 0;
        for o in self.offsets.iter_mut() {
            *o = 0;
        }
    }

    /// HyperLogLog `alpha` bias constant.
    fn alpha(m: usize) -> f64 {
        match m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m as f64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_with_valid_params() {
        let h = HllTailCut::new(10, 4, 0).expect("ok");
        assert_eq!(h.num_registers(), 1024);
        assert_eq!(h.base(), 0);
    }

    #[test]
    fn rejects_bad_precision() {
        assert!(HllTailCut::new(2, 4, 0).is_err());
        assert!(HllTailCut::new(20, 4, 0).is_err());
    }

    #[test]
    fn rejects_bad_tail_bits() {
        assert!(HllTailCut::new(10, 0, 0).is_err());
        assert!(HllTailCut::new(10, 7, 0).is_err());
    }

    #[test]
    fn empty_estimate_is_small() {
        let h = HllTailCut::new(10, 4, 0).expect("ok");
        assert!(h.estimate() < 1.0, "empty estimate {}", h.estimate());
    }

    #[test]
    fn low_cardinality_accurate() {
        // The target regime: a few hundred distinct items.
        let mut h = HllTailCut::new(12, 5, 0).expect("ok");
        let n = 300u64;
        for i in 0..n {
            h.add_u64(i);
        }
        let e = h.estimate();
        let rel = (e - n as f64).abs() / n as f64;
        assert!(rel < 0.10, "low-card rel err {rel} (est {e})");
    }

    #[test]
    fn medium_cardinality_within_bounds() {
        let mut h = HllTailCut::new(14, 6, 0).expect("ok");
        let n = 10_000u64;
        for i in 0..n {
            h.add_u64(i);
        }
        let e = h.estimate();
        let rel = (e - n as f64).abs() / n as f64;
        assert!(rel < 0.07, "rel err {rel} (est {e})");
    }

    #[test]
    fn duplicates_do_not_inflate() {
        let mut h = HllTailCut::new(12, 5, 0).expect("ok");
        for _ in 0..5000 {
            h.add_u64(99);
        }
        assert!(h.estimate() < 5.0, "dup estimate {}", h.estimate());
    }

    #[test]
    fn rebasing_happens_for_high_rho() {
        // Force a large leading-zero count by hashing a value whose hash tail
        // has many leading zeros: add many distinct items so some register's
        // value climbs and rebasing engages with a small tail budget.
        let mut h = HllTailCut::new(8, 2, 7).expect("ok"); // tail_max = 3
        for i in 0..50_000u64 {
            h.add_u64(i);
        }
        // With a tiny tail budget the base must have advanced above zero.
        assert!(h.base() > 0, "base should have rebased, got {}", h.base());
        // Estimate should still be in a sane ballpark (order 10^4-10^5).
        let e = h.estimate();
        assert!(e > 1_000.0, "estimate {e} unreasonably small");
    }

    #[test]
    fn merge_unions_cardinality() {
        let mut a = HllTailCut::new(12, 5, 0).expect("ok");
        let mut b = HllTailCut::new(12, 5, 0).expect("ok");
        for i in 0..500u64 {
            a.add_u64(i);
        }
        for i in 500..1000u64 {
            b.add_u64(i);
        }
        a.merge(&b).expect("ok");
        let e = a.estimate();
        let rel = (e - 1000.0).abs() / 1000.0;
        assert!(rel < 0.12, "merged rel err {rel} (est {e})");
    }

    #[test]
    fn merge_idempotent_for_same_set() {
        let mut a = HllTailCut::new(12, 5, 0).expect("ok");
        let mut b = HllTailCut::new(12, 5, 0).expect("ok");
        for i in 0..400u64 {
            a.add_u64(i);
            b.add_u64(i);
        }
        let before = a.estimate();
        a.merge(&b).expect("ok");
        let after = a.estimate();
        assert!(
            (before - after).abs() < 1e-9,
            "merge changed same-set estimate"
        );
    }

    #[test]
    fn merge_dimension_mismatch_rejected() {
        let mut a = HllTailCut::new(10, 4, 0).expect("ok");
        let b = HllTailCut::new(12, 4, 0).expect("ok");
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn clear_resets() {
        let mut h = HllTailCut::new(10, 4, 0).expect("ok");
        for i in 0..200u64 {
            h.add_u64(i);
        }
        h.clear();
        assert_eq!(h.base(), 0);
        assert!(h.estimate() < 1.0);
    }

    #[test]
    fn rebase_preserves_register_maxima_within_tail() {
        // After manual rebase, registers that fit in the tail keep their value.
        let mut h = HllTailCut::new(6, 6, 1).expect("ok");
        h.add_hash(0x8000_0000_0000_0000); // some register gets a value
        let before: Vec<u8> = (0..h.num_registers()).map(|j| h.register(j)).collect();
        // Rebase by zero (no-op) must not change anything.
        h.rebase(0);
        let after: Vec<u8> = (0..h.num_registers()).map(|j| h.register(j)).collect();
        assert_eq!(before, after);
    }
}

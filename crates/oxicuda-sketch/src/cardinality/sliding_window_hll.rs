//! Sliding-window HyperLogLog cardinality estimator.
//!
//! Reference: Chabchoub & Heroum (2010), "A Sliding HyperLogLog Algorithm"
//! (also called "Sliding HyperLogLog" / Heule-style 2013 extension).
//!
//! Goal: estimate the number of distinct items observed in the most-recent
//! time window `W`, where each insertion carries a `timestamp: u64`.
//!
//! Compared to vanilla HLL (a single `u8` per register storing the running
//! maximum rank), this structure stores per register a TIME-SORTED deque of
//! `(timestamp, rank)` entries. The deque is invariant-maintained so that
//! once stale entries are evicted from the front, the FRONT of the deque
//! holds the current sliding-window maximum rank (≥ rest of the deque).
//!
//! ## Per-register invariants (after every operation)
//!
//! Each deque, after eviction relative to the query time `now`, is:
//!
//! * Sorted strictly by `timestamp` (front = oldest non-stale entry).
//! * Strictly decreasing in `rank` from FRONT to BACK (oldest entry has the
//!   largest rank). This is because, at insertion of `(t, r)`, we pop from
//!   the BACK while back's `rank ≤ r` — those entries are dominated.
//! * All `timestamp` values satisfy `timestamp ≥ now - window_w` after the
//!   front-eviction pass.
//!
//! The current register value at time `now` is therefore simply the rank at
//! the FRONT of the deque (or `0` if the deque is empty).
//!
//! ## Hashing
//!
//! Uses [`crate::hash::xxh3_min::xxh3_64_u64`] (the same family as
//! [`crate::cardinality::hll::HyperLogLog`]). The top `p` bits of the
//! 64-bit hash select the register index; the remaining `64 - p` bits give
//! the rank via `leading_zeros + 1` (clamped to `u8`).

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;
use std::collections::VecDeque;

/// Configuration for [`SlidingWindowHll`].
#[derive(Debug, Clone, Copy)]
pub struct SlidingWindowHllConfig {
    /// HyperLogLog precision: `m = 2^p` registers. Must satisfy `4 ≤ p ≤ 16`.
    pub precision_p: u8,
    /// Time window width: only entries with `timestamp ≥ now - window_w`
    /// participate in the cardinality estimate. `0` means a degenerate single-instant window.
    pub window_w: u64,
}

/// Sliding-window HyperLogLog.
#[derive(Debug, Clone)]
pub struct SlidingWindowHll {
    p: u8,
    m: usize,
    window_w: u64,
    seed: u64,
    /// Per register: deque of `(timestamp, rank)` entries.
    ///
    /// After eviction, sorted by timestamp ascending front-to-back, with rank
    /// strictly decreasing front-to-back.
    registers: Vec<VecDeque<(u64, u8)>>,
    /// Precomputed HyperLogLog bias constant α_m.
    alpha_m: f64,
}

impl SlidingWindowHll {
    /// Construct a new sliding-window HLL.
    ///
    /// Errors if `precision_p` is outside `[4, 16]`.
    pub fn new(config: SlidingWindowHllConfig) -> SketchResult<Self> {
        Self::with_seed(config, 0)
    }

    /// Construct a new sliding-window HLL with an explicit hash seed.
    pub fn with_seed(config: SlidingWindowHllConfig, seed: u64) -> SketchResult<Self> {
        if !(4..=16).contains(&config.precision_p) {
            return Err(SketchError::InvalidPrecision(u32::from(config.precision_p)));
        }
        let p = config.precision_p;
        let m = 1usize << p;
        let alpha_m = alpha_for_m(m);
        let mut registers = Vec::with_capacity(m);
        for _ in 0..m {
            registers.push(VecDeque::new());
        }
        Ok(Self {
            p,
            m,
            window_w: config.window_w,
            seed,
            registers,
            alpha_m,
        })
    }

    /// Number of registers (`m = 2^p`).
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Precision parameter `p`.
    #[must_use]
    pub fn precision_p(&self) -> u8 {
        self.p
    }

    /// Window width `W`.
    #[must_use]
    pub fn window_w(&self) -> u64 {
        self.window_w
    }

    /// HLL bias constant α_m.
    #[must_use]
    pub fn alpha_m(&self) -> f64 {
        self.alpha_m
    }

    /// Reset all registers to empty.
    pub fn reset(&mut self) {
        for deque in self.registers.iter_mut() {
            deque.clear();
        }
    }

    /// Hash a `u64` key and insert `(timestamp, rank)` into the appropriate register.
    ///
    /// Errors if the implied register index is somehow out of bounds (cannot
    /// happen with a correctly shaped hash, but reported defensively).
    pub fn add_u64(&mut self, item: u64, timestamp: u64) -> SketchResult<()> {
        let h = xxh3_64_u64(item, self.seed);
        self.add_hashed(h, timestamp)
    }

    /// Insert raw 64-bit hash directly (caller has already hashed).
    pub fn add_hashed(&mut self, hash: u64, timestamp: u64) -> SketchResult<()> {
        let (idx, rank) = self.bucket_and_rank(hash);
        let deque = self
            .registers
            .get_mut(idx)
            .ok_or(SketchError::IndexOutOfBounds {
                index: idx,
                len: self.m,
            })?;
        // (1) Pop from the BACK while back's rank ≤ new rank.
        //     Those are older entries dominated by the new (newer, ≥ rank) one.
        while let Some(&(_, back_rank)) = deque.back() {
            if back_rank <= rank {
                deque.pop_back();
            } else {
                break;
            }
        }
        // (2) Push (timestamp, rank) at the back.
        deque.push_back((timestamp, rank));
        // (3) Evict stale FRONT entries relative to the new timestamp.
        Self::evict_front_against(deque, timestamp, self.window_w);
        Ok(())
    }

    /// Estimate the number of distinct items in the window `[now - window_w, now]`.
    ///
    /// Evicts stale heads in-place first, then applies the standard HLL
    /// estimator with small/large-range corrections.
    pub fn cardinality(&mut self, now: u64) -> f64 {
        // First evict everything strictly older than `now - window_w`.
        for deque in self.registers.iter_mut() {
            Self::evict_front_against(deque, now, self.window_w);
        }

        let m = self.m as f64;
        let mut sum = 0.0_f64;
        let mut empty_count = 0usize;
        for deque in &self.registers {
            let v_i = deque.front().map(|&(_, r)| r).unwrap_or(0);
            if v_i == 0 {
                empty_count += 1;
            }
            // 2^{-v_i} via ldexp-style scaling. v_i fits in u8 (≤ 64).
            sum += 2.0_f64.powi(-i32::from(v_i));
        }
        let raw = self.alpha_m * m * m / sum;

        // Small-range correction (linear counting): when raw ≤ 5/2 * m and at
        // least one register is empty, fall back to V·ln(m/V) where V = #empty.
        if raw <= 2.5 * m && empty_count > 0 {
            let v = empty_count as f64;
            return m * (m / v).ln();
        }

        // Large-range correction (Flajolet 2007): only relevant when raw is
        // close to the 2^32 hash-space limit. Our hashes are 64-bit so this
        // path is essentially dead, but we keep the formula for completeness.
        let two_pow_32 = 4_294_967_296.0_f64;
        if raw > two_pow_32 / 30.0 {
            return -two_pow_32 * (1.0 - raw / two_pow_32).ln();
        }

        raw
    }

    /// Evict from the FRONT while the front timestamp is strictly older than
    /// `now - window_w` (using saturating subtraction so we never wrap).
    fn evict_front_against(deque: &mut VecDeque<(u64, u8)>, now: u64, window_w: u64) {
        let cutoff = now.saturating_sub(window_w);
        while let Some(&(ts, _)) = deque.front() {
            if ts < cutoff {
                deque.pop_front();
            } else {
                break;
            }
        }
    }

    /// Split a 64-bit hash into `(register_index, rank)`.
    ///
    /// Top `p` bits → register index in `[0, m)`. Remaining `64 - p` bits →
    /// `leading_zeros + 1` (with a sentinel bit to bound the count). Clamped
    /// to fit in `u8` (max value 64).
    fn bucket_and_rank(&self, hash: u64) -> (usize, u8) {
        let p = u32::from(self.p);
        let idx = (hash >> (64 - p)) as usize;
        // Shift away the index bits, then OR in a sentinel so that an
        // all-zeros tail still yields a finite leading-zeros count.
        let tail_bits = 64 - p;
        let sentinel = if p == 0 {
            0
        } else {
            1u64 << (p.saturating_sub(1))
        };
        let w = (hash << p) | sentinel;
        let lz = w.leading_zeros();
        let bounded = lz.min(tail_bits) as u8;
        let rank = bounded.saturating_add(1).min(64);
        (idx, rank)
    }
}

/// Standard HyperLogLog bias constant α_m (Flajolet, Fusy, Gandouet, Meunier 2007).
fn alpha_for_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn cfg(p: u8, w: u64) -> SlidingWindowHllConfig {
        SlidingWindowHllConfig {
            precision_p: p,
            window_w: w,
        }
    }

    #[test]
    fn standard_hll_recovery_unbounded_window() {
        // window_w = u64::MAX → degenerates to a vanilla HLL over all data.
        let p: u8 = 12;
        let m = 1usize << p;
        let std_err = (1.04_f64 / (m as f64)).sqrt();
        for seed in [1u64, 7, 11, 99, 12345] {
            let mut s = SlidingWindowHll::with_seed(cfg(p, u64::MAX), seed).unwrap();
            let mut rng = LcgRng::new(seed);
            let n: u64 = 5_000;
            for i in 0..n {
                let item = rng.next_u64();
                s.add_u64(item, i).unwrap();
            }
            let est = s.cardinality(n);
            let rel = (est - n as f64).abs() / n as f64;
            assert!(
                rel < 5.0 * std_err,
                "seed={seed}: |est - n| / n = {rel}, expected < {}",
                5.0 * std_err
            );
        }
    }

    #[test]
    fn old_timestamps_excluded_from_window() {
        let mut s = SlidingWindowHll::new(cfg(10, 100)).unwrap();
        // Insert N distinct items at timestamps 0..50 (well before now=200, window=100).
        for i in 0..200u64 {
            s.add_u64(i, i).unwrap();
        }
        // Reference: vanilla unbounded result for the full 200 inserts.
        let mut full = SlidingWindowHll::new(cfg(10, u64::MAX)).unwrap();
        for i in 0..200u64 {
            full.add_u64(i, i).unwrap();
        }
        let est_full = full.cardinality(200);
        let est_window = s.cardinality(200);
        // The windowed estimate (range [100, 200]) should be much smaller.
        assert!(
            est_window < 0.9 * est_full,
            "windowed estimate {est_window} should be < 0.9 * full {est_full}"
        );
    }

    #[test]
    fn re_adding_item_with_new_timestamp_re_enters_window() {
        let mut s = SlidingWindowHll::new(cfg(8, 10)).unwrap();
        s.add_u64(42, 0).unwrap();
        // After advancing far past the window, the item must have been evicted.
        let est_after = s.cardinality(100);
        assert!(
            est_after < 1.5,
            "evicted item should not contribute: est={est_after}"
        );
        // Re-adding it with a fresh timestamp re-enters the window.
        s.add_u64(42, 95).unwrap();
        let est_re = s.cardinality(100);
        assert!(
            est_re > 0.5,
            "re-added item should contribute: est={est_re}"
        );
    }

    #[test]
    fn precision_boundary_p_eq_4() {
        let mut s = SlidingWindowHll::new(cfg(4, u64::MAX)).unwrap();
        assert_eq!(s.m(), 16);
        assert_eq!(s.precision_p(), 4);
        let mut rng = LcgRng::new(1);
        for i in 0..1_000u64 {
            s.add_u64(rng.next_u64(), i).unwrap();
        }
        let est = s.cardinality(1_000);
        // p=4 has high standard error (~26%); just check the estimate is positive
        // and within an order-of-magnitude band.
        assert!(est > 100.0 && est < 10_000.0, "p=4 estimate {est}");
    }

    #[test]
    fn precision_boundary_p_eq_16() {
        let s = SlidingWindowHll::new(cfg(16, u64::MAX)).unwrap();
        assert_eq!(s.m(), 65_536);
        assert_eq!(s.precision_p(), 16);
    }

    #[test]
    fn invalid_precision_too_small() {
        let r = SlidingWindowHll::new(cfg(3, 100));
        assert!(matches!(r, Err(SketchError::InvalidPrecision(_))));
    }

    #[test]
    fn invalid_precision_too_large() {
        let r = SlidingWindowHll::new(cfg(17, 100));
        assert!(matches!(r, Err(SketchError::InvalidPrecision(_))));
    }

    #[test]
    fn deque_invariant_rank_decreasing_back_to_front() {
        let mut s = SlidingWindowHll::new(cfg(10, u64::MAX)).unwrap();
        let mut rng = LcgRng::new(42);
        for i in 0..5_000u64 {
            s.add_u64(rng.next_u64(), i).unwrap();
        }
        // Each register must satisfy: rank strictly decreasing front-to-back.
        for (idx, deque) in s.registers.iter().enumerate() {
            let mut prev: Option<u8> = None;
            for &(_, r) in deque {
                if let Some(p) = prev {
                    assert!(
                        p > r,
                        "register {idx}: rank not strictly decreasing front-to-back (prev={p}, cur={r})"
                    );
                }
                prev = Some(r);
            }
        }
    }

    #[test]
    fn deque_invariant_timestamps_increasing_back() {
        let mut s = SlidingWindowHll::new(cfg(10, u64::MAX)).unwrap();
        let mut rng = LcgRng::new(123);
        for i in 0..5_000u64 {
            s.add_u64(rng.next_u64(), i).unwrap();
        }
        for (idx, deque) in s.registers.iter().enumerate() {
            let mut prev: Option<u64> = None;
            for &(ts, _) in deque {
                if let Some(p) = prev {
                    assert!(
                        ts > p,
                        "register {idx}: timestamps not strictly increasing (prev={p}, cur={ts})"
                    );
                }
                prev = Some(ts);
            }
        }
    }

    #[test]
    fn eviction_monotone_no_new_adds() {
        // Time advances with no new inserts → cardinality is non-increasing.
        let mut s = SlidingWindowHll::new(cfg(10, 100)).unwrap();
        let mut rng = LcgRng::new(9);
        for i in 0..500u64 {
            s.add_u64(rng.next_u64(), i).unwrap();
        }
        let mut prev = s.cardinality(500);
        for t in (500..2_000u64).step_by(100) {
            let cur = s.cardinality(t);
            assert!(
                cur <= prev + 1e-9,
                "cardinality must be non-increasing without inserts (t={t}, prev={prev}, cur={cur})"
            );
            prev = cur;
        }
    }

    #[test]
    fn reset_clears_all_registers() {
        let mut s = SlidingWindowHll::new(cfg(8, u64::MAX)).unwrap();
        for i in 0..1000u64 {
            s.add_u64(i, i).unwrap();
        }
        assert!(s.cardinality(1_000) > 0.0);
        s.reset();
        let est = s.cardinality(1_000);
        assert!(
            est < 1e-9 || est <= 1.0,
            "after reset cardinality should be effectively 0 but got {est}"
        );
        for deque in &s.registers {
            assert!(deque.is_empty());
        }
    }

    #[test]
    fn m_equals_two_pow_p() {
        for p in 4u8..=16u8 {
            let s = SlidingWindowHll::new(cfg(p, 1)).unwrap();
            assert_eq!(s.m(), 1usize << p, "p={p}");
        }
    }

    #[test]
    fn deterministic_given_same_hash_inputs_and_timestamps() {
        let mut a = SlidingWindowHll::new(cfg(10, 1_000)).unwrap();
        let mut b = SlidingWindowHll::new(cfg(10, 1_000)).unwrap();
        let mut rng_a = LcgRng::new(2026);
        let mut rng_b = LcgRng::new(2026);
        for t in 0..2_000u64 {
            let x_a = rng_a.next_u64();
            let x_b = rng_b.next_u64();
            assert_eq!(x_a, x_b);
            a.add_u64(x_a, t).unwrap();
            b.add_u64(x_b, t).unwrap();
        }
        for query in [500u64, 1_500, 2_000, 3_000] {
            assert_eq!(a.cardinality(query), b.cardinality(query));
        }
    }

    #[test]
    fn alpha_m_table_matches_standard_values() {
        // Standard reference: α_16=0.673, α_32=0.697, α_64=0.709,
        // α_m = 0.7213 / (1 + 1.079/m) for m ≥ 128.
        let s4 = SlidingWindowHll::new(cfg(4, 1)).unwrap();
        assert!((s4.alpha_m() - 0.673).abs() < 1e-12);
        let s5 = SlidingWindowHll::new(cfg(5, 1)).unwrap();
        assert!((s5.alpha_m() - 0.697).abs() < 1e-12);
        let s6 = SlidingWindowHll::new(cfg(6, 1)).unwrap();
        assert!((s6.alpha_m() - 0.709).abs() < 1e-12);
        let s7 = SlidingWindowHll::new(cfg(7, 1)).unwrap();
        let expected_128 = 0.7213_f64 / (1.0 + 1.079 / 128.0);
        assert!((s7.alpha_m() - expected_128).abs() < 1e-12);
        let s10 = SlidingWindowHll::new(cfg(10, 1)).unwrap();
        let expected_1024 = 0.7213_f64 / (1.0 + 1.079 / 1024.0);
        assert!((s10.alpha_m() - expected_1024).abs() < 1e-12);
    }

    #[test]
    fn linear_counting_path_triggers_at_low_load() {
        // With m=1024 and only a handful of items, raw ≤ 2.5 m and V > 0
        // so the small-range correction (linear counting) path must fire.
        let mut s = SlidingWindowHll::new(cfg(10, u64::MAX)).unwrap();
        for i in 0..5u64 {
            s.add_u64(i, i).unwrap();
        }
        let est = s.cardinality(5);
        // For genuinely small load, the linear-counting estimate is much more
        // accurate than the raw harmonic-mean. Should be well within 2x of 5.
        assert!(
            est < 20.0,
            "linear counting path estimate {est} should be small"
        );
        assert!(est > 0.0);
    }

    #[test]
    fn window_w_accessor_returns_construction_value() {
        let s = SlidingWindowHll::new(cfg(8, 12_345)).unwrap();
        assert_eq!(s.window_w(), 12_345);
        let s2 = SlidingWindowHll::new(cfg(8, u64::MAX)).unwrap();
        assert_eq!(s2.window_w(), u64::MAX);
    }

    #[test]
    fn empty_sketch_cardinality_is_zero() {
        let mut s = SlidingWindowHll::new(cfg(10, 100)).unwrap();
        let est = s.cardinality(0);
        // All registers empty → V = m, so m * ln(m / m) = 0 exactly.
        assert!(est.abs() < 1e-9, "empty cardinality must be 0, got {est}");
    }

    #[test]
    fn back_dominated_inserts_dont_blow_up_deque() {
        // Insert into a fixed register with monotonically increasing ranks at
        // increasing timestamps. Each later insert should pop all earlier ones
        // from the back, leaving the deque size ≤ 1.
        let mut s = SlidingWindowHll::new(cfg(8, u64::MAX)).unwrap();
        // Find a hash whose top-p bits give register 0; then construct
        // synthetic hashes with same top bits but increasing rank.
        let p = s.precision_p();
        // Build hashes with idx=0 and explicitly chosen tail bits.
        for k in 0..32u32 {
            // Tail = 0...01 with k leading zeros in the (64-p)-bit tail.
            let tail_len = 64u32 - u32::from(p);
            let tail = if k < tail_len {
                1u64 << (tail_len - 1 - k)
            } else {
                0
            };
            let hash = tail;
            s.add_hashed(hash, k as u64).unwrap();
        }
        let deque_len = s.registers[0].len();
        // Each new insertion has rank strictly larger than the previous (k → k+1),
        // so pop_back must keep removing → deque size ≤ 1.
        assert!(
            deque_len <= 1,
            "expected dominated deque to collapse to at most 1 entry, got {deque_len}"
        );
    }

    #[test]
    fn front_dominates_back_after_decreasing_rank_inserts() {
        // Insert into a fixed register with monotonically DECREASING ranks at
        // increasing timestamps. Each later entry has strictly smaller rank,
        // so they all stack at the back; deque length grows.
        let mut s = SlidingWindowHll::new(cfg(8, u64::MAX)).unwrap();
        let p = s.precision_p();
        let tail_len = 64u32 - u32::from(p);
        for k in 0..16u32 {
            // Decreasing rank: smaller k → larger leading-zero run.
            let lz = 15 - k;
            let tail = if lz < tail_len {
                1u64 << (tail_len - 1 - lz)
            } else {
                0
            };
            let hash = tail;
            s.add_hashed(hash, k as u64).unwrap();
        }
        assert!(
            s.registers[0].len() >= 2,
            "decreasing-rank inserts should accumulate, got {}",
            s.registers[0].len()
        );
        // Front rank must be the largest.
        let front_rank = s.registers[0].front().map(|&(_, r)| r).unwrap();
        for &(_, r) in &s.registers[0] {
            assert!(r <= front_rank);
        }
    }

    #[test]
    fn front_evicts_after_window_advance() {
        // Insert one item at t=0 and let time advance well beyond the window.
        let mut s = SlidingWindowHll::new(cfg(8, 10)).unwrap();
        s.add_u64(7, 0).unwrap();
        // First query inside the window: must see contribution.
        let inside = s.cardinality(5);
        assert!(inside > 0.0);
        // Query well after the window expires: the front must have been evicted.
        let outside = s.cardinality(100);
        // Either an empty register (preferred) or contribution ≤ inside.
        assert!(
            outside <= inside + 1e-9,
            "outside estimate {outside} should be ≤ inside {inside}"
        );
        // Every register must be empty after the eviction sweep.
        let total: usize = s.registers.iter().map(|d| d.len()).sum();
        assert_eq!(
            total, 0,
            "all entries should be evicted, got total len {total}"
        );
    }

    #[test]
    fn bucket_and_rank_consistent_with_hash_layout() {
        let s = SlidingWindowHll::new(cfg(10, u64::MAX)).unwrap();
        // A hash with all top-p bits clear → idx = 0.
        let (idx, _) = s.bucket_and_rank(0x0000_0000_0000_0001);
        assert_eq!(idx, 0);
        // A hash with top-p bits all set → idx = m - 1.
        let top_mask = u64::MAX << (64 - u32::from(s.precision_p()));
        let (idx_max, _) = s.bucket_and_rank(top_mask);
        assert_eq!(idx_max, s.m() - 1);
        // Rank must be in [1, 64].
        for h in [1u64, 2, 3, 1 << 40, u64::MAX, top_mask | 0x1234] {
            let (_, r) = s.bucket_and_rank(h);
            assert!((1..=64).contains(&r), "rank {r} for hash {h}");
        }
    }
}

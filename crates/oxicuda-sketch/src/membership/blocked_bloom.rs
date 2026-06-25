//! Blocked Bloom filter ("Bloom-1") — Putze, Sanders, Singler (JEA 2009),
//! "Cache-, Hash- and Space-Efficient Bloom Filters".
//!
//! A classic Bloom filter sets `k` bits at positions scattered across the whole
//! `m`-bit array, so each membership test touches up to `k` distinct cache
//! lines. The *blocked* variant partitions the array into fixed-size **blocks**
//! (one machine cache line each) and confines all `k` bits of an element to a
//! single block selected by a first hash. Every insert/query then touches
//! exactly one block ⇒ one cache miss, which is the whole point of the design
//! for very large filters.
//!
//! ## Layout
//!
//! * The filter holds `n_blocks` blocks of `BLOCK_BITS = 512` bits
//!   (`8 × u64` words, i.e. a typical 64-byte cache line).
//! * An element `x` first picks a block `blk = h0(x) mod n_blocks`, then sets
//!   `k` bits *inside that block* using double hashing
//!   `(h1 + i·h2) mod BLOCK_BITS`.
//!
//! ## False-positive rate
//!
//! Confining bits to a block makes the per-block load non-uniform (a Poisson
//! number of items land in each block), so the FP rate is slightly **worse**
//! than a global Bloom filter of the same size. A good engineering
//! approximation averages the standard `(1 − e^{−k·λ/B})^k` curve over the
//! Poisson block-occupancy distribution, where `B = BLOCK_BITS`,
//! `λ = n / n_blocks` is the mean items per block. [`crate::membership::blocked_bloom::BlockedBloomFilter::expected_fp_rate`]
//! evaluates exactly this Poisson-averaged expression.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

/// Bits per block (one 64-byte cache line).
const BLOCK_BITS: usize = 512;
/// `u64` words per block.
const BLOCK_WORDS: usize = BLOCK_BITS / 64;

/// Blocked ("Bloom-1") Bloom filter.
#[derive(Debug, Clone)]
pub struct BlockedBloomFilter {
    /// Number of cache-line blocks.
    pub n_blocks: usize,
    /// Hash functions per element (bits set within the chosen block).
    pub k: usize,
    /// Bit storage: `n_blocks · BLOCK_WORDS` words, block-major.
    pub bits: Vec<u64>,
    /// Base seed mixed into all hashes.
    pub seed_base: u64,
}

impl BlockedBloomFilter {
    /// Construct a blocked Bloom filter with `n_blocks` blocks and `k` hashes
    /// per element. Total capacity is `n_blocks × 512` bits.
    pub fn new(n_blocks: usize, k: usize, seed_base: u64) -> SketchResult<Self> {
        if n_blocks == 0 || k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(n_blocks, k)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            n_blocks,
            k,
            bits: vec![0u64; n_blocks * BLOCK_WORDS],
            seed_base,
        })
    }

    /// Choose parameters to hold `n_expected` items at target FP rate `p`.
    ///
    /// We size the total bit budget with the classic
    /// `m = −n·ln(p)/(ln 2)²`, `k = (m/n)·ln 2` formulae, then round the block
    /// count up so `n_blocks · 512 ≥ m`. Because blocking inflates the FP rate
    /// modestly, the realised rate is bounded by [`Self::expected_fp_rate`];
    /// for a tighter target supply a smaller `p`.
    pub fn with_expected_fp(n_expected: usize, p: f64, seed_base: u64) -> SketchResult<Self> {
        if !(0.0 < p && p < 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "p".to_string(),
                reason: "must be in (0,1)".to_string(),
            });
        }
        if n_expected == 0 {
            return Err(SketchError::InvalidParameter {
                name: "n_expected".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let ln2 = std::f64::consts::LN_2;
        let m = (-(n_expected as f64) * p.ln() / (ln2 * ln2)).ceil() as usize;
        let k = ((m as f64 / n_expected as f64) * ln2).round() as usize;
        let n_blocks = m.max(BLOCK_BITS).div_ceil(BLOCK_BITS);
        Self::new(n_blocks.max(1), k.max(1), seed_base)
    }

    /// Total number of bits across all blocks.
    #[must_use]
    pub fn m(&self) -> usize {
        self.n_blocks * BLOCK_BITS
    }

    /// Select the block index for `x`.
    fn block_of(&self, x: u64) -> usize {
        let h0 = xxh3_64_u64(x, self.seed_base);
        (h0 as usize) % self.n_blocks
    }

    /// The `k` in-block bit positions (`0 ≤ pos < 512`) for `x`, via double
    /// hashing with two independent derived hashes.
    fn in_block_positions(&self, x: u64) -> impl Iterator<Item = usize> + '_ {
        let h1 = xxh3_64_u64(x, self.seed_base.wrapping_add(0x9E37_79B9_7F4A_7C15));
        let h2 = xxh3_64_u64(x, self.seed_base.wrapping_add(0xC2B2_AE3D_27D4_EB4F)) | 1; // odd ⇒ coprime with 512, full bit coverage
        (0..self.k)
            .map(move |i| (h1.wrapping_add((i as u64).wrapping_mul(h2)) as usize) % BLOCK_BITS)
    }

    /// Insert an item.
    pub fn insert(&mut self, x: u64) {
        let blk = self.block_of(x);
        let base = blk * BLOCK_WORDS;
        let positions: Vec<usize> = self.in_block_positions(x).collect();
        for pos in positions {
            self.bits[base + pos / 64] |= 1u64 << (pos % 64);
        }
    }

    /// Test membership. Never returns a false negative.
    #[must_use]
    pub fn contains(&self, x: u64) -> bool {
        let blk = self.block_of(x);
        let base = blk * BLOCK_WORDS;
        for pos in self.in_block_positions(x) {
            if (self.bits[base + pos / 64] >> (pos % 64)) & 1 == 0 {
                return false;
            }
        }
        true
    }

    /// Number of set bits across the whole filter.
    #[must_use]
    pub fn popcount(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Poisson-averaged false-positive rate after `n` insertions.
    ///
    /// With `λ = n / n_blocks` items per block on average and `B = 512` bits
    /// per block, the per-block FP rate when `j` items land there is
    /// `(1 − e^{−k·j/B})^k`. Averaging over `j ∼ Poisson(λ)` gives
    ///
    /// ```text
    ///     FP ≈ Σ_{j≥0} e^{−λ} λ^j / j! · (1 − e^{−k·j/B})^k .
    /// ```
    #[must_use]
    pub fn expected_fp_rate(&self, n: usize) -> f64 {
        let lambda = n as f64 / self.n_blocks as f64;
        if lambda <= 0.0 {
            return 0.0;
        }
        // Truncate the Poisson sum well beyond the mean for accuracy.
        let jmax = (lambda + 12.0 * lambda.sqrt()).ceil() as usize + 16;
        let mut term = (-lambda).exp(); // P(j = 0)
        let mut acc = 0.0;
        for j in 0..=jmax {
            let fp_j = if j == 0 {
                0.0
            } else {
                let load = -(self.k as f64) * (j as f64) / (BLOCK_BITS as f64);
                (1.0 - load.exp()).powi(self.k as i32)
            };
            acc += term * fp_j;
            // Advance Poisson pmf: P(j+1) = P(j) · λ / (j+1).
            term *= lambda / (j as f64 + 1.0);
        }
        acc.clamp(0.0, 1.0)
    }

    /// Reset to empty.
    pub fn clear(&mut self) {
        for w in self.bits.iter_mut() {
            *w = 0;
        }
    }

    /// Bitwise-OR merge with another blocked filter of identical geometry.
    pub fn merge(&mut self, other: &BlockedBloomFilter) -> SketchResult<()> {
        if self.n_blocks != other.n_blocks || self.k != other.k {
            return Err(SketchError::DimensionMismatch {
                a: self.n_blocks,
                b: other.n_blocks,
            });
        }
        for i in 0..self.bits.len() {
            self.bits[i] |= other.bits[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_constructs() {
        let bf = BlockedBloomFilter::new(64, 7, 0).expect("ok");
        assert_eq!(bf.n_blocks, 64);
        assert_eq!(bf.k, 7);
        assert_eq!(bf.m(), 64 * 512);
        assert_eq!(bf.bits.len(), 64 * BLOCK_WORDS);
    }

    #[test]
    fn blocked_invalid_params() {
        assert!(BlockedBloomFilter::new(0, 4, 0).is_err());
        assert!(BlockedBloomFilter::new(4, 0, 0).is_err());
        assert!(BlockedBloomFilter::with_expected_fp(1000, 0.0, 0).is_err());
        assert!(BlockedBloomFilter::with_expected_fp(0, 0.01, 0).is_err());
    }

    #[test]
    fn blocked_no_false_negatives() {
        let mut bf = BlockedBloomFilter::new(128, 7, 0).expect("ok");
        for i in 0..2000u64 {
            bf.insert(i);
        }
        for i in 0..2000u64 {
            assert!(bf.contains(i), "missing inserted item {i}");
        }
    }

    #[test]
    fn blocked_all_bits_in_one_block() {
        // After inserting a single element, exactly one block holds set bits.
        let mut bf = BlockedBloomFilter::new(32, 7, 123).expect("ok");
        bf.insert(0xABCD_1234);
        let mut non_empty_blocks = 0usize;
        for blk in 0..bf.n_blocks {
            let base = blk * BLOCK_WORDS;
            let any = (0..BLOCK_WORDS).any(|w| bf.bits[base + w] != 0);
            if any {
                non_empty_blocks += 1;
            }
        }
        assert_eq!(non_empty_blocks, 1, "all k bits must live in one block");
    }

    #[test]
    fn blocked_fp_rate_reasonable() {
        let n = 4000usize;
        let mut bf = BlockedBloomFilter::with_expected_fp(n, 0.01, 7).expect("ok");
        for i in 0..n as u64 {
            bf.insert(i);
        }
        let mut fp = 0usize;
        let trials = 20_000u64;
        for i in 1_000_000..1_000_000 + trials {
            if bf.contains(i) {
                fp += 1;
            }
        }
        let rate = fp as f64 / trials as f64;
        let predicted = bf.expected_fp_rate(n);
        // Empirical rate must track the Poisson-averaged prediction (blocking
        // inflates over the global 1% target but stays modest).
        assert!(rate < 0.06, "blocked FP rate {rate} too high");
        assert!(
            predicted < 0.06 && predicted > 0.0,
            "predicted {predicted} out of expected band"
        );
        assert!(
            (rate - predicted).abs() < 0.04,
            "empirical {rate} vs predicted {predicted}"
        );
    }

    #[test]
    fn blocked_fp_worse_than_global_for_same_size() {
        // Sanity: the Poisson-averaged rate is >= the idealised uniform rate.
        let bf = BlockedBloomFilter::new(50, 6, 0).expect("ok");
        let n = 3000usize;
        let lambda = n as f64 / bf.n_blocks as f64;
        let uniform = (1.0 - (-(bf.k as f64) * lambda / 512.0).exp()).powi(bf.k as i32);
        let poisson_avg = bf.expected_fp_rate(n);
        assert!(
            poisson_avg >= uniform - 1e-9,
            "blocked {poisson_avg} should be >= uniform {uniform}"
        );
    }

    #[test]
    fn blocked_merge() {
        let mut a = BlockedBloomFilter::new(16, 5, 11).expect("ok");
        let mut b = BlockedBloomFilter::new(16, 5, 11).expect("ok");
        a.insert(1);
        b.insert(2);
        a.merge(&b).expect("ok");
        assert!(a.contains(1));
        assert!(a.contains(2));
        let mut wrong = BlockedBloomFilter::new(8, 5, 11).expect("ok");
        assert!(a.merge(&wrong).is_err());
        wrong.clear();
    }

    #[test]
    fn blocked_clear_empties() {
        let mut bf = BlockedBloomFilter::new(8, 5, 0).expect("ok");
        bf.insert(42);
        assert!(bf.popcount() > 0);
        bf.clear();
        assert_eq!(bf.popcount(), 0);
        assert!(!bf.contains(42));
    }
}

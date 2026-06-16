//! Bloomier filter — a function-valued Bloom filter (Chazelle, Kilian,
//! Rubinfeld & Tal 2004, *The Bloomier Filter: An Efficient Data Structure for
//! Static Support Lookup Tables*).
//!
//! A classic Bloom filter answers *membership* (`x ∈ S?`). A Bloomier filter
//! generalises this to a *partial function* `f : S → V`: for keys in the support
//! `S` it returns the correct value `f(x)`; for keys outside `S` it returns a
//! "don't-know" sentinel (`None`) with high probability, and only with small
//! probability returns a spurious value.
//!
//! # Construction (greedy hypergraph peeling)
//!
//! Each key `x` hashes to `k` cells `h₁(x), …, h_k(x)` in a table of `m` cells
//! and to a `w`-bit value-mask `M(x)`. The static support table is built by an
//! **order-and-match** procedure (the "matching" of CKRT, equivalent to the
//! peeling used in XOR/cuckoo/ribbon filters):
//!
//! 1. Repeatedly find a key that owns a cell occupied by no other not-yet-placed
//!    key (a *singleton* cell). Place that key and remove it. This yields an
//!    ordering in which every key has a private cell.
//! 2. Process keys in **reverse** placement order, assigning the table so that
//!    for key `x` with private cell `ℓ`:
//!
//!    ```text
//!    table[ℓ] = encode(value(x)) XOR M(x) XOR ⊕_{j ≠ ℓ} table[h_j(x)] .
//!    ```
//!
//!    Lookup recomputes `r = M(x) XOR ⊕_j table[h_j(x)]`; the low `value_bits`
//!    of `r` are the decoded value and the high bits act as a checksum that, if
//!    non-zero, flags `x` as not in the support.
//!
//! Construction can fail if the random hypergraph is not peelable (probability
//! shrinks rapidly when `m ≥ 1.23·n` for `k = 3`); callers retry with a fresh
//! seed. Values are stored as `u64` and must fit in `value_bits` bits.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

/// A static Bloomier filter mapping a fixed key set to `u64` values.
#[derive(Debug, Clone)]
pub struct BloomierFilter {
    /// Number of table cells.
    m: usize,
    /// Number of hash probes per key.
    k: usize,
    /// Bits used to store a value (the remaining high bits form a checksum).
    value_bits: u32,
    /// Hash seed base.
    seed_base: u64,
    /// The encoded table.
    table: Vec<u64>,
}

impl BloomierFilter {
    /// `k` *distinct* probe locations for key `x`. Distinctness keeps the XOR
    /// encoding consistent with the peeling (which works on distinct cells).
    /// When `m` is at least `k` (guaranteed by `build`), `k` distinct cells
    /// always exist; collisions are resolved by linear probing.
    fn locations(m: usize, k: usize, seed_base: u64, x: u64) -> Vec<usize> {
        let h1 = xxh3_64_u64(x, seed_base);
        let h2 = xxh3_64_u64(x, seed_base.wrapping_add(0x9E37_79B9_7F4A_7C15));
        let mut out: Vec<usize> = Vec::with_capacity(k);
        for i in 0..k {
            let mut c = h1.wrapping_add((i as u64).wrapping_mul(h2)) as usize % m;
            // Linear-probe forward until a fresh cell is found.
            while out.contains(&c) {
                c = (c + 1) % m;
            }
            out.push(c);
        }
        out
    }

    /// The value-mask `M(x)`: a pseudo-random `u64` mixed from a distinct seed.
    fn value_mask(seed_base: u64, x: u64) -> u64 {
        xxh3_64_u64(x, seed_base.wrapping_add(0xD1B5_4A32_D192_ED03))
    }

    /// Encode a stored value with a non-zero high-bit checksum so out-of-support
    /// keys are flagged. The value occupies the low `value_bits`; the checksum
    /// (a fixed function of the value) occupies the rest.
    fn encode(value_bits: u32, value: u64) -> u64 {
        let value_mask = if value_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << value_bits) - 1
        };
        let lo = value & value_mask;
        // Checksum: a non-zero pattern derived from the value, placed above the
        // value bits. For in-support keys the recovered checksum matches; for
        // spurious keys it almost never matches, flagging "not present".
        let checksum = if value_bits >= 64 {
            0
        } else {
            let cs =
                (lo.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1) & ((1u64 << (64 - value_bits)) - 1);
            cs << value_bits
        };
        lo | checksum
    }

    /// Decode a recovered word; returns `Some(value)` if its checksum matches.
    fn decode(value_bits: u32, word: u64) -> Option<u64> {
        let value_mask = if value_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << value_bits) - 1
        };
        let lo = word & value_mask;
        let expected = Self::encode(value_bits, lo);
        if expected == word { Some(lo) } else { None }
    }

    /// Build a Bloomier filter for the given `(key, value)` pairs.
    ///
    /// * `m_factor` sets the table size `m = ceil(m_factor · n)` (≥ ~1.23 for
    ///   `k = 3` peelability).
    /// * `k` is the number of probes (3 is the classic choice).
    /// * `value_bits` is how many bits each value uses (`value < 2^value_bits`).
    /// * `seed_base` seeds the hash family; vary it across retries.
    ///
    /// Returns `Err(NotConverged)` if the random hypergraph was not peelable for
    /// this seed (retry with another seed), or `InvalidParameter` on bad inputs.
    pub fn build(
        pairs: &[(u64, u64)],
        m_factor: f64,
        k: usize,
        value_bits: u32,
        seed_base: u64,
    ) -> SketchResult<Self> {
        if pairs.is_empty() {
            return Err(SketchError::EmptyStream);
        }
        if k < 2 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be >= 2".to_string(),
            });
        }
        if value_bits == 0 || value_bits > 56 {
            return Err(SketchError::InvalidParameter {
                name: "value_bits".to_string(),
                reason: "must be in 1..=56 (leaving room for a checksum)".to_string(),
            });
        }
        if !(m_factor.is_finite() && m_factor > 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "m_factor".to_string(),
                reason: "must be > 1".to_string(),
            });
        }
        let value_cap = 1u64 << value_bits;
        for &(_, v) in pairs {
            if v >= value_cap {
                return Err(SketchError::InvalidParameter {
                    name: "value".to_string(),
                    reason: format!("value {v} does not fit in {value_bits} bits"),
                });
            }
        }
        // Reject duplicate keys (a static function must be single-valued).
        {
            let mut keys: Vec<u64> = pairs.iter().map(|&(kk, _)| kk).collect();
            keys.sort_unstable();
            if keys.windows(2).any(|w| w[0] == w[1]) {
                return Err(SketchError::InvalidParameter {
                    name: "keys".to_string(),
                    reason: "duplicate keys are not allowed".to_string(),
                });
            }
        }

        let n = pairs.len();
        let m = ((m_factor * n as f64).ceil() as usize).max(k);

        // Pre-compute each key's probe locations.
        let locs: Vec<Vec<usize>> = pairs
            .iter()
            .map(|&(kk, _)| Self::locations(m, k, seed_base, kk))
            .collect();

        // Greedy peeling: cell → count of live keys touching it, and an XOR of
        // their indices so a singleton cell reveals its unique owner.
        let mut cell_count = vec![0u32; m];
        let mut cell_xor = vec![0usize; m];
        for (idx, loc) in locs.iter().enumerate() {
            // A key may probe the same cell twice; only distinct cells count.
            let mut distinct = loc.clone();
            distinct.sort_unstable();
            distinct.dedup();
            for &c in &distinct {
                cell_count[c] += 1;
                cell_xor[c] ^= idx;
            }
        }

        let mut placed = vec![false; n];
        // Order of (key index, private cell) discovered by peeling.
        let mut order: Vec<(usize, usize)> = Vec::with_capacity(n);

        // Queue of currently-singleton cells.
        let mut queue: Vec<usize> = (0..m).filter(|&c| cell_count[c] == 1).collect();

        while let Some(c) = queue.pop() {
            if cell_count[c] != 1 {
                continue;
            }
            let key_idx = cell_xor[c];
            if placed[key_idx] {
                continue;
            }
            placed[key_idx] = true;
            order.push((key_idx, c));

            // Remove this key from all its cells; newly-singleton cells re-queue.
            let mut distinct = locs[key_idx].clone();
            distinct.sort_unstable();
            distinct.dedup();
            for &cc in &distinct {
                cell_count[cc] -= 1;
                cell_xor[cc] ^= key_idx;
                if cell_count[cc] == 1 {
                    queue.push(cc);
                }
            }
        }

        if order.len() != n {
            // Not peelable for this seed.
            return Err(SketchError::NotConverged { iter: order.len() });
        }

        // Assign the table in reverse peel order.
        let mut table = vec![0u64; m];
        for &(key_idx, cell) in order.iter().rev() {
            let (_, value) = pairs[key_idx];
            let encoded = Self::encode(value_bits, value);
            let mask = Self::value_mask(seed_base, pairs[key_idx].0);
            // XOR of the other probed cells (excluding `cell`, counted once).
            let mut acc = 0u64;
            let mut seen_private = false;
            for &c in &locs[key_idx] {
                if c == cell && !seen_private {
                    seen_private = true;
                    continue;
                }
                acc ^= table[c];
            }
            table[cell] = encoded ^ mask ^ acc;
        }

        Ok(Self {
            m,
            k,
            value_bits,
            seed_base,
            table,
        })
    }

    /// Look up the value associated with `key`.
    ///
    /// Returns `Some(value)` if `key` is (very likely) in the support, or `None`
    /// if it is not present (or, with small probability, a false "absent" never
    /// happens for stored keys — stored keys always decode correctly).
    #[must_use]
    pub fn get(&self, key: u64) -> Option<u64> {
        let locs = Self::locations(self.m, self.k, self.seed_base, key);
        let mask = Self::value_mask(self.seed_base, key);
        // Recover `r = M(x) XOR ⊕_j table[h_j(x)]` over every probe occurrence,
        // exactly mirroring the construction's XOR accumulation.
        let mut acc = mask;
        for &c in &locs {
            acc ^= self.table[c];
        }
        Self::decode(self.value_bits, acc)
    }

    /// Whether `key` is (probably) a member of the support.
    #[must_use]
    pub fn contains(&self, key: u64) -> bool {
        self.get(key).is_some()
    }

    /// Number of table cells.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.m
    }

    /// Number of probes per key.
    #[must_use]
    pub fn probes(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build with seed-retry so peeling almost surely succeeds for the tests.
    fn build_retry(pairs: &[(u64, u64)], value_bits: u32) -> BloomierFilter {
        for s in 0..64u64 {
            if let Ok(bf) = BloomierFilter::build(pairs, 1.3, 3, value_bits, 1000 + s) {
                return bf;
            }
        }
        panic!("peeling failed for all seeds");
    }

    #[test]
    fn stored_keys_return_correct_values() {
        let pairs: Vec<(u64, u64)> = (0..200u64).map(|i| (i * 7 + 3, i % 100)).collect();
        let bf = build_retry(&pairs, 8);
        for &(k, v) in &pairs {
            assert_eq!(bf.get(k), Some(v), "key {k} should map to {v}");
        }
    }

    #[test]
    fn stored_keys_are_members() {
        let pairs: Vec<(u64, u64)> = (0..150u64).map(|i| (i * 13, i % 50)).collect();
        let bf = build_retry(&pairs, 8);
        for &(k, _) in &pairs {
            assert!(bf.contains(k), "key {k} must be a member");
        }
    }

    #[test]
    fn absent_keys_mostly_return_none() {
        let pairs: Vec<(u64, u64)> = (0..300u64).map(|i| (i * 11 + 1, i % 64)).collect();
        let bf = build_retry(&pairs, 6);
        // Query far-away keys not in support.
        let mut false_positives = 0usize;
        let total = 5_000usize;
        for q in 1_000_000u64..(1_000_000 + total as u64) {
            if bf.get(q).is_some() {
                false_positives += 1;
            }
        }
        // With ~10 checksum bits the spurious-value rate is well under 5%.
        let rate = false_positives as f64 / total as f64;
        assert!(rate < 0.05, "spurious rate {rate} too high");
    }

    #[test]
    fn single_pair_roundtrips() {
        let bf = build_retry(&[(42, 7)], 8);
        assert_eq!(bf.get(42), Some(7));
    }

    #[test]
    fn values_can_be_zero() {
        let pairs: Vec<(u64, u64)> = (0..50u64).map(|i| (i * 3 + 1, 0)).collect();
        let bf = build_retry(&pairs, 8);
        for &(k, _) in &pairs {
            assert_eq!(bf.get(k), Some(0));
        }
    }

    #[test]
    fn distinct_values_preserved() {
        let pairs = vec![(1u64, 11u64), (2, 22), (3, 33), (4, 44), (5, 55)];
        let bf = build_retry(&pairs, 8);
        for &(k, v) in &pairs {
            assert_eq!(bf.get(k), Some(v));
        }
    }

    #[test]
    fn rejects_empty() {
        let res = BloomierFilter::build(&[], 1.3, 3, 8, 0);
        assert!(matches!(res, Err(SketchError::EmptyStream)));
    }

    #[test]
    fn rejects_value_overflow() {
        // value_bits = 4 → values must be < 16.
        let res = BloomierFilter::build(&[(1u64, 100u64)], 1.3, 3, 4, 0);
        assert!(matches!(res, Err(SketchError::InvalidParameter { .. })));
    }

    #[test]
    fn rejects_duplicate_keys() {
        let res = BloomierFilter::build(&[(1u64, 2u64), (1, 3)], 1.3, 3, 8, 0);
        assert!(matches!(res, Err(SketchError::InvalidParameter { .. })));
    }

    #[test]
    fn rejects_bad_k() {
        let res = BloomierFilter::build(&[(1u64, 2u64)], 1.3, 1, 8, 0);
        assert!(matches!(res, Err(SketchError::InvalidParameter { .. })));
    }

    #[test]
    fn rejects_bad_m_factor() {
        let res = BloomierFilter::build(&[(1u64, 2u64)], 0.5, 3, 8, 0);
        assert!(matches!(res, Err(SketchError::InvalidParameter { .. })));
    }

    #[test]
    fn capacity_scales_with_m_factor() {
        let pairs: Vec<(u64, u64)> = (0..100u64).map(|i| (i + 1, i % 10)).collect();
        // Try a few seeds to get a successful build.
        let mut built = None;
        for s in 0..64u64 {
            if let Ok(bf) = BloomierFilter::build(&pairs, 1.5, 3, 8, s) {
                built = Some(bf);
                break;
            }
        }
        let bf = built.expect("build");
        assert!(bf.capacity() >= 150, "capacity {} too small", bf.capacity());
        assert_eq!(bf.probes(), 3);
    }

    #[test]
    fn large_value_bits_roundtrip() {
        let pairs: Vec<(u64, u64)> = (0..80u64).map(|i| (i * 5 + 1, i * 1000)).collect();
        let bf = build_retry(&pairs, 24);
        for &(k, v) in &pairs {
            assert_eq!(bf.get(k), Some(v));
        }
    }
}

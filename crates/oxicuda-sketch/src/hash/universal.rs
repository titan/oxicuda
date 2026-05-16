//! Universal-style hash helpers built on multiplicative mixing.
//!
//! `hash_u64_to_range(x, m, seed)` returns a uniform-ish integer in `[0, m)` derived from
//! `x` using a high-quality multiplicative mixer (Knuth) plus avalanche from xxh3-min.

use crate::hash::xxh3_min::xxh3_64_u64;

/// Map a `u64` deterministically into `[0, m)` using xxh3-min with the supplied seed.
#[must_use]
pub fn hash_u64_to_range(x: u64, m: u64, seed: u64) -> u64 {
    let h = xxh3_64_u64(x, seed);
    h % m.max(1)
}

/// Map a byte slice into `[0, m)` using xxh3-min.
#[must_use]
pub fn hash_bytes_to_range(bytes: &[u8], m: u64, seed: u64) -> u64 {
    let h = crate::hash::xxh3_min::xxh3_64(bytes, seed);
    h % m.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_in_range() {
        for x in 0..1000u64 {
            let r = hash_u64_to_range(x, 16, 0);
            assert!(r < 16);
        }
    }

    #[test]
    fn hash_bytes_in_range() {
        for i in 0..100u32 {
            let r = hash_bytes_to_range(&i.to_le_bytes(), 7, 11);
            assert!(r < 7);
        }
    }
}

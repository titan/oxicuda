//! Murmur3-32 hash implementation (Austin Appleby, public-domain reference).
//!
//! Pure Rust, no external dependencies. Produces a 32-bit hash from arbitrary bytes.

const C1: u32 = 0xcc9e_2d51;
const C2: u32 = 0x1b87_3593;
const R1: u32 = 15;
const R2: u32 = 13;
const M: u32 = 5;
const N: u32 = 0xe654_6b64;

/// Compute a Murmur3-32 hash over a byte slice with the given 32-bit seed.
#[must_use]
pub fn murmur3_32_bytes(bytes: &[u8], seed: u32) -> u32 {
    let mut h = seed;
    let nblocks = bytes.len() / 4;

    // Body: process 4-byte blocks.
    for i in 0..nblocks {
        let off = i * 4;
        let mut k =
            u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        k = k.wrapping_mul(C1);
        k = k.rotate_left(R1);
        k = k.wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(R2);
        h = h.wrapping_mul(M).wrapping_add(N);
    }

    // Tail
    let tail_off = nblocks * 4;
    let tail_len = bytes.len() - tail_off;
    let mut k1: u32 = 0;
    if tail_len >= 3 {
        k1 ^= (bytes[tail_off + 2] as u32) << 16;
    }
    if tail_len >= 2 {
        k1 ^= (bytes[tail_off + 1] as u32) << 8;
    }
    if tail_len >= 1 {
        k1 ^= bytes[tail_off] as u32;
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(R1);
        k1 = k1.wrapping_mul(C2);
        h ^= k1;
    }

    // Finalize
    h ^= bytes.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// Compute a Murmur3-32 hash of a `u64` (encoded little-endian).
#[must_use]
pub fn murmur3_32(value: u64, seed: u32) -> u32 {
    murmur3_32_bytes(&value.to_le_bytes(), seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn murmur3_known_empty() {
        // Reference: murmur3_32(b"", 0) = 0
        assert_eq!(murmur3_32_bytes(b"", 0), 0);
    }

    #[test]
    fn murmur3_consistent() {
        let h1 = murmur3_32_bytes(b"hello", 42);
        let h2 = murmur3_32_bytes(b"hello", 42);
        assert_eq!(h1, h2);
    }

    #[test]
    fn murmur3_different_seeds() {
        let h1 = murmur3_32_bytes(b"world", 1);
        let h2 = murmur3_32_bytes(b"world", 2);
        assert_ne!(h1, h2);
    }

    #[test]
    fn murmur3_u64_diff_inputs() {
        let h1 = murmur3_32(42, 0);
        let h2 = murmur3_32(43, 0);
        assert_ne!(h1, h2);
    }
}

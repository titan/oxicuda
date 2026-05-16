//! Simplified 64-bit xxHash3-like hash (NOT bit-exact with reference xxh3-64).
//!
//! Pure Rust, suitable as a fast multiplicative mixer for stream sketches.
//! The construction mixes the input through three rounds of multiplications + shifts
//! using the original xxHash 64-bit primes.

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline]
fn rotl(x: u64, n: u32) -> u64 {
    x.rotate_left(n)
}

fn mix(mut a: u64) -> u64 {
    a = a.wrapping_mul(P2);
    a = rotl(a, 31);
    a = a.wrapping_mul(P1);
    a
}

/// Compute a simplified 64-bit hash over a byte slice with a 64-bit seed.
#[must_use]
pub fn xxh3_64(bytes: &[u8], seed: u64) -> u64 {
    let mut acc = seed.wrapping_add(P5).wrapping_add(bytes.len() as u64);

    // Process 8-byte blocks.
    let mut i = 0;
    while i + 8 <= bytes.len() {
        let chunk = u64::from_le_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
            bytes[i + 4],
            bytes[i + 5],
            bytes[i + 6],
            bytes[i + 7],
        ]);
        acc ^= mix(chunk);
        acc = rotl(acc, 27).wrapping_mul(P1).wrapping_add(P4);
        i += 8;
    }

    // 4-byte tail
    if i + 4 <= bytes.len() {
        let chunk = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as u64;
        acc ^= chunk.wrapping_mul(P1);
        acc = rotl(acc, 23).wrapping_mul(P2).wrapping_add(P3);
        i += 4;
    }

    // Remaining bytes
    while i < bytes.len() {
        let chunk = bytes[i] as u64;
        acc ^= chunk.wrapping_mul(P5);
        acc = rotl(acc, 11).wrapping_mul(P1);
        i += 1;
    }

    // Final avalanche
    acc ^= acc >> 33;
    acc = acc.wrapping_mul(P2);
    acc ^= acc >> 29;
    acc = acc.wrapping_mul(P3);
    acc ^= acc >> 32;
    acc
}

/// Hash a `u64` value with a 64-bit seed.
#[must_use]
pub fn xxh3_64_u64(value: u64, seed: u64) -> u64 {
    xxh3_64(&value.to_le_bytes(), seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxh3_empty_nonzero() {
        let h = xxh3_64(b"", 0);
        // Empty input mixes seed+P5 then avalanches => non-zero.
        assert_ne!(h, 0);
    }

    #[test]
    fn xxh3_seed_changes_output() {
        let h1 = xxh3_64(b"hello", 0);
        let h2 = xxh3_64(b"hello", 1);
        assert_ne!(h1, h2);
    }

    #[test]
    fn xxh3_diff_input() {
        let h1 = xxh3_64(b"hello", 0);
        let h2 = xxh3_64(b"world", 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn xxh3_consistency() {
        let h1 = xxh3_64(b"abcdefgh12345", 7);
        let h2 = xxh3_64(b"abcdefgh12345", 7);
        assert_eq!(h1, h2);
    }
}

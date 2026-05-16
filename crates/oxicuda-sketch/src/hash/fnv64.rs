//! FNV-1a 64-bit hash implementation (Fowler-Noll-Vo).
//!
//! Reference constants: offset_basis = 0xcbf29ce484222325, prime = 0x100000001b3.

const FNV_OFFSET_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;

/// Compute a 64-bit FNV-1a hash over a byte slice.
#[must_use]
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET_64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME_64);
    }
    h
}

/// Compute a 64-bit FNV-1a hash of a `u64` (little-endian bytes).
#[must_use]
pub fn fnv1a_64_u64(value: u64) -> u64 {
    fnv1a_64(&value.to_le_bytes())
}

/// Seeded FNV-1a 64-bit by XOR-folding the seed into the offset basis.
#[must_use]
pub fn fnv1a_64_seeded(bytes: &[u8], seed: u64) -> u64 {
    let mut h = FNV_OFFSET_64 ^ seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME_64);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv64_known_empty() {
        // FNV-1a of empty string is the offset basis.
        assert_eq!(fnv1a_64(b""), FNV_OFFSET_64);
    }

    #[test]
    fn fnv64_avalanche_one_byte() {
        let h1 = fnv1a_64(b"a");
        let h2 = fnv1a_64(b"b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn fnv64_seeded_changes_output() {
        let h1 = fnv1a_64_seeded(b"x", 0);
        let h2 = fnv1a_64_seeded(b"x", 1);
        assert_ne!(h1, h2);
    }
}

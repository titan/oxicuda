//! Tabulation hashing: an extremely strong (4-wise independent) hash family.
//!
//! For each byte position, precompute a 256-entry random table; the hash of a multi-byte
//! key is the XOR of `table[i][byte_i(key)]`. See Patrascu-Thorup 2012.

use crate::handle::LcgRng;

/// 8-byte tabulation hash: 8 tables × 256 entries × u64.
#[derive(Debug, Clone)]
pub struct TabulationHash {
    pub tables: [[u64; 256]; 8],
}

impl TabulationHash {
    /// Build a new random tabulation hash from an RNG.
    #[must_use]
    pub fn new(rng: &mut LcgRng) -> Self {
        let mut tables = [[0u64; 256]; 8];
        for table in tables.iter_mut() {
            for entry in table.iter_mut() {
                *entry = rng.next_u64();
            }
        }
        Self { tables }
    }

    /// Hash a `u64` to a `u64`.
    #[must_use]
    pub fn hash(&self, x: u64) -> u64 {
        let bytes = x.to_le_bytes();
        let mut h = 0u64;
        for (i, &b) in bytes.iter().enumerate() {
            h ^= self.tables[i][b as usize];
        }
        h
    }

    /// Hash bytes by chunking 8 bytes at a time, XOR-folding tail.
    #[must_use]
    pub fn hash_bytes(&self, bytes: &[u8]) -> u64 {
        let mut h = 0u64;
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
            h ^= self.hash(chunk);
            i += 8;
        }
        // Tail: pad with zeros into a u64 then hash and XOR-fold.
        if i < bytes.len() {
            let mut tail = [0u8; 8];
            let rem = bytes.len() - i;
            tail[..rem].copy_from_slice(&bytes[i..]);
            h ^= self.hash(u64::from_le_bytes(tail));
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabulation_deterministic() {
        let mut rng = LcgRng::new(11);
        let h = TabulationHash::new(&mut rng);
        let v1 = h.hash(42);
        let v2 = h.hash(42);
        assert_eq!(v1, v2);
    }

    #[test]
    fn tabulation_different_inputs() {
        let mut rng = LcgRng::new(11);
        let h = TabulationHash::new(&mut rng);
        assert_ne!(h.hash(0), h.hash(1));
        assert_ne!(h.hash(100), h.hash(101));
    }

    #[test]
    fn tabulation_bytes_consistent() {
        let mut rng = LcgRng::new(11);
        let h = TabulationHash::new(&mut rng);
        let h1 = h.hash_bytes(b"hello world");
        let h2 = h.hash_bytes(b"hello world");
        assert_eq!(h1, h2);
    }
}

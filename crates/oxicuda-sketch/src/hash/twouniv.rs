//! 2-universal hash family using the Carter-Wegman construction.
//!
//! `h_{a,b}(x) = ((a * x + b) mod p) mod m` where `p = 2^61 - 1` (Mersenne prime),
//! `a` drawn from `[1, p-1]`, `b` drawn from `[0, p-1]`.

use crate::handle::LcgRng;

/// Mersenne prime 2^61 - 1.
pub const PRIME_MERSENNE_61: u64 = (1u64 << 61) - 1;

/// One row of a 2-universal hash family with output range `[0, m)`.
#[derive(Debug, Clone)]
pub struct TwoUniversal {
    pub a: u64,
    pub b: u64,
    pub m: u64,
}

impl TwoUniversal {
    /// Construct a new random 2-universal hash with output range `[0, m)`.
    #[must_use]
    pub fn new(rng: &mut LcgRng, m: u64) -> Self {
        // a is in [1, p-1], b is in [0, p-1].
        let a = 1 + rng.next_u64() % (PRIME_MERSENNE_61 - 1);
        let b = rng.next_u64() % PRIME_MERSENNE_61;
        Self { a, b, m: m.max(1) }
    }

    /// Construct with explicit `a` and `b` (clamped to valid range). Useful for tests.
    #[must_use]
    pub fn with_coeffs(a: u64, b: u64, m: u64) -> Self {
        let a = (a % (PRIME_MERSENNE_61 - 1)).max(1);
        let b = b % PRIME_MERSENNE_61;
        Self { a, b, m: m.max(1) }
    }

    /// Evaluate the hash on a `u64` input.
    #[must_use]
    pub fn hash(&self, x: u64) -> u64 {
        // Compute (a*x + b) mod p using 128-bit intermediates.
        let prod = (self.a as u128).wrapping_mul(x as u128);
        let sum = prod + self.b as u128;
        // Fast modulo for Mersenne prime: (low_61 + high) modded again.
        let p = PRIME_MERSENNE_61 as u128;
        let r = (sum & p) + (sum >> 61);
        let r = if r >= p { r - p } else { r };
        (r as u64) % self.m
    }

    /// Build `d` independent hash functions.
    #[must_use]
    pub fn many(rng: &mut LcgRng, d: usize, m: u64) -> Vec<TwoUniversal> {
        (0..d).map(|_| TwoUniversal::new(rng, m)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twouniv_in_range() {
        let mut rng = LcgRng::new(11);
        let h = TwoUniversal::new(&mut rng, 1024);
        for x in 0..1000u64 {
            assert!(h.hash(x) < 1024);
        }
    }

    #[test]
    fn twouniv_deterministic() {
        let mut rng = LcgRng::new(11);
        let h = TwoUniversal::new(&mut rng, 1024);
        let v1 = h.hash(42);
        let v2 = h.hash(42);
        assert_eq!(v1, v2);
    }

    #[test]
    fn twouniv_balanced_buckets() {
        // Check distribution: M=8 buckets, hash 8000 distinct keys, no bucket should be too imbalanced.
        let mut rng = LcgRng::new(7);
        let h = TwoUniversal::new(&mut rng, 8);
        let mut counts = [0usize; 8];
        for x in 0..8000u64 {
            counts[h.hash(x) as usize] += 1;
        }
        // Allow generous slack; each bucket expected count = 1000.
        for &c in &counts {
            assert!(c > 700 && c < 1300, "bucket count {c} out of bounds");
        }
    }

    #[test]
    fn twouniv_explicit_coeffs() {
        let h = TwoUniversal::with_coeffs(3, 7, 100);
        // (3 * 5 + 7) mod p = 22, mod 100 = 22
        assert_eq!(h.hash(5), 22);
    }
}

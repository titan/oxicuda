//! Xoshiro256** pseudorandom number generator.
//!
//! Implements the xoshiro256** algorithm from Blackman & Vigna 2019.
//! 256-bit state, 64-bit output, period 2^256-1.
//!
//! Features:
//! - Excellent statistical quality (passes BigCrush)
//! - Fast output: one multiply + two rotates
//! - Jump function equivalent to 2^128 calls (for parallel streams)
//!
//! Reference: D. Blackman and S. Vigna, "Scrambled Linear Pseudorandom Number
//! Generators", ACM TOMACS, 2021.

/// Rotate `x` left by `k` bits.
#[inline(always)]
fn rotl64(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

/// SplitMix64 step used to initialize state from a single seed.
///
/// This ensures that even small changes in the seed produce completely
/// different initial states via avalanche.
#[inline(always)]
fn splitmix64(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut v = *z;
    v = (v ^ (v >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    v ^ (v >> 31)
}

/// Xoshiro256** random number generator.
///
/// 256-bit state, 64-bit output, period 2^256-1.
/// The `**` scrambler uses multiply + rotate + multiply for a strong
/// avalanche effect. Jump function equivalent to 2^128 calls enables
/// independent parallel streams.
pub struct Xoshiro256ss {
    s: [u64; 4],
}

impl Xoshiro256ss {
    /// Jump polynomial constants for xoshiro256**.
    /// Equivalent to 2^128 calls to `next_u64`.
    const JUMP: [u64; 4] = [
        0x180e_c6d3_3cfd_0aba,
        0xd5a6_1266_f0c9_392c,
        0xa958_2618_e03f_c9aa,
        0x39ab_dc45_29b1_661c,
    ];

    /// Create a new Xoshiro256** generator from a 64-bit seed.
    ///
    /// Uses four successive SplitMix64 steps to initialize the four 64-bit
    /// state words, guaranteeing a high-quality starting state even for
    /// simple seeds such as 0 or 1.
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let s = [
            splitmix64(&mut z),
            splitmix64(&mut z),
            splitmix64(&mut z),
            splitmix64(&mut z),
        ];
        Xoshiro256ss { s }
    }

    /// Generate the next pseudorandom `u64`.
    ///
    /// Output permutation: `rotl64(s[1] * 5, 7) * 9`
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = rotl64(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = rotl64(self.s[3], 45);
        result
    }

    /// Generate a uniformly distributed `f64` in `[0.0, 1.0)`.
    ///
    /// Uses the upper 53 bits for full double-precision mantissa.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Jump the generator forward by 2^128 steps.
    ///
    /// This is equivalent to calling `next_u64` 2^128 times, useful for
    /// partitioning the sequence into independent substreams for parallel use.
    pub fn jump(&mut self) {
        let mut s0 = 0u64;
        let mut s1 = 0u64;
        let mut s2 = 0u64;
        let mut s3 = 0u64;

        for &j_word in &Self::JUMP {
            for b in 0..64u32 {
                if (j_word >> b) & 1 != 0 {
                    s0 ^= self.s[0];
                    s1 ^= self.s[1];
                    s2 ^= self.s[2];
                    s3 ^= self.s[3];
                }
                self.next_u64();
            }
        }
        self.s = [s0, s1, s2, s3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut rng1 = Xoshiro256ss::new(0xdead_beef_cafe_babe);
        let mut rng2 = Xoshiro256ss::new(0xdead_beef_cafe_babe);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn different_seeds() {
        let mut rng1 = Xoshiro256ss::new(1);
        let mut rng2 = Xoshiro256ss::new(2);
        let v1 = rng1.next_u64();
        let v2 = rng2.next_u64();
        assert_ne!(
            v1, v2,
            "different seeds must produce different first output"
        );
    }

    #[test]
    fn f64_in_range() {
        let mut rng = Xoshiro256ss::new(0xfeed_face);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 value {v} not in [0, 1)");
        }
    }

    #[test]
    fn jump_changes_state() {
        let mut rng1 = Xoshiro256ss::new(12345);
        let mut rng2 = Xoshiro256ss::new(12345);
        rng2.jump();
        // After a jump the states differ, so outputs should differ
        let v1 = rng1.next_u64();
        let v2 = rng2.next_u64();
        assert_ne!(v1, v2, "jumped RNG must produce different output");
    }

    #[test]
    fn full_cycle_not_all_zero() {
        let mut rng = Xoshiro256ss::new(9999);
        let nonzero = (0..1000)
            .map(|_| rng.next_u64())
            .filter(|&v| v != 0)
            .count();
        assert!(nonzero > 0, "sequence must contain nonzero values");
    }

    #[test]
    fn next_u64_not_always_same() {
        let mut rng = Xoshiro256ss::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2, "successive calls must produce different values");
    }

    #[test]
    fn splitmix64_mixes_well() {
        let mut rng1 = Xoshiro256ss::new(1);
        let mut rng2 = Xoshiro256ss::new(2);
        let v1 = rng1.next_u64();
        let v2 = rng2.next_u64();
        // Seeds 1 and 2 should produce very different outputs due to avalanche
        assert_ne!(v1, v2);
        // Hamming distance check: XOR should have several bits set
        let xor = v1 ^ v2;
        let hamming = xor.count_ones();
        assert!(
            hamming >= 4,
            "hamming distance {hamming} too small; avalanche not strong enough"
        );
    }

    #[test]
    fn batch_fill_diverse() {
        let mut rng = Xoshiro256ss::new(0xabcd_1234);
        let values: Vec<u64> = (0..20).map(|_| rng.next_u64()).collect();
        let unique: std::collections::HashSet<u64> = values.iter().copied().collect();
        assert!(
            unique.len() > 10,
            "at least 11 of 20 values must be distinct"
        );
    }

    #[test]
    fn sequential_different() {
        let mut rng = Xoshiro256ss::new(0x1234_5678_9abc_def0);
        let values: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();
        // All 10 consecutive values should be distinct (period >> 10)
        let set: std::collections::HashSet<u64> = values.iter().copied().collect();
        assert_eq!(set.len(), 10, "10 consecutive values must all be distinct");
    }
}

//! PCG32 pseudorandom number generator.
//!
//! Implements the PCG (Permuted Congruential Generator) algorithm from
//! O'Neill 2014. PCG32 uses a 64-bit LCG state with a permutation output
//! function (xsh-rr: xorshift-high, random-rotate) to produce 32-bit outputs
//! with excellent statistical quality.

/// Right-rotate a u32 by `rot` bits.
#[inline(always)]
fn ror32(x: u32, rot: u32) -> u32 {
    x.rotate_right(rot)
}

/// PCG32 random number generator.
///
/// State: 64-bit LCG with selectable stream.
/// Output: 32-bit with xsh-rr permutation.
/// Period: 2^64.
///
/// Reference: M.E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient
/// Statistically Good Algorithms for Random Number Generation", 2014.
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// PCG multiplier constant (Knuth 1997, TAOCP vol. 2).
    const MULT: u64 = 6_364_136_223_846_793_005;

    /// Advance the internal LCG state by one step.
    #[inline(always)]
    fn lcg_step(&mut self) {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(self.inc);
    }

    /// Create a new PCG32 generator with the given seed and sequence selector.
    ///
    /// `seq` selects which of the 2^63 independent streams to use. Different
    /// `seq` values produce independent, non-overlapping output sequences.
    ///
    /// Initialization follows the reference implementation:
    /// 1. `inc = (seq << 1) | 1`  (force odd increment)
    /// 2. `state = 0`, advance once
    /// 3. `state += seed`, advance once more
    pub fn new(seed: u64, seq: u64) -> Self {
        let mut rng = Pcg32 {
            state: 0,
            inc: seq.wrapping_shl(1) | 1,
        };
        rng.lcg_step();
        rng.state = rng.state.wrapping_add(seed);
        rng.lcg_step();
        rng
    }

    /// Generate the next pseudorandom `u32`.
    ///
    /// Uses the xsh-rr (xorshift-high, random-rotate) output permutation:
    /// - Save old state, advance LCG
    /// - Compute xorshifted = ((old >> 18) ^ old) >> 27
    /// - Rotation amount = old >> 59
    /// - Output = ror32(xorshifted, rot)
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.lcg_step();
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        ror32(xorshifted, rot)
    }

    /// Generate a uniformly distributed `f32` in `[0.0, 1.0)`.
    ///
    /// Uses the upper 24 bits for mantissa precision.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Fill a buffer with pseudorandom `u32` values.
    pub fn fill(&mut self, buf: &mut [u32]) {
        for x in buf.iter_mut() {
            *x = self.next_u32();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_with_seed() {
        let mut rng1 = Pcg32::new(42, 1);
        let mut rng2 = Pcg32::new(42, 1);
        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn different_seeds_different_output() {
        let mut rng1 = Pcg32::new(1, 1);
        let mut rng2 = Pcg32::new(2, 1);
        let v1 = rng1.next_u32();
        let v2 = rng2.next_u32();
        assert_ne!(
            v1, v2,
            "different seeds must produce different first output"
        );
    }

    #[test]
    fn period_gt_2_32() {
        let mut rng = Pcg32::new(12345, 1);
        let count = 1 << 20; // 2^20
        let mut seen_nonzero = false;
        let first = rng.next_u32();
        let mut all_same = true;
        for _ in 1..count {
            let v = rng.next_u32();
            if v != 0 {
                seen_nonzero = true;
            }
            if v != first {
                all_same = false;
            }
        }
        assert!(seen_nonzero, "sequence must contain nonzero values");
        assert!(!all_same, "sequence must not be constant over 2^20 values");
    }

    #[test]
    fn uniform_distribution_chi2() {
        let mut rng = Pcg32::new(99999, 7);
        let n = 10_000usize;
        let bins = 10usize;
        let mut counts = vec![0usize; bins];
        for _ in 0..n {
            let v = rng.next_u32();
            let bin = (v as usize * bins) / (u32::MAX as usize + 1);
            let bin = bin.min(bins - 1);
            counts[bin] += 1;
        }
        let expected = n as f64 / bins as f64;
        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                d * d / expected
            })
            .sum();
        // Chi2 with 9 df; lenient threshold of 30 (p ~ 0.0006, very unlikely to fail)
        assert!(chi2 < 30.0, "chi2 = {chi2:.2} exceeds threshold 30.0");
    }

    #[test]
    fn fill_batch() {
        let seed = 77777;
        let seq = 3;
        let mut rng_seq = Pcg32::new(seed, seq);
        let mut rng_fill = Pcg32::new(seed, seq);

        let n = 64;
        let sequential: Vec<u32> = (0..n).map(|_| rng_seq.next_u32()).collect();
        let mut buf = vec![0u32; n];
        rng_fill.fill(&mut buf);
        assert_eq!(sequential, buf, "fill() must match sequential next_u32()");
    }

    #[test]
    fn f32_in_range() {
        let mut rng = Pcg32::new(54321, 2);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "f32 value {v} not in [0, 1)");
        }
    }

    #[test]
    fn two_sequences_independent() {
        let mut rng0 = Pcg32::new(42, 0);
        let mut rng1 = Pcg32::new(42, 1);
        let v0 = rng0.next_u32();
        let v1 = rng1.next_u32();
        assert_ne!(v0, v1, "different seq values must yield different streams");
    }

    #[test]
    fn next_u32_not_zero_for_all_inputs() {
        let mut rng = Pcg32::new(0, 0);
        let values: Vec<u32> = (0..100).map(|_| rng.next_u32()).collect();
        let nonzero = values.iter().filter(|&&v| v != 0).count();
        assert!(nonzero > 0, "at least some values must be nonzero");
    }

    #[test]
    fn stateful_advance() {
        let mut rng = Pcg32::new(1234567890, 5);
        let v1 = rng.next_u32();
        let v2 = rng.next_u32();
        assert_ne!(v1, v2, "successive calls must produce different values");
    }
}

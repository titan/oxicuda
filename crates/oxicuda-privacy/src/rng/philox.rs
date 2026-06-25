//! Philox 4×32-10 counter-based RNG (Random123).
//!
//! Reference: Salmon JK, Moraes MA, Dror RO, Shaw DE (2011), "Parallel Random
//! Numbers: As Easy as 1, 2, 3", SC '11 (Random123 library) — the canonical
//! `philox4x32` generator with the standard 10 rounds.
//!
//! # Why counter-based?
//! A counter-based RNG (CBRNG) is a *stateless* keyed bijection `b(key, counter)`
//! whose output is a function of the (key, counter) pair only. This makes the
//! stream trivially seekable and reproducible: the `n`-th 128-bit output block is
//! `philox(key, n)` with no need to iterate through blocks `0 … n−1`. That is
//! exactly the property differential-privacy noise replay needs — a given
//! (key, counter) deterministically reproduces the same noise draw on any
//! machine, bit-for-bit, regardless of how many other draws happened. CBRNGs are
//! a CPU algorithm: there is no device-state dependency, only integer multiply /
//! xor / add.
//!
//! # Algorithm (philox4x32-10)
//! State is a 128-bit counter `(c0, c1, c2, c3)` (four `u32`) and a 64-bit key
//! `(k0, k1)`. Each of the 10 rounds applies, with the 32×32→64 multiplies
//!
//! ```text
//! (hi0, lo0) = mulhilo(0xD2511F53, c0)
//! (hi1, lo1) = mulhilo(0xCD9E8D57, c2)
//! c0' = hi1 ^ c1 ^ k0
//! c1' = lo1
//! c2' = hi0 ^ c3 ^ k1
//! c3' = lo0
//! ```
//!
//! and bumps the key by the Weyl constants `k0 += 0x9E3779B9`,
//! `k1 += 0xBB67AE85` between rounds. After 10 rounds the permuted counter is
//! the four-word random output.

/// Philox first multiplier constant `M0`.
const PHILOX_M4X32_0: u32 = 0xD251_1F53;
/// Philox second multiplier constant `M1`.
const PHILOX_M4X32_1: u32 = 0xCD9E_8D57;
/// Philox key Weyl increment `W0` (golden-ratio fractional bits).
const PHILOX_W32_0: u32 = 0x9E37_79B9;
/// Philox key Weyl increment `W1` (√3 − 1 fractional bits).
const PHILOX_W32_1: u32 = 0xBB67_AE85;
/// Standard round count for the cryptographically-strong `philox4x32` variant.
const PHILOX_ROUNDS: usize = 10;

/// Reciprocal of `2³²`, mapping a full-range `u32` into `[0, 1)`.
const INV_2_32: f64 = 1.0 / 4_294_967_296.0;

/// 32×32 → (hi, lo) widening multiply.
#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64) * (b as u64);
    ((product >> 32) as u32, product as u32)
}

/// One Philox round on the 128-bit counter given the (per-round) key words.
#[inline]
fn philox_round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(PHILOX_M4X32_0, ctr[0]);
    let (hi1, lo1) = mulhilo(PHILOX_M4X32_1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Evaluate the raw philox4x32-10 bijection for a (key, counter) pair.
///
/// Pure function: identical inputs always yield identical 128-bit (four-word)
/// output. This is the primitive behind both block generation and seeking.
#[must_use]
pub fn philox4x32_10(key: [u32; 2], counter: [u32; 4]) -> [u32; 4] {
    let mut ctr = counter;
    let mut key = key;
    for round in 0..PHILOX_ROUNDS {
        ctr = philox_round(ctr, key);
        // Bump the key for the *next* round (no bump needed after the last).
        if round + 1 < PHILOX_ROUNDS {
            key[0] = key[0].wrapping_add(PHILOX_W32_0);
            key[1] = key[1].wrapping_add(PHILOX_W32_1);
        }
    }
    ctr
}

/// Counter-based Philox 4×32-10 random number generator.
///
/// Produces a deterministic stream of `u32` (and derived `f32` / `f64` uniforms)
/// parallel to [`crate::handle::LcgRng`]. The stream is driven by a 64-bit key
/// and a 128-bit counter; the counter is treated as a little-endian 128-bit
/// integer that increments by one per consumed 128-bit (four-word) block. Each
/// block yields four `u32` outputs, buffered so callers can pull one word at a
/// time.
#[derive(Debug, Clone)]
pub struct PhiloxRng {
    key: [u32; 2],
    /// Little-endian 128-bit block counter (index of the *next* block to emit).
    block_counter: u128,
    /// The four words of the current block.
    buffer: [u32; 4],
    /// Index `0..4` of the next unconsumed word in `buffer` (`4` ⇒ empty).
    pos: usize,
}

impl PhiloxRng {
    /// Create a Philox RNG from a 64-bit seed (used as the key), starting at
    /// block counter 0.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self::with_key([seed as u32, (seed >> 32) as u32])
    }

    /// Create a Philox RNG from an explicit two-word key, starting at block 0.
    #[must_use]
    pub fn with_key(key: [u32; 2]) -> Self {
        Self {
            key,
            block_counter: 0,
            buffer: [0; 4],
            pos: 4, // force a refill on first draw
        }
    }

    /// Split the little-endian 128-bit block counter into four `u32` words.
    #[inline]
    fn counter_words(block: u128) -> [u32; 4] {
        [
            block as u32,
            (block >> 32) as u32,
            (block >> 64) as u32,
            (block >> 96) as u32,
        ]
    }

    /// Generate the block at the current `block_counter`, advance the counter,
    /// and reset the read position to the start of the freshly-filled buffer.
    #[inline]
    fn refill(&mut self) {
        let ctr = Self::counter_words(self.block_counter);
        self.buffer = philox4x32_10(self.key, ctr);
        self.block_counter = self.block_counter.wrapping_add(1);
        self.pos = 0;
    }

    /// Seek to absolute word index `word_index`: the next [`Self::next_u32`]
    /// returns the `word_index`-th word of the stream (word 0 being the first
    /// word of block 0). Reading sequentially from a freshly-seeked generator is
    /// identical to having consumed `word_index` words from the start.
    pub fn seek(&mut self, word_index: u128) {
        self.block_counter = word_index / 4;
        let within = (word_index % 4) as usize;
        // Materialise the target block, then position inside it.
        self.refill();
        self.pos = within;
    }

    /// Draw the next `u32` from the stream, refilling the 128-bit block buffer
    /// when it is exhausted.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= 4 {
            self.refill();
        }
        let v = self.buffer[self.pos];
        self.pos += 1;
        v
    }

    /// Draw the next `u64` by concatenating two consecutive `u32` words
    /// (low word first).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// Return an `f64` uniformly distributed in `[0, 1)` using the full 32-bit
    /// output range (÷2³²).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        f64::from(self.next_u32()) * INV_2_32
    }

    /// Return an `f32` uniformly distributed in `[0, 1)` (÷2³² then narrowed).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// Fill a slice with uniform `[0, 1)` `f64` draws.
    pub fn fill_uniform(&mut self, buf: &mut [f64]) {
        for v in buf.iter_mut() {
            *v = self.next_f64();
        }
    }

    /// A pair of standard-normal `N(0,1)` samples via the Box-Muller transform,
    /// matching [`crate::handle::LcgRng::normal_pair`] so the same downstream
    /// noise code can be driven by either generator.
    pub fn normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_f64().max(f64::EPSILON);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Known-answer test vectors from the Random123 reference suite
    //    (`kat_vectors`): philox4x32 with 10 rounds.
    #[test]
    fn philox_known_answer_vectors() {
        // key = {0,0}, ctr = {0,0,0,0}
        assert_eq!(
            philox4x32_10([0x0000_0000, 0x0000_0000], [0, 0, 0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        // key = {0xffffffff,0xffffffff}, ctr = {0xffffffff × 4}.
        // (First three words are the canonical Random123 KAT; the fourth,
        // 0x6d5451fd, is cross-checked against an independent reference
        // implementation of philox4x32-10.)
        assert_eq!(
            philox4x32_10(
                [0xffff_ffff, 0xffff_ffff],
                [0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff]
            ),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        // key = {0xa4093822, 0x299f31d0}, ctr = digits of π / e (Random123 KAT).
        assert_eq!(
            philox4x32_10(
                [0xa409_3822, 0x299f_31d0],
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344]
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    // 2. Same (key, counter) ⇒ identical stream, bit-for-bit.
    #[test]
    fn deterministic_same_key_counter() {
        let mut a = PhiloxRng::new(0xDEAD_BEEF_CAFE_F00D);
        let mut b = PhiloxRng::new(0xDEAD_BEEF_CAFE_F00D);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    // 3. Different keys ⇒ different streams (overwhelmingly).
    #[test]
    fn different_keys_differ() {
        let mut a = PhiloxRng::new(1);
        let mut b = PhiloxRng::new(2);
        let mut differences = 0;
        for _ in 0..256 {
            if a.next_u32() != b.next_u32() {
                differences += 1;
            }
        }
        assert!(differences > 250, "keys barely differ: {differences}/256");
    }

    // 4. Seeking to word N then reading equals reading N+ values sequentially.
    #[test]
    fn seek_equals_sequential() {
        let mut sequential = PhiloxRng::new(0x1234_5678);
        let mut consumed = Vec::new();
        for _ in 0..40 {
            consumed.push(sequential.next_u32());
        }
        // Seek to several offsets and confirm the next reads line up.
        for &offset in &[0u128, 1, 3, 4, 5, 7, 8, 17, 39] {
            let mut seeked = PhiloxRng::new(0x1234_5678);
            seeked.seek(offset);
            for (k, &want) in consumed.iter().enumerate().skip(offset as usize) {
                assert_eq!(
                    seeked.next_u32(),
                    want,
                    "seek({offset}) mismatch at word {k}"
                );
            }
        }
    }

    // 5. next_u64 concatenates two consecutive words (low first).
    #[test]
    fn u64_concatenates_words() {
        let mut words = PhiloxRng::new(7);
        let w0 = words.next_u32();
        let w1 = words.next_u32();
        let mut wide = PhiloxRng::new(7);
        let v = wide.next_u64();
        assert_eq!(v, ((w1 as u64) << 32) | (w0 as u64));
    }

    // 6. Uniform sanity: mean ≈ 0.5 and full [0,1) range via ÷2³².
    #[test]
    fn uniform_mean_and_range() {
        let mut rng = PhiloxRng::new(0xABCD_1234_5678_9F01);
        let n = 100_000;
        let mut sum = 0.0;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for _ in 0..n {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u), "out of [0,1): {u}");
            sum += u;
            lo = lo.min(u);
            hi = hi.max(u);
        }
        let mean = sum / f64::from(n);
        assert!((mean - 0.5).abs() < 0.01, "mean {mean} ≉ 0.5");
        assert!(lo < 0.01, "min {lo} not near 0");
        assert!(hi > 0.99, "max {hi} not near 1");
    }

    // 7. f32 draws are in [0,1) and exercise the narrowed path.
    #[test]
    fn f32_in_unit_interval() {
        let mut rng = PhiloxRng::new(99);
        for _ in 0..10_000 {
            let u = rng.next_f32();
            assert!((0.0..1.0).contains(&u), "f32 out of [0,1): {u}");
        }
    }

    // 8. with_key and new agree when the key words match the split seed.
    #[test]
    fn with_key_matches_seed_split() {
        let seed = 0x0102_0304_0506_0708u64;
        let mut from_seed = PhiloxRng::new(seed);
        let mut from_key = PhiloxRng::with_key([seed as u32, (seed >> 32) as u32]);
        for _ in 0..16 {
            assert_eq!(from_seed.next_u32(), from_key.next_u32());
        }
    }

    // 9. Block boundary: the 4th and 5th words come from consecutive blocks and
    //    match a direct evaluation of the bijection.
    #[test]
    fn block_boundary_matches_bijection() {
        let key = [0x1111_2222u32, 0x3333_4444u32];
        let block0 = philox4x32_10(key, [0, 0, 0, 0]);
        let block1 = philox4x32_10(key, [1, 0, 0, 0]);
        let mut rng = PhiloxRng::with_key(key);
        for &expected in &block0 {
            assert_eq!(rng.next_u32(), expected);
        }
        for &expected in &block1 {
            assert_eq!(rng.next_u32(), expected);
        }
    }

    // 10. fill_uniform fills the whole slice within range.
    #[test]
    fn fill_uniform_fills_slice() {
        let mut rng = PhiloxRng::new(5);
        let mut buf = [0.0f64; 37];
        rng.fill_uniform(&mut buf);
        assert!(buf.iter().all(|&u| (0.0..1.0).contains(&u)));
    }
}

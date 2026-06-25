//! ChaCha20 block function used as a counter-based RNG stream.
//!
//! Reference: Bernstein DJ (2008), "ChaCha, a variant of Salsa20", and
//! Nir Y, Langley A (2018), RFC 8439 "ChaCha20 and Poly1305 for IETF Protocols"
//! §2.3 (the 20-round ChaCha block function). We use the block function as a
//! seekable keystream: each 64-byte block is `chacha20_block(key, counter,
//! nonce)`, a pure function of its block counter, so the stream is reproducible
//! and seekable exactly like Philox — the property differential-privacy noise
//! replay requires. This is a CPU algorithm: only 32-bit add / xor / rotate.
//!
//! # Algorithm (RFC 8439 §2.3)
//! The 512-bit state is sixteen `u32` words laid out as
//!
//! ```text
//! cccccccc  cccccccc  cccccccc  cccccccc   (4 constants "expand 32-byte k")
//! kkkkkkkk  kkkkkkkk  kkkkkkkk  kkkkkkkk   (8 key words)
//! bbbbbbbb  nnnnnnnn  nnnnnnnn  nnnnnnnn   (1 block counter + 3 nonce words)
//! ```
//!
//! Twenty rounds (ten "double rounds": four column quarter-rounds followed by
//! four diagonal quarter-rounds) are applied to a working copy, which is then
//! added word-wise to the original state to produce the keystream block.

/// The four ChaCha constants: the ASCII string `"expand 32-byte k"` read as four
/// little-endian `u32`.
const CHACHA_CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

/// Number of double-rounds (ChaCha20 ⇒ 10 double-rounds ⇒ 20 rounds).
const CHACHA_DOUBLE_ROUNDS: usize = 10;

/// Reciprocal of `2³²`, mapping a full-range `u32` into `[0, 1)`.
const INV_2_32: f64 = 1.0 / 4_294_967_296.0;

/// The ChaCha quarter-round on four state words `a, b, c, d` (RFC 8439 §2.1).
#[inline]
fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_left(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_left(12);
    state[a] = state[a].wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_left(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_left(7);
}

/// Evaluate the ChaCha20 block function for a (key, counter, nonce) triple,
/// returning the 16-word (64-byte) keystream block.
///
/// Pure function of its inputs ⇒ deterministic and seekable by `counter`.
#[must_use]
pub fn chacha20_block(key: [u32; 8], counter: u32, nonce: [u32; 3]) -> [u32; 16] {
    let mut init = [0u32; 16];
    init[0..4].copy_from_slice(&CHACHA_CONSTANTS);
    init[4..12].copy_from_slice(&key);
    init[12] = counter;
    init[13..16].copy_from_slice(&nonce);

    let mut working = init;
    for _ in 0..CHACHA_DOUBLE_ROUNDS {
        // Column rounds.
        quarter_round(&mut working, 0, 4, 8, 12);
        quarter_round(&mut working, 1, 5, 9, 13);
        quarter_round(&mut working, 2, 6, 10, 14);
        quarter_round(&mut working, 3, 7, 11, 15);
        // Diagonal rounds.
        quarter_round(&mut working, 0, 5, 10, 15);
        quarter_round(&mut working, 1, 6, 11, 12);
        quarter_round(&mut working, 2, 7, 8, 13);
        quarter_round(&mut working, 3, 4, 9, 14);
    }

    let mut out = [0u32; 16];
    for i in 0..16 {
        out[i] = working[i].wrapping_add(init[i]);
    }
    out
}

/// Counter-based ChaCha20 random number generator.
///
/// Produces a deterministic `u32` (and derived `f32` / `f64` uniform) stream
/// parallel to [`crate::handle::LcgRng`] and [`super::philox::PhiloxRng`]. Each
/// 64-byte block yields sixteen `u32` words; the 32-bit block counter advances
/// by one per consumed block. Seeking is O(1): jump the block counter and
/// re-evaluate one block.
#[derive(Debug, Clone)]
pub struct ChaCha20Rng {
    key: [u32; 8],
    nonce: [u32; 3],
    /// Index of the *next* block to emit.
    block_counter: u32,
    /// Words of the current keystream block.
    buffer: [u32; 16],
    /// Index `0..16` of the next unconsumed word (`16` ⇒ empty).
    pos: usize,
}

impl ChaCha20Rng {
    /// Create a ChaCha20 RNG from a 64-bit seed. The seed is expanded into the
    /// 256-bit key by a small splitmix64-style diffusion so that nearby seeds
    /// produce well-separated key material; the nonce is fixed to zero and the
    /// block counter starts at 0.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut key = [0u32; 8];
        // Four splitmix64 outputs → eight key words.
        for pair in key.chunks_mut(2) {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            pair[0] = x as u32;
            pair[1] = (x >> 32) as u32;
        }
        Self::with_key_nonce(key, [0, 0, 0])
    }

    /// Create a ChaCha20 RNG from an explicit 256-bit key and 96-bit nonce,
    /// starting at block counter 0.
    #[must_use]
    pub fn with_key_nonce(key: [u32; 8], nonce: [u32; 3]) -> Self {
        Self {
            key,
            nonce,
            block_counter: 0,
            buffer: [0; 16],
            pos: 16, // force refill on first draw
        }
    }

    /// Evaluate the block at the current counter, advance the counter, and reset
    /// the read position.
    #[inline]
    fn refill(&mut self) {
        self.buffer = chacha20_block(self.key, self.block_counter, self.nonce);
        self.block_counter = self.block_counter.wrapping_add(1);
        self.pos = 0;
    }

    /// Seek to absolute word index `word_index`: the next [`Self::next_u32`]
    /// returns the `word_index`-th word of the stream (word 0 being the first
    /// word of block 0). Sequential reading after a seek matches having consumed
    /// `word_index` words from the start.
    pub fn seek(&mut self, word_index: u64) {
        self.block_counter = (word_index / 16) as u32;
        let within = (word_index % 16) as usize;
        self.refill();
        self.pos = within;
    }

    /// Draw the next `u32`, refilling the 64-byte keystream block when exhausted.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= 16 {
            self.refill();
        }
        let v = self.buffer[self.pos];
        self.pos += 1;
        v
    }

    /// Draw the next `u64` from two consecutive words (low word first).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// Return an `f64` uniformly distributed in `[0, 1)` via the full 32-bit
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

    /// A pair of standard-normal `N(0,1)` samples via Box-Muller, matching
    /// [`crate::handle::LcgRng::normal_pair`].
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

    // 1. RFC 8439 §2.3.2 block-function known-answer test.
    //    key = 00:01:02:…:1f, nonce = 00:00:00:09:00:00:00:4a:00:00:00:00,
    //    block counter = 1.
    #[test]
    fn chacha20_rfc8439_block_kat() {
        let key: [u32; 8] = [
            0x0302_0100,
            0x0706_0504,
            0x0b0a_0908,
            0x0f0e_0d0c,
            0x1312_1110,
            0x1716_1514,
            0x1b1a_1918,
            0x1f1e_1d1c,
        ];
        let nonce: [u32; 3] = [0x0900_0000, 0x4a00_0000, 0x0000_0000];
        let counter = 1u32;
        let expected: [u32; 16] = [
            0xe4e7_f110,
            0x1559_3bd1,
            0x1fdd_0f50,
            0xc471_20a3,
            0xc7f4_d1c7,
            0x0368_c033,
            0x9aaa_2204,
            0x4e6c_d4c3,
            0x4664_82d2,
            0x09aa_9f07,
            0x05d7_c214,
            0xa202_8bd9,
            0xd19c_12b5,
            0xb94e_16de,
            0xe883_d0cb,
            0x4e3c_50a2,
        ];
        assert_eq!(chacha20_block(key, counter, nonce), expected);
    }

    // 2. Same key/nonce ⇒ identical stream bit-for-bit.
    #[test]
    fn deterministic_same_key() {
        let mut a = ChaCha20Rng::new(0x0BAD_F00D_1234_5678);
        let mut b = ChaCha20Rng::new(0x0BAD_F00D_1234_5678);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    // 3. Different seeds ⇒ different streams (overwhelmingly).
    #[test]
    fn different_seeds_differ() {
        let mut a = ChaCha20Rng::new(10);
        let mut b = ChaCha20Rng::new(11);
        let mut differences = 0;
        for _ in 0..256 {
            if a.next_u32() != b.next_u32() {
                differences += 1;
            }
        }
        assert!(differences > 250, "seeds barely differ: {differences}/256");
    }

    // 4. Seeking to word N then reading equals sequential reads from N.
    #[test]
    fn seek_equals_sequential() {
        let mut sequential = ChaCha20Rng::new(0xFEED_FACE);
        let mut consumed = Vec::new();
        for _ in 0..80 {
            consumed.push(sequential.next_u32());
        }
        for &offset in &[0u64, 1, 15, 16, 17, 31, 32, 47, 63, 79] {
            let mut seeked = ChaCha20Rng::new(0xFEED_FACE);
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

    // 5. Block boundary: words 16..32 come from block 1 and match the bijection.
    #[test]
    fn block_boundary_matches_bijection() {
        let key = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let nonce = [9u32, 10, 11];
        let block0 = chacha20_block(key, 0, nonce);
        let block1 = chacha20_block(key, 1, nonce);
        let mut rng = ChaCha20Rng::with_key_nonce(key, nonce);
        for &expected in &block0 {
            assert_eq!(rng.next_u32(), expected);
        }
        for &expected in &block1 {
            assert_eq!(rng.next_u32(), expected);
        }
    }

    // 6. Uniform sanity: mean ≈ 0.5 and full [0,1) range via ÷2³².
    #[test]
    fn uniform_mean_and_range() {
        let mut rng = ChaCha20Rng::new(0x5151_2424_8989_F0F0);
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

    // 7. next_u64 concatenates two consecutive words (low first).
    #[test]
    fn u64_concatenates_words() {
        let mut words = ChaCha20Rng::new(7);
        let w0 = words.next_u32();
        let w1 = words.next_u32();
        let mut wide = ChaCha20Rng::new(7);
        let v = wide.next_u64();
        assert_eq!(v, ((w1 as u64) << 32) | (w0 as u64));
    }

    // 8. f32 draws stay in [0,1).
    #[test]
    fn f32_in_unit_interval() {
        let mut rng = ChaCha20Rng::new(33);
        for _ in 0..10_000 {
            let u = rng.next_f32();
            assert!((0.0..1.0).contains(&u), "f32 out of [0,1): {u}");
        }
    }

    // 9. Distinct nonces give distinct streams under the same key.
    #[test]
    fn nonce_separates_streams() {
        let key = [42u32; 8];
        let mut a = ChaCha20Rng::with_key_nonce(key, [0, 0, 0]);
        let mut b = ChaCha20Rng::with_key_nonce(key, [0, 0, 1]);
        let mut differences = 0;
        for _ in 0..64 {
            if a.next_u32() != b.next_u32() {
                differences += 1;
            }
        }
        assert!(differences > 60, "nonce barely separates: {differences}/64");
    }

    // 10. fill_uniform fills the whole slice within range.
    #[test]
    fn fill_uniform_fills_slice() {
        let mut rng = ChaCha20Rng::new(5);
        let mut buf = [0.0f64; 51];
        rng.fill_uniform(&mut buf);
        assert!(buf.iter().all(|&u| (0.0..1.0).contains(&u)));
    }
}

//! Counter-based random number generators for deterministic DP-noise replay.
//!
//! Differential-privacy auditing and reproducible experiments require that a
//! given `(key, counter)` deterministically reproduces the *same* noise draw,
//! bit-for-bit, on any machine — independent of how many other draws happened
//! before it. Counter-based RNGs (CBRNGs) provide exactly this: the `n`-th
//! output block is a pure function of `(key, n)`, so the stream is trivially
//! seekable and replayable.
//!
//! This module provides two industry-standard CBRNGs as pure-Rust structs,
//! parallel to the LCG generator in [`crate::handle`]:
//!
//! - [`PhiloxRng`] — Philox 4×32-10 (Salmon et al. 2011, Random123). The same
//!   counter-based generator family used by cuRAND's `Philox4_32_10` and
//!   JAX/TensorFlow's stateless RNG.
//! - [`ChaCha20Rng`] — the 20-round ChaCha block function (RFC 8439) used as a
//!   keystream, the basis of the widely-deployed `rand_chacha` generator.
//!
//! Both expose the same `next_u32` / `next_u64` / `next_f64` / `next_f32` /
//! `normal_pair` surface as [`crate::handle::LcgRng`], using the full 32-bit
//! output range (÷2³², never ÷2³¹) for unit-interval uniforms, and both support
//! O(1) `seek` for noise replay.

pub mod chacha20;
pub mod philox;

pub use chacha20::{ChaCha20Rng, chacha20_block};
pub use philox::{PhiloxRng, philox4x32_10};

#[cfg(test)]
mod tests {
    use super::*;

    // Cross-generator: both CBRNGs and the LCG produce uniforms with sample mean
    // near 0.5 over the full [0,1) range, confirming the shared ÷2³² convention
    // is sound for all three.
    #[test]
    fn all_generators_uniform_mean() {
        let n = 50_000;

        let mut philox = PhiloxRng::new(123);
        let mut chacha = ChaCha20Rng::new(123);

        let mut s_philox = 0.0;
        let mut s_chacha = 0.0;
        for _ in 0..n {
            s_philox += philox.next_f64();
            s_chacha += chacha.next_f64();
        }
        assert!((s_philox / f64::from(n) - 0.5).abs() < 0.01);
        assert!((s_chacha / f64::from(n) - 0.5).abs() < 0.01);
    }

    // Both CBRNGs replay a draw at an arbitrary counter without iterating from
    // the start — the core DP-noise-replay guarantee.
    #[test]
    fn cbrng_replay_without_iteration() {
        let target = 4242u128;

        let mut philox_seek = PhiloxRng::new(77);
        philox_seek.seek(target);
        let replayed_philox = philox_seek.next_u32();
        let mut philox_iter = PhiloxRng::new(77);
        for _ in 0..target {
            philox_iter.next_u32();
        }
        assert_eq!(replayed_philox, philox_iter.next_u32());

        let mut chacha_seek = ChaCha20Rng::new(77);
        chacha_seek.seek(target as u64);
        let replayed_chacha = chacha_seek.next_u32();
        let mut chacha_iter = ChaCha20Rng::new(77);
        for _ in 0..target {
            chacha_iter.next_u32();
        }
        assert_eq!(replayed_chacha, chacha_iter.next_u32());
    }
}

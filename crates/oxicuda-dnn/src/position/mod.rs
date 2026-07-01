//! CPU-reference positional-encoding primitives.
//!
//! These are dependency-free `f32` reference implementations used for
//! correctness checking and host-side prototyping, complementing the
//! device-resident encodings in [`crate::attn`]:
//!
//! - [`rope`] — Rotary Position Embedding (Su et al. 2021), cached cos/sin
//!   rotation of query/key pairs.
//! - [`alibi`] — Attention with Linear Biases (Press et al. 2022), per-head
//!   linear distance penalties added to attention scores.
//!
//! All tensors use flat row-major `Vec<f32>` / `[f32]` layouts.

pub mod alibi;
pub mod rope;

pub use alibi::{AlibiBias, alibi_slope};
pub use rope::{NeoXRopeConfig, Rope, RopeConfig, RopeStyle, apply_rope_neox_half_split};

// ---------------------------------------------------------------------------
// DnnRng — minimal host-side PRNG for the CPU-reference modules
// ---------------------------------------------------------------------------

/// Minimal full-period 64-bit LCG (Knuth MMIX constants) with `f32` and
/// standard-normal sampling, used to initialise the CPU-reference modules in
/// [`crate::position`] and [`crate::activation`].
///
/// The crate's serving-side [`crate::LcgRng`] only exposes `u64`/`f64`
/// categorical sampling; this variant adds the `f32` Box–Muller normals those
/// host-side layers need without perturbing the serving RNG's stream.
#[derive(Debug, Clone)]
pub struct DnnRng {
    state: u64,
}

impl DnnRng {
    const MUL: u64 = 6_364_136_223_846_793_005;
    const ADD: u64 = 1_442_695_040_888_963_407;

    /// Create a new generator seeded with `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(Self::ADD),
        }
    }

    /// Advance the state and return a `u32` drawn from the high bits.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(Self::MUL).wrapping_add(Self::ADD);
        ((self.state >> 33) ^ self.state) as u32
    }

    /// Return a uniform `f32` in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    /// Sample one standard-normal `f32` via the Box–Muller transform.
    #[inline]
    pub fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_f32() + 1e-10).min(1.0 - 1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Fill `buf` with standard-normal samples.
    pub fn fill_normal(&mut self, buf: &mut [f32]) {
        for v in buf.iter_mut() {
            *v = self.next_normal();
        }
    }
}

#[cfg(test)]
mod rng_tests {
    use super::DnnRng;

    #[test]
    fn dnn_rng_deterministic() {
        let mut a = DnnRng::new(42);
        let mut b = DnnRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn dnn_rng_f32_in_range() {
        let mut rng = DnnRng::new(7);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn dnn_rng_normal_finite() {
        let mut rng = DnnRng::new(13);
        let mut buf = vec![0.0_f32; 64];
        rng.fill_normal(&mut buf);
        assert!(buf.iter().all(|v| v.is_finite()));
    }
}

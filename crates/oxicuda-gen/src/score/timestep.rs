//! Timestep embedding for denoising score networks.
//!
//! Provides sinusoidal and Fourier-based timestep embeddings for
//! conditioning diffusion networks on the noise level.

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;

// ─── SinusoidalEmbedding ──────────────────────────────────────────────────────

/// Sinusoidal positional embedding for continuous timestep conditioning.
///
/// Embeds scalar timesteps `t` into a `dim`-dimensional vector using
/// sinusoidal functions at different frequencies:
/// - `e[2i] = sin(t * scale / max_period^(2i/d))`
/// - `e[2i+1] = cos(t * scale / max_period^(2i/d))`
///
/// This is the standard Transformer-style position embedding adapted for
/// diffusion model timestep conditioning.
#[derive(Debug, Clone)]
pub struct SinusoidalEmbedding {
    dim: usize,
    max_period: f32,
    scale: f32,
}

impl SinusoidalEmbedding {
    /// Create a sinusoidal embedding with default parameters.
    ///
    /// Defaults: `max_period = 10000`, `scale = 1.0`.
    ///
    /// # Errors
    /// - `EmptyInput` if `dim == 0` or `dim % 2 != 0`
    pub fn new(dim: usize) -> GenResult<Self> {
        if dim == 0 {
            return Err(GenError::EmptyInput("dim must be > 0"));
        }
        if dim % 2 != 0 {
            return Err(GenError::DimensionMismatch {
                expected: dim + 1,
                got: dim,
            });
        }
        Ok(Self {
            dim,
            max_period: 10000.0,
            scale: 1.0,
        })
    }

    /// Create with custom parameters.
    ///
    /// # Errors
    /// - `EmptyInput` if `dim == 0`
    /// - `DimensionMismatch` if `dim` is odd
    pub fn with_params(dim: usize, max_period: f32, scale: f32) -> GenResult<Self> {
        if dim == 0 {
            return Err(GenError::EmptyInput("dim must be > 0"));
        }
        if dim % 2 != 0 {
            return Err(GenError::DimensionMismatch {
                expected: dim + 1,
                got: dim,
            });
        }
        Ok(Self {
            dim,
            max_period: max_period.max(1.0),
            scale,
        })
    }

    /// Embed a single timestep `t` into a `dim`-dimensional vector.
    ///
    /// For `i ∈ [0, dim/2)`:
    /// - `e[2i] = sin(t * scale / max_period^(2i/dim))`
    /// - `e[2i+1] = cos(t * scale / max_period^(2i/dim))`
    pub fn embed_timestep(&self, t: f32) -> Vec<f32> {
        let half_dim = self.dim / 2;
        let mut emb = vec![0.0_f32; self.dim];
        for i in 0..half_dim {
            let exp = -2.0 * i as f32 / self.dim as f32 * self.max_period.ln();
            let freq = (exp).exp();
            let angle = t * self.scale * freq;
            emb[2 * i] = angle.sin();
            emb[2 * i + 1] = angle.cos();
        }
        emb
    }

    /// Embed a batch of timesteps.
    ///
    /// # Returns
    /// A flat vector of shape `[n × dim]`.
    pub fn embed_batch(&self, timesteps: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(timesteps.len() * self.dim);
        for &t in timesteps {
            out.extend(self.embed_timestep(t));
        }
        out
    }

    /// Return the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the max period.
    pub fn max_period(&self) -> f32 {
        self.max_period
    }

    /// Return the scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

// ─── FourierEmbedding ─────────────────────────────────────────────────────────

/// Random Fourier feature embedding for continuous timestep conditioning.
///
/// Embeds `t` as `[sin(2π*f_0*t), cos(2π*f_0*t), sin(2π*f_1*t), ...]`
/// where frequencies `f_i` are drawn from a Gaussian distribution.
///
/// This is the "random Fourier features" approach for scalable kernel
/// approximation, adapted from Rahimi & Recht (2007).
#[derive(Debug, Clone)]
pub struct FourierEmbedding {
    /// Random frequencies: `[dim/2]`.
    freqs: Vec<f32>,
    dim: usize,
}

impl FourierEmbedding {
    /// Create a Fourier embedding with `dim/2` random frequencies.
    ///
    /// Frequencies are drawn from `N(0, 1)`.
    ///
    /// # Errors
    /// - `EmptyInput` if `dim == 0`
    /// - `DimensionMismatch` if `dim` is odd
    pub fn new(dim: usize, rng: &mut LcgRng) -> GenResult<Self> {
        if dim == 0 {
            return Err(GenError::EmptyInput("dim must be > 0"));
        }
        if dim % 2 != 0 {
            return Err(GenError::DimensionMismatch {
                expected: dim + 1,
                got: dim,
            });
        }
        let half = dim / 2;
        let mut freqs = vec![0.0_f32; half];
        rng.fill_normal(&mut freqs);
        Ok(Self { freqs, dim })
    }

    /// Create from pre-specified frequencies.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `freqs.len() * 2 != dim`
    pub fn from_freqs(freqs: Vec<f32>, dim: usize) -> GenResult<Self> {
        if freqs.len() * 2 != dim {
            return Err(GenError::DimensionMismatch {
                expected: dim / 2,
                got: freqs.len(),
            });
        }
        Ok(Self { freqs, dim })
    }

    /// Embed a single timestep `t`.
    ///
    /// `e[2i] = sin(2π * f_i * t)`, `e[2i+1] = cos(2π * f_i * t)`.
    pub fn embed(&self, t: f32) -> Vec<f32> {
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut emb = vec![0.0_f32; self.dim];
        for (i, &f) in self.freqs.iter().enumerate() {
            let angle = two_pi * f * t;
            emb[2 * i] = angle.sin();
            emb[2 * i + 1] = angle.cos();
        }
        emb
    }

    /// Embed a batch of timesteps.
    pub fn embed_batch(&self, timesteps: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(timesteps.len() * self.dim);
        for &t in timesteps {
            out.extend(self.embed(t));
        }
        out
    }

    /// Return the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the frequency vector.
    pub fn freqs(&self) -> &[f32] {
        &self.freqs
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn sinusoidal_dim_correct() {
        let emb = SinusoidalEmbedding::new(64).expect("new should succeed");
        let out = emb.embed_timestep(1.0);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn sinusoidal_at_t0_sin_is_zero() {
        // sin(0) = 0 for all frequencies; cos(0) = 1
        let emb = SinusoidalEmbedding::new(8).expect("new should succeed");
        let out = emb.embed_timestep(0.0);
        for i in 0..4 {
            assert!(
                out[2 * i].abs() < EPS,
                "sin at t=0 should be 0: {}",
                out[2 * i]
            );
            assert!(
                (out[2 * i + 1] - 1.0).abs() < EPS,
                "cos at t=0 should be 1: {}",
                out[2 * i + 1]
            );
        }
    }

    #[test]
    fn sinusoidal_output_finite() {
        let emb = SinusoidalEmbedding::new(64).expect("new should succeed");
        for t in [0.0_f32, 0.5, 1.0, 100.0, 1000.0] {
            let out = emb.embed_timestep(t);
            assert!(out.iter().all(|v| v.is_finite()), "non-finite at t={t}");
        }
    }

    #[test]
    fn sinusoidal_batch_shape() {
        let emb = SinusoidalEmbedding::new(32).expect("new should succeed");
        let ts = vec![0.0_f32, 1.0, 2.0];
        let out = emb.embed_batch(&ts);
        assert_eq!(out.len(), 3 * 32);
    }

    #[test]
    fn sinusoidal_output_bounded() {
        // sin and cos values should be in [-1, 1]
        let emb = SinusoidalEmbedding::new(16).expect("new should succeed");
        let out = emb.embed_timestep(500.0);
        for &v in &out {
            assert!((-1.0 - EPS..=1.0 + EPS).contains(&v), "out of [-1,1]: {v}");
        }
    }

    #[test]
    fn sinusoidal_sin_cos_identity() {
        // For each pair: sin²(θ) + cos²(θ) ≈ 1
        let emb = SinusoidalEmbedding::new(8).expect("new should succeed");
        let out = emb.embed_timestep(42.0);
        for i in 0..4 {
            let s = out[2 * i];
            let c = out[2 * i + 1];
            assert!((s * s + c * c - 1.0).abs() < 1e-4, "sin²+cos²≠1 at i={i}");
        }
    }

    #[test]
    fn sinusoidal_odd_dim_rejected() {
        assert!(matches!(
            SinusoidalEmbedding::new(7),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fourier_dim_correct() {
        let mut rng = LcgRng::new(1234);
        let emb = FourierEmbedding::new(64, &mut rng).expect("new should succeed");
        let out = emb.embed(1.0);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn fourier_output_bounded() {
        let mut rng = LcgRng::new(5);
        let emb = FourierEmbedding::new(32, &mut rng).expect("new should succeed");
        let out = emb.embed(100.0);
        for &v in &out {
            assert!((-1.0 - EPS..=1.0 + EPS).contains(&v), "out of [-1,1]: {v}");
        }
    }

    #[test]
    fn fourier_sin_cos_identity() {
        let mut rng = LcgRng::new(99);
        let emb = FourierEmbedding::new(16, &mut rng).expect("new should succeed");
        let out = emb.embed(std::f32::consts::PI);
        for i in 0..8 {
            let s = out[2 * i];
            let c = out[2 * i + 1];
            assert!((s * s + c * c - 1.0).abs() < 1e-4, "sin²+cos²≠1 at i={i}");
        }
    }

    #[test]
    fn fourier_batch_shape() {
        let mut rng = LcgRng::new(77);
        let emb = FourierEmbedding::new(32, &mut rng).expect("new should succeed");
        let ts = vec![0.0_f32, 0.5, 1.0, 2.0, 5.0];
        let out = emb.embed_batch(&ts);
        assert_eq!(out.len(), 5 * 32);
    }
}

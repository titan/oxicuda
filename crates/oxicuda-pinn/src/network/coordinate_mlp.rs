//! Sinusoidal Fourier feature network (coordinate MLP with random features).

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;
use crate::network::mlp::{Mlp, MlpConfig};

/// Configuration for Fourier Feature Network.
pub struct FourierFeatureConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Number of Fourier features (each gives sin + cos → 2·n_fourier features).
    pub n_fourier: usize,
    /// Frequency scale: entries of B ~ N(0, scale²).
    pub scale: f32,
    /// Downstream MLP config (input dim should be 2·n_fourier).
    pub mlp: MlpConfig,
}

/// Fourier Feature Network.
///
/// Maps `x → [sin(2π·B·x); cos(2π·B·x)] → MLP → output`.
pub struct FourierFeatureNetwork {
    /// Random frequency matrix B: [n_fourier × input_dim], row-major.
    b: Vec<f32>,
    mlp: Mlp,
    input_dim: usize,
    n_fourier: usize,
}

impl FourierFeatureNetwork {
    /// Construct a new Fourier Feature Network.
    pub fn new(config: FourierFeatureConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        let n = config.n_fourier;
        let d = config.input_dim;
        let scale = config.scale;

        // B ~ N(0, scale²): sample via Box-Muller, then scale
        let mut b = vec![0.0_f32; n * d];
        rng.fill_normal(&mut b);
        for v in &mut b {
            *v *= scale;
        }

        // Check that MLP input dim = 2·n_fourier
        let mlp_in = config.mlp.layer_widths.first().copied().unwrap_or(0);
        if mlp_in != 2 * n {
            return Err(PinnError::DimensionMismatch {
                expected: 2 * n,
                got: mlp_in,
            });
        }

        let mlp = Mlp::new(config.mlp, rng)?;
        Ok(Self {
            b,
            mlp,
            input_dim: d,
            n_fourier: n,
        })
    }

    /// Encode `x [input_dim]` → `[sin(2π·B·x); cos(2π·B·x)]` of length `2·n_fourier`.
    pub fn encode(&self, x: &[f32]) -> Vec<f32> {
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut out = vec![0.0_f32; 2 * self.n_fourier];
        for i in 0..self.n_fourier {
            let dot: f32 = (0..self.input_dim)
                .map(|j| self.b[i * self.input_dim + j] * x[j])
                .sum();
            out[i] = (two_pi * dot).sin();
            out[self.n_fourier + i] = (two_pi * dot).cos();
        }
        out
    }

    /// Full forward: encode then MLP.
    pub fn forward(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let encoded = self.encode(x);
        self.mlp.forward(&encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::mlp::Activation;

    fn make_ffn(
        input_dim: usize,
        n_fourier: usize,
        scale: f32,
    ) -> PinnResult<FourierFeatureNetwork> {
        let mut rng = LcgRng::new(42);
        let cfg = FourierFeatureConfig {
            input_dim,
            n_fourier,
            scale,
            mlp: MlpConfig {
                layer_widths: vec![2 * n_fourier, 32, 1],
                activation: Activation::Tanh,
                omega_0: 1.0,
            },
        };
        FourierFeatureNetwork::new(cfg, &mut rng)
    }

    #[test]
    fn fourier_network_construct() {
        assert!(make_ffn(2, 8, 1.0).is_ok());
    }

    #[test]
    fn encode_length() {
        let ffn = make_ffn(2, 8, 1.0).unwrap();
        let enc = ffn.encode(&[0.3, 0.7]);
        assert_eq!(enc.len(), 16, "Encoded length should be 2 * n_fourier = 16");
    }

    #[test]
    fn encode_sin_cos_bounded() {
        let ffn = make_ffn(1, 4, 1.0).unwrap();
        let enc = ffn.encode(&[0.5]);
        for v in &enc {
            assert!(v.abs() <= 1.0 + 1e-5, "sin/cos must be in [-1,1], got {v}");
        }
    }

    #[test]
    fn forward_shape() {
        let ffn = make_ffn(2, 16, 5.0).unwrap();
        let out = ffn.forward(&[0.3, 0.7]).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn forward_finite() {
        let ffn = make_ffn(1, 8, 1.0).unwrap();
        for i in 0..10 {
            let x = i as f32 * 0.1;
            let out = ffn.forward(&[x]).unwrap();
            assert!(out[0].is_finite(), "Output not finite at x={x}");
        }
    }

    #[test]
    fn forward_dim_mismatch_error() {
        let ffn = make_ffn(2, 8, 1.0).unwrap();
        let result = ffn.forward(&[0.5]); // expects 2 inputs
        assert!(result.is_err());
    }

    #[test]
    fn wrong_mlp_input_dim_error() {
        let mut rng = LcgRng::new(1);
        let cfg = FourierFeatureConfig {
            input_dim: 2,
            n_fourier: 8,
            scale: 1.0,
            mlp: MlpConfig {
                layer_widths: vec![10, 8, 1], // wrong! should be 16
                activation: Activation::Tanh,
                omega_0: 1.0,
            },
        };
        let result = FourierFeatureNetwork::new(cfg, &mut rng);
        assert!(result.is_err());
    }
}

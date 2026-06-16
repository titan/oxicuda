//! VAE encoder with ResNet-style residual blocks.
//!
//! Implements a configurable encoder that maps inputs to a Gaussian
//! latent distribution `q(z|x) = N(μ, σ²I)`.

use crate::error::{GenError, GenResult};
use crate::vae::kl::GaussianLatent;

// ─── Activation functions ─────────────────────────────────────────────────────

/// GELU activation: `x * Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.
fn gelu(x: f32) -> f32 {
    let k = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = k * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Apply GELU elementwise to a slice.
fn gelu_slice(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| gelu(v)).collect()
}

/// GroupNorm: normalise within groups of `group_size`, then scale and shift.
///
/// Operates on a flat buffer of shape `[n_groups * group_size]`.
fn group_norm(x: &[f32], group_size: usize, scale: f32, shift: f32) -> Vec<f32> {
    if x.is_empty() || group_size == 0 {
        return x.to_vec();
    }
    let n_groups = x.len() / group_size.max(1);
    let mut out = vec![0.0_f32; x.len()];
    for g in 0..n_groups {
        let start = g * group_size;
        let end = (start + group_size).min(x.len());
        let group = &x[start..end];
        let n = group.len() as f32;
        let mean = group.iter().sum::<f32>() / n;
        let var = group.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for (i, &v) in group.iter().enumerate() {
            out[start + i] = (v - mean) * inv_std * scale + shift;
        }
    }
    // Handle remainder
    let covered = n_groups * group_size;
    for i in covered..x.len() {
        out[i] = x[i] * scale + shift;
    }
    out
}

/// Dense (linear) layer: `y = x @ W^T` where `W: [out × in]`, `x: [batch × in]`.
fn linear(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize, batch: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * out_dim];
    for b in 0..batch {
        for o in 0..out_dim {
            let mut acc = 0.0_f32;
            for i in 0..in_dim {
                acc += x[b * in_dim + i] * w[o * in_dim + i];
            }
            out[b * out_dim + o] = acc;
        }
    }
    out
}

// ─── EncoderConfig ────────────────────────────────────────────────────────────

/// Configuration for the VAE encoder.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Number of input channels/features.
    pub in_channels: usize,
    /// Base channel count (first block width).
    pub base_channels: usize,
    /// Channel multipliers per resolution level.
    pub channel_mult: Vec<usize>,
    /// Number of residual blocks per level.
    pub num_res_blocks: usize,
    /// Latent space dimensionality (half of final output, split into μ and log σ²).
    pub latent_dim: usize,
}

impl EncoderConfig {
    /// Create a minimal encoder config.
    pub fn new(
        in_channels: usize,
        base_channels: usize,
        channel_mult: Vec<usize>,
        num_res_blocks: usize,
        latent_dim: usize,
    ) -> GenResult<Self> {
        if in_channels == 0 || base_channels == 0 || latent_dim == 0 {
            return Err(GenError::EmptyInput("channel dimensions must be > 0"));
        }
        if channel_mult.is_empty() {
            return Err(GenError::EmptyInput("channel_mult must not be empty"));
        }
        Ok(Self {
            in_channels,
            base_channels,
            channel_mult,
            num_res_blocks,
            latent_dim,
        })
    }

    /// Compute the channel width at each level.
    pub fn channels_at_level(&self, level: usize) -> usize {
        self.base_channels * self.channel_mult.get(level).copied().unwrap_or(1)
    }

    /// Total number of residual blocks.
    pub fn total_blocks(&self) -> usize {
        self.channel_mult.len() * self.num_res_blocks
    }
}

// ─── ResidualBlock ────────────────────────────────────────────────────────────

/// A ResNet-style residual block.
///
/// Architecture: `GroupNorm → GELU → Linear → GroupNorm → GELU → Linear + skip`
#[derive(Debug, Clone)]
pub struct ResidualBlock {
    pub in_channels: usize,
    pub out_channels: usize,
}

impl ResidualBlock {
    /// Create a new residual block.
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `x`: Input of shape `[batch × in_channels]`.
    /// - `w1`: Weight matrix of shape `[out_channels × in_channels]`.
    /// - `w2`: Weight matrix of shape `[out_channels × out_channels]`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if weight shapes don't match
    pub fn forward(&self, x: &[f32], w1: &[f32], w2: &[f32]) -> GenResult<Vec<f32>> {
        if x.is_empty() {
            return Err(GenError::EmptyInput("x is empty"));
        }
        if x.len() % self.in_channels != 0 {
            return Err(GenError::DimensionMismatch {
                expected: x.len() - x.len() % self.in_channels,
                got: x.len(),
            });
        }
        let batch = x.len() / self.in_channels;
        // Validate weight shapes
        let expected_w1 = self.out_channels * self.in_channels;
        let expected_w2 = self.out_channels * self.out_channels;
        if w1.len() != expected_w1 {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![self.out_channels, self.in_channels],
                input: vec![w1.len()],
            });
        }
        if w2.len() != expected_w2 {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![self.out_channels, self.out_channels],
                input: vec![w2.len()],
            });
        }
        // First norm-activation-linear
        let h = group_norm(x, self.in_channels.max(1), 1.0, 0.0);
        let h = gelu_slice(&h);
        let h = linear(&h, w1, self.in_channels, self.out_channels, batch);
        // Second norm-activation-linear
        let h = group_norm(&h, self.out_channels.max(1), 1.0, 0.0);
        let h = gelu_slice(&h);
        let h = linear(&h, w2, self.out_channels, self.out_channels, batch);
        // Skip connection: project input if dimensions differ
        let skip: Vec<f32> = if self.in_channels == self.out_channels {
            x.to_vec()
        } else {
            // Zero-pad or truncate (simple channel adaptation)
            let mut s = vec![0.0_f32; batch * self.out_channels];
            let min_ch = self.in_channels.min(self.out_channels);
            for b in 0..batch {
                for c in 0..min_ch {
                    s[b * self.out_channels + c] = x[b * self.in_channels + c];
                }
            }
            s
        };
        let out = h.iter().zip(&skip).map(|(&a, &b)| a + b).collect();
        Ok(out)
    }
}

// ─── EncoderWeights ───────────────────────────────────────────────────────────

/// Weight container for the encoder.
#[derive(Debug, Clone)]
pub struct EncoderWeights {
    /// Block weights: w1 per block.
    pub block_w1: Vec<Vec<f32>>,
    /// Block weights: w2 per block.
    pub block_w2: Vec<Vec<f32>>,
    /// Projection to mean: `[latent_dim × final_channels]`.
    pub proj_mu: Vec<f32>,
    /// Projection to log-variance: `[latent_dim × final_channels]`.
    pub proj_logvar: Vec<f32>,
}

impl EncoderWeights {
    /// Create zero-initialised weights for the given config.
    ///
    /// Matches exactly the block structure produced by `Encoder::new`:
    /// - For each level and residual block, if it is the first block in the level
    ///   (`res == 0`), `in_ch = in_ch_level`, otherwise `in_ch = out_ch`.
    pub fn zeros(config: &EncoderConfig) -> Self {
        let n_blocks = config.total_blocks();
        let mut block_w1 = Vec::with_capacity(n_blocks);
        let mut block_w2 = Vec::with_capacity(n_blocks);
        for level in 0..config.channel_mult.len() {
            let in_ch_level = if level == 0 {
                config.in_channels
            } else {
                config.channels_at_level(level - 1)
            };
            let out_ch = config.channels_at_level(level);
            for res in 0..config.num_res_blocks {
                let in_c = if res == 0 { in_ch_level } else { out_ch };
                block_w1.push(vec![0.0_f32; out_ch * in_c]);
                block_w2.push(vec![0.0_f32; out_ch * out_ch]);
            }
        }
        let final_ch = config.channels_at_level(config.channel_mult.len() - 1);
        let proj_mu = vec![0.0_f32; config.latent_dim * final_ch];
        let proj_logvar = vec![0.0_f32; config.latent_dim * final_ch];
        Self {
            block_w1,
            block_w2,
            proj_mu,
            proj_logvar,
        }
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

/// VAE encoder.
///
/// Maps input `x` through a series of residual blocks and projects
/// to Gaussian parameters `(μ, log σ²)`.
#[derive(Debug, Clone)]
pub struct Encoder {
    config: EncoderConfig,
    blocks: Vec<ResidualBlock>,
}

impl Encoder {
    /// Create a new encoder from the given config.
    ///
    /// # Errors
    /// - `EmptyInput` if config has zero-size dimensions
    pub fn new(config: EncoderConfig) -> GenResult<Self> {
        if config.in_channels == 0 {
            return Err(GenError::EmptyInput("in_channels must be > 0"));
        }
        let mut blocks = Vec::new();
        let n_levels = config.channel_mult.len();
        let mut block_idx = 0;
        for level in 0..n_levels {
            let out_ch = config.channels_at_level(level);
            let in_ch_level = if level == 0 {
                config.in_channels
            } else {
                config.channels_at_level(level - 1)
            };
            for res in 0..config.num_res_blocks {
                let in_ch = if res == 0 { in_ch_level } else { out_ch };
                blocks.push(ResidualBlock::new(in_ch, out_ch));
                block_idx += 1;
            }
        }
        let _ = block_idx; // suppress unused warning
        Ok(Self { config, blocks })
    }

    /// Run the encoder forward pass.
    ///
    /// # Arguments
    /// - `x`: Input of shape `[batch × in_channels]`.
    /// - `weights`: Pretrained or zero weights.
    ///
    /// # Returns
    /// A `GaussianLatent` with `μ` and `log σ²` of shape `[batch × latent_dim]`.
    ///
    /// # Errors
    /// - `DimensionMismatch` on shape mismatch
    /// - `EmptyInput` if `x` is empty
    pub fn encode(&self, x: &[f32], weights: &EncoderWeights) -> GenResult<GaussianLatent> {
        if x.is_empty() {
            return Err(GenError::EmptyInput("x is empty"));
        }
        if x.len() % self.config.in_channels != 0 {
            return Err(GenError::DimensionMismatch {
                expected: x.len() - x.len() % self.config.in_channels,
                got: x.len(),
            });
        }
        let batch = x.len() / self.config.in_channels;
        if weights.block_w1.len() != self.blocks.len() {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![self.blocks.len()],
                input: vec![weights.block_w1.len()],
            });
        }
        // Run through residual blocks
        let mut h = x.to_vec();
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &weights.block_w1[i], &weights.block_w2[i])?;
        }
        // Project to μ and log σ²
        let final_ch = if self.blocks.is_empty() {
            self.config.in_channels
        } else {
            self.blocks
                .last()
                .map(|b| b.out_channels)
                .unwrap_or(self.config.in_channels)
        };
        let mu = linear(
            &h,
            &weights.proj_mu,
            final_ch,
            self.config.latent_dim,
            batch,
        );
        let logvar = linear(
            &h,
            &weights.proj_logvar,
            final_ch,
            self.config.latent_dim,
            batch,
        );
        GaussianLatent::new(mu, logvar)
    }

    /// Return the encoder config.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Estimate the total number of parameters.
    pub fn num_params(&self) -> usize {
        let block_params: usize = self
            .blocks
            .iter()
            .map(|b| b.in_channels * b.out_channels + b.out_channels * b.out_channels)
            .sum();
        let final_ch = self
            .blocks
            .last()
            .map(|b| b.out_channels)
            .unwrap_or(self.config.in_channels);
        let proj_params = 2 * self.config.latent_dim * final_ch;
        block_params + proj_params
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> EncoderConfig {
        EncoderConfig::new(4, 8, vec![1, 2], 1, 16).expect("new should succeed")
    }

    #[test]
    fn encoder_new_valid() {
        let config = make_config();
        let enc = Encoder::new(config).expect("new should succeed");
        assert!(!enc.blocks.is_empty());
    }

    #[test]
    fn encoder_forward_output_shape() {
        let config = make_config();
        let weights = EncoderWeights::zeros(&config);
        let enc = Encoder::new(config.clone()).expect("value should be present");
        let batch = 2;
        let x = vec![0.5_f32; batch * config.in_channels];
        let latent = enc.encode(&x, &weights).expect("encode should succeed");
        assert_eq!(latent.mu.len(), batch * config.latent_dim);
        assert_eq!(latent.logvar.len(), batch * config.latent_dim);
    }

    #[test]
    fn encoder_forward_all_zeros_weights() {
        // With zero weights, output should be zero (bias-free)
        let config = make_config();
        let weights = EncoderWeights::zeros(&config);
        let enc = Encoder::new(config.clone()).expect("value should be present");
        let x = vec![1.0_f32; config.in_channels];
        let latent = enc.encode(&x, &weights).expect("encode should succeed");
        // With zero weights, linear layers output 0
        for &v in &latent.mu {
            assert!(
                v.abs() < 1e-5,
                "expected zero output with zero weights: {v}"
            );
        }
    }

    #[test]
    fn encoder_empty_input_rejected() {
        let config = make_config();
        let weights = EncoderWeights::zeros(&config);
        let enc = Encoder::new(config).expect("new should succeed");
        assert!(matches!(
            enc.encode(&[], &weights),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn residual_block_forward_shape() {
        let block = ResidualBlock::new(4, 8);
        let x = vec![1.0_f32; 4]; // batch=1, in=4
        let w1 = vec![0.0_f32; 8 * 4];
        let w2 = vec![0.0_f32; 8 * 8];
        let out = block.forward(&x, &w1, &w2).expect("forward should succeed");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn residual_block_same_channels() {
        let block = ResidualBlock::new(4, 4);
        let x = vec![1.0_f32; 4];
        let w1 = vec![0.0_f32; 4 * 4];
        let w2 = vec![0.0_f32; 4 * 4];
        let out = block.forward(&x, &w1, &w2).expect("forward should succeed");
        // With zero weights, output = skip connection = input
        for (&o, &xi) in out.iter().zip(&x) {
            assert!(
                (o - xi).abs() < 1e-5,
                "skip should pass through: {o} vs {xi}"
            );
        }
    }

    #[test]
    fn gelu_zero_input() {
        assert!(gelu(0.0).abs() < 1e-5);
    }

    #[test]
    fn gelu_positive_for_positive_input() {
        assert!(gelu(1.0) > 0.0);
        assert!(gelu(2.0) > gelu(1.0));
    }

    #[test]
    fn encoder_num_params_positive() {
        let config = make_config();
        let enc = Encoder::new(config).expect("new should succeed");
        assert!(enc.num_params() > 0);
    }
}

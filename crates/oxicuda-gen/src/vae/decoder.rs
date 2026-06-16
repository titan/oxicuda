//! VAE decoder with ResNet-style residual blocks.
//!
//! Implements a configurable decoder that maps latent vectors back to
//! the data domain via a series of upsampling residual blocks.

use crate::error::{GenError, GenResult};

// ─── Shared utilities (re-implemented to avoid cross-module private access) ───

fn gelu(x: f32) -> f32 {
    let k = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = k * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn gelu_slice(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| gelu(v)).collect()
}

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
    let covered = n_groups * group_size;
    for i in covered..x.len() {
        out[i] = x[i] * scale + shift;
    }
    out
}

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

// ─── DecoderConfig ────────────────────────────────────────────────────────────

/// Configuration for the VAE decoder.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Dimensionality of the input latent vector.
    pub latent_dim: usize,
    /// Base channel count.
    pub base_channels: usize,
    /// Channel multipliers per resolution level (applied in reverse for upsampling).
    pub channel_mult: Vec<usize>,
    /// Number of residual blocks per level.
    pub num_res_blocks: usize,
    /// Number of output channels/features.
    pub out_channels: usize,
}

impl DecoderConfig {
    /// Create a new decoder config.
    pub fn new(
        latent_dim: usize,
        base_channels: usize,
        channel_mult: Vec<usize>,
        num_res_blocks: usize,
        out_channels: usize,
    ) -> GenResult<Self> {
        if latent_dim == 0 || base_channels == 0 || out_channels == 0 {
            return Err(GenError::EmptyInput("channel dimensions must be > 0"));
        }
        if channel_mult.is_empty() {
            return Err(GenError::EmptyInput("channel_mult must not be empty"));
        }
        Ok(Self {
            latent_dim,
            base_channels,
            channel_mult,
            num_res_blocks,
            out_channels,
        })
    }

    /// Compute the channel width at a given level.
    pub fn channels_at_level(&self, level: usize) -> usize {
        self.base_channels * self.channel_mult.get(level).copied().unwrap_or(1)
    }

    /// Total residual blocks.
    pub fn total_blocks(&self) -> usize {
        self.channel_mult.len() * self.num_res_blocks
    }
}

// ─── DecoderWeights ───────────────────────────────────────────────────────────

/// Weight container for the decoder.
#[derive(Debug, Clone)]
pub struct DecoderWeights {
    /// Input projection: `[first_channels × latent_dim]`.
    pub proj_in: Vec<f32>,
    /// Block first-layer weights.
    pub block_w1: Vec<Vec<f32>>,
    /// Block second-layer weights.
    pub block_w2: Vec<Vec<f32>>,
    /// Output projection: `[out_channels × last_channels]`.
    pub proj_out: Vec<f32>,
}

impl DecoderWeights {
    /// Create zero-initialised weights for the given config.
    pub fn zeros(config: &DecoderConfig) -> Self {
        let n_levels = config.channel_mult.len();
        // Start from the highest multiplier, then decrease (mirror of encoder)
        let first_ch = config.channels_at_level(n_levels - 1);
        let proj_in = vec![0.0_f32; first_ch * config.latent_dim];
        let mut block_w1 = Vec::new();
        let mut block_w2 = Vec::new();
        for level in (0..n_levels).rev() {
            let out_ch = config.channels_at_level(level);
            let in_ch_level = if level == n_levels - 1 {
                first_ch
            } else {
                config.channels_at_level(level + 1)
            };
            for res in 0..config.num_res_blocks {
                let in_ch = if res == 0 { in_ch_level } else { out_ch };
                block_w1.push(vec![0.0_f32; out_ch * in_ch]);
                block_w2.push(vec![0.0_f32; out_ch * out_ch]);
            }
        }
        let last_ch = config.channels_at_level(0);
        let proj_out = vec![0.0_f32; config.out_channels * last_ch];
        Self {
            proj_in,
            block_w1,
            block_w2,
            proj_out,
        }
    }
}

// ─── Decoder ──────────────────────────────────────────────────────────────────

/// VAE decoder.
///
/// Maps latent vectors `z` through a series of residual blocks and projects
/// to the output domain.
#[derive(Debug, Clone)]
pub struct Decoder {
    config: DecoderConfig,
    /// Block structure: `(in_channels, out_channels)`.
    blocks: Vec<(usize, usize)>,
}

impl Decoder {
    /// Create a new decoder from the given config.
    ///
    /// # Errors
    /// - `EmptyInput` if config has zero-size dimensions
    pub fn new(config: DecoderConfig) -> GenResult<Self> {
        if config.latent_dim == 0 {
            return Err(GenError::EmptyInput("latent_dim must be > 0"));
        }
        let n_levels = config.channel_mult.len();
        let first_ch = config.channels_at_level(n_levels - 1);
        let mut blocks = Vec::new();
        for level in (0..n_levels).rev() {
            let out_ch = config.channels_at_level(level);
            let in_ch_level = if level == n_levels - 1 {
                first_ch
            } else {
                config.channels_at_level(level + 1)
            };
            for res in 0..config.num_res_blocks {
                let in_ch = if res == 0 { in_ch_level } else { out_ch };
                blocks.push((in_ch, out_ch));
            }
        }
        Ok(Self { config, blocks })
    }

    /// Run the decoder forward pass.
    ///
    /// # Arguments
    /// - `z`: Latent vector of shape `[batch × latent_dim]`.
    /// - `weights`: Decoder weights.
    ///
    /// # Returns
    /// Output of shape `[batch × out_channels]`.
    ///
    /// # Errors
    /// - `DimensionMismatch` on shape mismatch
    /// - `EmptyInput` if `z` is empty
    pub fn decode(&self, z: &[f32], weights: &DecoderWeights) -> GenResult<Vec<f32>> {
        if z.is_empty() {
            return Err(GenError::EmptyInput("z is empty"));
        }
        if z.len() % self.config.latent_dim != 0 {
            return Err(GenError::DimensionMismatch {
                expected: z.len() - z.len() % self.config.latent_dim,
                got: z.len(),
            });
        }
        let batch = z.len() / self.config.latent_dim;
        let n_levels = self.config.channel_mult.len();
        let first_ch = self.config.channels_at_level(n_levels - 1);
        // Input projection: latent → first_ch
        if weights.proj_in.len() != first_ch * self.config.latent_dim {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![first_ch, self.config.latent_dim],
                input: vec![weights.proj_in.len()],
            });
        }
        let mut h = linear(z, &weights.proj_in, self.config.latent_dim, first_ch, batch);
        // Run residual blocks
        if weights.block_w1.len() != self.blocks.len() {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![self.blocks.len()],
                input: vec![weights.block_w1.len()],
            });
        }
        for (i, &(in_ch, out_ch)) in self.blocks.iter().enumerate() {
            let w1 = &weights.block_w1[i];
            let w2 = &weights.block_w2[i];
            // Norm-act-linear-norm-act-linear + skip
            let hn = group_norm(&h, in_ch.max(1), 1.0, 0.0);
            let hn = gelu_slice(&hn);
            let hn = if w1.len() == out_ch * in_ch {
                linear(&hn, w1, in_ch, out_ch, batch)
            } else {
                return Err(GenError::WeightShapeMismatch {
                    weight: vec![out_ch, in_ch],
                    input: vec![w1.len()],
                });
            };
            let hn = group_norm(&hn, out_ch.max(1), 1.0, 0.0);
            let hn = gelu_slice(&hn);
            let hn = if w2.len() == out_ch * out_ch {
                linear(&hn, w2, out_ch, out_ch, batch)
            } else {
                return Err(GenError::WeightShapeMismatch {
                    weight: vec![out_ch, out_ch],
                    input: vec![w2.len()],
                });
            };
            // Skip connection
            let skip: Vec<f32> = if in_ch == out_ch {
                h.clone()
            } else {
                let mut s = vec![0.0_f32; batch * out_ch];
                let min_ch = in_ch.min(out_ch);
                for b in 0..batch {
                    for c in 0..min_ch {
                        s[b * out_ch + c] = h[b * in_ch + c];
                    }
                }
                s
            };
            h = hn.iter().zip(&skip).map(|(&a, &b)| a + b).collect();
        }
        // Output projection
        let last_ch = self.config.channels_at_level(0);
        if weights.proj_out.len() != self.config.out_channels * last_ch {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![self.config.out_channels, last_ch],
                input: vec![weights.proj_out.len()],
            });
        }
        let out = linear(
            &h,
            &weights.proj_out,
            last_ch,
            self.config.out_channels,
            batch,
        );
        Ok(out)
    }

    /// Return the decoder config.
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Return the number of residual blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> DecoderConfig {
        DecoderConfig::new(16, 8, vec![1, 2], 1, 4)
            .expect("valid decoder config with sensible dimensions should construct without error")
    }

    #[test]
    fn decoder_new_valid() {
        let config = make_config();
        let dec = Decoder::new(config).expect("new should succeed");
        assert!(dec.num_blocks() > 0);
    }

    #[test]
    fn decoder_forward_output_shape() {
        let config = make_config();
        let weights = DecoderWeights::zeros(&config);
        let dec = Decoder::new(config.clone()).expect("value should be present");
        let batch = 3;
        let z = vec![0.5_f32; batch * config.latent_dim];
        let out = dec.decode(&z, &weights).expect("decode should succeed");
        assert_eq!(out.len(), batch * config.out_channels);
    }

    #[test]
    fn decoder_forward_zero_weights() {
        // With zero weights, output should be zero
        let config = make_config();
        let weights = DecoderWeights::zeros(&config);
        let dec = Decoder::new(config.clone()).expect("value should be present");
        let z = vec![1.0_f32; config.latent_dim];
        let out = dec.decode(&z, &weights).expect("decode should succeed");
        for &v in &out {
            assert!(v.abs() < 1e-5, "expected zero with zero weights: {v}");
        }
    }

    #[test]
    fn decoder_empty_input_rejected() {
        let config = make_config();
        let weights = DecoderWeights::zeros(&config);
        let dec = Decoder::new(config).expect("new should succeed");
        assert!(matches!(
            dec.decode(&[], &weights),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn decoder_output_finite() {
        let config = make_config();
        let weights = DecoderWeights::zeros(&config);
        let dec = Decoder::new(config.clone()).expect("value should be present");
        let z: Vec<f32> = (0..config.latent_dim).map(|i| i as f32 * 0.01).collect();
        let out = dec.decode(&z, &weights).expect("decode should succeed");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output contains non-finite values"
        );
    }

    #[test]
    fn gelu_monotone_for_large_positive() {
        let g1 = gelu(1.0);
        let g2 = gelu(2.0);
        assert!(g2 > g1, "GELU should be monotone for large positive x");
    }

    #[test]
    fn group_norm_zero_variance() {
        // Constant input → mean = value, var = 0, norm = 0, scaled = shift
        let x = vec![5.0_f32; 8];
        let out = group_norm(&x, 8, 1.0, 0.0);
        for &v in &out {
            assert!(v.abs() < 1e-4, "group_norm of constant should be ~0: {v}");
        }
    }

    #[test]
    fn decoder_config_channels_at_level() {
        let config = make_config();
        assert_eq!(config.channels_at_level(0), 8); // base * mult[0] = 8*1
        assert_eq!(config.channels_at_level(1), 16); // base * mult[1] = 8*2
    }

    #[test]
    fn decoder_weights_zeros_shape() {
        let config = make_config();
        let weights = DecoderWeights::zeros(&config);
        assert!(!weights.proj_in.is_empty());
        assert!(!weights.proj_out.is_empty());
        assert_eq!(weights.block_w1.len(), config.total_blocks());
    }
}

//! TCN encoder: a stack of residual blocks with exponentially doubling dilations.
//!
//! Dilation schedule: block `i` uses dilation `2^i`, so the receptive field
//! grows exponentially.  The first block maps `in_channels → hidden_channels`,
//! interior blocks preserve `hidden_channels`, and the last block maps to
//! `out_channels`.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;
use crate::tcn::temporal_block::TcnBlock;

/// Configuration for a `TcnEncoder`.
#[derive(Debug, Clone)]
pub struct TcnConfig {
    /// Number of input channels.
    pub in_channels: usize,
    /// Width of intermediate layers.
    pub hidden_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Convolutional kernel size (same for every block).
    pub kernel_size: usize,
    /// Number of residual blocks.
    pub num_layers: usize,
}

impl TcnConfig {
    /// Small configuration suitable for fast unit tests.
    ///
    /// `in=4, hidden=16, out=16, kernel=3, layers=4`
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            hidden_channels: 16,
            out_channels: 16,
            kernel_size: 3,
            num_layers: 4,
        }
    }

    /// Practical default configuration.
    ///
    /// `in=7, hidden=64, out=64, kernel=3, layers=8`
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            in_channels: 7,
            hidden_channels: 64,
            out_channels: 64,
            kernel_size: 3,
            num_layers: 8,
        }
    }
}

/// Stacked TCN residual encoder.
#[derive(Debug, Clone)]
pub struct TcnEncoder {
    /// Residual blocks in order.
    pub blocks: Vec<TcnBlock>,
    /// Config used to build this encoder.
    pub config: TcnConfig,
}

impl TcnEncoder {
    /// Build a `TcnEncoder` from a `TcnConfig`.
    ///
    /// Block dilation schedule: `dilation[i] = 2^i`.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`TcnBlock::new`] (kernel/dilation zero).
    pub fn new(config: TcnConfig, rng: &mut LcgRng) -> TsResult<Self> {
        let n = config.num_layers;
        let mut blocks = Vec::with_capacity(n);

        for i in 0..n {
            let dilation = 1_usize << i; // 2^i

            let (c_in, c_out) = if n == 1 {
                (config.in_channels, config.out_channels)
            } else if i == 0 {
                (config.in_channels, config.hidden_channels)
            } else if i == n - 1 {
                (config.hidden_channels, config.out_channels)
            } else {
                (config.hidden_channels, config.hidden_channels)
            };

            blocks.push(TcnBlock::new(
                c_in,
                c_out,
                config.kernel_size,
                dilation,
                rng,
            )?);
        }

        Ok(Self { blocks, config })
    }

    /// Run all blocks sequentially on a `[T, in_channels]` input.
    ///
    /// Returns `[T, out_channels]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != t * in_channels`.
    pub fn forward(&self, x: &[f32], t: usize) -> TsResult<Vec<f32>> {
        let expected = t * self.config.in_channels;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let mut h = x.to_vec();
        for block in &self.blocks {
            h = block.forward(&h, t)?;
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(99)
    }

    #[test]
    fn tcn_encoder_tiny_output_shape() {
        let mut rng = make_rng();
        let enc = TcnEncoder::new(TcnConfig::tiny(), &mut rng).expect("ok");
        let t = 32;
        let x = vec![0.1_f32; t * 4];
        let out = enc.forward(&x, t).expect("ok");
        assert_eq!(out.len(), t * 16);
    }

    #[test]
    fn tcn_encoder_output_finite() {
        let mut rng = make_rng();
        let enc = TcnEncoder::new(TcnConfig::tiny(), &mut rng).expect("ok");
        let t = 48;
        let mut x = vec![0.0_f32; t * 4];
        rng.fill_normal(&mut x);
        let out = enc.forward(&x, t).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn tcn_encoder_output_nonneg() {
        // All outputs should be >= 0 because the last block ends with ReLU.
        let mut rng = make_rng();
        let enc = TcnEncoder::new(TcnConfig::tiny(), &mut rng).expect("ok");
        let t = 20;
        let mut x = vec![0.0_f32; t * 4];
        rng.fill_normal(&mut x);
        let out = enc.forward(&x, t).expect("ok");
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn tcn_encoder_correct_block_count() {
        let mut rng = make_rng();
        let enc = TcnEncoder::new(TcnConfig::tiny(), &mut rng).expect("ok");
        assert_eq!(enc.blocks.len(), 4);
    }

    #[test]
    fn tcn_encoder_exponential_dilations() {
        let mut rng = make_rng();
        let enc = TcnEncoder::new(TcnConfig::tiny(), &mut rng).expect("ok");
        let expected_dilations = [1_usize, 2, 4, 8];
        for (i, (block, &exp_d)) in enc.blocks.iter().zip(expected_dilations.iter()).enumerate() {
            assert_eq!(
                block.dilation, exp_d,
                "block {i}: expected dilation {exp_d}, got {}",
                block.dilation
            );
        }
    }

    #[test]
    fn tcn_encoder_dim_mismatch_error() {
        let mut rng = make_rng();
        let enc = TcnEncoder::new(TcnConfig::tiny(), &mut rng).expect("ok");
        let x = vec![0.0_f32; 7]; // wrong
        assert!(matches!(
            enc.forward(&x, 2).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn tcn_encoder_single_layer_channels() {
        // When num_layers=1, first block should be in_channels → out_channels directly
        let mut rng = make_rng();
        let cfg = TcnConfig {
            in_channels: 3,
            hidden_channels: 16,
            out_channels: 8,
            kernel_size: 3,
            num_layers: 1,
        };
        let enc = TcnEncoder::new(cfg, &mut rng).expect("ok");
        assert_eq!(enc.blocks.len(), 1);
        assert_eq!(enc.blocks[0].c_in, 3);
        assert_eq!(enc.blocks[0].c_out, 8);
    }

    #[test]
    fn tcn_encoder_default_config_output_shape() {
        let mut rng = make_rng();
        let enc = TcnEncoder::new(TcnConfig::default_config(), &mut rng).expect("ok");
        let t = 16;
        let mut x = vec![0.0_f32; t * 7];
        rng.fill_normal(&mut x);
        let out = enc.forward(&x, t).expect("ok");
        assert_eq!(out.len(), t * 64);
    }
}

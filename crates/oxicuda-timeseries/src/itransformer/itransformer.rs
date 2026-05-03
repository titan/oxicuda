//! iTransformer (Liu et al. 2024): inverted attention over variates.
//!
//! Instead of attending over the time axis, each variate is first projected
//! from `[T]` to a token `[D]`, and the Transformer then attends over the C
//! variate tokens.  This makes the model agnostic to sequence length and
//! enables explicit multivariate correlation modelling.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;
use crate::itransformer::inverted_block::InvertedBlock;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Full configuration for an iTransformer model.
#[derive(Debug, Clone)]
pub struct ITransformerConfig {
    /// Number of variates.
    pub c: usize,
    /// Input sequence length.
    pub t: usize,
    /// Forecast horizon (steps).
    pub horizon: usize,
    /// Embedding dimension per variate token.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of inverted Transformer blocks.
    pub n_layers: usize,
    /// FFN expansion factor (default 4).
    pub ffn_expansion: usize,
}

impl ITransformerConfig {
    /// Small configuration: `d=64, heads=4, layers=2, expansion=4`.
    pub fn tiny(c: usize, t: usize, horizon: usize) -> Self {
        Self {
            c,
            t,
            horizon,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            ffn_expansion: 4,
        }
    }

    /// Standard configuration: `d=128, heads=8, layers=3, expansion=4`.
    pub fn base(c: usize, t: usize, horizon: usize) -> Self {
        Self {
            c,
            t,
            horizon,
            d_model: 128,
            n_heads: 8,
            n_layers: 3,
            ffn_expansion: 4,
        }
    }
}

// ─── Model ────────────────────────────────────────────────────────────────────

/// iTransformer forecasting model.
///
/// Embeds each variate over its full time history, attends across variates,
/// and decodes per-variate to the forecast horizon.
#[derive(Debug, Clone)]
pub struct ITransformer {
    /// Variate embedding weight `[D, T]`.
    pub embed_w: Vec<f32>,
    /// Variate embedding bias `[D]`.
    pub embed_b: Vec<f32>,
    /// Stack of inverted Transformer blocks.
    pub blocks: Vec<InvertedBlock>,
    /// Forecast head weight `[C * horizon, D]`.
    pub head_w: Vec<f32>,
    /// Forecast head bias `[C * horizon]`.
    pub head_b: Vec<f32>,
    /// Model configuration.
    pub config: ITransformerConfig,
}

impl ITransformer {
    /// Build an iTransformer from config.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidEmbedDim`] when `d_model == 0`.
    /// - [`TsError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`TsError::HeadDimMismatch`] when `d_model % n_heads != 0`.
    pub fn new(config: ITransformerConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.d_model == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if config.n_heads == 0 {
            return Err(TsError::InvalidNumHeads(0));
        }
        if config.d_model % config.n_heads != 0 {
            return Err(TsError::HeadDimMismatch {
                embed_dim: config.d_model,
                n_heads: config.n_heads,
            });
        }

        let d = config.d_model;
        let t = config.t;

        let embed_scale = (6.0_f32 / (t + d) as f32).sqrt();
        let mut embed_w = vec![0.0_f32; d * t];
        rng.fill_normal(&mut embed_w);
        for w in &mut embed_w {
            *w *= embed_scale;
        }
        let embed_b = vec![0.0_f32; d];

        let blocks = (0..config.n_layers)
            .map(|_| InvertedBlock::new(d, config.n_heads, rng))
            .collect::<TsResult<Vec<_>>>()?;

        let head_out = config.c * config.horizon;
        let head_scale = (6.0_f32 / (d + config.horizon) as f32).sqrt();
        let mut head_w = vec![0.0_f32; head_out * d];
        rng.fill_normal(&mut head_w);
        for w in &mut head_w {
            *w *= head_scale;
        }
        let head_b = vec![0.0_f32; head_out];

        Ok(Self {
            embed_w,
            embed_b,
            blocks,
            head_w,
            head_b,
            config,
        })
    }

    /// Forecast `x: [T, C]` → `[horizon, C]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != t * c`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let cfg = &self.config;
        let expected = cfg.t * cfg.c;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let d = cfg.d_model;
        let mut tokens = vec![0.0_f32; cfg.c * d];

        for ci in 0..cfg.c {
            for di in 0..d {
                let mut acc = self.embed_b[di];
                for ti in 0..cfg.t {
                    acc += x[ti * cfg.c + ci] * self.embed_w[di * cfg.t + ti];
                }
                tokens[ci * d + di] = acc;
            }
        }

        for block in &self.blocks {
            tokens = block.forward(&tokens, cfg.c)?;
        }

        let mut out_ch_first = vec![0.0_f32; cfg.c * cfg.horizon];
        for ci in 0..cfg.c {
            for hi in 0..cfg.horizon {
                let row = ci * cfg.horizon + hi;
                let mut val = self.head_b[row];
                for di in 0..d {
                    val += self.head_w[row * d + di] * tokens[ci * d + di];
                }
                out_ch_first[ci * cfg.horizon + hi] = val;
            }
        }

        let mut forecast = vec![0.0_f32; cfg.horizon * cfg.c];
        for ci in 0..cfg.c {
            for hi in 0..cfg.horizon {
                forecast[hi * cfg.c + ci] = out_ch_first[ci * cfg.horizon + hi];
            }
        }

        Ok(forecast)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(77)
    }

    #[test]
    fn itransformer_tiny_output_shape() {
        let mut rng = make_rng();
        let cfg = ITransformerConfig::tiny(4, 48, 12);
        let model = ITransformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.5_f32; cfg.t * cfg.c];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.horizon * cfg.c);
    }

    #[test]
    fn itransformer_base_output_shape() {
        let mut rng = make_rng();
        let cfg = ITransformerConfig::base(3, 64, 24);
        let model = ITransformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![1.0_f32; cfg.t * cfg.c];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.horizon * cfg.c);
    }

    #[test]
    fn itransformer_output_finite() {
        let mut rng = make_rng();
        let cfg = ITransformerConfig::tiny(5, 32, 8);
        let model = ITransformer::new(cfg.clone(), &mut rng).expect("build");
        let mut x = vec![0.0_f32; cfg.t * cfg.c];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn itransformer_horizon_first_layout() {
        let mut rng = make_rng();
        let cfg = ITransformerConfig::tiny(2, 32, 6);
        let model = ITransformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.0_f32; cfg.t * cfg.c];
        let out = model.forward(&x).expect("forward");
        // [horizon=6, C=2] → 12 elements
        assert_eq!(out.len(), 12);
    }

    #[test]
    fn itransformer_error_invalid_embed_dim() {
        let mut rng = make_rng();
        let cfg = ITransformerConfig {
            d_model: 0,
            ..ITransformerConfig::tiny(2, 32, 6)
        };
        assert!(matches!(
            ITransformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidEmbedDim(0)
        ));
    }

    #[test]
    fn itransformer_error_invalid_num_heads() {
        let mut rng = make_rng();
        let cfg = ITransformerConfig {
            n_heads: 0,
            ..ITransformerConfig::tiny(2, 32, 6)
        };
        assert!(matches!(
            ITransformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidNumHeads(0)
        ));
    }

    #[test]
    fn itransformer_error_head_dim_mismatch() {
        let mut rng = make_rng();
        let cfg = ITransformerConfig {
            d_model: 65,
            n_heads: 4,
            ..ITransformerConfig::tiny(2, 32, 6)
        };
        assert!(matches!(
            ITransformer::new(cfg, &mut rng).unwrap_err(),
            TsError::HeadDimMismatch { .. }
        ));
    }
}

//! ViT encoder: a stack of `depth` transformer blocks followed by a
//! final layer normalisation.
//!
//! The encoder operates on flat `[n_tokens, embed_dim]` tensors and applies
//! each `ViTBlock` sequentially with skip connections already handled inside
//! each block.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    vit::vit_block::{ViTBlock, ViTBlockConfig, layer_norm},
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the ViT encoder stack.
#[derive(Debug, Clone, PartialEq)]
pub struct ViTEncoderConfig {
    /// Shared configuration for every transformer block.
    pub block_cfg: ViTBlockConfig,
    /// Number of transformer blocks (encoder depth).
    pub depth: usize,
}

impl ViTEncoderConfig {
    /// Create and validate a `ViTEncoderConfig`.
    ///
    /// # Errors
    /// Propagates errors from `ViTBlockConfig::new` (embed/head validation).
    /// Also returns `Internal` if `depth == 0`.
    pub fn new(
        embed_dim: usize,
        n_heads: usize,
        mlp_ratio: usize,
        depth: usize,
    ) -> VisionResult<Self> {
        if depth == 0 {
            return Err(VisionError::Internal("encoder depth must be > 0".into()));
        }
        let block_cfg = ViTBlockConfig::new(embed_dim, n_heads, mlp_ratio)?;
        Ok(Self { block_cfg, depth })
    }
}

// ─── ViTEncoder ──────────────────────────────────────────────────────────────

/// Encoder stack: `depth` ViT blocks + a final layer norm.
pub struct ViTEncoder {
    /// The individual transformer blocks.
    pub blocks: Vec<ViTBlock>,
    /// Final LayerNorm scale: `[embed_dim]`.
    pub final_ln_weight: Vec<f32>,
    /// Final LayerNorm bias: `[embed_dim]`.
    pub final_ln_bias: Vec<f32>,
}

impl ViTEncoder {
    /// Construct the encoder: `depth` blocks with independent weight
    /// initialisations from the shared `rng`.
    ///
    /// The final layer-norm is initialised with weight=1, bias=0.
    pub fn new(cfg: ViTEncoderConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let e = cfg.block_cfg.embed_dim;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for _ in 0..cfg.depth {
            blocks.push(ViTBlock::new(cfg.block_cfg.clone(), rng));
        }
        let final_ln_weight = vec![1.0f32; e];
        let final_ln_bias = vec![0.0f32; e];
        Ok(Self {
            blocks,
            final_ln_weight,
            final_ln_bias,
        })
    }

    /// Forward pass through all blocks then the final layer norm.
    ///
    /// `tokens`: flat `[n_tokens, embed_dim]`.
    /// Returns `[n_tokens, embed_dim]`.
    pub fn forward(&self, tokens: &[f32], n_tokens: usize) -> VisionResult<Vec<f32>> {
        let e = self
            .blocks
            .first()
            .map(|b| b.config.embed_dim)
            .ok_or_else(|| VisionError::Internal("encoder has no blocks".into()))?;

        if tokens.len() != n_tokens * e {
            return Err(VisionError::DimensionMismatch {
                expected: n_tokens * e,
                got: tokens.len(),
            });
        }
        if n_tokens == 0 {
            return Err(VisionError::EmptyInput("tokens"));
        }

        // Apply each block sequentially
        let mut h: Vec<f32> = tokens.to_vec();
        for block in &self.blocks {
            h = block.forward(&h, n_tokens)?;
        }

        // Final layer norm
        let out = layer_norm(
            &h,
            &self.final_ln_weight,
            &self.final_ln_bias,
            n_tokens,
            e,
            1e-5,
        );
        Ok(out)
    }

    /// Embedding dimension (read from the first block config).
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        self.blocks.first().map_or(0, |b| b.config.embed_dim)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_enc(depth: usize) -> ViTEncoder {
        let cfg = ViTEncoderConfig::new(64, 4, 4, depth).expect("valid encoder config");
        let mut rng = LcgRng::new(42);
        ViTEncoder::new(cfg, &mut rng).expect("encoder created")
    }

    // ── Config ────────────────────────────────────────────────────────────────

    #[test]
    fn config_valid() {
        let cfg = ViTEncoderConfig::new(64, 4, 4, 3).expect("valid");
        assert_eq!(cfg.depth, 3);
        assert_eq!(cfg.block_cfg.embed_dim, 64);
    }

    #[test]
    fn config_depth_zero_errors() {
        let r = ViTEncoderConfig::new(64, 4, 4, 0);
        assert!(matches!(r, Err(VisionError::Internal(_))));
    }

    #[test]
    fn config_propagates_block_error() {
        // embed_dim not divisible by n_heads
        let r = ViTEncoderConfig::new(65, 4, 4, 2);
        assert!(matches!(r, Err(VisionError::HeadDimMismatch { .. })));
    }

    // ── Forward shape ─────────────────────────────────────────────────────────

    #[test]
    fn depth1_output_shape() {
        let enc = make_enc(1);
        let e = enc.embed_dim();
        let n_tokens = 17;
        let tokens = vec![0.1f32; n_tokens * e];
        let out = enc.forward(&tokens, n_tokens).expect("forward ok");
        assert_eq!(out.len(), n_tokens * e);
    }

    #[test]
    fn depth2_output_shape() {
        let enc = make_enc(2);
        let e = enc.embed_dim();
        let n_tokens = 17;
        let tokens = vec![0.1f32; n_tokens * e];
        let out = enc.forward(&tokens, n_tokens).expect("forward ok");
        assert_eq!(out.len(), n_tokens * e);
    }

    #[test]
    fn depth4_output_shape() {
        let enc = make_enc(4);
        let e = enc.embed_dim();
        let n_tokens = 9;
        let mut rng = LcgRng::new(11);
        let mut tokens = vec![0.0f32; n_tokens * e];
        rng.fill_normal(&mut tokens);
        let out = enc.forward(&tokens, n_tokens).expect("forward ok");
        assert_eq!(out.len(), n_tokens * e);
    }

    // ── Finite outputs ────────────────────────────────────────────────────────

    #[test]
    fn output_finite_random_input() {
        let enc = make_enc(2);
        let e = enc.embed_dim();
        let n_tokens = 17;
        let mut rng = LcgRng::new(7);
        let mut tokens = vec![0.0f32; n_tokens * e];
        rng.fill_normal(&mut tokens);
        let out = enc.forward(&tokens, n_tokens).expect("forward ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite encoder output"
        );
    }

    // ── Final LN runs ─────────────────────────────────────────────────────────

    #[test]
    fn final_ln_weight_bias_correct_size() {
        let enc = make_enc(1);
        assert_eq!(enc.final_ln_weight.len(), enc.embed_dim());
        assert_eq!(enc.final_ln_bias.len(), enc.embed_dim());
    }

    #[test]
    fn final_ln_weight_initialised_one() {
        let enc = make_enc(1);
        assert!(enc.final_ln_weight.iter().all(|&v| (v - 1.0).abs() < 1e-9));
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn dimension_mismatch_errors() {
        let enc = make_enc(1);
        let e = enc.embed_dim();
        // Wrong token count (n_tokens=5 but slice says 3 tokens)
        let tokens = vec![0.0f32; 3 * e];
        let r = enc.forward(&tokens, 5);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn empty_tokens_errors() {
        let enc = make_enc(1);
        let tokens: Vec<f32> = vec![];
        let r = enc.forward(&tokens, 0);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }

    // ── Block count ───────────────────────────────────────────────────────────

    #[test]
    fn correct_number_of_blocks() {
        for d in [1, 2, 4, 6, 12] {
            let enc = make_enc(d);
            assert_eq!(enc.blocks.len(), d, "wrong block count for depth={d}");
        }
    }
}

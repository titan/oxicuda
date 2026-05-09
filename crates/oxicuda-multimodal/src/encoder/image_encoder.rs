//! ViT-based image encoder with CLS-pool.
//!
//! Patchifies an image, embeds patches linearly, prepends a CLS token,
//! adds positional embeddings, and applies N transformer blocks.
//! The CLS token at position 0 is returned as the image representation.

use crate::cross_attn::cross_attention::{
    CrossAttention, CrossAttnConfig, CrossAttnWeights, softmax_rows_inplace,
};
use crate::cross_attn::self_cross_block::LayerNorm;
use crate::error::{MmResult, MultiModalError};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the ViT image encoder.
#[derive(Debug, Clone)]
pub struct ViTEncoderConfig {
    /// Image height and width in pixels (square image assumed).
    pub img_size: usize,
    /// Patch height and width in pixels (square patch assumed).
    pub patch_size: usize,
    /// Number of image channels (1 or 3).
    pub n_channels: usize,
    /// Model hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Feed-forward intermediate dimension.
    pub d_ff: usize,
}

impl ViTEncoderConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            img_size: 32,
            patch_size: 4,
            n_channels: 3,
            d_model: 8,
            n_heads: 2,
            n_layers: 2,
            d_ff: 16,
        }
    }

    /// Number of patches per spatial dimension.
    #[must_use]
    pub fn n_patches_side(&self) -> usize {
        self.img_size / self.patch_size
    }

    /// Total number of patches.
    #[must_use]
    pub fn n_patches(&self) -> usize {
        self.n_patches_side() * self.n_patches_side()
    }

    /// Patch embedding input dimension: `patch_size² × n_channels`.
    #[must_use]
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.n_channels
    }

    /// Validate configuration.
    pub fn validate(&self) -> MmResult<()> {
        if self.img_size == 0 || self.patch_size == 0 || self.img_size % self.patch_size != 0 {
            return Err(MultiModalError::InvalidPatchCount { n_patches: 0 });
        }
        if self.d_model == 0 || self.d_model % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: self.d_model,
            });
        }
        if self.n_layers == 0 {
            return Err(MultiModalError::InvalidLayerCount);
        }
        Ok(())
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Weights for a single ViT transformer block.
#[derive(Debug, Clone)]
pub struct ViTBlockWeights {
    /// Self-attention weights.
    pub attn: CrossAttnWeights,
    /// FFN W1: `[d_model × d_ff]`.
    pub ffn_w1: Vec<f32>,
    /// FFN b1: `[d_ff]`.
    pub ffn_b1: Vec<f32>,
    /// FFN W2: `[d_ff × d_model]`.
    pub ffn_w2: Vec<f32>,
    /// FFN b2: `[d_model]`.
    pub ffn_b2: Vec<f32>,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
}

impl ViTBlockWeights {
    #[must_use]
    pub fn zeros(cfg: &ViTEncoderConfig) -> Self {
        let d = cfg.d_model;
        let f = cfg.d_ff;
        let attn_cfg = CrossAttnConfig {
            n_heads: cfg.n_heads,
            d_model: d,
            d_k: d / cfg.n_heads,
            d_v: d / cfg.n_heads,
            dropout_rate: 0.0,
        };
        Self {
            attn: CrossAttnWeights::zeros(&attn_cfg),
            ffn_w1: vec![0.0_f32; d * f],
            ffn_b1: vec![0.0_f32; f],
            ffn_w2: vec![0.0_f32; f * d],
            ffn_b2: vec![0.0_f32; d],
            ln1: LayerNorm::ones(d),
            ln2: LayerNorm::ones(d),
        }
    }
}

/// Full ViT encoder weights.
#[derive(Debug, Clone)]
pub struct ViTEncoderWeights {
    /// Patch embedding weight: `[patch_dim × d_model]`.
    pub patch_embed: Vec<f32>,
    /// Patch embedding bias: `[d_model]`.
    pub patch_bias: Vec<f32>,
    /// CLS token embedding: `[d_model]`.
    pub cls_token: Vec<f32>,
    /// Positional embedding table: `[(1 + n_patches) × d_model]`.
    pub pos_embed: Vec<f32>,
    /// Transformer block weights.
    pub blocks: Vec<ViTBlockWeights>,
    /// Final layer norm weight: `[d_model]`.
    pub final_ln_weight: Vec<f32>,
    /// Final layer norm bias: `[d_model]`.
    pub final_ln_bias: Vec<f32>,
}

impl ViTEncoderWeights {
    /// Create with zeros.
    #[must_use]
    pub fn zeros(cfg: &ViTEncoderConfig) -> Self {
        let d = cfg.d_model;
        let n_pos = 1 + cfg.n_patches();
        Self {
            patch_embed: vec![0.0_f32; cfg.patch_dim() * d],
            patch_bias: vec![0.0_f32; d],
            cls_token: vec![0.0_f32; d],
            pos_embed: vec![0.0_f32; n_pos * d],
            blocks: (0..cfg.n_layers)
                .map(|_| ViTBlockWeights::zeros(cfg))
                .collect(),
            final_ln_weight: vec![1.0_f32; d],
            final_ln_bias: vec![0.0_f32; d],
        }
    }
}

// ─── ViTEncoder ──────────────────────────────────────────────────────────────

/// ViT-based image encoder.
pub struct ViTEncoder;

impl ViTEncoder {
    /// Encode an image to a CLS-pooled `[d_model]` vector.
    ///
    /// `image`: flat `[n_channels × img_size × img_size]` (CHW format).
    pub fn forward(
        image: &[f32],
        cfg: &ViTEncoderConfig,
        weights: &ViTEncoderWeights,
    ) -> MmResult<Vec<f32>> {
        cfg.validate()?;

        let expected = cfg.n_channels * cfg.img_size * cfg.img_size;
        if image.len() != expected {
            return Err(MultiModalError::DimensionMismatch {
                expected,
                got: image.len(),
            });
        }

        let d = cfg.d_model;
        let patch_dim = cfg.patch_dim();
        let n_patches = cfg.n_patches();
        let n_side = cfg.n_patches_side();
        let ps = cfg.patch_size;

        // Extract patches in row-major order, CHW → patch embedding
        // For each patch position (py, px), extract all channels
        let mut patches = vec![0.0_f32; n_patches * patch_dim];
        for py in 0..n_side {
            for px in 0..n_side {
                let patch_idx = py * n_side + px;
                let mut pi = 0;
                for c in 0..cfg.n_channels {
                    for ky in 0..ps {
                        for kx in 0..ps {
                            let y = py * ps + ky;
                            let x = px * ps + kx;
                            // CHW: index = c * img_size² + y * img_size + x
                            patches[patch_idx * patch_dim + pi] =
                                image[c * cfg.img_size * cfg.img_size + y * cfg.img_size + x];
                            pi += 1;
                        }
                    }
                }
            }
        }

        // Patch linear embedding: [n_patches × d_model]
        let mut patch_embeddings = vec![0.0_f32; n_patches * d];
        for p in 0..n_patches {
            for di in 0..d {
                let mut acc = weights.patch_bias[di];
                for pi in 0..patch_dim {
                    acc += patches[p * patch_dim + pi] * weights.patch_embed[pi * d + di];
                }
                patch_embeddings[p * d + di] = acc;
            }
        }

        // Prepend CLS token → [(1 + n_patches) × d_model]
        let n_tokens = 1 + n_patches;
        let mut tokens = vec![0.0_f32; n_tokens * d];
        tokens[..d].copy_from_slice(&weights.cls_token);
        tokens[d..].copy_from_slice(&patch_embeddings);

        // Add positional embeddings
        for i in 0..n_tokens {
            for di in 0..d {
                tokens[i * d + di] += weights.pos_embed[i * d + di];
            }
        }

        // Apply transformer blocks
        for block_w in &weights.blocks {
            tokens = vit_block_forward(&tokens, n_tokens, cfg, block_w)?;
        }

        // Final layer norm
        let ln = LayerNorm {
            weight: weights.final_ln_weight.clone(),
            bias: weights.final_ln_bias.clone(),
            d_model: d,
        };
        tokens = ln.forward(&tokens, n_tokens)?;

        // CLS token = position 0
        let cls = tokens[..d].to_vec();
        Ok(cls)
    }
}

/// Apply a single ViT transformer block (pre-norm style).
fn vit_block_forward(
    input: &[f32],
    n_tokens: usize,
    cfg: &ViTEncoderConfig,
    w: &ViTBlockWeights,
) -> MmResult<Vec<f32>> {
    let d = cfg.d_model;
    let h = cfg.n_heads;
    let d_k = d / h;
    let f = cfg.d_ff;

    // Pre-norm + self-attention + residual
    let normed1 = w.ln1.forward(input, n_tokens)?;
    let attn_cfg = CrossAttnConfig {
        n_heads: h,
        d_model: d,
        d_k,
        d_v: d_k,
        dropout_rate: 0.0,
    };
    let attn = CrossAttention::with_weights(attn_cfg, w.attn.clone());
    let sa_out = attn.forward(&normed1, &normed1, &normed1, n_tokens, n_tokens)?;
    let mut x: Vec<f32> = input
        .iter()
        .zip(sa_out.iter())
        .map(|(a, b)| a + b)
        .collect();

    // Pre-norm + FFN + residual
    let normed2 = w.ln2.forward(&x, n_tokens)?;
    let mut hidden = vec![0.0_f32; n_tokens * f];
    for s in 0..n_tokens {
        for fi in 0..f {
            let mut acc = w.ffn_b1[fi];
            for di in 0..d {
                acc += normed2[s * d + di] * w.ffn_w1[di * f + fi];
            }
            hidden[s * f + fi] = vit_gelu(acc);
        }
    }
    let mut ffn_out = vec![0.0_f32; n_tokens * d];
    for s in 0..n_tokens {
        for di in 0..d {
            let mut acc = w.ffn_b2[di];
            for fi in 0..f {
                acc += hidden[s * f + fi] * w.ffn_w2[fi * d + di];
            }
            ffn_out[s * d + di] = acc;
        }
    }
    for (xi, fi) in x.iter_mut().zip(ffn_out.iter()) {
        *xi += fi;
    }

    Ok(x)
}

#[inline]
fn vit_gelu(x: f32) -> f32 {
    let _ = softmax_rows_inplace; // bring into scope to avoid dead-code warning
    let k = 0.044_715_f32;
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + k * x.powi(3))).tanh())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vit_config_tiny_valid() {
        let cfg = ViTEncoderConfig::tiny();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.n_patches(), 64); // (32/4)^2
        assert_eq!(cfg.patch_dim(), 48); // 4*4*3
    }

    #[test]
    fn vit_config_invalid_patch_size() {
        let mut cfg = ViTEncoderConfig::tiny();
        cfg.patch_size = 7; // 32 % 7 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn vit_encoder_output_shape() {
        let cfg = ViTEncoderConfig::tiny();
        let weights = ViTEncoderWeights::zeros(&cfg);
        let image = vec![0.5_f32; 3 * 32 * 32];
        let out = ViTEncoder::forward(&image, &cfg, &weights).unwrap();
        assert_eq!(out.len(), cfg.d_model);
    }

    #[test]
    fn vit_encoder_output_finite() {
        let cfg = ViTEncoderConfig::tiny();
        let mut weights = ViTEncoderWeights::zeros(&cfg);
        // Set non-trivial patch embed (small random-ish)
        for (i, w) in weights.patch_embed.iter_mut().enumerate() {
            *w = (i as f32 * 0.01).sin() * 0.1;
        }
        weights.cls_token = vec![0.1_f32; cfg.d_model];
        let image: Vec<f32> = (0..3 * 32 * 32).map(|i| (i as f32 * 0.001).sin()).collect();
        let out = ViTEncoder::forward(&image, &cfg, &weights).unwrap();
        assert_eq!(out.len(), cfg.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn vit_encoder_zero_weights_zero_output() {
        let cfg = ViTEncoderConfig::tiny();
        let weights = ViTEncoderWeights::zeros(&cfg);
        // With zero patch embed and zero CLS token and zero pos embed,
        // all tokens start at 0, and zero attention weights keep them at 0.
        let image = vec![1.0_f32; 3 * 32 * 32];
        let out = ViTEncoder::forward(&image, &cfg, &weights).unwrap();
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }

    #[test]
    fn vit_encoder_wrong_image_size() {
        let cfg = ViTEncoderConfig::tiny();
        let weights = ViTEncoderWeights::zeros(&cfg);
        let image = vec![0.0_f32; 100]; // wrong size
        let err = ViTEncoder::forward(&image, &cfg, &weights).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }
}

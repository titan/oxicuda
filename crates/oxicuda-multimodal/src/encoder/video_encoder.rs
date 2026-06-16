//! Temporal ViT video encoder.
//!
//! Applies a spatial ViT to each frame independently, then runs temporal
//! self-attention over the per-frame CLS tokens, then mean-pools over frames.

use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::self_cross_block::LayerNorm;
use crate::encoder::image_encoder::{ViTEncoder, ViTEncoderConfig, ViTEncoderWeights};
use crate::error::{MmResult, MultiModalError};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the temporal ViT video encoder.
#[derive(Debug, Clone)]
pub struct VideoEncoderConfig {
    /// Spatial ViT configuration (applied per frame).
    pub spatial: ViTEncoderConfig,
    /// Number of temporal attention heads (over frame tokens).
    pub temporal_heads: usize,
    /// Number of temporal transformer layers.
    pub temporal_layers: usize,
    /// Temporal FFN intermediate dimension.
    pub temporal_d_ff: usize,
}

impl VideoEncoderConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        let spatial = ViTEncoderConfig::tiny();
        let d = spatial.d_model;
        Self {
            temporal_heads: 2,
            temporal_layers: 1,
            temporal_d_ff: d * 2,
            spatial,
        }
    }

    /// Model dimension (inherited from spatial ViT).
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.spatial.d_model
    }

    /// Validate configuration.
    pub fn validate(&self) -> MmResult<()> {
        self.spatial.validate()?;
        let d = self.spatial.d_model;
        if d % self.temporal_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.temporal_heads,
                d_model: d,
            });
        }
        if self.temporal_layers == 0 {
            return Err(MultiModalError::InvalidLayerCount);
        }
        Ok(())
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Weights for one temporal transformer block.
#[derive(Debug, Clone)]
pub struct TemporalBlockWeights {
    pub attn: CrossAttnWeights,
    pub ffn_w1: Vec<f32>,
    pub ffn_b1: Vec<f32>,
    pub ffn_w2: Vec<f32>,
    pub ffn_b2: Vec<f32>,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
}

impl TemporalBlockWeights {
    #[must_use]
    pub fn zeros(cfg: &VideoEncoderConfig) -> Self {
        let d = cfg.d_model();
        let f = cfg.temporal_d_ff;
        let h = cfg.temporal_heads;
        let attn_cfg = CrossAttnConfig {
            n_heads: h,
            d_model: d,
            d_k: d / h,
            d_v: d / h,
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

/// Full video encoder weights.
#[derive(Debug, Clone)]
pub struct VideoEncoderWeights {
    /// Spatial ViT weights (shared across frames).
    pub spatial: ViTEncoderWeights,
    /// Temporal positional embedding: `[max_frames × d_model]`.
    pub temporal_pos_embed: Vec<f32>,
    /// Temporal transformer blocks.
    pub temporal_blocks: Vec<TemporalBlockWeights>,
    /// Final layer norm weight.
    pub final_ln_weight: Vec<f32>,
    /// Final layer norm bias.
    pub final_ln_bias: Vec<f32>,
}

impl VideoEncoderWeights {
    /// Create with zeros; `max_frames` limits temporal positional embeddings.
    #[must_use]
    pub fn zeros(cfg: &VideoEncoderConfig, max_frames: usize) -> Self {
        let d = cfg.d_model();
        Self {
            spatial: ViTEncoderWeights::zeros(&cfg.spatial),
            temporal_pos_embed: vec![0.0_f32; max_frames * d],
            temporal_blocks: (0..cfg.temporal_layers)
                .map(|_| TemporalBlockWeights::zeros(cfg))
                .collect(),
            final_ln_weight: vec![1.0_f32; d],
            final_ln_bias: vec![0.0_f32; d],
        }
    }
}

// ─── VideoEncoder ────────────────────────────────────────────────────────────

/// Temporal ViT video encoder.
pub struct VideoEncoder;

impl VideoEncoder {
    /// Encode a video to a `[d_model]` mean-pooled representation.
    ///
    /// `frames`: `[n_frames × n_channels × img_size × img_size]` flat buffer.
    ///
    /// Returns `[d_model]`.
    pub fn forward(
        frames: &[f32],
        n_frames: usize,
        cfg: &VideoEncoderConfig,
        weights: &VideoEncoderWeights,
    ) -> MmResult<Vec<f32>> {
        cfg.validate()?;
        if n_frames == 0 {
            return Err(MultiModalError::EmptyInput);
        }

        let frame_size = cfg.spatial.n_channels * cfg.spatial.img_size * cfg.spatial.img_size;
        if frames.len() != n_frames * frame_size {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_frames * frame_size,
                got: frames.len(),
            });
        }

        let d = cfg.d_model();

        // Apply spatial ViT per frame → per-frame CLS token [n_frames × d_model]
        let mut frame_tokens = vec![0.0_f32; n_frames * d];
        for t in 0..n_frames {
            let frame = &frames[t * frame_size..(t + 1) * frame_size];
            let cls = ViTEncoder::forward(frame, &cfg.spatial, &weights.spatial)?;
            frame_tokens[t * d..(t + 1) * d].copy_from_slice(&cls);
        }

        // Add temporal positional embeddings
        let max_frames = weights.temporal_pos_embed.len() / d;
        for t in 0..n_frames {
            let pos_t = t.min(max_frames.saturating_sub(1));
            for di in 0..d {
                frame_tokens[t * d + di] += weights.temporal_pos_embed[pos_t * d + di];
            }
        }

        // Temporal transformer blocks
        for block_w in &weights.temporal_blocks {
            frame_tokens = temporal_block_forward(&frame_tokens, n_frames, cfg, block_w)?;
        }

        // Final layer norm
        let ln = LayerNorm {
            weight: weights.final_ln_weight.clone(),
            bias: weights.final_ln_bias.clone(),
            d_model: d,
        };
        frame_tokens = ln.forward(&frame_tokens, n_frames)?;

        // Mean pool over frames
        let mut mean = vec![0.0_f32; d];
        for t in 0..n_frames {
            for di in 0..d {
                mean[di] += frame_tokens[t * d + di];
            }
        }
        let inv_n = 1.0 / n_frames as f32;
        for v in mean.iter_mut() {
            *v *= inv_n;
        }

        Ok(mean)
    }
}

/// Apply a single temporal transformer block (pre-norm).
fn temporal_block_forward(
    input: &[f32],
    n_frames: usize,
    cfg: &VideoEncoderConfig,
    w: &TemporalBlockWeights,
) -> MmResult<Vec<f32>> {
    let d = cfg.d_model();
    let h = cfg.temporal_heads;
    let d_k = d / h;
    let f = cfg.temporal_d_ff;

    // Self-attention over frame dimension
    let normed1 = w.ln1.forward(input, n_frames)?;
    let attn_cfg = CrossAttnConfig {
        n_heads: h,
        d_model: d,
        d_k,
        d_v: d_k,
        dropout_rate: 0.0,
    };
    let attn = CrossAttention::with_weights(attn_cfg, w.attn.clone());
    let sa_out = attn.forward(&normed1, &normed1, &normed1, n_frames, n_frames)?;

    let mut x: Vec<f32> = input
        .iter()
        .zip(sa_out.iter())
        .map(|(a, b)| a + b)
        .collect();

    // FFN
    let normed2 = w.ln2.forward(&x, n_frames)?;
    let mut hidden = vec![0.0_f32; n_frames * f];
    for t in 0..n_frames {
        for fi in 0..f {
            let mut acc = w.ffn_b1[fi];
            for di in 0..d {
                acc += normed2[t * d + di] * w.ffn_w1[di * f + fi];
            }
            hidden[t * f + fi] = temporal_gelu(acc);
        }
    }
    let mut ffn_out = vec![0.0_f32; n_frames * d];
    for t in 0..n_frames {
        for di in 0..d {
            let mut acc = w.ffn_b2[di];
            for fi in 0..f {
                acc += hidden[t * f + fi] * w.ffn_w2[fi * d + di];
            }
            ffn_out[t * d + di] = acc;
        }
    }
    for (xi, fi) in x.iter_mut().zip(ffn_out.iter()) {
        *xi += fi;
    }

    Ok(x)
}

#[inline]
fn temporal_gelu(x: f32) -> f32 {
    let k = 0.044_715_f32;
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + k * x.powi(3))).tanh())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_config_tiny_valid() {
        let cfg = VideoEncoderConfig::tiny();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn video_encoder_output_shape() {
        let cfg = VideoEncoderConfig::tiny();
        let weights = VideoEncoderWeights::zeros(&cfg, 16);
        let frame_size = 3 * 32 * 32;
        let n_frames = 4;
        let frames = vec![0.1_f32; n_frames * frame_size];
        let out = VideoEncoder::forward(&frames, n_frames, &cfg, &weights)
            .expect("forward should succeed");
        assert_eq!(out.len(), cfg.d_model());
    }

    #[test]
    fn video_encoder_output_finite() {
        let cfg = VideoEncoderConfig::tiny();
        let weights = VideoEncoderWeights::zeros(&cfg, 16);
        let frame_size = 3 * 32 * 32;
        let n_frames = 3;
        let frames: Vec<f32> = (0..n_frames * frame_size)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let out = VideoEncoder::forward(&frames, n_frames, &cfg, &weights)
            .expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn video_encoder_empty_frames_error() {
        let cfg = VideoEncoderConfig::tiny();
        let weights = VideoEncoderWeights::zeros(&cfg, 8);
        let err = VideoEncoder::forward(&[], 0, &cfg, &weights).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    #[test]
    fn video_encoder_wrong_frame_buffer_size() {
        let cfg = VideoEncoderConfig::tiny();
        let weights = VideoEncoderWeights::zeros(&cfg, 8);
        // n_frames=2 but buffer has only 100 elements
        let frames = vec![0.0_f32; 100];
        let err = VideoEncoder::forward(&frames, 2, &cfg, &weights).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn video_encoder_single_frame() {
        let cfg = VideoEncoderConfig::tiny();
        let weights = VideoEncoderWeights::zeros(&cfg, 8);
        let frame_size = 3 * 32 * 32;
        let frames = vec![0.5_f32; frame_size];
        let out =
            VideoEncoder::forward(&frames, 1, &cfg, &weights).expect("forward should succeed");
        assert_eq!(out.len(), cfg.d_model());
    }
}

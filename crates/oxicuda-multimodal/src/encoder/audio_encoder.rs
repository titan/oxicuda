//! Conformer-style audio encoder with statistics pooling.
//!
//! Processes mel-spectrogram features through a stack of conformer blocks,
//! then pools to `[mean ‖ std]` producing a `[2 * d_model]` embedding.

use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::self_cross_block::LayerNorm;
use crate::error::{MmResult, MultiModalError};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the audio Conformer encoder.
#[derive(Debug, Clone)]
pub struct AudioEncoderConfig {
    /// Number of mel filter-bank bins.
    pub n_mels: usize,
    /// Model hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of conformer layers.
    pub n_layers: usize,
    /// Convolution kernel size for the conv module.
    pub kernel_size: usize,
    /// Feed-forward intermediate dimension.
    pub d_ff: usize,
}

impl AudioEncoderConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            n_mels: 16,
            d_model: 8,
            n_heads: 2,
            n_layers: 2,
            kernel_size: 3,
            d_ff: 16,
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> MmResult<()> {
        if self.d_model == 0 || self.d_model % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: self.d_model,
            });
        }
        if self.n_layers == 0 {
            return Err(MultiModalError::InvalidLayerCount);
        }
        if self.n_mels == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        Ok(())
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Weights for a single conformer block.
///
/// Simplified conformer (without depthwise conv for pure-CPU efficiency):
/// `FFN → MHSA → Conv-module (depthwise) → FFN → LN`
#[derive(Debug, Clone)]
pub struct ConformerBlockWeights {
    /// Multi-head self-attention weights.
    pub attn: CrossAttnWeights,
    /// FFN1 W1: `[d_model × d_ff]`.
    pub ffn1_w1: Vec<f32>,
    /// FFN1 b1: `[d_ff]`.
    pub ffn1_b1: Vec<f32>,
    /// FFN1 W2: `[d_ff × d_model]`.
    pub ffn1_w2: Vec<f32>,
    /// FFN1 b2: `[d_model]`.
    pub ffn1_b2: Vec<f32>,
    /// FFN2 W1: `[d_model × d_ff]`.
    pub ffn2_w1: Vec<f32>,
    /// FFN2 b1: `[d_ff]`.
    pub ffn2_b1: Vec<f32>,
    /// FFN2 W2: `[d_ff × d_model]`.
    pub ffn2_w2: Vec<f32>,
    /// FFN2 b2: `[d_model]`.
    pub ffn2_b2: Vec<f32>,
    /// Depthwise conv weight: `[d_model × kernel_size]`.
    pub conv_dw: Vec<f32>,
    /// Depthwise conv bias: `[d_model]`.
    pub conv_dw_bias: Vec<f32>,
    pub ln_attn: LayerNorm,
    pub ln_ffn1: LayerNorm,
    pub ln_ffn2: LayerNorm,
    pub ln_conv: LayerNorm,
    pub ln_final: LayerNorm,
}

impl ConformerBlockWeights {
    #[must_use]
    pub fn zeros(cfg: &AudioEncoderConfig) -> Self {
        let d = cfg.d_model;
        let f = cfg.d_ff;
        let k = cfg.kernel_size;
        let attn_cfg = CrossAttnConfig {
            n_heads: cfg.n_heads,
            d_model: d,
            d_k: d / cfg.n_heads,
            d_v: d / cfg.n_heads,
            dropout_rate: 0.0,
        };
        Self {
            attn: CrossAttnWeights::zeros(&attn_cfg),
            ffn1_w1: vec![0.0_f32; d * f],
            ffn1_b1: vec![0.0_f32; f],
            ffn1_w2: vec![0.0_f32; f * d],
            ffn1_b2: vec![0.0_f32; d],
            ffn2_w1: vec![0.0_f32; d * f],
            ffn2_b1: vec![0.0_f32; f],
            ffn2_w2: vec![0.0_f32; f * d],
            ffn2_b2: vec![0.0_f32; d],
            conv_dw: vec![0.0_f32; d * k],
            conv_dw_bias: vec![0.0_f32; d],
            ln_attn: LayerNorm::ones(d),
            ln_ffn1: LayerNorm::ones(d),
            ln_ffn2: LayerNorm::ones(d),
            ln_conv: LayerNorm::ones(d),
            ln_final: LayerNorm::ones(d),
        }
    }
}

/// Full audio encoder weights.
#[derive(Debug, Clone)]
pub struct AudioEncoderWeights {
    /// Input linear projection: `[n_mels × d_model]`.
    pub input_proj: Vec<f32>,
    /// Input projection bias: `[d_model]`.
    pub input_proj_bias: Vec<f32>,
    /// Conformer block weights.
    pub blocks: Vec<ConformerBlockWeights>,
}

impl AudioEncoderWeights {
    #[must_use]
    pub fn zeros(cfg: &AudioEncoderConfig) -> Self {
        Self {
            input_proj: vec![0.0_f32; cfg.n_mels * cfg.d_model],
            input_proj_bias: vec![0.0_f32; cfg.d_model],
            blocks: (0..cfg.n_layers)
                .map(|_| ConformerBlockWeights::zeros(cfg))
                .collect(),
        }
    }
}

// ─── AudioEncoder ─────────────────────────────────────────────────────────────

/// Conformer-style audio encoder.
pub struct AudioEncoder;

impl AudioEncoder {
    /// Encode mel spectrogram features to `[2 * d_model]` stats-pooled embedding.
    ///
    /// `mel_features`: `[n_frames × n_mels]` row-major.
    ///
    /// Returns `[mean ‖ std]` = `[2 * d_model]`.
    pub fn forward(
        mel_features: &[f32],
        n_frames: usize,
        cfg: &AudioEncoderConfig,
        weights: &AudioEncoderWeights,
    ) -> MmResult<Vec<f32>> {
        cfg.validate()?;
        if n_frames == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if mel_features.len() != n_frames * cfg.n_mels {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_frames * cfg.n_mels,
                got: mel_features.len(),
            });
        }

        let d = cfg.d_model;

        // Linear projection: [n_frames × d_model]
        let mut hidden = vec![0.0_f32; n_frames * d];
        for t in 0..n_frames {
            for di in 0..d {
                let mut acc = weights.input_proj_bias[di];
                for m in 0..cfg.n_mels {
                    acc += mel_features[t * cfg.n_mels + m] * weights.input_proj[m * d + di];
                }
                hidden[t * d + di] = acc;
            }
        }

        // Conformer blocks
        for block_w in &weights.blocks {
            hidden = conformer_block_forward(&hidden, n_frames, cfg, block_w)?;
        }

        // Statistics pooling: compute [mean ‖ std] over temporal dimension
        let mut mean = vec![0.0_f32; d];
        for t in 0..n_frames {
            for di in 0..d {
                mean[di] += hidden[t * d + di];
            }
        }
        let inv_n = 1.0 / n_frames as f32;
        for v in mean.iter_mut() {
            *v *= inv_n;
        }

        let mut std_dev = vec![0.0_f32; d];
        for t in 0..n_frames {
            for di in 0..d {
                let diff = hidden[t * d + di] - mean[di];
                std_dev[di] += diff * diff;
            }
        }
        for v in std_dev.iter_mut() {
            *v = (*v * inv_n).sqrt();
        }

        let mut out = Vec::with_capacity(2 * d);
        out.extend_from_slice(&mean);
        out.extend_from_slice(&std_dev);
        Ok(out)
    }
}

/// Apply a single conformer block.
fn conformer_block_forward(
    input: &[f32],
    n_frames: usize,
    cfg: &AudioEncoderConfig,
    w: &ConformerBlockWeights,
) -> MmResult<Vec<f32>> {
    let d = cfg.d_model;
    let f = cfg.d_ff;
    let h = cfg.n_heads;
    let d_k = d / h;
    let ks = cfg.kernel_size;

    // Module 1: Half-step FFN1
    let normed1 = w.ln_ffn1.forward(input, n_frames)?;
    let ffn1_out = apply_ffn(
        &normed1, n_frames, d, f, &w.ffn1_w1, &w.ffn1_b1, &w.ffn1_w2, &w.ffn1_b2,
    )?;
    // half-step: x += 0.5 * ffn1_out
    let mut x: Vec<f32> = input
        .iter()
        .zip(ffn1_out.iter())
        .map(|(a, b)| a + 0.5 * b)
        .collect();

    // Module 2: Multi-head self-attention
    let normed_attn = w.ln_attn.forward(&x, n_frames)?;
    let attn_cfg = CrossAttnConfig {
        n_heads: h,
        d_model: d,
        d_k,
        d_v: d_k,
        dropout_rate: 0.0,
    };
    let attn = CrossAttention::with_weights(attn_cfg, w.attn.clone());
    let sa_out = attn.forward(&normed_attn, &normed_attn, &normed_attn, n_frames, n_frames)?;
    for (xi, si) in x.iter_mut().zip(sa_out.iter()) {
        *xi += si;
    }

    // Module 3: Depthwise convolution (1D, causal, per-channel)
    let normed_conv = w.ln_conv.forward(&x, n_frames)?;
    let pad = ks / 2;
    let mut conv_out = vec![0.0_f32; n_frames * d];
    for t in 0..n_frames {
        for di in 0..d {
            let mut acc = w.conv_dw_bias[di];
            for ki in 0..ks {
                let src_t = t as isize + ki as isize - pad as isize;
                if src_t >= 0 && (src_t as usize) < n_frames {
                    acc += normed_conv[src_t as usize * d + di] * w.conv_dw[di * ks + ki];
                }
            }
            conv_out[t * d + di] = swish(acc);
        }
    }
    for (xi, ci) in x.iter_mut().zip(conv_out.iter()) {
        *xi += ci;
    }

    // Module 4: Half-step FFN2
    let normed2 = w.ln_ffn2.forward(&x, n_frames)?;
    let ffn2_out = apply_ffn(
        &normed2, n_frames, d, f, &w.ffn2_w1, &w.ffn2_b1, &w.ffn2_w2, &w.ffn2_b2,
    )?;
    for (xi, fi) in x.iter_mut().zip(ffn2_out.iter()) {
        *xi += 0.5 * fi;
    }

    // Final layer norm
    x = w.ln_final.forward(&x, n_frames)?;

    Ok(x)
}

/// Apply a 2-layer FFN with SiLU (swish) activation.
fn apply_ffn(
    input: &[f32],
    seq: usize,
    d: usize,
    f: usize,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> MmResult<Vec<f32>> {
    let mut hidden = vec![0.0_f32; seq * f];
    for s in 0..seq {
        for fi in 0..f {
            let mut acc = b1[fi];
            for di in 0..d {
                acc += input[s * d + di] * w1[di * f + fi];
            }
            hidden[s * f + fi] = swish(acc);
        }
    }
    let mut out = vec![0.0_f32; seq * d];
    for s in 0..seq {
        for di in 0..d {
            let mut acc = b2[di];
            for fi in 0..f {
                acc += hidden[s * f + fi] * w2[fi * d + di];
            }
            out[s * d + di] = acc;
        }
    }
    Ok(out)
}

/// SiLU / Swish activation: `x * sigmoid(x)`.
#[inline]
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_config_tiny_valid() {
        let cfg = AudioEncoderConfig::tiny();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn audio_encoder_output_shape() {
        let cfg = AudioEncoderConfig::tiny();
        let weights = AudioEncoderWeights::zeros(&cfg);
        let n_frames = 20;
        let mel = vec![0.5_f32; n_frames * cfg.n_mels];
        let out = AudioEncoder::forward(&mel, n_frames, &cfg, &weights).unwrap();
        assert_eq!(out.len(), 2 * cfg.d_model);
    }

    #[test]
    fn audio_encoder_output_finite() {
        let cfg = AudioEncoderConfig::tiny();
        let mut weights = AudioEncoderWeights::zeros(&cfg);
        // Non-trivial projection
        for (i, w) in weights.input_proj.iter_mut().enumerate() {
            *w = (i as f32 * 0.05).sin() * 0.1;
        }
        let n_frames = 10;
        let mel: Vec<f32> = (0..n_frames * cfg.n_mels)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let out = AudioEncoder::forward(&mel, n_frames, &cfg, &weights).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "output must be finite");
    }

    #[test]
    fn audio_encoder_stats_pool_shape() {
        let cfg = AudioEncoderConfig::tiny();
        let weights = AudioEncoderWeights::zeros(&cfg);
        let n_frames = 8;
        let mel = vec![1.0_f32; n_frames * cfg.n_mels];
        let out = AudioEncoder::forward(&mel, n_frames, &cfg, &weights).unwrap();
        // First half = mean, second half = std
        assert_eq!(out.len(), 2 * cfg.d_model);
        // std of constant signal = 0
        let std_part = &out[cfg.d_model..];
        for &v in std_part {
            assert!(v.abs() < 1e-5, "std of constant should be ~0: {v}");
        }
    }

    #[test]
    fn audio_encoder_empty_frames_error() {
        let cfg = AudioEncoderConfig::tiny();
        let weights = AudioEncoderWeights::zeros(&cfg);
        let err = AudioEncoder::forward(&[], 0, &cfg, &weights).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    #[test]
    fn audio_encoder_wrong_shape_error() {
        let cfg = AudioEncoderConfig::tiny();
        let weights = AudioEncoderWeights::zeros(&cfg);
        let mel = vec![0.0_f32; 10]; // wrong size
        let err = AudioEncoder::forward(&mel, 3, &cfg, &weights).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn swish_zero_is_zero() {
        assert!(swish(0.0).abs() < 1e-6);
    }

    #[test]
    fn swish_positive_positive() {
        // swish is positive for positive x
        assert!(swish(1.0) > 0.0);
    }

    #[test]
    fn swish_negative_small_negative() {
        // swish(-1) is slightly negative
        let v = swish(-1.0);
        assert!(v < 0.0 && v > -0.3);
    }
}

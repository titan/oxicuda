//! Alternating self-attention + cross-attention transformer block.
//!
//! Architecture (pre-norm):
//! ```text
//! x = LN1(x) → self_attn(x, x, x) + residual
//! x = LN2(x) → cross_attn(x, ctx, ctx) + residual
//! x = LN3(x) → FFN(x) + residual
//! ```

use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
use crate::error::{MmResult, MultiModalError};

// ─── LayerNorm ────────────────────────────────────────────────────────────────

/// Simple learnable layer normalisation over the last dimension.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Learnable scale (gamma), length = d_model.
    pub weight: Vec<f32>,
    /// Learnable bias (beta), length = d_model.
    pub bias: Vec<f32>,
    pub d_model: usize,
}

impl LayerNorm {
    /// Initialise with weight=1, bias=0 (standard initialisation).
    #[must_use]
    pub fn ones(d_model: usize) -> Self {
        Self {
            weight: vec![1.0_f32; d_model],
            bias: vec![0.0_f32; d_model],
            d_model,
        }
    }

    /// Normalise `input [seq × d_model]` in-place; returns normalised output.
    pub fn forward(&self, input: &[f32], seq: usize) -> MmResult<Vec<f32>> {
        let d = self.d_model;
        if input.len() != seq * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: seq * d,
                got: input.len(),
            });
        }
        let mut out = vec![0.0_f32; seq * d];
        for s in 0..seq {
            let row = &input[s * d..(s + 1) * d];
            let mean = row.iter().sum::<f32>() / d as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
            let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
            for i in 0..d {
                out[s * d + i] = (row[i] - mean) * inv_std * self.weight[i] + self.bias[i];
            }
        }
        Ok(out)
    }
}

// ─── Feed-Forward Network ────────────────────────────────────────────────────

/// Two-layer feed-forward network with GELU activation.
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// W1: [d_model × d_ff]
    pub w1: Vec<f32>,
    /// b1: \[d_ff\]
    pub b1: Vec<f32>,
    /// W2: [d_ff × d_model]
    pub w2: Vec<f32>,
    /// b2: \[d_model\]
    pub b2: Vec<f32>,
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForward {
    /// Create with zeros.
    #[must_use]
    pub fn zeros(d_model: usize, d_ff: usize) -> Self {
        Self {
            w1: vec![0.0_f32; d_model * d_ff],
            b1: vec![0.0_f32; d_ff],
            w2: vec![0.0_f32; d_ff * d_model],
            b2: vec![0.0_f32; d_model],
            d_model,
            d_ff,
        }
    }

    /// Create identity-like (pass-through): W1 maps d→d, W2 maps back.
    /// Only valid when `d_ff == d_model`.
    #[must_use]
    pub fn identity(d_model: usize) -> Self {
        let mut w = vec![0.0_f32; d_model * d_model];
        for i in 0..d_model {
            w[i * d_model + i] = 1.0;
        }
        Self {
            w1: w.clone(),
            b1: vec![0.0_f32; d_model],
            w2: w,
            b2: vec![0.0_f32; d_model],
            d_model,
            d_ff: d_model,
        }
    }

    /// Forward: `[seq × d_model]` → `[seq × d_model]`.
    pub fn forward(&self, input: &[f32], seq: usize) -> MmResult<Vec<f32>> {
        let d = self.d_model;
        let f = self.d_ff;
        // Hidden layer: x1 = GELU(input × W1 + b1)   [seq × d_ff]
        let mut hidden = vec![0.0_f32; seq * f];
        for s in 0..seq {
            for fi in 0..f {
                let mut acc = self.b1[fi];
                for di in 0..d {
                    acc += input[s * d + di] * self.w1[di * f + fi];
                }
                hidden[s * f + fi] = gelu(acc);
            }
        }
        // Output layer: out = hidden × W2 + b2   [seq × d_model]
        let mut out = vec![0.0_f32; seq * d];
        for s in 0..seq {
            for di in 0..d {
                let mut acc = self.b2[di];
                for fi in 0..f {
                    acc += hidden[s * f + fi] * self.w2[fi * d + di];
                }
                out[s * d + di] = acc;
            }
        }
        Ok(out)
    }
}

/// Approximate GELU: `x * Φ(x)` using the `tanh` approximation.
#[inline]
fn gelu(x: f32) -> f32 {
    let k = 0.044_715_f32;
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + k * x.powi(3));
    0.5 * x * (1.0 + inner.tanh())
}

// ─── SelfCrossBlock weights ──────────────────────────────────────────────────

/// Weights for a full self-cross block.
#[derive(Debug, Clone)]
pub struct SelfCrossBlockWeights {
    pub self_attn: CrossAttnWeights,
    pub cross_attn: CrossAttnWeights,
    pub ffn: FeedForward,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub ln3: LayerNorm,
}

impl SelfCrossBlockWeights {
    /// Create zero-initialised weights.
    #[must_use]
    pub fn zeros(cfg: &CrossAttnConfig) -> Self {
        let d = cfg.d_model;
        Self {
            self_attn: CrossAttnWeights::zeros(cfg),
            cross_attn: CrossAttnWeights::zeros(cfg),
            ffn: FeedForward::zeros(d, d * 4),
            ln1: LayerNorm::ones(d),
            ln2: LayerNorm::ones(d),
            ln3: LayerNorm::ones(d),
        }
    }
}

// ─── SelfCrossBlock ──────────────────────────────────────────────────────────

/// Alternating self-attention + cross-attention block.
///
/// Pre-norm layout:
/// 1. `y = self_attn(LN1(x), LN1(x), LN1(x)) + x`
/// 2. `y = cross_attn(LN2(y), LN2(ctx), LN2(ctx)) + y`
/// 3. `y = ffn(LN3(y)) + y`
pub struct SelfCrossBlock {
    pub self_attn: CrossAttention,
    pub cross_attn: CrossAttention,
    pub ffn: FeedForward,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub ln3: LayerNorm,
}

impl SelfCrossBlock {
    /// Create a new block.
    #[must_use]
    pub fn new(cfg: CrossAttnConfig) -> Self {
        let d = cfg.d_model;
        let weights = SelfCrossBlockWeights::zeros(&cfg);
        Self {
            self_attn: CrossAttention::with_weights(cfg.clone(), weights.self_attn),
            cross_attn: CrossAttention::with_weights(cfg, weights.cross_attn),
            ffn: FeedForward::zeros(d, d * 4),
            ln1: LayerNorm::ones(d),
            ln2: LayerNorm::ones(d),
            ln3: LayerNorm::ones(d),
        }
    }

    /// Create a block with explicit weights.
    #[must_use]
    pub fn with_weights(cfg: CrossAttnConfig, w: SelfCrossBlockWeights) -> Self {
        let self_attn = CrossAttention::with_weights(cfg.clone(), w.self_attn);
        let cross_attn = CrossAttention::with_weights(cfg, w.cross_attn);
        Self {
            self_attn,
            cross_attn,
            ffn: w.ffn,
            ln1: w.ln1,
            ln2: w.ln2,
            ln3: w.ln3,
        }
    }

    /// Forward pass.
    ///
    /// - `query_seq`: `[q_len × d_model]` — the query sequence (modality A).
    /// - `context_seq`: `[ctx_len × d_model]` — the context sequence (modality B).
    ///
    /// Returns `[q_len × d_model]`.
    pub fn forward(
        &self,
        query_seq: &[f32],
        context_seq: &[f32],
        q_len: usize,
        ctx_len: usize,
    ) -> MmResult<Vec<f32>> {
        let d = self.self_attn.cfg.d_model;

        // ── Step 1: Self-attention with pre-norm ─────────────────────────────
        let ln1_q = self.ln1.forward(query_seq, q_len)?;
        let self_out = self
            .self_attn
            .forward(&ln1_q, &ln1_q, &ln1_q, q_len, q_len)?;
        // Residual add
        let mut x: Vec<f32> = query_seq
            .iter()
            .zip(self_out.iter())
            .map(|(a, b)| a + b)
            .collect();

        // ── Step 2: Cross-attention over context ─────────────────────────────
        let ln2_q = self.ln2.forward(&x, q_len)?;
        let ln2_ctx = self.ln2.forward(context_seq, ctx_len)?;
        let cross_out = self
            .cross_attn
            .forward(&ln2_q, &ln2_ctx, &ln2_ctx, q_len, ctx_len)?;
        for (xi, ci) in x.iter_mut().zip(cross_out.iter()) {
            *xi += ci;
        }

        // ── Step 3: FFN with pre-norm ────────────────────────────────────────
        let ln3_x = self.ln3.forward(&x, q_len)?;
        let ffn_out = self.ffn.forward(&ln3_x, q_len)?;
        for (xi, fi) in x.iter_mut().zip(ffn_out.iter()) {
            *xi += fi;
        }

        // Verify output dimension
        if x.len() != q_len * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: q_len * d,
                got: x.len(),
            });
        }

        Ok(x)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_output_shape() {
        let ln = LayerNorm::ones(8);
        let input = vec![1.0_f32; 4 * 8];
        let out = ln.forward(&input, 4).expect("forward should succeed");
        assert_eq!(out.len(), 4 * 8);
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let ln = LayerNorm::ones(4);
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = ln.forward(&input, 1).expect("forward should succeed");
        let mean = out.iter().sum::<f32>() / 4.0;
        let var = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "mean={mean}");
        assert!((var - 1.0).abs() < 0.05, "var={var}");
    }

    #[test]
    fn layer_norm_dimension_mismatch() {
        let ln = LayerNorm::ones(8);
        let input = vec![0.0_f32; 3 * 9]; // wrong dim
        let err = ln.forward(&input, 3).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn feed_forward_output_shape() {
        let ffn = FeedForward::zeros(8, 16);
        let input = vec![1.0_f32; 3 * 8];
        let out = ffn.forward(&input, 3).expect("forward should succeed");
        assert_eq!(out.len(), 3 * 8);
    }

    #[test]
    fn gelu_zero_input() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_positive_input() {
        // GELU(1.0) ≈ 0.841
        assert!(gelu(1.0) > 0.8 && gelu(1.0) < 0.9);
    }

    #[test]
    fn self_cross_block_output_shape() {
        let cfg = CrossAttnConfig::tiny();
        let d = cfg.d_model;
        let block = SelfCrossBlock::new(cfg);

        let q = vec![0.1_f32; 4 * d];
        let ctx = vec![0.2_f32; 6 * d];
        let out = block
            .forward(&q, &ctx, 4, 6)
            .expect("forward should succeed");
        assert_eq!(out.len(), 4 * d);
    }

    #[test]
    fn self_cross_block_output_finite() {
        let cfg = CrossAttnConfig::tiny();
        let d = cfg.d_model;
        let block = SelfCrossBlock::new(cfg);

        let q = vec![0.1_f32; 3 * d];
        let ctx = vec![0.2_f32; 5 * d];
        let out = block
            .forward(&q, &ctx, 3, 5)
            .expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn self_cross_block_residual_preserves_scale() {
        // With zero attention weights, output ≈ input (residual dominates)
        let cfg = CrossAttnConfig::tiny();
        let d = cfg.d_model;
        let block = SelfCrossBlock::new(cfg); // zero attn weights

        let q: Vec<f32> = (0..(4 * d)).map(|i| (i as f32) * 0.01).collect();
        let ctx = vec![0.5_f32; 3 * d];
        let out = block
            .forward(&q, &ctx, 4, 3)
            .expect("forward should succeed");
        // Output should still contain the residual-added input signal
        assert_eq!(out.len(), 4 * d);
        // At least the norms should not blow up
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm.is_finite() && norm < 1e4);
    }
}

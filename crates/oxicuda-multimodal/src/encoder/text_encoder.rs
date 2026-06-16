//! BERT-style multi-layer transformer text encoder.
//!
//! Produces a CLS-pooled `[d_model]` embedding from token ID sequences.

use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::self_cross_block::LayerNorm;
use crate::error::{MmResult, MultiModalError};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the BERT-style text encoder.
#[derive(Debug, Clone)]
pub struct BertConfig {
    /// Vocabulary size (number of token types).
    pub vocab_size: usize,
    /// Maximum sequence length (for positional embeddings).
    pub max_seq_len: usize,
    /// Model hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer encoder layers.
    pub n_layers: usize,
    /// Feed-forward intermediate dimension.
    pub d_ff: usize,
}

impl BertConfig {
    /// Tiny preset for fast unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            vocab_size: 32,
            max_seq_len: 16,
            d_model: 8,
            n_heads: 2,
            n_layers: 2,
            d_ff: 16,
        }
    }

    /// Validate config.
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
        Ok(())
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Weights for a single BERT encoder layer.
#[derive(Debug, Clone)]
pub struct BertLayerWeights {
    /// Combined Q, K, V projection: `[d_model × 3*d_model]`.
    pub self_attn_qkv: Vec<f32>,
    /// Output projection: `[d_model × d_model]`.
    pub self_attn_out: Vec<f32>,
    /// FFN W1: `[d_model × d_ff]`.
    pub ffn_w1: Vec<f32>,
    /// FFN b1: `[d_ff]`.
    pub ffn_b1: Vec<f32>,
    /// FFN W2: `[d_ff × d_model]`.
    pub ffn_w2: Vec<f32>,
    /// FFN b2: `[d_model]`.
    pub ffn_b2: Vec<f32>,
    /// LN1 weight (gamma), `[d_model]`.
    pub ln1_weight: Vec<f32>,
    /// LN1 bias (beta), `[d_model]`.
    pub ln1_bias: Vec<f32>,
    /// LN2 weight (gamma), `[d_model]`.
    pub ln2_weight: Vec<f32>,
    /// LN2 bias (beta), `[d_model]`.
    pub ln2_bias: Vec<f32>,
}

impl BertLayerWeights {
    /// Create with zeros (except LN weights = 1).
    #[must_use]
    pub fn zeros(cfg: &BertConfig) -> Self {
        let d = cfg.d_model;
        let f = cfg.d_ff;
        Self {
            self_attn_qkv: vec![0.0_f32; d * 3 * d],
            self_attn_out: vec![0.0_f32; d * d],
            ffn_w1: vec![0.0_f32; d * f],
            ffn_b1: vec![0.0_f32; f],
            ffn_w2: vec![0.0_f32; f * d],
            ffn_b2: vec![0.0_f32; d],
            ln1_weight: vec![1.0_f32; d],
            ln1_bias: vec![0.0_f32; d],
            ln2_weight: vec![1.0_f32; d],
            ln2_bias: vec![0.0_f32; d],
        }
    }

    /// Create with ones for attn weights (identity-like behaviour).
    #[must_use]
    pub fn ones(cfg: &BertConfig) -> Self {
        let d = cfg.d_model;
        let f = cfg.d_ff;
        // Create identity-like QKV weights (Q, K, V each = identity)
        let mut qkv = vec![0.0_f32; d * 3 * d];
        for i in 0..d {
            qkv[i * 3 * d + i] = 1.0; // Q part
            qkv[i * 3 * d + d + i] = 1.0; // K part
            qkv[i * 3 * d + 2 * d + i] = 1.0; // V part
        }
        let mut out = vec![0.0_f32; d * d];
        for i in 0..d {
            out[i * d + i] = 1.0;
        }
        Self {
            self_attn_qkv: qkv,
            self_attn_out: out,
            ffn_w1: vec![0.0_f32; d * f],
            ffn_b1: vec![0.0_f32; f],
            ffn_w2: vec![0.0_f32; f * d],
            ffn_b2: vec![0.0_f32; d],
            ln1_weight: vec![1.0_f32; d],
            ln1_bias: vec![0.0_f32; d],
            ln2_weight: vec![1.0_f32; d],
            ln2_bias: vec![0.0_f32; d],
        }
    }
}

/// Full BERT encoder weights.
#[derive(Debug, Clone)]
pub struct BertWeights {
    /// Token embedding table: `[vocab_size × d_model]`.
    pub token_embed: Vec<f32>,
    /// Positional embedding table: `[max_seq_len × d_model]`.
    pub pos_embed: Vec<f32>,
    /// Per-layer weights.
    pub layers: Vec<BertLayerWeights>,
    /// Final layer norm (weight).
    pub final_ln_weight: Vec<f32>,
    /// Final layer norm (bias).
    pub final_ln_bias: Vec<f32>,
}

impl BertWeights {
    /// Create with zeros.
    #[must_use]
    pub fn zeros(cfg: &BertConfig) -> Self {
        let d = cfg.d_model;
        Self {
            token_embed: vec![0.0_f32; cfg.vocab_size * d],
            pos_embed: vec![0.0_f32; cfg.max_seq_len * d],
            layers: (0..cfg.n_layers)
                .map(|_| BertLayerWeights::zeros(cfg))
                .collect(),
            final_ln_weight: vec![1.0_f32; d],
            final_ln_bias: vec![0.0_f32; d],
        }
    }

    /// Create with ones (identity-like).
    #[must_use]
    pub fn ones(cfg: &BertConfig) -> Self {
        let d = cfg.d_model;
        // Positional embeddings: small sine waves
        let mut pos_embed = vec![0.0_f32; cfg.max_seq_len * d];
        for pos in 0..cfg.max_seq_len {
            for i in 0..d {
                pos_embed[pos * d + i] = ((pos + 1) as f32 * (i + 1) as f32 * 0.01).sin();
            }
        }
        Self {
            token_embed: vec![0.1_f32; cfg.vocab_size * d],
            pos_embed,
            layers: (0..cfg.n_layers)
                .map(|_| BertLayerWeights::ones(cfg))
                .collect(),
            final_ln_weight: vec![1.0_f32; d],
            final_ln_bias: vec![0.0_f32; d],
        }
    }
}

// ─── BertEncoder ─────────────────────────────────────────────────────────────

/// BERT-style encoder forward pass.
pub struct BertEncoder;

impl BertEncoder {
    /// Encode a token sequence to a CLS-pooled `[d_model]` vector.
    ///
    /// `token_ids`: sequence of token IDs (integers in `[0, vocab_size)`).
    ///
    /// Returns `[d_model]` CLS token output.
    pub fn forward(
        token_ids: &[u32],
        weights: &BertWeights,
        cfg: &BertConfig,
    ) -> MmResult<Vec<f32>> {
        cfg.validate()?;

        let d = cfg.d_model;
        let seq_len = token_ids.len();

        if seq_len == 0 {
            return Err(MultiModalError::EmptyInput);
        }

        // Validate token IDs
        for &tid in token_ids {
            if tid as usize >= cfg.vocab_size {
                return Err(MultiModalError::TokenOutOfRange {
                    token_id: tid,
                    vocab_size: cfg.vocab_size,
                });
            }
        }

        // Token + position embeddings: [seq_len × d_model]
        let mut hidden = vec![0.0_f32; seq_len * d];
        for (pos, &tid) in token_ids.iter().enumerate() {
            let tok_row = &weights.token_embed[tid as usize * d..(tid as usize + 1) * d];
            let pos_row = &weights.pos_embed
                [pos.min(cfg.max_seq_len - 1) * d..(pos.min(cfg.max_seq_len - 1) + 1) * d];
            for i in 0..d {
                hidden[pos * d + i] = tok_row[i] + pos_row[i];
            }
        }

        // N transformer layers
        for layer_w in &weights.layers {
            hidden = bert_layer_forward(&hidden, seq_len, cfg, layer_w)?;
        }

        // Final layer norm
        let ln = LayerNorm {
            weight: weights.final_ln_weight.clone(),
            bias: weights.final_ln_bias.clone(),
            d_model: d,
        };
        hidden = ln.forward(&hidden, seq_len)?;

        // CLS token = position 0
        let cls = hidden[..d].to_vec();
        Ok(cls)
    }
}

/// Apply a single BERT transformer layer.
fn bert_layer_forward(
    input: &[f32],
    seq: usize,
    cfg: &BertConfig,
    w: &BertLayerWeights,
) -> MmResult<Vec<f32>> {
    let d = cfg.d_model;
    let h = cfg.n_heads;
    let d_k = d / h;

    // Pre-norm LN1
    let ln1 = LayerNorm {
        weight: w.ln1_weight.clone(),
        bias: w.ln1_bias.clone(),
        d_model: d,
    };
    let normed = ln1.forward(input, seq)?;

    // Extract Q, K, V from combined QKV weight: [d × 3d]
    // W_qkv[i, j] for j in [0,d) = Q, [d, 2d) = K, [2d, 3d) = V
    let mut w_q = vec![0.0_f32; d * d];
    let mut w_k = vec![0.0_f32; d * d];
    let mut w_v = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..d {
            w_q[i * d + j] = w.self_attn_qkv[i * 3 * d + j];
            w_k[i * d + j] = w.self_attn_qkv[i * 3 * d + d + j];
            w_v[i * d + j] = w.self_attn_qkv[i * 3 * d + 2 * d + j];
        }
    }

    let attn_cfg = CrossAttnConfig {
        n_heads: h,
        d_model: d,
        d_k,
        d_v: d_k,
        dropout_rate: 0.0,
    };
    let attn_weights = CrossAttnWeights {
        w_q,
        w_k,
        w_v,
        w_o: w.self_attn_out.clone(),
    };
    let attn = CrossAttention::with_weights(attn_cfg, attn_weights);

    // Self-attention
    let sa_out = attn.forward(&normed, &normed, &normed, seq, seq)?;

    // Residual 1
    let mut x: Vec<f32> = input
        .iter()
        .zip(sa_out.iter())
        .map(|(a, b)| a + b)
        .collect();

    // Pre-norm LN2 + FFN + residual
    let ln2 = LayerNorm {
        weight: w.ln2_weight.clone(),
        bias: w.ln2_bias.clone(),
        d_model: d,
    };
    let normed2 = ln2.forward(&x, seq)?;

    // FFN: W1 [d × d_ff], GELU, W2 [d_ff × d]
    let f = cfg.d_ff;
    let mut hidden = vec![0.0_f32; seq * f];
    for s in 0..seq {
        for fi in 0..f {
            let mut acc = w.ffn_b1[fi];
            for di in 0..d {
                acc += normed2[s * d + di] * w.ffn_w1[di * f + fi];
            }
            hidden[s * f + fi] = bert_gelu(acc);
        }
    }
    let mut ffn_out = vec![0.0_f32; seq * d];
    for s in 0..seq {
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

/// GELU approximation (tanh-based).
#[inline]
fn bert_gelu(x: f32) -> f32 {
    let k = 0.044_715_f32;
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + k * x.powi(3))).tanh())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bert_config_tiny_valid() {
        let cfg = BertConfig::tiny();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn bert_config_invalid_heads() {
        let mut cfg = BertConfig::tiny();
        cfg.n_heads = 3; // 8 % 3 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn bert_forward_output_shape() {
        let cfg = BertConfig::tiny();
        let weights = BertWeights::zeros(&cfg);
        let token_ids = [0_u32, 1, 2, 3];
        let out = BertEncoder::forward(&token_ids, &weights, &cfg).expect("forward should succeed");
        assert_eq!(out.len(), cfg.d_model);
    }

    #[test]
    fn bert_forward_output_finite() {
        let cfg = BertConfig::tiny();
        let weights = BertWeights::ones(&cfg);
        let token_ids = [0_u32, 1, 2, 3, 4];
        let out = BertEncoder::forward(&token_ids, &weights, &cfg).expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bert_forward_zero_weights_zero_cls() {
        let cfg = BertConfig::tiny();
        let weights = BertWeights::zeros(&cfg);
        let token_ids = [0_u32, 1, 2];
        let out = BertEncoder::forward(&token_ids, &weights, &cfg).expect("forward should succeed");
        // With zero embeddings and zero weights, all outputs are zero
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }

    #[test]
    fn bert_token_out_of_range_error() {
        let cfg = BertConfig::tiny();
        let weights = BertWeights::zeros(&cfg);
        let token_ids = [0_u32, 100]; // vocab_size = 32
        let err = BertEncoder::forward(&token_ids, &weights, &cfg).unwrap_err();
        assert!(matches!(err, MultiModalError::TokenOutOfRange { .. }));
    }

    #[test]
    fn bert_empty_input_error() {
        let cfg = BertConfig::tiny();
        let weights = BertWeights::zeros(&cfg);
        let err = BertEncoder::forward(&[], &weights, &cfg).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    #[test]
    fn bert_layer_weights_ones_attn_identity() {
        let cfg = BertConfig::tiny();
        let w = BertLayerWeights::ones(&cfg);
        // QKV should have non-zero diagonal entries
        assert!(w.self_attn_qkv.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn bert_weights_zeros_all_zero() {
        let cfg = BertConfig::tiny();
        let w = BertWeights::zeros(&cfg);
        assert!(w.token_embed.iter().all(|&v| v == 0.0));
        assert!(w.pos_embed.iter().all(|&v| v == 0.0));
    }
}

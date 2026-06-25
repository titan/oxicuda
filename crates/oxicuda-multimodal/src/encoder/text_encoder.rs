//! BERT-style multi-layer transformer text encoder.
//!
//! Produces a CLS-pooled `[d_model]` embedding from token ID sequences.

use crate::cross_attn::cross_attention::{
    CrossAttention, CrossAttnConfig, CrossAttnWeights, softmax_rows_inplace,
};
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

    /// Encode a token sequence with an explicit key-padding mask.
    ///
    /// `attention_mask[i] == true` marks position `i` as a **real** token;
    /// `false` marks it as padding. Padded positions are excluded from every
    /// query's attention (their pre-softmax scores are set to `-∞`), so the
    /// representation of the real tokens is identical to the representation that
    /// would be produced if the padding had never been appended. This matches the
    /// Hugging Face `attention_mask` convention (1 = keep, 0 = pad).
    ///
    /// Position 0 (CLS) must be a real token. The mask length must equal the
    /// sequence length.
    ///
    /// # Errors
    /// - Every error of [`BertEncoder::forward`].
    /// - [`MultiModalError::MismatchedSeqLens`] when `attention_mask.len()` does
    ///   not equal `token_ids.len()`.
    /// - [`MultiModalError::EmptyInput`] when no position is a real token.
    pub fn forward_masked(
        token_ids: &[u32],
        attention_mask: &[bool],
        weights: &BertWeights,
        cfg: &BertConfig,
    ) -> MmResult<Vec<f32>> {
        cfg.validate()?;
        let d = cfg.d_model;
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if attention_mask.len() != seq_len {
            return Err(MultiModalError::MismatchedSeqLens {
                q_len: seq_len,
                kv_len: attention_mask.len(),
            });
        }
        if !attention_mask.iter().any(|&m| m) {
            return Err(MultiModalError::EmptyInput);
        }
        for &tid in token_ids {
            if tid as usize >= cfg.vocab_size {
                return Err(MultiModalError::TokenOutOfRange {
                    token_id: tid,
                    vocab_size: cfg.vocab_size,
                });
            }
        }

        let mut hidden = vec![0.0_f32; seq_len * d];
        for (pos, &tid) in token_ids.iter().enumerate() {
            let tok_row = &weights.token_embed[tid as usize * d..(tid as usize + 1) * d];
            let pos_row = &weights.pos_embed
                [pos.min(cfg.max_seq_len - 1) * d..(pos.min(cfg.max_seq_len - 1) + 1) * d];
            for i in 0..d {
                hidden[pos * d + i] = tok_row[i] + pos_row[i];
            }
        }

        for layer_w in &weights.layers {
            hidden = bert_layer_forward_masked(&hidden, seq_len, attention_mask, cfg, layer_w)?;
        }

        let ln = LayerNorm {
            weight: weights.final_ln_weight.clone(),
            bias: weights.final_ln_bias.clone(),
            d_model: d,
        };
        hidden = ln.forward(&hidden, seq_len)?;

        let cls = hidden[..d].to_vec();
        Ok(cls)
    }
}

/// BERT transformer layer with a key-padding mask applied to self-attention.
fn bert_layer_forward_masked(
    input: &[f32],
    seq: usize,
    mask: &[bool],
    cfg: &BertConfig,
    w: &BertLayerWeights,
) -> MmResult<Vec<f32>> {
    let d = cfg.d_model;
    let h = cfg.n_heads;
    let d_k = d / h;

    let ln1 = LayerNorm {
        weight: w.ln1_weight.clone(),
        bias: w.ln1_bias.clone(),
        d_model: d,
    };
    let normed = ln1.forward(input, seq)?;

    // Split the combined QKV weight into Q/K/V projections.
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
    let proj_q = linear_rows(&normed, &w_q, seq, d, d);
    let proj_k = linear_rows(&normed, &w_k, seq, d, d);
    let proj_v = linear_rows(&normed, &w_v, seq, d, d);

    let scale = 1.0 / (d_k as f32).sqrt();
    let mut head_outputs = vec![0.0_f32; seq * d];
    for head in 0..h {
        let col = head * d_k;
        let mut scores = vec![0.0_f32; seq * seq];
        for qi in 0..seq {
            for ki in 0..seq {
                if !mask[ki] {
                    scores[qi * seq + ki] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0_f32;
                for di in 0..d_k {
                    dot += proj_q[qi * d + col + di] * proj_k[ki * d + col + di];
                }
                scores[qi * seq + ki] = dot * scale;
            }
        }
        softmax_rows_inplace(&mut scores, seq, seq);
        for qi in 0..seq {
            for vi in 0..d_k {
                let mut s = 0.0_f32;
                for ki in 0..seq {
                    s += scores[qi * seq + ki] * proj_v[ki * d + col + vi];
                }
                head_outputs[qi * d + col + vi] = s;
            }
        }
    }
    let sa_out = linear_rows(&head_outputs, &w.self_attn_out, seq, d, d);

    let mut x: Vec<f32> = input
        .iter()
        .zip(sa_out.iter())
        .map(|(a, b)| a + b)
        .collect();

    let ln2 = LayerNorm {
        weight: w.ln2_weight.clone(),
        bias: w.ln2_bias.clone(),
        d_model: d,
    };
    let normed2 = ln2.forward(&x, seq)?;

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

/// `A [rows × in_dim] · W [in_dim × out_dim]` → `[rows × out_dim]`, `W` row-major.
fn linear_rows(a: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0.0_f32;
            for i in 0..in_dim {
                acc += a[r * in_dim + i] * w[i * out_dim + o];
            }
            out[r * out_dim + o] = acc;
        }
    }
    out
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

    /// Build a deterministic, non-trivial BERT so that masking actually changes
    /// the output (zero/ones presets are too degenerate to test invariance).
    fn random_bert(cfg: &BertConfig, seed: u64) -> BertWeights {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(seed);
        let mut w = BertWeights::zeros(cfg);
        rng.fill_normal(&mut w.token_embed);
        rng.fill_normal(&mut w.pos_embed);
        for layer in w.layers.iter_mut() {
            rng.fill_normal(&mut layer.self_attn_qkv);
            rng.fill_normal(&mut layer.self_attn_out);
            rng.fill_normal(&mut layer.ffn_w1);
            rng.fill_normal(&mut layer.ffn_w2);
            // Keep init scale small so activations stay well-conditioned.
            for v in layer.self_attn_qkv.iter_mut() {
                *v *= 0.2;
            }
            for v in layer.self_attn_out.iter_mut() {
                *v *= 0.2;
            }
        }
        w
    }

    #[test]
    fn bert_masked_all_true_equals_unmasked() {
        let cfg = BertConfig::tiny();
        let w = random_bert(&cfg, 1);
        let token_ids = [0_u32, 5, 9, 13, 2];
        let mask = [true; 5];
        let masked =
            BertEncoder::forward_masked(&token_ids, &mask, &w, &cfg).expect("masked forward");
        let plain = BertEncoder::forward(&token_ids, &w, &cfg).expect("forward");
        for (a, b) in masked.iter().zip(plain.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "all-true mask must match forward: {a} vs {b}"
            );
        }
    }

    #[test]
    fn bert_padding_is_invariant() {
        // The CLS embedding of a real sequence must not change when padding tokens
        // are appended and masked out. (Position 0 = CLS is real.)
        let cfg = BertConfig::tiny();
        let w = random_bert(&cfg, 2);

        let real = [0_u32, 7, 3];
        let real_mask = [true, true, true];
        let cls_real =
            BertEncoder::forward_masked(&real, &real_mask, &w, &cfg).expect("real forward");

        // Append two padding tokens (arbitrary ids) flagged false in the mask.
        let padded = [0_u32, 7, 3, 11, 4];
        let padded_mask = [true, true, true, false, false];
        let cls_padded =
            BertEncoder::forward_masked(&padded, &padded_mask, &w, &cfg).expect("padded forward");

        for (a, b) in cls_real.iter().zip(cls_padded.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "padding changed the CLS embedding: {a} vs {b}"
            );
        }
    }

    #[test]
    fn bert_masking_changes_output_vs_full_attention() {
        // Masking a token out should produce a *different* CLS than attending to
        // it, confirming the mask actually takes effect.
        let cfg = BertConfig::tiny();
        let w = random_bert(&cfg, 3);
        let token_ids = [0_u32, 7, 3, 11];
        let full = [true, true, true, true];
        let drop_last = [true, true, true, false];
        let cls_full = BertEncoder::forward_masked(&token_ids, &full, &w, &cfg).expect("full");
        let cls_drop = BertEncoder::forward_masked(&token_ids, &drop_last, &w, &cfg).expect("drop");
        let diff: f32 = cls_full
            .iter()
            .zip(cls_drop.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-5,
            "masking a token should change the output, diff={diff}"
        );
    }

    #[test]
    fn bert_masked_wrong_mask_len_errors() {
        let cfg = BertConfig::tiny();
        let w = BertWeights::zeros(&cfg);
        let token_ids = [0_u32, 1, 2];
        let mask = [true, true]; // wrong length
        assert!(matches!(
            BertEncoder::forward_masked(&token_ids, &mask, &w, &cfg),
            Err(MultiModalError::MismatchedSeqLens { .. })
        ));
    }

    #[test]
    fn bert_masked_all_false_errors() {
        let cfg = BertConfig::tiny();
        let w = BertWeights::zeros(&cfg);
        let token_ids = [0_u32, 1, 2];
        let mask = [false, false, false];
        assert!(matches!(
            BertEncoder::forward_masked(&token_ids, &mask, &w, &cfg),
            Err(MultiModalError::EmptyInput)
        ));
    }
}

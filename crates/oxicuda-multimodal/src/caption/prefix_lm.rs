//! Prefix-LM for cross-modal captioning and generation.
//!
//! Greedy-decodes text tokens conditioned on a visual prefix via cross-attention.
//! Architecture: each decode step cross-attends text tokens over the visual prefix,
//! then produces a distribution over the vocabulary via a head projection.

use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::self_cross_block::LayerNorm;
use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the Prefix-LM decoder.
#[derive(Debug, Clone)]
pub struct PrefixLmConfig {
    /// Model hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of decoder layers.
    pub n_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum generation length.
    pub max_gen_len: usize,
}

impl PrefixLmConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            d_model: 8,
            n_heads: 2,
            n_layers: 2,
            vocab_size: 32,
            max_gen_len: 10,
        }
    }

    /// Validate the configuration.
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
        if self.vocab_size == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        Ok(())
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Weights for a single Prefix-LM decoder layer.
#[derive(Debug, Clone)]
pub struct PrefixLmLayerWeights {
    /// Self-attention weights (over generated tokens).
    pub self_attn: CrossAttnWeights,
    /// Cross-attention weights (text tokens over visual prefix).
    pub cross_attn: CrossAttnWeights,
    /// FFN W1: `[d_model × 4*d_model]`.
    pub ffn_w1: Vec<f32>,
    /// FFN b1: `[4*d_model]`.
    pub ffn_b1: Vec<f32>,
    /// FFN W2: `[4*d_model × d_model]`.
    pub ffn_w2: Vec<f32>,
    /// FFN b2: `[d_model]`.
    pub ffn_b2: Vec<f32>,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub ln3: LayerNorm,
}

impl PrefixLmLayerWeights {
    #[must_use]
    pub fn zeros(cfg: &PrefixLmConfig) -> Self {
        let d = cfg.d_model;
        let f = d * 4;
        let attn_cfg = CrossAttnConfig {
            n_heads: cfg.n_heads,
            d_model: d,
            d_k: d / cfg.n_heads,
            d_v: d / cfg.n_heads,
            dropout_rate: 0.0,
        };
        Self {
            self_attn: CrossAttnWeights::zeros(&attn_cfg),
            cross_attn: CrossAttnWeights::zeros(&attn_cfg),
            ffn_w1: vec![0.0_f32; d * f],
            ffn_b1: vec![0.0_f32; f],
            ffn_w2: vec![0.0_f32; f * d],
            ffn_b2: vec![0.0_f32; d],
            ln1: LayerNorm::ones(d),
            ln2: LayerNorm::ones(d),
            ln3: LayerNorm::ones(d),
        }
    }
}

/// Full Prefix-LM weights.
#[derive(Debug, Clone)]
pub struct PrefixLmWeights {
    /// Token embedding table: `[vocab_size × d_model]`.
    pub token_embed: Vec<f32>,
    /// Positional embedding: `[max_gen_len × d_model]`.
    pub pos_embed: Vec<f32>,
    /// Per-layer decoder weights.
    pub layers: Vec<PrefixLmLayerWeights>,
    /// Language model head: `[d_model × vocab_size]`.
    pub lm_head: Vec<f32>,
    /// LM head bias: `[vocab_size]`.
    pub lm_head_bias: Vec<f32>,
    /// Final layer norm weight.
    pub final_ln_weight: Vec<f32>,
    /// Final layer norm bias.
    pub final_ln_bias: Vec<f32>,
}

impl PrefixLmWeights {
    #[must_use]
    pub fn zeros(cfg: &PrefixLmConfig) -> Self {
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        Self {
            token_embed: vec![0.0_f32; v * d],
            pos_embed: vec![0.0_f32; cfg.max_gen_len * d],
            layers: (0..cfg.n_layers)
                .map(|_| PrefixLmLayerWeights::zeros(cfg))
                .collect(),
            lm_head: vec![0.0_f32; d * v],
            lm_head_bias: vec![0.0_f32; v],
            final_ln_weight: vec![1.0_f32; d],
            final_ln_bias: vec![0.0_f32; d],
        }
    }
}

// ─── PrefixLm ─────────────────────────────────────────────────────────────────

/// Prefix-LM decoder for cross-modal generation.
pub struct PrefixLm;

impl PrefixLm {
    /// Greedy-decode `max_new_tokens` tokens conditioned on `visual_prefix`.
    ///
    /// `visual_prefix`: `[prefix_len × d_model]` — visual context tokens.
    ///
    /// Returns generated token IDs (length ≤ `max_new_tokens`).
    pub fn generate(
        visual_prefix: &[f32],
        prefix_len: usize,
        max_new_tokens: usize,
        cfg: &PrefixLmConfig,
        weights: &PrefixLmWeights,
        rng: &mut LcgRng,
    ) -> MmResult<Vec<u32>> {
        cfg.validate()?;
        if prefix_len == 0 || visual_prefix.len() != prefix_len * cfg.d_model {
            return Err(MultiModalError::DimensionMismatch {
                expected: prefix_len * cfg.d_model,
                got: visual_prefix.len(),
            });
        }

        let d = cfg.d_model;
        let vocab = cfg.vocab_size;

        // Start with a single BOS token (token 0)
        let mut generated: Vec<u32> = vec![0];

        for step in 0..max_new_tokens {
            let cur_len = generated.len();
            if cur_len >= cfg.max_gen_len {
                break;
            }

            // Build token embeddings
            let mut token_hidden = vec![0.0_f32; cur_len * d];
            for (pos, &tid) in generated.iter().enumerate() {
                let tid_clamped = (tid as usize).min(vocab - 1);
                let tok_row = &weights.token_embed[tid_clamped * d..(tid_clamped + 1) * d];
                let pos_clamped = pos.min(cfg.max_gen_len - 1);
                let pos_row = &weights.pos_embed[pos_clamped * d..(pos_clamped + 1) * d];
                for i in 0..d {
                    token_hidden[pos * d + i] = tok_row[i] + pos_row[i];
                }
            }

            // Run decoder layers
            for layer_w in &weights.layers {
                token_hidden = prefix_lm_layer(
                    &token_hidden,
                    visual_prefix,
                    cur_len,
                    prefix_len,
                    cfg,
                    layer_w,
                )?;
            }

            // Final LN
            let ln = LayerNorm {
                weight: weights.final_ln_weight.clone(),
                bias: weights.final_ln_bias.clone(),
                d_model: d,
            };
            token_hidden = ln.forward(&token_hidden, cur_len)?;

            // LM head on last token
            let last = &token_hidden[(cur_len - 1) * d..cur_len * d];
            let mut logits = weights.lm_head_bias.clone();
            for v in 0..vocab {
                for di in 0..d {
                    logits[v] += last[di] * weights.lm_head[di * vocab + v];
                }
            }

            // Greedy decode: argmax with small temperature noise from RNG
            // (for variety in generation even with zero weights)
            let next_token = if logits.iter().all(|&v| v == logits[0]) {
                // All logits equal — use RNG to pick
                rng.next_usize(vocab) as u32
            } else {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            };

            generated.push(next_token);

            // EOS check: token 1 = EOS (convention)
            if next_token == 1 {
                break;
            }

            let _ = step; // suppress unused warning
        }

        // Remove BOS token (position 0), return generated tokens
        Ok(generated[1..].to_vec())
    }
}

/// Apply a single Prefix-LM decoder layer.
fn prefix_lm_layer(
    token_hidden: &[f32],
    visual_prefix: &[f32],
    cur_len: usize,
    prefix_len: usize,
    cfg: &PrefixLmConfig,
    w: &PrefixLmLayerWeights,
) -> MmResult<Vec<f32>> {
    let d = cfg.d_model;
    let h = cfg.n_heads;
    let d_k = d / h;
    let f = d * 4;

    // Pre-norm + self-attention + residual
    let normed1 = w.ln1.forward(token_hidden, cur_len)?;
    let self_attn_cfg = CrossAttnConfig {
        n_heads: h,
        d_model: d,
        d_k,
        d_v: d_k,
        dropout_rate: 0.0,
    };
    let self_attn = CrossAttention::with_weights(self_attn_cfg.clone(), w.self_attn.clone());
    let sa_out = self_attn.forward(&normed1, &normed1, &normed1, cur_len, cur_len)?;
    let mut x: Vec<f32> = token_hidden
        .iter()
        .zip(sa_out.iter())
        .map(|(a, b)| a + b)
        .collect();

    // Pre-norm + cross-attention over visual prefix + residual
    let normed2 = w.ln2.forward(&x, cur_len)?;
    let normed_prefix = w.ln2.forward(visual_prefix, prefix_len)?;
    let cross_attn = CrossAttention::with_weights(self_attn_cfg, w.cross_attn.clone());
    let ca_out = cross_attn.forward(
        &normed2,
        &normed_prefix,
        &normed_prefix,
        cur_len,
        prefix_len,
    )?;
    for (xi, ci) in x.iter_mut().zip(ca_out.iter()) {
        *xi += ci;
    }

    // Pre-norm + FFN + residual
    let normed3 = w.ln3.forward(&x, cur_len)?;
    let mut hidden = vec![0.0_f32; cur_len * f];
    for s in 0..cur_len {
        for fi in 0..f {
            let mut acc = w.ffn_b1[fi];
            for di in 0..d {
                acc += normed3[s * d + di] * w.ffn_w1[di * f + fi];
            }
            hidden[s * f + fi] = plm_gelu(acc);
        }
    }
    let mut ffn_out = vec![0.0_f32; cur_len * d];
    for s in 0..cur_len {
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
fn plm_gelu(x: f32) -> f32 {
    let k = 0.044_715_f32;
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + k * x.powi(3))).tanh())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_lm_generates_correct_length() {
        let cfg = PrefixLmConfig::tiny();
        let weights = PrefixLmWeights::zeros(&cfg);
        let mut rng = LcgRng::new(42);
        let prefix = vec![0.5_f32; 4 * cfg.d_model];
        let tokens = PrefixLm::generate(&prefix, 4, 5, &cfg, &weights, &mut rng)
            .expect("generate should succeed");
        assert!(
            tokens.len() <= 5,
            "generated {} tokens, expected ≤ 5",
            tokens.len()
        );
    }

    #[test]
    fn prefix_lm_tokens_in_vocab_range() {
        let cfg = PrefixLmConfig::tiny();
        let weights = PrefixLmWeights::zeros(&cfg);
        let mut rng = LcgRng::new(0);
        let prefix = vec![0.3_f32; 3 * cfg.d_model];
        let tokens = PrefixLm::generate(&prefix, 3, 8, &cfg, &weights, &mut rng)
            .expect("generate should succeed");
        for &tid in &tokens {
            assert!(
                (tid as usize) < cfg.vocab_size,
                "token {tid} out of vocab {}",
                cfg.vocab_size
            );
        }
    }

    #[test]
    fn prefix_lm_empty_prefix_error() {
        let cfg = PrefixLmConfig::tiny();
        let weights = PrefixLmWeights::zeros(&cfg);
        let mut rng = LcgRng::new(0);
        let err = PrefixLm::generate(&[], 0, 5, &cfg, &weights, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn prefix_lm_deterministic_with_same_rng() {
        let cfg = PrefixLmConfig::tiny();
        let weights = PrefixLmWeights::zeros(&cfg);
        let prefix = vec![0.1_f32; 4 * cfg.d_model];
        let mut rng1 = LcgRng::new(7);
        let mut rng2 = LcgRng::new(7);
        let t1 = PrefixLm::generate(&prefix, 4, 5, &cfg, &weights, &mut rng1)
            .expect("generate should succeed");
        let t2 = PrefixLm::generate(&prefix, 4, 5, &cfg, &weights, &mut rng2)
            .expect("generate should succeed");
        assert_eq!(t1, t2);
    }

    #[test]
    fn prefix_lm_config_validate() {
        let cfg = PrefixLmConfig::tiny();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn prefix_lm_config_invalid_heads() {
        let mut cfg = PrefixLmConfig::tiny();
        cfg.n_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn prefix_lm_weights_zeros_correct_shape() {
        let cfg = PrefixLmConfig::tiny();
        let w = PrefixLmWeights::zeros(&cfg);
        assert_eq!(w.token_embed.len(), cfg.vocab_size * cfg.d_model);
        assert_eq!(w.lm_head.len(), cfg.d_model * cfg.vocab_size);
        assert_eq!(w.layers.len(), cfg.n_layers);
    }
}

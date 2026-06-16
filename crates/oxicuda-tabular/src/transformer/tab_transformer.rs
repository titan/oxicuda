//! TabTransformer: Transformer over Categorical Embeddings (Huang et al. 2020).
//!
//! Architecture: categorical features are embedded and processed by stacked transformer
//! blocks (multi-head self-attention + FFN). The transformed embeddings are concatenated
//! with raw continuous features and fed to an MLP head.

use crate::attention::saint::multihead_attention;
use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── TabTransformerConfig ─────────────────────────────────────────────────────

/// Configuration for `TabTransformer`.
#[derive(Debug, Clone)]
pub struct TabTransformerConfig {
    /// Number of categorical input features.
    pub n_cat_features: usize,
    /// Number of categories per categorical feature.
    pub cat_n_categories: Vec<usize>,
    /// Number of continuous input features.
    pub n_cont_features: usize,
    /// Embedding dimension per categorical feature (d).
    pub embed_dim: usize,
    /// Number of attention heads (`embed_dim % n_heads == 0`).
    pub n_heads: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// FFN hidden dimension within each block.
    pub ffn_hidden: usize,
    /// Hidden layer sizes for the MLP classification head.
    pub mlp_hidden_sizes: Vec<usize>,
    /// Output classes (1 for regression).
    pub n_classes: usize,
    /// Dropout rate (stored but not applied during inference).
    pub dropout_rate: f32,
}

// ─── Weight types ─────────────────────────────────────────────────────────────

/// Weight bundle for one transformer block.
#[derive(Debug, Clone)]
pub struct TabTransformerBlockWeights {
    /// Q projection: `[embed_dim × embed_dim]`.
    pub wq: Vec<f32>,
    /// K projection: `[embed_dim × embed_dim]`.
    pub wk: Vec<f32>,
    /// V projection: `[embed_dim × embed_dim]`.
    pub wv: Vec<f32>,
    /// Output projection: `[embed_dim × embed_dim]`.
    pub wo: Vec<f32>,
    /// LayerNorm 1 scale: `[embed_dim]`.
    pub ln1_scale: Vec<f32>,
    /// LayerNorm 1 bias: `[embed_dim]`.
    pub ln1_bias: Vec<f32>,
    /// FFN linear 1 weight: `[embed_dim × ffn_hidden]`.
    pub ffn_w1: Vec<f32>,
    /// FFN linear 1 bias: `[ffn_hidden]`.
    pub ffn_b1: Vec<f32>,
    /// FFN linear 2 weight: `[ffn_hidden × embed_dim]`.
    pub ffn_w2: Vec<f32>,
    /// FFN linear 2 bias: `[embed_dim]`.
    pub ffn_b2: Vec<f32>,
    /// LayerNorm 2 scale: `[embed_dim]`.
    pub ln2_scale: Vec<f32>,
    /// LayerNorm 2 bias: `[embed_dim]`.
    pub ln2_bias: Vec<f32>,
}

/// Full weight bundle for `TabTransformer`.
#[derive(Debug, Clone)]
pub struct TabTransformerWeights {
    /// Categorical embedding tables: `cat_embeddings[i]` is `[cat_n_categories[i] × embed_dim]`.
    pub cat_embeddings: Vec<Vec<f32>>,
    /// Column-wise LayerNorm scales before transformer: `[n_cat_features]` (one per column).
    pub col_ln_scale: Vec<f32>,
    /// Column-wise LayerNorm biases before transformer: `[n_cat_features]`.
    pub col_ln_bias: Vec<f32>,
    /// Transformer block weights (one per layer).
    pub blocks: Vec<TabTransformerBlockWeights>,
    /// MLP head layers: each element is (weight `[in × out]`, bias `[out]`).
    pub mlp_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Final output weight: `[mlp_last_dim × n_classes]` (or flat concat dim if no MLP layers).
    pub output_w: Vec<f32>,
    /// Final output bias: `[n_classes]`.
    pub output_b: Vec<f32>,
}

// ─── TabTransformer ───────────────────────────────────────────────────────────

/// TabTransformer model (Huang et al. 2020).
pub struct TabTransformer {
    pub cfg: TabTransformerConfig,
}

impl TabTransformer {
    const LN_EPS: f32 = 1e-5;

    /// Construct and validate a `TabTransformer` from the given config.
    pub fn new(cfg: TabTransformerConfig) -> TabularResult<Self> {
        if cfg.n_cat_features == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if cfg.cat_n_categories.len() != cfg.n_cat_features {
            return Err(TabularError::DimensionMismatch {
                expected: cfg.n_cat_features,
                got: cfg.cat_n_categories.len(),
            });
        }
        if cfg.embed_dim == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if cfg.n_heads == 0 || !cfg.embed_dim.is_multiple_of(cfg.n_heads) {
            return Err(TabularError::InvalidAttentionDim { dim: cfg.embed_dim });
        }
        Ok(Self { cfg })
    }

    /// Initialise all weights using Kaiming-uniform: U(-√(6/fan_in), √(6/fan_in)).
    pub fn init_weights(&self, rng: &mut LcgRng) -> TabTransformerWeights {
        let cfg = &self.cfg;

        let kaiming = |fan_in: usize, size: usize, rng: &mut LcgRng| -> Vec<f32> {
            let k = (6.0_f32 / fan_in as f32).sqrt();
            (0..size).map(|_| rng.next_f32() * 2.0 * k - k).collect()
        };

        let cat_embeddings: Vec<Vec<f32>> = cfg
            .cat_n_categories
            .iter()
            .map(|&nc| kaiming(nc, nc * cfg.embed_dim, rng))
            .collect();

        let col_ln_scale = vec![1.0_f32; cfg.n_cat_features];
        let col_ln_bias = vec![0.0_f32; cfg.n_cat_features];

        let blocks: Vec<TabTransformerBlockWeights> = (0..cfg.n_layers)
            .map(|_| {
                let d = cfg.embed_dim;
                let h = cfg.ffn_hidden;
                TabTransformerBlockWeights {
                    wq: kaiming(d, d * d, rng),
                    wk: kaiming(d, d * d, rng),
                    wv: kaiming(d, d * d, rng),
                    wo: kaiming(d, d * d, rng),
                    ln1_scale: vec![1.0_f32; d],
                    ln1_bias: vec![0.0_f32; d],
                    ffn_w1: kaiming(d, d * h, rng),
                    ffn_b1: vec![0.0_f32; h],
                    ffn_w2: kaiming(h, h * d, rng),
                    ffn_b2: vec![0.0_f32; d],
                    ln2_scale: vec![1.0_f32; d],
                    ln2_bias: vec![0.0_f32; d],
                }
            })
            .collect();

        // MLP head layer dimensions
        let flat_cat_dim = cfg.n_cat_features * cfg.embed_dim;
        let concat_dim = flat_cat_dim + cfg.n_cont_features;
        let mut in_dim = concat_dim;

        let mlp_layers: Vec<(Vec<f32>, Vec<f32>)> = cfg
            .mlp_hidden_sizes
            .iter()
            .map(|&out| {
                let w = kaiming(in_dim, in_dim * out, rng);
                let b = vec![0.0_f32; out];
                in_dim = out;
                (w, b)
            })
            .collect();

        let output_w = kaiming(in_dim, in_dim * cfg.n_classes, rng);
        let output_b = vec![0.0_f32; cfg.n_classes];

        TabTransformerWeights {
            cat_embeddings,
            col_ln_scale,
            col_ln_bias,
            blocks,
            mlp_layers,
            output_w,
            output_b,
        }
    }

    /// Forward pass through TabTransformer.
    ///
    /// - `cat_ids`: one index per categorical feature; each must be `< cat_n_categories[i]`.
    /// - `cont_feats`: raw continuous feature values.
    /// - Returns logits of length `n_classes` (no final softmax).
    pub fn forward(
        &self,
        cat_ids: &[usize],
        cont_feats: &[f32],
        weights: &TabTransformerWeights,
    ) -> TabularResult<Vec<f32>> {
        let cfg = &self.cfg;

        if cat_ids.len() != cfg.n_cat_features {
            return Err(TabularError::DimensionMismatch {
                expected: cfg.n_cat_features,
                got: cat_ids.len(),
            });
        }
        if cont_feats.len() != cfg.n_cont_features {
            return Err(TabularError::DimensionMismatch {
                expected: cfg.n_cont_features,
                got: cont_feats.len(),
            });
        }

        // 1. Categorical embedding lookup
        let mut embeds: Vec<f32> = Vec::with_capacity(cfg.n_cat_features * cfg.embed_dim);
        for (i, &cat_id) in cat_ids.iter().enumerate() {
            if cat_id >= cfg.cat_n_categories[i] {
                return Err(TabularError::CategoricalOutOfRange {
                    feat: i,
                    val: cat_id,
                    n: cfg.cat_n_categories[i],
                });
            }
            let table = &weights.cat_embeddings[i];
            let start = cat_id * cfg.embed_dim;
            embeds.extend_from_slice(&table[start..start + cfg.embed_dim]);
        }

        // 2. Column-wise LayerNorm (per feature, across embed_dim)
        for i in 0..cfg.n_cat_features {
            let row = &mut embeds[i * cfg.embed_dim..(i + 1) * cfg.embed_dim];
            let scale_val = weights.col_ln_scale[i];
            let bias_val = weights.col_ln_bias[i];
            let scale_slice = vec![scale_val; cfg.embed_dim];
            let bias_slice = vec![bias_val; cfg.embed_dim];
            let normed = Self::layer_norm(row, &scale_slice, &bias_slice);
            row.copy_from_slice(&normed);
        }

        // 3. Transformer blocks
        let mut x = embeds;
        for block_w in &weights.blocks {
            x = Self::transformer_block(
                &x,
                cfg.n_cat_features,
                cfg.embed_dim,
                cfg.n_heads,
                block_w,
            )?;
        }

        // 4. Flatten transformed embeddings: [n_cat * embed_dim]
        // x is already flat

        // 5. Concatenate with continuous features
        let mut combined = x;
        combined.extend_from_slice(cont_feats);

        // 6. MLP head
        let mut h = combined;
        for (w, b) in &weights.mlp_layers {
            let in_dim = h.len();
            let out_dim = b.len();
            let projected = Self::linear(&h, w, b, in_dim, out_dim);
            h = Self::relu(&projected);
        }

        // 7. Output projection (logits, no softmax)
        let in_dim = h.len();
        let out_dim = cfg.n_classes;
        let logits = Self::linear(&h, &weights.output_w, &weights.output_b, in_dim, out_dim);

        Ok(logits)
    }

    /// Pre-LN transformer block: x = x + attn(LN1(x)); x = x + FFN(LN2(x)).
    pub fn transformer_block(
        x: &[f32],
        n_cat: usize,
        embed_dim: usize,
        n_heads: usize,
        w: &TabTransformerBlockWeights,
    ) -> TabularResult<Vec<f32>> {
        if x.len() != n_cat * embed_dim {
            return Err(TabularError::DimensionMismatch {
                expected: n_cat * embed_dim,
                got: x.len(),
            });
        }

        // Pre-LN: apply layer norm per token before attention
        let mut x_ln1 = vec![0.0_f32; n_cat * embed_dim];
        for i in 0..n_cat {
            let row = &x[i * embed_dim..(i + 1) * embed_dim];
            let normed = Self::layer_norm(row, &w.ln1_scale, &w.ln1_bias);
            x_ln1[i * embed_dim..(i + 1) * embed_dim].copy_from_slice(&normed);
        }

        // Multi-head self-attention using saint's multihead_attention
        // Build Q, K, V projections manually then call MHSA
        let attn_out = multihead_attention(
            &x_ln1, &w.wq, &w.wk, &w.wv, &w.wo, n_cat, embed_dim, n_heads,
        )?;

        // Residual connection after attention
        let mut x_after_attn: Vec<f32> = x
            .iter()
            .zip(attn_out.iter())
            .map(|(&xi, &ai)| xi + ai)
            .collect();

        // Pre-LN before FFN
        let mut x_ln2 = vec![0.0_f32; n_cat * embed_dim];
        for i in 0..n_cat {
            let row = &x_after_attn[i * embed_dim..(i + 1) * embed_dim];
            let normed = Self::layer_norm(row, &w.ln2_scale, &w.ln2_bias);
            x_ln2[i * embed_dim..(i + 1) * embed_dim].copy_from_slice(&normed);
        }

        // FFN: apply per token
        let ffn_hidden = w.ffn_b1.len();
        let mut ffn_out = vec![0.0_f32; n_cat * embed_dim];
        for i in 0..n_cat {
            let token = &x_ln2[i * embed_dim..(i + 1) * embed_dim];
            // W1 is stored as [embed_dim × ffn_hidden] row-major
            let h = Self::linear(token, &w.ffn_w1, &w.ffn_b1, embed_dim, ffn_hidden);
            let h_gelu = Self::gelu(&h);
            // W2 is stored as [ffn_hidden × embed_dim] row-major
            let out = Self::linear(&h_gelu, &w.ffn_w2, &w.ffn_b2, ffn_hidden, embed_dim);
            ffn_out[i * embed_dim..(i + 1) * embed_dim].copy_from_slice(&out);
        }

        // Residual connection after FFN
        for (xa, fo) in x_after_attn.iter_mut().zip(ffn_out.iter()) {
            *xa += fo;
        }

        Ok(x_after_attn)
    }

    /// LayerNorm: `(x - mean) / sqrt(var + eps) * scale + bias`.
    pub fn layer_norm(x: &[f32], scale: &[f32], bias: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let denom = (var + Self::LN_EPS).sqrt();
        x.iter()
            .zip(scale.iter().zip(bias.iter()))
            .map(|(&xi, (&s, &b))| (xi - mean) / denom * s + b)
            .collect()
    }

    /// GELU approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³))).
    pub fn gelu(x: &[f32]) -> Vec<f32> {
        const C: f32 = 0.797_884_56; // sqrt(2/π)
        x.iter()
            .map(|&v| 0.5 * v * (1.0 + (C * (v + 0.044715 * v * v * v)).tanh()))
            .collect()
    }

    /// ReLU element-wise.
    pub fn relu(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
    }

    /// Linear projection: `y = x @ W + b` where W is `[in_dim × out_dim]` row-major.
    pub fn linear(x: &[f32], w: &[f32], b: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut out = b.to_vec();
        for j in 0..out_dim {
            let mut acc = 0.0_f32;
            for i in 0..in_dim {
                acc += x[i] * w[i * out_dim + j];
            }
            out[j] += acc;
        }
        out
    }

    /// Softmax in-place (numerically stable).
    pub fn softmax(x: &mut [f32]) {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for v in x.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let denom = if sum < 1e-30 { 1e-30 } else { sum };
        for v in x.iter_mut() {
            *v /= denom;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(
        n_cat: usize,
        n_cont: usize,
        embed_dim: usize,
        n_heads: usize,
        n_layers: usize,
        n_classes: usize,
    ) -> TabTransformerConfig {
        TabTransformerConfig {
            n_cat_features: n_cat,
            cat_n_categories: vec![4; n_cat],
            n_cont_features: n_cont,
            embed_dim,
            n_heads,
            n_layers,
            ffn_hidden: embed_dim * 2,
            mlp_hidden_sizes: vec![16],
            n_classes,
            dropout_rate: 0.1,
        }
    }

    #[test]
    fn tab_transformer_forward_output_shape() {
        let cfg = make_cfg(3, 2, 8, 2, 1, 4);
        let model = TabTransformer::new(cfg).expect("new should succeed");
        let mut rng = LcgRng::new(1);
        let w = model.init_weights(&mut rng);
        let logits = model
            .forward(&[0, 1, 2], &[0.5, -0.3], &w)
            .expect("forward should succeed");
        assert_eq!(logits.len(), 4);
    }

    #[test]
    fn tab_transformer_forward_finite() {
        let cfg = make_cfg(3, 2, 8, 2, 1, 2);
        let model = TabTransformer::new(cfg).expect("new should succeed");
        let mut rng = LcgRng::new(2);
        let w = model.init_weights(&mut rng);
        let logits = model
            .forward(&[0, 0, 0], &[1.0, 2.0], &w)
            .expect("forward should succeed");
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn layer_norm_zero_mean() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let scale = vec![1.0_f32; 5];
        let bias = vec![0.0_f32; 5];
        let out = TabTransformer::layer_norm(&x, &scale, &bias);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-5, "mean={mean}");
    }

    #[test]
    fn layer_norm_unit_variance() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let scale = vec![1.0_f32; 5];
        let bias = vec![0.0_f32; 5];
        let out = TabTransformer::layer_norm(&x, &scale, &bias);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / out.len() as f32;
        assert!((var - 1.0).abs() < 1e-4, "var={var}");
    }

    #[test]
    fn transformer_block_output_shape() {
        let d = 8;
        let n_cat = 3;
        let mut rng = LcgRng::new(10);
        let cfg = make_cfg(n_cat, 0, d, 2, 1, 2);
        let model = TabTransformer::new(cfg).expect("new should succeed");
        let w = model.init_weights(&mut rng);
        let x: Vec<f32> = (0..n_cat * d).map(|_| rng.next_f32()).collect();
        let out = TabTransformer::transformer_block(&x, n_cat, d, 2, &w.blocks[0])
            .expect("transformer_block should succeed");
        assert_eq!(out.len(), n_cat * d);
    }

    #[test]
    fn transformer_block_output_finite() {
        let d = 8;
        let n_cat = 4;
        let mut rng = LcgRng::new(11);
        let cfg = make_cfg(n_cat, 0, d, 2, 1, 2);
        let model = TabTransformer::new(cfg).expect("new should succeed");
        let w = model.init_weights(&mut rng);
        let x: Vec<f32> = (0..n_cat * d).map(|_| rng.next_f32() - 0.5).collect();
        let out = TabTransformer::transformer_block(&x, n_cat, d, 2, &w.blocks[0])
            .expect("transformer_block should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn linear_matmul_check() {
        // linear(ones, I_4, 0) = ones
        let x = vec![1.0_f32; 4];
        let mut w = vec![0.0_f32; 16];
        for i in 0..4 {
            w[i * 4 + i] = 1.0;
        }
        let b = vec![0.0_f32; 4];
        let out = TabTransformer::linear(&x, &w, &b, 4, 4);
        for (o, e) in out.iter().zip(x.iter()) {
            assert!((o - e).abs() < 1e-6);
        }
    }

    #[test]
    fn relu_zeros_negative() {
        let x = vec![-1.0_f32, 0.0, 1.0];
        let out = TabTransformer::relu(&x);
        assert!((out[0] - 0.0).abs() < 1e-7);
        assert!((out[1] - 0.0).abs() < 1e-7);
        assert!((out[2] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        TabTransformer::softmax(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "sum={s}");
    }

    #[test]
    fn softmax_max_dominates() {
        let mut x = vec![0.1_f32, 0.5, 10.0, 0.2];
        let argmax_before = 2usize;
        TabTransformer::softmax(&mut x);
        let argmax_after = x
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(argmax_after, argmax_before);
    }

    #[test]
    fn tab_transformer_no_cat_features_err() {
        let cfg = TabTransformerConfig {
            n_cat_features: 0,
            cat_n_categories: vec![],
            n_cont_features: 2,
            embed_dim: 8,
            n_heads: 2,
            n_layers: 1,
            ffn_hidden: 16,
            mlp_hidden_sizes: vec![],
            n_classes: 2,
            dropout_rate: 0.0,
        };
        assert!(TabTransformer::new(cfg).is_err());
    }

    #[test]
    fn tab_transformer_single_cat_single_cont() {
        let cfg = TabTransformerConfig {
            n_cat_features: 1,
            cat_n_categories: vec![3],
            n_cont_features: 1,
            embed_dim: 4,
            n_heads: 2,
            n_layers: 1,
            ffn_hidden: 8,
            mlp_hidden_sizes: vec![8],
            n_classes: 2,
            dropout_rate: 0.0,
        };
        let model = TabTransformer::new(cfg).expect("new should succeed");
        let mut rng = LcgRng::new(99);
        let w = model.init_weights(&mut rng);
        let logits = model
            .forward(&[1], &[0.7], &w)
            .expect("forward should succeed");
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tab_transformer_multi_class_output() {
        let cfg = make_cfg(2, 2, 8, 2, 1, 5);
        let model = TabTransformer::new(cfg).expect("new should succeed");
        let mut rng = LcgRng::new(42);
        let w = model.init_weights(&mut rng);
        let logits = model
            .forward(&[0, 1], &[1.0, -1.0], &w)
            .expect("forward should succeed");
        assert_eq!(logits.len(), 5);
    }

    #[test]
    fn tab_transformer_cat_id_bounds() {
        let cfg = make_cfg(2, 1, 8, 2, 1, 2);
        let model = TabTransformer::new(cfg).expect("new should succeed");
        let mut rng = LcgRng::new(7);
        let w = model.init_weights(&mut rng);
        // cat_n_categories = [4, 4], so index 4 is out of bounds
        let result = model.forward(&[0, 4], &[0.5], &w);
        assert!(result.is_err());
    }

    #[test]
    fn init_weights_correct_shapes() {
        let cfg = make_cfg(3, 2, 8, 2, 2, 4);
        let model = TabTransformer::new(cfg.clone()).expect("value should be present");
        let mut rng = LcgRng::new(55);
        let w = model.init_weights(&mut rng);

        assert_eq!(w.cat_embeddings.len(), 3);
        for (i, emb) in w.cat_embeddings.iter().enumerate() {
            assert_eq!(emb.len(), cfg.cat_n_categories[i] * cfg.embed_dim);
        }
        assert_eq!(w.col_ln_scale.len(), 3);
        assert_eq!(w.col_ln_bias.len(), 3);
        assert_eq!(w.blocks.len(), 2);
        for b in &w.blocks {
            assert_eq!(b.wq.len(), cfg.embed_dim * cfg.embed_dim);
            assert_eq!(b.ffn_w1.len(), cfg.embed_dim * cfg.ffn_hidden);
            assert_eq!(b.ffn_b1.len(), cfg.ffn_hidden);
            assert_eq!(b.ffn_w2.len(), cfg.ffn_hidden * cfg.embed_dim);
        }
        assert_eq!(w.output_b.len(), 4);
    }

    #[test]
    fn tab_transformer_embed_dim_head_div_err() {
        let cfg = TabTransformerConfig {
            n_cat_features: 2,
            cat_n_categories: vec![3, 4],
            n_cont_features: 1,
            embed_dim: 9,
            n_heads: 2, // 9 % 2 != 0 → error
            n_layers: 1,
            ffn_hidden: 18,
            mlp_hidden_sizes: vec![],
            n_classes: 2,
            dropout_rate: 0.0,
        };
        assert!(TabTransformer::new(cfg).is_err());
    }

    #[test]
    fn tab_transformer_two_layers() {
        let cfg = make_cfg(2, 2, 8, 2, 2, 2);
        let model = TabTransformer::new(cfg).expect("new should succeed");
        let mut rng = LcgRng::new(77);
        let w = model.init_weights(&mut rng);
        let logits = model
            .forward(&[1, 2], &[0.0, 1.0], &w)
            .expect("forward should succeed");
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tab_transformer_zero_cont_features() {
        let cfg = TabTransformerConfig {
            n_cat_features: 3,
            cat_n_categories: vec![4, 4, 4],
            n_cont_features: 0,
            embed_dim: 8,
            n_heads: 2,
            n_layers: 1,
            ffn_hidden: 16,
            mlp_hidden_sizes: vec![16],
            n_classes: 2,
            dropout_rate: 0.0,
        };
        let model = TabTransformer::new(cfg).expect("new should succeed");
        let mut rng = LcgRng::new(88);
        let w = model.init_weights(&mut rng);
        let logits = model
            .forward(&[0, 1, 2], &[], &w)
            .expect("forward should succeed");
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}

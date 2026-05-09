//! FT-Transformer: Feature Tokenizer + Transformer (Gorishniy et al. 2021).
//!
//! Architecture:
//! - `FeatureTokenizer`: maps continuous features → linear tokens, categorical → embedding lookups.
//! - A learnable CLS token is prepended to the feature tokens.
//! - Standard Transformer blocks (Pre-LN MHSA + Pre-LN FFN) are applied.
//! - The CLS token output is passed through a classification head.

use crate::attention::saint::multihead_attention;
use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── FtConfig ────────────────────────────────────────────────────────────────

/// Configuration for `FtTransformer`.
pub struct FtConfig {
    /// Number of continuous input features.
    pub n_cont_features: usize,
    /// Number of categories for each categorical feature.
    pub cat_n_categories: Vec<usize>,
    /// Token embedding dimension `d_token`.
    pub embed_dim: usize,
    /// Number of attention heads (`embed_dim % n_heads == 0`).
    pub n_heads: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// FFN hidden dimension.
    pub ffn_hidden: usize,
    /// Dropout rate (stored; not applied during inference).
    pub dropout_rate: f32,
    /// Output classes (1 for regression).
    pub n_classes: usize,
}

// ─── FeatureTokenizer ─────────────────────────────────────────────────────────

/// Maps raw features to per-feature embedding tokens.
///
/// - Continuous feature j: `token_j = x_j * w_j + b_j` (element-wise per embedding dim).
/// - Categorical feature j: embedding lookup from a learnt table.
pub struct FeatureTokenizer {
    // Continuous: per-feature scale (weight) and bias, each [embed_dim]
    cont_w: Vec<f32>,
    cont_b: Vec<f32>,
    // Categorical: embedding tables per feature [n_cat_j * embed_dim]
    cat_embeds: Vec<Vec<f32>>,
    n_cont: usize,
    embed_dim: usize,
}

impl FeatureTokenizer {
    /// Construct a new `FeatureTokenizer` with random initialisation.
    pub fn new(n_cont: usize, cat_sizes: &[usize], embed_dim: usize, rng: &mut LcgRng) -> Self {
        let mut cont_w = vec![0.0_f32; n_cont * embed_dim];
        let mut cont_b = vec![0.0_f32; n_cont * embed_dim];
        let std = (2.0_f32 / (embed_dim as f32 + 1.0)).sqrt();
        rng.fill_normal_scaled(&mut cont_w, std);
        rng.fill_normal_scaled(&mut cont_b, std);

        let cat_embeds: Vec<Vec<f32>> = cat_sizes
            .iter()
            .map(|&nc| {
                let mut emb = vec![0.0_f32; nc * embed_dim];
                rng.fill_normal_scaled(&mut emb, std);
                emb
            })
            .collect();

        Self {
            cont_w,
            cont_b,
            cat_embeds,
            n_cont,
            embed_dim,
        }
    }

    /// Number of feature tokens produced (continuous + categorical).
    pub fn n_features(&self) -> usize {
        self.n_cont + self.cat_embeds.len()
    }

    /// Tokenise a single sample.
    ///
    /// - `x_cont`: `[n_cont]` continuous values.
    /// - `x_cat`: `[n_cat]` categorical indices.
    /// - Returns: `[(n_cont + n_cat) * embed_dim]` flat token matrix.
    pub fn tokenize(&self, x_cont: &[f32], x_cat: &[usize]) -> TabularResult<Vec<f32>> {
        if x_cont.len() != self.n_cont {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_cont,
                got: x_cont.len(),
            });
        }
        if x_cat.len() != self.cat_embeds.len() {
            return Err(TabularError::DimensionMismatch {
                expected: self.cat_embeds.len(),
                got: x_cat.len(),
            });
        }
        let ed = self.embed_dim;
        let n_total = self.n_features();
        let mut tokens = vec![0.0_f32; n_total * ed];

        // Continuous tokens: token_j[d] = x_j * w_j[d] + b_j[d]
        for j in 0..self.n_cont {
            for d in 0..ed {
                tokens[j * ed + d] = x_cont[j] * self.cont_w[j * ed + d] + self.cont_b[j * ed + d];
            }
        }

        // Categorical tokens: embedding lookup
        for (i, (emb, &cat_idx)) in self.cat_embeds.iter().zip(x_cat.iter()).enumerate() {
            let n_cats = emb.len() / ed;
            if cat_idx >= n_cats {
                return Err(TabularError::CategoricalOutOfRange {
                    feat: self.n_cont + i,
                    val: cat_idx,
                    n: n_cats,
                });
            }
            let tok_out = &mut tokens[(self.n_cont + i) * ed..(self.n_cont + i) * ed + ed];
            tok_out.copy_from_slice(&emb[cat_idx * ed..(cat_idx + 1) * ed]);
        }
        Ok(tokens)
    }
}

// ─── Layer norm helper ────────────────────────────────────────────────────────

fn ln(x: &[f32], g: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    x.iter()
        .zip(g.iter().zip(b.iter()))
        .map(|(&xi, (&gi, &bi))| (xi - mean) / (var + 1e-5).sqrt() * gi + bi)
        .collect()
}

// ─── FtTransformer ───────────────────────────────────────────────────────────

/// FT-Transformer: Feature Tokenizer + standard Transformer.
pub struct FtTransformer {
    tokenizer: FeatureTokenizer,
    cls_token: Vec<f32>,
    // Attention projection matrices: [n_layers][embed_dim * embed_dim]
    wq: Vec<Vec<f32>>,
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    wo: Vec<Vec<f32>>,
    // FFN: W1 [embed_dim → ffn_hidden], W2 [ffn_hidden → embed_dim]
    ffn_w1: Vec<Vec<f32>>,
    ffn_b1: Vec<Vec<f32>>,
    ffn_w2: Vec<Vec<f32>>,
    ffn_b2: Vec<Vec<f32>>,
    // Pre-LN gamma/beta [n_layers], each [embed_dim]
    ln1_g: Vec<Vec<f32>>,
    ln1_b: Vec<Vec<f32>>,
    ln2_g: Vec<Vec<f32>>,
    ln2_b: Vec<Vec<f32>>,
    // Head: [n_classes * embed_dim]
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    config: FtConfig,
}

impl FtTransformer {
    /// Construct a new `FtTransformer` with Xavier weight initialisation.
    pub fn new(cfg: FtConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if cfg.embed_dim == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if !cfg.embed_dim.is_multiple_of(cfg.n_heads) {
            return Err(TabularError::InvalidAttentionDim { dim: cfg.embed_dim });
        }

        let ed = cfg.embed_dim;
        let fh = cfg.ffn_hidden;
        let nl = cfg.n_layers;
        let std = (2.0_f32 / (ed + ed) as f32).sqrt();
        let std_ffn = (2.0_f32 / (ed + fh) as f32).sqrt();

        let tokenizer = FeatureTokenizer::new(cfg.n_cont_features, &cfg.cat_n_categories, ed, rng);

        let mut cls_token = vec![0.0_f32; ed];
        rng.fill_normal_scaled(&mut cls_token, std);

        let mut wq = Vec::with_capacity(nl);
        let mut wk = Vec::with_capacity(nl);
        let mut wv = Vec::with_capacity(nl);
        let mut wo = Vec::with_capacity(nl);
        let mut ffn_w1 = Vec::with_capacity(nl);
        let mut ffn_b1 = Vec::with_capacity(nl);
        let mut ffn_w2 = Vec::with_capacity(nl);
        let mut ffn_b2 = Vec::with_capacity(nl);
        let mut ln1_g = Vec::with_capacity(nl);
        let mut ln1_b = Vec::with_capacity(nl);
        let mut ln2_g = Vec::with_capacity(nl);
        let mut ln2_b = Vec::with_capacity(nl);

        for _ in 0..nl {
            let mut w = vec![0.0_f32; ed * ed];
            rng.fill_normal_scaled(&mut w, std);
            wq.push(w.clone());
            rng.fill_normal_scaled(&mut w, std);
            wk.push(w.clone());
            rng.fill_normal_scaled(&mut w, std);
            wv.push(w.clone());
            rng.fill_normal_scaled(&mut w, std);
            wo.push(w);

            let mut w1 = vec![0.0_f32; fh * ed];
            rng.fill_normal_scaled(&mut w1, std_ffn);
            ffn_w1.push(w1);
            ffn_b1.push(vec![0.0_f32; fh]);

            let mut w2 = vec![0.0_f32; ed * fh];
            rng.fill_normal_scaled(&mut w2, std_ffn);
            ffn_w2.push(w2);
            ffn_b2.push(vec![0.0_f32; ed]);

            ln1_g.push(vec![1.0_f32; ed]);
            ln1_b.push(vec![0.0_f32; ed]);
            ln2_g.push(vec![1.0_f32; ed]);
            ln2_b.push(vec![0.0_f32; ed]);
        }

        let mut head_w = vec![0.0_f32; cfg.n_classes * ed];
        rng.fill_normal_scaled(&mut head_w, std);
        let head_b = vec![0.0_f32; cfg.n_classes];

        Ok(Self {
            tokenizer,
            cls_token,
            wq,
            wk,
            wv,
            wo,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
            ln1_g,
            ln1_b,
            ln2_g,
            ln2_b,
            head_w,
            head_b,
            config: cfg,
        })
    }

    fn apply_ffn(
        x: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        ed: usize,
        fh: usize,
    ) -> Vec<f32> {
        // h = GELU(W1 x + b1)
        let mut h = b1.to_vec();
        for o in 0..fh {
            for i in 0..ed {
                h[o] += w1[o * ed + i] * x[i];
            }
        }
        for v in &mut h {
            *v *= 1.0 / (1.0 + (-1.702 * *v).exp());
        }
        // out = W2 h + b2
        let mut out = b2.to_vec();
        for o in 0..ed {
            for i in 0..fh {
                out[o] += w2[o * fh + i] * h[i];
            }
        }
        out
    }

    /// Forward pass for a single sample.
    ///
    /// Returns logits `[n_classes]`.
    pub fn forward(&self, x_cont: &[f32], x_cat: &[usize]) -> TabularResult<Vec<f32>> {
        let cfg = &self.config;
        let ed = cfg.embed_dim;

        // 1. Tokenize features → [(n_feat) * ed]
        let feat_tokens = self.tokenizer.tokenize(x_cont, x_cat)?;
        let n_feat = self.tokenizer.n_features();

        // 2. Prepend CLS token → [(n_feat + 1) * ed]
        let seq_len = n_feat + 1;
        let mut h = Vec::with_capacity(seq_len * ed);
        h.extend_from_slice(&self.cls_token);
        h.extend_from_slice(&feat_tokens);

        // 3. Transformer layers
        for layer in 0..cfg.n_layers {
            // Pre-LN MHSA
            let mut pre_ln1 = vec![0.0_f32; seq_len * ed];
            for s in 0..seq_len {
                let tok = &h[s * ed..(s + 1) * ed];
                let normed = ln(tok, &self.ln1_g[layer], &self.ln1_b[layer]);
                pre_ln1[s * ed..(s + 1) * ed].copy_from_slice(&normed);
            }

            let attn_out = multihead_attention(
                &pre_ln1,
                &self.wq[layer],
                &self.wk[layer],
                &self.wv[layer],
                &self.wo[layer],
                seq_len,
                ed,
                cfg.n_heads,
            )?;

            // Residual
            for i in 0..seq_len * ed {
                h[i] += attn_out[i];
            }

            // Pre-LN FFN per token
            let mut new_h = Vec::with_capacity(seq_len * ed);
            for s in 0..seq_len {
                let tok = &h[s * ed..(s + 1) * ed];
                let normed = ln(tok, &self.ln2_g[layer], &self.ln2_b[layer]);
                let ffn_out = Self::apply_ffn(
                    &normed,
                    &self.ffn_w1[layer],
                    &self.ffn_b1[layer],
                    &self.ffn_w2[layer],
                    &self.ffn_b2[layer],
                    ed,
                    cfg.ffn_hidden,
                );
                // Residual
                let residual: Vec<f32> = tok
                    .iter()
                    .zip(ffn_out.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                new_h.extend_from_slice(&residual);
            }
            h = new_h;
        }

        // 4. Extract CLS token [0..ed] and apply head
        let cls = &h[0..ed];
        let mut logits = self.head_b.clone();
        for (o, lo) in logits.iter_mut().enumerate() {
            for (d, &cv) in cls.iter().enumerate() {
                *lo += self.head_w[o * ed + d] * cv;
            }
        }
        Ok(logits)
    }

    /// Batch forward: `x_cont` is `[batch * n_cont]`, `x_cat` is `[batch * n_cat]`.
    ///
    /// Returns logits `[batch * n_classes]`.
    pub fn forward_batch(
        &self,
        x_cont: &[f32],
        x_cat: &[usize],
        batch_size: usize,
    ) -> TabularResult<Vec<f32>> {
        let n_cont = self.config.n_cont_features;
        let n_cat = self.config.cat_n_categories.len();
        if x_cont.len() != batch_size * n_cont {
            return Err(TabularError::DimensionMismatch {
                expected: batch_size * n_cont,
                got: x_cont.len(),
            });
        }
        if x_cat.len() != batch_size * n_cat {
            return Err(TabularError::DimensionMismatch {
                expected: batch_size * n_cat,
                got: x_cat.len(),
            });
        }

        let mut all_logits = Vec::with_capacity(batch_size * self.config.n_classes);
        for b in 0..batch_size {
            let cont_row = &x_cont[b * n_cont..(b + 1) * n_cont];
            let cat_row = &x_cat[b * n_cat..(b + 1) * n_cat];
            let logits = self.forward(cont_row, cat_row)?;
            all_logits.extend_from_slice(&logits);
        }
        Ok(all_logits)
    }
}

// ─── Exported alias so users can call self_attention directly ─────────────────
// (already pub in saint module; re-exported via prelude)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn ft_transformer_forward_finite() {
        let mut rng = LcgRng::new(42);
        let cfg = FtConfig {
            n_cont_features: 4,
            cat_n_categories: vec![5, 3],
            embed_dim: 8,
            n_heads: 2,
            n_layers: 2,
            ffn_hidden: 16,
            dropout_rate: 0.1,
            n_classes: 3,
        };
        let model = FtTransformer::new(cfg, &mut rng).unwrap();
        let x_cont = vec![0.5_f32; 4];
        let x_cat = vec![1usize, 0];
        let logits = model.forward(&x_cont, &x_cat).unwrap();
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn feature_tokenizer_shape() {
        let mut rng = LcgRng::new(7);
        let tok = FeatureTokenizer::new(4, &[5, 3], 8, &mut rng);
        let tokens = tok.tokenize(&[0.1, 0.2, 0.3, 0.4], &[1, 2]).unwrap();
        assert_eq!(tokens.len(), (4 + 2) * 8);
    }

    #[test]
    fn ft_transformer_batch() {
        let mut rng = LcgRng::new(13);
        let cfg = FtConfig {
            n_cont_features: 2,
            cat_n_categories: vec![4],
            embed_dim: 4,
            n_heads: 2,
            n_layers: 1,
            ffn_hidden: 8,
            dropout_rate: 0.0,
            n_classes: 2,
        };
        let model = FtTransformer::new(cfg, &mut rng).unwrap();
        let x_cont = vec![0.5_f32; 3 * 2];
        let x_cat = vec![0usize; 3];
        let logits = model.forward_batch(&x_cont, &x_cat, 3).unwrap();
        assert_eq!(logits.len(), 3 * 2);
    }
}

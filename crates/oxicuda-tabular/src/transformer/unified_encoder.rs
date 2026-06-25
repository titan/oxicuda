//! Joint categorical + continuous unified feature encoder.
//!
//! A single [`FeatureTokenizer`](crate::transformer::ft_transformer::FeatureTokenizer)-style
//! path embeds **both** kinds of tabular feature into one merged token sequence
//! that a multi-head self-attention stack can consume:
//!
//! - **Continuous** feature `j`: a learned per-feature rank-1 token
//!   `token_j[d] = x_j · w_j[d] + b_j[d]` (Gorishniy et al. 2021).
//! - **Categorical** feature `k`: a per-category lookup embedding
//!   `token_k = E_k[idx]`.
//!
//! The tokens are concatenated **continuous-first, then categorical**, an
//! optional CLS token is prepended, and the resulting `[seq_len × embed_dim]`
//! matrix is passed through Pre-LN transformer blocks.  The encoder returns the
//! full contextualised token sequence; callers attach their own task head.
//!
//! This differs from the FT-Transformer module by exposing the **merged token
//! sequence** directly (so downstream code can pool / gather any subset) and by
//! making the continuous and categorical contributions independently inspectable
//! via [`UnifiedEncoder::tokenize`].

use crate::attention::saint::multihead_attention;
use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── Config ─────────────────────────────────────────────────────────────────

/// Configuration for [`UnifiedEncoder`].
pub struct UnifiedEncoderConfig {
    /// Number of continuous input features.
    pub n_cont: usize,
    /// Cardinality (number of categories) of each categorical feature.
    pub cat_cardinalities: Vec<usize>,
    /// Token / embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads (`embed_dim % n_heads == 0`).
    pub n_heads: usize,
    /// Number of Pre-LN transformer blocks (may be 0 for tokenisation-only).
    pub n_layers: usize,
    /// FFN hidden dimension.
    pub ffn_hidden: usize,
    /// Whether to prepend a learnable CLS token.
    pub use_cls: bool,
}

// ─── Joint tokenizer ──────────────────────────────────────────────────────────

/// Joint continuous + categorical tokenizer producing one merged token matrix.
pub struct JointTokenizer {
    cont_w: Vec<f32>, // [n_cont * embed_dim]
    cont_b: Vec<f32>, // [n_cont * embed_dim]
    cat_embeds: Vec<Vec<f32>>,
    n_cont: usize,
    embed_dim: usize,
}

impl JointTokenizer {
    /// Construct with N(0, σ) initialisation, `σ = sqrt(2 / (embed_dim + 1))`.
    pub fn new(
        n_cont: usize,
        cat_cardinalities: &[usize],
        embed_dim: usize,
        rng: &mut LcgRng,
    ) -> Self {
        let std = (2.0_f32 / (embed_dim as f32 + 1.0)).sqrt();
        let mut cont_w = vec![0.0_f32; n_cont * embed_dim];
        let mut cont_b = vec![0.0_f32; n_cont * embed_dim];
        rng.fill_normal_scaled(&mut cont_w, std);
        rng.fill_normal_scaled(&mut cont_b, std);
        let cat_embeds = cat_cardinalities
            .iter()
            .map(|&c| {
                let mut e = vec![0.0_f32; c * embed_dim];
                rng.fill_normal_scaled(&mut e, std);
                e
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

    /// Number of feature tokens (continuous + categorical), excluding any CLS.
    pub fn n_tokens(&self) -> usize {
        self.n_cont + self.cat_embeds.len()
    }

    /// Tokenise one sample into a flat `[(n_cont + n_cat) * embed_dim]` matrix,
    /// continuous tokens first, then categorical.
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
        let mut tokens = vec![0.0_f32; self.n_tokens() * ed];

        // Continuous: rank-1 affine token.
        for j in 0..self.n_cont {
            for d in 0..ed {
                tokens[j * ed + d] = x_cont[j] * self.cont_w[j * ed + d] + self.cont_b[j * ed + d];
            }
        }
        // Categorical: embedding lookup.
        for (i, (emb, &idx)) in self.cat_embeds.iter().zip(x_cat.iter()).enumerate() {
            let card = emb.len() / ed;
            if idx >= card {
                return Err(TabularError::CategoricalOutOfRange {
                    feat: self.n_cont + i,
                    val: idx,
                    n: card,
                });
            }
            let base = (self.n_cont + i) * ed;
            tokens[base..base + ed].copy_from_slice(&emb[idx * ed..idx * ed + ed]);
        }
        Ok(tokens)
    }
}

// ─── LayerNorm helper ─────────────────────────────────────────────────────────

fn ln(x: &[f32], g: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter()
        .zip(g.iter().zip(b.iter()))
        .map(|(&xi, (&gi, &bi))| (xi - mean) * inv * gi + bi)
        .collect()
}

// ─── UnifiedEncoder ───────────────────────────────────────────────────────────

/// Joint continuous + categorical encoder with Pre-LN transformer blocks.
pub struct UnifiedEncoder {
    tokenizer: JointTokenizer,
    cls_token: Option<Vec<f32>>,
    wq: Vec<Vec<f32>>,
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    wo: Vec<Vec<f32>>,
    ffn_w1: Vec<Vec<f32>>,
    ffn_b1: Vec<Vec<f32>>,
    ffn_w2: Vec<Vec<f32>>,
    ffn_b2: Vec<Vec<f32>>,
    ln1_g: Vec<Vec<f32>>,
    ln1_b: Vec<Vec<f32>>,
    ln2_g: Vec<Vec<f32>>,
    ln2_b: Vec<Vec<f32>>,
    config: UnifiedEncoderConfig,
}

impl UnifiedEncoder {
    /// Construct a new encoder with Xavier-style initialisation.
    pub fn new(cfg: UnifiedEncoderConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if cfg.embed_dim == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if cfg.n_heads == 0 || !cfg.embed_dim.is_multiple_of(cfg.n_heads) {
            return Err(TabularError::InvalidAttentionDim { dim: cfg.embed_dim });
        }
        if cfg.n_cont == 0 && cfg.cat_cardinalities.is_empty() {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }

        let ed = cfg.embed_dim;
        let fh = cfg.ffn_hidden;
        let nl = cfg.n_layers;
        let std = (2.0_f32 / (ed + ed) as f32).sqrt();
        let std_ffn = (2.0_f32 / (ed + fh) as f32).sqrt();

        let tokenizer = JointTokenizer::new(cfg.n_cont, &cfg.cat_cardinalities, ed, rng);

        let cls_token = if cfg.use_cls {
            let mut c = vec![0.0_f32; ed];
            rng.fill_normal_scaled(&mut c, std);
            Some(c)
        } else {
            None
        };

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
            config: cfg,
        })
    }

    /// Number of tokens in the sequence the encoder operates on (including CLS
    /// if enabled).
    pub fn seq_len(&self) -> usize {
        self.tokenizer.n_tokens() + usize::from(self.cls_token.is_some())
    }

    /// Tokenise only (no attention).  Returns the merged token matrix, with the
    /// CLS token prepended when configured: `[seq_len * embed_dim]`.
    pub fn tokenize(&self, x_cont: &[f32], x_cat: &[usize]) -> TabularResult<Vec<f32>> {
        let ed = self.config.embed_dim;
        let feat = self.tokenizer.tokenize(x_cont, x_cat)?;
        if let Some(cls) = &self.cls_token {
            let mut out = Vec::with_capacity((self.tokenizer.n_tokens() + 1) * ed);
            out.extend_from_slice(cls);
            out.extend_from_slice(&feat);
            Ok(out)
        } else {
            Ok(feat)
        }
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
        let mut h = b1.to_vec();
        for o in 0..fh {
            for (i, &xi) in x.iter().enumerate() {
                h[o] += w1[o * ed + i] * xi;
            }
        }
        for v in &mut h {
            *v *= 1.0 / (1.0 + (-1.702 * *v).exp());
        }
        let mut out = b2.to_vec();
        for o in 0..ed {
            for i in 0..fh {
                out[o] += w2[o * fh + i] * h[i];
            }
        }
        out
    }

    /// Full forward: tokenise → (optional CLS) → Pre-LN MHSA blocks.
    ///
    /// Returns the contextualised token sequence `[seq_len * embed_dim]`.
    pub fn forward(&self, x_cont: &[f32], x_cat: &[usize]) -> TabularResult<Vec<f32>> {
        let cfg = &self.config;
        let ed = cfg.embed_dim;
        let mut h = self.tokenize(x_cont, x_cat)?;
        let seq = self.seq_len();

        for layer in 0..cfg.n_layers {
            // Pre-LN MHSA
            let mut pre = vec![0.0_f32; seq * ed];
            for s in 0..seq {
                let normed = ln(
                    &h[s * ed..(s + 1) * ed],
                    &self.ln1_g[layer],
                    &self.ln1_b[layer],
                );
                pre[s * ed..(s + 1) * ed].copy_from_slice(&normed);
            }
            let attn = multihead_attention(
                &pre,
                &self.wq[layer],
                &self.wk[layer],
                &self.wv[layer],
                &self.wo[layer],
                seq,
                ed,
                cfg.n_heads,
            )?;
            for i in 0..seq * ed {
                h[i] += attn[i];
            }
            // Pre-LN FFN
            let mut new_h = Vec::with_capacity(seq * ed);
            for s in 0..seq {
                let tok = &h[s * ed..(s + 1) * ed];
                let normed = ln(tok, &self.ln2_g[layer], &self.ln2_b[layer]);
                let ffn = Self::apply_ffn(
                    &normed,
                    &self.ffn_w1[layer],
                    &self.ffn_b1[layer],
                    &self.ffn_w2[layer],
                    &self.ffn_b2[layer],
                    ed,
                    cfg.ffn_hidden,
                );
                for (a, b) in tok.iter().zip(ffn.iter()) {
                    new_h.push(a + b);
                }
            }
            h = new_h;
        }
        Ok(h)
    }

    /// Mean-pool the feature tokens (excluding any CLS) into `[embed_dim]`.
    pub fn pooled(&self, x_cont: &[f32], x_cat: &[usize]) -> TabularResult<Vec<f32>> {
        let ed = self.config.embed_dim;
        let seq = self.forward(x_cont, x_cat)?;
        let start = usize::from(self.cls_token.is_some());
        let n = self.tokenizer.n_tokens();
        let mut pooled = vec![0.0_f32; ed];
        for t in start..start + n {
            for d in 0..ed {
                pooled[d] += seq[t * ed + d];
            }
        }
        for v in &mut pooled {
            *v /= n as f32;
        }
        Ok(pooled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(use_cls: bool) -> UnifiedEncoderConfig {
        UnifiedEncoderConfig {
            n_cont: 3,
            cat_cardinalities: vec![5, 4],
            embed_dim: 8,
            n_heads: 2,
            n_layers: 2,
            ffn_hidden: 16,
            use_cls,
        }
    }

    #[test]
    fn token_count_matches_mixed_input() {
        let mut rng = LcgRng::new(1);
        let enc = UnifiedEncoder::new(cfg(false), &mut rng).expect("new");
        let toks = enc.tokenize(&[0.1, 0.2, 0.3], &[1, 2]).expect("tokenize");
        // 3 continuous + 2 categorical, no CLS → 5 tokens × 8 dims.
        assert_eq!(toks.len(), (3 + 2) * 8);
        assert_eq!(enc.seq_len(), 5);
    }

    #[test]
    fn cls_prepended_increases_seq() {
        let mut rng = LcgRng::new(2);
        let enc = UnifiedEncoder::new(cfg(true), &mut rng).expect("new");
        assert_eq!(enc.seq_len(), 6); // 5 feature tokens + CLS
        let toks = enc.tokenize(&[0.0, 0.0, 0.0], &[0, 0]).expect("tok");
        assert_eq!(toks.len(), 6 * 8);
    }

    #[test]
    fn continuous_path_contributes() {
        // Two samples differing only in continuous features must produce
        // different continuous tokens but identical categorical tokens.
        let mut rng = LcgRng::new(3);
        let enc = UnifiedEncoder::new(cfg(false), &mut rng).expect("new");
        let a = enc.tokenize(&[0.1, 0.2, 0.3], &[1, 2]).expect("a");
        let b = enc.tokenize(&[0.9, -0.4, 1.7], &[1, 2]).expect("b");
        let ed = 8;
        // Continuous tokens (first 3) differ.
        let cont_differs = (0..3 * ed).any(|i| (a[i] - b[i]).abs() > 1e-6);
        assert!(cont_differs, "continuous tokens should depend on x_cont");
        // Categorical tokens (last 2) identical (same indices).
        for i in 3 * ed..5 * ed {
            assert!((a[i] - b[i]).abs() < 1e-9, "categorical token changed");
        }
    }

    #[test]
    fn categorical_path_contributes() {
        // Differing categorical index changes only the relevant categorical token.
        let mut rng = LcgRng::new(4);
        let enc = UnifiedEncoder::new(cfg(false), &mut rng).expect("new");
        let a = enc.tokenize(&[0.5, 0.5, 0.5], &[1, 2]).expect("a");
        let b = enc.tokenize(&[0.5, 0.5, 0.5], &[3, 2]).expect("b");
        let ed = 8;
        // First categorical token (index 3) changed.
        let cat0_differs = (3 * ed..4 * ed).any(|i| (a[i] - b[i]).abs() > 1e-6);
        assert!(cat0_differs, "changing cat index should change its token");
        // Continuous tokens unchanged.
        for i in 0..3 * ed {
            assert!((a[i] - b[i]).abs() < 1e-9, "continuous token changed");
        }
        // Second categorical token (unchanged index) identical.
        for i in 4 * ed..5 * ed {
            assert!((a[i] - b[i]).abs() < 1e-9, "second cat token changed");
        }
    }

    #[test]
    fn forward_finite_and_deterministic() {
        let mut rng = LcgRng::new(5);
        let enc = UnifiedEncoder::new(cfg(true), &mut rng).expect("new");
        let out1 = enc.forward(&[0.3, -0.2, 0.8], &[2, 1]).expect("fwd1");
        let out2 = enc.forward(&[0.3, -0.2, 0.8], &[2, 1]).expect("fwd2");
        assert_eq!(out1.len(), 6 * 8);
        assert!(
            out1.iter().all(|v| v.is_finite()),
            "encoder output must be finite"
        );
        // Deterministic for the same input.
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-12, "forward must be deterministic");
        }
    }

    #[test]
    fn pooled_shape() {
        let mut rng = LcgRng::new(6);
        let enc = UnifiedEncoder::new(cfg(true), &mut rng).expect("new");
        let p = enc.pooled(&[0.1, 0.2, 0.3], &[0, 1]).expect("pooled");
        assert_eq!(p.len(), 8);
        assert!(p.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cont_only_and_cat_only_ok() {
        // Continuous-only.
        let mut rng = LcgRng::new(7);
        let c = UnifiedEncoderConfig {
            n_cont: 4,
            cat_cardinalities: vec![],
            embed_dim: 4,
            n_heads: 2,
            n_layers: 1,
            ffn_hidden: 8,
            use_cls: false,
        };
        let enc = UnifiedEncoder::new(c, &mut rng).expect("new cont-only");
        let out = enc.forward(&[0.1, 0.2, 0.3, 0.4], &[]).expect("fwd");
        assert_eq!(out.len(), 4 * 4);

        // Categorical-only.
        let c2 = UnifiedEncoderConfig {
            n_cont: 0,
            cat_cardinalities: vec![3, 3],
            embed_dim: 4,
            n_heads: 1,
            n_layers: 1,
            ffn_hidden: 8,
            use_cls: true,
        };
        let enc2 = UnifiedEncoder::new(c2, &mut rng).expect("new cat-only");
        let out2 = enc2.forward(&[], &[1, 2]).expect("fwd2");
        assert_eq!(out2.len(), (2 + 1) * 4);
    }

    #[test]
    fn rejects_oob_category() {
        let mut rng = LcgRng::new(8);
        let enc = UnifiedEncoder::new(cfg(false), &mut rng).expect("new");
        // Second categorical has cardinality 4 → index 4 is out of range.
        assert!(enc.tokenize(&[0.0, 0.0, 0.0], &[0, 4]).is_err());
    }
}

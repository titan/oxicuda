//! TabPFN-style prior-data-fitted network for small-dataset classification.
//!
//! Reference: Hollmann et al. "TabPFN: A Transformer That Solves Small Tabular
//! Classification Problems in a Second" (ICLR 2023).
//!
//! TabPFN performs *in-context learning*: a single, dataset-agnostic transformer
//! receives a labelled **support set** and an **unlabelled query** in the same
//! sequence, then predicts the query labels by attending over the support
//! examples — without any gradient step on the new dataset.  This module
//! implements the CPU forward engine of that mechanism:
//!
//! 1. Each row is embedded with a shared linear feature encoder.
//! 2. Support rows additionally receive a learnable per-class **label
//!    embedding**, added to their token; query rows receive a dedicated
//!    "unknown-label" embedding.
//! 3. One or more transformer blocks mix the tokens. Query tokens may attend to
//!    every support token (and to themselves) but support tokens never attend to
//!    queries — an attention mask that makes the support set a fixed context.
//! 4. A linear classification head maps each query token to class logits.
//!
//! The weights here are randomly initialised (the "prior fit" stage that trains
//! them on synthetic prior datasets is gradient training, which is out of scope
//! for this inference-oriented crate); the forward pass, the in-context masking,
//! and the label-conditioning are the genuinely novel, exactly-implemented part.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;
use crate::transformer::autoint::layer_norm;

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for the TabPFN in-context classifier.
#[derive(Debug, Clone)]
pub struct TabPfnConfig {
    /// Number of input features per row.
    pub n_features: usize,
    /// Maximum number of classes the model can distinguish.
    pub n_classes: usize,
    /// Token embedding dimension (divisible by `n_heads`).
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// FFN hidden dimension.
    pub ffn_hidden: usize,
}

impl Default for TabPfnConfig {
    fn default() -> Self {
        Self {
            n_features: 8,
            n_classes: 3,
            embed_dim: 16,
            n_heads: 2,
            n_layers: 2,
            ffn_hidden: 32,
        }
    }
}

// ─── Block weights ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PfnBlock {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
    ffn_w1: Vec<f32>,
    ffn_b1: Vec<f32>,
    ffn_w2: Vec<f32>,
    ffn_b2: Vec<f32>,
}

impl PfnBlock {
    fn new(cfg: &TabPfnConfig, rng: &mut LcgRng) -> Self {
        let ed = cfg.embed_dim;
        let k = (6.0_f32 / ed as f32).sqrt();
        let k_ffn = (6.0_f32 / (ed + cfg.ffn_hidden) as f32).sqrt();
        let fill = |n: usize, b: f32, rng: &mut LcgRng| -> Vec<f32> {
            (0..n).map(|_| rng.next_f32() * 2.0 * b - b).collect()
        };
        let n = ed * ed;
        Self {
            wq: fill(n, k, rng),
            wk: fill(n, k, rng),
            wv: fill(n, k, rng),
            wo: fill(n, k, rng),
            ln1_g: vec![1.0; ed],
            ln1_b: vec![0.0; ed],
            ln2_g: vec![1.0; ed],
            ln2_b: vec![0.0; ed],
            ffn_w1: fill(ed * cfg.ffn_hidden, k_ffn, rng),
            ffn_b1: vec![0.0; cfg.ffn_hidden],
            ffn_w2: fill(cfg.ffn_hidden * ed, k_ffn, rng),
            ffn_b2: vec![0.0; ed],
        }
    }
}

// ─── Model ────────────────────────────────────────────────────────────────────

/// TabPFN in-context classifier.
#[derive(Debug, Clone)]
pub struct TabPfn {
    config: TabPfnConfig,
    /// Feature encoder weight `[embed_dim × n_features]`.
    enc_w: Vec<f32>,
    /// Feature encoder bias `[embed_dim]`.
    enc_b: Vec<f32>,
    /// Per-class label embedding `[n_classes × embed_dim]`.
    label_emb: Vec<f32>,
    /// Embedding for query (unknown-label) tokens `[embed_dim]`.
    query_emb: Vec<f32>,
    blocks: Vec<PfnBlock>,
    /// Classification head weight `[n_classes × embed_dim]`.
    head_w: Vec<f32>,
    /// Classification head bias `[n_classes]`.
    head_b: Vec<f32>,
}

impl TabPfn {
    /// Build a TabPFN model with randomly initialised prior weights.
    ///
    /// # Errors
    /// Returns [`TabularError::InvalidAttentionDim`] if `embed_dim` is not
    /// divisible by `n_heads`, or [`TabularError::InvalidParameter`] /
    /// [`TabularError::InvalidFeatureCount`] for other degenerate values.
    pub fn new(config: TabPfnConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if config.n_features == 0 {
            return Err(TabularError::InvalidFeatureCount {
                n: config.n_features,
            });
        }
        if config.n_classes < 2 {
            return Err(TabularError::InvalidParameter {
                name: "n_classes".into(),
                msg: "must be >= 2".into(),
            });
        }
        if config.embed_dim == 0
            || config.n_heads == 0
            || !config.embed_dim.is_multiple_of(config.n_heads)
        {
            return Err(TabularError::InvalidAttentionDim {
                dim: config.embed_dim,
            });
        }
        if config.n_layers == 0 || config.ffn_hidden == 0 {
            return Err(TabularError::InvalidParameter {
                name: "n_layers/ffn_hidden".into(),
                msg: "must be > 0".into(),
            });
        }
        let ed = config.embed_dim;
        let k = (6.0_f32 / (config.n_features + ed) as f32).sqrt();
        let fill = |n: usize, b: f32, rng: &mut LcgRng| -> Vec<f32> {
            (0..n).map(|_| rng.next_f32() * 2.0 * b - b).collect()
        };
        let enc_w = fill(ed * config.n_features, k, rng);
        let enc_b = vec![0.0_f32; ed];
        let k_lab = (6.0_f32 / ed as f32).sqrt();
        let label_emb = fill(config.n_classes * ed, k_lab, rng);
        let query_emb = fill(ed, k_lab, rng);
        let blocks: Vec<PfnBlock> = (0..config.n_layers)
            .map(|_| PfnBlock::new(&config, rng))
            .collect();
        let k_head = (6.0_f32 / (ed + config.n_classes) as f32).sqrt();
        let head_w = fill(config.n_classes * ed, k_head, rng);
        let head_b = vec![0.0_f32; config.n_classes];
        Ok(Self {
            config,
            enc_w,
            enc_b,
            label_emb,
            query_emb,
            blocks,
            head_w,
            head_b,
        })
    }

    /// Number of input features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.config.n_features
    }

    /// Number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.config.n_classes
    }

    /// Encode one feature row to an `embed_dim` token: `enc_w · x + enc_b`.
    fn encode_row(&self, x: &[f32]) -> Vec<f32> {
        let ed = self.config.embed_dim;
        let nf = self.config.n_features;
        let mut t = vec![0.0_f32; ed];
        for (o, to) in t.iter_mut().enumerate() {
            let w_row = &self.enc_w[o * nf..(o + 1) * nf];
            let acc: f32 = w_row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
            *to = self.enc_b[o] + acc;
        }
        t
    }

    /// In-context prediction: classify `query` rows given a labelled support set.
    ///
    /// * `support_x`: `[n_support × n_features]` row-major.
    /// * `support_y`: `[n_support]` integer class labels in `[0, n_classes)`.
    /// * `query_x`: `[n_query × n_features]` row-major.
    ///
    /// Returns `[n_query × n_classes]` softmax probabilities, row-major.
    ///
    /// # Errors
    /// Returns an error on any shape mismatch or out-of-range label.
    pub fn predict(
        &self,
        support_x: &[f32],
        support_y: &[usize],
        n_support: usize,
        query_x: &[f32],
        n_query: usize,
    ) -> TabularResult<Vec<f32>> {
        let nf = self.config.n_features;
        let ed = self.config.embed_dim;
        let nc = self.config.n_classes;
        if n_support == 0 {
            return Err(TabularError::InsufficientSamples { need: 1, got: 0 });
        }
        if n_query == 0 {
            return Err(TabularError::EmptyInput);
        }
        if support_x.len() != n_support * nf {
            return Err(TabularError::DimensionMismatch {
                expected: n_support * nf,
                got: support_x.len(),
            });
        }
        if support_y.len() != n_support {
            return Err(TabularError::DimensionMismatch {
                expected: n_support,
                got: support_y.len(),
            });
        }
        if query_x.len() != n_query * nf {
            return Err(TabularError::DimensionMismatch {
                expected: n_query * nf,
                got: query_x.len(),
            });
        }
        let seq = n_support + n_query;
        // Build the token matrix [seq × ed]: support rows first, then queries.
        let mut tokens = vec![0.0_f32; seq * ed];
        for s in 0..n_support {
            let label = support_y[s];
            if label >= nc {
                return Err(TabularError::LabelOutOfRange {
                    label,
                    n_classes: nc,
                });
            }
            let enc = self.encode_row(&support_x[s * nf..(s + 1) * nf]);
            for d in 0..ed {
                tokens[s * ed + d] = enc[d] + self.label_emb[label * ed + d];
            }
        }
        for q in 0..n_query {
            let enc = self.encode_row(&query_x[q * nf..(q + 1) * nf]);
            let row = n_support + q;
            for d in 0..ed {
                tokens[row * ed + d] = enc[d] + self.query_emb[d];
            }
        }
        // Transformer blocks with in-context attention mask.
        for block in &self.blocks {
            tokens = self.block_forward(block, &tokens, seq, n_support)?;
        }
        // Classify each query token with a softmax head.
        let mut probs = vec![0.0_f32; n_query * nc];
        for q in 0..n_query {
            let row = &tokens[(n_support + q) * ed..(n_support + q + 1) * ed];
            let mut logits = vec![0.0_f32; nc];
            for (c, lc) in logits.iter_mut().enumerate() {
                let w_row = &self.head_w[c * ed..(c + 1) * ed];
                let acc: f32 = w_row.iter().zip(row.iter()).map(|(&w, &x)| w * x).sum();
                *lc = self.head_b[c] + acc;
            }
            let sm = softmax(&logits);
            probs[q * nc..(q + 1) * nc].copy_from_slice(&sm);
        }
        Ok(probs)
    }

    /// Predict the single most-likely class for each query row.
    ///
    /// # Errors
    /// Propagates any error from [`TabPfn::predict`].
    pub fn predict_labels(
        &self,
        support_x: &[f32],
        support_y: &[usize],
        n_support: usize,
        query_x: &[f32],
        n_query: usize,
    ) -> TabularResult<Vec<usize>> {
        let nc = self.config.n_classes;
        let probs = self.predict(support_x, support_y, n_support, query_x, n_query)?;
        let mut labels = vec![0usize; n_query];
        for (q, lab) in labels.iter_mut().enumerate() {
            let row = &probs[q * nc..(q + 1) * nc];
            let mut best = 0usize;
            let mut best_v = row[0];
            for (c, &p) in row.iter().enumerate() {
                if p > best_v {
                    best_v = p;
                    best = c;
                }
            }
            *lab = best;
        }
        Ok(labels)
    }

    /// One Pre-LN transformer block with the in-context attention mask:
    /// support tokens attend only to support tokens; query tokens attend to all
    /// support tokens plus themselves.
    fn block_forward(
        &self,
        block: &PfnBlock,
        tokens: &[f32],
        seq: usize,
        n_support: usize,
    ) -> TabularResult<Vec<f32>> {
        let ed = self.config.embed_dim;
        // Pre-LN.
        let mut normed = vec![0.0_f32; seq * ed];
        for t in 0..seq {
            let ln = layer_norm(
                &tokens[t * ed..(t + 1) * ed],
                &block.ln1_g,
                &block.ln1_b,
                1e-5,
            );
            normed[t * ed..(t + 1) * ed].copy_from_slice(&ln);
        }
        let attn = self.masked_attention(block, &normed, seq, n_support);
        // Residual.
        let mut h = vec![0.0_f32; seq * ed];
        for i in 0..seq * ed {
            h[i] = tokens[i] + attn[i];
        }
        // FFN with Pre-LN + residual.
        let mut out = vec![0.0_f32; seq * ed];
        for t in 0..seq {
            let ln = layer_norm(&h[t * ed..(t + 1) * ed], &block.ln2_g, &block.ln2_b, 1e-5);
            let mut hid = vec![0.0_f32; self.config.ffn_hidden];
            for (j, hj) in hid.iter_mut().enumerate() {
                let w_row = &block.ffn_w1[j * ed..(j + 1) * ed];
                let acc: f32 = w_row.iter().zip(ln.iter()).map(|(&w, &x)| w * x).sum();
                *hj = (block.ffn_b1[j] + acc).max(0.0);
            }
            for d in 0..ed {
                let mut acc = block.ffn_b2[d];
                for (j, &hj) in hid.iter().enumerate() {
                    acc += block.ffn_w2[d * self.config.ffn_hidden + j] * hj;
                }
                out[t * ed + d] = h[t * ed + d] + acc;
            }
        }
        Ok(out)
    }

    /// Masked multi-head self-attention. Position `i` may attend to position `j`
    /// iff `j` is a support token, or `j == i` (a query attending to itself).
    fn masked_attention(
        &self,
        block: &PfnBlock,
        x: &[f32],
        seq: usize,
        n_support: usize,
    ) -> Vec<f32> {
        let ed = self.config.embed_dim;
        let n_heads = self.config.n_heads;
        let head_dim = ed / n_heads;
        let q = self.project(&block.wq, x, seq);
        let k = self.project(&block.wk, x, seq);
        let v = self.project(&block.wv, x, seq);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut concat = vec![0.0_f32; seq * ed];
        for h in 0..n_heads {
            let h0 = h * head_dim;
            for i in 0..seq {
                let mut scores = vec![f32::NEG_INFINITY; seq];
                for (j, sj) in scores.iter_mut().enumerate() {
                    let visible = j < n_support || j == i;
                    if !visible {
                        continue;
                    }
                    let mut d = 0.0_f32;
                    for c in 0..head_dim {
                        d += q[i * ed + h0 + c] * k[j * ed + h0 + c];
                    }
                    *sj = d * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores
                    .iter()
                    .map(|&z| if z.is_finite() { (z - max).exp() } else { 0.0 })
                    .collect();
                let denom: f32 = exps.iter().sum::<f32>().max(1e-30);
                for c in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for j in 0..seq {
                        acc += (exps[j] / denom) * v[j * ed + h0 + c];
                    }
                    concat[i * ed + h0 + c] = acc;
                }
            }
        }
        self.project(&block.wo, &concat, seq)
    }

    /// `y = x · Wᵀ` for row-major `[seq × ed]` input and `[ed × ed]` weight.
    fn project(&self, w: &[f32], x: &[f32], seq: usize) -> Vec<f32> {
        let ed = self.config.embed_dim;
        let mut out = vec![0.0_f32; seq * ed];
        for t in 0..seq {
            for o in 0..ed {
                let mut acc = 0.0_f32;
                for i in 0..ed {
                    acc += x[t * ed + i] * w[o * ed + i];
                }
                out[t * ed + o] = acc;
            }
        }
        out
    }
}

/// Numerically-stable softmax.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut e: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let s: f32 = e.iter().sum::<f32>().max(1e-30);
    for v in &mut e {
        *v /= s;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> TabPfnConfig {
        TabPfnConfig {
            n_features: 4,
            n_classes: 3,
            embed_dim: 16,
            n_heads: 4,
            n_layers: 2,
            ffn_hidden: 32,
        }
    }

    #[test]
    fn rejects_bad_embed_dim() {
        let mut rng = LcgRng::new(1);
        let cfg = TabPfnConfig {
            embed_dim: 10,
            n_heads: 4,
            ..small_cfg()
        };
        assert!(TabPfn::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn rejects_single_class() {
        let mut rng = LcgRng::new(1);
        let cfg = TabPfnConfig {
            n_classes: 1,
            ..small_cfg()
        };
        assert!(TabPfn::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn predict_shape_and_simplex() {
        let mut rng = LcgRng::new(2);
        let cfg = small_cfg();
        let model = TabPfn::new(cfg.clone(), &mut rng).expect("new");
        let n_support = 6;
        let n_query = 3;
        let nf = cfg.n_features;
        let mut sx = vec![0.0_f32; n_support * nf];
        rng.fill_normal(&mut sx);
        let sy: Vec<usize> = (0..n_support).map(|i| i % cfg.n_classes).collect();
        let mut qx = vec![0.0_f32; n_query * nf];
        rng.fill_normal(&mut qx);
        let probs = model
            .predict(&sx, &sy, n_support, &qx, n_query)
            .expect("predict");
        assert_eq!(probs.len(), n_query * cfg.n_classes);
        for q in 0..n_query {
            let row = &probs[q * cfg.n_classes..(q + 1) * cfg.n_classes];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {q} sums to {sum}");
            assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
    }

    #[test]
    fn out_of_range_label_errors() {
        let mut rng = LcgRng::new(3);
        let cfg = small_cfg();
        let model = TabPfn::new(cfg.clone(), &mut rng).expect("new");
        let nf = cfg.n_features;
        let sx = vec![0.0_f32; 2 * nf];
        let sy = vec![0usize, cfg.n_classes + 1]; // out of range
        let qx = vec![0.0_f32; nf];
        assert!(model.predict(&sx, &sy, 2, &qx, 1).is_err());
    }

    #[test]
    fn predict_labels_in_range() {
        let mut rng = LcgRng::new(4);
        let cfg = small_cfg();
        let model = TabPfn::new(cfg.clone(), &mut rng).expect("new");
        let nf = cfg.n_features;
        let n_support = 5;
        let mut sx = vec![0.0_f32; n_support * nf];
        rng.fill_normal(&mut sx);
        let sy: Vec<usize> = (0..n_support).map(|i| i % cfg.n_classes).collect();
        let qx = vec![0.3_f32; 2 * nf];
        let labels = model
            .predict_labels(&sx, &sy, n_support, &qx, 2)
            .expect("labels");
        assert_eq!(labels.len(), 2);
        assert!(labels.iter().all(|&l| l < cfg.n_classes));
    }

    #[test]
    fn determinism_same_seed() {
        let cfg = small_cfg();
        let mut r1 = LcgRng::new(55);
        let mut r2 = LcgRng::new(55);
        let m1 = TabPfn::new(cfg.clone(), &mut r1).expect("new");
        let m2 = TabPfn::new(cfg.clone(), &mut r2).expect("new");
        let nf = cfg.n_features;
        let sx = vec![0.1_f32; 3 * nf];
        let sy = vec![0usize, 1, 2];
        let qx = vec![0.2_f32; nf];
        let p1 = m1.predict(&sx, &sy, 3, &qx, 1).expect("p");
        let p2 = m2.predict(&sx, &sy, 3, &qx, 1).expect("p");
        assert_eq!(p1, p2);
    }

    #[test]
    fn query_does_not_leak_into_support_context() {
        // Two identical support sets but different *query* rows must yield the
        // SAME prediction for a shared third query — because support tokens never
        // attend to query tokens, so adding extra queries cannot change an
        // existing query's context-derived logits.
        let mut rng = LcgRng::new(6);
        let cfg = small_cfg();
        let model = TabPfn::new(cfg.clone(), &mut rng).expect("new");
        let nf = cfg.n_features;
        let n_support = 4;
        let mut sx = vec![0.0_f32; n_support * nf];
        rng.fill_normal(&mut sx);
        let sy: Vec<usize> = vec![0, 1, 2, 0];
        let shared_q = vec![0.5_f32; nf];

        // Case A: just the shared query.
        let a = model.predict(&sx, &sy, n_support, &shared_q, 1).expect("a");

        // Case B: shared query followed by another, different query.
        let mut qx = shared_q.clone();
        qx.extend(vec![-0.7_f32; nf]);
        let b = model.predict(&sx, &sy, n_support, &qx, 2).expect("b");

        // The shared query's probabilities (row 0) must match across A and B.
        for c in 0..cfg.n_classes {
            assert!(
                (a[c] - b[c]).abs() < 1e-5,
                "class {c}: {} vs {}",
                a[c],
                b[c]
            );
        }
    }
}

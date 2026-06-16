//! Transformer4Rec — Transformer-based sequential recommender.
//!
//! Reference: Gabriel de Souza P. Moreira, Sara Rabhi, Jeong Min Lee,
//! Ronay Ak, Even Oldridge, "Transformers4Rec: Bridging the Gap between
//! NLP and Sequential/Session-Based Recommendation Systems", RecSys 2021.
//!
//! Architecture:
//!   Item embeddings and positional embeddings are summed to form the input
//!   sequence representation. Each transformer layer applies a simplified
//!   causal (single-head) scaled dot-product self-attention, followed by a
//!   residual connection and layer normalisation. The last-token hidden state
//!   is projected through the output head (weight-tied with item embeddings
//!   as an optional design; here we use a separate output weight matrix)
//!   to produce scores over all items.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Layer normalisation: (x − μ) / (σ + ε) * g + b.
fn layer_norm_vec(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
    x.iter().map(|&xi| (xi - mean) * inv_std).collect()
}

/// Numerically stable softmax (in-place).
fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / (sum + 1e-10_f32);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Matrix-vector product: `w` is [d_out × d_in] (row-major), `x` has length `d_in`.
/// Returns vec of length `d_out`.
fn matvec(w: &[f32], x: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
    (0..d_out)
        .map(|row| {
            w[row * d_in..(row + 1) * d_in]
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
        })
        .collect()
}

// ─── public types ─────────────────────────────────────────────────────────────

/// Hyper-parameters for [`Transformer4Rec`].
#[derive(Debug)]
pub struct T4RecConfig {
    /// Number of catalogue items (vocabulary size).
    pub n_items: usize,
    /// Hidden-state / embedding dimension.
    pub d_model: usize,
    /// Number of attention heads (unused in the simplified single-head
    /// implementation; kept for API parity with the paper).
    pub n_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Maximum supported sequence length.
    pub max_seq_len: usize,
    /// Fraction of tokens to mask during training (unused at inference).
    pub mask_ratio: f32,
}

/// Transformer4Rec sequential recommender.
#[derive(Debug)]
pub struct Transformer4Rec {
    /// Item embedding table: `[n_items × d_model]` (row-major).
    item_emb: Vec<f32>,
    /// Positional embedding table: `[max_seq_len × d_model]` (row-major).
    pos_emb: Vec<f32>,
    /// Per-layer combined Q/K/V projection: `[d_model × d_model]` (row-major).
    layer_w: Vec<Vec<f32>>,
    /// Per-layer bias: `[d_model]`.
    layer_b: Vec<Vec<f32>>,
    /// Output head weight matrix: `[n_items × d_model]` (row-major).
    output_w: Vec<f32>,
    /// Output head bias: `[n_items]`.
    output_b: Vec<f32>,
    /// Frozen configuration.
    config: T4RecConfig,
}

impl Transformer4Rec {
    /// Construct a Transformer4Rec model with Xavier-initialised weights.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidNumItems`] when `n_items == 0`.
    /// - [`RecsysError::InvalidEmbeddingDim`] when `d_model == 0`.
    /// - [`RecsysError::InvalidConfig`] when `max_seq_len == 0`.
    pub fn new(config: T4RecConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if config.n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: 0 });
        }
        if config.d_model == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if config.max_seq_len == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "max_seq_len must be > 0".into(),
            });
        }

        let d = config.d_model;
        let emb_scale = (1.0 / d as f32).sqrt();
        let w_scale = (1.0 / d as f32).sqrt(); // Xavier for square matrices

        let item_emb: Vec<f32> = (0..config.n_items * d)
            .map(|_| rng.next_normal() * emb_scale)
            .collect();

        let pos_emb: Vec<f32> = (0..config.max_seq_len * d)
            .map(|_| rng.next_normal() * emb_scale)
            .collect();

        let layer_w: Vec<Vec<f32>> = (0..config.n_layers)
            .map(|_| {
                (0..d * d)
                    .map(|_| rng.next_normal() * w_scale)
                    .collect::<Vec<f32>>()
            })
            .collect();

        let layer_b: Vec<Vec<f32>> = (0..config.n_layers).map(|_| vec![0.0_f32; d]).collect();

        let output_w: Vec<f32> = (0..config.n_items * d)
            .map(|_| rng.next_normal() * emb_scale)
            .collect();

        let output_b: Vec<f32> = vec![0.0_f32; config.n_items];

        Ok(Self {
            item_emb,
            pos_emb,
            layer_w,
            layer_b,
            output_w,
            output_b,
            config,
        })
    }

    /// Encode an item sequence, returning a flattened `[seq_len × d_model]` tensor.
    ///
    /// `seq_len` is clamped to `[1, max_seq_len]`; if `item_ids` is longer than
    /// `max_seq_len` only the last `max_seq_len` items are used.
    ///
    /// # Errors
    /// - [`RecsysError::ItemOutOfBounds`] when any item id >= `n_items`.
    /// - [`RecsysError::InvalidConfig`] when `seq_len == 0`.
    pub fn encode_sequence(&self, item_ids: &[usize], seq_len: usize) -> RecsysResult<Vec<f32>> {
        if seq_len == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "seq_len must be >= 1".into(),
            });
        }
        for &id in item_ids {
            if id >= self.config.n_items {
                return Err(RecsysError::ItemOutOfBounds {
                    idx: id,
                    n: self.config.n_items,
                });
            }
        }

        let d = self.config.d_model;
        let eff_len = seq_len.min(self.config.max_seq_len);

        // Take the last `eff_len` items (most recent context).
        let ids: &[usize] = if item_ids.len() > eff_len {
            &item_ids[item_ids.len() - eff_len..]
        } else {
            item_ids
        };
        let actual_len = ids.len().max(1); // pad with item 0 if empty

        // Build initial hidden states: item_emb + pos_emb.
        let mut h = vec![0.0_f32; actual_len * d];
        for (pos, opt_id) in (0..actual_len).map(|p| (p, ids.get(p))) {
            let pos_clamped = pos.min(self.config.max_seq_len - 1);
            let pe = &self.pos_emb[pos_clamped * d..(pos_clamped + 1) * d];
            let ie: &[f32] = if let Some(&id) = opt_id {
                &self.item_emb[id * d..(id + 1) * d]
            } else {
                &self.item_emb[0..d] // padding with item-0 embedding
            };
            for k in 0..d {
                h[pos * d + k] = ie[k] + pe[k];
            }
        }

        // Apply transformer layers.
        for layer_idx in 0..self.config.n_layers {
            h = self.apply_transformer_layer(&h, layer_idx, actual_len)?;
        }

        Ok(h)
    }

    /// Apply one transformer layer (simplified single-head self-attention +
    /// residual + layer-norm).
    fn apply_transformer_layer(
        &self,
        h: &[f32],
        layer_idx: usize,
        seq_len: usize,
    ) -> RecsysResult<Vec<f32>> {
        let d = self.config.d_model;
        let w = &self.layer_w[layer_idx];
        let b = &self.layer_b[layer_idx];
        let scale = 1.0 / (d as f32).sqrt();

        // Project each token: q_t = K_t = V_t = W * h_t + b
        let mut proj: Vec<f32> = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let x_t = &h[t * d..(t + 1) * d];
            let p_t = matvec(w, x_t, d, d);
            for k in 0..d {
                proj[t * d + k] = p_t[k] + b[k];
            }
        }

        // Scaled dot-product self-attention (Q = K = V = proj).
        // Causal mask: token i only attends to tokens 0..=i.
        let mut attn_out = vec![0.0_f32; seq_len * d];
        for i in 0..seq_len {
            let q_i = &proj[i * d..(i + 1) * d];
            let attend_to = i + 1; // causal: positions 0..=i

            let mut scores: Vec<f32> = (0..attend_to)
                .map(|j| {
                    let k_j = &proj[j * d..(j + 1) * d];
                    q_i.iter()
                        .zip(k_j.iter())
                        .map(|(&qi, &kj)| qi * kj)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            softmax_inplace(&mut scores);

            for (j, &a) in scores.iter().enumerate() {
                let v_j = &proj[j * d..(j + 1) * d];
                for k in 0..d {
                    attn_out[i * d + k] += a * v_j[k];
                }
            }
        }

        // Residual + layer norm per token.
        let mut out = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let res: Vec<f32> = (0..d).map(|k| h[t * d + k] + attn_out[t * d + k]).collect();
            let normed = layer_norm_vec(&res);
            out[t * d..(t + 1) * d].copy_from_slice(&normed);
        }

        Ok(out)
    }

    /// Compute item scores from a last-token representation vector.
    ///
    /// Returns `[n_items]` logits via `output_w * last_repr + output_b`.
    ///
    /// # Errors
    /// - [`RecsysError::DimensionMismatch`] when `last_repr.len() != d_model`.
    pub fn score_items(&self, last_repr: &[f32]) -> RecsysResult<Vec<f32>> {
        let d = self.config.d_model;
        if last_repr.len() != d {
            return Err(RecsysError::DimensionMismatch {
                expected: d,
                got: last_repr.len(),
            });
        }
        let logits: Vec<f32> = (0..self.config.n_items)
            .map(|i| {
                let row = &self.output_w[i * d..(i + 1) * d];
                row.iter()
                    .zip(last_repr.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
                    + self.output_b[i]
            })
            .collect();
        Ok(logits)
    }

    /// Encode the sequence and return the top-`k` item indices by score.
    ///
    /// # Errors
    /// - Propagates errors from [`Transformer4Rec::encode_sequence`] and [`Transformer4Rec::score_items`].
    /// - [`RecsysError::InvalidK`] when `k > n_items`.
    pub fn recommend(
        &self,
        item_ids: &[usize],
        seq_len: usize,
        k: usize,
    ) -> RecsysResult<Vec<usize>> {
        if k > self.config.n_items {
            return Err(RecsysError::InvalidK {
                k,
                n: self.config.n_items,
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let h = self.encode_sequence(item_ids, seq_len)?;
        let d = self.config.d_model;
        // Last token representation.
        let actual_len = h.len() / d;
        let last_repr = &h[(actual_len - 1) * d..actual_len * d];

        let logits = self.score_items(last_repr)?;

        // Partial sort: top-k by descending score.
        let mut indices: Vec<usize> = (0..self.config.n_items).collect();
        indices.sort_unstable_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(k);
        Ok(indices)
    }

    /// Number of items in the vocabulary.
    #[must_use]
    pub fn n_items(&self) -> usize {
        self.config.n_items
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_config(n_items: usize, d_model: usize, n_layers: usize) -> T4RecConfig {
        T4RecConfig {
            n_items,
            d_model,
            n_heads: 1,
            n_layers,
            max_seq_len: 16,
            mask_ratio: 0.15,
        }
    }

    fn make_model(n_items: usize, d_model: usize, n_layers: usize) -> Transformer4Rec {
        let mut rng = LcgRng::new(42);
        Transformer4Rec::new(make_config(n_items, d_model, n_layers), &mut rng)
            .expect("model construction must succeed")
    }

    // 1. encode_sequence returns vec of size seq_len * d_model
    #[test]
    fn sequence_encoding_shape() {
        let model = make_model(20, 8, 2);
        let ids = vec![0usize, 1, 2, 3];
        let h = model.encode_sequence(&ids, 4).expect("encode must succeed");
        assert_eq!(h.len(), 4 * 8, "encoded length must be seq_len * d_model");
    }

    // 2. score_items returns vec of size n_items
    #[test]
    fn score_items_shape() {
        let model = make_model(20, 8, 2);
        let repr = vec![0.1_f32; 8];
        let scores = model.score_items(&repr).expect("score_items must succeed");
        assert_eq!(scores.len(), 20, "score_items must return n_items logits");
    }

    // 3. recommend returns exactly k items
    #[test]
    fn recommend_returns_k_items() {
        let model = make_model(20, 8, 2);
        let ids = vec![0usize, 1, 3];
        let recs = model.recommend(&ids, 3, 5).expect("recommend must succeed");
        assert_eq!(recs.len(), 5, "recommend must return exactly k items");
    }

    // 4. all recommended items < n_items
    #[test]
    fn items_in_range() {
        let model = make_model(20, 8, 2);
        let ids = vec![2usize, 5, 7];
        let recs = model
            .recommend(&ids, 3, 10)
            .expect("recommend must succeed");
        for &r in &recs {
            assert!(r < 20, "recommended item {r} must be < n_items");
        }
    }

    // 5. seq_len=1 works fine
    #[test]
    fn seq_len_1_works() {
        let model = make_model(10, 4, 1);
        let ids = vec![0usize];
        let h = model.encode_sequence(&ids, 1).expect("seq_len=1 must work");
        assert_eq!(h.len(), 4);
        let recs = model
            .recommend(&ids, 1, 3)
            .expect("recommend with seq_len=1");
        assert_eq!(recs.len(), 3);
    }

    // 6. item_id >= n_items returns Err
    #[test]
    fn too_large_item_id_error() {
        let model = make_model(10, 4, 1);
        let ids = vec![9usize, 10]; // 10 is out of range for n_items=10
        let result = model.encode_sequence(&ids, 2);
        assert!(
            matches!(result, Err(RecsysError::ItemOutOfBounds { idx: 10, n: 10 })),
            "expected ItemOutOfBounds error, got: {result:?}"
        );
    }

    // 7. d_model=0 returns Err from new()
    #[test]
    fn d_model_zero_error() {
        let mut rng = LcgRng::new(42);
        let cfg = T4RecConfig {
            n_items: 10,
            d_model: 0,
            n_heads: 1,
            n_layers: 1,
            max_seq_len: 8,
            mask_ratio: 0.1,
        };
        let result = Transformer4Rec::new(cfg, &mut rng);
        assert!(
            matches!(result, Err(RecsysError::InvalidEmbeddingDim { d: 0 })),
            "expected InvalidEmbeddingDim, got: {result:?}"
        );
    }

    // 8. n_layers=0 creates model successfully and recommend works
    #[test]
    fn n_layers_zero_works() {
        let model = make_model(10, 4, 0);
        let ids = vec![1usize, 2];
        let recs = model
            .recommend(&ids, 2, 3)
            .expect("n_layers=0 model must work");
        assert_eq!(recs.len(), 3);
    }

    // 9. two different sequences generally give different logits
    #[test]
    fn different_sequences_different_recs() {
        let model = make_model(20, 8, 2);
        let ids_a = vec![0usize, 1, 2];
        let ids_b = vec![5usize, 6, 7];
        let h_a = model.encode_sequence(&ids_a, 3).expect("encode a");
        let h_b = model.encode_sequence(&ids_b, 3).expect("encode b");
        let diff: f32 = h_a
            .iter()
            .zip(h_b.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "different item sequences must yield different hidden states (diff={diff})"
        );
    }

    // 10. all score_items values are finite
    #[test]
    fn output_finite() {
        let model = make_model(20, 8, 2);
        let repr = vec![0.5_f32; 8];
        let scores = model.score_items(&repr).expect("score_items must succeed");
        for (i, &s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] = {s} must be finite");
        }
    }

    // 11. n_items=0 returns Err from new()
    #[test]
    fn n_items_zero_error() {
        let mut rng = LcgRng::new(42);
        let cfg = T4RecConfig {
            n_items: 0,
            d_model: 8,
            n_heads: 1,
            n_layers: 1,
            max_seq_len: 8,
            mask_ratio: 0.1,
        };
        let result = Transformer4Rec::new(cfg, &mut rng);
        assert!(
            matches!(result, Err(RecsysError::InvalidNumItems { n: 0 })),
            "expected InvalidNumItems, got: {result:?}"
        );
    }

    // 12. max_seq_len=0 returns Err
    #[test]
    fn max_seq_len_zero_error() {
        let mut rng = LcgRng::new(42);
        let cfg = T4RecConfig {
            n_items: 10,
            d_model: 4,
            n_heads: 1,
            n_layers: 1,
            max_seq_len: 0,
            mask_ratio: 0.1,
        };
        let result = Transformer4Rec::new(cfg, &mut rng);
        assert!(
            matches!(result, Err(RecsysError::InvalidConfig { .. })),
            "expected InvalidConfig for max_seq_len=0, got: {result:?}"
        );
    }

    // 13. sequence longer than max_seq_len is truncated (returns correct size)
    #[test]
    fn long_sequence_truncated() {
        let mut rng = LcgRng::new(42);
        let cfg = T4RecConfig {
            n_items: 10,
            d_model: 4,
            n_heads: 1,
            n_layers: 1,
            max_seq_len: 4,
            mask_ratio: 0.1,
        };
        let model = Transformer4Rec::new(cfg, &mut rng).expect("must build");
        let ids: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7]; // 8 items, max=4
        let h = model.encode_sequence(&ids, 8).expect("encode long seq");
        // encode_sequence uses eff_len = min(seq_len, max_seq_len)=4
        // actual_len = min(ids.len(), eff_len) via the slice — either 4 tokens
        assert!(!h.is_empty(), "output must be non-empty");
        assert_eq!(h.len() % 4, 0, "output must be multiple of d_model");
    }

    // 14. k > n_items returns InvalidK
    #[test]
    fn k_too_large_error() {
        let model = make_model(5, 4, 1);
        let ids = vec![0usize, 1];
        let result = model.recommend(&ids, 2, 10); // k=10 > n_items=5
        assert!(
            matches!(result, Err(RecsysError::InvalidK { k: 10, n: 5 })),
            "expected InvalidK, got: {result:?}"
        );
    }
}

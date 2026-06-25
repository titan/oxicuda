//! FT-Transformer variant with rotary position embeddings (RoPE) and learnable
//! relative attention bias.
//!
//! The canonical FT-Transformer treats feature tokens as an unordered set, so it
//! omits positional information entirely.  This variant injects two orthogonal
//! position-aware mechanisms over the feature-token axis:
//!
//! # Rotary Position Embedding (Su et al. 2021, RoFormer)
//!
//! Each query / key head vector is split into `head_dim / 2` 2-D pairs.  Pair
//! `p` at token position `t` is rotated by angle `t · θ_p`, where
//! `θ_p = base^{−2p / head_dim}`.  Rotation acts as a relative encoding: the
//! attention dot product between positions `t_i` and `t_j` depends only on the
//! difference `t_i − t_j`.
//!
//! ```text
//! [q'_{2p}, q'_{2p+1}] = R(t·θ_p) · [q_{2p}, q_{2p+1}]
//! R(φ) = [[cos φ, −sin φ], [sin φ, cos φ]]
//! ```
//!
//! # Learnable relative attention bias (T5-style, Raffel et al. 2020)
//!
//! A per-head table `bias[h][Δ]` indexed by the signed token offset
//! `Δ = j − i ∈ [−(S−1), S−1]` is added to the pre-softmax logits.  This lets
//! the model learn data-dependent feature-pair preferences that survive the
//! permutation symmetry that pure RoPE preserves.
//!
//! Both are applied inside a Pre-LayerNorm transformer block stack with a
//! prepended CLS token; the CLS row of the final layer drives a linear head.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;
use crate::transformer::autoint::layer_norm;

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for the RoPE FT-Transformer.
#[derive(Debug, Clone)]
pub struct FtRopeConfig {
    /// Number of continuous features (each becomes one token).
    pub n_cont_features: usize,
    /// Embedding dimension per token (must be even and divisible by `n_heads`).
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Hidden dimension of the position-wise FFN.
    pub ffn_hidden: usize,
    /// Number of output classes (1 for regression).
    pub n_classes: usize,
    /// RoPE frequency base (10000 in the original RoFormer).
    pub rope_base: f32,
}

impl Default for FtRopeConfig {
    fn default() -> Self {
        Self {
            n_cont_features: 8,
            embed_dim: 16,
            n_heads: 2,
            n_layers: 2,
            ffn_hidden: 32,
            n_classes: 2,
            rope_base: 10_000.0,
        }
    }
}

// ─── Per-layer weights ─────────────────────────────────────────────────────────

/// Weights for a single RoPE transformer block.
#[derive(Debug, Clone)]
struct BlockWeights {
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
    /// Per-head relative bias table, length `n_heads * (2*max_offset + 1)`.
    rel_bias: Vec<f32>,
}

impl BlockWeights {
    fn new(cfg: &FtRopeConfig, n_offsets: usize, rng: &mut LcgRng) -> Self {
        let ed = cfg.embed_dim;
        let k = (6.0_f32 / ed as f32).sqrt();
        let fill = |size: usize, bound: f32, rng: &mut LcgRng| -> Vec<f32> {
            (0..size)
                .map(|_| rng.next_f32() * 2.0 * bound - bound)
                .collect()
        };
        let n = ed * ed;
        let k_ffn = (6.0_f32 / (ed + cfg.ffn_hidden) as f32).sqrt();
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
            // Relative bias initialised to zero (T5 starts from no preference).
            rel_bias: vec![0.0; cfg.n_heads * n_offsets],
        }
    }
}

// ─── Model ────────────────────────────────────────────────────────────────────

/// FT-Transformer with rotary position embeddings + learnable relative bias.
#[derive(Debug, Clone)]
pub struct FtRopeTransformer {
    config: FtRopeConfig,
    /// Continuous tokenizer scale: `[n_cont * embed_dim]`.
    tok_w: Vec<f32>,
    /// Continuous tokenizer bias: `[n_cont * embed_dim]`.
    tok_b: Vec<f32>,
    /// CLS token embedding: `[embed_dim]`.
    cls: Vec<f32>,
    blocks: Vec<BlockWeights>,
    /// Final classifier weight: `[n_classes * embed_dim]`.
    head_w: Vec<f32>,
    /// Final classifier bias: `[n_classes]`.
    head_b: Vec<f32>,
    /// Sequence length including CLS (= n_cont + 1).
    seq_len: usize,
    /// Number of signed offsets in the relative bias table.
    n_offsets: usize,
}

impl FtRopeTransformer {
    /// Build a RoPE FT-Transformer with randomly initialised weights.
    ///
    /// # Errors
    /// Returns [`TabularError::InvalidAttentionDim`] if `embed_dim` is not even
    /// or not divisible by `n_heads`, and [`TabularError::InvalidParameter`] for
    /// other degenerate configurations.
    pub fn new(config: FtRopeConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if config.embed_dim == 0
            || !config.embed_dim.is_multiple_of(2)
            || !config.embed_dim.is_multiple_of(config.n_heads.max(1))
            || config.n_heads == 0
        {
            return Err(TabularError::InvalidAttentionDim {
                dim: config.embed_dim,
            });
        }
        if config.n_cont_features == 0 {
            return Err(TabularError::InvalidFeatureCount {
                n: config.n_cont_features,
            });
        }
        if config.n_classes == 0 {
            return Err(TabularError::InvalidParameter {
                name: "n_classes".into(),
                msg: "must be > 0".into(),
            });
        }
        if config.rope_base <= 1.0 {
            return Err(TabularError::InvalidParameter {
                name: "rope_base".into(),
                msg: "must be > 1".into(),
            });
        }
        let ed = config.embed_dim;
        let seq_len = config.n_cont_features + 1;
        let n_offsets = 2 * (seq_len - 1) + 1;
        let k = (6.0_f32 / (ed as f32 + 1.0)).sqrt();
        let fill = |size: usize, bound: f32, rng: &mut LcgRng| -> Vec<f32> {
            (0..size)
                .map(|_| rng.next_f32() * 2.0 * bound - bound)
                .collect()
        };
        let tok_w = fill(config.n_cont_features * ed, k, rng);
        let tok_b = vec![0.0_f32; config.n_cont_features * ed];
        let cls = fill(ed, k, rng);
        let blocks: Vec<BlockWeights> = (0..config.n_layers)
            .map(|_| BlockWeights::new(&config, n_offsets, rng))
            .collect();
        let k_head = (6.0_f32 / (ed + config.n_classes) as f32).sqrt();
        let head_w = fill(config.n_classes * ed, k_head, rng);
        let head_b = vec![0.0_f32; config.n_classes];
        Ok(Self {
            config,
            tok_w,
            tok_b,
            cls,
            blocks,
            head_w,
            head_b,
            seq_len,
            n_offsets,
        })
    }

    /// Sequence length including the CLS token.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Tokenize continuous features into `[seq_len × embed_dim]` with a CLS row
    /// at index 0.
    fn tokenize(&self, x_cont: &[f32]) -> TabularResult<Vec<f32>> {
        let ed = self.config.embed_dim;
        let nc = self.config.n_cont_features;
        if x_cont.len() != nc {
            return Err(TabularError::DimensionMismatch {
                expected: nc,
                got: x_cont.len(),
            });
        }
        let mut tokens = vec![0.0_f32; self.seq_len * ed];
        tokens[0..ed].copy_from_slice(&self.cls);
        for f in 0..nc {
            let xv = x_cont[f];
            for d in 0..ed {
                let idx = f * ed + d;
                tokens[(f + 1) * ed + d] = xv * self.tok_w[idx] + self.tok_b[idx];
            }
        }
        Ok(tokens)
    }

    /// Forward pass returning class logits (length `n_classes`).
    ///
    /// # Errors
    /// Returns an error if `x_cont.len() != n_cont_features`.
    pub fn forward(&self, x_cont: &[f32]) -> TabularResult<Vec<f32>> {
        let ed = self.config.embed_dim;
        let mut tokens = self.tokenize(x_cont)?;
        for block in &self.blocks {
            tokens = self.block_forward(block, &tokens)?;
        }
        // CLS head.
        let cls_row = &tokens[0..ed];
        let mut logits = vec![0.0_f32; self.config.n_classes];
        for (c, lc) in logits.iter_mut().enumerate() {
            let w_row = &self.head_w[c * ed..(c + 1) * ed];
            let acc: f32 = w_row.iter().zip(cls_row.iter()).map(|(&w, &x)| w * x).sum();
            *lc = self.head_b[c] + acc;
        }
        Ok(logits)
    }

    /// One Pre-LN transformer block: RoPE+bias MHSA, residual, FFN, residual.
    fn block_forward(&self, block: &BlockWeights, tokens: &[f32]) -> TabularResult<Vec<f32>> {
        let ed = self.config.embed_dim;
        let s = self.seq_len;
        // Pre-LN.
        let mut normed = vec![0.0_f32; s * ed];
        for t in 0..s {
            let row = &tokens[t * ed..(t + 1) * ed];
            let ln = layer_norm(row, &block.ln1_g, &block.ln1_b, 1e-5);
            normed[t * ed..(t + 1) * ed].copy_from_slice(&ln);
        }
        let attn = self.rope_attention(block, &normed)?;
        // Residual.
        let mut h = vec![0.0_f32; s * ed];
        for i in 0..s * ed {
            h[i] = tokens[i] + attn[i];
        }
        // FFN with Pre-LN.
        let mut out = vec![0.0_f32; s * ed];
        for t in 0..s {
            let row = &h[t * ed..(t + 1) * ed];
            let ln = layer_norm(row, &block.ln2_g, &block.ln2_b, 1e-5);
            // FFN: ReLU(W1·ln + b1) then W2·· + b2
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

    /// Multi-head self-attention with rotary Q/K and additive relative bias.
    fn rope_attention(&self, block: &BlockWeights, x: &[f32]) -> TabularResult<Vec<f32>> {
        let ed = self.config.embed_dim;
        let s = self.seq_len;
        let n_heads = self.config.n_heads;
        let head_dim = ed / n_heads;
        // Project to Q, K, V (row-major [s × ed]).
        let q = self.project(&block.wq, x);
        let k = self.project(&block.wk, x);
        let v = self.project(&block.wv, x);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mid = self.n_offsets / 2; // offset 0 lives at the midpoint
        let mut concat = vec![0.0_f32; s * ed];
        for h in 0..n_heads {
            let h0 = h * head_dim;
            // Apply RoPE per token to this head's Q and K slices.
            let mut q_h = vec![0.0_f32; s * head_dim];
            let mut k_h = vec![0.0_f32; s * head_dim];
            for t in 0..s {
                let qs = &q[t * ed + h0..t * ed + h0 + head_dim];
                let ks = &k[t * ed + h0..t * ed + h0 + head_dim];
                apply_rope(
                    qs,
                    t,
                    self.config.rope_base,
                    &mut q_h[t * head_dim..(t + 1) * head_dim],
                );
                apply_rope(
                    ks,
                    t,
                    self.config.rope_base,
                    &mut k_h[t * head_dim..(t + 1) * head_dim],
                );
            }
            // Scores + relative bias + softmax.
            let bias_row = &block.rel_bias[h * self.n_offsets..(h + 1) * self.n_offsets];
            for i in 0..s {
                let mut scores = vec![0.0_f32; s];
                for (j, sj) in scores.iter_mut().enumerate() {
                    let mut d = 0.0_f32;
                    for c in 0..head_dim {
                        d += q_h[i * head_dim + c] * k_h[j * head_dim + c];
                    }
                    // Δ = j − i, mapped into [0, n_offsets).
                    let delta = j as isize - i as isize;
                    let bidx = (mid as isize + delta) as usize;
                    *sj = d * scale + bias_row[bidx];
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|&z| (z - max).exp()).collect();
                let denom: f32 = exps.iter().sum::<f32>().max(1e-30);
                for c in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for j in 0..s {
                        acc += (exps[j] / denom) * v[j * ed + h0 + c];
                    }
                    concat[i * ed + h0 + c] = acc;
                }
            }
        }
        // Output projection.
        Ok(self.project(&block.wo, &concat))
    }

    /// Project `[s × ed]` rows by a row-major `[ed × ed]` weight: `y = x · Wᵀ`.
    fn project(&self, w: &[f32], x: &[f32]) -> Vec<f32> {
        let ed = self.config.embed_dim;
        let s = self.seq_len;
        let mut out = vec![0.0_f32; s * ed];
        for t in 0..s {
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

/// Apply rotary position embedding to a single head vector `v` at position `pos`.
///
/// Consecutive (even, odd) channel pairs are rotated by `pos · base^{−2p/dim}`.
fn apply_rope(v: &[f32], pos: usize, base: f32, out: &mut [f32]) {
    let dim = v.len();
    let half = dim / 2;
    for p in 0..half {
        let inv_freq = base.powf(-2.0 * p as f32 / dim as f32);
        let angle = pos as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let a = v[2 * p];
        let b = v[2 * p + 1];
        out[2 * p] = a * cos - b * sin;
        out[2 * p + 1] = a * sin + b * cos;
    }
    // If dim is odd (should not happen given the even check) copy the tail.
    if dim % 2 == 1 {
        out[dim - 1] = v[dim - 1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_odd_embed_dim() {
        let mut rng = LcgRng::new(1);
        let cfg = FtRopeConfig {
            embed_dim: 15,
            ..Default::default()
        };
        assert!(FtRopeTransformer::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn rejects_non_divisible_heads() {
        let mut rng = LcgRng::new(1);
        let cfg = FtRopeConfig {
            embed_dim: 16,
            n_heads: 3,
            ..Default::default()
        };
        assert!(FtRopeTransformer::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn forward_shape_and_finite() {
        let mut rng = LcgRng::new(2);
        let cfg = FtRopeConfig {
            n_cont_features: 5,
            embed_dim: 16,
            n_heads: 4,
            n_layers: 3,
            ffn_hidden: 32,
            n_classes: 4,
            rope_base: 10_000.0,
        };
        let model = FtRopeTransformer::new(cfg, &mut rng).expect("new");
        let logits = model
            .forward(&[0.1, -0.5, 0.3, 0.8, -0.2])
            .expect("forward");
        assert_eq!(logits.len(), 4);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rope_preserves_vector_norm() {
        // A rotation is orthogonal: ‖R·v‖ = ‖v‖.
        let v = [0.3_f32, -0.7, 1.2, 0.5];
        let mut out = vec![0.0_f32; 4];
        apply_rope(&v, 3, 10_000.0, &mut out);
        let nv: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let no: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((nv - no).abs() < 1e-5, "‖v‖={nv} ‖Rv‖={no}");
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let v = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f32; 4];
        apply_rope(&v, 0, 10_000.0, &mut out);
        for (a, b) in v.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_relative_dot_product_invariance() {
        // RoPE dot product depends only on relative offset:
        // ⟨R(t_i)q, R(t_j)k⟩ == ⟨R(t_i+s)q, R(t_j+s)k⟩.
        let q = [0.4_f32, 0.1, -0.3, 0.9, 0.2, -0.6];
        let k = [0.7_f32, -0.2, 0.5, 0.1, -0.4, 0.8];
        let base = 10_000.0;
        let dot_at = |ti: usize, tj: usize| -> f32 {
            let mut rq = vec![0.0_f32; q.len()];
            let mut rk = vec![0.0_f32; k.len()];
            apply_rope(&q, ti, base, &mut rq);
            apply_rope(&k, tj, base, &mut rk);
            rq.iter().zip(rk.iter()).map(|(&a, &b)| a * b).sum()
        };
        let d0 = dot_at(1, 4);
        let d1 = dot_at(3, 6); // same offset (j − i = 3), shifted by +2
        assert!((d0 - d1).abs() < 1e-4, "{d0} vs {d1}");
    }

    #[test]
    fn determinism_same_seed() {
        let cfg = FtRopeConfig::default();
        let mut r1 = LcgRng::new(123);
        let mut r2 = LcgRng::new(123);
        let m1 = FtRopeTransformer::new(cfg.clone(), &mut r1).expect("new");
        let m2 = FtRopeTransformer::new(cfg.clone(), &mut r2).expect("new");
        let x = vec![0.2_f32; cfg.n_cont_features];
        assert_eq!(m1.forward(&x).expect("f"), m2.forward(&x).expect("f"));
    }

    #[test]
    fn wrong_input_length_errors() {
        let mut rng = LcgRng::new(4);
        let cfg = FtRopeConfig::default();
        let model = FtRopeTransformer::new(cfg.clone(), &mut rng).expect("new");
        let bad = vec![0.0_f32; cfg.n_cont_features + 2];
        assert!(model.forward(&bad).is_err());
    }
}

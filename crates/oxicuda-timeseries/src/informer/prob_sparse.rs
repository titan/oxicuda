//! Informer ProbSparse self-attention (Zhou et al. 2021 AAAI Best Paper).
//!
//! Reduces standard O(L²) attention to O(L log L) via sparse query selection:
//! only the top-u queries by KL-divergence dominance score attend to all keys,
//! while the remaining queries are filled with mean(V).
//!
//! Reference: "Informer: Beyond Efficient Transformer for Long Sequence
//! Time-Series Forecasting", Zhou et al., AAAI 2021.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Config & Weights ────────────────────────────────────────────────────────

/// Configuration for a single ProbSparse attention block.
#[derive(Debug, Clone)]
pub struct ProbSparseConfig {
    /// Total embedding dimension (must be divisible by `n_heads`).
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Sampling factor c: u = min(factor * ceil(ln(L_K)), L_K).
    pub factor: usize,
    /// Dropout rate (0.0 = no dropout; applied in training for reference).
    pub dropout_rate: f32,
    /// Input encoder sequence length L_Q.
    pub seq_len: usize,
    /// Start token length for decoder cross-attention (informational).
    pub label_len: usize,
    /// Prediction horizon H.
    pub pred_len: usize,
}

/// Learnable weights for one ProbSparse attention block.
#[derive(Debug, Clone)]
pub struct ProbSparseWeights {
    /// Query projection `[embed_dim × embed_dim]`.
    pub w_q: Vec<f32>,
    /// Key projection `[embed_dim × embed_dim]`.
    pub w_k: Vec<f32>,
    /// Value projection `[embed_dim × embed_dim]`.
    pub w_v: Vec<f32>,
    /// Output projection `[embed_dim × embed_dim]`.
    pub w_o: Vec<f32>,
    /// Query bias `[embed_dim]`.
    pub b_q: Vec<f32>,
    /// Key bias `[embed_dim]`.
    pub b_k: Vec<f32>,
    /// Value bias `[embed_dim]`.
    pub b_v: Vec<f32>,
    /// Output bias `[embed_dim]`.
    pub b_o: Vec<f32>,
}

impl ProbSparseWeights {
    /// Kaiming-uniform initialisation (fan-in = embed_dim).
    fn new(embed_dim: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0_f32 / embed_dim as f32).sqrt();
        let mut mat = |rows: usize, cols: usize| -> Vec<f32> {
            let mut v = vec![0.0_f32; rows * cols];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= scale;
            }
            v
        };
        let d = embed_dim;
        Self {
            w_q: mat(d, d),
            w_k: mat(d, d),
            w_v: mat(d, d),
            w_o: mat(d, d),
            b_q: vec![0.0_f32; d],
            b_k: vec![0.0_f32; d],
            b_v: vec![0.0_f32; d],
            b_o: vec![0.0_f32; d],
        }
    }
}

// ─── InformerBlock ───────────────────────────────────────────────────────────

/// Result produced by a single ProbSparse forward pass.
#[derive(Debug, Clone)]
pub struct InformerResult {
    /// Output tensor `[seq_len_q × embed_dim]`.
    pub output: Vec<f32>,
    /// Top-u selected query indices per head (concatenated).
    pub selected_query_idx: Vec<usize>,
}

/// Single-layer ProbSparse attention block.
#[derive(Debug, Clone)]
pub struct InformerBlock {
    /// Block configuration.
    pub cfg: ProbSparseConfig,
    /// Learnable parameters.
    pub weights: ProbSparseWeights,
}

impl InformerBlock {
    /// Construct an `InformerBlock` with Kaiming-uniform weight initialisation.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidEmbedDim`] when `embed_dim == 0`.
    /// - [`TsError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`TsError::HeadDimMismatch`] when `embed_dim % n_heads != 0`.
    /// - [`TsError::InvalidSequenceLength`] when `seq_len == 0`.
    /// - [`TsError::InvalidHorizon`] when `factor == 0`.
    pub fn new(cfg: ProbSparseConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if cfg.embed_dim == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if cfg.n_heads == 0 {
            return Err(TsError::InvalidNumHeads(0));
        }
        if cfg.embed_dim % cfg.n_heads != 0 {
            return Err(TsError::HeadDimMismatch {
                embed_dim: cfg.embed_dim,
                n_heads: cfg.n_heads,
            });
        }
        if cfg.seq_len == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if cfg.factor == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        let weights = ProbSparseWeights::new(cfg.embed_dim, rng);
        Ok(Self { cfg, weights })
    }

    /// ProbSparse attention forward pass.
    ///
    /// # Arguments
    ///
    /// * `q` — `[seq_len_q × embed_dim]` query input.
    /// * `seq_len_q` — number of query tokens.
    /// * `k`, `v` — `[seq_len_kv × embed_dim]` key/value inputs.
    /// * `seq_len_kv` — number of key/value tokens.
    ///
    /// Returns `seq_len_q × embed_dim` output concatenating sparse attentions
    /// across all heads, projected through `W_o`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] on size mismatches.
    /// - [`TsError::InvalidSequenceLength`] when any sequence length is zero.
    pub fn forward(
        &self,
        q: &[f32],
        seq_len_q: usize,
        k: &[f32],
        v: &[f32],
        seq_len_kv: usize,
        rng: &mut LcgRng,
    ) -> TsResult<InformerResult> {
        let d = self.cfg.embed_dim;
        let n_h = self.cfg.n_heads;
        let head_dim = d / n_h;

        if seq_len_q == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if seq_len_kv == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if q.len() != seq_len_q * d {
            return Err(TsError::DimensionMismatch {
                expected: seq_len_q * d,
                got: q.len(),
            });
        }
        if k.len() != seq_len_kv * d {
            return Err(TsError::DimensionMismatch {
                expected: seq_len_kv * d,
                got: k.len(),
            });
        }
        if v.len() != seq_len_kv * d {
            return Err(TsError::DimensionMismatch {
                expected: seq_len_kv * d,
                got: v.len(),
            });
        }

        // Project Q, K, V: [seq × embed_dim] → [seq × embed_dim].
        let q_proj = linear_proj(q, seq_len_q, d, &self.weights.w_q, &self.weights.b_q);
        let k_proj = linear_proj(k, seq_len_kv, d, &self.weights.w_k, &self.weights.b_k);
        let v_proj = linear_proj(v, seq_len_kv, d, &self.weights.w_v, &self.weights.b_v);

        // Multi-head ProbSparse attention.
        let mut concat_out = vec![0.0_f32; seq_len_q * d];
        let mut all_selected_idx: Vec<usize> = Vec::new();

        for h in 0..n_h {
            let h_off = h * head_dim;

            // Extract per-head slices: [seq × head_dim].
            let q_head = extract_head(&q_proj, seq_len_q, d, h_off, head_dim);
            let k_head = extract_head(&k_proj, seq_len_kv, d, h_off, head_dim);
            let v_head = extract_head(&v_proj, seq_len_kv, d, h_off, head_dim);

            let (head_out, sel_idx) = Self::prob_sparse_attention(
                &q_head,
                &k_head,
                &v_head,
                seq_len_q,
                seq_len_kv,
                head_dim,
                self.cfg.factor,
                rng,
            );

            // Write head output into concat buffer at head offset.
            for qi in 0..seq_len_q {
                for hd in 0..head_dim {
                    concat_out[qi * d + h_off + hd] = head_out[qi * head_dim + hd];
                }
            }
            all_selected_idx.extend_from_slice(&sel_idx);
        }

        // Output projection: [seq_len_q × embed_dim].
        let output = linear_proj(
            &concat_out,
            seq_len_q,
            d,
            &self.weights.w_o,
            &self.weights.b_o,
        );

        Ok(InformerResult {
            output,
            selected_query_idx: all_selected_idx,
        })
    }

    /// ProbSparse attention for a single head.
    ///
    /// Computes dominance scores for each query using a random sample of
    /// `u = min(factor * ceil(ln(L_K)), L_K)` key vectors, selects the top-u
    /// queries, applies full scaled dot-product attention for those queries,
    /// and fills the remaining queries with `mean(V)`.
    ///
    /// # Returns
    ///
    /// `(output [L_Q × head_dim], selected_query_indices [u])`
    pub fn prob_sparse_attention(
        q_head: &[f32],
        k_head: &[f32],
        v_head: &[f32],
        l_q: usize,
        l_k: usize,
        head_dim: usize,
        factor: usize,
        rng: &mut LcgRng,
    ) -> (Vec<f32>, Vec<usize>) {
        // Number of keys to sample for dominance scoring.
        let ln_lk = if l_k > 1 {
            (l_k as f32).ln().ceil() as usize
        } else {
            1
        };
        let u_sample = (factor * ln_lk).min(l_k);
        let u_select = u_sample.min(l_q);

        // Sample u_sample key indices without replacement (Fisher-Yates on index vec).
        let mut k_indices: Vec<usize> = (0..l_k).collect();
        rng.shuffle(&mut k_indices);
        let sampled_k_idx = &k_indices[..u_sample];

        // Build sampled key matrix [u_sample × head_dim].
        let mut k_sample = vec![0.0_f32; u_sample * head_dim];
        for (si, &ki) in sampled_k_idx.iter().enumerate() {
            for hd in 0..head_dim {
                k_sample[si * head_dim + hd] = k_head[ki * head_dim + hd];
            }
        }

        // Compute dominance scores for every query.
        let mut scores: Vec<(usize, f32)> = (0..l_q)
            .map(|qi| {
                let q_row = &q_head[qi * head_dim..(qi + 1) * head_dim];
                let m = Self::query_dominance(q_row, &k_sample, u_sample, head_dim);
                (qi, m)
            })
            .collect();

        // Select top-u_select queries (descending dominance).
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected_idx: Vec<usize> = scores[..u_select].iter().map(|&(i, _)| i).collect();

        // Compute mean(V) across all key positions → [head_dim].
        let mut mean_v = vec![0.0_f32; head_dim];
        for ki in 0..l_k {
            for hd in 0..head_dim {
                mean_v[hd] += v_head[ki * head_dim + hd];
            }
        }
        let inv_lk = (l_k as f32).recip();
        for v in &mut mean_v {
            *v *= inv_lk;
        }

        // Build output: unselected get mean(V), selected get sparse attention.
        let mut output = vec![0.0_f32; l_q * head_dim];

        // Fill all rows with mean(V) first.
        for qi in 0..l_q {
            for hd in 0..head_dim {
                output[qi * head_dim + hd] = mean_v[hd];
            }
        }

        // Compute full attention for selected queries.
        let mut q_selected = vec![0.0_f32; u_select * head_dim];
        for (si, &qi) in selected_idx.iter().enumerate() {
            for hd in 0..head_dim {
                q_selected[si * head_dim + hd] = q_head[qi * head_dim + hd];
            }
        }

        let sparse_out =
            Self::sparse_attention(&q_selected, k_head, v_head, u_select, l_k, head_dim);

        // Write sparse attention results back to selected query positions.
        for (si, &qi) in selected_idx.iter().enumerate() {
            for hd in 0..head_dim {
                output[qi * head_dim + hd] = sparse_out[si * head_dim + hd];
            }
        }

        (output, selected_idx)
    }

    /// Query dominance score M(q_i, K̃) = max_j(q_i · k̃_j/√d) - mean_j(q_i · k̃_j/√d).
    ///
    /// Approximates the KL divergence between the query's attention distribution
    /// and a uniform distribution using a sampled key matrix K̃.
    pub fn query_dominance(q: &[f32], k_sample: &[f32], l_k_sample: usize, head_dim: usize) -> f32 {
        if l_k_sample == 0 || head_dim == 0 {
            return 0.0;
        }
        let scale = (head_dim as f32).sqrt().recip();
        let mut max_dot = f32::NEG_INFINITY;
        let mut sum_dot = 0.0_f32;

        for ki in 0..l_k_sample {
            let mut dot = 0.0_f32;
            for hd in 0..head_dim {
                dot += q[hd] * k_sample[ki * head_dim + hd];
            }
            let scaled = dot * scale;
            if scaled > max_dot {
                max_dot = scaled;
            }
            sum_dot += scaled;
        }
        let mean_dot = sum_dot / l_k_sample as f32;
        max_dot - mean_dot
    }

    /// Standard scaled dot-product attention on u selected queries.
    ///
    /// # Arguments
    ///
    /// * `q_selected` — `[u × head_dim]` selected query rows.
    /// * `k` — `[l_k × head_dim]` full key matrix.
    /// * `v` — `[l_k × head_dim]` full value matrix.
    ///
    /// Returns `[u × head_dim]` attended output.
    pub fn sparse_attention(
        q_selected: &[f32],
        k: &[f32],
        v: &[f32],
        u: usize,
        l_k: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        if u == 0 || l_k == 0 || head_dim == 0 {
            return vec![0.0_f32; u * head_dim];
        }
        let scale = (head_dim as f32).sqrt().recip();
        let mut attn_scores = vec![0.0_f32; u * l_k];

        // Compute Q_sel @ K^T / sqrt(d): [u × l_k].
        for qi in 0..u {
            for ki in 0..l_k {
                let mut dot = 0.0_f32;
                for hd in 0..head_dim {
                    dot += q_selected[qi * head_dim + hd] * k[ki * head_dim + hd];
                }
                attn_scores[qi * l_k + ki] = dot * scale;
            }
        }

        // Softmax over l_k axis for each query.
        for qi in 0..u {
            softmax_row(&mut attn_scores[qi * l_k..(qi + 1) * l_k]);
        }

        // attn @ V: [u × head_dim].
        let mut out = vec![0.0_f32; u * head_dim];
        for qi in 0..u {
            for hd in 0..head_dim {
                let mut acc = 0.0_f32;
                for ki in 0..l_k {
                    acc += attn_scores[qi * l_k + ki] * v[ki * head_dim + hd];
                }
                out[qi * head_dim + hd] = acc;
            }
        }
        out
    }
}

// ─── InformerEncoder ─────────────────────────────────────────────────────────

/// Configuration for a stacked Informer encoder.
#[derive(Debug, Clone)]
pub struct InformerEncoderConfig {
    /// Total embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of encoder layers.
    pub n_layers: usize,
    /// Sampling factor c.
    pub factor: usize,
    /// Feed-forward hidden dimension (usually 4 × embed_dim).
    pub ff_dim: usize,
    /// Input sequence length.
    pub seq_len: usize,
    /// Dropout rate (stored for reference; not applied in this pure-Rust impl).
    pub dropout_rate: f32,
}

/// Learnable weights for all layers of an Informer encoder.
#[derive(Debug, Clone)]
pub struct InformerEncoderWeights {
    /// Per-layer ProbSparse attention weights.
    pub attn_weights: Vec<ProbSparseWeights>,
    /// Per-layer FFN first weight `[ff_dim × embed_dim]`.
    pub ff_w1: Vec<Vec<f32>>,
    /// Per-layer FFN first bias `[ff_dim]`.
    pub ff_b1: Vec<Vec<f32>>,
    /// Per-layer FFN second weight `[embed_dim × ff_dim]`.
    pub ff_w2: Vec<Vec<f32>>,
    /// Per-layer FFN second bias `[embed_dim]`.
    pub ff_b2: Vec<Vec<f32>>,
    /// Per-layer LayerNorm gain `[embed_dim]`.
    pub norm_gamma: Vec<Vec<f32>>,
    /// Per-layer LayerNorm bias `[embed_dim]`.
    pub norm_beta: Vec<Vec<f32>>,
}

/// Multi-layer Informer encoder: stacked ProbSparse attention + FFN sublayers.
#[derive(Debug, Clone)]
pub struct InformerEncoder {
    /// Encoder configuration.
    pub cfg: InformerEncoderConfig,
    /// Learnable parameters for all layers.
    pub weights: InformerEncoderWeights,
}

impl InformerEncoder {
    /// Construct an `InformerEncoder` with randomised weights.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidEmbedDim`] when `embed_dim == 0`.
    /// - [`TsError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`TsError::HeadDimMismatch`] when `embed_dim % n_heads != 0`.
    /// - [`TsError::InvalidSequenceLength`] when `seq_len == 0`.
    /// - [`TsError::InvalidHorizon`] when `factor == 0`.
    /// - [`TsError::ShapeMismatch`] when `n_layers == 0`.
    pub fn new(cfg: InformerEncoderConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if cfg.embed_dim == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if cfg.n_heads == 0 {
            return Err(TsError::InvalidNumHeads(0));
        }
        if cfg.embed_dim % cfg.n_heads != 0 {
            return Err(TsError::HeadDimMismatch {
                embed_dim: cfg.embed_dim,
                n_heads: cfg.n_heads,
            });
        }
        if cfg.seq_len == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if cfg.factor == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if cfg.n_layers == 0 {
            return Err(TsError::ShapeMismatch {
                msg: "n_layers must be >= 1".into(),
            });
        }
        if cfg.ff_dim == 0 {
            return Err(TsError::ShapeMismatch {
                msg: "ff_dim must be >= 1".into(),
            });
        }

        let d = cfg.embed_dim;
        let ff = cfg.ff_dim;
        let scale_attn = (2.0_f32 / d as f32).sqrt();
        let scale_ff1 = (2.0_f32 / d as f32).sqrt();

        let mut attn_weights = Vec::with_capacity(cfg.n_layers);
        let mut ff_w1 = Vec::with_capacity(cfg.n_layers);
        let mut ff_b1 = Vec::with_capacity(cfg.n_layers);
        let mut ff_w2 = Vec::with_capacity(cfg.n_layers);
        let mut ff_b2 = Vec::with_capacity(cfg.n_layers);
        let mut norm_gamma = Vec::with_capacity(cfg.n_layers);
        let mut norm_beta = Vec::with_capacity(cfg.n_layers);

        let _ = scale_attn; // used in ProbSparseWeights::new

        for _ in 0..cfg.n_layers {
            attn_weights.push(ProbSparseWeights::new(d, rng));

            // FFN W1: [ff_dim × embed_dim] Kaiming.
            let mut w1 = vec![0.0_f32; ff * d];
            rng.fill_normal(&mut w1);
            for w in &mut w1 {
                *w *= scale_ff1;
            }
            ff_w1.push(w1);
            ff_b1.push(vec![0.0_f32; ff]);

            // FFN W2: [embed_dim × ff_dim] Kaiming (fan_in = ff_dim).
            let scale_ff2 = (2.0_f32 / ff as f32).sqrt();
            let mut w2 = vec![0.0_f32; d * ff];
            rng.fill_normal(&mut w2);
            for w in &mut w2 {
                *w *= scale_ff2;
            }
            ff_w2.push(w2);
            ff_b2.push(vec![0.0_f32; d]);

            norm_gamma.push(vec![1.0_f32; d]);
            norm_beta.push(vec![0.0_f32; d]);
        }

        Ok(Self {
            cfg,
            weights: InformerEncoderWeights {
                attn_weights,
                ff_w1,
                ff_b1,
                ff_w2,
                ff_b2,
                norm_gamma,
                norm_beta,
            },
        })
    }

    /// Forward pass: `x [seq_len × embed_dim]` → `[seq_len × embed_dim]`.
    ///
    /// Each layer applies:
    /// 1. ProbSparse self-attention + residual + LayerNorm.
    /// 2. Feed-forward (GELU) + residual + LayerNorm.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != seq_len * embed_dim`.
    pub fn forward(&self, x: &[f32], rng: &mut LcgRng) -> TsResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        let seq = self.cfg.seq_len;

        if x.len() != seq * d {
            return Err(TsError::DimensionMismatch {
                expected: seq * d,
                got: x.len(),
            });
        }

        let mut h = x.to_vec();

        for layer in 0..self.cfg.n_layers {
            // Build a temporary InformerBlock for this layer's attention weights.
            let blk_cfg = ProbSparseConfig {
                embed_dim: d,
                n_heads: self.cfg.n_heads,
                factor: self.cfg.factor,
                dropout_rate: self.cfg.dropout_rate,
                seq_len: seq,
                label_len: 0,
                pred_len: 0,
            };
            let blk = InformerBlock {
                cfg: blk_cfg,
                weights: self.weights.attn_weights[layer].clone(),
            };

            // ProbSparse self-attention (Q=K=V=h).
            let attn_res = blk.forward(&h, seq, &h, &h, seq, rng)?;
            // Residual + LayerNorm.
            let after_attn: Vec<f32> = h
                .iter()
                .zip(attn_res.output.iter())
                .map(|(a, b)| a + b)
                .collect();
            let normed_attn = Self::layer_norm(
                &after_attn,
                &self.weights.norm_gamma[layer],
                &self.weights.norm_beta[layer],
                seq,
                d,
            );

            // Feed-forward sublayer.
            let ff_out = Self::feed_forward(
                &normed_attn,
                seq,
                &self.weights.ff_w1[layer],
                &self.weights.ff_b1[layer],
                self.cfg.ff_dim,
                &self.weights.ff_w2[layer],
                &self.weights.ff_b2[layer],
                d,
            );

            // Residual + LayerNorm (second norm over the same gamma/beta pair).
            let after_ff: Vec<f32> = normed_attn
                .iter()
                .zip(ff_out.iter())
                .map(|(a, b)| a + b)
                .collect();
            h = Self::layer_norm(
                &after_ff,
                &self.weights.norm_gamma[layer],
                &self.weights.norm_beta[layer],
                seq,
                d,
            );
        }

        Ok(h)
    }

    /// Row-wise LayerNorm of `[seq_len × embed_dim]` with learnable affine.
    ///
    /// Each row is normalised to zero mean and unit variance (eps = 1e-5),
    /// then scaled by `gamma` and shifted by `beta`.
    pub fn layer_norm(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        seq_len: usize,
        embed_dim: usize,
    ) -> Vec<f32> {
        let mut out = x.to_vec();
        for t in 0..seq_len {
            let row = &mut out[t * embed_dim..(t + 1) * embed_dim];
            let mean: f32 = row.iter().sum::<f32>() / embed_dim as f32;
            let var: f32 = row
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                / embed_dim as f32;
            let inv_std = (var + 1e-5_f32).sqrt().recip();
            for (j, v) in row.iter_mut().enumerate() {
                *v = (*v - mean) * inv_std * gamma[j] + beta[j];
            }
        }
        out
    }

    /// Feed-forward sublayer: GELU(x @ W1 + b1) @ W2 + b2.
    ///
    /// # Arguments
    ///
    /// * `x` — `[seq_len × embed_dim]` input.
    /// * `w1` — `[ff_dim × embed_dim]`.
    /// * `b1` — `[ff_dim]`.
    /// * `w2` — `[embed_dim × ff_dim]`.
    /// * `b2` — `[embed_dim]`.
    ///
    /// Returns `[seq_len × embed_dim]`.
    pub fn feed_forward(
        x: &[f32],
        seq_len: usize,
        w1: &[f32],
        b1: &[f32],
        ff_dim: usize,
        w2: &[f32],
        b2: &[f32],
        embed_dim: usize,
    ) -> Vec<f32> {
        // Hidden layer: [seq_len × ff_dim].
        let mut hidden = vec![0.0_f32; seq_len * ff_dim];
        for t in 0..seq_len {
            for fi in 0..ff_dim {
                let mut acc = b1[fi];
                for k in 0..embed_dim {
                    acc += x[t * embed_dim + k] * w1[fi * embed_dim + k];
                }
                hidden[t * ff_dim + fi] = Self::gelu(acc);
            }
        }

        // Output layer: [seq_len × embed_dim].
        let mut out = vec![0.0_f32; seq_len * embed_dim];
        for t in 0..seq_len {
            for di in 0..embed_dim {
                let mut acc = b2[di];
                for fi in 0..ff_dim {
                    acc += hidden[t * ff_dim + fi] * w2[di * ff_dim + fi];
                }
                out[t * embed_dim + di] = acc;
            }
        }
        out
    }

    /// GELU activation using the tanh approximation.
    ///
    /// `GELU(x) = 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))`
    #[inline]
    pub fn gelu(x: f32) -> f32 {
        let c = std::f32::consts::FRAC_2_SQRT_PI * std::f32::consts::FRAC_1_SQRT_2;
        // √(2/π) = √2 · (1/√π) — standard Gaussian CDF approximation constant.
        let inner = c * (x + 0.044_715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Linear projection: `[seq × d_in] @ W^T + b → [seq × d_out]`.
///
/// `w` is `[d_out × d_in]` row-major; `b` is `[d_out]`.
fn linear_proj(x: &[f32], seq: usize, d: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let d_out = b.len();
    let mut out = vec![0.0_f32; seq * d_out];
    for t in 0..seq {
        for oi in 0..d_out {
            let mut acc = b[oi];
            for k in 0..d {
                acc += x[t * d + k] * w[oi * d + k];
            }
            out[t * d_out + oi] = acc;
        }
    }
    out
}

/// Extract one head slice from a multi-head projected matrix.
///
/// Copies column slice `[h_off .. h_off + head_dim]` from each row of
/// `[seq × d]` into a new `[seq × head_dim]` buffer.
fn extract_head(x: &[f32], seq: usize, d: usize, h_off: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; seq * head_dim];
    for t in 0..seq {
        for hd in 0..head_dim {
            out[t * head_dim + hd] = x[t * d + h_off + hd];
        }
    }
    out
}

/// Numerically stable in-place softmax over a row.
fn softmax_row(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = if sum == 0.0 { 1.0 } else { sum.recip() };
    for v in row.iter_mut() {
        *v *= inv_sum;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(2024)
    }

    fn tiny_cfg(seq: usize) -> ProbSparseConfig {
        ProbSparseConfig {
            embed_dim: 16,
            n_heads: 2,
            factor: 2,
            dropout_rate: 0.0,
            seq_len: seq,
            label_len: 4,
            pred_len: 8,
        }
    }

    // 1. Output shape is seq_len_q × embed_dim.
    #[test]
    fn prob_sparse_output_shape() {
        let mut rng = make_rng();
        let cfg = tiny_cfg(12);
        let block = InformerBlock::new(cfg, &mut rng).expect("build");
        let d = block.cfg.embed_dim;
        let seq_q = 12usize;
        let seq_kv = 12usize;
        let q = vec![0.1_f32; seq_q * d];
        let k = vec![0.1_f32; seq_kv * d];
        let v = vec![0.1_f32; seq_kv * d];
        let res = block
            .forward(&q, seq_q, &k, &v, seq_kv, &mut rng)
            .expect("forward");
        assert_eq!(res.output.len(), seq_q * d);
    }

    // 2. Output is finite (no NaN/inf).
    #[test]
    fn prob_sparse_output_finite() {
        let mut rng = make_rng();
        let cfg = tiny_cfg(10);
        let block = InformerBlock::new(cfg, &mut rng).expect("build");
        let d = block.cfg.embed_dim;
        let seq = 10usize;
        let mut q = vec![0.0_f32; seq * d];
        let mut k = vec![0.0_f32; seq * d];
        let mut v = vec![0.0_f32; seq * d];
        rng.fill_normal(&mut q);
        rng.fill_normal(&mut k);
        rng.fill_normal(&mut v);
        let res = block
            .forward(&q, seq, &k, &v, seq, &mut rng)
            .expect("forward");
        assert!(
            res.output.iter().all(|x| x.is_finite()),
            "non-finite output"
        );
    }

    // 3. Single head output shape is l_q × head_dim.
    #[test]
    fn single_head_output_shape() {
        let mut rng = make_rng();
        let l_q = 8usize;
        let l_k = 8usize;
        let head_dim = 8usize;
        let q = vec![0.1_f32; l_q * head_dim];
        let k = vec![0.1_f32; l_k * head_dim];
        let v = vec![0.1_f32; l_k * head_dim];
        let (out, _) =
            InformerBlock::prob_sparse_attention(&q, &k, &v, l_q, l_k, head_dim, 2, &mut rng);
        assert_eq!(out.len(), l_q * head_dim);
    }

    // 4. Selected indices count == u_select = min(factor*ceil(ln(l_k)), l_k, l_q).
    #[test]
    fn selected_indices_count() {
        let mut rng = make_rng();
        let l_q = 16usize;
        let l_k = 16usize;
        let head_dim = 8usize;
        let factor = 3usize;
        let q = vec![0.2_f32; l_q * head_dim];
        let k = vec![0.3_f32; l_k * head_dim];
        let v = vec![0.1_f32; l_k * head_dim];
        let (_, sel_idx) =
            InformerBlock::prob_sparse_attention(&q, &k, &v, l_q, l_k, head_dim, factor, &mut rng);
        let ln_lk = (l_k as f32).ln().ceil() as usize;
        let u_expected = (factor * ln_lk).min(l_k).min(l_q);
        assert_eq!(sel_idx.len(), u_expected);
    }

    // 5. u ≤ l_q (selected never exceeds number of queries).
    #[test]
    fn u_at_most_l_q() {
        let mut rng = make_rng();
        let l_q = 4usize;
        let l_k = 100usize;
        let head_dim = 4usize;
        let factor = 50usize;
        let q = vec![0.1_f32; l_q * head_dim];
        let k = vec![0.1_f32; l_k * head_dim];
        let v = vec![0.1_f32; l_k * head_dim];
        let (_, sel_idx) =
            InformerBlock::prob_sparse_attention(&q, &k, &v, l_q, l_k, head_dim, factor, &mut rng);
        assert!(
            sel_idx.len() <= l_q,
            "selected {} > l_q {}",
            sel_idx.len(),
            l_q
        );
    }

    // 6. Unselected positions get exactly mean(V).
    #[test]
    fn mean_fill_for_unselected() {
        let mut rng = make_rng();
        let l_q = 8usize;
        let l_k = 4usize;
        let head_dim = 4usize;
        let factor = 1usize; // small factor → few selected
        // Use a fixed V for easy mean computation.
        let v: Vec<f32> = (0..l_k * head_dim).map(|i| i as f32 * 0.1).collect();
        let q = vec![0.5_f32; l_q * head_dim];
        let k = vec![0.5_f32; l_k * head_dim];
        let (out, sel_idx) =
            InformerBlock::prob_sparse_attention(&q, &k, &v, l_q, l_k, head_dim, factor, &mut rng);

        // Compute expected mean(V) per head dim.
        let mut mean_v = vec![0.0_f32; head_dim];
        for ki in 0..l_k {
            for hd in 0..head_dim {
                mean_v[hd] += v[ki * head_dim + hd];
            }
        }
        for mv in &mut mean_v {
            *mv /= l_k as f32;
        }

        let sel_set: std::collections::HashSet<usize> = sel_idx.into_iter().collect();
        for qi in 0..l_q {
            if !sel_set.contains(&qi) {
                for hd in 0..head_dim {
                    let got = out[qi * head_dim + hd];
                    let exp = mean_v[hd];
                    assert!(
                        (got - exp).abs() < 1e-5,
                        "unselected qi={qi} hd={hd}: got={got} exp={exp}"
                    );
                }
            }
        }
    }

    // 7. Large factor → all queries selected (full attention).
    #[test]
    fn full_attention_when_factor_large() {
        let mut rng = make_rng();
        let l_q = 6usize;
        let l_k = 6usize;
        let head_dim = 4usize;
        let factor = 1000usize; // ensures u >= l_q
        let q = vec![0.1_f32; l_q * head_dim];
        let k = vec![0.1_f32; l_k * head_dim];
        let v = vec![0.1_f32; l_k * head_dim];
        let (_, sel_idx) =
            InformerBlock::prob_sparse_attention(&q, &k, &v, l_q, l_k, head_dim, factor, &mut rng);
        assert_eq!(sel_idx.len(), l_q, "expected all {} queries selected", l_q);
    }

    // 8. Encoder forward shape: seq_len × embed_dim.
    #[test]
    fn encoder_forward_shape() {
        let mut rng = make_rng();
        let seq = 12usize;
        let d = 16usize;
        let enc_cfg = InformerEncoderConfig {
            embed_dim: d,
            n_heads: 2,
            n_layers: 2,
            factor: 2,
            ff_dim: 32,
            seq_len: seq,
            dropout_rate: 0.0,
        };
        let enc = InformerEncoder::new(enc_cfg, &mut rng).expect("build");
        let x = vec![0.1_f32; seq * d];
        let out = enc.forward(&x, &mut rng).expect("forward");
        assert_eq!(out.len(), seq * d);
    }

    // 9. Encoder forward output is finite.
    #[test]
    fn encoder_forward_finite() {
        let mut rng = make_rng();
        let seq = 8usize;
        let d = 16usize;
        let enc_cfg = InformerEncoderConfig {
            embed_dim: d,
            n_heads: 2,
            n_layers: 2,
            factor: 2,
            ff_dim: 32,
            seq_len: seq,
            dropout_rate: 0.0,
        };
        let enc = InformerEncoder::new(enc_cfg, &mut rng).expect("build");
        let mut x = vec![0.0_f32; seq * d];
        rng.fill_normal(&mut x);
        let out = enc.forward(&x, &mut rng).expect("forward");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "encoder produced non-finite output"
        );
    }

    // 10. LayerNorm: mean of each row ≈ 0.
    #[test]
    fn layer_norm_mean_approx_zero() {
        let seq = 5usize;
        let d = 8usize;
        let x: Vec<f32> = (0..seq * d).map(|i| i as f32 * 0.3 - 6.0).collect();
        let gamma = vec![1.0_f32; d];
        let beta = vec![0.0_f32; d];
        let out = InformerEncoder::layer_norm(&x, &gamma, &beta, seq, d);
        for t in 0..seq {
            let mean: f32 = (0..d).map(|k| out[t * d + k]).sum::<f32>() / d as f32;
            assert!(mean.abs() < 1e-4, "row {t} mean={mean}");
        }
    }

    // 11. GELU(0) ≈ 0.
    #[test]
    fn gelu_at_zero() {
        let v = InformerEncoder::gelu(0.0);
        assert!(v.abs() < 1e-6, "gelu(0)={v}");
    }

    // 12. GELU(5) ≈ 5.0 (large x → GELU ≈ identity).
    #[test]
    fn gelu_positive_for_large_x() {
        let v = InformerEncoder::gelu(5.0);
        assert!((v - 5.0).abs() < 0.01, "gelu(5.0)={v}");
    }

    // 13. embed_dim == 0 → InvalidEmbedDim.
    #[test]
    fn err_embed_dim_zero() {
        let mut rng = make_rng();
        let cfg = ProbSparseConfig {
            embed_dim: 0,
            n_heads: 2,
            factor: 2,
            dropout_rate: 0.0,
            seq_len: 8,
            label_len: 2,
            pred_len: 4,
        };
        assert!(matches!(
            InformerBlock::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidEmbedDim(0)
        ));
    }

    // 14. n_heads == 0 → InvalidNumHeads.
    #[test]
    fn err_n_heads_zero() {
        let mut rng = make_rng();
        let cfg = ProbSparseConfig {
            embed_dim: 16,
            n_heads: 0,
            factor: 2,
            dropout_rate: 0.0,
            seq_len: 8,
            label_len: 2,
            pred_len: 4,
        };
        assert!(matches!(
            InformerBlock::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidNumHeads(0)
        ));
    }

    // 15. embed_dim % n_heads != 0 → HeadDimMismatch.
    #[test]
    fn err_head_dim_not_divisible() {
        let mut rng = make_rng();
        let cfg = ProbSparseConfig {
            embed_dim: 15,
            n_heads: 4,
            factor: 2,
            dropout_rate: 0.0,
            seq_len: 8,
            label_len: 2,
            pred_len: 4,
        };
        assert!(matches!(
            InformerBlock::new(cfg, &mut rng).unwrap_err(),
            TsError::HeadDimMismatch { .. }
        ));
    }

    // 16. seq_len == 0 → InvalidSequenceLength.
    #[test]
    fn err_seq_len_zero() {
        let mut rng = make_rng();
        let cfg = ProbSparseConfig {
            embed_dim: 16,
            n_heads: 2,
            factor: 2,
            dropout_rate: 0.0,
            seq_len: 0,
            label_len: 0,
            pred_len: 4,
        };
        assert!(matches!(
            InformerBlock::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }
}

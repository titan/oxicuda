//! SAINT: Self-Attention and Intersample Attention Transformer (Somepalli et al. 2021).
//!
//! Two types of attention alternate per block:
//! 1. **Row-wise**: standard MHSA across features (each feature is a token) for a single row.
//! 2. **Intersample**: MHSA across rows (samples) for each feature position.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── SaintConfig ─────────────────────────────────────────────────────────────

/// Configuration for `SaintLayer`.
pub struct SaintConfig {
    /// Number of input features.
    pub n_features: usize,
    /// Feature embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads (`embed_dim % n_heads == 0`).
    pub n_heads: usize,
    /// Number of alternating block pairs.
    pub n_layers: usize,
    /// FFN hidden dimension (typically `4 * embed_dim`).
    pub ffn_hidden: usize,
    /// Output classes.
    pub n_classes: usize,
}

// ─── Layer normalisation ─────────────────────────────────────────────────────

fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    x.iter()
        .zip(gamma.iter().zip(beta.iter()))
        .map(|(&xi, (&g, &b))| (xi - mean) / (var + eps).sqrt() * g + b)
        .collect()
}

// ─── Scaled dot-product attention ────────────────────────────────────────────

/// Single-head scaled dot-product attention.
///
/// - `q`, `k`, `v`: flat `[seq_len * head_dim]` row-major matrices.
/// - Returns: `[seq_len * head_dim]`.
pub fn self_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> TabularResult<Vec<f32>> {
    if q.len() != seq_len * head_dim {
        return Err(TabularError::DimensionMismatch {
            expected: seq_len * head_dim,
            got: q.len(),
        });
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    // scores[i, j] = dot(q[i], k[j]) * scale
    let mut scores = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }

    // Stable softmax per row
    let mut attn = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        let row = &scores[i * seq_len..(i + 1) * seq_len];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&s| (s - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let denom = if sum < 1e-30 { 1e-30 } else { sum };
        for j in 0..seq_len {
            attn[i * seq_len + j] = exps[j] / denom;
        }
    }

    // out = attn · V
    let mut out = vec![0.0_f32; seq_len * head_dim];
    for i in 0..seq_len {
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for j in 0..seq_len {
                acc += attn[i * seq_len + j] * v[j * head_dim + d];
            }
            out[i * head_dim + d] = acc;
        }
    }
    Ok(out)
}

/// Matrix multiply helper: `C = A * B` where A is `[m×k]`, B is `[k×n]`, all row-major.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Multi-head self-attention.
///
/// - `x`: `[seq_len * embed_dim]`
/// - `wq`, `wk`, `wv`, `wo`: `[embed_dim * embed_dim]`
/// - Returns: `[seq_len * embed_dim]`
pub fn multihead_attention(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    seq_len: usize,
    embed_dim: usize,
    n_heads: usize,
) -> TabularResult<Vec<f32>> {
    if !embed_dim.is_multiple_of(n_heads) {
        return Err(TabularError::InvalidAttentionDim { dim: embed_dim });
    }
    let head_dim = embed_dim / n_heads;

    // Project x to Q, K, V
    let q = matmul(x, wq, seq_len, embed_dim, embed_dim);
    let k = matmul(x, wk, seq_len, embed_dim, embed_dim);
    let v = matmul(x, wv, seq_len, embed_dim, embed_dim);

    // Per-head attention and concatenate
    let mut concat = vec![0.0_f32; seq_len * embed_dim];
    for h in 0..n_heads {
        let h_start = h * head_dim;
        // Extract head slices
        let mut q_h = vec![0.0_f32; seq_len * head_dim];
        let mut k_h = vec![0.0_f32; seq_len * head_dim];
        let mut v_h = vec![0.0_f32; seq_len * head_dim];
        for s in 0..seq_len {
            let src = s * embed_dim + h_start;
            q_h[s * head_dim..s * head_dim + head_dim].copy_from_slice(&q[src..src + head_dim]);
            k_h[s * head_dim..s * head_dim + head_dim].copy_from_slice(&k[src..src + head_dim]);
            v_h[s * head_dim..s * head_dim + head_dim].copy_from_slice(&v[src..src + head_dim]);
        }
        let head_out = self_attention(&q_h, &k_h, &v_h, seq_len, head_dim)?;
        // Write back into concat buffer
        for s in 0..seq_len {
            concat[s * embed_dim + h_start..s * embed_dim + h_start + head_dim]
                .copy_from_slice(&head_out[s * head_dim..s * head_dim + head_dim]);
        }
    }

    // Output projection
    let out = matmul(&concat, wo, seq_len, embed_dim, embed_dim);
    Ok(out)
}

/// Intersample attention: MHSA across `n_samples` for each feature position.
///
/// - `x`: `[n_samples * n_features * embed_dim]`
/// - `wq`, `wk`, `wv`, `wo`: `[embed_dim * embed_dim]`
/// - Returns: `[n_samples * n_features * embed_dim]`
pub fn intersample_attention(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    n_samples: usize,
    n_features: usize,
    embed_dim: usize,
) -> TabularResult<Vec<f32>> {
    if x.len() != n_samples * n_features * embed_dim {
        return Err(TabularError::DimensionMismatch {
            expected: n_samples * n_features * embed_dim,
            got: x.len(),
        });
    }

    let mut out = vec![0.0_f32; n_samples * n_features * embed_dim];

    // For each feature position f, gather the n_samples embeddings and run MHSA
    let mut feat_seq = vec![0.0_f32; n_samples * embed_dim];
    for f in 0..n_features {
        // Extract feature f across all samples
        for s in 0..n_samples {
            let src = s * n_features * embed_dim + f * embed_dim;
            feat_seq[s * embed_dim..(s + 1) * embed_dim].copy_from_slice(&x[src..src + embed_dim]);
        }
        // Single-head MHSA (embed_dim == head_dim for n_heads=1 here; caller passes appropriate Ws)
        let attn_out = multihead_attention(&feat_seq, wq, wk, wv, wo, n_samples, embed_dim, 1)?;
        // Scatter back
        for s in 0..n_samples {
            let dst = s * n_features * embed_dim + f * embed_dim;
            out[dst..dst + embed_dim]
                .copy_from_slice(&attn_out[s * embed_dim..(s + 1) * embed_dim]);
        }
    }
    Ok(out)
}

// ─── Xavier init helper ───────────────────────────────────────────────────────

fn xavier(rng: &mut LcgRng, fan_in: usize, fan_out: usize, n: usize) -> Vec<f32> {
    let std_dev = (2.0_f32 / (fan_in + fan_out) as f32).sqrt();
    let mut w = vec![0.0_f32; n];
    rng.fill_normal_scaled(&mut w, std_dev);
    w
}

// ─── SaintLayer ──────────────────────────────────────────────────────────────

/// SAINT model layer with row + intersample attention blocks.
pub struct SaintLayer {
    // Per-layer row attention weights [n_layers][embed_dim * embed_dim]
    row_wq: Vec<Vec<f32>>,
    row_wk: Vec<Vec<f32>>,
    row_wv: Vec<Vec<f32>>,
    row_wo: Vec<Vec<f32>>,
    // Per-layer intersample attention weights
    inter_wq: Vec<Vec<f32>>,
    inter_wk: Vec<Vec<f32>>,
    inter_wv: Vec<Vec<f32>>,
    inter_wo: Vec<Vec<f32>>,
    // FFN weights per layer: W1 [embed_dim → ffn_hidden], W2 [ffn_hidden → embed_dim]
    ffn_w1: Vec<Vec<f32>>,
    ffn_b1: Vec<Vec<f32>>,
    ffn_w2: Vec<Vec<f32>>,
    ffn_b2: Vec<Vec<f32>>,
    // Layer norms: 4 per layer (before row_attn, before inter_attn, before ffn1, before ffn2)
    // gamma[layer * 4 + which], beta[layer * 4 + which]
    ln_gamma: Vec<Vec<f32>>,
    ln_beta: Vec<Vec<f32>>,
    // Output head: [embed_dim → n_classes]
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    config: SaintConfig,
}

impl SaintLayer {
    /// Construct a new `SaintLayer` with Xavier initialisation.
    pub fn new(cfg: SaintConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if cfg.n_features == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if cfg.embed_dim == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if !cfg.embed_dim.is_multiple_of(cfg.n_heads) {
            return Err(TabularError::InvalidAttentionDim { dim: cfg.embed_dim });
        }

        let ed = cfg.embed_dim;
        let ffn_h = cfg.ffn_hidden;
        let nl = cfg.n_layers;

        let mut row_wq = Vec::with_capacity(nl);
        let mut row_wk = Vec::with_capacity(nl);
        let mut row_wv = Vec::with_capacity(nl);
        let mut row_wo = Vec::with_capacity(nl);
        let mut inter_wq = Vec::with_capacity(nl);
        let mut inter_wk = Vec::with_capacity(nl);
        let mut inter_wv = Vec::with_capacity(nl);
        let mut inter_wo = Vec::with_capacity(nl);
        let mut ffn_w1 = Vec::with_capacity(nl);
        let mut ffn_b1 = Vec::with_capacity(nl);
        let mut ffn_w2 = Vec::with_capacity(nl);
        let mut ffn_b2 = Vec::with_capacity(nl);
        let mut ln_gamma = Vec::with_capacity(nl * 4);
        let mut ln_beta = Vec::with_capacity(nl * 4);

        for _ in 0..nl {
            row_wq.push(xavier(rng, ed, ed, ed * ed));
            row_wk.push(xavier(rng, ed, ed, ed * ed));
            row_wv.push(xavier(rng, ed, ed, ed * ed));
            row_wo.push(xavier(rng, ed, ed, ed * ed));

            inter_wq.push(xavier(rng, ed, ed, ed * ed));
            inter_wk.push(xavier(rng, ed, ed, ed * ed));
            inter_wv.push(xavier(rng, ed, ed, ed * ed));
            inter_wo.push(xavier(rng, ed, ed, ed * ed));

            ffn_w1.push(xavier(rng, ed, ffn_h, ed * ffn_h));
            ffn_b1.push(vec![0.0_f32; ffn_h]);
            ffn_w2.push(xavier(rng, ffn_h, ed, ffn_h * ed));
            ffn_b2.push(vec![0.0_f32; ed]);

            // 4 layer norms per block: row_pre, inter_pre, ffn1_pre, ffn2_pre
            for _ in 0..4 {
                ln_gamma.push(vec![1.0_f32; ed]);
                ln_beta.push(vec![0.0_f32; ed]);
            }
        }

        let head_w = xavier(rng, ed, cfg.n_classes, ed * cfg.n_classes);
        let head_b = vec![0.0_f32; cfg.n_classes];

        Ok(Self {
            row_wq,
            row_wk,
            row_wv,
            row_wo,
            inter_wq,
            inter_wk,
            inter_wv,
            inter_wo,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
            ln_gamma,
            ln_beta,
            head_w,
            head_b,
            config: cfg,
        })
    }

    /// Apply a FFN: GELU activation, `W2(GELU(W1 x + b1)) + b2`.
    fn ffn(
        x: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        // h = W1 x + b1, shape [hidden]
        let mut h = b1.to_vec();
        for o in 0..hidden {
            for i in 0..in_dim {
                h[o] += w1[o * in_dim + i] * x[i];
            }
        }
        // GELU: h = h * sigmoid(1.702 * h)
        for v in &mut h {
            *v *= 1.0 / (1.0 + (-1.702 * *v).exp());
        }
        // out = W2 h + b2, shape [out_dim]
        let mut out = b2.to_vec();
        for o in 0..out_dim {
            for i in 0..hidden {
                out[o] += w2[o * hidden + i] * h[i];
            }
        }
        out
    }

    /// Add two vectors element-wise (residual connection).
    fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    /// Forward pass for a batch of `n_samples` samples.
    ///
    /// - `x`: `[n_samples * n_features * embed_dim]` — feature tokens per sample.
    /// - Returns: logits `[n_samples * n_classes]`.
    pub fn forward(&self, x: &[f32], n_samples: usize) -> TabularResult<Vec<f32>> {
        let cfg = &self.config;
        let ed = cfg.embed_dim;
        let nf = cfg.n_features;
        let total = n_samples * nf * ed;

        if x.len() != total {
            return Err(TabularError::DimensionMismatch {
                expected: total,
                got: x.len(),
            });
        }

        let mut h = x.to_vec();

        for layer in 0..cfg.n_layers {
            let ln_row = layer * 4;
            let ln_inter = layer * 4 + 1;
            let ln_ffn1 = layer * 4 + 2;
            let ln_ffn2 = layer * 4 + 3;

            // ── Row-wise self-attention ──────────────────────────────────────
            // For each sample: apply MHSA across features
            let mut row_attn_out = vec![0.0_f32; n_samples * nf * ed];
            for s in 0..n_samples {
                let sample_start = s * nf * ed;
                let sample_slice = &h[sample_start..sample_start + nf * ed];

                // Pre-LN each token independently
                let mut pre_ln = vec![0.0_f32; nf * ed];
                for f in 0..nf {
                    let tok = &sample_slice[f * ed..(f + 1) * ed];
                    let normed =
                        layer_norm(tok, &self.ln_gamma[ln_row], &self.ln_beta[ln_row], 1e-5);
                    pre_ln[f * ed..(f + 1) * ed].copy_from_slice(&normed);
                }

                let attn = multihead_attention(
                    &pre_ln,
                    &self.row_wq[layer],
                    &self.row_wk[layer],
                    &self.row_wv[layer],
                    &self.row_wo[layer],
                    nf,
                    ed,
                    cfg.n_heads,
                )?;
                // Residual
                for i in 0..nf * ed {
                    row_attn_out[sample_start + i] = sample_slice[i] + attn[i];
                }
            }

            // ── Intersample attention ────────────────────────────────────────
            // Pre-LN per token
            let mut pre_ln_inter = vec![0.0_f32; n_samples * nf * ed];
            for idx in 0..n_samples * nf {
                let tok = &row_attn_out[idx * ed..(idx + 1) * ed];
                let normed =
                    layer_norm(tok, &self.ln_gamma[ln_inter], &self.ln_beta[ln_inter], 1e-5);
                pre_ln_inter[idx * ed..(idx + 1) * ed].copy_from_slice(&normed);
            }

            let inter_out = intersample_attention(
                &pre_ln_inter,
                &self.inter_wq[layer],
                &self.inter_wk[layer],
                &self.inter_wv[layer],
                &self.inter_wo[layer],
                n_samples,
                nf,
                ed,
            )?;

            // Residual
            let mut after_inter: Vec<f32> = row_attn_out
                .iter()
                .zip(inter_out.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            // ── FFN per token ─────────────────────────────────────────────────
            let mut after_ffn = vec![0.0_f32; n_samples * nf * ed];
            for idx in 0..n_samples * nf {
                let tok = &after_inter[idx * ed..(idx + 1) * ed];
                let normed = layer_norm(tok, &self.ln_gamma[ln_ffn1], &self.ln_beta[ln_ffn1], 1e-5);
                let ffn_out = Self::ffn(
                    &normed,
                    &self.ffn_w1[layer],
                    &self.ffn_b1[layer],
                    &self.ffn_w2[layer],
                    &self.ffn_b2[layer],
                    ed,
                    cfg.ffn_hidden,
                    ed,
                );
                // Second pre-LN before residual add
                let normed2 = layer_norm(
                    &ffn_out,
                    &self.ln_gamma[ln_ffn2],
                    &self.ln_beta[ln_ffn2],
                    1e-5,
                );
                let tok_out = Self::add_vecs(tok, &normed2);
                after_ffn[idx * ed..(idx + 1) * ed].copy_from_slice(&tok_out);
            }

            // Update h for next layer
            after_inter = after_ffn;
            h = after_inter;
        }

        // Classification head: pool by averaging features per sample, then project
        let mut logits = Vec::with_capacity(n_samples * cfg.n_classes);
        for s in 0..n_samples {
            let sample_start = s * nf * ed;
            // Average over feature tokens
            let mut pooled = vec![0.0_f32; ed];
            for f in 0..nf {
                let tok = &h[sample_start + f * ed..sample_start + f * ed + ed];
                for (d, &tv) in tok.iter().enumerate() {
                    pooled[d] += tv;
                }
            }
            for v in &mut pooled {
                *v /= nf as f32;
            }
            // Linear head
            let mut sample_logits = self.head_b.clone();
            for (o, sl) in sample_logits.iter_mut().enumerate() {
                for (d, &pv) in pooled.iter().enumerate() {
                    *sl += self.head_w[o * ed + d] * pv;
                }
            }
            logits.extend_from_slice(&sample_logits);
        }
        Ok(logits)
    }
}

// ─── Crate-internal accessors for the analytic backward pass ──────────────────
// The backward implementation lives in `saint_grad.rs`.

impl SaintLayer {
    pub(crate) fn config_ref(&self) -> &SaintConfig {
        &self.config
    }
    pub(crate) fn row_wq_ref(&self, l: usize) -> &[f32] {
        &self.row_wq[l]
    }
    pub(crate) fn row_wk_ref(&self, l: usize) -> &[f32] {
        &self.row_wk[l]
    }
    pub(crate) fn row_wv_ref(&self, l: usize) -> &[f32] {
        &self.row_wv[l]
    }
    pub(crate) fn row_wo_ref(&self, l: usize) -> &[f32] {
        &self.row_wo[l]
    }
    pub(crate) fn inter_wq_ref(&self, l: usize) -> &[f32] {
        &self.inter_wq[l]
    }
    pub(crate) fn inter_wk_ref(&self, l: usize) -> &[f32] {
        &self.inter_wk[l]
    }
    pub(crate) fn inter_wv_ref(&self, l: usize) -> &[f32] {
        &self.inter_wv[l]
    }
    pub(crate) fn inter_wo_ref(&self, l: usize) -> &[f32] {
        &self.inter_wo[l]
    }
    pub(crate) fn ffn_w1_ref(&self, l: usize) -> &[f32] {
        &self.ffn_w1[l]
    }
    pub(crate) fn ffn_b1_ref(&self, l: usize) -> &[f32] {
        &self.ffn_b1[l]
    }
    pub(crate) fn ffn_w2_ref(&self, l: usize) -> &[f32] {
        &self.ffn_w2[l]
    }
    pub(crate) fn ffn_b2_ref(&self, l: usize) -> &[f32] {
        &self.ffn_b2[l]
    }
    pub(crate) fn ln_gamma_ref(&self, idx: usize) -> &[f32] {
        &self.ln_gamma[idx]
    }
    pub(crate) fn ln_beta_ref(&self, idx: usize) -> &[f32] {
        &self.ln_beta[idx]
    }
    pub(crate) fn head_w_ref(&self) -> &[f32] {
        &self.head_w
    }
    pub(crate) fn head_b_ref(&self) -> &[f32] {
        &self.head_b
    }

    /// Read a single scalar parameter (test-only, for finite-difference checks).
    #[cfg(test)]
    pub(crate) fn param_get(&self, p: &crate::attention::saint_grad::SaintParam) -> f32 {
        use crate::attention::saint_grad::SaintParam as P;
        match *p {
            P::RowWq(l, i) => self.row_wq[l][i],
            P::RowWk(l, i) => self.row_wk[l][i],
            P::RowWv(l, i) => self.row_wv[l][i],
            P::RowWo(l, i) => self.row_wo[l][i],
            P::InterWq(l, i) => self.inter_wq[l][i],
            P::InterWk(l, i) => self.inter_wk[l][i],
            P::InterWv(l, i) => self.inter_wv[l][i],
            P::InterWo(l, i) => self.inter_wo[l][i],
            P::FfnW1(l, i) => self.ffn_w1[l][i],
            P::FfnB1(l, i) => self.ffn_b1[l][i],
            P::FfnW2(l, i) => self.ffn_w2[l][i],
            P::FfnB2(l, i) => self.ffn_b2[l][i],
            P::LnGamma(idx, i) => self.ln_gamma[idx][i],
            P::LnBeta(idx, i) => self.ln_beta[idx][i],
            P::HeadW(i) => self.head_w[i],
            P::HeadB(i) => self.head_b[i],
        }
    }

    /// Write a single scalar parameter (test-only, for finite-difference checks).
    #[cfg(test)]
    pub(crate) fn param_set(&mut self, p: &crate::attention::saint_grad::SaintParam, val: f32) {
        use crate::attention::saint_grad::SaintParam as P;
        match *p {
            P::RowWq(l, i) => self.row_wq[l][i] = val,
            P::RowWk(l, i) => self.row_wk[l][i] = val,
            P::RowWv(l, i) => self.row_wv[l][i] = val,
            P::RowWo(l, i) => self.row_wo[l][i] = val,
            P::InterWq(l, i) => self.inter_wq[l][i] = val,
            P::InterWk(l, i) => self.inter_wk[l][i] = val,
            P::InterWv(l, i) => self.inter_wv[l][i] = val,
            P::InterWo(l, i) => self.inter_wo[l][i] = val,
            P::FfnW1(l, i) => self.ffn_w1[l][i] = val,
            P::FfnB1(l, i) => self.ffn_b1[l][i] = val,
            P::FfnW2(l, i) => self.ffn_w2[l][i] = val,
            P::FfnB2(l, i) => self.ffn_b2[l][i] = val,
            P::LnGamma(idx, i) => self.ln_gamma[idx][i] = val,
            P::LnBeta(idx, i) => self.ln_beta[idx][i] = val,
            P::HeadW(i) => self.head_w[i] = val,
            P::HeadB(i) => self.head_b[i] = val,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn saint_forward_shape() {
        let mut rng = LcgRng::new(42);
        let cfg = SaintConfig {
            n_features: 4,
            embed_dim: 8,
            n_heads: 2,
            n_layers: 2,
            ffn_hidden: 16,
            n_classes: 3,
        };
        let model = SaintLayer::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.1_f32; 2 * 4 * 8]; // 2 samples, 4 features, embed_dim=8
        let out = model.forward(&x, 2).expect("forward should succeed");
        assert_eq!(out.len(), 2 * 3);
    }

    #[test]
    fn self_attention_output_shape() {
        let q = vec![0.1_f32; 4 * 8];
        let k = vec![0.1_f32; 4 * 8];
        let v = vec![0.1_f32; 4 * 8];
        let out = self_attention(&q, &k, &v, 4, 8).expect("self_attention should succeed");
        assert_eq!(out.len(), 4 * 8);
    }
}

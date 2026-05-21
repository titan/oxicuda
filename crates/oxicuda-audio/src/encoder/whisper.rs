//! Whisper-style speech encoder (Radford et al., 2023).
//!
//! Implements the encoder branch from OpenAI Whisper:
//!
//! ```text
//! mel [n_mels, seq_in]
//!   → Conv1d (n_mels → d_model, k=3, stride=1, same-pad) → GELU
//!   → Conv1d (d_model → d_model, k=3, stride=2, same-pad) → GELU
//!   → transpose to [seq_out, d_model]
//!   → + sinusoidal positional embedding [seq_out, d_model]
//!   → N × pre-norm transformer encoder block
//!         · LN → MHSA → +residual
//!         · LN → FFN (GELU)  → +residual
//!   → final LayerNorm
//!   → [seq_out, d_model]
//! ```
//!
//! The sinusoidal positional embedding follows the original Vaswani et al.
//! 2017 formulation:
//!
//! ```text
//! pe[pos, 2i]   = sin(pos / 10000^(2i / d_model))
//! pe[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
//! ```
//!
//! References:
//! - Radford et al. 2023, "Robust Speech Recognition via Large-Scale Weak Supervision"
//! - Vaswani et al. 2017, "Attention Is All You Need"

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for [`WhisperEncoder`].
#[derive(Debug, Clone)]
pub struct WhisperEncoderConfig {
    /// Number of mel filterbank channels (input feature dimension).
    pub n_mels: usize,
    /// Transformer model dimension `D`.
    pub d_model: usize,
    /// Number of attention heads (must divide `d_model`).
    pub n_heads: usize,
    /// Number of stacked transformer encoder blocks.
    pub n_layers: usize,
    /// Hidden dimension of the FFN sub-layer.
    pub ffn_dim: usize,
    /// Maximum sequence length supported by the cached positional table.
    pub max_seq_len: usize,
}

impl WhisperEncoderConfig {
    /// Tiny configuration suitable for unit tests:
    /// `n_mels=8, d_model=16, n_heads=2, n_layers=2, ffn_dim=32, max_seq_len=64`.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            n_mels: 8,
            d_model: 16,
            n_heads: 2,
            n_layers: 2,
            ffn_dim: 32,
            max_seq_len: 64,
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns the appropriate [`AudioError`] for invalid fields.
    pub fn validate(&self) -> AudioResult<()> {
        if self.n_mels == 0 {
            return Err(AudioError::InvalidNumMels(0));
        }
        if self.d_model == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if self.n_heads == 0 {
            return Err(AudioError::InvalidNumHeads(0));
        }
        if self.d_model % self.n_heads != 0 {
            return Err(AudioError::HeadDimMismatch {
                embed_dim: self.d_model,
                n_heads: self.n_heads,
            });
        }
        if self.n_layers == 0 {
            return Err(AudioError::EmptyInput {
                msg: "WhisperEncoderConfig: n_layers must be ≥ 1".into(),
            });
        }
        if self.ffn_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if self.max_seq_len == 0 {
            return Err(AudioError::InvalidSequenceLength(0));
        }
        Ok(())
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Tanh-approximation GELU activation
/// (matches the convention used by the rest of the crate).
#[inline]
fn gelu_approx(x: f32) -> f32 {
    let inner = 0.797_884_6_f32 * (x + 0.044_715_f32 * x * x * x);
    0.5_f32 * x * (1.0_f32 + inner.tanh())
}

/// Layer normalisation over the last dimension of a `[T, D]` flat buffer.
fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let d = weight.len();
    let t = x.len().checked_div(d).unwrap_or(0);
    let mut out = vec![0.0_f32; x.len()];
    for ti in 0..t {
        let row = &x[ti * d..(ti + 1) * d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv_std = 1.0_f32 / (var + eps).sqrt();
        for (di, (&xv, (&w, &b))) in row.iter().zip(weight.iter().zip(bias.iter())).enumerate() {
            out[ti * d + di] = (xv - mean) * inv_std * w + b;
        }
    }
    out
}

/// Numerically stable in-place softmax over a contiguous slice.
fn softmax_inplace(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in scores.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0_f32 / sum;
        for v in scores.iter_mut() {
            *v *= inv;
        }
    }
}

/// Dense matrix multiply `C = A @ B` (sizes `[m, k] × [k, n] → [m, n]`,
/// row-major).
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// Transpose a flat `[rows, cols]` matrix to `[cols, rows]`.
fn transpose_2d(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

/// Apply a linear layer `y = x @ wᵀ + b` for `n` tokens.
///
/// `x` — `[n, in_d]`, `w` — `[out_d, in_d]`, `b` — `[out_d]`.
/// Returns `[n, out_d]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], in_d: usize, out_d: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * out_d];
    for tok in 0..n {
        for od in 0..out_d {
            let mut acc = b[od];
            let x_row = &x[tok * in_d..(tok + 1) * in_d];
            let w_row = &w[od * in_d..(od + 1) * in_d];
            for (xv, wv) in x_row.iter().zip(w_row.iter()) {
                acc += xv * wv;
            }
            out[tok * out_d + od] = acc;
        }
    }
    out
}

/// Xavier-uniform initialisation: limit = sqrt(6 / (fan_in + fan_out)).
#[inline]
fn xavier_limit(fan_in: usize, fan_out: usize) -> f32 {
    (6.0_f32 / (fan_in + fan_out) as f32).sqrt()
}

/// Fill `buf` with Xavier-uniform values in `[-limit, +limit]`.
fn fill_xavier(buf: &mut [f32], limit: f32, rng: &mut LcgRng) {
    for v in buf.iter_mut() {
        *v = (rng.next_f32() * 2.0_f32 - 1.0_f32) * limit;
    }
}

/// 1-D convolution with same padding (stride 1) or stride-2 same-pad,
/// kernel size 3.
///
/// `input`   — `[in_channels, in_len]` flat row-major (channels-first).
/// `weight`  — `[out_channels, in_channels, 3]` flat row-major.
/// `bias`    — `[out_channels]`.
///
/// With `padding = 1`, `kernel = 3`, `stride = s`:
/// `out_len = floor((in_len + 2 − 3) / s + 1) = floor((in_len − 1) / s) + 1`.
///
/// For `s = 1` this is `in_len`; for `s = 2` this is `ceil(in_len / 2)`.
fn conv1d_pad1_k3(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_channels: usize,
    in_len: usize,
    out_channels: usize,
    stride: usize,
) -> Vec<f32> {
    let kernel = 3_usize;
    let out_len = (in_len + 2 - kernel) / stride + 1;
    let mut out = vec![0.0_f32; out_channels * out_len];
    for oc in 0..out_channels {
        let b = bias[oc];
        for t_out in 0..out_len {
            let mut acc = b;
            // The centre of the kernel sits at position `t_out * stride`
            // in the unpadded input. With padding=1 the kernel taps are at
            // `t_in - 1, t_in, t_in + 1`.
            let t_centre = (t_out * stride) as isize;
            for ic in 0..in_channels {
                let w_off = (oc * in_channels + ic) * kernel;
                let in_off = ic * in_len;
                for k in 0..kernel {
                    let src = t_centre + (k as isize) - 1;
                    if src >= 0 && (src as usize) < in_len {
                        acc += weight[w_off + k] * input[in_off + src as usize];
                    }
                    // else: zero-padding contribution.
                }
            }
            out[oc * out_len + t_out] = acc;
        }
    }
    out
}

// ─── Public weight containers ────────────────────────────────────────────────

/// Conv-stem weight container (two stacked Conv1d layers).
#[derive(Debug, Clone)]
pub struct ConvStemWeights {
    /// First conv kernel `[d_model, n_mels, 3]`.
    pub w1: Vec<f32>,
    /// First conv bias `[d_model]`.
    pub b1: Vec<f32>,
    /// Second conv kernel `[d_model, d_model, 3]`.
    pub w2: Vec<f32>,
    /// Second conv bias `[d_model]`.
    pub b2: Vec<f32>,
}

/// Multi-head self-attention weights (pre-norm style, no relative bias).
#[derive(Debug, Clone)]
pub struct WhisperAttentionWeights {
    /// LN scale `[d_model]`.
    pub ln_weight: Vec<f32>,
    /// LN bias `[d_model]`.
    pub ln_bias: Vec<f32>,
    /// Query projection `[d_model, d_model]`.
    pub q_proj: Vec<f32>,
    /// Key projection `[d_model, d_model]`.
    pub k_proj: Vec<f32>,
    /// Value projection `[d_model, d_model]`.
    pub v_proj: Vec<f32>,
    /// Output projection `[d_model, d_model]`.
    pub out_proj: Vec<f32>,
    /// Query bias `[d_model]`.
    pub q_bias: Vec<f32>,
    /// Key bias `[d_model]` (Whisper uses no bias on keys; we still allocate
    /// a zero vector for layout uniformity).
    pub k_bias: Vec<f32>,
    /// Value bias `[d_model]`.
    pub v_bias: Vec<f32>,
    /// Output bias `[d_model]`.
    pub out_bias: Vec<f32>,
}

/// Feed-forward sub-layer weights (pre-norm style).
#[derive(Debug, Clone)]
pub struct WhisperFfnWeights {
    /// LN scale `[d_model]`.
    pub ln_weight: Vec<f32>,
    /// LN bias `[d_model]`.
    pub ln_bias: Vec<f32>,
    /// First linear weight `[ffn_dim, d_model]`.
    pub w1: Vec<f32>,
    /// First linear bias `[ffn_dim]`.
    pub b1: Vec<f32>,
    /// Second linear weight `[d_model, ffn_dim]`.
    pub w2: Vec<f32>,
    /// Second linear bias `[d_model]`.
    pub b2: Vec<f32>,
}

/// A single pre-norm transformer encoder block (attention + FFN sub-layers).
#[derive(Debug, Clone)]
pub struct WhisperBlock {
    /// Multi-head self-attention sub-layer.
    pub attn: WhisperAttentionWeights,
    /// Feed-forward sub-layer.
    pub ffn: WhisperFfnWeights,
}

// ─── Public encoder ──────────────────────────────────────────────────────────

/// Whisper-style speech encoder.
#[derive(Debug, Clone)]
pub struct WhisperEncoder {
    /// Construction configuration.
    pub cfg: WhisperEncoderConfig,
    /// Conv-stem weights.
    pub conv: ConvStemWeights,
    /// Cached sinusoidal positional embedding `[max_seq_len, d_model]`.
    pos_embed: Vec<f32>,
    /// Stacked transformer encoder blocks (length = `n_layers`).
    pub blocks: Vec<WhisperBlock>,
    /// Final layer-norm scale `[d_model]`.
    pub final_ln_weight: Vec<f32>,
    /// Final layer-norm bias `[d_model]`.
    pub final_ln_bias: Vec<f32>,
}

impl WhisperEncoder {
    /// Build a fresh `WhisperEncoder` with randomly-initialised weights.
    ///
    /// # Errors
    ///
    /// Propagates [`WhisperEncoderConfig::validate`] errors.
    pub fn new(cfg: WhisperEncoderConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        cfg.validate()?;

        let d = cfg.d_model;
        let n_mels = cfg.n_mels;
        let ffn_dim = cfg.ffn_dim;
        let kernel = 3_usize;

        // ── Conv-stem weights (Xavier uniform) ───────────────────────────────
        let lim1 = xavier_limit(n_mels * kernel, d * kernel);
        let lim2 = xavier_limit(d * kernel, d * kernel);
        let mut w1 = vec![0.0_f32; d * n_mels * kernel];
        fill_xavier(&mut w1, lim1, rng);
        let mut b1 = vec![0.0_f32; d];
        fill_xavier(&mut b1, lim1, rng);
        let mut w2 = vec![0.0_f32; d * d * kernel];
        fill_xavier(&mut w2, lim2, rng);
        let mut b2 = vec![0.0_f32; d];
        fill_xavier(&mut b2, lim2, rng);

        let conv = ConvStemWeights { w1, b1, w2, b2 };

        // ── Sinusoidal positional embedding ──────────────────────────────────
        let pos_embed = Self::build_sinusoidal_table(cfg.max_seq_len, d);

        // ── Transformer blocks ───────────────────────────────────────────────
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            blocks.push(init_block(d, ffn_dim, rng));
        }

        let final_ln_weight = vec![1.0_f32; d];
        let final_ln_bias = vec![0.0_f32; d];

        Ok(Self {
            cfg,
            conv,
            pos_embed,
            blocks,
            final_ln_weight,
            final_ln_bias,
        })
    }

    /// Length of the encoder output sequence after the strided conv stem:
    /// `ceil(seq_in / 2)` (with padding=1, kernel=3, stride=2).
    #[must_use]
    pub fn output_seq_len(&self, seq_in: usize) -> usize {
        seq_in.div_ceil(2)
    }

    /// Return the cached sinusoidal positional embedding as a flat row-major
    /// `[max_seq_len, d_model]` vector.
    #[must_use]
    pub fn sinusoidal_position_embedding(&self) -> Vec<f32> {
        self.pos_embed.clone()
    }

    /// Build a fresh sinusoidal positional table of shape `[max_seq_len, d_model]`.
    ///
    /// Formula:
    /// ```text
    /// pe[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    /// pe[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
    /// ```
    #[must_use]
    pub fn build_sinusoidal_table(max_seq_len: usize, d_model: usize) -> Vec<f32> {
        let mut pe = vec![0.0_f32; max_seq_len * d_model];
        if d_model == 0 || max_seq_len == 0 {
            return pe;
        }
        let half = d_model / 2;
        // Precompute inverse frequencies for each (2i) index.
        let mut inv_freq = vec![0.0_f32; half];
        for (i, freq) in inv_freq.iter_mut().enumerate() {
            let exp = (2_usize * i) as f32 / d_model as f32;
            *freq = 1.0_f32 / 10_000.0_f32.powf(exp);
        }
        for pos in 0..max_seq_len {
            let pos_f = pos as f32;
            for (i, &w) in inv_freq.iter().enumerate() {
                let theta = pos_f * w;
                pe[pos * d_model + 2 * i] = theta.sin();
                let cos_idx = 2 * i + 1;
                if cos_idx < d_model {
                    pe[pos * d_model + cos_idx] = theta.cos();
                }
            }
        }
        pe
    }

    /// Run the convolutional stem.
    ///
    /// `mel` — `[n_mels, seq_in]` flat row-major (channels-first).
    ///
    /// Returns `[d_model, seq_out]` flat row-major, where
    /// `seq_out = ceil(seq_in / 2)`.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] when `seq_in == 0`.
    /// - [`AudioError::ShapeMismatch`] when `mel.len() != n_mels * seq_in`.
    pub fn conv_stem(&self, mel: &[f32], seq_in: usize) -> AudioResult<Vec<f32>> {
        if seq_in == 0 {
            return Err(AudioError::EmptyInput {
                msg: "WhisperEncoder::conv_stem: seq_in == 0".into(),
            });
        }
        let n_mels = self.cfg.n_mels;
        let d = self.cfg.d_model;
        if mel.len() != n_mels * seq_in {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "WhisperEncoder::conv_stem: mel.len()={} != n_mels*seq_in={}",
                    mel.len(),
                    n_mels * seq_in
                ),
            });
        }

        // ── Conv1: n_mels → d_model, k=3, stride=1, pad=1 ────────────────────
        let mut h1 = conv1d_pad1_k3(mel, &self.conv.w1, &self.conv.b1, n_mels, seq_in, d, 1);
        for v in h1.iter_mut() {
            *v = gelu_approx(*v);
        }

        // ── Conv2: d_model → d_model, k=3, stride=2, pad=1 ───────────────────
        let mut h2 = conv1d_pad1_k3(&h1, &self.conv.w2, &self.conv.b2, d, seq_in, d, 2);
        for v in h2.iter_mut() {
            *v = gelu_approx(*v);
        }

        Ok(h2)
    }

    /// Full encoder forward pass.
    ///
    /// `mel` — `[n_mels, seq_in]` row-major.
    /// Returns `[seq_out, d_model]` row-major where `seq_out = ceil(seq_in / 2)`.
    ///
    /// # Errors
    ///
    /// - Propagates [`Self::conv_stem`] errors.
    /// - [`AudioError::InvalidSequenceLength`] when `seq_out > max_seq_len`.
    pub fn forward(&self, mel: &[f32], seq_in: usize) -> AudioResult<Vec<f32>> {
        let d = self.cfg.d_model;
        let seq_out = self.output_seq_len(seq_in);
        if seq_out > self.cfg.max_seq_len {
            return Err(AudioError::InvalidSequenceLength(seq_out));
        }

        // 1. Conv stem  → [d_model, seq_out]
        let h_cf = self.conv_stem(mel, seq_in)?;

        // 2. Transpose to [seq_out, d_model] (token-major).
        let mut h = transpose_2d(&h_cf, d, seq_out);

        // 3. Add sinusoidal positional embedding.
        for tok in 0..seq_out {
            let dst_off = tok * d;
            let src_off = tok * d;
            for di in 0..d {
                h[dst_off + di] += self.pos_embed[src_off + di];
            }
        }

        // 4. Stacked pre-norm transformer encoder blocks.
        let n_heads = self.cfg.n_heads;
        for block in &self.blocks {
            // Attention sub-layer (pre-norm + residual).
            let attn_out = whisper_attention(&h, seq_out, d, n_heads, &block.attn);
            for (h_v, &dv) in h.iter_mut().zip(attn_out.iter()) {
                *h_v += dv;
            }
            // FFN sub-layer (pre-norm + residual).
            let ffn_out = whisper_ffn(&h, seq_out, d, &block.ffn);
            for (h_v, &dv) in h.iter_mut().zip(ffn_out.iter()) {
                *h_v += dv;
            }
        }

        // 5. Final layer norm.
        let out = layer_norm(&h, &self.final_ln_weight, &self.final_ln_bias, 1e-5);
        Ok(out)
    }
}

// ─── Block / sub-layer constructors and forward passes ───────────────────────

fn init_block(d: usize, ffn_dim: usize, rng: &mut LcgRng) -> WhisperBlock {
    // Attention.
    let proj_len = d * d;
    let lim_proj = xavier_limit(d, d);

    let mut q_proj = vec![0.0_f32; proj_len];
    fill_xavier(&mut q_proj, lim_proj, rng);
    let mut k_proj = vec![0.0_f32; proj_len];
    fill_xavier(&mut k_proj, lim_proj, rng);
    let mut v_proj = vec![0.0_f32; proj_len];
    fill_xavier(&mut v_proj, lim_proj, rng);
    let mut out_proj = vec![0.0_f32; proj_len];
    fill_xavier(&mut out_proj, lim_proj, rng);

    let mut q_bias = vec![0.0_f32; d];
    fill_xavier(&mut q_bias, lim_proj, rng);
    let k_bias = vec![0.0_f32; d]; // Whisper: no bias on K (kept as zeros).
    let mut v_bias = vec![0.0_f32; d];
    fill_xavier(&mut v_bias, lim_proj, rng);
    let mut out_bias = vec![0.0_f32; d];
    fill_xavier(&mut out_bias, lim_proj, rng);

    let attn = WhisperAttentionWeights {
        ln_weight: vec![1.0_f32; d],
        ln_bias: vec![0.0_f32; d],
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        q_bias,
        k_bias,
        v_bias,
        out_bias,
    };

    // FFN.
    let lim1 = xavier_limit(d, ffn_dim);
    let lim2 = xavier_limit(ffn_dim, d);
    let mut w1 = vec![0.0_f32; ffn_dim * d];
    fill_xavier(&mut w1, lim1, rng);
    let mut b1 = vec![0.0_f32; ffn_dim];
    fill_xavier(&mut b1, lim1, rng);
    let mut w2 = vec![0.0_f32; d * ffn_dim];
    fill_xavier(&mut w2, lim2, rng);
    let mut b2 = vec![0.0_f32; d];
    fill_xavier(&mut b2, lim2, rng);

    let ffn = WhisperFfnWeights {
        ln_weight: vec![1.0_f32; d],
        ln_bias: vec![0.0_f32; d],
        w1,
        b1,
        w2,
        b2,
    };

    WhisperBlock { attn, ffn }
}

/// Pre-norm multi-head self-attention.
///
/// `x` — `[t, d]` row-major.
/// Returns `[t, d]` row-major (the *residual delta*, not yet added to `x`).
fn whisper_attention(
    x: &[f32],
    t: usize,
    d: usize,
    n_heads: usize,
    w: &WhisperAttentionWeights,
) -> Vec<f32> {
    // 1. Pre-norm.
    let normed = layer_norm(x, &w.ln_weight, &w.ln_bias, 1e-5);

    // 2. Project to Q, K, V — each [t, d].
    let q = linear(&normed, &w.q_proj, &w.q_bias, d, d, t);
    let k = linear(&normed, &w.k_proj, &w.k_bias, d, d, t);
    let v = linear(&normed, &w.v_proj, &w.v_bias, d, d, t);

    // 3. Per-head attention.
    let head_dim = d / n_heads;
    let inv_sqrt_hd = 1.0_f32 / (head_dim as f32).sqrt();

    let mut ctx_concat = vec![0.0_f32; t * d];

    for h in 0..n_heads {
        let h_off = h * head_dim;
        // Extract Q_h, K_h, V_h slices.
        let mut q_h = vec![0.0_f32; t * head_dim];
        let mut k_h = vec![0.0_f32; t * head_dim];
        let mut v_h = vec![0.0_f32; t * head_dim];
        for tok in 0..t {
            let src = tok * d + h_off;
            let dst = tok * head_dim;
            q_h[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
            k_h[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
            v_h[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
        }

        // K_hᵀ: [head_dim, t].
        let k_h_t = transpose_2d(&k_h, t, head_dim);
        // scores [t, t] = Q_h @ K_hᵀ
        let mut scores = matmul(&q_h, &k_h_t, t, head_dim, t);
        for s in scores.iter_mut() {
            *s *= inv_sqrt_hd;
        }
        // Softmax per query row.
        for qi in 0..t {
            softmax_inplace(&mut scores[qi * t..(qi + 1) * t]);
        }
        // ctx_h [t, head_dim] = scores @ V_h
        let ctx_h = matmul(&scores, &v_h, t, t, head_dim);
        // Write back into ctx_concat.
        for tok in 0..t {
            let dst = tok * d + h_off;
            let src = tok * head_dim;
            ctx_concat[dst..dst + head_dim].copy_from_slice(&ctx_h[src..src + head_dim]);
        }
    }

    // 4. Output projection: out = ctx_concat @ out_projᵀ + out_bias.
    linear(&ctx_concat, &w.out_proj, &w.out_bias, d, d, t)
}

/// Pre-norm feed-forward sub-layer.
///
/// `x` — `[t, d]` row-major.
/// Returns `[t, d]` row-major (the residual delta).
fn whisper_ffn(x: &[f32], t: usize, d: usize, w: &WhisperFfnWeights) -> Vec<f32> {
    let ffn_dim = w.w1.len() / d;
    let normed = layer_norm(x, &w.ln_weight, &w.ln_bias, 1e-5);

    // h = GELU(normed @ w1ᵀ + b1) — [t, ffn_dim].
    let mut h = linear(&normed, &w.w1, &w.b1, d, ffn_dim, t);
    for v in h.iter_mut() {
        *v = gelu_approx(*v);
    }

    // out = h @ w2ᵀ + b2 — [t, d].
    linear(&h, &w.w2, &w.b2, ffn_dim, d, t)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> WhisperEncoderConfig {
        WhisperEncoderConfig {
            n_mels: 8,
            d_model: 16,
            n_heads: 2,
            n_layers: 2,
            ffn_dim: 32,
            max_seq_len: 64,
        }
    }

    #[test]
    fn tiny_config_validates() {
        let cfg = WhisperEncoderConfig::tiny();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn build_tiny_ok() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(42);
        let enc = WhisperEncoder::new(cfg, &mut rng);
        assert!(enc.is_ok(), "WhisperEncoder::new failed: {enc:?}");
    }

    #[test]
    fn sinusoidal_table_length() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(1);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let pe = enc.sinusoidal_position_embedding();
        assert_eq!(pe.len(), cfg.max_seq_len * cfg.d_model);
    }

    #[test]
    fn sinusoidal_first_row_is_sin_t() {
        // pe[t, 0] = sin(t / 10000^0) = sin(t).
        let cfg = small_cfg();
        let mut rng = LcgRng::new(1);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let pe = enc.sinusoidal_position_embedding();
        for t in 0..cfg.max_seq_len.min(10) {
            let expected = (t as f32).sin();
            let got = pe[t * cfg.d_model];
            assert!(
                (expected - got).abs() < 1e-5,
                "pe[{t}, 0]={got}, sin({t})={expected}"
            );
        }
    }

    #[test]
    fn sinusoidal_second_col_is_cos_t() {
        // pe[t, 1] = cos(t).
        let cfg = small_cfg();
        let mut rng = LcgRng::new(2);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let pe = enc.sinusoidal_position_embedding();
        for t in 0..cfg.max_seq_len.min(5) {
            let expected = (t as f32).cos();
            let got = pe[t * cfg.d_model + 1];
            assert!(
                (expected - got).abs() < 1e-5,
                "pe[{t}, 1]={got}, cos({t})={expected}"
            );
        }
    }

    #[test]
    fn sinusoidal_zero_position_pattern() {
        // For pos=0: sin(0)=0, cos(0)=1.
        let cfg = small_cfg();
        let mut rng = LcgRng::new(3);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let pe = enc.sinusoidal_position_embedding();
        for (i, &v) in pe.iter().enumerate().take(cfg.d_model) {
            if i % 2 == 0 {
                assert!(v.abs() < 1e-6, "pe[0, {i}]={v} (expected sin(0)=0)");
            } else {
                assert!((v - 1.0).abs() < 1e-6, "pe[0, {i}]={v} (expected cos(0)=1)");
            }
        }
    }

    #[test]
    fn output_seq_len_ceil_div() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(4);
        let enc = WhisperEncoder::new(cfg, &mut rng).expect("ok");
        assert_eq!(enc.output_seq_len(8), 4);
        assert_eq!(enc.output_seq_len(9), 5);
        assert_eq!(enc.output_seq_len(1), 1);
        assert_eq!(enc.output_seq_len(2), 1);
        assert_eq!(enc.output_seq_len(3), 2);
        assert_eq!(enc.output_seq_len(10), 5);
        assert_eq!(enc.output_seq_len(11), 6);
    }

    #[test]
    fn conv_stem_output_length() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(5);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        for &seq_in in &[1_usize, 4, 7, 16, 33] {
            let mel = vec![0.1_f32; cfg.n_mels * seq_in];
            let out = enc.conv_stem(&mel, seq_in).expect("conv_stem");
            let expected = cfg.d_model * enc.output_seq_len(seq_in);
            assert_eq!(out.len(), expected, "seq_in={seq_in}");
        }
    }

    #[test]
    fn conv_stem_finite() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(6);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let seq_in = 12_usize;
        let mut mel = vec![0.0_f32; cfg.n_mels * seq_in];
        rng.fill_normal(&mut mel);
        let out = enc.conv_stem(&mel, seq_in).expect("conv_stem");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_output_length() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(7);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        for &seq_in in &[2_usize, 5, 10, 16] {
            let mel = vec![0.1_f32; cfg.n_mels * seq_in];
            let out = enc.forward(&mel, seq_in).expect("forward");
            let expected = enc.output_seq_len(seq_in) * cfg.d_model;
            assert_eq!(out.len(), expected, "seq_in={seq_in}");
        }
    }

    #[test]
    fn forward_finite() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(8);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let seq_in = 10_usize;
        let mut mel = vec![0.0_f32; cfg.n_mels * seq_in];
        rng.fill_normal(&mut mel);
        let out = enc.forward(&mel, seq_in).expect("forward");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite encoder output"
        );
    }

    #[test]
    fn forward_deterministic() {
        // Same seed → same encoder → same output for same input.
        let cfg = small_cfg();
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let enc_a = WhisperEncoder::new(cfg.clone(), &mut rng_a).expect("ok");
        let enc_b = WhisperEncoder::new(cfg.clone(), &mut rng_b).expect("ok");
        let seq_in = 8_usize;
        let mel: Vec<f32> = (0..cfg.n_mels * seq_in)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let out_a = enc_a.forward(&mel, seq_in).expect("a");
        let out_b = enc_b.forward(&mel, seq_in).expect("b");
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn forward_changes_with_input() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(11);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let seq_in = 6_usize;
        let mel_a: Vec<f32> = (0..cfg.n_mels * seq_in).map(|i| i as f32 * 0.01).collect();
        let mel_b: Vec<f32> = (0..cfg.n_mels * seq_in)
            .map(|i| (i as f32) * 0.01 + 1.0)
            .collect();
        let out_a = enc.forward(&mel_a, seq_in).expect("a");
        let out_b = enc.forward(&mel_b, seq_in).expect("b");
        let diff: f32 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-3,
            "encoder appears constant in input: diff={diff}"
        );
    }

    // ── Error paths ──────────────────────────────────────────────────────────

    #[test]
    fn err_d_model_not_divisible_by_n_heads() {
        let cfg = WhisperEncoderConfig {
            n_mels: 4,
            d_model: 17,
            n_heads: 4,
            n_layers: 1,
            ffn_dim: 16,
            max_seq_len: 16,
        };
        let mut rng = LcgRng::new(0);
        let r = WhisperEncoder::new(cfg, &mut rng);
        assert!(matches!(r.unwrap_err(), AudioError::HeadDimMismatch { .. }));
    }

    #[test]
    fn err_n_layers_zero() {
        let cfg = WhisperEncoderConfig {
            n_mels: 4,
            d_model: 8,
            n_heads: 2,
            n_layers: 0,
            ffn_dim: 16,
            max_seq_len: 16,
        };
        let mut rng = LcgRng::new(0);
        let r = WhisperEncoder::new(cfg, &mut rng);
        assert!(matches!(r.unwrap_err(), AudioError::EmptyInput { .. }));
    }

    #[test]
    fn err_n_mels_zero() {
        let cfg = WhisperEncoderConfig {
            n_mels: 0,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            ffn_dim: 16,
            max_seq_len: 16,
        };
        let mut rng = LcgRng::new(0);
        let r = WhisperEncoder::new(cfg, &mut rng);
        assert!(matches!(r.unwrap_err(), AudioError::InvalidNumMels(0)));
    }

    #[test]
    fn err_max_seq_len_too_small() {
        // max_seq_len=2, but seq_in=8 → seq_out=4 > 2 → expect error.
        let cfg = WhisperEncoderConfig {
            n_mels: 4,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            ffn_dim: 16,
            max_seq_len: 2,
        };
        let mut rng = LcgRng::new(1);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("build ok");
        let mel = vec![0.1_f32; cfg.n_mels * 8];
        let r = enc.forward(&mel, 8);
        assert!(matches!(
            r.unwrap_err(),
            AudioError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn err_mel_wrong_length() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(2);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let bad = vec![0.0_f32; cfg.n_mels * 5 + 1]; // wrong length
        let r = enc.forward(&bad, 5);
        assert!(matches!(r.unwrap_err(), AudioError::ShapeMismatch { .. }));
    }

    #[test]
    fn err_conv_stem_empty() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(3);
        let enc = WhisperEncoder::new(cfg, &mut rng).expect("ok");
        let r = enc.conv_stem(&[], 0);
        assert!(matches!(r.unwrap_err(), AudioError::EmptyInput { .. }));
    }

    // ── Config-variation tests ───────────────────────────────────────────────

    #[test]
    fn n_layers_one_works() {
        let cfg = WhisperEncoderConfig {
            n_mels: 4,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            ffn_dim: 16,
            max_seq_len: 16,
        };
        let mut rng = LcgRng::new(33);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        assert_eq!(enc.blocks.len(), 1);
        let seq_in = 8_usize;
        let mel = vec![0.1_f32; cfg.n_mels * seq_in];
        let out = enc.forward(&mel, seq_in).expect("forward");
        assert_eq!(out.len(), enc.output_seq_len(seq_in) * cfg.d_model);
    }

    #[test]
    fn single_mel_channel_works() {
        let cfg = WhisperEncoderConfig {
            n_mels: 1,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            ffn_dim: 16,
            max_seq_len: 16,
        };
        let mut rng = LcgRng::new(44);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let seq_in = 8_usize;
        let mel: Vec<f32> = (0..seq_in).map(|i| (i as f32).sin()).collect();
        let out = enc.forward(&mel, seq_in).expect("forward");
        assert_eq!(out.len(), enc.output_seq_len(seq_in) * cfg.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn build_sinusoidal_table_static() {
        // Standalone construction (no rng) should match the encoder's cached pe.
        let cfg = small_cfg();
        let mut rng = LcgRng::new(101);
        let enc = WhisperEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let direct = WhisperEncoder::build_sinusoidal_table(cfg.max_seq_len, cfg.d_model);
        let cached = enc.sinusoidal_position_embedding();
        assert_eq!(direct, cached);
    }
}

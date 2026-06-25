//! FastSpeech 2 acoustic-model core: the deterministic variance adaptor plus a
//! feed-forward Transformer (FFT) encoder / decoder that maps phoneme
//! embeddings to a mel spectrogram.
//!
//! # Architecture
//!
//! ```text
//! phonemes [T_phon, D]
//!   → FFT encoder  (depth × FftBlock)                         → [T_phon, D]
//!   → DurationPredictor                                       → log-durations [T_phon]
//!   → LengthRegulator (expand each frame by round(exp(log_d)))→ [T_mel, D]
//!   → pitch + energy VariancePredictors (on expanded seq),
//!     quantised → bins → embedded → ADDED to hidden states    → [T_mel, D]
//!   → FFT decoder  (depth × FftBlock)                         → [T_mel, D]
//!   → linear mel projection                                   → [T_mel, n_mels]
//! ```
//!
//! Each [`FftBlock`] is a feed-forward Transformer block: a full
//! (bidirectional, non-causal) multi-head self-attention sub-layer followed by
//! the FastSpeech position-wise FFN that uses **1-D convolutions** (kernel `9`,
//! `same` padding) instead of dense layers, with a ReLU between the two convs.
//! Both sub-layers use a **post-norm** residual layout
//! (`x = LayerNorm(x + Sublayer(x))`), matching the original FastSpeech /
//! FastSpeech 2 reference where layer normalisation follows the residual add.
//!
//! The model is fully deterministic for a fixed seed and input; the GAN
//! vocoder and the VITS normalising-flow stack are out of scope here (the
//! WaveNet / HiFi-GAN / Griffin-Lim vocoders live in [`crate::vocoder`]).
//!
//! # References
//!
//! - Y. Ren, C. Hu, X. Tan, T. Qin, S. Zhao, Z. Zhao, T.-Y. Liu,
//!   "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech",
//!   ICLR 2021. <https://arxiv.org/abs/2006.04558>
//! - Y. Ren, Y. Ruan, X. Tan, T. Qin, S. Zhao, Z. Zhao, T.-Y. Liu,
//!   "FastSpeech: Fast, Robust and Controllable Text to Speech",
//!   NeurIPS 2019. <https://arxiv.org/abs/1905.09263>

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Layer normalisation over the last dimension of a `[T * D]` flat buffer.
///
/// Each row (timestep) is normalised independently.
fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    let t = x.len().checked_div(d).unwrap_or(0);
    let mut out = vec![0.0_f32; x.len()];
    for ti in 0..t {
        let row = &x[ti * d..(ti + 1) * d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (di, (&xv, (&wv, &bv))) in row.iter().zip(w.iter().zip(b.iter())).enumerate() {
            out[ti * d + di] = (xv - mean) * inv_std * wv + bv;
        }
    }
    out
}

/// Rectified-linear activation.
#[inline]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// Dense matrix multiply: `C = A * B` where A is `[m, k]`, B is `[k, n]`.
///
/// Inputs and output are flat row-major.
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

/// Numerically stable in-place softmax over a contiguous slice.
fn softmax_inplace(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in scores.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in scores.iter_mut() {
            *v /= sum;
        }
    }
}

/// Transpose a `[rows, cols]` flat matrix to `[cols, rows]`.
fn transpose_2d(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

/// Apply a linear projection `y = x * wᵀ + b` where `w` is `[d_out, d_in]`.
///
/// `x` is `[t, d_in]`; the result is `[t, d_out]`.
fn linear(x: &[f32], t: usize, d_in: usize, d_out: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let w_t = transpose_2d(w, d_out, d_in);
    let mut out = matmul(x, &w_t, t, d_in, d_out);
    for ti in 0..t {
        for di in 0..d_out {
            out[ti * d_out + di] += b[di];
        }
    }
    out
}

/// Xavier uniform limit: `sqrt(6 / (fan_in + fan_out))`.
#[inline]
fn xavier_limit(fan_in: usize, fan_out: usize) -> f32 {
    (6.0 / (fan_in + fan_out) as f32).sqrt()
}

/// Allocate a weight matrix of `sz` elements drawn from `N(0, 1)` scaled by
/// `scale` (deterministically via `rng`).
fn make_normal_vec(sz: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut w = vec![0.0_f32; sz];
    rng.fill_normal(&mut w);
    for v in w.iter_mut() {
        *v *= scale;
    }
    w
}

/// 1-D convolution with `same` padding over a `[t, channels_in]` sequence,
/// producing `[t, channels_out]`.
///
/// `weight` is laid out `[channels_out, channels_in, kernel]` (row-major) and
/// `bias` is `[channels_out]`. The kernel is centred (`padding = kernel / 2`),
/// so an odd `kernel` keeps the time length unchanged. Out-of-range taps are
/// treated as zero (zero padding).
fn conv1d_same(
    x: &[f32],
    t: usize,
    channels_in: usize,
    channels_out: usize,
    kernel: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let pad = kernel / 2;
    let mut out = vec![0.0_f32; t * channels_out];
    for ti in 0..t {
        for co in 0..channels_out {
            let mut acc = bias[co];
            for kk in 0..kernel {
                // Input time index for this kernel tap (centred kernel).
                let src = ti as isize + kk as isize - pad as isize;
                if src < 0 || src >= t as isize {
                    continue;
                }
                let src = src as usize;
                let w_base = (co * channels_in) * kernel + kk;
                for ci in 0..channels_in {
                    acc += x[src * channels_in + ci] * weight[w_base + ci * kernel];
                }
            }
            out[ti * channels_out + co] = acc;
        }
    }
    out
}

// ─── Multi-head self-attention ─────────────────────────────────────────────────

/// Weights for a bidirectional multi-head self-attention sub-layer.
#[derive(Debug, Clone)]
pub struct SelfAttnWeights {
    /// Query projection `[embed_dim, embed_dim]`.
    pub w_q: Vec<f32>,
    /// Key projection `[embed_dim, embed_dim]`.
    pub w_k: Vec<f32>,
    /// Value projection `[embed_dim, embed_dim]`.
    pub w_v: Vec<f32>,
    /// Output projection `[embed_dim, embed_dim]`.
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

impl SelfAttnWeights {
    /// Initialise attention weights with Xavier-uniform-scaled normal draws.
    fn new(embed_dim: usize, rng: &mut LcgRng) -> Self {
        let scale = xavier_limit(embed_dim, embed_dim) / 1.732_050_8; // ≈ unit-variance scale
        let proj = embed_dim * embed_dim;
        Self {
            w_q: make_normal_vec(proj, scale, rng),
            w_k: make_normal_vec(proj, scale, rng),
            w_v: make_normal_vec(proj, scale, rng),
            w_o: make_normal_vec(proj, scale, rng),
            b_q: vec![0.0_f32; embed_dim],
            b_k: vec![0.0_f32; embed_dim],
            b_v: vec![0.0_f32; embed_dim],
            b_o: vec![0.0_f32; embed_dim],
        }
    }
}

/// Full (non-causal) scaled-dot-product multi-head self-attention.
///
/// `x` is `[t, embed_dim]`; the result is `[t, embed_dim]` (pre-residual).
fn self_attention(
    x: &[f32],
    t: usize,
    embed_dim: usize,
    n_heads: usize,
    w: &SelfAttnWeights,
) -> Vec<f32> {
    let head_dim = embed_dim / n_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = linear(x, t, embed_dim, embed_dim, &w.w_q, &w.b_q);
    let k = linear(x, t, embed_dim, embed_dim, &w.w_k, &w.b_k);
    let v = linear(x, t, embed_dim, embed_dim, &w.w_v, &w.b_v);

    let mut ctx = vec![0.0_f32; t * embed_dim];

    for h in 0..n_heads {
        let h_off = h * head_dim;

        // Per-head slices [t, head_dim].
        let mut q_h = vec![0.0_f32; t * head_dim];
        let mut k_h = vec![0.0_f32; t * head_dim];
        let mut v_h = vec![0.0_f32; t * head_dim];
        for ti in 0..t {
            let dst = ti * head_dim;
            let src = ti * embed_dim + h_off;
            q_h[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
            k_h[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
            v_h[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
        }

        // Scores [t, t] = Q_h · K_hᵀ · scale (full, no masking).
        let k_h_t = transpose_2d(&k_h, t, head_dim);
        let mut scores = matmul(&q_h, &k_h_t, t, head_dim, t);
        for s in scores.iter_mut() {
            *s *= scale;
        }
        for qi in 0..t {
            softmax_inplace(&mut scores[qi * t..(qi + 1) * t]);
        }

        // Context [t, head_dim] = scores · V_h.
        let ctx_h = matmul(&scores, &v_h, t, t, head_dim);
        for ti in 0..t {
            let dst = ti * embed_dim + h_off;
            ctx[dst..dst + head_dim].copy_from_slice(&ctx_h[ti * head_dim..(ti + 1) * head_dim]);
        }
    }

    linear(&ctx, t, embed_dim, embed_dim, &w.w_o, &w.b_o)
}

// ─── Conv FFN ──────────────────────────────────────────────────────────────────

/// FastSpeech convolutional position-wise feed-forward network.
///
/// Two 1-D convolutions (`same` padding) with a ReLU between them:
/// `conv1 → ReLU → conv2`. The first conv expands `embed_dim → conv_dim`, the
/// second contracts `conv_dim → embed_dim`.
#[derive(Debug, Clone)]
pub struct ConvFfnWeights {
    /// First conv weight `[conv_dim, embed_dim, kernel]`.
    pub conv1_weight: Vec<f32>,
    /// First conv bias `[conv_dim]`.
    pub conv1_bias: Vec<f32>,
    /// Second conv weight `[embed_dim, conv_dim, kernel]`.
    pub conv2_weight: Vec<f32>,
    /// Second conv bias `[embed_dim]`.
    pub conv2_bias: Vec<f32>,
    /// Convolution kernel size (odd).
    pub kernel: usize,
    /// Hidden convolution channel count.
    pub conv_dim: usize,
}

impl ConvFfnWeights {
    /// Initialise a conv-FFN with deterministic small-normal weights.
    fn new(embed_dim: usize, conv_dim: usize, kernel: usize, rng: &mut LcgRng) -> Self {
        let s1 = 1.0 / ((embed_dim * kernel) as f32).sqrt();
        let s2 = 1.0 / ((conv_dim * kernel) as f32).sqrt();
        Self {
            conv1_weight: make_normal_vec(conv_dim * embed_dim * kernel, s1, rng),
            conv1_bias: vec![0.0_f32; conv_dim],
            conv2_weight: make_normal_vec(embed_dim * conv_dim * kernel, s2, rng),
            conv2_bias: vec![0.0_f32; embed_dim],
            kernel,
            conv_dim,
        }
    }

    /// Run the conv FFN on `x` of `[t, embed_dim]`, returning `[t, embed_dim]`.
    fn forward(&self, x: &[f32], t: usize, embed_dim: usize) -> Vec<f32> {
        let mut h = conv1d_same(
            x,
            t,
            embed_dim,
            self.conv_dim,
            self.kernel,
            &self.conv1_weight,
            &self.conv1_bias,
        );
        for v in h.iter_mut() {
            *v = relu(*v);
        }
        conv1d_same(
            &h,
            t,
            self.conv_dim,
            embed_dim,
            self.kernel,
            &self.conv2_weight,
            &self.conv2_bias,
        )
    }
}

// ─── FFT block ─────────────────────────────────────────────────────────────────

/// A single feed-forward Transformer (FFT) block.
///
/// Layout (post-norm):
///
/// ```text
/// x → LayerNorm(x + SelfAttention(x))
///   → LayerNorm(x + ConvFFN(x))
/// ```
#[derive(Debug, Clone)]
pub struct FftBlock {
    /// Self-attention sub-layer weights.
    pub attn: SelfAttnWeights,
    /// Layer-norm scale after attention `[embed_dim]`.
    pub attn_ln_weight: Vec<f32>,
    /// Layer-norm bias after attention `[embed_dim]`.
    pub attn_ln_bias: Vec<f32>,
    /// Convolutional FFN sub-layer weights.
    pub ffn: ConvFfnWeights,
    /// Layer-norm scale after FFN `[embed_dim]`.
    pub ffn_ln_weight: Vec<f32>,
    /// Layer-norm bias after FFN `[embed_dim]`.
    pub ffn_ln_bias: Vec<f32>,
    /// Model (embedding) dimension.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
}

impl FftBlock {
    /// Construct an `FftBlock` with deterministic Xavier/fan-in initialisation.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidEmbedDim`], [`AudioError::InvalidNumHeads`],
    /// [`AudioError::HeadDimMismatch`] or [`AudioError::InvalidKernelSize`] when
    /// the supplied dimensions are inconsistent.
    pub fn new(
        embed_dim: usize,
        n_heads: usize,
        conv_dim: usize,
        kernel: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        if embed_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if n_heads == 0 {
            return Err(AudioError::InvalidNumHeads(0));
        }
        if embed_dim % n_heads != 0 {
            return Err(AudioError::HeadDimMismatch { embed_dim, n_heads });
        }
        if kernel == 0 || kernel % 2 == 0 {
            return Err(AudioError::InvalidKernelSize(kernel));
        }
        Ok(Self {
            attn: SelfAttnWeights::new(embed_dim, rng),
            attn_ln_weight: vec![1.0_f32; embed_dim],
            attn_ln_bias: vec![0.0_f32; embed_dim],
            ffn: ConvFfnWeights::new(embed_dim, conv_dim, kernel, rng),
            ffn_ln_weight: vec![1.0_f32; embed_dim],
            ffn_ln_bias: vec![0.0_f32; embed_dim],
            embed_dim,
            n_heads,
        })
    }

    /// Apply the FFT block to `x` of shape `[t, embed_dim]` (flat row-major).
    ///
    /// # Returns
    ///
    /// `[t, embed_dim]` flat output.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::EmptyInput`] when `t == 0`, or
    /// [`AudioError::ShapeMismatch`] when `x.len() != t * embed_dim`.
    pub fn forward(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        let d = self.embed_dim;
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "FftBlock: t == 0".into(),
            });
        }
        if x.len() != t * d {
            return Err(AudioError::ShapeMismatch {
                msg: format!("FftBlock::forward: x.len()={} != t*d={}", x.len(), t * d),
            });
        }

        // Attention sub-layer with post-norm residual.
        let attn_out = self_attention(x, t, d, self.n_heads, &self.attn);
        let mut h = vec![0.0_f32; t * d];
        for i in 0..t * d {
            h[i] = x[i] + attn_out[i];
        }
        let h = layer_norm(&h, &self.attn_ln_weight, &self.attn_ln_bias, 1e-5);

        // Conv-FFN sub-layer with post-norm residual.
        let ffn_out = self.ffn.forward(&h, t, d);
        let mut g = vec![0.0_f32; t * d];
        for i in 0..t * d {
            g[i] = h[i] + ffn_out[i];
        }
        let out = layer_norm(&g, &self.ffn_ln_weight, &self.ffn_ln_bias, 1e-5);
        Ok(out)
    }
}

// ─── Variance predictor (duration / pitch / energy) ────────────────────────────

/// A FastSpeech 2 variance predictor: two `Conv1d → ReLU → LayerNorm` stages
/// followed by a linear projection to a scalar per frame.
///
/// Used for the duration predictor (operating in the log domain) and for the
/// pitch / energy predictors (continuous value per frame).
#[derive(Debug, Clone)]
pub struct VariancePredictor {
    /// First conv weight `[hidden, in_dim, kernel]`.
    pub conv1_weight: Vec<f32>,
    /// First conv bias `[hidden]`.
    pub conv1_bias: Vec<f32>,
    /// Layer-norm scale after conv1 `[hidden]`.
    pub ln1_weight: Vec<f32>,
    /// Layer-norm bias after conv1 `[hidden]`.
    pub ln1_bias: Vec<f32>,
    /// Second conv weight `[hidden, hidden, kernel]`.
    pub conv2_weight: Vec<f32>,
    /// Second conv bias `[hidden]`.
    pub conv2_bias: Vec<f32>,
    /// Layer-norm scale after conv2 `[hidden]`.
    pub ln2_weight: Vec<f32>,
    /// Layer-norm bias after conv2 `[hidden]`.
    pub ln2_bias: Vec<f32>,
    /// Output projection weight `[1, hidden]`.
    pub proj_weight: Vec<f32>,
    /// Output projection bias `[1]`.
    pub proj_bias: Vec<f32>,
    /// Input feature dimension.
    pub in_dim: usize,
    /// Hidden convolution channel count.
    pub hidden: usize,
    /// Convolution kernel size (odd).
    pub kernel: usize,
}

impl VariancePredictor {
    /// Construct a variance predictor with deterministic small-normal init.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidEmbedDim`] when `in_dim == 0`,
    /// [`AudioError::InvalidKernelSize`] when `kernel` is `0` or even, or
    /// [`AudioError::Internal`] when `hidden == 0`.
    pub fn new(in_dim: usize, hidden: usize, kernel: usize, rng: &mut LcgRng) -> AudioResult<Self> {
        if in_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if hidden == 0 {
            return Err(AudioError::Internal(
                "VariancePredictor: hidden == 0".into(),
            ));
        }
        if kernel == 0 || kernel % 2 == 0 {
            return Err(AudioError::InvalidKernelSize(kernel));
        }
        let s1 = 1.0 / ((in_dim * kernel) as f32).sqrt();
        let s2 = 1.0 / ((hidden * kernel) as f32).sqrt();
        let sp = 1.0 / (hidden as f32).sqrt();
        Ok(Self {
            conv1_weight: make_normal_vec(hidden * in_dim * kernel, s1, rng),
            conv1_bias: vec![0.0_f32; hidden],
            ln1_weight: vec![1.0_f32; hidden],
            ln1_bias: vec![0.0_f32; hidden],
            conv2_weight: make_normal_vec(hidden * hidden * kernel, s2, rng),
            conv2_bias: vec![0.0_f32; hidden],
            ln2_weight: vec![1.0_f32; hidden],
            ln2_bias: vec![0.0_f32; hidden],
            proj_weight: make_normal_vec(hidden, sp, rng),
            proj_bias: vec![0.0_f32; 1],
            in_dim,
            hidden,
            kernel,
        })
    }

    /// Predict one scalar per frame for `x` of shape `[t, in_dim]`.
    ///
    /// The pipeline is `Conv1d → ReLU → LayerNorm` (twice) followed by a linear
    /// projection to a scalar. Dropout is identity at inference and therefore
    /// omitted.
    ///
    /// # Returns
    ///
    /// A `[t]` vector (one scalar per frame). For the duration predictor these
    /// scalars are interpreted in the **log** domain.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::EmptyInput`] when `t == 0`, or
    /// [`AudioError::ShapeMismatch`] when `x.len() != t * in_dim`.
    pub fn predict(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "VariancePredictor: t == 0".into(),
            });
        }
        if x.len() != t * self.in_dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "VariancePredictor::predict: x.len()={} != t*in_dim={}",
                    x.len(),
                    t * self.in_dim
                ),
            });
        }

        // Stage 1: conv → ReLU → layer norm.
        let mut h = conv1d_same(
            x,
            t,
            self.in_dim,
            self.hidden,
            self.kernel,
            &self.conv1_weight,
            &self.conv1_bias,
        );
        for v in h.iter_mut() {
            *v = relu(*v);
        }
        let h = layer_norm(&h, &self.ln1_weight, &self.ln1_bias, 1e-5);

        // Stage 2: conv → ReLU → layer norm.
        let mut g = conv1d_same(
            &h,
            t,
            self.hidden,
            self.hidden,
            self.kernel,
            &self.conv2_weight,
            &self.conv2_bias,
        );
        for v in g.iter_mut() {
            *v = relu(*v);
        }
        let g = layer_norm(&g, &self.ln2_weight, &self.ln2_bias, 1e-5);

        // Linear projection to a scalar per frame.
        let out = linear(&g, t, self.hidden, 1, &self.proj_weight, &self.proj_bias);
        Ok(out)
    }
}

// ─── Duration predictor ────────────────────────────────────────────────────────

/// Phoneme-level duration predictor.
///
/// Wraps a [`VariancePredictor`] and exposes the FastSpeech 2 inference
/// convention: the network predicts `log(duration)` and the integer frame count
/// is recovered with `round(exp(log_d)).max(0)` (then clamped to a configured
/// maximum). Predicting in the log domain keeps the regression target on a
/// compressed scale and guarantees non-negative durations after the exp.
#[derive(Debug, Clone)]
pub struct DurationPredictor {
    /// Underlying conv variance predictor (output = log-duration).
    pub predictor: VariancePredictor,
    /// Hard upper bound on a single phoneme's frame count after rounding.
    pub max_duration: usize,
}

impl DurationPredictor {
    /// Construct a duration predictor over an `embed_dim`-wide phoneme sequence.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`VariancePredictor::new`].
    pub fn new(
        embed_dim: usize,
        hidden: usize,
        kernel: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        Ok(Self {
            predictor: VariancePredictor::new(embed_dim, hidden, kernel, rng)?,
            max_duration: 1_000,
        })
    }

    /// Predict per-phoneme **log-durations** for `x` of `[t, embed_dim]`.
    ///
    /// # Returns
    ///
    /// A `[t]` vector of predicted `log(duration)` values.
    ///
    /// # Errors
    ///
    /// Propagates shape / empty-input errors from [`VariancePredictor::predict`].
    pub fn predict(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        self.predictor.predict(x, t)
    }

    /// Predict integer per-phoneme durations for `x` of `[t, embed_dim]`.
    ///
    /// Applies the FastSpeech 2 inference rule `round(exp(log_d))`, clamps the
    /// result to `[0, max_duration]`, and maps any non-finite prediction to `0`.
    ///
    /// # Returns
    ///
    /// A `[t]` vector of non-negative frame counts.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`DurationPredictor::predict`].
    pub fn predict_rounded(&self, x: &[f32], t: usize) -> AudioResult<Vec<usize>> {
        let log_d = self.predict(x, t)?;
        let mut out = Vec::with_capacity(t);
        for &ld in &log_d {
            let d = ld.exp();
            let rounded = if d.is_finite() {
                let r = d.round();
                if r <= 0.0 {
                    0
                } else if r >= self.max_duration as f32 {
                    self.max_duration
                } else {
                    r as usize
                }
            } else {
                0
            };
            out.push(rounded);
        }
        Ok(out)
    }
}

// ─── Length regulator ──────────────────────────────────────────────────────────

/// Expand a phoneme sequence by repeating each frame `durations[i]` times.
///
/// This is the core FastSpeech length-regulation operator. Given hidden states
/// `x` of shape `[t, dim]` and per-phoneme integer `durations` (`durations.len()
/// == t`), it produces `[sum(durations), dim]` where phoneme `i` is copied
/// `durations[i]` times consecutively along the time axis. Phonemes with
/// duration `0` are skipped entirely.
///
/// # Errors
///
/// - [`AudioError::EmptyInput`] when `t == 0`.
/// - [`AudioError::ShapeMismatch`] when `x.len() != t * dim` or
///   `durations.len() != t`.
/// - [`AudioError::InvalidSequenceLength`] when **every** duration is `0`
///   (the expanded sequence would be empty, which is not a valid mel target).
pub fn length_regulate(
    x: &[f32],
    t: usize,
    dim: usize,
    durations: &[usize],
) -> AudioResult<Vec<f32>> {
    if t == 0 {
        return Err(AudioError::EmptyInput {
            msg: "length_regulate: t == 0".into(),
        });
    }
    if dim == 0 {
        return Err(AudioError::InvalidEmbedDim(0));
    }
    if x.len() != t * dim {
        return Err(AudioError::ShapeMismatch {
            msg: format!("length_regulate: x.len()={} != t*dim={}", x.len(), t * dim),
        });
    }
    if durations.len() != t {
        return Err(AudioError::ShapeMismatch {
            msg: format!(
                "length_regulate: durations.len()={} != t={}",
                durations.len(),
                t
            ),
        });
    }

    let total: usize = durations.iter().sum();
    if total == 0 {
        return Err(AudioError::InvalidSequenceLength(0));
    }

    let mut out = vec![0.0_f32; total * dim];
    let mut frame = 0;
    for (i, &d) in durations.iter().enumerate() {
        let src = &x[i * dim..(i + 1) * dim];
        for _ in 0..d {
            out[frame * dim..(frame + 1) * dim].copy_from_slice(src);
            frame += 1;
        }
    }
    Ok(out)
}

/// Length-regulate while rescaling durations by `1 / pace`.
///
/// `pace > 1` yields a faster (shorter) utterance, `pace < 1` a slower (longer)
/// one. Each scaled duration is rounded with `round(d / pace)` and clamped to a
/// minimum of `1` for phonemes whose original duration was non-zero, so that no
/// audible phoneme is dropped purely through pace scaling. Phonemes with an
/// original duration of `0` stay at `0`.
///
/// # Errors
///
/// - [`AudioError::Internal`] when `pace` is not finite or `<= 0`.
/// - All errors that [`length_regulate`] can return (after rescaling), including
///   [`AudioError::InvalidSequenceLength`] if every duration became `0`.
pub fn length_regulate_with_pace(
    x: &[f32],
    t: usize,
    dim: usize,
    durations: &[usize],
    pace: f32,
) -> AudioResult<Vec<f32>> {
    if !pace.is_finite() || pace <= 0.0 {
        return Err(AudioError::Internal(format!(
            "length_regulate_with_pace: invalid pace {pace}"
        )));
    }
    if durations.len() != t {
        return Err(AudioError::ShapeMismatch {
            msg: format!(
                "length_regulate_with_pace: durations.len()={} != t={}",
                durations.len(),
                t
            ),
        });
    }
    let scaled = scale_durations(durations, pace);
    length_regulate(x, t, dim, &scaled)
}

/// Rescale integer durations by `1 / pace`, keeping non-zero phonemes audible.
///
/// A phoneme with original duration `> 0` is clamped to a minimum of `1` after
/// rounding; a phoneme with duration `0` stays `0`.
fn scale_durations(durations: &[usize], pace: f32) -> Vec<usize> {
    durations
        .iter()
        .map(|&d| {
            if d == 0 {
                0
            } else {
                let scaled = (d as f32 / pace).round();
                if scaled < 1.0 { 1 } else { scaled as usize }
            }
        })
        .collect()
}

// ─── Pitch / energy quantisation + embedding ──────────────────────────────────

/// Bucketise continuous `values` into bin indices given sorted `boundaries`.
///
/// Returns, for each value `v`, the count of boundaries strictly less than `v`
/// (i.e. the number of crossed thresholds). With `boundaries.len() == n - 1`
/// internal edges this yields indices in `[0, n - 1]`: a value below all
/// boundaries maps to bin `0`, a value above all boundaries maps to the last
/// bin, and the mapping is monotone non-decreasing in `v`. `boundaries` is
/// assumed sorted ascending.
#[must_use]
pub fn quantize_to_bins(values: &[f32], boundaries: &[f32]) -> Vec<usize> {
    values
        .iter()
        .map(|&v| {
            let mut bin = 0;
            for &edge in boundaries {
                if v >= edge {
                    bin += 1;
                } else {
                    break;
                }
            }
            bin
        })
        .collect()
}

/// Add a learned per-bin embedding row into each frame of a hidden sequence.
///
/// For each frame `f` the row `embedding[bin_ids[f] * dim .. +dim]` is added
/// element-wise into `hidden[f * dim .. +dim]`. This realises the FastSpeech 2
/// "predict → quantise → embed → add" pitch / energy injection on the expanded
/// (mel-length) sequence.
///
/// # Errors
///
/// - [`AudioError::ShapeMismatch`] when `hidden.len() != t * dim` or
///   `bin_ids.len() != t`.
/// - [`AudioError::WeightShapeMismatch`] when any bin index is out of range for
///   the supplied `embedding` table (`[n_bins, dim]`).
pub fn embed_and_add(
    hidden: &mut [f32],
    t: usize,
    dim: usize,
    bin_ids: &[usize],
    embedding: &[f32],
) -> AudioResult<()> {
    if hidden.len() != t * dim {
        return Err(AudioError::ShapeMismatch {
            msg: format!(
                "embed_and_add: hidden.len()={} != t*dim={}",
                hidden.len(),
                t * dim
            ),
        });
    }
    if bin_ids.len() != t {
        return Err(AudioError::ShapeMismatch {
            msg: format!("embed_and_add: bin_ids.len()={} != t={}", bin_ids.len(), t),
        });
    }
    if dim == 0 {
        return Err(AudioError::InvalidEmbedDim(0));
    }
    let n_bins = embedding.len() / dim;
    for (f, &bin) in bin_ids.iter().enumerate() {
        if bin >= n_bins {
            return Err(AudioError::WeightShapeMismatch {
                msg: format!("embed_and_add: bin {bin} >= n_bins {n_bins}"),
            });
        }
        let emb = &embedding[bin * dim..(bin + 1) * dim];
        let dst = &mut hidden[f * dim..(f + 1) * dim];
        for (h, &e) in dst.iter_mut().zip(emb.iter()) {
            *h += e;
        }
    }
    Ok(())
}

/// Build `n_bins - 1` evenly spaced internal boundaries over `[lo, hi]`.
fn linspace_boundaries(lo: f32, hi: f32, n_bins: usize) -> Vec<f32> {
    if n_bins <= 1 {
        return Vec::new();
    }
    let step = (hi - lo) / n_bins as f32;
    (1..n_bins).map(|i| lo + step * i as f32).collect()
}

// ─── FastSpeech 2 configuration ────────────────────────────────────────────────

/// Configuration for a [`FastSpeech2`] acoustic model.
#[derive(Debug, Clone)]
pub struct FastSpeech2Config {
    /// Model (phoneme-embedding / hidden) dimension `D`.
    pub embed_dim: usize,
    /// Number of self-attention heads (must divide `embed_dim`).
    pub n_heads: usize,
    /// Hidden channel count inside each FFT block's conv-FFN.
    pub conv_dim: usize,
    /// FFT-block conv-FFN kernel size (odd).
    pub ffn_kernel: usize,
    /// Number of FFT blocks in the encoder (and, separately, the decoder).
    pub depth: usize,
    /// Hidden channel count inside each variance predictor.
    pub var_hidden: usize,
    /// Variance-predictor conv kernel size (odd).
    pub var_kernel: usize,
    /// Number of pitch quantisation bins.
    pub pitch_bins: usize,
    /// Number of energy quantisation bins.
    pub energy_bins: usize,
    /// Number of output mel-spectrogram channels.
    pub n_mels: usize,
}

impl FastSpeech2Config {
    /// Tiny preset suitable for unit tests
    /// (`D=16, H=2, conv_dim=32, depth=2, n_mels=16`).
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            embed_dim: 16,
            n_heads: 2,
            conv_dim: 32,
            ffn_kernel: 9,
            depth: 2,
            var_hidden: 16,
            var_kernel: 3,
            pitch_bins: 8,
            energy_bins: 8,
            n_mels: 16,
        }
    }

    /// Construct and validate a configuration.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`FastSpeech2Config::validate`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embed_dim: usize,
        n_heads: usize,
        conv_dim: usize,
        ffn_kernel: usize,
        depth: usize,
        var_hidden: usize,
        var_kernel: usize,
        pitch_bins: usize,
        energy_bins: usize,
        n_mels: usize,
    ) -> AudioResult<Self> {
        let cfg = Self {
            embed_dim,
            n_heads,
            conv_dim,
            ffn_kernel,
            depth,
            var_hidden,
            var_kernel,
            pitch_bins,
            energy_bins,
            n_mels,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidEmbedDim`] when `embed_dim == 0`.
    /// - [`AudioError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`AudioError::HeadDimMismatch`] when `embed_dim % n_heads != 0`.
    /// - [`AudioError::InvalidKernelSize`] when `ffn_kernel` or `var_kernel` is
    ///   `0` or even.
    /// - [`AudioError::InvalidNumMels`] when `n_mels == 0`.
    /// - [`AudioError::Internal`] when `conv_dim`, `var_hidden`, `depth`,
    ///   `pitch_bins`, or `energy_bins` is `0`.
    pub fn validate(&self) -> AudioResult<()> {
        if self.embed_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if self.n_heads == 0 {
            return Err(AudioError::InvalidNumHeads(0));
        }
        if self.embed_dim % self.n_heads != 0 {
            return Err(AudioError::HeadDimMismatch {
                embed_dim: self.embed_dim,
                n_heads: self.n_heads,
            });
        }
        if self.ffn_kernel == 0 || self.ffn_kernel % 2 == 0 {
            return Err(AudioError::InvalidKernelSize(self.ffn_kernel));
        }
        if self.var_kernel == 0 || self.var_kernel % 2 == 0 {
            return Err(AudioError::InvalidKernelSize(self.var_kernel));
        }
        if self.n_mels == 0 {
            return Err(AudioError::InvalidNumMels(0));
        }
        if self.conv_dim == 0 {
            return Err(AudioError::Internal("conv_dim == 0".into()));
        }
        if self.var_hidden == 0 {
            return Err(AudioError::Internal("var_hidden == 0".into()));
        }
        if self.depth == 0 {
            return Err(AudioError::Internal("depth == 0".into()));
        }
        if self.pitch_bins == 0 {
            return Err(AudioError::Internal("pitch_bins == 0".into()));
        }
        if self.energy_bins == 0 {
            return Err(AudioError::Internal("energy_bins == 0".into()));
        }
        Ok(())
    }
}

// ─── FastSpeech 2 model ────────────────────────────────────────────────────────

/// FastSpeech 2 acoustic model: FFT encoder → variance adaptor → FFT decoder →
/// mel projection.
///
/// All parameters are initialised deterministically from a seeded [`LcgRng`].
pub struct FastSpeech2 {
    /// FFT encoder blocks over the phoneme sequence.
    pub encoder: Vec<FftBlock>,
    /// Phoneme-level duration predictor.
    pub duration: DurationPredictor,
    /// Frame-level pitch predictor (operates on the expanded sequence).
    pub pitch: VariancePredictor,
    /// Frame-level energy predictor (operates on the expanded sequence).
    pub energy: VariancePredictor,
    /// Pitch bin embedding table `[pitch_bins, embed_dim]`.
    pub pitch_embedding: Vec<f32>,
    /// Energy bin embedding table `[energy_bins, embed_dim]`.
    pub energy_embedding: Vec<f32>,
    /// Pitch quantisation boundaries `[pitch_bins - 1]`.
    pub pitch_boundaries: Vec<f32>,
    /// Energy quantisation boundaries `[energy_bins - 1]`.
    pub energy_boundaries: Vec<f32>,
    /// FFT decoder blocks over the expanded sequence.
    pub decoder: Vec<FftBlock>,
    /// Mel projection weight `[n_mels, embed_dim]`.
    pub mel_proj_weight: Vec<f32>,
    /// Mel projection bias `[n_mels]`.
    pub mel_proj_bias: Vec<f32>,
    /// Configuration this model was built from.
    pub config: FastSpeech2Config,
}

impl FastSpeech2 {
    /// Construct a FastSpeech 2 model with deterministic initialisation.
    ///
    /// # Errors
    ///
    /// Returns any error produced by [`FastSpeech2Config::validate`] or by the
    /// inner block / predictor constructors.
    pub fn new(config: FastSpeech2Config, rng: &mut LcgRng) -> AudioResult<Self> {
        config.validate()?;
        let d = config.embed_dim;

        let mut encoder = Vec::with_capacity(config.depth);
        for _ in 0..config.depth {
            encoder.push(FftBlock::new(
                d,
                config.n_heads,
                config.conv_dim,
                config.ffn_kernel,
                rng,
            )?);
        }

        let duration = DurationPredictor::new(d, config.var_hidden, config.var_kernel, rng)?;
        let pitch = VariancePredictor::new(d, config.var_hidden, config.var_kernel, rng)?;
        let energy = VariancePredictor::new(d, config.var_hidden, config.var_kernel, rng)?;

        let pitch_embedding = make_normal_vec(config.pitch_bins * d, 0.02, rng);
        let energy_embedding = make_normal_vec(config.energy_bins * d, 0.02, rng);

        // Default quantisation ranges (standardised pitch/energy in roughly
        // [-3, 3] σ for pitch and [0, 6] for energy magnitude).
        let pitch_boundaries = linspace_boundaries(-3.0, 3.0, config.pitch_bins);
        let energy_boundaries = linspace_boundaries(0.0, 6.0, config.energy_bins);

        let mut decoder = Vec::with_capacity(config.depth);
        for _ in 0..config.depth {
            decoder.push(FftBlock::new(
                d,
                config.n_heads,
                config.conv_dim,
                config.ffn_kernel,
                rng,
            )?);
        }

        let mel_proj_weight = make_normal_vec(config.n_mels * d, 1.0 / (d as f32).sqrt(), rng);
        let mel_proj_bias = vec![0.0_f32; config.n_mels];

        Ok(Self {
            encoder,
            duration,
            pitch,
            energy,
            pitch_embedding,
            energy_embedding,
            pitch_boundaries,
            energy_boundaries,
            decoder,
            mel_proj_weight,
            mel_proj_bias,
            config,
        })
    }

    /// Run the FFT encoder stack over `x` of `[t, embed_dim]`.
    fn encode(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        let mut h = x.to_vec();
        for block in &self.encoder {
            h = block.forward(&h, t)?;
        }
        Ok(h)
    }

    /// Run the FFT decoder stack and mel projection over the expanded sequence.
    ///
    /// `x` is `[t_mel, embed_dim]`; the result is `[t_mel, n_mels]`.
    fn decode(&self, x: &[f32], t_mel: usize) -> AudioResult<Vec<f32>> {
        let d = self.config.embed_dim;
        let mut h = x.to_vec();
        for block in &self.decoder {
            h = block.forward(&h, t_mel)?;
        }
        let mel = linear(
            &h,
            t_mel,
            d,
            self.config.n_mels,
            &self.mel_proj_weight,
            &self.mel_proj_bias,
        );
        Ok(mel)
    }

    /// Predict, quantise, embed and add pitch + energy on the expanded sequence.
    ///
    /// Mutates `hidden` (`[t_mel, embed_dim]`) in place by adding the pitch and
    /// energy bin embeddings derived from the model's own predictors.
    fn add_variance(&self, hidden: &mut [f32], t_mel: usize) -> AudioResult<()> {
        let d = self.config.embed_dim;

        let pitch_vals = self.pitch.predict(hidden, t_mel)?;
        let pitch_bins = quantize_to_bins(&pitch_vals, &self.pitch_boundaries);
        embed_and_add(hidden, t_mel, d, &pitch_bins, &self.pitch_embedding)?;

        let energy_vals = self.energy.predict(hidden, t_mel)?;
        let energy_bins = quantize_to_bins(&energy_vals, &self.energy_boundaries);
        embed_and_add(hidden, t_mel, d, &energy_bins, &self.energy_embedding)?;

        Ok(())
    }

    /// Validate that a phoneme buffer matches `[t, embed_dim]`.
    fn check_phon(&self, phon: &[f32], t: usize) -> AudioResult<()> {
        let d = self.config.embed_dim;
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "FastSpeech2: t == 0".into(),
            });
        }
        if phon.len() != t * d {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "FastSpeech2: phon.len()={} != t*embed_dim={}",
                    phon.len(),
                    t * d
                ),
            });
        }
        Ok(())
    }

    /// Training forward pass with **teacher-forced** ground-truth durations.
    ///
    /// `phon` is `[t, embed_dim]`; `gt_durations` are the ground-truth
    /// per-phoneme frame counts (`gt_durations.len() == t`). The phoneme
    /// sequence is encoded, length-regulated with the ground-truth durations,
    /// has its (predicted) pitch and energy added, decoded, and projected to a
    /// mel spectrogram of shape `[sum(gt_durations), n_mels]`.
    ///
    /// Pitch and energy are taken from the model's own predictors on the
    /// expanded sequence (the same path used at inference); ground-truth
    /// pitch / energy targets would be supplied separately by a training loop
    /// for the variance losses and are not required to produce the mel output.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] / [`AudioError::ShapeMismatch`] for bad
    ///   `phon` shape.
    /// - All errors from [`length_regulate`] (including
    ///   [`AudioError::InvalidSequenceLength`] when every ground-truth duration
    ///   is `0`).
    pub fn forward_train(
        &self,
        phon: &[f32],
        t: usize,
        gt_durations: &[usize],
    ) -> AudioResult<Vec<f32>> {
        self.check_phon(phon, t)?;
        let d = self.config.embed_dim;
        if gt_durations.len() != t {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "forward_train: gt_durations.len()={} != t={}",
                    gt_durations.len(),
                    t
                ),
            });
        }

        let enc = self.encode(phon, t)?;
        let mut expanded = length_regulate(&enc, t, d, gt_durations)?;
        let t_mel = expanded.len() / d;
        self.add_variance(&mut expanded, t_mel)?;
        self.decode(&expanded, t_mel)
    }

    /// Inference forward pass: predict durations, length-regulate, add pitch /
    /// energy, decode to a mel spectrogram.
    ///
    /// `phon` is `[t, embed_dim]`. Returns the mel spectrogram
    /// `[sum(pred_durations), n_mels]` together with the predicted per-phoneme
    /// durations (`pred_durations.len() == t`).
    ///
    /// When the duration predictor rounds **every** phoneme to `0` (degenerate
    /// at initialisation), each phoneme is floored to a single frame so a valid
    /// non-empty mel can still be produced.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] / [`AudioError::ShapeMismatch`] for bad
    ///   `phon` shape.
    /// - Propagates errors from the encoder, decoder, or variance adaptor.
    pub fn forward_infer(&self, phon: &[f32], t: usize) -> AudioResult<(Vec<f32>, Vec<usize>)> {
        self.check_phon(phon, t)?;
        let d = self.config.embed_dim;

        let enc = self.encode(phon, t)?;
        let mut durations = self.duration.predict_rounded(&enc, t)?;

        // Guarantee at least one output frame: if all durations rounded to 0,
        // floor every phoneme to a single frame.
        if durations.iter().all(|&d| d == 0) {
            for slot in durations.iter_mut() {
                *slot = 1;
            }
        }

        let mut expanded = length_regulate(&enc, t, d, &durations)?;
        let t_mel = expanded.len() / d;
        self.add_variance(&mut expanded, t_mel)?;
        let mel = self.decode(&expanded, t_mel)?;
        Ok((mel, durations))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Private helpers ───────────────────────────────────────────────────────

    #[test]
    fn matmul_simple() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]].
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let c = matmul(&a, &b, 2, 2, 2);
        assert!((c[0] - 19.0).abs() < 1e-4);
        assert!((c[1] - 22.0).abs() < 1e-4);
        assert!((c[2] - 43.0).abs() < 1e-4);
        assert!((c[3] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut s = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax_inplace(&mut s);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
    }

    #[test]
    fn relu_clamps_negative() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(2.5), 2.5);
    }

    #[test]
    fn conv1d_same_preserves_length() {
        // Single channel, identity-ish kernel.
        let t = 5;
        let x: Vec<f32> = (0..t).map(|i| i as f32).collect();
        // kernel=3, weight picks only the centre tap = 1.0.
        let weight = vec![0.0_f32, 1.0, 0.0];
        let bias = vec![0.0_f32];
        let out = conv1d_same(&x, t, 1, 1, 3, &weight, &bias);
        assert_eq!(out.len(), t);
        assert_eq!(out, x, "centre-tap conv should be identity");
    }

    #[test]
    fn conv1d_same_left_shift() {
        // kernel=3 with weight on the right tap reads x[t+1] (zero-padded edge).
        let t = 4;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![0.0_f32, 0.0, 1.0]; // tap index 2 → src = ti+1
        let bias = vec![0.0_f32];
        let out = conv1d_same(&x, t, 1, 1, 3, &weight, &bias);
        assert_eq!(out, vec![2.0, 3.0, 4.0, 0.0]);
    }

    // ── FftBlock ──────────────────────────────────────────────────────────────

    #[test]
    fn fft_block_output_shape() {
        let mut rng = LcgRng::new(1);
        let block = FftBlock::new(16, 2, 32, 9, &mut rng).expect("new");
        for t in [1usize, 3, 8, 20] {
            let x = vec![0.1_f32; t * 16];
            let out = block.forward(&x, t).expect("forward");
            assert_eq!(out.len(), t * 16, "shape wrong for t={t}");
        }
    }

    #[test]
    fn fft_block_output_finite() {
        let mut rng = LcgRng::new(2);
        let block = FftBlock::new(16, 2, 32, 9, &mut rng).expect("new");
        let t = 12usize;
        let mut x = vec![0.0_f32; t * 16];
        rng.fill_normal(&mut x);
        let out = block.forward(&x, t).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite FFT output");
    }

    #[test]
    fn fft_block_empty_t_err() {
        let mut rng = LcgRng::new(3);
        let block = FftBlock::new(16, 2, 32, 9, &mut rng).expect("new");
        assert!(block.forward(&[], 0).is_err());
    }

    #[test]
    fn fft_block_even_kernel_err() {
        let mut rng = LcgRng::new(4);
        assert!(FftBlock::new(16, 2, 32, 8, &mut rng).is_err());
    }

    #[test]
    fn fft_block_head_mismatch_err() {
        let mut rng = LcgRng::new(5);
        // 16 % 3 != 0
        assert!(FftBlock::new(16, 3, 32, 9, &mut rng).is_err());
    }

    // ── Length regulator ──────────────────────────────────────────────────────

    #[test]
    fn length_regulate_basic_expansion() {
        // 3 phonemes, dim 4, durations [2, 0, 3] → 5 frames.
        let dim = 4;
        let x: Vec<f32> = vec![
            // phoneme 0
            1.0, 1.1, 1.2, 1.3, // phoneme 1
            2.0, 2.1, 2.2, 2.3, // phoneme 2
            3.0, 3.1, 3.2, 3.3,
        ];
        let durations = vec![2usize, 0, 3];
        let out = length_regulate(&x, 3, dim, &durations).expect("regulate");
        assert_eq!(out.len(), 5 * dim);

        // Frames 0,1 == phoneme 0.
        assert_eq!(&out[0..4], &x[0..4]);
        assert_eq!(&out[4..8], &x[0..4]);
        // Frames 2,3,4 == phoneme 2 (phoneme 1 with dur 0 is skipped).
        assert_eq!(&out[8..12], &x[8..12]);
        assert_eq!(&out[12..16], &x[8..12]);
        assert_eq!(&out[16..20], &x[8..12]);
    }

    #[test]
    fn length_regulate_total_equals_sum() {
        let dim = 3;
        let durations = vec![1usize, 4, 2, 0, 5];
        let t = durations.len();
        let x: Vec<f32> = (0..t * dim).map(|i| i as f32).collect();
        let out = length_regulate(&x, t, dim, &durations).expect("regulate");
        let total: usize = durations.iter().sum();
        assert_eq!(out.len(), total * dim);
    }

    #[test]
    fn length_regulate_with_pace_changes_length() {
        let dim = 2;
        let durations = vec![4usize, 4, 4]; // sum 12
        let t = durations.len();
        let x: Vec<f32> = (0..t * dim).map(|i| i as f32).collect();

        // pace 2.0 → roughly half the frames (4/2 = 2 each → 6 total).
        let faster = length_regulate_with_pace(&x, t, dim, &durations, 2.0).expect("pace");
        assert_eq!(faster.len(), 6 * dim);

        // pace 0.5 → roughly double (4/0.5 = 8 each → 24 total).
        let slower = length_regulate_with_pace(&x, t, dim, &durations, 0.5).expect("pace");
        assert_eq!(slower.len(), 24 * dim);
    }

    #[test]
    fn length_regulate_with_pace_keeps_nonzero_audible() {
        // A short phoneme should never vanish to 0 frames from pace scaling.
        let dim = 1;
        let durations = vec![1usize, 1, 1];
        let t = durations.len();
        let x = vec![1.0_f32, 2.0, 3.0];
        let out = length_regulate_with_pace(&x, t, dim, &durations, 10.0).expect("pace");
        // 1/10 rounds to 0 but is clamped to 1 each → 3 frames.
        assert_eq!(out.len(), 3 * dim);
    }

    #[test]
    fn length_regulate_len_mismatch_err() {
        let dim = 2;
        let x = vec![0.0_f32; 3 * dim];
        let durations = vec![1usize, 1]; // len 2 != t 3
        assert!(length_regulate(&x, 3, dim, &durations).is_err());
    }

    #[test]
    fn length_regulate_all_zero_err() {
        let dim = 2;
        let x = vec![0.0_f32; 3 * dim];
        let durations = vec![0usize, 0, 0];
        let r = length_regulate(&x, 3, dim, &durations);
        assert_eq!(r, Err(AudioError::InvalidSequenceLength(0)));
    }

    #[test]
    fn length_regulate_pace_invalid_err() {
        let dim = 2;
        let x = vec![0.0_f32; 2 * dim];
        let durations = vec![1usize, 1];
        assert!(length_regulate_with_pace(&x, 2, dim, &durations, 0.0).is_err());
        assert!(length_regulate_with_pace(&x, 2, dim, &durations, f32::NAN).is_err());
    }

    // ── Duration predictor ────────────────────────────────────────────────────

    #[test]
    fn duration_predictor_shape_and_finite() {
        let mut rng = LcgRng::new(10);
        let dp = DurationPredictor::new(16, 16, 3, &mut rng).expect("new");
        let t = 7usize;
        let mut x = vec![0.0_f32; t * 16];
        rng.fill_normal(&mut x);
        let log_d = dp.predict(&x, t).expect("predict");
        assert_eq!(log_d.len(), t);
        assert!(log_d.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn duration_predictor_rounded_nonnegative_and_deterministic() {
        let mut rng = LcgRng::new(11);
        let dp = DurationPredictor::new(16, 16, 3, &mut rng).expect("new");
        let t = 9usize;
        let mut x = vec![0.0_f32; t * 16];
        rng.fill_normal(&mut x);
        let a = dp.predict_rounded(&x, t).expect("a");
        let b = dp.predict_rounded(&x, t).expect("b");
        assert_eq!(a.len(), t);
        assert_eq!(a, b, "predict_rounded must be deterministic");
        // usize is inherently non-negative; assert each is within the cap.
        assert!(a.iter().all(|&d| d <= dp.max_duration));
    }

    // ── Quantisation ──────────────────────────────────────────────────────────

    #[test]
    fn quantize_below_above_and_monotone() {
        let boundaries = vec![-1.0_f32, 0.0, 1.0]; // 4 bins: (..-1)(−1..0)(0..1)(1..)
        // Below all → bin 0; above all → last bin (3).
        let bins = quantize_to_bins(&[-5.0, -0.5, 0.5, 5.0], &boundaries);
        assert_eq!(bins, vec![0, 1, 2, 3]);

        // Monotone non-decreasing across a sweep.
        let sweep: Vec<f32> = (-30..30).map(|i| i as f32 * 0.1).collect();
        let swept = quantize_to_bins(&sweep, &boundaries);
        for w in swept.windows(2) {
            assert!(w[1] >= w[0], "quantisation not monotone");
        }
        assert_eq!(*swept.first().unwrap(), 0);
        assert_eq!(*swept.last().unwrap(), 3);
    }

    #[test]
    fn quantize_empty_boundaries_all_zero() {
        let bins = quantize_to_bins(&[-1.0, 0.0, 1.0], &[]);
        assert_eq!(bins, vec![0, 0, 0]);
    }

    // ── Variance predictor ────────────────────────────────────────────────────

    #[test]
    fn variance_predictor_shape_and_finite() {
        let mut rng = LcgRng::new(12);
        let vp = VariancePredictor::new(16, 16, 3, &mut rng).expect("new");
        let t = 11usize;
        let mut x = vec![0.0_f32; t * 16];
        rng.fill_normal(&mut x);
        let out = vp.predict(&x, t).expect("predict");
        assert_eq!(out.len(), t);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn variance_predictor_bad_shape_err() {
        let mut rng = LcgRng::new(13);
        let vp = VariancePredictor::new(16, 16, 3, &mut rng).expect("new");
        // t says 4 but buffer is sized for 3.
        assert!(vp.predict(&[0.0_f32; 3 * 16], 4).is_err());
    }

    // ── embed_and_add ─────────────────────────────────────────────────────────

    #[test]
    fn embed_and_add_adds_correct_row() {
        let dim = 3;
        let t = 2;
        let mut hidden = vec![0.0_f32; t * dim];
        // n_bins = 3 distinct, easily-identifiable rows.
        let embedding = vec![
            10.0, 11.0, 12.0, // bin 0
            20.0, 21.0, 22.0, // bin 1
            30.0, 31.0, 32.0, // bin 2
        ];
        let bin_ids = vec![2usize, 0];
        embed_and_add(&mut hidden, t, dim, &bin_ids, &embedding).expect("add");
        // Frame 0 += bin 2, frame 1 += bin 0.
        assert_eq!(&hidden[0..3], &[30.0, 31.0, 32.0]);
        assert_eq!(&hidden[3..6], &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn embed_and_add_accumulates_on_existing() {
        let dim = 2;
        let t = 1;
        let mut hidden = vec![1.0_f32, 2.0];
        let embedding = vec![5.0_f32, 6.0]; // single bin
        embed_and_add(&mut hidden, t, dim, &[0], &embedding).expect("add");
        assert_eq!(hidden, vec![6.0, 8.0]);
    }

    #[test]
    fn embed_and_add_out_of_range_err() {
        let dim = 2;
        let mut hidden = vec![0.0_f32; 2];
        let embedding = vec![1.0_f32, 2.0]; // 1 bin only
        let r = embed_and_add(&mut hidden, 1, dim, &[5], &embedding);
        assert!(matches!(r, Err(AudioError::WeightShapeMismatch { .. })));
    }

    #[test]
    fn embed_and_add_shape_mismatch_err() {
        let dim = 2;
        let mut hidden = vec![0.0_f32; 4];
        let embedding = vec![1.0_f32, 2.0];
        // bin_ids len 1 but t 2.
        assert!(embed_and_add(&mut hidden, 2, dim, &[0], &embedding).is_err());
    }

    // ── FastSpeech2 config ────────────────────────────────────────────────────

    #[test]
    fn config_tiny_valid() {
        assert!(FastSpeech2Config::tiny().validate().is_ok());
    }

    #[test]
    fn config_head_mismatch_err() {
        let mut cfg = FastSpeech2Config::tiny();
        cfg.n_heads = 3; // 16 % 3 != 0
        assert_eq!(
            cfg.validate(),
            Err(AudioError::HeadDimMismatch {
                embed_dim: 16,
                n_heads: 3
            })
        );
    }

    #[test]
    fn config_zero_mels_err() {
        let mut cfg = FastSpeech2Config::tiny();
        cfg.n_mels = 0;
        assert_eq!(cfg.validate(), Err(AudioError::InvalidNumMels(0)));
    }

    #[test]
    fn config_new_validates() {
        // Even ffn kernel must be rejected by `new`.
        let r = FastSpeech2Config::new(16, 2, 32, 8, 2, 16, 3, 8, 8, 16);
        assert!(r.is_err());
    }

    // ── FastSpeech2 forward ───────────────────────────────────────────────────

    #[test]
    fn fastspeech2_build_ok() {
        let mut rng = LcgRng::new(100);
        let model = FastSpeech2::new(FastSpeech2Config::tiny(), &mut rng);
        assert!(model.is_ok(), "build failed: {:?}", model.err());
    }

    #[test]
    fn fastspeech2_infer_shapes() {
        let cfg = FastSpeech2Config::tiny();
        let mut rng = LcgRng::new(101);
        let model = FastSpeech2::new(cfg.clone(), &mut rng).expect("new");
        let t = 5usize;
        let mut phon = vec![0.0_f32; t * cfg.embed_dim];
        rng.fill_normal(&mut phon);

        let (mel, dur) = model.forward_infer(&phon, t).expect("infer");
        assert_eq!(dur.len(), t, "one duration per phoneme");
        let total: usize = dur.iter().sum();
        assert_eq!(mel.len(), total * cfg.n_mels, "mel length wrong");
        assert!(mel.iter().all(|v| v.is_finite()), "non-finite mel");
        // Floor guarantees at least one frame.
        assert!(total >= 1);
    }

    #[test]
    fn fastspeech2_train_uses_gt_durations() {
        let cfg = FastSpeech2Config::tiny();
        let mut rng = LcgRng::new(102);
        let model = FastSpeech2::new(cfg.clone(), &mut rng).expect("new");
        let t = 4usize;
        let mut phon = vec![0.0_f32; t * cfg.embed_dim];
        rng.fill_normal(&mut phon);

        let gt = vec![2usize, 3, 0, 4]; // sum 9
        let mel = model.forward_train(&phon, t, &gt).expect("train");
        let total: usize = gt.iter().sum();
        assert_eq!(mel.len(), total * cfg.n_mels);
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fastspeech2_train_gt_len_mismatch_err() {
        let cfg = FastSpeech2Config::tiny();
        let mut rng = LcgRng::new(103);
        let model = FastSpeech2::new(cfg.clone(), &mut rng).expect("new");
        let t = 4usize;
        let phon = vec![0.0_f32; t * cfg.embed_dim];
        // gt length 3 != t 4.
        assert!(model.forward_train(&phon, t, &[1, 1, 1]).is_err());
    }

    #[test]
    fn fastspeech2_infer_deterministic() {
        let cfg = FastSpeech2Config::tiny();
        let mut rng_a = LcgRng::new(2024);
        let model_a = FastSpeech2::new(cfg.clone(), &mut rng_a).expect("a");
        let mut rng_b = LcgRng::new(2024);
        let model_b = FastSpeech2::new(cfg.clone(), &mut rng_b).expect("b");

        let t = 6usize;
        // Deterministic input from a fixed seed.
        let mut seed_rng = LcgRng::new(555);
        let mut phon = vec![0.0_f32; t * cfg.embed_dim];
        seed_rng.fill_normal(&mut phon);

        let (mel_a, dur_a) = model_a.forward_infer(&phon, t).expect("a");
        let (mel_b, dur_b) = model_b.forward_infer(&phon, t).expect("b");
        assert_eq!(dur_a, dur_b, "durations must match for same seed");
        assert_eq!(mel_a, mel_b, "mel must be bit-identical for same seed");
    }

    #[test]
    fn fastspeech2_empty_input_err() {
        let cfg = FastSpeech2Config::tiny();
        let mut rng = LcgRng::new(104);
        let model = FastSpeech2::new(cfg, &mut rng).expect("new");
        assert!(model.forward_infer(&[], 0).is_err());
    }

    #[test]
    fn fastspeech2_bad_phon_shape_err() {
        let cfg = FastSpeech2Config::tiny();
        let mut rng = LcgRng::new(105);
        let model = FastSpeech2::new(cfg.clone(), &mut rng).expect("new");
        // t says 5 but buffer sized for 4.
        let phon = vec![0.0_f32; 4 * cfg.embed_dim];
        assert!(model.forward_infer(&phon, 5).is_err());
    }
}

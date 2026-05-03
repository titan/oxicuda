//! Conformer encoder block with macaron-style FFN halving.
//!
//! Each block applies the following pipeline:
//!
//! ```text
//! x → ½ · FFN₁(LN(x)) + x
//!   → MHSA-with-rel-pos(LN(x)) + x
//!   → ConvModule(x) + x
//!   → ½ · FFN₂(LN(x)) + x
//!   → LN(x)
//! ```
//!
//! `ConformerEncoder` stacks `depth` blocks and appends a final layer norm.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Layer normalisation over the last dimension of a `[T * D]` flat buffer.
///
/// Normalises each row (timestep) independently.
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

/// Tanh-approximation GELU activation.
#[inline]
fn gelu_approx(x: f32) -> f32 {
    let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Dense matrix multiply: `C = A * B` where A is `[m, k]`, B is `[k, n]`.
///
/// Both inputs and output are flat row-major.
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

/// Feed-forward sub-layer for a single block.
///
/// `x` — `[T, D]` flat, returns `[T, D]` flat (pre-residual, pre-halving).
fn ffn_forward(x: &[f32], t: usize, embed_dim: usize, weights: &FfnWeights) -> Vec<f32> {
    let ffn_dim = weights.w1.len() / embed_dim;

    // Layer norm [T, D].
    let normed = layer_norm(x, &weights.ln_weight, &weights.ln_bias, 1e-5);

    // Linear 1: [T, D] × [D, ffn_dim]ᵀ → [T, ffn_dim].
    // w1 is [ffn_dim, embed_dim] so we compute normed × w1ᵀ.
    let mut h = vec![0.0_f32; t * ffn_dim];
    for ti in 0..t {
        for fi in 0..ffn_dim {
            let mut acc = weights.b1[fi];
            for di in 0..embed_dim {
                acc += normed[ti * embed_dim + di] * weights.w1[fi * embed_dim + di];
            }
            h[ti * ffn_dim + fi] = gelu_approx(acc);
        }
    }

    // Linear 2: [T, ffn_dim] × [ffn_dim, embed_dim]ᵀ → [T, embed_dim].
    // w2 is [embed_dim, ffn_dim].
    let mut out = vec![0.0_f32; t * embed_dim];
    for ti in 0..t {
        for di in 0..embed_dim {
            let mut acc = weights.b2[di];
            for fi in 0..ffn_dim {
                acc += h[ti * ffn_dim + fi] * weights.w2[di * ffn_dim + fi];
            }
            out[ti * embed_dim + di] = acc;
        }
    }
    out
}

/// Multi-head self-attention with relative position bias.
///
/// Uses sinusoidal-style table lookup: `scores[q, k] += rel_pos_table[k - q + max_len - 1]`
/// (clamped to `[0, 2*max_len - 2]`).
///
/// `x` — `[T, D]` flat, returns `[T, D]` flat (pre-residual).
fn mhsa_forward(
    x: &[f32],
    t: usize,
    embed_dim: usize,
    n_heads: usize,
    weights: &MhsaWeights,
) -> Vec<f32> {
    let head_dim = embed_dim / n_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_len = weights.max_len;

    // Layer norm.
    let normed = layer_norm(x, &weights.ln_weight, &weights.ln_bias, 1e-5);

    // Project Q, K, V — each [T, D].
    let q = project_qkv(&normed, t, embed_dim, &weights.q_proj, &weights.q_bias);
    let k = project_qkv(&normed, t, embed_dim, &weights.k_proj, &weights.k_bias);
    let v = project_qkv(&normed, t, embed_dim, &weights.v_proj, &weights.v_bias);

    // Per-head attention.
    let mut ctx = vec![0.0_f32; t * embed_dim];

    for h in 0..n_heads {
        let h_off = h * head_dim;

        // Extract [T, head_dim] slices for this head.
        let mut q_h = vec![0.0_f32; t * head_dim];
        let mut k_h = vec![0.0_f32; t * head_dim];
        let mut v_h = vec![0.0_f32; t * head_dim];
        for ti in 0..t {
            q_h[ti * head_dim..ti * head_dim + head_dim]
                .copy_from_slice(&q[ti * embed_dim + h_off..ti * embed_dim + h_off + head_dim]);
            k_h[ti * head_dim..ti * head_dim + head_dim]
                .copy_from_slice(&k[ti * embed_dim + h_off..ti * embed_dim + h_off + head_dim]);
            v_h[ti * head_dim..ti * head_dim + head_dim]
                .copy_from_slice(&v[ti * embed_dim + h_off..ti * embed_dim + h_off + head_dim]);
        }

        // Attention scores [T, T] = Q_h × K_hᵀ × scale.
        // Transpose K from [T, head_dim] → [head_dim, T] so that
        // matmul(Q_h, K_hᵀ) gives [T, T].
        let k_h_t = transpose_2d(&k_h, t, head_dim);
        let mut scores_mat = matmul(&q_h, &k_h_t, t, head_dim, t);

        for v_s in scores_mat.iter_mut() {
            *v_s *= scale;
        }

        // Add relative position bias.
        let table_len = weights.rel_pos_table.len();
        for qi in 0..t {
            for ki in 0..t {
                let rel = ki as isize - qi as isize + max_len as isize - 1;
                let idx = rel.clamp(0, table_len as isize - 1) as usize;
                scores_mat[qi * t + ki] += weights.rel_pos_table[idx];
            }
        }

        // Softmax per query.
        for qi in 0..t {
            softmax_inplace(&mut scores_mat[qi * t..(qi + 1) * t]);
        }

        // Context: [T, T] × [T, head_dim] → [T, head_dim].
        let ctx_h = matmul(&scores_mat, &v_h, t, t, head_dim);

        // Write back into ctx [T, embed_dim].
        for ti in 0..t {
            ctx[ti * embed_dim + h_off..ti * embed_dim + h_off + head_dim]
                .copy_from_slice(&ctx_h[ti * head_dim..(ti + 1) * head_dim]);
        }
    }

    // Output projection: [T, D] × [D, D]ᵀ → [T, D].
    let out_w_t = transpose_2d(&weights.out_proj, embed_dim, embed_dim);
    let mut out = matmul(&ctx, &out_w_t, t, embed_dim, embed_dim);
    for ti in 0..t {
        for di in 0..embed_dim {
            out[ti * embed_dim + di] += weights.out_bias[di];
        }
    }
    out
}

/// Applies a linear projection `y = x * wᵀ + b` where `w` is `[D, D]`.
fn project_qkv(x: &[f32], t: usize, d: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let w_t = transpose_2d(w, d, d);
    let mut out = matmul(x, &w_t, t, d, d);
    for ti in 0..t {
        for di in 0..d {
            out[ti * d + di] += b[di];
        }
    }
    out
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

/// Add `scale * delta` into `acc` in-place (used for residual connections).
fn add_scaled_residual(acc: &mut [f32], delta: &[f32], scale: f32) {
    for (a, d) in acc.iter_mut().zip(delta.iter()) {
        *a += scale * d;
    }
}

/// Xavier uniform limit: `sqrt(6 / (fan_in + fan_out))`.
#[inline]
fn xavier_limit(fan_in: usize, fan_out: usize) -> f32 {
    (6.0 / (fan_in + fan_out) as f32).sqrt()
}

// ─── Public types ────────────────────────────────────────────────────────────

/// Feed-forward sub-network weights for a Conformer block.
#[derive(Debug)]
pub struct FfnWeights {
    /// Layer-norm scale `[D]`.
    pub ln_weight: Vec<f32>,
    /// Layer-norm bias `[D]`.
    pub ln_bias: Vec<f32>,
    /// First linear weight `[ffn_dim, embed_dim]`.
    pub w1: Vec<f32>,
    /// First linear bias `[ffn_dim]`.
    pub b1: Vec<f32>,
    /// Second linear weight `[embed_dim, ffn_dim]`.
    pub w2: Vec<f32>,
    /// Second linear bias `[embed_dim]`.
    pub b2: Vec<f32>,
}

/// Multi-head self-attention weights for a Conformer block.
#[derive(Debug)]
pub struct MhsaWeights {
    /// Layer-norm scale `[D]`.
    pub ln_weight: Vec<f32>,
    /// Layer-norm bias `[D]`.
    pub ln_bias: Vec<f32>,
    /// Query projection `[embed_dim, embed_dim]`.
    pub q_proj: Vec<f32>,
    /// Key projection `[embed_dim, embed_dim]`.
    pub k_proj: Vec<f32>,
    /// Value projection `[embed_dim, embed_dim]`.
    pub v_proj: Vec<f32>,
    /// Output projection `[embed_dim, embed_dim]`.
    pub out_proj: Vec<f32>,
    /// Query bias `[embed_dim]`.
    pub q_bias: Vec<f32>,
    /// Key bias `[embed_dim]`.
    pub k_bias: Vec<f32>,
    /// Value bias `[embed_dim]`.
    pub v_bias: Vec<f32>,
    /// Output bias `[embed_dim]`.
    pub out_bias: Vec<f32>,
    /// Relative position table `[2 * max_len - 1]`.
    pub rel_pos_table: Vec<f32>,
    /// Maximum sequence length encoded in the table.
    pub max_len: usize,
}

/// A single Conformer block (macaron FFN + MHSA + Conv + macaron FFN).
#[derive(Debug)]
pub struct ConformerBlock {
    /// First half-step FFN.
    pub ffn1: FfnWeights,
    /// Multi-head self-attention sub-layer.
    pub mhsa: MhsaWeights,
    /// Convolution sub-module.
    pub conv: super::conv_module::ConvModule,
    /// Second half-step FFN.
    pub ffn2: FfnWeights,
    /// Final block layer-norm scale `[D]`.
    pub final_ln_weight: Vec<f32>,
    /// Final block layer-norm bias `[D]`.
    pub final_ln_bias: Vec<f32>,
    /// Model (embedding) dimension.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
}

/// Configuration for a [`ConformerEncoder`].
#[derive(Debug, Clone)]
pub struct ConformerConfig {
    /// Embedding dimension `D`.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Depthwise convolution kernel size inside each block.
    pub conv_kernel: usize,
    /// FFN hidden dimension multiplier: `ffn_dim = embed_dim * ffn_expansion`.
    pub ffn_expansion: usize,
    /// Number of stacked Conformer blocks.
    pub depth: usize,
    /// Maximum input sequence length (determines relative-position table size).
    pub max_len: usize,
}

impl ConformerConfig {
    /// Tiny test configuration: `D=64, H=4, K=15, expansion=4, depth=2, max_len=256`.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            embed_dim: 64,
            n_heads: 4,
            conv_kernel: 15,
            ffn_expansion: 4,
            depth: 2,
            max_len: 256,
        }
    }

    /// Validate the configuration, returning an error for any illegal combination.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::InvalidEmbedDim`, `AudioError::InvalidNumHeads`,
    /// `AudioError::HeadDimMismatch`, or `AudioError::InvalidKernelSize`
    /// when the configuration is inconsistent.
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
        if self.conv_kernel == 0 {
            return Err(AudioError::InvalidKernelSize(0));
        }
        Ok(())
    }
}

/// Stack of [`ConformerBlock`]s with a final layer norm.
pub struct ConformerEncoder {
    /// Ordered blocks (block 0 is closest to the input).
    pub blocks: Vec<ConformerBlock>,
    /// Final encoder layer-norm scale `[D]`.
    pub final_norm_weight: Vec<f32>,
    /// Final encoder layer-norm bias `[D]`.
    pub final_norm_bias: Vec<f32>,
    /// Configuration used to build this encoder.
    pub config: ConformerConfig,
}

// ─── Constructors ────────────────────────────────────────────────────────────

/// Initialise an `FfnWeights` struct with Xavier-uniform weights.
fn init_ffn_weights(embed_dim: usize, ffn_dim: usize, rng: &mut LcgRng) -> FfnWeights {
    let lim1 = xavier_limit(embed_dim, ffn_dim);
    let lim2 = xavier_limit(ffn_dim, embed_dim);

    let mut w1 = vec![0.0_f32; ffn_dim * embed_dim];
    for v in w1.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * lim1;
    }
    let mut b1 = vec![0.0_f32; ffn_dim];
    for v in b1.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * lim1;
    }
    let mut w2 = vec![0.0_f32; embed_dim * ffn_dim];
    for v in w2.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * lim2;
    }
    let mut b2 = vec![0.0_f32; embed_dim];
    for v in b2.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * lim2;
    }

    FfnWeights {
        ln_weight: vec![1.0_f32; embed_dim],
        ln_bias: vec![0.0_f32; embed_dim],
        w1,
        b1,
        w2,
        b2,
    }
}

/// Allocate a weight matrix of `sz` elements initialised with Xavier-uniform values.
fn make_xavier_vec(sz: usize, lim: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut w = vec![0.0_f32; sz];
    for v in w.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * lim;
    }
    w
}

/// Initialise an `MhsaWeights` struct with Xavier-uniform weights.
fn init_mhsa_weights(embed_dim: usize, max_len: usize, rng: &mut LcgRng) -> MhsaWeights {
    let lim = xavier_limit(embed_dim, embed_dim);
    let proj_len = embed_dim * embed_dim;

    let q_proj = make_xavier_vec(proj_len, lim, rng);
    let k_proj = make_xavier_vec(proj_len, lim, rng);
    let v_proj = make_xavier_vec(proj_len, lim, rng);
    let out_proj = make_xavier_vec(proj_len, lim, rng);
    let q_bias = make_xavier_vec(embed_dim, lim, rng);
    let k_bias = make_xavier_vec(embed_dim, lim, rng);
    let v_bias = make_xavier_vec(embed_dim, lim, rng);
    let out_bias = make_xavier_vec(embed_dim, lim, rng);

    // Relative position table: length = 2*max_len - 1.
    let table_len = 2 * max_len - 1;
    let mut rel_pos_table = vec![0.0_f32; table_len];
    rng.fill_normal(&mut rel_pos_table);
    // Scale down so attention logits stay reasonable at init.
    for v in rel_pos_table.iter_mut() {
        *v *= 0.01;
    }

    MhsaWeights {
        ln_weight: vec![1.0_f32; embed_dim],
        ln_bias: vec![0.0_f32; embed_dim],
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        q_bias,
        k_bias,
        v_bias,
        out_bias,
        rel_pos_table,
        max_len,
    }
}

impl ConformerBlock {
    /// Construct a new `ConformerBlock` from `config` using Xavier-uniform init.
    ///
    /// # Errors
    ///
    /// Returns an error if `config.validate()` fails or if the inner
    /// [`ConvModule`][super::conv_module::ConvModule] constructor fails.
    pub fn new(config: &ConformerConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        config.validate()?;
        let d = config.embed_dim;
        let ffn_dim = d * config.ffn_expansion;

        let ffn1 = init_ffn_weights(d, ffn_dim, rng);
        let mhsa = init_mhsa_weights(d, config.max_len, rng);
        let conv = super::conv_module::ConvModule::new(d, config.conv_kernel, rng)?;
        let ffn2 = init_ffn_weights(d, ffn_dim, rng);

        Ok(Self {
            ffn1,
            mhsa,
            conv,
            ffn2,
            final_ln_weight: vec![1.0_f32; d],
            final_ln_bias: vec![0.0_f32; d],
            embed_dim: d,
            n_heads: config.n_heads,
        })
    }

    /// Apply the Conformer block to `x` of shape `[T, D]` (flat row-major).
    ///
    /// # Returns
    ///
    /// `[T, D]` flat output tensor.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::ShapeMismatch` when `x.len() != t * embed_dim`,
    /// or `AudioError::EmptyInput` when `t == 0`.
    pub fn forward(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        let d = self.embed_dim;
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "ConformerBlock: t == 0".into(),
            });
        }
        if x.len() != t * d {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "ConformerBlock::forward: x.len()={} != t*d={}",
                    x.len(),
                    t * d
                ),
            });
        }

        // Residual stream starts as a copy of the input.
        let mut h = x.to_vec();

        // ── ½ · FFN₁ + residual ─────────────────────────────────────────────
        let ffn1_out = ffn_forward(&h, t, d, &self.ffn1);
        add_scaled_residual(&mut h, &ffn1_out, 0.5);

        // ── MHSA + residual ──────────────────────────────────────────────────
        let mhsa_out = mhsa_forward(&h, t, d, self.n_heads, &self.mhsa);
        add_scaled_residual(&mut h, &mhsa_out, 1.0);

        // ── ConvModule + residual ────────────────────────────────────────────
        let conv_out = self.conv.forward(&h, t)?;
        add_scaled_residual(&mut h, &conv_out, 1.0);

        // ── ½ · FFN₂ + residual ─────────────────────────────────────────────
        let ffn2_out = ffn_forward(&h, t, d, &self.ffn2);
        add_scaled_residual(&mut h, &ffn2_out, 0.5);

        // ── Final layer norm ─────────────────────────────────────────────────
        let out = layer_norm(&h, &self.final_ln_weight, &self.final_ln_bias, 1e-5);

        Ok(out)
    }
}

impl ConformerEncoder {
    /// Construct a `ConformerEncoder` with `config.depth` stacked blocks.
    ///
    /// # Errors
    ///
    /// Returns any error produced by [`ConformerBlock::new`].
    pub fn new(config: ConformerConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        config.validate()?;
        let d = config.embed_dim;
        let mut blocks = Vec::with_capacity(config.depth);
        for _ in 0..config.depth {
            blocks.push(ConformerBlock::new(&config, rng)?);
        }
        Ok(Self {
            blocks,
            final_norm_weight: vec![1.0_f32; d],
            final_norm_bias: vec![0.0_f32; d],
            config,
        })
    }

    /// Run the full encoder on `x` of shape `[T, D]` (flat row-major).
    ///
    /// # Returns
    ///
    /// `[T, D]` flat output after all blocks and the final layer norm.
    ///
    /// # Errors
    ///
    /// Returns errors from any block's `forward`, or `AudioError::ShapeMismatch`
    /// when the input size is inconsistent.
    pub fn forward(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        let d = self.config.embed_dim;
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "ConformerEncoder: t == 0".into(),
            });
        }
        if x.len() != t * d {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "ConformerEncoder::forward: x.len()={} != t*d={}",
                    x.len(),
                    t * d
                ),
            });
        }

        let mut h = x.to_vec();
        for block in &self.blocks {
            h = block.forward(&h, t)?;
        }
        let out = layer_norm(&h, &self.final_norm_weight, &self.final_norm_bias, 1e-5);
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    #[test]
    fn matmul_identity() {
        // A × I = A.
        let a = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2×2
        let eye = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let c = matmul(&a, &eye, 2, 2, 2);
        assert_eq!(c, a);
    }

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
        let mut scores = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax_inplace(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
    }

    #[test]
    fn softmax_max_dominates() {
        let mut scores = vec![0.0_f32, 0.0, 10.0, 0.0];
        softmax_inplace(&mut scores);
        assert!(scores[2] > 0.99, "max should dominate: {}", scores[2]);
    }

    #[test]
    fn softmax_empty_noop() {
        let mut scores: Vec<f32> = vec![];
        softmax_inplace(&mut scores);
        assert!(scores.is_empty());
    }

    #[test]
    fn gelu_approx_zero() {
        assert!(gelu_approx(0.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_approx_large_positive() {
        assert!((gelu_approx(10.0) - 10.0).abs() < 0.01);
    }

    // ── ConformerConfig ───────────────────────────────────────────────────────

    #[test]
    fn conformer_config_tiny_valid() {
        let cfg = ConformerConfig::tiny();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn conformer_config_bad_heads_err() {
        let cfg = ConformerConfig {
            embed_dim: 64,
            n_heads: 7, // 64 % 7 != 0
            conv_kernel: 15,
            ffn_expansion: 4,
            depth: 1,
            max_len: 128,
        };
        assert!(cfg.validate().is_err());
    }

    // ── ConformerBlock ────────────────────────────────────────────────────────

    #[test]
    fn conformer_tiny_build_ok() {
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(42);
        let block = ConformerBlock::new(&cfg, &mut rng);
        assert!(block.is_ok(), "ConformerBlock::new failed: {block:?}");
    }

    #[test]
    fn conformer_block_output_shape() {
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(7);
        let block = ConformerBlock::new(&cfg, &mut rng).expect("new");

        for t in [1usize, 4, 16, 50] {
            let x = vec![0.1_f32; t * cfg.embed_dim];
            let out = block.forward(&x, t).expect("forward");
            assert_eq!(out.len(), t * cfg.embed_dim, "shape wrong for t={t}");
        }
    }

    #[test]
    fn conformer_block_output_finite() {
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(99);
        let block = ConformerBlock::new(&cfg, &mut rng).expect("new");
        let t = 20usize;
        let mut x = vec![0.0_f32; t * cfg.embed_dim];
        rng.fill_normal(&mut x);
        let out = block.forward(&x, t).expect("forward");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite in block output"
        );
    }

    #[test]
    fn conformer_block_empty_t_err() {
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(1);
        let block = ConformerBlock::new(&cfg, &mut rng).expect("new");
        let r = block.forward(&[], 0);
        assert!(r.is_err());
    }

    #[test]
    fn ffn_residual_changes_input() {
        // After ½·FFN+residual the output must differ from a zero-init input
        // (the Xavier init weights will produce non-trivial activations).
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(13);
        let block = ConformerBlock::new(&cfg, &mut rng).expect("new");
        let t = 8usize;
        let d = cfg.embed_dim;
        let x: Vec<f32> = (0..t * d).map(|i| (i as f32) * 0.001).collect();
        let ffn_delta = ffn_forward(&x, t, d, &block.ffn1);
        // The FFN must have done something non-trivial.
        let non_zero = ffn_delta.iter().any(|v| v.abs() > 1e-8);
        assert!(non_zero, "FFN produced all-zero output (suspicious)");
    }

    #[test]
    fn mhsa_output_shape() {
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(55);
        let block = ConformerBlock::new(&cfg, &mut rng).expect("new");
        let t = 12usize;
        let d = cfg.embed_dim;
        let x = vec![0.5_f32; t * d];
        let out = mhsa_forward(&x, t, d, cfg.n_heads, &block.mhsa);
        assert_eq!(out.len(), t * d, "MHSA output shape wrong");
    }

    // ── ConformerEncoder ──────────────────────────────────────────────────────

    #[test]
    fn conformer_encoder_forward_shape() {
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(77);
        let enc = ConformerEncoder::new(cfg.clone(), &mut rng).expect("new");
        let t = 24usize;
        let x = vec![0.1_f32; t * cfg.embed_dim];
        let out = enc.forward(&x, t).expect("forward");
        assert_eq!(out.len(), t * cfg.embed_dim);
    }

    #[test]
    fn conformer_encoder_forward_finite() {
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(11);
        let enc = ConformerEncoder::new(cfg.clone(), &mut rng).expect("new");
        let t = 10usize;
        let mut x = vec![0.0_f32; t * cfg.embed_dim];
        rng.fill_normal(&mut x);
        let out = enc.forward(&x, t).expect("forward");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite in encoder output"
        );
    }

    #[test]
    fn conformer_encoder_depth_matches_config() {
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(3);
        let enc = ConformerEncoder::new(cfg.clone(), &mut rng).expect("new");
        assert_eq!(enc.blocks.len(), cfg.depth);
    }
}

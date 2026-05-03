//! Multi-head self-attention with additive relative-position bias.
//!
//! Implements the Transformer self-attention mechanism augmented with the
//! Shaw et al. (2018) / Dai et al. (2019) relative-position encoding.  The
//! relative-position bias is looked up from a [`RelPosEncoding`] table and
//! added to the raw dot-product scores before softmax.

use crate::attention::rel_pos_encoding::RelPosEncoding;
use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private arithmetic helpers ──────────────────────────────────────────────

/// Dense matrix multiply: `C = A @ B`, sizes `[M, K] × [K, N] → [M, N]`.
///
/// Row-major storage throughout.
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

/// Numerically stable in-place softmax.
///
/// Subtracts the maximum element before exponentiation to avoid overflow.
fn softmax_inplace_stable(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Apply `X @ W^T + b` for each of `n` tokens.
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

// ─── Weight container ────────────────────────────────────────────────────────

/// Packed projection weight matrices and bias vectors for [`RelPosAttention`].
#[derive(Debug, Clone)]
pub struct RelPosAttentionWeights {
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
    /// Output-projection bias `[embed_dim]`.
    pub out_bias: Vec<f32>,
}

impl RelPosAttentionWeights {
    fn new(embed_dim: usize, rng: &mut LcgRng) -> Self {
        let proj_len = embed_dim * embed_dim;
        let scale = 1.0 / (embed_dim as f32).sqrt();

        let mut init_proj = |len: usize| -> Vec<f32> {
            let mut buf = vec![0.0_f32; len];
            rng.fill_normal(&mut buf);
            for v in &mut buf {
                *v *= scale;
            }
            buf
        };

        let q_proj = init_proj(proj_len);
        let k_proj = init_proj(proj_len);
        let v_proj = init_proj(proj_len);
        let out_proj = init_proj(proj_len);

        let init_bias = |len: usize| -> Vec<f32> { vec![0.0_f32; len] };

        let q_bias = init_bias(embed_dim);
        let k_bias = init_bias(embed_dim);
        let v_bias = init_bias(embed_dim);
        let out_bias = init_bias(embed_dim);

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_bias,
            k_bias,
            v_bias,
            out_bias,
        }
    }
}

// ─── RelPosAttention ─────────────────────────────────────────────────────────

/// Multi-head self-attention with additive relative-position bias.
///
/// Each head computes:
/// ```text
/// scores[h, q, k] = Q[h,q,:] · K[h,k,:] / sqrt(head_dim) + rel_pos_bias[q,k]
/// attn[h,q,:]     = softmax(scores[h,q,:])
/// out[h,q,:]      = attn[h,q,:] @ V[h,:,:]
/// ```
/// Heads are concatenated and projected through `out_proj`.
#[derive(Debug, Clone)]
pub struct RelPosAttention {
    /// Total embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dimension per head (`embed_dim / n_heads`).
    pub head_dim: usize,
    /// Projection weights and biases.
    pub weights: RelPosAttentionWeights,
    /// Relative-position encoding table.
    pub pos_enc: RelPosEncoding,
}

impl RelPosAttention {
    /// Construct a new `RelPosAttention` with randomly initialised weights.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidEmbedDim`] when `embed_dim == 0`.
    /// - [`AudioError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`AudioError::HeadDimMismatch`] when `embed_dim % n_heads != 0`.
    pub fn new(
        embed_dim: usize,
        n_heads: usize,
        max_len: usize,
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
        let head_dim = embed_dim / n_heads;
        let weights = RelPosAttentionWeights::new(embed_dim, rng);
        let pos_enc = RelPosEncoding::new(max_len, rng)?;
        Ok(Self {
            embed_dim,
            n_heads,
            head_dim,
            weights,
            pos_enc,
        })
    }

    /// Run multi-head relative-position attention on a sequence.
    ///
    /// # Arguments
    ///
    /// * `x` — input tensor `[T, embed_dim]` row-major.
    /// * `t` — sequence length `T` (must satisfy `x.len() == t * embed_dim`).
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::DimensionMismatch`] when `x.len() != t * embed_dim`.
    pub fn forward(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        let d = self.embed_dim;
        let expected = t * d;
        if x.len() != expected {
            return Err(AudioError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        if t == 0 {
            return Ok(vec![]);
        }

        let w = &self.weights;
        let h = self.n_heads;
        let hd = self.head_dim;

        // ── 1. Project to Q, K, V ── [T, D]
        let q_flat = linear(x, &w.q_proj, &w.q_bias, d, d, t);
        let k_flat = linear(x, &w.k_proj, &w.k_bias, d, d, t);
        let v_flat = linear(x, &w.v_proj, &w.v_bias, d, d, t);

        // ── 2. Build relative-position bias matrix [T, T]
        let rel_bias = self.pos_enc.bias_matrix(t, t);

        // ── 3–5. Per-head attention ──────────────────────────────────────────
        // Accumulator for concatenated head outputs: [T, D]
        let mut concat_out = vec![0.0_f32; t * d];

        let inv_sqrt_hd = 1.0 / (hd as f32).sqrt();

        for head in 0..h {
            let h_off = head * hd; // channel offset for this head

            // Extract head slices: Q_h[t, hd], K_h[t, hd], V_h[t, hd]
            let mut q_h = vec![0.0_f32; t * hd];
            let mut k_h = vec![0.0_f32; t * hd];
            let mut v_h = vec![0.0_f32; t * hd];

            for tok in 0..t {
                let src_off = tok * d + h_off;
                let dst_off = tok * hd;
                q_h[dst_off..dst_off + hd].copy_from_slice(&q_flat[src_off..src_off + hd]);
                k_h[dst_off..dst_off + hd].copy_from_slice(&k_flat[src_off..src_off + hd]);
                v_h[dst_off..dst_off + hd].copy_from_slice(&v_flat[src_off..src_off + hd]);
            }

            // scores[q, k] = Q_h[q,:] · K_h[k,:] * inv_sqrt_hd + rel_bias[q,k]
            // K_h^T: [hd, T]
            let mut k_h_t = vec![0.0_f32; hd * t];
            for tok in 0..t {
                for dim in 0..hd {
                    k_h_t[dim * t + tok] = k_h[tok * hd + dim];
                }
            }

            // raw_scores [T, T] = Q_h @ K_h^T  (matmul M=t, K=hd, N=t)
            let mut scores = matmul(&q_h, &k_h_t, t, hd, t);

            // Scale and add relative-position bias
            for (s, rb) in scores.iter_mut().zip(rel_bias.iter()) {
                *s = *s * inv_sqrt_hd + rb;
            }

            // Softmax over key dimension for each query
            for q_idx in 0..t {
                let row = &mut scores[q_idx * t..(q_idx + 1) * t];
                softmax_inplace_stable(row);
            }

            // context [T, hd] = scores @ V_h
            let context = matmul(&scores, &v_h, t, t, hd);

            // Write context into correct head slice of concat_out
            for tok in 0..t {
                let dst_off = tok * d + h_off;
                let src_off = tok * hd;
                concat_out[dst_off..dst_off + hd].copy_from_slice(&context[src_off..src_off + hd]);
            }
        }

        // ── 6. Output projection ── [T, D]
        let out = linear(&concat_out, &w.out_proj, &w.out_bias, d, d, t);
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn rel_pos_attention_new_ok() {
        let mut rng = make_rng();
        let attn = RelPosAttention::new(64, 4, 16, &mut rng);
        assert!(attn.is_ok());
        let a = attn.expect("ok");
        assert_eq!(a.head_dim, 16);
    }

    #[test]
    fn rel_pos_attention_head_dim_mismatch() {
        let mut rng = make_rng();
        let err = RelPosAttention::new(65, 4, 16, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::HeadDimMismatch { .. }));
    }

    #[test]
    fn rel_pos_attention_zero_embed_dim() {
        let mut rng = make_rng();
        let err = RelPosAttention::new(0, 4, 16, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::InvalidEmbedDim(0)));
    }

    #[test]
    fn rel_pos_attention_zero_heads() {
        let mut rng = make_rng();
        let err = RelPosAttention::new(64, 0, 16, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::InvalidNumHeads(0)));
    }

    #[test]
    fn rel_pos_attention_output_shape() {
        let mut rng = make_rng();
        let attn = RelPosAttention::new(32, 4, 20, &mut rng).expect("new ok");
        let t = 7_usize;
        let d = 32_usize;
        let mut x = vec![0.0_f32; t * d];
        rng.fill_normal(&mut x);
        let out = attn.forward(&x, t).expect("forward ok");
        assert_eq!(out.len(), t * d);
    }

    #[test]
    fn rel_pos_attention_output_finite() {
        let mut rng = make_rng();
        let attn = RelPosAttention::new(32, 4, 20, &mut rng).expect("new ok");
        let t = 5_usize;
        let d = 32_usize;
        let mut x = vec![0.0_f32; t * d];
        rng.fill_normal(&mut x);
        let out = attn.forward(&x, t).expect("forward ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite value found");
    }

    #[test]
    fn rel_pos_attention_single_token() {
        let mut rng = make_rng();
        let attn = RelPosAttention::new(16, 2, 8, &mut rng).expect("new ok");
        let x = vec![1.0_f32; 16];
        let out = attn.forward(&x, 1).expect("single token ok");
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softmax_inplace_sums_to_one() {
        let mut v = vec![1.0_f32, 2.0, 3.0, 0.5, -1.0];
        softmax_inplace_stable(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn softmax_inplace_empty() {
        let mut v: Vec<f32> = vec![];
        softmax_inplace_stable(&mut v); // must not panic
    }

    #[test]
    fn matmul_basic() {
        // 2×3 @ 3×2 = 2×2
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul(&a, &b, 2, 3, 2);
        // row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c.len(), 4);
        assert!((c[0] - 58.0).abs() < 1e-4);
        assert!((c[1] - 64.0).abs() < 1e-4);
        assert!((c[2] - 139.0).abs() < 1e-4);
        assert!((c[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn linear_output_shape() {
        // 3 tokens, in_d=4, out_d=6
        let n = 3_usize;
        let in_d = 4_usize;
        let out_d = 6_usize;
        let x = vec![1.0_f32; n * in_d];
        let w = vec![0.5_f32; out_d * in_d];
        let b = vec![0.1_f32; out_d];
        let out = linear(&x, &w, &b, in_d, out_d, n);
        assert_eq!(out.len(), n * out_d);
    }
}

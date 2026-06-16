//! Rotary Positional Embedding (RoPE) self-attention.
//!
//! Implements the rotary position encoding of Su et al. (2021),
//! "RoFormer: Enhanced Transformer with Rotary Position Embedding".
//!
//! Rather than adding a position embedding to the token features, RoPE
//! *rotates* consecutive pairs of channels in the query and key vectors by an
//! angle proportional to the token position:
//!
//! ```text
//!   [q_{2i}, q_{2i+1}]  ↦  R(θ_i · pos) · [q_{2i}, q_{2i+1}]
//! ```
//!
//! with per-pair frequencies `θ_i = base^{-2i / head_dim}`.  Because each `R`
//! is a planar rotation, RoPE is **norm-preserving**, and because
//! `⟨R(m)·q, R(n)·k⟩ = ⟨q, R(n − m)·k⟩`, the attention score depends only on the
//! **relative** position `n − m` — the headline property of RoPE.
//!
//! This module mirrors [`crate::score::unet_block::SelfAttentionBlock`] but
//! injects the rotation into `Q` and `K` (the values `V` are left untouched).

use crate::error::{GenError, GenResult};
use crate::score::unet_block::SelfAttentionBlock;

// ─── Local math helpers ──────────────────────────────────────────────────────────

/// Row-major matrix multiply `C[m×n] = A[m×k] · B[k×n]`.
fn matmul_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Per-token layer normalisation (mean/variance over the feature axis).
fn layer_norm_rows(x: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / (dim as f32);
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / (dim as f32);
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        let dst = &mut out[r * dim..(r + 1) * dim];
        for (o, &v) in dst.iter_mut().zip(row) {
            *o = (v - mean) * inv_std;
        }
    }
    out
}

// ─── RotaryEmbedding ─────────────────────────────────────────────────────────────

/// Precomputed rotary embedding for a single attention head.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    head_dim: usize,
    base: f32,
    /// Inverse frequencies `θ_i = base^{-2i / head_dim}`, length `head_dim / 2`.
    inv_freq: Vec<f32>,
}

impl RotaryEmbedding {
    /// Create a rotary embedding for vectors of length `head_dim` (must be even).
    ///
    /// `base` is the RoPE wavelength base (10000 in the original paper).
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] if `head_dim == 0`.
    /// * [`GenError::DimensionMismatch`] if `head_dim` is odd.
    pub fn new(head_dim: usize, base: f32) -> GenResult<Self> {
        if head_dim == 0 {
            return Err(GenError::EmptyInput("head_dim must be > 0"));
        }
        if head_dim % 2 != 0 {
            return Err(GenError::DimensionMismatch {
                expected: head_dim + 1,
                got: head_dim,
            });
        }
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| base.powf(-2.0 * (i as f32) / (head_dim as f32)))
            .collect();
        Ok(Self {
            head_dim,
            base,
            inv_freq,
        })
    }

    /// Rotate a single `head_dim`-length vector in place by `position`.
    ///
    /// Pair `(v[2i], v[2i+1])` is rotated by `position · θ_i`.
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] if `vec.len() != head_dim`.
    pub fn rotate_in_place(&self, vec: &mut [f32], position: usize) -> GenResult<()> {
        if vec.len() != self.head_dim {
            return Err(GenError::DimensionMismatch {
                expected: self.head_dim,
                got: vec.len(),
            });
        }
        let pos = position as f32;
        for (i, &theta) in self.inv_freq.iter().enumerate() {
            let (sin, cos) = (pos * theta).sin_cos();
            let a = vec[2 * i];
            let b = vec[2 * i + 1];
            vec[2 * i] = a * cos - b * sin;
            vec[2 * i + 1] = a * sin + b * cos;
        }
        Ok(())
    }

    /// Return a rotated copy of `vec`.
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] if `vec.len() != head_dim`.
    pub fn rotated(&self, vec: &[f32], position: usize) -> GenResult<Vec<f32>> {
        let mut out = vec.to_vec();
        self.rotate_in_place(&mut out, position)?;
        Ok(out)
    }

    /// Head dimensionality.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// RoPE wavelength base.
    #[must_use]
    pub fn base(&self) -> f32 {
        self.base
    }
}

// ─── RopeSelfAttention ───────────────────────────────────────────────────────────

/// Multi-head self-attention with rotary positional embedding on Q and K.
#[derive(Debug, Clone)]
pub struct RopeSelfAttention {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    scale: f32,
    rotary: RotaryEmbedding,
}

impl RopeSelfAttention {
    /// Create a RoPE self-attention block.
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] if a dimension is `0`.
    /// * [`GenError::DimensionMismatch`] if `embed_dim % num_heads != 0` or the
    ///   resulting `head_dim` is odd (RoPE rotates channel pairs).
    pub fn new(embed_dim: usize, num_heads: usize, rope_base: f32) -> GenResult<Self> {
        if embed_dim == 0 || num_heads == 0 {
            return Err(GenError::EmptyInput("dimensions must be > 0"));
        }
        if embed_dim % num_heads != 0 {
            return Err(GenError::DimensionMismatch {
                expected: embed_dim - embed_dim % num_heads,
                got: embed_dim,
            });
        }
        let head_dim = embed_dim / num_heads;
        let rotary = RotaryEmbedding::new(head_dim, rope_base)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self {
            num_heads,
            head_dim,
            embed_dim,
            scale,
            rotary,
        })
    }

    /// Forward pass with positions `0, 1, …, seq_len − 1`.
    ///
    /// See [`RopeSelfAttention::forward_with_offset`] for argument semantics.
    ///
    /// # Errors
    /// Propagates shape errors.
    pub fn forward(
        &self,
        x: &[f32],
        qkv_weight: &[f32],
        out_weight: &[f32],
        seq_len: usize,
    ) -> GenResult<Vec<f32>> {
        self.forward_with_offset(x, qkv_weight, out_weight, seq_len, 0)
    }

    /// Forward pass using positions `offset, offset+1, …, offset+seq_len−1`.
    ///
    /// * `x`          — input `[seq_len × embed_dim]`.
    /// * `qkv_weight` — fused QKV projection `[3·embed_dim × embed_dim]`.
    /// * `out_weight` — output projection `[embed_dim × embed_dim]`.
    ///
    /// Returns `[seq_len × embed_dim]` (residual added).
    ///
    /// Because RoPE scores depend only on relative position, the output is
    /// **invariant** to a common `offset` applied to all positions.
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] / [`GenError::WeightShapeMismatch`] on
    /// shape mismatch.
    pub fn forward_with_offset(
        &self,
        x: &[f32],
        qkv_weight: &[f32],
        out_weight: &[f32],
        seq_len: usize,
        offset: usize,
    ) -> GenResult<Vec<f32>> {
        if x.is_empty() {
            return Err(GenError::EmptyInput("x is empty"));
        }
        if x.len() != seq_len * self.embed_dim {
            return Err(GenError::DimensionMismatch {
                expected: seq_len * self.embed_dim,
                got: x.len(),
            });
        }
        if qkv_weight.len() != 3 * self.embed_dim * self.embed_dim {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![3 * self.embed_dim, self.embed_dim],
                input: vec![qkv_weight.len()],
            });
        }
        if out_weight.len() != self.embed_dim * self.embed_dim {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![self.embed_dim, self.embed_dim],
                input: vec![out_weight.len()],
            });
        }

        let x_norm = layer_norm_rows(x, seq_len, self.embed_dim);
        let qkv = matmul_nn(
            &x_norm,
            qkv_weight,
            seq_len,
            self.embed_dim,
            3 * self.embed_dim,
        );
        let q = &qkv[..seq_len * self.embed_dim];
        let k = &qkv[seq_len * self.embed_dim..2 * seq_len * self.embed_dim];
        let v = &qkv[2 * seq_len * self.embed_dim..];

        let attended = self.rope_attention(q, k, v, seq_len, offset)?;
        let out = matmul_nn(
            &attended,
            out_weight,
            seq_len,
            self.embed_dim,
            self.embed_dim,
        );
        Ok(x.iter().zip(&out).map(|(&xi, &oi)| xi + oi).collect())
    }

    /// Multi-head attention with RoPE applied to per-head Q and K slices.
    fn rope_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        offset: usize,
    ) -> GenResult<Vec<f32>> {
        let mut out = vec![0.0_f32; seq_len * self.embed_dim];
        for h in 0..self.num_heads {
            let head_off = h * self.head_dim;

            // Gather + rotate per-head Q, K; gather V (unrotated).
            let mut q_head = vec![0.0_f32; seq_len * self.head_dim];
            let mut k_head = vec![0.0_f32; seq_len * self.head_dim];
            let mut v_head = vec![0.0_f32; seq_len * self.head_dim];
            for t in 0..seq_len {
                let dst = t * self.head_dim;
                let src = t * self.embed_dim + head_off;
                q_head[dst..dst + self.head_dim].copy_from_slice(&q[src..src + self.head_dim]);
                k_head[dst..dst + self.head_dim].copy_from_slice(&k[src..src + self.head_dim]);
                v_head[dst..dst + self.head_dim].copy_from_slice(&v[src..src + self.head_dim]);
                let pos = offset + t;
                self.rotary
                    .rotate_in_place(&mut q_head[dst..dst + self.head_dim], pos)?;
                self.rotary
                    .rotate_in_place(&mut k_head[dst..dst + self.head_dim], pos)?;
            }

            // Scores, softmax, weighted value sum.
            let mut logits = vec![0.0_f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut acc = 0.0_f32;
                    for d in 0..self.head_dim {
                        acc += q_head[i * self.head_dim + d] * k_head[j * self.head_dim + d];
                    }
                    logits[i * seq_len + j] = acc * self.scale;
                }
                SelfAttentionBlock::softmax_row_pub(&mut logits[i * seq_len..(i + 1) * seq_len]);
            }
            for i in 0..seq_len {
                for d in 0..self.head_dim {
                    let mut acc = 0.0_f32;
                    for j in 0..seq_len {
                        acc += logits[i * seq_len + j] * v_head[j * self.head_dim + d];
                    }
                    out[i * self.embed_dim + head_off + d] += acc;
                }
            }
        }
        Ok(out)
    }

    /// Number of attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Per-head dimensionality.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Embedding dimensionality.
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Borrow the rotary embedding.
    #[must_use]
    pub fn rotary(&self) -> &RotaryEmbedding {
        &self.rotary
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    fn norm(a: &[f32]) -> f32 {
        dot(a, a).sqrt()
    }

    fn rand_vec(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn rope_preserves_norm() {
        let rope = RotaryEmbedding::new(8, 10000.0)
            .expect("creating RotaryEmbedding with valid even head_dim=8 should succeed");
        let mut rng = LcgRng::new(1);
        let q = rand_vec(&mut rng, 8);
        let before = norm(&q);
        for pos in [0_usize, 1, 7, 100] {
            let rotated = rope.rotated(&q, pos).expect("rotated should succeed");
            let after = norm(&rotated);
            assert!(
                (before - after).abs() < 1e-5,
                "RoPE must preserve norm: {before} vs {after} (pos={pos})"
            );
        }
    }

    #[test]
    fn rope_relative_position_dot_product() {
        // ⟨R(m)q, R(n)k⟩ depends only on (n − m): shifting both positions by the
        // same offset leaves the score unchanged.
        let rope = RotaryEmbedding::new(16, 10000.0).expect("new should succeed");
        let mut rng = LcgRng::new(2);
        let q = rand_vec(&mut rng, 16);
        let k = rand_vec(&mut rng, 16);
        let m = 3_usize;
        let n = 7_usize;
        let base_score = dot(
            &rope.rotated(&q, m).expect("rotated should succeed"),
            &rope.rotated(&k, n).expect("rotated should succeed"),
        );
        for s in [1_usize, 4, 20] {
            let shifted = dot(
                &rope.rotated(&q, m + s).expect("rotated should succeed"),
                &rope.rotated(&k, n + s).expect("rotated should succeed"),
            );
            assert!(
                (base_score - shifted).abs() < 1e-3,
                "score must depend only on relative position: {base_score} vs {shifted} (s={s})"
            );
        }
    }

    #[test]
    fn rope_zero_position_is_identity() {
        let rope = RotaryEmbedding::new(8, 10000.0).expect("new should succeed");
        let mut rng = LcgRng::new(3);
        let q = rand_vec(&mut rng, 8);
        let rotated = rope.rotated(&q, 0).expect("rotated should succeed");
        for (&a, &b) in q.iter().zip(&rotated) {
            assert!((a - b).abs() < 1e-6, "pos=0 should be identity");
        }
    }

    #[test]
    fn rope_odd_head_dim_errors() {
        assert!(RotaryEmbedding::new(7, 10000.0).is_err());
        assert!(RotaryEmbedding::new(0, 10000.0).is_err());
    }

    #[test]
    fn attention_output_shape_and_finite() {
        let attn = RopeSelfAttention::new(16, 4, 10000.0).expect("new should succeed");
        let mut rng = LcgRng::new(4);
        let x = rand_vec(&mut rng, 6 * 16);
        let qkv = rand_vec(&mut rng, 3 * 16 * 16);
        let out_w = rand_vec(&mut rng, 16 * 16);
        let y = attn
            .forward(&x, &qkv, &out_w, 6)
            .expect("forward should succeed");
        assert_eq!(y.len(), 6 * 16);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn attention_relative_position_invariance() {
        // Self-attention output is invariant to a global position offset, because
        // every pairwise score depends only on relative positions and V is not
        // rotated.
        let attn = RopeSelfAttention::new(16, 2, 10000.0).expect("new should succeed");
        let mut rng = LcgRng::new(5);
        let seq = 5;
        let x = rand_vec(&mut rng, seq * 16);
        let qkv = rand_vec(&mut rng, 3 * 16 * 16);
        let out_w = rand_vec(&mut rng, 16 * 16);
        let base = attn
            .forward(&x, &qkv, &out_w, seq)
            .expect("forward should succeed");
        for offset in [1_usize, 10, 50] {
            let shifted = attn
                .forward_with_offset(&x, &qkv, &out_w, seq, offset)
                .expect("value should be present");
            let err = base
                .iter()
                .zip(&shifted)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                err < 1e-4,
                "attention output must be offset-invariant: err={err} (offset={offset})"
            );
        }
    }

    #[test]
    fn attention_invalid_dims_error() {
        assert!(RopeSelfAttention::new(7, 4, 10000.0).is_err()); // not divisible
        // embed_dim/num_heads = 5 (odd) ⇒ RoPE pairing impossible.
        assert!(RopeSelfAttention::new(10, 2, 10000.0).is_err());
    }
}

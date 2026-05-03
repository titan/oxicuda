//! UNet building blocks for denoising score networks.
//!
//! Implements self-attention, cross-attention, and residual blocks
//! used in diffusion model architectures (e.g. Stable Diffusion's UNet).

use crate::error::{GenError, GenResult};

// ─── Math utilities ───────────────────────────────────────────────────────────

fn layer_norm(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    x.iter().map(|&v| (v - mean) * inv_std).collect()
}

fn silu_slice(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// Matrix multiplication: `C[m, n] = A[m, k] @ B[k, n]`.
/// Stored row-major.
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

/// Softmax over the last axis in-place.
fn softmax_row(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum.max(1e-10);
    for v in logits.iter_mut() {
        *v *= inv_sum;
    }
}

// ─── SelfAttentionBlock ───────────────────────────────────────────────────────

/// Multi-head self-attention block.
///
/// Computes: `Attn(Q, K, V) = softmax(QK^T / √d_head) * V`
/// where `Q = x @ W_Q`, `K = x @ W_K`, `V = x @ W_V`.
#[derive(Debug, Clone)]
pub struct SelfAttentionBlock {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    scale: f32,
}

impl SelfAttentionBlock {
    /// Create a new self-attention block.
    ///
    /// # Errors
    /// - `EmptyInput` if any dimension is 0
    /// - `DimensionMismatch` if `embed_dim % num_heads != 0`
    pub fn new(embed_dim: usize, num_heads: usize) -> GenResult<Self> {
        if embed_dim == 0 {
            return Err(GenError::EmptyInput("embed_dim must be > 0"));
        }
        if num_heads == 0 {
            return Err(GenError::EmptyInput("num_heads must be > 0"));
        }
        if embed_dim % num_heads != 0 {
            return Err(GenError::DimensionMismatch {
                expected: embed_dim - embed_dim % num_heads,
                got: embed_dim,
            });
        }
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self {
            num_heads,
            head_dim,
            embed_dim,
            scale,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `x`: Input of shape `[seq_len × embed_dim]`.
    /// - `qkv_weight`: QKV projection weights `[3*embed_dim × embed_dim]`.
    /// - `out_weight`: Output projection weights `[embed_dim × embed_dim]`.
    /// - `seq_len`: Sequence length.
    ///
    /// # Returns
    /// Output of shape `[seq_len × embed_dim]`.
    ///
    /// # Errors
    /// - `DimensionMismatch` on shape mismatch
    pub fn forward(
        &self,
        x: &[f32],
        qkv_weight: &[f32],
        out_weight: &[f32],
        seq_len: usize,
    ) -> GenResult<Vec<f32>> {
        if x.is_empty() {
            return Err(GenError::EmptyInput("x is empty"));
        }
        let expected_x = seq_len * self.embed_dim;
        if x.len() != expected_x {
            return Err(GenError::DimensionMismatch {
                expected: expected_x,
                got: x.len(),
            });
        }
        let expected_qkv = 3 * self.embed_dim * self.embed_dim;
        if qkv_weight.len() != expected_qkv {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![3 * self.embed_dim, self.embed_dim],
                input: vec![qkv_weight.len()],
            });
        }
        let expected_out = self.embed_dim * self.embed_dim;
        if out_weight.len() != expected_out {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![self.embed_dim, self.embed_dim],
                input: vec![out_weight.len()],
            });
        }
        // Layer norm
        let x_norm = layer_norm(x);
        // QKV projection: [seq × 3*embed] via matmul with [3*embed × embed]
        let qkv = matmul_nn(
            &x_norm,
            qkv_weight,
            seq_len,
            self.embed_dim,
            3 * self.embed_dim,
        );
        // Split into Q, K, V each [seq × embed]
        let q = &qkv[..seq_len * self.embed_dim];
        let k = &qkv[seq_len * self.embed_dim..2 * seq_len * self.embed_dim];
        let v = &qkv[2 * seq_len * self.embed_dim..];
        // Compute multi-head attention
        let attended = self.multi_head_attention(q, k, v, seq_len, seq_len)?;
        // Output projection
        let out = matmul_nn(
            &attended,
            out_weight,
            seq_len,
            self.embed_dim,
            self.embed_dim,
        );
        // Residual connection
        let result = x.iter().zip(&out).map(|(&xi, &oi)| xi + oi).collect();
        Ok(result)
    }

    /// Compute multi-head scaled dot-product attention.
    fn multi_head_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_len: usize,
        kv_len: usize,
    ) -> GenResult<Vec<f32>> {
        let mut out = vec![0.0_f32; q_len * self.embed_dim];
        for h in 0..self.num_heads {
            let head_off = h * self.head_dim;
            // Extract head slices (non-contiguous in general, but we stride manually)
            let mut q_head = vec![0.0_f32; q_len * self.head_dim];
            let mut k_head = vec![0.0_f32; kv_len * self.head_dim];
            let mut v_head = vec![0.0_f32; kv_len * self.head_dim];
            for t in 0..q_len {
                for d in 0..self.head_dim {
                    q_head[t * self.head_dim + d] = q[t * self.embed_dim + head_off + d];
                }
            }
            for t in 0..kv_len {
                for d in 0..self.head_dim {
                    k_head[t * self.head_dim + d] = k[t * self.embed_dim + head_off + d];
                    v_head[t * self.head_dim + d] = v[t * self.embed_dim + head_off + d];
                }
            }
            // Attention logits: Q @ K^T / sqrt(d_head) → [q_len × kv_len]
            let mut logits = vec![0.0_f32; q_len * kv_len];
            for i in 0..q_len {
                for j in 0..kv_len {
                    let mut acc = 0.0_f32;
                    for d in 0..self.head_dim {
                        acc += q_head[i * self.head_dim + d] * k_head[j * self.head_dim + d];
                    }
                    logits[i * kv_len + j] = acc * self.scale;
                }
            }
            // Row-wise softmax
            for i in 0..q_len {
                softmax_row(&mut logits[i * kv_len..(i + 1) * kv_len]);
            }
            // Weighted sum: logits @ V → [q_len × head_dim]
            for i in 0..q_len {
                for d in 0..self.head_dim {
                    let mut acc = 0.0_f32;
                    for j in 0..kv_len {
                        acc += logits[i * kv_len + j] * v_head[j * self.head_dim + d];
                    }
                    out[i * self.embed_dim + head_off + d] += acc;
                }
            }
        }
        Ok(out)
    }

    /// Return the number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Return the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Return the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Return the attention scale factor.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Expose the softmax utility for testing.
    pub fn softmax_row_pub(logits: &mut [f32]) {
        softmax_row(logits);
    }
}

// ─── CrossAttentionBlock ──────────────────────────────────────────────────────

/// Multi-head cross-attention block.
///
/// Queries come from `x` (e.g. image features) while keys and values
/// come from `context` (e.g. text embeddings).
#[derive(Debug, Clone)]
pub struct CrossAttentionBlock {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    context_dim: usize,
    scale: f32,
}

impl CrossAttentionBlock {
    /// Create a new cross-attention block.
    ///
    /// # Errors
    /// - `EmptyInput` if any dimension is 0
    /// - `DimensionMismatch` if `embed_dim % num_heads != 0`
    pub fn new(embed_dim: usize, context_dim: usize, num_heads: usize) -> GenResult<Self> {
        if embed_dim == 0 || context_dim == 0 {
            return Err(GenError::EmptyInput("dimensions must be > 0"));
        }
        if num_heads == 0 {
            return Err(GenError::EmptyInput("num_heads must be > 0"));
        }
        if embed_dim % num_heads != 0 {
            return Err(GenError::DimensionMismatch {
                expected: embed_dim - embed_dim % num_heads,
                got: embed_dim,
            });
        }
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self {
            num_heads,
            head_dim,
            embed_dim,
            context_dim,
            scale,
        })
    }

    /// Forward cross-attention pass.
    ///
    /// # Arguments
    /// - `x`: Query source, shape `[seq_len × embed_dim]`.
    /// - `context`: Key/value source, shape `[ctx_len × context_dim]`.
    /// - `seq_len`: Length of the query sequence.
    /// - `ctx_len`: Length of the context sequence.
    /// - `q_weight`: Query projection `[embed_dim × embed_dim]`.
    /// - `kv_weight`: Key/value projection `[2*embed_dim × context_dim]`.
    /// - `out_weight`: Output projection `[embed_dim × embed_dim]`.
    ///
    /// # Errors
    /// - `DimensionMismatch` on shape mismatch
    pub fn forward(
        &self,
        x: &[f32],
        context: &[f32],
        seq_len: usize,
        ctx_len: usize,
        q_weight: &[f32],
        kv_weight: &[f32],
        out_weight: &[f32],
    ) -> GenResult<Vec<f32>> {
        if x.is_empty() || context.is_empty() {
            return Err(GenError::EmptyInput("x or context is empty"));
        }
        let expected_x = seq_len * self.embed_dim;
        if x.len() != expected_x {
            return Err(GenError::DimensionMismatch {
                expected: expected_x,
                got: x.len(),
            });
        }
        let expected_ctx = ctx_len * self.context_dim;
        if context.len() != expected_ctx {
            return Err(GenError::DimensionMismatch {
                expected: expected_ctx,
                got: context.len(),
            });
        }
        // Q = x @ W_Q^T: [seq × embed]
        // W_Q: [embed × embed], so x: [seq × embed], W_Q: [embed × embed]
        let q = matmul_nn(x, q_weight, seq_len, self.embed_dim, self.embed_dim);
        // KV = ctx @ W_KV^T: [ctx × 2*embed]
        let kv = matmul_nn(
            context,
            kv_weight,
            ctx_len,
            self.context_dim,
            2 * self.embed_dim,
        );
        let k = &kv[..ctx_len * self.embed_dim];
        let v = &kv[ctx_len * self.embed_dim..];
        // Attention
        let self_attn = SelfAttentionBlock {
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            embed_dim: self.embed_dim,
            scale: self.scale,
        };
        let attended = self_attn.multi_head_attention(&q, k, v, seq_len, ctx_len)?;
        // Output projection
        let out = matmul_nn(
            &attended,
            out_weight,
            seq_len,
            self.embed_dim,
            self.embed_dim,
        );
        // Residual
        let result = x.iter().zip(&out).map(|(&xi, &oi)| xi + oi).collect();
        Ok(result)
    }

    /// Return the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Return the context dimension.
    pub fn context_dim(&self) -> usize {
        self.context_dim
    }

    /// Return the number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

// ─── UNetResBlock ─────────────────────────────────────────────────────────────

/// UNet residual block with SiLU activation and time embedding injection.
///
/// Architecture:
/// 1. `x → LayerNorm → SiLU → Linear → h`
/// 2. `time_emb → SiLU → Linear → scale/shift`
/// 3. `h = h * scale + shift`
/// 4. `h → LayerNorm → SiLU → Linear → h`
/// 5. `out = h + skip(x)`
#[derive(Debug, Clone)]
pub struct UNetResBlock {
    in_channels: usize,
    out_channels: usize,
    time_emb_dim: usize,
}

impl UNetResBlock {
    /// Create a new UNet residual block.
    pub fn new(in_channels: usize, out_channels: usize, time_emb_dim: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            time_emb_dim,
        }
    }

    /// Forward pass with SiLU activation.
    ///
    /// # Arguments
    /// - `x`: Input of shape `[batch × in_channels]`.
    /// - `time_emb`: Time embedding of shape `[batch × time_emb_dim]`.
    /// - `w1`: First linear weight `[out_channels × in_channels]`.
    /// - `w2`: Second linear weight `[out_channels × out_channels]`.
    /// - `wt`: Time embedding projection weight `[2*out_channels × time_emb_dim]`.
    ///
    /// # Returns
    /// Output of shape `[batch × out_channels]`.
    ///
    /// # Errors
    /// - `DimensionMismatch` on shape mismatch
    pub fn forward(
        &self,
        x: &[f32],
        time_emb: &[f32],
        w1: &[f32],
        w2: &[f32],
        wt: &[f32],
    ) -> GenResult<Vec<f32>> {
        if x.is_empty() {
            return Err(GenError::EmptyInput("x is empty"));
        }
        if x.len() % self.in_channels != 0 {
            return Err(GenError::DimensionMismatch {
                expected: x.len() - x.len() % self.in_channels,
                got: x.len(),
            });
        }
        let batch = x.len() / self.in_channels;
        let expected_temb = batch * self.time_emb_dim;
        if time_emb.len() != expected_temb {
            return Err(GenError::DimensionMismatch {
                expected: expected_temb,
                got: time_emb.len(),
            });
        }
        // Validate weight shapes
        if w1.len() != self.out_channels * self.in_channels {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![self.out_channels, self.in_channels],
                input: vec![w1.len()],
            });
        }
        if w2.len() != self.out_channels * self.out_channels {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![self.out_channels, self.out_channels],
                input: vec![w2.len()],
            });
        }
        if wt.len() != 2 * self.out_channels * self.time_emb_dim {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![2 * self.out_channels, self.time_emb_dim],
                input: vec![wt.len()],
            });
        }
        // First: LayerNorm → SiLU → Linear
        let h_norm = layer_norm(x);
        let h_act = silu_slice(&h_norm);
        let mut h = vec![0.0_f32; batch * self.out_channels];
        for b in 0..batch {
            for o in 0..self.out_channels {
                let mut acc = 0.0_f32;
                for i in 0..self.in_channels {
                    acc += h_act[b * self.in_channels + i] * w1[o * self.in_channels + i];
                }
                h[b * self.out_channels + o] = acc;
            }
        }
        // Time embedding: SiLU → Linear → scale, shift
        let te_act = silu_slice(time_emb);
        let mut scale_shift = vec![0.0_f32; batch * 2 * self.out_channels];
        for b in 0..batch {
            for o in 0..2 * self.out_channels {
                let mut acc = 0.0_f32;
                for t in 0..self.time_emb_dim {
                    acc += te_act[b * self.time_emb_dim + t] * wt[o * self.time_emb_dim + t];
                }
                scale_shift[b * 2 * self.out_channels + o] = acc;
            }
        }
        // Apply scale and shift: h = h * (1 + scale) + shift
        for b in 0..batch {
            for o in 0..self.out_channels {
                let scale = scale_shift[b * 2 * self.out_channels + o];
                let shift = scale_shift[b * 2 * self.out_channels + self.out_channels + o];
                h[b * self.out_channels + o] = h[b * self.out_channels + o] * (1.0 + scale) + shift;
            }
        }
        // Second: LayerNorm → SiLU → Linear
        let h_norm2 = layer_norm(&h);
        let h_act2 = silu_slice(&h_norm2);
        let mut h2 = vec![0.0_f32; batch * self.out_channels];
        for b in 0..batch {
            for o in 0..self.out_channels {
                let mut acc = 0.0_f32;
                for i in 0..self.out_channels {
                    acc += h_act2[b * self.out_channels + i] * w2[o * self.out_channels + i];
                }
                h2[b * self.out_channels + o] = acc;
            }
        }
        // Skip connection (project if needed)
        let skip: Vec<f32> = if self.in_channels == self.out_channels {
            x.to_vec()
        } else {
            let mut s = vec![0.0_f32; batch * self.out_channels];
            let min_ch = self.in_channels.min(self.out_channels);
            for b in 0..batch {
                for c in 0..min_ch {
                    s[b * self.out_channels + c] = x[b * self.in_channels + c];
                }
            }
            s
        };
        let out = h2.iter().zip(&skip).map(|(&a, &b)| a + b).collect();
        Ok(out)
    }

    #[cfg(test)]
    fn silu(x: &[f32]) -> Vec<f32> {
        silu_slice(x)
    }

    #[cfg(test)]
    fn layer_norm(x: &[f32]) -> Vec<f32> {
        layer_norm(x)
    }

    /// Return the input channel count.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Return the output channel count.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Return the time embedding dimension.
    pub fn time_emb_dim(&self) -> usize {
        self.time_emb_dim
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn self_attention_output_shape() {
        let attn = SelfAttentionBlock::new(16, 4).unwrap();
        let x = vec![0.1_f32; 8 * 16]; // seq=8, embed=16
        let qkv = vec![0.0_f32; 3 * 16 * 16];
        let out_w = vec![0.0_f32; 16 * 16];
        let out = attn.forward(&x, &qkv, &out_w, 8).unwrap();
        assert_eq!(out.len(), 8 * 16);
    }

    #[test]
    fn self_attention_residual() {
        // With zero weights, out = 0 + x (residual), so output = x
        let attn = SelfAttentionBlock::new(8, 2).unwrap();
        let x = vec![1.0_f32; 4 * 8]; // seq=4
        let qkv = vec![0.0_f32; 3 * 8 * 8];
        let out_w = vec![0.0_f32; 8 * 8];
        let out = attn.forward(&x, &qkv, &out_w, 4).unwrap();
        for (&o, &xi) in out.iter().zip(&x) {
            assert!((o - xi).abs() < EPS, "residual: {o} vs {xi}");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        SelfAttentionBlock::softmax_row_pub(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "softmax sum: {sum}");
    }

    #[test]
    fn softmax_non_negative() {
        let mut logits = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        SelfAttentionBlock::softmax_row_pub(&mut logits);
        assert!(
            logits.iter().all(|&v| v >= 0.0),
            "softmax values must be >= 0"
        );
    }

    #[test]
    fn cross_attention_output_shape() {
        let attn = CrossAttentionBlock::new(16, 32, 4).unwrap();
        let x = vec![0.1_f32; 4 * 16]; // seq=4, embed=16
        let ctx = vec![0.1_f32; 8 * 32]; // ctx=8, ctx_dim=32
        let q_w = vec![0.0_f32; 16 * 16];
        let kv_w = vec![0.0_f32; 2 * 16 * 32];
        let out_w = vec![0.0_f32; 16 * 16];
        let out = attn.forward(&x, &ctx, 4, 8, &q_w, &kv_w, &out_w).unwrap();
        assert_eq!(out.len(), 4 * 16);
    }

    #[test]
    fn cross_attention_residual() {
        let attn = CrossAttentionBlock::new(8, 16, 2).unwrap();
        let x = vec![1.5_f32; 3 * 8];
        let ctx = vec![0.0_f32; 5 * 16];
        let q_w = vec![0.0_f32; 8 * 8];
        let kv_w = vec![0.0_f32; 2 * 8 * 16];
        let out_w = vec![0.0_f32; 8 * 8];
        let out = attn.forward(&x, &ctx, 3, 5, &q_w, &kv_w, &out_w).unwrap();
        // With zero weights, projected is 0, residual = x
        for (&o, &xi) in out.iter().zip(&x) {
            assert!((o - xi).abs() < EPS, "residual mismatch: {o} vs {xi}");
        }
    }

    #[test]
    fn unet_resblock_output_shape() {
        let block = UNetResBlock::new(8, 16, 32);
        let x = vec![0.5_f32; 2 * 8]; // batch=2
        let time_emb = vec![0.1_f32; 2 * 32];
        let w1 = vec![0.0_f32; 16 * 8];
        let w2 = vec![0.0_f32; 16 * 16];
        let wt = vec![0.0_f32; 2 * 16 * 32];
        let out = block.forward(&x, &time_emb, &w1, &w2, &wt).unwrap();
        assert_eq!(out.len(), 2 * 16);
    }

    #[test]
    fn unet_resblock_skip_connection() {
        // With same in/out channels and zero weights, output should be input (skip)
        let block = UNetResBlock::new(8, 8, 16);
        let x = vec![2.0_f32; 8];
        let time_emb = vec![0.0_f32; 16];
        let w1 = vec![0.0_f32; 8 * 8];
        let w2 = vec![0.0_f32; 8 * 8];
        let wt = vec![0.0_f32; 2 * 8 * 16];
        let out = block.forward(&x, &time_emb, &w1, &w2, &wt).unwrap();
        for (&o, &xi) in out.iter().zip(&x) {
            assert!((o - xi).abs() < EPS, "skip mismatch: {o} vs {xi}");
        }
    }

    #[test]
    fn silu_positive_input() {
        // SiLU(x) = x / (1 + exp(-x)) ≈ x for large x
        let x = vec![10.0_f32];
        let out = UNetResBlock::silu(&x);
        assert!((out[0] - 10.0).abs() < 0.001, "SiLU(10) ≈ 10: {}", out[0]);
    }

    #[test]
    fn silu_zero_input() {
        let x = vec![0.0_f32];
        let out = UNetResBlock::silu(&x);
        assert_eq!(out[0], 0.0, "SiLU(0) = 0");
    }

    #[test]
    fn layer_norm_zero_output_for_constant() {
        // Constant input → variance = 0 → output = 0 (before scale/shift)
        let x = vec![5.0_f32; 8];
        let out = UNetResBlock::layer_norm(&x);
        for &v in &out {
            assert!(v.abs() < EPS, "layer_norm of constant should be ~0: {v}");
        }
    }

    #[test]
    fn self_attention_invalid_embed_dim() {
        // embed_dim % num_heads != 0 should fail
        assert!(SelfAttentionBlock::new(7, 4).is_err());
    }

    #[test]
    fn cross_attention_invalid_embed_dim() {
        assert!(CrossAttentionBlock::new(7, 32, 4).is_err());
    }
}

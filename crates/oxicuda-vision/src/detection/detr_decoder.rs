//! DETR (DEtection TRansformer) decoder.
//!
//! Implements the DETR decoder as described in "End-to-End Object Detection with
//! Transformers" (Carion et al., 2020).  Each decoder layer applies:
//!
//! 1. **Self-attention** over the object query embeddings (pre-norm).
//! 2. **Cross-attention** from object queries to encoder memory (pre-norm).
//! 3. **Feed-forward network** (two-layer MLP with GELU, pre-norm).
//!
//! The decoder stacks `depth` such layers sequentially.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

// ─── DetrConfig ───────────────────────────────────────────────────────────────

/// DETR decoder hyper-parameters.
#[derive(Debug, Clone)]
pub struct DetrConfig {
    /// Number of object query vectors.
    pub n_queries: usize,
    /// Embedding dimension for all tokens (queries and encoder features).
    pub embed_dim: usize,
    /// Number of attention heads (must divide `embed_dim`).
    pub n_heads: usize,
    /// Number of decoder layers.
    pub depth: usize,
    /// MLP expansion factor: `mlp_dim = mlp_ratio * embed_dim`.
    pub mlp_ratio: usize,
}

impl DetrConfig {
    /// Construct a validated `DetrConfig`.
    ///
    /// # Errors
    /// - `InvalidEmbedDim` if `embed_dim == 0`.
    /// - `InvalidNumHeads` if `n_heads == 0`.
    /// - `HeadDimMismatch` if `embed_dim % n_heads != 0`.
    /// - `DimensionMismatch` if `n_queries == 0`, `depth == 0`, or `mlp_ratio == 0`.
    pub fn new(
        n_queries: usize,
        embed_dim: usize,
        n_heads: usize,
        depth: usize,
        mlp_ratio: usize,
    ) -> VisionResult<Self> {
        if embed_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(embed_dim));
        }
        if n_heads == 0 {
            return Err(VisionError::InvalidNumHeads(n_heads));
        }
        if embed_dim % n_heads != 0 {
            return Err(VisionError::HeadDimMismatch { n_heads, embed_dim });
        }
        if n_queries == 0 {
            return Err(VisionError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if depth == 0 {
            return Err(VisionError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if mlp_ratio == 0 {
            return Err(VisionError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        Ok(Self {
            n_queries,
            embed_dim,
            n_heads,
            depth,
            mlp_ratio,
        })
    }

    /// A tiny configuration for unit tests.
    ///
    /// `n_queries=4, embed_dim=32, n_heads=4, depth=1, mlp_ratio=4`.
    pub fn tiny() -> Self {
        Self {
            n_queries: 4,
            embed_dim: 32,
            n_heads: 4,
            depth: 1,
            mlp_ratio: 4,
        }
    }

    /// MLP hidden dimension.
    #[inline]
    pub fn mlp_dim(&self) -> usize {
        self.mlp_ratio * self.embed_dim
    }

    /// Per-head dimension.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.n_heads
    }
}

// ─── DetrDecoderLayerWeights ──────────────────────────────────────────────────

/// All learnable weights for a single DETR decoder layer.
pub struct DetrDecoderLayerWeights {
    // ── Self-attention (queries attend to queries) ────────────────────────────
    /// Fused QKV projection: `[3 × embed_dim × embed_dim]`.
    pub self_qkv_weight: Vec<f32>,
    /// Fused QKV bias: `[3 × embed_dim]`.
    pub self_qkv_bias: Vec<f32>,
    /// Output projection: `[embed_dim × embed_dim]`.
    pub self_out_weight: Vec<f32>,
    /// Output projection bias: `[embed_dim]`.
    pub self_out_bias: Vec<f32>,

    // ── Cross-attention (queries attend to encoder memory) ────────────────────
    /// Query projection: `[embed_dim × embed_dim]`.
    pub cross_q_weight: Vec<f32>,
    /// Query projection bias: `[embed_dim]`.
    pub cross_q_bias: Vec<f32>,
    /// Fused Key+Value projection from encoder: `[2 × embed_dim × embed_dim]`.
    pub cross_kv_weight: Vec<f32>,
    /// Fused KV bias: `[2 × embed_dim]`.
    pub cross_kv_bias: Vec<f32>,
    /// Cross-attention output projection: `[embed_dim × embed_dim]`.
    pub cross_out_weight: Vec<f32>,
    /// Cross-attention output bias: `[embed_dim]`.
    pub cross_out_bias: Vec<f32>,

    // ── Feed-forward network ─────────────────────────────────────────────────
    /// FFN first layer: `[mlp_dim × embed_dim]`.
    pub ffn1_weight: Vec<f32>,
    /// FFN first layer bias: `[mlp_dim]`.
    pub ffn1_bias: Vec<f32>,
    /// FFN second layer: `[embed_dim × mlp_dim]`.
    pub ffn2_weight: Vec<f32>,
    /// FFN second layer bias: `[embed_dim]`.
    pub ffn2_bias: Vec<f32>,

    // ── Layer normalisation (three norms per layer) ───────────────────────────
    /// LN after self-attention: scale `[embed_dim]`.
    pub ln1_weight: Vec<f32>,
    /// LN after self-attention: bias `[embed_dim]`.
    pub ln1_bias: Vec<f32>,
    /// LN before cross-attention: scale `[embed_dim]`.
    pub ln2_weight: Vec<f32>,
    /// LN before cross-attention: bias `[embed_dim]`.
    pub ln2_bias: Vec<f32>,
    /// LN before FFN: scale `[embed_dim]`.
    pub ln3_weight: Vec<f32>,
    /// LN before FFN: bias `[embed_dim]`.
    pub ln3_bias: Vec<f32>,
}

impl DetrDecoderLayerWeights {
    /// Xavier-style default initialisation.
    ///
    /// Attention/FFN weights: N(0, 1/√embed_dim); biases: zeros;
    /// LayerNorm weights: ones; biases: zeros.
    pub fn default_init(cfg: &DetrConfig, rng: &mut LcgRng) -> Self {
        let e = cfg.embed_dim;
        let mlp = cfg.mlp_dim();
        let scale = 1.0_f32 / (e as f32).sqrt();

        let fill_scaled = |rng: &mut LcgRng, n: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= scale;
            }
            v
        };

        // Self-attention
        let self_qkv_weight = fill_scaled(rng, 3 * e * e);
        let self_qkv_bias = vec![0.0f32; 3 * e];
        let self_out_weight = fill_scaled(rng, e * e);
        let self_out_bias = vec![0.0f32; e];

        // Cross-attention
        let cross_q_weight = fill_scaled(rng, e * e);
        let cross_q_bias = vec![0.0f32; e];
        let cross_kv_weight = fill_scaled(rng, 2 * e * e);
        let cross_kv_bias = vec![0.0f32; 2 * e];
        let cross_out_weight = fill_scaled(rng, e * e);
        let cross_out_bias = vec![0.0f32; e];

        // FFN
        let ffn1_weight = fill_scaled(rng, mlp * e);
        let ffn1_bias = vec![0.0f32; mlp];
        let ffn2_weight = fill_scaled(rng, e * mlp);
        let ffn2_bias = vec![0.0f32; e];

        // Layer norms
        let ln1_weight = vec![1.0f32; e];
        let ln1_bias = vec![0.0f32; e];
        let ln2_weight = vec![1.0f32; e];
        let ln2_bias = vec![0.0f32; e];
        let ln3_weight = vec![1.0f32; e];
        let ln3_bias = vec![0.0f32; e];

        Self {
            self_qkv_weight,
            self_qkv_bias,
            self_out_weight,
            self_out_bias,
            cross_q_weight,
            cross_q_bias,
            cross_kv_weight,
            cross_kv_bias,
            cross_out_weight,
            cross_out_bias,
            ffn1_weight,
            ffn1_bias,
            ffn2_weight,
            ffn2_bias,
            ln1_weight,
            ln1_bias,
            ln2_weight,
            ln2_bias,
            ln3_weight,
            ln3_bias,
        }
    }
}

// ─── DetrDecoderLayer ─────────────────────────────────────────────────────────

/// A single DETR decoder layer.
pub struct DetrDecoderLayer {
    /// Decoder configuration (n_queries, embed_dim, n_heads, …).
    pub config: DetrConfig,
    /// Learned weights for this layer.
    pub weights: DetrDecoderLayerWeights,
}

impl DetrDecoderLayer {
    /// Construct a new decoder layer with Xavier-initialised weights.
    pub fn new(cfg: DetrConfig, rng: &mut LcgRng) -> Self {
        let weights = DetrDecoderLayerWeights::default_init(&cfg, rng);
        Self {
            config: cfg,
            weights,
        }
    }

    /// Forward pass for one decoder layer.
    ///
    /// Pre-norm residual scheme:
    /// ```text
    /// q1  = self_attn(LN1(queries)) + queries
    /// q2  = cross_attn(LN2(q1), key=encoder, val=encoder) + q1
    /// out = FFN(LN3(q2)) + q2
    /// ```
    ///
    /// # Parameters
    /// - `queries`:       flat `[n_queries × embed_dim]`.
    /// - `encoder_feats`: flat `[n_enc_tokens × embed_dim]`.
    /// - `n_enc_tokens`:  number of encoder feature tokens.
    ///
    /// # Returns
    /// Updated queries: flat `[n_queries × embed_dim]`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if input tensor lengths are inconsistent.
    /// - `NonFinite` if NaN/Inf appear in attention output.
    pub fn forward(
        &self,
        queries: &[f32],
        encoder_feats: &[f32],
        n_enc_tokens: usize,
    ) -> VisionResult<Vec<f32>> {
        let e = self.config.embed_dim;
        let nq = self.config.n_queries;
        let nh = self.config.n_heads;
        let w = &self.weights;

        // Validate input sizes.
        let expected_q = nq * e;
        if queries.len() != expected_q {
            return Err(VisionError::DimensionMismatch {
                expected: expected_q,
                got: queries.len(),
            });
        }
        let expected_enc = n_enc_tokens * e;
        if encoder_feats.len() != expected_enc {
            return Err(VisionError::DimensionMismatch {
                expected: expected_enc,
                got: encoder_feats.len(),
            });
        }
        if n_enc_tokens == 0 {
            return Err(VisionError::EmptyInput("encoder features"));
        }

        // ── Step 1: Self-attention ────────────────────────────────────────────
        // Pre-norm: LN1(queries)
        let queries_normed = layer_norm(queries, &w.ln1_weight, &w.ln1_bias, nq, e, 1e-5);
        // Self-attn: Q=K=V=queries_normed
        let sa_out = mhsa_self(
            &queries_normed,
            nq,
            e,
            nh,
            &w.self_qkv_weight,
            &w.self_qkv_bias,
            &w.self_out_weight,
            &w.self_out_bias,
        )?;
        // Residual 1: queries + self_attn_out
        let q1: Vec<f32> = queries
            .iter()
            .zip(sa_out.iter())
            .map(|(a, b)| a + b)
            .collect();

        // ── Step 2: Cross-attention ───────────────────────────────────────────
        // Pre-norm: LN2(q1)
        let q1_normed = layer_norm(&q1, &w.ln2_weight, &w.ln2_bias, nq, e, 1e-5);
        // Cross-attn: Q from normed queries, K/V from encoder
        let ca_out = mhsa_cross(
            &q1_normed,
            nq,
            encoder_feats,
            n_enc_tokens,
            e,
            nh,
            &w.cross_q_weight,
            &w.cross_q_bias,
            &w.cross_kv_weight,
            &w.cross_kv_bias,
            &w.cross_out_weight,
            &w.cross_out_bias,
        )?;
        // Residual 2: q1 + cross_attn_out
        let q2: Vec<f32> = q1.iter().zip(ca_out.iter()).map(|(a, b)| a + b).collect();

        // ── Step 3: FFN ───────────────────────────────────────────────────────
        // Pre-norm: LN3(q2)
        let q2_normed = layer_norm(&q2, &w.ln3_weight, &w.ln3_bias, nq, e, 1e-5);
        let mlp_dim = self.config.mlp_dim();
        // Linear1 → GELU
        let ffn_mid = linear(&q2_normed, &w.ffn1_weight, &w.ffn1_bias, e, mlp_dim);
        let ffn_mid: Vec<f32> = ffn_mid.iter().map(|&v| gelu_approx(v)).collect();
        // Linear2
        let ffn_out = linear(&ffn_mid, &w.ffn2_weight, &w.ffn2_bias, mlp_dim, e);
        // Residual 3: q2 + ffn_out
        let out: Vec<f32> = q2.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

        Ok(out)
    }
}

// ─── DetrDecoder ─────────────────────────────────────────────────────────────

/// Multi-layer DETR decoder: stacks `config.depth` decoder layers.
pub struct DetrDecoder {
    /// Decoder layers in order of application.
    pub layers: Vec<DetrDecoderLayer>,
}

impl DetrDecoder {
    /// Build a new `DetrDecoder` with `cfg.depth` layers, all Xavier-initialised.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `cfg.depth == 0`.
    /// - Propagates errors from `DetrConfig` validation (via cloning).
    pub fn new(cfg: DetrConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        if cfg.depth == 0 {
            return Err(VisionError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let depth = cfg.depth;
        let mut layers = Vec::with_capacity(depth);
        for _ in 0..depth {
            layers.push(DetrDecoderLayer::new(cfg.clone(), rng));
        }
        Ok(Self { layers })
    }

    /// Apply all decoder layers in sequence.
    ///
    /// # Parameters
    /// - `queries`:       flat `[n_queries × embed_dim]`.
    /// - `encoder_feats`: flat `[n_enc_tokens × embed_dim]`.
    /// - `n_enc_tokens`:  number of encoder memory tokens.
    ///
    /// # Returns
    /// Final queries: flat `[n_queries × embed_dim]`.
    pub fn forward(
        &self,
        queries: &[f32],
        encoder_feats: &[f32],
        n_enc_tokens: usize,
    ) -> VisionResult<Vec<f32>> {
        let mut current = queries.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current, encoder_feats, n_enc_tokens)?;
        }
        Ok(current)
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Per-row layer normalisation.
///
/// For each of `n` rows of length `d`:
/// ```text
/// out[i, j] = (x[i, j] - mean_i) / sqrt(var_i + eps) * weight[j] + bias[j]
/// ```
fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], n: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        let o = &mut out[i * d..(i + 1) * d];
        for j in 0..d {
            o[j] = (row[j] - mean) * inv_std * weight[j] + bias[j];
        }
    }
    out
}

/// Dense linear transform: `y = x W^T + b`.
///
/// - `x`: `[batch × n_in]`.
/// - `w`: `[n_out × n_in]`.
/// - `b`: `[n_out]`.
///
/// Returns `[batch × n_out]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let batch = x.len() / n_in;
    let mut out = vec![0.0f32; batch * n_out];
    for bi in 0..batch {
        let xrow = &x[bi * n_in..(bi + 1) * n_in];
        let orow = &mut out[bi * n_out..(bi + 1) * n_out];
        for oi in 0..n_out {
            let wrow = &w[oi * n_in..(oi + 1) * n_in];
            let mut acc = b[oi];
            for k in 0..n_in {
                acc += xrow[k] * wrow[k];
            }
            orow[oi] = acc;
        }
    }
    out
}

/// GELU activation via tanh approximation.
///
/// ```text
/// GELU(x) ≈ x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// ```
#[inline]
fn gelu_approx(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    x * 0.5 * (1.0 + inner.tanh())
}

/// Row-wise softmax with max subtraction for numerical stability.
fn softmax_rows(logits: &mut [f32], n_rows: usize, n_cols: usize) {
    for i in 0..n_rows {
        let row = &mut logits[i * n_cols..(i + 1) * n_cols];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Multi-head **self**-attention: Q, K, V all from the same token sequence.
///
/// Uses a fused `[3 * embed_dim × embed_dim]` QKV projection matrix.
#[allow(clippy::too_many_arguments)]
fn mhsa_self(
    tokens: &[f32],
    n_tokens: usize,
    embed_dim: usize,
    n_heads: usize,
    qkv_weight: &[f32],
    qkv_bias: &[f32],
    out_weight: &[f32],
    out_bias: &[f32],
) -> VisionResult<Vec<f32>> {
    let head_dim = embed_dim / n_heads;
    // Fused QKV projection: [n_tokens × 3*embed_dim]
    let qkv = linear(tokens, qkv_weight, qkv_bias, embed_dim, 3 * embed_dim);

    // Split into Q, K, V each [n_tokens × embed_dim]
    let mut q = vec![0.0f32; n_tokens * embed_dim];
    let mut k = vec![0.0f32; n_tokens * embed_dim];
    let mut v = vec![0.0f32; n_tokens * embed_dim];
    for t in 0..n_tokens {
        let src = &qkv[t * 3 * embed_dim..(t + 1) * 3 * embed_dim];
        q[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&src[..embed_dim]);
        k[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&src[embed_dim..2 * embed_dim]);
        v[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&src[2 * embed_dim..]);
    }

    compute_attention(
        &q, n_tokens, &k, n_tokens, &v, embed_dim, n_heads, head_dim, out_weight, out_bias,
    )
}

/// Multi-head **cross**-attention: Q from queries, K/V from encoder memory.
///
/// `q_weight`: `[embed_dim × embed_dim]`
/// `kv_weight`: `[2 * embed_dim × embed_dim]` (first half = K, second half = V)
#[allow(clippy::too_many_arguments)]
fn mhsa_cross(
    queries: &[f32],
    n_queries: usize,
    encoder: &[f32],
    n_enc: usize,
    embed_dim: usize,
    n_heads: usize,
    q_weight: &[f32],
    q_bias: &[f32],
    kv_weight: &[f32],
    kv_bias: &[f32],
    out_weight: &[f32],
    out_bias: &[f32],
) -> VisionResult<Vec<f32>> {
    let head_dim = embed_dim / n_heads;

    // Q projection: [n_queries × embed_dim]
    let q = linear(queries, q_weight, q_bias, embed_dim, embed_dim);

    // KV fused projection: [n_enc × 2*embed_dim]
    let kv = linear(encoder, kv_weight, kv_bias, embed_dim, 2 * embed_dim);

    // Split KV into K and V each [n_enc × embed_dim]
    let mut k = vec![0.0f32; n_enc * embed_dim];
    let mut v = vec![0.0f32; n_enc * embed_dim];
    for t in 0..n_enc {
        let src = &kv[t * 2 * embed_dim..(t + 1) * 2 * embed_dim];
        k[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&src[..embed_dim]);
        v[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&src[embed_dim..]);
    }

    compute_attention(
        &q, n_queries, &k, n_enc, &v, embed_dim, n_heads, head_dim, out_weight, out_bias,
    )
}

/// Core scaled dot-product attention computation.
///
/// Given already-projected Q `[n_q × embed_dim]`, K `[n_k × embed_dim]`,
/// V `[n_k × embed_dim]`, computes:
/// ```text
/// scores = Q @ K^T / sqrt(head_dim)  [n_q × n_k] per head
/// attn   = softmax(scores) @ V
/// out    = concat(attn_heads) @ out_weight + out_bias
/// ```
#[allow(clippy::too_many_arguments)]
fn compute_attention(
    q: &[f32],
    n_q: usize,
    k: &[f32],
    n_k: usize,
    v: &[f32],
    embed_dim: usize,
    n_heads: usize,
    head_dim: usize,
    out_weight: &[f32],
    out_bias: &[f32],
) -> VisionResult<Vec<f32>> {
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut concat = vec![0.0f32; n_q * embed_dim];
    let mut scores = vec![0.0f32; n_q * n_k];

    for h in 0..n_heads {
        let hd_off = h * head_dim;

        // Compute scores[i, j] = scale * dot(Q[i, h*hd..], K[j, h*hd..])
        for i in 0..n_q {
            for j in 0..n_k {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * embed_dim + hd_off + d] * k[j * embed_dim + hd_off + d];
                }
                scores[i * n_k + j] = dot * scale;
            }
        }

        // Row-wise softmax over keys
        softmax_rows(&mut scores, n_q, n_k);

        // Weighted value sum: out[i, h*hd + d] = Σ_j scores[i,j] * V[j, h*hd + d]
        for i in 0..n_q {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..n_k {
                    acc += scores[i * n_k + j] * v[j * embed_dim + hd_off + d];
                }
                concat[i * embed_dim + hd_off + d] = acc;
            }
        }
    }

    let out = linear(&concat, out_weight, out_bias, embed_dim, embed_dim);

    if out.iter().any(|v| !v.is_finite()) {
        return Err(VisionError::NonFinite("DETR decoder attention output"));
    }

    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── DetrConfig ─────────────────────────────────────────────────────────────

    #[test]
    fn detr_config_tiny() {
        let cfg = DetrConfig::tiny();
        assert_eq!(cfg.n_queries, 4);
        assert_eq!(cfg.embed_dim, 32);
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.depth, 1);
        assert_eq!(cfg.mlp_ratio, 4);
        assert_eq!(cfg.mlp_dim(), 128);
        assert_eq!(cfg.head_dim(), 8);
    }

    #[test]
    fn detr_config_invalid_embed_dim_zero() {
        let r = DetrConfig::new(4, 0, 4, 1, 4);
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(0))));
    }

    #[test]
    fn detr_config_invalid_heads_zero() {
        let r = DetrConfig::new(4, 32, 0, 1, 4);
        assert!(matches!(r, Err(VisionError::InvalidNumHeads(0))));
    }

    #[test]
    fn detr_config_head_dim_mismatch() {
        let r = DetrConfig::new(4, 32, 3, 1, 4); // 32 % 3 != 0
        assert!(matches!(r, Err(VisionError::HeadDimMismatch { .. })));
    }

    #[test]
    fn detr_config_zero_queries_errors() {
        let r = DetrConfig::new(0, 32, 4, 1, 4);
        assert!(r.is_err());
    }

    // ── Single layer forward ───────────────────────────────────────────────────

    #[test]
    fn single_layer_forward_shape() {
        let mut rng = make_rng();
        let cfg = DetrConfig::tiny();
        let nq = cfg.n_queries;
        let e = cfg.embed_dim;
        let layer = DetrDecoderLayer::new(cfg, &mut rng);

        let queries = vec![0.1f32; nq * e];
        let encoder = vec![0.2f32; 8 * e]; // 8 encoder tokens
        let out = layer.forward(&queries, &encoder, 8).expect("forward ok");

        assert_eq!(out.len(), nq * e, "output shape [n_queries × embed_dim]");
    }

    #[test]
    fn single_layer_forward_finite() {
        let mut rng = make_rng();
        let cfg = DetrConfig::tiny();
        let nq = cfg.n_queries;
        let e = cfg.embed_dim;
        let layer = DetrDecoderLayer::new(cfg, &mut rng);

        let mut queries = vec![0.0f32; nq * e];
        rng.fill_normal(&mut queries);
        let mut encoder = vec![0.0f32; 16 * e];
        rng.fill_normal(&mut encoder);

        let out = layer.forward(&queries, &encoder, 16).expect("forward ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite in output");
    }

    #[test]
    fn single_layer_forward_wrong_query_size_errors() {
        let mut rng = make_rng();
        let cfg = DetrConfig::tiny();
        let e = cfg.embed_dim;
        let layer = DetrDecoderLayer::new(cfg, &mut rng);

        // Provide wrong number of elements for queries
        let queries = vec![0.0f32; 3 * e]; // should be 4 * e
        let encoder = vec![0.0f32; 8 * e];
        let r = layer.forward(&queries, &encoder, 8);
        assert!(
            matches!(r, Err(VisionError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }

    #[test]
    fn single_layer_forward_empty_encoder_errors() {
        let mut rng = make_rng();
        let cfg = DetrConfig::tiny();
        let nq = cfg.n_queries;
        let e = cfg.embed_dim;
        let layer = DetrDecoderLayer::new(cfg, &mut rng);

        let queries = vec![0.0f32; nq * e];
        let r = layer.forward(&queries, &[], 0);
        assert!(r.is_err(), "expected error for empty encoder");
    }

    // ── Multi-layer decoder ────────────────────────────────────────────────────

    #[test]
    fn multi_layer_decoder_forward_shape() {
        let mut rng = make_rng();
        let cfg = DetrConfig::new(4, 32, 4, 3, 4).expect("valid config");
        let nq = cfg.n_queries;
        let e = cfg.embed_dim;
        let decoder = DetrDecoder::new(cfg, &mut rng).expect("valid decoder");

        let queries = vec![0.1f32; nq * e];
        let encoder = vec![0.2f32; 12 * e];
        let out = decoder
            .forward(&queries, &encoder, 12)
            .expect("multi-layer ok");

        assert_eq!(out.len(), nq * e, "multi-layer output shape preserved");
    }

    #[test]
    fn multi_layer_decoder_forward_finite() {
        let mut rng = make_rng();
        let cfg = DetrConfig::new(8, 32, 4, 2, 4).expect("valid config");
        let nq = cfg.n_queries;
        let e = cfg.embed_dim;
        let decoder = DetrDecoder::new(cfg, &mut rng).expect("valid decoder");

        let mut queries = vec![0.0f32; nq * e];
        rng.fill_normal(&mut queries);
        let mut encoder = vec![0.0f32; 6 * e];
        rng.fill_normal(&mut encoder);

        let out = decoder.forward(&queries, &encoder, 6).expect("forward ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite in multi-layer output"
        );
    }

    // ── layer_norm ────────────────────────────────────────────────────────────

    #[test]
    fn layer_norm_constant_row_is_zero() {
        let x = vec![5.0f32; 32];
        let w = vec![1.0f32; 32];
        let b = vec![0.0f32; 32];
        let out = layer_norm(&x, &w, &b, 1, 32, 1e-5);
        for v in &out {
            assert!(v.abs() < 1e-5, "expected near-zero, got {v}");
        }
    }

    // ── gelu_approx ───────────────────────────────────────────────────────────

    #[test]
    fn gelu_zero() {
        assert!((gelu_approx(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_large_pos() {
        assert!((gelu_approx(10.0) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn gelu_large_neg() {
        assert!(gelu_approx(-10.0).abs() < 1e-3);
    }
}

//! ViT block: pre-norm multi-head self-attention + MLP with GELU.
//!
//! Layout for all weight tensors is row-major C-contiguous `Vec<f32>`.
//!
//! ## Forward pass (pre-norm variant)
//! ```text
//! h  = LN1(x)
//! h  = MHSA(h) + x           (residual 1)
//! h2 = LN2(h)
//! out = MLP(h2) + h           (residual 2)
//! ```

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a single ViT transformer block.
#[derive(Debug, Clone, PartialEq)]
pub struct ViTBlockConfig {
    /// Token embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads. Must divide `embed_dim`.
    pub n_heads: usize,
    /// MLP hidden-dim multiplier: `mlp_dim = mlp_ratio * embed_dim`.
    pub mlp_ratio: usize,
}

impl ViTBlockConfig {
    /// Create and validate a `ViTBlockConfig`.
    ///
    /// # Errors
    /// - `embed_dim == 0` → `InvalidEmbedDim`
    /// - `n_heads == 0`   → `InvalidNumHeads`
    /// - `embed_dim % n_heads != 0` → `HeadDimMismatch`
    pub fn new(embed_dim: usize, n_heads: usize, mlp_ratio: usize) -> VisionResult<Self> {
        if embed_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(embed_dim));
        }
        if n_heads == 0 {
            return Err(VisionError::InvalidNumHeads(n_heads));
        }
        if embed_dim % n_heads != 0 {
            return Err(VisionError::HeadDimMismatch { n_heads, embed_dim });
        }
        Ok(Self {
            embed_dim,
            n_heads,
            mlp_ratio,
        })
    }

    /// Dimension per attention head.
    #[must_use]
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.n_heads
    }

    /// MLP hidden dimension.
    #[must_use]
    #[inline]
    pub fn mlp_dim(&self) -> usize {
        self.mlp_ratio * self.embed_dim
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Learnable weights for one ViT transformer block.
///
/// All tensors are flat row-major `Vec<f32>`.
pub struct ViTBlockWeights {
    /// QKV projection kernel: `[3 * embed_dim, embed_dim]` stored flat.
    /// Equivalent to concatenating W_Q, W_K, W_V along the output-dim axis.
    pub qkv_weight: Vec<f32>,
    /// QKV projection bias: `[3 * embed_dim]`.
    pub qkv_bias: Vec<f32>,

    /// Output projection kernel: `[embed_dim, embed_dim]`.
    pub out_weight: Vec<f32>,
    /// Output projection bias: `[embed_dim]`.
    pub out_bias: Vec<f32>,

    /// MLP first linear kernel: `[mlp_dim, embed_dim]`.
    pub mlp1_weight: Vec<f32>,
    /// MLP first linear bias: `[mlp_dim]`.
    pub mlp1_bias: Vec<f32>,

    /// MLP second linear kernel: `[embed_dim, mlp_dim]`.
    pub mlp2_weight: Vec<f32>,
    /// MLP second linear bias: `[embed_dim]`.
    pub mlp2_bias: Vec<f32>,

    /// LayerNorm 1 scale: `[embed_dim]` (init 1).
    pub ln1_weight: Vec<f32>,
    /// LayerNorm 1 bias: `[embed_dim]` (init 0).
    pub ln1_bias: Vec<f32>,

    /// LayerNorm 2 scale: `[embed_dim]` (init 1).
    pub ln2_weight: Vec<f32>,
    /// LayerNorm 2 bias: `[embed_dim]` (init 0).
    pub ln2_bias: Vec<f32>,
}

impl ViTBlockWeights {
    /// Xavier-style default initialisation.
    ///
    /// - Attention & MLP weights: N(0, 1/√embed_dim)
    /// - Biases: zeros
    /// - LayerNorm weights: ones; biases: zeros
    pub fn default_init(cfg: &ViTBlockConfig, rng: &mut LcgRng) -> Self {
        let e = cfg.embed_dim;
        let mlp = cfg.mlp_dim();
        let scale = 1.0 / (e as f32).sqrt();

        let fill_scaled = |rng: &mut LcgRng, n: usize, sc: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= sc;
            }
            v
        };

        let qkv_weight = fill_scaled(rng, 3 * e * e, scale);
        let qkv_bias = vec![0.0f32; 3 * e];
        let out_weight = fill_scaled(rng, e * e, scale);
        let out_bias = vec![0.0f32; e];
        let mlp1_weight = fill_scaled(rng, mlp * e, scale);
        let mlp1_bias = vec![0.0f32; mlp];
        let mlp2_weight = fill_scaled(rng, e * mlp, scale);
        let mlp2_bias = vec![0.0f32; e];
        let ln1_weight = vec![1.0f32; e];
        let ln1_bias = vec![0.0f32; e];
        let ln2_weight = vec![1.0f32; e];
        let ln2_bias = vec![0.0f32; e];

        Self {
            qkv_weight,
            qkv_bias,
            out_weight,
            out_bias,
            mlp1_weight,
            mlp1_bias,
            mlp2_weight,
            mlp2_bias,
            ln1_weight,
            ln1_bias,
            ln2_weight,
            ln2_bias,
        }
    }
}

// ─── ViTBlock ─────────────────────────────────────────────────────────────────

/// A single pre-norm ViT transformer block.
pub struct ViTBlock {
    pub config: ViTBlockConfig,
    pub weights: ViTBlockWeights,
}

impl ViTBlock {
    /// Construct a new block with Xavier-initialised weights.
    pub fn new(cfg: ViTBlockConfig, rng: &mut LcgRng) -> Self {
        let weights = ViTBlockWeights::default_init(&cfg, rng);
        Self {
            config: cfg,
            weights,
        }
    }

    /// Forward pass.
    ///
    /// `tokens` is flat `[n_tokens, embed_dim]`.
    /// Returns `[n_tokens * embed_dim]`.
    ///
    /// Pipeline:
    /// ```text
    /// h   = LN1(tokens)
    /// h   = MHSA(h)  + tokens   (residual)
    /// h2  = LN2(h)
    /// out = MLP(h2)  + h        (residual)
    /// ```
    pub fn forward(&self, tokens: &[f32], n_tokens: usize) -> VisionResult<Vec<f32>> {
        let e = self.config.embed_dim;
        if tokens.len() != n_tokens * e {
            return Err(VisionError::DimensionMismatch {
                expected: n_tokens * e,
                got: tokens.len(),
            });
        }
        if n_tokens == 0 {
            return Err(VisionError::EmptyInput("tokens"));
        }

        let w = &self.weights;
        let cfg = &self.config;

        // Pre-norm 1
        let h = layer_norm(tokens, &w.ln1_weight, &w.ln1_bias, n_tokens, e, 1e-5);

        // Multi-head self-attention
        let attn_out = mhsa(
            &h,
            n_tokens,
            e,
            cfg.n_heads,
            cfg.head_dim(),
            &w.qkv_weight,
            &w.qkv_bias,
            &w.out_weight,
            &w.out_bias,
        )?;

        // Residual 1: attn_out + tokens
        let mut h: Vec<f32> = attn_out
            .iter()
            .zip(tokens.iter())
            .map(|(a, b)| a + b)
            .collect();

        // Pre-norm 2
        let h2 = layer_norm(&h, &w.ln2_weight, &w.ln2_bias, n_tokens, e, 1e-5);

        // MLP: Linear → GELU → Linear
        let mlp_dim = cfg.mlp_dim();
        let mid = linear(&h2, &w.mlp1_weight, &w.mlp1_bias, e, mlp_dim);
        let mid: Vec<f32> = mid.into_iter().map(gelu_exact).collect();
        let mlp_out = linear(&mid, &w.mlp2_weight, &w.mlp2_bias, mlp_dim, e);

        // Residual 2: mlp_out + h
        for (o, m) in h.iter_mut().zip(mlp_out.iter()) {
            *o += m;
        }

        Ok(h)
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Per-row layer normalisation.
///
/// For each of `n` rows of length `d`:
/// ```text
/// mean = Σx / d
/// var  = Σ(x - mean)² / d
/// out  = (x - mean) / sqrt(var + eps) * weight + bias
/// ```
pub(crate) fn layer_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    n: usize,
    d: usize,
    eps: f32,
) -> Vec<f32> {
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
/// `x`:  `[batch, n_in]` flat
/// `w`:  `[n_out, n_in]` flat (each row is one output filter)
/// `b`:  `[n_out]`
/// Returns `[batch, n_out]` flat.
pub(crate) fn linear(x: &[f32], w: &[f32], b: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
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

/// Exact GELU activation via the tanh approximation.
///
/// ```text
/// GELU(x) ≈ x * 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x³)))
/// ```
#[inline]
pub(crate) fn gelu_exact(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    x * 0.5 * (1.0 + inner.tanh())
}

/// Multi-head self-attention.
///
/// # Steps
/// 1. Project `tokens` → Q, K, V via a single fused `[3*embed, embed]` matrix.
/// 2. Split into three `[n_tokens, embed]` tensors.
/// 3. Reshape each to `[n_heads, n_tokens, head_dim]`.
/// 4. Scaled dot-product per head: `S = Q @ K^T / sqrt(head_dim)`.
/// 5. Row-wise softmax of `S`.
/// 6. Weighted sum: `A = S @ V`.
/// 7. Reshape `A` → `[n_tokens, embed]`, apply output projection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mhsa(
    tokens: &[f32],
    n_tokens: usize,
    embed_dim: usize,
    n_heads: usize,
    head_dim: usize,
    qkv_weight: &[f32],
    qkv_bias: &[f32],
    out_weight: &[f32],
    out_bias: &[f32],
) -> VisionResult<Vec<f32>> {
    // Step 1: fused QKV projection → [n_tokens, 3*embed]
    let qkv = linear(tokens, qkv_weight, qkv_bias, embed_dim, 3 * embed_dim);

    // Step 2: split into Q, K, V each [n_tokens, embed_dim]
    let mut q = vec![0.0f32; n_tokens * embed_dim];
    let mut k = vec![0.0f32; n_tokens * embed_dim];
    let mut v = vec![0.0f32; n_tokens * embed_dim];
    for t in 0..n_tokens {
        let src = &qkv[t * 3 * embed_dim..(t + 1) * 3 * embed_dim];
        let qd = &mut q[t * embed_dim..(t + 1) * embed_dim];
        let kd = &mut k[t * embed_dim..(t + 1) * embed_dim];
        let vd = &mut v[t * embed_dim..(t + 1) * embed_dim];
        qd.copy_from_slice(&src[..embed_dim]);
        kd.copy_from_slice(&src[embed_dim..2 * embed_dim]);
        vd.copy_from_slice(&src[2 * embed_dim..]);
    }

    // Steps 3-6: per-head scaled dot-product attention
    // We compute head-by-head to avoid one giant allocation.
    let scale = 1.0 / (head_dim as f32).sqrt();
    // Output buffer: [n_tokens, embed_dim]
    let mut concat = vec![0.0f32; n_tokens * embed_dim];

    // attn_scores: [n_tokens, n_tokens] scratch
    let mut scores = vec![0.0f32; n_tokens * n_tokens];

    for h in 0..n_heads {
        let hd_off = h * head_dim; // column offset within embed_dim

        // Compute scores[i, j] = scale * Σ_d Q[i, h*hd + d] * K[j, h*hd + d]
        for i in 0..n_tokens {
            for j in 0..n_tokens {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * embed_dim + hd_off + d] * k[j * embed_dim + hd_off + d];
                }
                scores[i * n_tokens + j] = dot * scale;
            }
        }

        // In-place row-wise softmax
        softmax_rows(&mut scores, n_tokens, n_tokens);

        // Weighted sum: A[i, d] = Σ_j scores[i,j] * V[j, h*hd + d]
        for i in 0..n_tokens {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..n_tokens {
                    acc += scores[i * n_tokens + j] * v[j * embed_dim + hd_off + d];
                }
                concat[i * embed_dim + hd_off + d] = acc;
            }
        }
    }

    // Step 7: output projection [embed, embed]
    let out = linear(&concat, out_weight, out_bias, embed_dim, embed_dim);

    // Validate all outputs are finite (catch NaN from degenerate inputs)
    if out.iter().any(|v| !v.is_finite()) {
        return Err(VisionError::NonFinite("mhsa output"));
    }

    Ok(out)
}

/// In-place row-wise softmax with numerical stability (max subtraction).
pub(crate) fn softmax_rows(logits: &mut [f32], n_rows: usize, n_cols: usize) {
    for i in 0..n_rows {
        let row = &mut logits[i * n_cols..(i + 1) * n_cols];
        // Find max
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // Shift and exp
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        // Normalise
        let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg() -> ViTBlockConfig {
        ViTBlockConfig::new(64, 4, 4).expect("valid config")
    }

    // ── Config validation ─────────────────────────────────────────────────────

    #[test]
    fn config_valid() {
        let cfg = make_cfg();
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.mlp_dim(), 256);
    }

    #[test]
    fn config_invalid_embed_zero() {
        let r = ViTBlockConfig::new(0, 4, 4);
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(0))));
    }

    #[test]
    fn config_invalid_heads_zero() {
        let r = ViTBlockConfig::new(64, 0, 4);
        assert!(matches!(r, Err(VisionError::InvalidNumHeads(0))));
    }

    #[test]
    fn config_head_dim_mismatch() {
        let r = ViTBlockConfig::new(64, 3, 4); // 64 % 3 != 0
        assert!(matches!(
            r,
            Err(VisionError::HeadDimMismatch {
                n_heads: 3,
                embed_dim: 64
            })
        ));
    }

    // ── layer_norm ────────────────────────────────────────────────────────────

    #[test]
    fn layer_norm_zero_input_with_identity_affine() {
        // LN of all-zeros with weight=1, bias=0 → output is all 0.
        // mean=0, var=0 → normalized = 0/sqrt(eps) * 1 + 0 = 0.
        let x = vec![0.0f32; 8];
        let w = vec![1.0f32; 8];
        let b = vec![0.0f32; 8];
        let out = layer_norm(&x, &w, &b, 1, 8, 1e-5);
        assert!(
            out.iter().all(|&v| v.abs() < 1e-4),
            "expected near-zero: {out:?}"
        );
    }

    #[test]
    fn layer_norm_constant_row_normalises_to_zero() {
        // All same value → mean = value, var = 0 → normalised = 0.
        let x = vec![5.0f32; 16];
        let w = vec![1.0f32; 16];
        let b = vec![0.0f32; 16];
        let out = layer_norm(&x, &w, &b, 1, 16, 1e-5);
        assert!(out.iter().all(|&v| v.abs() < 1e-4));
    }

    #[test]
    fn layer_norm_output_shape() {
        let x = vec![1.0f32; 4 * 64];
        let w = vec![1.0f32; 64];
        let b = vec![0.0f32; 64];
        let out = layer_norm(&x, &w, &b, 4, 64, 1e-5);
        assert_eq!(out.len(), 4 * 64);
    }

    #[test]
    fn layer_norm_standard_normal_output() {
        // After LN, each row should have ~mean 0 and ~var 1.
        let mut rng = LcgRng::new(77);
        let mut x = vec![0.0f32; 128];
        rng.fill_normal(&mut x);
        let w = vec![1.0f32; 128];
        let b = vec![0.0f32; 128];
        let out = layer_norm(&x, &w, &b, 1, 128, 1e-5);
        let mean: f32 = out.iter().sum::<f32>() / 128.0;
        let var: f32 = out.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 128.0;
        assert!(mean.abs() < 1e-4, "mean too large: {mean}");
        assert!((var - 1.0).abs() < 1e-3, "var not ~1: {var}");
    }

    // ── MHSA ──────────────────────────────────────────────────────────────────

    #[test]
    fn mhsa_output_shape() {
        let cfg = make_cfg();
        let e = cfg.embed_dim;
        let n_tokens = 17;
        let mut rng = LcgRng::new(1);
        let w = ViTBlockWeights::default_init(&cfg, &mut rng);
        let tokens = vec![0.1f32; n_tokens * e];
        let out = mhsa(
            &tokens,
            n_tokens,
            e,
            cfg.n_heads,
            cfg.head_dim(),
            &w.qkv_weight,
            &w.qkv_bias,
            &w.out_weight,
            &w.out_bias,
        )
        .expect("mhsa ok");
        assert_eq!(out.len(), n_tokens * e);
    }

    #[test]
    fn mhsa_output_finite() {
        let cfg = make_cfg();
        let e = cfg.embed_dim;
        let n_tokens = 10;
        let mut rng = LcgRng::new(2);
        let w = ViTBlockWeights::default_init(&cfg, &mut rng);
        let mut tokens = vec![0.0f32; n_tokens * e];
        rng.fill_normal(&mut tokens);
        let out = mhsa(
            &tokens,
            n_tokens,
            e,
            cfg.n_heads,
            cfg.head_dim(),
            &w.qkv_weight,
            &w.qkv_bias,
            &w.out_weight,
            &w.out_bias,
        )
        .expect("mhsa ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite in mhsa output"
        );
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    #[test]
    fn forward_output_shape() {
        let cfg = make_cfg();
        let e = cfg.embed_dim;
        let n_tokens = 17; // 16 patches + 1 CLS
        let mut rng = LcgRng::new(3);
        let block = ViTBlock::new(cfg, &mut rng);
        let tokens = vec![0.0f32; n_tokens * e];
        let out = block.forward(&tokens, n_tokens).expect("forward ok");
        assert_eq!(out.len(), n_tokens * e);
    }

    #[test]
    fn forward_output_finite() {
        let cfg = make_cfg();
        let e = cfg.embed_dim;
        let n_tokens = 17;
        let mut rng = LcgRng::new(4);
        let block = ViTBlock::new(cfg, &mut rng);
        let mut tokens = vec![0.0f32; n_tokens * e];
        rng.fill_normal(&mut tokens);
        let out = block.forward(&tokens, n_tokens).expect("forward ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite in block output"
        );
    }

    #[test]
    fn forward_dimension_mismatch_errors() {
        let cfg = make_cfg();
        let n_tokens = 5;
        let mut rng = LcgRng::new(5);
        let block = ViTBlock::new(cfg, &mut rng);
        // Deliberately wrong length
        let tokens = vec![0.0f32; n_tokens * 32]; // embed=64 expected
        let r = block.forward(&tokens, n_tokens);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_residual_not_trivially_zero() {
        // Even with zero tokens, the block output should be non-zero due to biases
        // and LN scaling (though with all-zero tokens, LN may collapse to 0 — biases are 0).
        // Use random tokens to verify the residual path changes values.
        let cfg = make_cfg();
        let e = cfg.embed_dim;
        let n_tokens = 8;
        let mut rng = LcgRng::new(6);
        let block = ViTBlock::new(cfg, &mut rng);
        let mut tokens = vec![0.0f32; n_tokens * e];
        rng.fill_normal(&mut tokens);
        let out = block.forward(&tokens, n_tokens).expect("forward ok");
        // Output should differ from input (the block is not identity-initialised).
        let diff: f32 = out
            .iter()
            .zip(tokens.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "block did not change tokens (diff={diff})");
    }

    // ── gelu_exact ────────────────────────────────────────────────────────────

    #[test]
    fn gelu_zero() {
        // GELU(0) = 0
        assert!((gelu_exact(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_large_positive_approx_identity() {
        // For large x, GELU(x) ≈ x
        let x = 10.0f32;
        assert!(
            (gelu_exact(x) - x).abs() < 1e-3,
            "GELU({x}) = {}",
            gelu_exact(x)
        );
    }

    #[test]
    fn gelu_large_negative_approx_zero() {
        // For large negative x, GELU(x) ≈ 0
        let x = -10.0f32;
        assert!(gelu_exact(x).abs() < 1e-3, "GELU({x}) = {}", gelu_exact(x));
    }
}

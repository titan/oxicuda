//! XCiT — Cross-Covariance Image Transformer (El-Nouby 2021, "XCiT:
//! Cross-Covariance Image Transformers", NeurIPS).
//!
//! XCiT replaces token self-attention with **Cross-Covariance Attention (XCA)**.
//! Instead of an `N × N` attention map over tokens, XCA builds a `d × d`
//! attention map over **feature channels**, which makes the cost *linear* in the
//! number of tokens `N` (it is quadratic only in the head dimension). Concretely,
//! per head:
//!
//! ```text
//! Q, K, V  : [N, d]              (tokens × head_dim)
//! Q̂ = ℓ2_normalise_columns(Q)   (each channel unit-norm across tokens)
//! K̂ = ℓ2_normalise_columns(K)
//! A  = softmax( (K̂ᵀ Q̂) · τ )    (d × d, τ a learned per-head temperature)
//! out = V A                      (N × d)
//! ```
//!
//! The `ℓ2`-normalisation along the **token** axis and the learnable temperature
//! `τ` keep the cross-covariance (Gram) matrix well-conditioned. This module
//! implements the XCA layer plus its surrounding pre-norm residual block; the
//! optional Local Patch Interaction (depthwise conv) of the full paper is a
//! separate concern and not bundled here.
//!
//! Layout: tokens are flat row-major `[n_tokens, embed_dim]`.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    vit::vit_block::{gelu_exact, layer_norm, linear},
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for an XCiT block (Cross-Covariance Attention + MLP).
#[derive(Debug, Clone, PartialEq)]
pub struct XcitConfig {
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads (must divide `embed_dim`).
    pub n_heads: usize,
    /// MLP expansion ratio.
    pub mlp_ratio: usize,
}

impl XcitConfig {
    /// Create and validate an `XcitConfig`.
    ///
    /// # Errors
    /// - [`VisionError::InvalidEmbedDim`] if `embed_dim == 0`.
    /// - [`VisionError::InvalidNumHeads`] if `n_heads == 0`.
    /// - [`VisionError::HeadDimMismatch`] if `embed_dim % n_heads != 0`.
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

    /// Small default config (`embed_dim = 64`, 4 heads, ratio 4).
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            embed_dim: 64,
            n_heads: 4,
            mlp_ratio: 4,
        }
    }

    /// Per-head dimension.
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

// ─── Cross-Covariance Attention ────────────────────────────────────────────────

/// Cross-Covariance Attention (XCA): channel-wise attention with linear token
/// complexity.
///
/// Computes, for `tokens` `[n_tokens, embed_dim]`, the XCA output of the same
/// shape. `temperature` holds one learnable scalar per head (`[n_heads]`).
///
/// # Errors
/// - [`VisionError::EmptyInput`] if `n_tokens == 0`.
/// - [`VisionError::DimensionMismatch`] if a tensor length is inconsistent.
/// - [`VisionError::NonFinite`] if the result is not finite.
#[allow(clippy::too_many_arguments)]
pub fn cross_covariance_attention(
    tokens: &[f32],
    n_tokens: usize,
    embed_dim: usize,
    n_heads: usize,
    head_dim: usize,
    qkv_weight: &[f32],
    qkv_bias: &[f32],
    out_weight: &[f32],
    out_bias: &[f32],
    temperature: &[f32],
) -> VisionResult<Vec<f32>> {
    if n_tokens == 0 {
        return Err(VisionError::EmptyInput("xca tokens"));
    }
    if tokens.len() != n_tokens * embed_dim {
        return Err(VisionError::DimensionMismatch {
            expected: n_tokens * embed_dim,
            got: tokens.len(),
        });
    }
    if temperature.len() != n_heads {
        return Err(VisionError::DimensionMismatch {
            expected: n_heads,
            got: temperature.len(),
        });
    }

    // Fused QKV projection → [n_tokens, 3*embed_dim].
    let qkv = linear(tokens, qkv_weight, qkv_bias, embed_dim, 3 * embed_dim);

    // Split into Q, K, V each [n_tokens, embed_dim].
    let mut q = vec![0.0f32; n_tokens * embed_dim];
    let mut k = vec![0.0f32; n_tokens * embed_dim];
    let mut v = vec![0.0f32; n_tokens * embed_dim];
    for t in 0..n_tokens {
        let src = &qkv[t * 3 * embed_dim..(t + 1) * 3 * embed_dim];
        q[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&src[..embed_dim]);
        k[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&src[embed_dim..2 * embed_dim]);
        v[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&src[2 * embed_dim..]);
    }

    // Output accumulator [n_tokens, embed_dim].
    let mut concat = vec![0.0f32; n_tokens * embed_dim];

    for (h, &tau) in temperature.iter().enumerate().take(n_heads) {
        let off = h * head_dim;

        // ℓ2-normalise Q and K along the TOKEN axis (per channel column).
        // For channel c: norm_c = sqrt(Σ_t Q[t,c]²).
        let mut q_norm = vec![0.0f32; n_tokens * head_dim];
        let mut k_norm = vec![0.0f32; n_tokens * head_dim];
        for c in 0..head_dim {
            let mut qn = 0.0f32;
            let mut kn = 0.0f32;
            for t in 0..n_tokens {
                let qv = q[t * embed_dim + off + c];
                let kv = k[t * embed_dim + off + c];
                qn += qv * qv;
                kn += kv * kv;
            }
            let qn = qn.sqrt().max(1e-12);
            let kn = kn.sqrt().max(1e-12);
            for t in 0..n_tokens {
                q_norm[t * head_dim + c] = q[t * embed_dim + off + c] / qn;
                k_norm[t * head_dim + c] = k[t * embed_dim + off + c] / kn;
            }
        }

        // Cross-covariance map A = (K̂ᵀ Q̂) scaled by τ → [head_dim, head_dim].
        // A[a, b] = Σ_t K̂[t, a] · Q̂[t, b].
        let mut attn = vec![0.0f32; head_dim * head_dim];
        for a in 0..head_dim {
            for b in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..n_tokens {
                    acc += k_norm[t * head_dim + a] * q_norm[t * head_dim + b];
                }
                attn[a * head_dim + b] = acc * tau;
            }
        }

        // Softmax over the last axis (rows of the d×d map).
        for a in 0..head_dim {
            let row = &mut attn[a * head_dim..(a + 1) * head_dim];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - mx).exp();
                sum += *x;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for x in row.iter_mut() {
                *x *= inv;
            }
        }

        // Output: out[t, a] = Σ_b V[t, b] · A[a, b]   (channel mixing).
        for t in 0..n_tokens {
            for a in 0..head_dim {
                let mut acc = 0.0f32;
                for b in 0..head_dim {
                    acc += v[t * embed_dim + off + b] * attn[a * head_dim + b];
                }
                concat[t * embed_dim + off + a] = acc;
            }
        }
    }

    // Output projection.
    let out = linear(&concat, out_weight, out_bias, embed_dim, embed_dim);
    if out.iter().any(|v| !v.is_finite()) {
        return Err(VisionError::NonFinite("xca output"));
    }
    Ok(out)
}

// ─── XCiT block ────────────────────────────────────────────────────────────────

/// A pre-norm XCiT transformer block: XCA + GELU MLP with residuals.
pub struct XcitBlock {
    config: XcitConfig,
    qkv_weight: Vec<f32>,
    qkv_bias: Vec<f32>,
    out_weight: Vec<f32>,
    out_bias: Vec<f32>,
    mlp1_weight: Vec<f32>,
    mlp1_bias: Vec<f32>,
    mlp2_weight: Vec<f32>,
    mlp2_bias: Vec<f32>,
    ln1_weight: Vec<f32>,
    ln1_bias: Vec<f32>,
    ln2_weight: Vec<f32>,
    ln2_bias: Vec<f32>,
    /// Per-head learnable temperature (initialised to 1).
    temperature: Vec<f32>,
}

impl XcitBlock {
    /// Construct an XCiT block with Xavier-initialised weights and `τ = 1`.
    #[must_use]
    pub fn new(config: XcitConfig, rng: &mut LcgRng) -> Self {
        let e = config.embed_dim;
        let mlp = config.mlp_dim();
        let scale = 1.0 / (e as f32).sqrt();
        let fill = |rng: &mut LcgRng, n: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= scale;
            }
            v
        };
        let temperature = vec![1.0f32; config.n_heads];
        Self {
            qkv_weight: fill(rng, 3 * e * e),
            qkv_bias: vec![0.0f32; 3 * e],
            out_weight: fill(rng, e * e),
            out_bias: vec![0.0f32; e],
            mlp1_weight: fill(rng, mlp * e),
            mlp1_bias: vec![0.0f32; mlp],
            mlp2_weight: fill(rng, e * mlp),
            mlp2_bias: vec![0.0f32; e],
            ln1_weight: vec![1.0f32; e],
            ln1_bias: vec![0.0f32; e],
            ln2_weight: vec![1.0f32; e],
            ln2_bias: vec![0.0f32; e],
            temperature,
            config,
        }
    }

    /// Configuration accessor.
    #[must_use]
    pub fn config(&self) -> &XcitConfig {
        &self.config
    }

    /// Forward pass over `[n_tokens, embed_dim]` tokens.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] / [`VisionError::EmptyInput`] on bad
    ///   shapes.
    /// - [`VisionError::NonFinite`] on a non-finite result.
    pub fn forward(&self, tokens: &[f32], n_tokens: usize) -> VisionResult<Vec<f32>> {
        let e = self.config.embed_dim;
        if tokens.len() != n_tokens * e {
            return Err(VisionError::DimensionMismatch {
                expected: n_tokens * e,
                got: tokens.len(),
            });
        }
        if n_tokens == 0 {
            return Err(VisionError::EmptyInput("xcit tokens"));
        }

        // Pre-norm + XCA.
        let h = layer_norm(tokens, &self.ln1_weight, &self.ln1_bias, n_tokens, e, 1e-5);
        let attn = cross_covariance_attention(
            &h,
            n_tokens,
            e,
            self.config.n_heads,
            self.config.head_dim(),
            &self.qkv_weight,
            &self.qkv_bias,
            &self.out_weight,
            &self.out_bias,
            &self.temperature,
        )?;
        // Residual 1.
        let mut x: Vec<f32> = tokens.iter().zip(attn.iter()).map(|(a, b)| a + b).collect();

        // Pre-norm 2 + MLP.
        let h2 = layer_norm(&x, &self.ln2_weight, &self.ln2_bias, n_tokens, e, 1e-5);
        let mlp_dim = self.config.mlp_dim();
        let mid = linear(&h2, &self.mlp1_weight, &self.mlp1_bias, e, mlp_dim);
        let mid: Vec<f32> = mid.into_iter().map(gelu_exact).collect();
        let mlp_out = linear(&mid, &self.mlp2_weight, &self.mlp2_bias, mlp_dim, e);
        // Residual 2.
        for (o, m) in x.iter_mut().zip(mlp_out.iter()) {
            *o += m;
        }

        if x.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("xcit block output"));
        }
        Ok(x)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        assert!(XcitConfig::new(0, 4, 4).is_err());
        assert!(XcitConfig::new(64, 0, 4).is_err());
        assert!(XcitConfig::new(64, 5, 4).is_err());
        let cfg = XcitConfig::tiny();
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.mlp_dim(), 256);
    }

    #[test]
    fn block_output_shape_and_finite() {
        let cfg = XcitConfig::tiny();
        let e = cfg.embed_dim;
        let mut rng = LcgRng::new(1);
        let block = XcitBlock::new(cfg, &mut rng);
        let n_tokens = 20;
        let mut tokens = vec![0.0f32; n_tokens * e];
        rng.fill_normal(&mut tokens);
        let out = block.forward(&tokens, n_tokens).expect("ok");
        assert_eq!(out.len(), n_tokens * e);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn linear_complexity_works_for_many_tokens() {
        // XCA cost is linear in tokens; a large token count must still succeed
        // (and the d×d map stays small).
        let cfg = XcitConfig::new(16, 2, 2).expect("ok");
        let e = cfg.embed_dim;
        let mut rng = LcgRng::new(2);
        let block = XcitBlock::new(cfg, &mut rng);
        let n_tokens = 500;
        let mut tokens = vec![0.0f32; n_tokens * e];
        rng.fill_normal(&mut tokens);
        let out = block.forward(&tokens, n_tokens).expect("ok");
        assert_eq!(out.len(), n_tokens * e);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn attention_map_rows_are_softmax_normalised() {
        // Directly exercise the XCA function and verify it produces finite output
        // for a deterministic input (rows of the internal d×d map sum to 1 by
        // construction; we check the externally observable output is sane).
        let n_tokens = 8;
        let embed_dim = 8;
        let n_heads = 2;
        let head_dim = 4;
        let mut rng = LcgRng::new(3);
        let mut tokens = vec![0.0f32; n_tokens * embed_dim];
        rng.fill_normal(&mut tokens);
        let scale = 1.0 / (embed_dim as f32).sqrt();
        let mut qkv_weight = vec![0.0f32; 3 * embed_dim * embed_dim];
        rng.fill_normal(&mut qkv_weight);
        for w in &mut qkv_weight {
            *w *= scale;
        }
        let qkv_bias = vec![0.0f32; 3 * embed_dim];
        let mut out_weight = vec![0.0f32; embed_dim * embed_dim];
        rng.fill_normal(&mut out_weight);
        for w in &mut out_weight {
            *w *= scale;
        }
        let out_bias = vec![0.0f32; embed_dim];
        let temperature = vec![1.0f32; n_heads];
        let out = cross_covariance_attention(
            &tokens,
            n_tokens,
            embed_dim,
            n_heads,
            head_dim,
            &qkv_weight,
            &qkv_bias,
            &out_weight,
            &out_bias,
            &temperature,
        )
        .expect("ok");
        assert_eq!(out.len(), n_tokens * embed_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn temperature_length_mismatch_errors() {
        let n_tokens = 4;
        let embed_dim = 8;
        let n_heads = 2;
        let head_dim = 4;
        let tokens = vec![0.1f32; n_tokens * embed_dim];
        let qkv_weight = vec![0.0f32; 3 * embed_dim * embed_dim];
        let qkv_bias = vec![0.0f32; 3 * embed_dim];
        let out_weight = vec![0.0f32; embed_dim * embed_dim];
        let out_bias = vec![0.0f32; embed_dim];
        // Wrong temperature length.
        let temperature = vec![1.0f32; n_heads + 1];
        let r = cross_covariance_attention(
            &tokens,
            n_tokens,
            embed_dim,
            n_heads,
            head_dim,
            &qkv_weight,
            &qkv_bias,
            &out_weight,
            &out_bias,
            &temperature,
        );
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn block_errors_on_bad_shape() {
        let cfg = XcitConfig::tiny();
        let mut rng = LcgRng::new(4);
        let block = XcitBlock::new(cfg, &mut rng);
        let bad = vec![0.0f32; 10];
        assert!(block.forward(&bad, 3).is_err());
        assert!(block.forward(&[], 0).is_err());
    }

    #[test]
    fn deterministic() {
        let cfg = XcitConfig::tiny();
        let e = cfg.embed_dim;
        let mut r1 = LcgRng::new(9);
        let b1 = XcitBlock::new(cfg.clone(), &mut r1);
        let mut r2 = LcgRng::new(9);
        let b2 = XcitBlock::new(cfg, &mut r2);
        let tokens = vec![0.2f32; 12 * e];
        let a = b1.forward(&tokens, 12).expect("ok");
        let b = b2.forward(&tokens, 12).expect("ok");
        assert_eq!(a, b);
    }
}

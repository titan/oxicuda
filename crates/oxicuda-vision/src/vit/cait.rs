//! CaiT — Class-Attention in image Transformers (Touvron 2021, "Going deeper
//! with Image Transformers", ICCV).
//!
//! CaiT splits a ViT into two stages:
//! 1. **Self-attention stage**: ordinary patch-token self-attention *without*
//!    the class token (handled by the existing [`crate::vit::vit_encoder`]).
//! 2. **Class-attention stage**: a small number of **Class-Attention (CA)**
//!    layers in which **only the class token is the query**, attending to the
//!    frozen patch tokens (keys/values). The patch tokens are *not* updated;
//!    only the class embedding is refined. This decouples "summarising for
//!    classification" from "patch interaction" and lets the network go deeper
//!    stably.
//!
//! Two ingredients make deep CaiT trainable, both implemented here:
//! - **LayerScale**: each residual branch is multiplied by a per-channel learned
//!   diagonal `γ` initialised to a tiny value `ε` (e.g. `1e-4`), so residual
//!   branches start near-identity ([`LayerScale`]).
//! - **Class-Attention** as above ([`ClassAttention`]).
//!
//! Layout: all tensors are flat row-major `f32`. Patch tokens are
//! `[n_patches, embed_dim]`; the class token is `[embed_dim]`.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    vit::vit_block::{gelu_exact, layer_norm, linear},
};

// ─── LayerScale ────────────────────────────────────────────────────────────────

/// Per-channel diagonal LayerScale `γ ⊙ x` with a learnable `[dim]` vector.
///
/// Initialised to a small constant so that, at the start of training, the
/// residual branch contributes almost nothing and the block is near-identity —
/// the mechanism that lets CaiT stack many layers without divergence.
#[derive(Debug, Clone)]
pub struct LayerScale {
    /// Per-channel scale `γ` of length `dim`.
    pub gamma: Vec<f32>,
}

impl LayerScale {
    /// Create a LayerScale initialised to the constant `init` on every channel.
    ///
    /// # Errors
    /// - [`VisionError::InvalidEmbedDim`] if `dim == 0`.
    /// - [`VisionError::NonFinite`] if `init` is not finite.
    pub fn new(dim: usize, init: f32) -> VisionResult<Self> {
        if dim == 0 {
            return Err(VisionError::InvalidEmbedDim(dim));
        }
        if !init.is_finite() {
            return Err(VisionError::NonFinite("layer_scale init"));
        }
        Ok(Self {
            gamma: vec![init; dim],
        })
    }

    /// Apply LayerScale in place to a `[rows, dim]` tensor.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `x.len()` is not a multiple of
    ///   `dim` (length `gamma.len()`).
    pub fn apply(&self, x: &mut [f32]) -> VisionResult<()> {
        let dim = self.gamma.len();
        if dim == 0 || x.len() % dim != 0 {
            return Err(VisionError::DimensionMismatch {
                expected: dim,
                got: x.len(),
            });
        }
        let rows = x.len() / dim;
        for r in 0..rows {
            let row = &mut x[r * dim..(r + 1) * dim];
            for (v, &g) in row.iter_mut().zip(self.gamma.iter()) {
                *v *= g;
            }
        }
        Ok(())
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a CaiT Class-Attention layer.
#[derive(Debug, Clone, PartialEq)]
pub struct CaitConfig {
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads (must divide `embed_dim`).
    pub n_heads: usize,
    /// MLP expansion ratio.
    pub mlp_ratio: usize,
    /// LayerScale initial value `ε`.
    pub layer_scale_init: f32,
}

impl CaitConfig {
    /// Create and validate a `CaitConfig`.
    ///
    /// # Errors
    /// - [`VisionError::InvalidEmbedDim`] if `embed_dim == 0`.
    /// - [`VisionError::InvalidNumHeads`] if `n_heads == 0`.
    /// - [`VisionError::HeadDimMismatch`] if `embed_dim % n_heads != 0`.
    /// - [`VisionError::NonFinite`] if `layer_scale_init` is not finite.
    pub fn new(
        embed_dim: usize,
        n_heads: usize,
        mlp_ratio: usize,
        layer_scale_init: f32,
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
        if !layer_scale_init.is_finite() {
            return Err(VisionError::NonFinite("layer_scale_init"));
        }
        Ok(Self {
            embed_dim,
            n_heads,
            mlp_ratio,
            layer_scale_init,
        })
    }

    /// A small default config (`embed_dim = 64`, 4 heads, ratio 4, `ε = 1e-4`).
    #[must_use]
    pub fn tiny() -> Self {
        // Constructed from validated literals.
        Self {
            embed_dim: 64,
            n_heads: 4,
            mlp_ratio: 4,
            layer_scale_init: 1e-4,
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

// ─── ClassAttention layer ──────────────────────────────────────────────────────

/// A CaiT Class-Attention layer: refine the class token by attending to patch
/// tokens, with LayerScale on both the attention and MLP residual branches.
///
/// The class token is the **only** query; the patch tokens (and the class token
/// itself, following the reference implementation) form the keys/values.
pub struct ClassAttention {
    config: CaitConfig,
    // Separate Q (class only) and KV projections (CaiT uses distinct q / k / v).
    q_weight: Vec<f32>,
    q_bias: Vec<f32>,
    k_weight: Vec<f32>,
    k_bias: Vec<f32>,
    v_weight: Vec<f32>,
    v_bias: Vec<f32>,
    proj_weight: Vec<f32>,
    proj_bias: Vec<f32>,
    ln1_weight: Vec<f32>,
    ln1_bias: Vec<f32>,
    ln2_weight: Vec<f32>,
    ln2_bias: Vec<f32>,
    mlp1_weight: Vec<f32>,
    mlp1_bias: Vec<f32>,
    mlp2_weight: Vec<f32>,
    mlp2_bias: Vec<f32>,
    ls_attn: LayerScale,
    ls_mlp: LayerScale,
}

impl ClassAttention {
    /// Construct a Class-Attention layer with Xavier-initialised weights.
    ///
    /// # Errors
    /// Propagates [`LayerScale::new`] validation (only fails on a bad config,
    /// which `CaitConfig::new` already prevents).
    pub fn new(config: CaitConfig, rng: &mut LcgRng) -> VisionResult<Self> {
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

        let ls_attn = LayerScale::new(e, config.layer_scale_init)?;
        let ls_mlp = LayerScale::new(e, config.layer_scale_init)?;

        Ok(Self {
            q_weight: fill(rng, e * e),
            q_bias: vec![0.0f32; e],
            k_weight: fill(rng, e * e),
            k_bias: vec![0.0f32; e],
            v_weight: fill(rng, e * e),
            v_bias: vec![0.0f32; e],
            proj_weight: fill(rng, e * e),
            proj_bias: vec![0.0f32; e],
            ln1_weight: vec![1.0f32; e],
            ln1_bias: vec![0.0f32; e],
            ln2_weight: vec![1.0f32; e],
            ln2_bias: vec![0.0f32; e],
            mlp1_weight: fill(rng, mlp * e),
            mlp1_bias: vec![0.0f32; mlp],
            mlp2_weight: fill(rng, e * mlp),
            mlp2_bias: vec![0.0f32; e],
            ls_attn,
            ls_mlp,
            config,
        })
    }

    /// Configuration accessor.
    #[must_use]
    pub fn config(&self) -> &CaitConfig {
        &self.config
    }

    /// Forward a single Class-Attention layer.
    ///
    /// - `cls`: class token `[embed_dim]`.
    /// - `patches`: patch tokens `[n_patches, embed_dim]` (treated as constants;
    ///   only used as keys/values).
    ///
    /// Returns the **updated class token** `[embed_dim]`. Patch tokens are
    /// returned unchanged by the caller (CA does not modify them).
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `cls` or `patches` lengths are
    ///   inconsistent with `embed_dim`.
    /// - [`VisionError::EmptyInput`] if there are no patches.
    /// - [`VisionError::NonFinite`] if the result is not finite.
    pub fn forward(&self, cls: &[f32], patches: &[f32]) -> VisionResult<Vec<f32>> {
        let e = self.config.embed_dim;
        if cls.len() != e {
            return Err(VisionError::DimensionMismatch {
                expected: e,
                got: cls.len(),
            });
        }
        if patches.is_empty() {
            return Err(VisionError::EmptyInput("cait patches"));
        }
        if patches.len() % e != 0 {
            return Err(VisionError::DimensionMismatch {
                expected: e,
                got: patches.len(),
            });
        }
        let n_patches = patches.len() / e;

        // Build the key/value context: [cls; patches] → (n_patches + 1) tokens.
        // CaiT attends over the concatenation of the class token and patches.
        let n_ctx = n_patches + 1;
        let mut ctx = Vec::with_capacity(n_ctx * e);
        ctx.extend_from_slice(cls);
        ctx.extend_from_slice(patches);

        // Pre-norm.
        let cls_norm = layer_norm(cls, &self.ln1_weight, &self.ln1_bias, 1, e, 1e-5);
        let ctx_norm = layer_norm(&ctx, &self.ln1_weight, &self.ln1_bias, n_ctx, e, 1e-5);

        // Projections: Q from the (single) class token, K/V from the context.
        let q = linear(&cls_norm, &self.q_weight, &self.q_bias, e, e); // [1, e]
        let k = linear(&ctx_norm, &self.k_weight, &self.k_bias, e, e); // [n_ctx, e]
        let v = linear(&ctx_norm, &self.v_weight, &self.v_bias, e, e); // [n_ctx, e]

        // Per-head scaled dot-product attention with one query.
        let n_heads = self.config.n_heads;
        let hd = self.config.head_dim();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn = vec![0.0f32; e];

        for h in 0..n_heads {
            let off = h * hd;
            // scores over the n_ctx keys.
            let mut scores = vec![0.0f32; n_ctx];
            let mut mx = f32::NEG_INFINITY;
            for (j, s) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[off + d] * k[j * e + off + d];
                }
                *s = dot * scale;
                if *s > mx {
                    mx = *s;
                }
            }
            // softmax.
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            // weighted sum of V.
            for d in 0..hd {
                let mut acc = 0.0f32;
                for (j, &s) in scores.iter().enumerate() {
                    acc += s * inv * v[j * e + off + d];
                }
                attn[off + d] = acc;
            }
        }

        // Output projection.
        let mut attn_out = linear(&attn, &self.proj_weight, &self.proj_bias, e, e);
        // LayerScale on the attention branch, then residual.
        self.ls_attn.apply(&mut attn_out)?;
        let mut x: Vec<f32> = cls
            .iter()
            .zip(attn_out.iter())
            .map(|(c, a)| c + a)
            .collect();

        // MLP branch with pre-norm + LayerScale.
        let h2 = layer_norm(&x, &self.ln2_weight, &self.ln2_bias, 1, e, 1e-5);
        let mlp_dim = self.config.mlp_dim();
        let mid = linear(&h2, &self.mlp1_weight, &self.mlp1_bias, e, mlp_dim);
        let mid: Vec<f32> = mid.into_iter().map(gelu_exact).collect();
        let mut mlp_out = linear(&mid, &self.mlp2_weight, &self.mlp2_bias, mlp_dim, e);
        self.ls_mlp.apply(&mut mlp_out)?;
        for (o, m) in x.iter_mut().zip(mlp_out.iter()) {
            *o += m;
        }

        if x.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("cait class-attention output"));
        }
        Ok(x)
    }
}

// ─── ClassAttention stack ──────────────────────────────────────────────────────

/// A stack of CaiT Class-Attention layers refining a single class token.
pub struct ClassAttentionStack {
    layers: Vec<ClassAttention>,
}

impl ClassAttentionStack {
    /// Build `depth` independently-initialised Class-Attention layers.
    ///
    /// # Errors
    /// - [`VisionError::Internal`] if `depth == 0`.
    /// - Propagates [`ClassAttention::new`].
    pub fn new(config: CaitConfig, depth: usize, rng: &mut LcgRng) -> VisionResult<Self> {
        if depth == 0 {
            return Err(VisionError::Internal("cait depth must be > 0".into()));
        }
        let mut layers = Vec::with_capacity(depth);
        for _ in 0..depth {
            layers.push(ClassAttention::new(config.clone(), rng)?);
        }
        Ok(Self { layers })
    }

    /// Number of Class-Attention layers.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Run all Class-Attention layers, returning the refined class token.
    ///
    /// # Errors
    /// Propagates [`ClassAttention::forward`].
    pub fn forward(&self, cls: &[f32], patches: &[f32]) -> VisionResult<Vec<f32>> {
        let mut c = cls.to_vec();
        for layer in &self.layers {
            c = layer.forward(&c, patches)?;
        }
        Ok(c)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_scale_init_and_apply() {
        let ls = LayerScale::new(4, 0.5).expect("ok");
        assert_eq!(ls.gamma, vec![0.5; 4]);
        let mut x = vec![2.0f32, 4.0, 6.0, 8.0, 1.0, 1.0, 1.0, 1.0];
        ls.apply(&mut x).expect("ok");
        assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn layer_scale_validation() {
        assert!(LayerScale::new(0, 1e-4).is_err());
        assert!(LayerScale::new(4, f32::NAN).is_err());
        let ls = LayerScale::new(4, 1.0).expect("ok");
        let mut bad = vec![1.0f32; 5];
        assert!(ls.apply(&mut bad).is_err());
    }

    #[test]
    fn config_validation() {
        assert!(CaitConfig::new(0, 4, 4, 1e-4).is_err());
        assert!(CaitConfig::new(64, 0, 4, 1e-4).is_err());
        assert!(CaitConfig::new(64, 3, 4, 1e-4).is_err());
        assert!(CaitConfig::new(64, 4, 4, f32::INFINITY).is_err());
        let cfg = CaitConfig::tiny();
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.mlp_dim(), 256);
    }

    #[test]
    fn class_attention_output_shape() {
        let cfg = CaitConfig::tiny();
        let e = cfg.embed_dim;
        let mut rng = LcgRng::new(1);
        let ca = ClassAttention::new(cfg, &mut rng).expect("ok");
        let n_patches = 16;
        let mut cls = vec![0.0f32; e];
        let mut patches = vec![0.0f32; n_patches * e];
        rng.fill_normal(&mut cls);
        rng.fill_normal(&mut patches);
        let out = ca.forward(&cls, &patches).expect("ok");
        assert_eq!(out.len(), e);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn class_attention_near_identity_at_init() {
        // With LayerScale ε = 1e-4, the refined class token should be very close
        // to the input class token (residual branches barely contribute).
        let cfg = CaitConfig::new(64, 4, 4, 1e-4).expect("ok");
        let e = cfg.embed_dim;
        let mut rng = LcgRng::new(2);
        let ca = ClassAttention::new(cfg, &mut rng).expect("ok");
        let mut cls = vec![0.0f32; e];
        let mut patches = vec![0.0f32; 16 * e];
        rng.fill_normal(&mut cls);
        rng.fill_normal(&mut patches);
        let out = ca.forward(&cls, &patches).expect("ok");
        let max_diff = cls
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-1,
            "class token moved too far at init: {max_diff}"
        );
    }

    #[test]
    fn class_attention_errors() {
        let cfg = CaitConfig::tiny();
        let e = cfg.embed_dim;
        let mut rng = LcgRng::new(3);
        let ca = ClassAttention::new(cfg, &mut rng).expect("ok");
        // wrong cls length.
        assert!(ca.forward(&vec![0.0f32; e - 1], &vec![0.0f32; e]).is_err());
        // empty patches.
        assert!(ca.forward(&vec![0.0f32; e], &[]).is_err());
        // ragged patches.
        assert!(ca.forward(&vec![0.0f32; e], &vec![0.0f32; e + 1]).is_err());
    }

    #[test]
    fn stack_depth_and_forward() {
        let cfg = CaitConfig::tiny();
        let e = cfg.embed_dim;
        let mut rng = LcgRng::new(4);
        let stack = ClassAttentionStack::new(cfg, 3, &mut rng).expect("ok");
        assert_eq!(stack.depth(), 3);
        let mut cls = vec![0.0f32; e];
        let mut patches = vec![0.0f32; 9 * e];
        rng.fill_normal(&mut cls);
        rng.fill_normal(&mut patches);
        let out = stack.forward(&cls, &patches).expect("ok");
        assert_eq!(out.len(), e);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stack_zero_depth_errors() {
        let cfg = CaitConfig::tiny();
        let mut rng = LcgRng::new(5);
        assert!(ClassAttentionStack::new(cfg, 0, &mut rng).is_err());
    }

    #[test]
    fn deterministic() {
        let cfg = CaitConfig::tiny();
        let e = cfg.embed_dim;
        let mut r1 = LcgRng::new(7);
        let ca1 = ClassAttention::new(cfg.clone(), &mut r1).expect("ok");
        let mut r2 = LcgRng::new(7);
        let ca2 = ClassAttention::new(cfg, &mut r2).expect("ok");
        let cls = vec![0.3f32; e];
        let patches = vec![0.1f32; 8 * e];
        let a = ca1.forward(&cls, &patches).expect("ok");
        let b = ca2.forward(&cls, &patches).expect("ok");
        assert_eq!(a, b);
    }
}

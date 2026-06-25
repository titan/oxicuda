//! EVA / EVA-CLIP variant configurations (Fang 2022, "EVA: Exploring the Limits
//! of Masked Visual Representation Learning at Scale", CVPR; Sun 2023,
//! "EVA-CLIP").
//!
//! EVA is a family of large-scale ViT backbones distilled from a CLIP teacher
//! with MIM pre-training. Architecturally, an EVA backbone is a standard ViT
//! *trunk* (so the heavy lifting reuses [`crate::vit::ViTEncoder`] and friends)
//! with two recipe-level differences this module captures:
//!
//! 1. **Configuration presets** — the published EVA / EVA-02 / EVA-CLIP sizes
//!    (e.g. EVA-g/14 with `embed_dim = 1408`, `depth = 40`, `patch = 14`). These
//!    are exposed as validated [`ViTConfig`] builders so the rest of the crate
//!    can instantiate them directly.
//! 2. **Mean-pool representation head** — EVA-CLIP forms the image embedding by
//!    **mean-pooling the patch tokens** (optionally followed by a post-LayerNorm)
//!    rather than reading a `[CLS]` token, then projecting into the joint
//!    image-text space. [`EvaPoolHead`] implements that pooling + LayerNorm +
//!    linear projection.
//!
//! The presets are intentionally small *here* only in that they are validated;
//! the dimensions match the real EVA configurations. Constructing the full
//! billion-parameter weights is a memory question for the caller, not a
//! correctness one — the config arithmetic (patch count, head dim divisibility)
//! is all checked.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    vit::{vit_block::linear, vit_model::ViTConfig},
};

// ─── EVA preset enumeration ────────────────────────────────────────────────────

/// A published EVA / EVA-CLIP backbone size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaVariant {
    /// EVA-02-Ti/14 — tiny EVA-02 (`embed 192`, `depth 12`, `heads 3`).
    Eva02Tiny,
    /// EVA-02-S/14 — small EVA-02 (`embed 384`, `depth 12`, `heads 6`).
    Eva02Small,
    /// EVA-02-B/14 — base EVA-02 (`embed 768`, `depth 12`, `heads 12`).
    Eva02Base,
    /// EVA-02-L/14 — large EVA-02 (`embed 1024`, `depth 24`, `heads 16`).
    Eva02Large,
    /// EVA-g/14 — the giant EVA-CLIP vision tower
    /// (`embed 1408`, `depth 40`, `heads 16`).
    EvaGiant,
}

impl EvaVariant {
    /// `(embed_dim, depth, n_heads, mlp_ratio, patch_size, img_size)` for the
    /// variant. The MLP ratio is integerised (EVA-02 uses ~`2.6×` SwiGLU; we use
    /// the nearest standard integer `4` for the GELU-MLP trunk reused here).
    #[must_use]
    fn spec(self) -> (usize, usize, usize, usize, usize, usize) {
        match self {
            EvaVariant::Eva02Tiny => (192, 12, 3, 4, 14, 224),
            EvaVariant::Eva02Small => (384, 12, 6, 4, 14, 224),
            EvaVariant::Eva02Base => (768, 12, 12, 4, 14, 224),
            EvaVariant::Eva02Large => (1024, 24, 16, 4, 14, 224),
            EvaVariant::EvaGiant => (1408, 40, 16, 4, 14, 224),
        }
    }

    /// Embedding (width) dimension of this variant.
    #[must_use]
    pub fn embed_dim(self) -> usize {
        self.spec().0
    }

    /// Transformer depth of this variant.
    #[must_use]
    pub fn depth(self) -> usize {
        self.spec().1
    }

    /// Number of attention heads.
    #[must_use]
    pub fn n_heads(self) -> usize {
        self.spec().2
    }

    /// Patch size (EVA uses a 14×14 patch).
    #[must_use]
    pub fn patch_size(self) -> usize {
        self.spec().4
    }

    /// Build a validated [`ViTConfig`] for this variant.
    ///
    /// `in_chans` and `n_classes` are caller-supplied (EVA-CLIP towers usually
    /// drop the classification head, so any positive `n_classes` is accepted as
    /// a placeholder projection size).
    ///
    /// # Errors
    /// Propagates [`ViTConfig::new`] (patch divisibility, head divisibility,
    /// positive classes).
    pub fn vit_config(self, in_chans: usize, n_classes: usize) -> VisionResult<ViTConfig> {
        let (embed_dim, depth, n_heads, mlp_ratio, patch_size, img_size) = self.spec();
        ViTConfig::new(
            img_size, patch_size, in_chans, embed_dim, depth, n_heads, mlp_ratio, n_classes,
        )
    }

    /// Joint image-text embedding dimension conventionally paired with this
    /// vision tower in EVA-CLIP (the projection output size).
    #[must_use]
    pub fn clip_proj_dim(self) -> usize {
        match self {
            EvaVariant::Eva02Tiny | EvaVariant::Eva02Small => 512,
            EvaVariant::Eva02Base => 512,
            EvaVariant::Eva02Large => 768,
            EvaVariant::EvaGiant => 1024,
        }
    }
}

// ─── EVA-CLIP mean-pool head ───────────────────────────────────────────────────

/// EVA-CLIP representation head: mean-pool patch tokens → LayerNorm → linear
/// projection into the joint embedding space.
///
/// Unlike vanilla ViT which reads the `[CLS]` token, EVA-CLIP averages the patch
/// tokens (global average pooling over the sequence) to form the image
/// representation; this head reproduces that pooling, an optional post-LayerNorm,
/// and the joint-space projection.
pub struct EvaPoolHead {
    embed_dim: usize,
    proj_dim: usize,
    /// LayerNorm scale `[embed_dim]`.
    ln_weight: Vec<f32>,
    /// LayerNorm bias `[embed_dim]`.
    ln_bias: Vec<f32>,
    /// Projection kernel `[proj_dim, embed_dim]`.
    proj_weight: Vec<f32>,
    /// Projection bias `[proj_dim]`.
    proj_bias: Vec<f32>,
}

impl EvaPoolHead {
    /// Construct an EVA pooling head with LayerNorm (scale 1, bias 0) and an
    /// Xavier-initialised projection.
    ///
    /// # Errors
    /// - [`VisionError::InvalidEmbedDim`] if `embed_dim == 0`.
    /// - [`VisionError::InvalidProjDim`] if `proj_dim == 0`.
    pub fn new(embed_dim: usize, proj_dim: usize, rng: &mut LcgRng) -> VisionResult<Self> {
        if embed_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(embed_dim));
        }
        if proj_dim == 0 {
            return Err(VisionError::InvalidProjDim(proj_dim));
        }
        let scale = 1.0 / (embed_dim as f32).sqrt();
        let mut proj_weight = vec![0.0f32; proj_dim * embed_dim];
        rng.fill_normal(&mut proj_weight);
        for w in &mut proj_weight {
            *w *= scale;
        }
        Ok(Self {
            embed_dim,
            proj_dim,
            ln_weight: vec![1.0f32; embed_dim],
            ln_bias: vec![0.0f32; embed_dim],
            proj_weight,
            proj_bias: vec![0.0f32; proj_dim],
        })
    }

    /// Embedding dimension.
    #[must_use]
    #[inline]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Projection output dimension.
    #[must_use]
    #[inline]
    pub fn proj_dim(&self) -> usize {
        self.proj_dim
    }

    /// Mean-pool `[n_tokens, embed_dim]` patch tokens, LayerNorm, then project.
    ///
    /// Returns the joint-space embedding `[proj_dim]`.
    ///
    /// # Errors
    /// - [`VisionError::EmptyInput`] if `n_tokens == 0`.
    /// - [`VisionError::DimensionMismatch`] if `tokens.len() != n_tokens·embed`.
    /// - [`VisionError::NonFinite`] if the result is not finite.
    pub fn forward(&self, tokens: &[f32], n_tokens: usize) -> VisionResult<Vec<f32>> {
        let e = self.embed_dim;
        if n_tokens == 0 {
            return Err(VisionError::EmptyInput("eva pool tokens"));
        }
        if tokens.len() != n_tokens * e {
            return Err(VisionError::DimensionMismatch {
                expected: n_tokens * e,
                got: tokens.len(),
            });
        }

        // Mean-pool over the token axis.
        let mut pooled = vec![0.0f32; e];
        for t in 0..n_tokens {
            let row = &tokens[t * e..(t + 1) * e];
            for (p, &v) in pooled.iter_mut().zip(row.iter()) {
                *p += v;
            }
        }
        let inv_n = 1.0 / n_tokens as f32;
        for p in &mut pooled {
            *p *= inv_n;
        }

        // Post-LayerNorm.
        let normed = layer_norm_vec(&pooled, &self.ln_weight, &self.ln_bias, 1e-5);

        // Joint-space projection.
        let out = linear(
            &normed,
            &self.proj_weight,
            &self.proj_bias,
            e,
            self.proj_dim,
        );
        if out.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("eva pool head output"));
        }
        Ok(out)
    }
}

/// Single-vector LayerNorm (length-`d` input).
fn layer_norm_vec(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let mean: f32 = x.iter().sum::<f32>() / d as f32;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .map(|((&v, &w), &b)| (v - mean) * inv_std * w + b)
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_variants_produce_valid_configs() {
        let variants = [
            EvaVariant::Eva02Tiny,
            EvaVariant::Eva02Small,
            EvaVariant::Eva02Base,
            EvaVariant::Eva02Large,
            EvaVariant::EvaGiant,
        ];
        for v in variants {
            let cfg = v.vit_config(3, 1000).expect("valid eva config");
            assert_eq!(cfg.embed_dim, v.embed_dim());
            assert_eq!(cfg.depth, v.depth());
            assert_eq!(cfg.n_heads, v.n_heads());
            assert_eq!(cfg.patch_size, v.patch_size());
            // head divisibility holds (ViTConfig::new would have rejected it).
            assert_eq!(cfg.embed_dim % cfg.n_heads, 0);
            // 224 / 14 = 16 → 256 patches.
            assert_eq!(cfg.n_patches(), 16 * 16);
        }
    }

    #[test]
    fn giant_spec_matches_paper() {
        let g = EvaVariant::EvaGiant;
        assert_eq!(g.embed_dim(), 1408);
        assert_eq!(g.depth(), 40);
        assert_eq!(g.n_heads(), 16);
        assert_eq!(g.clip_proj_dim(), 1024);
    }

    #[test]
    fn clip_proj_dims_present() {
        assert_eq!(EvaVariant::Eva02Large.clip_proj_dim(), 768);
        assert_eq!(EvaVariant::Eva02Base.clip_proj_dim(), 512);
    }

    #[test]
    fn pool_head_validation() {
        let mut rng = LcgRng::new(1);
        assert!(EvaPoolHead::new(0, 16, &mut rng).is_err());
        assert!(EvaPoolHead::new(32, 0, &mut rng).is_err());
        let head = EvaPoolHead::new(32, 16, &mut rng).expect("ok");
        assert_eq!(head.embed_dim(), 32);
        assert_eq!(head.proj_dim(), 16);
    }

    #[test]
    fn pool_head_output_shape_and_finite() {
        let mut rng = LcgRng::new(2);
        let head = EvaPoolHead::new(64, 32, &mut rng).expect("ok");
        let n_tokens = 17;
        let mut tokens = vec![0.0f32; n_tokens * 64];
        rng.fill_normal(&mut tokens);
        let out = head.forward(&tokens, n_tokens).expect("ok");
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pool_head_mean_pooling_is_correct() {
        // With identity LN-ish behaviour hard to guarantee, instead check the
        // pooled vector feeds projection: if all tokens are identical, the result
        // must equal the projection of that single (LayerNorm'd) token.
        let mut rng = LcgRng::new(3);
        let head = EvaPoolHead::new(8, 4, &mut rng).expect("ok");
        let token = [1.0f32, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let n_tokens = 5;
        let mut tokens = vec![0.0f32; n_tokens * 8];
        for t in 0..n_tokens {
            tokens[t * 8..(t + 1) * 8].copy_from_slice(&token);
        }
        let out_many = head.forward(&tokens, n_tokens).expect("ok");
        let out_one = head.forward(&token, 1).expect("ok");
        // Mean of identical rows == the row → identical projection.
        for (a, b) in out_many.iter().zip(out_one.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn pool_head_errors() {
        let mut rng = LcgRng::new(4);
        let head = EvaPoolHead::new(16, 8, &mut rng).expect("ok");
        assert!(head.forward(&[], 0).is_err());
        assert!(head.forward(&[0.0f32; 10], 3).is_err()); // 3*16 != 10
    }

    #[test]
    fn deterministic() {
        let mut r1 = LcgRng::new(7);
        let h1 = EvaPoolHead::new(32, 16, &mut r1).expect("ok");
        let mut r2 = LcgRng::new(7);
        let h2 = EvaPoolHead::new(32, 16, &mut r2).expect("ok");
        let tokens = vec![0.3f32; 9 * 32];
        let a = h1.forward(&tokens, 9).expect("ok");
        let b = h2.forward(&tokens, 9).expect("ok");
        assert_eq!(a, b);
    }
}

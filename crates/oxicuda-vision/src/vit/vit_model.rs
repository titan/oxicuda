//! Full Vision Transformer (ViT) model.
//!
//! ## Pipeline
//! ```text
//! image [C, H, W]
//!   → PatchEmbed         → [n_patches, embed_dim]
//!   → prepend_cls        → [n_patches + 1, embed_dim]
//!   → add_pos_embed      → [n_patches + 1, embed_dim]
//!   → ViTEncoder         → [n_patches + 1, embed_dim]
//!   → CLS token [0]      → [embed_dim]
//!   → Linear head        → [n_classes]   (logits)
//! ```

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    patch_embed::{LearnablePosEmbed, PatchEmbed, PatchEmbedConfig, add_pos_embed, prepend_cls},
    vit::{
        vit_block::linear,
        vit_encoder::{ViTEncoder, ViTEncoderConfig},
    },
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Top-level ViT configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ViTConfig {
    /// Square spatial resolution of the input image (H = W).
    pub img_size: usize,
    /// Patch size (stride = kernel size = `patch_size`).
    pub patch_size: usize,
    /// Number of input channels (e.g. 3 for RGB).
    pub in_chans: usize,
    /// Token embedding dimension.
    pub embed_dim: usize,
    /// Number of transformer blocks.
    pub depth: usize,
    /// Number of attention heads per block.
    pub n_heads: usize,
    /// MLP hidden-dim multiplier.
    pub mlp_ratio: usize,
    /// Number of output classes for the classification head.
    pub n_classes: usize,
}

impl ViTConfig {
    /// Tiny ViT: suitable for CIFAR-10 style 32×32 RGB images.
    ///
    /// `img_size=32`, `patch_size=4`, `in_chans=3`, `embed_dim=64`,
    /// `depth=2`, `n_heads=4`, `mlp_ratio=4`, `n_classes=10`.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            img_size: 32,
            patch_size: 4,
            in_chans: 3,
            embed_dim: 64,
            depth: 2,
            n_heads: 4,
            mlp_ratio: 4,
            n_classes: 10,
        }
    }

    /// Validate and construct a `ViTConfig`.
    ///
    /// # Errors
    /// - `n_classes == 0` → `InvalidNumClasses`
    /// - patch or image size issues → propagated from `PatchEmbedConfig`
    /// - `embed_dim % n_heads != 0` → `HeadDimMismatch`
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        depth: usize,
        n_heads: usize,
        mlp_ratio: usize,
        n_classes: usize,
    ) -> VisionResult<Self> {
        if n_classes == 0 {
            return Err(VisionError::InvalidNumClasses(n_classes));
        }
        if depth == 0 {
            return Err(VisionError::Internal("depth must be > 0".into()));
        }
        // Delegate patch / embed / head validation to their own constructors.
        PatchEmbedConfig::new(img_size, patch_size, in_chans, embed_dim)?;
        ViTEncoderConfig::new(embed_dim, n_heads, mlp_ratio, depth)?;

        Ok(Self {
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            n_heads,
            mlp_ratio,
            n_classes,
        })
    }

    /// Number of non-overlapping patches for this image / patch size.
    #[must_use]
    pub fn n_patches(&self) -> usize {
        let grid = self.img_size / self.patch_size;
        grid * grid
    }

    /// Total sequence length including the CLS token.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.n_patches() + 1
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Learnable weights for the ViT classification head.
///
/// The head is a single linear projection from `embed_dim` to `n_classes`.
pub struct ViTModelWeights {
    /// Head projection kernel: `[n_classes, embed_dim]` flat.
    pub head_weight: Vec<f32>,
    /// Head projection bias: `[n_classes]`.
    pub head_bias: Vec<f32>,
}

impl ViTModelWeights {
    fn default_init(cfg: &ViTConfig, rng: &mut LcgRng) -> Self {
        let scale = 1.0 / (cfg.embed_dim as f32).sqrt();
        let mut head_weight = vec![0.0f32; cfg.n_classes * cfg.embed_dim];
        rng.fill_normal(&mut head_weight);
        for v in &mut head_weight {
            *v *= scale;
        }
        let head_bias = vec![0.0f32; cfg.n_classes];
        Self {
            head_weight,
            head_bias,
        }
    }
}

// ─── ViTModel ─────────────────────────────────────────────────────────────────

/// Full Vision Transformer model.
pub struct ViTModel {
    /// Top-level model configuration.
    pub config: ViTConfig,
    /// Strided conv2d patch embedder.
    pub patch_embed: PatchEmbed,
    /// Learnable positional embeddings for `seq_len` positions (CLS + patches).
    pub pos_embed: LearnablePosEmbed,
    /// Transformer encoder stack.
    pub encoder: ViTEncoder,
    /// Classification head weights.
    pub weights: ViTModelWeights,
}

impl ViTModel {
    /// Build and initialise a full ViT model.
    ///
    /// All sub-modules share the same `rng` stream for reproducibility.
    pub fn new(cfg: ViTConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let patch_cfg =
            PatchEmbedConfig::new(cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)?;
        let patch_embed = PatchEmbed::new(patch_cfg, rng);

        // Positional embedding covers CLS token + all patch tokens.
        let seq_len = cfg.seq_len();
        let pos_embed = LearnablePosEmbed::new(seq_len, cfg.embed_dim, rng)?;

        let enc_cfg = ViTEncoderConfig::new(cfg.embed_dim, cfg.n_heads, cfg.mlp_ratio, cfg.depth)?;
        let encoder = ViTEncoder::new(enc_cfg, rng)?;

        let weights = ViTModelWeights::default_init(&cfg, rng);

        Ok(Self {
            config: cfg,
            patch_embed,
            pos_embed,
            encoder,
            weights,
        })
    }

    /// Forward pass.
    ///
    /// `image` is flat CHW: `[in_chans, img_size, img_size]`.
    /// Returns logits: `[n_classes]`.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `image.len()` does not match
    /// `in_chans * img_size * img_size`.
    pub fn forward(&self, image: &[f32]) -> VisionResult<Vec<f32>> {
        let cfg = &self.config;
        let expected_img = cfg.in_chans * cfg.img_size * cfg.img_size;
        if image.len() != expected_img {
            return Err(VisionError::DimensionMismatch {
                expected: expected_img,
                got: image.len(),
            });
        }

        // Step 1: patch embedding → [n_patches, embed_dim]
        let patch_tokens = self.patch_embed.forward(image)?;

        // Step 2: prepend CLS token → [n_patches + 1, embed_dim]
        let cls_token = &self.patch_embed.weights.cls_token;
        let mut tokens = prepend_cls(&patch_tokens, cls_token, cfg.embed_dim)?;

        // Step 3: add positional embeddings (all seq_len positions)
        add_pos_embed(&mut tokens, &self.pos_embed.table, cfg.embed_dim)?;

        // Step 4: transformer encoder → [seq_len, embed_dim]
        let seq_len = cfg.seq_len();
        let encoded = self.encoder.forward(&tokens, seq_len)?;

        // Step 5: extract CLS token (first row)
        let cls_repr = &encoded[..cfg.embed_dim];

        // Step 6: classification head → [n_classes]
        let logits = linear(
            cls_repr,
            &self.weights.head_weight,
            &self.weights.head_bias,
            cfg.embed_dim,
            cfg.n_classes,
        );

        Ok(logits)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tiny_model() -> ViTModel {
        let cfg = ViTConfig::tiny();
        let mut rng = LcgRng::new(42);
        ViTModel::new(cfg, &mut rng).expect("tiny model created")
    }

    // ── Config ────────────────────────────────────────────────────────────────

    #[test]
    fn tiny_config_values() {
        let cfg = ViTConfig::tiny();
        assert_eq!(cfg.img_size, 32);
        assert_eq!(cfg.patch_size, 4);
        assert_eq!(cfg.in_chans, 3);
        assert_eq!(cfg.embed_dim, 64);
        assert_eq!(cfg.depth, 2);
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.mlp_ratio, 4);
        assert_eq!(cfg.n_classes, 10);
    }

    #[test]
    fn tiny_config_n_patches() {
        let cfg = ViTConfig::tiny();
        // (32/4)^2 = 8^2 = 64 patches
        assert_eq!(cfg.n_patches(), 64);
        assert_eq!(cfg.seq_len(), 65);
    }

    #[test]
    fn config_zero_classes_errors() {
        let r = ViTConfig::new(32, 4, 3, 64, 2, 4, 4, 0);
        assert!(matches!(r, Err(VisionError::InvalidNumClasses(0))));
    }

    #[test]
    fn config_invalid_patch_size_errors() {
        let r = ViTConfig::new(32, 5, 3, 64, 2, 4, 4, 10); // 32 % 5 != 0
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn config_head_dim_mismatch_errors() {
        let r = ViTConfig::new(32, 4, 3, 63, 2, 4, 4, 10); // 63 % 4 != 0
        assert!(matches!(r, Err(VisionError::HeadDimMismatch { .. })));
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    #[test]
    fn forward_returns_ten_logits() {
        let model = make_tiny_model();
        let image = vec![0.0f32; 3 * 32 * 32];
        let logits = model.forward(&image).expect("forward ok");
        assert_eq!(logits.len(), 10, "expected 10 logits, got {}", logits.len());
    }

    #[test]
    fn forward_logits_finite() {
        let model = make_tiny_model();
        let mut rng = LcgRng::new(7);
        let mut image = vec![0.0f32; 3 * 32 * 32];
        rng.fill_normal(&mut image);
        let logits = model.forward(&image).expect("forward ok");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite logits: {logits:?}"
        );
    }

    #[test]
    fn forward_random_input_not_constant_logits() {
        // Different images → different logits
        let model = make_tiny_model();
        let mut rng = LcgRng::new(13);
        let mut img1 = vec![0.0f32; 3 * 32 * 32];
        let mut img2 = vec![0.0f32; 3 * 32 * 32];
        rng.fill_normal(&mut img1);
        rng.fill_normal(&mut img2);
        let l1 = model.forward(&img1).expect("ok");
        let l2 = model.forward(&img2).expect("ok");
        let diff: f32 = l1.iter().zip(l2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "logits did not change between different images (diff={diff})"
        );
    }

    #[test]
    fn forward_wrong_image_size_errors() {
        let model = make_tiny_model();
        // Too small
        let image = vec![0.0f32; 3 * 32 * 31]; // wrong
        let r = model.forward(&image);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_correct_image_size_passes() {
        let model = make_tiny_model();
        let image = vec![0.5f32; 3 * 32 * 32];
        let logits = model
            .forward(&image)
            .expect("forward ok with constant image");
        assert_eq!(logits.len(), 10);
    }

    // ── Structural checks ─────────────────────────────────────────────────────

    #[test]
    fn pos_embed_has_correct_positions() {
        let model = make_tiny_model();
        // seq_len = 65 (64 patches + 1 CLS)
        assert_eq!(model.pos_embed.n_positions, 65);
        assert_eq!(model.pos_embed.embed_dim, 64);
    }

    #[test]
    fn encoder_has_correct_depth() {
        let model = make_tiny_model();
        assert_eq!(model.encoder.blocks.len(), 2);
    }

    #[test]
    fn head_weights_correct_size() {
        let model = make_tiny_model();
        assert_eq!(model.weights.head_weight.len(), 10 * 64);
        assert_eq!(model.weights.head_bias.len(), 10);
    }

    #[test]
    fn different_seeds_produce_different_outputs() {
        let cfg = ViTConfig::tiny();
        let mut rng_a = LcgRng::new(1);
        let mut rng_b = LcgRng::new(2);
        let model_a = ViTModel::new(cfg.clone(), &mut rng_a).expect("ok");
        let model_b = ViTModel::new(cfg, &mut rng_b).expect("ok");
        let image = vec![0.5f32; 3 * 32 * 32];
        let la = model_a.forward(&image).expect("ok");
        let lb = model_b.forward(&image).expect("ok");
        let diff: f32 = la.iter().zip(lb.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "different seeds should yield different logits (diff={diff})"
        );
    }
}

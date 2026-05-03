//! CLIP vision encoder.
//!
//! Wraps a ViT encoder with CLIP-specific construction conveniences.
//! A single CLS token is prepended, positional embeddings are added, and
//! the encoder output at the CLS position is returned as the image embedding.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    patch_embed::{LearnablePosEmbed, PatchEmbed, PatchEmbedConfig, add_pos_embed, prepend_cls},
    vit::{ViTConfig, ViTEncoder, ViTEncoderConfig},
};

// ─── ClipVisionConfig ────────────────────────────────────────────────────────

/// Configuration for the CLIP vision encoder.
///
/// Wraps a [`ViTConfig`] so that CLIP can share the same architectural
/// hyper-parameter vocabulary as a standalone ViT model.
#[derive(Debug, Clone)]
pub struct ClipVisionConfig {
    /// Underlying ViT hyper-parameters (image size, patch size, depth, …).
    pub vit_config: ViTConfig,
}

impl ClipVisionConfig {
    /// Create a [`ClipVisionConfig`] from an existing [`ViTConfig`].
    #[must_use]
    pub fn new(vit_config: ViTConfig) -> Self {
        Self { vit_config }
    }

    /// Convenience constructor for a tiny CLIP encoder suitable for tests.
    ///
    /// Delegates to [`ViTConfig::tiny`].
    #[must_use]
    pub fn tiny() -> Self {
        Self::new(ViTConfig::tiny())
    }
}

// ─── ClipVisionEncoder ───────────────────────────────────────────────────────

/// CLIP vision encoder: ViT-backbone that produces a single `embed_dim`
/// CLS-token embedding per image.
///
/// Pipeline:
/// ```text
/// image [C × H × W]
///   → patch_embed    → [n_patches, embed_dim]
///   → prepend_cls    → [n_patches + 1, embed_dim]
///   → add_pos_embed  → [n_patches + 1, embed_dim]
///   → encoder        → [n_patches + 1, embed_dim]
///   → tokens[0]      → [embed_dim]   (CLS token output)
/// ```
pub struct ClipVisionEncoder {
    /// Full configuration.
    pub config: ClipVisionConfig,
    /// Strided Conv2D patch embedder.
    pub patch_embed: PatchEmbed,
    /// Learnable positional embeddings: `n_patches + 1` positions (incl. CLS).
    pub pos_embed: LearnablePosEmbed,
    /// Stack of ViT transformer blocks with final layer-norm.
    pub encoder: ViTEncoder,
    /// CLS token: flat `[embed_dim]`, Gaussian-initialised with scale 0.02.
    pub cls_token: Vec<f32>,
}

impl ClipVisionEncoder {
    /// Construct a new CLIP vision encoder.
    ///
    /// Initialises:
    /// - Patch embedder (Conv2D kernel, bias).
    /// - Learnable positional embedding table with `n_patches + 1` rows.
    /// - ViT encoder stack.
    /// - CLS token vector (N(0, 0.02²)).
    ///
    /// # Errors
    /// Propagates any errors from the sub-component constructors.
    pub fn new(cfg: ClipVisionConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let vc = &cfg.vit_config;

        // ── Patch embedder ────────────────────────────────────────────────────
        let pe_cfg = PatchEmbedConfig::new(vc.img_size, vc.patch_size, vc.in_chans, vc.embed_dim)?;
        let patch_embed = PatchEmbed::new(pe_cfg.clone(), rng);

        // ── Positional embeddings: n_patches + 1 positions (CLS slot at 0) ──
        let n_patches = pe_cfg.n_patches();
        let n_positions = n_patches + 1;
        let pos_embed = LearnablePosEmbed::new(n_positions, vc.embed_dim, rng)?;

        // ── Encoder stack ─────────────────────────────────────────────────────
        let enc_cfg = ViTEncoderConfig::new(vc.embed_dim, vc.n_heads, vc.mlp_ratio, vc.depth)?;
        let encoder = ViTEncoder::new(enc_cfg, rng)?;

        // ── CLS token: N(0, 0.02²) ───────────────────────────────────────────
        let mut cls_token = vec![0.0f32; vc.embed_dim];
        rng.fill_normal(&mut cls_token);
        for v in &mut cls_token {
            *v *= 0.02;
        }

        Ok(Self {
            config: cfg,
            patch_embed,
            pos_embed,
            encoder,
            cls_token,
        })
    }

    /// Run the encoder on a single image and return the CLS embedding.
    ///
    /// # Parameters
    /// - `image`: flat `[in_chans × img_size × img_size]` CHW buffer.
    ///
    /// # Returns
    /// `[embed_dim]` CLS-token embedding.
    ///
    /// # Errors
    /// Returns [`VisionError::DimensionMismatch`] if the image size does not
    /// match the configured dimensions.
    pub fn forward_single(&self, image: &[f32]) -> VisionResult<Vec<f32>> {
        let embed_dim = self.config.vit_config.embed_dim;

        // 1. Patch embedding → [n_patches, embed_dim]
        let patch_tokens = self.patch_embed.forward(image)?;

        // 2. Prepend CLS token → [n_patches + 1, embed_dim]
        let mut tokens = prepend_cls(&patch_tokens, &self.cls_token, embed_dim)?;

        // 3. Add positional embeddings in-place.
        add_pos_embed(&mut tokens, &self.pos_embed.table, embed_dim)?;

        // 4. ViT encoder → [n_patches + 1, embed_dim]
        let n_tokens = tokens.len() / embed_dim;
        let encoded = self.encoder.forward(&tokens, n_tokens)?;

        // 5. Extract CLS token (first row).
        let cls_out = encoded[..embed_dim].to_vec();

        Ok(cls_out)
    }

    /// Run the encoder on a batch of images.
    ///
    /// # Parameters
    /// - `images`: flat `[batch × in_chans × img_size × img_size]` buffer.
    /// - `batch_size`: number of images.
    ///
    /// # Returns
    /// `Vec<Vec<f32>>` of length `batch_size`, each element is `[embed_dim]`.
    ///
    /// # Errors
    /// Returns [`VisionError::DimensionMismatch`] if the flat buffer length
    /// does not match `batch_size × in_chans × img_size × img_size`, or if
    /// any individual forward pass fails.
    pub fn forward_batch(&self, images: &[f32], batch_size: usize) -> VisionResult<Vec<Vec<f32>>> {
        let vc = &self.config.vit_config;
        let single_len = vc.in_chans * vc.img_size * vc.img_size;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let expected = batch_size * single_len;
        if images.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: images.len(),
            });
        }

        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let slice = &images[b * single_len..(b + 1) * single_len];
            results.push(self.forward_single(slice)?);
        }

        Ok(results)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a tiny encoder for tests.
    fn make_tiny_encoder(seed: u64) -> (ClipVisionEncoder, usize) {
        let mut rng = LcgRng::new(seed);
        let cfg = ClipVisionConfig::tiny();
        let embed_dim = cfg.vit_config.embed_dim;
        let encoder = ClipVisionEncoder::new(cfg, &mut rng).expect("tiny encoder ok");
        (encoder, embed_dim)
    }

    /// Fill an image buffer with ramp values (deterministic, finite).
    fn make_image(in_chans: usize, img_size: usize) -> Vec<f32> {
        let len = in_chans * img_size * img_size;
        (0..len).map(|i| i as f32 / len as f32).collect()
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn tiny_encoder_constructs() {
        let (enc, _) = make_tiny_encoder(1);
        // CLS token has the right dimension.
        let vc = &enc.config.vit_config;
        assert_eq!(enc.cls_token.len(), vc.embed_dim);
        // Positional embed has n_patches + 1 positions.
        let n_patches = (vc.img_size / vc.patch_size).pow(2);
        assert_eq!(enc.pos_embed.n_positions, n_patches + 1);
    }

    #[test]
    fn config_new_wraps_vit_config() {
        let vit_cfg = ViTConfig::tiny();
        let clip_cfg = ClipVisionConfig::new(vit_cfg.clone());
        assert_eq!(clip_cfg.vit_config.embed_dim, vit_cfg.embed_dim);
    }

    // ── forward_single ───────────────────────────────────────────────────────

    #[test]
    fn forward_single_output_shape() {
        let (enc, embed_dim) = make_tiny_encoder(2);
        let vc = &enc.config.vit_config;
        let img = make_image(vc.in_chans, vc.img_size);
        let z = enc.forward_single(&img).expect("forward_single ok");
        assert_eq!(
            z.len(),
            embed_dim,
            "forward_single output should be embed_dim"
        );
    }

    #[test]
    fn forward_single_output_finite() {
        let (enc, _) = make_tiny_encoder(3);
        let vc = &enc.config.vit_config;
        let img = make_image(vc.in_chans, vc.img_size);
        let z = enc.forward_single(&img).expect("ok");
        assert!(
            z.iter().all(|v| v.is_finite()),
            "forward_single output must be finite"
        );
    }

    #[test]
    fn forward_single_error_wrong_image_size() {
        let (enc, _) = make_tiny_encoder(4);
        let wrong_img = vec![0.0f32; 10]; // definitely wrong
        let r = enc.forward_single(&wrong_img);
        assert!(
            matches!(r, Err(VisionError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {:?}",
            r
        );
    }

    #[test]
    fn forward_single_deterministic() {
        // Same encoder + same image → same output.
        let (enc, _) = make_tiny_encoder(5);
        let vc = &enc.config.vit_config;
        let img = make_image(vc.in_chans, vc.img_size);
        let z1 = enc.forward_single(&img).expect("ok");
        let z2 = enc.forward_single(&img).expect("ok");
        assert_eq!(z1, z2, "forward_single should be deterministic");
    }

    // ── forward_batch ────────────────────────────────────────────────────────

    #[test]
    fn forward_batch_output_count() {
        let (enc, _) = make_tiny_encoder(6);
        let vc = &enc.config.vit_config;
        let single_len = vc.in_chans * vc.img_size * vc.img_size;
        let batch_size = 3_usize;
        let images = make_image(vc.in_chans * batch_size, vc.img_size);
        // Manually pad to exact batch length.
        let mut flat = images.clone();
        flat.resize(batch_size * single_len, 0.0);
        let results = enc
            .forward_batch(&flat, batch_size)
            .expect("forward_batch ok");
        assert_eq!(results.len(), batch_size, "batch result count mismatch");
    }

    #[test]
    fn forward_batch_each_embedding_has_embed_dim() {
        let (enc, embed_dim) = make_tiny_encoder(7);
        let vc = &enc.config.vit_config;
        let single_len = vc.in_chans * vc.img_size * vc.img_size;
        let batch_size = 4_usize;
        let flat = vec![0.5f32; batch_size * single_len];
        let results = enc.forward_batch(&flat, batch_size).expect("ok");
        for (i, z) in results.iter().enumerate() {
            assert_eq!(z.len(), embed_dim, "embedding {i} has wrong size");
        }
    }

    #[test]
    fn forward_batch_zero_batch_returns_empty() {
        let (enc, _) = make_tiny_encoder(8);
        let results = enc.forward_batch(&[], 0).expect("zero batch ok");
        assert!(results.is_empty(), "zero batch should return empty Vec");
    }

    #[test]
    fn forward_batch_error_wrong_total_length() {
        let (enc, _) = make_tiny_encoder(9);
        let vc = &enc.config.vit_config;
        let single_len = vc.in_chans * vc.img_size * vc.img_size;
        // One pixel too few for batch_size=2.
        let flat = vec![0.0f32; 2 * single_len - 1];
        let r = enc.forward_batch(&flat, 2);
        assert!(
            matches!(r, Err(VisionError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {:?}",
            r
        );
    }

    #[test]
    fn forward_batch_matches_individual() {
        // batch forward should equal individual forward calls.
        let (enc, embed_dim) = make_tiny_encoder(10);
        let vc = &enc.config.vit_config;
        let single_len = vc.in_chans * vc.img_size * vc.img_size;
        let batch_size = 2_usize;
        let flat: Vec<f32> = (0..batch_size * single_len)
            .map(|i| i as f32 / (batch_size * single_len) as f32)
            .collect();

        let batch_results = enc.forward_batch(&flat, batch_size).expect("batch ok");

        for b in 0..batch_size {
            let single = enc
                .forward_single(&flat[b * single_len..(b + 1) * single_len])
                .expect("single ok");
            for d in 0..embed_dim {
                assert!(
                    (batch_results[b][d] - single[d]).abs() < 1e-6,
                    "batch[{b}][{d}] = {} ≠ single[{d}] = {}",
                    batch_results[b][d],
                    single[d]
                );
            }
        }
    }
}

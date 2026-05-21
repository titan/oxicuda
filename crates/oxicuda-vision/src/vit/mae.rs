//! MAE — Masked Autoencoder (He et al. 2022 CVPR
//! "Masked Autoencoders Are Scalable Vision Learners").
//!
//! Implements the canonical self-supervised vision pre-training recipe:
//!
//! 1. Split the image into a flat sequence of `n_patches` patches of length
//!    `patch_pixels = patch_size² · in_channels`.
//! 2. Apply a learnable linear `patch_embed` projection to every patch and
//!    add an `encoder_pos_embed`.
//! 3. Randomly select a subset of `n_visible = n_patches − n_masked` indices
//!    (via partial Fisher–Yates) and keep only those tokens.
//! 4. Run the kept tokens through a deep `ViTBlock` stack (the encoder).
//! 5. Project encoded visible tokens to `decoder_dim`, scatter them back to
//!    their original positions, fill masked positions with a learnable
//!    `mask_token`, add a separate `decoder_pos_embed`, and run a (typically
//!    shallower) ViTBlock stack (the decoder).
//! 6. Project the decoder output to per-patch pixel space via
//!    `decoder_pred`.
//! 7. Loss = MEAN squared error **only over masked positions** (the canonical
//!    MAE objective — visible reconstructions do not contribute).
//!
//! ## RNG safety
//! The crate's `LcgRng::next_f32()` is biased — its output spans only
//! `[0, ~0.5)` because `next_u32()` returns the high 31 bits. We therefore
//! never call `next_f32() < mask_ratio` for masking. The random mask is built
//! with a partial Fisher–Yates shuffle driven by `next_usize(n)`, which gives
//! an exact (deterministic, count-correct) selection of `round(mask_ratio · n)`
//! masked indices. Weight initialisation uses
//! `(next_u32() as f32) / 2_147_483_648.0 − 0.5` (the genuine `[-0.5, 0.5)`
//! recipe).

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    vit::{
        vit_block::{ViTBlock, ViTBlockConfig, layer_norm, linear},
        vit_encoder::ViTEncoderConfig,
    },
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for an MAE (Masked Autoencoder) model.
#[derive(Debug, Clone, PartialEq)]
pub struct MaeConfig {
    /// Square spatial resolution of the input image (H = W).
    pub img_size: usize,
    /// Patch size (must divide `img_size`).
    pub patch_size: usize,
    /// Number of input channels (e.g. 3 for RGB).
    pub in_channels: usize,
    /// Encoder embedding dimension.
    pub encoder_dim: usize,
    /// Number of encoder transformer blocks.
    pub encoder_depth: usize,
    /// Number of encoder attention heads.
    pub encoder_heads: usize,
    /// Decoder embedding dimension (usually smaller than `encoder_dim`).
    pub decoder_dim: usize,
    /// Number of decoder transformer blocks.
    pub decoder_depth: usize,
    /// Number of decoder attention heads.
    pub decoder_heads: usize,
    /// MLP hidden-dim multiplier (shared by encoder + decoder blocks).
    pub mlp_ratio: usize,
    /// Fraction of patches to mask, in `[0, 1]` (e.g. 0.75).
    pub mask_ratio: f32,
}

impl MaeConfig {
    /// Build and validate a new `MaeConfig`.
    ///
    /// # Errors
    /// - `img_size % patch_size != 0` → `InvalidPatchSize`
    /// - any zero-sized dimension → `InvalidEmbedDim` / `EmptyInput`
    /// - `mask_ratio` outside `[0, 1]` → `Internal`
    /// - `encoder_dim % encoder_heads != 0` or
    ///   `decoder_dim % decoder_heads != 0` → `HeadDimMismatch`
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        encoder_dim: usize,
        encoder_depth: usize,
        encoder_heads: usize,
        decoder_dim: usize,
        decoder_depth: usize,
        decoder_heads: usize,
        mlp_ratio: usize,
        mask_ratio: f32,
    ) -> VisionResult<Self> {
        if patch_size == 0 || img_size == 0 || img_size % patch_size != 0 {
            return Err(VisionError::InvalidPatchSize {
                patch_size,
                img_size,
            });
        }
        if in_channels == 0 {
            return Err(VisionError::EmptyInput("in_channels"));
        }
        if encoder_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(encoder_dim));
        }
        if decoder_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(decoder_dim));
        }
        if encoder_depth == 0 {
            return Err(VisionError::Internal("encoder_depth must be > 0".into()));
        }
        if decoder_depth == 0 {
            return Err(VisionError::Internal("decoder_depth must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&mask_ratio) || !mask_ratio.is_finite() {
            return Err(VisionError::Internal(format!(
                "mask_ratio {mask_ratio} not in [0, 1]"
            )));
        }
        // Block configs validate head divisibility:
        let _ = ViTBlockConfig::new(encoder_dim, encoder_heads, mlp_ratio)?;
        let _ = ViTBlockConfig::new(decoder_dim, decoder_heads, mlp_ratio)?;
        Ok(Self {
            img_size,
            patch_size,
            in_channels,
            encoder_dim,
            encoder_depth,
            encoder_heads,
            decoder_dim,
            decoder_depth,
            decoder_heads,
            mlp_ratio,
            mask_ratio,
        })
    }

    /// Number of non-overlapping image patches.
    #[must_use]
    pub fn n_patches(&self) -> usize {
        let grid = self.img_size / self.patch_size;
        grid * grid
    }

    /// Per-patch flat pixel count (`patch_size² · in_channels`).
    #[must_use]
    pub fn patch_pixels(&self) -> usize {
        self.patch_size * self.patch_size * self.in_channels
    }
}

// ─── MaskMeta ────────────────────────────────────────────────────────────────

/// Bookkeeping for a random mask over `n_patches` positions.
///
/// Both vectors are sorted ascending for deterministic downstream use, and
/// `visible_ids ∪ masked_ids = 0..n_patches` (disjoint).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskMeta {
    /// Sorted indices of patches kept (encoder sees these).
    pub visible_ids: Vec<usize>,
    /// Sorted indices of patches replaced by the mask token.
    pub masked_ids: Vec<usize>,
}

// ─── generate_random_mask ────────────────────────────────────────────────────

/// Build a deterministic random mask over `n_patches` positions.
///
/// Uses a **partial Fisher–Yates shuffle** — the only RNG-safe technique with
/// this crate's biased `LcgRng::next_f32` (see module-level doc-comment).
///
/// The shuffle picks exactly `n_masked = round(mask_ratio · n_patches)` masked
/// indices regardless of `mask_ratio`'s decimal value; the returned counts are
/// not stochastic.
///
/// # Errors
/// - `n_patches == 0` → `EmptyInput`
/// - `mask_ratio ∉ [0, 1]` or NaN/Inf → `Internal`
pub fn generate_random_mask(
    n_patches: usize,
    mask_ratio: f32,
    rng: &mut LcgRng,
) -> VisionResult<MaskMeta> {
    if n_patches == 0 {
        return Err(VisionError::EmptyInput("n_patches"));
    }
    if !(0.0..=1.0).contains(&mask_ratio) || !mask_ratio.is_finite() {
        return Err(VisionError::Internal(format!(
            "mask_ratio {mask_ratio} not in [0, 1]"
        )));
    }
    let mut ids: Vec<usize> = (0..n_patches).collect();
    let n_masked = (mask_ratio * (n_patches as f32)).round() as usize;
    let n_masked = n_masked.min(n_patches);

    // Partial Fisher–Yates: for i in 0..n_masked, pick j uniformly in [i, n_patches)
    // and swap. The first n_masked positions become a uniform random sample without
    // replacement.
    for i in 0..n_masked {
        let remaining = n_patches - i;
        // next_usize(remaining) returns a value in [0, remaining); add i for [i, n_patches)
        let j = i + rng.next_usize(remaining);
        ids.swap(i, j);
    }

    let mut masked_ids: Vec<usize> = ids[..n_masked].to_vec();
    let mut visible_ids: Vec<usize> = ids[n_masked..].to_vec();
    masked_ids.sort_unstable();
    visible_ids.sort_unstable();
    Ok(MaskMeta {
        visible_ids,
        masked_ids,
    })
}

// ─── Mae ─────────────────────────────────────────────────────────────────────

/// Masked Autoencoder model (He et al. 2022).
///
/// Weights are stored as flat row-major `Vec<f32>` consistent with the rest of
/// the crate's transformer modules.
pub struct Mae {
    /// Top-level configuration.
    pub config: MaeConfig,
    /// Patch-embedding kernel: `[encoder_dim, patch_pixels]` row-major.
    pub patch_embed_weights: Vec<f32>,
    /// Patch-embedding bias: `[encoder_dim]`.
    pub patch_embed_bias: Vec<f32>,
    /// Encoder positional embeddings: `[n_patches, encoder_dim]`.
    pub encoder_pos_embed: Vec<f32>,
    /// Encoder ViT block stack.
    pub encoder_blocks: Vec<ViTBlock>,
    /// Final encoder LayerNorm scale (encoder_dim).
    pub encoder_norm_gamma: Vec<f32>,
    /// Final encoder LayerNorm bias  (encoder_dim).
    pub encoder_norm_beta: Vec<f32>,
    /// Decoder embed projection kernel: `[decoder_dim, encoder_dim]`.
    pub decoder_embed_weights: Vec<f32>,
    /// Decoder embed bias: `[decoder_dim]`.
    pub decoder_embed_bias: Vec<f32>,
    /// Learnable mask token shared by every masked position (decoder_dim).
    pub mask_token: Vec<f32>,
    /// Decoder positional embeddings: `[n_patches, decoder_dim]`.
    pub decoder_pos_embed: Vec<f32>,
    /// Decoder ViT block stack.
    pub decoder_blocks: Vec<ViTBlock>,
    /// Final decoder LayerNorm scale (decoder_dim).
    pub decoder_norm_gamma: Vec<f32>,
    /// Final decoder LayerNorm bias  (decoder_dim).
    pub decoder_norm_beta: Vec<f32>,
    /// Per-patch pixel projection kernel: `[patch_pixels, decoder_dim]`.
    pub decoder_pred_weights: Vec<f32>,
    /// Per-patch pixel projection bias: `[patch_pixels]`.
    pub decoder_pred_bias: Vec<f32>,
}

/// Hazard-safe `[-0.5, 0.5)` uniform sample using the high 31 bits of LcgRng.
///
/// `next_u32()` already returns values in `[0, 2³¹)`, so dividing by `2³¹`
/// gives a true `[0, 1)` sample; subtracting 0.5 centres it.
#[inline]
fn safe_centered_uniform(rng: &mut LcgRng) -> f32 {
    (rng.next_u32() as f32) / 2_147_483_648.0 - 0.5
}

/// Fill `buf` with i.i.d. `[-scale, scale)` samples using the hazard-safe
/// recipe (NOT `next_f32`, which is biased to `[0, ~0.5)`).
fn fill_centered_uniform(buf: &mut [f32], scale: f32, rng: &mut LcgRng) {
    for v in buf.iter_mut() {
        *v = safe_centered_uniform(rng) * 2.0 * scale;
    }
}

impl Mae {
    /// Build and initialise a fresh MAE model.
    ///
    /// Weight initialisation:
    /// - Linear / projection kernels: uniform `[-scale, scale)` with
    ///   `scale = 1 / sqrt(fan_in)` (Xavier-like).
    /// - Biases: zeros.
    /// - LayerNorm gammas: ones, betas: zeros.
    /// - Positional embeddings: small-magnitude uniform.
    /// - `mask_token`: small-magnitude uniform.
    ///
    /// # Errors
    /// Propagates any block-config validation error.
    pub fn new(cfg: MaeConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let n_patches = cfg.n_patches();
        let pp = cfg.patch_pixels();
        let edim = cfg.encoder_dim;
        let ddim = cfg.decoder_dim;

        let enc_scale = 1.0 / (pp as f32).sqrt();
        let mut patch_embed_weights = vec![0.0f32; edim * pp];
        fill_centered_uniform(&mut patch_embed_weights, enc_scale, rng);
        let patch_embed_bias = vec![0.0f32; edim];

        let pos_scale = 0.02f32; // canonical ViT pos-embed init magnitude
        let mut encoder_pos_embed = vec![0.0f32; n_patches * edim];
        fill_centered_uniform(&mut encoder_pos_embed, pos_scale, rng);

        // Encoder block stack — reuse ViTEncoderConfig to validate and create blocks.
        let enc_block_cfg =
            ViTEncoderConfig::new(edim, cfg.encoder_heads, cfg.mlp_ratio, cfg.encoder_depth)?;
        let mut encoder_blocks = Vec::with_capacity(cfg.encoder_depth);
        for _ in 0..cfg.encoder_depth {
            encoder_blocks.push(ViTBlock::new(enc_block_cfg.block_cfg.clone(), rng));
        }
        let encoder_norm_gamma = vec![1.0f32; edim];
        let encoder_norm_beta = vec![0.0f32; edim];

        let dec_in_scale = 1.0 / (edim as f32).sqrt();
        let mut decoder_embed_weights = vec![0.0f32; ddim * edim];
        fill_centered_uniform(&mut decoder_embed_weights, dec_in_scale, rng);
        let decoder_embed_bias = vec![0.0f32; ddim];

        let mut mask_token = vec![0.0f32; ddim];
        fill_centered_uniform(&mut mask_token, pos_scale, rng);

        let mut decoder_pos_embed = vec![0.0f32; n_patches * ddim];
        fill_centered_uniform(&mut decoder_pos_embed, pos_scale, rng);

        let dec_block_cfg =
            ViTEncoderConfig::new(ddim, cfg.decoder_heads, cfg.mlp_ratio, cfg.decoder_depth)?;
        let mut decoder_blocks = Vec::with_capacity(cfg.decoder_depth);
        for _ in 0..cfg.decoder_depth {
            decoder_blocks.push(ViTBlock::new(dec_block_cfg.block_cfg.clone(), rng));
        }
        let decoder_norm_gamma = vec![1.0f32; ddim];
        let decoder_norm_beta = vec![0.0f32; ddim];

        let pred_scale = 1.0 / (ddim as f32).sqrt();
        let mut decoder_pred_weights = vec![0.0f32; pp * ddim];
        fill_centered_uniform(&mut decoder_pred_weights, pred_scale, rng);
        let decoder_pred_bias = vec![0.0f32; pp];

        Ok(Self {
            config: cfg,
            patch_embed_weights,
            patch_embed_bias,
            encoder_pos_embed,
            encoder_blocks,
            encoder_norm_gamma,
            encoder_norm_beta,
            decoder_embed_weights,
            decoder_embed_bias,
            mask_token,
            decoder_pos_embed,
            decoder_blocks,
            decoder_norm_gamma,
            decoder_norm_beta,
            decoder_pred_weights,
            decoder_pred_bias,
        })
    }

    /// Encode an already-patchified image.
    ///
    /// `image_patches` is `[n_patches, patch_pixels]` row-major. Returns the
    /// `[n_visible, encoder_dim]` features of visible tokens plus the
    /// generated `MaskMeta`.
    ///
    /// # Errors
    /// - `image_patches.len() != n_patches · patch_pixels` → `DimensionMismatch`
    /// - mask-generation errors propagated
    pub fn encode(
        &self,
        image_patches: &[f32],
        rng: &mut LcgRng,
    ) -> VisionResult<(Vec<f32>, MaskMeta)> {
        let n_patches = self.config.n_patches();
        let pp = self.config.patch_pixels();
        let edim = self.config.encoder_dim;

        if n_patches == 0 {
            return Err(VisionError::EmptyInput("n_patches"));
        }
        let expected = n_patches * pp;
        if image_patches.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: image_patches.len(),
            });
        }

        // Patch embed: y = X · W^T + b — output [n_patches, encoder_dim]
        let mut embedded = linear(
            image_patches,
            &self.patch_embed_weights,
            &self.patch_embed_bias,
            pp,
            edim,
        );

        // Add encoder positional embeddings.
        for (i, v) in embedded.iter_mut().enumerate() {
            *v += self
                .encoder_pos_embed
                .get(i)
                .copied()
                .ok_or(VisionError::Internal(
                    "encoder_pos_embed shorter than embedded".into(),
                ))?;
        }

        // Generate mask and gather visible tokens.
        let mask_meta = generate_random_mask(n_patches, self.config.mask_ratio, rng)?;
        let n_visible = mask_meta.visible_ids.len();

        let mut visible_tokens = vec![0.0f32; n_visible * edim];
        for (out_i, &src_i) in mask_meta.visible_ids.iter().enumerate() {
            let src = embedded
                .get(src_i * edim..(src_i + 1) * edim)
                .ok_or(VisionError::Internal("visible idx out of range".into()))?;
            let dst = visible_tokens
                .get_mut(out_i * edim..(out_i + 1) * edim)
                .ok_or(VisionError::Internal(
                    "visible_tokens slice out of range".into(),
                ))?;
            dst.copy_from_slice(src);
        }

        // Encoder block stack (only on visible tokens — the MAE speed-up).
        // ViTBlock::forward rejects n_tokens == 0, so for mask_ratio == 1 we
        // produce an empty Vec without touching the blocks.
        let encoded = if n_visible == 0 {
            Vec::new()
        } else {
            let mut h = visible_tokens;
            for block in &self.encoder_blocks {
                h = block.forward(&h, n_visible)?;
            }
            layer_norm(
                &h,
                &self.encoder_norm_gamma,
                &self.encoder_norm_beta,
                n_visible,
                edim,
                1e-5,
            )
        };

        Ok((encoded, mask_meta))
    }

    /// Decode visible features and reconstruct per-patch pixels.
    ///
    /// Returns `[n_patches, patch_pixels]` (full sequence including
    /// reconstructions of the originally-visible positions).
    ///
    /// # Errors
    /// - `encoded_visible.len() != |visible_ids| · encoder_dim` →
    ///   `DimensionMismatch`
    pub fn decode(&self, encoded_visible: &[f32], mask_meta: &MaskMeta) -> VisionResult<Vec<f32>> {
        let n_patches = self.config.n_patches();
        let edim = self.config.encoder_dim;
        let ddim = self.config.decoder_dim;
        let pp = self.config.patch_pixels();
        let n_visible = mask_meta.visible_ids.len();
        let n_masked = mask_meta.masked_ids.len();

        if n_visible + n_masked != n_patches {
            return Err(VisionError::Internal(
                "MaskMeta visible + masked sizes do not sum to n_patches".into(),
            ));
        }
        if encoded_visible.len() != n_visible * edim {
            return Err(VisionError::DimensionMismatch {
                expected: n_visible * edim,
                got: encoded_visible.len(),
            });
        }

        // Project visible features encoder_dim → decoder_dim (if any).
        let visible_dec = if n_visible == 0 {
            Vec::new()
        } else {
            linear(
                encoded_visible,
                &self.decoder_embed_weights,
                &self.decoder_embed_bias,
                edim,
                ddim,
            )
        };

        // Scatter into the full-length [n_patches, ddim] sequence: visible at
        // their original ids, mask_token at masked ids.
        let mut full = vec![0.0f32; n_patches * ddim];
        for (vis_i, &dst_i) in mask_meta.visible_ids.iter().enumerate() {
            let src = visible_dec
                .get(vis_i * ddim..(vis_i + 1) * ddim)
                .ok_or(VisionError::Internal("visible_dec slice".into()))?;
            let dst = full
                .get_mut(dst_i * ddim..(dst_i + 1) * ddim)
                .ok_or(VisionError::Internal("full slice (visible)".into()))?;
            dst.copy_from_slice(src);
        }
        for &dst_i in &mask_meta.masked_ids {
            let dst = full
                .get_mut(dst_i * ddim..(dst_i + 1) * ddim)
                .ok_or(VisionError::Internal("full slice (masked)".into()))?;
            dst.copy_from_slice(&self.mask_token);
        }

        // Add decoder positional embeddings.
        for (i, v) in full.iter_mut().enumerate() {
            *v += self
                .decoder_pos_embed
                .get(i)
                .copied()
                .ok_or(VisionError::Internal("decoder_pos_embed".into()))?;
        }

        // Decoder block stack on the FULL sequence.
        let mut h = full;
        for block in &self.decoder_blocks {
            h = block.forward(&h, n_patches)?;
        }
        let post_norm = layer_norm(
            &h,
            &self.decoder_norm_gamma,
            &self.decoder_norm_beta,
            n_patches,
            ddim,
            1e-5,
        );

        // Per-patch pixel projection [n_patches, ddim] → [n_patches, patch_pixels].
        let reconstructed = linear(
            &post_norm,
            &self.decoder_pred_weights,
            &self.decoder_pred_bias,
            ddim,
            pp,
        );
        Ok(reconstructed)
    }
}

// ─── mae_loss ────────────────────────────────────────────────────────────────

/// Canonical MAE reconstruction loss.
///
/// Mean squared error averaged over **masked positions only** (visible
/// positions are intentionally ignored — this is the core MAE insight).
///
/// # Errors
/// - `reconstructed.len() != ground_truth_patches.len()` → `DimensionMismatch`
/// - lengths not divisible by `n_patches` → `DimensionMismatch`
/// - any masked index out of range → `Internal`
/// - empty `masked_ids` → returns `Ok(0.0)`  (no error: with mask_ratio=0
///   there is nothing to score; downstream code should treat 0 as "no signal")
pub fn mae_loss(
    reconstructed: &[f32],
    ground_truth_patches: &[f32],
    mask_meta: &MaskMeta,
) -> VisionResult<f32> {
    if reconstructed.len() != ground_truth_patches.len() {
        return Err(VisionError::DimensionMismatch {
            expected: reconstructed.len(),
            got: ground_truth_patches.len(),
        });
    }
    let n_patches = mask_meta.visible_ids.len() + mask_meta.masked_ids.len();
    if n_patches == 0 {
        return Err(VisionError::EmptyInput("mask_meta n_patches"));
    }
    if reconstructed.len() % n_patches != 0 {
        return Err(VisionError::DimensionMismatch {
            expected: n_patches,
            got: reconstructed.len(),
        });
    }
    let pp = reconstructed.len() / n_patches;
    if mask_meta.masked_ids.is_empty() {
        return Ok(0.0);
    }
    let mut sum_sq = 0.0f64;
    let mut count: u64 = 0;
    for &mi in &mask_meta.masked_ids {
        let r = reconstructed
            .get(mi * pp..(mi + 1) * pp)
            .ok_or(VisionError::Internal("loss: masked idx".into()))?;
        let g = ground_truth_patches
            .get(mi * pp..(mi + 1) * pp)
            .ok_or(VisionError::Internal("loss: masked idx (gt)".into()))?;
        for (rv, gv) in r.iter().zip(g.iter()) {
            let d = (*rv - *gv) as f64;
            sum_sq += d * d;
            count += 1;
        }
    }
    let mean = if count == 0 {
        0.0
    } else {
        sum_sq / (count as f64)
    };
    Ok(mean as f32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn make_tiny_cfg() -> MaeConfig {
        // img 8 / patch 4 = 2x2 = 4 patches, in_chans 3, patch_pixels = 48
        MaeConfig::new(8, 4, 3, 16, 2, 4, 8, 1, 4, 2, 0.5).expect("valid tiny cfg")
    }

    fn make_medium_cfg() -> MaeConfig {
        // img 16 / patch 4 = 4x4 = 16 patches, in_chans 3, patch_pixels = 48
        MaeConfig::new(16, 4, 3, 32, 2, 4, 16, 1, 4, 2, 0.75).expect("valid med cfg")
    }

    // ── generate_random_mask ──────────────────────────────────────────────────

    #[test]
    fn mask_union_and_disjoint() {
        let mut rng = LcgRng::new(1);
        let n = 16;
        let m = generate_random_mask(n, 0.5, &mut rng).expect("ok");
        let v: HashSet<usize> = m.visible_ids.iter().copied().collect();
        let k: HashSet<usize> = m.masked_ids.iter().copied().collect();
        assert!(v.is_disjoint(&k));
        let union: HashSet<usize> = v.union(&k).copied().collect();
        let expected: HashSet<usize> = (0..n).collect();
        assert_eq!(union, expected);
        assert_eq!(v.len() + k.len(), n);
    }

    #[test]
    fn mask_count_matches_round() {
        let mut rng = LcgRng::new(2);
        // 0.75 * 100 = 75 exactly
        let m = generate_random_mask(100, 0.75, &mut rng).expect("ok");
        assert_eq!(m.masked_ids.len(), 75);
        assert_eq!(m.visible_ids.len(), 25);
    }

    #[test]
    fn mask_count_rounds_correctly() {
        let mut rng = LcgRng::new(3);
        // 0.7 * 10 = 7 exactly
        let m = generate_random_mask(10, 0.7, &mut rng).expect("ok");
        assert_eq!(m.masked_ids.len(), 7);
    }

    #[test]
    fn mask_ratio_zero_all_visible() {
        let mut rng = LcgRng::new(4);
        let m = generate_random_mask(8, 0.0, &mut rng).expect("ok");
        assert_eq!(m.masked_ids.len(), 0);
        assert_eq!(m.visible_ids.len(), 8);
        assert_eq!(m.visible_ids, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn mask_ratio_one_all_masked() {
        let mut rng = LcgRng::new(5);
        let m = generate_random_mask(8, 1.0, &mut rng).expect("ok");
        assert_eq!(m.masked_ids.len(), 8);
        assert_eq!(m.visible_ids.len(), 0);
        assert_eq!(m.masked_ids, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn mask_deterministic_same_seed() {
        let mut a = LcgRng::new(42);
        let mut b = LcgRng::new(42);
        let ma = generate_random_mask(64, 0.75, &mut a).expect("ok");
        let mb = generate_random_mask(64, 0.75, &mut b).expect("ok");
        assert_eq!(ma, mb);
    }

    #[test]
    fn mask_sorted_ascending() {
        let mut rng = LcgRng::new(6);
        let m = generate_random_mask(50, 0.6, &mut rng).expect("ok");
        for w in m.visible_ids.windows(2) {
            assert!(w[0] < w[1]);
        }
        for w in m.masked_ids.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn mask_invalid_ratio_errors() {
        let mut rng = LcgRng::new(7);
        assert!(generate_random_mask(8, -0.1, &mut rng).is_err());
        assert!(generate_random_mask(8, 1.5, &mut rng).is_err());
        assert!(generate_random_mask(8, f32::NAN, &mut rng).is_err());
    }

    #[test]
    fn mask_n_patches_zero_errors() {
        let mut rng = LcgRng::new(8);
        let r = generate_random_mask(0, 0.5, &mut rng);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }

    // ── Mae construction / config ─────────────────────────────────────────────

    #[test]
    fn cfg_patch_not_divisible_errors() {
        let r = MaeConfig::new(7, 4, 3, 16, 1, 4, 8, 1, 4, 2, 0.5);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn cfg_zero_channels_errors() {
        let r = MaeConfig::new(8, 4, 0, 16, 1, 4, 8, 1, 4, 2, 0.5);
        assert!(r.is_err());
    }

    #[test]
    fn cfg_zero_encoder_dim_errors() {
        let r = MaeConfig::new(8, 4, 3, 0, 1, 4, 8, 1, 4, 2, 0.5);
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(0))));
    }

    #[test]
    fn cfg_zero_decoder_dim_errors() {
        let r = MaeConfig::new(8, 4, 3, 16, 1, 4, 0, 1, 4, 2, 0.5);
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(0))));
    }

    #[test]
    fn cfg_zero_depth_errors() {
        let r1 = MaeConfig::new(8, 4, 3, 16, 0, 4, 8, 1, 4, 2, 0.5);
        let r2 = MaeConfig::new(8, 4, 3, 16, 1, 4, 8, 0, 4, 2, 0.5);
        assert!(r1.is_err());
        assert!(r2.is_err());
    }

    #[test]
    fn cfg_mask_ratio_out_of_range_errors() {
        let r1 = MaeConfig::new(8, 4, 3, 16, 1, 4, 8, 1, 4, 2, -0.1);
        let r2 = MaeConfig::new(8, 4, 3, 16, 1, 4, 8, 1, 4, 2, 1.5);
        assert!(r1.is_err());
        assert!(r2.is_err());
    }

    #[test]
    fn cfg_n_patches_and_pixels() {
        let cfg = make_tiny_cfg();
        assert_eq!(cfg.n_patches(), 4);
        assert_eq!(cfg.patch_pixels(), 4 * 4 * 3);
    }

    // ── encode / decode shapes ────────────────────────────────────────────────

    #[test]
    fn encode_shape() {
        let cfg = make_medium_cfg();
        let mut rng = LcgRng::new(11);
        let mae = Mae::new(cfg.clone(), &mut rng).expect("ok");
        let n_patches = cfg.n_patches();
        let pp = cfg.patch_pixels();
        let edim = cfg.encoder_dim;
        let patches = vec![0.1f32; n_patches * pp];
        let mut rng2 = LcgRng::new(99);
        let (enc, mask) = mae.encode(&patches, &mut rng2).expect("ok");
        assert_eq!(enc.len(), mask.visible_ids.len() * edim);
    }

    #[test]
    fn decode_shape_matches_patches() {
        let cfg = make_medium_cfg();
        let mut rng = LcgRng::new(13);
        let mae = Mae::new(cfg.clone(), &mut rng).expect("ok");
        let n_patches = cfg.n_patches();
        let pp = cfg.patch_pixels();
        let patches = vec![0.1f32; n_patches * pp];
        let mut rng2 = LcgRng::new(101);
        let (enc, mask) = mae.encode(&patches, &mut rng2).expect("ok");
        let recon = mae.decode(&enc, &mask).expect("ok");
        assert_eq!(recon.len(), n_patches * pp);
    }

    #[test]
    fn full_pipeline_deterministic_same_seed() {
        let cfg = make_medium_cfg();
        let mut rng_a = LcgRng::new(33);
        let mae_a = Mae::new(cfg.clone(), &mut rng_a).expect("ok");
        let mut rng_b = LcgRng::new(33);
        let mae_b = Mae::new(cfg.clone(), &mut rng_b).expect("ok");

        let n_patches = cfg.n_patches();
        let pp = cfg.patch_pixels();
        let mut patches = vec![0.0f32; n_patches * pp];
        let mut rin = LcgRng::new(5);
        for v in patches.iter_mut() {
            *v = (rin.next_u32() as f32) / 2_147_483_648.0;
        }

        let mut r_a = LcgRng::new(77);
        let mut r_b = LcgRng::new(77);
        let (ea, ma) = mae_a.encode(&patches, &mut r_a).expect("ok");
        let (eb, mb) = mae_b.encode(&patches, &mut r_b).expect("ok");
        assert_eq!(ma, mb);
        for (a, b) in ea.iter().zip(eb.iter()) {
            assert!((a - b).abs() < 1e-6, "encode differs: {a} vs {b}");
        }
        let recon_a = mae_a.decode(&ea, &ma).expect("ok");
        let recon_b = mae_b.decode(&eb, &mb).expect("ok");
        for (a, b) in recon_a.iter().zip(recon_b.iter()) {
            assert!((a - b).abs() < 1e-6, "decode differs: {a} vs {b}");
        }
    }

    // ── encode / decode error paths ───────────────────────────────────────────

    #[test]
    fn encode_dimension_mismatch_errors() {
        let cfg = make_tiny_cfg();
        let mut rng = LcgRng::new(15);
        let mae = Mae::new(cfg.clone(), &mut rng).expect("ok");
        // wrong size: 1 patch short
        let pp = cfg.patch_pixels();
        let patches = vec![0.0f32; (cfg.n_patches() - 1) * pp];
        let mut rng2 = LcgRng::new(16);
        let r = mae.encode(&patches, &mut rng2);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn decode_wrong_visible_length_errors() {
        let cfg = make_tiny_cfg();
        let mut rng = LcgRng::new(17);
        let mae = Mae::new(cfg.clone(), &mut rng).expect("ok");
        // Build a valid mask, then truncate features.
        let mut rng_m = LcgRng::new(18);
        let mask = generate_random_mask(cfg.n_patches(), 0.5, &mut rng_m).expect("ok");
        let wrong = vec![0.0f32; mask.visible_ids.len() * cfg.encoder_dim - 1];
        let r = mae.decode(&wrong, &mask);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    // ── mae_loss ──────────────────────────────────────────────────────────────

    #[test]
    fn loss_zero_when_match_on_masked_positions() {
        // Ground truth and reconstructed only need to agree on masked positions.
        let mask = MaskMeta {
            visible_ids: vec![0, 2],
            masked_ids: vec![1, 3],
        };
        let pp = 5;
        let n_patches = 4;
        let mut gt = vec![0.0f32; n_patches * pp];
        let mut recon = vec![0.0f32; n_patches * pp];
        for (i, g) in gt.iter_mut().enumerate() {
            *g = i as f32;
        }
        // Match at masked positions (1, 3):
        for &mi in &mask.masked_ids {
            for k in 0..pp {
                recon[mi * pp + k] = gt[mi * pp + k];
            }
        }
        // Differ on visible positions (0, 2):
        for k in 0..pp {
            recon[k] = 999.0;
            recon[2 * pp + k] = -777.0;
        }
        let loss = mae_loss(&recon, &gt, &mask).expect("ok");
        assert!(
            loss.abs() < 1e-6,
            "loss should be 0 when masked match: {loss}"
        );
    }

    #[test]
    fn loss_independent_of_visible_positions() {
        let mask = MaskMeta {
            visible_ids: vec![0, 2],
            masked_ids: vec![1, 3],
        };
        let pp = 3;
        let n_patches = 4;
        let mut gt = vec![0.0f32; n_patches * pp];
        let mut recon_a = vec![0.0f32; n_patches * pp];
        let mut recon_b = vec![0.0f32; n_patches * pp];
        for i in 0..n_patches * pp {
            gt[i] = (i as f32) * 0.1;
            recon_a[i] = gt[i] + 0.5; // off everywhere
            recon_b[i] = gt[i] + 0.5;
        }
        // Now alter ONLY visible positions in recon_b drastically:
        for &vi in &mask.visible_ids {
            for k in 0..pp {
                recon_b[vi * pp + k] = 1234.0;
            }
        }
        let la = mae_loss(&recon_a, &gt, &mask).expect("ok");
        let lb = mae_loss(&recon_b, &gt, &mask).expect("ok");
        assert!(
            (la - lb).abs() < 1e-6,
            "loss depends on visible: {la} vs {lb}"
        );
    }

    #[test]
    fn loss_dimension_mismatch_errors() {
        let mask = MaskMeta {
            visible_ids: vec![0],
            masked_ids: vec![1],
        };
        let r = mae_loss(&[0.0; 4], &[0.0; 5], &mask);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn loss_mask_ratio_zero_returns_zero() {
        let mut rng = LcgRng::new(21);
        let m = generate_random_mask(6, 0.0, &mut rng).expect("ok");
        let r = vec![1.0f32; 6 * 4];
        let g = vec![2.0f32; 6 * 4];
        let l = mae_loss(&r, &g, &m).expect("ok");
        assert!(l.abs() < 1e-6, "no masked → loss = 0; got {l}");
    }

    #[test]
    fn loss_positive_when_recon_off() {
        let mask = MaskMeta {
            visible_ids: vec![0],
            masked_ids: vec![1],
        };
        let pp = 4;
        let gt = vec![0.0f32; 2 * pp];
        let mut recon = vec![0.0f32; 2 * pp];
        for k in 0..pp {
            recon[pp + k] = 1.0; // squared error of 1 per element
        }
        let l = mae_loss(&recon, &gt, &mask).expect("ok");
        assert!((l - 1.0).abs() < 1e-6, "expected MSE=1, got {l}");
    }

    // ── encode produces finite outputs ────────────────────────────────────────

    #[test]
    fn encode_decode_finite() {
        let cfg = make_medium_cfg();
        let mut rng = LcgRng::new(45);
        let mae = Mae::new(cfg.clone(), &mut rng).expect("ok");
        let n_patches = cfg.n_patches();
        let pp = cfg.patch_pixels();
        let mut patches = vec![0.0f32; n_patches * pp];
        let mut rin = LcgRng::new(55);
        rin.fill_normal(&mut patches);
        let mut r2 = LcgRng::new(66);
        let (enc, mask) = mae.encode(&patches, &mut r2).expect("ok");
        assert!(enc.iter().all(|v| v.is_finite()));
        let recon = mae.decode(&enc, &mask).expect("ok");
        assert!(recon.iter().all(|v| v.is_finite()));
    }

    // ── identity-ish weight round trip ────────────────────────────────────────

    #[test]
    fn identity_decoder_reconstructs_mask_token_at_masked() {
        // Hand-craft the decoder so that, for any visible token, the value of
        // the reconstructed patch at MASKED positions depends ONLY on
        // mask_token and decoder_pos_embed, not on visible content. We do this
        // by:
        //   1. Setting decoder Q/K/V projections to zero → MHSA output is 0.
        //   2. Setting MLP projections to zero → MLP residual contributes 0.
        //   3. Pre-norm LN1/LN2 weights to 1 / biases to 0 (default).
        //   4. final layer norm gamma=1, beta=0.
        //   5. decoder_pred_weights = identity over decoder_dim → patch_pixels
        //      identity when patch_pixels == decoder_dim, biases zero.
        //   6. decoder_pos_embed = 0 to eliminate its contribution.
        //
        // Then at any masked position i: input to decoder is mask_token; after
        // all the (post-norm) zero residuals it remains mask_token; LN
        // normalises it; the linear projection passes it through. We check
        // shape and that the masked-position outputs are independent of the
        // visible token content by perturbing the visible input.
        //
        // To make the projection "identity-ish" cleanly, pick patch_pixels =
        // decoder_dim = 4. So img_size = 2, patch_size = 2, in_channels = 1.
        let mut cfg = MaeConfig::new(2, 2, 1, 4, 1, 1, 4, 1, 1, 1, 0.5).expect("ok");
        cfg.mask_ratio = 0.5;
        let mut rng = LcgRng::new(123);
        let mut mae = Mae::new(cfg.clone(), &mut rng).expect("ok");

        // Zero out all decoder block weights (Q/K/V, output, MLP1, MLP2)
        for block in mae.decoder_blocks.iter_mut() {
            for v in block.weights.qkv_weight.iter_mut() {
                *v = 0.0;
            }
            for v in block.weights.qkv_bias.iter_mut() {
                *v = 0.0;
            }
            for v in block.weights.out_weight.iter_mut() {
                *v = 0.0;
            }
            for v in block.weights.out_bias.iter_mut() {
                *v = 0.0;
            }
            for v in block.weights.mlp1_weight.iter_mut() {
                *v = 0.0;
            }
            for v in block.weights.mlp1_bias.iter_mut() {
                *v = 0.0;
            }
            for v in block.weights.mlp2_weight.iter_mut() {
                *v = 0.0;
            }
            for v in block.weights.mlp2_bias.iter_mut() {
                *v = 0.0;
            }
            // LayerNorms: gamma=1, beta=0 already.
        }
        // Final norm: gamma=1, beta=0 already.
        // decoder_pos_embed → 0 (no positional contribution at masked sites)
        for v in mae.decoder_pos_embed.iter_mut() {
            *v = 0.0;
        }
        // decoder_pred = identity (patch_pixels == decoder_dim == 4)
        for v in mae.decoder_pred_weights.iter_mut() {
            *v = 0.0;
        }
        for i in 0..4 {
            mae.decoder_pred_weights[i * 4 + i] = 1.0;
        }
        for v in mae.decoder_pred_bias.iter_mut() {
            *v = 0.0;
        }
        // mask_token set to a known value
        mae.mask_token = vec![0.1, -0.2, 0.3, -0.4];

        // Run encode with TWO different visible-token contents but the SAME
        // mask (achieved by reusing the same RNG seed for encode).
        let n_patches = cfg.n_patches();
        let pp = cfg.patch_pixels();

        let patches_a = vec![1.0f32; n_patches * pp];
        let mut patches_b = vec![1.0f32; n_patches * pp];
        // Different content at every pixel:
        for v in patches_b.iter_mut() {
            *v = 7.7;
        }

        let mut r_a = LcgRng::new(2024);
        let mut r_b = LcgRng::new(2024);
        let (enc_a, ma) = mae.encode(&patches_a, &mut r_a).expect("ok");
        let (enc_b, mb) = mae.encode(&patches_b, &mut r_b).expect("ok");
        assert_eq!(ma, mb, "same RNG seed must produce same mask");

        let recon_a = mae.decode(&enc_a, &ma).expect("ok");
        let recon_b = mae.decode(&enc_b, &mb).expect("ok");
        // At masked positions, reconstruction must be identical (zero
        // attention prevents visible info from reaching mask sites, and zero
        // decoder_pos_embed eliminates positional bias).
        //
        // Compute LN(mask_token) for the expected value: with 4 elements
        // (0.1, -0.2, 0.3, -0.4): mean=-0.05, centred=(0.15,-0.15,0.35,-0.35).
        // var = (.0225 + .0225 + .1225 + .1225)/4 = .0725
        // inv_std = 1/sqrt(.0725 + 1e-5) ≈ 3.71...
        let mean = (0.1f32 + (-0.2) + 0.3 + (-0.4)) / 4.0;
        let centred = [0.1f32 - mean, -0.2 - mean, 0.3 - mean, -0.4 - mean];
        let var = centred.iter().map(|c| c * c).sum::<f32>() / 4.0;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        let expected_at_mask: Vec<f32> = centred.iter().map(|c| c * inv_std).collect();

        for &mi in &ma.masked_ids {
            for k in 0..pp {
                let a = recon_a[mi * pp + k];
                let b = recon_b[mi * pp + k];
                assert!(
                    (a - b).abs() < 1e-5,
                    "masked pos {mi} k={k}: a={a} b={b} (depends on visible!)"
                );
                let exp = expected_at_mask[k];
                assert!(
                    (a - exp).abs() < 1e-4,
                    "masked pos {mi} k={k}: got {a} expected {exp}"
                );
            }
        }
    }

    // ── extra coverage ────────────────────────────────────────────────────────

    #[test]
    fn mask_full_ratio_encoder_skipped() {
        // mask_ratio == 1: no visible tokens — encoder returns empty Vec
        // without panicking and decode reconstructs all-mask_token paths.
        let cfg = MaeConfig::new(4, 2, 1, 4, 1, 1, 4, 1, 1, 1, 1.0).expect("ok");
        let mut rng = LcgRng::new(31);
        let mae = Mae::new(cfg.clone(), &mut rng).expect("ok");
        let pp = cfg.patch_pixels();
        let n = cfg.n_patches();
        let patches = vec![0.0f32; n * pp];
        let mut r2 = LcgRng::new(32);
        let (enc, mask) = mae.encode(&patches, &mut r2).expect("ok");
        assert_eq!(enc.len(), 0);
        assert_eq!(mask.masked_ids.len(), n);
        let recon = mae.decode(&enc, &mask).expect("ok");
        assert_eq!(recon.len(), n * pp);
        assert!(recon.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mask_zero_ratio_full_encoder() {
        let cfg = MaeConfig::new(4, 2, 1, 4, 1, 1, 4, 1, 1, 1, 0.0).expect("ok");
        let mut rng = LcgRng::new(41);
        let mae = Mae::new(cfg.clone(), &mut rng).expect("ok");
        let pp = cfg.patch_pixels();
        let n = cfg.n_patches();
        let patches = vec![0.1f32; n * pp];
        let mut r2 = LcgRng::new(42);
        let (enc, mask) = mae.encode(&patches, &mut r2).expect("ok");
        assert_eq!(mask.masked_ids.len(), 0);
        assert_eq!(mask.visible_ids.len(), n);
        assert_eq!(enc.len(), n * cfg.encoder_dim);
    }
}

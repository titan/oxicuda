//! NaViT — "Native Resolution ViT" (Dehghani et al. 2023): patch-n-pack.
//!
//! A standard ViT forces every image to a fixed square resolution. NaViT instead
//! processes images at their *native* aspect ratio and resolution by **packing**
//! the patch tokens of several variable-sized images into a single sequence
//! (example packing), exactly like packing variable-length text examples into one
//! row. Two pieces of machinery make this work and are implemented here:
//!
//! 1. **Factorised positional embeddings.** Because grid shapes vary, NaViT uses
//!    separable learned row/column position embeddings `pos = pos_row[i] +
//!    pos_col[j]` so the same table serves any `H×W` grid up to a maximum side.
//!    See [`NaViTConfig::factorised_pos`].
//! 2. **Block-diagonal (example) attention mask.** When patches from several
//!    images share one sequence, self-attention must be confined to within each
//!    image. [`packed_attention_mask`] builds the additive mask
//!    (`0` allowed / `-inf` blocked) from the per-image token counts.
//!
//! The [`NaViT::patchify_pack`] routine flattens and linearly embeds a batch of
//! variable-resolution images, adds the factorised positions, and concatenates
//! them into one packed `[total_tokens × d_model]` tensor together with the
//! segment ids needed to build the mask.

use crate::error::{MmResult, MultiModalError};

/// One image's spatial description for NaViT packing.
#[derive(Debug, Clone, Copy)]
pub struct ImageShape {
    /// Image height in pixels (must be a multiple of `patch_size`).
    pub height: usize,
    /// Image width in pixels (must be a multiple of `patch_size`).
    pub width: usize,
    /// Channel count.
    pub channels: usize,
}

impl ImageShape {
    /// Number of patch rows.
    #[must_use]
    pub fn grid_rows(&self, patch: usize) -> usize {
        self.height / patch
    }
    /// Number of patch columns.
    #[must_use]
    pub fn grid_cols(&self, patch: usize) -> usize {
        self.width / patch
    }
    /// Total patches for this image.
    #[must_use]
    pub fn n_patches(&self, patch: usize) -> usize {
        self.grid_rows(patch) * self.grid_cols(patch)
    }
}

/// NaViT configuration with factorised positional tables.
#[derive(Debug, Clone)]
pub struct NaViTConfig {
    /// Square patch side in pixels.
    pub patch_size: usize,
    /// Channel count of input images.
    pub channels: usize,
    /// Model / token dimension.
    pub d_model: usize,
    /// Maximum number of patch rows the row table supports.
    pub max_rows: usize,
    /// Maximum number of patch columns the column table supports.
    pub max_cols: usize,
}

impl NaViTConfig {
    /// Whether this config exposes factorised positions (always true; provided
    /// for API symmetry / documentation).
    #[must_use]
    pub fn factorised_pos(&self) -> bool {
        true
    }

    /// Tiny preset for tests.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            patch_size: 4,
            channels: 3,
            d_model: 8,
            max_rows: 16,
            max_cols: 16,
        }
    }

    /// Patch vector length (`channels * patch²`).
    #[must_use]
    pub fn patch_dim(&self) -> usize {
        self.channels * self.patch_size * self.patch_size
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// - [`MultiModalError::InvalidFeatureDim`] when `d_model == 0`, `channels
    ///   == 0`, or `patch_size == 0`.
    /// - [`MultiModalError::InvalidPatchCount`] when `max_rows == 0` or
    ///   `max_cols == 0`.
    pub fn validate(&self) -> MmResult<()> {
        if self.d_model == 0 || self.channels == 0 || self.patch_size == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if self.max_rows == 0 || self.max_cols == 0 {
            return Err(MultiModalError::InvalidPatchCount { n_patches: 0 });
        }
        Ok(())
    }
}

/// Learned weights for NaViT patch embedding + factorised positions.
#[derive(Debug, Clone)]
pub struct NaViTWeights {
    /// Patch projection `[patch_dim × d_model]` row-major.
    pub patch_proj: Vec<f32>,
    /// Row positional table `[max_rows × d_model]`.
    pub pos_row: Vec<f32>,
    /// Column positional table `[max_cols × d_model]`.
    pub pos_col: Vec<f32>,
}

impl NaViTWeights {
    /// All-zero weights for the given config.
    #[must_use]
    pub fn zeros(cfg: &NaViTConfig) -> Self {
        Self {
            patch_proj: vec![0.0_f32; cfg.patch_dim() * cfg.d_model],
            pos_row: vec![0.0_f32; cfg.max_rows * cfg.d_model],
            pos_col: vec![0.0_f32; cfg.max_cols * cfg.d_model],
        }
    }
}

/// Output of [`NaViT::patchify_pack`].
#[derive(Debug, Clone, PartialEq)]
pub struct PackedSequence {
    /// Packed token features `[total_tokens × d_model]` row-major.
    pub tokens: Vec<f32>,
    /// Per-token image (segment) id, length `total_tokens`.
    pub segment_ids: Vec<usize>,
    /// Number of tokens contributed by each packed image.
    pub seq_lens: Vec<usize>,
}

/// NaViT patch-and-pack front-end.
pub struct NaViT;

impl NaViT {
    /// Flatten, embed and pack a batch of variable-resolution images.
    ///
    /// `images[i]` is the row-major pixel buffer for image `shapes[i]`
    /// (`channels × height × width`). All images share the config's channel count
    /// and patch size. Returns the packed token sequence plus segmentation.
    ///
    /// # Errors
    /// - Any error from [`NaViTConfig::validate`].
    /// - [`MultiModalError::EmptyInput`] when `images` is empty.
    /// - [`MultiModalError::DimensionMismatch`] when a buffer length or channel
    ///   count is inconsistent with its declared shape.
    /// - [`MultiModalError::InvalidPatchCount`] when a side is not a multiple of
    ///   the patch size, or a grid exceeds the positional tables.
    pub fn patchify_pack(
        images: &[&[f32]],
        shapes: &[ImageShape],
        cfg: &NaViTConfig,
        weights: &NaViTWeights,
    ) -> MmResult<PackedSequence> {
        cfg.validate()?;
        if images.is_empty() || shapes.is_empty() {
            return Err(MultiModalError::EmptyInput);
        }
        if images.len() != shapes.len() {
            return Err(MultiModalError::DimensionMismatch {
                expected: shapes.len(),
                got: images.len(),
            });
        }
        let p = cfg.patch_size;
        let d = cfg.d_model;
        let patch_dim = cfg.patch_dim();

        let mut tokens: Vec<f32> = Vec::new();
        let mut segment_ids: Vec<usize> = Vec::new();
        let mut seq_lens: Vec<usize> = Vec::new();

        for (img_idx, (&img, &shape)) in images.iter().zip(shapes.iter()).enumerate() {
            if shape.channels != cfg.channels {
                return Err(MultiModalError::DimensionMismatch {
                    expected: cfg.channels,
                    got: shape.channels,
                });
            }
            if shape.height % p != 0 || shape.width % p != 0 {
                return Err(MultiModalError::InvalidPatchCount { n_patches: 0 });
            }
            let g_rows = shape.grid_rows(p);
            let g_cols = shape.grid_cols(p);
            if g_rows > cfg.max_rows || g_cols > cfg.max_cols {
                return Err(MultiModalError::InvalidPatchCount {
                    n_patches: g_rows * g_cols,
                });
            }
            let expected_len = shape.channels * shape.height * shape.width;
            if img.len() != expected_len {
                return Err(MultiModalError::DimensionMismatch {
                    expected: expected_len,
                    got: img.len(),
                });
            }

            let mut count = 0usize;
            for gr in 0..g_rows {
                for gc in 0..g_cols {
                    // Gather the patch pixels into a [patch_dim] vector in
                    // channel-major, row-major order.
                    let mut patch = vec![0.0_f32; patch_dim];
                    let mut pi = 0usize;
                    for c in 0..shape.channels {
                        for py in 0..p {
                            let y = gr * p + py;
                            for px in 0..p {
                                let x = gc * p + px;
                                let off = (c * shape.height + y) * shape.width + x;
                                patch[pi] = img[off];
                                pi += 1;
                            }
                        }
                    }
                    // Linear embed: token = patch · patch_proj.
                    let mut tok = vec![0.0_f32; d];
                    for (o, slot) in tok.iter_mut().enumerate() {
                        let mut acc = 0.0_f32;
                        for k in 0..patch_dim {
                            acc += patch[k] * weights.patch_proj[k * d + o];
                        }
                        *slot = acc;
                    }
                    // Factorised position: + pos_row[gr] + pos_col[gc].
                    for o in 0..d {
                        tok[o] += weights.pos_row[gr * d + o] + weights.pos_col[gc * d + o];
                    }
                    tokens.extend_from_slice(&tok);
                    segment_ids.push(img_idx);
                    count += 1;
                }
            }
            seq_lens.push(count);
        }

        if tokens.iter().any(|v| !v.is_finite()) {
            return Err(MultiModalError::NanEncountered {
                location: "navit_patchify_pack",
            });
        }
        Ok(PackedSequence {
            tokens,
            segment_ids,
            seq_lens,
        })
    }
}

/// Build the additive block-diagonal (example) attention mask for a packed
/// sequence described by `seq_lens`.
///
/// The returned `[total × total]` matrix holds `0.0` where attention is allowed
/// (both positions belong to the same image) and `f32::NEG_INFINITY` where it is
/// blocked (cross-image). Adding it to the pre-softmax scores yields per-image
/// self-attention within one packed sequence.
///
/// # Errors
/// [`MultiModalError::EmptyInput`] when `seq_lens` is empty or sums to zero.
pub fn packed_attention_mask(seq_lens: &[usize]) -> MmResult<Vec<f32>> {
    if seq_lens.is_empty() {
        return Err(MultiModalError::EmptyInput);
    }
    let total: usize = seq_lens.iter().sum();
    if total == 0 {
        return Err(MultiModalError::EmptyInput);
    }
    // Segment id per position.
    let mut seg = Vec::with_capacity(total);
    for (s, &len) in seq_lens.iter().enumerate() {
        for _ in 0..len {
            seg.push(s);
        }
    }
    let mut mask = vec![0.0_f32; total * total];
    for i in 0..total {
        for j in 0..total {
            if seg[i] != seg[j] {
                mask[i * total + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_image(shape: ImageShape, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0_f32; shape.channels * shape.height * shape.width];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn packs_variable_resolution_images() {
        let cfg = NaViTConfig::tiny(); // patch 4, ch 3, d 8
        let w = NaViTWeights::zeros(&cfg);
        // Two different shapes: 8x4 (2x1 grid = 2 patches), 4x8 (1x2 = 2 patches).
        let s0 = ImageShape {
            height: 8,
            width: 4,
            channels: 3,
        };
        let s1 = ImageShape {
            height: 4,
            width: 8,
            channels: 3,
        };
        let i0 = make_image(s0, 1);
        let i1 = make_image(s1, 2);
        let packed =
            NaViT::patchify_pack(&[&i0, &i1], &[s0, s1], &cfg, &w).expect("pack should succeed");
        assert_eq!(packed.seq_lens, vec![2, 2]);
        assert_eq!(packed.tokens.len(), 4 * cfg.d_model);
        assert_eq!(packed.segment_ids, vec![0, 0, 1, 1]);
    }

    #[test]
    fn factorised_positions_added() {
        // With a zero patch projection, tokens equal pos_row[gr] + pos_col[gc].
        let cfg = NaViTConfig::tiny();
        let mut w = NaViTWeights::zeros(&cfg);
        // Distinguish positions: pos_row[r][0] = 100*r, pos_col[c][0] = c.
        for r in 0..cfg.max_rows {
            w.pos_row[r * cfg.d_model] = 100.0 * r as f32;
        }
        for c in 0..cfg.max_cols {
            w.pos_col[c * cfg.d_model] = c as f32;
        }
        let shape = ImageShape {
            height: 8,
            width: 8,
            channels: 3,
        }; // 2x2 grid
        let img = vec![0.0_f32; 3 * 8 * 8];
        let packed = NaViT::patchify_pack(&[&img], &[shape], &cfg, &w).expect("pack");
        // Row-major patch order: (0,0)->0, (0,1)->1, (1,0)->100, (1,1)->101.
        let d = cfg.d_model;
        assert!((packed.tokens[0] - 0.0).abs() < 1e-5);
        assert!((packed.tokens[d] - 1.0).abs() < 1e-5);
        assert!((packed.tokens[2 * d] - 100.0).abs() < 1e-5);
        assert!((packed.tokens[3 * d] - 101.0).abs() < 1e-5);
    }

    #[test]
    fn patch_projection_applied() {
        // patch_proj that sums the patch into channel 0; constant image → known.
        let cfg = NaViTConfig {
            patch_size: 2,
            channels: 1,
            d_model: 4,
            max_rows: 8,
            max_cols: 8,
        };
        let mut w = NaViTWeights::zeros(&cfg);
        // patch_dim = 1*2*2 = 4. proj[k][0] = 1 → token[0] = sum(patch).
        for k in 0..cfg.patch_dim() {
            w.patch_proj[k * cfg.d_model] = 1.0;
        }
        let shape = ImageShape {
            height: 2,
            width: 2,
            channels: 1,
        }; // single 1-patch image
        let img = vec![1.0, 2.0, 3.0, 4.0]; // sum = 10
        let packed = NaViT::patchify_pack(&[&img], &[shape], &cfg, &w).expect("pack");
        assert!(
            (packed.tokens[0] - 10.0).abs() < 1e-5,
            "got {}",
            packed.tokens[0]
        );
    }

    #[test]
    fn attention_mask_is_block_diagonal() {
        // Two images of 2 and 3 tokens → 5x5 mask blocking cross-image entries.
        let mask = packed_attention_mask(&[2, 3]).expect("mask");
        let n = 5;
        // Within image 0 (0,1): allowed.
        assert_eq!(mask[1], 0.0);
        // Within image 1 (2,3,4): allowed.
        assert_eq!(mask[2 * n + 4], 0.0);
        // Cross-image (0 ↔ 3): blocked.
        assert_eq!(mask[3], f32::NEG_INFINITY);
        assert_eq!(mask[4 * n + 1], f32::NEG_INFINITY);
    }

    #[test]
    fn mask_diagonal_always_allowed() {
        let mask = packed_attention_mask(&[1, 1, 1]).expect("mask");
        let n = 3;
        for i in 0..n {
            assert_eq!(mask[i * n + i], 0.0, "self-attention must be allowed");
        }
    }

    #[test]
    fn non_multiple_side_errors() {
        let cfg = NaViTConfig::tiny(); // patch 4
        let w = NaViTWeights::zeros(&cfg);
        let shape = ImageShape {
            height: 5, // not a multiple of 4
            width: 8,
            channels: 3,
        };
        let img = vec![0.0_f32; 3 * 5 * 8];
        assert!(matches!(
            NaViT::patchify_pack(&[&img], &[shape], &cfg, &w),
            Err(MultiModalError::InvalidPatchCount { .. })
        ));
    }

    #[test]
    fn wrong_buffer_length_errors() {
        let cfg = NaViTConfig::tiny();
        let w = NaViTWeights::zeros(&cfg);
        let shape = ImageShape {
            height: 8,
            width: 8,
            channels: 3,
        };
        let img = vec![0.0_f32; 10]; // wrong length
        assert!(matches!(
            NaViT::patchify_pack(&[&img], &[shape], &cfg, &w),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_batch_errors() {
        let cfg = NaViTConfig::tiny();
        let w = NaViTWeights::zeros(&cfg);
        assert!(matches!(
            NaViT::patchify_pack(&[], &[], &cfg, &w),
            Err(MultiModalError::EmptyInput)
        ));
    }

    #[test]
    fn grid_exceeds_table_errors() {
        let cfg = NaViTConfig {
            patch_size: 1,
            channels: 1,
            d_model: 4,
            max_rows: 2,
            max_cols: 2,
        };
        let w = NaViTWeights::zeros(&cfg);
        let shape = ImageShape {
            height: 3, // 3 rows > max_rows 2
            width: 1,
            channels: 1,
        };
        let img = vec![0.0_f32; 3];
        assert!(matches!(
            NaViT::patchify_pack(&[&img], &[shape], &cfg, &w),
            Err(MultiModalError::InvalidPatchCount { .. })
        ));
    }

    #[test]
    fn deterministic_packing() {
        let cfg = NaViTConfig::tiny();
        let mut w = NaViTWeights::zeros(&cfg);
        let mut rng = LcgRng::new(5);
        rng.fill_normal(&mut w.patch_proj);
        let shape = ImageShape {
            height: 8,
            width: 8,
            channels: 3,
        };
        let img = make_image(shape, 9);
        let a = NaViT::patchify_pack(&[&img], &[shape], &cfg, &w).expect("a");
        let b = NaViT::patchify_pack(&[&img], &[shape], &cfg, &w).expect("b");
        assert_eq!(a, b);
    }
}

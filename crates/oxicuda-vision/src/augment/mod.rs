//! Image augmentation pipeline for CHW tensors.
//!
//! Provides an enum-dispatched set of operations (`AugOp`) and a composable
//! `Pipeline` that applies them in sequence.  All operations work on flat
//! `[channels × h × w]` row-major `f32` buffers; dimensions are tracked
//! as `(channels, h, w)` tuples so that spatial-modifying operations (crop,
//! resize) can propagate updated dimensions to later stages.

pub mod geometric;
pub mod normalize;
pub mod photometric;

use crate::{error::VisionResult, handle::LcgRng};

use geometric::{center_crop, random_crop, random_horizontal_flip, resize_bilinear};
use normalize::normalize_chw;
use photometric::{color_jitter, random_grayscale};

// ─── AugOp ───────────────────────────────────────────────────────────────────

/// Enum-dispatched augmentation operations — no `dyn Trait`, no heap boxing.
///
/// Each variant carries the hyperparameters it needs.  Stochastic operations
/// receive a mutable `LcgRng` reference at call time via [`AugOp::apply`].
#[derive(Debug, Clone)]
pub enum AugOp {
    /// Randomly crop to `[channels, crop_size, crop_size]`.
    RandomCrop { crop_size: usize },

    /// Deterministic centre crop to `[channels, crop_size, crop_size]`.
    CenterCrop { crop_size: usize },

    /// Randomly flip the image horizontally with the given probability.
    HorizontalFlip { prob: f32 },

    /// Bilinear resize to `[channels, target, target]`.
    Resize { target: usize },

    /// Colour jitter: brightness, contrast, saturation perturbation magnitudes.
    ColorJitter {
        brightness: f32,
        contrast: f32,
        saturation: f32,
    },

    /// Convert to grayscale with the given probability (RGB images only).
    RandomGrayscale { prob: f32 },

    /// Channel-wise normalisation: `(x - mean[c]) / std[c]`.
    Normalize { mean: [f32; 3], std: [f32; 3] },
}

impl AugOp {
    /// Apply this augmentation to a CHW image.
    ///
    /// # Parameters
    /// - `img`: flat `[channels × h × w]` input buffer.
    /// - `channels`: number of channels (e.g., 3 for RGB).
    /// - `h`, `w`: spatial height and width of the input image.
    /// - `rng`: mutable RNG for stochastic operations; deterministic ops ignore it.
    ///
    /// # Returns
    /// `(new_img, new_h, new_w)` — the transformed image and its (possibly
    /// updated) spatial dimensions.  `channels` is never changed.
    ///
    /// # Errors
    /// Propagates errors from the underlying operation functions (invalid
    /// dimensions, mismatched buffers, non-positive std, etc.).
    pub fn apply(
        &self,
        img: &[f32],
        channels: usize,
        h: usize,
        w: usize,
        rng: &mut LcgRng,
    ) -> VisionResult<(Vec<f32>, usize, usize)> {
        match self {
            AugOp::RandomCrop { crop_size } => {
                let out = random_crop(img, channels, h, w, *crop_size, rng)?;
                Ok((out, *crop_size, *crop_size))
            }
            AugOp::CenterCrop { crop_size } => {
                let out = center_crop(img, channels, h, w, *crop_size)?;
                Ok((out, *crop_size, *crop_size))
            }
            AugOp::HorizontalFlip { prob } => {
                let out = random_horizontal_flip(img, channels, h, w, *prob, rng);
                Ok((out, h, w))
            }
            AugOp::Resize { target } => {
                let out = resize_bilinear(img, channels, h, w, *target)?;
                Ok((out, *target, *target))
            }
            AugOp::ColorJitter {
                brightness,
                contrast,
                saturation,
            } => {
                let out = color_jitter(
                    img,
                    channels,
                    h,
                    w,
                    *brightness,
                    *contrast,
                    *saturation,
                    rng,
                );
                Ok((out, h, w))
            }
            AugOp::RandomGrayscale { prob } => {
                let out = random_grayscale(img, channels, h, w, *prob, rng);
                Ok((out, h, w))
            }
            AugOp::Normalize { mean, std } => {
                let out = normalize_chw(img, channels, h, w, mean, std)?;
                Ok((out, h, w))
            }
        }
    }
}

// ─── Pipeline ────────────────────────────────────────────────────────────────

/// A sequence of augmentation operations applied in order.
///
/// `Pipeline` owns a `Vec<AugOp>` and threads `(img, channels, h, w)` through
/// each operation, updating the spatial dimensions as needed (e.g., after a
/// crop or resize).
///
/// # Example
/// ```rust,ignore
/// let pipeline = Pipeline::new()
///     .push(AugOp::Resize { target: 256 })
///     .push(AugOp::RandomCrop { crop_size: 224 })
///     .push(AugOp::HorizontalFlip { prob: 0.5 })
///     .push(AugOp::Normalize { mean: IMAGENET_MEAN, std: IMAGENET_STD });
/// ```
// Note: method is named `push` (not `add`) to avoid confusion with std::ops::Add::add.
#[derive(Debug, Clone, Default)]
pub struct Pipeline {
    /// Ordered list of augmentation operations.
    pub ops: Vec<AugOp>,
}

impl Pipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append an operation to the pipeline (builder pattern).
    #[must_use]
    pub fn push(mut self, op: AugOp) -> Self {
        self.ops.push(op);
        self
    }

    /// Apply all operations in sequence, threading the output through.
    ///
    /// Returns the final `(image, h, w)` after all augmentations, or the
    /// first error encountered.  If the pipeline is empty the image and
    /// dimensions are returned unchanged (cloning the input slice).
    pub fn apply(
        &self,
        img: &[f32],
        channels: usize,
        h: usize,
        w: usize,
        rng: &mut LcgRng,
    ) -> VisionResult<(Vec<f32>, usize, usize)> {
        if self.ops.is_empty() {
            return Ok((img.to_vec(), h, w));
        }

        // Apply first op to the original input.
        let (mut cur_img, mut cur_h, mut cur_w) = self.ops[0].apply(img, channels, h, w, rng)?;

        // Apply subsequent ops to the evolving (cur_img, cur_h, cur_w).
        for op in &self.ops[1..] {
            let (next_img, next_h, next_w) = op.apply(&cur_img, channels, cur_h, cur_w, rng)?;
            cur_img = next_img;
            cur_h = next_h;
            cur_w = next_w;
        }

        Ok((cur_img, cur_h, cur_w))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use normalize::{IMAGENET_MEAN, IMAGENET_STD};

    fn ramp_rgb(h: usize, w: usize) -> Vec<f32> {
        let hw = h * w;
        (0..3 * hw).map(|i| i as f32 / (3 * hw) as f32).collect()
    }

    // ── AugOp::RandomCrop ────────────────────────────────────────────────────

    #[test]
    fn aug_op_random_crop_updates_dims() {
        let img = ramp_rgb(32, 32);
        let mut rng = LcgRng::new(1);
        let op = AugOp::RandomCrop { crop_size: 24 };
        let (out, new_h, new_w) = op.apply(&img, 3, 32, 32, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (24, 24));
        assert_eq!(out.len(), 3 * 24 * 24);
    }

    // ── AugOp::CenterCrop ────────────────────────────────────────────────────

    #[test]
    fn aug_op_center_crop_updates_dims() {
        let img = ramp_rgb(32, 32);
        let mut rng = LcgRng::new(2);
        let op = AugOp::CenterCrop { crop_size: 16 };
        let (out, new_h, new_w) = op.apply(&img, 3, 32, 32, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (16, 16));
        assert_eq!(out.len(), 3 * 16 * 16);
    }

    // ── AugOp::HorizontalFlip ────────────────────────────────────────────────

    #[test]
    fn aug_op_flip_preserves_dims() {
        let img = ramp_rgb(16, 16);
        let mut rng = LcgRng::new(3);
        let op = AugOp::HorizontalFlip { prob: 0.5 };
        let (out, new_h, new_w) = op.apply(&img, 3, 16, 16, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (16, 16));
        assert_eq!(out.len(), img.len());
    }

    // ── AugOp::Resize ────────────────────────────────────────────────────────

    #[test]
    fn aug_op_resize_updates_dims() {
        let img = ramp_rgb(64, 64);
        let mut rng = LcgRng::new(4);
        let op = AugOp::Resize { target: 32 };
        let (out, new_h, new_w) = op.apply(&img, 3, 64, 64, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (32, 32));
        assert_eq!(out.len(), 3 * 32 * 32);
    }

    // ── AugOp::ColorJitter ───────────────────────────────────────────────────

    #[test]
    fn aug_op_color_jitter_preserves_dims() {
        let img = ramp_rgb(16, 16);
        let mut rng = LcgRng::new(5);
        let op = AugOp::ColorJitter {
            brightness: 0.2,
            contrast: 0.2,
            saturation: 0.2,
        };
        let (out, new_h, new_w) = op.apply(&img, 3, 16, 16, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (16, 16));
        assert_eq!(out.len(), img.len());
    }

    // ── AugOp::RandomGrayscale ───────────────────────────────────────────────

    #[test]
    fn aug_op_grayscale_preserves_dims() {
        let img = ramp_rgb(16, 16);
        let mut rng = LcgRng::new(6);
        let op = AugOp::RandomGrayscale { prob: 0.5 };
        let (out, new_h, new_w) = op.apply(&img, 3, 16, 16, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (16, 16));
        assert_eq!(out.len(), img.len());
    }

    // ── AugOp::Normalize ─────────────────────────────────────────────────────

    #[test]
    fn aug_op_normalize_preserves_dims() {
        let img = ramp_rgb(16, 16);
        let mut rng = LcgRng::new(7);
        let op = AugOp::Normalize {
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
        };
        let (out, new_h, new_w) = op.apply(&img, 3, 16, 16, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (16, 16));
        assert_eq!(out.len(), img.len());
    }

    // ── Pipeline ─────────────────────────────────────────────────────────────

    #[test]
    fn pipeline_empty_returns_clone() {
        let img = ramp_rgb(16, 16);
        let pipeline = Pipeline::new();
        let mut rng = LcgRng::new(8);
        let (out, new_h, new_w) = pipeline.apply(&img, 3, 16, 16, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (16, 16));
        assert_eq!(out, img);
    }

    #[test]
    fn pipeline_single_op() {
        let img = ramp_rgb(32, 32);
        let pipeline = Pipeline::new().push(AugOp::Resize { target: 16 });
        let mut rng = LcgRng::new(9);
        let (out, new_h, new_w) = pipeline.apply(&img, 3, 32, 32, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (16, 16));
        assert_eq!(out.len(), 3 * 16 * 16);
    }

    #[test]
    fn pipeline_multi_op_dims_chain() {
        // Resize 64→32, then CenterCrop 32→24.
        let img = ramp_rgb(64, 64);
        let pipeline = Pipeline::new()
            .push(AugOp::Resize { target: 32 })
            .push(AugOp::CenterCrop { crop_size: 24 });
        let mut rng = LcgRng::new(10);
        let (out, new_h, new_w) = pipeline.apply(&img, 3, 64, 64, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (24, 24));
        assert_eq!(out.len(), 3 * 24 * 24);
    }

    #[test]
    fn pipeline_full_augmentation_chain() {
        // Typical training augmentation: resize → random_crop → flip → jitter → normalize.
        let img: Vec<f32> = (0..3 * 256 * 256)
            .map(|i| i as f32 / (3.0 * 256.0 * 256.0))
            .collect();
        let pipeline = Pipeline::new()
            .push(AugOp::Resize { target: 256 })
            .push(AugOp::RandomCrop { crop_size: 224 })
            .push(AugOp::HorizontalFlip { prob: 0.5 })
            .push(AugOp::ColorJitter {
                brightness: 0.1,
                contrast: 0.1,
                saturation: 0.1,
            })
            .push(AugOp::Normalize {
                mean: IMAGENET_MEAN,
                std: IMAGENET_STD,
            });
        let mut rng = LcgRng::new(11);
        let (out, new_h, new_w) = pipeline.apply(&img, 3, 256, 256, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (224, 224));
        assert_eq!(out.len(), 3 * 224 * 224);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "pipeline output must be finite"
        );
    }

    #[test]
    fn pipeline_add_is_builder() {
        // The builder pattern should accumulate ops correctly.
        let p = Pipeline::new()
            .push(AugOp::HorizontalFlip { prob: 1.0 })
            .push(AugOp::HorizontalFlip { prob: 1.0 });
        assert_eq!(p.ops.len(), 2);

        // Two horizontal flips at prob=1 should recover original.
        let img = ramp_rgb(8, 8);
        let mut rng = LcgRng::new(12);
        let (out, _, _) = p.apply(&img, 3, 8, 8, &mut rng).expect("ok");
        for (i, (&a, &b)) in img.iter().zip(out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "pixel {i}: double-flip should be identity"
            );
        }
    }

    #[test]
    fn pipeline_clone_is_independent() {
        let p1 = Pipeline::new().push(AugOp::Resize { target: 16 });
        let p2 = p1.clone();
        assert_eq!(p1.ops.len(), p2.ops.len());
    }

    #[test]
    fn aug_op_error_propagated_through_pipeline() {
        // A crop larger than the image should produce an error.
        let img = ramp_rgb(16, 16);
        let pipeline = Pipeline::new().push(AugOp::CenterCrop { crop_size: 32 }); // 32 > 16
        let mut rng = LcgRng::new(13);
        let r = pipeline.apply(&img, 3, 16, 16, &mut rng);
        assert!(
            r.is_err(),
            "oversized crop through pipeline should propagate error"
        );
    }
}

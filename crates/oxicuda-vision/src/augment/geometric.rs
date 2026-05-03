//! Geometric image augmentations for CHW tensors.
//!
//! All operations work on flat `[channels × h × w]` row-major buffers.
//! Spatial dimensions are described by `(channels, h, w)` parameters rather
//! than embedded in a separate tensor type, keeping the API allocation-minimal.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Validate that a CHW image buffer has the expected size and all dimensions
/// are non-zero.
#[inline]
fn validate_chw(img: &[f32], channels: usize, h: usize, w: usize) -> VisionResult<()> {
    if channels == 0 || h == 0 || w == 0 {
        return Err(VisionError::InvalidImageSize {
            height: h,
            width: w,
            channels,
        });
    }
    let expected = channels * h * w;
    if img.len() != expected {
        return Err(VisionError::DimensionMismatch {
            expected,
            got: img.len(),
        });
    }
    Ok(())
}

// ─── random_crop ─────────────────────────────────────────────────────────────

/// Random crop of a CHW image to `[channels, crop_size, crop_size]`.
///
/// A top-left corner `(top, left)` is drawn uniformly at random from the
/// feasible region `[0, h - crop_size] × [0, w - crop_size]`.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] if any dimension is zero.
/// Returns [`VisionError::DimensionMismatch`] if the image buffer length is wrong.
/// Returns [`VisionError::InvalidPatchSize`] if `crop_size > h` or `crop_size > w`.
pub fn random_crop(
    img: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    crop_size: usize,
    rng: &mut LcgRng,
) -> VisionResult<Vec<f32>> {
    validate_chw(img, channels, h, w)?;
    if crop_size == 0 || crop_size > h || crop_size > w {
        return Err(VisionError::InvalidPatchSize {
            patch_size: crop_size,
            img_size: h.min(w),
        });
    }

    // Feasible region for the top-left corner.
    let max_top = h - crop_size;
    let max_left = w - crop_size;

    // Draw uniform random offsets; if the range is 0 the only choice is 0.
    let top = if max_top == 0 {
        0
    } else {
        rng.next_usize(max_top + 1)
    };
    let left = if max_left == 0 {
        0
    } else {
        rng.next_usize(max_left + 1)
    };

    let mut out = vec![0.0f32; channels * crop_size * crop_size];

    for c in 0..channels {
        for oy in 0..crop_size {
            for ox in 0..crop_size {
                let sy = top + oy;
                let sx = left + ox;
                let src_idx = c * h * w + sy * w + sx;
                let dst_idx = c * crop_size * crop_size + oy * crop_size + ox;
                out[dst_idx] = img[src_idx];
            }
        }
    }

    Ok(out)
}

// ─── center_crop ─────────────────────────────────────────────────────────────

/// Center crop of a CHW image to `[channels, crop_size, crop_size]`.
///
/// The crop is anchored at `((h - crop_size) / 2, (w - crop_size) / 2)`.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] if any dimension is zero.
/// Returns [`VisionError::DimensionMismatch`] if the image buffer length is wrong.
/// Returns [`VisionError::InvalidPatchSize`] if `crop_size > h` or `crop_size > w`.
pub fn center_crop(
    img: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    crop_size: usize,
) -> VisionResult<Vec<f32>> {
    validate_chw(img, channels, h, w)?;
    if crop_size == 0 || crop_size > h || crop_size > w {
        return Err(VisionError::InvalidPatchSize {
            patch_size: crop_size,
            img_size: h.min(w),
        });
    }

    let top = (h - crop_size) / 2;
    let left = (w - crop_size) / 2;

    let mut out = vec![0.0f32; channels * crop_size * crop_size];

    for c in 0..channels {
        for oy in 0..crop_size {
            for ox in 0..crop_size {
                let sy = top + oy;
                let sx = left + ox;
                let src_idx = c * h * w + sy * w + sx;
                let dst_idx = c * crop_size * crop_size + oy * crop_size + ox;
                out[dst_idx] = img[src_idx];
            }
        }
    }

    Ok(out)
}

// ─── random_horizontal_flip ──────────────────────────────────────────────────

/// Random horizontal flip with probability `prob` ∈ [0, 1].
///
/// If `prob <= 0` the image is never flipped; if `prob >= 1` it is always
/// flipped.  For intermediate values a single Bernoulli trial decides.
///
/// Does **not** validate the image buffer (accepts any slice length); callers
/// are responsible for consistent `(channels, h, w)`.
pub fn random_horizontal_flip(
    img: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    prob: f32,
    rng: &mut LcgRng,
) -> Vec<f32> {
    // Determine whether to flip.
    let flip = if prob <= 0.0 {
        false
    } else if prob >= 1.0 {
        true
    } else {
        rng.next_f32() < prob
    };

    if !flip {
        return img.to_vec();
    }

    let mut out = vec![0.0f32; channels * h * w];
    for c in 0..channels {
        for y in 0..h {
            for x in 0..w {
                let src_idx = c * h * w + y * w + (w - 1 - x);
                let dst_idx = c * h * w + y * w + x;
                out[dst_idx] = img[src_idx];
            }
        }
    }
    out
}

// ─── resize_bilinear ─────────────────────────────────────────────────────────

/// Bilinear resize to `[channels, target, target]`.
///
/// Uses the half-pixel convention:
/// ```text
/// src_y = (oy + 0.5) * h / target - 0.5
/// src_x = (ox + 0.5) * w / target - 0.5
/// ```
/// which aligns sample centres rather than corners and avoids boundary
/// over-sampling artefacts.  Source coordinates are clamped to `[0, h-1]`
/// and `[0, w-1]` before bilinear interpolation.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] if any dimension is zero.
/// Returns [`VisionError::DimensionMismatch`] if the image buffer length is wrong.
/// Returns [`VisionError::InvalidPatchSize`] if `target == 0`.
pub fn resize_bilinear(
    img: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    target: usize,
) -> VisionResult<Vec<f32>> {
    validate_chw(img, channels, h, w)?;
    if target == 0 {
        return Err(VisionError::InvalidPatchSize {
            patch_size: 0,
            img_size: h.min(w),
        });
    }

    let h_f = h as f32;
    let w_f = w as f32;
    let t_f = target as f32;

    let mut out = vec![0.0f32; channels * target * target];

    for c in 0..channels {
        let c_base_src = c * h * w;
        let c_base_dst = c * target * target;

        for oy in 0..target {
            // Map output pixel centre to source coordinate.
            let sy_f = (oy as f32 + 0.5) * h_f / t_f - 0.5;
            let sy_f = sy_f.clamp(0.0, h_f - 1.0);

            let sy0 = sy_f.floor() as usize;
            let sy1 = (sy0 + 1).min(h - 1);
            let wy1 = sy_f - sy_f.floor(); // weight for sy1
            let wy0 = 1.0 - wy1;

            for ox in 0..target {
                let sx_f = (ox as f32 + 0.5) * w_f / t_f - 0.5;
                let sx_f = sx_f.clamp(0.0, w_f - 1.0);

                let sx0 = sx_f.floor() as usize;
                let sx1 = (sx0 + 1).min(w - 1);
                let wx1 = sx_f - sx_f.floor();
                let wx0 = 1.0 - wx1;

                // Bilinear interpolation from 4 neighbours.
                let v00 = img[c_base_src + sy0 * w + sx0];
                let v01 = img[c_base_src + sy0 * w + sx1];
                let v10 = img[c_base_src + sy1 * w + sx0];
                let v11 = img[c_base_src + sy1 * w + sx1];

                let val = wy0 * wx0 * v00 + wy0 * wx1 * v01 + wy1 * wx0 * v10 + wy1 * wx1 * v11;

                out[c_base_dst + oy * target + ox] = val;
            }
        }
    }

    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Ramp image: pixel value = flat index cast to f32.
    fn ramp_image(channels: usize, h: usize, w: usize) -> Vec<f32> {
        (0..channels * h * w).map(|i| i as f32).collect()
    }

    // ── random_crop ──────────────────────────────────────────────────────────

    #[test]
    fn random_crop_output_size() {
        let mut rng = LcgRng::new(1);
        let img = ramp_image(3, 32, 32);
        let out = random_crop(&img, 3, 32, 32, 24, &mut rng).expect("random_crop ok");
        assert_eq!(out.len(), 3 * 24 * 24, "wrong output length");
    }

    #[test]
    fn random_crop_exact_size_identity() {
        let mut rng = LcgRng::new(2);
        let img = ramp_image(3, 16, 16);
        let out = random_crop(&img, 3, 16, 16, 16, &mut rng).expect("ok");
        assert_eq!(
            out, img,
            "crop == image size should return identical values"
        );
    }

    #[test]
    fn random_crop_deterministic_with_seed() {
        let img = ramp_image(3, 32, 32);
        let mut rng1 = LcgRng::new(77);
        let mut rng2 = LcgRng::new(77);
        let out1 = random_crop(&img, 3, 32, 32, 20, &mut rng1).expect("ok");
        let out2 = random_crop(&img, 3, 32, 32, 20, &mut rng2).expect("ok");
        assert_eq!(out1, out2, "same seed must produce same crop");
    }

    #[test]
    fn random_crop_error_crop_larger_than_image() {
        let mut rng = LcgRng::new(0);
        let img = ramp_image(3, 16, 16);
        let r = random_crop(&img, 3, 16, 16, 32, &mut rng);
        assert!(
            matches!(r, Err(VisionError::InvalidPatchSize { .. })),
            "expected InvalidPatchSize for oversized crop"
        );
    }

    #[test]
    fn random_crop_error_zero_crop_size() {
        let mut rng = LcgRng::new(0);
        let img = ramp_image(1, 8, 8);
        let r = random_crop(&img, 1, 8, 8, 0, &mut rng);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn random_crop_multiple_calls_vary() {
        // With a large feasible region, repeated calls should (very likely) yield different crops.
        let img: Vec<f32> = (0..3 * 64 * 64).map(|i| i as f32).collect();
        let mut rng = LcgRng::new(42);
        let out1 = random_crop(&img, 3, 64, 64, 32, &mut rng).expect("ok");
        let out2 = random_crop(&img, 3, 64, 64, 32, &mut rng).expect("ok");
        // Not strictly guaranteed, but the probability of collision is astronomically low.
        assert_ne!(out1, out2, "two consecutive random crops should differ");
    }

    // ── center_crop ──────────────────────────────────────────────────────────

    #[test]
    fn center_crop_output_size() {
        let img = ramp_image(3, 32, 32);
        let out = center_crop(&img, 3, 32, 32, 24).expect("center_crop ok");
        assert_eq!(out.len(), 3 * 24 * 24);
    }

    #[test]
    fn center_crop_symmetry() {
        // With a 4×4 image and crop_size=2, the center crop should be
        // deterministic and equal to [[5,6],[9,10]] per channel for a ramp image.
        // (top=1, left=1)
        let img = ramp_image(1, 4, 4); // 0..15
        let out = center_crop(&img, 1, 4, 4, 2).expect("ok");
        // top=(4-2)/2=1, left=(4-2)/2=1
        // out[0] = img[1*4+1] = 5
        // out[1] = img[1*4+2] = 6
        // out[2] = img[2*4+1] = 9
        // out[3] = img[2*4+2] = 10
        assert_eq!(out, vec![5.0, 6.0, 9.0, 10.0], "center crop values wrong");
    }

    #[test]
    fn center_crop_exact_size_identity() {
        let img = ramp_image(3, 16, 16);
        let out = center_crop(&img, 3, 16, 16, 16).expect("ok");
        assert_eq!(out, img);
    }

    #[test]
    fn center_crop_error_oversized() {
        let img = ramp_image(2, 8, 8);
        let r = center_crop(&img, 2, 8, 8, 16);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    // ── random_horizontal_flip ───────────────────────────────────────────────

    #[test]
    fn horizontal_flip_preserves_shape() {
        let img = ramp_image(3, 32, 32);
        let mut rng = LcgRng::new(5);
        let out = random_horizontal_flip(&img, 3, 32, 32, 0.5, &mut rng);
        assert_eq!(out.len(), img.len(), "flip should preserve flat length");
    }

    #[test]
    fn horizontal_flip_prob_zero_returns_clone() {
        let img = ramp_image(3, 8, 8);
        let mut rng = LcgRng::new(6);
        let out = random_horizontal_flip(&img, 3, 8, 8, 0.0, &mut rng);
        assert_eq!(out, img, "prob=0 should never flip");
    }

    #[test]
    fn horizontal_flip_prob_one_always_flips() {
        let img = ramp_image(1, 1, 4); // single row: [0, 1, 2, 3]
        let mut rng = LcgRng::new(7);
        let out = random_horizontal_flip(&img, 1, 1, 4, 1.0, &mut rng);
        // Mirrored: [3, 2, 1, 0]
        assert_eq!(out, vec![3.0, 2.0, 1.0, 0.0], "prob=1 should always flip");
    }

    #[test]
    fn horizontal_flip_double_flip_identity() {
        let img = ramp_image(3, 16, 16);
        let mut rng = LcgRng::new(9);
        let flipped = random_horizontal_flip(&img, 3, 16, 16, 1.0, &mut rng);
        let double = random_horizontal_flip(&flipped, 3, 16, 16, 1.0, &mut rng);
        assert_eq!(double, img, "two flips should recover original");
    }

    #[test]
    fn horizontal_flip_reverses_columns_correctly() {
        // Manually verify a 2-channel 2×3 image.
        // Channel 0 row 0: [10, 11, 12]  → after flip: [12, 11, 10]
        // Channel 0 row 1: [13, 14, 15]  → after flip: [15, 14, 13]
        let img = vec![
            10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, // channel 0
            20.0, 21.0, 22.0, 23.0, 24.0, 25.0, // channel 1
        ];
        let mut rng = LcgRng::new(0);
        let out = random_horizontal_flip(&img, 2, 2, 3, 1.0, &mut rng);
        let expected = vec![
            12.0f32, 11.0, 10.0, 15.0, 14.0, 13.0, // channel 0 reversed cols
            22.0, 21.0, 20.0, 25.0, 24.0, 23.0, // channel 1 reversed cols
        ];
        assert_eq!(out, expected);
    }

    // ── resize_bilinear ──────────────────────────────────────────────────────

    #[test]
    fn resize_bilinear_output_size() {
        let img = ramp_image(3, 32, 32);
        let out = resize_bilinear(&img, 3, 32, 32, 16).expect("resize ok");
        assert_eq!(out.len(), 3 * 16 * 16, "wrong output length after resize");
    }

    #[test]
    fn resize_bilinear_upscale_output_size() {
        let img = ramp_image(3, 8, 8);
        let out = resize_bilinear(&img, 3, 8, 8, 32).expect("resize upscale ok");
        assert_eq!(out.len(), 3 * 32 * 32);
    }

    #[test]
    fn resize_bilinear_same_size_approx_identity() {
        let img = ramp_image(1, 8, 8);
        let out = resize_bilinear(&img, 1, 8, 8, 8).expect("ok");
        // With half-pixel convention target==source, values should be preserved.
        for (i, (&a, &b)) in img.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "pixel {i}: source={a}, resized={b}");
        }
    }

    #[test]
    fn resize_bilinear_constant_image_stays_constant() {
        // A constant image should remain constant after resize.
        let img = vec![7.0f32; 3 * 16 * 16];
        let out = resize_bilinear(&img, 3, 16, 16, 32).expect("ok");
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 7.0).abs() < 1e-5, "pixel {i} should be 7.0 but is {v}");
        }
    }

    #[test]
    fn resize_bilinear_output_finite() {
        let mut rng = LcgRng::new(11);
        let mut img = vec![0.0f32; 3 * 16 * 16];
        rng.fill_normal(&mut img);
        let out = resize_bilinear(&img, 3, 16, 16, 32).expect("ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite output after resize"
        );
    }

    #[test]
    fn resize_bilinear_error_zero_target() {
        let img = ramp_image(3, 16, 16);
        let r = resize_bilinear(&img, 3, 16, 16, 0);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn resize_bilinear_error_zero_dimension() {
        let img: Vec<f32> = vec![];
        let r = resize_bilinear(&img, 3, 0, 16, 8);
        assert!(matches!(r, Err(VisionError::InvalidImageSize { .. })));
    }

    #[test]
    fn resize_bilinear_error_wrong_buffer_length() {
        let img = vec![0.0f32; 100]; // wrong size for 3×16×16
        let r = resize_bilinear(&img, 3, 16, 16, 8);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }
}

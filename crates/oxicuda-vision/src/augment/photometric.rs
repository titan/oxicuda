//! Photometric image augmentations for CHW tensors.
//!
//! Provides colour jitter (brightness, contrast, saturation) and random
//! grayscale conversion.  All operations work on flat `[channels × h × w]`
//! row-major buffers and are applied in-place (returning a new `Vec<f32>`).

use crate::handle::LcgRng;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Sample a perturbation factor from `Uniform(1 - mag, 1 + mag)` clamped so
/// the factor stays non-negative.  `mag` is expected to be in `[0, 1)`.
#[inline]
fn sample_factor(mag: f32, rng: &mut LcgRng) -> f32 {
    let lo = (1.0 - mag).max(0.0);
    let hi = 1.0 + mag;
    lo + rng.next_f32() * (hi - lo)
}

// ─── color_jitter ────────────────────────────────────────────────────────────

/// Apply brightness, contrast, and saturation jitter to a CHW image.
///
/// For each augmentation type, a scalar factor `f` is sampled uniformly from
/// `[1 - magnitude, 1 + magnitude]` (clamped to non-negative).  The
/// transformations are applied sequentially:
///
/// 1. **Brightness**: `x' = f_b * x`
/// 2. **Contrast**: `x' = mean(x) + f_c * (x - mean(x))`, where `mean` is
///    computed over all pixels and channels combined.
/// 3. **Saturation** (only when `channels == 3`): blend toward per-pixel
///    YIQ luminance: `x' = (1 - f_s) * gray + f_s * x'`
///
/// If `channels != 3` the saturation step is silently skipped because
/// grayscale conversion is undefined for non-RGB images.
///
/// Magnitudes equal to `0.0` leave the corresponding property unchanged (the
/// sampled factor is exactly `1.0` when `lo == hi`).
///
/// # Parameters
/// - `img`: flat `[channels × h × w]` input.
/// - `channels`, `h`, `w`: spatial dimensions.
/// - `brightness`, `contrast`, `saturation`: perturbation magnitudes ∈ `[0, 1)`.
/// - `rng`: source of randomness.
pub fn color_jitter(
    img: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    rng: &mut LcgRng,
) -> Vec<f32> {
    let n_pixels = channels * h * w;
    let mut out: Vec<f32> = img.to_vec();

    // ── 1. Brightness ────────────────────────────────────────────────────────
    {
        let fb = sample_factor(brightness, rng);
        for v in &mut out {
            *v *= fb;
        }
    }

    // ── 2. Contrast ──────────────────────────────────────────────────────────
    {
        let fc = sample_factor(contrast, rng);
        // Mean over all pixels and channels.
        let mean: f32 = if n_pixels == 0 {
            0.0
        } else {
            out.iter().sum::<f32>() / n_pixels as f32
        };
        for v in &mut out {
            *v = mean + fc * (*v - mean);
        }
    }

    // ── 3. Saturation (RGB only) ─────────────────────────────────────────────
    if channels == 3 {
        let fs = sample_factor(saturation, rng);
        let hw = h * w;
        // Compute per-pixel YIQ luminance: Y = 0.299 R + 0.587 G + 0.114 B
        // and blend: x' = (1 - fs) * Y + fs * x
        for i in 0..hw {
            let r = out[i];
            let g = out[hw + i];
            let b = out[2 * hw + i];
            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            let one_minus_fs = 1.0 - fs;
            out[i] = one_minus_fs * gray + fs * r;
            out[hw + i] = one_minus_fs * gray + fs * g;
            out[2 * hw + i] = one_minus_fs * gray + fs * b;
        }
    }

    out
}

// ─── random_grayscale ────────────────────────────────────────────────────────

/// Convert a CHW image to grayscale with probability `prob`.
///
/// Uses YIQ luminance weights: `Y = 0.299 R + 0.587 G + 0.114 B`.
/// The result is a 3-channel image where all three channels contain the same
/// luminance value.
///
/// If `channels != 3` the image is returned unchanged (a clone is still
/// produced for API consistency).  If `prob <= 0` the image is never
/// converted; if `prob >= 1` it is always converted.
pub fn random_grayscale(
    img: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    prob: f32,
    rng: &mut LcgRng,
) -> Vec<f32> {
    // Only applicable to 3-channel RGB images.
    if channels != 3 {
        return img.to_vec();
    }

    // Bernoulli trial.
    let do_gray = if prob <= 0.0 {
        false
    } else if prob >= 1.0 {
        true
    } else {
        rng.next_f32() < prob
    };

    if !do_gray {
        return img.to_vec();
    }

    let hw = h * w;
    let mut out = vec![0.0f32; 3 * hw];

    for i in 0..hw {
        let r = img[i];
        let g = img[hw + i];
        let b = img[2 * hw + i];
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        out[i] = y;
        out[hw + i] = y;
        out[2 * hw + i] = y;
    }

    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Create a simple constant 3-channel, H×W image.
    fn const_rgb_image(r: f32, g: f32, b: f32, h: usize, w: usize) -> Vec<f32> {
        let hw = h * w;
        let mut img = vec![0.0f32; 3 * hw];
        for i in 0..hw {
            img[i] = r;
            img[hw + i] = g;
            img[2 * hw + i] = b;
        }
        img
    }

    // ── color_jitter ─────────────────────────────────────────────────────────

    #[test]
    fn color_jitter_output_finite() {
        let img = const_rgb_image(0.5, 0.4, 0.3, 8, 8);
        let mut rng = LcgRng::new(42);
        let out = color_jitter(&img, 3, 8, 8, 0.2, 0.2, 0.2, &mut rng);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "color_jitter produced non-finite values"
        );
    }

    #[test]
    fn color_jitter_zero_magnitude_preserves_values() {
        // With mag=0.0, factor is exactly 1.0 for brightness and contrast.
        // Saturation with mag=0.0 blends with factor 1.0 → also identity.
        let img = const_rgb_image(0.5, 0.5, 0.5, 4, 4);
        let mut rng = LcgRng::new(1);
        let out = color_jitter(&img, 3, 4, 4, 0.0, 0.0, 0.0, &mut rng);
        for (i, (&a, &b)) in img.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "pixel {i}: expected {a}, got {b}");
        }
    }

    #[test]
    fn color_jitter_output_shape_preserved() {
        let img: Vec<f32> = (0..3 * 16 * 16).map(|i| i as f32 / 100.0).collect();
        let mut rng = LcgRng::new(99);
        let out = color_jitter(&img, 3, 16, 16, 0.4, 0.4, 0.4, &mut rng);
        assert_eq!(
            out.len(),
            img.len(),
            "color_jitter must preserve buffer length"
        );
    }

    #[test]
    fn color_jitter_single_channel_skips_saturation() {
        // For a 1-channel image, saturation should be skipped (no panic).
        let img = vec![0.5f32; 8 * 8];
        let mut rng = LcgRng::new(7);
        let out = color_jitter(&img, 1, 8, 8, 0.1, 0.1, 0.1, &mut rng);
        assert_eq!(out.len(), img.len());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn color_jitter_brightness_scales_uniformly() {
        // A constant image + only brightness jitter → output should be uniform.
        let img = vec![0.5f32; 3 * 4 * 4];
        let mut rng = LcgRng::new(3);
        // Use only brightness, zero contrast and saturation magnitude.
        let out = color_jitter(&img, 3, 4, 4, 0.5, 0.0, 0.0, &mut rng);
        // All values should be identical (uniform scaling of constant input).
        let first = out[0];
        assert!(
            out.iter().all(|&v| (v - first).abs() < 1e-6),
            "brightness jitter should preserve uniformity of constant image"
        );
    }

    #[test]
    fn color_jitter_contrast_constant_image_unchanged() {
        // For a constant image, contrast jitter is identity (deviation = 0).
        let img = vec![0.8f32; 3 * 4 * 4];
        let mut rng = LcgRng::new(5);
        // Zero brightness to isolate contrast effect.
        let out = color_jitter(&img, 3, 4, 4, 0.0, 0.8, 0.0, &mut rng);
        // Contrast of a constant image: mean + f*(v - mean) = v for any f.
        for (i, (&a, &b)) in img.iter().zip(out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "pixel {i}: constant image should be unchanged by contrast jitter"
            );
        }
    }

    // ── random_grayscale ─────────────────────────────────────────────────────

    #[test]
    fn grayscale_outputs_equal_channels() {
        // With prob=1.0, all three channels should be equal (luminance).
        let img = const_rgb_image(0.8, 0.5, 0.2, 8, 8);
        let mut rng = LcgRng::new(10);
        let out = random_grayscale(&img, 3, 8, 8, 1.0, &mut rng);
        let hw = 8 * 8;
        for i in 0..hw {
            let r_out = out[i];
            let g_out = out[hw + i];
            let b_out = out[2 * hw + i];
            assert!(
                (r_out - g_out).abs() < 1e-6 && (g_out - b_out).abs() < 1e-6,
                "pixel {i}: R={r_out}, G={g_out}, B={b_out} not equal after grayscale"
            );
        }
    }

    #[test]
    fn grayscale_prob_zero_returns_unchanged() {
        let img = const_rgb_image(0.9, 0.3, 0.1, 4, 4);
        let mut rng = LcgRng::new(11);
        let out = random_grayscale(&img, 3, 4, 4, 0.0, &mut rng);
        assert_eq!(out, img, "prob=0 should not modify image");
    }

    #[test]
    fn grayscale_non_rgb_returns_clone() {
        // 1-channel image should pass through unchanged.
        let img = vec![0.5f32; 8 * 8];
        let mut rng = LcgRng::new(12);
        let out = random_grayscale(&img, 1, 8, 8, 1.0, &mut rng);
        assert_eq!(out, img, "non-3-channel image should be returned unchanged");
    }

    #[test]
    fn grayscale_output_shape_preserved() {
        let img = const_rgb_image(0.6, 0.4, 0.2, 16, 16);
        let mut rng = LcgRng::new(13);
        let out = random_grayscale(&img, 3, 16, 16, 0.5, &mut rng);
        assert_eq!(
            out.len(),
            img.len(),
            "grayscale output should preserve buffer length"
        );
    }

    #[test]
    fn grayscale_luminance_correct() {
        // Single pixel: R=1, G=0, B=0 → Y = 0.299.
        let img = vec![1.0f32, 0.0, 0.0]; // 3 × 1 × 1
        let mut rng = LcgRng::new(14);
        let out = random_grayscale(&img, 3, 1, 1, 1.0, &mut rng);
        let expected_y = 0.299_f32;
        assert!(
            (out[0] - expected_y).abs() < 1e-5,
            "R channel: expected {expected_y}, got {}",
            out[0]
        );
        assert!(
            (out[1] - expected_y).abs() < 1e-5,
            "G channel: expected {expected_y}, got {}",
            out[1]
        );
        assert!(
            (out[2] - expected_y).abs() < 1e-5,
            "B channel: expected {expected_y}, got {}",
            out[2]
        );
    }

    #[test]
    fn grayscale_output_finite() {
        let mut rng_gen = LcgRng::new(50);
        let mut img = vec![0.0f32; 3 * 32 * 32];
        rng_gen.fill_normal(&mut img);
        let mut rng = LcgRng::new(51);
        let out = random_grayscale(&img, 3, 32, 32, 1.0, &mut rng);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "grayscale output not finite"
        );
    }
}

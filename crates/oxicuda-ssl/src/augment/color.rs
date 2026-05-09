//! Color augmentation primitives operating on `[C, H, W]` (CHW) RGB images.
//!
//! Standard SSL pretraining recipes (SimCLR, BYOL, MoCo v2/v3, MAE) include
//! per-channel brightness/contrast/saturation/hue jitter and a probability of
//! random grayscale conversion. These pure-CPU helpers reproduce the
//! numerically dominant terms; downstream callers can compose them with their
//! own augmentation pipeline.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

/// Multiplicative brightness/contrast/saturation jitter on a CHW RGB image.
///
/// `image` is `[3 × H × W]` row-major; each channel is multiplied by a sampled
/// factor in `[1 − strength, 1 + strength]`. Output is clamped to `[0, 1]`.
///
/// # Errors
/// - [`SslError::DimensionMismatch`] if length is not `3·H·W`.
/// - [`SslError::EmptyInput`] if `H == 0` or `W == 0`.
/// - [`SslError::InvalidLossWeight`] if `strength` is non-finite or negative.
pub fn color_jitter(
    image: &mut [f32],
    h: usize,
    w: usize,
    strength: f32,
    rng: &mut LcgRng,
) -> SslResult<()> {
    if h == 0 || w == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(strength.is_finite() && strength >= 0.0) {
        return Err(SslError::InvalidLossWeight { weight: strength });
    }
    if image.len() != 3 * h * w {
        return Err(SslError::DimensionMismatch {
            expected: 3 * h * w,
            got: image.len(),
        });
    }
    let plane = h * w;
    for c in 0..3 {
        let factor = 1.0 - strength + 2.0 * strength * rng.next_f32();
        let chan = &mut image[c * plane..(c + 1) * plane];
        for v in chan.iter_mut() {
            *v = (*v * factor).clamp(0.0, 1.0);
        }
    }
    Ok(())
}

/// With probability `p`, replace a CHW RGB image with its luminance-equal
/// grayscale broadcast across all 3 channels.
///
/// Uses the BT.601 luminance weights `(0.299, 0.587, 0.114)`.
///
/// Returns `true` if the image was converted, `false` otherwise.
///
/// # Errors
/// - [`SslError::DimensionMismatch`] if length is not `3·H·W`.
/// - [`SslError::EmptyInput`] if `H == 0` or `W == 0`.
/// - [`SslError::InvalidLossWeight`] if `p ∉ [0, 1]`.
pub fn random_grayscale_chw(
    image: &mut [f32],
    h: usize,
    w: usize,
    p: f32,
    rng: &mut LcgRng,
) -> SslResult<bool> {
    if h == 0 || w == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
        return Err(SslError::InvalidLossWeight { weight: p });
    }
    if image.len() != 3 * h * w {
        return Err(SslError::DimensionMismatch {
            expected: 3 * h * w,
            got: image.len(),
        });
    }
    if rng.next_f32() >= p {
        return Ok(false);
    }
    let plane = h * w;
    for i in 0..plane {
        let r = image[i];
        let g = image[plane + i];
        let b = image[2 * plane + i];
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        image[i] = y;
        image[plane + i] = y;
        image[2 * plane + i] = y;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_image(h: usize, w: usize) -> Vec<f32> {
        let mut img = vec![0.0_f32; 3 * h * w];
        for (i, v) in img.iter_mut().enumerate() {
            *v = (i as f32 % 10.0) / 10.0;
        }
        img
    }

    #[test]
    fn color_jitter_zero_strength_unchanged() {
        let mut rng = LcgRng::new(0);
        let h = 4;
        let w = 4;
        let mut img = sample_image(h, w);
        let original = img.clone();
        color_jitter(&mut img, h, w, 0.0, &mut rng).unwrap();
        assert_eq!(img, original);
    }

    #[test]
    fn color_jitter_clips_to_unit_interval() {
        let mut rng = LcgRng::new(0);
        let h = 4;
        let w = 4;
        let mut img = vec![0.9_f32; 3 * h * w];
        color_jitter(&mut img, h, w, 1.0, &mut rng).unwrap();
        for v in &img {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn color_jitter_rejects_negative_strength() {
        let mut rng = LcgRng::new(0);
        let mut img = sample_image(4, 4);
        let r = color_jitter(&mut img, 4, 4, -0.1, &mut rng);
        assert!(r.is_err());
    }

    #[test]
    fn color_jitter_rejects_zero_dims() {
        let mut rng = LcgRng::new(0);
        let mut img: Vec<f32> = vec![];
        let r = color_jitter(&mut img, 0, 0, 0.5, &mut rng);
        assert!(r.is_err());
    }

    #[test]
    fn random_grayscale_zero_prob_never_converts() {
        let mut rng = LcgRng::new(0);
        let h = 4;
        let w = 4;
        let mut img = sample_image(h, w);
        let original = img.clone();
        let converted = random_grayscale_chw(&mut img, h, w, 0.0, &mut rng).unwrap();
        assert!(!converted);
        assert_eq!(img, original);
    }

    #[test]
    fn random_grayscale_full_prob_always_converts() {
        let mut rng = LcgRng::new(0);
        let h = 2;
        let w = 2;
        let mut img = vec![
            1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5,
        ];
        let converted = random_grayscale_chw(&mut img, h, w, 1.0, &mut rng).unwrap();
        assert!(converted);
        // All channels equal after conversion.
        let plane = h * w;
        for i in 0..plane {
            let r = img[i];
            let g = img[plane + i];
            let b = img[2 * plane + i];
            assert!((r - g).abs() < 1e-6);
            assert!((g - b).abs() < 1e-6);
        }
    }

    #[test]
    fn random_grayscale_rejects_invalid_p() {
        let mut rng = LcgRng::new(0);
        let mut img = sample_image(2, 2);
        assert!(random_grayscale_chw(&mut img, 2, 2, 1.5, &mut rng).is_err());
        assert!(random_grayscale_chw(&mut img, 2, 2, -0.1, &mut rng).is_err());
    }
}

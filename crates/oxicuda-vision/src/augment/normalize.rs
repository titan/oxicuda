//! Channel-wise normalisation for CHW image tensors.
//!
//! Provides the standard `(x - mean) / std` transformation used before
//! feeding images to neural networks, applied independently per channel.

use crate::error::{VisionError, VisionResult};

// ─── Constants ───────────────────────────────────────────────────────────────

/// ImageNet per-channel mean (RGB order, values pre-scaled to [0, 1]).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet per-channel standard deviation (RGB order).
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

// ─── normalize_chw ───────────────────────────────────────────────────────────

/// Normalize a CHW image channel-wise: `output[c, h, w] = (input[c, h, w] - mean[c]) / std[c]`.
///
/// # Parameters
/// - `img`: flat `[channels × h × w]` input buffer.
/// - `channels`: number of channels; must equal `mean.len()` and `std.len()`.
/// - `h`: image height in pixels.
/// - `w`: image width in pixels.
/// - `mean`: per-channel mean values; length must equal `channels`.
/// - `std`: per-channel standard deviation values; length must equal `channels`.
///   Each element must be positive.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] if any dimension is zero.
/// Returns [`VisionError::DimensionMismatch`] if `img.len() != channels * h * w`.
/// Returns [`VisionError::ShapeMismatch`] if `mean.len() != channels` or `std.len() != channels`.
/// Returns [`VisionError::NonFinite`] if any `std[c] <= 0` (would produce NaN/Inf).
pub fn normalize_chw(
    img: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    mean: &[f32],
    std: &[f32],
) -> VisionResult<Vec<f32>> {
    if channels == 0 || h == 0 || w == 0 {
        return Err(VisionError::InvalidImageSize {
            height: h,
            width: w,
            channels,
        });
    }
    let expected_len = channels * h * w;
    if img.len() != expected_len {
        return Err(VisionError::DimensionMismatch {
            expected: expected_len,
            got: img.len(),
        });
    }
    if mean.len() != channels {
        return Err(VisionError::ShapeMismatch {
            lhs: vec![channels],
            rhs: vec![mean.len()],
        });
    }
    if std.len() != channels {
        return Err(VisionError::ShapeMismatch {
            lhs: vec![channels],
            rhs: vec![std.len()],
        });
    }
    // Validate std values before proceeding (avoid silently producing NaN/Inf).
    for (c, &s) in std.iter().enumerate() {
        if s <= 0.0 || !s.is_finite() {
            return Err(VisionError::NonFinite(
                // We use a single static string; the channel index is
                // implicit (detailed validation error).
                if c == 0 {
                    "std[0] non-positive"
                } else if c == 1 {
                    "std[1] non-positive"
                } else if c == 2 {
                    "std[2] non-positive"
                } else {
                    "std[c] non-positive"
                },
            ));
        }
    }

    let hw = h * w;
    let mut out = vec![0.0f32; expected_len];

    for c in 0..channels {
        let m = mean[c];
        let s = std[c];
        let inv_s = 1.0 / s;
        let base = c * hw;
        for i in 0..hw {
            out[base + i] = (img[base + i] - m) * inv_s;
        }
    }

    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple single-channel, 4-pixel image: values [1, 2, 3, 4].
    fn make_single_channel_img() -> (Vec<f32>, usize, usize, usize) {
        let img = vec![1.0f32, 2.0, 3.0, 4.0];
        (img, 1, 2, 2) // (data, channels, h, w)
    }

    #[test]
    fn normalized_mean_approx_zero() {
        // For a channel with values [1,2,3,4], mean=2.5, std=1.118...
        // After normalization, sample mean ≈ 0.
        let (img, channels, h, w) = make_single_channel_img();
        let sample_mean = img.iter().sum::<f32>() / img.len() as f32; // 2.5
        let variance =
            img.iter().map(|&v| (v - sample_mean).powi(2)).sum::<f32>() / img.len() as f32;
        let sample_std = variance.sqrt(); // ~1.118

        let out = normalize_chw(&img, channels, h, w, &[sample_mean], &[sample_std])
            .expect("normalize_chw ok");

        let out_mean = out.iter().sum::<f32>() / out.len() as f32;
        assert!(
            out_mean.abs() < 1e-5,
            "expected near-zero mean after normalization, got {out_mean}"
        );
    }

    #[test]
    fn normalized_std_approx_one() {
        let (img, channels, h, w) = make_single_channel_img();
        let sample_mean = img.iter().sum::<f32>() / img.len() as f32;
        let variance =
            img.iter().map(|&v| (v - sample_mean).powi(2)).sum::<f32>() / img.len() as f32;
        let sample_std = variance.sqrt();

        let out = normalize_chw(&img, channels, h, w, &[sample_mean], &[sample_std])
            .expect("normalize_chw ok");

        let out_mean = out.iter().sum::<f32>() / out.len() as f32;
        let out_var = out.iter().map(|&v| (v - out_mean).powi(2)).sum::<f32>() / out.len() as f32;
        let out_std = out_var.sqrt();
        assert!(
            (out_std - 1.0).abs() < 1e-5,
            "expected std ≈ 1.0 after normalization, got {out_std}"
        );
    }

    #[test]
    fn multi_channel_normalization_per_channel() {
        // 3 channels, 1×1 spatial (trivial sizes to verify math).
        // Channel c contains value (c as f32 + 1) * 10.
        let img = vec![10.0f32, 20.0, 30.0]; // 3 × 1 × 1
        let mean = [5.0f32, 15.0, 25.0];
        let std = [2.5f32, 2.5, 2.5];

        let out = normalize_chw(&img, 3, 1, 1, &mean, &std).expect("ok");
        // channel 0: (10 - 5) / 2.5 = 2.0
        // channel 1: (20 - 15) / 2.5 = 2.0
        // channel 2: (30 - 25) / 2.5 = 2.0
        assert!((out[0] - 2.0).abs() < 1e-6, "c0: {}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-6, "c1: {}", out[1]);
        assert!((out[2] - 2.0).abs() < 1e-6, "c2: {}", out[2]);
    }

    #[test]
    fn error_on_zero_height() {
        let img = vec![1.0f32; 3];
        let r = normalize_chw(&img, 3, 0, 1, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
        assert!(
            matches!(r, Err(VisionError::InvalidImageSize { .. })),
            "expected InvalidImageSize, got {:?}",
            r
        );
    }

    #[test]
    fn error_on_zero_channels() {
        let img: Vec<f32> = vec![];
        let r = normalize_chw(&img, 0, 4, 4, &[], &[]);
        assert!(matches!(r, Err(VisionError::InvalidImageSize { .. })));
    }

    #[test]
    fn error_on_wrong_image_length() {
        let img = vec![1.0f32; 10]; // should be 3*4*4=48
        let r = normalize_chw(&img, 3, 4, 4, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn error_on_mean_length_mismatch() {
        let img = vec![0.0f32; 3 * 2 * 2];
        let r = normalize_chw(&img, 3, 2, 2, &[0.0, 0.0], &[1.0, 1.0, 1.0]);
        assert!(matches!(r, Err(VisionError::ShapeMismatch { .. })));
    }

    #[test]
    fn error_on_std_length_mismatch() {
        let img = vec![0.0f32; 3 * 2 * 2];
        let r = normalize_chw(&img, 3, 2, 2, &[0.0, 0.0, 0.0], &[1.0, 1.0]);
        assert!(matches!(r, Err(VisionError::ShapeMismatch { .. })));
    }

    #[test]
    fn error_on_nonpositive_std() {
        let img = vec![1.0f32; 2 * 2];
        let r = normalize_chw(&img, 1, 2, 2, &[0.5], &[0.0]);
        assert!(matches!(r, Err(VisionError::NonFinite(_))));
    }

    #[test]
    fn error_on_negative_std() {
        let img = vec![1.0f32; 2 * 2];
        let r = normalize_chw(&img, 1, 2, 2, &[0.5], &[-0.5]);
        assert!(matches!(r, Err(VisionError::NonFinite(_))));
    }

    #[test]
    fn imagenet_constants_valid() {
        assert_eq!(IMAGENET_MEAN.len(), 3);
        assert_eq!(IMAGENET_STD.len(), 3);
        // All positive
        assert!(IMAGENET_STD.iter().all(|&v| v > 0.0));
        // Mean in [0, 1]
        assert!(IMAGENET_MEAN.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn imagenet_normalization_output_finite() {
        // Typical ImageNet input after /255 rescaling.
        let img: Vec<f32> = (0..3 * 224 * 224)
            .map(|i| ((i % 256) as f32) / 255.0)
            .collect();
        let out = normalize_chw(&img, 3, 224, 224, &IMAGENET_MEAN, &IMAGENET_STD)
            .expect("imagenet normalize ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite after imagenet normalize"
        );
    }
}

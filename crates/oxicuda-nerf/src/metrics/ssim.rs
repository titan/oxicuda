//! Structural Similarity Index (SSIM) image-quality metric.
//!
//! Wang, Bovik, Sheikh & Simoncelli (2004) "Image Quality Assessment: From Error
//! Visibility to Structural Similarity", IEEE Transactions on Image Processing,
//! vol. 13(4), pp. 600-612.
//!
//! SSIM complements PSNR by measuring *perceived* structural similarity rather than
//! pixelwise error. Over a local window it combines three comparisons — luminance,
//! contrast and structure — into
//!
//! ```text
//! SSIM(x, y) = (2 μ_x μ_y + C₁)(2 σ_xy + C₂)
//!              ─────────────────────────────────
//!              (μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂)
//! ```
//!
//! where `μ`, `σ²`, `σ_xy` are the local mean, variance and covariance, and
//! `C₁ = (k₁ L)²`, `C₂ = (k₂ L)²` stabilise the ratio (`L` = dynamic range, default
//! `1.0` for `[0, 1]` images; `k₁ = 0.01`, `k₂ = 0.03`). The global score is the
//! mean SSIM over all windows (MSSIM). This implementation uses a uniform (box)
//! window — the standard simplified variant used by most NeRF evaluation code — over
//! a single-channel image; multi-channel images are scored per channel and averaged.

use crate::error::{NerfError, NerfResult};

/// SSIM stabilisation / window parameters.
#[derive(Debug, Clone, Copy)]
pub struct SsimConfig {
    /// Side length of the square sliding window (odd, `>= 1`). Default `7`.
    pub window: usize,
    /// Dynamic range `L` of the pixel values (e.g. `1.0` for `[0, 1]`). Default `1.0`.
    pub data_range: f32,
    /// Luminance stabiliser constant `k₁`. Default `0.01`.
    pub k1: f32,
    /// Contrast stabiliser constant `k₂`. Default `0.03`.
    pub k2: f32,
}

impl Default for SsimConfig {
    fn default() -> Self {
        Self {
            window: 7,
            data_range: 1.0,
            k1: 0.01,
            k2: 0.03,
        }
    }
}

/// Mean SSIM (MSSIM) between two single-channel images of size `width × height`
/// (row-major), using a uniform window.
///
/// # Errors
/// - [`NerfError::EmptyInput`] if either image is empty.
/// - [`NerfError::DimensionMismatch`] if the lengths differ or do not equal
///   `width * height`.
/// - [`NerfError::InvalidGridResolution`] if `window == 0`, even, or larger than
///   the image.
/// - [`NerfError::NanEncountered`] on non-finite inputs.
pub fn ssim_gray(
    a: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    cfg: &SsimConfig,
) -> NerfResult<f32> {
    if a.is_empty() || b.is_empty() {
        return Err(NerfError::EmptyInput);
    }
    if a.len() != b.len() || a.len() != width * height {
        return Err(NerfError::DimensionMismatch {
            expected: width * height,
            got: a.len().min(b.len()),
        });
    }
    if cfg.window == 0 || cfg.window.is_multiple_of(2) || cfg.window > width || cfg.window > height
    {
        return Err(NerfError::InvalidGridResolution { res: cfg.window });
    }
    if a.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return Err(NerfError::NanEncountered {
            context: "ssim_gray: non-finite pixel".into(),
        });
    }

    let c1 = (cfg.k1 * cfg.data_range).powi(2);
    let c2 = (cfg.k2 * cfg.data_range).powi(2);
    let win = cfg.window;
    let n_win = (win * win) as f32;

    let mut acc = 0.0_f64;
    let mut count = 0_usize;

    // Slide the window so it stays fully inside the image (valid convolution).
    for top in 0..=(height - win) {
        for left in 0..=(width - win) {
            let mut sum_a = 0.0_f32;
            let mut sum_b = 0.0_f32;
            let mut sum_aa = 0.0_f32;
            let mut sum_bb = 0.0_f32;
            let mut sum_ab = 0.0_f32;
            for dy in 0..win {
                let row = (top + dy) * width + left;
                for dx in 0..win {
                    let va = a[row + dx];
                    let vb = b[row + dx];
                    sum_a += va;
                    sum_b += vb;
                    sum_aa += va * va;
                    sum_bb += vb * vb;
                    sum_ab += va * vb;
                }
            }
            let mu_a = sum_a / n_win;
            let mu_b = sum_b / n_win;
            // Unbiased-style covariance/variance via E[x²] − E[x]²
            // (population estimate, matching the reference implementation).
            let var_a = (sum_aa / n_win - mu_a * mu_a).max(0.0);
            let var_b = (sum_bb / n_win - mu_b * mu_b).max(0.0);
            let cov_ab = sum_ab / n_win - mu_a * mu_b;

            let numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2);
            let denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2);
            acc += (numerator / denominator) as f64;
            count += 1;
        }
    }

    if count == 0 {
        return Err(NerfError::InvalidGridResolution { res: win });
    }
    let mssim = (acc / count as f64) as f32;
    if !mssim.is_finite() {
        return Err(NerfError::NanEncountered {
            context: "ssim_gray: non-finite result".into(),
        });
    }
    Ok(mssim)
}

/// Mean SSIM for an interleaved multi-channel image (`width × height × channels`,
/// channel-last) — computed per channel and averaged.
///
/// # Errors
/// - [`NerfError::InvalidFeatureDim`] if `channels == 0`.
/// - Propagates errors from [`ssim_gray`].
pub fn ssim_image(
    a: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    cfg: &SsimConfig,
) -> NerfResult<f32> {
    if channels == 0 {
        return Err(NerfError::InvalidFeatureDim { dim: channels });
    }
    if a.len() != b.len() || a.len() != width * height * channels {
        return Err(NerfError::DimensionMismatch {
            expected: width * height * channels,
            got: a.len().min(b.len()),
        });
    }
    let plane = width * height;
    let mut total = 0.0_f32;
    let mut chan_a = vec![0.0_f32; plane];
    let mut chan_b = vec![0.0_f32; plane];
    for c in 0..channels {
        for p in 0..plane {
            chan_a[p] = a[p * channels + c];
            chan_b[p] = b[p * channels + c];
        }
        total += ssim_gray(&chan_a, &chan_b, width, height, cfg)?;
    }
    Ok(total / channels as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn ssim_identical_is_one() {
        let img = vec![0.5_f32; 8 * 8];
        let cfg = SsimConfig::default();
        let s = ssim_gray(&img, &img, 8, 8, &cfg).expect("ssim_gray should succeed");
        assert!(approx(s, 1.0, 1e-5), "SSIM(x,x) = {s}");
    }

    #[test]
    fn ssim_identical_textured_is_one() {
        // Non-constant image still gives SSIM = 1 with itself.
        let img: Vec<f32> = (0..16 * 16).map(|i| ((i % 16) as f32) / 16.0).collect();
        let cfg = SsimConfig::default();
        let s = ssim_gray(&img, &img, 16, 16, &cfg).expect("ssim_gray should succeed");
        assert!(approx(s, 1.0, 1e-5), "SSIM(x,x) = {s}");
    }

    #[test]
    fn ssim_in_valid_range() {
        let a: Vec<f32> = (0..12 * 12)
            .map(|i| (i as f32 * 0.013).sin().abs())
            .collect();
        let b: Vec<f32> = (0..12 * 12)
            .map(|i| (i as f32 * 0.021).cos().abs())
            .collect();
        let cfg = SsimConfig::default();
        let s = ssim_gray(&a, &b, 12, 12, &cfg).expect("ssim_gray should succeed");
        assert!((-1.0..=1.0).contains(&s), "SSIM out of range: {s}");
    }

    #[test]
    fn ssim_decreases_with_noise() {
        let gt: Vec<f32> = (0..20 * 20).map(|i| ((i * 7 % 20) as f32) / 20.0).collect();
        let small: Vec<f32> = gt.iter().map(|&v| (v + 0.02).min(1.0)).collect();
        let large: Vec<f32> = gt
            .iter()
            .enumerate()
            .map(|(i, &v)| (v + if i % 2 == 0 { 0.3 } else { -0.3 }).clamp(0.0, 1.0))
            .collect();
        let cfg = SsimConfig::default();
        let s_small = ssim_gray(&gt, &small, 20, 20, &cfg).expect("ssim_gray should succeed");
        let s_large = ssim_gray(&gt, &large, 20, 20, &cfg).expect("ssim_gray should succeed");
        assert!(
            s_small > s_large,
            "more distortion should lower SSIM: {s_small} vs {s_large}"
        );
    }

    #[test]
    fn ssim_symmetric() {
        let a: Vec<f32> = (0..10 * 10)
            .map(|i| (i as f32 * 0.03).sin().abs())
            .collect();
        let b: Vec<f32> = (0..10 * 10)
            .map(|i| (i as f32 * 0.05).cos().abs())
            .collect();
        let cfg = SsimConfig::default();
        let s_ab = ssim_gray(&a, &b, 10, 10, &cfg).expect("ssim_gray should succeed");
        let s_ba = ssim_gray(&b, &a, 10, 10, &cfg).expect("ssim_gray should succeed");
        assert!(approx(s_ab, s_ba, 1e-5), "SSIM not symmetric");
    }

    #[test]
    fn ssim_constant_shift_high() {
        // A small constant offset keeps structure → SSIM stays high.
        let gt: Vec<f32> = (0..16 * 16).map(|i| ((i % 16) as f32) / 16.0).collect();
        let shifted: Vec<f32> = gt.iter().map(|&v| (v + 0.05).min(1.0)).collect();
        let cfg = SsimConfig::default();
        let s = ssim_gray(&gt, &shifted, 16, 16, &cfg).expect("ssim_gray should succeed");
        assert!(
            s > 0.5,
            "structure-preserving shift should keep SSIM high: {s}"
        );
    }

    #[test]
    fn ssim_empty_errors() {
        let cfg = SsimConfig::default();
        assert!(matches!(
            ssim_gray(&[], &[], 0, 0, &cfg),
            Err(NerfError::EmptyInput)
        ));
    }

    #[test]
    fn ssim_dim_mismatch_errors() {
        let cfg = SsimConfig::default();
        let a = vec![0.0_f32; 64];
        let b = vec![0.0_f32; 49];
        assert!(matches!(
            ssim_gray(&a, &b, 8, 8, &cfg),
            Err(NerfError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn ssim_window_too_large_errors() {
        let cfg = SsimConfig {
            window: 9,
            ..Default::default()
        };
        let img = vec![0.5_f32; 4 * 4];
        assert!(matches!(
            ssim_gray(&img, &img, 4, 4, &cfg),
            Err(NerfError::InvalidGridResolution { .. })
        ));
    }

    #[test]
    fn ssim_even_window_errors() {
        let cfg = SsimConfig {
            window: 4,
            ..Default::default()
        };
        let img = vec![0.5_f32; 8 * 8];
        assert!(ssim_gray(&img, &img, 8, 8, &cfg).is_err());
    }

    #[test]
    fn ssim_nan_errors() {
        let cfg = SsimConfig::default();
        let mut a = vec![0.5_f32; 8 * 8];
        a[0] = f32::NAN;
        let b = vec![0.5_f32; 8 * 8];
        assert!(matches!(
            ssim_gray(&a, &b, 8, 8, &cfg),
            Err(NerfError::NanEncountered { .. })
        ));
    }

    #[test]
    fn ssim_image_multichannel_identical() {
        let img = vec![0.4_f32; 8 * 8 * 3];
        let cfg = SsimConfig::default();
        let s = ssim_image(&img, &img, 8, 8, 3, &cfg).expect("ssim_image should succeed");
        assert!(approx(s, 1.0, 1e-5), "multichannel SSIM(x,x) = {s}");
    }

    #[test]
    fn ssim_image_zero_channels_errors() {
        let cfg = SsimConfig::default();
        assert!(matches!(
            ssim_image(&[], &[], 8, 8, 0, &cfg),
            Err(NerfError::InvalidFeatureDim { .. })
        ));
    }
}

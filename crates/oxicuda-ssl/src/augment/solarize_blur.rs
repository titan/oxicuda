//! Gaussian blur, solarization, and Gaussian-noise augmentations for CHW images.
//!
//! # Layout convention
//!
//! All functions operate on a flat `[C × H × W]` row-major buffer where channel `c`,
//! row `y`, and column `x` maps to index `c * H * W + y * W + x`.  Pixel values are
//! `f32` in `[0.0, 1.0]`.
//!
//! # Key references
//! - Chen et al., "A Simple Framework for Contrastive Learning of Visual
//!   Representations", ICML 2020 (SimCLR) — Gaussian blur with σ ∈ [0.1, 2.0].
//! - Grill et al., "Bootstrap Your Own Latent", NeurIPS 2020 (BYOL) — solarization
//!   at threshold 0.5 with probability 0.2 for one view.
//! - Separable 2-D Gaussian convolution reduces complexity from O(H·W·k²) to
//!   O(H·W·k) by applying two independent 1-D passes.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Gaussian kernel construction ─────────────────────────────────────────────

/// Build a normalised 1-D Gaussian kernel of radius `r = ceil(3σ)`.
///
/// The kernel has `2r + 1` coefficients; `k[i] = exp(-(i - r)² / (2σ²))`,
/// normalised so that the sum equals `1.0` exactly (floating-point).
///
/// # Errors
/// Returns [`SslError::InvalidParameter`] when `sigma <= 0` or `!sigma.is_finite()`.
fn build_gaussian_kernel(sigma: f64) -> SslResult<Vec<f32>> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(SslError::InvalidParameter {
            name: "sigma".into(),
            reason: format!("must be positive and finite, got {sigma}"),
        });
    }
    let r = (3.0 * sigma).ceil() as usize;
    let k = 2 * r + 1;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut weights: Vec<f64> = (0..k)
        .map(|i| {
            let d = i as f64 - r as f64;
            (-d * d / two_sigma_sq).exp()
        })
        .collect();
    let sum: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum;
    }
    Ok(weights.iter().map(|&w| w as f32).collect())
}

// ─── Separable 2-D Gaussian blur ─────────────────────────────────────────────

/// Convolve a single-channel `H × W` plane with a 1-D kernel along the
/// horizontal axis (axis = 1, i.e. columns), using clamp-to-edge padding.
///
/// `src` has length `H * W`.  The result is stored into `dst` (same length).
fn convolve_horizontal(src: &[f32], dst: &mut [f32], height: usize, width: usize, kernel: &[f32]) {
    let r = kernel.len() / 2;
    for y in 0..height {
        let row_off = y * width;
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                // Map kernel index ki to source column; clamp to [0, W-1].
                let src_x = (x + ki).saturating_sub(r).min(width - 1);
                acc += src[row_off + src_x] * kw;
            }
            dst[row_off + x] = acc;
        }
    }
}

/// Convolve a single-channel `H × W` plane with a 1-D kernel along the
/// vertical axis (axis = 0, i.e. rows), using clamp-to-edge padding.
///
/// `src` has length `H * W`.  The result is stored into `dst` (same length).
fn convolve_vertical(src: &[f32], dst: &mut [f32], height: usize, width: usize, kernel: &[f32]) {
    let r = kernel.len() / 2;
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                // Map kernel index ki to source row; clamp to [0, H-1].
                let src_y = (y + ki).saturating_sub(r).min(height - 1);
                acc += src[src_y * width + x] * kw;
            }
            dst[y * width + x] = acc;
        }
    }
}

/// Gaussian blur on a CHW image (C channels, H rows, W cols).
///
/// Applies a separable 1-D Gaussian kernel first horizontally then vertically
/// for each channel independently.  Edge handling uses clamp-to-edge (the
/// border pixel is repeated instead of zero-padding).
///
/// # Parameters
/// - `pixels` — flat `[C × H × W]` row-major input, `f32` in `[0, 1]`.
/// - `channels` — number of channels `C`.
/// - `height` — number of rows `H`.
/// - `width`   — number of columns `W`.
/// - `sigma`   — Gaussian standard deviation; kernel radius = `ceil(3σ)`.
///
/// # Errors
/// - [`SslError::EmptyInput`] if `channels == 0 || height == 0 || width == 0`.
/// - [`SslError::DimensionMismatch`] if `pixels.len() != C·H·W`.
/// - [`SslError::InvalidParameter`] if `sigma <= 0` or not finite.
pub fn gaussian_blur_chw(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    sigma: f64,
) -> SslResult<Vec<f32>> {
    if channels == 0 || height == 0 || width == 0 {
        return Err(SslError::EmptyInput);
    }
    let expected = channels * height * width;
    if pixels.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: pixels.len(),
        });
    }
    let kernel = build_gaussian_kernel(sigma)?;
    let plane = height * width;

    // Intermediate and output buffers; we reuse for every channel.
    let mut after_h = vec![0.0_f32; plane];
    let mut output = vec![0.0_f32; expected];

    for c in 0..channels {
        let src_plane = &pixels[c * plane..(c + 1) * plane];
        let dst_plane = &mut output[c * plane..(c + 1) * plane];

        // Pass 1: horizontal convolution → after_h
        convolve_horizontal(src_plane, &mut after_h, height, width, &kernel);
        // Pass 2: vertical convolution → dst_plane
        convolve_vertical(&after_h, dst_plane, height, width, &kernel);
    }
    Ok(output)
}

/// Randomly select `σ ∈ [sigma_min, sigma_max]` and apply Gaussian blur.
///
/// The sigma is sampled uniformly: `σ = sigma_min + U * (sigma_max - sigma_min)`
/// where `U ~ Uniform[0, 1)`.
///
/// # Errors
/// Propagates all errors from [`gaussian_blur_chw`], plus
/// [`SslError::InvalidParameter`] when `sigma_min >= sigma_max` or either bound
/// is non-positive.
pub fn random_gaussian_blur_chw(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    sigma_min: f64,
    sigma_max: f64,
    rng: &mut LcgRng,
) -> SslResult<Vec<f32>> {
    if sigma_min <= 0.0 || !sigma_min.is_finite() {
        return Err(SslError::InvalidParameter {
            name: "sigma_min".into(),
            reason: format!("must be positive and finite, got {sigma_min}"),
        });
    }
    if sigma_max <= sigma_min || !sigma_max.is_finite() {
        return Err(SslError::InvalidParameter {
            name: "sigma_max".into(),
            reason: format!("must be > sigma_min ({sigma_min}) and finite, got {sigma_max}"),
        });
    }
    let u = rng.next_f32() as f64;
    let sigma = sigma_min + u * (sigma_max - sigma_min);
    gaussian_blur_chw(pixels, channels, height, width, sigma)
}

// ─── Solarization ─────────────────────────────────────────────────────────────

/// Solarize a CHW image in `[0, 1]`.
///
/// For each pixel `p`:
/// - `p → p`       if `p < threshold`
/// - `p → 1 - p`   if `p >= threshold`
///
/// This matches the behaviour of `PIL.ImageOps.solarize` when the byte-space
/// threshold is mapped to normalised `f32` space.
///
/// # Parameters
/// - `pixels`    — flat `[C × H × W]` row-major input.
/// - `channels`  — number of channels `C`.
/// - `height`    — number of rows `H`.
/// - `width`     — number of columns `W`.
/// - `threshold` — inversion threshold in `[0, 1]`.
///
/// # Errors
/// - [`SslError::EmptyInput`] if any spatial dimension is zero.
/// - [`SslError::DimensionMismatch`] if slice length != `C·H·W`.
/// - [`SslError::InvalidParameter`] if `threshold` is outside `[0, 1]` or NaN.
pub fn solarize(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    threshold: f32,
) -> SslResult<Vec<f32>> {
    if channels == 0 || height == 0 || width == 0 {
        return Err(SslError::EmptyInput);
    }
    let expected = channels * height * width;
    if pixels.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: pixels.len(),
        });
    }
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(SslError::InvalidParameter {
            name: "threshold".into(),
            reason: format!("must be in [0, 1] and finite, got {threshold}"),
        });
    }
    let out: Vec<f32> = pixels
        .iter()
        .map(|&p| if p < threshold { p } else { 1.0 - p })
        .collect();
    Ok(out)
}

/// Apply solarization with probability `probability` using `rng`.
///
/// With probability `1 - probability` the image is returned unchanged (clone).
/// With probability `probability` the full solarize transform is applied.
///
/// # Errors
/// - [`SslError::InvalidParameter`] if `probability` is outside `[0, 1]` or NaN.
/// - All errors from [`solarize`].
pub fn random_solarize(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    threshold: f32,
    probability: f32,
    rng: &mut LcgRng,
) -> SslResult<Vec<f32>> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(SslError::InvalidParameter {
            name: "probability".into(),
            reason: format!("must be in [0, 1] and finite, got {probability}"),
        });
    }
    if rng.next_f32() >= probability {
        // Pass-through: validate dims but return clone without modification.
        if channels == 0 || height == 0 || width == 0 {
            return Err(SslError::EmptyInput);
        }
        let expected = channels * height * width;
        if pixels.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: pixels.len(),
            });
        }
        return Ok(pixels.to_vec());
    }
    solarize(pixels, channels, height, width, threshold)
}

// ─── Gaussian noise ───────────────────────────────────────────────────────────

/// Add isotropic zero-mean Gaussian noise with the given `std_dev` to every
/// pixel in a CHW image.
///
/// The noise is sampled via Box-Muller (the `LcgRng::next_normal_pair` method)
/// and scaled by `std_dev`.  The output is clamped to `[0, 1]`.
///
/// # Parameters
/// - `pixels`  — input flat CHW buffer (any length); returned as a new `Vec`.
/// - `std_dev` — standard deviation of the additive noise; must be `>= 0` and finite.
/// - `rng`     — mutable reference to the deterministic LCG generator.
///
/// # Errors
/// - [`SslError::InvalidParameter`] if `std_dev < 0` or not finite.
/// - [`SslError::EmptyInput`] if `pixels` is empty.
pub fn add_gaussian_noise(pixels: &[f32], std_dev: f32, rng: &mut LcgRng) -> SslResult<Vec<f32>> {
    if pixels.is_empty() {
        return Err(SslError::EmptyInput);
    }
    if !std_dev.is_finite() || std_dev < 0.0 {
        return Err(SslError::InvalidParameter {
            name: "std_dev".into(),
            reason: format!("must be >= 0 and finite, got {std_dev}"),
        });
    }
    let n = pixels.len();
    let mut out = pixels.to_vec();
    // Consume pairs; handle odd length via the second element of the last pair.
    let mut i = 0;
    while i + 1 < n {
        let (a, b) = rng.next_normal_pair();
        out[i] = (pixels[i] + a * std_dev).clamp(0.0, 1.0);
        out[i + 1] = (pixels[i + 1] + b * std_dev).clamp(0.0, 1.0);
        i += 2;
    }
    if i < n {
        let (a, _) = rng.next_normal_pair();
        out[i] = (pixels[i] + a * std_dev).clamp(0.0, 1.0);
    }
    Ok(out)
}

// ─── SimCLR / MoCo augmentation pipeline ─────────────────────────────────────

/// Configuration for the combined SimCLR / BYOL blur + solarization step.
///
/// Default values reproduce the SimCLR view-1 augmentation:
/// - Gaussian blur: σ ∈ [0.1, 2.0], applied with probability 0.5.
/// - Solarization:  threshold 0.5, disabled (`solar_prob = 0.0`).
///
/// For the BYOL view-2 recipe set `solar_prob = 0.2` and `blur_prob = 0.1`.
#[derive(Debug, Clone, PartialEq)]
pub struct SimClrBlurSolarConfig {
    /// Lower bound of the uniform sigma range (exclusive 0.0).
    pub blur_sigma_min: f64,
    /// Upper bound of the uniform sigma range.
    pub blur_sigma_max: f64,
    /// Probability with which Gaussian blur is applied (`[0, 1]`).
    pub blur_prob: f32,
    /// Solarization threshold in `[0, 1]`.
    pub solar_threshold: f32,
    /// Probability with which solarization is applied (`[0, 1]`).
    pub solar_prob: f32,
}

impl Default for SimClrBlurSolarConfig {
    fn default() -> Self {
        Self {
            blur_sigma_min: 0.1,
            blur_sigma_max: 2.0,
            blur_prob: 0.5,
            solar_threshold: 0.5,
            solar_prob: 0.0,
        }
    }
}

/// Apply the SimCLR / BYOL blur + solarization augmentation sequence to a CHW
/// image.
///
/// The pipeline executes two independent probabilistic transforms in order:
/// 1. Random Gaussian blur with σ sampled from
///    `[config.blur_sigma_min, config.blur_sigma_max]`, applied with probability
///    `config.blur_prob`.
/// 2. Solarization with `config.solar_threshold`, applied with probability
///    `config.solar_prob`.
///
/// Each transform is applied independently; both, one, or neither may fire on a
/// given call.  The function is pure (returns a new `Vec`; never mutates the
/// input).
///
/// # Errors
/// - [`SslError::EmptyInput`] if any dimension is zero.
/// - [`SslError::DimensionMismatch`] if `pixels.len() != C·H·W`.
/// - [`SslError::InvalidParameter`] for invalid config values.
pub fn simclr_blur_solar(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    config: &SimClrBlurSolarConfig,
    rng: &mut LcgRng,
) -> SslResult<Vec<f32>> {
    // Validate spatial dims and slice length once up-front.
    if channels == 0 || height == 0 || width == 0 {
        return Err(SslError::EmptyInput);
    }
    let expected = channels * height * width;
    if pixels.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: pixels.len(),
        });
    }
    // Validate config fields eagerly so we never silently mis-apply.
    if !config.blur_prob.is_finite() || !(0.0..=1.0).contains(&config.blur_prob) {
        return Err(SslError::InvalidParameter {
            name: "blur_prob".into(),
            reason: format!("must be in [0, 1], got {}", config.blur_prob),
        });
    }
    if !config.solar_prob.is_finite() || !(0.0..=1.0).contains(&config.solar_prob) {
        return Err(SslError::InvalidParameter {
            name: "solar_prob".into(),
            reason: format!("must be in [0, 1], got {}", config.solar_prob),
        });
    }

    // Step 1 — Gaussian blur (probabilistic).
    let after_blur: Vec<f32> = if rng.next_f32() < config.blur_prob {
        random_gaussian_blur_chw(
            pixels,
            channels,
            height,
            width,
            config.blur_sigma_min,
            config.blur_sigma_max,
            rng,
        )?
    } else {
        pixels.to_vec()
    };

    // Step 2 — Solarization (probabilistic).
    let after_solar: Vec<f32> = if rng.next_f32() < config.solar_prob {
        solarize(&after_blur, channels, height, width, config.solar_threshold)?
    } else {
        after_blur
    };

    Ok(after_solar)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Variance of a flat slice.
    fn variance(v: &[f32]) -> f32 {
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n
    }

    // ── Gaussian blur tests ───────────────────────────────────────────────────

    #[test]
    fn gaussian_blur_preserves_mean() {
        // Uniform image: sum should be unchanged (no energy loss).
        let (c, h, w) = (1, 32, 32);
        let pixels = vec![0.5_f32; c * h * w];
        let blurred = gaussian_blur_chw(&pixels, c, h, w, 1.0).unwrap();
        let orig_sum: f32 = pixels.iter().sum();
        let blur_sum: f32 = blurred.iter().sum();
        // With clamp-to-edge the sums are exactly equal for a constant image.
        assert!(
            (blur_sum - orig_sum).abs() < 1e-3,
            "orig_sum={orig_sum}, blur_sum={blur_sum}"
        );
    }

    #[test]
    fn gaussian_blur_shape_preserved() {
        let (c, h, w) = (3, 16, 24);
        let pixels = vec![0.3_f32; c * h * w];
        let out = gaussian_blur_chw(&pixels, c, h, w, 0.8).unwrap();
        assert_eq!(out.len(), c * h * w);
    }

    #[test]
    fn gaussian_blur_identity_sigma_near_zero() {
        // Very small sigma → kernel is essentially a Dirac delta → output ≈ input.
        let (c, h, w) = (1, 8, 8);
        let mut rng = LcgRng::new(42);
        let pixels: Vec<f32> = (0..c * h * w).map(|_| rng.next_f32()).collect();
        let out = gaussian_blur_chw(&pixels, c, h, w, 1e-5).unwrap();
        for (a, b) in pixels.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-4, "a={a} b={b}");
        }
    }

    #[test]
    fn gaussian_blur_decreases_variance() {
        // Checkerboard: alternating 0.0 and 1.0 — high variance.
        let (c, h, w) = (1, 16, 16);
        let pixels: Vec<f32> = (0..c * h * w)
            .map(|i| {
                if (i + i / w) % 2 == 0 {
                    0.0_f32
                } else {
                    1.0_f32
                }
            })
            .collect();
        let orig_var = variance(&pixels);
        let blurred = gaussian_blur_chw(&pixels, c, h, w, 2.0).unwrap();
        let blur_var = variance(&blurred);
        assert!(
            blur_var < orig_var,
            "expected variance reduction: orig={orig_var}, blurred={blur_var}"
        );
    }

    #[test]
    fn gaussian_blur_negative_sigma_error() {
        // Shape [C=1, H=4, W=4] flattened: 16 pixels.
        let pixels = vec![0.5_f32; 16];
        let result = gaussian_blur_chw(&pixels, 1, 4, 4, -1.0);
        assert!(
            matches!(result, Err(SslError::InvalidParameter { .. })),
            "expected InvalidParameter, got {:?}",
            result
        );
    }

    #[test]
    fn gaussian_blur_zero_sigma_error() {
        // Shape [C=1, H=4, W=4] flattened: 16 pixels.
        let pixels = vec![0.5_f32; 16];
        let result = gaussian_blur_chw(&pixels, 1, 4, 4, 0.0);
        assert!(matches!(result, Err(SslError::InvalidParameter { .. })));
    }

    #[test]
    fn gaussian_blur_multichannel_independent() {
        // Each channel has a different constant value; blur should preserve them.
        let (c, h, w) = (3, 10, 10);
        let mut pixels = vec![0.0_f32; c * h * w];
        let plane = h * w;
        for ch in 0..c {
            for p in pixels[ch * plane..(ch + 1) * plane].iter_mut() {
                *p = (ch as f32 + 1.0) * 0.25;
            }
        }
        let out = gaussian_blur_chw(&pixels, c, h, w, 1.5).unwrap();
        for ch in 0..c {
            let expected_val = (ch as f32 + 1.0) * 0.25;
            for &v in &out[ch * plane..(ch + 1) * plane] {
                assert!(
                    (v - expected_val).abs() < 1e-4,
                    "ch={ch} expected={expected_val} got={v}"
                );
            }
        }
    }

    // ── Solarization tests ────────────────────────────────────────────────────

    #[test]
    fn solarize_below_threshold_unchanged() {
        // Pixel 0.3 with threshold 0.5 → stays 0.3.
        // Shape [C=1, H=4, W=4] flattened: 16 pixels.
        let pixels = vec![0.3_f32; 16];
        let out = solarize(&pixels, 1, 4, 4, 0.5).unwrap();
        for &v in &out {
            assert!((v - 0.3).abs() < 1e-6, "v={v}");
        }
    }

    #[test]
    fn solarize_above_threshold_inverted() {
        // Pixel 0.8 with threshold 0.5 → 1.0 - 0.8 = 0.2.
        // Shape [C=1, H=4, W=4] flattened: 16 pixels.
        let pixels = vec![0.8_f32; 16];
        let out = solarize(&pixels, 1, 4, 4, 0.5).unwrap();
        for &v in &out {
            assert!((v - 0.2).abs() < 1e-6, "v={v}");
        }
    }

    #[test]
    fn solarize_at_threshold_inverted() {
        // Pixel exactly equal to threshold → 1 - threshold.
        let threshold = 0.5_f32;
        // Shape [C=1, H=4, W=4] flattened: 16 pixels.
        let pixels = vec![threshold; 16];
        let out = solarize(&pixels, 1, 4, 4, threshold).unwrap();
        for &v in &out {
            assert!((v - (1.0 - threshold)).abs() < 1e-6, "v={v}");
        }
    }

    #[test]
    fn solarize_preserves_shape() {
        let (c, h, w) = (3, 8, 12);
        let pixels = vec![0.6_f32; c * h * w];
        let out = solarize(&pixels, c, h, w, 0.5).unwrap();
        assert_eq!(out.len(), c * h * w);
    }

    #[test]
    fn random_solarize_prob_zero() {
        // With probability 0 the image is always returned unchanged.
        let pixels: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let mut rng = LcgRng::new(99);
        let out = random_solarize(&pixels, 1, 8, 8, 0.5, 0.0, &mut rng).unwrap();
        assert_eq!(out, pixels);
    }

    #[test]
    fn random_solarize_prob_one() {
        // With probability 1 the image is always fully solarized.
        let pixels: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let mut rng = LcgRng::new(7);
        let out = random_solarize(&pixels, 1, 8, 8, 0.5, 1.0, &mut rng).unwrap();
        let expected = solarize(&pixels, 1, 8, 8, 0.5).unwrap();
        assert_eq!(out, expected);
    }

    // ── Gaussian noise tests ──────────────────────────────────────────────────

    #[test]
    fn add_gaussian_noise_output_in_range() {
        let pixels = vec![0.5_f32; 3 * 16 * 16];
        let mut rng = LcgRng::new(123);
        let out = add_gaussian_noise(&pixels, 0.2, &mut rng).unwrap();
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "out-of-range pixel: {v}");
        }
    }

    #[test]
    fn add_gaussian_noise_nonzero_std_changes_image() {
        // Shape [C=1, H=32, W=32] flattened: 1024 pixels.
        let pixels = vec![0.5_f32; 1024];
        let mut rng = LcgRng::new(55);
        let out = add_gaussian_noise(&pixels, 0.1, &mut rng).unwrap();
        // With std_dev 0.1, it is astronomically unlikely that all pixels are
        // unchanged, so at least one should differ.
        let changed = pixels
            .iter()
            .zip(out.iter())
            .filter(|(a, b)| *a != *b)
            .count();
        assert!(changed > 0, "no pixels changed with std_dev=0.1");
    }

    #[test]
    fn add_gaussian_noise_zero_std_unchanged() {
        // std_dev = 0.0 → noise term is 0 → output equals input.
        let pixels = vec![0.4_f32; 64];
        let mut rng = LcgRng::new(11);
        let out = add_gaussian_noise(&pixels, 0.0, &mut rng).unwrap();
        for (a, b) in pixels.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-7, "a={a} b={b}");
        }
    }

    #[test]
    fn add_gaussian_noise_negative_std_error() {
        let pixels = vec![0.5_f32; 16];
        let mut rng = LcgRng::new(1);
        let result = add_gaussian_noise(&pixels, -0.1, &mut rng);
        assert!(matches!(result, Err(SslError::InvalidParameter { .. })));
    }

    // ── SimCLR pipeline tests ─────────────────────────────────────────────────

    #[test]
    fn simclr_blur_solar_shape() {
        let (c, h, w) = (3, 16, 16);
        let pixels = vec![0.5_f32; c * h * w];
        let cfg = SimClrBlurSolarConfig::default();
        let mut rng = LcgRng::new(42);
        let out = simclr_blur_solar(&pixels, c, h, w, &cfg, &mut rng).unwrap();
        assert_eq!(out.len(), c * h * w);
    }

    #[test]
    fn simclr_blur_solar_both_probs_one() {
        // With blur_prob=1 and solar_prob=1 both transforms always fire.
        let (c, h, w) = (1, 8, 8);
        let pixels: Vec<f32> = (0..c * h * w).map(|i| (i as f32 * 0.01).min(1.0)).collect();
        let cfg = SimClrBlurSolarConfig {
            blur_sigma_min: 0.5,
            blur_sigma_max: 1.0,
            blur_prob: 1.0,
            solar_threshold: 0.5,
            solar_prob: 1.0,
        };
        let mut rng = LcgRng::new(77);
        let out = simclr_blur_solar(&pixels, c, h, w, &cfg, &mut rng).unwrap();
        assert_eq!(out.len(), pixels.len());
        // The output must differ from the original because we blur then solarize.
        let identical = pixels
            .iter()
            .zip(out.iter())
            .all(|(a, b)| (a - b).abs() < 1e-8);
        assert!(!identical, "expected pipeline to modify the image");
    }

    #[test]
    fn simclr_blur_solar_probs_zero_passthrough() {
        // With both probs = 0, neither transform fires and output == input.
        let (c, h, w) = (1, 4, 4);
        let pixels: Vec<f32> = (0..c * h * w)
            .map(|i| i as f32 / (c * h * w) as f32)
            .collect();
        let cfg = SimClrBlurSolarConfig {
            blur_sigma_min: 0.1,
            blur_sigma_max: 2.0,
            blur_prob: 0.0,
            solar_threshold: 0.5,
            solar_prob: 0.0,
        };
        let mut rng = LcgRng::new(3);
        let out = simclr_blur_solar(&pixels, c, h, w, &cfg, &mut rng).unwrap();
        assert_eq!(out, pixels);
    }

    // ── Empty / dim-mismatch edge cases ───────────────────────────────────────

    #[test]
    fn empty_image_error_blur() {
        let result = gaussian_blur_chw(&[], 0, 8, 8, 1.0);
        assert!(matches!(result, Err(SslError::EmptyInput)));
    }

    #[test]
    fn empty_image_error_solarize() {
        let result = solarize(&[], 0, 8, 8, 0.5);
        assert!(matches!(result, Err(SslError::EmptyInput)));
    }

    #[test]
    fn dim_mismatch_blur() {
        let pixels = vec![0.5_f32; 10]; // wrong length for 1×4×4 = 16
        let result = gaussian_blur_chw(&pixels, 1, 4, 4, 1.0);
        assert!(matches!(result, Err(SslError::DimensionMismatch { .. })));
    }

    #[test]
    fn random_gaussian_blur_sigma_range_error() {
        // Shape [C=1, H=4, W=4] flattened: 16 pixels.
        let pixels = vec![0.5_f32; 16];
        let mut rng = LcgRng::new(1);
        // sigma_min >= sigma_max → error.
        let result = random_gaussian_blur_chw(&pixels, 1, 4, 4, 2.0, 0.5, &mut rng);
        assert!(matches!(result, Err(SslError::InvalidParameter { .. })));
    }
}

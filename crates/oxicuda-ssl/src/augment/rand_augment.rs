//! RandAugment and AutoAugment augmentation policies for CHW images.
//!
//! # Layout convention
//!
//! All functions operate on a flat `[C × H × W]` row-major buffer where
//! channel `c`, row `y`, and column `x` maps to index `c * H * W + y * W + x`.
//! Pixel values are `f32` in `[0.0, 1.0]`.
//!
//! # References
//! - Cubuk et al., "RandAugment: Practical automated data augmentation with a
//!   reduced search space", NeurIPS 2020.
//! - Cubuk et al., "AutoAugment: Learning Augmentation Policies from Data",
//!   CVPR 2019.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Augmentation operation enum ──────────────────────────────────────────────

/// The 14 canonical RandAugment operations.
#[derive(Debug, Clone, PartialEq)]
pub enum AugOp {
    /// Pass the image through unchanged.
    Identity,
    /// Stretch per-channel histogram to [0, 1].
    AutoContrast,
    /// Histogram equalization per channel.
    Equalize,
    /// Rotate by ±30° scaled by magnitude.
    Rotate,
    /// Invert pixels above a magnitude-derived threshold.
    Solarize,
    /// Blend between grayscale and original (saturation adjust).
    Color,
    /// Reduce effective bit depth.
    Posterize,
    /// Blend between channel-mean image and original.
    Contrast,
    /// Blend between black and original.
    Brightness,
    /// Sharpen via unsharp masking.
    Sharpness,
    /// Shear horizontally.
    ShearX,
    /// Shear vertically.
    ShearY,
    /// Translate horizontally.
    TranslateX,
    /// Translate vertically.
    TranslateY,
}

/// Default set of all 14 RandAugment operations in canonical order.
pub fn all_aug_ops() -> Vec<AugOp> {
    vec![
        AugOp::Identity,
        AugOp::AutoContrast,
        AugOp::Equalize,
        AugOp::Rotate,
        AugOp::Solarize,
        AugOp::Color,
        AugOp::Posterize,
        AugOp::Contrast,
        AugOp::Brightness,
        AugOp::Sharpness,
        AugOp::ShearX,
        AugOp::ShearY,
        AugOp::TranslateX,
        AugOp::TranslateY,
    ]
}

// ─── Configuration types ──────────────────────────────────────────────────────

/// Configuration for the RandAugment policy (Cubuk et al., NeurIPS 2020).
#[derive(Debug, Clone)]
pub struct RandAugmentConfig {
    /// N: number of operations to sample and apply per image (default: 2).
    pub n_ops: usize,
    /// M: shared magnitude on a 0–30 scale (default: 9.0).
    pub magnitude: f32,
    /// Fill value for geometric transforms when sampling outside the image boundary.
    pub fill_value: f32,
    /// Pool of operations to sample from (default: all 14).
    pub ops: Vec<AugOp>,
}

impl Default for RandAugmentConfig {
    fn default() -> Self {
        Self {
            n_ops: 2,
            magnitude: 9.0,
            fill_value: 0.5,
            ops: all_aug_ops(),
        }
    }
}

impl RandAugmentConfig {
    /// Validate that the config is self-consistent.
    pub fn validate(&self) -> SslResult<()> {
        if !(self.magnitude.is_finite() && (0.0..=30.0).contains(&self.magnitude)) {
            return Err(SslError::InvalidParameter {
                name: "magnitude".into(),
                reason: format!("must be in [0, 30] and finite, got {}", self.magnitude),
            });
        }
        if !(self.fill_value.is_finite() && (0.0..=1.0).contains(&self.fill_value)) {
            return Err(SslError::InvalidParameter {
                name: "fill_value".into(),
                reason: format!("must be in [0, 1] and finite, got {}", self.fill_value),
            });
        }
        if self.ops.is_empty() {
            return Err(SslError::InvalidParameter {
                name: "ops".into(),
                reason: "must contain at least one operation".into(),
            });
        }
        Ok(())
    }
}

/// AutoAugment sub-policy: two sequential operations each with a probability
/// and discrete magnitude level.
///
/// Each element is `(op, probability, magnitude_level)`.  Probability is in
/// `[0.0, 1.0]`; magnitude level is an integer in `[0, 10]` (AutoAugment
/// convention) and is internally remapped to the 0–30 RandAugment magnitude
/// scale before being passed to [`apply_aug_op`].
pub type SubPolicy = ((AugOp, f32, usize), (AugOp, f32, usize));

/// Built-in AutoAugment dataset policies.
#[derive(Debug, Clone)]
pub enum AutoAugPolicy {
    /// The 25 sub-policies from the original ImageNet AutoAugment paper.
    ImageNet,
    /// The 25 sub-policies from the original CIFAR-10 AutoAugment paper.
    Cifar10,
    /// User-defined collection of sub-policies.
    Custom(Vec<SubPolicy>),
}

/// Configuration for the AutoAugment policy (Cubuk et al., CVPR 2019).
#[derive(Debug, Clone)]
pub struct AutoAugmentConfig {
    /// Which policy (set of sub-policies) to use.
    pub policy: AutoAugPolicy,
    /// Fill value for geometric transforms.
    pub fill_value: f32,
}

impl Default for AutoAugmentConfig {
    fn default() -> Self {
        Self {
            policy: AutoAugPolicy::ImageNet,
            fill_value: 0.5,
        }
    }
}

// ─── Primitive image operations ───────────────────────────────────────────────

/// Index into a CHW buffer: `c * H * W + y * W + x`.
#[inline]
fn chw_idx(c: usize, y: usize, x: usize, height: usize, width: usize) -> usize {
    c * height * width + y * width + x
}

/// Bilinear sample from a single-channel plane of size `H × W`.
///
/// Coordinates outside `[0, H-1] × [0, W-1]` return `fill_value`.
fn bilinear_sample(
    plane: &[f32],
    height: usize,
    width: usize,
    fy: f32,
    fx: f32,
    fill_value: f32,
) -> f32 {
    if fy < 0.0 || fx < 0.0 || fy > (height - 1) as f32 || fx > (width - 1) as f32 {
        return fill_value;
    }
    let y0 = fy.floor() as usize;
    let x0 = fx.floor() as usize;
    let y1 = (y0 + 1).min(height - 1);
    let x1 = (x0 + 1).min(width - 1);
    let dy = fy - y0 as f32;
    let dx = fx - x0 as f32;

    let v00 = plane[y0 * width + x0];
    let v01 = plane[y0 * width + x1];
    let v10 = plane[y1 * width + x0];
    let v11 = plane[y1 * width + x1];

    let top = v00 * (1.0 - dx) + v01 * dx;
    let bot = v10 * (1.0 - dx) + v11 * dx;
    top * (1.0 - dy) + bot * dy
}

/// Apply an affine warp to all channels of a CHW image.
///
/// For each output pixel `(y, x)`, the source coordinate is:
/// ```text
///   src_y = y + dy_coeff * y + dyx_coeff * x + shift_y
///   src_x = x + dx_coeff * x + dxy_coeff * y + shift_x
/// ```
/// where `dy_coeff`, `dx_coeff`, `dxy_coeff`, `dyx_coeff` encode shear, and
/// `shift_x`, `shift_y` encode translation.  Pixels outside the source image
/// are filled with `fill_value`.
#[allow(clippy::too_many_arguments)]
fn warp_affine(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    // Inverse affine coefficients for the source lookup
    a00: f32, // src_x += a00 * x
    a01: f32, // src_x += a01 * y
    a02: f32, // src_x += a02 (translation x)
    a10: f32, // src_y += a10 * x
    a11: f32, // src_y += a11 * y
    a12: f32, // src_y += a12 (translation y)
    fill_value: f32,
) -> Vec<f32> {
    let plane = height * width;
    let mut out = vec![fill_value; channels * plane];
    for c in 0..channels {
        let src_plane = &pixels[c * plane..(c + 1) * plane];
        let dst_plane = &mut out[c * plane..(c + 1) * plane];
        for y in 0..height {
            for x in 0..width {
                let fx = a00 * x as f32 + a01 * y as f32 + a02;
                let fy = a10 * x as f32 + a11 * y as f32 + a12;
                dst_plane[y * width + x] =
                    bilinear_sample(src_plane, height, width, fy, fx, fill_value);
            }
        }
    }
    out
}

/// Auto-contrast: per-channel linear stretch to [0, 1].
fn op_auto_contrast(pixels: &[f32], channels: usize, height: usize, width: usize) -> Vec<f32> {
    let plane = height * width;
    let mut out = pixels.to_vec();
    for c in 0..channels {
        let ch = &pixels[c * plane..(c + 1) * plane];
        let min_v = ch.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_v = ch.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if (max_v - min_v).abs() < 1e-7 {
            continue; // uniform channel — leave as-is
        }
        let range = max_v - min_v;
        for (dst, &src) in out[c * plane..(c + 1) * plane].iter_mut().zip(ch.iter()) {
            *dst = ((src - min_v) / range).clamp(0.0, 1.0);
        }
    }
    out
}

/// Histogram equalization per channel.
///
/// Pixels are quantized into 256 bins, a CDF is computed, and each pixel is
/// remapped via the CDF so that the output histogram is approximately uniform.
fn op_equalize(pixels: &[f32], channels: usize, height: usize, width: usize) -> Vec<f32> {
    const BINS: usize = 256;
    let plane = height * width;
    let mut out = pixels.to_vec();
    for c in 0..channels {
        let ch = &pixels[c * plane..(c + 1) * plane];
        let mut hist = [0u32; BINS];
        for &p in ch.iter() {
            let bin = ((p * (BINS as f32 - 1.0)).round() as usize).min(BINS - 1);
            hist[bin] += 1;
        }
        // Compute CDF
        let mut cdf = [0u32; BINS];
        cdf[0] = hist[0];
        for i in 1..BINS {
            cdf[i] = cdf[i - 1] + hist[i];
        }
        let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
        let total = plane as u32;
        let denom = total.saturating_sub(cdf_min);
        // Build the LUT
        let mut lut = [0.0_f32; BINS];
        for (i, lut_v) in lut.iter_mut().enumerate() {
            if denom == 0 {
                *lut_v = i as f32 / (BINS as f32 - 1.0);
            } else {
                let mapped = (cdf[i].saturating_sub(cdf_min)) as f32 / denom as f32;
                *lut_v = mapped.clamp(0.0, 1.0);
            }
        }
        for (dst, &src) in out[c * plane..(c + 1) * plane].iter_mut().zip(ch.iter()) {
            let bin = ((src * (BINS as f32 - 1.0)).round() as usize).min(BINS - 1);
            *dst = lut[bin];
        }
    }
    out
}

/// Rotation by `angle_deg` degrees with bilinear interpolation (center pivot).
fn op_rotate(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    angle_deg: f32,
    fill_value: f32,
) -> Vec<f32> {
    let angle_rad = angle_deg * std::f32::consts::PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    let cx = (width as f32 - 1.0) / 2.0;
    let cy = (height as f32 - 1.0) / 2.0;
    // Inverse rotation: given output (x, y), find source.
    // src_x = cos_a * (x - cx) + sin_a * (y - cy) + cx
    // src_y = -sin_a * (x - cx) + cos_a * (y - cy) + cy
    let a00 = cos_a;
    let a01 = sin_a;
    let a02 = -cos_a * cx - sin_a * cy + cx;
    let a10 = -sin_a;
    let a11 = cos_a;
    let a12 = sin_a * cx - cos_a * cy + cy;
    warp_affine(
        pixels, channels, height, width, a00, a01, a02, a10, a11, a12, fill_value,
    )
}

/// Solarize: invert pixels at or above `threshold`.
fn op_solarize(pixels: &[f32], threshold: f32) -> Vec<f32> {
    pixels
        .iter()
        .map(|&p| if p >= threshold { 1.0 - p } else { p })
        .collect()
}

/// Color (saturation) adjustment.
///
/// `alpha` in `[0, 1]`: 0 = grayscale, 1 = original.
/// Uses BT.601 luminance weights.
fn op_color(pixels: &[f32], channels: usize, height: usize, width: usize, alpha: f32) -> Vec<f32> {
    if channels != 3 {
        // For non-RGB images, no-op.
        return pixels.to_vec();
    }
    let plane = height * width;
    let mut out = pixels.to_vec();
    for i in 0..plane {
        let r = pixels[i];
        let g = pixels[plane + i];
        let b = pixels[2 * plane + i];
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        out[i] = (alpha * r + (1.0 - alpha) * y).clamp(0.0, 1.0);
        out[plane + i] = (alpha * g + (1.0 - alpha) * y).clamp(0.0, 1.0);
        out[2 * plane + i] = (alpha * b + (1.0 - alpha) * y).clamp(0.0, 1.0);
    }
    out
}

/// Posterize: keep the top `k` bits of each pixel value (quantized to 8-bit).
///
/// `k` ranges from 4–8; lower = more posterized.
fn op_posterize(pixels: &[f32], k: u32) -> Vec<f32> {
    // k bits: mask is 0xFF with the lower (8-k) bits zeroed.
    let shift = 8u32.saturating_sub(k);
    let mask = if shift >= 8 { 0u8 } else { 0xFFu8 << shift };
    pixels
        .iter()
        .map(|&p| {
            let byte = (p * 255.0).round().clamp(0.0, 255.0) as u8;
            let masked = byte & mask;
            (masked as f32 / 255.0).clamp(0.0, 1.0)
        })
        .collect()
}

/// Contrast: blend between channel-mean and original.
fn op_contrast(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    alpha: f32,
) -> Vec<f32> {
    let plane = height * width;
    let mut out = pixels.to_vec();
    for c in 0..channels {
        let ch = &pixels[c * plane..(c + 1) * plane];
        let mean = ch.iter().sum::<f32>() / plane as f32;
        for (dst, &src) in out[c * plane..(c + 1) * plane].iter_mut().zip(ch.iter()) {
            *dst = ((1.0 - alpha) * mean + alpha * src).clamp(0.0, 1.0);
        }
    }
    out
}

/// Brightness: blend between black (0) and original.
fn op_brightness(pixels: &[f32], strength: f32) -> Vec<f32> {
    pixels
        .iter()
        .map(|&p| (strength * p).clamp(0.0, 1.0))
        .collect()
}

/// Sharpness: blend between blurred (3×3 box) and original.
fn op_sharpness(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    alpha: f32,
) -> Vec<f32> {
    // Build a blurred version using a 3×3 box filter.
    let plane = height * width;
    let mut blurred = vec![0.0_f32; channels * plane];
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0_f32;
                let mut count = 0u32;
                for dy in 0..3usize {
                    let ny = y + dy;
                    if ny == 0 || ny > height {
                        continue;
                    }
                    let ny = ny - 1;
                    for dx in 0..3usize {
                        let nx = x + dx;
                        if nx == 0 || nx > width {
                            continue;
                        }
                        let nx = nx - 1;
                        acc += pixels[chw_idx(c, ny, nx, height, width)];
                        count += 1;
                    }
                }
                blurred[chw_idx(c, y, x, height, width)] =
                    if count > 0 { acc / count as f32 } else { 0.0 };
            }
        }
    }
    // Blend: alpha * original + (1 - alpha) * blurred
    pixels
        .iter()
        .zip(blurred.iter())
        .map(|(&orig, &blur)| (alpha * orig + (1.0 - alpha) * blur).clamp(0.0, 1.0))
        .collect()
}

/// Horizontal shear by `shear` radians (inverse warp).
fn op_shear_x(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    shear: f32,
    fill_value: f32,
) -> Vec<f32> {
    // For output (x, y): src_x = x - shear * y; src_y = y.
    warp_affine(
        pixels, channels, height, width, 1.0, -shear, 0.0, // src_x coefficients
        0.0, 1.0, 0.0, // src_y coefficients
        fill_value,
    )
}

/// Vertical shear by `shear` radians (inverse warp).
fn op_shear_y(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    shear: f32,
    fill_value: f32,
) -> Vec<f32> {
    // For output (x, y): src_x = x; src_y = y - shear * x.
    warp_affine(
        pixels, channels, height, width, 1.0, 0.0, 0.0, // src_x coefficients
        -shear, 1.0, 0.0, // src_y coefficients
        fill_value,
    )
}

/// Horizontal translation by `shift_x` pixels.
fn op_translate_x(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    shift_x: f32,
    fill_value: f32,
) -> Vec<f32> {
    warp_affine(
        pixels, channels, height, width, 1.0, 0.0, -shift_x, 0.0, 1.0, 0.0, fill_value,
    )
}

/// Vertical translation by `shift_y` pixels.
fn op_translate_y(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    shift_y: f32,
    fill_value: f32,
) -> Vec<f32> {
    warp_affine(
        pixels, channels, height, width, 1.0, 0.0, 0.0, 0.0, 1.0, -shift_y, fill_value,
    )
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Apply a single augmentation operation with the given magnitude and fill value.
///
/// # Parameters
/// - `pixels`     — flat `[C × H × W]` CHW input in `[0, 1]`.
/// - `channels`   — number of channels `C`.
/// - `height`     — image height `H`.
/// - `width`      — image width `W`.
/// - `op`         — which [`AugOp`] to apply.
/// - `magnitude`  — shared magnitude in `[0, 30]`.
/// - `fill_value` — fill for geometric OOB pixels.
///
/// # Errors
/// - [`SslError::EmptyInput`] if any dimension is zero.
/// - [`SslError::DimensionMismatch`] if `pixels.len() != C·H·W`.
/// - [`SslError::InvalidParameter`] for invalid magnitude or fill_value.
pub fn apply_aug_op(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    op: &AugOp,
    magnitude: f32,
    fill_value: f32,
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
    if !(magnitude.is_finite() && (0.0..=30.0).contains(&magnitude)) {
        return Err(SslError::InvalidParameter {
            name: "magnitude".into(),
            reason: format!("must be in [0, 30] and finite, got {magnitude}"),
        });
    }
    if !(fill_value.is_finite() && (0.0..=1.0).contains(&fill_value)) {
        return Err(SslError::InvalidParameter {
            name: "fill_value".into(),
            reason: format!("must be in [0, 1] and finite, got {fill_value}"),
        });
    }

    let m = magnitude / 30.0; // normalised to [0, 1]

    let result = match op {
        AugOp::Identity => pixels.to_vec(),

        AugOp::AutoContrast => op_auto_contrast(pixels, channels, height, width),

        AugOp::Equalize => op_equalize(pixels, channels, height, width),

        AugOp::Rotate => {
            // ±30° max; use a signed direction encoded by (m >= 0.5).
            // In RandAugment the sign is sampled externally; here we use the
            // direct magnitude linearly in [0°, 30°] (caller picks sign).
            let angle = m * 30.0;
            op_rotate(pixels, channels, height, width, angle, fill_value)
        }

        AugOp::Solarize => {
            // threshold = 1 - m: magnitude=0 → threshold=1 (nothing flipped);
            // magnitude=30 → threshold=0 (all pixels flipped).
            let threshold = (1.0 - m).clamp(0.0, 1.0);
            op_solarize(pixels, threshold)
        }

        AugOp::Color => {
            // alpha=1 at magnitude=0 (original); alpha decreases with magnitude.
            let alpha = (1.0 - m * 0.9).clamp(0.0, 1.0);
            op_color(pixels, channels, height, width, alpha)
        }

        AugOp::Posterize => {
            // k = 8 - floor(m * 4): range [4, 8].
            let k = 8 - (m * 4.0).floor() as u32;
            let k = k.max(1);
            op_posterize(pixels, k)
        }

        AugOp::Contrast => {
            // alpha=1 at magnitude=0 (original); blend toward mean.
            let alpha = (1.0 - m * 0.9).clamp(0.0, 1.0);
            op_contrast(pixels, channels, height, width, alpha)
        }

        AugOp::Brightness => {
            // strength = m * 0.9 + 0.1 so at m=0 strength≈0.1 (dim) and m=1 strength=1.0.
            let strength = (m * 0.9 + 0.1).clamp(0.0, 1.0);
            op_brightness(pixels, strength)
        }

        AugOp::Sharpness => {
            // alpha=1 → sharp (original); alpha=0 → fully blurred.
            let alpha = m.clamp(0.0, 1.0);
            op_sharpness(pixels, channels, height, width, alpha)
        }

        AugOp::ShearX => {
            let shear = m * 0.3;
            op_shear_x(pixels, channels, height, width, shear, fill_value)
        }

        AugOp::ShearY => {
            let shear = m * 0.3;
            op_shear_y(pixels, channels, height, width, shear, fill_value)
        }

        AugOp::TranslateX => {
            let shift = m * 0.33 * width as f32;
            op_translate_x(pixels, channels, height, width, shift, fill_value)
        }

        AugOp::TranslateY => {
            let shift = m * 0.33 * height as f32;
            op_translate_y(pixels, channels, height, width, shift, fill_value)
        }
    };

    Ok(result)
}

/// Apply the RandAugment policy to a CHW image.
///
/// Randomly samples `config.n_ops` operations (with replacement) from
/// `config.ops` and applies each in sequence using `config.magnitude`.
/// When `n_ops == 0` the image is returned unchanged.
///
/// # Errors
/// - [`SslError::EmptyInput`] if any dimension is zero.
/// - [`SslError::DimensionMismatch`] if slice length != `C·H·W`.
/// - [`SslError::InvalidParameter`] if config is invalid.
pub fn rand_augment(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    config: &RandAugmentConfig,
    rng: &mut LcgRng,
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
    config.validate()?;

    if config.n_ops == 0 {
        return Ok(pixels.to_vec());
    }

    let n_pool = config.ops.len();
    let mut current = pixels.to_vec();

    for _ in 0..config.n_ops {
        let idx = rng.next_usize(n_pool);
        let op = &config.ops[idx];
        current = apply_aug_op(
            &current,
            channels,
            height,
            width,
            op,
            config.magnitude,
            config.fill_value,
        )?;
    }
    Ok(current)
}

// ─── AutoAugment policy tables ────────────────────────────────────────────────

/// Build the 25 ImageNet AutoAugment sub-policies from Cubuk et al., CVPR 2019.
///
/// Each entry is `((op, prob, mag_level), (op, prob, mag_level))` where
/// `mag_level` is in `[0, 10]` (scaled ×3 to reach the 0–30 magnitude range).
fn imagenet_sub_policies() -> Vec<SubPolicy> {
    use AugOp::*;
    vec![
        ((Posterize, 0.4, 8), (Rotate, 0.6, 9)),
        ((Solarize, 0.6, 5), (AutoContrast, 0.6, 5)),
        ((Equalize, 0.8, 8), (Equalize, 0.6, 3)),
        ((Posterize, 0.6, 7), (Posterize, 0.6, 6)),
        ((Equalize, 0.4, 7), (Solarize, 0.2, 4)),
        ((Equalize, 0.4, 4), (Rotate, 0.8, 8)),
        ((Solarize, 0.6, 3), (Equalize, 0.6, 7)),
        ((Posterize, 0.8, 5), (Equalize, 1.0, 2)),
        ((Rotate, 0.2, 3), (Solarize, 0.6, 8)),
        ((Equalize, 0.6, 8), (Posterize, 0.4, 6)),
        ((Rotate, 0.8, 8), (Color, 1.0, 2)),
        ((Rotate, 0.9, 9), (Equalize, 1.0, 2)),
        ((Equalize, 0.6, 7), (Equalize, 0.6, 3)),
        ((Equalize, 0.6, 4), (Rotate, 0.6, 4)),
        ((Solarize, 0.6, 7), (Rotate, 0.6, 3)),
        ((ShearX, 0.8, 8), (Solarize, 0.8, 4)),
        ((Color, 0.8, 3), (Color, 1.0, 7)),
        ((Color, 0.4, 1), (Rotate, 0.6, 8)),
        ((Color, 0.8, 8), (Solarize, 0.8, 8)),
        ((Equalize, 0.4, 8), (Equalize, 0.8, 3)),
        ((Posterize, 0.4, 6), (Rotate, 0.4, 3)),
        ((Equalize, 0.6, 7), (Color, 0.4, 4)),
        ((Color, 0.4, 9), (Equalize, 0.6, 3)),
        ((Color, 0.8, 8), (Contrast, 0.6, 1)),
        ((Rotate, 0.8, 8), (Contrast, 1.0, 2)),
    ]
}

/// Build the 25 CIFAR-10 AutoAugment sub-policies from Cubuk et al., CVPR 2019.
fn cifar10_sub_policies() -> Vec<SubPolicy> {
    use AugOp::*;
    vec![
        ((Equalize, 0.1, 8), (ShearY, 0.6, 4)),
        ((Color, 0.6, 1), (Equalize, 0.6, 2)),
        ((Sharpness, 0.6, 7), (Brightness, 0.6, 6)),
        ((AutoContrast, 0.4, 0), (Equalize, 0.6, 0)),
        ((Equalize, 1.0, 9), (ShearY, 0.6, 3)),
        ((Color, 0.4, 3), (AutoContrast, 0.6, 1)),
        ((ShearX, 0.8, 5), (Color, 1.0, 3)),
        ((ShearX, 0.4, 4), (Posterize, 0.4, 7)),
        ((Color, 0.4, 3), (Brightness, 0.6, 7)),
        ((ShearY, 0.6, 4), (Color, 1.0, 9)),
        ((Equalize, 0.6, 9), (Posterize, 0.4, 6)),
        ((Solarize, 0.4, 9), (AutoContrast, 0.6, 3)),
        ((AutoContrast, 0.6, 1), (Posterize, 0.6, 9)),
        ((Equalize, 0.4, 9), (Solarize, 0.4, 5)),
        ((Brightness, 0.2, 1), (Equalize, 0.6, 2)),
        ((Equalize, 0.0, 0), (Equalize, 1.0, 0)),
        ((AutoContrast, 0.2, 0), (Equalize, 0.6, 0)),
        ((Equalize, 0.2, 0), (AutoContrast, 0.6, 0)),
        ((Contrast, 0.2, 0), (Equalize, 0.6, 0)),
        ((Brightness, 0.6, 5), (Contrast, 0.6, 6)),
        ((AutoContrast, 0.8, 5), (Rotate, 0.6, 2)),
        ((Solarize, 0.4, 3), (Brightness, 0.8, 9)),
        ((Rotate, 0.6, 6), (Color, 1.0, 1)),
        ((Equalize, 0.4, 5), (AutoContrast, 0.6, 5)),
        ((Rotate, 0.6, 6), (Posterize, 0.8, 8)),
    ]
}

/// Apply the AutoAugment policy to a CHW image.
///
/// 1. Uniformly samples one sub-policy from the policy's list.
/// 2. For each of the two operations in the sub-policy, applies it with the
///    corresponding probability and magnitude level.
///
/// AutoAugment magnitude levels are integers in `[0, 10]`; they are scaled ×3
/// to map into the `[0, 30]` range expected by [`apply_aug_op`].
///
/// # Errors
/// - [`SslError::EmptyInput`] if any dimension is zero.
/// - [`SslError::DimensionMismatch`] if `pixels.len() != C·H·W`.
/// - [`SslError::InvalidParameter`] if policy has no sub-policies.
pub fn auto_augment(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    config: &AutoAugmentConfig,
    rng: &mut LcgRng,
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
    if !(config.fill_value.is_finite() && (0.0..=1.0).contains(&config.fill_value)) {
        return Err(SslError::InvalidParameter {
            name: "fill_value".into(),
            reason: format!("must be in [0, 1] and finite, got {}", config.fill_value),
        });
    }

    let sub_policies: Vec<SubPolicy> = match &config.policy {
        AutoAugPolicy::ImageNet => imagenet_sub_policies(),
        AutoAugPolicy::Cifar10 => cifar10_sub_policies(),
        AutoAugPolicy::Custom(v) => v.clone(),
    };

    if sub_policies.is_empty() {
        return Err(SslError::InvalidParameter {
            name: "policy".into(),
            reason: "policy contains no sub-policies".into(),
        });
    }

    // Sample one sub-policy.
    let sp_idx = rng.next_usize(sub_policies.len());
    let ((op1, prob1, mag_level1), (op2, prob2, mag_level2)) = &sub_policies[sp_idx];

    // Scale magnitude level [0, 10] → [0, 30].
    let mag1 = (*mag_level1 as f32 * 3.0).clamp(0.0, 30.0);
    let mag2 = (*mag_level2 as f32 * 3.0).clamp(0.0, 30.0);

    let mut current = pixels.to_vec();

    if rng.next_f32() < *prob1 {
        current = apply_aug_op(
            &current,
            channels,
            height,
            width,
            op1,
            mag1,
            config.fill_value,
        )?;
    }
    if rng.next_f32() < *prob2 {
        current = apply_aug_op(
            &current,
            channels,
            height,
            width,
            op2,
            mag2,
            config.fill_value,
        )?;
    }
    Ok(current)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Create a deterministic gradient CHW image.
    fn gradient_image(channels: usize, height: usize, width: usize) -> Vec<f32> {
        let n = channels * height * width;
        (0..n)
            .map(|i| {
                let v = (i as f32) / (n as f32);
                v.clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Assert all pixels in `[0, 1]`.
    fn assert_unit_range(pixels: &[f32], label: &str) {
        for (i, &v) in pixels.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "{label}: pixel[{i}] = {v} out of [0, 1]"
            );
        }
    }

    // ── Test 1: Output shape always matches input ──────────────────────────────

    #[test]
    fn output_shape_equals_input_for_all_ops() {
        let (c, h, w) = (3, 16, 16);
        let img = gradient_image(c, h, w);
        let expected_len = c * h * w;

        for op in all_aug_ops() {
            let out = apply_aug_op(&img, c, h, w, &op, 15.0, 0.5).unwrap();
            assert_eq!(out.len(), expected_len, "shape mismatch for op {:?}", op);
        }
    }

    // ── Test 2: All pixels in [0, 1] after any operation ─────────────────────

    #[test]
    fn all_pixels_in_unit_range_for_all_ops() {
        let (c, h, w) = (3, 16, 16);
        let img = gradient_image(c, h, w);

        for op in all_aug_ops() {
            let out = apply_aug_op(&img, c, h, w, &op, 20.0, 0.5).unwrap();
            assert_unit_range(&out, &format!("{op:?}"));
        }
    }

    // ── Test 3: Identity op returns exact copy ────────────────────────────────

    #[test]
    fn identity_op_returns_exact_copy() {
        let (c, h, w) = (3, 8, 8);
        let img = gradient_image(c, h, w);
        let out = apply_aug_op(&img, c, h, w, &AugOp::Identity, 15.0, 0.5).unwrap();
        assert_eq!(out, img, "Identity must return exact copy");
    }

    // ── Test 4: AutoContrast stretches to [0, 1] per channel ─────────────────

    #[test]
    fn auto_contrast_stretches_to_unit() {
        // Create image with known range per channel.
        let (c, h, w) = (3, 4, 4);
        let plane = h * w;
        let mut img = vec![0.0_f32; c * plane];
        // Channel 0: range [0.2, 0.8]
        for v in img[0..plane].iter_mut() {
            *v = 0.5;
        }
        img[0] = 0.2;
        img[plane - 1] = 0.8;
        // Channel 1: range [0.1, 0.9]
        for v in img[plane..2 * plane].iter_mut() {
            *v = 0.5;
        }
        img[plane] = 0.1;
        img[2 * plane - 1] = 0.9;
        // Channel 2: constant → should be left alone.
        for v in img[2 * plane..].iter_mut() {
            *v = 0.3;
        }

        let out = apply_aug_op(&img, c, h, w, &AugOp::AutoContrast, 0.0, 0.5).unwrap();
        // Channel 0: min should become 0, max should become 1.
        let ch0_min = out[..plane].iter().cloned().fold(f32::INFINITY, f32::min);
        let ch0_max = out[..plane]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(ch0_min.abs() < 1e-5, "ch0 min = {ch0_min}");
        assert!((ch0_max - 1.0).abs() < 1e-5, "ch0 max = {ch0_max}");
        // Channel 2: constant → should stay ≈ 0.3.
        for &v in &out[2 * plane..] {
            assert!((v - 0.3).abs() < 1e-5, "constant channel changed: {v}");
        }
    }

    // ── Test 5: Equalize outputs in [0, 1] ────────────────────────────────────

    #[test]
    fn equalize_output_in_unit_range() {
        let (c, h, w) = (1, 32, 32);
        let img = gradient_image(c, h, w);
        let out = apply_aug_op(&img, c, h, w, &AugOp::Equalize, 0.0, 0.5).unwrap();
        assert_unit_range(&out, "Equalize");
        assert_eq!(out.len(), c * h * w);
    }

    // ── Test 6: Rotate by 0° returns original ────────────────────────────────

    #[test]
    fn rotate_zero_degrees_approx_identity() {
        let (c, h, w) = (1, 8, 8);
        let img = gradient_image(c, h, w);
        // magnitude = 0 → angle = 0°.
        let out = apply_aug_op(&img, c, h, w, &AugOp::Rotate, 0.0, 0.5).unwrap();
        for (i, (&a, &b)) in img.iter().zip(out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "rotate(0°): pixel[{i}]: input={a} output={b}"
            );
        }
    }

    // ── Test 7: Solarize with threshold=1.0 leaves all pixels unchanged ───────

    #[test]
    fn solarize_threshold_one_unchanged() {
        // magnitude=0 → threshold = 1 - 0 = 1.0.
        // No pixel in [0,1] is ≥ 1.0 (strictly), so nothing flips.
        let (c, h, w) = (3, 8, 8);
        let img = gradient_image(c, h, w);
        let out = apply_aug_op(&img, c, h, w, &AugOp::Solarize, 0.0, 0.5).unwrap();
        // Pixels < 1.0 are unchanged; pixel at exactly 1.0 (if any) gets flipped to 0.
        for (i, (&a, &b)) in img.iter().zip(out.iter()).enumerate() {
            if a < 1.0 {
                assert!(
                    (a - b).abs() < 1e-6,
                    "solarize(threshold=1): pixel[{i}] changed: {a}→{b}"
                );
            }
        }
    }

    // ── Test 8: RandAugment with N=0 returns unchanged image ─────────────────

    #[test]
    fn rand_augment_zero_ops_unchanged() {
        let (c, h, w) = (3, 8, 8);
        let img = gradient_image(c, h, w);
        let config = RandAugmentConfig {
            n_ops: 0,
            magnitude: 9.0,
            fill_value: 0.5,
            ops: all_aug_ops(),
        };
        let mut rng = LcgRng::new(42);
        let out = rand_augment(&img, c, h, w, &config, &mut rng).unwrap();
        assert_eq!(out, img, "n_ops=0 must return exact input copy");
    }

    // ── Test 9: RandAugment applies exactly N ops (implicit via shape) ────────

    #[test]
    fn rand_augment_output_valid_shape_and_range() {
        let (c, h, w) = (3, 16, 16);
        let img = gradient_image(c, h, w);
        let config = RandAugmentConfig {
            n_ops: 3,
            magnitude: 15.0,
            fill_value: 0.5,
            ops: all_aug_ops(),
        };
        let mut rng = LcgRng::new(7);
        let out = rand_augment(&img, c, h, w, &config, &mut rng).unwrap();
        assert_eq!(out.len(), c * h * w);
        assert_unit_range(&out, "RandAugment(N=3)");
    }

    // ── Test 10: AutoAugment ImageNet policy: output is finite and valid ──────

    #[test]
    fn auto_augment_imagenet_output_finite_and_valid() {
        let (c, h, w) = (3, 16, 16);
        let img = gradient_image(c, h, w);
        let config = AutoAugmentConfig {
            policy: AutoAugPolicy::ImageNet,
            fill_value: 0.5,
        };
        let mut rng = LcgRng::new(13);
        let out = auto_augment(&img, c, h, w, &config, &mut rng).unwrap();
        assert_eq!(out.len(), c * h * w);
        assert_unit_range(&out, "AutoAugment(ImageNet)");
        for &v in &out {
            assert!(v.is_finite(), "non-finite pixel in AutoAugment output");
        }
    }

    // ── Test 11: Different seeds → different augmentations ───────────────────

    #[test]
    fn different_seeds_produce_different_outputs() {
        let (c, h, w) = (3, 16, 16);
        let img = gradient_image(c, h, w);
        let config = RandAugmentConfig::default();

        let mut rng_a = LcgRng::new(1);
        let mut rng_b = LcgRng::new(999);
        let out_a = rand_augment(&img, c, h, w, &config, &mut rng_a).unwrap();
        let out_b = rand_augment(&img, c, h, w, &config, &mut rng_b).unwrap();

        // It is overwhelmingly unlikely that two different random seeds produce
        // identical augmented outputs; if they do, the test catches a RNG bug.
        let identical = out_a
            .iter()
            .zip(out_b.iter())
            .all(|(a, b)| (a - b).abs() < 1e-8);
        assert!(!identical, "different seeds must produce different outputs");
    }

    // ── Test 12: Same seed → same output (deterministic) ─────────────────────

    #[test]
    fn same_seed_produces_same_output() {
        let (c, h, w) = (3, 16, 16);
        let img = gradient_image(c, h, w);
        let config = RandAugmentConfig::default();

        let mut rng_a = LcgRng::new(42);
        let mut rng_b = LcgRng::new(42);
        let out_a = rand_augment(&img, c, h, w, &config, &mut rng_a).unwrap();
        let out_b = rand_augment(&img, c, h, w, &config, &mut rng_b).unwrap();
        assert_eq!(out_a, out_b, "same seed must produce identical output");
    }

    // ── Test 13: Brightness at magnitude=0 dims image significantly ──────────

    #[test]
    fn brightness_low_magnitude_dims_image() {
        let (c, h, w) = (3, 8, 8);
        let img = vec![0.8_f32; c * h * w];
        // magnitude=0 → strength = 0*0.9 + 0.1 = 0.1.
        let out = apply_aug_op(&img, c, h, w, &AugOp::Brightness, 0.0, 0.5).unwrap();
        let mean_out: f32 = out.iter().sum::<f32>() / out.len() as f32;
        // 0.8 * 0.1 = 0.08; allow tolerance.
        assert!(
            mean_out < 0.2,
            "Brightness(mag=0) should produce near-black image, got mean={mean_out}"
        );
    }

    // ── Test 14: apply_aug_op valid for all 14 ops without panic ─────────────

    #[test]
    fn all_14_ops_run_without_error() {
        let (c, h, w) = (3, 12, 12);
        let img = gradient_image(c, h, w);
        for mag in [0.0_f32, 9.0, 15.0, 30.0] {
            for op in all_aug_ops() {
                let result = apply_aug_op(&img, c, h, w, &op, mag, 0.5);
                assert!(
                    result.is_ok(),
                    "op {:?} at magnitude={mag} returned error: {:?}",
                    op,
                    result
                );
                assert_unit_range(&result.unwrap(), &format!("{op:?}@{mag}"));
            }
        }
    }

    // ── Test 15: AutoAugment CIFAR-10 policy ─────────────────────────────────

    #[test]
    fn auto_augment_cifar10_output_valid() {
        let (c, h, w) = (3, 32, 32);
        let img = gradient_image(c, h, w);
        let config = AutoAugmentConfig {
            policy: AutoAugPolicy::Cifar10,
            fill_value: 0.5,
        };
        let mut rng = LcgRng::new(77);
        let out = auto_augment(&img, c, h, w, &config, &mut rng).unwrap();
        assert_eq!(out.len(), c * h * w);
        assert_unit_range(&out, "AutoAugment(Cifar10)");
    }

    // ── Test 16: Custom AutoAugment policy ────────────────────────────────────

    #[test]
    fn auto_augment_custom_policy_identity_always() {
        // A custom policy with a single sub-policy: Identity at prob=1.
        let (c, h, w) = (3, 8, 8);
        let img = gradient_image(c, h, w);
        let config = AutoAugmentConfig {
            policy: AutoAugPolicy::Custom(vec![(
                (AugOp::Identity, 1.0, 0),
                (AugOp::Identity, 1.0, 0),
            )]),
            fill_value: 0.5,
        };
        let mut rng = LcgRng::new(1);
        let out = auto_augment(&img, c, h, w, &config, &mut rng).unwrap();
        assert_eq!(
            out, img,
            "custom Identity × Identity should return exact copy"
        );
    }

    // ── Test 17: Error on empty input ────────────────────────────────────────

    #[test]
    fn error_on_empty_input() {
        let result = apply_aug_op(&[], 0, 8, 8, &AugOp::Identity, 0.0, 0.5);
        assert!(matches!(result, Err(SslError::EmptyInput)));
    }

    // ── Test 18: Error on dimension mismatch ─────────────────────────────────

    #[test]
    fn error_on_dimension_mismatch() {
        let img = vec![0.5_f32; 10]; // wrong for 3×4×4=48
        let result = apply_aug_op(&img, 3, 4, 4, &AugOp::Identity, 0.0, 0.5);
        assert!(matches!(result, Err(SslError::DimensionMismatch { .. })));
    }

    // ── Test 19: Posterize at full magnitude quantizes heavily ────────────────

    #[test]
    fn posterize_full_magnitude_reduces_unique_values() {
        let (c, h, w) = (1, 16, 16);
        let img = gradient_image(c, h, w);
        // magnitude=30 → k = 8 - floor(1.0 * 4) = 4 bits.
        let out = apply_aug_op(&img, c, h, w, &AugOp::Posterize, 30.0, 0.5).unwrap();
        // With 4-bit posterization we expect at most 16 distinct values.
        let mut values: Vec<u32> = out.iter().map(|&v| (v * 255.0).round() as u32).collect();
        values.sort_unstable();
        values.dedup();
        assert!(
            values.len() <= 16,
            "expected ≤16 distinct values after 4-bit posterize, got {}",
            values.len()
        );
    }

    // ── Test 20: Sharpness at magnitude=1 returns original ───────────────────

    #[test]
    fn sharpness_full_magnitude_is_original() {
        let (c, h, w) = (3, 8, 8);
        let img = gradient_image(c, h, w);
        // alpha = magnitude/30 = 1.0 → pure original, no blur blended in.
        let out = apply_aug_op(&img, c, h, w, &AugOp::Sharpness, 30.0, 0.5).unwrap();
        for (i, (&a, &b)) in img.iter().zip(out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Sharpness(1.0): pixel[{i}] input={a} output={b}"
            );
        }
    }
}

//! Image-quality / reconstruction metrics: PSNR, MSE, SSIM, MS-SSIM.
//!
//! These metrics quantify the fidelity of a reconstructed (or compressed,
//! denoised, super-resolved) image relative to a reference. They are widely
//! used both as evaluation criteria and — in their differentiable SSIM /
//! MS-SSIM form — as training objectives for image-to-image models.
//!
//! All images are flat `[channels × h × w]` row-major `f32` buffers.
//!
//! * [`mse`] — mean-squared error `(1/N) Σ (a−b)²`.
//! * [`psnr`] — peak signal-to-noise ratio `10·log10(max_val² / mse)` in dB.
//! * [`ssim`] — structural similarity (Wang et al. 2004) using a single global
//!   (or per-window, sliding) Gaussian-free uniform window, averaged over the
//!   image and channels.
//! * [`ms_ssim`] — multi-scale SSIM (Wang et al. 2003) combining the contrast /
//!   structure terms across `levels` dyadic down-sampling stages with the
//!   canonical exponent weights.
//!
//! The SSIM window is a flat (box) window of side `window` rather than a
//! Gaussian; this keeps the implementation dependency-free while preserving the
//! `[0, 1]` boundedness and the perfect-match `SSIM = 1` property.

use crate::error::{VisionError, VisionResult};

/// Default `C1 = (K1·L)²` stabiliser with `K1 = 0.01`, `L = 1.0`.
pub const SSIM_C1: f32 = 0.01 * 0.01;
/// Default `C2 = (K2·L)²` stabiliser with `K2 = 0.03`, `L = 1.0`.
pub const SSIM_C2: f32 = 0.03 * 0.03;

/// Canonical MS-SSIM per-scale weights (Wang et al. 2003, Table 1, 5 scales).
pub const MS_SSIM_WEIGHTS: [f32; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

// ─── Validation ───────────────────────────────────────────────────────────────

#[inline]
fn validate_pair(a: &[f32], b: &[f32], channels: usize, h: usize, w: usize) -> VisionResult<()> {
    if channels == 0 || h == 0 || w == 0 {
        return Err(VisionError::InvalidImageSize {
            height: h,
            width: w,
            channels,
        });
    }
    let expected = channels * h * w;
    if a.len() != expected {
        return Err(VisionError::DimensionMismatch {
            expected,
            got: a.len(),
        });
    }
    if b.len() != expected {
        return Err(VisionError::ShapeMismatch {
            lhs: vec![a.len()],
            rhs: vec![b.len()],
        });
    }
    Ok(())
}

// ─── MSE / PSNR ────────────────────────────────────────────────────────────────

/// Mean-squared error between two equally-shaped images.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] / [`VisionError::DimensionMismatch`]
/// / [`VisionError::ShapeMismatch`] on a shape problem.
pub fn mse(a: &[f32], b: &[f32], channels: usize, h: usize, w: usize) -> VisionResult<f32> {
    validate_pair(a, b, channels, h, w)?;
    let n = a.len() as f32;
    let acc: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum();
    Ok(acc / n)
}

/// Peak signal-to-noise ratio in decibels: `10·log10(max_val² / MSE)`.
///
/// When the two images are identical the MSE is zero and PSNR is mathematically
/// infinite; this function returns [`f32::INFINITY`] in that case.
///
/// # Errors
/// Propagates shape errors from [`mse`]; returns
/// [`VisionError::NonPositiveTemperature`] reuse is avoided — instead a
/// non-positive `max_val` yields [`VisionError::Internal`].
pub fn psnr(
    a: &[f32],
    b: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    max_val: f32,
) -> VisionResult<f32> {
    if !max_val.is_finite() || max_val <= 0.0 {
        return Err(VisionError::Internal(format!(
            "psnr max_val must be positive and finite, got {max_val}"
        )));
    }
    let e = mse(a, b, channels, h, w)?;
    if e <= 0.0 {
        return Ok(f32::INFINITY);
    }
    Ok(10.0 * (max_val * max_val / e).log10())
}

// ─── SSIM ──────────────────────────────────────────────────────────────────────

/// Structural similarity index over a sliding box window.
///
/// For every window position the local means `μ_a, μ_b`, variances
/// `σ_a², σ_b²` and covariance `σ_ab` are formed, and the local SSIM map value
///
/// ```text
/// (2 μ_a μ_b + C1)(2 σ_ab + C2)
/// ─────────────────────────────────────────
/// (μ_a² + μ_b² + C1)(σ_a² + σ_b² + C2)
/// ```
///
/// is accumulated; the returned scalar is the mean over all window positions
/// and channels (the "mean SSIM", MSSIM).
///
/// `window` is the side length of the square box window; it is clamped so it is
/// never larger than the smaller spatial dimension and never smaller than 1.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] / [`VisionError::DimensionMismatch`]
/// / [`VisionError::ShapeMismatch`] on a shape problem.
pub fn ssim(
    a: &[f32],
    b: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    window: usize,
    c1: f32,
    c2: f32,
) -> VisionResult<f32> {
    validate_pair(a, b, channels, h, w)?;
    let win = window.clamp(1, h.min(w));
    let win_area = (win * win) as f32;

    let mut acc = 0.0_f32;
    let mut count = 0_usize;

    for c in 0..channels {
        let base = c * h * w;
        // Slide the window so it stays fully inside the image.
        for top in 0..=(h - win) {
            for left in 0..=(w - win) {
                let mut sum_a = 0.0_f32;
                let mut sum_b = 0.0_f32;
                let mut sum_aa = 0.0_f32;
                let mut sum_bb = 0.0_f32;
                let mut sum_ab = 0.0_f32;
                for dy in 0..win {
                    let row = base + (top + dy) * w + left;
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
                let mu_a = sum_a / win_area;
                let mu_b = sum_b / win_area;
                // Population variance / covariance over the window.
                let var_a = (sum_aa / win_area - mu_a * mu_a).max(0.0);
                let var_b = (sum_bb / win_area - mu_b * mu_b).max(0.0);
                let cov_ab = sum_ab / win_area - mu_a * mu_b;

                let numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2);
                let denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2);
                acc += numerator / denominator;
                count += 1;
            }
        }
    }

    if count == 0 {
        return Err(VisionError::Internal(
            "ssim produced no windows (window too large)".to_string(),
        ));
    }
    Ok(acc / count as f32)
}

/// Convenience wrapper using the default stabilisers [`SSIM_C1`] / [`SSIM_C2`]
/// and a window of `7`.
///
/// # Errors
/// Propagates from [`ssim`].
pub fn ssim_default(
    a: &[f32],
    b: &[f32],
    channels: usize,
    h: usize,
    w: usize,
) -> VisionResult<f32> {
    ssim(a, b, channels, h, w, 7, SSIM_C1, SSIM_C2)
}

// ─── MS-SSIM ─────────────────────────────────────────────────────────────────

/// Box-downsample a CHW image by a factor of 2 (2×2 average pooling, stride 2).
///
/// Returns `(out, new_h, new_w)`; the output drops odd trailing rows/columns.
fn downsample_2x(img: &[f32], channels: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let nh = h / 2;
    let nw = w / 2;
    let mut out = vec![0.0_f32; channels * nh * nw];
    for c in 0..channels {
        let src_base = c * h * w;
        let dst_base = c * nh * nw;
        for oy in 0..nh {
            for ox in 0..nw {
                let r0 = src_base + (2 * oy) * w + 2 * ox;
                let r1 = src_base + (2 * oy + 1) * w + 2 * ox;
                let s = img[r0] + img[r0 + 1] + img[r1] + img[r1 + 1];
                out[dst_base + oy * nw + ox] = s * 0.25;
            }
        }
    }
    (out, nh, nw)
}

/// Compute the per-window mean of the contrast-structure term
/// `(2 σ_ab + C2) / (σ_a² + σ_b² + C2)` and the full SSIM (luminance included)
/// over the image, returning `(mean_full_ssim, mean_cs)`.
fn ssim_cs(
    a: &[f32],
    b: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    window: usize,
    c1: f32,
    c2: f32,
) -> Option<(f32, f32)> {
    // MS-SSIM uses a *fixed* window across scales: if a scale becomes smaller
    // than the window the multi-scale chain cannot continue (caller errors).
    let win = window.max(1);
    if h < win || w < win {
        return None;
    }
    let win_area = (win * win) as f32;
    let mut acc_full = 0.0_f32;
    let mut acc_cs = 0.0_f32;
    let mut count = 0_usize;
    for c in 0..channels {
        let base = c * h * w;
        for top in 0..=(h - win) {
            for left in 0..=(w - win) {
                let mut sum_a = 0.0_f32;
                let mut sum_b = 0.0_f32;
                let mut sum_aa = 0.0_f32;
                let mut sum_bb = 0.0_f32;
                let mut sum_ab = 0.0_f32;
                for dy in 0..win {
                    let row = base + (top + dy) * w + left;
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
                let mu_a = sum_a / win_area;
                let mu_b = sum_b / win_area;
                let var_a = (sum_aa / win_area - mu_a * mu_a).max(0.0);
                let var_b = (sum_bb / win_area - mu_b * mu_b).max(0.0);
                let cov_ab = sum_ab / win_area - mu_a * mu_b;

                let cs = (2.0 * cov_ab + c2) / (var_a + var_b + c2);
                let luminance = (2.0 * mu_a * mu_b + c1) / (mu_a * mu_a + mu_b * mu_b + c1);
                acc_cs += cs;
                acc_full += luminance * cs;
                count += 1;
            }
        }
    }
    if count == 0 {
        return None;
    }
    Some((acc_full / count as f32, acc_cs / count as f32))
}

/// Multi-scale SSIM (Wang et al. 2003).
///
/// At each of the first `levels−1` scales only the contrast-structure (cs)
/// term is taken; at the coarsest scale the full SSIM (including luminance) is
/// used. The scale outputs are combined as
///
/// ```text
/// MS-SSIM = Π_{j<L-1} cs_j^{w_j} · ssim_{L-1}^{w_{L-1}}
/// ```
///
/// with `w_j` the canonical [`MS_SSIM_WEIGHTS`] (renormalised to the chosen
/// `levels`). Negative cs values are clamped to a small positive floor so the
/// weighted geometric mean stays well-defined.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] for empty inputs, and
/// [`VisionError::Internal`] if the image becomes smaller than the window
/// before all levels are processed or if `levels` is `0` / exceeds `5`.
pub fn ms_ssim(
    a: &[f32],
    b: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    levels: usize,
    window: usize,
) -> VisionResult<f32> {
    validate_pair(a, b, channels, h, w)?;
    if levels == 0 || levels > MS_SSIM_WEIGHTS.len() {
        return Err(VisionError::Internal(format!(
            "ms_ssim levels must be in 1..={}, got {levels}",
            MS_SSIM_WEIGHTS.len()
        )));
    }
    // Renormalise the first `levels` canonical weights so they sum to 1.
    let weight_sum: f32 = MS_SSIM_WEIGHTS[..levels].iter().sum();
    let weights: Vec<f32> = MS_SSIM_WEIGHTS[..levels]
        .iter()
        .map(|&x| x / weight_sum)
        .collect();

    let win = window.clamp(1, 11);
    let floor = 1e-6_f32;

    let mut cur_a = a.to_vec();
    let mut cur_b = b.to_vec();
    let mut cur_h = h;
    let mut cur_w = w;

    let mut log_acc = 0.0_f32;
    for (level, &weight) in weights.iter().enumerate() {
        let (full, cs) = ssim_cs(
            &cur_a, &cur_b, channels, cur_h, cur_w, win, SSIM_C1, SSIM_C2,
        )
        .ok_or_else(|| {
            VisionError::Internal(format!(
                "ms_ssim: image {cur_h}×{cur_w} smaller than window {win} at level {level}"
            ))
        })?;
        let term = if level == levels - 1 {
            full.max(floor)
        } else {
            cs.max(floor)
        };
        log_acc += weight * term.ln();

        if level + 1 < levels {
            let (na, nh, nw) = downsample_2x(&cur_a, channels, cur_h, cur_w);
            let (nb, _, _) = downsample_2x(&cur_b, channels, cur_h, cur_w);
            cur_a = na;
            cur_b = nb;
            cur_h = nh;
            cur_w = nw;
        }
    }
    Ok(log_acc.exp())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn ramp(channels: usize, h: usize, w: usize) -> Vec<f32> {
        let n = channels * h * w;
        (0..n).map(|i| (i as f32) / (n as f32)).collect()
    }

    #[test]
    fn mse_identical_is_zero() {
        let img = ramp(3, 8, 8);
        let e = mse(&img, &img, 3, 8, 8).expect("ok");
        assert!(
            e.abs() < 1e-9,
            "mse of identical images should be 0, got {e}"
        );
    }

    #[test]
    fn mse_constant_offset() {
        let a = vec![0.2_f32; 16];
        let b = vec![0.5_f32; 16];
        let e = mse(&a, &b, 1, 4, 4).expect("ok");
        // (0.3)^2 = 0.09
        assert!((e - 0.09).abs() < 1e-6, "mse={e}");
    }

    #[test]
    fn mse_shape_mismatch_errors() {
        let a = vec![0.0_f32; 16];
        let b = vec![0.0_f32; 8];
        assert!(matches!(
            mse(&a, &b, 1, 4, 4),
            Err(VisionError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn psnr_identical_is_infinite() {
        let img = ramp(1, 8, 8);
        let p = psnr(&img, &img, 1, 8, 8, 1.0).expect("ok");
        assert!(
            p.is_infinite() && p > 0.0,
            "psnr of identical = +inf, got {p}"
        );
    }

    #[test]
    fn psnr_known_value() {
        // MSE = 0.01, max=1 → PSNR = 10*log10(1/0.01) = 20 dB.
        let a = vec![0.0_f32; 16];
        let b = vec![0.1_f32; 16];
        let p = psnr(&a, &b, 1, 4, 4, 1.0).expect("ok");
        assert!((p - 20.0).abs() < 1e-4, "psnr={p}, expected 20 dB");
    }

    #[test]
    fn psnr_decreases_with_error() {
        let a = ramp(1, 8, 8);
        let mut b_small = a.clone();
        let mut b_large = a.clone();
        for v in &mut b_small {
            *v += 0.01;
        }
        for v in &mut b_large {
            *v += 0.1;
        }
        let p_small = psnr(&a, &b_small, 1, 8, 8, 1.0).expect("ok");
        let p_large = psnr(&a, &b_large, 1, 8, 8, 1.0).expect("ok");
        assert!(
            p_small > p_large,
            "smaller error must yield higher PSNR: {p_small} vs {p_large}"
        );
    }

    #[test]
    fn psnr_nonpositive_max_errors() {
        let a = vec![0.0_f32; 16];
        let b = vec![0.1_f32; 16];
        assert!(matches!(
            psnr(&a, &b, 1, 4, 4, 0.0),
            Err(VisionError::Internal(_))
        ));
    }

    #[test]
    fn ssim_identical_is_one() {
        let img = ramp(1, 16, 16);
        let s = ssim(&img, &img, 1, 16, 16, 7, SSIM_C1, SSIM_C2).expect("ok");
        assert!((s - 1.0).abs() < 1e-4, "ssim of identical = 1, got {s}");
    }

    #[test]
    fn ssim_default_identical_is_one() {
        let img = ramp(3, 16, 16);
        let s = ssim_default(&img, &img, 3, 16, 16).expect("ok");
        assert!(
            (s - 1.0).abs() < 1e-4,
            "ssim_default identical = 1, got {s}"
        );
    }

    #[test]
    fn ssim_in_unit_range() {
        let mut rng = LcgRng::new(1);
        let mut a = vec![0.0_f32; 16 * 16];
        let mut b = vec![0.0_f32; 16 * 16];
        for v in &mut a {
            *v = rng.next_f32();
        }
        for v in &mut b {
            *v = rng.next_f32();
        }
        let s = ssim(&a, &b, 1, 16, 16, 7, SSIM_C1, SSIM_C2).expect("ok");
        assert!((-1.0..=1.0001).contains(&s), "ssim out of range: {s}");
    }

    #[test]
    fn ssim_lower_for_noisier() {
        let img = ramp(1, 24, 24);
        let mut rng = LcgRng::new(5);
        let mut noisy_small = img.clone();
        let mut noisy_large = img.clone();
        for v in &mut noisy_small {
            *v += 0.02 * (rng.next_f32() - 0.5);
        }
        for v in &mut noisy_large {
            *v += 0.3 * (rng.next_f32() - 0.5);
        }
        let s_small = ssim(&img, &noisy_small, 1, 24, 24, 7, SSIM_C1, SSIM_C2).expect("ok");
        let s_large = ssim(&img, &noisy_large, 1, 24, 24, 7, SSIM_C1, SSIM_C2).expect("ok");
        assert!(
            s_small > s_large,
            "less noise → higher SSIM: {s_small} vs {s_large}"
        );
    }

    #[test]
    fn ssim_symmetric() {
        let mut rng = LcgRng::new(9);
        let mut a = vec![0.0_f32; 12 * 12];
        let mut b = vec![0.0_f32; 12 * 12];
        for v in &mut a {
            *v = rng.next_f32();
        }
        for v in &mut b {
            *v = rng.next_f32();
        }
        let s_ab = ssim(&a, &b, 1, 12, 12, 5, SSIM_C1, SSIM_C2).expect("ok");
        let s_ba = ssim(&b, &a, 1, 12, 12, 5, SSIM_C1, SSIM_C2).expect("ok");
        assert!((s_ab - s_ba).abs() < 1e-5, "ssim must be symmetric");
    }

    #[test]
    fn ssim_window_clamped() {
        // window larger than the image is clamped, not an error.
        let img = ramp(1, 4, 4);
        let s = ssim(&img, &img, 1, 4, 4, 16, SSIM_C1, SSIM_C2).expect("ok");
        assert!((s - 1.0).abs() < 1e-4);
    }

    #[test]
    fn ssim_empty_errors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(matches!(
            ssim(&a, &b, 0, 0, 0, 7, SSIM_C1, SSIM_C2),
            Err(VisionError::InvalidImageSize { .. })
        ));
    }

    #[test]
    fn ms_ssim_identical_is_one() {
        let img = ramp(1, 32, 32);
        let s = ms_ssim(&img, &img, 1, 32, 32, 3, 7).expect("ok");
        assert!((s - 1.0).abs() < 1e-3, "ms_ssim of identical = 1, got {s}");
    }

    #[test]
    fn ms_ssim_lower_for_noisier() {
        let img = ramp(1, 32, 32);
        let mut rng = LcgRng::new(11);
        let mut small = img.clone();
        let mut large = img.clone();
        for v in &mut small {
            *v += 0.02 * (rng.next_f32() - 0.5);
        }
        for v in &mut large {
            *v += 0.4 * (rng.next_f32() - 0.5);
        }
        let s_small = ms_ssim(&img, &small, 1, 32, 32, 3, 7).expect("ok");
        let s_large = ms_ssim(&img, &large, 1, 32, 32, 3, 7).expect("ok");
        assert!(s_small > s_large, "{s_small} vs {s_large}");
    }

    #[test]
    fn ms_ssim_invalid_levels_errors() {
        let img = ramp(1, 32, 32);
        assert!(matches!(
            ms_ssim(&img, &img, 1, 32, 32, 0, 7),
            Err(VisionError::Internal(_))
        ));
        assert!(matches!(
            ms_ssim(&img, &img, 1, 32, 32, 99, 7),
            Err(VisionError::Internal(_))
        ));
    }

    #[test]
    fn ms_ssim_too_small_errors() {
        // 8×8 image cannot survive 3 levels of /2 down-sampling with window 7.
        let img = ramp(1, 8, 8);
        assert!(matches!(
            ms_ssim(&img, &img, 1, 8, 8, 4, 7),
            Err(VisionError::Internal(_))
        ));
    }

    #[test]
    fn downsample_halves_dims() {
        let img = ramp(2, 8, 8);
        let (out, nh, nw) = downsample_2x(&img, 2, 8, 8);
        assert_eq!((nh, nw), (4, 4));
        assert_eq!(out.len(), 2 * 4 * 4);
    }

    #[test]
    fn downsample_constant_preserved() {
        let img = vec![0.5_f32; 16];
        let (out, _, _) = downsample_2x(&img, 1, 4, 4);
        assert!(out.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }
}

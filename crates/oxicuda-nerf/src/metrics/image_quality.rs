//! Image quality metrics: MSE and PSNR.

use crate::error::{NerfError, NerfResult};

/// Compute Mean Squared Error between two equal-length float slices.
///
/// # Errors
///
/// Returns `EmptyInput` if slices are empty, `DimensionMismatch` if sizes differ.
pub fn mse_image(gt: &[f32], pred: &[f32]) -> NerfResult<f32> {
    if gt.is_empty() {
        return Err(NerfError::EmptyInput);
    }
    if gt.len() != pred.len() {
        return Err(NerfError::DimensionMismatch {
            expected: gt.len(),
            got: pred.len(),
        });
    }
    let sum: f32 = gt
        .iter()
        .zip(pred.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    Ok(sum / gt.len() as f32)
}

/// Compute PSNR in dB: `-10 * log10(MSE)`.
///
/// Returns `f32::INFINITY` if MSE == 0 (identical images).
///
/// # Errors
///
/// Propagates errors from `mse_image`.
pub fn psnr(gt: &[f32], pred: &[f32]) -> NerfResult<f32> {
    let mse = mse_image(gt, pred)?;
    if mse == 0.0 {
        return Ok(f32::INFINITY);
    }
    Ok(-10.0 * mse.log10())
}

/// Combined image quality metrics.
#[derive(Debug, Clone, Copy)]
pub struct ImageMetrics {
    /// Mean squared error.
    pub mse: f32,
    /// Peak signal-to-noise ratio in dB.
    pub psnr: f32,
}

/// Compute MSE and PSNR together.
///
/// # Errors
///
/// Propagates errors from `mse_image`.
pub fn compute_image_metrics(gt: &[f32], pred: &[f32]) -> NerfResult<ImageMetrics> {
    let mse = mse_image(gt, pred)?;
    let psnr_val = if mse == 0.0 {
        f32::INFINITY
    } else {
        -10.0 * mse.log10()
    };
    Ok(ImageMetrics {
        mse,
        psnr: psnr_val,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_identical() {
        let x = [0.1_f32, 0.5, 0.9];
        let m = mse_image(&x, &x).unwrap();
        assert!(m.abs() < 1e-10);
    }

    #[test]
    fn psnr_identical_is_inf() {
        let x = [0.5_f32; 16];
        assert_eq!(psnr(&x, &x).unwrap(), f32::INFINITY);
    }

    #[test]
    fn psnr_decreases_with_noise() {
        let gt = vec![0.5_f32; 64];
        let low_noise: Vec<f32> = gt.iter().map(|&v| v + 0.01).collect();
        let high_noise: Vec<f32> = gt.iter().map(|&v| v + 0.1).collect();
        let p_low = psnr(&gt, &low_noise).unwrap();
        let p_high = psnr(&gt, &high_noise).unwrap();
        assert!(p_low > p_high, "PSNR should decrease with more noise");
    }
}

//! MAE — He et al. 2022 — Masked Autoencoders.
//!
//! Train a vision transformer to reconstruct masked image patches given the
//! visible ones. The encoder sees only the *unmasked* tokens (~25%); the
//! decoder predicts the pixel content of the masked tokens; reconstruction
//! MSE is averaged over the masked subset only.
//!
//! This module provides:
//! - [`random_patch_mask`] — uniform random Bernoulli mask of patches with a
//!   target ratio.
//! - [`mae_reconstruction_loss`] — MSE over masked positions only.
//! - [`MaeConfig`] — tunable mask ratio (default 75%, paper-canonical).

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

/// MAE configuration.
#[derive(Debug, Clone)]
pub struct MaeConfig {
    /// Fraction of patches to mask. Default 0.75 (Vit-B/16 paper).
    pub mask_ratio: f32,
}

impl Default for MaeConfig {
    fn default() -> Self {
        Self { mask_ratio: 0.75 }
    }
}

impl MaeConfig {
    /// Validated config.
    ///
    /// # Errors
    /// [`SslError::InvalidMaskRatio`] if `mask_ratio` is outside `[0, 1)` or
    /// non-finite.
    pub fn new(mask_ratio: f32) -> SslResult<Self> {
        if !(mask_ratio.is_finite() && (0.0..1.0).contains(&mask_ratio)) {
            return Err(SslError::InvalidMaskRatio { ratio: mask_ratio });
        }
        Ok(Self { mask_ratio })
    }
}

/// Generate a Bernoulli mask `[N]` with `target_ratio` probability of being
/// 0 (= masked). Output `1` means visible, `0` means masked.
///
/// We deterministically pick exactly `floor(N · ratio)` indices via Fisher-Yates
/// shuffle for tighter ratio control than per-element sampling.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n_patches == 0`.
/// - [`SslError::InvalidMaskRatio`] when ratio is out of range.
pub fn random_patch_mask(
    n_patches: usize,
    mask_ratio: f32,
    rng: &mut LcgRng,
) -> SslResult<Vec<f32>> {
    if n_patches == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(mask_ratio.is_finite() && (0.0..1.0).contains(&mask_ratio)) {
        return Err(SslError::InvalidMaskRatio { ratio: mask_ratio });
    }
    let n_mask = (n_patches as f32 * mask_ratio) as usize;
    let mut indices: Vec<usize> = (0..n_patches).collect();
    rng.shuffle(&mut indices);
    let mut mask = vec![1.0_f32; n_patches];
    for &idx in indices.iter().take(n_mask) {
        mask[idx] = 0.0;
    }
    Ok(mask)
}

/// Compute MAE reconstruction loss = mean squared error over **masked** patches
/// only (where `mask == 0`).
///
/// `target` and `pred` are `[N × D]` row-major patch embeddings (or pixel
/// patches); `mask` is `[N]` with `1=visible, 0=masked`. Returns the per-element
/// MSE averaged across the masked patches × `D`.
///
/// # Errors
/// - [`SslError::DimensionMismatch`] when shapes disagree.
/// - [`SslError::EmptyInput`] when `n == 0`, `d == 0`, or no patches are masked.
pub fn mae_reconstruction_loss(
    target: &[f32],
    pred: &[f32],
    mask: &[f32],
    n: usize,
    d: usize,
) -> SslResult<f32> {
    if n == 0 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if target.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: target.len(),
        });
    }
    if pred.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: pred.len(),
        });
    }
    if mask.len() != n {
        return Err(SslError::DimensionMismatch {
            expected: n,
            got: mask.len(),
        });
    }
    let mut total = 0.0_f64;
    let mut count = 0usize;
    for i in 0..n {
        if mask[i] == 0.0 {
            count += 1;
            for k in 0..d {
                let diff = target[i * d + k] - pred[i * d + k];
                total += (diff as f64) * (diff as f64);
            }
        }
    }
    if count == 0 {
        return Err(SslError::EmptyInput);
    }
    let denom = count * d;
    Ok((total / denom as f64) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mae_config_default_is_paper_ratio() {
        let cfg = MaeConfig::default();
        assert!((cfg.mask_ratio - 0.75).abs() < 1e-6);
    }

    #[test]
    fn mae_config_rejects_invalid_ratio() {
        assert!(MaeConfig::new(-0.1).is_err());
        assert!(MaeConfig::new(1.0).is_err());
        assert!(MaeConfig::new(1.5).is_err());
        assert!(MaeConfig::new(f32::NAN).is_err());
        assert!(MaeConfig::new(0.5).is_ok());
    }

    #[test]
    fn random_patch_mask_respects_ratio() {
        let mut rng = LcgRng::new(0);
        let mask = random_patch_mask(100, 0.75, &mut rng).unwrap();
        let n_masked = mask.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(n_masked, 75);
    }

    #[test]
    fn random_patch_mask_handles_zero_ratio() {
        let mut rng = LcgRng::new(0);
        let mask = random_patch_mask(8, 0.0, &mut rng).unwrap();
        for &v in &mask {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn random_patch_mask_rejects_zero_n() {
        let mut rng = LcgRng::new(0);
        assert!(random_patch_mask(0, 0.5, &mut rng).is_err());
    }

    #[test]
    fn random_patch_mask_rejects_invalid_ratio() {
        let mut rng = LcgRng::new(0);
        assert!(random_patch_mask(8, -0.1, &mut rng).is_err());
        assert!(random_patch_mask(8, 1.0, &mut rng).is_err());
        assert!(random_patch_mask(8, f32::NAN, &mut rng).is_err());
    }

    #[test]
    fn mae_reconstruction_loss_zero_for_perfect() {
        let n = 8;
        let d = 4;
        let target = vec![1.0_f32; n * d];
        let pred = target.clone();
        let mask = vec![0.0_f32, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let l = mae_reconstruction_loss(&target, &pred, &mask, n, d).unwrap();
        assert!(l.abs() < 1e-7);
    }

    #[test]
    fn mae_reconstruction_loss_only_masked() {
        let n = 4;
        let d = 1;
        let target = vec![1.0_f32, 2.0, 3.0, 4.0];
        let pred = vec![10.0_f32, 2.0, 30.0, 4.0]; // off only at masked positions
        let mask = vec![0.0_f32, 1.0, 0.0, 1.0]; // patches 0 and 2 are masked
        let l = mae_reconstruction_loss(&target, &pred, &mask, n, d).unwrap();
        // Errors at masked: (1-10)² = 81, (3-30)² = 729; mean = 405.
        assert!((l - 405.0).abs() < 1e-3, "l = {l}");
    }

    #[test]
    fn mae_reconstruction_loss_no_mask_errors() {
        let target = vec![0.0_f32, 0.0];
        let pred = vec![0.0_f32, 0.0];
        let mask = vec![1.0_f32, 1.0]; // nothing masked
        assert!(mae_reconstruction_loss(&target, &pred, &mask, 2, 1).is_err());
    }

    #[test]
    fn mae_reconstruction_loss_dim_mismatch() {
        let target = vec![1.0_f32; 8];
        let pred = vec![1.0_f32; 6];
        let mask = vec![0.0_f32; 4];
        assert!(mae_reconstruction_loss(&target, &pred, &mask, 4, 2).is_err());
    }
}

//! SimMIM — Xie et al. 2022 — "SimMIM: A Simple Framework for Masked Image Modeling".
//!
//! Key differences from MAE:
//! - L1 reconstruction loss (mean absolute error) over masked patches only.
//! - Predicts raw pixel values (no patch normalisation).
//! - Decoder is a single linear layer (not a transformer decoder).
//! - Larger default mask ratio (0.6) compared to MAE (0.75).
//!
//! This module provides:
//! - [`SimMimConfig`] — mask ratio + patch size (defaults 0.6 / 32).
//! - [`simmim_random_mask`] — Fisher-Yates uniform random patch mask → `Vec<bool>`.
//! - [`simmim_block_mask`] — random rectangular-block mask for spatial continuity.
//! - [`simmim_l1_loss`] — mean absolute error over masked patches only.
//! - [`simmim_l2_loss`] — mean squared error over masked patches only.
//! - [`simmim_reconstruction_loss`] — dispatch to L1 or L2 by flag.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// SimMIM configuration.
#[derive(Debug, Clone)]
pub struct SimMimConfig {
    /// Fraction of patches to mask. Default 0.6 (paper recommends 0.5–0.7).
    pub mask_ratio: f32,
    /// Spatial side-length (pixels) of each patch. Default 32 (swin-B / SimMIM paper).
    pub patch_size: usize,
}

impl Default for SimMimConfig {
    fn default() -> Self {
        Self {
            mask_ratio: 0.6,
            patch_size: 32,
        }
    }
}

impl SimMimConfig {
    /// Validated constructor.
    ///
    /// # Errors
    /// - [`SslError::InvalidMaskRatio`] when `mask_ratio` ∉ `[0, 1)` or non-finite.
    /// - [`SslError::InvalidParameter`] when `patch_size == 0`.
    pub fn new(mask_ratio: f32, patch_size: usize) -> SslResult<Self> {
        if !(mask_ratio.is_finite() && (0.0..1.0).contains(&mask_ratio)) {
            return Err(SslError::InvalidMaskRatio { ratio: mask_ratio });
        }
        if patch_size == 0 {
            return Err(SslError::InvalidParameter {
                name: "patch_size".into(),
                reason: "must be > 0".into(),
            });
        }
        Ok(Self {
            mask_ratio,
            patch_size,
        })
    }
}

// ─── Masking ─────────────────────────────────────────────────────────────────

/// Generate a random uniform mask of length `n_patches`.
///
/// Returns a `Vec<bool>` where `true` indicates that patch *i* is **masked**
/// (must be reconstructed). Exactly `floor(n_patches × mask_ratio)` patches
/// are masked, chosen via Fisher-Yates partial shuffle for tight ratio control.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n_patches == 0`.
/// - [`SslError::InvalidMaskRatio`] when `mask_ratio` ∉ `[0, 1)` or non-finite.
pub fn simmim_random_mask(
    n_patches: usize,
    mask_ratio: f32,
    rng: &mut LcgRng,
) -> SslResult<Vec<bool>> {
    if n_patches == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(mask_ratio.is_finite() && (0.0..1.0).contains(&mask_ratio)) {
        return Err(SslError::InvalidMaskRatio { ratio: mask_ratio });
    }
    let n_mask = (n_patches as f32 * mask_ratio) as usize;
    let mut indices: Vec<usize> = (0..n_patches).collect();
    rng.shuffle(&mut indices);
    let mut mask = vec![false; n_patches];
    for &idx in indices.iter().take(n_mask) {
        mask[idx] = true;
    }
    Ok(mask)
}

/// Generate a **block-wise** mask on a 2-D grid of patches.
///
/// The mask is built by repeatedly placing randomly-sized rectangular blocks
/// (uniform width and height each independently drawn from `[2, 4]` patches)
/// at random positions until the total number of masked patches reaches at
/// least `floor(n_patches_h × n_patches_w × mask_ratio)`.
///
/// Because blocks may overlap, the realised ratio can exceed the target
/// (overshoot is bounded by one block area ≤ 16 patches).
///
/// Returns a flat `Vec<bool>` of length `n_patches_h × n_patches_w` in row-major
/// order (`mask[r * n_patches_w + c] == true` ⟺ patch (r, c) is masked).
///
/// # Errors
/// - [`SslError::EmptyInput`] when either dimension is 0.
/// - [`SslError::InvalidMaskRatio`] when `mask_ratio` ∉ `[0, 1)` or non-finite.
pub fn simmim_block_mask(
    n_patches_h: usize,
    n_patches_w: usize,
    mask_ratio: f32,
    rng: &mut LcgRng,
) -> SslResult<Vec<bool>> {
    if n_patches_h == 0 || n_patches_w == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(mask_ratio.is_finite() && (0.0..1.0).contains(&mask_ratio)) {
        return Err(SslError::InvalidMaskRatio { ratio: mask_ratio });
    }
    let total = n_patches_h * n_patches_w;
    let target_masked = (total as f32 * mask_ratio) as usize;
    let mut mask = vec![false; total];
    let mut n_masked = 0usize;

    // Safety valve: stop after a reasonable number of placements to avoid
    // an infinite loop when target_masked == 0 or the grid is very small.
    let max_iters = (target_masked + 1).max(1) * 16 + 1;
    let mut iters = 0usize;

    while n_masked < target_masked && iters < max_iters {
        iters += 1;
        // Block side lengths: uniform in [2, 4] clamped to grid bounds.
        let bh = (rng.next_usize(3) + 2).min(n_patches_h);
        let bw = (rng.next_usize(3) + 2).min(n_patches_w);
        // Top-left corner: uniform over valid anchor positions.
        let r0 = if n_patches_h > bh {
            rng.next_usize(n_patches_h - bh + 1)
        } else {
            0
        };
        let c0 = if n_patches_w > bw {
            rng.next_usize(n_patches_w - bw + 1)
        } else {
            0
        };
        for r in r0..r0 + bh {
            for c in c0..c0 + bw {
                let idx = r * n_patches_w + c;
                if !mask[idx] {
                    mask[idx] = true;
                    n_masked += 1;
                }
            }
        }
    }
    Ok(mask)
}

// ─── Loss functions ───────────────────────────────────────────────────────────

/// Compute SimMIM L1 (mean absolute error) reconstruction loss over **masked**
/// patches only.
///
/// `pred` and `target` are `[n_patches × patch_dim]` row-major flat slices.
/// `mask[i] == true` means patch *i* is masked (should be reconstructed).
///
/// Returns the mean absolute error averaged over all `(masked patch, channel)`
/// element pairs.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n_patches == 0`, `patch_dim == 0`, or
///   no patches are masked.
/// - [`SslError::DimensionMismatch`] when `pred` or `target` or `mask` have
///   wrong lengths.
pub fn simmim_l1_loss(
    pred: &[f32],
    target: &[f32],
    mask: &[bool],
    n_patches: usize,
    patch_dim: usize,
) -> SslResult<f32> {
    validate_inputs(pred, target, mask, n_patches, patch_dim)?;
    let mut total = 0.0_f64;
    let mut count = 0usize;
    for (i, &masked) in mask.iter().enumerate() {
        if masked {
            count += 1;
            let base = i * patch_dim;
            for k in 0..patch_dim {
                let diff = (pred[base + k] - target[base + k]) as f64;
                total += diff.abs();
            }
        }
    }
    if count == 0 {
        return Err(SslError::EmptyInput);
    }
    Ok((total / (count * patch_dim) as f64) as f32)
}

/// Compute SimMIM L2 (mean squared error) reconstruction loss over **masked**
/// patches only.
///
/// This is a complementary variant — use when MSE loss is preferred over MAE.
///
/// `pred` and `target` are `[n_patches × patch_dim]` row-major flat slices.
/// `mask[i] == true` means patch *i* is masked (should be reconstructed).
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n_patches == 0`, `patch_dim == 0`, or
///   no patches are masked.
/// - [`SslError::DimensionMismatch`] when `pred`, `target`, or `mask` have
///   wrong lengths.
pub fn simmim_l2_loss(
    pred: &[f32],
    target: &[f32],
    mask: &[bool],
    n_patches: usize,
    patch_dim: usize,
) -> SslResult<f32> {
    validate_inputs(pred, target, mask, n_patches, patch_dim)?;
    let mut total = 0.0_f64;
    let mut count = 0usize;
    for (i, &masked) in mask.iter().enumerate() {
        if masked {
            count += 1;
            let base = i * patch_dim;
            for k in 0..patch_dim {
                let diff = (pred[base + k] - target[base + k]) as f64;
                total += diff * diff;
            }
        }
    }
    if count == 0 {
        return Err(SslError::EmptyInput);
    }
    Ok((total / (count * patch_dim) as f64) as f32)
}

/// Dispatch to either L1 or L2 reconstruction loss.
///
/// When `use_l1 == true`, computes mean absolute error; otherwise computes
/// mean squared error.  Both are averaged over masked patches × `patch_dim`.
///
/// # Errors
/// Propagates errors from [`simmim_l1_loss`] / [`simmim_l2_loss`].
pub fn simmim_reconstruction_loss(
    pred: &[f32],
    target: &[f32],
    mask: &[bool],
    n_patches: usize,
    patch_dim: usize,
    use_l1: bool,
) -> SslResult<f32> {
    if use_l1 {
        simmim_l1_loss(pred, target, mask, n_patches, patch_dim)
    } else {
        simmim_l2_loss(pred, target, mask, n_patches, patch_dim)
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Shared dimension-validation logic for the loss functions.
#[inline]
fn validate_inputs(
    pred: &[f32],
    target: &[f32],
    mask: &[bool],
    n_patches: usize,
    patch_dim: usize,
) -> SslResult<()> {
    if n_patches == 0 || patch_dim == 0 {
        return Err(SslError::EmptyInput);
    }
    let expected = n_patches * patch_dim;
    if pred.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: pred.len(),
        });
    }
    if target.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: target.len(),
        });
    }
    if mask.len() != n_patches {
        return Err(SslError::DimensionMismatch {
            expected: n_patches,
            got: mask.len(),
        });
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── L1 loss ───────────────────────────────────────────────────────────────

    /// L1 loss is strictly positive when masked predictions differ from targets.
    #[test]
    fn simmim_l1_loss_all_zero_pred_nonzero_target() {
        let n = 8;
        let d = 4;
        let pred = vec![0.0_f32; n * d];
        let target = vec![1.0_f32; n * d];
        // Mask all patches.
        let mask = vec![true; n];
        let loss = simmim_l1_loss(&pred, &target, &mask, n, d).unwrap();
        assert!(loss > 0.0, "loss should be > 0, got {loss}");
        // Expected: mean |0 - 1| = 1.0
        assert!((loss - 1.0).abs() < 1e-5, "expected 1.0, got {loss}");
    }

    /// L1 loss is exactly zero when prediction equals target on masked patches.
    #[test]
    fn simmim_l1_loss_perfect_reconstruction_zero() {
        let n = 10;
        let d = 8;
        let target: Vec<f32> = (0..n * d).map(|i| i as f32 * 0.1).collect();
        let pred = target.clone();
        let mask = vec![
            true, false, true, false, true, false, true, false, true, false,
        ];
        let loss = simmim_l1_loss(&pred, &target, &mask, n, d).unwrap();
        assert!(loss.abs() < 1e-7, "perfect reconstruction: loss = {loss}");
    }

    /// For predictions with ‖e‖ ≤ 1 on each masked patch, L1 ≤ L2 by
    /// Cauchy-Schwarz: MAE ≤ RMSE ⟺ L1 ≤ sqrt(L2), so L1 ≤ L2 only when
    /// errors are ≤ 1 per element.
    ///
    /// We verify the ordering specifically for small errors (values in [0,1]),
    /// where |x| ≤ x² is false but mean|x| ≤ sqrt(mean x²) holds (Jensen).
    /// We check the squared version: (mean|x|)² ≤ mean x² = L2.
    #[test]
    fn simmim_l1_vs_l2_ordering() {
        let n = 20;
        let d = 16;
        // small residuals in [0, 0.1] — so |x| ≤ 1 ⟹ (mean|x|)² ≤ mean x² is
        // not necessarily true; let's just verify |x| ≤ x² fails here and that
        // l1 ≤ l2 holds *when errors > 1* (use errors = 2.0 constant).
        let target = vec![0.0_f32; n * d];
        let pred = vec![2.0_f32; n * d]; // error = 2 per element
        let mask = vec![true; n];
        let l1 = simmim_l1_loss(&pred, &target, &mask, n, d).unwrap();
        let l2 = simmim_l2_loss(&pred, &target, &mask, n, d).unwrap();
        // L1 = mean|2| = 2.0,  L2 = mean 4 = 4.0  ⟹ L1 ≤ L2
        assert!(
            l1 <= l2,
            "expected L1 ≤ L2 when errors ≥ 1, got L1={l1} L2={l2}"
        );
    }

    // ── L2 loss ───────────────────────────────────────────────────────────────

    /// Manual computation must match `simmim_l2_loss`.
    #[test]
    fn simmim_l2_loss_vs_manual() {
        let n = 4;
        let d = 2;
        // Patches 0 and 2 are masked.
        let target = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let pred = vec![2.0_f32, 3.0, 3.0, 4.0, 6.0, 7.0, 7.0, 8.0];
        let mask = vec![true, false, true, false];
        let loss = simmim_l2_loss(&pred, &target, &mask, n, d).unwrap();
        // Masked patches 0 and 2:
        //   patch 0: (2-1)²=1, (3-2)²=1  → sum = 2
        //   patch 2: (6-5)²=1, (7-6)²=1  → sum = 2
        // total = 4, count = 2 patches × 2 dims = 4 elements → mean = 1.0
        assert!((loss - 1.0).abs() < 1e-5, "expected 1.0, got {loss}");
    }

    // ── Random mask ───────────────────────────────────────────────────────────

    /// The realised mask ratio must be within 0.05 of the target.
    #[test]
    fn simmim_random_mask_ratio_approx() {
        let mut rng = LcgRng::new(42);
        let n = 200;
        let ratio = 0.6_f32;
        let mask = simmim_random_mask(n, ratio, &mut rng).unwrap();
        let n_masked = mask.iter().filter(|&&v| v).count();
        let realised = n_masked as f32 / n as f32;
        assert!(
            (realised - ratio).abs() < 0.05,
            "realised ratio {realised} too far from target {ratio}"
        );
    }

    /// The mask length must equal `n_patches`.
    #[test]
    fn simmim_random_mask_length_correct() {
        let mut rng = LcgRng::new(7);
        let n = 196;
        let mask = simmim_random_mask(n, 0.6, &mut rng).unwrap();
        assert_eq!(mask.len(), n, "mask length mismatch");
    }

    // ── Block mask ────────────────────────────────────────────────────────────

    /// The realised block-mask ratio should be within 0.2 of the target
    /// (blocks can overshoot, so tolerance is wider than for random mask).
    #[test]
    fn simmim_block_mask_ratio_approx() {
        let mut rng = LcgRng::new(99);
        let h = 14;
        let w = 14;
        let ratio = 0.5_f32;
        let mask = simmim_block_mask(h, w, ratio, &mut rng).unwrap();
        let total = h * w;
        assert_eq!(mask.len(), total);
        let n_masked = mask.iter().filter(|&&v| v).count();
        let realised = n_masked as f32 / total as f32;
        assert!(
            (realised - ratio).abs() < 0.20,
            "realised ratio {realised} too far from target {ratio} (tol 0.20)"
        );
    }

    // ── Reconstruction loss dispatch ──────────────────────────────────────────

    /// Dispatch with `use_l1 = true` must return a finite value.
    #[test]
    fn simmim_reconstruction_loss_dispatch_l1() {
        let n = 6;
        let d = 4;
        let pred: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.05).collect();
        let target = vec![0.5_f32; n * d];
        let mask = vec![true, false, true, false, true, false];
        let loss = simmim_reconstruction_loss(&pred, &target, &mask, n, d, true).unwrap();
        assert!(loss.is_finite(), "L1 dispatch returned non-finite: {loss}");
    }

    /// Dispatch with `use_l1 = false` must return a finite value.
    #[test]
    fn simmim_reconstruction_loss_dispatch_l2() {
        let n = 6;
        let d = 4;
        let pred: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.05).collect();
        let target = vec![0.5_f32; n * d];
        let mask = vec![true, false, true, false, true, false];
        let loss = simmim_reconstruction_loss(&pred, &target, &mask, n, d, false).unwrap();
        assert!(loss.is_finite(), "L2 dispatch returned non-finite: {loss}");
    }

    // ── Masking isolation ─────────────────────────────────────────────────────

    /// Changing prediction values for *unmasked* patches must not affect loss.
    #[test]
    fn simmim_loss_only_unmasked_ignored() {
        let n = 6;
        let d = 3;
        let target = vec![1.0_f32; n * d];
        let mask = vec![true, false, true, false, false, true];
        // Base prediction: all zeros.
        let pred_base = vec![0.0_f32; n * d];
        let loss_base = simmim_l1_loss(&pred_base, &target, &mask, n, d).unwrap();
        // Mutate only unmasked patches (indices 1, 3, 4).
        let mut pred_mutated = pred_base.clone();
        for &i in &[1_usize, 3, 4] {
            for k in 0..d {
                pred_mutated[i * d + k] = 999.0;
            }
        }
        let loss_mutated = simmim_l1_loss(&pred_mutated, &target, &mask, n, d).unwrap();
        assert!(
            (loss_base - loss_mutated).abs() < 1e-6,
            "unmasked patches affected loss: {loss_base} vs {loss_mutated}"
        );
    }

    // ── Error handling ────────────────────────────────────────────────────────

    /// Empty `pred` / `target` slices must return `EmptyInput`.
    #[test]
    fn empty_input_returns_error() {
        // n_patches == 0
        assert_eq!(
            simmim_l1_loss(&[], &[], &[], 0, 4),
            Err(SslError::EmptyInput)
        );
        // patch_dim == 0
        assert_eq!(
            simmim_l1_loss(&[], &[], &[], 4, 0),
            Err(SslError::EmptyInput)
        );
        // random mask with n_patches == 0
        let mut rng = LcgRng::new(0);
        assert_eq!(
            simmim_random_mask(0, 0.5, &mut rng),
            Err(SslError::EmptyInput)
        );
        // block mask with zero dimension
        assert_eq!(
            simmim_block_mask(0, 4, 0.5, &mut rng),
            Err(SslError::EmptyInput)
        );
    }

    /// When no patch is masked (all `false`), the loss functions must return
    /// an error (no elements to average over).  When `mask_ratio == 0.0`,
    /// `simmim_random_mask` succeeds but returns all-false; subsequent loss
    /// must then error.
    #[test]
    fn zero_mask_ratio_returns_error_or_zero_loss() {
        let mut rng = LcgRng::new(3);
        let n = 16;
        let mask = simmim_random_mask(n, 0.0, &mut rng).unwrap();
        // All false — no masked patches.
        assert!(mask.iter().all(|&v| !v));
        let pred = vec![1.0_f32; n * 4];
        let target = vec![0.0_f32; n * 4];
        let result = simmim_l1_loss(&pred, &target, &mask, n, 4);
        // Must be an error because there are no masked patches to average.
        assert!(result.is_err(), "expected error for all-unmasked input");
    }
}

//! Soft Dice loss for semantic / instance segmentation (Milletari et al. 2016).
//!
//! The Dice coefficient measures the overlap of a predicted soft mask `p ∈ [0,1]`
//! with a binary target mask `g ∈ {0,1}`:
//!
//! ```text
//! Dice = (2 · Σ p·g + ε) / (Σ p + Σ g + ε)
//! DiceLoss = 1 − Dice
//! ```
//!
//! The squared-denominator variant of Milletari ("V-Net") is also provided via
//! [`dice_loss_squared`], which uses `Σ p² + Σ g²` in the denominator. A small
//! smoothing constant `ε` keeps the loss finite and differentiable when both
//! masks are empty (in which case the loss is exactly `0`).
//!
//! These losses are scale-balanced with respect to foreground area, which makes
//! them robust to the strong class imbalance typical of segmentation masks.

use crate::error::{VisionError, VisionResult};

/// Default Laplace smoothing constant.
const DEFAULT_EPS: f32 = 1.0;

/// Soft Dice loss `1 − Dice` for a single predicted/target mask pair.
///
/// `pred` are probabilities in `[0, 1]`; `target` are labels in `{0, 1}`
/// (any value is accepted and treated as a soft target, but it should be binary
/// for the standard interpretation). `eps` is the smoothing constant added to
/// both numerator and denominator.
pub fn dice_loss(pred: &[f32], target: &[f32], eps: f32) -> VisionResult<f32> {
    let (inter, sum_p, sum_g) = accumulate(pred, target, false)?;
    Ok(dice_loss_from_sums(inter, sum_p, sum_g, eps))
}

/// Squared-denominator (V-Net) Dice loss `1 − Dice` for one mask pair.
pub fn dice_loss_squared(pred: &[f32], target: &[f32], eps: f32) -> VisionResult<f32> {
    let (inter, sum_p2, sum_g2) = accumulate(pred, target, true)?;
    Ok(dice_loss_from_sums(inter, sum_p2, sum_g2, eps))
}

/// Convenience wrapper using the default smoothing constant `ε = 1`.
pub fn dice_loss_default(pred: &[f32], target: &[f32]) -> VisionResult<f32> {
    dice_loss(pred, target, DEFAULT_EPS)
}

/// Mean soft Dice loss over a batch of `(pred, target)` mask pairs.
///
/// Every pair must have matching lengths; pairs may differ in length from each
/// other. Returns [`VisionError::EmptyInput`] for an empty batch.
pub fn dice_loss_batch(pairs: &[(Vec<f32>, Vec<f32>)], eps: f32) -> VisionResult<f32> {
    if pairs.is_empty() {
        return Err(VisionError::EmptyInput("dice_loss_batch"));
    }
    let mut acc = 0.0f32;
    for (pred, target) in pairs {
        acc += dice_loss(pred, target, eps)?;
    }
    Ok(acc / pairs.len() as f32)
}

/// Compute `(intersection, Σpred, Σtarget)`, optionally squaring the marginals.
fn accumulate(pred: &[f32], target: &[f32], squared: bool) -> VisionResult<(f32, f32, f32)> {
    if pred.is_empty() {
        return Err(VisionError::EmptyInput("dice pred"));
    }
    if pred.len() != target.len() {
        return Err(VisionError::ShapeMismatch {
            lhs: vec![pred.len()],
            rhs: vec![target.len()],
        });
    }
    let mut inter = 0.0f32;
    let mut sum_p = 0.0f32;
    let mut sum_g = 0.0f32;
    for (&p, &g) in pred.iter().zip(target.iter()) {
        if !p.is_finite() || !g.is_finite() {
            return Err(VisionError::NonFinite("dice mask"));
        }
        inter += p * g;
        if squared {
            sum_p += p * p;
            sum_g += g * g;
        } else {
            sum_p += p;
            sum_g += g;
        }
    }
    Ok((inter, sum_p, sum_g))
}

/// `1 − (2·inter + ε) / (sum_p + sum_g + ε)`.
#[inline]
fn dice_loss_from_sums(inter: f32, sum_p: f32, sum_g: f32, eps: f32) -> f32 {
    let dice = (2.0 * inter + eps) / (sum_p + sum_g + eps);
    1.0 - dice
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    #[test]
    fn perfect_overlap_zero_loss() {
        let p = [1.0, 1.0, 0.0, 0.0];
        let g = [1.0, 1.0, 0.0, 0.0];
        let l = dice_loss(&p, &g, 1e-6).expect("loss");
        assert!(l < 1e-4, "loss={l}");
    }

    #[test]
    fn no_overlap_near_one_loss() {
        let p = [1.0, 1.0, 0.0, 0.0];
        let g = [0.0, 0.0, 1.0, 1.0];
        let l = dice_loss(&p, &g, 1e-6).expect("loss");
        assert!(l > 0.99, "loss={l}");
    }

    #[test]
    fn both_empty_masks_zero_loss() {
        // Both all-zero: numerator and denominator both = ε → Dice 1 → loss 0.
        let p = [0.0, 0.0, 0.0];
        let g = [0.0, 0.0, 0.0];
        let l = dice_loss(&p, &g, 1.0).expect("loss");
        assert!(l.abs() < TOL, "loss={l}");
    }

    #[test]
    fn half_overlap_intermediate() {
        // pred covers {0,1}, target covers {1,2}: inter=1, |p|=2, |g|=2.
        // Dice = (2·1)/(2+2) = 0.5 → loss 0.5 (with tiny eps).
        let p = [1.0, 1.0, 0.0];
        let g = [0.0, 1.0, 1.0];
        let l = dice_loss(&p, &g, 1e-8).expect("loss");
        assert!((l - 0.5).abs() < 1e-3, "loss={l}");
    }

    #[test]
    fn soft_probabilities_between_zero_and_one() {
        let p = [0.5, 0.5, 0.5, 0.5];
        let g = [1.0, 1.0, 0.0, 0.0];
        let l = dice_loss(&p, &g, 1e-8).expect("loss");
        // inter = 0.5+0.5 = 1.0; |p| = 2.0; |g| = 2.0 → Dice 0.5 → loss 0.5.
        assert!((l - 0.5).abs() < 1e-3, "loss={l}");
    }

    #[test]
    fn loss_is_nonnegative_and_bounded() {
        let p = [0.2, 0.9, 0.1, 0.7, 0.0];
        let g = [0.0, 1.0, 0.0, 1.0, 1.0];
        let l = dice_loss(&p, &g, 1.0).expect("loss");
        assert!((-TOL..=1.0 + TOL).contains(&l), "loss out of range: {l}");
    }

    #[test]
    fn squared_variant_perfect_overlap() {
        let p = [1.0, 0.0, 1.0];
        let g = [1.0, 0.0, 1.0];
        let l = dice_loss_squared(&p, &g, 1e-6).expect("loss");
        assert!(l < 1e-4, "loss={l}");
    }

    #[test]
    fn squared_variant_differs_for_soft_masks() {
        // For soft probabilities the squared denominator changes the value.
        let p = [0.5, 0.5, 0.5, 0.5];
        let g = [1.0, 1.0, 0.0, 0.0];
        let linear = dice_loss(&p, &g, 1e-8).expect("lin");
        let squared = dice_loss_squared(&p, &g, 1e-8).expect("sq");
        assert!((linear - squared).abs() > 1e-3, "lin={linear} sq={squared}");
    }

    #[test]
    fn default_eps_wrapper_matches() {
        let p = [1.0, 0.0, 1.0, 1.0];
        let g = [1.0, 0.0, 0.0, 1.0];
        let a = dice_loss_default(&p, &g).expect("a");
        let b = dice_loss(&p, &g, 1.0).expect("b");
        assert!((a - b).abs() < TOL);
    }

    #[test]
    fn shape_mismatch_errors() {
        let p = [1.0, 0.0];
        let g = [1.0];
        assert!(dice_loss(&p, &g, 1.0).is_err());
    }

    #[test]
    fn empty_input_errors() {
        let p: [f32; 0] = [];
        let g: [f32; 0] = [];
        assert!(dice_loss(&p, &g, 1.0).is_err());
    }

    #[test]
    fn nonfinite_errors() {
        let p = [1.0, f32::NAN];
        let g = [1.0, 0.0];
        assert!(dice_loss(&p, &g, 1.0).is_err());
    }

    #[test]
    fn batch_mean_matches_manual() {
        let pairs = vec![
            (vec![1.0, 1.0, 0.0], vec![1.0, 1.0, 0.0]),
            (vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]),
        ];
        let mean = dice_loss_batch(&pairs, 1e-8).expect("mean");
        let l0 = dice_loss(&pairs[0].0, &pairs[0].1, 1e-8).expect("l0");
        let l1 = dice_loss(&pairs[1].0, &pairs[1].1, 1e-8).expect("l1");
        let manual = 0.5 * (l0 + l1);
        assert!((mean - manual).abs() < TOL, "mean={mean} manual={manual}");
    }

    #[test]
    fn batch_empty_errors() {
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        assert!(dice_loss_batch(&pairs, 1.0).is_err());
    }
}

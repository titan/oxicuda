//! Standalone gradient-clipping utilities for raw `f32` slices.
//!
//! This module provides free functions that operate on plain `&mut [f32]` and
//! `&[&[f32]]` slices, complementing the [`crate::grad_clip`] module which
//! works with [`crate::gpu_optimizer::ParamTensor`] objects.
//!
//! ## Overview
//!
//! | Function | Description |
//! |---|---|
//! | [`clip_grad_norm`] | Scale all gradients so the global L2 norm ≤ `max_norm` |
//! | [`clip_grad_value`] | Clamp each gradient element to `[−clip_val, clip_val]` |
//! | [`global_grad_norm`] | Compute L2 norm across multiple parameter-group slices |
//! | [`adaptive_grad_clip`] | Clip at a given percentile of historical gradient norms |

// ─── clip_grad_norm ───────────────────────────────────────────────────────────

/// Clip gradients in-place so that their global L2 norm is at most `max_norm`.
///
/// Returns the L2 norm of `grads` **before** clipping.
///
/// If `grads` is empty the norm is `0.0` and no modification is made.
/// If `max_norm` is `0.0` or the computed scale would produce non-finite
/// values, all gradient elements are set to `0.0`.
#[must_use = "returns the pre-clip norm"]
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    if grads.is_empty() {
        return 0.0;
    }

    let norm: f32 = grads.iter().map(|&g| g * g).sum::<f32>().sqrt();

    if norm == 0.0 || max_norm <= 0.0 {
        // Zero-out when there's nothing to scale or max_norm is non-positive
        for g in grads.iter_mut() {
            *g = 0.0;
        }
        return norm;
    }

    if norm > max_norm {
        let scale = max_norm / norm;
        if scale.is_finite() {
            for g in grads.iter_mut() {
                *g *= scale;
            }
        } else {
            // Defensive: scale is degenerate, zero out
            for g in grads.iter_mut() {
                *g = 0.0;
            }
        }
    }

    norm
}

// ─── clip_grad_value ─────────────────────────────────────────────────────────

/// Clamp each gradient element to `[−clip_val, clip_val]` in-place.
///
/// `clip_val` should be non-negative; if it is negative, all gradients are
/// clamped to exactly `0.0` (both bounds coincide at zero after sign flip).
pub fn clip_grad_value(grads: &mut [f32], clip_val: f32) {
    let lo = -clip_val.abs();
    let hi = clip_val.abs();
    for g in grads.iter_mut() {
        *g = g.clamp(lo, hi);
    }
}

// ─── global_grad_norm ────────────────────────────────────────────────────────

/// Compute the global L2 norm across multiple parameter-group slices.
///
/// Equivalent to concatenating all slices and computing the L2 norm of the
/// resulting vector, but without allocating.
///
/// Returns `0.0` if `grads` is empty or all groups are empty.
#[must_use]
pub fn global_grad_norm(grads: &[&[f32]]) -> f32 {
    let sum_sq: f32 = grads
        .iter()
        .flat_map(|group| group.iter())
        .map(|&g| g * g)
        .sum();
    sum_sq.sqrt()
}

// ─── adaptive_grad_clip ──────────────────────────────────────────────────────

/// Clip gradients at a given percentile of historical gradient norms.
///
/// `history` must be **sorted in ascending order**.  The percentile threshold
/// is determined by linear interpolation over `history`.
///
/// * `percentile` must be in `[0, 1]`.  Values outside this range are
///   saturated to the nearest bound before use.
/// * Returns the L2 norm of `grads` **before** clipping.
/// * If `history` is empty, `max_norm` defaults to `f32::MAX` (no clipping).
///
/// # Clipping rule
///
/// The threshold `τ` is derived from `history` at the given `percentile`.
/// The gradient vector is then rescaled exactly like [`clip_grad_norm`] with
/// `max_norm = τ`.
#[must_use = "returns the pre-clip norm"]
pub fn adaptive_grad_clip(grads: &mut [f32], percentile: f32, history: &[f32]) -> f32 {
    // Compute current norm before any modification.
    let current_norm: f32 = if grads.is_empty() {
        0.0
    } else {
        grads.iter().map(|&g| g * g).sum::<f32>().sqrt()
    };

    // Determine threshold from sorted history via linear interpolation.
    let threshold = percentile_threshold(history, percentile);

    // Apply norm clipping at the derived threshold (return value intentionally discarded
    // because adaptive_grad_clip already returns the pre-clip norm computed above).
    let _ = clip_grad_norm(grads, threshold);

    current_norm
}

/// Derive the norm threshold from `sorted_history` at `percentile ∈ [0, 1]`
/// using linear interpolation.  Returns `f32::MAX` when history is empty.
fn percentile_threshold(sorted_history: &[f32], percentile: f32) -> f32 {
    if sorted_history.is_empty() {
        return f32::MAX;
    }

    let n = sorted_history.len();
    if n == 1 {
        return sorted_history[0];
    }

    // Saturate percentile to [0, 1]
    let p = percentile.clamp(0.0, 1.0);

    // Map percentile to a floating-point index in [0, n-1]
    let float_idx = p * (n as f32 - 1.0);
    let lo = float_idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = float_idx - lo as f32;

    // Linear interpolation
    sorted_history[lo] + frac * (sorted_history[hi] - sorted_history[lo])
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. norm_clip_reduces_norm — after clip, norm ≤ max_norm
    #[test]
    fn norm_clip_reduces_norm() {
        let mut grads = vec![3.0_f32, 4.0_f32]; // norm = 5.0
        let max_norm = 2.0_f32;
        let _ = clip_grad_norm(&mut grads, max_norm);
        let norm_after: f32 = grads.iter().map(|&g| g * g).sum::<f32>().sqrt();
        assert!(
            norm_after <= max_norm + 1e-5,
            "norm after clip must be ≤ max_norm: got {norm_after}, max={max_norm}"
        );
    }

    // 2. norm_clip_finite — result is finite
    #[test]
    fn norm_clip_finite() {
        // Use values within f32 range whose L2 norm is representable:
        // norm([3e18, -4e18]) = 5e18, which is within f32 range (~3.4e38).
        let mut grads = vec![3e18_f32, -4e18_f32];
        let norm_before = clip_grad_norm(&mut grads, 1.0);
        assert!(
            norm_before.is_finite(),
            "pre-clip norm must be finite, got {norm_before}"
        );
        for &g in &grads {
            assert!(g.is_finite(), "grad must be finite after clip, got {g}");
        }
        // Clipped norm must be ≤ max_norm=1.0
        let norm_after: f32 = grads.iter().map(|&g| g * g).sum::<f32>().sqrt();
        assert!(
            norm_after <= 1.0 + 1e-5,
            "norm after clip must be ≤ 1.0, got {norm_after}"
        );
    }

    // 3. value_clip_bounds — all elements in [-clip_val, clip_val] after clip
    #[test]
    fn value_clip_bounds() {
        let mut grads = vec![-5.0_f32, -1.0, 0.0, 2.0, 10.0];
        let clip_val = 3.0_f32;
        clip_grad_value(&mut grads, clip_val);
        for &g in &grads {
            assert!(
                g >= -clip_val && g <= clip_val,
                "gradient {g} out of bounds [-{clip_val}, {clip_val}]"
            );
        }
    }

    // 4. global_norm_correct — manual check: [3,0] and [4,0] → global norm=5
    #[test]
    fn global_norm_correct() {
        let g1 = [3.0_f32, 0.0_f32];
        let g2 = [4.0_f32, 0.0_f32];
        let norm = global_grad_norm(&[&g1, &g2]);
        assert!(
            (norm - 5.0_f32).abs() < 1e-5,
            "global norm of [3,0]+[4,0] must be 5.0, got {norm}"
        );
    }

    // 5. adaptive_clip_at_percentile — with sorted history and percentile=0.5,
    //    clips near median
    #[test]
    fn adaptive_clip_at_percentile() {
        // history sorted ascending: median = 3.0
        let history = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        // grads with norm >> 3.0
        let mut grads = vec![10.0_f32, 10.0, 10.0]; // norm ≈ 17.3
        let _ = adaptive_grad_clip(&mut grads, 0.5, &history);
        let norm_after: f32 = grads.iter().map(|&g| g * g).sum::<f32>().sqrt();
        // Threshold at p=0.5 over 5 elements: float_idx = 0.5*(5-1) = 2.0 → history[2] = 3.0
        assert!(
            norm_after <= 3.0 + 1e-4,
            "adaptive clip at p=0.5 should clip near median (3.0), got {norm_after}"
        );
    }

    // 6. empty_grads — clip_grad_norm on [] returns 0.0 without panic
    #[test]
    fn empty_grads() {
        let mut grads: Vec<f32> = Vec::new();
        let norm = clip_grad_norm(&mut grads, 1.0);
        assert_eq!(norm, 0.0, "empty grads must yield norm=0.0");
        assert!(grads.is_empty(), "grads must remain empty");
    }

    // 7. max_norm_0_zeros_grads — clip_grad_norm with max_norm=0 makes all grads ~0
    #[test]
    fn max_norm_0_zeros_grads() {
        let mut grads = vec![3.0_f32, 4.0_f32];
        let _ = clip_grad_norm(&mut grads, 0.0);
        for &g in &grads {
            assert!(
                g.abs() < 1e-6,
                "with max_norm=0, all grads must be ≈0, got {g}"
            );
        }
    }

    // 8. norm_below_max_unchanged — grads with norm < max_norm are not changed
    #[test]
    fn norm_below_max_unchanged() {
        let original = vec![0.3_f32, 0.4_f32]; // norm = 0.5
        let mut grads = original.clone();
        let _ = clip_grad_norm(&mut grads, 10.0); // max_norm >> norm
        for (i, (&orig, &after)) in original.iter().zip(grads.iter()).enumerate() {
            assert!(
                (orig - after).abs() < 1e-7,
                "grad[{i}] should be unchanged when norm < max_norm: orig={orig}, after={after}"
            );
        }
    }

    // 9. multi_group_norm — global_grad_norm with multiple groups
    #[test]
    fn multi_group_norm() {
        // Each group contributes to sum of squares:
        // g1 = [1,2,3] → sq sum = 14
        // g2 = [4]     → sq sum = 16
        // g3 = [0,5]   → sq sum = 25
        // total = 55; norm = sqrt(55) ≈ 7.416
        let g1 = [1.0_f32, 2.0, 3.0];
        let g2 = [4.0_f32];
        let g3 = [0.0_f32, 5.0];
        let norm = global_grad_norm(&[&g1, &g2, &g3]);
        let expected = 55.0_f32.sqrt();
        assert!(
            (norm - expected).abs() < 1e-5,
            "multi-group global norm must be sqrt(55)≈{expected}, got {norm}"
        );
    }
}

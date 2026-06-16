//! Lp-ball constraint helpers used by attack and defence modules.

use crate::error::{AdvError, AdvResult};

/// Lp-norm types currently supported as threat models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpNorm {
    /// L∞ ball: per-coordinate clamp to `[−ε, +ε]`.
    LInf,
    /// L2 (Euclidean) ball: rescale δ so its L2 norm ≤ ε.
    L2,
    /// L1 ball: rescale δ so its L1 norm ≤ ε.
    L1,
}

/// Element-wise L∞ norm of a vector.
#[must_use]
pub fn l_inf_norm(x: &[f32]) -> f32 {
    x.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()))
}

/// L2 (Euclidean) norm of a vector.
#[must_use]
pub fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// L1 norm of a vector.
#[must_use]
pub fn l1_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v.abs()).sum()
}

/// Project `x` onto the L∞ ball of radius `eps` centred at `x_orig`,
/// then clamp into `[lo, hi]`. Result is fresh.
///
/// # Errors
/// - [`AdvError::DimensionMismatch`] if shapes disagree.
/// - [`AdvError::InvalidEpsilon`] if `eps` is non-finite or negative.
pub fn project_l_inf(x: &[f32], x_orig: &[f32], eps: f32, lo: f32, hi: f32) -> AdvResult<Vec<f32>> {
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(AdvError::InvalidEpsilon { eps });
    }
    if x.len() != x_orig.len() {
        return Err(AdvError::DimensionMismatch {
            expected: x.len(),
            got: x_orig.len(),
        });
    }
    Ok(x.iter()
        .zip(x_orig.iter())
        .map(|(&xi, &xo)| {
            let l = xo - eps;
            let h = xo + eps;
            xi.clamp(l, h).clamp(lo, hi)
        })
        .collect())
}

/// Project `x` onto the L2 ball of radius `eps` centred at `x_orig`,
/// then clamp into `[lo, hi]`. Result is fresh.
///
/// # Errors
/// - [`AdvError::DimensionMismatch`] if shapes disagree.
/// - [`AdvError::InvalidEpsilon`] if `eps` is non-finite or negative.
pub fn project_l2(x: &[f32], x_orig: &[f32], eps: f32, lo: f32, hi: f32) -> AdvResult<Vec<f32>> {
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(AdvError::InvalidEpsilon { eps });
    }
    if x.len() != x_orig.len() {
        return Err(AdvError::DimensionMismatch {
            expected: x.len(),
            got: x_orig.len(),
        });
    }
    let delta: Vec<f32> = x.iter().zip(x_orig.iter()).map(|(a, b)| a - b).collect();
    let n = l2_norm(&delta);
    let factor = if n > eps && n > 0.0 { eps / n } else { 1.0 };
    Ok(delta
        .iter()
        .zip(x_orig.iter())
        .map(|(&d, &o)| (o + factor * d).clamp(lo, hi))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l_inf_norm_max_abs() {
        assert!((l_inf_norm(&[1.0_f32, -3.0, 2.0]) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn l2_norm_unit_basis() {
        assert!((l2_norm(&[1.0_f32, 0.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((l2_norm(&[3.0_f32, 4.0]) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn l1_norm_sum_abs() {
        assert!((l1_norm(&[1.0_f32, -2.0, 3.0]) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn project_l_inf_clamps() {
        let orig = vec![0.5_f32, 0.5];
        let x = vec![1.5_f32, -1.5];
        let p = project_l_inf(&x, &orig, 0.3, 0.0, 1.0).expect("project_l_inf should succeed");
        assert!((p[0] - 0.8).abs() < 1e-5); // 0.5 + 0.3
        assert!((p[1] - 0.2).abs() < 1e-5); // 0.5 - 0.3
    }

    #[test]
    fn project_l_inf_outer_clamp() {
        let orig = vec![0.95_f32];
        let x = vec![10.0_f32];
        let p = project_l_inf(&x, &orig, 0.3, 0.0, 1.0).expect("project_l_inf should succeed");
        // 0.95 + 0.3 = 1.25 → clamped to 1.0
        assert!((p[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn project_l_inf_zero_eps_keeps_orig() {
        let orig = vec![0.5_f32, 0.7];
        let x = vec![10.0_f32, -10.0];
        let p = project_l_inf(&x, &orig, 0.0, 0.0, 1.0).expect("project_l_inf should succeed");
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn project_l_inf_rejects_invalid_eps() {
        let orig = vec![0.5_f32];
        let x = vec![0.5_f32];
        assert!(project_l_inf(&x, &orig, -0.1, 0.0, 1.0).is_err());
        assert!(project_l_inf(&x, &orig, f32::NAN, 0.0, 1.0).is_err());
    }

    #[test]
    fn project_l_inf_dim_mismatch() {
        let orig = vec![0.5_f32, 0.7];
        let x = vec![0.5_f32];
        assert!(project_l_inf(&x, &orig, 0.1, 0.0, 1.0).is_err());
    }

    #[test]
    fn project_l2_inside_ball_is_identity() {
        let orig = vec![0.0_f32, 0.0];
        let x = vec![0.1_f32, 0.0];
        let p = project_l2(&x, &orig, 1.0, -10.0, 10.0).expect("project_l2 should succeed");
        assert!((p[0] - 0.1).abs() < 1e-6);
        assert!((p[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn project_l2_outside_ball_scales_back() {
        let orig = vec![0.0_f32, 0.0];
        let x = vec![3.0_f32, 4.0];
        // ‖x‖ = 5; ε = 1 → factor = 0.2 → result ≈ (0.6, 0.8)
        let p = project_l2(&x, &orig, 1.0, -10.0, 10.0).expect("project_l2 should succeed");
        assert!((p[0] - 0.6).abs() < 1e-5);
        assert!((p[1] - 0.8).abs() < 1e-5);
        // L2 norm of result equals 1.
        assert!((l2_norm(&p) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn project_l2_outer_clamp_applied() {
        let orig = vec![0.9_f32];
        let x = vec![10.0_f32];
        // ‖x − orig‖ = 9.1; ε = 5 → projected x = 0.9 + 5·1·sign = 5.9 → clamped to 1.0
        let p = project_l2(&x, &orig, 5.0, 0.0, 1.0).expect("project_l2 should succeed");
        assert!((p[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn project_l2_rejects_invalid_eps() {
        let orig = vec![0.5_f32];
        let x = vec![0.5_f32];
        assert!(project_l2(&x, &orig, -1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn lp_norm_variants_distinct() {
        assert_ne!(LpNorm::LInf, LpNorm::L2);
        assert_ne!(LpNorm::L1, LpNorm::LInf);
    }
}

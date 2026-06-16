//! Scene contraction from Mip-NeRF 360 (Barron et al. 2022 CVPR).
//!
//! Maps unbounded 3D space into a bounded ball of radius `2 * inner_radius`,
//! enabling NeRF to handle scenes with objects at arbitrary distances.
//!
//! # Contraction formula
//!
//! For a 3D point `x ∈ R³` and inner radius `r`:
//! ```text
//! contract(x) = x                                  if |x|₂ ≤ r
//! contract(x) = (2 - r/|x|₂) * x / |x|₂ * r      if |x|₂ > r
//! ```
//! This maps all of R³ into the ball `B(0, 2r)`.
//!
//! # Inverse (uncontraction)
//!
//! ```text
//! uncontract(y) = y                                     if |y|₂ ≤ r
//! uncontract(y) = r² / (|y|₂ * (2r - |y|₂)) * y      if r < |y|₂ < 2r
//! ```

use crate::error::{NerfError, NerfResult};

// ─── ContractionConfig ────────────────────────────────────────────────────────

/// Configuration for Mip-NeRF 360 scene contraction.
#[derive(Debug, Clone, Copy)]
pub struct ContractionConfig {
    /// Inner sphere radius — region mapped identically. Default `1.0`.
    pub inner_radius: f32,
    /// Epsilon for numerical stability when computing `1/norm`. Default `1e-8`.
    pub eps: f32,
}

impl Default for ContractionConfig {
    fn default() -> Self {
        Self {
            inner_radius: 1.0,
            eps: 1e-8,
        }
    }
}

// ─── Validation helpers ───────────────────────────────────────────────────────

/// Return an error if any value in `slice` is NaN or infinite.
fn check_finite_3(x: &[f32], context: &str) -> NerfResult<()> {
    for &v in x {
        if !v.is_finite() {
            return Err(NerfError::NanEncountered {
                context: context.to_string(),
            });
        }
    }
    Ok(())
}

/// Check that `x.len() == 3`, then that all values are finite.
fn check_point(x: &[f32], context: &str) -> NerfResult<()> {
    if x.len() != 3 {
        return Err(NerfError::DimensionMismatch {
            expected: 3,
            got: x.len(),
        });
    }
    check_finite_3(x, context)
}

// ─── contract_point ───────────────────────────────────────────────────────────

/// Contract a single 3D point into the ball of radius `2 * cfg.inner_radius`.
///
/// - If `|x|₂ ≤ inner_radius`: returns `x` unchanged (identity in inner sphere).
/// - Otherwise: applies the Mip-NeRF 360 contraction mapping.
///
/// # Errors
///
/// - [`NerfError::DimensionMismatch`] if `x.len() ≠ 3`.
/// - [`NerfError::NanEncountered`] if `x` contains NaN or infinity.
pub fn contract_point(x: &[f32], cfg: &ContractionConfig) -> NerfResult<Vec<f32>> {
    check_point(x, "x")?;

    let norm = l2_norm_3(x);
    let r = cfg.inner_radius;

    if norm <= r {
        return Ok(x.to_vec());
    }

    // For |x| = norm > r:
    //   contract(x) = (2 - r/norm) * (x / norm)
    //
    // Derivation: let u = x/r, |u| = norm/r > 1.
    //   contract_unit(u) = (2 - 1/|u|) * u/|u|  [standard r=1 formula]
    //   contract(x) = r * contract_unit(x/r)
    //               = r * (2 - r/norm) * (x/r) / (norm/r)
    //               = (2 - r/norm) * x / norm
    let safe_norm = norm.max(cfg.eps);
    let scale = (2.0 * r - r * r / safe_norm) / safe_norm;

    Ok(vec![x[0] * scale, x[1] * scale, x[2] * scale])
}

// ─── contract_batch ───────────────────────────────────────────────────────────

/// Contract a batch of N 3D points given as a flat array of length `N * 3`.
///
/// Input layout: `[x0, y0, z0, x1, y1, z1, ...]`.
///
/// # Errors
///
/// - [`NerfError::InvalidSampleCount`] if `pts.len() % 3 ≠ 0`.
/// - [`NerfError::NanEncountered`] if any coordinate is NaN or infinity.
pub fn contract_batch(pts: &[f32], cfg: &ContractionConfig) -> NerfResult<Vec<f32>> {
    if !pts.len().is_multiple_of(3) {
        return Err(NerfError::InvalidSampleCount { n: pts.len() });
    }
    check_finite_3(pts, "pts")?;

    let n = pts.len() / 3;
    let mut out = vec![0.0_f32; pts.len()];

    for i in 0..n {
        let off = i * 3;
        let p = &pts[off..off + 3];
        let contracted = contract_point(p, cfg)?;
        out[off] = contracted[0];
        out[off + 1] = contracted[1];
        out[off + 2] = contracted[2];
    }

    Ok(out)
}

// ─── uncontract_point ────────────────────────────────────────────────────────

/// Invert the contraction for a single 3D point `y` within the contracted ball.
///
/// - If `|y|₂ ≤ inner_radius`: returns `y` unchanged.
/// - If `inner_radius < |y|₂ < 2 * inner_radius`: returns the pre-image.
/// - If `|y|₂ ≥ 2 * inner_radius` or `|y|₂ == 0` (with `|y| > r`): returns
///   [`NerfError::InvalidBounds`].
///
/// # Errors
///
/// - [`NerfError::DimensionMismatch`] if `y.len() ≠ 3`.
/// - [`NerfError::NanEncountered`] if `y` contains NaN or infinity.
/// - [`NerfError::InvalidBounds`] if `|y|₂ ≥ 2 * inner_radius`.
pub fn uncontract_point(y: &[f32], cfg: &ContractionConfig) -> NerfResult<Vec<f32>> {
    check_point(y, "y")?;

    let norm_y = l2_norm_3(y);
    let r = cfg.inner_radius;

    if norm_y <= r {
        return Ok(y.to_vec());
    }

    // Must be strictly inside the outer shell
    if norm_y >= 2.0 * r - cfg.eps {
        return Err(NerfError::InvalidBounds {
            near: norm_y,
            far: 2.0 * r,
        });
    }

    // Derivation:
    //   contracted: y = (2 - r/|x|) * x / |x|
    //   |y| = 2 - r/|x|  →  |x| = r / (2 - |y|/r) = r² / (2r - |y|)
    //   x = y * |x| / |y| = y * r² / (|y| * (2r - |y|))
    let denom = norm_y * (2.0 * r - norm_y);
    // denom > 0 because norm_y > 0 and (2r - norm_y) > 0 (checked above)
    let scale = r * r / denom;

    Ok(vec![y[0] * scale, y[1] * scale, y[2] * scale])
}

// ─── uncontract_batch ────────────────────────────────────────────────────────

/// Invert the contraction for a batch of N 3D points given as a flat array.
///
/// Input layout: `[x0, y0, z0, x1, y1, z1, ...]`.
///
/// # Errors
///
/// - [`NerfError::InvalidSampleCount`] if `pts.len() % 3 ≠ 0`.
/// - Propagates any error from [`uncontract_point`].
pub fn uncontract_batch(pts: &[f32], cfg: &ContractionConfig) -> NerfResult<Vec<f32>> {
    if !pts.len().is_multiple_of(3) {
        return Err(NerfError::InvalidSampleCount { n: pts.len() });
    }
    check_finite_3(pts, "pts")?;

    let n = pts.len() / 3;
    let mut out = vec![0.0_f32; pts.len()];

    for i in 0..n {
        let off = i * 3;
        let p = &pts[off..off + 3];
        let unc = uncontract_point(p, cfg)?;
        out[off] = unc[0];
        out[off + 1] = unc[1];
        out[off + 2] = unc[2];
    }

    Ok(out)
}

// ─── contracted_norm ──────────────────────────────────────────────────────────

/// Compute the contracted norm: `|contract(x)|₂` for a 3D point.
///
/// Useful for checking whether a point falls in the inner or outer contracted region.
///
/// # Errors
///
/// - [`NerfError::DimensionMismatch`] if `x.len() ≠ 3`.
/// - [`NerfError::NanEncountered`] if `x` contains NaN or infinity.
pub fn contracted_norm(x: &[f32], cfg: &ContractionConfig) -> NerfResult<f32> {
    let c = contract_point(x, cfg)?;
    Ok(l2_norm_3(&c))
}

// ─── is_inner ─────────────────────────────────────────────────────────────────

/// Check whether a 3D point lies within the inner sphere (not contracted).
///
/// Returns `true` iff `|x|₂ ≤ cfg.inner_radius`.
///
/// # Errors
///
/// - [`NerfError::DimensionMismatch`] if `x.len() ≠ 3`.
/// - [`NerfError::NanEncountered`] if `x` contains NaN or infinity.
pub fn is_inner(x: &[f32], cfg: &ContractionConfig) -> NerfResult<bool> {
    check_point(x, "x")?;
    Ok(l2_norm_3(x) <= cfg.inner_radius)
}

// ─── Internal utilities ───────────────────────────────────────────────────────

/// Compute the L₂ norm of a 3-element slice.
#[inline]
fn l2_norm_3(v: &[f32]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: ContractionConfig = ContractionConfig {
        inner_radius: 1.0,
        eps: 1e-8,
    };

    fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() <= tol, "{msg}: expected {b} ± {tol}, got {a}");
    }

    fn assert_vec_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (ai - bi).abs() <= tol,
                "{msg}[{i}]: expected {bi} ± {tol}, got {ai}"
            );
        }
    }

    // 1. Point inside inner sphere is unchanged
    #[test]
    fn contract_point_inner_unchanged() {
        let x = [0.3_f32, 0.4, 0.0]; // |x| = 0.5 ≤ 1.0
        let y = contract_point(&x, &CFG).expect("contract_point should succeed");
        assert_vec_close(&y, &x, 1e-7, "inner point identity");
    }

    // 2. Point exactly on boundary is unchanged (|x| = 1.0)
    #[test]
    fn contract_point_exactly_on_boundary() {
        let x = [1.0_f32, 0.0, 0.0];
        let y = contract_point(&x, &CFG).expect("contract_point should succeed");
        assert_vec_close(&y, &x, 1e-7, "boundary identity");
    }

    // 3. Point outside inner sphere: |contract(x)| < 2.0
    #[test]
    fn contract_point_outer_norm_lt_2() {
        let x = [2.0_f32, 0.0, 0.0]; // |x| = 2 > 1
        let y = contract_point(&x, &CFG).expect("contract_point should succeed");
        let norm_y = l2_norm_3(&y);
        assert!(norm_y < 2.0, "|contract(x)| must be < 2, got {norm_y}");
    }

    // 4. Very large x: |contract(x)| < 2.0
    #[test]
    fn contract_point_infinity_norm_lt_2() {
        let x = [
            100.0_f32 / 3.0_f32.sqrt(),
            100.0 / 3.0_f32.sqrt(),
            100.0 / 3.0_f32.sqrt(),
        ]; // |x| ≈ 100
        let y = contract_point(&x, &CFG).expect("contract_point should succeed");
        let norm_y = l2_norm_3(&y);
        assert!(
            norm_y < 2.0,
            "|contract(x)| must be < 2 for large x, got {norm_y}"
        );
    }

    // 5. Direction is preserved (contract(x) is parallel to x for outer region)
    #[test]
    fn contract_point_direction_preserved() {
        let x = [3.0_f32, 4.0, 0.0]; // |x| = 5 > 1
        let y = contract_point(&x, &CFG).expect("contract_point should succeed");
        // y = scale * x, so cross product x × y = 0
        // Check unit direction: y/|y| == x/|x|
        let norm_x = l2_norm_3(&x);
        let norm_y = l2_norm_3(&y);
        let ux = [x[0] / norm_x, x[1] / norm_x, x[2] / norm_x];
        let uy = [y[0] / norm_y, y[1] / norm_y, y[2] / norm_y];
        assert_vec_close(&ux, &uy, 1e-6, "direction preserved");
    }

    // 6. Zero vector: |x| = 0 ≤ inner_radius → unchanged
    #[test]
    fn contract_point_zero() {
        let x = [0.0_f32, 0.0, 0.0];
        let y = contract_point(&x, &CFG).expect("contract_point should succeed");
        assert_vec_close(&y, &x, 1e-9, "zero point identity");
    }

    // 7. Axis-aligned: x = [2,0,0] → contract = [(2-0.5),0,0] = [1.5,0,0]
    //    scale = (2 - r/norm)/norm = (2 - 1/2)/2 = 1.5/2 = 0.75
    //    y = [2*0.75, 0, 0] = [1.5, 0, 0]
    #[test]
    fn contract_point_axis_aligned() {
        let x = [2.0_f32, 0.0, 0.0];
        let y = contract_point(&x, &CFG).expect("contract_point should succeed");
        assert_vec_close(&y, &[1.5, 0.0, 0.0], 1e-6, "axis-aligned contraction");
    }

    // 8. Uncontract: inner point unchanged
    #[test]
    fn uncontract_point_inner_unchanged() {
        let y = [0.5_f32, 0.0, 0.0];
        let x = uncontract_point(&y, &CFG).expect("uncontract_point should succeed");
        assert_vec_close(&x, &y, 1e-7, "inner uncontract identity");
    }

    // 9. Roundtrip inner: uncontract(contract(x)) ≈ x for |x|=0.7
    #[test]
    fn uncontract_point_roundtrip_inner() {
        let x = [0.7_f32, 0.0, 0.0];
        let contracted = contract_point(&x, &CFG).expect("contract_point should succeed");
        let recovered =
            uncontract_point(&contracted, &CFG).expect("uncontract_point should succeed");
        assert_vec_close(&recovered, &x, 1e-5, "roundtrip inner");
    }

    // 10. Roundtrip outer: uncontract(contract(x)) ≈ x for |x|=3.0
    #[test]
    fn uncontract_point_roundtrip_outer() {
        let x = [3.0_f32, 0.0, 0.0];
        let contracted = contract_point(&x, &CFG).expect("contract_point should succeed");
        let recovered =
            uncontract_point(&contracted, &CFG).expect("uncontract_point should succeed");
        assert_vec_close(&recovered, &x, 1e-5, "roundtrip outer |x|=3");
    }

    // 11. Roundtrip far: uncontract(contract(x)) ≈ x for |x|=50.0
    #[test]
    fn uncontract_point_roundtrip_far() {
        let x = [50.0_f32, 0.0, 0.0];
        let contracted = contract_point(&x, &CFG).expect("contract_point should succeed");
        let recovered =
            uncontract_point(&contracted, &CFG).expect("uncontract_point should succeed");
        assert_vec_close(&recovered, &x, 1e-4, "roundtrip far |x|=50");
    }

    // 12. Batch shape: 5 points → output len == 15
    #[test]
    fn contract_batch_shape() {
        let pts: Vec<f32> = (0..15).map(|i| i as f32 * 0.1).collect();
        let out = contract_batch(&pts, &CFG).expect("contract_batch should succeed");
        assert_eq!(out.len(), 15, "batch output length");
    }

    // 13. Batch roundtrip: contract then uncontract, all within 1e-5
    #[test]
    fn uncontract_batch_roundtrip() {
        // 4 points: 2 inner, 2 outer
        let pts = [
            0.3_f32, 0.0, 0.0, // inner
            0.0, 0.7, 0.0, // inner
            2.0, 0.0, 0.0, // outer
            0.0, 0.0, 3.0, // outer
        ];
        let contracted = contract_batch(&pts, &CFG).expect("contract_batch should succeed");
        let recovered =
            uncontract_batch(&contracted, &CFG).expect("uncontract_batch should succeed");
        assert_vec_close(&recovered, &pts, 1e-5, "batch roundtrip");
    }

    // 14. contracted_norm for inner: equals |x|
    #[test]
    fn contracted_norm_inner() {
        let x = [0.6_f32, 0.0, 0.0];
        let cn = contracted_norm(&x, &CFG).expect("contracted_norm should succeed");
        assert_close(cn, 0.6, 1e-7, "contracted_norm inner");
    }

    // 15. contracted_norm for outer: < 2.0
    #[test]
    fn contracted_norm_outer() {
        let x = [5.0_f32, 0.0, 0.0];
        let cn = contracted_norm(&x, &CFG).expect("contracted_norm should succeed");
        assert!(cn < 2.0, "contracted_norm outer must be < 2.0, got {cn}");
    }

    // 16. is_inner true for inner point
    #[test]
    fn is_inner_true() {
        let x = [0.5_f32, 0.0, 0.0];
        assert!(
            is_inner(&x, &CFG).expect("is_inner should succeed"),
            "should be inner"
        );
    }

    // 17. is_inner false for outer point
    #[test]
    fn is_inner_false() {
        let x = [2.0_f32, 0.0, 0.0];
        assert!(
            !is_inner(&x, &CFG).expect("is_inner should succeed"),
            "should not be inner"
        );
    }

    // 18. NanEncountered for NaN input
    #[test]
    fn contract_err_nan() {
        let x = [f32::NAN, 0.0, 0.0];
        let result = contract_point(&x, &CFG);
        assert!(
            matches!(result, Err(NerfError::NanEncountered { .. })),
            "expected NanEncountered"
        );
    }

    // 19. InvalidBounds for |y| ≥ 2*inner_radius
    #[test]
    fn uncontract_err_out_of_ball() {
        let y = [2.0_f32, 0.0, 0.0]; // |y| = 2 ≥ 2 * 1.0
        let result = uncontract_point(&y, &CFG);
        assert!(
            matches!(result, Err(NerfError::InvalidBounds { .. })),
            "expected InvalidBounds for |y|=2"
        );
    }

    // 20. InvalidSampleCount when pts.len() % 3 ≠ 0
    #[test]
    fn contract_batch_err_not_multiple_of_3() {
        let pts = [0.0_f32, 1.0]; // len=2, not divisible by 3
        let result = contract_batch(&pts, &CFG);
        assert!(
            matches!(result, Err(NerfError::InvalidSampleCount { .. })),
            "expected InvalidSampleCount"
        );
    }

    // 21. Analytical check: x=[3,0,0] → contract = [(2 - 1/3), 0, 0] = [5/3, 0, 0]
    //     scale = (2 - r/norm)/norm = (2 - 1/3)/3 = (5/3)/3 = 5/9
    //     y = [3 * 5/9, 0, 0] = [5/3, 0, 0]
    #[test]
    fn contract_axis_x() {
        let x = [3.0_f32, 0.0, 0.0];
        let y = contract_point(&x, &CFG).expect("contract_point should succeed");
        let expected = 5.0_f32 / 3.0;
        assert_close(y[0], expected, 1e-6, "x=[3,0,0] → contract[0] = 5/3");
        assert_close(y[1], 0.0, 1e-9, "x=[3,0,0] → contract[1] = 0");
        assert_close(y[2], 0.0, 1e-9, "x=[3,0,0] → contract[2] = 0");
    }
}

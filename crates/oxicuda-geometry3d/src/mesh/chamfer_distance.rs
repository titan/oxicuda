//! Chamfer distance between two point clouds.

use crate::error::{Geom3dError, Geom3dResult};

/// Bidirectional Chamfer Distance between A `[na×3]` and B `[nb×3]`.
///
/// `CD(A,B) = (1/|A|)Σ min_b ||a-b||² + (1/|B|)Σ min_a ||a-b||²`
pub fn chamfer_distance(a: &[f32], na: usize, b: &[f32], nb: usize) -> Geom3dResult<f32> {
    if na == 0 || nb == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if a.len() != na * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: na * 3,
            got: a.len(),
        });
    }
    if b.len() != nb * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: nb * 3,
            got: b.len(),
        });
    }

    let mut sum_ab = 0.0_f64;
    for ai in 0..na {
        let ax = a[ai * 3];
        let ay = a[ai * 3 + 1];
        let az = a[ai * 3 + 2];
        let mut min_d = f32::INFINITY;
        for bi in 0..nb {
            let dx = ax - b[bi * 3];
            let dy = ay - b[bi * 3 + 1];
            let dz = az - b[bi * 3 + 2];
            let d = dx * dx + dy * dy + dz * dz;
            if d < min_d {
                min_d = d;
            }
        }
        sum_ab += min_d as f64;
    }

    let mut sum_ba = 0.0_f64;
    for bi in 0..nb {
        let bx = b[bi * 3];
        let by = b[bi * 3 + 1];
        let bz = b[bi * 3 + 2];
        let mut min_d = f32::INFINITY;
        for ai in 0..na {
            let dx = bx - a[ai * 3];
            let dy = by - a[ai * 3 + 1];
            let dz = bz - a[ai * 3 + 2];
            let d = dx * dx + dy * dy + dz * dz;
            if d < min_d {
                min_d = d;
            }
        }
        sum_ba += min_d as f64;
    }

    let cd = (sum_ab / na as f64 + sum_ba / nb as f64) as f32;

    if !cd.is_finite() {
        return Err(Geom3dError::NanEncountered {
            location: "chamfer_distance",
        });
    }

    Ok(cd)
}

/// Returns per-point gradients `(grad_a [na×3], grad_b [nb×3])`.
///
/// `grad_a[i] = 2*(a_i - b_nearest[i]) / na`
pub fn chamfer_distance_grad(
    a: &[f32],
    na: usize,
    b: &[f32],
    nb: usize,
) -> Geom3dResult<(Vec<f32>, Vec<f32>)> {
    if na == 0 || nb == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if a.len() != na * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: na * 3,
            got: a.len(),
        });
    }
    if b.len() != nb * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: nb * 3,
            got: b.len(),
        });
    }

    let mut grad_a = vec![0.0_f32; na * 3];
    let mut grad_b = vec![0.0_f32; nb * 3];

    // A→B: for each a[i], find nearest b[j]
    for ai in 0..na {
        let ax = a[ai * 3];
        let ay = a[ai * 3 + 1];
        let az = a[ai * 3 + 2];

        let mut min_d = f32::INFINITY;
        let mut nearest_b = 0usize;
        for bi in 0..nb {
            let dx = ax - b[bi * 3];
            let dy = ay - b[bi * 3 + 1];
            let dz = az - b[bi * 3 + 2];
            let d = dx * dx + dy * dy + dz * dz;
            if d < min_d {
                min_d = d;
                nearest_b = bi;
            }
        }
        let scale = 2.0 / na as f32;
        grad_a[ai * 3] = scale * (ax - b[nearest_b * 3]);
        grad_a[ai * 3 + 1] = scale * (ay - b[nearest_b * 3 + 1]);
        grad_a[ai * 3 + 2] = scale * (az - b[nearest_b * 3 + 2]);
    }

    // B→A: for each b[j], find nearest a[i]
    for bi in 0..nb {
        let bx = b[bi * 3];
        let by = b[bi * 3 + 1];
        let bz = b[bi * 3 + 2];

        let mut min_d = f32::INFINITY;
        let mut nearest_a = 0usize;
        for ai in 0..na {
            let dx = bx - a[ai * 3];
            let dy = by - a[ai * 3 + 1];
            let dz = bz - a[ai * 3 + 2];
            let d = dx * dx + dy * dy + dz * dz;
            if d < min_d {
                min_d = d;
                nearest_a = ai;
            }
        }
        let scale = 2.0 / nb as f32;
        grad_b[bi * 3] = scale * (bx - a[nearest_a * 3]);
        grad_b[bi * 3 + 1] = scale * (by - a[nearest_a * 3 + 1]);
        grad_b[bi * 3 + 2] = scale * (bz - a[nearest_a * 3 + 2]);
    }

    Ok((grad_a, grad_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts_from_iter(n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .flat_map(|i| vec![i as f32 * scale, 0.0, 0.0])
            .collect()
    }

    #[test]
    fn chamfer_self_distance_zero() {
        let pts = pts_from_iter(10, 1.0);
        let cd = chamfer_distance(&pts, 10, &pts, 10).expect("chamfer_distance should succeed");
        assert!(cd.abs() < 1e-5, "CD(A,A) must be 0, got {cd}");
    }

    #[test]
    fn chamfer_nonnegative() {
        let a = pts_from_iter(5, 1.0);
        let b = pts_from_iter(8, 0.7);
        let cd = chamfer_distance(&a, 5, &b, 8).expect("chamfer_distance should succeed");
        assert!(cd >= 0.0, "Chamfer distance must be non-negative");
    }

    #[test]
    fn chamfer_symmetric_equal_sizes() {
        let a: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let b: Vec<f32> = vec![0.5, 0.0, 0.0, 1.5, 0.0, 0.0];
        let cd_ab = chamfer_distance(&a, 2, &b, 2).expect("chamfer_distance should succeed");
        let cd_ba = chamfer_distance(&b, 2, &a, 2).expect("chamfer_distance should succeed");
        assert!(
            (cd_ab - cd_ba).abs() < 1e-5,
            "Chamfer must be symmetric for equal sizes"
        );
    }

    #[test]
    fn chamfer_empty_error() {
        assert!(chamfer_distance(&[], 0, &[1.0, 0.0, 0.0], 1).is_err());
    }

    #[test]
    fn chamfer_grad_shape() {
        let a: Vec<f32> = pts_from_iter(5, 1.0);
        let b: Vec<f32> = pts_from_iter(7, 0.8);
        let (ga, gb) =
            chamfer_distance_grad(&a, 5, &b, 7).expect("chamfer_distance_grad should succeed");
        assert_eq!(ga.len(), 5 * 3);
        assert_eq!(gb.len(), 7 * 3);
    }

    #[test]
    fn chamfer_grad_at_self_near_zero() {
        // Gradient of CD(A,A) should be near zero
        let pts: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (ga, gb) =
            chamfer_distance_grad(&pts, 3, &pts, 3).expect("chamfer_distance_grad should succeed");
        for &v in ga.iter().chain(gb.iter()) {
            assert!(v.abs() < 1e-5, "Gradient of CD(A,A) should be ~0, got {v}");
        }
    }

    #[test]
    fn chamfer_increases_with_distance() {
        let a: Vec<f32> = vec![0.0, 0.0, 0.0];
        let b_near: Vec<f32> = vec![1.0, 0.0, 0.0];
        let b_far: Vec<f32> = vec![10.0, 0.0, 0.0];
        let cd_near = chamfer_distance(&a, 1, &b_near, 1).expect("chamfer_distance should succeed");
        let cd_far = chamfer_distance(&a, 1, &b_far, 1).expect("chamfer_distance should succeed");
        assert!(cd_far > cd_near);
    }
}

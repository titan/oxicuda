//! Robust plane fitting via RANSAC plus total-least-squares refinement.
//!
//! Fitting a plane to a noisy point cloud with outliers (e.g. a LiDAR ground
//! sweep or a table surface in an RGB-D scan) is a textbook RANSAC application:
//!
//! 1. Randomly sample three points and form their candidate plane.
//! 2. Count *inliers* — points within `distance_threshold` of the plane.
//! 3. Keep the model with the most inliers across `max_iterations` trials.
//! 4. Re-fit the plane to all inliers by total least squares (PCA: the plane
//!    normal is the smallest-eigenvalue eigenvector of the inlier covariance).
//!
//! A plane is stored as `(normal, d)` with `normal` unit-length and the implicit
//! equation `n · x + d = 0`; the signed distance of a point is simply
//! `n · x + d`. Point clouds follow the crate convention: a flat `&[f32]` of
//! length `3 · n_points` in row-major `[x, y, z]` order.

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

/// A fitted plane `n · x + d = 0` with unit normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    /// Unit-length plane normal `(a, b, c)`.
    pub normal: [f32; 3],
    /// Plane offset `d` such that `a·x + b·y + c·z + d = 0`.
    pub d: f32,
}

impl Plane {
    /// Signed point-to-plane distance `n · p + d` (positive on the `+normal` side).
    #[must_use]
    pub fn signed_distance(&self, p: [f32; 3]) -> f32 {
        self.normal[0] * p[0] + self.normal[1] * p[1] + self.normal[2] * p[2] + self.d
    }
}

/// Result of [`fit_plane_ransac`].
#[derive(Debug, Clone)]
pub struct PlaneFitResult {
    /// The best plane found.
    pub plane: Plane,
    /// Indices of points classified as inliers under the final model.
    pub inliers: Vec<usize>,
    /// Number of inliers (== `inliers.len()`).
    pub n_inliers: usize,
    /// Iterations actually executed.
    pub iterations: usize,
}

#[inline]
fn point(cloud: &[f32], i: usize) -> [f32; 3] {
    [cloud[i * 3], cloud[i * 3 + 1], cloud[i * 3 + 2]]
}

/// Form the plane through three points, returning `None` if they are collinear.
fn plane_from_three(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<Plane> {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    // Cross product = plane normal.
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len < 1e-12 {
        return None; // degenerate / collinear sample
    }
    let normal = [n[0] / len, n[1] / len, n[2] / len];
    let d = -(normal[0] * a[0] + normal[1] * a[1] + normal[2] * a[2]);
    Some(Plane { normal, d })
}

/// Total-least-squares plane fit of the inlier set via PCA (f64 accumulation).
///
/// Computes the centroid and `3×3` covariance, then takes the smallest
/// eigenvalue's eigenvector as the normal (the direction of least variance). The
/// offset is `d = −n · centroid`.
fn refit_plane(cloud: &[f32], inliers: &[usize]) -> Option<Plane> {
    if inliers.len() < 3 {
        return None;
    }
    let n = inliers.len() as f64;
    let mut mean = [0.0f64; 3];
    for &i in inliers {
        let p = point(cloud, i);
        mean[0] += p[0] as f64;
        mean[1] += p[1] as f64;
        mean[2] += p[2] as f64;
    }
    mean[0] /= n;
    mean[1] /= n;
    mean[2] /= n;
    let mut cov = [0.0f64; 9];
    for &i in inliers {
        let p = point(cloud, i);
        let d = [
            p[0] as f64 - mean[0],
            p[1] as f64 - mean[1],
            p[2] as f64 - mean[2],
        ];
        for r in 0..3 {
            for c in 0..3 {
                cov[r * 3 + c] += d[r] * d[c];
            }
        }
    }
    let normal = smallest_eigenvector_sym3(&cov)?;
    let nf = [normal[0] as f32, normal[1] as f32, normal[2] as f32];
    let d = -(normal[0] * mean[0] + normal[1] * mean[1] + normal[2] * mean[2]) as f32;
    Some(Plane { normal: nf, d })
}

/// Eigenvector of the smallest eigenvalue of a symmetric `3×3` matrix.
///
/// Uses inverse power iteration on `B = (λ_max · I − A)` (so the smallest-A
/// direction becomes the dominant-B direction), seeded from three canonical
/// axes for robustness. Returns `None` if the matrix is numerically zero.
fn smallest_eigenvector_sym3(cov: &[f64; 9]) -> Option<[f64; 3]> {
    // Gershgorin upper bound on the largest eigenvalue.
    let lambda_max = {
        let mut bound = f64::NEG_INFINITY;
        for r in 0..3 {
            let radius = cov[r * 3].abs() + cov[r * 3 + 1].abs() + cov[r * 3 + 2].abs();
            bound = bound.max(radius);
        }
        bound
    };
    if !lambda_max.is_finite() || lambda_max < 1e-300 {
        return None;
    }
    let b = [
        lambda_max - cov[0],
        -cov[1],
        -cov[2],
        -cov[3],
        lambda_max - cov[4],
        -cov[5],
        -cov[6],
        -cov[7],
        lambda_max - cov[8],
    ];
    let starts = [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut best_v = None;
    let mut best_rq = f64::INFINITY;
    for start in &starts {
        let mut v = *start;
        for _ in 0..100 {
            let vn = [
                b[0] * v[0] + b[1] * v[1] + b[2] * v[2],
                b[3] * v[0] + b[4] * v[1] + b[5] * v[2],
                b[6] * v[0] + b[7] * v[1] + b[8] * v[2],
            ];
            let len = (vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]).sqrt();
            if len < 1e-14 {
                break;
            }
            v = [vn[0] / len, vn[1] / len, vn[2] / len];
        }
        let av = [
            cov[0] * v[0] + cov[1] * v[1] + cov[2] * v[2],
            cov[3] * v[0] + cov[4] * v[1] + cov[5] * v[2],
            cov[6] * v[0] + cov[7] * v[1] + cov[8] * v[2],
        ];
        let rq = v[0] * av[0] + v[1] * av[1] + v[2] * av[2];
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 0.5 && rq < best_rq {
            best_rq = rq;
            best_v = Some(v);
        }
    }
    best_v
}

/// Fit a plane to a point cloud robustly via RANSAC.
///
/// Runs up to `max_iterations` minimal-sample trials (seeded by `rng` for full
/// determinism), tracks the model with the most inliers within
/// `distance_threshold`, then refines that model with a total-least-squares fit
/// over its inlier set. `distance_threshold` must be positive.
///
/// # Errors
/// * [`Geom3dError::EmptyPointCloud`] if `cloud` is empty.
/// * [`Geom3dError::InvalidPointDim`] if `cloud.len()` is not a multiple of 3.
/// * [`Geom3dError::InvalidSampleCount`] if there are fewer than 3 points.
/// * [`Geom3dError::InvalidRadius`] if `distance_threshold` is not positive and
///   finite.
pub fn fit_plane_ransac(
    cloud: &[f32],
    distance_threshold: f32,
    max_iterations: usize,
    rng: &mut LcgRng,
) -> Geom3dResult<PlaneFitResult> {
    if cloud.is_empty() {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if cloud.len() % 3 != 0 {
        return Err(Geom3dError::InvalidPointDim {
            dim: cloud.len() % 3,
        });
    }
    let n_points = cloud.len() / 3;
    if n_points < 3 {
        return Err(Geom3dError::InvalidSampleCount {
            requested: 3,
            available: n_points,
        });
    }
    if !(distance_threshold.is_finite() && distance_threshold > 0.0) {
        return Err(Geom3dError::InvalidRadius {
            radius: distance_threshold,
        });
    }

    let count_inliers = |plane: &Plane| -> Vec<usize> {
        let mut out = Vec::new();
        for i in 0..n_points {
            if plane.signed_distance(point(cloud, i)).abs() <= distance_threshold {
                out.push(i);
            }
        }
        out
    };

    let mut best_plane: Option<Plane> = None;
    let mut best_inliers: Vec<usize> = Vec::new();
    let iterations = max_iterations.max(1);
    for _ in 0..iterations {
        // Draw three distinct indices.
        let i0 = rng.next_usize(n_points);
        let mut i1 = rng.next_usize(n_points);
        let mut guard = 0;
        while i1 == i0 && guard < 8 {
            i1 = rng.next_usize(n_points);
            guard += 1;
        }
        let mut i2 = rng.next_usize(n_points);
        guard = 0;
        while (i2 == i0 || i2 == i1) && guard < 8 {
            i2 = rng.next_usize(n_points);
            guard += 1;
        }
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }
        let candidate = match plane_from_three(point(cloud, i0), point(cloud, i1), point(cloud, i2))
        {
            Some(p) => p,
            None => continue,
        };
        let inliers = count_inliers(&candidate);
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_plane = Some(candidate);
        }
    }

    let best_plane = best_plane.ok_or(Geom3dError::Internal(
        "RANSAC found no valid plane (all samples degenerate)".into(),
    ))?;

    // Total-least-squares refinement on the inlier set, then a final inlier pass.
    let refined = refit_plane(cloud, &best_inliers).unwrap_or(best_plane);
    let final_inliers = count_inliers(&refined);
    // Keep whichever model explains more points.
    let (plane, inliers) = if final_inliers.len() >= best_inliers.len() {
        (refined, final_inliers)
    } else {
        (best_plane, best_inliers)
    };

    let n_inliers = inliers.len();
    Ok(PlaneFitResult {
        plane,
        inliers,
        n_inliers,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a planar grid on `z = 0` of `side × side` points, then optionally
    /// append `n_outliers` points far from the plane.
    fn planar_cloud(side: usize, n_outliers: usize) -> Vec<f32> {
        let mut v = Vec::new();
        for i in 0..side {
            for j in 0..side {
                v.push(i as f32 * 0.1);
                v.push(j as f32 * 0.1);
                v.push(0.0);
            }
        }
        for k in 0..n_outliers {
            v.push(k as f32 * 0.05);
            v.push(k as f32 * 0.03);
            v.push(5.0 + k as f32); // far above the plane
        }
        v
    }

    #[test]
    fn plane_from_three_axis_plane() {
        let p = plane_from_three([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            .expect("plane_from_three should succeed");
        // Normal should be ±z.
        assert!(p.normal[2].abs() > 0.999, "n={:?}", p.normal);
        assert!(p.d.abs() < 1e-6);
    }

    #[test]
    fn plane_from_three_collinear_is_none() {
        assert!(plane_from_three([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn signed_distance_sign() {
        let plane = Plane {
            normal: [0.0, 0.0, 1.0],
            d: 0.0,
        };
        assert!((plane.signed_distance([0.0, 0.0, 2.0]) - 2.0).abs() < 1e-6);
        assert!((plane.signed_distance([0.0, 0.0, -3.0]) + 3.0).abs() < 1e-6);
    }

    #[test]
    fn ransac_empty_errors() {
        let mut rng = LcgRng::new(1);
        assert!(fit_plane_ransac(&[], 0.01, 100, &mut rng).is_err());
    }

    #[test]
    fn ransac_bad_dim_errors() {
        let mut rng = LcgRng::new(1);
        assert!(fit_plane_ransac(&[0.0, 1.0], 0.01, 100, &mut rng).is_err());
    }

    #[test]
    fn ransac_too_few_points_errors() {
        let mut rng = LcgRng::new(1);
        let cloud = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // only 2 points
        assert!(fit_plane_ransac(&cloud, 0.01, 100, &mut rng).is_err());
    }

    #[test]
    fn ransac_bad_threshold_errors() {
        let mut rng = LcgRng::new(1);
        let cloud = planar_cloud(4, 0);
        assert!(fit_plane_ransac(&cloud, 0.0, 100, &mut rng).is_err());
        assert!(fit_plane_ransac(&cloud, -1.0, 100, &mut rng).is_err());
        assert!(fit_plane_ransac(&cloud, f32::NAN, 100, &mut rng).is_err());
    }

    #[test]
    fn ransac_recovers_z0_plane() {
        let mut rng = LcgRng::new(42);
        let cloud = planar_cloud(8, 0);
        let res =
            fit_plane_ransac(&cloud, 0.01, 200, &mut rng).expect("fit_plane_ransac should succeed");
        // All 64 points are inliers; the normal is ±z.
        assert_eq!(res.n_inliers, 64);
        assert!(res.plane.normal[2].abs() > 0.99, "n={:?}", res.plane.normal);
        assert!(res.plane.d.abs() < 1e-3, "d={}", res.plane.d);
    }

    #[test]
    fn ransac_rejects_outliers() {
        // 64 plane points + 10 far outliers ⇒ inliers should be exactly the 64.
        let mut rng = LcgRng::new(7);
        let cloud = planar_cloud(8, 10);
        let res =
            fit_plane_ransac(&cloud, 0.05, 400, &mut rng).expect("fit_plane_ransac should succeed");
        assert_eq!(res.n_inliers, 64, "should exclude the 10 outliers");
        // Every inlier index is < 64 (the planar block).
        assert!(res.inliers.iter().all(|&i| i < 64));
    }

    #[test]
    fn ransac_tilted_plane() {
        // Plane x + y + z = 0 normal (1,1,1)/√3.
        let mut v = Vec::new();
        let inv = 1.0f32 / 3.0f32.sqrt();
        for i in 0..8 {
            for j in 0..8 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let z = -(x + y); // on the plane
                v.push(x);
                v.push(y);
                v.push(z);
            }
        }
        let mut rng = LcgRng::new(123);
        let res =
            fit_plane_ransac(&v, 0.01, 300, &mut rng).expect("fit_plane_ransac should succeed");
        assert_eq!(res.n_inliers, 64);
        // Normal aligned with (1,1,1)/√3 up to sign.
        let dot =
            (res.plane.normal[0] * inv + res.plane.normal[1] * inv + res.plane.normal[2] * inv)
                .abs();
        assert!(dot > 0.99, "normal={:?} dot={dot}", res.plane.normal);
    }

    #[test]
    fn ransac_deterministic_same_seed() {
        let cloud = planar_cloud(6, 5);
        let mut a = LcgRng::new(99);
        let mut b = LcgRng::new(99);
        let ra =
            fit_plane_ransac(&cloud, 0.02, 100, &mut a).expect("fit_plane_ransac should succeed");
        let rb =
            fit_plane_ransac(&cloud, 0.02, 100, &mut b).expect("fit_plane_ransac should succeed");
        assert_eq!(ra.n_inliers, rb.n_inliers);
        assert_eq!(ra.inliers, rb.inliers);
    }

    #[test]
    fn ransac_refit_improves_normal_under_noise() {
        // Planar block with small z-noise; TLS refit should yield a near-z normal.
        let mut v = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let idx: usize = i * 10 + j;
                let noise = ((idx.wrapping_mul(2654435761)) & 0xff) as f32 / 255.0 * 0.004 - 0.002;
                v.push(i as f32 * 0.1);
                v.push(j as f32 * 0.1);
                v.push(noise);
            }
        }
        let mut rng = LcgRng::new(2024);
        let res =
            fit_plane_ransac(&v, 0.01, 300, &mut rng).expect("fit_plane_ransac should succeed");
        assert!(res.n_inliers >= 95, "inliers={}", res.n_inliers);
        assert!(
            res.plane.normal[2].abs() > 0.99,
            "refined normal={:?}",
            res.plane.normal
        );
    }

    #[test]
    fn ransac_inliers_within_threshold() {
        let mut rng = LcgRng::new(55);
        let cloud = planar_cloud(7, 3);
        let res =
            fit_plane_ransac(&cloud, 0.02, 200, &mut rng).expect("fit_plane_ransac should succeed");
        for &i in &res.inliers {
            let dist = res.plane.signed_distance(point(&cloud, i)).abs();
            assert!(
                dist <= 0.02 + 1e-5,
                "inlier {i} dist {dist} exceeds threshold"
            );
        }
    }
}

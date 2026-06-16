//! Axis-aligned (AABB) and oriented (OBB) bounding boxes with PCA fitting and
//! intersection tests (interval overlap for AABBs, the separating-axis theorem
//! for OBBs).
//!
//! All geometry is `f64`. An OBB is stored by its centre, three orthonormal axes,
//! and the half-extent along each axis, following Ericson's *Real-Time Collision
//! Detection* (§4.4).
//!
//! # References
//! - Ericson, C. (2005). *Real-Time Collision Detection*, §4.2 (AABB),
//!   §4.4 (OBB), §4.4.1 (OBB-OBB separating-axis test).
//! - Gottschalk, S., Lin, M.C. & Manocha, D. (1996). "OBBTree: A hierarchical
//!   structure for rapid interference detection". *SIGGRAPH '96*.

use crate::error::{Geom3dError, Geom3dResult};

/// Axis-aligned bounding box `[min, max]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Per-axis minimum corner.
    pub min: [f64; 3],
    /// Per-axis maximum corner.
    pub max: [f64; 3],
}

impl Aabb {
    /// Build the tight AABB of a flat `[n×3]` point buffer.
    ///
    /// # Errors
    /// - [`Geom3dError::EmptyPointCloud`] if `n == 0`.
    /// - [`Geom3dError::DimensionMismatch`] if `points.len() != 3 * n`.
    pub fn from_points(points: &[f64], n: usize) -> Geom3dResult<Self> {
        if n == 0 {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if points.len() != 3 * n {
            return Err(Geom3dError::DimensionMismatch {
                expected: 3 * n,
                got: points.len(),
            });
        }
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for i in 0..n {
            for d in 0..3 {
                let v = points[3 * i + d];
                min[d] = min[d].min(v);
                max[d] = max[d].max(v);
            }
        }
        Ok(Self { min, max })
    }

    /// Box centre.
    #[inline]
    pub fn center(&self) -> [f64; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Per-axis half-extents.
    #[inline]
    pub fn half_extents(&self) -> [f64; 3] {
        [
            0.5 * (self.max[0] - self.min[0]),
            0.5 * (self.max[1] - self.min[1]),
            0.5 * (self.max[2] - self.min[2]),
        ]
    }

    /// Box volume.
    #[inline]
    pub fn volume(&self) -> f64 {
        (self.max[0] - self.min[0]).max(0.0)
            * (self.max[1] - self.min[1]).max(0.0)
            * (self.max[2] - self.min[2]).max(0.0)
    }

    /// Whether a point lies inside (or on the boundary of) the box.
    pub fn contains(&self, p: [f64; 3]) -> bool {
        (0..3).all(|d| p[d] >= self.min[d] && p[d] <= self.max[d])
    }

    /// AABB-AABB overlap test (touching boxes count as intersecting).
    pub fn intersects(&self, other: &Aabb) -> bool {
        (0..3).all(|d| self.min[d] <= other.max[d] && self.max[d] >= other.min[d])
    }
}

/// Oriented bounding box: centre + orthonormal axes + half-extents.
#[derive(Debug, Clone, Copy)]
pub struct Obb {
    /// Box centre.
    pub center: [f64; 3],
    /// Three orthonormal local axes (rows): `axes[k]` is the `k`-th axis.
    pub axes: [[f64; 3]; 3],
    /// Half-extent along each local axis.
    pub half_extents: [f64; 3],
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

impl Obb {
    /// Construct an OBB from an [`Aabb`] (axis-aligned ⇒ identity orientation).
    pub fn from_aabb(aabb: &Aabb) -> Self {
        Self {
            center: aabb.center(),
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            half_extents: aabb.half_extents(),
        }
    }

    /// Fit an OBB to a point cloud by principal-component analysis: the box axes
    /// are the eigenvectors of the `3×3` covariance matrix, and the extents are
    /// the projection ranges of the points onto those axes.
    ///
    /// # Errors
    /// - [`Geom3dError::EmptyPointCloud`] if `n == 0`.
    /// - [`Geom3dError::DimensionMismatch`] if `points.len() != 3 * n`.
    pub fn fit_pca(points: &[f64], n: usize) -> Geom3dResult<Self> {
        if n == 0 {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if points.len() != 3 * n {
            return Err(Geom3dError::DimensionMismatch {
                expected: 3 * n,
                got: points.len(),
            });
        }

        // Mean.
        let mut mean = [0.0_f64; 3];
        for i in 0..n {
            for d in 0..3 {
                mean[d] += points[3 * i + d];
            }
        }
        for d in &mut mean {
            *d /= n as f64;
        }

        // Covariance (symmetric 3×3).
        let mut cov = [[0.0_f64; 3]; 3];
        for i in 0..n {
            let c = [
                points[3 * i] - mean[0],
                points[3 * i + 1] - mean[1],
                points[3 * i + 2] - mean[2],
            ];
            for r in 0..3 {
                for col in 0..3 {
                    cov[r][col] += c[r] * c[col];
                }
            }
        }
        let inv_n = 1.0 / n as f64;
        for row in &mut cov {
            for entry in row.iter_mut() {
                *entry *= inv_n;
            }
        }

        let mut axes = jacobi_eigen_3x3(cov);
        // Orthonormalise defensively (Jacobi yields orthonormal vectors, but
        // re-normalise to wipe out round-off and guarantee a right-handed frame).
        for axis in &mut axes {
            let l = norm(*axis);
            if l > 1e-300 {
                axis[0] /= l;
                axis[1] /= l;
                axis[2] /= l;
            }
        }
        // Make the frame right-handed: axes[2] = axes[0] × axes[1].
        axes[2] = cross(axes[0], axes[1]);

        // Project points onto axes to get min/max extents and the true centre.
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for i in 0..n {
            let p = [points[3 * i], points[3 * i + 1], points[3 * i + 2]];
            for (k, axis) in axes.iter().enumerate() {
                let proj = dot(p, *axis);
                lo[k] = lo[k].min(proj);
                hi[k] = hi[k].max(proj);
            }
        }
        let half_extents = [
            0.5 * (hi[0] - lo[0]),
            0.5 * (hi[1] - lo[1]),
            0.5 * (hi[2] - lo[2]),
        ];
        // Centre is the midpoint of the projected ranges, mapped back to world.
        let mid = [
            0.5 * (hi[0] + lo[0]),
            0.5 * (hi[1] + lo[1]),
            0.5 * (hi[2] + lo[2]),
        ];
        let center = [
            mid[0] * axes[0][0] + mid[1] * axes[1][0] + mid[2] * axes[2][0],
            mid[0] * axes[0][1] + mid[1] * axes[1][1] + mid[2] * axes[2][1],
            mid[0] * axes[0][2] + mid[1] * axes[1][2] + mid[2] * axes[2][2],
        ];

        Ok(Self {
            center,
            axes,
            half_extents,
        })
    }

    /// OBB volume.
    #[inline]
    pub fn volume(&self) -> f64 {
        8.0 * self.half_extents[0] * self.half_extents[1] * self.half_extents[2]
    }

    /// Whether a world-space point lies inside the OBB.
    pub fn contains(&self, p: [f64; 3]) -> bool {
        let d = sub(p, self.center);
        (0..3).all(|k| dot(d, self.axes[k]).abs() <= self.half_extents[k] + 1e-12)
    }

    /// OBB-OBB intersection via the separating-axis theorem (15 candidate axes:
    /// the 3 face normals of each box plus the 9 edge-edge cross products).
    ///
    /// Returns `true` if the boxes overlap (touching counts as overlapping).
    pub fn intersects(&self, other: &Obb) -> bool {
        const EPS: f64 = 1e-9;
        // Rotation matrix expressing other's axes in this box's frame: R[i][j] =
        // a_i · b_j.
        let mut r = [[0.0_f64; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                *entry = dot(self.axes[i], other.axes[j]);
            }
        }
        // Translation between centres, in this box's frame.
        let t_world = sub(other.center, self.center);
        let t = [
            dot(t_world, self.axes[0]),
            dot(t_world, self.axes[1]),
            dot(t_world, self.axes[2]),
        ];
        // Absolute rotation matrix with an epsilon to counter near-parallel edges.
        let mut abs_r = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                abs_r[i][j] = r[i][j].abs() + EPS;
            }
        }
        let a = self.half_extents;
        let b = other.half_extents;

        // L = A0, A1, A2 (this box's face normals).
        for i in 0..3 {
            let ra = a[i];
            let rb = b[0] * abs_r[i][0] + b[1] * abs_r[i][1] + b[2] * abs_r[i][2];
            if t[i].abs() > ra + rb {
                return false;
            }
        }
        // L = B0, B1, B2 (other box's face normals).
        for j in 0..3 {
            let ra = a[0] * abs_r[0][j] + a[1] * abs_r[1][j] + a[2] * abs_r[2][j];
            let rb = b[j];
            let tj = t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j];
            if tj.abs() > ra + rb {
                return false;
            }
        }

        // Nine edge-edge axes L = A_i × B_j.
        // A0 × B0..B2
        if (t[2] * r[1][0] - t[1] * r[2][0]).abs()
            > a[1] * abs_r[2][0] + a[2] * abs_r[1][0] + b[1] * abs_r[0][2] + b[2] * abs_r[0][1]
        {
            return false;
        }
        if (t[2] * r[1][1] - t[1] * r[2][1]).abs()
            > a[1] * abs_r[2][1] + a[2] * abs_r[1][1] + b[0] * abs_r[0][2] + b[2] * abs_r[0][0]
        {
            return false;
        }
        if (t[2] * r[1][2] - t[1] * r[2][2]).abs()
            > a[1] * abs_r[2][2] + a[2] * abs_r[1][2] + b[0] * abs_r[0][1] + b[1] * abs_r[0][0]
        {
            return false;
        }
        // A1 × B0..B2
        if (t[0] * r[2][0] - t[2] * r[0][0]).abs()
            > a[0] * abs_r[2][0] + a[2] * abs_r[0][0] + b[1] * abs_r[1][2] + b[2] * abs_r[1][1]
        {
            return false;
        }
        if (t[0] * r[2][1] - t[2] * r[0][1]).abs()
            > a[0] * abs_r[2][1] + a[2] * abs_r[0][1] + b[0] * abs_r[1][2] + b[2] * abs_r[1][0]
        {
            return false;
        }
        if (t[0] * r[2][2] - t[2] * r[0][2]).abs()
            > a[0] * abs_r[2][2] + a[2] * abs_r[0][2] + b[0] * abs_r[1][1] + b[1] * abs_r[1][0]
        {
            return false;
        }
        // A2 × B0..B2
        if (t[1] * r[0][0] - t[0] * r[1][0]).abs()
            > a[0] * abs_r[1][0] + a[1] * abs_r[0][0] + b[1] * abs_r[2][2] + b[2] * abs_r[2][1]
        {
            return false;
        }
        if (t[1] * r[0][1] - t[0] * r[1][1]).abs()
            > a[0] * abs_r[1][1] + a[1] * abs_r[0][1] + b[0] * abs_r[2][2] + b[2] * abs_r[2][0]
        {
            return false;
        }
        if (t[1] * r[0][2] - t[0] * r[1][2]).abs()
            > a[0] * abs_r[1][2] + a[1] * abs_r[0][2] + b[0] * abs_r[2][1] + b[1] * abs_r[2][0]
        {
            return false;
        }

        true
    }
}

/// Symmetric `3×3` Jacobi eigenvalue decomposition; returns the three
/// eigenvectors as rows, sorted by descending eigenvalue.
fn jacobi_eigen_3x3(mut a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Accumulated rotation (eigenvectors as columns of `v`).
    let mut v = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        // Find the largest off-diagonal magnitude.
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = a[0][1].abs();
        if a[0][2].abs() > max_off {
            max_off = a[0][2].abs();
            p = 0;
            q = 2;
        }
        if a[1][2].abs() > max_off {
            max_off = a[1][2].abs();
            p = 1;
            q = 2;
        }
        if max_off < 1e-15 {
            break;
        }
        // Jacobi rotation angle that zeroes a[p][q]. With the rotation
        // J = [[c, s], [-s, c]] in the (p, q) plane, applied as A ← Jᵀ A J, the
        // updated off-diagonal is a'[p][q] = ½ sin(2θ)(a_pp − a_qq) + cos(2θ) a_pq.
        // Forcing it to zero gives tan(2θ) = 2 a_pq / (a_qq − a_pp), i.e.
        //   θ = ½ atan2(2 a_pq, a_qq − a_pp).
        // (Using a_pp − a_qq here would flip the sign of θ and *double* the
        // off-diagonal each step instead of annihilating it.)
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let theta = if apq.abs() < 1e-300 {
            0.0
        } else {
            0.5 * f64::atan2(2.0 * apq, aqq - app)
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation J^T A J.
        let mut a_new = a;
        for i in 0..3 {
            a_new[i][p] = c * a[i][p] - s * a[i][q];
            a_new[i][q] = s * a[i][p] + c * a[i][q];
        }
        let a_tmp = a_new;
        for i in 0..3 {
            a_new[p][i] = c * a_tmp[p][i] - s * a_tmp[q][i];
            a_new[q][i] = s * a_tmp[p][i] + c * a_tmp[q][i];
        }
        a = a_new;

        // Accumulate eigenvectors.
        let mut v_new = v;
        for i in 0..3 {
            v_new[i][p] = c * v[i][p] - s * v[i][q];
            v_new[i][q] = s * v[i][p] + c * v[i][q];
        }
        v = v_new;
    }

    // Eigenvalues are the diagonal; sort axes by descending eigenvalue.
    let mut idx = [0usize, 1, 2];
    let eig = [a[0][0], a[1][1], a[2][2]];
    idx.sort_by(|&i, &j| {
        eig[j]
            .partial_cmp(&eig[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Eigenvectors are columns of v; return them as rows in sorted order.
    let mut out = [[0.0_f64; 3]; 3];
    for (k, &col) in idx.iter().enumerate() {
        out[k] = [v[0][col], v[1][col], v[2][col]];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_scaled(sx: f64, sy: f64, sz: f64) -> Vec<f64> {
        let mut v = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    v.push(x as f64 * sx);
                    v.push(y as f64 * sy);
                    v.push(z as f64 * sz);
                }
            }
        }
        v
    }

    // ── AABB ──────────────────────────────────────────────────────────────────

    #[test]
    fn aabb_from_points_bounds() {
        let pts = cube_scaled(2.0, 3.0, 4.0);
        let aabb = Aabb::from_points(&pts, 8).expect("aabb");
        assert_eq!(aabb.min, [0.0, 0.0, 0.0]);
        assert_eq!(aabb.max, [2.0, 3.0, 4.0]);
        assert!((aabb.volume() - 24.0).abs() < 1e-12);
        assert_eq!(aabb.center(), [1.0, 1.5, 2.0]);
    }

    #[test]
    fn aabb_empty_and_bad_length_error() {
        assert!(matches!(
            Aabb::from_points(&[], 0),
            Err(Geom3dError::EmptyPointCloud)
        ));
        assert!(matches!(
            Aabb::from_points(&[0.0, 0.0], 4),
            Err(Geom3dError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn aabb_contains_and_intersects() {
        let a = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        assert!(a.contains([0.5, 0.5, 0.5]));
        assert!(!a.contains([1.5, 0.5, 0.5]));
        let b = Aabb {
            min: [0.5, 0.5, 0.5],
            max: [2.0, 2.0, 2.0],
        };
        assert!(a.intersects(&b));
        let c = Aabb {
            min: [2.0, 2.0, 2.0],
            max: [3.0, 3.0, 3.0],
        };
        assert!(!a.intersects(&c));
        // Touching faces count as intersecting.
        let d = Aabb {
            min: [1.0, 0.0, 0.0],
            max: [2.0, 1.0, 1.0],
        };
        assert!(a.intersects(&d));
    }

    // ── OBB PCA fit ─────────────────────────────────────────────────────────────

    #[test]
    fn obb_from_aabb_matches() {
        let aabb = Aabb {
            min: [-1.0, -2.0, -3.0],
            max: [1.0, 2.0, 3.0],
        };
        let obb = Obb::from_aabb(&aabb);
        assert_eq!(obb.center, [0.0, 0.0, 0.0]);
        assert_eq!(obb.half_extents, [1.0, 2.0, 3.0]);
        assert!((obb.volume() - aabb.volume()).abs() < 1e-12);
    }

    #[test]
    fn obb_pca_axis_aligned_box() {
        // A box stretched along x should yield half-extents matching its shape,
        // and a centre at the box centre.
        let pts = cube_scaled(4.0, 1.0, 1.0);
        let obb = Obb::fit_pca(&pts, 8).expect("obb");
        // Volume should match the AABB volume (4·1·1 = 4) since axis-aligned.
        assert!((obb.volume() - 4.0).abs() < 1e-6, "vol={}", obb.volume());
        // Largest extent direction must be the x-axis (±).
        let principal = obb.axes[0];
        assert!(
            principal[0].abs() > 0.99,
            "principal axis should be x, got {principal:?}"
        );
        // Centre should be at (2, 0.5, 0.5).
        assert!((obb.center[0] - 2.0).abs() < 1e-6);
        assert!((obb.center[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn obb_pca_rotated_box_tight() {
        // Build a thin box, then rotate it 45° about z. PCA should recover a tight
        // OBB whose volume matches the un-rotated AABB volume.
        let base = cube_scaled(4.0, 1.0, 2.0);
        let n = 8;
        let theta = std::f64::consts::FRAC_PI_4;
        let (c, s) = (theta.cos(), theta.sin());
        let mut pts = vec![0.0; n * 3];
        for i in 0..n {
            let x = base[3 * i];
            let y = base[3 * i + 1];
            let z = base[3 * i + 2];
            pts[3 * i] = c * x - s * y;
            pts[3 * i + 1] = s * x + c * y;
            pts[3 * i + 2] = z;
        }
        let obb = Obb::fit_pca(&pts, n).expect("obb");
        assert!(
            (obb.volume() - 8.0).abs() < 1e-5,
            "tight OBB volume should equal 4·1·2 = 8, got {}",
            obb.volume()
        );
        // Every original point must be inside the fitted OBB.
        for i in 0..n {
            let p = [pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]];
            assert!(obb.contains(p), "point {i} outside fitted OBB");
        }
    }

    #[test]
    fn obb_empty_and_bad_length_error() {
        assert!(matches!(
            Obb::fit_pca(&[], 0),
            Err(Geom3dError::EmptyPointCloud)
        ));
        assert!(matches!(
            Obb::fit_pca(&[0.0, 0.0], 4),
            Err(Geom3dError::DimensionMismatch { .. })
        ));
    }

    // ── OBB-OBB SAT ─────────────────────────────────────────────────────────────

    fn unit_obb(center: [f64; 3]) -> Obb {
        Obb {
            center,
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            half_extents: [0.5, 0.5, 0.5],
        }
    }

    #[test]
    fn obb_overlap_when_coincident() {
        let a = unit_obb([0.0, 0.0, 0.0]);
        let b = unit_obb([0.0, 0.0, 0.0]);
        assert!(a.intersects(&b));
    }

    #[test]
    fn obb_separated_along_axis() {
        let a = unit_obb([0.0, 0.0, 0.0]);
        let b = unit_obb([2.0, 0.0, 0.0]); // gap of 1 between faces
        assert!(!a.intersects(&b));
        let c = unit_obb([0.9, 0.0, 0.0]); // overlapping
        assert!(a.intersects(&c));
    }

    #[test]
    fn obb_rotated_edge_separation() {
        // A unit box and a 45°-rotated unit box whose centres are far enough that
        // an edge-edge axis separates them.
        let a = unit_obb([0.0, 0.0, 0.0]);
        let theta = std::f64::consts::FRAC_PI_4;
        let (c, s) = (theta.cos(), theta.sin());
        let rotated = Obb {
            center: [1.8, 0.0, 0.0],
            axes: [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
            half_extents: [0.5, 0.5, 0.5],
        };
        // Rotated box has half-diagonal ~0.707 in x; centre at 1.8 ⇒ separated.
        assert!(!a.intersects(&rotated));
        // Move it closer so they overlap.
        let close = Obb {
            center: [1.0, 0.0, 0.0],
            axes: [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
            half_extents: [0.5, 0.5, 0.5],
        };
        assert!(a.intersects(&close));
    }

    #[test]
    fn obb_contains_point() {
        let theta = std::f64::consts::FRAC_PI_4;
        let (c, s) = (theta.cos(), theta.sin());
        let obb = Obb {
            center: [0.0, 0.0, 0.0],
            axes: [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
            half_extents: [1.0, 0.5, 0.5],
        };
        // Point along the local x-axis at distance 0.9 < 1.0 ⇒ inside.
        let p = [0.9 * c, 0.9 * s, 0.0];
        assert!(obb.contains(p));
        // Just past the local x half-extent ⇒ outside.
        let q = [1.1 * c, 1.1 * s, 0.0];
        assert!(!obb.contains(q));
    }

    #[test]
    fn jacobi_diagonal_identity() {
        // Eigen of a diagonal matrix returns its axes (largest eigenvalue first).
        let diag = [[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let axes = jacobi_eigen_3x3(diag);
        // Principal axis (eig 5) is x, then y, then z (up to sign).
        assert!(axes[0][0].abs() > 0.99);
        assert!(axes[1][1].abs() > 0.99);
        assert!(axes[2][2].abs() > 0.99);
    }
}

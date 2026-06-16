//! Incremental 3D convex hull (and a 2D Andrew's-monotone-chain fallback).
//!
//! The 3D routine builds the hull directly by the incremental / "beneath-beyond"
//! method: seed an initial tetrahedron, then add the remaining points one at a
//! time, deleting every face the new point can "see" and stitching the resulting
//! horizon loop to the point with fresh outward-oriented triangles. This is
//! distinct from [`crate::mesh::delaunay3d`], which extracts hull faces as a
//! by-product of a full Bowyer-Watson tetrahedralization; the algorithm here
//! never builds interior tetrahedra.
//!
//! All geometry is `f64`. The robust signed-volume predicate
//! [`crate::mesh::delaunay3d::orient3d`] is reused so the two modules agree on
//! coplanarity handling.
//!
//! # References
//! - Preparata, F.P. & Shamos, M.I. (1985). *Computational Geometry: An
//!   Introduction*, §3.4 (beneath-beyond).
//! - Andrew, A.M. (1979). "Another efficient algorithm for convex hulls in two
//!   dimensions". *Inf. Process. Lett.* 9(5):216-219 (the 2D monotone chain).

use crate::error::{Geom3dError, Geom3dResult};
use crate::mesh::delaunay3d::orient3d;

/// A triangular face of a convex hull, as an index triple into the input cloud.
///
/// Faces are oriented counter-clockwise when viewed from *outside* the hull, so
/// the outward normal is `(v1−v0) × (v2−v0)`.
pub type HullFace = [usize; 3];

/// Result of a 3D convex-hull construction.
#[derive(Debug, Clone)]
pub struct ConvexHull3d {
    /// Outward-oriented triangular faces (indices into the original point cloud).
    pub faces: Vec<HullFace>,
    /// Indices of the points that lie on the hull (the extreme points).
    pub vertices: Vec<usize>,
}

#[inline]
fn point(points: &[f64], i: usize) -> [f64; 3] {
    [points[3 * i], points[3 * i + 1], points[3 * i + 2]]
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Build the 3D convex hull of a flat point buffer `points = [n×3]`.
///
/// Returns outward-oriented triangular faces and the list of extreme vertices.
/// Degenerate inputs (all points coplanar or collinear) are rejected, since a
/// 3D hull is not defined; use [`convex_hull_2d`] for planar data.
///
/// # Errors
/// - [`Geom3dError::EmptyPointCloud`] if `n == 0`.
/// - [`Geom3dError::DimensionMismatch`] if `points.len() != 3 * n`.
/// - [`Geom3dError::InvalidTopology`] if fewer than 4 points or all points are
///   coplanar (no full-dimensional hull).
pub fn convex_hull_3d(points: &[f64], n: usize) -> Geom3dResult<ConvexHull3d> {
    if n == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if points.len() != 3 * n {
        return Err(Geom3dError::DimensionMismatch {
            expected: 3 * n,
            got: points.len(),
        });
    }
    if n < 4 {
        return Err(Geom3dError::InvalidTopology {
            reason: "convex_hull_3d needs at least 4 points",
        });
    }

    // ── 1. Seed tetrahedron: pick four affinely-independent points. ───────────
    // i0/i1: the two points with extreme x (guaranteed distinct unless all equal).
    let mut i0 = 0usize;
    let mut i1 = 0usize;
    for i in 1..n {
        if points[3 * i] < points[3 * i0] {
            i0 = i;
        }
        if points[3 * i] > points[3 * i1] {
            i1 = i;
        }
    }
    if i0 == i1 {
        return Err(Geom3dError::InvalidTopology {
            reason: "convex_hull_3d: all points coincide",
        });
    }
    let p0 = point(points, i0);
    let p1 = point(points, i1);

    // i2: farthest point from the line p0-p1 (max |(p-p0) × (p1-p0)|).
    let line = sub(p1, p0);
    let mut i2 = usize::MAX;
    let mut best = 0.0_f64;
    for i in 0..n {
        if i == i0 || i == i1 {
            continue;
        }
        let c = cross(sub(point(points, i), p0), line);
        let d = dot(c, c);
        if d > best {
            best = d;
            i2 = i;
        }
    }
    if i2 == usize::MAX {
        return Err(Geom3dError::InvalidTopology {
            reason: "convex_hull_3d: all points are collinear",
        });
    }
    let p2 = point(points, i2);

    // i3: farthest point from the plane (p0,p1,p2) by |orient3d|.
    let mut i3 = usize::MAX;
    let mut best_vol = 0.0_f64;
    for i in 0..n {
        if i == i0 || i == i1 || i == i2 {
            continue;
        }
        let v = orient3d(p0, p1, p2, point(points, i)).abs();
        if v > best_vol {
            best_vol = v;
            i3 = i;
        }
    }
    if i3 == usize::MAX {
        return Err(Geom3dError::InvalidTopology {
            reason: "convex_hull_3d: all points are coplanar",
        });
    }
    let p3 = point(points, i3);

    // Build the seed tetrahedron's four faces, each oriented so its outward
    // normal points away from the opposite vertex.
    let mut faces: Vec<HullFace> = Vec::new();
    let mut add_face = |a: usize, b: usize, c: usize, opposite: [f64; 3]| {
        let va = point(points, a);
        let vb = point(points, b);
        let vc = point(points, c);
        // Ensure `opposite` is on the negative side of (a,b,c).
        if orient3d(va, vb, vc, opposite) > 0.0 {
            faces.push([a, c, b]);
        } else {
            faces.push([a, b, c]);
        }
    };
    add_face(i0, i1, i2, p3);
    add_face(i0, i1, i3, p2);
    add_face(i0, i2, i3, p1);
    add_face(i1, i2, i3, p0);

    // ── 2. Incrementally add the remaining points. ───────────────────────────
    let seed = [i0, i1, i2, i3];
    for i in 0..n {
        if seed.contains(&i) {
            continue;
        }
        let p = point(points, i);

        // Collect faces visible from p (p strictly above the face plane).
        let mut visible = vec![false; faces.len()];
        let mut any_visible = false;
        for (fi, f) in faces.iter().enumerate() {
            let va = point(points, f[0]);
            let vb = point(points, f[1]);
            let vc = point(points, f[2]);
            // Outward normal is (vb-va)×(vc-va); p is "above" if orient3d>0
            // (p on the same side as the outward normal).
            if orient3d(va, vb, vc, p) > 0.0 {
                visible[fi] = true;
                any_visible = true;
            }
        }
        if !any_visible {
            continue; // p is inside or on the hull.
        }

        // Horizon = edges shared by exactly one visible and one hidden face.
        // Count directed edges of visible faces; a boundary edge appears once.
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for (fi, f) in faces.iter().enumerate() {
            if !visible[fi] {
                continue;
            }
            let edges = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])];
            for &(a, b) in &edges {
                // The opposite directed edge (b,a) belongs to the neighbouring
                // face. If that neighbour is NOT visible, (a,b) is on the horizon.
                let neighbour_visible = faces.iter().enumerate().any(|(gj, g)| {
                    visible[gj]
                        && gj != fi
                        && [(g[0], g[1]), (g[1], g[2]), (g[2], g[0])].contains(&(b, a))
                });
                if !neighbour_visible {
                    horizon.push((a, b));
                }
            }
        }

        // Remove visible faces.
        let mut kept: Vec<HullFace> = Vec::with_capacity(faces.len());
        for (fi, f) in faces.iter().enumerate() {
            if !visible[fi] {
                kept.push(*f);
            }
        }
        faces = kept;

        // Stitch each horizon edge (a,b) to p as a new face (a,b,i). The horizon
        // edge is oriented along the boundary of the removed cap, so (a,b,i)
        // already faces outward.
        for (a, b) in horizon {
            faces.push([a, b, i]);
        }
    }

    // ── 3. Collect the distinct hull vertices. ───────────────────────────────
    let mut seen = vec![false; n];
    let mut vertices = Vec::new();
    for f in &faces {
        for &v in f {
            if !seen[v] {
                seen[v] = true;
                vertices.push(v);
            }
        }
    }
    vertices.sort_unstable();

    Ok(ConvexHull3d { faces, vertices })
}

impl ConvexHull3d {
    /// Number of hull faces.
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Total surface area of the hull.
    pub fn surface_area(&self, points: &[f64]) -> f64 {
        let mut area = 0.0_f64;
        for f in &self.faces {
            let a = point(points, f[0]);
            let b = point(points, f[1]);
            let c = point(points, f[2]);
            let n = cross(sub(b, a), sub(c, a));
            area += 0.5 * dot(n, n).sqrt();
        }
        area
    }

    /// Enclosed volume of the hull via the divergence-theorem tetrahedron sum
    /// `Σ (v0 · (v1 × v2)) / 6` over outward-oriented faces.
    pub fn volume(&self, points: &[f64]) -> f64 {
        let mut vol = 0.0_f64;
        for f in &self.faces {
            let a = point(points, f[0]);
            let b = point(points, f[1]);
            let c = point(points, f[2]);
            vol += dot(a, cross(b, c));
        }
        (vol / 6.0).abs()
    }
}

/// Andrew's monotone-chain convex hull of a planar point set.
///
/// Input is a flat `[n×2]` buffer; the result is the indices of the hull points
/// in counter-clockwise order, starting from the lexicographically smallest
/// point, **without** repeating the first point at the end.
///
/// # Errors
/// - [`Geom3dError::EmptyPointCloud`] if `n == 0`.
/// - [`Geom3dError::DimensionMismatch`] if `points.len() != 2 * n`.
pub fn convex_hull_2d(points: &[f64], n: usize) -> Geom3dResult<Vec<usize>> {
    if n == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if points.len() != 2 * n {
        return Err(Geom3dError::DimensionMismatch {
            expected: 2 * n,
            got: points.len(),
        });
    }
    if n <= 2 {
        return Ok((0..n).collect());
    }

    // Sort indices lexicographically by (x, y).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        let (xi, yi) = (points[2 * i], points[2 * i + 1]);
        let (xj, yj) = (points[2 * j], points[2 * j + 1]);
        xi.partial_cmp(&xj)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(yi.partial_cmp(&yj).unwrap_or(std::cmp::Ordering::Equal))
    });

    // 2D cross product of OA×OB for orientation (>0 ⇒ counter-clockwise turn).
    let cross2 = |o: usize, a: usize, b: usize| -> f64 {
        let ox = points[2 * o];
        let oy = points[2 * o + 1];
        (points[2 * a] - ox) * (points[2 * b + 1] - oy)
            - (points[2 * a + 1] - oy) * (points[2 * b] - ox)
    };

    let mut hull: Vec<usize> = Vec::with_capacity(2 * n);
    // Lower hull.
    for &idx in &order {
        while hull.len() >= 2 && cross2(hull[hull.len() - 2], hull[hull.len() - 1], idx) <= 0.0 {
            hull.pop();
        }
        hull.push(idx);
    }
    // Upper hull.
    let lower_len = hull.len() + 1;
    for &idx in order.iter().rev().skip(1) {
        while hull.len() >= lower_len
            && cross2(hull[hull.len() - 2], hull[hull.len() - 1], idx) <= 0.0
        {
            hull.pop();
        }
        hull.push(idx);
    }
    // The last point equals the first; drop it.
    hull.pop();
    Ok(hull)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The eight corners of the unit cube as a flat buffer.
    fn cube() -> Vec<f64> {
        let mut v = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    v.push(x as f64);
                    v.push(y as f64);
                    v.push(z as f64);
                }
            }
        }
        v
    }

    #[test]
    fn hull3d_rejects_small_input() {
        assert!(matches!(
            convex_hull_3d(&[0.0, 0.0, 0.0], 1),
            Err(Geom3dError::InvalidTopology { .. })
        ));
        assert!(matches!(
            convex_hull_3d(&[], 0),
            Err(Geom3dError::EmptyPointCloud)
        ));
    }

    #[test]
    fn hull3d_rejects_bad_length() {
        assert!(matches!(
            convex_hull_3d(&[0.0, 0.0], 4),
            Err(Geom3dError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn hull3d_rejects_coplanar() {
        // Four points in the z=0 plane — no 3D hull.
        let pts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        assert!(matches!(
            convex_hull_3d(&pts, 4),
            Err(Geom3dError::InvalidTopology { .. })
        ));
    }

    #[test]
    fn hull3d_tetrahedron_has_four_faces() {
        let pts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let hull = convex_hull_3d(&pts, 4).expect("hull");
        assert_eq!(hull.n_faces(), 4);
        assert_eq!(hull.vertices.len(), 4);
    }

    #[test]
    fn hull3d_cube_euler_characteristic() {
        // A triangulated cube hull: V − E + F = 2 (Euler). 8 vertices, 12 faces.
        let pts = cube();
        let hull = convex_hull_3d(&pts, 8).expect("hull");
        assert_eq!(hull.vertices.len(), 8, "all 8 cube corners are extreme");
        assert_eq!(hull.n_faces(), 12, "triangulated cube has 12 faces");
        // Each triangle has 3 edges, each shared by 2 faces ⇒ E = 3F/2 = 18.
        let e = 3 * hull.n_faces() / 2;
        let v = hull.vertices.len();
        assert_eq!(v as i64 - e as i64 + hull.n_faces() as i64, 2);
    }

    #[test]
    fn hull3d_cube_volume_and_area() {
        let pts = cube();
        let hull = convex_hull_3d(&pts, 8).expect("hull");
        assert!(
            (hull.volume(&pts) - 1.0).abs() < 1e-9,
            "unit cube volume = 1"
        );
        assert!(
            (hull.surface_area(&pts) - 6.0).abs() < 1e-9,
            "cube area = 6"
        );
    }

    #[test]
    fn hull3d_ignores_interior_points() {
        // Cube corners + a point at the centre; the centre must not be a vertex.
        let mut pts = cube();
        pts.extend_from_slice(&[0.5, 0.5, 0.5]);
        let hull = convex_hull_3d(&pts, 9).expect("hull");
        assert_eq!(hull.vertices.len(), 8);
        assert!(!hull.vertices.contains(&8), "centre point is interior");
        assert!((hull.volume(&pts) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hull3d_faces_outward_oriented() {
        // For every face, the hull centroid must be on the negative (inside) side.
        let pts = cube();
        let hull = convex_hull_3d(&pts, 8).expect("hull");
        let centroid = {
            let mut c = [0.0_f64; 3];
            for i in 0..8 {
                c[0] += pts[3 * i];
                c[1] += pts[3 * i + 1];
                c[2] += pts[3 * i + 2];
            }
            [c[0] / 8.0, c[1] / 8.0, c[2] / 8.0]
        };
        for f in &hull.faces {
            let a = point(&pts, f[0]);
            let b = point(&pts, f[1]);
            let c = point(&pts, f[2]);
            // Outward normal ⇒ centroid is below the plane ⇒ orient3d < 0.
            assert!(
                orient3d(a, b, c, centroid) < 0.0,
                "face {f:?} is not outward-oriented"
            );
        }
    }

    #[test]
    fn hull3d_random_sphere_all_on_hull() {
        // Points on a sphere are all extreme; the hull should contain them all.
        let mut pts = Vec::new();
        let n = 30;
        for i in 0..n {
            let t = i as f64 * 2.399963; // golden-angle spiral
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            pts.push(r * t.cos());
            pts.push(r * t.sin());
            pts.push(z);
        }
        let hull = convex_hull_3d(&pts, n).expect("hull");
        assert_eq!(hull.vertices.len(), n, "all sphere points are extreme");
        // Closed triangulated surface ⇒ V − E + F = 2.
        let f = hull.n_faces();
        let e = 3 * f / 2;
        assert_eq!(n as i64 - e as i64 + f as i64, 2);
    }

    // ── 2D ──────────────────────────────────────────────────────────────────

    #[test]
    fn hull2d_square_corners() {
        // Square corners + an interior point; interior point excluded.
        let pts = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.5];
        let hull = convex_hull_2d(&pts, 5).expect("hull");
        assert_eq!(hull.len(), 4);
        assert!(!hull.contains(&4));
    }

    #[test]
    fn hull2d_counter_clockwise() {
        let pts = [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0];
        let hull = convex_hull_2d(&pts, 4).expect("hull");
        // Signed area (shoelace) must be positive for CCW orientation.
        let mut area = 0.0_f64;
        for k in 0..hull.len() {
            let i = hull[k];
            let j = hull[(k + 1) % hull.len()];
            area += pts[2 * i] * pts[2 * j + 1] - pts[2 * j] * pts[2 * i + 1];
        }
        assert!(area > 0.0, "hull should be counter-clockwise");
    }

    #[test]
    fn hull2d_collinear_points() {
        // Collinear points: monotone chain returns the two extreme endpoints.
        let pts = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let hull = convex_hull_2d(&pts, 4).expect("hull");
        assert!(hull.len() <= 2, "collinear hull degenerates to a segment");
    }

    #[test]
    fn hull2d_rejects_bad_length() {
        assert!(matches!(
            convex_hull_2d(&[0.0], 4),
            Err(Geom3dError::DimensionMismatch { .. })
        ));
    }
}

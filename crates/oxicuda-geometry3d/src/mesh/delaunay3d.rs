//! Incremental 3D Delaunay tetrahedralization (Bowyer-Watson).
//!
//! Builds a Delaunay tetrahedralization of a 3D point set by inserting points
//! one at a time into an enclosing *super-tetrahedron*, removing every tetra-
//! hedron whose circumsphere contains the new point (the "cavity"), and
//! re-triangulating the cavity boundary by connecting each boundary face to the
//! new point. After all insertions, every tetrahedron referencing a super-tet
//! vertex is discarded, leaving the Delaunay tetrahedralization of the input.
//!
//! All geometric predicates ([`orient3d`], [`in_sphere`]) are evaluated in
//! `f64` from the (possibly `f32`-sourced) coordinates. Degeneracies are
//! handled with a relative epsilon: a point exactly on a circumsphere is *not*
//! treated as strictly inside, so co-spherical / co-planar inputs degrade
//! gracefully (an empty or partial tetrahedralization) rather than panicking.
//!
//! # References
//! - A. Bowyer, "Computing Dirichlet tessellations", Comput. J. 24(2), 1981.
//! - D. F. Watson, "Computing the n-dimensional Delaunay tessellation with
//!   application to Voronoi polytopes", Comput. J. 24(2), 1981.
//! - J. R. Shewchuk, "Robust Adaptive Floating-Point Geometric Predicates",
//!   1996 (orientation / in-sphere determinant forms).

use crate::error::{Geom3dError, Geom3dResult};
use std::collections::HashMap;

/// Relative epsilon used to classify (near-)degenerate orientation and
/// in-sphere determinants. Scaled by the cube of the local coordinate
/// magnitude so the test is invariant to the overall point-cloud scale.
const PREDICATE_EPS: f64 = 1e-12;

/// A 3D Delaunay tetrahedralization.
#[derive(Debug, Clone)]
pub struct Delaunay3d {
    /// All vertices used during construction, flat as `[m]` of `[x,y,z]`.
    ///
    /// Indices `0..n` are the original input points (in input order); indices
    /// `n..n+4` are the four super-tetrahedron vertices. The super-tet vertices
    /// are retained so [`circumcenter`](Delaunay3d::circumcenter) and other
    /// queries can reference them, but no *output* tetrahedron uses them.
    pub vertices: Vec<[f64; 3]>,
    /// Output tetrahedra as quadruples of indices into `vertices`.
    ///
    /// Each tetrahedron has positive orientation (`orient3d > 0`).
    pub tetrahedra: Vec<[usize; 4]>,
}

/// Compute `3×3` determinant of the matrix whose rows are `r0,r1,r2`.
#[inline]
fn det3(r0: [f64; 3], r1: [f64; 3], r2: [f64; 3]) -> f64 {
    r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0])
        + r0[2] * (r1[0] * r2[1] - r1[1] * r2[0])
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Signed orientation of the tetrahedron `(a,b,c,d)`.
///
/// Returns the determinant of `[b-a; c-a; d-a]`. Positive means `d` is on the
/// positive side of the oriented plane `(a,b,c)` (i.e. `(a,b,c)` is counter-
/// clockwise when viewed from `d`); negative means the opposite side; values
/// within the scaled epsilon are reported as `0.0` (coplanar / degenerate).
pub fn orient3d(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let det = det3(sub(b, a), sub(c, a), sub(d, a));
    let scale = magnitude_scale(&[a, b, c, d]);
    if det.abs() <= PREDICATE_EPS * scale {
        0.0
    } else {
        det
    }
}

/// In-sphere test for point `p` against the circumsphere of tet `(a,b,c,d)`.
///
/// `(a,b,c,d)` is assumed positively oriented (`orient3d(a,b,c,d) > 0`). The
/// standard lifted `4×4` determinant is returned:
///
/// ```text
/// | ax-px  ay-py  az-pz  (ax-px)²+(ay-py)²+(az-pz)² |
/// | bx-px  ...                                       |
/// | cx-px  ...                                       |
/// | dx-px  ...                                       |
/// ```
///
/// A *positive* result means `p` lies strictly **inside** the circumsphere; a
/// negative result means strictly outside; values within the scaled epsilon
/// are reported as `0.0` (on the sphere, treated as outside by callers).
pub fn in_sphere(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3], p: [f64; 3]) -> f64 {
    let da = sub(a, p);
    let db = sub(b, p);
    let dc = sub(c, p);
    let dd = sub(d, p);
    let la = da[0] * da[0] + da[1] * da[1] + da[2] * da[2];
    let lb = db[0] * db[0] + db[1] * db[1] + db[2] * db[2];
    let lc = dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2];
    let ld = dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2];

    // Expand the 4×4 determinant along the last column (the lifted coordinate),
    // each minor being a 3×3 of the first three (shifted) coordinates.
    let m_a = det3(
        [db[0], db[1], db[2]],
        [dc[0], dc[1], dc[2]],
        [dd[0], dd[1], dd[2]],
    );
    let m_b = det3(
        [da[0], da[1], da[2]],
        [dc[0], dc[1], dc[2]],
        [dd[0], dd[1], dd[2]],
    );
    let m_c = det3(
        [da[0], da[1], da[2]],
        [db[0], db[1], db[2]],
        [dd[0], dd[1], dd[2]],
    );
    let m_d = det3(
        [da[0], da[1], da[2]],
        [db[0], db[1], db[2]],
        [dc[0], dc[1], dc[2]],
    );

    let det = la * m_a - lb * m_b + lc * m_c - ld * m_d;

    let scale = magnitude_scale(&[a, b, c, d, p]);
    // The lifted determinant scales like coordinate^5.
    let s5 = scale * scale * scale * scale * scale;
    if det.abs() <= PREDICATE_EPS * s5 {
        0.0
    } else {
        det
    }
}

/// Maximum coordinate magnitude across `pts`, floored at `1.0`, used to scale
/// the relative degeneracy epsilon for the predicates.
#[inline]
fn magnitude_scale(pts: &[[f64; 3]]) -> f64 {
    let mut m = 1.0_f64;
    for p in pts {
        m = m.max(p[0].abs()).max(p[1].abs()).max(p[2].abs());
    }
    m
}

/// An undirected triangular face key (sorted vertex indices) for cavity
/// boundary bookkeeping.
type FaceKey = [usize; 3];

#[inline]
fn face_key(a: usize, b: usize, c: usize) -> FaceKey {
    let mut k = [a, b, c];
    k.sort_unstable();
    k
}

/// Tetrahedralize `n` points given flat as `points = [n*3]` (row-major
/// `x,y,z`).
///
/// The coordinates may originate from `f32` data; they are promoted to `f64`
/// for the predicates. Returns a [`Delaunay3d`] whose `tetrahedra` form the
/// Delaunay tetrahedralization of the input (no super-tet vertices remain).
///
/// # Errors
/// - [`Geom3dError::EmptyPointCloud`] if `n == 0`.
/// - [`Geom3dError::DimensionMismatch`] if `points.len() != n*3`.
/// - [`Geom3dError::InvalidPointDim`] if `n < 4` (a tetrahedron needs 4 points).
/// - [`Geom3dError::NanEncountered`] if any coordinate is non-finite.
///
/// Degenerate inputs (all coplanar / all collinear / fewer than 4 points in
/// general position) yield an **empty** `tetrahedra` list rather than an error
/// once the basic shape checks pass, except where the explicit checks above
/// apply.
pub fn tetrahedralize(points: &[f64], n: usize) -> Geom3dResult<Delaunay3d> {
    if n == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if points.len() != n * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: n * 3,
            got: points.len(),
        });
    }
    if n < 4 {
        return Err(Geom3dError::InvalidPointDim { dim: n });
    }
    for &v in points {
        if !v.is_finite() {
            return Err(Geom3dError::NanEncountered {
                location: "delaunay3d::tetrahedralize input",
            });
        }
    }

    // ── Bounding box ─────────────────────────────────────────────────────────
    let mut bmin = [f64::INFINITY; 3];
    let mut bmax = [f64::NEG_INFINITY; 3];
    for i in 0..n {
        for axis in 0..3 {
            let c = points[i * 3 + axis];
            bmin[axis] = bmin[axis].min(c);
            bmax[axis] = bmax[axis].max(c);
        }
    }
    let center = [
        0.5 * (bmin[0] + bmax[0]),
        0.5 * (bmin[1] + bmax[1]),
        0.5 * (bmin[2] + bmax[2]),
    ];
    let diag = {
        let dx = bmax[0] - bmin[0];
        let dy = bmax[1] - bmin[1];
        let dz = bmax[2] - bmin[2];
        (dx * dx + dy * dy + dz * dz).sqrt().max(1.0)
    };
    // Expand by ~10× the bbox diagonal so every input point is strictly inside
    // the super-tetrahedron's circumsphere.
    let r = 10.0 * diag;

    // ── Vertices: copy inputs, then append 4 super-tet corners. ──────────────
    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(n + 4);
    for i in 0..n {
        vertices.push([points[i * 3], points[i * 3 + 1], points[i * 3 + 2]]);
    }
    // A regular tetrahedron centered at `center`, circumradius ≈ r. The four
    // canonical regular-tet directions are the alternate corners of a cube.
    let s = r / 3.0_f64.sqrt();
    let super_dirs = [
        [1.0_f64, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    let super_base = n;
    for d in &super_dirs {
        vertices.push([
            center[0] + s * d[0],
            center[1] + s * d[1],
            center[2] + s * d[2],
        ]);
    }

    // Ensure positive orientation of the initial super-tetrahedron.
    let mut super_tet = [super_base, super_base + 1, super_base + 2, super_base + 3];
    if orient3d(
        vertices[super_tet[0]],
        vertices[super_tet[1]],
        vertices[super_tet[2]],
        vertices[super_tet[3]],
    ) < 0.0
    {
        super_tet.swap(2, 3);
    }

    // ── Incremental insertion ────────────────────────────────────────────────
    let mut tets: Vec<[usize; 4]> = vec![super_tet];

    for i in 0..n {
        let p = vertices[i];

        // Find "bad" tetrahedra whose circumsphere strictly contains p.
        let mut bad: Vec<usize> = Vec::new();
        for (ti, tet) in tets.iter().enumerate() {
            let a = vertices[tet[0]];
            let b = vertices[tet[1]];
            let c = vertices[tet[2]];
            let d = vertices[tet[3]];
            // Tets are stored positively oriented; in_sphere > 0 ⇒ inside.
            if in_sphere(a, b, c, d, p) > 0.0 {
                bad.push(ti);
            }
        }

        if bad.is_empty() {
            // Point coincided with an existing vertex or fell on a sphere
            // boundary (degenerate). Skip — it adds no new tetrahedron.
            continue;
        }

        // Boundary faces of the cavity = faces belonging to exactly one bad tet.
        let mut face_count: HashMap<FaceKey, (usize, [usize; 3])> = HashMap::new();
        for &ti in &bad {
            let t = tets[ti];
            // The 4 faces of a tetrahedron (opposite each vertex), kept as the
            // *ordered* triple so the new tet can inherit a consistent winding.
            let faces = [
                [t[1], t[2], t[3]],
                [t[0], t[3], t[2]],
                [t[0], t[1], t[3]],
                [t[0], t[2], t[1]],
            ];
            for f in faces {
                let key = face_key(f[0], f[1], f[2]);
                let entry = face_count.entry(key).or_insert((0, f));
                entry.0 += 1;
            }
        }

        // Remove bad tets (descending index to keep positions valid).
        bad.sort_unstable_by(|x, y| y.cmp(x));
        for &ti in &bad {
            tets.swap_remove(ti);
        }

        // Re-triangulate: connect each boundary face to p.
        for (_, (count, face)) in face_count {
            if count != 1 {
                continue;
            }
            let mut new_tet = [face[0], face[1], face[2], i];
            // Fix orientation so the new tet is positively oriented.
            if orient3d(
                vertices[new_tet[0]],
                vertices[new_tet[1]],
                vertices[new_tet[2]],
                vertices[new_tet[3]],
            ) < 0.0
            {
                new_tet.swap(0, 1);
            }
            tets.push(new_tet);
        }
    }

    // ── Strip every tet that references a super-tet vertex (index ≥ n). ──────
    tets.retain(|t| t.iter().all(|&v| v < n));

    Ok(Delaunay3d {
        vertices,
        tetrahedra: tets,
    })
}

impl Delaunay3d {
    /// Number of tetrahedra in the tetrahedralization.
    #[inline]
    pub fn num_tetrahedra(&self) -> usize {
        self.tetrahedra.len()
    }

    /// Circumcenter of tetrahedron `tet` (index into `tetrahedra`).
    ///
    /// Solves the `3×3` linear system for the point equidistant from the four
    /// vertices. Returns the (finite) circumcenter; for a (near-)degenerate
    /// tetrahedron the centroid is returned as a stable fallback so the result
    /// is never NaN/Inf.
    pub fn circumcenter(&self, tet: usize) -> [f64; 3] {
        let t = self.tetrahedra[tet];
        let a = self.vertices[t[0]];
        let b = self.vertices[t[1]];
        let c = self.vertices[t[2]];
        let d = self.vertices[t[3]];
        circumcenter_of(a, b, c, d)
    }

    /// Faces of the convex hull as outward-consistent vertex-index triples.
    ///
    /// In a Delaunay tetrahedralization the hull boundary is exactly the set of
    /// triangular faces that belong to a **single** tetrahedron (interior faces
    /// are shared by two). Each returned triple uses the winding inherited from
    /// its owning tetrahedron's outward face.
    pub fn convex_hull_faces(&self) -> Vec<[usize; 3]> {
        let mut count: HashMap<FaceKey, (usize, [usize; 3])> = HashMap::new();
        for t in &self.tetrahedra {
            let faces = [
                [t[1], t[2], t[3]],
                [t[0], t[3], t[2]],
                [t[0], t[1], t[3]],
                [t[0], t[2], t[1]],
            ];
            for f in faces {
                let key = face_key(f[0], f[1], f[2]);
                let entry = count.entry(key).or_insert((0, f));
                entry.0 += 1;
                // Keep the ordered triple of the last writer; for a hull face
                // there is only one owner so its winding is the outward one.
                entry.1 = f;
            }
        }
        count
            .into_iter()
            .filter_map(|(_, (c, f))| if c == 1 { Some(f) } else { None })
            .collect()
    }

    /// Signed volume of tetrahedron `tet` (always non-negative because output
    /// tetrahedra are positively oriented). Equals `|det[b-a;c-a;d-a]| / 6`.
    pub fn tet_volume(&self, tet: usize) -> f64 {
        let t = self.tetrahedra[tet];
        let a = self.vertices[t[0]];
        let b = self.vertices[t[1]];
        let c = self.vertices[t[2]];
        let d = self.vertices[t[3]];
        det3(sub(b, a), sub(c, a), sub(d, a)).abs() / 6.0
    }

    /// Total volume tiled by all tetrahedra.
    pub fn total_volume(&self) -> f64 {
        (0..self.tetrahedra.len()).map(|t| self.tet_volume(t)).sum()
    }
}

/// Solve for the circumcenter of tetrahedron `(a,b,c,d)`.
///
/// The circumcenter `x` satisfies `|x-a|² = |x-b|² = |x-c|² = |x-d|²`, which
/// linearizes to `2 (b-a)·x = |b|²-|a|²`, etc. We solve the resulting `3×3`
/// system via Cramer's rule; a degenerate (coplanar) tet falls back to the
/// centroid.
fn circumcenter_of(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> [f64; 3] {
    let row = |q: [f64; 3]| -> [f64; 3] {
        [
            2.0 * (q[0] - a[0]),
            2.0 * (q[1] - a[1]),
            2.0 * (q[2] - a[2]),
        ]
    };
    let rhs = |q: [f64; 3]| -> f64 {
        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) - (a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    };
    let m0 = row(b);
    let m1 = row(c);
    let m2 = row(d);
    let r = [rhs(b), rhs(c), rhs(d)];

    let det = det3(m0, m1, m2);
    if det.abs() < 1e-18 {
        // Degenerate: return centroid as a stable, finite fallback.
        return [
            (a[0] + b[0] + c[0] + d[0]) / 4.0,
            (a[1] + b[1] + c[1] + d[1]) / 4.0,
            (a[2] + b[2] + c[2] + d[2]) / 4.0,
        ];
    }
    // Cramer's rule: replace each column with r.
    let dx = det3(
        [r[0], m0[1], m0[2]],
        [r[1], m1[1], m1[2]],
        [r[2], m2[1], m2[2]],
    );
    let dy = det3(
        [m0[0], r[0], m0[2]],
        [m1[0], r[1], m1[2]],
        [m2[0], r[2], m2[2]],
    );
    let dz = det3(
        [m0[0], m0[1], r[0]],
        [m1[0], m1[1], r[1]],
        [m2[0], m2[1], r[2]],
    );
    [dx / det, dy / det, dz / det]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng_points(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        let mut p = vec![0.0_f64; n * 3];
        for v in &mut p {
            *v = (rng.next_f32() as f64) * 2.0 - 1.0;
        }
        p
    }

    #[test]
    fn single_tet_from_four_points() {
        // A non-degenerate tetrahedron → exactly one output tet.
        let pts = [
            0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let d = tetrahedralize(&pts, 4).expect("tetrahedralize should succeed");
        assert_eq!(
            d.num_tetrahedra(),
            1,
            "4 points in general position → 1 tet"
        );
        // Its volume must equal the analytic 1/6.
        assert!((d.total_volume() - 1.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn empty_circumsphere_property() {
        // The defining Delaunay property: no input point lies strictly inside
        // any tetrahedron's circumsphere.
        for seed in [1_u64, 7, 42] {
            let n = 20;
            let pts = rng_points(n, seed);
            let d = tetrahedralize(&pts, n).expect("tetrahedralize should succeed");
            assert!(d.num_tetrahedra() > 0, "should produce tets (seed {seed})");
            for t in &d.tetrahedra {
                let a = d.vertices[t[0]];
                let b = d.vertices[t[1]];
                let c = d.vertices[t[2]];
                let dd = d.vertices[t[3]];
                for (idx, _) in pts.chunks_exact(3).enumerate() {
                    if idx == t[0] || idx == t[1] || idx == t[2] || idx == t[3] {
                        continue;
                    }
                    let p = d.vertices[idx];
                    let s = in_sphere(a, b, c, dd, p);
                    assert!(
                        s <= 0.0,
                        "point {idx} strictly inside circumsphere (in_sphere={s}, seed={seed})"
                    );
                }
            }
        }
    }

    #[test]
    fn cube_corners_tile_unit_volume() {
        // The 8 corners of the unit cube should be tiled by 5 or 6 tets whose
        // volumes sum to 1.
        let mut pts = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    pts.push(x as f64);
                    pts.push(y as f64);
                    pts.push(z as f64);
                }
            }
        }
        let d = tetrahedralize(&pts, 8).expect("tetrahedralize should succeed");
        assert!(d.num_tetrahedra() >= 5, "cube needs ≥5 tets");
        assert!(
            (d.total_volume() - 1.0).abs() < 1e-9,
            "tets must tile the unit cube, got volume {}",
            d.total_volume()
        );
    }

    #[test]
    fn coplanar_input_is_graceful() {
        // All z=0 → no valid tetrahedron, but must not panic / produce Inf.
        let pts = [
            0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0,
        ];
        let d = tetrahedralize(&pts, 5).expect("tetrahedralize should succeed");
        assert_eq!(d.num_tetrahedra(), 0, "coplanar input → no tets");
        // Volume sum is trivially finite & zero.
        assert_eq!(d.total_volume(), 0.0);
    }

    #[test]
    fn convex_hull_faces_of_single_tet() {
        let pts = [
            0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let d = tetrahedralize(&pts, 4).expect("tetrahedralize should succeed");
        let hull = d.convex_hull_faces();
        assert_eq!(hull.len(), 4, "a single tet's hull has exactly 4 faces");
    }

    #[test]
    fn circumcenter_equidistant() {
        let pts = [
            0.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0,
        ];
        let d = tetrahedralize(&pts, 4).expect("tetrahedralize should succeed");
        let cc = d.circumcenter(0);
        let t = d.tetrahedra[0];
        let mut r0 = f64::NAN;
        for &vi in &t {
            let v = d.vertices[vi];
            let dist =
                ((v[0] - cc[0]).powi(2) + (v[1] - cc[1]).powi(2) + (v[2] - cc[2]).powi(2)).sqrt();
            if r0.is_nan() {
                r0 = dist;
            } else {
                assert!((dist - r0).abs() < 1e-9, "circumcenter not equidistant");
            }
        }
    }

    #[test]
    fn errors_on_bad_shape() {
        assert!(matches!(
            tetrahedralize(&[], 0),
            Err(Geom3dError::EmptyPointCloud)
        ));
        assert!(matches!(
            tetrahedralize(&[0.0, 0.0, 0.0], 1),
            Err(Geom3dError::InvalidPointDim { dim: 1 })
        ));
        assert!(matches!(
            tetrahedralize(&[0.0; 6], 4),
            Err(Geom3dError::DimensionMismatch { .. })
        ));
        let mut bad = vec![0.0_f64; 12];
        bad[0] = f64::NAN;
        assert!(matches!(
            tetrahedralize(&bad, 4),
            Err(Geom3dError::NanEncountered { .. })
        ));
    }

    #[test]
    fn orient3d_sign_conventions() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let above = [0.0, 0.0, 1.0];
        let below = [0.0, 0.0, -1.0];
        assert!(orient3d(a, b, c, above) > 0.0);
        assert!(orient3d(a, b, c, below) < 0.0);
        // Coplanar → exactly 0.
        assert_eq!(orient3d(a, b, c, [0.5, 0.5, 0.0]), 0.0);
    }

    #[test]
    fn in_sphere_sign_conventions() {
        // Positively oriented unit-corner tet.
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        assert!(orient3d(a, b, c, d) > 0.0);
        // Centroid is inside the circumsphere.
        let inside = [0.25, 0.25, 0.25];
        assert!(in_sphere(a, b, c, d, inside) > 0.0);
        // A far point is outside.
        let outside = [10.0, 10.0, 10.0];
        assert!(in_sphere(a, b, c, d, outside) < 0.0);
    }
}

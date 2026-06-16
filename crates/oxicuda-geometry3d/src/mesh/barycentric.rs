//! Barycentric coordinates for triangles and tetrahedra, with point-containment
//! tests and barycentric interpolation of vertex attributes.
//!
//! Barycentric coordinates express a point `p` as an affine combination
//! `p = Σ λ_i v_i` with `Σ λ_i = 1`. For a triangle a point lies inside the
//! (closed) triangle iff all three coordinates are non-negative; for a
//! tetrahedron the same holds for all four. The triangle routine works for 3D
//! triangles by projecting onto the triangle plane (Ericson's area method).
//!
//! All geometry is `f64`.
//!
//! # References
//! - Ericson, C. (2005). *Real-Time Collision Detection*, §3.4 (barycentric
//!   coordinates) and §5.4.2 (point-in-triangle via barycentrics).
//! - Möbius, A.F. (1827). *Der barycentrische Calcul*.

use crate::error::{Geom3dError, Geom3dResult};

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

/// Barycentric coordinates `(λ0, λ1, λ2)` of point `p` with respect to the
/// triangle `(a, b, c)`, computed by Ericson's projected-area method (works for
/// 3D triangles; if `p` is off the plane the coordinates are those of its
/// orthogonal projection).
///
/// # Errors
/// - [`Geom3dError::InvalidTopology`] if the triangle is degenerate (zero area).
pub fn barycentric_triangle(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> Geom3dResult<[f64; 3]> {
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let v2 = sub(p, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-300 {
        return Err(Geom3dError::InvalidTopology {
            reason: "barycentric_triangle: degenerate (zero-area) triangle",
        });
    }
    let lambda1 = (d11 * d20 - d01 * d21) / denom;
    let lambda2 = (d00 * d21 - d01 * d20) / denom;
    let lambda0 = 1.0 - lambda1 - lambda2;
    Ok([lambda0, lambda1, lambda2])
}

/// Whether `p` lies inside (or on the boundary of) the triangle `(a, b, c)`.
///
/// For 3D triangles this also requires `p` to be (approximately) coplanar with
/// the triangle, within `plane_tol` distance of the triangle plane.
///
/// # Errors
/// - [`Geom3dError::InvalidTopology`] if the triangle is degenerate.
pub fn point_in_triangle(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    plane_tol: f64,
) -> Geom3dResult<bool> {
    let lambda = barycentric_triangle(p, a, b, c)?;
    // Check coplanarity: distance of p from the plane.
    let normal = cross(sub(b, a), sub(c, a));
    let n_len = dot(normal, normal).sqrt();
    let dist = if n_len < 1e-300 {
        0.0
    } else {
        dot(sub(p, a), normal).abs() / n_len
    };
    let inside = lambda[0] >= -1e-12 && lambda[1] >= -1e-12 && lambda[2] >= -1e-12;
    Ok(inside && dist <= plane_tol)
}

/// Interpolate a per-vertex scalar attribute `(fa, fb, fc)` at the barycentric
/// position of `p` inside triangle `(a, b, c)`.
///
/// # Errors
/// - [`Geom3dError::InvalidTopology`] if the triangle is degenerate.
pub fn interpolate_triangle(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    fa: f64,
    fb: f64,
    fc: f64,
) -> Geom3dResult<f64> {
    let l = barycentric_triangle(p, a, b, c)?;
    Ok(l[0] * fa + l[1] * fb + l[2] * fc)
}

/// Signed volume of the tetrahedron `(a, b, c, d)` scaled by 6
/// (= the `3×3` determinant `det[b−a, c−a, d−a]`).
#[inline]
fn signed_volume6(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let u = sub(b, a);
    let v = sub(c, a);
    let w = sub(d, a);
    dot(u, cross(v, w))
}

/// Barycentric coordinates `(λ0, λ1, λ2, λ3)` of `p` with respect to the
/// tetrahedron `(a, b, c, d)`, via the signed-sub-volume ratios.
///
/// # Errors
/// - [`Geom3dError::InvalidTopology`] if the tetrahedron is degenerate
///   (zero volume / coplanar vertices).
pub fn barycentric_tetrahedron(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    d: [f64; 3],
) -> Geom3dResult<[f64; 4]> {
    let v_total = signed_volume6(a, b, c, d);
    if v_total.abs() < 1e-300 {
        return Err(Geom3dError::InvalidTopology {
            reason: "barycentric_tetrahedron: degenerate (zero-volume) tetrahedron",
        });
    }
    // λ_i = vol of the tet with vertex i replaced by p, divided by total volume.
    let l0 = signed_volume6(p, b, c, d) / v_total;
    let l1 = signed_volume6(a, p, c, d) / v_total;
    let l2 = signed_volume6(a, b, p, d) / v_total;
    let l3 = signed_volume6(a, b, c, p) / v_total;
    Ok([l0, l1, l2, l3])
}

/// Whether `p` lies inside (or on the boundary of) the tetrahedron
/// `(a, b, c, d)`.
///
/// # Errors
/// - [`Geom3dError::InvalidTopology`] if the tetrahedron is degenerate.
pub fn point_in_tetrahedron(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    d: [f64; 3],
) -> Geom3dResult<bool> {
    let l = barycentric_tetrahedron(p, a, b, c, d)?;
    Ok(l.iter().all(|&v| v >= -1e-12))
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: [f64; 3] = [0.0, 0.0, 0.0];
    const B: [f64; 3] = [1.0, 0.0, 0.0];
    const C: [f64; 3] = [0.0, 1.0, 0.0];

    // ── Triangle barycentrics ──────────────────────────────────────────────────

    #[test]
    fn bary_triangle_vertices() {
        let la = barycentric_triangle(A, A, B, C).expect("ok");
        assert!((la[0] - 1.0).abs() < 1e-12 && la[1].abs() < 1e-12 && la[2].abs() < 1e-12);
        let lb = barycentric_triangle(B, A, B, C).expect("ok");
        assert!((lb[1] - 1.0).abs() < 1e-12);
        let lc = barycentric_triangle(C, A, B, C).expect("ok");
        assert!((lc[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bary_triangle_centroid() {
        let centroid = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let l = barycentric_triangle(centroid, A, B, C).expect("ok");
        for &v in &l {
            assert!((v - 1.0 / 3.0).abs() < 1e-12, "centroid λ should be 1/3");
        }
    }

    #[test]
    fn bary_triangle_sums_to_one() {
        let p = [0.2, 0.3, 0.0];
        let l = barycentric_triangle(p, A, B, C).expect("ok");
        assert!((l[0] + l[1] + l[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bary_triangle_degenerate_errors() {
        // Three collinear points.
        let d1 = [0.0, 0.0, 0.0];
        let d2 = [1.0, 0.0, 0.0];
        let d3 = [2.0, 0.0, 0.0];
        assert!(matches!(
            barycentric_triangle([0.5, 0.0, 0.0], d1, d2, d3),
            Err(Geom3dError::InvalidTopology { .. })
        ));
    }

    #[test]
    fn point_in_triangle_inside_outside() {
        assert!(point_in_triangle([0.25, 0.25, 0.0], A, B, C, 1e-9).expect("ok"));
        // Outside (λ1+λ2 > 1).
        assert!(!point_in_triangle([0.8, 0.8, 0.0], A, B, C, 1e-9).expect("ok"));
        // On an edge counts as inside.
        assert!(point_in_triangle([0.5, 0.0, 0.0], A, B, C, 1e-9).expect("ok"));
    }

    #[test]
    fn point_in_triangle_off_plane_rejected() {
        // Directly above the centroid but off the plane ⇒ not "in" with tol.
        assert!(!point_in_triangle([0.25, 0.25, 0.5], A, B, C, 1e-6).expect("ok"));
        // With a generous tolerance it is accepted.
        assert!(point_in_triangle([0.25, 0.25, 0.5], A, B, C, 1.0).expect("ok"));
    }

    #[test]
    fn interpolate_triangle_corner_and_centroid() {
        // Attribute equals 10 at B, 0 elsewhere; centroid value = 10/3.
        let v =
            interpolate_triangle([1.0 / 3.0, 1.0 / 3.0, 0.0], A, B, C, 0.0, 10.0, 0.0).expect("ok");
        assert!((v - 10.0 / 3.0).abs() < 1e-12);
        // At vertex B exactly, value = 10.
        let vb = interpolate_triangle(B, A, B, C, 0.0, 10.0, 0.0).expect("ok");
        assert!((vb - 10.0).abs() < 1e-12);
    }

    #[test]
    fn bary_triangle_3d_plane() {
        // A triangle lying in a tilted plane; the centroid still maps to 1/3.
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 1.0];
        let c = [0.0, 1.0, 1.0];
        let centroid = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        let l = barycentric_triangle(centroid, a, b, c).expect("ok");
        for &v in &l {
            assert!((v - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    // ── Tetrahedron barycentrics ────────────────────────────────────────────────

    const TA: [f64; 3] = [0.0, 0.0, 0.0];
    const TB: [f64; 3] = [1.0, 0.0, 0.0];
    const TC: [f64; 3] = [0.0, 1.0, 0.0];
    const TD: [f64; 3] = [0.0, 0.0, 1.0];

    #[test]
    fn bary_tet_vertices() {
        let l = barycentric_tetrahedron(TA, TA, TB, TC, TD).expect("ok");
        assert!((l[0] - 1.0).abs() < 1e-12);
        let l3 = barycentric_tetrahedron(TD, TA, TB, TC, TD).expect("ok");
        assert!((l3[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bary_tet_centroid_sums_to_one() {
        let centroid = [0.25, 0.25, 0.25];
        let l = barycentric_tetrahedron(centroid, TA, TB, TC, TD).expect("ok");
        for &v in &l {
            assert!((v - 0.25).abs() < 1e-12, "tet centroid λ should be 1/4");
        }
        assert!((l.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bary_tet_degenerate_errors() {
        // Four coplanar points (all z=0).
        let d = [1.0, 1.0, 0.0];
        assert!(matches!(
            barycentric_tetrahedron([0.25, 0.25, 0.0], TA, TB, TC, d),
            Err(Geom3dError::InvalidTopology { .. })
        ));
    }

    #[test]
    fn point_in_tetrahedron_inside_outside() {
        assert!(point_in_tetrahedron([0.1, 0.1, 0.1], TA, TB, TC, TD).expect("ok"));
        // Outside: sum of x,y,z > 1.
        assert!(!point_in_tetrahedron([0.5, 0.5, 0.5], TA, TB, TC, TD).expect("ok"));
        // On a face (z=0 plane, inside the base triangle) counts as inside.
        assert!(point_in_tetrahedron([0.2, 0.2, 0.0], TA, TB, TC, TD).expect("ok"));
        // Just outside a vertex.
        assert!(!point_in_tetrahedron([-0.01, 0.0, 0.0], TA, TB, TC, TD).expect("ok"));
    }

    #[test]
    fn point_in_tetrahedron_orientation_independent() {
        // Swapping two vertices flips the sign of the total volume but the
        // containment result must be unchanged.
        let p = [0.1, 0.1, 0.1];
        let normal = point_in_tetrahedron(p, TA, TB, TC, TD).expect("ok");
        let swapped = point_in_tetrahedron(p, TB, TA, TC, TD).expect("ok");
        assert_eq!(normal, swapped);
    }
}

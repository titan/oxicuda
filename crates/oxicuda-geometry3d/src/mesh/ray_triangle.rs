//! Ray-triangle intersection (Möller-Trumbore), point-triangle closest point
//! (Ericson Voronoi-region), ray-AABB (slab), and their mesh-level reductions.
//!
//! All routines work in `f64`. Triangles are passed as three `[f64;3]` corners;
//! meshes are a flat vertex buffer `vertices = [v*3]` plus a triangle index list
//! `&[[usize;3]]`.
//!
//! # References
//! - T. Möller, B. Trumbore, "Fast, Minimum Storage Ray/Triangle Intersection",
//!   Journal of Graphics Tools 2(1), 1997.
//! - C. Ericson, "Real-Time Collision Detection", 2005 — §5.1.5
//!   `ClosestPtPointTriangle` (Voronoi-region barycentric form) and §5.3.3
//!   ray/slab test.

/// Epsilon below which a ray is treated as parallel to a triangle's plane.
const RAY_EPS: f64 = 1e-12;

/// A ray-triangle hit record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Ray parameter: `point = orig + t * dir` (in units of `dir`'s length).
    pub t: f64,
    /// Barycentric coordinate along `edge1 = v1 - v0`.
    pub u: f64,
    /// Barycentric coordinate along `edge2 = v2 - v0`.
    pub v: f64,
    /// World-space intersection point.
    pub point: [f64; 3],
}

#[inline]
fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Möller-Trumbore ray-triangle intersection.
///
/// Returns `Some(hit)` with `t > RAY_EPS` (intersection in front of the ray
/// origin) when the ray crosses the triangle interior (inclusive of edges), or
/// `None` if the ray is parallel to the plane, misses the triangle, or the hit
/// is behind the origin. The test is double-sided (front and back faces both
/// hit).
pub fn ray_triangle_intersect(
    orig: &[f64; 3],
    dir: &[f64; 3],
    v0: &[f64; 3],
    v1: &[f64; 3],
    v2: &[f64; 3],
) -> Option<RayHit> {
    let edge1 = sub(v1, v0);
    let edge2 = sub(v2, v0);
    let h = cross(dir, &edge2);
    let a = dot(&edge1, &h);
    if a.abs() < RAY_EPS {
        return None; // Parallel to the triangle plane.
    }
    let f = 1.0 / a;
    let s = sub(orig, v0);
    let u = f * dot(&s, &h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(&s, &edge1);
    let v = f * dot(dir, &q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * dot(&edge2, &q);
    if t > RAY_EPS {
        Some(RayHit {
            t,
            u,
            v,
            point: [
                orig[0] + t * dir[0],
                orig[1] + t * dir[1],
                orig[2] + t * dir[2],
            ],
        })
    } else {
        None
    }
}

/// Closest point on triangle `(a,b,c)` to `p`, with its squared distance.
///
/// Implements the Voronoi-region method from Ericson's *Real-Time Collision
/// Detection* (§5.1.5): the result lies in the triangle's interior, on one of
/// its edges, or at a vertex, depending on which feature's Voronoi region
/// contains the projection of `p`. Returns `(closest, sq_dist)`.
pub fn closest_point_on_triangle(
    p: &[f64; 3],
    a: &[f64; 3],
    b: &[f64; 3],
    c: &[f64; 3],
) -> ([f64; 3], f64) {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);

    // Vertex region outside A.
    let d1 = dot(&ab, &ap);
    let d2 = dot(&ac, &ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return finalize(*a, p);
    }

    // Vertex region outside B.
    let bp = sub(p, b);
    let d3 = dot(&ab, &bp);
    let d4 = dot(&ac, &bp);
    if d3 >= 0.0 && d4 <= d3 {
        return finalize(*b, p);
    }

    // Edge region of AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let closest = [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
        return finalize(closest, p);
    }

    // Vertex region outside C.
    let cp = sub(p, c);
    let d5 = dot(&ab, &cp);
    let d6 = dot(&ac, &cp);
    if d6 >= 0.0 && d5 <= d6 {
        return finalize(*c, p);
    }

    // Edge region of AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let closest = [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
        return finalize(closest, p);
    }

    // Edge region of BC.
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = sub(c, b);
        let closest = [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]];
        return finalize(closest, p);
    }

    // Inside face region. Compute barycentric coordinates (u,v,w).
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let closest = [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ];
    finalize(closest, p)
}

#[inline]
fn finalize(closest: [f64; 3], p: &[f64; 3]) -> ([f64; 3], f64) {
    let d = sub(&closest, p);
    (closest, dot(&d, &d))
}

/// Ray-AABB intersection via the slab method.
///
/// Returns `Some((t_near, t_far))` — the entry/exit ray parameters of the
/// segment of the ray that lies inside the box `[bmin, bmax]` — clamped so that
/// `t_near` is non-negative-meaningful for the caller (the raw near/far are
/// returned; `t_far >= 0` and `t_near <= t_far` is guaranteed on `Some`).
/// Returns `None` if the ray misses the box or the box lies entirely behind the
/// origin. Components of `dir` may be zero (the ray runs parallel to that slab).
pub fn ray_aabb_intersect(
    orig: &[f64; 3],
    dir: &[f64; 3],
    bmin: &[f64; 3],
    bmax: &[f64; 3],
) -> Option<(f64, f64)> {
    let mut t_near = f64::NEG_INFINITY;
    let mut t_far = f64::INFINITY;
    for axis in 0..3 {
        if dir[axis].abs() < RAY_EPS {
            // Parallel to this slab: must already be within the slab bounds.
            if orig[axis] < bmin[axis] || orig[axis] > bmax[axis] {
                return None;
            }
        } else {
            let inv = 1.0 / dir[axis];
            let mut t1 = (bmin[axis] - orig[axis]) * inv;
            let mut t2 = (bmax[axis] - orig[axis]) * inv;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_near = t_near.max(t1);
            t_far = t_far.min(t2);
            if t_near > t_far {
                return None;
            }
        }
    }
    if t_far < 0.0 {
        return None; // Box entirely behind the ray origin.
    }
    Some((t_near, t_far))
}

/// Nearest ray-mesh intersection.
///
/// Tests `dir` against every triangle, returning the `(triangle_index, hit)`
/// with the smallest positive `t`, or `None` if no triangle is hit. Triangle
/// indices that reference out-of-range vertices are skipped.
pub fn ray_mesh_intersect(
    orig: &[f64; 3],
    dir: &[f64; 3],
    vertices: &[f64],
    triangles: &[[usize; 3]],
) -> Option<(usize, RayHit)> {
    let nv = vertices.len() / 3;
    let mut best: Option<(usize, RayHit)> = None;
    for (ti, tri) in triangles.iter().enumerate() {
        if tri[0] >= nv || tri[1] >= nv || tri[2] >= nv {
            continue;
        }
        let v0 = fetch(vertices, tri[0]);
        let v1 = fetch(vertices, tri[1]);
        let v2 = fetch(vertices, tri[2]);
        if let Some(hit) = ray_triangle_intersect(orig, dir, &v0, &v1, &v2) {
            match &best {
                Some((_, b)) if hit.t >= b.t => {}
                _ => best = Some((ti, hit)),
            }
        }
    }
    best
}

/// Closest point on a triangle mesh to `p`.
///
/// Returns `(triangle_index, closest_point, sq_dist)` for the nearest surface
/// point across all triangles. If the mesh has no usable triangle the result is
/// `(usize::MAX, p, +inf)`.
pub fn closest_point_on_mesh(
    p: &[f64; 3],
    vertices: &[f64],
    triangles: &[[usize; 3]],
) -> (usize, [f64; 3], f64) {
    let nv = vertices.len() / 3;
    let mut best_idx = usize::MAX;
    let mut best_pt = *p;
    let mut best_sq = f64::INFINITY;
    for (ti, tri) in triangles.iter().enumerate() {
        if tri[0] >= nv || tri[1] >= nv || tri[2] >= nv {
            continue;
        }
        let a = fetch(vertices, tri[0]);
        let b = fetch(vertices, tri[1]);
        let c = fetch(vertices, tri[2]);
        let (cp, sq) = closest_point_on_triangle(p, &a, &b, &c);
        if sq < best_sq {
            best_sq = sq;
            best_pt = cp;
            best_idx = ti;
        }
    }
    (best_idx, best_pt, best_sq)
}

#[inline]
fn fetch(vertices: &[f64], i: usize) -> [f64; 3] {
    [vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRI_V0: [f64; 3] = [0.0, 0.0, 0.0];
    const TRI_V1: [f64; 3] = [1.0, 0.0, 0.0];
    const TRI_V2: [f64; 3] = [0.0, 1.0, 0.0];

    fn centroid() -> [f64; 3] {
        [
            (TRI_V0[0] + TRI_V1[0] + TRI_V2[0]) / 3.0,
            (TRI_V0[1] + TRI_V1[1] + TRI_V2[1]) / 3.0,
            (TRI_V0[2] + TRI_V1[2] + TRI_V2[2]) / 3.0,
        ]
    }

    #[test]
    fn ray_hits_centroid_along_minus_normal() {
        let c = centroid();
        let orig = [c[0], c[1], 2.0];
        let dir = [0.0, 0.0, -1.0]; // along -normal (triangle normal is +z)
        let hit =
            ray_triangle_intersect(&orig, &dir, &TRI_V0, &TRI_V1, &TRI_V2).expect("ray should hit");
        assert!((hit.t - 2.0).abs() < 1e-9, "t should be 2, got {}", hit.t);
        // Reconstruct centroid from barycentrics: u=v=1/3.
        assert!((hit.u - 1.0 / 3.0).abs() < 1e-9);
        assert!((hit.v - 1.0 / 3.0).abs() < 1e-9);
        for (hp, cc) in hit.point.iter().zip(c.iter()) {
            assert!((hp - cc).abs() < 1e-9);
        }
    }

    #[test]
    fn ray_parallel_returns_none() {
        let orig = [0.2, 0.2, 1.0];
        let dir = [1.0, 0.0, 0.0]; // parallel to z=0 plane
        assert!(ray_triangle_intersect(&orig, &dir, &TRI_V0, &TRI_V1, &TRI_V2).is_none());
    }

    #[test]
    fn ray_misses_outside() {
        let orig = [2.0, 2.0, 1.0];
        let dir = [0.0, 0.0, -1.0];
        assert!(ray_triangle_intersect(&orig, &dir, &TRI_V0, &TRI_V1, &TRI_V2).is_none());
    }

    #[test]
    fn ray_behind_origin_returns_none() {
        let c = centroid();
        let orig = [c[0], c[1], -2.0];
        let dir = [0.0, 0.0, -1.0]; // pointing away from the triangle
        assert!(ray_triangle_intersect(&orig, &dir, &TRI_V0, &TRI_V1, &TRI_V2).is_none());
    }

    #[test]
    fn closest_point_above_face_projects_down() {
        let p = [0.25, 0.25, 3.0]; // above the interior
        let (cp, sq) = closest_point_on_triangle(&p, &TRI_V0, &TRI_V1, &TRI_V2);
        assert!((cp[0] - 0.25).abs() < 1e-9);
        assert!((cp[1] - 0.25).abs() < 1e-9);
        assert!(cp[2].abs() < 1e-9);
        assert!((sq - 9.0).abs() < 1e-9, "sq_dist should be height² = 9");
    }

    #[test]
    fn closest_point_beyond_vertex() {
        let p = [-1.0, -1.0, 0.0]; // beyond vertex A
        let (cp, _) = closest_point_on_triangle(&p, &TRI_V0, &TRI_V1, &TRI_V2);
        for (got, want) in cp.iter().zip(TRI_V0.iter()) {
            assert!((got - want).abs() < 1e-9, "should clamp to vertex A");
        }
    }

    #[test]
    fn closest_point_over_edge_midpoint() {
        // Point just outside the midpoint of edge v0-v1 (the x-axis edge).
        let mid = [0.5, 0.0, 0.0];
        let p = [0.5, -1.0, 0.0];
        let (cp, _) = closest_point_on_triangle(&p, &TRI_V0, &TRI_V1, &TRI_V2);
        for (got, want) in cp.iter().zip(mid.iter()) {
            assert!((got - want).abs() < 1e-9, "should clamp to edge midpoint");
        }
    }

    #[test]
    fn aabb_slab_hit_and_miss() {
        let bmin = [0.0, 0.0, 0.0];
        let bmax = [1.0, 1.0, 1.0];
        // Hit: ray straight down through the center.
        let hit = ray_aabb_intersect(&[0.5, 0.5, 3.0], &[0.0, 0.0, -1.0], &bmin, &bmax);
        assert!(hit.is_some());
        let (tn, tf) = hit.expect("hit should be present");
        assert!((tn - 2.0).abs() < 1e-9 && (tf - 3.0).abs() < 1e-9);
        // Miss: parallel and outside the slab.
        let miss = ray_aabb_intersect(&[2.0, 0.5, 3.0], &[0.0, 0.0, -1.0], &bmin, &bmax);
        assert!(miss.is_none());
        // Miss: pointing away.
        let away = ray_aabb_intersect(&[0.5, 0.5, 3.0], &[0.0, 0.0, 1.0], &bmin, &bmax);
        assert!(away.is_none());
    }

    #[test]
    fn ray_mesh_picks_nearest() {
        // Two parallel triangles at z=1 and z=2; a downward ray hits the nearer.
        let vertices = vec![
            // z=2 triangle (indices 0..3)
            0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, // z=1 triangle (indices 3..6)
            0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
        ];
        let tris = [[0, 1, 2], [3, 4, 5]];
        let (idx, hit) =
            ray_mesh_intersect(&[0.25, 0.25, 5.0], &[0.0, 0.0, -1.0], &vertices, &tris)
                .expect("ray_mesh_intersect should succeed");
        assert_eq!(idx, 0, "nearest is the z=2 triangle");
        assert!((hit.t - 3.0).abs() < 1e-9);
    }

    #[test]
    fn closest_point_on_mesh_basic() {
        let vertices = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let tris = [[0_usize, 1, 2]];
        let (idx, cp, sq) = closest_point_on_mesh(&[0.25, 0.25, 2.0], &vertices, &tris);
        assert_eq!(idx, 0);
        assert!(cp[2].abs() < 1e-9);
        assert!((sq - 4.0).abs() < 1e-9);
    }

    #[test]
    fn closest_point_on_empty_mesh() {
        let (idx, _, sq) = closest_point_on_mesh(&[0.0, 0.0, 0.0], &[], &[]);
        assert_eq!(idx, usize::MAX);
        assert!(sq.is_infinite());
    }
}

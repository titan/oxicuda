//! 2D (constrained) Delaunay triangulation via the Bowyer–Watson algorithm.
//!
//! This module exposes the [`Delaunay`] zero-sized type with four `pub`
//! associated functions:
//!
//! * [`Delaunay::orient_2d`] — signed-area determinant of three points
//!   (positive for a CCW triple).
//! * [`Delaunay::in_circle`] — classical in-circle predicate (positive when
//!   the query point is strictly inside the circumscribed circle of the CCW
//!   triple).
//! * [`Delaunay::triangulate`] — unconstrained Delaunay triangulation by
//!   *incremental insertion* (Bowyer–Watson).
//! * [`Delaunay::constrained_triangulate`] — constrained Delaunay: after the
//!   plain triangulation each requested edge is recovered through a sequence
//!   of local edge swaps (flips).
//!
//! All algorithms operate on `f64` coordinates, use deterministic insertion
//! orders, and return triangles with consistently *counter-clockwise* vertex
//! ordering.

use std::collections::HashMap;

use crate::error::{PdeError, PdeResult};

/// Output of a Delaunay triangulation.
#[derive(Debug, Clone)]
pub struct DelaunayResult {
    /// One triple of input-point indices per triangle, CCW.
    pub triangles: Vec<[usize; 3]>,
    /// Number of input vertices used to compute the triangulation.
    pub n_input_vertices: usize,
}

/// Tag type that namespaces the public functions.
pub struct Delaunay;

/// Numerical noise floor used when comparing predicate determinants to zero.
const PREDICATE_EPS: f64 = 1.0e-12;
/// Maximum number of flip iterations during constrained-edge recovery.
/// `8 N²` is a generous bound; the loop normally terminates much sooner.
const MAX_FLIPS_PER_EDGE: usize = 4096;

// ── Geometric predicates ──────────────────────────────────────────────────────

impl Delaunay {
    /// Twice the signed area of triangle `(a, b, c)`.
    ///
    /// Positive  ⇒ CCW orientation.<br>
    /// Negative  ⇒ CW orientation.<br>
    /// Zero      ⇒ collinear.
    #[inline]
    pub fn orient_2d(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> f64 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }

    /// In-circle predicate.
    ///
    /// Returns the signed determinant
    /// `| ax−dx ay−dy (ax−dx)²+(ay−dy)² |`
    /// `| bx−dx by−dy (bx−dx)²+(by−dy)² |`
    /// `| cx−dx cy−dy (cx−dx)²+(cy−dy)² |`.
    /// When `(a, b, c)` are oriented CCW, the value is positive iff `d` is
    /// strictly inside the circumcircle.
    #[inline]
    pub fn in_circle(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2], d: &[f64; 2]) -> f64 {
        let ax = a[0] - d[0];
        let ay = a[1] - d[1];
        let bx = b[0] - d[0];
        let by = b[1] - d[1];
        let cx = c[0] - d[0];
        let cy = c[1] - d[1];
        let a2 = ax * ax + ay * ay;
        let b2 = bx * bx + by * by;
        let c2 = cx * cx + cy * cy;
        ax * (by * c2 - b2 * cy) - ay * (bx * c2 - b2 * cx) + a2 * (bx * cy - by * cx)
    }

    /// Compute the Delaunay triangulation of `points` by incremental
    /// Bowyer-Watson insertion.
    ///
    /// # Errors
    /// * `Err(InvalidParameter)` if `points.len() < 3` or any pair of points
    ///   coincides.
    /// * `Err(InvalidGrid)` if every input point is collinear (no valid
    ///   triangle can be produced).
    pub fn triangulate(points: &[[f64; 2]]) -> PdeResult<DelaunayResult> {
        triangulate_impl(points, &[])
    }

    /// Compute a *constrained* Delaunay triangulation: after the plain
    /// Bowyer-Watson pass each requested edge `(i, j)` is recovered through
    /// edge swaps along the line.
    ///
    /// `constrained_edges` must contain pairs with `i < j` and both indices
    /// `< points.len()`.
    pub fn constrained_triangulate(
        points: &[[f64; 2]],
        constrained_edges: &[(usize, usize)],
    ) -> PdeResult<DelaunayResult> {
        // Validate constraints first so the user gets clean errors before any
        // computation work.
        for &(i, j) in constrained_edges {
            if i >= points.len() || j >= points.len() {
                return Err(PdeError::IndexOutOfBounds {
                    index: i.max(j),
                    len: points.len(),
                });
            }
            if i >= j {
                return Err(PdeError::InvalidParameter {
                    name: "constrained_edges".to_string(),
                    reason: format!("expect i<j, got ({i},{j})"),
                });
            }
        }
        triangulate_impl(points, constrained_edges)
    }
}

// ── Core Bowyer-Watson driver ────────────────────────────────────────────────

/// Internal state: triangle = three vertex indices into a combined point list
/// that contains the user's `points` followed by 3 super-triangle vertices.
#[derive(Clone)]
struct Tri {
    v: [usize; 3],
    alive: bool,
}

fn triangulate_impl(
    points: &[[f64; 2]],
    constrained_edges: &[(usize, usize)],
) -> PdeResult<DelaunayResult> {
    if points.len() < 3 {
        return Err(PdeError::InvalidParameter {
            name: "points".to_string(),
            reason: format!("need >= 3 points, got {}", points.len()),
        });
    }
    // Detect coincident points up front.
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            if (points[i][0] - points[j][0]).abs() < PREDICATE_EPS
                && (points[i][1] - points[j][1]).abs() < PREDICATE_EPS
            {
                return Err(PdeError::InvalidParameter {
                    name: "points".to_string(),
                    reason: format!("duplicate point at indices {i} and {j}"),
                });
            }
        }
    }

    // Combined point storage: points[0..n] are user inputs;
    // points[n..n+3] are the super-triangle vertices.
    let n = points.len();
    let mut all_points: Vec<[f64; 2]> = points.to_vec();
    let (s0, s1, s2) = make_super_triangle(points);
    all_points.push(s0);
    all_points.push(s1);
    all_points.push(s2);

    // Initial mesh: the super-triangle (CCW).
    let super_idx = [n, n + 1, n + 2];
    let mut tris: Vec<Tri> = Vec::with_capacity(n * 2 + 4);
    tris.push(Tri {
        v: ccw_triple(&all_points, super_idx[0], super_idx[1], super_idx[2]),
        alive: true,
    });

    // Incremental insertion of the user points in input order.
    for p_idx in 0..n {
        insert_point(p_idx, &all_points, &mut tris)?;
    }

    // Enforce constrained edges (if any).  This stage operates on the full
    // mesh (super-triangle still present) so we can swap edges adjacent to
    // either side of the constraint freely; super-triangle triangles are
    // discarded afterwards.
    for &(i, j) in constrained_edges {
        enforce_constrained_edge(i, j, &all_points, &mut tris)?;
    }

    // Drop triangles that touch the super-triangle.
    let mut output: Vec<[usize; 3]> = Vec::new();
    for tri in tris.iter() {
        if !tri.alive {
            continue;
        }
        if tri.v.iter().any(|&v| v >= n) {
            continue;
        }
        // Ensure CCW orientation in the result (insertion preserves it, but
        // edge swaps can introduce inversions for degenerate constraint cases).
        let v0 = tri.v[0];
        let v1 = tri.v[1];
        let v2 = tri.v[2];
        let orient = Delaunay::orient_2d(&all_points[v0], &all_points[v1], &all_points[v2]);
        if orient > PREDICATE_EPS {
            output.push([v0, v1, v2]);
        } else if orient < -PREDICATE_EPS {
            output.push([v0, v2, v1]);
        }
        // Skip degenerate triangles (orient ≈ 0).
    }

    if output.is_empty() {
        return Err(PdeError::InvalidGrid(
            "Delaunay triangulation produced no triangles (collinear input?)".to_string(),
        ));
    }

    Ok(DelaunayResult {
        triangles: output,
        n_input_vertices: n,
    })
}

// ── Super-triangle construction ──────────────────────────────────────────────

/// Build a CCW super-triangle that strictly encloses every input point.
fn make_super_triangle(points: &[[f64; 2]]) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let mut x_min = f64::INFINITY;
    let mut y_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for p in points {
        if p[0] < x_min {
            x_min = p[0];
        }
        if p[1] < y_min {
            y_min = p[1];
        }
        if p[0] > x_max {
            x_max = p[0];
        }
        if p[1] > y_max {
            y_max = p[1];
        }
    }
    let dx = (x_max - x_min).max(1.0);
    let dy = (y_max - y_min).max(1.0);
    let mid_x = 0.5 * (x_min + x_max);
    let mid_y = 0.5 * (y_min + y_max);
    let r = 100.0 * dx.max(dy);
    // Equilateral-ish triangle around the bounding box, CCW.
    let s0 = [mid_x - 3.0 * r, mid_y - r];
    let s1 = [mid_x + 3.0 * r, mid_y - r];
    let s2 = [mid_x, mid_y + 3.0 * r];
    (s0, s1, s2)
}

/// Reorder `(a, b, c)` so the resulting triple is CCW.
fn ccw_triple(pts: &[[f64; 2]], a: usize, b: usize, c: usize) -> [usize; 3] {
    if Delaunay::orient_2d(&pts[a], &pts[b], &pts[c]) > 0.0 {
        [a, b, c]
    } else {
        [a, c, b]
    }
}

// ── Bowyer-Watson incremental insertion ──────────────────────────────────────

fn insert_point(p_idx: usize, pts: &[[f64; 2]], tris: &mut Vec<Tri>) -> PdeResult<()> {
    // Step 1: find every triangle whose circumcircle contains the new point.
    let p = pts[p_idx];
    let mut bad: Vec<usize> = Vec::new();
    for (i, tri) in tris.iter().enumerate() {
        if !tri.alive {
            continue;
        }
        // The triangle is already oriented CCW, so a strictly positive
        // in-circle value means the new point is inside the circumcircle.
        let v = tri.v;
        let ic = Delaunay::in_circle(&pts[v[0]], &pts[v[1]], &pts[v[2]], &p);
        if ic > PREDICATE_EPS {
            bad.push(i);
        }
    }
    if bad.is_empty() {
        // Numerical edge case: the point lies *on* every cavity boundary.
        // Fall back to a containing-triangle test so a hopelessly degenerate
        // input never silently succeeds.
        for (i, tri) in tris.iter().enumerate() {
            if !tri.alive {
                continue;
            }
            if point_in_triangle_strict(&p, &pts[tri.v[0]], &pts[tri.v[1]], &pts[tri.v[2]]) {
                bad.push(i);
                break;
            }
        }
    }
    if bad.is_empty() {
        return Err(PdeError::InvalidGrid(format!(
            "Bowyer-Watson: point {p_idx} outside super-triangle (unexpected)"
        )));
    }

    // Step 2: compute the cavity boundary edges (those appearing in exactly
    // one bad triangle).
    let mut edge_counts: HashMap<(usize, usize), usize> = HashMap::new();
    let mut edge_order: Vec<(usize, usize, usize, usize)> = Vec::new();
    //               (key.0, key.1, oriented_a, oriented_b)

    for &b in &bad {
        let v = tris[b].v;
        let edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])];
        for (a, c) in edges {
            let key = if a < c { (a, c) } else { (c, a) };
            let entry = edge_counts.entry(key).or_insert_with(|| {
                edge_order.push((key.0, key.1, a, c));
                0_usize
            });
            *entry += 1;
        }
    }

    // Step 3: mark bad triangles dead.
    for &b in &bad {
        tris[b].alive = false;
    }

    // Step 4: for every boundary edge (count == 1), add a new triangle
    // joining that edge to p_idx.  The original oriented (a, c) for that
    // edge keeps the new triangle CCW.
    for &(k0, k1, oa, oc) in &edge_order {
        if let Some(&count) = edge_counts.get(&(k0, k1)) {
            if count == 1 {
                let new_tri = Tri {
                    v: ccw_triple(pts, oa, oc, p_idx),
                    alive: true,
                };
                tris.push(new_tri);
            }
        }
    }
    Ok(())
}

/// Strict containment: returns true iff `p` is strictly inside the triangle
/// `(a, b, c)` (boundary excluded).
fn point_in_triangle_strict(p: &[f64; 2], a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> bool {
    let d1 = Delaunay::orient_2d(p, a, b);
    let d2 = Delaunay::orient_2d(p, b, c);
    let d3 = Delaunay::orient_2d(p, c, a);
    let has_neg = d1 < -PREDICATE_EPS || d2 < -PREDICATE_EPS || d3 < -PREDICATE_EPS;
    let has_pos = d1 > PREDICATE_EPS || d2 > PREDICATE_EPS || d3 > PREDICATE_EPS;
    !(has_neg && has_pos)
}

// ── Constrained-edge recovery via local flips ────────────────────────────────

fn enforce_constrained_edge(
    i: usize,
    j: usize,
    pts: &[[f64; 2]],
    tris: &mut [Tri],
) -> PdeResult<()> {
    // If the edge is already present, nothing to do.
    if has_edge(tris, i, j) {
        return Ok(());
    }
    for _ in 0..MAX_FLIPS_PER_EDGE {
        if has_edge(tris, i, j) {
            return Ok(());
        }
        // Find some triangle whose edge crosses segment (i, j) properly and
        // try to flip it with its neighbour.
        if !flip_one_crossing(i, j, pts, tris)? {
            return Err(PdeError::InvalidGrid(format!(
                "constrained Delaunay: failed to recover edge ({i},{j})"
            )));
        }
    }
    if has_edge(tris, i, j) {
        Ok(())
    } else {
        Err(PdeError::InvalidGrid(format!(
            "constrained Delaunay: edge ({i},{j}) not recovered in {MAX_FLIPS_PER_EDGE} flips"
        )))
    }
}

fn has_edge(tris: &[Tri], i: usize, j: usize) -> bool {
    for tri in tris {
        if !tri.alive {
            continue;
        }
        let v = tri.v;
        let pairs = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])];
        for (a, b) in pairs {
            if (a == i && b == j) || (a == j && b == i) {
                return true;
            }
        }
    }
    false
}

/// Look for one pair of adjacent triangles whose shared edge properly
/// crosses segment `(i, j)` and is *flippable* (the quadrilateral formed by
/// the two triangles is strictly convex).  Perform the flip in place.
/// Returns `Ok(true)` if a flip was made, `Ok(false)` if no candidate found.
fn flip_one_crossing(i: usize, j: usize, pts: &[[f64; 2]], tris: &mut [Tri]) -> PdeResult<bool> {
    // Build edge → list of (tri_index, opposite_vertex) for live edges.
    let mut edge_to_tris: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for (idx, tri) in tris.iter().enumerate() {
        if !tri.alive {
            continue;
        }
        let v = tri.v;
        let edges = [
            ((v[0], v[1]), v[2]),
            ((v[1], v[2]), v[0]),
            ((v[2], v[0]), v[1]),
        ];
        for ((a, b), c) in edges {
            let key = if a < b { (a, b) } else { (b, a) };
            edge_to_tris.entry(key).or_default().push((idx, c));
        }
    }

    // Look for a shared edge whose two endpoints differ from {i, j} and
    // whose segment properly crosses (i, j).
    for ((a, b), incident) in edge_to_tris {
        if a == i || a == j || b == i || b == j {
            continue;
        }
        if incident.len() != 2 {
            continue;
        }
        let pi = pts[i];
        let pj = pts[j];
        let pa = pts[a];
        let pb = pts[b];
        if !segments_properly_cross(&pi, &pj, &pa, &pb) {
            continue;
        }
        let (t1_idx, opp1) = incident[0];
        let (t2_idx, opp2) = incident[1];
        if opp1 == opp2 {
            continue;
        }
        // The quadrilateral (a, opp2, b, opp1) — diagonal (a, b) → swap to
        // (opp1, opp2).  Require strict convexity to keep all triangles
        // CCW after the flip.
        if !is_strictly_convex_quad(&pts[a], &pts[opp1], &pts[b], &pts[opp2]) {
            continue;
        }
        // Replace t1 = (a, b, ?) and t2 = (a, b, ?) with (a, opp1, opp2)
        // and (b, opp2, opp1), each oriented CCW.
        let new_t1 = ccw_triple(pts, a, opp1, opp2);
        let new_t2 = ccw_triple(pts, b, opp2, opp1);
        tris[t1_idx].v = new_t1;
        tris[t2_idx].v = new_t2;
        return Ok(true);
    }
    Ok(false)
}

/// Strict (non-collinear) proper crossing of open segments `(p1, p2)` and
/// `(p3, p4)`.
fn segments_properly_cross(p1: &[f64; 2], p2: &[f64; 2], p3: &[f64; 2], p4: &[f64; 2]) -> bool {
    let d1 = Delaunay::orient_2d(p3, p4, p1);
    let d2 = Delaunay::orient_2d(p3, p4, p2);
    let d3 = Delaunay::orient_2d(p1, p2, p3);
    let d4 = Delaunay::orient_2d(p1, p2, p4);
    let s1 = d1 > PREDICATE_EPS;
    let s2 = d2 > PREDICATE_EPS;
    let s3 = d3 > PREDICATE_EPS;
    let s4 = d4 > PREDICATE_EPS;
    let n1 = d1 < -PREDICATE_EPS;
    let n2 = d2 < -PREDICATE_EPS;
    let n3 = d3 < -PREDICATE_EPS;
    let n4 = d4 < -PREDICATE_EPS;
    (s1 != s2 || n1 != n2)
        && (s3 != s4 || n3 != n4)
        && (s1 || n1)
        && (s2 || n2)
        && (s3 || n3)
        && (s4 || n4)
}

/// Check that vertices `(p0, p1, p2, p3)` (in CCW order) form a strictly
/// convex quadrilateral.
fn is_strictly_convex_quad(p0: &[f64; 2], p1: &[f64; 2], p2: &[f64; 2], p3: &[f64; 2]) -> bool {
    let o1 = Delaunay::orient_2d(p0, p1, p2);
    let o2 = Delaunay::orient_2d(p1, p2, p3);
    let o3 = Delaunay::orient_2d(p2, p3, p0);
    let o4 = Delaunay::orient_2d(p3, p0, p1);
    let positive =
        o1 > PREDICATE_EPS && o2 > PREDICATE_EPS && o3 > PREDICATE_EPS && o4 > PREDICATE_EPS;
    let negative =
        o1 < -PREDICATE_EPS && o2 < -PREDICATE_EPS && o3 < -PREDICATE_EPS && o4 < -PREDICATE_EPS;
    positive || negative
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1.0e-9;

    // ── orient_2d ─────────────────────────────────────────────────────────

    #[test]
    fn orient_ccw_positive() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        assert!(
            Delaunay::orient_2d(&a, &b, &c) > 0.0,
            "CCW must be positive"
        );
    }

    #[test]
    fn orient_cw_negative() {
        let a = [0.0, 0.0];
        let b = [0.0, 1.0];
        let c = [1.0, 0.0];
        assert!(Delaunay::orient_2d(&a, &b, &c) < 0.0, "CW must be negative");
    }

    #[test]
    fn orient_collinear_zero() {
        let a = [0.0, 0.0];
        let b = [1.0, 1.0];
        let c = [2.0, 2.0];
        let o = Delaunay::orient_2d(&a, &b, &c);
        assert!(o.abs() < TOL, "collinear must be zero, got {o}");
    }

    // ── in_circle ─────────────────────────────────────────────────────────

    #[test]
    fn in_circle_center_of_equilateral_zero() {
        // Equilateral triangle centred at origin, circumradius 1.
        let a = [1.0_f64, 0.0];
        let third = 2.0 * std::f64::consts::PI / 3.0;
        let b = [third.cos(), third.sin()];
        let c = [(2.0 * third).cos(), (2.0 * third).sin()];
        let centre = [0.0_f64, 0.0];
        let v = Delaunay::in_circle(&a, &b, &c, &centre);
        // The centre of an equilateral triangle's circumcircle is inside; expected positive.
        assert!(v > 0.0, "centre must lie strictly inside circumcircle");
    }

    #[test]
    fn in_circle_on_circumcircle_is_zero() {
        // Right triangle (0,0)-(2,0)-(0,2); circumcircle centre (1,1), radius √2.
        let a = [0.0, 0.0];
        let b = [2.0, 0.0];
        let c = [0.0, 2.0];
        // (2,2) is on the circumcircle.
        let d = [2.0_f64, 2.0];
        let v = Delaunay::in_circle(&a, &b, &c, &d);
        assert!(
            v.abs() < 1.0e-9,
            "point on circumcircle: in_circle≈0, got {v}"
        );
    }

    #[test]
    fn in_circle_outside_negative() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        // Point far away.
        let d = [10.0, 10.0];
        let v = Delaunay::in_circle(&a, &b, &c, &d);
        assert!(v < 0.0, "far-away point should be outside circumcircle");
    }

    // ── triangulate basics ────────────────────────────────────────────────

    #[test]
    fn triangulate_three_points_single_triangle() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let result = Delaunay::triangulate(&pts).expect("ok");
        assert_eq!(result.triangles.len(), 1);
        assert_eq!(result.n_input_vertices, 3);
    }

    #[test]
    fn triangulate_square_two_triangles() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let result = Delaunay::triangulate(&pts).expect("ok");
        assert_eq!(result.triangles.len(), 2);
        // All triangles must reference only user-input vertex indices.
        for t in &result.triangles {
            for &v in t {
                assert!(v < pts.len());
            }
        }
    }

    #[test]
    fn triangulate_all_ccw() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]];
        let result = Delaunay::triangulate(&pts).expect("ok");
        for t in &result.triangles {
            let o = Delaunay::orient_2d(&pts[t[0]], &pts[t[1]], &pts[t[2]]);
            assert!(o > 0.0, "every output triangle must be CCW, got {o}");
        }
    }

    #[test]
    fn triangulate_euler_bound() {
        // n = 5 points in convex position; the bound 2n−5 (boundary triangles).
        let pts = vec![[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.5, 2.5], [-0.5, 1.0]];
        let result = Delaunay::triangulate(&pts).expect("ok");
        assert!(
            result.triangles.len() <= 2 * pts.len(),
            "Euler bound violated: {} > 2n",
            result.triangles.len()
        );
        // A convex 5-gon triangulates to 3 triangles.
        assert_eq!(result.triangles.len(), 3);
    }

    #[test]
    fn triangulate_five_point_convex_polygon() {
        let pts = vec![[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.5, 2.5], [-0.5, 1.0]];
        let result = Delaunay::triangulate(&pts).expect("ok");
        // Three triangles, total area = polygon area = 5.5 (shoelace).
        let mut total_area = 0.0;
        for t in &result.triangles {
            total_area += 0.5 * Delaunay::orient_2d(&pts[t[0]], &pts[t[1]], &pts[t[2]]).abs();
        }
        // Polygon area via shoelace.
        let mut polygon_area = 0.0;
        let n = pts.len();
        for i in 0..n {
            let j = (i + 1) % n;
            polygon_area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
        }
        let polygon_area = 0.5 * polygon_area.abs();
        assert!(
            (total_area - polygon_area).abs() < 1.0e-9,
            "Σ triangle area {total_area} ≠ polygon area {polygon_area}"
        );
    }

    #[test]
    fn triangulate_errors_too_few_points() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0]];
        let res = Delaunay::triangulate(&pts);
        assert!(res.is_err());
    }

    #[test]
    fn triangulate_errors_duplicate_points() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let res = Delaunay::triangulate(&pts);
        assert!(res.is_err());
    }

    #[test]
    fn triangulate_errors_collinear_input() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let res = Delaunay::triangulate(&pts);
        assert!(res.is_err(), "collinear inputs must error");
    }

    #[test]
    fn triangulate_deterministic() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]];
        let a = Delaunay::triangulate(&pts).expect("ok");
        let b = Delaunay::triangulate(&pts).expect("ok");
        assert_eq!(a.triangles, b.triangles);
    }

    // ── Constrained Delaunay ──────────────────────────────────────────────

    #[test]
    fn constrained_already_present_is_noop() {
        // Square + the diagonal that the unconstrained algorithm already picks.
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let plain = Delaunay::triangulate(&pts).expect("ok");
        // Find the actual diagonal used by plain Delaunay.
        let mut diag = (0_usize, 0_usize);
        for t in &plain.triangles {
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let is_boundary_edge = matches!((lo, hi), (0, 1) | (1, 2) | (2, 3) | (0, 3));
                if !is_boundary_edge {
                    diag = (lo, hi);
                }
            }
        }
        let constrained = Delaunay::constrained_triangulate(&pts, &[diag]).expect("ok");
        assert_eq!(constrained.triangles.len(), plain.triangles.len());
    }

    #[test]
    fn constrained_forces_non_delaunay_diagonal() {
        // A square has two possible diagonals; for a perfect square both are
        // equally Delaunay so plain triangulate is degenerate. Use a slightly
        // skewed square where one diagonal is strictly Delaunay and the other
        // is not.
        // Square slightly tilted so the Delaunay diagonal is (0,2). Force (1,3).
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.2, 1.0], [0.2, 1.0]];
        let constrained = Delaunay::constrained_triangulate(&pts, &[(1, 3)]).expect("ok");
        // Check the diagonal is now (1, 3).
        let mut has_13 = false;
        for t in &constrained.triangles {
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if (lo, hi) == (1, 3) {
                    has_13 = true;
                }
            }
        }
        assert!(
            has_13,
            "constrained edge (1,3) must appear in triangulation"
        );
    }

    #[test]
    fn constrained_errors_index_out_of_range() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let res = Delaunay::constrained_triangulate(&pts, &[(0, 99)]);
        assert!(res.is_err());
    }

    #[test]
    fn constrained_errors_i_geq_j() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let res = Delaunay::constrained_triangulate(&pts, &[(2, 1)]);
        assert!(res.is_err());
    }

    #[test]
    fn constrained_deterministic() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.2, 1.0], [0.2, 1.0], [0.5, 0.5]];
        let a = Delaunay::constrained_triangulate(&pts, &[(0, 2)]).expect("ok");
        let b = Delaunay::constrained_triangulate(&pts, &[(0, 2)]).expect("ok");
        assert_eq!(a.triangles, b.triangles);
    }
}

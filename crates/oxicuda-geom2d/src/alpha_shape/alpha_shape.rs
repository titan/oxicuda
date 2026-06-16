//! 2D alpha-shape construction over a Delaunay triangulation.
//!
//! # Convention: `alpha` is a RADIUS
//!
//! Throughout this module `alpha` is interpreted as a **radius** (the same units
//! as the input coordinates), following the radius formulation of
//! Edelsbrunner-Kirkpatrick-Seidel. A Delaunay triangle is part of the
//! `alpha`-complex iff its circumradius `r` satisfies `r <= alpha`. As
//! `alpha -> +inf` every Delaunay triangle is kept, so the boundary converges to
//! the convex hull. As `alpha -> 0` (or below the minimum triangle circumradius)
//! no triangle is kept and the shape is empty.
//!
//! Some texts parameterise alpha shapes by `alpha = 1 / r^2` (signed); the radius
//! convention used here is monotone in the *opposite* direction (larger radius =>
//! more inclusive), which is the most intuitive for "ball of radius alpha rolling
//! over the points".
//!
//! # Components produced
//!
//! * `triangles` — the retained Delaunay triangles (circumradius `<= alpha`),
//!   forming the *general* / *interior* part of the alpha-complex.
//! * `boundary_edges` — the boundary of the alpha-shape: undirected edges that
//!   are incident to **at most one** retained triangle. This is exactly the set
//!   of `alpha`-exposed edges of the retained region (the "regular" and
//!   "singular" boundary of the complex). Edges interior to the retained region
//!   (shared by two retained triangles) are *not* boundary edges.
//!
//! # Robustness
//!
//! Circumradii are computed from the circumcenter; near-degenerate (sliver)
//! triangles whose circumcenter is numerically unstable are treated as having an
//! effectively infinite circumradius so they are only ever admitted for very
//! large `alpha`, and are guarded against division blow-ups. All orientation /
//! degeneracy reasoning reuses the crate predicates.

use crate::error::{Geom2dError, Geom2dResult};
use crate::predicate::orientation::orient_value;
use crate::primitives::point::Point;
use crate::triangulation::bowyer_watson_delaunay::{Triangle, bowyer_watson};

/// Result of an alpha-shape computation.
#[derive(Debug, Clone)]
pub struct AlphaShape {
    /// Retained Delaunay triangles (circumradius `<= alpha`), as index triples
    /// into the original point slice.
    pub triangles: Vec<[usize; 3]>,
    /// Boundary edges of the alpha-shape as undirected index pairs `(lo, hi)`
    /// with `lo < hi`. Each is incident to at most one retained triangle.
    pub boundary_edges: Vec<[usize; 2]>,
}

/// Tolerance for degeneracy / circumcenter stability.
const EPS: f64 = 1e-12;

/// Circumradius of triangle `(a, b, c)`.
///
/// Returns `None` only for an *exactly* (or numerically indistinguishably)
/// degenerate triangle whose three points are collinear, for which the
/// circumradius is undefined (infinite). A thin-but-valid sliver triangle has a
/// large *finite* circumradius and is returned as such, so that at large `alpha`
/// every genuine Delaunay triangle is retained (and the convex hull is fully
/// recovered) rather than dropping boundary slivers.
#[must_use]
pub fn circumradius(a: Point, b: Point, c: Point) -> Option<f64> {
    // r = (|AB| * |BC| * |CA|) / (4 * area). Using the signed area keeps it robust.
    let area2 = orient_value(a, b, c); // 2 * signed area
    let ab = a.distance(b);
    let bc = b.distance(c);
    let ca = c.distance(a);
    // Relative collinearity floor: treat as degenerate only when the area is
    // negligible compared to the product of the longest two edges (a true
    // straight-line configuration), not merely thin.
    let scale = ab.max(bc).max(ca);
    let floor = (scale * scale).max(1.0) * 1e-300;
    if area2.abs() <= floor {
        return None;
    }
    let r = (ab * bc * ca) / (2.0 * area2.abs());
    if r.is_finite() { Some(r) } else { None }
}

/// Sort a pair into `[lo, hi]` ordering.
fn sorted_pair(u: usize, v: usize) -> [usize; 2] {
    if u < v { [u, v] } else { [v, u] }
}

/// Build the alpha-shape of `points` for radius parameter `alpha`.
///
/// Triangles of the Delaunay triangulation with circumradius `<= alpha` are
/// retained; boundary edges are those incident to at most one retained triangle.
///
/// # Errors
///
/// * [`Geom2dError::NotEnoughPoints`] if fewer than 3 points are supplied.
/// * [`Geom2dError::InvalidParameter`] if `alpha` is negative or not finite.
/// * [`Geom2dError::DegeneratePolygon`] (propagated from the triangulator) if
///   all points are collinear.
pub fn alpha_shape(points: &[Point], alpha: f64) -> Geom2dResult<AlphaShape> {
    if points.len() < 3 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 3,
            got: points.len(),
        });
    }
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(Geom2dError::InvalidParameter(
            "alpha must be a finite non-negative radius".into(),
        ));
    }

    let tris = bowyer_watson(points)?;
    Ok(build_from_triangulation(points, &tris, alpha))
}

/// Build the alpha-complex from an existing triangulation and radius `alpha`.
fn build_from_triangulation(points: &[Point], tris: &[Triangle], alpha: f64) -> AlphaShape {
    let mut kept: Vec<[usize; 3]> = Vec::new();
    // Count incidence of each undirected edge among retained triangles.
    let mut edge_incidence: std::collections::HashMap<[usize; 2], usize> =
        std::collections::HashMap::new();

    for t in tris {
        let pa = points[t.a];
        let pb = points[t.b];
        let pc = points[t.c];
        let keep = match circumradius(pa, pb, pc) {
            // Tolerant `<=` so triangles exactly at radius alpha are retained.
            Some(r) => r <= alpha + EPS.max(alpha * 1e-12),
            None => false, // sliver: infinite circumradius, never retained.
        };
        if keep {
            kept.push([t.a, t.b, t.c]);
            for (u, v) in [(t.a, t.b), (t.b, t.c), (t.c, t.a)] {
                *edge_incidence.entry(sorted_pair(u, v)).or_insert(0) += 1;
            }
        }
    }

    // Boundary edges: incident to exactly one retained triangle.
    let mut boundary_edges: Vec<[usize; 2]> = edge_incidence
        .into_iter()
        .filter(|&(_, count)| count == 1)
        .map(|(e, _)| e)
        .collect();
    boundary_edges.sort_unstable();

    AlphaShape {
        triangles: kept,
        boundary_edges,
    }
}

/// Spectrum of distinct triangle circumradii (ascending), i.e. the `alpha`
/// values at which retained-triangle membership changes.
///
/// Useful for sweeping the full family of alpha shapes; `alpha_shape_auto`
/// consults it to find the connectivity threshold.
///
/// # Errors
///
/// Propagates triangulation errors as in [`alpha_shape`].
pub fn alpha_spectrum(points: &[Point]) -> Geom2dResult<Vec<f64>> {
    if points.len() < 3 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 3,
            got: points.len(),
        });
    }
    let tris = bowyer_watson(points)?;
    let mut radii: Vec<f64> = Vec::with_capacity(tris.len());
    for t in &tris {
        if let Some(r) = circumradius(points[t.a], points[t.b], points[t.c]) {
            radii.push(r);
        }
    }
    radii.sort_by(|x, y| x.partial_cmp(y).unwrap_or(core::cmp::Ordering::Equal));
    radii.dedup_by(|x, y| (*x - *y).abs() < EPS);
    Ok(radii)
}

/// Result of [`alpha_shape_auto`]: the chosen `alpha` plus the resulting shape.
#[derive(Debug, Clone)]
pub struct AutoAlphaShape {
    /// The smallest `alpha` (a circumradius from the spectrum) for which every
    /// input point is covered by the retained triangulation (the shape is
    /// "connected" / fully supported).
    pub alpha: f64,
    /// The alpha-shape at that `alpha`.
    pub shape: AlphaShape,
}

/// Automatically choose `alpha` as the smallest spectrum value at which the
/// retained triangulation covers **every** input vertex (each point participates
/// in at least one retained triangle), then return that alpha-shape.
///
/// This is the standard "smallest alpha keeping the shape connected/supported"
/// heuristic: it yields the tightest non-convex outline that still spans all
/// points. If even the largest circumradius fails to cover all points (possible
/// only with duplicate / collinear strays), the convex-hull-equivalent maximal
/// alpha is returned.
///
/// # Errors
///
/// Propagates triangulation errors as in [`alpha_shape`].
pub fn alpha_shape_auto(points: &[Point]) -> Geom2dResult<AutoAlphaShape> {
    let spectrum = alpha_spectrum(points)?;
    let tris = bowyer_watson(points)?;
    let n = points.len();

    // Find the smallest spectrum threshold that covers all vertices.
    let mut chosen = spectrum.last().copied().unwrap_or(0.0);
    for &alpha in &spectrum {
        if covers_all_vertices(points, &tris, alpha, n) {
            chosen = alpha;
            break;
        }
    }
    let shape = build_from_triangulation(points, &tris, chosen);
    Ok(AutoAlphaShape {
        alpha: chosen,
        shape,
    })
}

/// True if every vertex index `0..n` belongs to at least one retained triangle
/// (circumradius `<= alpha`).
fn covers_all_vertices(points: &[Point], tris: &[Triangle], alpha: f64, n: usize) -> bool {
    let mut covered = vec![false; n];
    let tol = EPS.max(alpha * 1e-12);
    for t in tris {
        let keep = match circumradius(points[t.a], points[t.b], points[t.c]) {
            Some(r) => r <= alpha + tol,
            None => false,
        };
        if keep {
            covered[t.a] = true;
            covered[t.b] = true;
            covered[t.c] = true;
        }
    }
    covered.iter().all(|&c| c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::hull::andrew_monotone_chain::andrew_monotone_chain;

    fn square_pts() -> Vec<Point> {
        vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ]
    }

    fn total_kept_area(shape: &AlphaShape, pts: &[Point]) -> f64 {
        let mut area = 0.0;
        for t in &shape.triangles {
            area += orient_value(pts[t[0]], pts[t[1]], pts[t[2]]).abs() / 2.0;
        }
        area
    }

    // Oracle (a): very large alpha recovers the convex hull boundary exactly.
    #[test]
    fn large_alpha_recovers_hull() {
        let mut rng = LcgRng::new(7);
        let pts: Vec<Point> = (0..40)
            .map(|_| Point::new(rng.next_f64() * 10.0, rng.next_f64() * 10.0))
            .collect();
        let shape = alpha_shape(&pts, 1.0e6).expect("ok");
        // Boundary-edge vertex set must equal the convex-hull vertex set.
        let mut boundary_vertices: Vec<usize> = shape
            .boundary_edges
            .iter()
            .flat_map(|e| [e[0], e[1]])
            .collect();
        boundary_vertices.sort_unstable();
        boundary_vertices.dedup();

        let hull = andrew_monotone_chain(&pts).expect("hull");
        // Map hull points back to indices.
        let mut hull_idx: Vec<usize> = hull
            .iter()
            .map(|hp| {
                pts.iter()
                    .position(|p| (p.x - hp.x).abs() < 1e-9 && (p.y - hp.y).abs() < 1e-9)
                    .expect("hull point is an input point")
            })
            .collect();
        hull_idx.sort_unstable();
        hull_idx.dedup();
        assert_eq!(
            boundary_vertices, hull_idx,
            "large-alpha boundary must equal convex hull vertices"
        );

        // Retained-triangle total area must equal the convex-hull area.
        let hull_poly = crate::primitives::polygon::Polygon::new(hull).expect("poly");
        assert!((total_kept_area(&shape, &pts) - hull_poly.area()).abs() < 1e-7);
    }

    // Oracle (b): alpha below the minimum circumradius -> no triangles.
    #[test]
    fn small_alpha_empty() {
        let pts = square_pts();
        let shape = alpha_shape(&pts, 1e-6).expect("ok");
        assert!(shape.triangles.is_empty());
        assert!(shape.boundary_edges.is_empty());
    }

    // Oracle (c): every kept triangle has circumradius <= alpha; every boundary
    // edge is incident to <= 1 kept triangle.
    #[test]
    fn invariants_hold() {
        let mut rng = LcgRng::new(99);
        let pts: Vec<Point> = (0..30)
            .map(|_| Point::new(rng.next_f64() * 5.0, rng.next_f64() * 5.0))
            .collect();
        let alpha = 1.2;
        let shape = alpha_shape(&pts, alpha).expect("ok");
        for t in &shape.triangles {
            let r = circumradius(pts[t[0]], pts[t[1]], pts[t[2]]).expect("non-degenerate");
            assert!(
                r <= alpha + 1e-9,
                "kept triangle radius {r} > alpha {alpha}"
            );
        }
        // Recount incidence directly.
        let mut inc: std::collections::HashMap<[usize; 2], usize> =
            std::collections::HashMap::new();
        for t in &shape.triangles {
            for (u, v) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *inc.entry(sorted_pair(u, v)).or_insert(0) += 1;
            }
        }
        for e in &shape.boundary_edges {
            let count = inc.get(e).copied().unwrap_or(0);
            assert!(
                count <= 1,
                "boundary edge {e:?} incident to {count} triangles"
            );
            assert!(count >= 1, "boundary edge must belong to a kept triangle");
        }
    }

    // Oracle (d): a 4-point square at a known alpha gives the known complex.
    #[test]
    fn square_known_complex() {
        // The unit square triangulates into 2 triangles, each with circumradius
        // sqrt(2)/2 ~= 0.7071 (right isosceles, hypotenuse = diameter).
        let pts = square_pts();
        let r = std::f64::consts::SQRT_2 / 2.0;

        // alpha just below r: no triangle kept.
        let below = alpha_shape(&pts, r - 1e-3).expect("ok");
        assert!(below.triangles.is_empty());

        // alpha just above r: both triangles kept, boundary = 4 square edges
        // (the shared diagonal is interior, incident to 2 triangles).
        let above = alpha_shape(&pts, r + 1e-3).expect("ok");
        assert_eq!(above.triangles.len(), 2);
        assert_eq!(
            above.boundary_edges.len(),
            4,
            "square alpha-complex boundary has the 4 outer edges, not the diagonal"
        );
        // Boundary covers all 4 vertices.
        let mut vs: Vec<usize> = above
            .boundary_edges
            .iter()
            .flat_map(|e| [e[0], e[1]])
            .collect();
        vs.sort_unstable();
        vs.dedup();
        assert_eq!(vs, vec![0, 1, 2, 3]);
    }

    // Oracle (e): kept-triangle area is monotone non-decreasing in alpha and
    // never exceeds the convex-hull area.
    #[test]
    fn area_monotone_in_alpha() {
        let mut rng = LcgRng::new(2024);
        let pts: Vec<Point> = (0..35)
            .map(|_| Point::new(rng.next_f64() * 8.0, rng.next_f64() * 8.0))
            .collect();
        let hull = andrew_monotone_chain(&pts).expect("hull");
        let hull_area = crate::primitives::polygon::Polygon::new(hull)
            .expect("poly")
            .area();

        let alphas = [0.3_f64, 0.6, 1.0, 1.5, 2.5, 5.0, 50.0];
        let mut prev = -1.0;
        for &a in &alphas {
            let shape = alpha_shape(&pts, a).expect("ok");
            let area = total_kept_area(&shape, &pts);
            assert!(area >= prev - 1e-9, "area not monotone at alpha {a}");
            assert!(area <= hull_area + 1e-7, "area exceeds hull at alpha {a}");
            prev = area;
        }
        // At very large alpha the area equals the hull area.
        let full = alpha_shape(&pts, 1e6).expect("ok");
        assert!((total_kept_area(&full, &pts) - hull_area).abs() < 1e-7);
    }

    // Oracle (f): an annulus-like sample yields a non-convex boundary at an
    // intermediate alpha (more boundary edges than the convex hull).
    #[test]
    fn annulus_nonconvex_boundary() {
        // Sample points on two concentric rings; at a moderate alpha the inner
        // hole is preserved, producing more boundary edges than the hull.
        let mut pts: Vec<Point> = Vec::new();
        let outer_r = 5.0;
        let inner_r = 2.0;
        let count = 48;
        for i in 0..count {
            let theta = std::f64::consts::TAU * (i as f64) / (count as f64);
            pts.push(Point::new(outer_r * theta.cos(), outer_r * theta.sin()));
            pts.push(Point::new(inner_r * theta.cos(), inner_r * theta.sin()));
        }
        // Band (inner<->outer) triangles have circumradius ~1.5; any triangle
        // spanning the empty inner disk needs circumradius >= inner_r = 2. Choose
        // alpha between the two so the band is filled but the inner hole survives.
        let alpha = 1.8;
        let shape = alpha_shape(&pts, alpha).expect("ok");
        let hull = andrew_monotone_chain(&pts).expect("hull");
        // Convex hull of this set is the outer ring polygon (<= count edges).
        assert!(
            shape.boundary_edges.len() > hull.len(),
            "annulus alpha-shape boundary ({}) should exceed hull edge count ({})",
            shape.boundary_edges.len(),
            hull.len()
        );
        // The inner hole must be preserved: not all triangles are kept (the
        // hole-spanning triangles are excluded), so the shape is strictly
        // non-convex (boundary forms an outer ring plus an inner ring).
        let full = alpha_shape(&pts, 1.0e6).expect("ok");
        assert!(
            shape.triangles.len() < full.triangles.len(),
            "intermediate alpha must keep a strict subset of the full triangulation"
        );
    }

    #[test]
    fn auto_alpha_covers_all_points() {
        let mut rng = LcgRng::new(555);
        let pts: Vec<Point> = (0..40)
            .map(|_| Point::new(rng.next_f64() * 6.0, rng.next_f64() * 6.0))
            .collect();
        let auto = alpha_shape_auto(&pts).expect("ok");
        // Every point participates in some retained triangle.
        let mut covered = vec![false; pts.len()];
        for t in &auto.shape.triangles {
            covered[t[0]] = true;
            covered[t[1]] = true;
            covered[t[2]] = true;
        }
        assert!(covered.iter().all(|&c| c));
        // The chosen alpha is from the spectrum and positive.
        assert!(auto.alpha > 0.0);
    }

    #[test]
    fn too_few_points_errors() {
        let pts = vec![Point::new(0.0, 0.0), Point::new(1.0, 0.0)];
        assert!(alpha_shape(&pts, 1.0).is_err());
    }

    #[test]
    fn negative_alpha_errors() {
        let pts = square_pts();
        assert!(alpha_shape(&pts, -1.0).is_err());
    }

    #[test]
    fn circumradius_of_right_isosceles() {
        let r = circumradius(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
        )
        .expect("non-degenerate");
        assert!((r - std::f64::consts::SQRT_2 / 2.0).abs() < 1e-12);
    }

    #[test]
    fn circumradius_degenerate_none() {
        assert!(
            circumradius(
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(2.0, 0.0),
            )
            .is_none()
        );
    }
}

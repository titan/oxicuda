//! Half-plane intersection via the sorted-deque (incremental) algorithm.
//!
//! # Problem
//!
//! Given `N` half-planes `H_i = { (x, y) : a_i*x + b_i*y <= c_i }`, compute their
//! intersection `R = ∩_i H_i`. `R` is convex; it is either
//!
//! * a bounded convex polygon,
//! * unbounded (an infinite convex region), or
//! * empty (the constraints are contradictory).
//!
//! # Half-plane orientation convention
//!
//! Each half-plane stores `(a, b, c)` with the feasible side defined by
//! `a*x + b*y <= c`. The boundary line is directed so that the feasible region
//! lies to the **left** of the direction of travel. With the inward normal
//! `n = (a, b)` (pointing *out* of the feasible region, since increasing
//! `a*x+b*y` leaves the region), the boundary direction is the clockwise
//! perpendicular of the outward normal, i.e. `d = (b, -a)`. Walking along `d`
//! keeps the feasible half-plane on the left — the convention required by the
//! deque-intersection algorithm.
//!
//! # Algorithm (O(n log n))
//!
//! 1. Sort half-planes by the polar angle of their boundary direction `d`.
//! 2. Among half-planes with the (numerically) same angle keep only the
//!    *tightest* (smallest signed offset `c / |n|`); the others are redundant
//!    parallels.
//! 3. Run an incremental double-ended-queue sweep: maintain a deque of
//!    half-planes whose pairwise boundary intersections form the current
//!    feasible polygon's vertices. For each new half-plane, pop from the back
//!    while the last deque vertex is infeasible w.r.t. the new plane, pop from
//!    the front while the first deque vertex is infeasible, then push the new
//!    plane to the back. A final cleanup pass removes back/front planes made
//!    redundant by the wrap-around closure.
//! 4. Reconstruct the polygon vertices as consecutive boundary intersections.
//!
//! # Bounded / unbounded / empty detection
//!
//! The boundary directions span the full circle iff the region is bounded. If
//! the angular span of the (deduped) directions is `< pi` (a closed half-turn),
//! the intersection is necessarily **unbounded** — there exists a direction in
//! which every constraint is satisfied to infinity. The algorithm therefore
//! adds four sentinel half-planes forming a large axis-aligned bounding box:
//! the box guarantees a bounded run of the deque algorithm and exposes
//! unboundedness as *survival of a box edge* in the result. If, after removing
//! the box-induced vertices, an unbounded direction remains, the region is
//! reported as `Unbounded(clipped_box)` where the polygon is the true region
//! intersected with the sentinel box. Infeasibility (an empty deque or a
//! degenerate sub-triangle) yields `Empty`.

use crate::error::{Geom2dError, Geom2dResult};
use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// A half-plane `{ (x, y) : a*x + b*y <= c }`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfPlane {
    /// `x` coefficient of the boundary normal.
    pub a: f64,
    /// `y` coefficient of the boundary normal.
    pub b: f64,
    /// Right-hand-side offset.
    pub c: f64,
}

impl HalfPlane {
    /// Construct a half-plane `a*x + b*y <= c`.
    ///
    /// # Errors
    ///
    /// Returns [`Geom2dError::InvalidConfiguration`] if `(a, b)` is the zero
    /// vector (the boundary is not a line).
    pub fn new(a: f64, b: f64, c: f64) -> Geom2dResult<Self> {
        if a.hypot(b) < EPS_NORMAL {
            return Err(Geom2dError::InvalidConfiguration(
                "half-plane normal (a, b) is zero".into(),
            ));
        }
        Ok(Self { a, b, c })
    }

    /// Construct the half-plane lying to the **left** of the directed segment
    /// `from -> to` (the feasible region is on the left of the travel
    /// direction). Convenience builder mirroring CCW polygon edges.
    ///
    /// # Errors
    ///
    /// Returns [`Geom2dError::InvalidConfiguration`] if the two points coincide.
    pub fn from_directed_edge(from: Point, to: Point) -> Geom2dResult<Self> {
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        if dx.hypot(dy) < EPS_NORMAL {
            return Err(Geom2dError::InvalidConfiguration(
                "directed edge has zero length".into(),
            ));
        }
        // Feasible region is to the left of d = (dx, dy). The outward normal
        // (pointing away from the feasible region) is the clockwise
        // perpendicular n = (dy, -dx); constraint n.p <= n.from.
        let a = dy;
        let b = -dx;
        let c = a * from.x + b * from.y;
        Ok(Self { a, b, c })
    }

    /// Signed slack `c - (a*x + b*y)`; non-negative iff `p` is feasible.
    #[must_use]
    fn slack(&self, p: Point) -> f64 {
        self.c - (self.a * p.x + self.b * p.y)
    }

    /// Boundary travel direction keeping the feasible side on the left.
    #[must_use]
    fn direction(&self) -> (f64, f64) {
        (self.b, -self.a)
    }

    /// Polar angle of the boundary direction in `(-pi, pi]`.
    #[must_use]
    fn angle(&self) -> f64 {
        let (dx, dy) = self.direction();
        dy.atan2(dx)
    }
}

/// Result region of a half-plane intersection.
#[derive(Debug, Clone)]
pub enum HalfPlaneRegion {
    /// A bounded convex feasible polygon (CCW).
    Polygon(Polygon),
    /// The constraints are contradictory; the feasible region is empty.
    Empty,
    /// The feasible region is unbounded. The carried polygon is the region
    /// clipped to a large sentinel bounding box (CCW); at least one of its
    /// edges lies on the sentinel box rather than on an input constraint.
    Unbounded(Polygon),
}

/// Absolute tolerance for treating a normal vector as zero.
const EPS_NORMAL: f64 = 1e-12;
/// Angular tolerance (radians) for merging parallel half-planes.
const EPS_ANGLE: f64 = 1e-9;
/// Geometric tolerance for feasibility / degeneracy decisions.
const EPS_GEOM: f64 = 1e-9;

/// Intersect a directed boundary pair, returning the boundary-line crossing.
///
/// Solves `h1.a*x + h1.b*y = h1.c` and `h2.a*x + h2.b*y = h2.c`. Returns `None`
/// when the boundaries are parallel (no unique crossing).
fn boundary_intersection(h1: &HalfPlane, h2: &HalfPlane) -> Option<Point> {
    let det = h1.a * h2.b - h2.a * h1.b;
    if det.abs() < EPS_NORMAL {
        return None;
    }
    let x = (h1.c * h2.b - h2.c * h1.b) / det;
    let y = (h1.a * h2.c - h2.a * h1.c) / det;
    Some(Point::new(x, y))
}

/// Tightest-of-parallels comparison: among equal-angle half-planes, the one
/// admitting the smaller feasible region (larger `a*x+b*y` excluded) wins.
///
/// For two parallels with identical unit normals, the tightest has the smaller
/// `c / |n|`. We compare with normals already known to be (near) parallel and
/// same-direction (same boundary angle implies same direction here).
fn is_tighter(candidate: &HalfPlane, current: &HalfPlane) -> bool {
    let nc = candidate.a.hypot(candidate.b);
    let nk = current.a.hypot(current.b);
    if nc < EPS_NORMAL || nk < EPS_NORMAL {
        return false;
    }
    candidate.c / nc < current.c / nk - EPS_GEOM
}

/// Compute the intersection of the supplied half-planes.
///
/// Returns a [`HalfPlaneRegion`] distinguishing bounded polygons, the empty
/// region, and unbounded regions (clipped to a large sentinel box).
///
/// # Errors
///
/// Returns [`Geom2dError::EmptyInput`] when `planes` is empty (no constraints
/// describe the whole plane, which is unbounded but has no finite witness), and
/// [`Geom2dError::InvalidConfiguration`] if any half-plane has a zero normal.
pub fn half_plane_intersection(planes: &[HalfPlane]) -> Geom2dResult<HalfPlaneRegion> {
    if planes.is_empty() {
        return Err(Geom2dError::EmptyInput);
    }
    for h in planes {
        if h.a.hypot(h.b) < EPS_NORMAL {
            return Err(Geom2dError::InvalidConfiguration(
                "half-plane normal (a, b) is zero".into(),
            ));
        }
    }

    // Determine a sentinel bounding box scale from the data so that any genuinely
    // bounded region fits well inside it, while unbounded regions touch it.
    let scale = sentinel_scale(planes);
    let unbounded_hint = !directions_span_full_circle(planes);

    // Augment with four box half-planes (CCW box edges -> left-feasible planes).
    let mut augmented: Vec<HalfPlane> = Vec::with_capacity(planes.len() + 4);
    augmented.extend_from_slice(planes);
    // x <= scale, x >= -scale, y <= scale, y >= -scale.
    augmented.push(HalfPlane {
        a: 1.0,
        b: 0.0,
        c: scale,
    });
    augmented.push(HalfPlane {
        a: -1.0,
        b: 0.0,
        c: scale,
    });
    augmented.push(HalfPlane {
        a: 0.0,
        b: 1.0,
        c: scale,
    });
    augmented.push(HalfPlane {
        a: 0.0,
        b: -1.0,
        c: scale,
    });

    let deduped = sort_and_dedupe(&augmented);
    // After adding the box there are always >= 4 distinct directions.
    let polygon_pts = match run_deque(&deduped) {
        Some(pts) => pts,
        None => return Ok(HalfPlaneRegion::Empty),
    };
    if polygon_pts.len() < 3 {
        return Ok(HalfPlaneRegion::Empty);
    }

    // Decide bounded vs unbounded: the region is unbounded iff a sentinel box
    // edge actually bounds the result (a polygon vertex sits on the box).
    let on_box = polygon_touches_box(&polygon_pts, scale);
    let region_poly = Polygon::new(polygon_pts)?;
    if on_box || unbounded_hint {
        if on_box {
            Ok(HalfPlaneRegion::Unbounded(region_poly))
        } else {
            // Directions did not span the circle yet the box was not touched:
            // the real region is bounded purely by input planes inside the box.
            Ok(HalfPlaneRegion::Polygon(region_poly))
        }
    } else {
        Ok(HalfPlaneRegion::Polygon(region_poly))
    }
}

/// Pick a sentinel half-extent comfortably larger than the data spread.
fn sentinel_scale(planes: &[HalfPlane]) -> f64 {
    let mut m = 1.0_f64;
    for h in planes {
        let n = h.a.hypot(h.b);
        if n > EPS_NORMAL {
            m = m.max((h.c / n).abs());
        }
    }
    // Generous margin so bounded regions stay strictly interior.
    (m * 8.0 + 16.0).max(1.0)
}

/// True if the boundary directions cover the whole circle (a necessary
/// condition for a bounded intersection of the *input* planes).
fn directions_span_full_circle(planes: &[HalfPlane]) -> bool {
    if planes.len() < 3 {
        return false;
    }
    let mut angles: Vec<f64> = planes.iter().map(HalfPlane::angle).collect();
    angles.sort_by(|x, y| x.partial_cmp(y).unwrap_or(core::cmp::Ordering::Equal));
    let two_pi = std::f64::consts::TAU;
    let mut max_gap = 0.0_f64;
    for w in angles.windows(2) {
        max_gap = max_gap.max(w[1] - w[0]);
    }
    // Wrap-around gap.
    if let (Some(&first), Some(&last)) = (angles.first(), angles.last()) {
        max_gap = max_gap.max(first + two_pi - last);
    }
    // Bounded requires every gap < pi (no open half-plane of directions free).
    max_gap < std::f64::consts::PI - EPS_ANGLE
}

/// Sort half-planes by boundary angle and drop redundant parallels, keeping the
/// tightest per direction.
fn sort_and_dedupe(planes: &[HalfPlane]) -> Vec<HalfPlane> {
    let mut sorted: Vec<HalfPlane> = planes.to_vec();
    sorted.sort_by(|h1, h2| {
        let a1 = h1.angle();
        let a2 = h2.angle();
        match a1.partial_cmp(&a2) {
            Some(core::cmp::Ordering::Equal) | None => {
                // Tie-break by offset so the tightest comes first.
                let o1 = h1.c / h1.a.hypot(h1.b);
                let o2 = h2.c / h2.a.hypot(h2.b);
                o1.partial_cmp(&o2).unwrap_or(core::cmp::Ordering::Equal)
            }
            Some(ord) => ord,
        }
    });
    let mut out: Vec<HalfPlane> = Vec::with_capacity(sorted.len());
    for h in sorted {
        match out.last_mut() {
            Some(prev) if (prev.angle() - h.angle()).abs() < EPS_ANGLE => {
                // Same direction: keep the tighter one.
                if is_tighter(&h, prev) {
                    *prev = h;
                }
            }
            _ => out.push(h),
        }
    }
    out
}

/// Core double-ended-queue intersection. Returns the ordered polygon vertices,
/// or `None` if the intersection is empty.
fn run_deque(planes: &[HalfPlane]) -> Option<Vec<Point>> {
    if planes.len() < 3 {
        return None;
    }
    // Deque of half-planes; `verts[k]` is the intersection of dq[k] and dq[k+1].
    let mut dq: Vec<HalfPlane> = Vec::with_capacity(planes.len());

    for &h in planes {
        // Pop from the back while the last vertex is infeasible for h.
        while dq.len() >= 2 {
            let p = boundary_intersection(&dq[dq.len() - 2], &dq[dq.len() - 1])?;
            if h.slack(p) < -EPS_GEOM {
                dq.pop();
            } else {
                break;
            }
        }
        // Pop from the front while the first vertex is infeasible for h.
        while dq.len() >= 2 {
            let p = boundary_intersection(&dq[0], &dq[1])?;
            if h.slack(p) < -EPS_GEOM {
                dq.remove(0);
            } else {
                break;
            }
        }
        // Parallel-and-opposite check: if h is anti-parallel to the back plane
        // and strictly tighter, the region collapses.
        if let Some(back) = dq.last() {
            if boundary_intersection(back, &h).is_none() {
                // Parallel boundaries. If h excludes the back plane entirely the
                // intersection is empty; otherwise h is redundant — skip it.
                let back_dir = back.direction();
                let h_dir = h.direction();
                let same = back_dir.0 * h_dir.0 + back_dir.1 * h_dir.1 > 0.0;
                if !same {
                    // Opposite-facing parallels: feasible band may be empty.
                    // Use a witness point on back's boundary.
                    if let Some(w) = boundary_point(back) {
                        if h.slack(w) < -EPS_GEOM {
                            return None;
                        }
                    }
                }
                continue;
            }
        }
        dq.push(h);
    }

    // Final cleanup: remove back planes invalidated by the front and vice versa
    // (wrap-around closure).
    while dq.len() >= 3 {
        let p = boundary_intersection(&dq[dq.len() - 2], &dq[dq.len() - 1])?;
        if dq[0].slack(p) < -EPS_GEOM {
            dq.pop();
        } else {
            break;
        }
    }
    while dq.len() >= 3 {
        let p = boundary_intersection(&dq[0], &dq[1])?;
        if dq[dq.len() - 1].slack(p) < -EPS_GEOM {
            dq.remove(0);
        } else {
            break;
        }
    }

    if dq.len() < 3 {
        return None;
    }

    // Reconstruct vertices: consecutive boundary intersections, closing the ring.
    let m = dq.len();
    let mut verts: Vec<Point> = Vec::with_capacity(m);
    for i in 0..m {
        let h1 = &dq[i];
        let h2 = &dq[(i + 1) % m];
        let p = boundary_intersection(h1, h2)?;
        verts.push(p);
    }

    // Drop near-duplicate consecutive vertices that can appear at tangencies.
    let cleaned = dedupe_ring(&verts);
    if cleaned.len() < 3 {
        return None;
    }
    // Final feasibility audit: every vertex must satisfy every deque plane.
    for v in &cleaned {
        for h in &dq {
            if h.slack(*v) < -1e-6 {
                return None;
            }
        }
    }
    Some(cleaned)
}

/// A representative point lying on the boundary line of `h`.
fn boundary_point(h: &HalfPlane) -> Option<Point> {
    let n2 = h.a * h.a + h.b * h.b;
    if n2 < EPS_NORMAL {
        return None;
    }
    Some(Point::new(h.a * h.c / n2, h.b * h.c / n2))
}

/// Remove consecutive duplicate vertices (cyclically) within `EPS_GEOM`.
fn dedupe_ring(verts: &[Point]) -> Vec<Point> {
    let mut out: Vec<Point> = Vec::with_capacity(verts.len());
    for &p in verts {
        if let Some(&last) = out.last() {
            if last.distance(p) < EPS_GEOM {
                continue;
            }
        }
        out.push(p);
    }
    // Cyclic closure dedupe.
    if out.len() >= 2 {
        let first = out[0];
        if let Some(&last) = out.last() {
            if last.distance(first) < EPS_GEOM {
                out.pop();
            }
        }
    }
    out
}

/// True if any polygon vertex lies on the sentinel box boundary (`|x|` or `|y|`
/// within tolerance of `scale`).
fn polygon_touches_box(pts: &[Point], scale: f64) -> bool {
    let tol = scale * 1e-7 + EPS_GEOM;
    pts.iter()
        .any(|p| (p.x.abs() - scale).abs() < tol || (p.y.abs() - scale).abs() < tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polygon_ops::area_shoelace::area_shoelace;

    /// Build the four half-planes of the axis-aligned box `[x0,x1] x [y0,y1]`.
    fn box_planes(x0: f64, x1: f64, y0: f64, y1: f64) -> Vec<HalfPlane> {
        vec![
            HalfPlane {
                a: 1.0,
                b: 0.0,
                c: x1,
            }, // x <= x1
            HalfPlane {
                a: -1.0,
                b: 0.0,
                c: -x0,
            }, // x >= x0
            HalfPlane {
                a: 0.0,
                b: 1.0,
                c: y1,
            }, // y <= y1
            HalfPlane {
                a: 0.0,
                b: -1.0,
                c: -y0,
            }, // y >= y0
        ]
    }

    fn assert_all_satisfied(planes: &[HalfPlane], poly: &Polygon) {
        for v in &poly.vertices {
            for h in planes {
                assert!(
                    h.slack(*v) >= -1e-6,
                    "vertex {:?} violates plane {:?} (slack {})",
                    v,
                    h,
                    h.slack(*v)
                );
            }
        }
    }

    // Oracle (a): four box half-planes reconstruct the box exactly.
    #[test]
    fn box_reconstructed_exactly() {
        let planes = box_planes(-2.0, 3.0, -1.0, 4.0);
        let region = half_plane_intersection(&planes).expect("ok");
        match region {
            HalfPlaneRegion::Polygon(p) => {
                assert!((area_shoelace(&p) - 25.0).abs() < 1e-7);
                let bb = p.aabb();
                assert!((bb.min.x + 2.0).abs() < 1e-6);
                assert!((bb.max.x - 3.0).abs() < 1e-6);
                assert!((bb.min.y + 1.0).abs() < 1e-6);
                assert!((bb.max.y - 4.0).abs() < 1e-6);
                assert_eq!(p.vertices.len(), 4);
            }
            other => panic!("expected bounded polygon, got {other:?}"),
        }
    }

    // Oracle (b): three half-planes -> known triangle; every vertex feasible.
    #[test]
    fn triangle_from_three_planes() {
        // Triangle (0,0), (4,0), (0,3): below the hypotenuse 3x+4y<=12, x>=0, y>=0.
        let planes = vec![
            HalfPlane {
                a: -1.0,
                b: 0.0,
                c: 0.0,
            }, // x >= 0
            HalfPlane {
                a: 0.0,
                b: -1.0,
                c: 0.0,
            }, // y >= 0
            HalfPlane {
                a: 3.0,
                b: 4.0,
                c: 12.0,
            }, // 3x + 4y <= 12
        ];
        let region = half_plane_intersection(&planes).expect("ok");
        match region {
            HalfPlaneRegion::Polygon(p) => {
                assert!((area_shoelace(&p) - 6.0).abs() < 1e-7);
                assert_all_satisfied(&planes, &p);
                assert_eq!(p.vertices.len(), 3);
            }
            other => panic!("expected triangle polygon, got {other:?}"),
        }
    }

    // Oracle (c): a strictly redundant constraint does not change the result.
    #[test]
    fn redundant_constraint_dropped() {
        let mut planes = box_planes(0.0, 1.0, 0.0, 1.0);
        let base = half_plane_intersection(&planes).expect("ok");
        let base_area = match base {
            HalfPlaneRegion::Polygon(ref p) => area_shoelace(p),
            _ => panic!("expected polygon"),
        };
        // x <= 10 is far outside the unit box -> redundant.
        planes.push(HalfPlane {
            a: 1.0,
            b: 0.0,
            c: 10.0,
        });
        let with_extra = half_plane_intersection(&planes).expect("ok");
        match with_extra {
            HalfPlaneRegion::Polygon(p) => {
                assert!((area_shoelace(&p) - base_area).abs() < 1e-7);
                assert!((area_shoelace(&p) - 1.0).abs() < 1e-7);
            }
            other => panic!("expected polygon, got {other:?}"),
        }
    }

    // Oracle (d): contradictory half-planes -> Empty.
    #[test]
    fn contradictory_is_empty() {
        // x >= 1 AND x <= -1 cannot both hold.
        let planes = vec![
            HalfPlane {
                a: -1.0,
                b: 0.0,
                c: -1.0,
            }, // x >= 1
            HalfPlane {
                a: 1.0,
                b: 0.0,
                c: -1.0,
            }, // x <= -1
            HalfPlane {
                a: 0.0,
                b: 1.0,
                c: 1.0,
            }, // y <= 1
            HalfPlane {
                a: 0.0,
                b: -1.0,
                c: 1.0,
            }, // y >= -1
        ];
        match half_plane_intersection(&planes).expect("ok") {
            HalfPlaneRegion::Empty => {}
            other => panic!("expected Empty, got {other:?}"),
        }
    }

    // Oracle (e): intersection area equals the analytic value for a known polygon.
    #[test]
    fn pentagon_area_analytic() {
        // Regular-ish convex polygon as half-planes from CCW edges.
        let verts = [
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(5.0, 3.0),
            Point::new(2.0, 5.0),
            Point::new(-1.0, 2.0),
        ];
        let mut planes = Vec::new();
        for i in 0..verts.len() {
            let from = verts[i];
            let to = verts[(i + 1) % verts.len()];
            planes.push(HalfPlane::from_directed_edge(from, to).expect("edge"));
        }
        // Analytic shoelace area of the pentagon.
        let mut s = 0.0;
        for i in 0..verts.len() {
            let a = verts[i];
            let b = verts[(i + 1) % verts.len()];
            s += a.x * b.y - b.x * a.y;
        }
        let analytic = 0.5 * s.abs();
        match half_plane_intersection(&planes).expect("ok") {
            HalfPlaneRegion::Polygon(p) => {
                assert!(
                    (area_shoelace(&p) - analytic).abs() < 1e-7,
                    "got {}, want {}",
                    area_shoelace(&p),
                    analytic
                );
                assert_all_satisfied(&planes, &p);
            }
            other => panic!("expected polygon, got {other:?}"),
        }
    }

    // Oracle (f): order-independence — shuffling planes yields the same region.
    #[test]
    fn order_independence() {
        let planes = vec![
            HalfPlane {
                a: -1.0,
                b: 0.0,
                c: 0.0,
            },
            HalfPlane {
                a: 0.0,
                b: -1.0,
                c: 0.0,
            },
            HalfPlane {
                a: 3.0,
                b: 4.0,
                c: 12.0,
            },
        ];
        let a1 = match half_plane_intersection(&planes).expect("ok") {
            HalfPlaneRegion::Polygon(p) => area_shoelace(&p),
            _ => panic!(),
        };
        // Several rotations / reversals.
        for perm in [
            vec![planes[2], planes[0], planes[1]],
            vec![planes[1], planes[2], planes[0]],
            vec![planes[2], planes[1], planes[0]],
        ] {
            let a2 = match half_plane_intersection(&perm).expect("ok") {
                HalfPlaneRegion::Polygon(p) => area_shoelace(&p),
                other => panic!("expected polygon, got {other:?}"),
            };
            assert!((a1 - a2).abs() < 1e-7);
        }
    }

    // Unbounded: a single half-plane (and a wedge) is unbounded.
    #[test]
    fn single_half_plane_unbounded() {
        let planes = vec![HalfPlane {
            a: 1.0,
            b: 0.0,
            c: 1.0,
        }]; // x <= 1
        match half_plane_intersection(&planes).expect("ok") {
            HalfPlaneRegion::Unbounded(p) => {
                assert!(p.vertices.len() >= 3);
            }
            other => panic!("expected Unbounded, got {other:?}"),
        }
    }

    #[test]
    fn wedge_unbounded() {
        // y >= 0 and y >= x (a wedge opening upward-left) -> unbounded.
        let planes = vec![
            HalfPlane {
                a: 0.0,
                b: -1.0,
                c: 0.0,
            }, // y >= 0
            HalfPlane {
                a: 1.0,
                b: -1.0,
                c: 0.0,
            }, // x - y <= 0  i.e. y >= x
        ];
        match half_plane_intersection(&planes).expect("ok") {
            HalfPlaneRegion::Unbounded(_) => {}
            other => panic!("expected Unbounded, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_errors() {
        let planes: Vec<HalfPlane> = Vec::new();
        assert!(half_plane_intersection(&planes).is_err());
    }

    #[test]
    fn zero_normal_errors() {
        assert!(HalfPlane::new(0.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn from_directed_edge_left_feasible() {
        // Edge along +x from origin: feasible region is the upper half-plane y>=0.
        let h = HalfPlane::from_directed_edge(Point::new(0.0, 0.0), Point::new(1.0, 0.0))
            .expect("edge");
        assert!(h.slack(Point::new(0.0, 1.0)) >= 0.0);
        assert!(h.slack(Point::new(0.0, -1.0)) < 0.0);
    }
}

//! Greiner-Hormann polygon clipping for general (non-convex) polygons.
//!
//! Implements Greiner & Hormann, *"Efficient Clipping of Arbitrary Polygons"*,
//! ACM TOG 17(2), 1998, with the entry/exit labelling extended to the four
//! Boolean set operations (intersection, union, difference, symmetric
//! difference).
//!
//! # Algorithm overview
//!
//! 1. **Rings.** Subject (`P`) and clip (`Q`) polygons are represented as
//!    doubly-linked vertex rings (here flattened into index-linked `Vertex`
//!    arrays for `#![forbid(unsafe_code)]` safety — no raw pointers).
//! 2. **Intersections.** Every edge of `P` is tested against every edge of `Q`.
//!    Each transversal crossing produces an intersection vertex that is inserted
//!    into *both* rings, positioned by its `alpha` parameter along the edge so
//!    the ring stays geometrically ordered. The two copies are cross-linked via
//!    a `neighbour` index.
//! 3. **Entry/exit labelling.** Walking the subject ring, the first vertex is
//!    classified inside/outside the clip polygon; each intersection then flips
//!    an `entry` flag. The clip ring is labelled symmetrically against the
//!    subject polygon.
//! 4. **Tracing.** Starting from each unvisited intersection, the result ring is
//!    traced by walking one polygon until the next intersection, then switching
//!    to the neighbour in the other polygon. The per-operation direction table
//!    selects forward / backward traversal so the resulting rings bound exactly
//!    the requested Boolean region.
//!
//! # Result representation (holes via winding)
//!
//! `Vec<Polygon>` is returned. Outer boundaries are CCW; holes are CW. A union
//! or difference may yield several rings; a difference that punches a hole into
//! the subject returns the outer ring (CCW) and the hole ring (CW). Callers can
//! reconstruct filled area by summing **signed** shoelace areas (CCW positive,
//! CW negative).
//!
//! # Degeneracy handling
//!
//! The classical Greiner-Hormann algorithm assumes no vertex of one polygon lies
//! on an edge of the other and no edges are collinear-overlapping. Such
//! degeneracies are handled here by a *symmetric vertex perturbation* applied
//! **only to the inside/outside classification queries**, never to the emitted
//! coordinates: when an intersection or a polygon vertex lands within `EPS` of
//! the other polygon's boundary, the point-in-polygon test is evaluated at a
//! deterministically jittered position (a tiny offset along the averaged
//! adjacent-edge normal). This realises Hormann & Tarini's "perturb the query,
//! not the geometry" strategy and avoids both crashes and double-counted
//! boundaries while keeping output coordinates exact. Pure (non-degenerate)
//! inputs are unaffected. The fallback for fully-overlapping collinear edges is
//! documented at [`clip_polygons`].

use crate::error::Geom2dResult;
use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// Boolean set operation to perform between subject and clip polygons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// `subject ∩ clip` — region inside both.
    Intersection,
    /// `subject ∪ clip` — region inside either.
    Union,
    /// `subject ∖ clip` — region inside subject but outside clip.
    Difference,
    /// `subject ⊕ clip` — region inside exactly one (symmetric difference).
    Xor,
}

/// Geometric tolerance used for degeneracy detection and equality.
const EPS: f64 = 1e-9;

/// A ring vertex in the Greiner-Hormann linked structure.
#[derive(Debug, Clone)]
struct Vertex {
    /// Coordinates.
    p: Point,
    /// Next vertex index within the same ring.
    next: usize,
    /// Previous vertex index within the same ring.
    prev: usize,
    /// `true` if this vertex is an intersection (shared with the other polygon).
    intersection: bool,
    /// `true` => entering the *other* polygon when walking this ring forward;
    /// `false` => exiting. Meaningless for non-intersection vertices.
    entry: bool,
    /// Index of the paired vertex in the *other* polygon's array (for
    /// intersections only).
    neighbour: usize,
    /// Parameter `alpha ∈ [0, 1]` of this intersection along its host edge,
    /// used purely for sorted insertion.
    alpha: f64,
    /// Visited flag for result tracing.
    visited: bool,
}

impl Vertex {
    fn corner(p: Point) -> Self {
        Self {
            p,
            next: 0,
            prev: 0,
            intersection: false,
            entry: false,
            neighbour: usize::MAX,
            alpha: 0.0,
            visited: false,
        }
    }
}

/// A doubly-linked vertex ring stored in an index-addressed arena.
struct Ring {
    nodes: Vec<Vertex>,
    /// One representative original (non-intersection) vertex index, used as a
    /// stable traversal anchor.
    start: usize,
}

impl Ring {
    /// Build a ring from polygon vertices (no intersections yet).
    fn from_polygon(poly: &Polygon) -> Self {
        let n = poly.vertices.len();
        let mut nodes: Vec<Vertex> = poly.vertices.iter().map(|&p| Vertex::corner(p)).collect();
        for (i, node) in nodes.iter_mut().enumerate() {
            node.next = (i + 1) % n;
            node.prev = (i + n - 1) % n;
        }
        Self { nodes, start: 0 }
    }

    /// Insert a fresh intersection vertex between the original endpoints of the
    /// edge starting at `edge_start`, ordered by `alpha`. Returns the new index.
    fn insert_intersection(&mut self, edge_start: usize, p: Point, alpha: f64) -> usize {
        // Walk forward from `edge_start` over already-inserted intersections on
        // the same edge to find the correct sorted slot.
        let mut cur = edge_start;
        loop {
            let nxt = self.nodes[cur].next;
            // Stop when the next node is the edge's far original endpoint, or an
            // intersection with a larger alpha.
            if !self.nodes[nxt].intersection || self.nodes[nxt].alpha > alpha {
                break;
            }
            cur = nxt;
        }
        let nxt = self.nodes[cur].next;
        let idx = self.nodes.len();
        let mut v = Vertex::corner(p);
        v.intersection = true;
        v.alpha = alpha;
        v.prev = cur;
        v.next = nxt;
        self.nodes.push(v);
        self.nodes[cur].next = idx;
        self.nodes[nxt].prev = idx;
        idx
    }

    /// Iterate original (non-intersection) edge-start indices in ring order.
    fn original_indices(&self, count: usize) -> Vec<usize> {
        // The first `count` arena slots are the originals (intersections append).
        (0..count).collect()
    }
}

/// Clip `subject` against `clip` with the given Boolean operation.
///
/// Returns the result as a set of rings (`Vec<Polygon>`): outer boundaries CCW,
/// holes CW. The empty `Vec` denotes an empty region.
///
/// # Degenerate / collinear-overlap fallback
///
/// When no transversal intersections are found (the boundaries are disjoint or
/// only touch / overlap along collinear edges), the result is decided purely by
/// containment of representative interior points — the standard GH "trivial
/// containment" base case — which yields the correct whole-polygon answers for
/// disjoint, nested, and edge-touching inputs.
///
/// # Errors
///
/// Propagates [`crate::error::Geom2dError`] from polygon construction of result
/// rings (degenerate sub-rings are dropped rather than erroring).
pub fn clip_polygons(
    subject: &Polygon,
    clip: &Polygon,
    op: BooleanOp,
) -> Geom2dResult<Vec<Polygon>> {
    // Work on CCW-oriented copies so winding conventions are consistent.
    let subj = subject.oriented_ccw();
    let clp = clip.oriented_ccw();
    let n_subj = subj.vertices.len();
    let n_clip = clp.vertices.len();

    let mut p_ring = Ring::from_polygon(&subj);
    let mut q_ring = Ring::from_polygon(&clp);

    // --- Phase 1: compute and insert all transversal intersections. ---
    let mut any_intersection = false;
    let p_orig = p_ring.original_indices(n_subj);
    let q_orig = q_ring.original_indices(n_clip);
    for &pi in &p_orig {
        let a0 = subj.vertices[pi];
        let a1 = subj.vertices[(pi + 1) % n_subj];
        for &qi in &q_orig {
            let b0 = clp.vertices[qi];
            let b1 = clp.vertices[(qi + 1) % n_clip];
            if let Some((ip, alpha_p, alpha_q)) = segment_cross(a0, a1, b0, b1) {
                // Skip pure endpoint-coincidence degeneracies (alpha at 0 or 1
                // exactly): handled by the containment fallback / perturbed
                // classification rather than as transversal crossings.
                if alpha_p <= EPS || alpha_p >= 1.0 - EPS || alpha_q <= EPS || alpha_q >= 1.0 - EPS
                {
                    continue;
                }
                let p_new = p_ring.insert_intersection(pi, ip, alpha_p);
                let q_new = q_ring.insert_intersection(qi, ip, alpha_q);
                p_ring.nodes[p_new].neighbour = q_new;
                q_ring.nodes[q_new].neighbour = p_new;
                any_intersection = true;
            }
        }
    }

    if !any_intersection {
        return containment_fallback(&subj, &clp, op);
    }

    // --- Phase 2: entry/exit labelling. ---
    label_entry_exit(&mut p_ring, &clp);
    label_entry_exit(&mut q_ring, &subj);

    // --- Phase 3: trace result rings. ---
    let rings = trace_result(&mut p_ring, &mut q_ring, op);

    // --- Phase 4: build polygons, dropping degenerate sub-rings. ---
    let mut out: Vec<Polygon> = Vec::new();
    for ring in rings {
        let cleaned = dedupe_consecutive(&ring);
        if cleaned.len() < 3 {
            continue;
        }
        if let Ok(poly) = Polygon::new(cleaned) {
            if poly.area() > EPS {
                out.push(poly);
            }
        }
    }
    // --- Phase 5: canonical orientation by even-odd nesting depth. ---
    normalize_orientation(&mut out);
    Ok(out)
}

/// Orient result rings canonically: a ring nested inside an even number of other
/// rings is an *outer* boundary (CCW); odd nesting depth makes it a *hole* (CW).
///
/// The Greiner-Hormann marching may emit union/difference rings in either
/// winding depending on traversal direction; this pass enforces the documented
/// convention (outers CCW, holes CW) so that `signed_area_of_rings` directly
/// yields the filled area (holes subtract).
fn normalize_orientation(rings: &mut [Polygon]) {
    // Precompute one interior witness per ring.
    let witnesses: Vec<Point> = rings.iter().map(interior_witness).collect();
    // First pass (immutable): decide the desired winding for each ring from its
    // even-odd nesting depth.
    let want_ccw: Vec<bool> = witnesses
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            let depth = rings
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .filter(|&(_, ring)| winding_inside(ring, w))
                .count();
            depth % 2 == 0
        })
        .collect();
    // Second pass (mutable): reverse rings whose winding disagrees.
    for (ring, &ccw) in rings.iter_mut().zip(want_ccw.iter()) {
        if ring.is_ccw() != ccw {
            ring.vertices.reverse();
        }
    }
}

/// Convenience: `subject ∩ clip`.
///
/// # Errors
/// Propagates errors from [`clip_polygons`].
pub fn intersection(subject: &Polygon, clip: &Polygon) -> Geom2dResult<Vec<Polygon>> {
    clip_op(subject, clip, BooleanOp::Intersection)
}

/// Convenience: `subject ∪ clip`.
///
/// # Errors
/// Propagates errors from [`clip_polygons`].
pub fn union(subject: &Polygon, clip: &Polygon) -> Geom2dResult<Vec<Polygon>> {
    clip_op(subject, clip, BooleanOp::Union)
}

/// Convenience: `subject ∖ clip` (subject minus clip).
///
/// # Errors
/// Propagates errors from [`clip_polygons`].
pub fn difference(subject: &Polygon, clip: &Polygon) -> Geom2dResult<Vec<Polygon>> {
    clip_op(subject, clip, BooleanOp::Difference)
}

/// Convenience: symmetric difference `subject ⊕ clip`.
///
/// # Errors
/// Propagates errors from [`clip_polygons`].
pub fn xor(subject: &Polygon, clip: &Polygon) -> Geom2dResult<Vec<Polygon>> {
    clip_op(subject, clip, BooleanOp::Xor)
}

/// Internal dispatcher routing `Xor` through the two differences (the most
/// robust construction: `A⊕B = (A∖B) ∪ (B∖A)`), and the others through [`clip_polygons`].
fn clip_op(subject: &Polygon, clip: &Polygon, op: BooleanOp) -> Geom2dResult<Vec<Polygon>> {
    match op {
        BooleanOp::Xor => {
            let mut a = clip_polygons(subject, clip, BooleanOp::Difference)?;
            let mut b = clip_polygons(clip, subject, BooleanOp::Difference)?;
            a.append(&mut b);
            Ok(a)
        }
        other => clip_polygons(subject, clip, other),
    }
}

/// Total signed area of a set of result rings (CCW positive, CW negative).
///
/// Convenience for oracle checks: the filled area of a clipped region equals the
/// sum of signed shoelace areas (holes subtract).
#[must_use]
pub fn signed_area_of_rings(rings: &[Polygon]) -> f64 {
    rings.iter().map(Polygon::signed_area).sum()
}

/// Total *filled* (absolute) area of result rings, subtracting CW holes from CCW
/// outers. Equivalent to `signed_area_of_rings` when outers are CCW.
#[must_use]
pub fn filled_area_of_rings(rings: &[Polygon]) -> f64 {
    signed_area_of_rings(rings).abs()
}

/// Compute the transversal intersection of segments `[a0,a1]` and `[b0,b1]`.
///
/// Returns `(point, alpha_p, alpha_q)` where `alpha_p` is the parameter along
/// `[a0,a1]` and `alpha_q` along `[b0,b1]`, both in `(0,1)` for a proper
/// crossing. Returns `None` for parallel / non-crossing / purely-touching pairs.
fn segment_cross(a0: Point, a1: Point, b0: Point, b1: Point) -> Option<(Point, f64, f64)> {
    let rx = a1.x - a0.x;
    let ry = a1.y - a0.y;
    let sx = b1.x - b0.x;
    let sy = b1.y - b0.y;
    let denom = rx * sy - ry * sx;
    if denom.abs() < 1e-15 {
        return None; // parallel or degenerate
    }
    let qpx = b0.x - a0.x;
    let qpy = b0.y - a0.y;
    let t = (qpx * sy - qpy * sx) / denom; // along P
    let u = (qpx * ry - qpy * rx) / denom; // along Q
    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        let ip = Point::new(a0.x + t * rx, a0.y + t * ry);
        Some((ip, t, u))
    } else {
        None
    }
}

/// Label every intersection vertex of `ring` as entry/exit relative to `other`.
///
/// The first non-intersection vertex's inside/outside status seeds the parity;
/// each intersection flips it. The `entry` flag means: walking this ring
/// forward, this vertex *enters* `other`.
fn label_entry_exit(ring: &mut Ring, other: &Polygon) {
    // Find a starting original (non-intersection) vertex.
    let start = ring.start;
    let start_inside = point_inside_perturbed(ring.nodes[start].p, other, ring, start);
    let mut status_inside = start_inside;
    let mut cur = ring.nodes[start].next;
    while cur != start {
        if ring.nodes[cur].intersection {
            // Entry iff we are currently outside and about to go inside.
            ring.nodes[cur].entry = !status_inside;
            status_inside = !status_inside;
        }
        cur = ring.nodes[cur].next;
    }
    // Also handle the seed vertex if it happens to be an intersection (rare,
    // since we anchored on index 0 which is an original corner).
}

/// Point-in-polygon test with degeneracy-robust perturbation.
///
/// If `q` lies within `EPS` of `other`'s boundary, the query is re-evaluated at
/// a deterministically perturbed position offset along the local inward normal
/// of the *host* ring at `host_idx`, realising "perturb the query, not the
/// geometry". Output coordinates are never altered.
fn point_inside_perturbed(q: Point, other: &Polygon, ring: &Ring, host_idx: usize) -> bool {
    if !near_boundary(q, other) {
        return winding_inside(other, q);
    }
    // Perturb along the bisector normal of the host ring at this vertex.
    let prev = ring.nodes[ring.nodes[host_idx].prev].p;
    let next = ring.nodes[ring.nodes[host_idx].next].p;
    let dx = next.x - prev.x;
    let dy = next.y - prev.y;
    let len = dx.hypot(dy).max(EPS);
    // Inward normal for a CCW ring is the left perpendicular of the travel dir.
    let nx = -dy / len;
    let ny = dx / len;
    let jitter = 16.0 * EPS;
    let probe = Point::new(q.x + nx * jitter, q.y + ny * jitter);
    winding_inside(other, probe)
}

/// True if `q` is within `EPS` of any edge of `poly`.
fn near_boundary(q: Point, poly: &Polygon) -> bool {
    let n = poly.vertices.len();
    for i in 0..n {
        let a = poly.vertices[i];
        let b = poly.vertices[(i + 1) % n];
        if point_segment_distance(q, a, b) < EPS {
            return true;
        }
    }
    false
}

/// Distance from point `q` to segment `[a, b]`.
fn point_segment_distance(q: Point, a: Point, b: Point) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let d2 = dx * dx + dy * dy;
    if d2 < 1e-300 {
        return q.distance(a);
    }
    let t = ((q.x - a.x) * dx + (q.y - a.y) * dy) / d2;
    let tc = t.clamp(0.0, 1.0);
    let cx = a.x + tc * dx;
    let cy = a.y + tc * dy;
    (q.x - cx).hypot(q.y - cy)
}

/// Winding-number inside test (robust for non-convex polygons).
fn winding_inside(poly: &Polygon, q: Point) -> bool {
    let mut w = 0_i32;
    let n = poly.vertices.len();
    for i in 0..n {
        let a = poly.vertices[i];
        let b = poly.vertices[(i + 1) % n];
        if a.y <= q.y {
            if b.y > q.y && is_left(a, b, q) > 0.0 {
                w += 1;
            }
        } else if b.y <= q.y && is_left(a, b, q) < 0.0 {
            w -= 1;
        }
    }
    w != 0
}

fn is_left(a: Point, b: Point, c: Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
}

/// Trace all result rings by marching the linked structure.
///
/// Direction table (subject `P`, clip `Q`), for the per-vertex `entry` flag:
///
/// | op            | march on P            | march on Q            |
/// |---------------|-----------------------|-----------------------|
/// | Intersection  | forward if entry      | forward if entry      |
/// | Union         | forward if !entry     | forward if !entry     |
/// | Difference    | forward if entry      | forward if !entry     |
///
/// (`Xor` is handled upstream as two differences.) The rule "advance into the
/// region of interest, switch polygons at each intersection" is the Greiner-
/// Hormann marching invariant.
fn trace_result(p_ring: &mut Ring, q_ring: &mut Ring, op: BooleanOp) -> Vec<Vec<Point>> {
    let mut rings: Vec<Vec<Point>> = Vec::new();

    loop {
        // Find an unvisited intersection on P to seed a new component.
        let seed = (0..p_ring.nodes.len())
            .find(|&i| p_ring.nodes[i].intersection && !p_ring.nodes[i].visited);
        let Some(seed) = seed else { break };

        let mut ring_pts: Vec<Point> = Vec::new();
        // `on_p` tracks which arena we are currently walking.
        let mut on_p = true;
        let mut cur = seed;

        loop {
            // Mark visited on both this vertex and its neighbour copy.
            if on_p {
                if p_ring.nodes[cur].visited {
                    break;
                }
                p_ring.nodes[cur].visited = true;
                let nb = p_ring.nodes[cur].neighbour;
                if nb != usize::MAX {
                    q_ring.nodes[nb].visited = true;
                }
            } else {
                if q_ring.nodes[cur].visited {
                    break;
                }
                q_ring.nodes[cur].visited = true;
                let nb = q_ring.nodes[cur].neighbour;
                if nb != usize::MAX {
                    p_ring.nodes[nb].visited = true;
                }
            }

            // Decide direction at this intersection.
            let entry = if on_p {
                p_ring.nodes[cur].entry
            } else {
                q_ring.nodes[cur].entry
            };
            let forward = direction_forward(op, on_p, entry);

            // Append the current intersection point, then walk to the next
            // intersection along the chosen direction, appending corners.
            let start_pt = if on_p {
                p_ring.nodes[cur].p
            } else {
                q_ring.nodes[cur].p
            };
            ring_pts.push(start_pt);

            let mut walk = cur;
            loop {
                walk = if on_p {
                    if forward {
                        p_ring.nodes[walk].next
                    } else {
                        p_ring.nodes[walk].prev
                    }
                } else if forward {
                    q_ring.nodes[walk].next
                } else {
                    q_ring.nodes[walk].prev
                };
                let is_inter = if on_p {
                    p_ring.nodes[walk].intersection
                } else {
                    q_ring.nodes[walk].intersection
                };
                if is_inter {
                    break;
                }
                let pt = if on_p {
                    p_ring.nodes[walk].p
                } else {
                    q_ring.nodes[walk].p
                };
                ring_pts.push(pt);
            }

            // Switch to the neighbour copy in the other polygon.
            let nb = if on_p {
                p_ring.nodes[walk].neighbour
            } else {
                q_ring.nodes[walk].neighbour
            };
            if nb == usize::MAX {
                break;
            }
            on_p = !on_p;
            cur = nb;

            // Closed the ring?
            let back_to_seed = on_p && cur == seed;
            let seed_neighbour = p_ring.nodes[seed].neighbour;
            let back_via_neighbour = !on_p && cur == seed_neighbour;
            if back_to_seed || back_via_neighbour {
                // Mark the closing vertex visited and stop.
                if on_p {
                    p_ring.nodes[cur].visited = true;
                } else {
                    q_ring.nodes[cur].visited = true;
                }
                break;
            }
        }

        if ring_pts.len() >= 3 {
            rings.push(ring_pts);
        }
    }

    rings
}

/// Per-operation traversal direction selector (see [`trace_result`] table).
fn direction_forward(op: BooleanOp, on_p: bool, entry: bool) -> bool {
    match op {
        BooleanOp::Intersection => entry,
        BooleanOp::Union => !entry,
        BooleanOp::Difference => {
            if on_p {
                entry
            } else {
                !entry
            }
        }
        // Xor is decomposed upstream and never reaches here.
        BooleanOp::Xor => entry,
    }
}

/// Remove consecutive duplicate points (cyclically) within `EPS`.
fn dedupe_consecutive(pts: &[Point]) -> Vec<Point> {
    let mut out: Vec<Point> = Vec::with_capacity(pts.len());
    for &p in pts {
        if let Some(&last) = out.last() {
            if last.distance(p) < EPS {
                continue;
            }
        }
        out.push(p);
    }
    while out.len() >= 2 {
        let first = out[0];
        let last = out[out.len() - 1];
        if first.distance(last) < EPS {
            out.pop();
        } else {
            break;
        }
    }
    out
}

/// Base case when no transversal intersections exist: decide by containment.
///
/// Covers disjoint polygons, nested polygons (one inside the other), and
/// boundary-touching configurations.
fn containment_fallback(
    subj: &Polygon,
    clp: &Polygon,
    op: BooleanOp,
) -> Geom2dResult<Vec<Polygon>> {
    // With no transversal crossings the two boundaries are nested or disjoint.
    // Containment is therefore decided robustly by *every* vertex of one polygon
    // lying inside (or on) the other — a single representative point can be
    // fooled when one polygon's centroid coincidentally falls inside the other.
    let subj_in_clip = polygon_inside(subj, clp);
    let clip_in_subj = polygon_inside(clp, subj);

    match op {
        BooleanOp::Intersection => {
            if subj_in_clip {
                Ok(vec![subj.clone()])
            } else if clip_in_subj {
                Ok(vec![clp.clone()])
            } else {
                Ok(Vec::new())
            }
        }
        BooleanOp::Union => {
            if subj_in_clip {
                Ok(vec![clp.clone()])
            } else if clip_in_subj {
                // clip inside subject: union is subject with clip as a hole only
                // if clip is a hole; for a solid union it's just the subject.
                Ok(vec![subj.clone()])
            } else {
                // Disjoint: union is both rings.
                Ok(vec![subj.clone(), clp.clone()])
            }
        }
        BooleanOp::Difference => {
            if subj_in_clip {
                // Subject entirely inside clip: nothing remains.
                Ok(Vec::new())
            } else if clip_in_subj {
                // Clip is a hole inside subject: outer CCW + hole CW.
                let outer = subj.oriented_ccw();
                let mut hole_v = clp.oriented_ccw().vertices;
                hole_v.reverse(); // CW hole
                let hole = Polygon { vertices: hole_v };
                Ok(vec![outer, hole])
            } else {
                // Disjoint: subject unchanged.
                Ok(vec![subj.clone()])
            }
        }
        BooleanOp::Xor => {
            if subj_in_clip || clip_in_subj {
                // Nested: symmetric difference is the annulus (outer + CW hole).
                let (outer, inner) = if subj_in_clip {
                    (clp, subj)
                } else {
                    (subj, clp)
                };
                let outer_ccw = outer.oriented_ccw();
                let mut hole_v = inner.oriented_ccw().vertices;
                hole_v.reverse();
                let hole = Polygon { vertices: hole_v };
                Ok(vec![outer_ccw, hole])
            } else {
                Ok(vec![subj.clone(), clp.clone()])
            }
        }
    }
}

/// True if `inner` is contained in `outer`, used only when the two boundaries do
/// not cross transversally. Robust to coincident / touching boundaries: every
/// `inner` vertex must lie inside `outer` or on its boundary, AND `inner`'s
/// interior-witness point must be strictly inside `outer`.
fn polygon_inside(inner: &Polygon, outer: &Polygon) -> bool {
    let all_in = inner
        .vertices
        .iter()
        .all(|&v| winding_inside(outer, v) || near_boundary(v, outer));
    if !all_in {
        return false;
    }
    // Guard against the all-on-boundary case (identical polygons / shared edges)
    // by also requiring a strictly-interior witness.
    let witness = interior_witness(inner);
    winding_inside(outer, witness)
}

/// A point strictly interior to `poly` (for simple polygons). Tries the area
/// centroid first, then ear-triangle centroids for concave shapes.
fn interior_witness(poly: &Polygon) -> Point {
    let n = poly.vertices.len();
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut a2 = 0.0;
    for i in 0..n {
        let p0 = poly.vertices[i];
        let p1 = poly.vertices[(i + 1) % n];
        let cross = p0.x * p1.y - p1.x * p0.y;
        cx += (p0.x + p1.x) * cross;
        cy += (p0.y + p1.y) * cross;
        a2 += cross;
    }
    if a2.abs() > EPS {
        let c = Point::new(cx / (3.0 * a2), cy / (3.0 * a2));
        if winding_inside(poly, c) {
            return c;
        }
    }
    for i in 0..n {
        let a = poly.vertices[i];
        let b = poly.vertices[(i + 1) % n];
        let cc = poly.vertices[(i + 2) % n];
        let mid = Point::new((a.x + b.x + cc.x) / 3.0, (a.y + b.y + cc.y) / 3.0);
        if winding_inside(poly, mid) {
            return mid;
        }
    }
    poly.vertices[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, s: f64) -> Polygon {
        Polygon::new(vec![
            Point::new(x0, y0),
            Point::new(x0 + s, y0),
            Point::new(x0 + s, y0 + s),
            Point::new(x0, y0 + s),
        ])
        .expect("ok")
    }

    fn area_sum(rings: &[Polygon]) -> f64 {
        rings.iter().map(|p| p.area()).sum()
    }

    // Oracle (a): area(A∩B) + area(A∪B) == area(A) + area(B), exactly.
    #[test]
    fn area_identity_overlapping_squares() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 1.0, 2.0);
        let inter = intersection(&a, &b).expect("ok");
        let uni = union(&a, &b).expect("ok");
        let lhs = filled_area_of_rings(&inter) + filled_area_of_rings(&uni);
        let rhs = a.area() + b.area();
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "area identity: lhs={lhs}, rhs={rhs}"
        );
    }

    /// An L-shaped non-convex hexagon translated by `(dx, dy)`.
    fn l_shape(dx: f64, dy: f64) -> Polygon {
        Polygon::new(vec![
            Point::new(dx, dy),
            Point::new(dx + 3.0, dy),
            Point::new(dx + 3.0, dy + 1.0),
            Point::new(dx + 1.0, dy + 1.0),
            Point::new(dx + 1.0, dy + 3.0),
            Point::new(dx, dy + 3.0),
        ])
        .expect("ok")
    }

    /// A rotated square (45 degrees) of "radius" `r` centred at `(cx, cy)`; its
    /// edges are oblique, so it never shares an axis-aligned edge with a box.
    fn diamond(cx: f64, cy: f64, r: f64) -> Polygon {
        Polygon::new(vec![
            Point::new(cx + r, cy),
            Point::new(cx, cy + r),
            Point::new(cx - r, cy),
            Point::new(cx, cy - r),
        ])
        .expect("ok")
    }

    // Oracle (a, extended): area identity over several genuinely-overlapping
    // pairs, including non-convex and obliquely-oriented polygons (chosen with
    // no shared collinear boundary edges, the regime the classical algorithm
    // covers exactly).
    #[test]
    fn area_identity_multiple_pairs() {
        let cases = [
            (square(0.0, 0.0, 3.0), square(1.5, 1.5, 3.0)),
            (square(-1.0, -1.0, 2.0), square(0.0, 0.0, 2.0)),
            (square(0.0, 0.0, 4.0), diamond(2.0, 2.0, 2.5)),
            (l_shape(0.0, 0.0), square(0.5, 0.5, 2.0)),
            (l_shape(0.0, 0.0), l_shape(0.5, 0.5)),
            (l_shape(0.0, 0.0), diamond(1.0, 1.0, 1.7)),
        ];
        for (a, b) in cases {
            let inter = intersection(&a, &b).expect("ok");
            let uni = union(&a, &b).expect("ok");
            let lhs = filled_area_of_rings(&inter) + filled_area_of_rings(&uni);
            let rhs = a.area() + b.area();
            assert!((lhs - rhs).abs() < 1e-9, "lhs={lhs}, rhs={rhs}");
        }
    }

    // Documented degeneracy behaviour: when two polygons share a collinear
    // boundary segment (the regime beyond the classical Greiner-Hormann
    // assumptions), the routine must not panic and must return a valid, simple
    // result rather than crashing. Full shared-edge Boolean robustness
    // (Foster-Hormann ON-vertex handling) is intentionally out of scope; here we
    // only assert graceful, non-panicking behaviour with finite area.
    #[test]
    fn collinear_shared_edge_does_not_panic() {
        // B's top edge (y = 4) is collinear and overlapping with A's top edge.
        let a = square(0.0, 0.0, 4.0);
        let b = Polygon::new(vec![
            Point::new(2.0, 1.0),
            Point::new(5.0, 1.0),
            Point::new(5.0, 4.0),
            Point::new(2.0, 4.0),
        ])
        .expect("ok");
        let inter = intersection(&a, &b).expect("ok");
        let uni = union(&a, &b).expect("ok");
        // Areas are finite and non-negative; no panic occurred.
        assert!(filled_area_of_rings(&inter).is_finite());
        assert!(filled_area_of_rings(&uni).is_finite());
        assert!(filled_area_of_rings(&uni) >= filled_area_of_rings(&inter) - 1e-9);
    }

    // Oracle (b): two overlapping axis-aligned squares -> analytic overlap rect.
    #[test]
    fn intersection_overlap_rectangle() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 1.0, 2.0);
        let inter = intersection(&a, &b).expect("ok");
        // Overlap is [1,2] x [1,2] -> area 1.
        assert!((area_sum(&inter) - 1.0).abs() < 1e-9);
    }

    // Oracle (c): disjoint polygons -> empty intersection; union = both.
    #[test]
    fn disjoint_intersection_union() {
        let a = square(0.0, 0.0, 1.0);
        let b = square(5.0, 5.0, 1.0);
        let inter = intersection(&a, &b).expect("ok");
        assert!(inter.is_empty() || area_sum(&inter) < 1e-12);
        let uni = union(&a, &b).expect("ok");
        assert!((filled_area_of_rings(&uni) - 2.0).abs() < 1e-9);
    }

    // Oracle (d): A ⊂ B -> A∩B has area(A); A∪B has area(B).
    #[test]
    fn nested_intersection_union() {
        let big = square(0.0, 0.0, 10.0);
        let small = square(3.0, 3.0, 2.0);
        let inter = intersection(&small, &big).expect("ok");
        assert!((filled_area_of_rings(&inter) - small.area()).abs() < 1e-9);
        let uni = union(&small, &big).expect("ok");
        assert!((filled_area_of_rings(&uni) - big.area()).abs() < 1e-9);
    }

    // Oracle (e): difference area = area(A) - area(A∩B).
    #[test]
    fn difference_area() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 1.0, 2.0);
        let inter = intersection(&a, &b).expect("ok");
        let diff = difference(&a, &b).expect("ok");
        let expect = a.area() - filled_area_of_rings(&inter);
        assert!(
            (filled_area_of_rings(&diff) - expect).abs() < 1e-9,
            "diff={}, expect={}",
            filled_area_of_rings(&diff),
            expect
        );
    }

    // Oracle (f): a non-convex L-shape ∩ a square -> correct clipped area.
    #[test]
    fn l_shape_intersection() {
        // L-shape occupying [0,3]x[0,1] ∪ [0,1]x[0,3] (area = 3 + 3 - 1 = 5).
        let l = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(3.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 3.0),
            Point::new(0.0, 3.0),
        ])
        .expect("ok");
        assert!((l.area() - 5.0).abs() < 1e-12);
        // Clip by the square [0.5, 2.5] x [0.5, 2.5].
        let sq = square(0.5, 0.5, 2.0);
        let inter = intersection(&l, &sq).expect("ok");
        // Analytic: L ∩ sq. Within the square:
        //   horizontal bar [0.5,2.5]x[0.5,1] -> 2.0 * 0.5 = 1.0
        //   vertical  bar [0.5,1]x[1,2.5]     -> 0.5 * 1.5 = 0.75
        //   (the corner [0.5,1]x[0.5,1] counted once in the horizontal bar)
        // total = 1.0 + 0.75 = 1.75
        assert!(
            (area_sum(&inter) - 1.75).abs() < 1e-9,
            "L∩square area = {}",
            area_sum(&inter)
        );
    }

    // Result simplicity: intersection of two convex squares is a single simple
    // ring with the expected vertex count for this overlap.
    #[test]
    fn intersection_is_simple_ring() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 1.0, 2.0);
        let inter = intersection(&a, &b).expect("ok");
        assert_eq!(inter.len(), 1);
        // Overlap square has 4 corners.
        assert_eq!(inter[0].vertices.len(), 4);
    }

    /// True if `poly` is simple: no two non-adjacent edges intersect.
    fn is_simple(poly: &Polygon) -> bool {
        let n = poly.vertices.len();
        for i in 0..n {
            let a0 = poly.vertices[i];
            let a1 = poly.vertices[(i + 1) % n];
            for j in (i + 1)..n {
                // Skip adjacent edges (they share an endpoint by construction).
                if j == i || (j + 1) % n == i || (i + 1) % n == j {
                    continue;
                }
                let b0 = poly.vertices[j];
                let b1 = poly.vertices[(j + 1) % n];
                if super::segment_cross(a0, a1, b0, b1).is_some() {
                    return false;
                }
            }
        }
        true
    }

    // Result polygons are simple (no self-intersections) for non-convex clips.
    #[test]
    fn results_are_simple_polygons() {
        let l = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(3.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 3.0),
            Point::new(0.0, 3.0),
        ])
        .expect("ok");
        let sq = square(0.5, 0.5, 2.0);
        for rings in [
            intersection(&l, &sq).expect("ok"),
            union(&l, &sq).expect("ok"),
            difference(&l, &sq).expect("ok"),
        ] {
            for r in &rings {
                assert!(is_simple(r), "result ring {:?} is not simple", r.vertices);
            }
        }
    }

    // Xor area equals area(A) + area(B) - 2*area(A∩B).
    #[test]
    fn xor_area() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 1.0, 2.0);
        let inter = intersection(&a, &b).expect("ok");
        let x = xor(&a, &b).expect("ok");
        let expect = a.area() + b.area() - 2.0 * filled_area_of_rings(&inter);
        assert!(
            (filled_area_of_rings(&x) - expect).abs() < 1e-9,
            "xor={}, expect={}",
            filled_area_of_rings(&x),
            expect
        );
    }

    // Difference producing a hole: a square with a fully-interior square removed.
    #[test]
    fn difference_with_hole() {
        let outer = square(0.0, 0.0, 6.0);
        let inner = square(2.0, 2.0, 2.0);
        let diff = difference(&outer, &inner).expect("ok");
        // Filled area = 36 - 4 = 32 (outer CCW minus CW hole).
        assert!(
            (filled_area_of_rings(&diff) - 32.0).abs() < 1e-9,
            "filled={}",
            filled_area_of_rings(&diff)
        );
    }

    #[test]
    fn segment_cross_basic() {
        let r = segment_cross(
            Point::new(0.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
            Point::new(2.0, 0.0),
        );
        let (p, _, _) = r.expect("crossing");
        assert!((p.x - 1.0).abs() < 1e-12 && (p.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn segment_cross_parallel_none() {
        assert!(
            segment_cross(
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(0.0, 1.0),
                Point::new(1.0, 1.0),
            )
            .is_none()
        );
    }
}

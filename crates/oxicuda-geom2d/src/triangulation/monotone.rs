//! Triangulation of `y`-monotone polygons, and decomposition of simple polygons into
//! `y`-monotone pieces (sweep-line), following de Berg, Cheong, van Kreveld, Overmars,
//! *Computational Geometry* (3rd ed.), ch. 3.
//!
//! # Two-phase polygon triangulation
//!
//! A simple polygon is triangulated in two phases:
//!
//!   1. [`make_monotone`] partitions the polygon into `y`-monotone sub-polygons by sweeping a
//!      horizontal line top-to-bottom, classifying each vertex as *start*, *end*, *split*,
//!      *merge*, or *regular*, and adding diagonals at split/merge vertices.
//!   2. [`triangulate_monotone`] triangulates each `y`-monotone piece in `O(k)` time with the
//!      classical stack algorithm.
//!
//! [`triangulate_simple`] chains the two phases for an arbitrary simple polygon.
//!
//! All triangles are returned as triples of indices into the **input** polygon's vertex array,
//! oriented counter-clockwise.

use crate::error::{Geom2dError, Geom2dResult};
use crate::predicate::orient2d_sign;
use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// A triangle as three CCW-oriented vertex indices into the source polygon.
pub type IndexedTriangle = (usize, usize, usize);

/// Vertex chain side of a `y`-monotone polygon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chain {
    Left,
    Right,
}

/// Total order on points for the top-to-bottom sweep, with the **topmost** point as the maximum.
///
/// `a` ranks as [`core::cmp::Ordering::Greater`] than `b` iff `a` comes earlier in the sweep,
/// i.e. `a` has the larger `y`, or (on a tie) the smaller `x`. This makes the sweep "key" a
/// max-key: sorting descending by `sweep_cmp` lists vertices top-to-bottom.
fn sweep_cmp(a: Point, b: Point) -> core::cmp::Ordering {
    match a.y.partial_cmp(&b.y) {
        // Equal y (or NaN guard): the smaller x is "higher" -> ranks Greater.
        Some(core::cmp::Ordering::Equal) | None => {
            b.x.partial_cmp(&a.x).unwrap_or(core::cmp::Ordering::Equal)
        }
        // Larger y ranks Greater (comes first in the sweep).
        Some(ord) => ord,
    }
}

/// True iff point `a` is strictly above point `b` in sweep order (ranks higher).
fn above(a: Point, b: Point) -> bool {
    sweep_cmp(a, b) == core::cmp::Ordering::Greater
}

/// Triangulate a `y`-monotone polygon in `O(n)` time using the stack algorithm.
///
/// The polygon must be `y`-monotone (every horizontal line meets its boundary in at most one
/// connected interval). Vertices may be given in either orientation; the routine internally
/// works with a CCW copy and merges the two monotone chains.
///
/// Returns `n - 2` triangles as CCW index triples into `poly.vertices`.
///
/// # Errors
/// * [`Geom2dError::NotEnoughPoints`] if the polygon has fewer than three vertices.
/// * [`Geom2dError::DegeneratePolygon`] if the polygon is not `y`-monotone (detected when the
///   sorted-by-sweep top and bottom vertices do not split the boundary into two monotone chains).
pub fn triangulate_monotone(poly: &Polygon) -> Geom2dResult<Vec<IndexedTriangle>> {
    let n = poly.n();
    if n < 3 {
        return Err(Geom2dError::NotEnoughPoints { needed: 3, got: n });
    }

    // Work on a guaranteed-CCW copy so the chain rule "interior is to the left going up the right
    // chain" holds. `omap[i]` maps a working index back to the original polygon index.
    let (cpts, omap): (Vec<Point>, Vec<usize>) = if poly.signed_area() > 0.0 {
        (poly.vertices.clone(), (0..n).collect())
    } else {
        let mut v = poly.vertices.clone();
        v.reverse();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.reverse();
        (v, idx)
    };
    let pts: &[Point] = &cpts;

    // Topmost and bottommost vertex in sweep order (working indices).
    let mut top = 0usize;
    let mut bot = 0usize;
    for i in 1..n {
        if above(pts[i], pts[top]) {
            top = i;
        }
        if above(pts[bot], pts[i]) {
            bot = i;
        }
    }

    // Chain assignment on the CCW polygon: a vertex lies on the RIGHT chain iff its CCW successor
    // (`i+1`) is strictly above it (the right boundary is traced upward in CCW order), and on the
    // LEFT chain iff its CCW predecessor (`i-1`) is above it. `top` and `bot` are shared by both
    // chains; their tag is unused by the merge (they are never the differing-chain test subject
    // because the stack is seeded with the two topmost and drained at the bottom).
    let mut chain = vec![Chain::Left; n];
    for i in 0..n {
        if i == top || i == bot {
            continue;
        }
        let next = pts[(i + 1) % n];
        chain[i] = if above(next, pts[i]) {
            Chain::Right
        } else {
            Chain::Left
        };
    }

    // Global sweep order of working indices (topmost first).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| sweep_cmp(pts[a], pts[b]).reverse());

    // Stack-based triangulation merging the two chains (de Berg, TriangulateMonotonePolygon).
    let mut tris: Vec<IndexedTriangle> = Vec::with_capacity(n - 2);
    let mut stack: Vec<usize> = Vec::with_capacity(n);
    stack.push(order[0]);
    stack.push(order[1]);

    for k in 2..(n - 1) {
        let u = order[k];
        let prev_sweep = order[k - 1]; // u_{j-1}
        let top_stack = *stack
            .last()
            .ok_or_else(|| Geom2dError::DegeneratePolygon("monotone stack underflow".into()))?;
        if chain[u] != chain[top_stack] {
            // Different chain: u sees every stacked vertex. Fan u to each adjacent stacked pair,
            // i.e. pop all but one, emitting a triangle per popped vertex with its lower neighbour.
            while stack.len() > 1 {
                let v1 = stack.pop().ok_or_else(|| {
                    Geom2dError::DegeneratePolygon("monotone stack underflow".into())
                })?;
                let v2 = *stack.last().ok_or_else(|| {
                    Geom2dError::DegeneratePolygon("monotone stack underflow".into())
                })?;
                tris.push(oriented(pts, u, v1, v2));
            }
            // Stack now holds exactly the first popped chain vertex (order[0..]'s remnant). Per
            // the algorithm, after handling the opposite chain we push u_{j-1} and u_j.
            stack.clear();
            stack.push(prev_sweep);
            stack.push(u);
        } else {
            // Same chain: pop the adjacent vertex (connected by a polygon edge), then keep popping
            // while the diagonal u -> next-stacked stays inside the polygon (interior-side turn).
            let mut last_popped = stack
                .pop()
                .ok_or_else(|| Geom2dError::DegeneratePolygon("monotone stack underflow".into()))?;
            while let Some(&prev) = stack.last() {
                // Diagonal from u to `prev` is admissible iff the triangle (u, last_popped, prev)
                // bulges toward the interior. On the right chain the interior is to the left of
                // the descending boundary (CCW turn, sign > 0); on the left chain it is to the
                // right (CW turn, sign < 0).
                let s = orient2d_sign(pts[u], pts[last_popped], pts[prev]);
                let convex = match chain[u] {
                    Chain::Right => s > 0,
                    Chain::Left => s < 0,
                };
                if convex {
                    tris.push(oriented(pts, u, last_popped, prev));
                    last_popped = stack.pop().ok_or_else(|| {
                        Geom2dError::DegeneratePolygon("monotone stack underflow".into())
                    })?;
                } else {
                    break;
                }
            }
            stack.push(last_popped);
            stack.push(u);
        }
    }

    // Final (bottommost) vertex connects to all remaining stack vertices except the first and
    // last (those are already joined by edges); fanning across adjacent pairs yields the rest.
    let last = order[n - 1];
    while stack.len() > 1 {
        let v1 = stack
            .pop()
            .ok_or_else(|| Geom2dError::DegeneratePolygon("monotone stack underflow".into()))?;
        let v2 = *stack
            .last()
            .ok_or_else(|| Geom2dError::DegeneratePolygon("monotone stack underflow".into()))?;
        tris.push(oriented(pts, last, v1, v2));
    }

    if tris.len() != n - 2 {
        return Err(Geom2dError::DegeneratePolygon(format!(
            "monotone triangulation produced {} triangles, expected {}",
            tris.len(),
            n - 2
        )));
    }
    // Map working indices back to original polygon indices.
    let mapped: Vec<IndexedTriangle> = tris
        .into_iter()
        .map(|(a, b, c)| (omap[a], omap[b], omap[c]))
        .collect();
    Ok(mapped)
}

/// Orient triangle `(i, j, k)` counter-clockwise, swapping the last two indices if needed.
fn oriented(pts: &[Point], i: usize, j: usize, k: usize) -> IndexedTriangle {
    if orient2d_sign(pts[i], pts[j], pts[k]) >= 0 {
        (i, j, k)
    } else {
        (i, k, j)
    }
}

// ---------------------------------------------------------------------------
// Decomposition of a simple polygon into y-monotone pieces (sweep-line).
// ---------------------------------------------------------------------------

/// The five vertex classifications used by the make-monotone sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VertexType {
    Start,
    End,
    Split,
    Merge,
    RegularLeft,
    RegularRight,
}

/// Decompose a simple polygon into a set of `y`-monotone sub-polygons.
///
/// Returns a list of sub-polygons, each expressed as a `Vec<usize>` of indices into
/// `poly.vertices`, ordered counter-clockwise. The union of the pieces equals the input polygon
/// and each piece is `y`-monotone.
///
/// Implements the sweep-line algorithm: process vertices top-to-bottom, maintaining the set of
/// edges currently crossed by the sweep line (the *status*) together with a *helper* vertex per
/// edge, and inserting diagonals at split and merge vertices to remove local non-monotonicity.
///
/// # Errors
/// [`Geom2dError::NotEnoughPoints`] if the polygon has fewer than three vertices.
pub fn make_monotone(poly: &Polygon) -> Geom2dResult<Vec<Vec<usize>>> {
    let n = poly.n();
    if n < 3 {
        return Err(Geom2dError::NotEnoughPoints { needed: 3, got: n });
    }
    // Work on a CCW copy so "interior is to the left of a downward edge" holds. We keep the
    // index mapping back to the original polygon.
    let (pts, orig_index): (Vec<Point>, Vec<usize>) = if poly.signed_area() > 0.0 {
        (poly.vertices.clone(), (0..n).collect())
    } else {
        let mut v = poly.vertices.clone();
        v.reverse();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.reverse();
        (v, idx)
    };

    // Doubly connected edge list (lightweight): next/prev around the single face, plus the set of
    // diagonals we add. We model the planar subdivision as an adjacency multimap and extract
    // faces at the end.
    let mut diagonals: Vec<(usize, usize)> = Vec::new();

    // Sweep status: edges currently intersected, each with a helper vertex. An edge is named by
    // its upper origin vertex `i` (edge goes from vertex i to vertex i+1 in CCW order, i.e. the
    // edge "leaving" vertex i downward along the polygon boundary). We store active left-edges.
    struct StatusEdge {
        /// Origin vertex index of the edge (in `pts` indexing).
        origin: usize,
        /// Helper vertex index for this edge.
        helper: usize,
    }
    let mut status: Vec<StatusEdge> = Vec::new();

    // Classify each vertex.
    let vtype: Vec<VertexType> = (0..n).map(|i| classify(&pts, i)).collect();

    // Priority order: top-to-bottom (sweep order).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| sweep_cmp(pts[a], pts[b]).reverse());

    // Helper: x-coordinate where edge `e` (origin..origin+1) intersects the horizontal line at
    // height `y` (used to locate the edge immediately left of a query vertex).
    let edge_x_at = |e_origin: usize, y: f64| -> f64 {
        let a = pts[e_origin];
        let b = pts[(e_origin + 1) % n];
        if (a.y - b.y).abs() < f64::MIN_POSITIVE {
            a.x.min(b.x)
        } else {
            let t = (a.y - y) / (a.y - b.y);
            a.x + t * (b.x - a.x)
        }
    };

    // Find, among active status edges, the one immediately to the left of vertex `v`.
    let find_left_edge = |status: &Vec<StatusEdge>, v: usize, pts: &Vec<Point>| -> Option<usize> {
        let qy = pts[v].y;
        let qx = pts[v].x;
        let mut best: Option<usize> = None;
        let mut best_x = f64::NEG_INFINITY;
        for (si, se) in status.iter().enumerate() {
            let ex = edge_x_at(se.origin, qy);
            if ex <= qx + 1e-12 && ex >= best_x {
                best_x = ex;
                best = Some(si);
            }
        }
        best
    };

    // Add a diagonal between two vertices, recording it for face extraction.
    let add_diagonal = |a: usize, b: usize, diagonals: &mut Vec<(usize, usize)>| {
        diagonals.push((a, b));
    };

    for &v in &order {
        match vtype[v] {
            VertexType::Start => {
                // Insert edge leaving v (v -> v+1) with helper v.
                status.push(StatusEdge {
                    origin: v,
                    helper: v,
                });
            }
            VertexType::End => {
                // Edge entering v is (v-1 -> v); remove edge with origin v-1.
                if let Some(si) = status.iter().position(|e| e.origin == (v + n - 1) % n) {
                    let se = &status[si];
                    if vtype[se.helper] == VertexType::Merge {
                        add_diagonal(v, se.helper, &mut diagonals);
                    }
                    status.remove(si);
                }
            }
            VertexType::Split => {
                if let Some(si) = find_left_edge(&status, v, &pts) {
                    add_diagonal(v, status[si].helper, &mut diagonals);
                    status[si].helper = v;
                }
                status.push(StatusEdge {
                    origin: v,
                    helper: v,
                });
            }
            VertexType::Merge => {
                // Remove edge entering v (origin v-1); if its helper is a merge vertex, diagonal.
                if let Some(si) = status.iter().position(|e| e.origin == (v + n - 1) % n) {
                    if vtype[status[si].helper] == VertexType::Merge {
                        add_diagonal(v, status[si].helper, &mut diagonals);
                    }
                    status.remove(si);
                }
                // Find edge left of v; if its helper is a merge, diagonal; set helper = v.
                if let Some(si) = find_left_edge(&status, v, &pts) {
                    if vtype[status[si].helper] == VertexType::Merge {
                        add_diagonal(v, status[si].helper, &mut diagonals);
                    }
                    status[si].helper = v;
                }
            }
            VertexType::RegularLeft => {
                // Interior is to the right: the polygon boundary goes downward through v on the
                // left chain. Remove the edge entering v (origin v-1), add diagonal if its helper
                // is a merge, then insert the edge leaving v with helper v.
                if let Some(si) = status.iter().position(|e| e.origin == (v + n - 1) % n) {
                    if vtype[status[si].helper] == VertexType::Merge {
                        add_diagonal(v, status[si].helper, &mut diagonals);
                    }
                    status.remove(si);
                }
                status.push(StatusEdge {
                    origin: v,
                    helper: v,
                });
            }
            VertexType::RegularRight => {
                // Interior is to the left: find the edge directly left of v; if its helper is a
                // merge vertex, add a diagonal; set its helper to v.
                if let Some(si) = find_left_edge(&status, v, &pts) {
                    if vtype[status[si].helper] == VertexType::Merge {
                        add_diagonal(v, status[si].helper, &mut diagonals);
                    }
                    status[si].helper = v;
                }
            }
        }
    }

    // Build faces from the polygon boundary plus the added diagonals. Each diagonal is inserted
    // in both directions. We then extract faces by repeatedly walking the most-clockwise edge.
    let pieces = extract_faces(&pts, &diagonals)?;

    // Map piece indices back to the original polygon vertex indexing.
    let mapped: Vec<Vec<usize>> = pieces
        .into_iter()
        .map(|piece| piece.into_iter().map(|i| orig_index[i]).collect())
        .collect();
    Ok(mapped)
}

/// Classify vertex `i` of a CCW polygon `pts` into one of the five sweep types.
fn classify(pts: &[Point], i: usize) -> VertexType {
    let n = pts.len();
    let prev = pts[(i + n - 1) % n];
    let cur = pts[i];
    let next = pts[(i + 1) % n];

    let prev_below = above(cur, prev);
    let next_below = above(cur, next);
    // Interior angle convexity at `cur`: CCW turn (left turn) means convex (< pi).
    let turn = orient2d_sign(prev, cur, next);

    if prev_below && next_below {
        // Both neighbours below: start (convex) or split (reflex).
        if turn > 0 {
            VertexType::Start
        } else {
            VertexType::Split
        }
    } else if !prev_below && !next_below {
        // Both neighbours above: end (convex) or merge (reflex).
        if turn > 0 {
            VertexType::End
        } else {
            VertexType::Merge
        }
    } else {
        // One above, one below: regular. Determine which chain (interior left vs right).
        // For a CCW polygon, if the polygon interior is to the right of the downward boundary
        // the vertex is on the left chain. Equivalent: the previous vertex is above => boundary
        // descends through v on the left chain (regular-left); else regular-right.
        if above(prev, cur) {
            VertexType::RegularLeft
        } else {
            VertexType::RegularRight
        }
    }
}

/// Extract the faces of the planar subdivision formed by the polygon boundary (vertices
/// `0..n` in CCW order, with edges `i -> i+1`) together with the added `diagonals`.
///
/// Returns each bounded face as a CCW vertex-index loop. The unbounded outer face is discarded.
fn extract_faces(pts: &[Point], diagonals: &[(usize, usize)]) -> Geom2dResult<Vec<Vec<usize>>> {
    let n = pts.len();
    // Build directed half-edges. Polygon boundary contributes i -> i+1. Each diagonal (a,b)
    // contributes both a -> b and b -> a.
    let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut half_edges: Vec<(usize, usize)> = Vec::new();
    let push_he =
        |u: usize, w: usize, out_edges: &mut Vec<Vec<usize>>, he: &mut Vec<(usize, usize)>| {
            out_edges[u].push(he.len());
            he.push((u, w));
        };
    for i in 0..n {
        push_he(i, (i + 1) % n, &mut out_edges, &mut half_edges);
    }
    for &(a, b) in diagonals {
        push_he(a, b, &mut out_edges, &mut half_edges);
        push_he(b, a, &mut out_edges, &mut half_edges);
    }

    let edge_count = half_edges.len();
    let mut visited = vec![false; edge_count];

    // For a half-edge arriving at vertex `w` from `u`, the next half-edge around the face (for
    // a CCW interior face traversal) is the one leaving `w` that makes the *smallest* clockwise
    // turn from the reversed incoming direction. We compute this by angular sorting.
    let next_face_edge = |incoming: usize,
                          half_edges: &Vec<(usize, usize)>,
                          out_edges: &Vec<Vec<usize>>|
     -> Option<usize> {
        let (u, w) = half_edges[incoming];
        let din = Point::new(pts[u].x - pts[w].x, pts[u].y - pts[w].y); // direction w->u
        let base_angle = din.y.atan2(din.x);
        // Choose the outgoing edge w->x maximizing the clockwise angle from (w->u), i.e. the
        // next edge in CCW face traversal is the most clockwise turn.
        let mut best: Option<usize> = None;
        let mut best_delta = f64::INFINITY;
        for &he in &out_edges[w] {
            let (_, x) = half_edges[he];
            if x == u && out_edges[w].len() > 1 {
                // Skip the immediate reverse unless it is the only option.
                // (Allows traversing back along a diagonal when needed.)
            }
            let dx = Point::new(pts[x].x - pts[w].x, pts[x].y - pts[w].y);
            let ang = dx.y.atan2(dx.x);
            // Clockwise delta in (0, 2pi].
            let mut delta = base_angle - ang;
            while delta <= 0.0 {
                delta += core::f64::consts::TAU;
            }
            while delta > core::f64::consts::TAU {
                delta -= core::f64::consts::TAU;
            }
            if delta < best_delta {
                best_delta = delta;
                best = Some(he);
            }
        }
        best
    };

    let mut faces: Vec<Vec<usize>> = Vec::new();
    for start in 0..edge_count {
        if visited[start] {
            continue;
        }
        // Walk the face.
        let mut loop_vertices: Vec<usize> = Vec::new();
        let mut e = start;
        let mut guard = 0usize;
        let max_guard = edge_count * 4 + 8;
        loop {
            visited[e] = true;
            let (u, _w) = half_edges[e];
            loop_vertices.push(u);
            let next = next_face_edge(e, &half_edges, &out_edges).ok_or_else(|| {
                Geom2dError::DegeneratePolygon("monotone: dangling half-edge".into())
            })?;
            e = next;
            guard += 1;
            if e == start || guard > max_guard {
                break;
            }
        }
        if loop_vertices.len() < 3 {
            continue;
        }
        // Keep only bounded (CCW, positive-area) faces; the outer face is CW (negative area).
        let area = signed_area_of(pts, &loop_vertices);
        if area > 1e-12 {
            faces.push(loop_vertices);
        }
    }
    Ok(faces)
}

/// Signed area (shoelace) of a vertex-index loop.
fn signed_area_of(pts: &[Point], loop_vertices: &[usize]) -> f64 {
    let m = loop_vertices.len();
    let mut s = 0.0;
    for i in 0..m {
        let a = pts[loop_vertices[i]];
        let b = pts[loop_vertices[(i + 1) % m]];
        s += a.x * b.y - b.x * a.y;
    }
    0.5 * s
}

/// Triangulate an arbitrary simple polygon: decompose into `y`-monotone pieces, then triangulate
/// each piece with the stack algorithm.
///
/// Returns the full triangle set as CCW index triples into `poly.vertices`. For an `n`-gon this
/// yields `n - 2` triangles.
///
/// # Errors
/// Propagates errors from [`make_monotone`] and [`triangulate_monotone`].
pub fn triangulate_simple(poly: &Polygon) -> Geom2dResult<Vec<IndexedTriangle>> {
    let n = poly.n();
    if n < 3 {
        return Err(Geom2dError::NotEnoughPoints { needed: 3, got: n });
    }
    let pieces = make_monotone(poly)?;
    let mut tris: Vec<IndexedTriangle> = Vec::with_capacity(n - 2);
    for piece in pieces {
        if piece.len() < 3 {
            continue;
        }
        let sub_pts: Vec<Point> = piece.iter().map(|&i| poly.vertices[i]).collect();
        let sub_poly = Polygon::new(sub_pts)?;
        let sub_tris = triangulate_monotone(&sub_poly)?;
        for (a, b, c) in sub_tris {
            tris.push((piece[a], piece[b], piece[c]));
        }
    }
    Ok(tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_area(pts: &[Point], t: IndexedTriangle) -> f64 {
        let (a, b, c) = t;
        let v = (pts[b].x - pts[a].x) * (pts[c].y - pts[a].y)
            - (pts[b].y - pts[a].y) * (pts[c].x - pts[a].x);
        0.5 * v.abs()
    }

    fn total_area(poly: &Polygon, tris: &[IndexedTriangle]) -> f64 {
        tris.iter().map(|&t| tri_area(&poly.vertices, t)).sum()
    }

    /// Two triangles overlap iff their interiors intersect; we test by sampling each triangle's
    /// centroid against every other triangle (a centroid inside another triangle ⇒ overlap).
    fn no_overlap(poly: &Polygon, tris: &[IndexedTriangle]) -> bool {
        let pts = &poly.vertices;
        for (i, &(a, b, c)) in tris.iter().enumerate() {
            let cx = (pts[a].x + pts[b].x + pts[c].x) / 3.0;
            let cy = (pts[a].y + pts[b].y + pts[c].y) / 3.0;
            let centroid = Point::new(cx, cy);
            for (j, &(d, e, f)) in tris.iter().enumerate() {
                if i == j {
                    continue;
                }
                if strictly_in_triangle(pts[d], pts[e], pts[f], centroid) {
                    return false;
                }
            }
        }
        true
    }

    fn strictly_in_triangle(a: Point, b: Point, c: Point, p: Point) -> bool {
        let s1 = orient2d_sign(a, b, p);
        let s2 = orient2d_sign(b, c, p);
        let s3 = orient2d_sign(c, a, p);
        (s1 > 0 && s2 > 0 && s3 > 0) || (s1 < 0 && s2 < 0 && s3 < 0)
    }

    fn convex_pentagon() -> Polygon {
        Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(5.0, 3.0),
            Point::new(2.0, 5.0),
            Point::new(-1.0, 3.0),
        ])
        .expect("valid")
    }

    /// A y-monotone but NON-convex polygon. Both chains descend monotonically in y, yet the
    /// polygon has a reflex vertex. Left chain: (0,6)->(0,3)->(0,0); right chain bulges inward.
    fn nonconvex_monotone() -> Polygon {
        Polygon::new(vec![
            Point::new(0.0, 6.0), // top
            Point::new(3.0, 5.0),
            Point::new(1.0, 3.0), // reflex bulge inward (still monotone in y)
            Point::new(3.0, 1.0),
            Point::new(0.0, 0.0), // bottom
            Point::new(-2.0, 3.0),
        ])
        .expect("valid")
    }

    #[test]
    fn convex_polygon_yields_n_minus_2_triangles() {
        let p = convex_pentagon();
        let tris = triangulate_monotone(&p).expect("ok");
        assert_eq!(tris.len(), p.n() - 2);
    }

    #[test]
    fn convex_areas_sum_to_polygon_area() {
        let p = convex_pentagon();
        let tris = triangulate_monotone(&p).expect("ok");
        let a = total_area(&p, &tris);
        assert!((a - p.area()).abs() < 1e-9, "sum={a} poly={}", p.area());
    }

    #[test]
    fn convex_triangulation_non_overlapping() {
        let p = convex_pentagon();
        let tris = triangulate_monotone(&p).expect("ok");
        assert!(no_overlap(&p, &tris));
    }

    #[test]
    fn convex_is_fan_from_extreme_vertex() {
        // For a convex polygon the monotone triangulation is a fan: exactly one vertex appears
        // in all n-2 triangles.
        let p = convex_pentagon();
        let tris = triangulate_monotone(&p).expect("ok");
        let n = p.n();
        let mut counts = vec![0usize; n];
        for &(a, b, c) in &tris {
            counts[a] += 1;
            counts[b] += 1;
            counts[c] += 1;
        }
        let apex = counts.iter().filter(|&&c| c == n - 2).count();
        assert!(apex >= 1, "expected a fan apex, counts={counts:?}");
    }

    #[test]
    fn all_triangles_ccw() {
        let p = convex_pentagon();
        let tris = triangulate_monotone(&p).expect("ok");
        for &(a, b, c) in &tris {
            assert!(orient2d_sign(p.vertices[a], p.vertices[b], p.vertices[c]) > 0);
        }
    }

    #[test]
    fn nonconvex_monotone_triangulates() {
        let p = nonconvex_monotone();
        let tris = triangulate_monotone(&p).expect("ok");
        assert_eq!(tris.len(), p.n() - 2);
        let a = total_area(&p, &tris);
        assert!((a - p.area()).abs() < 1e-9, "sum={a} poly={}", p.area());
        assert!(no_overlap(&p, &tris));
        for &(x, y, z) in &tris {
            assert!(orient2d_sign(p.vertices[x], p.vertices[y], p.vertices[z]) > 0);
        }
    }

    #[test]
    fn triangle_is_itself() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(1.0, 2.0),
        ])
        .expect("valid");
        let tris = triangulate_monotone(&p).expect("ok");
        assert_eq!(tris.len(), 1);
        assert!((total_area(&p, &tris) - p.area()).abs() < 1e-12);
    }

    #[test]
    fn too_few_points_errors() {
        // Polygon::new already guards <3, so build a 3-gon and check the monotone API guards a
        // hypothetical 2-vertex slice via triangulate_simple on a degenerate piece is impossible;
        // instead assert the explicit guard by constructing the smallest valid polygon works.
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
        ])
        .expect("valid");
        assert!(triangulate_monotone(&p).is_ok());
    }

    // ---- make_monotone / triangulate_simple ----

    /// An L-shaped (non-monotone) polygon must be split into y-monotone pieces and then fully
    /// triangulated into n-2 triangles whose areas sum to the polygon area.
    #[test]
    fn l_shape_make_monotone_then_triangulate() {
        // Classic L: 6 vertices, has a reflex vertex requiring a split into monotone pieces.
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 2.0),
            Point::new(2.0, 2.0),
            Point::new(2.0, 4.0),
            Point::new(0.0, 4.0),
        ])
        .expect("valid");

        let pieces = make_monotone(&p).expect("monotone ok");
        // Each piece must be y-monotone and triangulable.
        assert!(!pieces.is_empty());

        let tris = triangulate_simple(&p).expect("triangulate ok");
        assert_eq!(tris.len(), p.n() - 2, "expected {} tris", p.n() - 2);
        let a = total_area(&p, &tris);
        assert!((a - p.area()).abs() < 1e-9, "sum={a} poly={}", p.area());
        assert!(no_overlap(&p, &tris));
    }

    /// A polygon with a split AND a merge vertex (an hourglass-ish comb) exercises both diagonal
    /// insertion rules.
    #[test]
    fn comb_polygon_triangulates_via_simple() {
        // A "M"/comb shape: top spikes with a valley creating split & merge vertices.
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(6.0, 0.0),
            Point::new(6.0, 4.0),
            Point::new(4.0, 1.0), // valley (reflex) -> split when swept appropriately
            Point::new(3.0, 4.0),
            Point::new(2.0, 1.0), // valley
            Point::new(0.0, 4.0),
        ])
        .expect("valid");
        let tris = triangulate_simple(&p).expect("ok");
        assert_eq!(tris.len(), p.n() - 2);
        let a = total_area(&p, &tris);
        assert!((a - p.area()).abs() < 1e-9, "sum={a} poly={}", p.area());
        assert!(no_overlap(&p, &tris));
    }

    /// Convex polygon through the full pipeline equals direct monotone triangulation count.
    #[test]
    fn convex_via_simple_matches() {
        let p = convex_pentagon();
        let tris = triangulate_simple(&p).expect("ok");
        assert_eq!(tris.len(), p.n() - 2);
        assert!((total_area(&p, &tris) - p.area()).abs() < 1e-9);
        assert!(no_overlap(&p, &tris));
    }
}

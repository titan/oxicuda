//! Bowyer-Watson incremental Delaunay triangulation.
//!
//! Algorithm:
//! 1. Construct a super-triangle enclosing all input points.
//! 2. For each input point p:
//!    a. Find all triangles whose circumcircle contains p ("bad triangles").
//!    b. Form the polygon hole = boundary of bad triangles (edges not shared between two bad triangles).
//!    c. Remove bad triangles; re-triangulate by connecting each hole edge to p.
//! 3. Remove triangles that share a vertex with the super-triangle.

use crate::error::{Geom2dError, Geom2dResult};
use crate::predicate::in_circle::in_circle_signed;
use crate::predicate::orientation::orient_value;
use crate::primitives::point::Point;

/// A triangle by three vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triangle {
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

impl Triangle {
    /// Build oriented CCW.
    pub fn new_ccw(a: usize, b: usize, c: usize, pts: &[Point]) -> Self {
        let pa = pts[a];
        let pb = pts[b];
        let pc = pts[c];
        if orient_value(pa, pb, pc) >= 0.0 {
            Self { a, b, c }
        } else {
            Self { a, b: c, c: b }
        }
    }

    /// Three undirected edges as (lo, hi).
    fn edges(self) -> [(usize, usize); 3] {
        [
            sorted_edge(self.a, self.b),
            sorted_edge(self.b, self.c),
            sorted_edge(self.c, self.a),
        ]
    }
}

fn sorted_edge(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// Compute Delaunay triangulation of points. Returns triangles with indices into `pts`.
///
/// Errors with `NotEnoughPoints` if `pts.len() < 3`, or `DegeneratePolygon` if all points
/// are collinear.
pub fn bowyer_watson(pts: &[Point]) -> Geom2dResult<Vec<Triangle>> {
    let n = pts.len();
    if n < 3 {
        return Err(Geom2dError::NotEnoughPoints { needed: 3, got: n });
    }
    // Check non-collinearity.
    let first = pts[0];
    let mut all_collinear = true;
    for i in 2..n {
        if orient_value(first, pts[1], pts[i]).abs() > 1e-12 {
            all_collinear = false;
            break;
        }
    }
    if all_collinear {
        return Err(Geom2dError::DegeneratePolygon(
            "all points collinear".into(),
        ));
    }

    // Build a super-triangle large enough to contain all points.
    let mut minx = pts[0].x;
    let mut maxx = pts[0].x;
    let mut miny = pts[0].y;
    let mut maxy = pts[0].y;
    for &p in pts {
        if p.x < minx {
            minx = p.x;
        }
        if p.x > maxx {
            maxx = p.x;
        }
        if p.y < miny {
            miny = p.y;
        }
        if p.y > maxy {
            maxy = p.y;
        }
    }
    let dx = (maxx - minx).max(1e-9);
    let dy = (maxy - miny).max(1e-9);
    let delta = dx.max(dy) * 20.0;
    let midx = (minx + maxx) / 2.0;
    let midy = (miny + maxy) / 2.0;
    let sa = Point::new(midx - delta, midy - delta);
    let sb = Point::new(midx + delta, midy - delta);
    let sc = Point::new(midx, midy + delta);

    let mut all_pts: Vec<Point> = pts.to_vec();
    let i_a = n;
    let i_b = n + 1;
    let i_c = n + 2;
    all_pts.push(sa);
    all_pts.push(sb);
    all_pts.push(sc);

    let mut tris: Vec<Triangle> = vec![Triangle::new_ccw(i_a, i_b, i_c, &all_pts)];

    for (p_idx, _) in pts.iter().enumerate() {
        let p = all_pts[p_idx];
        // Find triangles whose circumcircle contains p.
        let mut bad: Vec<usize> = Vec::new();
        for (ti, t) in tris.iter().enumerate() {
            let pa = all_pts[t.a];
            let pb = all_pts[t.b];
            let pc = all_pts[t.c];
            if in_circle_signed(pa, pb, pc, p) > 1e-12 {
                bad.push(ti);
            }
        }
        // Build the hole boundary (edges not shared between two bad triangles).
        let mut edge_count: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for &bi in &bad {
            for e in tris[bi].edges() {
                *edge_count.entry(e).or_insert(0) += 1;
            }
        }
        let boundary: Vec<(usize, usize)> = edge_count
            .into_iter()
            .filter(|&(_, c)| c == 1)
            .map(|(k, _)| k)
            .collect();
        // Remove bad triangles (descending indices).
        let mut bad_sorted = bad.clone();
        bad_sorted.sort_unstable_by(|a, b| b.cmp(a));
        for &bi in &bad_sorted {
            tris.remove(bi);
        }
        // Insert new triangles using p and each boundary edge.
        for (u, v) in boundary {
            // Skip degenerate triangles where p is collinear with the edge.
            let o = orient_value(all_pts[u], all_pts[v], p);
            if o.abs() < 1e-15 {
                continue;
            }
            tris.push(Triangle::new_ccw(u, v, p_idx, &all_pts));
        }
    }
    // Filter out triangles touching super-triangle vertices.
    tris.retain(|t| t.a < n && t.b < n && t.c < n);
    Ok(tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collinear_errs() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
        ];
        assert!(bowyer_watson(&pts).is_err());
    }

    #[test]
    fn three_points_one_triangle() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(1.0, 2.0),
        ];
        let t = bowyer_watson(&pts).expect("ok");
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn four_points_two_triangles() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let t = bowyer_watson(&pts).expect("ok");
        assert_eq!(t.len(), 2);
    }
}

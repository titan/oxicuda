//! Ear-clipping triangulation of a simple polygon. O(n^2).
//!
//! At each step, find an "ear" — a triangle formed by three consecutive vertices
//! `(v_{i-1}, v_i, v_{i+1})` such that:
//!   * triangle is CCW (convex at v_i for a CCW polygon)
//!   * no other polygon vertex lies inside the triangle
//!
//! Clip the ear, shrink the polygon, repeat until 3 vertices remain.

use crate::error::{Geom2dError, Geom2dResult};
use crate::predicate::orientation::orient_value;
use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// Triangle as three vertex indices into the original polygon.
pub type IndexedTriangle = (usize, usize, usize);

/// Triangulate a simple polygon into triangles by ear-clipping.
///
/// Returns a list of `(i, j, k)` triples referring to input polygon vertex indices.
pub fn ear_clipping(poly: &Polygon) -> Geom2dResult<Vec<IndexedTriangle>> {
    let mut indices: Vec<usize> = (0..poly.n()).collect();
    // Orient CCW.
    if !poly.is_ccw() {
        indices.reverse();
    }
    let pts: &[Point] = &poly.vertices;
    let mut out: Vec<IndexedTriangle> = Vec::with_capacity(poly.n().saturating_sub(2));
    let mut iter_count = 0_usize;
    let max_iters = poly.n() * poly.n() + 8;
    while indices.len() > 2 {
        iter_count += 1;
        if iter_count > max_iters {
            return Err(Geom2dError::NotConverged { iter: iter_count });
        }
        let n = indices.len();
        let mut found = false;
        for i in 0..n {
            let i_prev = indices[(i + n - 1) % n];
            let i_curr = indices[i];
            let i_next = indices[(i + 1) % n];
            let a = pts[i_prev];
            let b = pts[i_curr];
            let c = pts[i_next];
            let o = orient_value(a, b, c);
            if o <= 0.0 {
                continue; // reflex or collinear
            }
            // Check no other polygon vertex is inside triangle (a, b, c).
            let mut any_inside = false;
            for &k in &indices {
                if k == i_prev || k == i_curr || k == i_next {
                    continue;
                }
                let p = pts[k];
                if point_in_triangle_strict(a, b, c, p) {
                    any_inside = true;
                    break;
                }
            }
            if !any_inside {
                out.push((i_prev, i_curr, i_next));
                indices.remove(i);
                found = true;
                break;
            }
        }
        if !found {
            return Err(Geom2dError::DegeneratePolygon(
                "no ear found (non-simple polygon?)".into(),
            ));
        }
    }
    Ok(out)
}

/// True iff `p` lies in the closed triangle (a, b, c).
/// Includes the boundary so vertices lying on a candidate edge disqualify the ear.
fn point_in_triangle_strict(a: Point, b: Point, c: Point, p: Point) -> bool {
    let s1 = orient_value(a, b, p);
    let s2 = orient_value(b, c, p);
    let s3 = orient_value(c, a, p);
    let eps = 1e-12;
    let pos = s1 >= -eps && s2 >= -eps && s3 >= -eps;
    let neg = s1 <= eps && s2 <= eps && s3 <= eps;
    pos || neg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_two_triangles() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        let tris = ear_clipping(&p).expect("ok");
        assert_eq!(tris.len(), 2);
    }

    #[test]
    fn pentagon_three_triangles() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(-1.0, 1.0),
        ])
        .expect("ok");
        let tris = ear_clipping(&p).expect("ok");
        assert_eq!(tris.len(), 3);
    }

    #[test]
    fn concave_polygon_works() {
        // L-shape
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(0.0, 2.0),
        ])
        .expect("ok");
        let tris = ear_clipping(&p).expect("ok");
        assert_eq!(tris.len(), 4);
    }
}

//! Minkowski sum of two convex polygons.
//!
//! Both polygons must be convex and CCW-oriented. The sum is computed by merging
//! the edge vectors of both polygons sorted by polar angle.

use crate::error::{Geom2dError, Geom2dResult};
use crate::polygon_ops::convexity_test::is_convex;
use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// Compute the Minkowski sum `A (+) B` of two convex polygons.
pub fn minkowski_sum(a: &Polygon, b: &Polygon) -> Geom2dResult<Polygon> {
    if !is_convex(a) || !is_convex(b) {
        return Err(Geom2dError::NotConvex(
            "Minkowski sum requires convex polygons".into(),
        ));
    }
    let a = a.oriented_ccw();
    let b = b.oriented_ccw();
    // Rotate each polygon so its lowest-leftmost vertex comes first.
    let a_rot = rotated_to_lowest(&a);
    let b_rot = rotated_to_lowest(&b);
    let n = a_rot.len();
    let m = b_rot.len();
    let mut out: Vec<Point> = Vec::with_capacity(n + m);
    let mut i = 0_usize;
    let mut j = 0_usize;
    while i < n || j < m {
        let p_a_i = a_rot[i % n];
        let p_a_i1 = a_rot[(i + 1) % n];
        let p_b_j = b_rot[j % m];
        let p_b_j1 = b_rot[(j + 1) % m];
        out.push(Point::new(p_a_i.x + p_b_j.x, p_a_i.y + p_b_j.y));
        let dax = p_a_i1.x - p_a_i.x;
        let day = p_a_i1.y - p_a_i.y;
        let dbx = p_b_j1.x - p_b_j.x;
        let dby = p_b_j1.y - p_b_j.y;
        let cr = dax * dby - day * dbx;
        if i >= n {
            j += 1;
        } else if j >= m || cr > 0.0 {
            i += 1;
        } else if cr < 0.0 {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    Polygon::new(out)
}

fn rotated_to_lowest(poly: &Polygon) -> Vec<Point> {
    let n = poly.n();
    let mut min_idx = 0;
    for i in 1..n {
        let p = poly.vertices[i];
        let q = poly.vertices[min_idx];
        if p.y < q.y || (p.y == q.y && p.x < q.x) {
            min_idx = i;
        }
    }
    let mut out: Vec<Point> = Vec::with_capacity(n);
    for k in 0..n {
        out.push(poly.vertices[(min_idx + k) % n]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_squares_sum() {
        let a = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        let b = a.clone();
        let s = minkowski_sum(&a, &b).expect("ok");
        // The Minkowski sum of two unit squares is a 2x2 square with area 4.
        assert!((s.area() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn non_convex_errs() {
        let a = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(0.0, 2.0),
        ])
        .expect("ok");
        let b = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.5, 1.0),
        ])
        .expect("ok");
        assert!(minkowski_sum(&a, &b).is_err());
    }
}

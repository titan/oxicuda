//! O(log n) point-in-convex-polygon via binary search.
//!
//! Assumes polygon is convex and CCW-oriented. Returns true if `q` is strictly inside.

use crate::predicate::orientation::orient_value;
use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// True if `q` is strictly inside the convex CCW-oriented polygon `poly`.
///
/// Note: this routine assumes the caller has verified convexity. For non-convex polygons,
/// use winding-number or ray-casting tests instead.
#[must_use]
pub fn point_in_convex_polygon(poly: &Polygon, q: Point) -> bool {
    let n = poly.n();
    if n < 3 {
        return false;
    }
    let v0 = poly.vertices[0];
    // q must be on the same side as v2..v_{n-1} relative to v0.
    if orient_value(v0, poly.vertices[1], q) < 0.0 {
        return false;
    }
    if orient_value(v0, poly.vertices[n - 1], q) > 0.0 {
        return false;
    }
    // Binary search for the wedge containing q.
    let mut lo = 1_usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if orient_value(v0, poly.vertices[mid], q) >= 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    orient_value(poly.vertices[lo], poly.vertices[hi], q) > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pentagon() -> Polygon {
        let n = 5;
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let theta = std::f64::consts::TAU * (i as f64) / (n as f64);
            pts.push(Point::new(theta.cos(), theta.sin()));
        }
        Polygon::new(pts).expect("ok")
    }

    #[test]
    fn inside_origin() {
        assert!(point_in_convex_polygon(&pentagon(), Point::ORIGIN));
    }

    #[test]
    fn outside_far() {
        assert!(!point_in_convex_polygon(&pentagon(), Point::new(2.0, 0.0)));
    }

    #[test]
    fn square_center() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        assert!(point_in_convex_polygon(&p, Point::new(0.5, 0.5)));
        assert!(!point_in_convex_polygon(&p, Point::new(2.0, 2.0)));
    }
}

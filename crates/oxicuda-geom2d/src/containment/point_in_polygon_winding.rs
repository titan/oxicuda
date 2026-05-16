//! Winding number-based point-in-polygon test.
//!
//! Computes the signed crossing count: number of times the polygon winds CCW around the query.
//! Robust for non-convex and even self-intersecting polygons.

use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// Compute the winding number of `poly` around `q`.
#[must_use]
pub fn winding_number(poly: &Polygon, q: Point) -> i32 {
    let mut w = 0_i32;
    let n = poly.n();
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
    w
}

fn is_left(a: Point, b: Point, c: Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
}

/// True if `q` is strictly inside `poly` by the winding number test.
#[must_use]
pub fn point_in_polygon_winding(poly: &Polygon, q: Point) -> bool {
    winding_number(poly, q) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq() -> Polygon {
        Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok")
    }

    #[test]
    fn inside_center() {
        assert!(point_in_polygon_winding(&sq(), Point::new(0.5, 0.5)));
    }

    #[test]
    fn outside_far() {
        assert!(!point_in_polygon_winding(&sq(), Point::new(2.0, 2.0)));
    }

    #[test]
    fn concave_polygon() {
        // Star-shape concave polygon
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ])
        .expect("ok");
        // Point in the indent: (1.5, 1) should be outside (indent)
        assert!(point_in_polygon_winding(&p, Point::new(0.5, 1.0)));
        assert!(!point_in_polygon_winding(&p, Point::new(3.0, 1.0)));
    }
}

//! Ray-casting point-in-polygon test (Jordan curve theorem).
//!
//! Casts a horizontal ray from `q` to +inf and counts polygon edge crossings.
//! Odd count = inside.

use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// True if `q` lies strictly inside `poly` by ray casting.
#[must_use]
pub fn point_in_polygon_ray_cast(poly: &Polygon, q: Point) -> bool {
    let n = poly.n();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let vi = poly.vertices[i];
        let vj = poly.vertices[j];
        let cond = (vi.y > q.y) != (vj.y > q.y);
        if cond {
            let t = (q.y - vi.y) / (vj.y - vi.y);
            let x_intersect = vi.x + t * (vj.x - vi.x);
            if q.x < x_intersect {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
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
        assert!(point_in_polygon_ray_cast(&sq(), Point::new(0.5, 0.5)));
    }

    #[test]
    fn outside_far() {
        assert!(!point_in_polygon_ray_cast(&sq(), Point::new(2.0, 2.0)));
    }

    #[test]
    fn outside_left() {
        assert!(!point_in_polygon_ray_cast(&sq(), Point::new(-1.0, 0.5)));
    }

    #[test]
    fn concave_indent() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 4.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 4.0),
        ])
        .expect("ok");
        assert!(point_in_polygon_ray_cast(&p, Point::new(1.0, 1.0)));
        assert!(!point_in_polygon_ray_cast(&p, Point::new(2.0, 3.5)));
    }
}

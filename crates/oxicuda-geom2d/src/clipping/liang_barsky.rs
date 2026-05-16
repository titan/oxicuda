//! Liang-Barsky parametric line clipping against an AABB.

use crate::primitives::aabb::Aabb;
use crate::primitives::point::Point;

/// Clip segment `(p0, p1)` against `aabb` using the Liang-Barsky parametric method.
///
/// Returns `Some((p_lo, p_hi))` or `None` if the segment is entirely outside.
#[must_use]
pub fn liang_barsky(p0: Point, p1: Point, aabb: Aabb) -> Option<(Point, Point)> {
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let p = [-dx, dx, -dy, dy];
    let q = [
        p0.x - aabb.min.x,
        aabb.max.x - p0.x,
        p0.y - aabb.min.y,
        aabb.max.y - p0.y,
    ];
    let mut u1 = 0.0_f64;
    let mut u2 = 1.0_f64;
    for i in 0..4 {
        if p[i].abs() < 1e-15 {
            if q[i] < 0.0 {
                return None;
            }
            continue;
        }
        let t = q[i] / p[i];
        if p[i] < 0.0 {
            if t > u2 {
                return None;
            }
            if t > u1 {
                u1 = t;
            }
        } else {
            if t < u1 {
                return None;
            }
            if t < u2 {
                u2 = t;
            }
        }
    }
    let r0 = Point::new(p0.x + u1 * dx, p0.y + u1 * dy);
    let r1 = Point::new(p0.x + u2 * dx, p0.y + u2 * dy);
    Some((r0, r1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_segment() {
        let bb = Aabb::new(Point::new(0.0, 0.0), Point::new(10.0, 10.0));
        let r = liang_barsky(Point::new(2.0, 2.0), Point::new(8.0, 8.0), bb).expect("ok");
        assert!((r.0.x - 2.0).abs() < 1e-12);
    }

    #[test]
    fn outside_segment() {
        let bb = Aabb::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        assert!(liang_barsky(Point::new(2.0, 0.5), Point::new(3.0, 0.5), bb).is_none());
    }

    #[test]
    fn crossing_horizontal() {
        let bb = Aabb::new(Point::new(0.0, 0.0), Point::new(10.0, 10.0));
        let r = liang_barsky(Point::new(-5.0, 5.0), Point::new(15.0, 5.0), bb).expect("ok");
        assert!((r.0.x).abs() < 1e-12);
        assert!((r.1.x - 10.0).abs() < 1e-12);
    }
}

//! In-circle predicate via the 3x3 determinant.
//!
//! Given CCW-ordered triangle `(a, b, c)`, `in_circle_signed(a, b, c, d) > 0` iff `d` is strictly
//! inside the circumcircle of the triangle.

use crate::primitives::point::Point;

/// Signed value of the 4x4 in-circle determinant (>0 inside, <0 outside, =0 on boundary).
///
/// Assumes the triangle `(a, b, c)` is oriented CCW.
#[must_use]
pub fn in_circle_signed(a: Point, b: Point, c: Point, d: Point) -> f64 {
    let ax = a.x - d.x;
    let ay = a.y - d.y;
    let bx = b.x - d.x;
    let by = b.y - d.y;
    let cx = c.x - d.x;
    let cy = c.y - d.y;
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    // 3x3 cofactor expansion
    let det11 = bx * cy - by * cx;
    let det12 = ay * (b2 * cx - bx * c2) - ax * (b2 * cy - by * c2);
    // det = a2*det11 - ay*(bx*c2 - b2*cx) + ax*(by*c2 - b2*cy)
    // expanded form:
    a2 * det11 + (det12)
}

/// Decision form: returns `true` if `d` is strictly inside circumcircle of CCW triangle `(a,b,c)`.
#[must_use]
pub fn in_circle(a: Point, b: Point, c: Point, d: Point) -> bool {
    in_circle_signed(a, b, c, d) > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_unit_triangle_circumcircle() {
        // Triangle vertices on unit circle: circumcircle = unit circle.
        let a = Point::new(1.0, 0.0);
        let b = Point::new(-0.5, 3_f64.sqrt() / 2.0);
        let c = Point::new(-0.5, -3_f64.sqrt() / 2.0);
        // Origin lies inside.
        assert!(in_circle(a, b, c, Point::ORIGIN));
    }

    #[test]
    fn outside_unit_triangle_circumcircle() {
        let a = Point::new(1.0, 0.0);
        let b = Point::new(-0.5, 3_f64.sqrt() / 2.0);
        let c = Point::new(-0.5, -3_f64.sqrt() / 2.0);
        assert!(!in_circle(a, b, c, Point::new(2.0, 2.0)));
    }

    #[test]
    fn on_boundary_zero() {
        // unit-square corners' circumcircle is centered at (0.5,0.5), radius sqrt(2)/2
        let a = Point::new(0.0, 0.0);
        let b = Point::new(1.0, 0.0);
        let c = Point::new(1.0, 1.0);
        // (0,1) is on the same circle
        let v = in_circle_signed(a, b, c, Point::new(0.0, 1.0));
        assert!(v.abs() < 1e-10);
    }
}

//! Circle-circle intersection via radical line.

use crate::primitives::circle::Circle;
use crate::primitives::point::Point;

/// Result of intersecting two circles.
#[derive(Debug, Clone, PartialEq)]
pub enum CircleCircleIntersection {
    /// Disjoint or one strictly inside other.
    None,
    /// Tangent (single intersection point).
    One(Point),
    /// Two intersection points.
    Two(Point, Point),
    /// Identical circles (infinitely many points).
    Coincident,
}

/// Intersect two circles. Uses the standard radical-line formulation.
#[must_use]
pub fn intersect_circles(c1: Circle, c2: Circle) -> CircleCircleIntersection {
    let dx = c2.center.x - c1.center.x;
    let dy = c2.center.y - c1.center.y;
    let d2 = dx * dx + dy * dy;
    let d = d2.sqrt();
    if d < 1e-15 {
        if (c1.radius - c2.radius).abs() < 1e-12 {
            return CircleCircleIntersection::Coincident;
        }
        return CircleCircleIntersection::None;
    }
    let r1 = c1.radius;
    let r2 = c2.radius;
    if d > r1 + r2 + 1e-12 || d < (r1 - r2).abs() - 1e-12 {
        return CircleCircleIntersection::None;
    }
    let a = (r1 * r1 - r2 * r2 + d2) / (2.0 * d);
    let h2 = r1 * r1 - a * a;
    let h = h2.max(0.0).sqrt();
    let px = c1.center.x + a * dx / d;
    let py = c1.center.y + a * dy / d;
    let rx = -dy * h / d;
    let ry = dx * h / d;
    let p1 = Point::new(px + rx, py + ry);
    let p2 = Point::new(px - rx, py - ry);
    if p1.distance_sq(p2) < 1e-20 {
        CircleCircleIntersection::One(p1)
    } else {
        CircleCircleIntersection::Two(p1, p2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_circles_intersect() {
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(1.0, 0.0), 1.0);
        match intersect_circles(c1, c2) {
            CircleCircleIntersection::Two(a, b) => {
                assert!((a.x - 0.5).abs() < 1e-10);
                assert!((b.x - 0.5).abs() < 1e-10);
                assert!((a.y + b.y).abs() < 1e-10);
            }
            other => panic!("expected Two, got {other:?}"),
        }
    }

    #[test]
    fn tangent_external() {
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(2.0, 0.0), 1.0);
        match intersect_circles(c1, c2) {
            CircleCircleIntersection::One(p) => {
                assert!((p.x - 1.0).abs() < 1e-10 && p.y.abs() < 1e-10);
            }
            other => panic!("expected One, got {other:?}"),
        }
    }

    #[test]
    fn disjoint() {
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(5.0, 0.0), 1.0);
        assert_eq!(intersect_circles(c1, c2), CircleCircleIntersection::None);
    }

    #[test]
    fn coincident_circles() {
        let c1 = Circle::new(Point::new(1.0, 2.0), 3.0);
        let c2 = Circle::new(Point::new(1.0, 2.0), 3.0);
        assert_eq!(
            intersect_circles(c1, c2),
            CircleCircleIntersection::Coincident
        );
    }
}

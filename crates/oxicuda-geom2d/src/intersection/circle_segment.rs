//! Circle-segment intersection via quadratic equation.

use crate::primitives::circle::Circle;
use crate::primitives::point::Point;
use crate::primitives::segment::Segment;

/// Result of intersecting a segment with a circle.
#[derive(Debug, Clone, PartialEq)]
pub enum CircleSegmentIntersection {
    /// No intersection.
    None,
    /// Segment is tangent: one point.
    One(Point),
    /// Segment cuts circle in two points.
    Two(Point, Point),
}

/// Intersect segment `seg` with `circ`. Solves `|seg(t) - c|^2 = r^2` for `t in [0, 1]`.
#[must_use]
pub fn intersect_circle_segment(seg: Segment, circ: Circle) -> CircleSegmentIntersection {
    let d = seg.direction();
    let f_x = seg.a.x - circ.center.x;
    let f_y = seg.a.y - circ.center.y;
    let aa = d.x * d.x + d.y * d.y;
    let bb = 2.0 * (f_x * d.x + f_y * d.y);
    let cc = f_x * f_x + f_y * f_y - circ.radius * circ.radius;
    if aa.abs() < 1e-30 {
        // Degenerate segment
        if cc.abs() < 1e-12 {
            return CircleSegmentIntersection::One(seg.a);
        }
        return CircleSegmentIntersection::None;
    }
    let disc = bb * bb - 4.0 * aa * cc;
    if disc < -1e-12 {
        return CircleSegmentIntersection::None;
    }
    let disc = disc.max(0.0);
    let sq = disc.sqrt();
    let t1 = (-bb - sq) / (2.0 * aa);
    let t2 = (-bb + sq) / (2.0 * aa);
    let mut pts: Vec<Point> = Vec::new();
    for &t in &[t1, t2] {
        if (-1.0e-12..=1.0 + 1.0e-12).contains(&t) {
            pts.push(seg.point_at(t.clamp(0.0, 1.0)));
        }
    }
    match pts.len() {
        0 => CircleSegmentIntersection::None,
        1 => CircleSegmentIntersection::One(pts[0]),
        _ => {
            if pts[0].distance_sq(pts[1]) < 1e-20 {
                CircleSegmentIntersection::One(pts[0])
            } else {
                CircleSegmentIntersection::Two(pts[0], pts[1])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diameter_two_intersections() {
        let c = Circle::new(Point::ORIGIN, 1.0);
        let s = Segment::new(Point::new(-2.0, 0.0), Point::new(2.0, 0.0));
        match intersect_circle_segment(s, c) {
            CircleSegmentIntersection::Two(a, b) => {
                assert!(
                    a.distance_sq(Point::new(-1.0, 0.0))
                        .min(a.distance_sq(Point::new(1.0, 0.0)))
                        < 1e-12
                );
                assert!(
                    b.distance_sq(Point::new(-1.0, 0.0))
                        .min(b.distance_sq(Point::new(1.0, 0.0)))
                        < 1e-12
                );
            }
            other => panic!("expected Two, got {other:?}"),
        }
    }

    #[test]
    fn tangent_one_intersection() {
        let c = Circle::new(Point::ORIGIN, 1.0);
        let s = Segment::new(Point::new(-2.0, 1.0), Point::new(2.0, 1.0));
        match intersect_circle_segment(s, c) {
            CircleSegmentIntersection::One(p) => {
                assert!((p.y - 1.0).abs() < 1e-10 && p.x.abs() < 1e-10);
            }
            other => panic!("expected One, got {other:?}"),
        }
    }

    #[test]
    fn no_intersection() {
        let c = Circle::new(Point::ORIGIN, 1.0);
        let s = Segment::new(Point::new(-2.0, 2.0), Point::new(2.0, 2.0));
        assert_eq!(
            intersect_circle_segment(s, c),
            CircleSegmentIntersection::None
        );
    }
}

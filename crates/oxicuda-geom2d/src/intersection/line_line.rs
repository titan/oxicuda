//! Line-line intersection in 2D.

use crate::primitives::line::Line;
use crate::primitives::point::Point;

/// Result of intersecting two infinite lines.
#[derive(Debug, Clone, PartialEq)]
pub enum LineLineIntersection {
    /// Parallel, no intersection (or strictly different).
    Parallel,
    /// Coincident (same line).
    Coincident,
    /// A unique point.
    Point(Point),
}

/// Solve the 2x2 system for the intersection of two parametric lines.
#[must_use]
pub fn intersect_lines(l1: Line, l2: Line) -> LineLineIntersection {
    let p = l1.p;
    let r = l1.dir;
    let q = l2.p;
    let ss = l2.dir;
    let denom = r.x * ss.y - r.y * ss.x;
    let qp = q - p;
    if denom.abs() < 1e-15 {
        // Parallel
        let num_t = qp.x * ss.y - qp.y * ss.x;
        if num_t.abs() < 1e-12 {
            LineLineIntersection::Coincident
        } else {
            LineLineIntersection::Parallel
        }
    } else {
        let t = (qp.x * ss.y - qp.y * ss.x) / denom;
        LineLineIntersection::Point(Point::new(p.x + t * r.x, p.y + t * r.y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::vector::Vector;

    #[test]
    fn cross_unit() {
        let l1 = Line::new(Point::new(0.0, 0.0), Vector::new(1.0, 1.0));
        let l2 = Line::new(Point::new(0.0, 2.0), Vector::new(1.0, -1.0));
        match intersect_lines(l1, l2) {
            LineLineIntersection::Point(p) => {
                assert!((p.x - 1.0).abs() < 1e-12 && (p.y - 1.0).abs() < 1e-12);
            }
            other => panic!("expected Point, got {other:?}"),
        }
    }

    #[test]
    fn parallel_distinct() {
        let l1 = Line::new(Point::new(0.0, 0.0), Vector::UNIT_X);
        let l2 = Line::new(Point::new(0.0, 1.0), Vector::UNIT_X);
        assert_eq!(intersect_lines(l1, l2), LineLineIntersection::Parallel);
    }

    #[test]
    fn coincident_same_line() {
        let l1 = Line::new(Point::new(0.0, 0.0), Vector::new(2.0, 0.0));
        let l2 = Line::new(Point::new(5.0, 0.0), Vector::UNIT_X);
        assert_eq!(intersect_lines(l1, l2), LineLineIntersection::Coincident);
    }
}

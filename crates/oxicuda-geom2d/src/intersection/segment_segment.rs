//! Segment-segment intersection in 2D, handling collinear overlap.

use crate::primitives::point::Point;
use crate::primitives::segment::Segment;

/// Result of intersecting two segments.
#[derive(Debug, Clone, PartialEq)]
pub enum SegmentSegmentIntersection {
    /// Disjoint, no intersection.
    None,
    /// A single intersection point.
    Point(Point),
    /// A collinear overlapping subsegment.
    Overlap(Segment),
}

/// Test whether `q` lies on the closed segment `[a, b]` (assuming collinearity).
fn on_segment_collinear(a: Point, b: Point, q: Point) -> bool {
    let minx = a.x.min(b.x);
    let maxx = a.x.max(b.x);
    let miny = a.y.min(b.y);
    let maxy = a.y.max(b.y);
    q.x >= minx - 1e-15 && q.x <= maxx + 1e-15 && q.y >= miny - 1e-15 && q.y <= maxy + 1e-15
}

/// Intersect segments `s1 = [p, p2]` and `s2 = [q, q2]`.
///
/// Solves parametric system `p + t * (p2 - p) = q + s * (q2 - q)` with `t, s in [0, 1]`.
/// Handles collinear overlap and shared-endpoint cases.
#[must_use]
pub fn intersect_segments(s1: Segment, s2: Segment) -> SegmentSegmentIntersection {
    let p = s1.a;
    let p2 = s1.b;
    let q = s2.a;
    let q2 = s2.b;
    let r = p2 - p;
    let ss = q2 - q;
    let denom = r.x * ss.y - r.y * ss.x;
    let qp = q - p;
    let num_t = qp.x * ss.y - qp.y * ss.x;
    let num_s = qp.x * r.y - qp.y * r.x;

    if denom.abs() < 1e-15 {
        // Parallel
        if num_t.abs() > 1e-12 || num_s.abs() > 1e-12 {
            return SegmentSegmentIntersection::None;
        }
        // Collinear: project onto direction of s1
        let r2 = r.x * r.x + r.y * r.y;
        if r2 == 0.0 {
            // s1 is a point
            if on_segment_collinear(q, q2, p) {
                return SegmentSegmentIntersection::Point(p);
            }
            return SegmentSegmentIntersection::None;
        }
        let t0 = (qp.x * r.x + qp.y * r.y) / r2;
        let qp2 = q2 - p;
        let t1 = (qp2.x * r.x + qp2.y * r.y) / r2;
        let (tlo, thi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        let lo = tlo.max(0.0);
        let hi = thi.min(1.0);
        if lo > hi + 1e-15 {
            return SegmentSegmentIntersection::None;
        }
        let p_lo = Point::new(p.x + lo * r.x, p.y + lo * r.y);
        if (hi - lo).abs() < 1e-15 {
            return SegmentSegmentIntersection::Point(p_lo);
        }
        let p_hi = Point::new(p.x + hi * r.x, p.y + hi * r.y);
        return SegmentSegmentIntersection::Overlap(Segment::new(p_lo, p_hi));
    }

    let t = num_t / denom;
    let s = num_s / denom;
    if (-1.0e-12..=1.0 + 1.0e-12).contains(&t) && (-1.0e-12..=1.0 + 1.0e-12).contains(&s) {
        let pt = Point::new(p.x + t * r.x, p.y + t * r.y);
        SegmentSegmentIntersection::Point(pt)
    } else {
        SegmentSegmentIntersection::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_at_one_one() {
        let s1 = Segment::new(Point::new(0.0, 0.0), Point::new(2.0, 2.0));
        let s2 = Segment::new(Point::new(0.0, 2.0), Point::new(2.0, 0.0));
        let r = intersect_segments(s1, s2);
        match r {
            SegmentSegmentIntersection::Point(p) => {
                assert!((p.x - 1.0).abs() < 1e-12);
                assert!((p.y - 1.0).abs() < 1e-12);
            }
            other => panic!("expected Point, got {other:?}"),
        }
    }

    #[test]
    fn parallel_disjoint() {
        let s1 = Segment::new(Point::new(0.0, 0.0), Point::new(1.0, 0.0));
        let s2 = Segment::new(Point::new(0.0, 1.0), Point::new(1.0, 1.0));
        assert_eq!(intersect_segments(s1, s2), SegmentSegmentIntersection::None);
    }

    #[test]
    fn collinear_overlap() {
        let s1 = Segment::new(Point::new(0.0, 0.0), Point::new(2.0, 0.0));
        let s2 = Segment::new(Point::new(1.0, 0.0), Point::new(3.0, 0.0));
        match intersect_segments(s1, s2) {
            SegmentSegmentIntersection::Overlap(o) => {
                assert!((o.a.x - 1.0).abs() < 1e-12 && o.a.y.abs() < 1e-12);
                assert!((o.b.x - 2.0).abs() < 1e-12 && o.b.y.abs() < 1e-12);
            }
            other => panic!("expected overlap, got {other:?}"),
        }
    }

    #[test]
    fn collinear_disjoint() {
        let s1 = Segment::new(Point::new(0.0, 0.0), Point::new(1.0, 0.0));
        let s2 = Segment::new(Point::new(2.0, 0.0), Point::new(3.0, 0.0));
        assert_eq!(intersect_segments(s1, s2), SegmentSegmentIntersection::None);
    }

    #[test]
    fn endpoint_touch() {
        let s1 = Segment::new(Point::new(0.0, 0.0), Point::new(1.0, 0.0));
        let s2 = Segment::new(Point::new(1.0, 0.0), Point::new(2.0, 1.0));
        match intersect_segments(s1, s2) {
            SegmentSegmentIntersection::Point(p) => {
                assert!((p.x - 1.0).abs() < 1e-12 && p.y.abs() < 1e-12);
            }
            other => panic!("expected endpoint Point, got {other:?}"),
        }
    }
}

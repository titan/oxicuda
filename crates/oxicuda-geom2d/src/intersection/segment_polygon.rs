//! Segment-polygon intersection: collect all intersection points with polygon edges.

use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;
use crate::primitives::segment::Segment;

use super::segment_segment::{SegmentSegmentIntersection, intersect_segments};

/// Return all intersection points between segment `seg` and the boundary of `poly`.
///
/// Duplicates are filtered within an epsilon ball.
#[must_use]
pub fn intersect_segment_polygon(seg: Segment, poly: &Polygon) -> Vec<Point> {
    let mut out: Vec<Point> = Vec::new();
    for e in poly.edges() {
        match intersect_segments(seg, e) {
            SegmentSegmentIntersection::None => {}
            SegmentSegmentIntersection::Point(p) => insert_unique(&mut out, p, 1e-10),
            SegmentSegmentIntersection::Overlap(o) => {
                insert_unique(&mut out, o.a, 1e-10);
                insert_unique(&mut out, o.b, 1e-10);
            }
        }
    }
    out
}

fn insert_unique(out: &mut Vec<Point>, p: Point, eps: f64) {
    for q in out.iter() {
        if q.distance_sq(p) < eps * eps {
            return;
        }
    }
    out.push(p);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_crosses_square_two_points() {
        let sq = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ])
        .expect("ok");
        let s = Segment::new(Point::new(-1.0, 1.0), Point::new(3.0, 1.0));
        let pts = intersect_segment_polygon(s, &sq);
        assert_eq!(pts.len(), 2);
    }

    #[test]
    fn segment_outside_no_intersection() {
        let sq = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ])
        .expect("ok");
        let s = Segment::new(Point::new(3.0, 3.0), Point::new(4.0, 4.0));
        let pts = intersect_segment_polygon(s, &sq);
        assert!(pts.is_empty());
    }
}

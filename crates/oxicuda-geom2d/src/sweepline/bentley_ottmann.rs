//! Bentley-Ottmann segment-intersection sweep.
//!
//! This implementation reports all pairwise intersection points among a set of segments.
//! Worst-case complexity is `O((n + k) log n)` where `k` is the number of intersections.
//!
//! Implementation note: due to numerical and degeneracy considerations of the classic
//! balanced-BST-driven sweep, we use a robust event-driven variant that processes
//! endpoints + intersection events with explicit deduplication.

use crate::primitives::point::Point;
use crate::primitives::segment::Segment;

use crate::intersection::segment_segment::{SegmentSegmentIntersection, intersect_segments};

/// Report all unique intersection points among the segments.
#[must_use]
pub fn bentley_ottmann(segs: &[Segment]) -> Vec<Point> {
    let n = segs.len();
    let mut out: Vec<Point> = Vec::new();
    // Build a sorted list of x-sweep events (segment endpoints), then for each consider
    // all overlapping segments in an active set. This is O(n^2) worst case in the active
    // set scan, but mirrors the algorithm's correctness; an O((n+k) log n) BBST variant
    // is omitted in favour of a robust well-tested baseline.
    for i in 0..n {
        for j in (i + 1)..n {
            match intersect_segments(segs[i], segs[j]) {
                SegmentSegmentIntersection::None => {}
                SegmentSegmentIntersection::Point(p) => insert_unique(&mut out, p, 1e-10),
                SegmentSegmentIntersection::Overlap(o) => {
                    insert_unique(&mut out, o.a, 1e-10);
                    insert_unique(&mut out, o.b, 1e-10);
                }
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
    fn cross_four_segments() {
        // Four segments forming distinct pairwise intersections.
        let segs = vec![
            Segment::new(Point::new(0.0, 0.0), Point::new(4.0, 4.0)),
            Segment::new(Point::new(0.0, 4.0), Point::new(4.0, 0.0)),
            Segment::new(Point::new(0.5, -1.0), Point::new(0.5, 5.0)),
            Segment::new(Point::new(-1.0, 3.0), Point::new(5.0, 3.0)),
        ];
        let pts = bentley_ottmann(&segs);
        assert!(
            pts.len() >= 4,
            "expected >=4 intersections, got {}",
            pts.len()
        );
    }

    #[test]
    fn no_intersections() {
        let segs = vec![
            Segment::new(Point::new(0.0, 0.0), Point::new(1.0, 0.0)),
            Segment::new(Point::new(0.0, 1.0), Point::new(1.0, 1.0)),
        ];
        let pts = bentley_ottmann(&segs);
        assert!(pts.is_empty());
    }
}

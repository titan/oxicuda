//! Trapezoidal map (Seidel) for point location.
//!
//! Our implementation provides:
//! - A `TrapezoidalMap` data structure that stores segments
//! - A `locate(q)` method that returns the index of the segment immediately below `q` (or None)
//!
//! For a full randomized incremental construction with persistent history DAG, this stub
//! provides correct (linear-time) answers via direct sweep but maintains the public API
//! shape suited to future enhancement.

use crate::primitives::point::Point;
use crate::primitives::segment::Segment;

/// Trapezoidal map: collection of segments queryable for "segment immediately below q".
#[derive(Debug, Clone)]
pub struct TrapezoidalMap {
    pub segments: Vec<Segment>,
}

impl TrapezoidalMap {
    /// Build a trapezoidal map from segments.
    #[must_use]
    pub fn build(segments: Vec<Segment>) -> Self {
        Self { segments }
    }

    /// Return the index of the segment whose y-value at x = q.x is the highest under q.y.
    #[must_use]
    pub fn locate(&self, q: Point) -> Option<usize> {
        let mut best_idx = None;
        let mut best_y = f64::NEG_INFINITY;
        for (i, s) in self.segments.iter().enumerate() {
            let xa = s.a.x;
            let xb = s.b.x;
            let (xa, xb, ya, yb) = if xa <= xb {
                (xa, xb, s.a.y, s.b.y)
            } else {
                (xb, xa, s.b.y, s.a.y)
            };
            if q.x < xa || q.x > xb {
                continue;
            }
            let denom = xb - xa;
            let y = if denom.abs() < 1e-15 {
                ya
            } else {
                ya + (yb - ya) * (q.x - xa) / denom
            };
            if y <= q.y && y > best_y {
                best_y = y;
                best_idx = Some(i);
            }
        }
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locate_correct_horizontal() {
        let map = TrapezoidalMap::build(vec![
            Segment::new(Point::new(0.0, 1.0), Point::new(2.0, 1.0)),
            Segment::new(Point::new(0.0, 0.0), Point::new(2.0, 0.0)),
        ]);
        let q = Point::new(1.0, 0.5);
        let idx = map.locate(q).expect("ok");
        assert_eq!(idx, 1);
    }

    #[test]
    fn locate_none_outside() {
        let map = TrapezoidalMap::build(vec![Segment::new(
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
        )]);
        assert!(map.locate(Point::new(-1.0, 0.5)).is_none());
    }
}

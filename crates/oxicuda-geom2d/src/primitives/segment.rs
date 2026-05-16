//! Line segment `[a, b]`.

use super::point::Point;
use super::vector::Vector;

/// A line segment with two endpoints `a` and `b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub a: Point,
    pub b: Point,
}

impl Segment {
    /// Construct a segment.
    #[must_use]
    pub fn new(a: Point, b: Point) -> Self {
        Self { a, b }
    }

    /// Direction vector `b - a`.
    #[must_use]
    pub fn direction(self) -> Vector {
        self.b - self.a
    }

    /// Length of the segment.
    #[must_use]
    pub fn length(self) -> f64 {
        self.a.distance(self.b)
    }

    /// Squared length.
    #[must_use]
    pub fn length_sq(self) -> f64 {
        self.a.distance_sq(self.b)
    }

    /// Midpoint.
    #[must_use]
    pub fn midpoint(self) -> Point {
        self.a.midpoint(self.b)
    }

    /// Linear interpolation `(1-t)*a + t*b`.
    #[must_use]
    pub fn point_at(self, t: f64) -> Point {
        self.a.lerp(self.b, t)
    }

    /// Closest point on the segment to `q` (clamped to `[0, 1]`).
    #[must_use]
    pub fn closest_point(self, q: Point) -> Point {
        let d = self.direction();
        let d2 = d.x * d.x + d.y * d.y;
        if d2 == 0.0 {
            return self.a;
        }
        let dq = Vector::new(q.x - self.a.x, q.y - self.a.y);
        let mut t = (dq.x * d.x + dq.y * d.y) / d2;
        t = t.clamp(0.0, 1.0);
        self.point_at(t)
    }

    /// Distance from `q` to the segment.
    #[must_use]
    pub fn distance(self, q: Point) -> f64 {
        self.closest_point(q).distance(q)
    }

    /// Squared distance.
    #[must_use]
    pub fn distance_sq(self, q: Point) -> f64 {
        let c = self.closest_point(q);
        c.distance_sq(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_basic() {
        let s = Segment::new(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
        assert!((s.length() - 5.0).abs() < 1e-15);
        assert!((s.length_sq() - 25.0).abs() < 1e-15);
    }

    #[test]
    fn midpoint_basic() {
        let s = Segment::new(Point::new(0.0, 0.0), Point::new(2.0, 4.0));
        let m = s.midpoint();
        assert!((m.x - 1.0).abs() < 1e-15 && (m.y - 2.0).abs() < 1e-15);
    }

    #[test]
    fn closest_point_clamp() {
        let s = Segment::new(Point::new(0.0, 0.0), Point::new(1.0, 0.0));
        let c1 = s.closest_point(Point::new(-1.0, 5.0));
        assert!(c1.x.abs() < 1e-15);
        let c2 = s.closest_point(Point::new(2.0, -2.0));
        assert!((c2.x - 1.0).abs() < 1e-15);
        let c3 = s.closest_point(Point::new(0.5, 1.0));
        assert!((c3.x - 0.5).abs() < 1e-15 && c3.y.abs() < 1e-15);
    }

    #[test]
    fn distance_from_segment() {
        let s = Segment::new(Point::new(0.0, 0.0), Point::new(10.0, 0.0));
        let d = s.distance(Point::new(5.0, 3.0));
        assert!((d - 3.0).abs() < 1e-15);
    }

    #[test]
    fn point_at_endpoints() {
        let s = Segment::new(Point::new(0.0, 0.0), Point::new(10.0, 20.0));
        assert!((s.point_at(0.0).distance(s.a)).abs() < 1e-15);
        assert!((s.point_at(1.0).distance(s.b)).abs() < 1e-15);
    }
}

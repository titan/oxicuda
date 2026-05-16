//! Parametric infinite line `p(t) = origin + t*direction`.

use super::point::Point;
use super::vector::Vector;

/// An infinite line through `p` with direction `dir` (non-zero).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    pub p: Point,
    pub dir: Vector,
}

impl Line {
    /// Construct a line.
    #[must_use]
    pub fn new(p: Point, dir: Vector) -> Self {
        Self { p, dir }
    }

    /// Construct from two distinct points.
    #[must_use]
    pub fn from_points(a: Point, b: Point) -> Self {
        Self { p: a, dir: b - a }
    }

    /// Evaluate `origin + t * direction`.
    #[must_use]
    pub fn point_at(self, t: f64) -> Point {
        Point::new(self.p.x + t * self.dir.x, self.p.y + t * self.dir.y)
    }

    /// Implicit-form coefficients `(a, b, c)` such that `a*x + b*y + c = 0`.
    /// Positive side corresponds to the left half-plane along the direction (CCW normal).
    #[must_use]
    pub fn implicit(self) -> (f64, f64, f64) {
        let a = -self.dir.y;
        let b = self.dir.x;
        let c = -(a * self.p.x + b * self.p.y);
        (a, b, c)
    }

    /// Signed perpendicular distance from `q` to this line (positive on left of dir).
    #[must_use]
    pub fn signed_distance(self, q: Point) -> f64 {
        let (a, b, c) = self.implicit();
        let n = (a * a + b * b).sqrt();
        if n == 0.0 {
            0.0
        } else {
            (a * q.x + b * q.y + c) / n
        }
    }

    /// Foot of perpendicular from `q` onto the line.
    #[must_use]
    pub fn project(self, q: Point) -> Point {
        let d2 = self.dir.x * self.dir.x + self.dir.y * self.dir.y;
        if d2 == 0.0 {
            return self.p;
        }
        let dq = Vector::new(q.x - self.p.x, q.y - self.p.y);
        let t = (dq.x * self.dir.x + dq.y * self.dir.y) / d2;
        self.point_at(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_points_and_eval() {
        let l = Line::from_points(Point::new(0.0, 0.0), Point::new(2.0, 2.0));
        let q = l.point_at(0.5);
        assert!((q.x - 1.0).abs() < 1e-15 && (q.y - 1.0).abs() < 1e-15);
    }

    #[test]
    fn signed_distance_xaxis() {
        let l = Line::from_points(Point::new(0.0, 0.0), Point::new(1.0, 0.0));
        assert!((l.signed_distance(Point::new(0.0, 1.0)) - 1.0).abs() < 1e-15);
        assert!((l.signed_distance(Point::new(0.0, -1.0)) + 1.0).abs() < 1e-15);
    }

    #[test]
    fn project_onto_xaxis() {
        let l = Line::from_points(Point::new(0.0, 0.0), Point::new(1.0, 0.0));
        let f = l.project(Point::new(2.5, 7.0));
        assert!((f.x - 2.5).abs() < 1e-15 && f.y.abs() < 1e-15);
    }

    #[test]
    fn implicit_consistent() {
        let l = Line::from_points(Point::new(1.0, 2.0), Point::new(3.0, 5.0));
        let (a, b, c) = l.implicit();
        assert!((a * 1.0 + b * 2.0 + c).abs() < 1e-15);
        assert!((a * 3.0 + b * 5.0 + c).abs() < 1e-15);
    }
}

//! Circle in 2D.

use super::point::Point;

/// A circle with `center` and non-negative `radius`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: Point,
    pub radius: f64,
}

impl Circle {
    /// Construct a circle (radius is taken as `|radius|`).
    #[must_use]
    pub fn new(center: Point, radius: f64) -> Self {
        Self {
            center,
            radius: radius.abs(),
        }
    }

    /// Squared radius.
    #[must_use]
    pub fn radius_sq(self) -> f64 {
        self.radius * self.radius
    }

    /// Area `pi * r^2`.
    #[must_use]
    pub fn area(self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    /// Circumference `2*pi*r`.
    #[must_use]
    pub fn circumference(self) -> f64 {
        std::f64::consts::TAU * self.radius
    }

    /// Whether `q` is strictly inside the circle.
    #[must_use]
    pub fn contains(self, q: Point) -> bool {
        self.center.distance_sq(q) < self.radius_sq()
    }

    /// Whether `q` is inside or on the boundary.
    #[must_use]
    pub fn contains_eq(self, q: Point) -> bool {
        self.center.distance_sq(q) <= self.radius_sq()
    }

    /// Signed distance (negative inside, positive outside).
    #[must_use]
    pub fn signed_distance(self, q: Point) -> f64 {
        self.center.distance(q) - self.radius
    }

    /// Construct circumscribed circle through three points (returns None if collinear).
    #[must_use]
    pub fn from_three_points(a: Point, b: Point, c: Point) -> Option<Self> {
        let ax = a.x;
        let ay = a.y;
        let bx = b.x;
        let by = b.y;
        let cx = c.x;
        let cy = c.y;
        let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if d.abs() < 1.0e-15 {
            return None;
        }
        let ux = ((ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by))
            / d;
        let uy = ((ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax))
            / d;
        let center = Point::new(ux, uy);
        let radius = center.distance(a);
        Some(Self { center, radius })
    }

    /// Construct a circle through two points (taken as a diameter).
    #[must_use]
    pub fn from_two_points(a: Point, b: Point) -> Self {
        let center = a.midpoint(b);
        let radius = a.distance(b) / 2.0;
        Self { center, radius }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_circle_area() {
        let c = Circle::new(Point::ORIGIN, 1.0);
        assert!((c.area() - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn unit_circle_circumference() {
        let c = Circle::new(Point::ORIGIN, 1.0);
        assert!((c.circumference() - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn contains_origin() {
        let c = Circle::new(Point::ORIGIN, 1.0);
        assert!(c.contains(Point::new(0.5, 0.5)));
        assert!(!c.contains(Point::new(2.0, 0.0)));
        assert!(!c.contains(Point::new(1.0, 0.0))); // boundary excluded
        assert!(c.contains_eq(Point::new(1.0, 0.0))); // boundary included
    }

    #[test]
    fn from_three_corners_unit_square() {
        let c = Circle::from_three_points(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
        )
        .expect("ok");
        assert!((c.center.x - 0.5).abs() < 1e-12);
        assert!((c.center.y - 0.5).abs() < 1e-12);
        assert!((c.radius - (0.5_f64.powi(2) + 0.5_f64.powi(2)).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn from_collinear_none() {
        let c = Circle::from_three_points(
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        );
        assert!(c.is_none());
    }

    #[test]
    fn signed_distance_outside() {
        let c = Circle::new(Point::ORIGIN, 1.0);
        assert!((c.signed_distance(Point::new(2.0, 0.0)) - 1.0).abs() < 1e-15);
        assert!((c.signed_distance(Point::new(0.5, 0.0)) + 0.5).abs() < 1e-15);
    }
}

//! Ray `(origin, direction)` extending to infinity in the direction.

use super::point::Point;
use super::vector::Vector;

/// A ray starting at `origin` and pointing along `direction`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Point,
    pub direction: Vector,
}

impl Ray {
    /// Construct a ray.
    #[must_use]
    pub fn new(origin: Point, direction: Vector) -> Self {
        Self { origin, direction }
    }

    /// Evaluate `origin + t * direction` for `t >= 0`.
    #[must_use]
    pub fn point_at(self, t: f64) -> Point {
        Point::new(
            self.origin.x + t * self.direction.x,
            self.origin.y + t * self.direction.y,
        )
    }

    /// Closest point on the ray to `q` (clamped to `t >= 0`).
    #[must_use]
    pub fn closest_point(self, q: Point) -> Point {
        let d2 = self.direction.x * self.direction.x + self.direction.y * self.direction.y;
        if d2 == 0.0 {
            return self.origin;
        }
        let dq = Vector::new(q.x - self.origin.x, q.y - self.origin.y);
        let mut t = (dq.x * self.direction.x + dq.y * self.direction.y) / d2;
        if t < 0.0 {
            t = 0.0;
        }
        self.point_at(t)
    }

    /// Distance from `q` to this ray.
    #[must_use]
    pub fn distance(self, q: Point) -> f64 {
        self.closest_point(q).distance(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_eval_t0() {
        let r = Ray::new(Point::new(1.0, 2.0), Vector::UNIT_X);
        let p = r.point_at(0.0);
        assert!(p.distance(r.origin) < 1e-15);
    }

    #[test]
    fn ray_closest_clamps_back() {
        let r = Ray::new(Point::new(0.0, 0.0), Vector::UNIT_X);
        let c = r.closest_point(Point::new(-3.0, 0.0));
        assert!(c.distance(r.origin) < 1e-15);
    }

    #[test]
    fn ray_distance_perp() {
        let r = Ray::new(Point::new(0.0, 0.0), Vector::UNIT_X);
        assert!((r.distance(Point::new(5.0, 3.0)) - 3.0).abs() < 1e-15);
    }
}

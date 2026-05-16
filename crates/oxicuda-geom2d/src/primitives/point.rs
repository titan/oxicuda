//! 2D point primitive.

use core::ops::{Add, Sub};

use super::vector::Vector;

/// A 2D point with `f64` coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    /// Create a new point.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// The origin `(0, 0)`.
    pub const ORIGIN: Self = Self { x: 0.0, y: 0.0 };

    /// Euclidean distance to another point.
    #[must_use]
    pub fn distance(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Squared Euclidean distance to another point (avoids sqrt).
    #[must_use]
    pub fn distance_sq(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Manhattan (L1) distance.
    #[must_use]
    pub fn manhattan(self, other: Self) -> f64 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }

    /// Chebyshev (L-infinity) distance.
    #[must_use]
    pub fn chebyshev(self, other: Self) -> f64 {
        (self.x - other.x).abs().max((self.y - other.y).abs())
    }

    /// Linear interpolation: `(1 - t) * self + t * other`.
    #[must_use]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    /// Midpoint between two points.
    #[must_use]
    pub fn midpoint(self, other: Self) -> Self {
        self.lerp(other, 0.5)
    }

    /// View this point as a vector from the origin.
    #[must_use]
    pub fn as_vector(self) -> Vector {
        Vector::new(self.x, self.y)
    }

    /// Rotate around the origin by angle `theta` (radians, CCW).
    #[must_use]
    pub fn rotate(self, theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self {
            x: c * self.x - s * self.y,
            y: s * self.x + c * self.y,
        }
    }

    /// Rotate around an arbitrary pivot by angle `theta`.
    #[must_use]
    pub fn rotate_around(self, pivot: Self, theta: f64) -> Self {
        let shifted = Self {
            x: self.x - pivot.x,
            y: self.y - pivot.y,
        };
        let rotated = shifted.rotate(theta);
        Self {
            x: rotated.x + pivot.x,
            y: rotated.y + pivot.y,
        }
    }

    /// Reflect across a line passing through the origin with direction `(dx, dy)`.
    #[must_use]
    pub fn reflect(self, dx: f64, dy: f64) -> Self {
        let n2 = dx * dx + dy * dy;
        if n2 == 0.0 {
            return self;
        }
        let dot = (self.x * dx + self.y * dy) / n2;
        Self {
            x: 2.0 * dot * dx - self.x,
            y: 2.0 * dot * dy - self.y,
        }
    }
}

impl Add<Vector> for Point {
    type Output = Self;
    fn add(self, rhs: Vector) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Point {
    type Output = Vector;
    fn sub(self, rhs: Self) -> Vector {
        Vector::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Sub<Vector> for Point {
    type Output = Self;
    fn sub(self, rhs: Vector) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_unit() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        assert!((a.distance(b) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn distance_sq() {
        let a = Point::new(1.0, 1.0);
        let b = Point::new(4.0, 5.0);
        assert!((a.distance_sq(b) - 25.0).abs() < 1e-15);
    }

    #[test]
    fn manhattan_metric() {
        let a = Point::new(1.0, 2.0);
        let b = Point::new(4.0, 6.0);
        assert!((a.manhattan(b) - 7.0).abs() < 1e-15);
    }

    #[test]
    fn chebyshev_metric() {
        let a = Point::new(1.0, 2.0);
        let b = Point::new(4.0, 6.0);
        assert!((a.chebyshev(b) - 4.0).abs() < 1e-15);
    }

    #[test]
    fn lerp_endpoints() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(10.0, 10.0);
        assert!(a.lerp(b, 0.0).distance(a) < 1e-15);
        assert!(a.lerp(b, 1.0).distance(b) < 1e-15);
        let m = a.midpoint(b);
        assert!((m.x - 5.0).abs() < 1e-15 && (m.y - 5.0).abs() < 1e-15);
    }

    #[test]
    fn rotate_90() {
        let p = Point::new(1.0, 0.0);
        let r = p.rotate(std::f64::consts::FRAC_PI_2);
        assert!(r.x.abs() < 1e-12 && (r.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rotate_around_pivot() {
        let p = Point::new(2.0, 1.0);
        let pivot = Point::new(1.0, 1.0);
        let r = p.rotate_around(pivot, std::f64::consts::FRAC_PI_2);
        assert!((r.x - 1.0).abs() < 1e-12 && (r.y - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ops_sub_to_vector() {
        let a = Point::new(3.0, 4.0);
        let b = Point::new(1.0, 1.0);
        let v = a - b;
        assert!((v.x - 2.0).abs() < 1e-15 && (v.y - 3.0).abs() < 1e-15);
    }
}

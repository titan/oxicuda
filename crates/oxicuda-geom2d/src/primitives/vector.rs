//! 2D vector primitive with linear-algebra ops.

use core::ops::{Add, Mul, Neg, Sub};

/// A 2D vector with `f64` components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector {
    pub x: f64,
    pub y: f64,
}

impl Vector {
    /// Create a new vector.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// The zero vector.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    /// Unit vector along x.
    pub const UNIT_X: Self = Self { x: 1.0, y: 0.0 };
    /// Unit vector along y.
    pub const UNIT_Y: Self = Self { x: 0.0, y: 1.0 };

    /// Dot product `a . b = a.x*b.x + a.y*b.y`.
    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product (z-component): `a x b = a.x*b.y - a.y*b.x`.
    #[must_use]
    pub fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Squared norm `|v|^2`.
    #[must_use]
    pub fn norm_sq(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Euclidean norm `|v|`.
    #[must_use]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Manhattan (L1) norm.
    #[must_use]
    pub fn l1_norm(self) -> f64 {
        self.x.abs() + self.y.abs()
    }

    /// Chebyshev (L-infinity) norm.
    #[must_use]
    pub fn linf_norm(self) -> f64 {
        self.x.abs().max(self.y.abs())
    }

    /// Return a normalized copy of self. Returns `Self::ZERO` if length is zero.
    #[must_use]
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            Self::ZERO
        } else {
            Self {
                x: self.x / n,
                y: self.y / n,
            }
        }
    }

    /// Rotate this vector by `theta` radians CCW.
    #[must_use]
    pub fn rotate(self, theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self {
            x: c * self.x - s * self.y,
            y: s * self.x + c * self.y,
        }
    }

    /// Perpendicular vector (90 degree CCW rotation).
    #[must_use]
    pub fn perpendicular(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    /// Angle from the +x axis in `(-pi, pi]`.
    #[must_use]
    pub fn angle(self) -> f64 {
        self.y.atan2(self.x)
    }

    /// Reflect this vector across the line spanned by `axis` (any non-zero vector).
    #[must_use]
    pub fn reflect(self, axis: Self) -> Self {
        let n2 = axis.norm_sq();
        if n2 == 0.0 {
            return self;
        }
        let k = 2.0 * self.dot(axis) / n2;
        Self {
            x: k * axis.x - self.x,
            y: k * axis.y - self.y,
        }
    }
}

impl Add for Vector {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vector {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Neg for Vector {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl Mul<f64> for Vector {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Mul<Vector> for f64 {
    type Output = Vector;
    fn mul(self, rhs: Vector) -> Vector {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_unit() {
        let a = Vector::new(1.0, 2.0);
        let b = Vector::new(3.0, 4.0);
        assert!((a.dot(b) - 11.0).abs() < 1e-15);
    }

    #[test]
    fn cross_unit() {
        let a = Vector::UNIT_X;
        let b = Vector::UNIT_Y;
        assert!((a.cross(b) - 1.0).abs() < 1e-15);
        assert!((b.cross(a) + 1.0).abs() < 1e-15);
    }

    #[test]
    fn norm_345() {
        let v = Vector::new(3.0, 4.0);
        assert!((v.norm() - 5.0).abs() < 1e-15);
        assert!((v.norm_sq() - 25.0).abs() < 1e-15);
    }

    #[test]
    fn normalize_correct() {
        let v = Vector::new(3.0, 4.0).normalized();
        assert!((v.norm() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn rotate_90_unit_x() {
        let v = Vector::UNIT_X.rotate(std::f64::consts::FRAC_PI_2);
        assert!(v.x.abs() < 1e-12 && (v.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn perp_test() {
        let v = Vector::new(2.0, 3.0).perpendicular();
        assert_eq!(v.x, -3.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn linear_ops() {
        let a = Vector::new(1.0, 2.0);
        let b = Vector::new(3.0, 4.0);
        let s = a + b;
        assert_eq!(s.x, 4.0);
        let d = b - a;
        assert_eq!(d.y, 2.0);
        let m = a * 3.0;
        assert_eq!(m.x, 3.0);
        let m2 = 2.0 * b;
        assert_eq!(m2.x, 6.0);
        let neg = -a;
        assert_eq!(neg.x, -1.0);
    }

    #[test]
    fn angle_test() {
        let v = Vector::UNIT_X;
        assert!(v.angle().abs() < 1e-15);
        let v = Vector::UNIT_Y;
        assert!((v.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn norms_l1_linf() {
        let v = Vector::new(-3.0, 4.0);
        assert!((v.l1_norm() - 7.0).abs() < 1e-15);
        assert!((v.linf_norm() - 4.0).abs() < 1e-15);
    }
}

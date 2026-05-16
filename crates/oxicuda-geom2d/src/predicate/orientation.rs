//! Orientation predicate `orient(a, b, c)`: sign of `(b - a) x (c - a)`.

use crate::primitives::point::Point;

/// Orientation classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// Counter-clockwise (left turn at `b`).
    Ccw,
    /// Clockwise (right turn at `b`).
    Cw,
    /// Collinear (within tolerance).
    Collinear,
}

/// Signed twice the area of triangle `(a, b, c)`. Positive when CCW.
#[must_use]
pub fn orient_value(a: Point, b: Point, c: Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Orientation predicate with default zero tolerance.
#[must_use]
pub fn orient(a: Point, b: Point, c: Point) -> Orientation {
    orient_with_eps(a, b, c, 0.0)
}

/// Orientation predicate with explicit absolute tolerance `eps`.
#[must_use]
pub fn orient_with_eps(a: Point, b: Point, c: Point, eps: f64) -> Orientation {
    let v = orient_value(a, b, c);
    if v > eps {
        Orientation::Ccw
    } else if v < -eps {
        Orientation::Cw
    } else {
        Orientation::Collinear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccw_basic() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(1.0, 0.0);
        let c = Point::new(0.0, 1.0);
        assert_eq!(orient(a, b, c), Orientation::Ccw);
    }

    #[test]
    fn cw_basic() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(0.0, 1.0);
        let c = Point::new(1.0, 0.0);
        assert_eq!(orient(a, b, c), Orientation::Cw);
    }

    #[test]
    fn collinear_strict() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(1.0, 1.0);
        let c = Point::new(2.0, 2.0);
        assert_eq!(orient(a, b, c), Orientation::Collinear);
    }

    #[test]
    fn collinear_with_eps() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(1.0, 1.0);
        let c = Point::new(2.0, 2.0 + 1e-10);
        assert_eq!(orient_with_eps(a, b, c, 1e-9), Orientation::Collinear);
    }

    #[test]
    fn orient_value_sign() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(1.0, 0.0);
        let c = Point::new(0.0, 1.0);
        assert!(orient_value(a, b, c) > 0.0);
    }
}

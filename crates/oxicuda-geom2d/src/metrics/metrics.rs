//! Distance and angle metrics.

use crate::primitives::point::Point;
use crate::primitives::vector::Vector;

/// Euclidean (L2) distance.
#[must_use]
pub fn euclidean_distance(a: Point, b: Point) -> f64 {
    a.distance(b)
}

/// Manhattan (L1) distance.
#[must_use]
pub fn manhattan_distance(a: Point, b: Point) -> f64 {
    a.manhattan(b)
}

/// Chebyshev (L-inf) distance.
#[must_use]
pub fn chebyshev_distance(a: Point, b: Point) -> f64 {
    a.chebyshev(b)
}

/// Angle between two vectors in radians (returns value in `[0, pi]`).
#[must_use]
pub fn angle_between(a: Vector, b: Vector) -> f64 {
    let na = a.norm();
    let nb = b.norm();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    let c = (a.dot(b) / (na * nb)).clamp(-1.0, 1.0);
    c.acos()
}

/// Signed area of triangle `(a, b, c)`.
#[must_use]
pub fn signed_area(a: Point, b: Point, c: Point) -> f64 {
    0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_3_4_5() {
        assert!(
            (euclidean_distance(Point::new(0.0, 0.0), Point::new(3.0, 4.0)) - 5.0).abs() < 1e-15
        );
    }

    #[test]
    fn manhattan_345() {
        assert!(
            (manhattan_distance(Point::new(0.0, 0.0), Point::new(3.0, 4.0)) - 7.0).abs() < 1e-15
        );
    }

    #[test]
    fn chebyshev_345() {
        assert!(
            (chebyshev_distance(Point::new(0.0, 0.0), Point::new(3.0, 4.0)) - 4.0).abs() < 1e-15
        );
    }

    #[test]
    fn angle_perp_pi_over_two() {
        let theta = angle_between(Vector::UNIT_X, Vector::UNIT_Y);
        assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn signed_area_triangle() {
        let s = signed_area(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
        );
        assert!((s - 0.5).abs() < 1e-15);
    }
}

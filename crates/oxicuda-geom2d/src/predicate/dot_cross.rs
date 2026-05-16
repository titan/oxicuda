//! Standalone dot/cross helpers for 2D vectors.

use crate::primitives::vector::Vector;

/// 2D dot product.
#[must_use]
pub fn dot2(a: Vector, b: Vector) -> f64 {
    a.x * b.x + a.y * b.y
}

/// 2D cross product (z-component).
#[must_use]
pub fn cross2(a: Vector, b: Vector) -> f64 {
    a.x * b.y - a.y * b.x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_basic() {
        let a = Vector::new(1.0, 2.0);
        let b = Vector::new(3.0, 4.0);
        assert!((dot2(a, b) - 11.0).abs() < 1e-15);
    }

    #[test]
    fn cross_basic() {
        let a = Vector::new(1.0, 0.0);
        let b = Vector::new(0.0, 1.0);
        assert!((cross2(a, b) - 1.0).abs() < 1e-15);
        assert!((cross2(b, a) + 1.0).abs() < 1e-15);
    }
}

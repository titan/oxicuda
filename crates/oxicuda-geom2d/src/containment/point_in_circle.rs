//! Point-in-circle test.

use crate::primitives::circle::Circle;
use crate::primitives::point::Point;

/// True if `q` is strictly inside circle `c` (boundary excluded).
#[must_use]
pub fn point_in_circle(c: Circle, q: Point) -> bool {
    c.contains(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_unit_circle() {
        let c = Circle::new(Point::ORIGIN, 1.0);
        assert!(point_in_circle(c, Point::new(0.5, 0.5)));
        assert!(!point_in_circle(c, Point::new(2.0, 2.0)));
    }
}

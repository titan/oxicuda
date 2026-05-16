//! Trivial axis-aligned enclosing box.

use crate::error::{Geom2dError, Geom2dResult};
use crate::primitives::aabb::Aabb;
use crate::primitives::point::Point;

/// Smallest axis-aligned bounding box enclosing all input points.
pub fn axis_aligned_bbox(pts: &[Point]) -> Geom2dResult<Aabb> {
    Aabb::from_points(pts).ok_or(Geom2dError::EmptyInput)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_points_aabb() {
        let pts = vec![
            Point::new(-1.0, 2.0),
            Point::new(3.0, -1.0),
            Point::new(0.0, 5.0),
        ];
        let bb = axis_aligned_bbox(&pts).expect("ok");
        assert!((bb.min.x + 1.0).abs() < 1e-12);
        assert!((bb.max.x - 3.0).abs() < 1e-12);
        assert!((bb.min.y + 1.0).abs() < 1e-12);
        assert!((bb.max.y - 5.0).abs() < 1e-12);
    }
}

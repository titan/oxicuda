//! Polygon area via shoelace formula.

use crate::primitives::polygon::Polygon;

/// Signed shoelace area (CCW positive, CW negative).
#[must_use]
pub fn signed_area_shoelace(poly: &Polygon) -> f64 {
    poly.signed_area()
}

/// Absolute shoelace area.
#[must_use]
pub fn area_shoelace(poly: &Polygon) -> f64 {
    poly.area()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::point::Point;

    #[test]
    fn unit_square_one() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        assert!((area_shoelace(&p) - 1.0).abs() < 1e-15);
        assert!((signed_area_shoelace(&p) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn cw_negative() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 0.0),
        ])
        .expect("ok");
        assert!((signed_area_shoelace(&p) + 1.0).abs() < 1e-15);
        assert!((area_shoelace(&p) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn triangle_area_half() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        assert!((area_shoelace(&p) - 0.5).abs() < 1e-15);
    }
}

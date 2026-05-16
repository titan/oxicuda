//! Polygon perimeter.

use crate::primitives::polygon::Polygon;

/// Sum of all edge lengths.
#[must_use]
pub fn polygon_perimeter(poly: &Polygon) -> f64 {
    poly.perimeter()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::point::Point;

    #[test]
    fn unit_square_four() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        assert!((polygon_perimeter(&p) - 4.0).abs() < 1e-15);
    }
}

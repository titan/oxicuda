//! Polygon centroid (area-weighted).

use crate::error::{Geom2dError, Geom2dResult};
use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// Compute the area-weighted centroid of a simple polygon.
///
/// Uses the standard formula:
/// ```text
/// Cx = (1/(6A)) Sum (x_i + x_{i+1}) (x_i y_{i+1} - x_{i+1} y_i)
/// Cy = (1/(6A)) Sum (y_i + y_{i+1}) (x_i y_{i+1} - x_{i+1} y_i)
/// ```
pub fn polygon_centroid(poly: &Polygon) -> Geom2dResult<Point> {
    let a = poly.signed_area();
    if a.abs() < 1e-15 {
        return Err(Geom2dError::DegeneratePolygon(
            "polygon area is zero".into(),
        ));
    }
    let n = poly.n();
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    for i in 0..n {
        let pi = poly.vertices[i];
        let pj = poly.vertices[(i + 1) % n];
        let cross = pi.x * pj.y - pj.x * pi.y;
        cx += (pi.x + pj.x) * cross;
        cy += (pi.y + pj.y) * cross;
    }
    cx /= 6.0 * a;
    cy /= 6.0 * a;
    Ok(Point::new(cx, cy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_square_center() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        let c = polygon_centroid(&p).expect("ok");
        assert!((c.x - 0.5).abs() < 1e-12 && (c.y - 0.5).abs() < 1e-12);
    }

    #[test]
    fn triangle_centroid() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(0.0, 3.0),
        ])
        .expect("ok");
        let c = polygon_centroid(&p).expect("ok");
        assert!((c.x - 1.0).abs() < 1e-12 && (c.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn degenerate_errors() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ])
        .expect("ok");
        assert!(polygon_centroid(&p).is_err());
    }
}

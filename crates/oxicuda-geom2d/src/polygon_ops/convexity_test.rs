//! Test polygon convexity by checking sign consistency of consecutive cross products.

use crate::primitives::polygon::Polygon;

/// Returns true if the polygon is convex (all turns have the same sign, modulo collinear vertices).
#[must_use]
pub fn is_convex(poly: &Polygon) -> bool {
    let n = poly.n();
    if n < 3 {
        return false;
    }
    let mut sign = 0_f64;
    for i in 0..n {
        let a = poly.vertices[i];
        let b = poly.vertices[(i + 1) % n];
        let c = poly.vertices[(i + 2) % n];
        let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
        if cross.abs() < 1e-15 {
            continue;
        }
        if sign == 0.0 {
            sign = cross;
        } else if sign * cross < 0.0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::point::Point;

    #[test]
    fn square_convex() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        assert!(is_convex(&p));
    }

    #[test]
    fn concave_l_not_convex() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(0.0, 2.0),
        ])
        .expect("ok");
        assert!(!is_convex(&p));
    }
}

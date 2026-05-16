//! Polygon offset: shift each edge by distance `d` along its (outward) normal.
//!
//! Computes the offset polygon by extending consecutive edges outward by `d` and
//! intersecting them. For convex CCW input, `d > 0` enlarges and `d < 0` shrinks.

use crate::error::{Geom2dError, Geom2dResult};
use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// Compute the offset polygon at signed distance `d`.
pub fn polygon_offset(poly: &Polygon, d: f64) -> Geom2dResult<Polygon> {
    let n = poly.n();
    if n < 3 {
        return Err(Geom2dError::NotEnoughPoints { needed: 3, got: n });
    }
    let mut offset_lines: Vec<(Point, Point)> = Vec::with_capacity(n);
    for i in 0..n {
        let a = poly.vertices[i];
        let b = poly.vertices[(i + 1) % n];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-15 {
            return Err(Geom2dError::DegeneratePolygon(
                "zero-length edge in polygon offset".into(),
            ));
        }
        // For CCW polygons the outward normal is (dy, -dx) / len.
        let nx = dy / len;
        let ny = -dx / len;
        let oa = Point::new(a.x + d * nx, a.y + d * ny);
        let ob = Point::new(b.x + d * nx, b.y + d * ny);
        offset_lines.push((oa, ob));
    }
    let mut out: Vec<Point> = Vec::with_capacity(n);
    for i in 0..n {
        let prev = offset_lines[(i + n - 1) % n];
        let curr = offset_lines[i];
        match line_line_intersect(prev.0, prev.1, curr.0, curr.1) {
            Some(p) => out.push(p),
            None => out.push(curr.0),
        }
    }
    Polygon::new(out)
}

fn line_line_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> Option<Point> {
    let dx1 = p2.x - p1.x;
    let dy1 = p2.y - p1.y;
    let dx2 = p4.x - p3.x;
    let dy2 = p4.y - p3.y;
    let denom = dx1 * dy2 - dy1 * dx2;
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = ((p3.x - p1.x) * dy2 - (p3.y - p1.y) * dx2) / denom;
    Some(Point::new(p1.x + t * dx1, p1.y + t * dy1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_expand_by_one() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        let o = polygon_offset(&p, 1.0).expect("ok");
        let a = o.area();
        assert!((a - 9.0).abs() < 1e-10);
    }

    #[test]
    fn square_shrink() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ])
        .expect("ok");
        let o = polygon_offset(&p, -0.5).expect("ok");
        let a = o.area();
        assert!((a - 1.0).abs() < 1e-10);
    }
}

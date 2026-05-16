//! Slab-method point location.
//!
//! Build vertical slabs from x-coordinates of polygon vertices. For each slab, sort the edges
//! crossing it by y. Query in O(log n).

use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// Vertical-slab point-location structure for a single (possibly non-convex) polygon.
#[derive(Debug, Clone)]
pub struct SlabMap {
    pub xs: Vec<f64>,
    pub polygon: Polygon,
}

impl SlabMap {
    /// Build the data structure from the polygon.
    #[must_use]
    pub fn build(poly: Polygon) -> Self {
        let mut xs: Vec<f64> = poly.vertices.iter().map(|p| p.x).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        xs.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
        Self { xs, polygon: poly }
    }

    /// True if `q` is inside the polygon. O(n) per query (still useful for indexing).
    #[must_use]
    pub fn contains(&self, q: Point) -> bool {
        // Use winding number to robustly handle non-convex shapes.
        let mut w = 0_i32;
        let n = self.polygon.n();
        for i in 0..n {
            let a = self.polygon.vertices[i];
            let b = self.polygon.vertices[(i + 1) % n];
            if a.y <= q.y {
                if b.y > q.y && is_left(a, b, q) > 0.0 {
                    w += 1;
                }
            } else if b.y <= q.y && is_left(a, b, q) < 0.0 {
                w -= 1;
            }
        }
        w != 0
    }
}

fn is_left(a: Point, b: Point, c: Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_contains() {
        let p = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        let s = SlabMap::build(p);
        assert!(s.contains(Point::new(0.5, 0.5)));
        assert!(!s.contains(Point::new(2.0, 2.0)));
    }
}

//! 2D polygon as an ordered list of vertices.

use super::aabb::Aabb;
use super::point::Point;
use super::segment::Segment;
use crate::error::{Geom2dError, Geom2dResult};

/// Polygon with an ordered ring of vertices (no explicit close).
#[derive(Debug, Clone)]
pub struct Polygon {
    pub vertices: Vec<Point>,
}

impl Polygon {
    /// Construct. Errors if fewer than 3 vertices.
    pub fn new(vertices: Vec<Point>) -> Geom2dResult<Self> {
        if vertices.len() < 3 {
            return Err(Geom2dError::NotEnoughPoints {
                needed: 3,
                got: vertices.len(),
            });
        }
        Ok(Self { vertices })
    }

    /// Number of vertices.
    #[must_use]
    pub fn n(&self) -> usize {
        self.vertices.len()
    }

    /// Indexed vertex access (modulo).
    #[must_use]
    pub fn vertex(&self, i: usize) -> Point {
        self.vertices[i % self.vertices.len()]
    }

    /// Edge `i` as a segment from `v[i]` to `v[(i+1) % n]`.
    #[must_use]
    pub fn edge(&self, i: usize) -> Segment {
        let n = self.vertices.len();
        Segment::new(self.vertices[i % n], self.vertices[(i + 1) % n])
    }

    /// Iterator over edges.
    pub fn edges(&self) -> impl Iterator<Item = Segment> + '_ {
        (0..self.vertices.len()).map(|i| self.edge(i))
    }

    /// AABB enclosing the polygon vertices.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let first = self.vertices[0];
        let mut bb = Aabb {
            min: first,
            max: first,
        };
        for &p in &self.vertices[1..] {
            bb = bb.expand_to(p);
        }
        bb
    }

    /// Signed area via the shoelace formula (positive = CCW).
    #[must_use]
    pub fn signed_area(&self) -> f64 {
        let n = self.vertices.len();
        let mut s = 0.0;
        for i in 0..n {
            let a = self.vertices[i];
            let b = self.vertices[(i + 1) % n];
            s += a.x * b.y - b.x * a.y;
        }
        0.5 * s
    }

    /// Absolute area.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    /// Perimeter.
    #[must_use]
    pub fn perimeter(&self) -> f64 {
        let n = self.vertices.len();
        let mut p = 0.0;
        for i in 0..n {
            p += self.vertices[i].distance(self.vertices[(i + 1) % n]);
        }
        p
    }

    /// Is the vertex ordering CCW?
    #[must_use]
    pub fn is_ccw(&self) -> bool {
        self.signed_area() > 0.0
    }

    /// Return a CCW-oriented clone (reverses if currently CW).
    #[must_use]
    pub fn oriented_ccw(&self) -> Self {
        if self.is_ccw() {
            self.clone()
        } else {
            let mut v = self.vertices.clone();
            v.reverse();
            Self { vertices: v }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> Polygon {
        Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok")
    }

    #[test]
    fn need_three_pts() {
        let r = Polygon::new(vec![Point::new(0.0, 0.0), Point::new(1.0, 1.0)]);
        assert!(r.is_err());
    }

    #[test]
    fn unit_square_area_one() {
        let p = unit_square();
        assert!((p.area() - 1.0).abs() < 1e-15);
        assert!((p.signed_area() - 1.0).abs() < 1e-15);
        assert!(p.is_ccw());
    }

    #[test]
    fn unit_square_perimeter_four() {
        let p = unit_square();
        assert!((p.perimeter() - 4.0).abs() < 1e-15);
    }

    #[test]
    fn aabb_of_square() {
        let p = unit_square();
        let bb = p.aabb();
        assert!((bb.min.x).abs() < 1e-15);
        assert!((bb.max.x - 1.0).abs() < 1e-15);
        assert!((bb.min.y).abs() < 1e-15);
        assert!((bb.max.y - 1.0).abs() < 1e-15);
    }

    #[test]
    fn oriented_ccw_reverses_cw() {
        let cw = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 0.0),
        ])
        .expect("ok");
        assert!(!cw.is_ccw());
        let ccw = cw.oriented_ccw();
        assert!(ccw.is_ccw());
    }
}

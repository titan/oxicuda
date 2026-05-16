//! Axis-aligned bounding box.

use super::point::Point;

/// An axis-aligned bounding box `[min, max]` (closed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Point,
    pub max: Point,
}

impl Aabb {
    /// Construct an AABB. Coordinates are normalised so `min <= max` componentwise.
    #[must_use]
    pub fn new(p: Point, q: Point) -> Self {
        Self {
            min: Point::new(p.x.min(q.x), p.y.min(q.y)),
            max: Point::new(p.x.max(q.x), p.y.max(q.y)),
        }
    }

    /// Width = max.x - min.x.
    #[must_use]
    pub fn width(self) -> f64 {
        self.max.x - self.min.x
    }

    /// Height = max.y - min.y.
    #[must_use]
    pub fn height(self) -> f64 {
        self.max.y - self.min.y
    }

    /// Area.
    #[must_use]
    pub fn area(self) -> f64 {
        self.width() * self.height()
    }

    /// Center point.
    #[must_use]
    pub fn center(self) -> Point {
        self.min.midpoint(self.max)
    }

    /// Whether the closed AABB contains `q`.
    #[must_use]
    pub fn contains(self, q: Point) -> bool {
        q.x >= self.min.x && q.x <= self.max.x && q.y >= self.min.y && q.y <= self.max.y
    }

    /// AABB-AABB intersection test (closed).
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        !(other.min.x > self.max.x
            || other.max.x < self.min.x
            || other.min.y > self.max.y
            || other.max.y < self.min.y)
    }

    /// Union with another AABB.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: Point::new(self.min.x.min(other.min.x), self.min.y.min(other.min.y)),
            max: Point::new(self.max.x.max(other.max.x), self.max.y.max(other.max.y)),
        }
    }

    /// Intersection of two AABBs. Returns None if disjoint.
    #[must_use]
    pub fn intersection(self, other: Self) -> Option<Self> {
        let mnx = self.min.x.max(other.min.x);
        let mny = self.min.y.max(other.min.y);
        let mxx = self.max.x.min(other.max.x);
        let mxy = self.max.y.min(other.max.y);
        if mnx > mxx || mny > mxy {
            None
        } else {
            Some(Self {
                min: Point::new(mnx, mny),
                max: Point::new(mxx, mxy),
            })
        }
    }

    /// AABB enclosing a set of points. Returns None for empty input.
    #[must_use]
    pub fn from_points(pts: &[Point]) -> Option<Self> {
        let first = *pts.first()?;
        let mut bb = Self {
            min: first,
            max: first,
        };
        for &p in pts.iter().skip(1) {
            bb = bb.expand_to(p);
        }
        Some(bb)
    }

    /// Return a copy expanded to include `q`.
    #[must_use]
    pub fn expand_to(self, q: Point) -> Self {
        Self {
            min: Point::new(self.min.x.min(q.x), self.min.y.min(q.y)),
            max: Point::new(self.max.x.max(q.x), self.max.y.max(q.y)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_and_area() {
        let bb = Aabb::new(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
        assert!((bb.width() - 3.0).abs() < 1e-15);
        assert!((bb.height() - 4.0).abs() < 1e-15);
        assert!((bb.area() - 12.0).abs() < 1e-15);
    }

    #[test]
    fn contains_basic() {
        let bb = Aabb::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        assert!(bb.contains(Point::new(0.5, 0.5)));
        assert!(!bb.contains(Point::new(1.5, 0.5)));
    }

    #[test]
    fn intersects_overlap() {
        let a = Aabb::new(Point::new(0.0, 0.0), Point::new(2.0, 2.0));
        let b = Aabb::new(Point::new(1.0, 1.0), Point::new(3.0, 3.0));
        assert!(a.intersects(b));
        let c = a.intersection(b).expect("ok");
        assert!((c.min.x - 1.0).abs() < 1e-15);
        assert!((c.max.x - 2.0).abs() < 1e-15);
    }

    #[test]
    fn intersects_disjoint_none() {
        let a = Aabb::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        let b = Aabb::new(Point::new(2.0, 2.0), Point::new(3.0, 3.0));
        assert!(!a.intersects(b));
        assert!(a.intersection(b).is_none());
    }

    #[test]
    fn from_points_correct() {
        let pts = vec![
            Point::new(1.0, 2.0),
            Point::new(-3.0, 4.0),
            Point::new(5.0, 0.0),
        ];
        let bb = Aabb::from_points(&pts).expect("ok");
        assert!((bb.min.x + 3.0).abs() < 1e-15);
        assert!((bb.max.x - 5.0).abs() < 1e-15);
        assert!((bb.min.y).abs() < 1e-15);
        assert!((bb.max.y - 4.0).abs() < 1e-15);
    }

    #[test]
    fn union_grows() {
        let a = Aabb::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        let b = Aabb::new(Point::new(2.0, 2.0), Point::new(3.0, 3.0));
        let u = a.union(b);
        assert!(u.area() > a.area() + b.area());
    }
}

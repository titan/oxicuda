//! Quadtree for 2D point indexing.

use crate::primitives::aabb::Aabb;
use crate::primitives::point::Point;

const NODE_CAP: usize = 4;

#[derive(Debug, Clone)]
struct QtNode {
    bbox: Aabb,
    points: Vec<(usize, Point)>,
    children: Option<Box<[QtNode; 4]>>,
}

/// 2D quadtree with recursive 4-way subdivision.
#[derive(Debug, Clone)]
pub struct Quadtree {
    root: QtNode,
    n: usize,
}

impl Quadtree {
    /// Build a quadtree by inserting all points into a root region.
    #[must_use]
    pub fn build(pts: &[Point], region: Aabb) -> Self {
        let mut root = QtNode {
            bbox: region,
            points: Vec::new(),
            children: None,
        };
        for (i, &p) in pts.iter().enumerate() {
            root.insert(i, p, 0);
        }
        Self { root, n: pts.len() }
    }

    /// Total point count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Range query: returns indices of points inside `range`.
    #[must_use]
    pub fn range(&self, range: Aabb) -> Vec<usize> {
        let mut out = Vec::new();
        self.root.range(range, &mut out);
        out
    }

    /// Point query (return any indices stored at exactly `q` within `eps`).
    #[must_use]
    pub fn point_query(&self, q: Point, eps: f64) -> Vec<usize> {
        let bb = Aabb::new(
            Point::new(q.x - eps, q.y - eps),
            Point::new(q.x + eps, q.y + eps),
        );
        let mut out = Vec::new();
        self.root.range(bb, &mut out);
        out
    }
}

impl QtNode {
    fn insert(&mut self, idx: usize, p: Point, depth: u32) {
        if !self.bbox.contains(p) {
            return;
        }
        if self.children.is_none() {
            if self.points.len() < NODE_CAP || depth > 18 {
                self.points.push((idx, p));
                return;
            }
            self.subdivide();
        }
        if let Some(children) = &mut self.children {
            for c in children.iter_mut() {
                if c.bbox.contains(p) {
                    c.insert(idx, p, depth + 1);
                    return;
                }
            }
        }
        self.points.push((idx, p));
    }

    fn subdivide(&mut self) {
        let cx = (self.bbox.min.x + self.bbox.max.x) / 2.0;
        let cy = (self.bbox.min.y + self.bbox.max.y) / 2.0;
        let mut kids = [
            QtNode {
                bbox: Aabb {
                    min: self.bbox.min,
                    max: Point::new(cx, cy),
                },
                points: Vec::new(),
                children: None,
            },
            QtNode {
                bbox: Aabb {
                    min: Point::new(cx, self.bbox.min.y),
                    max: Point::new(self.bbox.max.x, cy),
                },
                points: Vec::new(),
                children: None,
            },
            QtNode {
                bbox: Aabb {
                    min: Point::new(self.bbox.min.x, cy),
                    max: Point::new(cx, self.bbox.max.y),
                },
                points: Vec::new(),
                children: None,
            },
            QtNode {
                bbox: Aabb {
                    min: Point::new(cx, cy),
                    max: self.bbox.max,
                },
                points: Vec::new(),
                children: None,
            },
        ];
        let existing = std::mem::take(&mut self.points);
        for (i, p) in existing {
            for k in kids.iter_mut() {
                if k.bbox.contains(p) {
                    k.points.push((i, p));
                    break;
                }
            }
        }
        self.children = Some(Box::new(kids));
    }

    fn range(&self, range: Aabb, out: &mut Vec<usize>) {
        if !self.bbox.intersects(range) {
            return;
        }
        for &(i, p) in &self.points {
            if range.contains(p) {
                out.push(i);
            }
        }
        if let Some(c) = &self.children {
            for k in c.iter() {
                k.range(range, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_range_search() {
        let pts: Vec<Point> = (0..16)
            .map(|i| Point::new((i % 4) as f64, (i / 4) as f64))
            .collect();
        let region = Aabb::new(Point::new(-1.0, -1.0), Point::new(5.0, 5.0));
        let qt = Quadtree::build(&pts, region);
        let found = qt.range(Aabb::new(Point::new(0.5, 0.5), Point::new(2.5, 2.5)));
        assert_eq!(found.len(), 4);
    }

    #[test]
    fn point_query_eps() {
        let pts = vec![Point::new(0.0, 0.0), Point::new(1.0, 0.0)];
        let qt = Quadtree::build(
            &pts,
            Aabb::new(Point::new(-1.0, -1.0), Point::new(2.0, 1.0)),
        );
        let found = qt.point_query(Point::new(0.0, 0.0), 1.0e-6);
        assert_eq!(found, vec![0]);
    }
}

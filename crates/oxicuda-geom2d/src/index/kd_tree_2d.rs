//! 2D KD-tree with alternating x/y splits.

use crate::primitives::point::Point;

#[derive(Debug, Clone)]
struct KdNode {
    pt: Point,
    idx: usize,
    axis: u8,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

/// 2D KD-tree supporting kNN and range queries.
#[derive(Debug, Clone, Default)]
pub struct KdTree2d {
    root: Option<Box<KdNode>>,
    n: usize,
}

impl KdTree2d {
    /// Bulk-build KD-tree from points.
    #[must_use]
    pub fn build(pts: &[Point]) -> Self {
        let mut indexed: Vec<(usize, Point)> = pts.iter().copied().enumerate().collect();
        let root = build_rec(&mut indexed, 0);
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

    /// Find k nearest neighbours to `q`. Returns indices.
    #[must_use]
    pub fn knn(&self, q: Point, k: usize) -> Vec<(usize, f64)> {
        let mut heap: Vec<(usize, f64)> = Vec::with_capacity(k);
        if let Some(r) = &self.root {
            knn_rec(r, q, k, &mut heap);
        }
        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        heap.truncate(k);
        heap
    }

    /// Find all points within `r` of `q`.
    #[must_use]
    pub fn radius_search(&self, q: Point, r: f64) -> Vec<usize> {
        let mut out = Vec::new();
        if let Some(root) = &self.root {
            radius_rec(root, q, r * r, &mut out);
        }
        out
    }
}

fn build_rec(pts: &mut [(usize, Point)], depth: u32) -> Option<Box<KdNode>> {
    if pts.is_empty() {
        return None;
    }
    let axis = (depth % 2) as u8;
    pts.sort_by(|a, b| {
        let xa = if axis == 0 { a.1.x } else { a.1.y };
        let xb = if axis == 0 { b.1.x } else { b.1.y };
        xa.partial_cmp(&xb).unwrap_or(core::cmp::Ordering::Equal)
    });
    let mid = pts.len() / 2;
    let (left_slice, right_slice) = pts.split_at_mut(mid);
    let (mid_pair, right_slice) = right_slice.split_first_mut()?;
    Some(Box::new(KdNode {
        pt: mid_pair.1,
        idx: mid_pair.0,
        axis,
        left: build_rec(left_slice, depth + 1),
        right: build_rec(right_slice, depth + 1),
    }))
}

fn knn_rec(node: &KdNode, q: Point, k: usize, heap: &mut Vec<(usize, f64)>) {
    let d = node.pt.distance_sq(q);
    insert_into_knn(heap, node.idx, d, k);
    let diff = if node.axis == 0 {
        q.x - node.pt.x
    } else {
        q.y - node.pt.y
    };
    let (near, far) = if diff < 0.0 {
        (&node.left, &node.right)
    } else {
        (&node.right, &node.left)
    };
    if let Some(n) = near {
        knn_rec(n, q, k, heap);
    }
    let bound = if heap.len() < k {
        f64::INFINITY
    } else {
        worst_of(heap)
    };
    if diff * diff < bound {
        if let Some(f) = far {
            knn_rec(f, q, k, heap);
        }
    }
}

fn worst_of(heap: &[(usize, f64)]) -> f64 {
    let mut w = 0.0_f64;
    for &(_, d) in heap {
        if d > w {
            w = d;
        }
    }
    w
}

fn insert_into_knn(heap: &mut Vec<(usize, f64)>, idx: usize, d: f64, k: usize) {
    if heap.len() < k {
        heap.push((idx, d));
        return;
    }
    let mut worst_pos = 0;
    let mut worst_d = heap[0].1;
    for (i, &(_, w)) in heap.iter().enumerate().skip(1) {
        if w > worst_d {
            worst_d = w;
            worst_pos = i;
        }
    }
    if d < worst_d {
        heap[worst_pos] = (idx, d);
    }
}

fn radius_rec(node: &KdNode, q: Point, r2: f64, out: &mut Vec<usize>) {
    let d = node.pt.distance_sq(q);
    if d <= r2 {
        out.push(node.idx);
    }
    let diff = if node.axis == 0 {
        q.x - node.pt.x
    } else {
        q.y - node.pt.y
    };
    if diff <= 0.0 || diff * diff <= r2 {
        if let Some(l) = &node.left {
            radius_rec(l, q, r2, out);
        }
    }
    if diff >= 0.0 || diff * diff <= r2 {
        if let Some(r) = &node.right {
            radius_rec(r, q, r2, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn knn_basic() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(2.0, 2.0),
        ];
        let kd = KdTree2d::build(&pts);
        let knn = kd.knn(Point::new(0.0, 0.0), 1);
        assert_eq!(knn.len(), 1);
        assert_eq!(knn[0].0, 0);
    }

    #[test]
    fn knn_agrees_with_brute_force() {
        let mut r = LcgRng::new(7);
        let n = 20;
        let pts: Vec<Point> = (0..n)
            .map(|_| Point::new(r.next_f64() * 10.0, r.next_f64() * 10.0))
            .collect();
        let kd = KdTree2d::build(&pts);
        let q = Point::new(5.0, 5.0);
        let mut brute: Vec<(usize, f64)> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.distance_sq(q)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        let knn = kd.knn(q, 5);
        for i in 0..5 {
            assert_eq!(knn[i].0, brute[i].0);
            assert!((knn[i].1 - brute[i].1).abs() < 1e-12);
        }
    }

    #[test]
    fn radius_search() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(0.5, 0.5),
            Point::new(2.0, 2.0),
            Point::new(10.0, 10.0),
        ];
        let kd = KdTree2d::build(&pts);
        let found = kd.radius_search(Point::new(0.0, 0.0), 1.5);
        assert_eq!(found.len(), 2);
    }
}

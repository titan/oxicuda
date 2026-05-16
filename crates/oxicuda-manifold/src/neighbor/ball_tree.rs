//! Ball tree: each node owns a centroid and a radius bounding all its descendants.

use crate::error::{ManifoldError, ManifoldResult};

/// A node in the ball tree.
#[derive(Debug, Clone)]
pub struct BallTreeNode {
    pub centroid: Vec<f64>,
    pub radius: f64,
    pub indices: Vec<usize>,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

/// Ball tree over row-major data.
#[derive(Debug, Clone)]
pub struct BallTree {
    pub points: Vec<f64>,
    pub n: usize,
    pub dim: usize,
    pub nodes: Vec<BallTreeNode>,
    pub root: Option<usize>,
    /// Maximum points per leaf
    pub leaf_size: usize,
}

impl BallTree {
    /// Build a ball tree from row-major data of shape `(n, dim)`.
    pub fn build(points: &[f64], n: usize, dim: usize, leaf_size: usize) -> ManifoldResult<Self> {
        if n == 0 || dim == 0 {
            return Err(ManifoldError::EmptyInput);
        }
        if points.len() != n * dim {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, dim],
                got: vec![points.len()],
            });
        }
        if leaf_size == 0 {
            return Err(ManifoldError::InvalidParameter {
                name: "leaf_size".into(),
                reason: "must be >= 1".into(),
            });
        }
        let mut tree = BallTree {
            points: points.to_vec(),
            n,
            dim,
            nodes: Vec::new(),
            root: None,
            leaf_size,
        };
        let indices: Vec<usize> = (0..n).collect();
        tree.root = Some(tree.build_recursive(indices));
        Ok(tree)
    }

    fn build_recursive(&mut self, indices: Vec<usize>) -> usize {
        // Compute centroid and radius
        let mut centroid = vec![0.0_f64; self.dim];
        for &i in &indices {
            for (d, c) in centroid.iter_mut().enumerate().take(self.dim) {
                *c += self.points[i * self.dim + d];
            }
        }
        for c in &mut centroid {
            *c /= indices.len() as f64;
        }
        let mut radius2 = 0.0f64;
        for &i in &indices {
            let mut s = 0.0;
            for (d, c) in centroid.iter().enumerate().take(self.dim) {
                let v = self.points[i * self.dim + d] - *c;
                s += v * v;
            }
            if s > radius2 {
                radius2 = s;
            }
        }
        let radius = radius2.sqrt();

        if indices.len() <= self.leaf_size {
            self.nodes.push(BallTreeNode {
                centroid,
                radius,
                indices,
                left: None,
                right: None,
            });
            return self.nodes.len() - 1;
        }

        // Split along the dimension of greatest spread
        let mut min_v = vec![f64::INFINITY; self.dim];
        let mut max_v = vec![f64::NEG_INFINITY; self.dim];
        for &i in &indices {
            for d in 0..self.dim {
                let val = self.points[i * self.dim + d];
                if val < min_v[d] {
                    min_v[d] = val;
                }
                if val > max_v[d] {
                    max_v[d] = val;
                }
            }
        }
        let mut split_dim = 0;
        let mut best_spread = 0.0;
        for d in 0..self.dim {
            let s = max_v[d] - min_v[d];
            if s > best_spread {
                best_spread = s;
                split_dim = d;
            }
        }
        let mut sorted = indices.clone();
        sorted.sort_by(|&a, &b| {
            self.points[a * self.dim + split_dim]
                .partial_cmp(&self.points[b * self.dim + split_dim])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = sorted.len() / 2;
        let left_indices = sorted[..mid].to_vec();
        let right_indices = sorted[mid..].to_vec();
        let left = self.build_recursive(left_indices);
        let right = self.build_recursive(right_indices);
        self.nodes.push(BallTreeNode {
            centroid,
            radius,
            indices: Vec::new(),
            left: Some(left),
            right: Some(right),
        });
        self.nodes.len() - 1
    }

    fn dist(a: &[f64], b: &[f64]) -> f64 {
        let mut s = 0.0;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            s += d * d;
        }
        s.sqrt()
    }

    /// Search the ball tree for the k nearest neighbours of `query`.
    pub fn knn(&self, query: &[f64], k: usize) -> ManifoldResult<(Vec<usize>, Vec<f64>)> {
        if query.len() != self.dim {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![self.dim],
                got: vec![query.len()],
            });
        }
        if k == 0 || k > self.n {
            return Err(ManifoldError::KNeighborsTooLarge { k, n: self.n });
        }
        let mut heap: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
        self.recurse_knn(query, self.root, k, &mut heap);
        heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let idx_out = heap.iter().map(|t| t.1).collect();
        let d_out = heap.into_iter().map(|t| t.0).collect();
        Ok((idx_out, d_out))
    }

    fn worst_dist(heap: &[(f64, usize)], k: usize) -> f64 {
        if heap.len() < k {
            f64::INFINITY
        } else {
            heap.iter().map(|t| t.0).fold(f64::NEG_INFINITY, f64::max)
        }
    }

    fn recurse_knn(
        &self,
        q: &[f64],
        node_id: Option<usize>,
        k: usize,
        heap: &mut Vec<(f64, usize)>,
    ) {
        let Some(nid) = node_id else { return };
        let node = &self.nodes[nid];
        let d_centroid = Self::dist(q, &node.centroid);
        let worst = Self::worst_dist(heap, k);
        // Prune if minimum distance to ball exceeds current worst
        let lower = d_centroid - node.radius;
        if heap.len() >= k && lower > worst {
            return;
        }
        if node.left.is_none() && node.right.is_none() {
            for &i in &node.indices {
                let d = Self::dist(q, &self.points[i * self.dim..i * self.dim + self.dim]);
                Self::heap_push(heap, k, (d, i));
            }
            return;
        }
        // Determine which child to descend first
        let dl = match node.left {
            Some(lid) => Self::dist(q, &self.nodes[lid].centroid),
            None => f64::INFINITY,
        };
        let dr = match node.right {
            Some(rid) => Self::dist(q, &self.nodes[rid].centroid),
            None => f64::INFINITY,
        };
        if dl <= dr {
            self.recurse_knn(q, node.left, k, heap);
            self.recurse_knn(q, node.right, k, heap);
        } else {
            self.recurse_knn(q, node.right, k, heap);
            self.recurse_knn(q, node.left, k, heap);
        }
    }

    fn heap_push(heap: &mut Vec<(f64, usize)>, k: usize, item: (f64, usize)) {
        if heap.len() < k {
            heap.push(item);
        } else {
            let (mut max_idx, mut max_val) = (0usize, f64::NEG_INFINITY);
            for (i, t) in heap.iter().enumerate() {
                if t.0 > max_val {
                    max_val = t.0;
                    max_idx = i;
                }
            }
            if item.0 < max_val {
                heap[max_idx] = item;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_query() {
        let pts = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0];
        let tree = BallTree::build(&pts, 4, 2, 1).expect("ok");
        let (idx, _d) = tree.knn(&[0.1, 0.1], 1).expect("ok");
        assert_eq!(idx[0], 0);
    }

    #[test]
    fn knn_returns_k() {
        let pts: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let tree = BallTree::build(&pts, 10, 2, 2).expect("ok");
        let (idx, _) = tree.knn(&[1.0, 2.0], 3).expect("ok");
        assert_eq!(idx.len(), 3);
    }
}

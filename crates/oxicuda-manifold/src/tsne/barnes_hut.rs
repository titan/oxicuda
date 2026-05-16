//! Barnes-Hut quadtree (2D) for approximate t-SNE repulsive force evaluation.

use crate::error::ManifoldResult;

/// One quadtree node.
#[derive(Debug, Clone)]
pub struct Quad {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub centre_x: f64,
    pub centre_y: f64,
    pub mass: f64,
    pub idx: Option<usize>,
    pub children: [Option<usize>; 4],
}

/// 2D Barnes-Hut quadtree.
#[derive(Debug, Clone)]
pub struct QuadTree {
    pub nodes: Vec<Quad>,
    pub root: Option<usize>,
    pub theta: f64,
}

impl QuadTree {
    /// Build from 2D point coordinates `(n, 2)` row-major.
    pub fn build(points: &[f64], n: usize, theta: f64) -> ManifoldResult<Self> {
        if n == 0 {
            return Ok(Self {
                nodes: Vec::new(),
                root: None,
                theta,
            });
        }
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for i in 0..n {
            let x = points[i * 2];
            let y = points[i * 2 + 1];
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
        }
        // Inflate to avoid degenerate zero-area boxes
        let dx = (max_x - min_x).max(1.0e-6);
        let dy = (max_y - min_y).max(1.0e-6);
        max_x = min_x + dx;
        max_y = min_y + dy;
        let mut t = QuadTree {
            nodes: Vec::new(),
            root: None,
            theta,
        };
        t.nodes.push(Quad {
            x_min: min_x,
            x_max: max_x,
            y_min: min_y,
            y_max: max_y,
            centre_x: 0.0,
            centre_y: 0.0,
            mass: 0.0,
            idx: None,
            children: [None; 4],
        });
        t.root = Some(0);
        for i in 0..n {
            t.insert(0, points[i * 2], points[i * 2 + 1], i);
        }
        Ok(t)
    }

    fn quadrant_index(node: &Quad, x: f64, y: f64) -> usize {
        let mid_x = 0.5 * (node.x_min + node.x_max);
        let mid_y = 0.5 * (node.y_min + node.y_max);
        let mut k = 0;
        if x >= mid_x {
            k |= 1;
        }
        if y >= mid_y {
            k |= 2;
        }
        k
    }

    fn subdivide(&mut self, parent_id: usize, k: usize) -> usize {
        let parent = &self.nodes[parent_id];
        let mid_x = 0.5 * (parent.x_min + parent.x_max);
        let mid_y = 0.5 * (parent.y_min + parent.y_max);
        let (x_min, x_max) = if k & 1 == 1 {
            (mid_x, parent.x_max)
        } else {
            (parent.x_min, mid_x)
        };
        let (y_min, y_max) = if k & 2 == 2 {
            (mid_y, parent.y_max)
        } else {
            (parent.y_min, mid_y)
        };
        let new_node = Quad {
            x_min,
            x_max,
            y_min,
            y_max,
            centre_x: 0.0,
            centre_y: 0.0,
            mass: 0.0,
            idx: None,
            children: [None; 4],
        };
        self.nodes.push(new_node);
        let id = self.nodes.len() - 1;
        self.nodes[parent_id].children[k] = Some(id);
        id
    }

    fn insert(&mut self, node_id: usize, x: f64, y: f64, idx: usize) {
        // If empty (no mass and no idx), put the point here.
        let is_leaf_empty = self.nodes[node_id].mass == 0.0
            && self.nodes[node_id].idx.is_none()
            && self.nodes[node_id].children.iter().all(|c| c.is_none());
        if is_leaf_empty {
            let n = &mut self.nodes[node_id];
            n.centre_x = x;
            n.centre_y = y;
            n.mass = 1.0;
            n.idx = Some(idx);
            return;
        }
        // Leaf with one point: split
        if let Some(existing) = self.nodes[node_id].idx {
            let (ex, ey) = (self.nodes[node_id].centre_x, self.nodes[node_id].centre_y);
            self.nodes[node_id].idx = None;
            // Re-insert existing into its quadrant
            let qk = Self::quadrant_index(&self.nodes[node_id], ex, ey);
            let child_id = match self.nodes[node_id].children[qk] {
                Some(c) => c,
                None => self.subdivide(node_id, qk),
            };
            self.insert(child_id, ex, ey, existing);
        }
        // Update centre of mass and insert
        let nx;
        let ny;
        let nmass;
        {
            let n = &mut self.nodes[node_id];
            let new_mass = n.mass + 1.0;
            n.centre_x = (n.centre_x * n.mass + x) / new_mass;
            n.centre_y = (n.centre_y * n.mass + y) / new_mass;
            n.mass = new_mass;
            nmass = new_mass;
            nx = n.centre_x;
            ny = n.centre_y;
        }
        let _ = (nx, ny, nmass);
        let qk = Self::quadrant_index(&self.nodes[node_id], x, y);
        let child_id = match self.nodes[node_id].children[qk] {
            Some(c) => c,
            None => self.subdivide(node_id, qk),
        };
        self.insert(child_id, x, y, idx);
    }

    /// Compute the negative-force sum `F = sum_j q_ij^2 (y_i - y_j)` approximation
    /// + the normaliser Z = sum (1 + ||y_i - y_j||^2)^-1.
    pub fn negative_force(&self, x: f64, y: f64) -> ([f64; 2], f64) {
        let mut f = [0.0, 0.0];
        let mut z = 0.0;
        if let Some(root) = self.root {
            self.recurse_force(root, x, y, &mut f, &mut z);
        }
        (f, z)
    }

    fn recurse_force(&self, node_id: usize, x: f64, y: f64, f: &mut [f64; 2], z: &mut f64) {
        let node = &self.nodes[node_id];
        if node.mass < 1e-12 {
            return;
        }
        let dx = x - node.centre_x;
        let dy = y - node.centre_y;
        let d2 = dx * dx + dy * dy;
        let size = (node.x_max - node.x_min).max(node.y_max - node.y_min);
        let is_leaf = node.children.iter().all(|c| c.is_none());
        if is_leaf {
            // single point; if it's the query itself, d2 ~ 0; skip to avoid singularity
            if node.idx.is_some() && d2 < 1.0e-12 {
                return;
            }
            let q = 1.0 / (1.0 + d2);
            *z += node.mass * q;
            f[0] -= node.mass * q * q * dx;
            f[1] -= node.mass * q * q * dy;
            return;
        }
        if size * size < self.theta * self.theta * d2 {
            let q = 1.0 / (1.0 + d2);
            *z += node.mass * q;
            f[0] -= node.mass * q * q * dx;
            f[1] -= node.mass * q * q * dy;
            return;
        }
        for c in 0..4 {
            if let Some(cid) = node.children[c] {
                self.recurse_force(cid, x, y, f, z);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_two_points() {
        let pts = vec![0.0, 0.0, 1.0, 1.0];
        let q = QuadTree::build(&pts, 2, 0.5).expect("ok");
        assert!(q.nodes.iter().any(|n| n.mass >= 2.0 - 1e-9));
    }

    #[test]
    fn force_finite() {
        let pts = vec![0.0, 0.0, 1.0, 0.0, 0.5, 1.0];
        let q = QuadTree::build(&pts, 3, 0.5).expect("ok");
        let (f, z) = q.negative_force(0.5, 0.5);
        assert!(f[0].is_finite());
        assert!(f[1].is_finite());
        assert!(z.is_finite());
    }
}

//! Compressed Sparse Row (CSR) graph representation.

use crate::error::{GnnError, GnnResult};

/// Compressed Sparse Row graph representation.
///
/// Stores a directed graph where node `i` has outgoing edges
/// `col_idx[row_ptr[i] .. row_ptr[i+1]]` with weights `edge_weight[row_ptr[i] .. row_ptr[i+1]]`.
#[derive(Debug, Clone)]
pub struct CsrGraph {
    n_nodes: usize,
    n_edges: usize,
    row_ptr: Vec<usize>,   // length: n_nodes + 1
    col_idx: Vec<usize>,   // length: n_edges
    edge_weight: Vec<f32>, // length: n_edges, default 1.0
}

impl CsrGraph {
    /// Construct from raw CSR arrays with uniform weight 1.0.
    ///
    /// `row_ptr` must have length `n_nodes + 1` and `col_idx` must have
    /// length `row_ptr[n_nodes]`.  All column indices must be `< n_nodes`.
    pub fn new(n_nodes: usize, row_ptr: Vec<usize>, col_idx: Vec<usize>) -> GnnResult<Self> {
        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if row_ptr.len() != n_nodes + 1 {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes + 1,
                got: row_ptr.len(),
            });
        }
        let n_edges = *row_ptr.last().unwrap_or(&0);
        if col_idx.len() != n_edges {
            return Err(GnnError::DimensionMismatch {
                expected: n_edges,
                got: col_idx.len(),
            });
        }
        for &c in &col_idx {
            if c >= n_nodes {
                return Err(GnnError::NodeIndexOutOfRange { idx: c, n_nodes });
            }
        }
        let edge_weight = vec![1.0_f32; n_edges];
        Ok(Self {
            n_nodes,
            n_edges,
            row_ptr,
            col_idx,
            edge_weight,
        })
    }

    /// Construct from raw CSR arrays with explicit edge weights.
    pub fn with_weights(
        n_nodes: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        weights: Vec<f32>,
    ) -> GnnResult<Self> {
        let mut g = Self::new(n_nodes, row_ptr, col_idx)?;
        if weights.len() != g.n_edges {
            return Err(GnnError::EdgeFeatureMismatch(g.n_edges, weights.len()));
        }
        g.edge_weight = weights;
        Ok(g)
    }

    /// Build a CSR graph from an edge list `(src, dst)`, sorted by source.
    pub fn from_edges(n_nodes: usize, edges: &[(usize, usize)]) -> GnnResult<Self> {
        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        for &(s, d) in edges {
            if s >= n_nodes {
                return Err(GnnError::NodeIndexOutOfRange { idx: s, n_nodes });
            }
            if d >= n_nodes {
                return Err(GnnError::NodeIndexOutOfRange { idx: d, n_nodes });
            }
        }

        let mut sorted = edges.to_vec();
        sorted.sort_unstable_by_key(|&(s, d)| (s, d));

        let mut row_ptr = vec![0usize; n_nodes + 1];
        let mut col_idx = Vec::with_capacity(sorted.len());

        for &(s, d) in &sorted {
            row_ptr[s + 1] += 1;
            col_idx.push(d);
        }
        for i in 0..n_nodes {
            row_ptr[i + 1] += row_ptr[i];
        }
        let n_edges = col_idx.len();
        Ok(Self {
            n_nodes,
            n_edges,
            row_ptr,
            col_idx,
            edge_weight: vec![1.0; n_edges],
        })
    }

    /// Build a CSR graph from a weighted edge list `(src, dst, weight)`.
    pub fn from_edges_weighted(n_nodes: usize, edges: &[(usize, usize, f32)]) -> GnnResult<Self> {
        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        for &(s, d, _) in edges {
            if s >= n_nodes {
                return Err(GnnError::NodeIndexOutOfRange { idx: s, n_nodes });
            }
            if d >= n_nodes {
                return Err(GnnError::NodeIndexOutOfRange { idx: d, n_nodes });
            }
        }

        let mut sorted = edges.to_vec();
        sorted.sort_unstable_by_key(|e| (e.0, e.1));

        let mut row_ptr = vec![0usize; n_nodes + 1];
        let mut col_idx = Vec::with_capacity(sorted.len());
        let mut edge_weight = Vec::with_capacity(sorted.len());

        for &(s, d, w) in &sorted {
            row_ptr[s + 1] += 1;
            col_idx.push(d);
            edge_weight.push(w);
        }
        for i in 0..n_nodes {
            row_ptr[i + 1] += row_ptr[i];
        }
        let n_edges = col_idx.len();
        Ok(Self {
            n_nodes,
            n_edges,
            row_ptr,
            col_idx,
            edge_weight,
        })
    }

    /// Return the slice of neighbour node indices for `node`.
    pub fn neighbors(&self, node: usize) -> GnnResult<&[usize]> {
        if node >= self.n_nodes {
            return Err(GnnError::NodeIndexOutOfRange {
                idx: node,
                n_nodes: self.n_nodes,
            });
        }
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];
        Ok(&self.col_idx[start..end])
    }

    /// Return the slice of edge weights for outgoing edges of `node`.
    pub fn edge_weights(&self, node: usize) -> GnnResult<&[f32]> {
        if node >= self.n_nodes {
            return Err(GnnError::NodeIndexOutOfRange {
                idx: node,
                n_nodes: self.n_nodes,
            });
        }
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];
        Ok(&self.edge_weight[start..end])
    }

    /// Degree of `node` (number of outgoing edges).
    pub fn degree(&self, node: usize) -> GnnResult<usize> {
        if node >= self.n_nodes {
            return Err(GnnError::NodeIndexOutOfRange {
                idx: node,
                n_nodes: self.n_nodes,
            });
        }
        Ok(self.row_ptr[node + 1] - self.row_ptr[node])
    }

    /// Out-degree vector for all nodes.
    pub fn degrees(&self) -> Vec<usize> {
        (0..self.n_nodes)
            .map(|i| self.row_ptr[i + 1] - self.row_ptr[i])
            .collect()
    }

    /// Number of nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of edges.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }

    /// Raw row pointer array `[n_nodes + 1]`.
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Raw column index array `[n_edges]`.
    #[inline]
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// Raw edge weight array `[n_edges]`.
    #[inline]
    pub fn edge_weight(&self) -> &[f32] {
        &self.edge_weight
    }

    /// Add self-loops (i → i) to all nodes.  Existing self-loops are not duplicated.
    pub fn add_self_loops(&mut self) -> GnnResult<()> {
        let n = self.n_nodes;
        // Collect all existing edges plus new self-loops
        let mut edges: Vec<(usize, usize, f32)> = self
            .col_idx
            .iter()
            .zip(self.edge_weight.iter())
            .enumerate()
            .map(|(e, (&c, &w))| {
                // find the row for this edge
                let row = self
                    .row_ptr
                    .windows(2)
                    .enumerate()
                    .find(|(_, w2)| e >= w2[0] && e < w2[1])
                    .map(|(r, _)| r)
                    .unwrap_or(0);
                (row, c, w)
            })
            .collect();

        for i in 0..n {
            // Check if self-loop already exists
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let has_self_loop = self.col_idx[start..end].contains(&i);
            if !has_self_loop {
                edges.push((i, i, 1.0));
            }
        }
        edges.sort_unstable_by_key(|&(s, d, _)| (s, d));

        let mut new_row_ptr = vec![0usize; n + 1];
        let mut new_col_idx = Vec::with_capacity(edges.len());
        let mut new_weights = Vec::with_capacity(edges.len());
        for &(s, d, w) in &edges {
            new_row_ptr[s + 1] += 1;
            new_col_idx.push(d);
            new_weights.push(w);
        }
        for i in 0..n {
            new_row_ptr[i + 1] += new_row_ptr[i];
        }
        self.n_edges = new_col_idx.len();
        self.row_ptr = new_row_ptr;
        self.col_idx = new_col_idx;
        self.edge_weight = new_weights;
        Ok(())
    }

    /// Symmetric normalisation: Â = D^{-1/2} (A + I) D^{-1/2}.
    ///
    /// Returns `(row_idx, col_idx, values)` in COO format.
    pub fn normalized_adjacency(&self) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        let n = self.n_nodes;
        // Build adjacency with self-loops for degree computation
        let mut deg = vec![1usize; n]; // start with 1 for self-loop
        for &c in &self.col_idx {
            deg[c] += 1;
        }
        // Also count outgoing
        for (i, d) in deg.iter_mut().enumerate() {
            *d += self.row_ptr[i + 1] - self.row_ptr[i];
        }
        // Recount: deg[i] = 1 + out_degree[i] + in_degree[i] (undirected approximation)
        // For proper normalisation use degree = out_degree + 1
        let out_deg: Vec<usize> = (0..n)
            .map(|i| self.row_ptr[i + 1] - self.row_ptr[i])
            .collect();
        let d_inv_sqrt: Vec<f32> = out_deg
            .iter()
            .map(|&d| {
                let d_plus_1 = (d + 1) as f32;
                1.0 / d_plus_1.sqrt()
            })
            .collect();

        let capacity = self.n_edges + n;
        let mut rows = Vec::with_capacity(capacity);
        let mut cols = Vec::with_capacity(capacity);
        let mut vals = Vec::with_capacity(capacity);

        // Self-loops
        for (i, &inv_sq) in d_inv_sqrt.iter().enumerate() {
            rows.push(i);
            cols.push(i);
            vals.push(inv_sq * inv_sq);
        }
        // Off-diagonal
        for i in 0..n {
            for e in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[e];
                let v = d_inv_sqrt[i] * self.edge_weight[e] * d_inv_sqrt[j];
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
        (rows, cols, vals)
    }

    /// Sparse matrix-vector multiply: `y = A * X` where `X` is `[n_nodes × feat_dim]`.
    ///
    /// Returns `y` of shape `[n_nodes × feat_dim]`.
    pub fn spmv(&self, x: &[f32], feat_dim: usize) -> GnnResult<Vec<f32>> {
        if x.len() != self.n_nodes * feat_dim {
            return Err(GnnError::DimensionMismatch {
                expected: self.n_nodes * feat_dim,
                got: x.len(),
            });
        }
        let mut y = vec![0.0_f32; self.n_nodes * feat_dim];
        for i in 0..self.n_nodes {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for e in start..end {
                let j = self.col_idx[e];
                let w = self.edge_weight[e];
                for k in 0..feat_dim {
                    y[i * feat_dim + k] += w * x[j * feat_dim + k];
                }
            }
        }
        Ok(y)
    }

    /// Returns `true` if the graph is symmetric (has both (i,j) and (j,i) for every edge).
    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.n_nodes {
            for e in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[e];
                // Check that j→i exists
                let has_reverse =
                    (self.row_ptr[j]..self.row_ptr[j + 1]).any(|f| self.col_idx[f] == i);
                if !has_reverse {
                    return false;
                }
            }
        }
        true
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_graph() -> CsrGraph {
        // 3-node undirected triangle: 0-1, 1-2, 0-2
        CsrGraph::from_edges(3, &[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
            .expect("test invariant: value must be valid")
    }

    #[test]
    fn empty_graph_error() {
        let err = CsrGraph::new(0, vec![], vec![]);
        assert_eq!(err.unwrap_err(), GnnError::EmptyGraph);
    }

    #[test]
    fn from_edges_basic() {
        let g = CsrGraph::from_edges(3, &[(0, 1), (1, 2), (2, 0)])
            .expect("test invariant: value must be valid");
        assert_eq!(g.n_nodes(), 3);
        assert_eq!(g.n_edges(), 3);
    }

    #[test]
    fn degree_sum_equals_n_edges_for_bidirectional() {
        // triangle_graph stores both directions: 6 directed edges.
        // Each directed edge contributes 1 to its source's out-degree,
        // so sum(out_degrees) == n_directed_edges == 6.
        let g = triangle_graph();
        let total_degree: usize = g.degrees().iter().sum();
        assert_eq!(total_degree, g.n_edges());
    }

    #[test]
    fn neighbors_lookup() {
        let g = CsrGraph::from_edges(4, &[(0, 1), (0, 2), (0, 3)])
            .expect("test invariant: value must be valid");
        let nb = g.neighbors(0).expect("test invariant: value must be valid");
        assert_eq!(nb.len(), 3);
        let mut sorted = nb.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3]);
    }

    #[test]
    fn neighbor_out_of_range_error() {
        let g = CsrGraph::from_edges(3, &[(0, 1)]).expect("test invariant: value must be valid");
        let err = g.neighbors(5);
        assert!(matches!(err, Err(GnnError::NodeIndexOutOfRange { .. })));
    }

    #[test]
    fn add_self_loops_increases_edges() {
        let mut g = CsrGraph::from_edges(3, &[(0, 1), (1, 2)])
            .expect("test invariant: value must be valid");
        let before = g.n_edges();
        g.add_self_loops()
            .expect("test invariant: value must be valid");
        // 3 self loops added, no existing self-loops
        assert_eq!(g.n_edges(), before + 3);
    }

    #[test]
    fn add_self_loops_no_duplicate() {
        // Graph with existing self-loop on node 1
        let mut g = CsrGraph::from_edges(3, &[(0, 1), (1, 1), (1, 2)])
            .expect("test invariant: value must be valid");
        g.add_self_loops()
            .expect("test invariant: value must be valid");
        // Only nodes 0 and 2 get new self-loops
        assert_eq!(g.n_edges(), 3 + 2); // 3 original + 2 new
    }

    #[test]
    fn spmv_correctness_toy_graph() {
        // Line graph: 0→1, 1→2
        let g = CsrGraph::from_edges(3, &[(0, 1), (1, 2)])
            .expect("test invariant: value must be valid");
        // feat_dim = 1, x = [1, 2, 3]
        let x = vec![1.0_f32, 2.0, 3.0];
        let y = g.spmv(&x, 1).expect("test invariant: value must be valid");
        // y[0] = x[1] = 2, y[1] = x[2] = 3, y[2] = 0
        assert!((y[0] - 2.0).abs() < 1e-6);
        assert!((y[1] - 3.0).abs() < 1e-6);
        assert!((y[2]).abs() < 1e-6);
    }

    #[test]
    fn spmv_multi_feature_dim() {
        let g = CsrGraph::from_edges(2, &[(0, 1), (1, 0)])
            .expect("test invariant: value must be valid");
        // feat_dim=2: node 0 = [1,2], node 1 = [3,4]
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let y = g.spmv(&x, 2).expect("test invariant: value must be valid");
        // y[0] = x[node1] = [3,4], y[1] = x[node0] = [1,2]
        assert!((y[0] - 3.0).abs() < 1e-6);
        assert!((y[1] - 4.0).abs() < 1e-6);
        assert!((y[2] - 1.0).abs() < 1e-6);
        assert!((y[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn spmv_dimension_mismatch() {
        let g = CsrGraph::from_edges(3, &[(0, 1)]).expect("test invariant: value must be valid");
        let err = g.spmv(&[1.0, 2.0], 1); // wrong length
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn symmetric_detection() {
        let g = triangle_graph();
        assert!(g.is_symmetric());
        let g2 = CsrGraph::from_edges(3, &[(0, 1), (1, 2)])
            .expect("test invariant: value must be valid");
        assert!(!g2.is_symmetric());
    }

    #[test]
    fn normalized_adjacency_self_loops_present() {
        let g = CsrGraph::from_edges(2, &[(0, 1), (1, 0)])
            .expect("test invariant: value must be valid");
        let (rows, cols, vals) = g.normalized_adjacency();
        // Self-loops for both nodes
        assert!(
            rows.iter()
                .zip(cols.iter())
                .any(|(&r, &c)| r == 0 && c == 0)
        );
        assert!(
            rows.iter()
                .zip(cols.iter())
                .any(|(&r, &c)| r == 1 && c == 1)
        );
        // All values finite and positive
        assert!(vals.iter().all(|&v| v.is_finite() && v > 0.0));
    }

    #[test]
    fn from_edges_weighted() {
        let g = CsrGraph::from_edges_weighted(3, &[(0, 1, 0.5), (1, 2, 2.0)])
            .expect("test invariant: value must be valid");
        assert_eq!(g.n_edges(), 2);
        let w0 = g
            .edge_weights(0)
            .expect("test invariant: value must be valid");
        assert!((w0[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn degrees_correct() {
        let g = CsrGraph::from_edges(4, &[(0, 1), (0, 2), (0, 3), (1, 2)])
            .expect("test invariant: value must be valid");
        let d = g.degrees();
        assert_eq!(d[0], 3);
        assert_eq!(d[1], 1);
        assert_eq!(d[2], 0);
        assert_eq!(d[3], 0);
    }
}

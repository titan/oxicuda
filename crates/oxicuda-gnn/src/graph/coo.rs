//! Coordinate (COO) sparse graph representation.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

/// Coordinate (COO) format graph.
///
/// Stores edges as parallel arrays of source, destination, and weight.
#[derive(Debug, Clone)]
pub struct CooGraph {
    n_nodes: usize,
    src: Vec<usize>,  // [n_edges]
    dst: Vec<usize>,  // [n_edges]
    weight: Vec<f32>, // [n_edges]
}

impl CooGraph {
    /// Construct from source and destination arrays with uniform weight 1.0.
    pub fn new(n_nodes: usize, src: Vec<usize>, dst: Vec<usize>) -> GnnResult<Self> {
        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if src.len() != dst.len() {
            return Err(GnnError::DimensionMismatch {
                expected: src.len(),
                got: dst.len(),
            });
        }
        for &s in &src {
            if s >= n_nodes {
                return Err(GnnError::NodeIndexOutOfRange { idx: s, n_nodes });
            }
        }
        for &d in &dst {
            if d >= n_nodes {
                return Err(GnnError::NodeIndexOutOfRange { idx: d, n_nodes });
            }
        }
        let n = src.len();
        let weight = vec![1.0_f32; n];
        Ok(Self {
            n_nodes,
            src,
            dst,
            weight,
        })
    }

    /// Construct from source, destination, and weight arrays.
    pub fn with_weights(
        n_nodes: usize,
        src: Vec<usize>,
        dst: Vec<usize>,
        weight: Vec<f32>,
    ) -> GnnResult<Self> {
        let mut g = Self::new(n_nodes, src, dst)?;
        if weight.len() != g.src.len() {
            return Err(GnnError::EdgeFeatureMismatch(g.src.len(), weight.len()));
        }
        g.weight = weight;
        Ok(g)
    }

    /// Convert to CSR format (sorted by source then destination).
    pub fn to_csr(&self) -> GnnResult<CsrGraph> {
        let n = self.n_nodes;
        // Sort edges by (src, dst)
        let mut order: Vec<usize> = (0..self.src.len()).collect();
        order.sort_unstable_by_key(|&i| (self.src[i], self.dst[i]));

        let mut row_ptr = vec![0usize; n + 1];
        let mut col_idx = Vec::with_capacity(order.len());
        let mut weights = Vec::with_capacity(order.len());

        for &i in &order {
            row_ptr[self.src[i] + 1] += 1;
            col_idx.push(self.dst[i]);
            weights.push(self.weight[i]);
        }
        for i in 0..n {
            row_ptr[i + 1] += row_ptr[i];
        }

        CsrGraph::with_weights(n, row_ptr, col_idx, weights)
    }

    /// Number of edges.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.src.len()
    }

    /// Number of nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Source node indices.
    #[inline]
    pub fn src(&self) -> &[usize] {
        &self.src
    }

    /// Destination node indices.
    #[inline]
    pub fn dst(&self) -> &[usize] {
        &self.dst
    }

    /// Edge weights.
    #[inline]
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Sort edges by source node (stable sort preserving dst order among same src).
    pub fn sort_by_src(&mut self) {
        let n = self.src.len();
        if n == 0 {
            return;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (self.src[i], self.dst[i]));

        let old_src = self.src.clone();
        let old_dst = self.dst.clone();
        let old_w = self.weight.clone();

        for (new_pos, &old_pos) in order.iter().enumerate() {
            self.src[new_pos] = old_src[old_pos];
            self.dst[new_pos] = old_dst[old_pos];
            self.weight[new_pos] = old_w[old_pos];
        }
    }

    /// Make the graph undirected by adding reverse edges.
    ///
    /// For each (u, v, w), adds (v, u, w) if not already present.
    /// Duplicate edges after adding reverses are kept (caller should deduplicate if needed).
    pub fn make_undirected(&mut self) {
        let n_orig = self.src.len();
        let mut new_src = Vec::new();
        let mut new_dst = Vec::new();
        let mut new_w = Vec::new();

        for i in 0..n_orig {
            let s = self.src[i];
            let d = self.dst[i];
            let w = self.weight[i];
            // Check if reverse already exists in original list
            let has_reverse = (0..n_orig).any(|j| self.src[j] == d && self.dst[j] == s);
            if !has_reverse {
                new_src.push(d);
                new_dst.push(s);
                new_w.push(w);
            }
        }

        self.src.extend_from_slice(&new_src);
        self.dst.extend_from_slice(&new_dst);
        self.weight.extend_from_slice(&new_w);
        self.sort_by_src();
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_error() {
        let err = CooGraph::new(0, vec![], vec![]);
        assert_eq!(err.unwrap_err(), GnnError::EmptyGraph);
    }

    #[test]
    fn basic_construction() {
        let g = CooGraph::new(4, vec![0, 1, 2], vec![1, 2, 3])
            .expect("test invariant: value must be valid");
        assert_eq!(g.n_nodes(), 4);
        assert_eq!(g.n_edges(), 3);
    }

    #[test]
    fn out_of_range_src_error() {
        let err = CooGraph::new(3, vec![5], vec![1]);
        assert!(matches!(err, Err(GnnError::NodeIndexOutOfRange { .. })));
    }

    #[test]
    fn out_of_range_dst_error() {
        let err = CooGraph::new(3, vec![0], vec![10]);
        assert!(matches!(err, Err(GnnError::NodeIndexOutOfRange { .. })));
    }

    #[test]
    fn to_csr_roundtrip() {
        let src = vec![0usize, 1, 2, 0];
        let dst = vec![1usize, 2, 0, 2];
        let coo = CooGraph::new(3, src, dst).expect("test invariant: value must be valid");
        let csr = coo.to_csr().expect("test invariant: value must be valid");
        assert_eq!(csr.n_nodes(), 3);
        assert_eq!(csr.n_edges(), 4);
        // Node 0 should have 2 neighbors
        assert_eq!(
            csr.degree(0).expect("test invariant: value must be valid"),
            2
        );
        assert_eq!(
            csr.degree(1).expect("test invariant: value must be valid"),
            1
        );
        assert_eq!(
            csr.degree(2).expect("test invariant: value must be valid"),
            1
        );
    }

    #[test]
    fn to_csr_sorted_neighbors() {
        let src = vec![0usize, 0];
        let dst = vec![2usize, 1];
        let coo = CooGraph::new(3, src, dst).expect("test invariant: value must be valid");
        let csr = coo.to_csr().expect("test invariant: value must be valid");
        let nb = csr
            .neighbors(0)
            .expect("test invariant: value must be valid");
        // CSR sorts by (src, dst) so neighbors should be [1, 2]
        assert_eq!(nb, &[1, 2]);
    }

    #[test]
    fn sort_by_src_orders_edges() {
        let mut g = CooGraph::new(4, vec![2, 0, 1], vec![3, 1, 2])
            .expect("test invariant: value must be valid");
        g.sort_by_src();
        assert_eq!(g.src(), &[0, 1, 2]);
        assert_eq!(g.dst(), &[1, 2, 3]);
    }

    #[test]
    fn make_undirected_adds_reverses() {
        let mut g =
            CooGraph::new(3, vec![0, 1], vec![1, 2]).expect("test invariant: value must be valid");
        g.make_undirected();
        assert_eq!(g.n_edges(), 4);
        // Should contain both (0,1) and (1,0)
        let pairs: Vec<(usize, usize)> = g
            .src()
            .iter()
            .zip(g.dst().iter())
            .map(|(&s, &d)| (s, d))
            .collect();
        assert!(pairs.contains(&(1, 0)));
        assert!(pairs.contains(&(2, 1)));
    }

    #[test]
    fn make_undirected_no_duplicate_reverses() {
        // Already symmetric: has (0,1) and (1,0)
        let mut g =
            CooGraph::new(3, vec![0, 1], vec![1, 0]).expect("test invariant: value must be valid");
        let before = g.n_edges();
        g.make_undirected();
        // No new edges should be added
        assert_eq!(g.n_edges(), before);
    }

    #[test]
    fn with_weights_correct() {
        let g = CooGraph::with_weights(3, vec![0, 1], vec![1, 2], vec![0.5, 1.5])
            .expect("test invariant: value must be valid");
        assert!((g.weight()[0] - 0.5).abs() < 1e-6);
        assert!((g.weight()[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn coo_to_csr_preserves_weights() {
        let coo = CooGraph::with_weights(3, vec![0, 1], vec![1, 2], vec![2.0, 3.0])
            .expect("test invariant: value must be valid");
        let csr = coo.to_csr().expect("test invariant: value must be valid");
        let w0 = csr
            .edge_weights(0)
            .expect("test invariant: value must be valid");
        assert!((w0[0] - 2.0).abs() < 1e-6);
        let w1 = csr
            .edge_weights(1)
            .expect("test invariant: value must be valid");
        assert!((w1[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn length_mismatch_error() {
        let err = CooGraph::new(3, vec![0, 1], vec![1]); // src.len != dst.len
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }
}

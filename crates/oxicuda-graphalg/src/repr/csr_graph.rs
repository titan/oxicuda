//! Compressed Sparse Row graph representation.

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

/// CSR-formatted directed graph: `row_ptr` of length `n+1` and `col_idx` of length `m`.
#[derive(Debug, Clone)]
pub struct CsrGraph {
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
}

impl CsrGraph {
    /// Create an empty CSR graph with `n` nodes and no edges.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            row_ptr: vec![0; n + 1],
            col_idx: Vec::new(),
        }
    }

    /// Build a CSR graph from an [`AdjacencyList`].
    pub fn from_adjacency_list(adj: &AdjacencyList) -> Self {
        let n = adj.n;
        let mut row_ptr = vec![0usize; n + 1];
        for (u, neigh) in adj.edges.iter().enumerate() {
            row_ptr[u + 1] = row_ptr[u] + neigh.len();
        }
        let m = row_ptr[n];
        let mut col_idx = Vec::with_capacity(m);
        for neigh in &adj.edges {
            for &v in neigh {
                col_idx.push(v);
            }
        }
        Self {
            n,
            row_ptr,
            col_idx,
        }
    }

    /// Number of directed edges.
    pub fn num_edges(&self) -> usize {
        self.col_idx.len()
    }

    /// Neighbors of vertex `u` in this CSR graph.
    pub fn neighbors(&self, u: usize) -> GraphalgResult<&[usize]> {
        if u >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u,
                len: self.n,
            });
        }
        let start = self.row_ptr[u];
        let end = self.row_ptr[u + 1];
        Ok(&self.col_idx[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_from_adj() {
        let mut adj = AdjacencyList::new(3);
        adj.add_edge(0, 1).expect("ok");
        adj.add_edge(0, 2).expect("ok");
        adj.add_edge(2, 1).expect("ok");
        let csr = CsrGraph::from_adjacency_list(&adj);
        assert_eq!(csr.num_edges(), 3);
        assert_eq!(csr.row_ptr, vec![0, 2, 2, 3]);
        let n0 = csr.neighbors(0).expect("ok");
        assert_eq!(n0, &[1, 2]);
    }
}

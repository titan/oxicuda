//! Adjacency-matrix graph representation (row-major boolean).

use crate::error::{GraphalgError, GraphalgResult};

/// Unweighted graph stored as a dense row-major boolean matrix.
#[derive(Debug, Clone)]
pub struct AdjacencyMatrix {
    pub n: usize,
    pub mat: Vec<bool>,
}

impl AdjacencyMatrix {
    /// Create an empty (all-false) matrix with `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            mat: vec![false; n * n],
        }
    }

    fn check_index(&self, u: usize, v: usize) -> GraphalgResult<()> {
        if u >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u,
                len: self.n,
            });
        }
        if v >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: v,
                len: self.n,
            });
        }
        Ok(())
    }

    /// Set the directed edge `u -> v` to `value`.
    pub fn set(&mut self, u: usize, v: usize, value: bool) -> GraphalgResult<()> {
        self.check_index(u, v)?;
        self.mat[u * self.n + v] = value;
        Ok(())
    }

    /// Add directed edge `u -> v`.
    pub fn add_edge(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        self.set(u, v, true)
    }

    /// Add an undirected edge `(u, v)`.
    pub fn add_undirected_edge(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        self.set(u, v, true)?;
        self.set(v, u, true)?;
        Ok(())
    }

    /// Get the value of the directed edge `u -> v`.
    pub fn has_edge(&self, u: usize, v: usize) -> GraphalgResult<bool> {
        self.check_index(u, v)?;
        Ok(self.mat[u * self.n + v])
    }

    /// Number of directed edges (true entries).
    pub fn num_edges(&self) -> usize {
        self.mat.iter().filter(|x| **x).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_matrix() {
        let m = AdjacencyMatrix::new(4);
        assert_eq!(m.num_edges(), 0);
    }

    #[test]
    fn set_and_get() {
        let mut m = AdjacencyMatrix::new(3);
        m.add_edge(0, 1).expect("ok");
        assert!(m.has_edge(0, 1).expect("ok"));
        assert!(!m.has_edge(1, 0).expect("ok"));
    }

    #[test]
    fn undirected_symmetric() {
        let mut m = AdjacencyMatrix::new(3);
        m.add_undirected_edge(0, 2).expect("ok");
        assert!(m.has_edge(0, 2).expect("ok"));
        assert!(m.has_edge(2, 0).expect("ok"));
    }
}

//! Adjacency-list graph representation.

use crate::error::{GraphalgError, GraphalgResult};

/// Unweighted graph stored as an adjacency list. Directed by default.
#[derive(Debug, Clone)]
pub struct AdjacencyList {
    pub n: usize,
    pub edges: Vec<Vec<usize>>,
}

impl AdjacencyList {
    /// Create an empty graph with `n` nodes and no edges.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: vec![Vec::new(); n],
        }
    }

    /// Add a directed edge `u -> v`. Returns error if endpoints are out of range.
    pub fn add_edge(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
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
        self.edges[u].push(v);
        Ok(())
    }

    /// Add an undirected edge `(u, v)`.
    pub fn add_undirected_edge(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        self.add_edge(u, v)?;
        if u != v {
            self.add_edge(v, u)?;
        }
        Ok(())
    }

    /// Total number of directed edges stored.
    pub fn num_edges(&self) -> usize {
        self.edges.iter().map(|adj| adj.len()).sum()
    }

    /// Out-neighbors of vertex `u`.
    pub fn neighbors(&self, u: usize) -> GraphalgResult<&[usize]> {
        if u >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u,
                len: self.n,
            });
        }
        Ok(&self.edges[u])
    }

    /// Compute the in-degree of each vertex.
    pub fn in_degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.n];
        for adj in &self.edges {
            for &v in adj {
                deg[v] += 1;
            }
        }
        deg
    }

    /// Compute the out-degree of each vertex.
    pub fn out_degrees(&self) -> Vec<usize> {
        self.edges.iter().map(|adj| adj.len()).collect()
    }

    /// Return the reverse graph (reverse all directed edges).
    pub fn reverse(&self) -> Self {
        let mut rev = Self::new(self.n);
        for (u, adj) in self.edges.iter().enumerate() {
            for &v in adj {
                rev.edges[v].push(u);
            }
        }
        rev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph() {
        let g = AdjacencyList::new(5);
        assert_eq!(g.n, 5);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn add_directed_edges() {
        let mut g = AdjacencyList::new(3);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        assert_eq!(g.num_edges(), 2);
    }

    #[test]
    fn add_undirected_edge_creates_two() {
        let mut g = AdjacencyList::new(2);
        g.add_undirected_edge(0, 1).expect("ok");
        assert_eq!(g.num_edges(), 2);
    }

    #[test]
    fn add_self_loop_undirected() {
        let mut g = AdjacencyList::new(2);
        g.add_undirected_edge(0, 0).expect("ok");
        assert_eq!(g.num_edges(), 1);
    }

    #[test]
    fn add_oob_errors() {
        let mut g = AdjacencyList::new(3);
        assert!(g.add_edge(3, 0).is_err());
        assert!(g.add_edge(0, 5).is_err());
    }

    #[test]
    fn degrees_consistent() {
        let mut g = AdjacencyList::new(3);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(0, 2).expect("ok");
        g.add_edge(1, 2).expect("ok");
        let out = g.out_degrees();
        let in_d = g.in_degrees();
        assert_eq!(out, vec![2, 1, 0]);
        assert_eq!(in_d, vec![0, 1, 2]);
    }

    #[test]
    fn reverse_graph() {
        let mut g = AdjacencyList::new(3);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        let r = g.reverse();
        assert_eq!(r.neighbors(1).expect("ok"), &[0usize]);
        assert_eq!(r.neighbors(2).expect("ok"), &[1usize]);
    }
}

//! Weighted graph representation (adjacency list of `(neighbor, weight)`).

use crate::error::{GraphalgError, GraphalgResult};

/// Weighted graph stored as adjacency list of `(neighbor, weight)` pairs.
#[derive(Debug, Clone)]
pub struct WeightedGraph {
    pub n: usize,
    pub edges: Vec<Vec<(usize, f64)>>,
}

impl WeightedGraph {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: vec![Vec::new(); n],
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) -> GraphalgResult<()> {
        if u >= self.n || v >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u.max(v),
                len: self.n,
            });
        }
        if w.is_nan() || w.is_infinite() {
            return Err(GraphalgError::InvalidEdgeWeight(format!(
                "edge ({u},{v}) weight not finite: {w}"
            )));
        }
        self.edges[u].push((v, w));
        Ok(())
    }

    pub fn add_undirected_edge(&mut self, u: usize, v: usize, w: f64) -> GraphalgResult<()> {
        self.add_edge(u, v, w)?;
        if u != v {
            self.add_edge(v, u, w)?;
        }
        Ok(())
    }

    pub fn num_edges(&self) -> usize {
        self.edges.iter().map(|adj| adj.len()).sum()
    }

    pub fn neighbors(&self, u: usize) -> GraphalgResult<&[(usize, f64)]> {
        if u >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u,
                len: self.n,
            });
        }
        Ok(&self.edges[u])
    }

    /// Iterate over all directed edges as `(u, v, w)`.
    pub fn all_edges(&self) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::with_capacity(self.num_edges());
        for (u, adj) in self.edges.iter().enumerate() {
            for &(v, w) in adj {
                out.push((u, v, w));
            }
        }
        out
    }

    /// Reverse the directed weighted graph.
    pub fn reverse(&self) -> Self {
        let mut rev = Self::new(self.n);
        for (u, adj) in self.edges.iter().enumerate() {
            for &(v, w) in adj {
                rev.edges[v].push((u, w));
            }
        }
        rev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_weighted() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 2.5).expect("ok");
        g.add_edge(1, 2, 4.0).expect("ok");
        assert_eq!(g.num_edges(), 2);
    }

    #[test]
    fn undirected_weighted() {
        let mut g = WeightedGraph::new(3);
        g.add_undirected_edge(0, 1, 1.5).expect("ok");
        assert_eq!(g.num_edges(), 2);
    }

    #[test]
    fn rejects_nan() {
        let mut g = WeightedGraph::new(2);
        assert!(g.add_edge(0, 1, f64::NAN).is_err());
    }
}

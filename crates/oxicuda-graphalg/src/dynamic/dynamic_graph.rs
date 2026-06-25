//! A mutable directed graph supporting incremental edge insertion and deletion.
//!
//! [`DynamicGraph`] is the substrate for the streaming-update algorithms in this
//! module (incremental PageRank, incremental SCC). Unlike the immutable
//! [`AdjacencyList`], it maintains
//! **both** the forward and reverse adjacency in lock-step and tracks the
//! *multiplicity* of every ordered pair so that `remove_edge` has well-defined
//! semantics on multigraphs: inserting a pair twice and removing it once leaves
//! the edge present with multiplicity one.
//!
//! The structure is deliberately a thin, allocation-light wrapper: out- and
//! in-neighbour lists are kept de-duplicated (one entry per distinct neighbour)
//! with the parallel multiplicity stored alongside, so neighbour iteration is
//! `O(deg)` and degree queries are `O(1)`-amortised.

use std::collections::HashMap;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

/// Mutable directed graph with incremental add/remove and a reverse index.
#[derive(Debug, Clone)]
pub struct DynamicGraph {
    n: usize,
    /// Distinct out-neighbours of each vertex (de-duplicated).
    out_adj: Vec<Vec<usize>>,
    /// Distinct in-neighbours of each vertex (de-duplicated).
    in_adj: Vec<Vec<usize>>,
    /// Multiplicity of each present ordered pair `(u, v)`.
    mult: HashMap<(usize, usize), u32>,
}

impl DynamicGraph {
    /// Create an edgeless dynamic graph on `n` vertices.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            out_adj: vec![Vec::new(); n],
            in_adj: vec![Vec::new(); n],
            mult: HashMap::new(),
        }
    }

    /// Build a dynamic graph from a static [`AdjacencyList`] (directed).
    ///
    /// Parallel edges in the source list increment the multiplicity.
    pub fn from_adjacency_list(g: &AdjacencyList) -> GraphalgResult<Self> {
        let mut dg = Self::new(g.n);
        for u in 0..g.n {
            for &v in g.neighbors(u)? {
                dg.add_edge(u, v)?;
            }
        }
        Ok(dg)
    }

    /// Number of vertices.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Number of *distinct* directed edges (ignoring multiplicity).
    pub fn num_distinct_edges(&self) -> usize {
        self.mult.len()
    }

    /// Total number of directed edges counting multiplicity.
    pub fn num_edges(&self) -> usize {
        self.mult.values().map(|&m| m as usize).sum()
    }

    /// Out-neighbours of `u` (distinct, unsorted).
    pub fn out_neighbors(&self, u: usize) -> GraphalgResult<&[usize]> {
        self.check(u)?;
        Ok(&self.out_adj[u])
    }

    /// In-neighbours of `u` (distinct, unsorted).
    pub fn in_neighbors(&self, u: usize) -> GraphalgResult<&[usize]> {
        self.check(u)?;
        Ok(&self.in_adj[u])
    }

    /// Out-degree of `u` counting multiplicity.
    pub fn out_degree(&self, u: usize) -> GraphalgResult<usize> {
        self.check(u)?;
        let mut d = 0usize;
        for &v in &self.out_adj[u] {
            d += *self.mult.get(&(u, v)).unwrap_or(&0) as usize;
        }
        Ok(d)
    }

    /// Multiplicity of the ordered pair `(u, v)` (zero if absent).
    pub fn edge_multiplicity(&self, u: usize, v: usize) -> u32 {
        *self.mult.get(&(u, v)).unwrap_or(&0)
    }

    /// Whether the directed edge `u -> v` is present.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.mult.contains_key(&(u, v))
    }

    /// Insert a directed edge `u -> v`, incrementing multiplicity.
    ///
    /// Returns `true` if this introduced a *new* distinct edge (the pair was not
    /// previously present), `false` if it only bumped an existing multiplicity.
    pub fn add_edge(&mut self, u: usize, v: usize) -> GraphalgResult<bool> {
        self.check(u)?;
        self.check(v)?;
        let entry = self.mult.entry((u, v)).or_insert(0);
        let was_new = *entry == 0;
        *entry += 1;
        if was_new {
            self.out_adj[u].push(v);
            self.in_adj[v].push(u);
        }
        Ok(was_new)
    }

    /// Remove one unit of multiplicity from the directed edge `u -> v`.
    ///
    /// Returns `true` if the distinct edge was *fully* removed (multiplicity hit
    /// zero), `false` if it merely decremented. Errors with [`GraphalgError::NoSolution`]
    /// when the edge is absent.
    pub fn remove_edge(&mut self, u: usize, v: usize) -> GraphalgResult<bool> {
        self.check(u)?;
        self.check(v)?;
        match self.mult.get_mut(&(u, v)) {
            None => Err(GraphalgError::NoSolution(format!(
                "edge ({u},{v}) is not present and cannot be removed"
            ))),
            Some(m) => {
                *m -= 1;
                if *m == 0 {
                    self.mult.remove(&(u, v));
                    remove_first(&mut self.out_adj[u], v);
                    remove_first(&mut self.in_adj[v], u);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Materialise the current state as a static [`AdjacencyList`], expanding
    /// each pair to its multiplicity (so parallel edges reappear).
    pub fn to_adjacency_list(&self) -> AdjacencyList {
        let mut g = AdjacencyList::new(self.n);
        for u in 0..self.n {
            for &v in &self.out_adj[u] {
                let m = *self.mult.get(&(u, v)).unwrap_or(&0);
                for _ in 0..m {
                    g.edges[u].push(v);
                }
            }
        }
        g
    }

    fn check(&self, u: usize) -> GraphalgResult<()> {
        if u >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u,
                len: self.n,
            });
        }
        Ok(())
    }
}

/// Remove the first occurrence of `target` from `vec` (swap-remove, order-free).
fn remove_first(vec: &mut Vec<usize>, target: usize) {
    if let Some(pos) = vec.iter().position(|&x| x == target) {
        vec.swap_remove(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_remove_round_trip() {
        let mut g = DynamicGraph::new(3);
        assert!(g.add_edge(0, 1).expect("add"));
        assert!(g.add_edge(1, 2).expect("add"));
        assert!(!g.add_edge(0, 1).expect("add"), "second add is not new");
        assert_eq!(g.num_distinct_edges(), 2);
        assert_eq!(g.num_edges(), 3);
        assert_eq!(g.edge_multiplicity(0, 1), 2);

        // first remove only decrements
        assert!(!g.remove_edge(0, 1).expect("rm"));
        assert!(g.has_edge(0, 1));
        // second fully removes
        assert!(g.remove_edge(0, 1).expect("rm"));
        assert!(!g.has_edge(0, 1));
        assert_eq!(g.num_distinct_edges(), 1);
    }

    #[test]
    fn reverse_index_consistent() {
        let mut g = DynamicGraph::new(4);
        g.add_edge(0, 2).expect("add");
        g.add_edge(1, 2).expect("add");
        g.add_edge(3, 2).expect("add");
        let mut ins = g.in_neighbors(2).expect("in").to_vec();
        ins.sort_unstable();
        assert_eq!(ins, vec![0, 1, 3]);
        g.remove_edge(1, 2).expect("rm");
        let mut ins2 = g.in_neighbors(2).expect("in").to_vec();
        ins2.sort_unstable();
        assert_eq!(ins2, vec![0, 3]);
        assert_eq!(g.out_neighbors(1).expect("out"), &[] as &[usize]);
    }

    #[test]
    fn out_degree_counts_multiplicity() {
        let mut g = DynamicGraph::new(2);
        g.add_edge(0, 1).expect("add");
        g.add_edge(0, 1).expect("add");
        g.add_edge(0, 1).expect("add");
        assert_eq!(g.out_degree(0).expect("deg"), 3);
        // distinct out-neighbours is still one entry
        assert_eq!(g.out_neighbors(0).expect("out").len(), 1);
    }

    #[test]
    fn remove_absent_errors() {
        let mut g = DynamicGraph::new(2);
        assert!(g.remove_edge(0, 1).is_err());
    }

    #[test]
    fn out_of_range_errors() {
        let mut g = DynamicGraph::new(2);
        assert!(g.add_edge(0, 5).is_err());
        assert!(g.out_neighbors(9).is_err());
    }

    #[test]
    fn round_trip_through_adjacency_list() {
        let mut src = AdjacencyList::new(3);
        src.add_edge(0, 1).expect("e");
        src.add_edge(0, 1).expect("e"); // parallel
        src.add_edge(1, 2).expect("e");
        let dg = DynamicGraph::from_adjacency_list(&src).expect("build");
        assert_eq!(dg.edge_multiplicity(0, 1), 2);
        let back = dg.to_adjacency_list();
        assert_eq!(back.num_edges(), 3);
        assert_eq!(back.out_degrees(), vec![2, 1, 0]);
    }
}

//! Bellman-Ford single-source shortest path with negative-cycle detection.

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

#[derive(Debug, Clone)]
pub struct BellmanFordOutput {
    pub dist: Vec<f64>,
    pub parent: Vec<usize>,
}

pub fn bellman_ford(g: &WeightedGraph, source: usize) -> GraphalgResult<BellmanFordOutput> {
    if source >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source,
            n: g.n,
        });
    }
    let mut dist = vec![f64::INFINITY; g.n];
    let mut parent = vec![usize::MAX; g.n];
    dist[source] = 0.0;
    parent[source] = source;
    let edges = g.all_edges();
    for _ in 0..g.n.saturating_sub(1) {
        let mut updated = false;
        for &(u, v, w) in &edges {
            if dist[u].is_finite() && dist[u] + w < dist[v] {
                dist[v] = dist[u] + w;
                parent[v] = u;
                updated = true;
            }
        }
        if !updated {
            break;
        }
    }
    // Negative-cycle check
    for &(u, v, w) in &edges {
        if dist[u].is_finite() && dist[u] + w < dist[v] {
            return Err(GraphalgError::NegativeCycle);
        }
    }
    Ok(BellmanFordOutput { dist, parent })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf_simple_path() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 2.0).expect("ok");
        g.add_edge(2, 3, 3.0).expect("ok");
        let out = bellman_ford(&g, 0).expect("ok");
        assert!((out.dist[3] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn bf_with_negative_edge() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 4.0).expect("ok");
        g.add_edge(0, 2, 5.0).expect("ok");
        g.add_edge(1, 2, -2.0).expect("ok");
        let out = bellman_ford(&g, 0).expect("ok");
        assert!((out.dist[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn bf_negative_cycle_detected() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, -3.0).expect("ok");
        g.add_edge(2, 0, 1.0).expect("ok");
        assert!(matches!(
            bellman_ford(&g, 0),
            Err(GraphalgError::NegativeCycle)
        ));
    }
}

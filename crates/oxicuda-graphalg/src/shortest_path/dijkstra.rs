//! Binary-heap Dijkstra single-source shortest path. Rejects negative weights.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

/// Result of a Dijkstra run from a single source.
#[derive(Debug, Clone)]
pub struct DijkstraOutput {
    pub dist: Vec<f64>,
    pub parent: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapItem {
    dist: f64,
    node: usize,
}

impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap by distance.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
    }
}

pub fn dijkstra(g: &WeightedGraph, source: usize) -> GraphalgResult<DijkstraOutput> {
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
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    heap.push(HeapItem {
        dist: 0.0,
        node: source,
    });
    while let Some(HeapItem { dist: d, node: u }) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        for &(v, w) in g.neighbors(u)? {
            if w < 0.0 {
                return Err(GraphalgError::NegativeWeight {
                    edge: (u, v),
                    weight: w,
                });
            }
            let nd = d + w;
            if nd < dist[v] {
                dist[v] = nd;
                parent[v] = u;
                heap.push(HeapItem { dist: nd, node: v });
            }
        }
    }
    Ok(DijkstraOutput { dist, parent })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dijkstra_4node() {
        // 0 -> 1 (1), 0 -> 2 (4), 1 -> 2 (2), 1 -> 3 (5), 2 -> 3 (1)
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(0, 2, 4.0).expect("ok");
        g.add_edge(1, 2, 2.0).expect("ok");
        g.add_edge(1, 3, 5.0).expect("ok");
        g.add_edge(2, 3, 1.0).expect("ok");
        let out = dijkstra(&g, 0).expect("ok");
        assert!((out.dist[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn dijkstra_rejects_negative() {
        let mut g = WeightedGraph::new(2);
        g.add_edge(0, 1, -1.0).expect("ok");
        assert!(dijkstra(&g, 0).is_err());
    }

    #[test]
    fn dijkstra_zero_weight_ok() {
        let mut g = WeightedGraph::new(2);
        g.add_edge(0, 1, 0.0).expect("ok");
        let out = dijkstra(&g, 0).expect("ok");
        assert!((out.dist[1] - 0.0).abs() < 1e-12);
    }
}

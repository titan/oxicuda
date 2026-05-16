//! Prim's MST algorithm with binary heap.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Item {
    weight: f64,
    node: usize,
    from: usize,
}
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.weight
            .partial_cmp(&self.weight)
            .unwrap_or(Ordering::Equal)
            .then(o.node.cmp(&self.node))
    }
}

/// Run Prim's MST starting from `source`. Returns the list of MST edges `(u, v, w)`.
/// Assumes graph is undirected and connected. Negative weights are allowed for MST.
pub fn prim_mst(g: &WeightedGraph, source: usize) -> GraphalgResult<Vec<(usize, usize, f64)>> {
    if source >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source,
            n: g.n,
        });
    }
    let mut in_tree = vec![false; g.n];
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    let mut heap: BinaryHeap<Item> = BinaryHeap::new();
    in_tree[source] = true;
    for &(v, w) in g.neighbors(source)? {
        heap.push(Item {
            weight: w,
            node: v,
            from: source,
        });
    }
    while let Some(Item { weight, node, from }) = heap.pop() {
        if in_tree[node] {
            continue;
        }
        in_tree[node] = true;
        edges.push((from, node, weight));
        for &(v, w) in g.neighbors(node)? {
            if !in_tree[v] {
                heap.push(Item {
                    weight: w,
                    node: v,
                    from: node,
                });
            }
        }
    }
    if edges.len() != g.n - 1 {
        return Err(GraphalgError::DisconnectedGraph(
            "Prim could not span the graph".to_string(),
        ));
    }
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> WeightedGraph {
        let mut g = WeightedGraph::new(4);
        g.add_undirected_edge(0, 1, 1.0).expect("ok");
        g.add_undirected_edge(0, 2, 4.0).expect("ok");
        g.add_undirected_edge(1, 2, 2.0).expect("ok");
        g.add_undirected_edge(1, 3, 5.0).expect("ok");
        g.add_undirected_edge(2, 3, 1.0).expect("ok");
        g
    }

    #[test]
    fn prim_total_weight() {
        let g = small();
        let mst = prim_mst(&g, 0).expect("ok");
        let total: f64 = mst.iter().map(|e| e.2).sum();
        assert!((total - 4.0).abs() < 1e-12);
    }

    #[test]
    fn prim_disconnected_err() {
        let mut g = WeightedGraph::new(4);
        g.add_undirected_edge(0, 1, 1.0).expect("ok");
        g.add_undirected_edge(2, 3, 1.0).expect("ok");
        assert!(prim_mst(&g, 0).is_err());
    }
}

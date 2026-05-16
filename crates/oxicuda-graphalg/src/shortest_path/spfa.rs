//! Shortest Path Faster Algorithm (queue-based Bellman-Ford variant).

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

use super::bellman_ford::BellmanFordOutput;

pub fn spfa(g: &WeightedGraph, source: usize) -> GraphalgResult<BellmanFordOutput> {
    if source >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source,
            n: g.n,
        });
    }
    let mut dist = vec![f64::INFINITY; g.n];
    let mut parent = vec![usize::MAX; g.n];
    let mut in_queue = vec![false; g.n];
    let mut count = vec![0usize; g.n];
    dist[source] = 0.0;
    parent[source] = source;
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(source);
    in_queue[source] = true;
    while let Some(u) = q.pop_front() {
        in_queue[u] = false;
        for &(v, w) in g.neighbors(u)? {
            let nd = dist[u] + w;
            if nd < dist[v] {
                dist[v] = nd;
                parent[v] = u;
                if !in_queue[v] {
                    q.push_back(v);
                    in_queue[v] = true;
                    count[v] += 1;
                    if count[v] > g.n {
                        return Err(GraphalgError::NegativeCycle);
                    }
                }
            }
        }
    }
    Ok(BellmanFordOutput { dist, parent })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spfa_simple() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 2.0).expect("ok");
        g.add_edge(2, 3, 3.0).expect("ok");
        let out = spfa(&g, 0).expect("ok");
        assert!((out.dist[3] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn spfa_negative_cycle() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, -3.0).expect("ok");
        g.add_edge(2, 0, 1.0).expect("ok");
        assert!(matches!(spfa(&g, 0), Err(GraphalgError::NegativeCycle)));
    }
}

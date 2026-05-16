//! A* search with admissible heuristic.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

#[derive(Debug, Clone)]
pub struct AStarOutput {
    pub dist: f64,
    pub path: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Item {
    f: f64,
    g: f64,
    node: usize,
}
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .partial_cmp(&self.f)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
    }
}

/// A* shortest path. `heuristic(node)` must be admissible (never overestimates).
pub fn a_star<F>(
    graph: &WeightedGraph,
    source: usize,
    target: usize,
    heuristic: F,
) -> GraphalgResult<AStarOutput>
where
    F: Fn(usize) -> f64,
{
    if source >= graph.n || target >= graph.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source.max(target),
            n: graph.n,
        });
    }
    let mut gscore = vec![f64::INFINITY; graph.n];
    let mut parent = vec![usize::MAX; graph.n];
    gscore[source] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(Item {
        f: heuristic(source),
        g: 0.0,
        node: source,
    });
    while let Some(Item {
        f: _,
        g: gu,
        node: u,
    }) = heap.pop()
    {
        if u == target {
            // Reconstruct
            let mut path = Vec::new();
            let mut cur = target;
            while cur != source {
                path.push(cur);
                let p = parent[cur];
                if p == usize::MAX {
                    return Err(GraphalgError::NumericalInstability(
                        "broken parent in A*".to_string(),
                    ));
                }
                cur = p;
            }
            path.push(source);
            path.reverse();
            return Ok(AStarOutput { dist: gu, path });
        }
        if gu > gscore[u] {
            continue;
        }
        for &(v, w) in graph.neighbors(u)? {
            if w < 0.0 {
                return Err(GraphalgError::NegativeWeight {
                    edge: (u, v),
                    weight: w,
                });
            }
            let cand = gu + w;
            if cand < gscore[v] {
                gscore[v] = cand;
                parent[v] = u;
                let fv = cand + heuristic(v);
                heap.push(Item {
                    f: fv,
                    g: cand,
                    node: v,
                });
            }
        }
    }
    Err(GraphalgError::NoSolution(format!(
        "A* failed to reach target {target}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_4() -> WeightedGraph {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(0, 2, 4.0).expect("ok");
        g.add_edge(1, 2, 2.0).expect("ok");
        g.add_edge(1, 3, 5.0).expect("ok");
        g.add_edge(2, 3, 1.0).expect("ok");
        g
    }

    #[test]
    fn a_star_zero_heuristic_eq_dijkstra() {
        let g = graph_4();
        let out = a_star(&g, 0, 3, |_| 0.0).expect("ok");
        assert!((out.dist - 4.0).abs() < 1e-12);
    }

    #[test]
    fn a_star_with_heuristic() {
        let g = graph_4();
        // Trivial heuristic 0 still works
        let out = a_star(&g, 0, 3, |_| 0.0).expect("ok");
        assert_eq!(*out.path.last().expect("ok"), 3usize);
    }

    #[test]
    fn a_star_no_path() {
        let g = WeightedGraph::new(3);
        assert!(a_star(&g, 0, 2, |_| 0.0).is_err());
    }
}

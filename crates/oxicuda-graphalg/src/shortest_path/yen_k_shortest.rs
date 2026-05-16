//! Yen's K-shortest loopless paths via deviation method.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

use super::dijkstra::dijkstra;

#[derive(Debug, Clone)]
pub struct YenPath {
    pub cost: f64,
    pub nodes: Vec<usize>,
}

#[derive(Debug, Clone)]
struct CandidateItem {
    cost: f64,
    nodes: Vec<usize>,
}
impl PartialEq for CandidateItem {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.nodes == other.nodes
    }
}
impl Eq for CandidateItem {}
impl PartialOrd for CandidateItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for CandidateItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.nodes.cmp(&self.nodes))
    }
}

/// Compute up to `k` shortest loopless paths from `source` to `target`.
pub fn yen_k_shortest_paths(
    graph: &WeightedGraph,
    source: usize,
    target: usize,
    k: usize,
) -> GraphalgResult<Vec<YenPath>> {
    if source >= graph.n || target >= graph.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source.max(target),
            n: graph.n,
        });
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    let mut paths: Vec<YenPath> = Vec::new();
    let mut candidates: BinaryHeap<CandidateItem> = BinaryHeap::new();
    // First shortest path
    let out = dijkstra(graph, source)?;
    if !out.dist[target].is_finite() {
        return Err(GraphalgError::NoSolution(format!(
            "no path from {source} to {target}"
        )));
    }
    let mut p = reconstruct(&out.parent, source, target)?;
    paths.push(YenPath {
        cost: out.dist[target],
        nodes: p.clone(),
    });
    for _ in 1..k {
        let prev = paths.last().cloned().unwrap_or(YenPath {
            cost: 0.0,
            nodes: Vec::new(),
        });
        for i in 0..prev.nodes.len() - 1 {
            let spur_node = prev.nodes[i];
            let root_path: Vec<usize> = prev.nodes[..=i].to_vec();
            // Build a modified graph: remove edges that share root prefix
            let mut removed_edges: Vec<(usize, usize, f64)> = Vec::new();
            let mut modified = graph.clone();
            for path in &paths {
                if path.nodes.len() > i + 1 && path.nodes[..=i] == root_path[..] {
                    let u = path.nodes[i];
                    let v = path.nodes[i + 1];
                    // remove edge u->v
                    let pos = modified.edges[u].iter().position(|&(x, _)| x == v);
                    if let Some(pos) = pos {
                        let (vv, ww) = modified.edges[u].remove(pos);
                        removed_edges.push((u, vv, ww));
                    }
                }
            }
            // Also remove root-path nodes (except spur)
            let mut blocked_nodes = vec![false; graph.n];
            for &n in &root_path[..root_path.len() - 1] {
                blocked_nodes[n] = true;
            }
            // Run dijkstra on the modified graph but with blocked nodes
            let spur_out = dijkstra_avoid(&modified, spur_node, &blocked_nodes)?;
            if spur_out.dist[target].is_finite() {
                let spur_path = reconstruct(&spur_out.parent, spur_node, target)?;
                let mut total_path = root_path.clone();
                total_path.extend(spur_path.into_iter().skip(1));
                let mut cost = 0.0;
                for w in total_path.windows(2) {
                    let u = w[0];
                    let v = w[1];
                    let edge = graph.edges[u]
                        .iter()
                        .find(|&&(x, _)| x == v)
                        .copied()
                        .unwrap_or((v, f64::INFINITY));
                    cost += edge.1;
                }
                if cost.is_finite() {
                    candidates.push(CandidateItem {
                        cost,
                        nodes: total_path,
                    });
                }
            }
            // Restore removed edges
            for (u, v, w) in removed_edges {
                modified.edges[u].push((v, w));
            }
        }
        match candidates.pop() {
            Some(c) => {
                if paths.iter().any(|p| p.nodes == c.nodes) {
                    continue;
                }
                p = c.nodes.clone();
                paths.push(YenPath {
                    cost: c.cost,
                    nodes: c.nodes,
                });
            }
            None => break,
        }
        let _ = p;
    }
    Ok(paths)
}

fn dijkstra_avoid(
    g: &WeightedGraph,
    source: usize,
    blocked: &[bool],
) -> GraphalgResult<super::dijkstra::DijkstraOutput> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct H {
        d: f64,
        v: usize,
    }
    impl Eq for H {}
    impl PartialOrd for H {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for H {
        fn cmp(&self, o: &Self) -> Ordering {
            o.d.partial_cmp(&self.d)
                .unwrap_or(Ordering::Equal)
                .then(o.v.cmp(&self.v))
        }
    }
    let mut dist = vec![f64::INFINITY; g.n];
    let mut parent = vec![usize::MAX; g.n];
    if blocked[source] {
        return Ok(super::dijkstra::DijkstraOutput { dist, parent });
    }
    dist[source] = 0.0;
    parent[source] = source;
    let mut heap: BinaryHeap<H> = BinaryHeap::new();
    heap.push(H { d: 0.0, v: source });
    while let Some(H { d, v: u }) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        for &(v, w) in g.neighbors(u)? {
            if blocked[v] {
                continue;
            }
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
                heap.push(H { d: nd, v });
            }
        }
    }
    Ok(super::dijkstra::DijkstraOutput { dist, parent })
}

fn reconstruct(parent: &[usize], source: usize, target: usize) -> GraphalgResult<Vec<usize>> {
    if target >= parent.len() || parent[target] == usize::MAX {
        return Err(GraphalgError::NoSolution(format!("no path to {target}")));
    }
    let mut path = Vec::new();
    let mut cur = target;
    while cur != source {
        path.push(cur);
        let p = parent[cur];
        if p == usize::MAX {
            return Err(GraphalgError::NumericalInstability(
                "broken parent".to_string(),
            ));
        }
        cur = p;
    }
    path.push(source);
    path.reverse();
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yen_two_paths() {
        // Two parallel paths from 0 to 3.
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 3, 1.0).expect("ok");
        g.add_edge(0, 2, 2.0).expect("ok");
        g.add_edge(2, 3, 2.0).expect("ok");
        let paths = yen_k_shortest_paths(&g, 0, 3, 2).expect("ok");
        assert_eq!(paths.len(), 2);
        assert!((paths[0].cost - 2.0).abs() < 1e-9);
        assert!((paths[1].cost - 4.0).abs() < 1e-9);
    }
}

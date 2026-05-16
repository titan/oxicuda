//! Christofides 1.5-approximation for metric TSP.
//!
//! Pipeline:
//!   1. Build a complete graph from the distance matrix.
//!   2. Compute an MST (Kruskal).
//!   3. Identify vertices of odd degree in the MST.
//!   4. Compute a minimum-weight perfect matching on the odd-degree subgraph
//!      (greedy approximation used here; full blossom is intentionally simplified).
//!   5. Combine MST + matching into a multigraph with all even degrees.
//!   6. Find an Eulerian circuit (Hierholzer).
//!   7. Shortcut repeated nodes to obtain a Hamiltonian cycle.

use crate::error::{GraphalgError, GraphalgResult};
use crate::mst::kruskal::kruskal_mst;
use crate::repr::weighted_graph::WeightedGraph;

#[derive(Debug, Clone)]
pub struct ChristofidesTour {
    pub order: Vec<usize>,
    pub cost: f64,
}

pub fn christofides_tour(dist: &[f64], n: usize) -> GraphalgResult<ChristofidesTour> {
    if dist.len() != n * n {
        return Err(GraphalgError::DimensionMismatch {
            a: dist.len(),
            b: n * n,
        });
    }
    if n == 0 {
        return Ok(ChristofidesTour {
            order: Vec::new(),
            cost: 0.0,
        });
    }
    if n == 1 {
        return Ok(ChristofidesTour {
            order: vec![0],
            cost: 0.0,
        });
    }
    // Build undirected complete graph
    let mut g = WeightedGraph::new(n);
    for i in 0..n {
        for j in i + 1..n {
            g.add_undirected_edge(i, j, dist[i * n + j])?;
        }
    }
    let mst = kruskal_mst(&g)?;
    // Identify odd-degree vertices in MST.
    let mut deg = vec![0usize; n];
    for &(u, v, _) in &mst {
        deg[u] += 1;
        deg[v] += 1;
    }
    let odd: Vec<usize> = (0..n).filter(|&i| deg[i] % 2 == 1).collect();
    // Greedy min-weight perfect matching on odd vertices.
    let mut matched = vec![false; odd.len()];
    let mut matching: Vec<(usize, usize)> = Vec::new();
    for i in 0..odd.len() {
        if matched[i] {
            continue;
        }
        let mut best = usize::MAX;
        let mut bestd = f64::INFINITY;
        for j in i + 1..odd.len() {
            if matched[j] {
                continue;
            }
            let d = dist[odd[i] * n + odd[j]];
            if d < bestd {
                bestd = d;
                best = j;
            }
        }
        if best != usize::MAX {
            matching.push((odd[i], odd[best]));
            matched[i] = true;
            matched[best] = true;
        }
    }
    // Build multigraph: MST + matching, stored as adjacency list of (neighbor, edge_id).
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut edge_used = vec![false; mst.len() + matching.len()];
    for (eid, &(u, v, _)) in mst.iter().enumerate() {
        adj[u].push((v, eid));
        adj[v].push((u, eid));
    }
    for (k, &(u, v)) in matching.iter().enumerate() {
        let eid = mst.len() + k;
        adj[u].push((v, eid));
        adj[v].push((u, eid));
    }
    // Hierholzer Eulerian circuit
    let mut circuit: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = vec![0];
    let mut idx = vec![0usize; n];
    while let Some(&u) = stack.last() {
        if idx[u] < adj[u].len() {
            // pop next edge from u
            let (v, eid) = adj[u][idx[u]];
            idx[u] += 1;
            if edge_used[eid] {
                continue;
            }
            edge_used[eid] = true;
            stack.push(v);
        } else {
            circuit.push(u);
            stack.pop();
        }
    }
    // Shortcut to Hamiltonian
    let mut seen = vec![false; n];
    let mut order: Vec<usize> = Vec::new();
    for &v in &circuit {
        if !seen[v] {
            seen[v] = true;
            order.push(v);
        }
    }
    if order.len() != n {
        return Err(GraphalgError::NoSolution(
            "Christofides shortcutting did not visit all nodes".to_string(),
        ));
    }
    let mut cost = 0.0;
    for i in 0..n {
        cost += dist[order[i] * n + order[(i + 1) % n]];
    }
    Ok(ChristofidesTour { order, cost })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn christofides_square() {
        let pts: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let n = 4;
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let dx = pts[i].0 - pts[j].0;
                let dy = pts[i].1 - pts[j].1;
                d[i * n + j] = (dx * dx + dy * dy).sqrt();
            }
        }
        let t = christofides_tour(&d, n).expect("ok");
        assert_eq!(t.order.len(), 4);
        assert!(t.cost <= 4.0 * 1.5 + 1e-9);
    }
}

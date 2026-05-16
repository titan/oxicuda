//! Hierholzer's Eulerian circuit, O(E).
//!
//! Assumes the graph is connected and every vertex has even degree.

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

pub fn eulerian_circuit_hierholzer(g: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    let n = g.n;
    if n == 0 {
        return Ok(Vec::new());
    }
    // Build mutable adjacency with edge ids.
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut total_edges = 0usize;
    // Each undirected edge counted twice in adj_list; pair them by edge id.
    let mut edge_pairs: Vec<(usize, usize)> = Vec::new();
    for (u, neigh) in g.edges.iter().enumerate() {
        for &v in neigh {
            if u <= v {
                let eid = edge_pairs.len();
                edge_pairs.push((u, v));
                adj[u].push((v, eid));
                if u != v {
                    adj[v].push((u, eid));
                }
                total_edges += 1;
            }
        }
    }
    let mut used = vec![false; total_edges];
    // Check degrees.
    for u in 0..n {
        if adj[u].len() % 2 != 0 {
            return Err(GraphalgError::InvalidGraph(format!(
                "node {u} has odd degree"
            )));
        }
    }
    if total_edges == 0 {
        return Ok(vec![0]);
    }
    let start = (0..n).find(|&u| !adj[u].is_empty()).unwrap_or(0);
    let mut stack: Vec<usize> = vec![start];
    let mut idx = vec![0usize; n];
    let mut circuit: Vec<usize> = Vec::new();
    while let Some(&u) = stack.last() {
        if idx[u] < adj[u].len() {
            let (v, eid) = adj[u][idx[u]];
            idx[u] += 1;
            if used[eid] {
                continue;
            }
            used[eid] = true;
            stack.push(v);
        } else {
            circuit.push(u);
            stack.pop();
        }
    }
    circuit.reverse();
    Ok(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_triangle() {
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        g.add_undirected_edge(0, 2).expect("ok");
        let c = eulerian_circuit_hierholzer(&g).expect("ok");
        assert_eq!(c.first(), c.last());
        assert_eq!(c.len(), 4);
    }

    #[test]
    fn euler_odd_degree_err() {
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        assert!(eulerian_circuit_hierholzer(&g).is_err());
    }
}

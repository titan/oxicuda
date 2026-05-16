//! Johnson's algorithm: all-pairs shortest path via reweighting (Bellman-Ford then Dijkstra).

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

use super::bellman_ford::bellman_ford;
use super::dijkstra::dijkstra;

/// Returns row-major n x n distance matrix using Johnson's reweighting trick.
pub fn johnson(g: &WeightedGraph) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    // Augment graph with extra source (index n) connected with weight 0 to all nodes.
    let mut aug = WeightedGraph::new(n + 1);
    for (u, adj) in g.edges.iter().enumerate() {
        for &(v, w) in adj {
            aug.add_edge(u, v, w)?;
        }
    }
    for v in 0..n {
        aug.add_edge(n, v, 0.0)?;
    }
    let h_out = bellman_ford(&aug, n)?;
    let h = h_out.dist;
    // Reweight: w'(u,v) = w(u,v) + h[u] - h[v]
    let mut reweighted = WeightedGraph::new(n);
    for (u, adj) in g.edges.iter().enumerate() {
        for &(v, w) in adj {
            let nw = w + h[u] - h[v];
            if nw < 0.0 {
                return Err(GraphalgError::NumericalInstability(
                    "reweighting produced negative edge".to_string(),
                ));
            }
            reweighted.add_edge(u, v, nw)?;
        }
    }
    let mut dist = vec![f64::INFINITY; n * n];
    for s in 0..n {
        let out = dijkstra(&reweighted, s)?;
        for t in 0..n {
            if out.dist[t].is_finite() {
                dist[s * n + t] = out.dist[t] - h[s] + h[t];
            }
        }
    }
    Ok(dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn johnson_matches_fw() {
        // Use only positive edges so it matches Floyd-Warshall easily.
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 2.0).expect("ok");
        g.add_edge(0, 2, 5.0).expect("ok");
        let d = johnson(&g).expect("ok");
        assert!((d[0 * 3 + 2] - 3.0).abs() < 1e-9);
    }
}

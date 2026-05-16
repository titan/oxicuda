//! Floyd-Warshall all-pairs shortest path. O(V^3).

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

/// Returns a row-major n x n distance matrix. Self-distance is 0. Unreachable -> +inf.
/// If a negative cycle exists, returns `NegativeCycle`.
pub fn floyd_warshall(g: &WeightedGraph) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    let mut d = vec![f64::INFINITY; n * n];
    for i in 0..n {
        d[i * n + i] = 0.0;
    }
    for (u, adj) in g.edges.iter().enumerate() {
        for &(v, w) in adj {
            let idx = u * n + v;
            if w < d[idx] {
                d[idx] = w;
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            let dik = d[i * n + k];
            if !dik.is_finite() {
                continue;
            }
            for j in 0..n {
                let dkj = d[k * n + j];
                if !dkj.is_finite() {
                    continue;
                }
                let cand = dik + dkj;
                let idx = i * n + j;
                if cand < d[idx] {
                    d[idx] = cand;
                }
            }
        }
    }
    for i in 0..n {
        if d[i * n + i] < 0.0 {
            return Err(GraphalgError::NegativeCycle);
        }
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fw_small_graph() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 2.0).expect("ok");
        g.add_edge(0, 2, 5.0).expect("ok");
        let d = floyd_warshall(&g).expect("ok");
        assert!((d[2] - 3.0).abs() < 1e-12); // 0->2 via 1
    }

    #[test]
    fn fw_negative_cycle() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, -3.0).expect("ok");
        g.add_edge(2, 0, 1.0).expect("ok");
        assert!(matches!(
            floyd_warshall(&g),
            Err(GraphalgError::NegativeCycle)
        ));
    }
}

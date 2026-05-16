//! Eigenvector centrality via power iteration on the adjacency matrix.

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

pub fn eigenvector_centrality(
    g: &AdjacencyList,
    max_iter: usize,
    tol: f64,
) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut x = vec![1.0f64 / (n as f64).sqrt(); n];
    let mut y = vec![0.0f64; n];
    for _ in 0..max_iter {
        for v in y.iter_mut() {
            *v = 0.0;
        }
        for u in 0..n {
            let xu = x[u];
            for &v in g.neighbors(u)? {
                y[v] += xu;
            }
        }
        let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-300 {
            return Err(GraphalgError::NumericalInstability(
                "zero eigenvector during power iteration".to_string(),
            ));
        }
        let mut diff = 0.0f64;
        for i in 0..n {
            let nv = y[i] / norm;
            diff += (nv - x[i]).abs();
            x[i] = nv;
        }
        if diff < tol {
            return Ok(x);
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_star_center_largest() {
        // Center of a star has largest eigenvector centrality.
        let mut g = AdjacencyList::new(4);
        for v in 1..4 {
            g.add_undirected_edge(0, v).expect("ok");
        }
        let c = eigenvector_centrality(&g, 200, 1e-10).expect("ok");
        assert!(c[0] > c[1]);
        assert!(c[0] > c[2]);
    }
}

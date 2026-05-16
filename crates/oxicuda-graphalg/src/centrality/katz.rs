//! Katz centrality via iterative fixed-point.
//!
//! Solves `c = alpha * A * c + beta * 1` by fixed-point iteration.

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

pub fn katz_centrality(
    g: &AdjacencyList,
    alpha: f64,
    beta: f64,
    max_iter: usize,
    tol: f64,
) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut c = vec![0.0f64; n];
    let mut c_next = vec![0.0f64; n];
    for _ in 0..max_iter {
        for v in c_next.iter_mut() {
            *v = beta;
        }
        for u in 0..n {
            let cu = c[u];
            for &v in g.neighbors(u)? {
                c_next[v] += alpha * cu;
            }
        }
        let diff: f64 = c
            .iter()
            .zip(c_next.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        std::mem::swap(&mut c, &mut c_next);
        if diff < tol {
            // Normalise to unit l2
            let norm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm < 1e-300 {
                return Err(GraphalgError::NumericalInstability(
                    "zero Katz vector".to_string(),
                ));
            }
            for v in c.iter_mut() {
                *v /= norm;
            }
            return Ok(c);
        }
    }
    let norm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 1e-300 {
        for v in c.iter_mut() {
            *v /= norm;
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn katz_runs() {
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        let c = katz_centrality(&g, 0.1, 1.0, 200, 1e-10).expect("ok");
        assert_eq!(c.len(), 3);
    }
}

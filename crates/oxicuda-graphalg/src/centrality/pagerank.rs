//! PageRank via power iteration with damping factor.

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

pub fn pagerank(
    g: &AdjacencyList,
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    if n == 0 {
        return Ok(Vec::new());
    }
    if !(0.0..=1.0).contains(&damping) {
        return Err(GraphalgError::InvalidParameter(format!(
            "damping out of [0,1]: {damping}"
        )));
    }
    let n_f = n as f64;
    let mut r = vec![1.0f64 / n_f; n];
    let mut r_next = vec![0.0f64; n];
    let out_degs = g.out_degrees();
    for _ in 0..max_iter {
        // dangling mass
        let mut dangling = 0.0f64;
        for v in 0..n {
            if out_degs[v] == 0 {
                dangling += r[v];
            }
        }
        let teleport = (1.0 - damping) / n_f + damping * dangling / n_f;
        for v in r_next.iter_mut() {
            *v = teleport;
        }
        for u in 0..n {
            let od = out_degs[u];
            if od == 0 {
                continue;
            }
            let contrib = damping * r[u] / od as f64;
            for &v in g.neighbors(u)? {
                r_next[v] += contrib;
            }
        }
        let diff: f64 = r
            .iter()
            .zip(r_next.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        std::mem::swap(&mut r, &mut r_next);
        if diff < tol {
            break;
        }
    }
    // Normalise to sum 1 (should be near 1 already).
    let s: f64 = r.iter().sum();
    if s.abs() > 1e-300 {
        for v in r.iter_mut() {
            *v /= s;
        }
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pagerank_sums_to_one() {
        let mut g = AdjacencyList::new(4);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g.add_edge(2, 0).expect("ok");
        g.add_edge(2, 3).expect("ok");
        let r = pagerank(&g, 0.85, 200, 1e-10).expect("ok");
        let s: f64 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }
}

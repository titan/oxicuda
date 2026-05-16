//! Isomap (Tenenbaum, 2000).
//!
//! 1. Build kNN graph with edge weights = Euclidean distance.
//! 2. Dijkstra shortest paths from each node to get geodesic distance matrix `G`.
//! 3. Apply classical MDS on `G`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::mds::classical_mds::classical_mds;
use crate::neighbor::knn_brute::knn_brute;

/// Isomap result.
pub struct IsomapResult {
    pub embedding: Vec<f64>,
    pub geodesic_distances: Vec<f64>,
}

/// Fit Isomap.
pub fn isomap_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    n_neighbors: usize,
    n_components: usize,
) -> ManifoldResult<IsomapResult> {
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![x.len()],
        });
    }
    if n_components == 0 || n_components >= n_samples {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..{n_samples}"),
        });
    }
    let n = n_samples;
    let k = n_neighbors;
    let (idx, d2) = knn_brute(x, n, dim, k)?;
    let dist: Vec<f64> = d2.iter().map(|v| v.sqrt()).collect();
    // Build adjacency as symmetric: include both i->j and j->i directions
    // Use Vec<Vec<(usize, f64)>>
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for jj in 0..k {
            let nb = idx[i * k + jj];
            let dv = dist[i * k + jj];
            adj[i].push((nb, dv));
            adj[nb].push((i, dv));
        }
    }
    // Deduplicate
    for v in &mut adj {
        v.sort_by_key(|t| t.0);
        v.dedup_by_key(|x| x.0);
    }
    let mut g = vec![f64::INFINITY; n * n];
    for i in 0..n {
        // Dijkstra from i
        let mut dist_i = vec![f64::INFINITY; n];
        dist_i[i] = 0.0;
        let mut visited = vec![false; n];
        for _ in 0..n {
            let mut min_d = f64::INFINITY;
            let mut min_u = usize::MAX;
            for u in 0..n {
                if !visited[u] && dist_i[u] < min_d {
                    min_d = dist_i[u];
                    min_u = u;
                }
            }
            if min_u == usize::MAX {
                break;
            }
            visited[min_u] = true;
            for &(v, w) in &adj[min_u] {
                if !visited[v] && dist_i[min_u] + w < dist_i[v] {
                    dist_i[v] = dist_i[min_u] + w;
                }
            }
        }
        for j in 0..n {
            g[i * n + j] = dist_i[j];
        }
    }
    // Check connectivity
    let mut disconnected = 0;
    for i in 0..n {
        for j in 0..n {
            if !g[i * n + j].is_finite() {
                disconnected += 1;
            }
        }
    }
    if disconnected > 0 {
        // Fallback: replace inf with diameter
        let mut max_finite = 0.0;
        for v in g.iter() {
            if v.is_finite() && *v > max_finite {
                max_finite = *v;
            }
        }
        let big = (max_finite * 2.0).max(1.0);
        for v in g.iter_mut() {
            if !v.is_finite() {
                *v = big;
            }
        }
    }
    let r = classical_mds(&g, n, n_components)?;
    Ok(IsomapResult {
        embedding: r.embedding,
        geodesic_distances: g,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isomap_line() {
        let n = 6;
        let dim = 2;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            x[i * dim] = i as f64;
            x[i * dim + 1] = 0.0;
        }
        let r = isomap_fit(&x, n, dim, 2, 1).expect("ok");
        assert_eq!(r.embedding.len(), n);
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }
}

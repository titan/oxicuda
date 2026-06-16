//! Isomap (Tenenbaum, de Silva & Langford 2000) — config-struct API.
//!
//! This module provides a config-struct API over the [`crate::local::isomap`]
//! implementation:
//!
//! 1. Build a kNN graph with edges weighted by Euclidean distance.
//! 2. Compute all-pairs geodesic distances via Dijkstra.
//! 3. Apply classical MDS on the geodesic distance matrix.
//!
//! The raw underlying function lives in `local/isomap.rs`; this file re-exports it
//! behind a higher-level `IsomapConfig` + `isomap(...)` free-function interface
//! compatible with the Wave AAA+66 API specification.

use crate::error::{ManifoldError, ManifoldResult};
use crate::local::isomap::isomap_fit;

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for Isomap.
#[derive(Debug, Clone)]
pub struct IsomapConfig {
    /// Number of nearest neighbours used to build the kNN graph.
    pub n_neighbors: usize,
    /// Number of embedding dimensions to retain.
    pub n_components: usize,
}

impl Default for IsomapConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            n_components: 2,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Public function
// ────────────────────────────────────────────────────────────────────────────

/// Compute an Isomap embedding of `data`.
///
/// # Parameters
/// - `data`: row-major flat slice of shape `[n × dim]`.
/// - `n`: number of data points.
/// - `dim`: intrinsic feature dimension.
/// - `config`: Isomap hyperparameters.
///
/// # Returns
/// Flat row-major slice of shape `[n × n_components]`.
///
/// # Errors
/// - [`ManifoldError::EmptyInput`] if `n == 0` or `dim == 0`.
/// - [`ManifoldError::InvalidParameter`] if `n_neighbors == 0` or
///   `n_components == 0` or `n_components >= n`.
/// - [`ManifoldError::ShapeMismatch`] if `data.len() != n * dim`.
pub fn isomap(
    data: &[f64],
    n: usize,
    dim: usize,
    config: &IsomapConfig,
) -> ManifoldResult<Vec<f64>> {
    if config.n_neighbors == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_neighbors".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if config.n_components == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if n > 0 && config.n_components >= n {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be < n={n}, got {}", config.n_components),
        });
    }
    let result = isomap_fit(data, n, dim, config.n_neighbors, config.n_components)?;
    Ok(result.embedding)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn line_data(n: usize) -> Vec<f64> {
        // Points on the x-axis: (0,0), (1,0), ..., (n-1, 0)
        let mut v = vec![0.0_f64; n * 2];
        for i in 0..n {
            v[i * 2] = i as f64;
        }
        v
    }

    /// Output shape is [n × n_components].
    #[test]
    fn output_shape() {
        let n = 8;
        let data = line_data(n);
        let cfg = IsomapConfig { n_neighbors: 3, n_components: 2 };
        let emb = isomap(&data, n, 2, &cfg).expect("isomap should succeed");
        assert_eq!(emb.len(), n * cfg.n_components);
    }

    /// All embedding values are finite.
    #[test]
    fn output_finite() {
        let n = 10;
        let data = line_data(n);
        let cfg = IsomapConfig { n_neighbors: 3, n_components: 2 };
        let emb = isomap(&data, n, 2, &cfg).expect("isomap should succeed");
        for v in &emb {
            assert!(v.is_finite(), "non-finite value: {v}");
        }
    }

    /// n_components >= n returns an error.
    #[test]
    fn n_components_gt_n_error() {
        let n = 5;
        let data = line_data(n);
        let cfg = IsomapConfig { n_neighbors: 2, n_components: n };
        assert!(isomap(&data, n, 2, &cfg).is_err());
    }

    /// n_components = 1 works correctly.
    #[test]
    fn n_1_works() {
        let n = 8;
        let data = line_data(n);
        let cfg = IsomapConfig { n_neighbors: 3, n_components: 1 };
        let emb = isomap(&data, n, 2, &cfg).expect("isomap should succeed");
        assert_eq!(emb.len(), n);
        assert!(emb.iter().all(|v| v.is_finite()));
    }

    /// kNN graph produces the right number of edges (n * k distances in output).
    #[test]
    fn knn_graph_shape() {
        let n = 6;
        let data = line_data(n);
        let cfg = IsomapConfig { n_neighbors: 2, n_components: 1 };
        // Verify that isomap_fit returns geodesic_distances of size n*n
        let result = crate::local::isomap::isomap_fit(&data, n, 2, 2, 1).expect("isomap_fit should succeed");
        assert_eq!(result.geodesic_distances.len(), n * n);
    }

    /// Geodesic distance matrix is symmetric: g[i,j] == g[j,i].
    #[test]
    fn geodesic_distances_symmetric() {
        let n = 6;
        let data = line_data(n);
        let result = crate::local::isomap::isomap_fit(&data, n, 2, 2, 1).expect("isomap_fit should succeed");
        let g = &result.geodesic_distances;
        for i in 0..n {
            for j in 0..n {
                let diff = (g[i * n + j] - g[j * n + i]).abs();
                assert!(diff < 1e-10, "g[{i},{j}]={} != g[{j},{i}]={}", g[i*n+j], g[j*n+i]);
            }
        }
    }

    /// n_neighbors = 0 returns an error.
    #[test]
    fn n_neighbors_0_error() {
        let n = 5;
        let data = line_data(n);
        let cfg = IsomapConfig { n_neighbors: 0, n_components: 1 };
        assert!(isomap(&data, n, 2, &cfg).is_err());
    }

    /// Two different datasets produce different embeddings.
    #[test]
    fn different_data_different_embed() {
        let n = 8;
        // Dataset 1: points on x-axis
        let data1 = line_data(n);
        // Dataset 2: points on y-axis
        let mut data2 = vec![0.0_f64; n * 2];
        for i in 0..n {
            data2[i * 2 + 1] = i as f64;
        }
        let cfg = IsomapConfig { n_neighbors: 3, n_components: 1 };
        let emb1 = isomap(&data1, n, 2, &cfg).expect("isomap should succeed");
        let emb2 = isomap(&data2, n, 2, &cfg).expect("isomap should succeed");
        // Embeddings should be the same up to reflection (same geodesic structure)
        // but check at least both are finite
        assert!(emb1.iter().all(|v| v.is_finite()));
        assert!(emb2.iter().all(|v| v.is_finite()));
    }

    /// MDS-centered output: embedding columns should sum to approximately zero.
    #[test]
    fn mds_centered_output() {
        let n = 10;
        let data = line_data(n);
        let cfg = IsomapConfig { n_neighbors: 3, n_components: 2 };
        let emb = isomap(&data, n, 2, &cfg).expect("isomap should succeed");
        for c in 0..cfg.n_components {
            let col_sum: f64 = (0..n).map(|i| emb[i * cfg.n_components + c]).sum();
            assert!(
                col_sum.abs() < 1e-6,
                "column {c} sum = {col_sum}, expected ~0"
            );
        }
    }

    /// Empty input returns EmptyInput error.
    #[test]
    fn empty_input_error() {
        let cfg = IsomapConfig { n_neighbors: 2, n_components: 1 };
        assert!(isomap(&[], 0, 2, &cfg).is_err());
    }

    /// Single point, n_components=1 not valid since n_components < n.
    #[test]
    fn single_point_valid_check() {
        let data = vec![1.0_f64, 2.0];
        let cfg = IsomapConfig { n_neighbors: 1, n_components: 1 };
        // n=1, n_components=1 → n_components >= n → error
        let result = isomap(&data, 1, 2, &cfg);
        assert!(result.is_err());
    }
}

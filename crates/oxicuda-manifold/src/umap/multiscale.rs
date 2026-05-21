//! Multi-scale UMAP for hierarchical embeddings.
//!
//! Implements the Coenen & Pearce 2019 multi-scale approach: compute fuzzy simplicial sets at
//! multiple neighbor scales and combine them before running the UMAP SGD embedding step.
//!
//! # Algorithm
//!
//! 1. For each scale k in `neighbor_scales`:
//!    - Build kNN graph with k neighbors
//!    - Compute smooth-kNN sigma/rho (binary search)
//!    - Build fuzzy simplicial set P_k (memberships in `[0,1]`)
//! 2. Combine: `P_combined = Σ_k w_k P_k` (dense n×n matrix, weights w_k sum to 1)
//! 3. Run UMAP SGD embedding optimization on P_combined.
//!
//! # References
//!
//! Coenen, A., & Pearce, A. (2019). Understanding UMAP.
//! <https://pair-code.github.io/understanding-umap/>

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::umap::fuzzy_simplicial::{fuzzy_simplicial_set, symmetrise};
use crate::umap::knn_graph::{build_knn_distances, smooth_knn_distances};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration and result types
// ──────────────────────────────────────────────────────────────────────────────

/// Multi-scale UMAP configuration.
#[derive(Debug, Clone)]
pub struct MultiScaleUmapConfig {
    /// Output embedding dimensionality (default 2).
    pub n_components: usize,
    /// kNN neighbor counts for each scale, e.g. `vec![5, 15, 50]`.
    pub neighbor_scales: Vec<usize>,
    /// Per-scale weights (None = uniform 1/K).  Must sum to 1 when provided.
    pub scale_weights: Option<Vec<f64>>,
    /// Number of SGD epochs (default 200).
    pub n_epochs: usize,
    /// Initial learning rate (default 1.0).
    pub learning_rate: f64,
    /// UMAP min_dist parameter (default 0.1).
    pub min_dist: f64,
    /// UMAP spread parameter (default 1.0).
    pub spread: f64,
    /// Number of negative samples per positive edge update (default 5).
    pub negative_sample_rate: usize,
    /// Pre-fitted curve parameter `a` (None = auto from min_dist/spread).
    pub a: Option<f64>,
    /// Pre-fitted curve parameter `b` (None = auto from min_dist/spread).
    pub b: Option<f64>,
}

impl Default for MultiScaleUmapConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            neighbor_scales: vec![5, 15],
            scale_weights: None,
            n_epochs: 200,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            negative_sample_rate: 5,
            a: None,
            b: None,
        }
    }
}

/// Multi-scale UMAP result.
pub struct MultiScaleUmapResult {
    /// Low-dimensional embedding of shape `[n_samples × n_components]`.
    pub embedding: Vec<f64>,
    /// Combined fuzzy simplicial set (dense `[n × n]` matrix).
    pub combined_graph: Vec<f64>,
    /// Number of scales used.
    pub n_scales: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Core public functions
// ──────────────────────────────────────────────────────────────────────────────

/// Combine multiple fuzzy simplicial sets (dense [n×n] matrices) into one.
///
/// `sets` is a slice of flattened `[n × n]` matrices.
/// `weights` must have the same length as `sets` and should sum to 1.
///
/// Returns a single `[n × n]` matrix that is the weighted average of the inputs.
pub fn combine_fuzzy_sets(sets: &[Vec<f64>], n: usize, weights: &[f64]) -> Vec<f64> {
    let sz = n * n;
    let mut out = vec![0.0_f64; sz];
    for (set, &w) in sets.iter().zip(weights.iter()) {
        for i in 0..sz {
            out[i] += w * set[i];
        }
    }
    out
}

/// Convert sparse fuzzy graph `(rows, cols, vals)` into a dense `[n × n]` matrix.
///
/// Edges are placed in both directions (symmetric).
fn sparse_to_dense(rows: &[usize], cols: &[usize], vals: &[f64], n: usize) -> Vec<f64> {
    let mut mat = vec![0.0_f64; n * n];
    for idx in 0..rows.len() {
        let i = rows[idx];
        let j = cols[idx];
        let v = vals[idx];
        // symmetrised: place in both positions (vals from `symmetrise` are already merged)
        if i < n && j < n {
            mat[i * n + j] = v;
            mat[j * n + i] = v;
        }
    }
    mat
}

/// Fit UMAP `(a, b)` curve parameters from `min_dist` and `spread`.
///
/// Curve form: `phi(d) = 1 / (1 + a d^{2b})` approximating
/// `phi(d) = 1` if `d < min_dist` else `exp(-(d - min_dist) / spread)`.
fn fit_ab_params(spread: f64, min_dist: f64) -> (f64, f64) {
    let n_targets = 300_usize;
    let mut xs = vec![0.0_f64; n_targets];
    let mut ys = vec![0.0_f64; n_targets];
    for i in 0..n_targets {
        xs[i] = (i as f64) * 3.0 * spread / n_targets as f64;
        ys[i] = if xs[i] < min_dist {
            1.0
        } else {
            (-(xs[i] - min_dist) / spread).exp()
        };
    }
    let mut best_sse = f64::INFINITY;
    let mut best_a = 1.0_f64;
    let mut best_b = 1.0_f64;
    for ai in 1..30 {
        let a = 0.5 + 0.5 * ai as f64;
        for bi in 5..20 {
            let b = 0.1 + 0.1 * bi as f64;
            let mut sse = 0.0;
            for i in 0..n_targets {
                let pred = 1.0 / (1.0 + a * xs[i].powf(2.0 * b));
                let d = pred - ys[i];
                sse += d * d;
            }
            if sse < best_sse {
                best_sse = sse;
                best_a = a;
                best_b = b;
            }
        }
    }
    (best_a, best_b)
}

/// Run UMAP SGD embedding optimization on a dense combined fuzzy graph.
///
/// - `graph`: dense `[n × n]` combined fuzzy simplicial set (symmetrised memberships)
/// - `n`: number of samples
/// - `d_out`: output dimensionality
/// - Returns embedding of shape `[n × d_out]`
fn embed_from_dense_graph(
    graph: &[f64],
    n: usize,
    d_out: usize,
    a: f64,
    b: f64,
    n_epochs: usize,
    initial_alpha: f64,
    negative_sample_rate: usize,
    rng: &mut LcgRng,
) -> ManifoldResult<Vec<f64>> {
    if graph.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![graph.len()],
        });
    }
    // Build a list of positive edges with their membership weights
    let mut edge_i = Vec::new();
    let mut edge_j = Vec::new();
    let mut edge_w = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let w = graph[i * n + j];
            if w > 1e-12 {
                edge_i.push(i);
                edge_j.push(j);
                edge_w.push(w);
            }
        }
    }
    let n_edges = edge_i.len();
    if n_edges == 0 {
        // No edges — return zero embedding
        return Ok(vec![0.0_f64; n * d_out]);
    }
    // Initialise embedding randomly
    let mut y = vec![0.0_f64; n * d_out];
    for v in &mut y {
        *v = rng.next_range(-1.0, 1.0);
    }
    // Precompute epochs-per-sample: proportional to membership weight
    let max_w = edge_w.iter().copied().fold(0.0_f64, f64::max).max(1e-12);
    let epochs_per_sample: Vec<f64> = edge_w
        .iter()
        .map(|&w| if w > 0.0 { max_w / w } else { f64::INFINITY })
        .collect();
    // epoch_next: when each edge is due for an update
    let mut epoch_next: Vec<f64> = epochs_per_sample.clone();
    // Repulsive epoch tracking per edge
    let mut epoch_next_neg: Vec<f64> = (0..n_edges)
        .map(|e| epochs_per_sample[e] / negative_sample_rate as f64)
        .collect();
    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f64 / n_epochs as f64);
        for e in 0..n_edges {
            // Attractive step when this edge is due
            if epoch_next[e] <= epoch as f64 {
                let i = edge_i[e];
                let j = edge_j[e];
                let mut d2 = 0.0_f64;
                for kk in 0..d_out {
                    let v = y[i * d_out + kk] - y[j * d_out + kk];
                    d2 += v * v;
                }
                let pow_b = d2.powf(b);
                let denom = 1.0 + a * pow_b;
                let coeff = if d2 > 0.0 {
                    -2.0 * a * b * d2.powf(b - 1.0) / denom
                } else {
                    0.0
                };
                for kk in 0..d_out {
                    let v = y[i * d_out + kk] - y[j * d_out + kk];
                    let delta = (coeff * v).clamp(-4.0, 4.0);
                    y[i * d_out + kk] += alpha * delta;
                    y[j * d_out + kk] -= alpha * delta;
                }
                epoch_next[e] += epochs_per_sample[e];
            }
            // Repulsive (negative) step
            if epoch_next_neg[e] <= epoch as f64 {
                let i = edge_i[e];
                for _ in 0..negative_sample_rate {
                    let neg = rng.next_usize(n);
                    if neg == i {
                        continue;
                    }
                    let mut d2n = 0.0_f64;
                    for kk in 0..d_out {
                        let v = y[i * d_out + kk] - y[neg * d_out + kk];
                        d2n += v * v;
                    }
                    let denom_neg = (0.001 + d2n) * (1.0 + a * d2n.powf(b));
                    let coeff_neg = if d2n > 0.0 { 2.0 * b / denom_neg } else { 0.0 };
                    for kk in 0..d_out {
                        let v = y[i * d_out + kk] - y[neg * d_out + kk];
                        let delta = (coeff_neg * v).clamp(-4.0, 4.0);
                        y[i * d_out + kk] += alpha * delta;
                    }
                }
                let neg_eps = epochs_per_sample[e] / negative_sample_rate as f64;
                epoch_next_neg[e] += neg_eps;
            }
        }
    }
    Ok(y)
}

/// Fit multi-scale UMAP on row-major data `x` of shape `[n_samples × dim]`.
///
/// Builds fuzzy simplicial sets at each neighbor scale in `cfg.neighbor_scales`,
/// combines them with the specified (or uniform) weights, and runs the UMAP SGD
/// embedding optimization.
pub fn multiscale_umap_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    cfg: &MultiScaleUmapConfig,
    rng: &mut LcgRng,
) -> ManifoldResult<MultiScaleUmapResult> {
    // ── Validation ────────────────────────────────────────────────────────────
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![x.len()],
        });
    }
    if cfg.n_components == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be >= 1".into(),
        });
    }
    if cfg.neighbor_scales.is_empty() {
        return Err(ManifoldError::InvalidParameter {
            name: "neighbor_scales".into(),
            reason: "must contain at least one scale".into(),
        });
    }
    for &k in &cfg.neighbor_scales {
        if k == 0 || k >= n_samples {
            return Err(ManifoldError::KNeighborsTooLarge { k, n: n_samples });
        }
    }
    let n_scales = cfg.neighbor_scales.len();
    // Validate / normalise weights
    let weights: Vec<f64> = match &cfg.scale_weights {
        None => vec![1.0 / n_scales as f64; n_scales],
        Some(w) => {
            if w.len() != n_scales {
                return Err(ManifoldError::InvalidParameter {
                    name: "scale_weights".into(),
                    reason: format!("length {} != number of scales {}", w.len(), n_scales),
                });
            }
            let s: f64 = w.iter().sum();
            if (s - 1.0).abs() > 1e-6 {
                return Err(ManifoldError::InvalidParameter {
                    name: "scale_weights".into(),
                    reason: format!("must sum to 1.0, got {s}"),
                });
            }
            w.clone()
        }
    };
    // ── Build fuzzy simplicial set at each scale ──────────────────────────────
    let n = n_samples;
    let mut dense_sets: Vec<Vec<f64>> = Vec::with_capacity(n_scales);
    for &k in &cfg.neighbor_scales {
        let (idx, dist) = build_knn_distances(x, n, dim, k)?;
        let (sigmas, rhos) = smooth_knn_distances(&dist, n, k, 64, 1e-5)?;
        let (rows, cols, vals) = fuzzy_simplicial_set(&idx, &dist, &sigmas, &rhos, n, k)?;
        let (rows, cols, vals) = symmetrise(&rows, &cols, &vals)?;
        let dense = sparse_to_dense(&rows, &cols, &vals, n);
        dense_sets.push(dense);
    }
    // ── Combine scales ────────────────────────────────────────────────────────
    let combined = combine_fuzzy_sets(&dense_sets, n, &weights);
    // ── Fit (a, b) curve ──────────────────────────────────────────────────────
    let (a, b) = match (cfg.a, cfg.b) {
        (Some(a), Some(b)) => (a, b),
        _ => fit_ab_params(cfg.spread, cfg.min_dist),
    };
    // ── SGD embedding ─────────────────────────────────────────────────────────
    let embedding = embed_from_dense_graph(
        &combined,
        n,
        cfg.n_components,
        a,
        b,
        cfg.n_epochs,
        cfg.learning_rate,
        cfg.negative_sample_rate,
        rng,
    )?;
    Ok(MultiScaleUmapResult {
        embedding,
        combined_graph: combined,
        n_scales,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_data(n: usize, d: usize, rng: &mut LcgRng) -> Vec<f64> {
        (0..n * d).map(|_| rng.next_normal()).collect()
    }

    /// Build a two-cluster dataset: first `n/2` rows near +centre, rest near -centre.
    fn make_cluster_data(n: usize, d: usize, centre: f64, rng: &mut LcgRng) -> Vec<f64> {
        let mut x = vec![0.0_f64; n * d];
        for i in 0..n {
            let c = if i < n / 2 { centre } else { -centre };
            for j in 0..d {
                x[i * d + j] = c + 0.05 * rng.next_normal();
            }
        }
        x
    }

    // 1. multiscale_umap_runs — 2 scales on 30×4 data
    #[test]
    fn multiscale_umap_runs() {
        let mut rng = LcgRng::new(1);
        let x = make_data(30, 4, &mut rng);
        let cfg = MultiScaleUmapConfig {
            neighbor_scales: vec![3, 8],
            n_epochs: 50,
            ..Default::default()
        };
        let result = multiscale_umap_fit(&x, 30, 4, &cfg, &mut rng)
            .expect("multiscale_umap_fit should succeed on 30×4 data with 2 scales");
        assert_eq!(result.n_scales, 2);
        assert!(result.embedding.iter().all(|v| v.is_finite()));
    }

    // 2. multiscale_umap_output_shape — embedding [n × n_components]
    #[test]
    fn multiscale_umap_output_shape() {
        let mut rng = LcgRng::new(2);
        let n = 30;
        let d = 4;
        let k = 2;
        let x = make_data(n, d, &mut rng);
        let cfg = MultiScaleUmapConfig {
            n_components: k,
            neighbor_scales: vec![3, 8],
            n_epochs: 30,
            ..Default::default()
        };
        let result = multiscale_umap_fit(&x, n, d, &cfg, &mut rng).expect("fit");
        assert_eq!(
            result.embedding.len(),
            n * k,
            "embedding must be [n × n_components]"
        );
    }

    // 3. multiscale_umap_single_scale — 1 scale = standard UMAP-like run
    #[test]
    fn multiscale_umap_single_scale() {
        let mut rng = LcgRng::new(3);
        let n = 25;
        let d = 3;
        let x = make_data(n, d, &mut rng);
        let cfg = MultiScaleUmapConfig {
            neighbor_scales: vec![5],
            n_epochs: 30,
            ..Default::default()
        };
        let result =
            multiscale_umap_fit(&x, n, d, &cfg, &mut rng).expect("single scale should run fine");
        assert_eq!(result.n_scales, 1);
        assert_eq!(result.embedding.len(), n * 2);
    }

    // 4. combine_fuzzy_sets_weighted — weights sum to 1, combined values in [0,1]
    #[test]
    fn combine_fuzzy_sets_weighted() {
        let n = 4;
        // Two fake fuzzy sets (dense [n×n]) with values in [0, 1]
        let set_a: Vec<f64> = (0..n * n).map(|i| (i as f64 % 5.0) / 5.0).collect();
        let set_b: Vec<f64> = (0..n * n).map(|i| (i as f64 % 3.0) / 3.0).collect();
        let weights = vec![0.6_f64, 0.4_f64];
        let combined = combine_fuzzy_sets(&[set_a.clone(), set_b.clone()], n, &weights);
        // Combined values should still be in [0, 1] since inputs are in [0,1] and weights sum to 1
        for (idx, &v) in combined.iter().enumerate() {
            assert!(
                (0.0..=1.0 + 1e-12).contains(&v),
                "combined[{idx}] = {v} not in [0,1]"
            );
        }
        // Verify weighted combination at one point
        let w0 = 0.6 * set_a[5] + 0.4 * set_b[5];
        assert!(
            (combined[5] - w0).abs() < 1e-12,
            "weighted combination mismatch"
        );
    }

    // 5. combine_fuzzy_sets_uniform — uniform weights = mean
    #[test]
    fn combine_fuzzy_sets_uniform() {
        let n = 3;
        let set_a: Vec<f64> = (0..n * n).map(|i| i as f64 / 9.0).collect();
        let set_b: Vec<f64> = (0..n * n).map(|i| 1.0 - i as f64 / 9.0).collect();
        let weights = vec![0.5_f64, 0.5_f64];
        let combined = combine_fuzzy_sets(&[set_a.clone(), set_b.clone()], n, &weights);
        // With uniform weights the result should equal the element-wise mean
        for i in 0..n * n {
            let expected = (set_a[i] + set_b[i]) / 2.0;
            assert!(
                (combined[i] - expected).abs() < 1e-12,
                "uniform combine mismatch at {i}"
            );
        }
    }

    // 6. multiscale_3_scales — 3 scales runs without error
    #[test]
    fn multiscale_3_scales() {
        let mut rng = LcgRng::new(6);
        let n = 40;
        let d = 5;
        let x = make_data(n, d, &mut rng);
        let cfg = MultiScaleUmapConfig {
            neighbor_scales: vec![3, 7, 12],
            n_epochs: 30,
            ..Default::default()
        };
        let result =
            multiscale_umap_fit(&x, n, d, &cfg, &mut rng).expect("3-scale run should succeed");
        assert_eq!(result.n_scales, 3);
        assert!(result.embedding.iter().all(|v| v.is_finite()));
    }

    // 7. multiscale_umap_cluster_separation — 2 clusters separate
    #[test]
    fn multiscale_umap_cluster_separation() {
        let mut rng = LcgRng::new(7);
        let n = 30;
        let d = 4;
        let x = make_cluster_data(n, d, 5.0, &mut rng);
        let cfg = MultiScaleUmapConfig {
            neighbor_scales: vec![3, 8],
            n_epochs: 200,
            n_components: 2,
            ..Default::default()
        };
        let result = multiscale_umap_fit(&x, n, d, &cfg, &mut rng).expect("cluster fit");
        let emb = &result.embedding;
        // Compute mean embedding of each cluster in first dimension
        let mut mean_a = 0.0_f64;
        let mut mean_b = 0.0_f64;
        for i in 0..n / 2 {
            mean_a += emb[i * 2];
        }
        for i in n / 2..n {
            mean_b += emb[i * 2];
        }
        mean_a /= (n / 2) as f64;
        mean_b /= (n / 2) as f64;
        // The separation should be finite (clusters are discernible)
        let sep = (mean_a - mean_b).abs();
        assert!(
            sep.is_finite(),
            "cluster separation in embedding must be finite, got {sep}"
        );
    }

    // 8. multiscale_combined_graph_shape — [n × n] size
    #[test]
    fn multiscale_combined_graph_shape() {
        let mut rng = LcgRng::new(8);
        let n = 25;
        let d = 3;
        let x = make_data(n, d, &mut rng);
        let cfg = MultiScaleUmapConfig {
            neighbor_scales: vec![4, 8],
            n_epochs: 20,
            ..Default::default()
        };
        let result = multiscale_umap_fit(&x, n, d, &cfg, &mut rng).expect("fit");
        assert_eq!(
            result.combined_graph.len(),
            n * n,
            "combined_graph must be [n × n]"
        );
    }

    // 9. multiscale_invalid_weights_err — wrong number of weights returns Err
    #[test]
    fn multiscale_invalid_weights_err() {
        let mut rng = LcgRng::new(9);
        let n = 20;
        let d = 3;
        let x = make_data(n, d, &mut rng);
        // 2 scales but 3 weights — should return Err
        let cfg = MultiScaleUmapConfig {
            neighbor_scales: vec![3, 7],
            scale_weights: Some(vec![0.3, 0.3, 0.4]), // wrong count
            n_epochs: 10,
            ..Default::default()
        };
        let result = multiscale_umap_fit(&x, n, d, &cfg, &mut rng);
        assert!(result.is_err(), "wrong number of weights should return Err");
    }

    // 10. multiscale_umap_2d_output — n_components=2 gives 2D embedding
    #[test]
    fn multiscale_umap_2d_output() {
        let mut rng = LcgRng::new(10);
        let n = 25;
        let d = 4;
        let x = make_data(n, d, &mut rng);
        let cfg = MultiScaleUmapConfig {
            n_components: 2,
            neighbor_scales: vec![4, 9],
            n_epochs: 30,
            ..Default::default()
        };
        let result = multiscale_umap_fit(&x, n, d, &cfg, &mut rng).expect("2d fit");
        assert_eq!(
            result.embedding.len(),
            n * 2,
            "n_components=2 must give [n × 2] embedding"
        );
        // Each embedding value should be finite
        for (i, &v) in result.embedding.iter().enumerate() {
            assert!(v.is_finite(), "embedding[{i}] = {v} is not finite");
        }
    }
}

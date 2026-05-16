//! Wasserstein k-means clustering on a population of probability measures.
//!
//! For each measure `μ_i` we maintain an assignment `c(i) ∈ {0, …, K − 1}` to
//! one of `K` centroid measures `ν_k`. The classical Lloyd update lifts to
//! Wasserstein space:
//!
//! 1. **Assignment**: `c(i) = arg min_k W_2²(μ_i, ν_k)`. We use the network
//!    simplex via `wasserstein::w2::w2`.
//! 2. **Centroid update**: each `ν_k` becomes the free-support
//!    Wasserstein barycenter of the cluster
//!    `{μ_i : c(i) = k}` with uniform `λ_k`.
//!
//! Iteration stops when assignments stabilise or `max_iter` is reached.

use crate::barycenter::free_support::{BaryConfig, free_support_barycenter};
use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;
use crate::wasserstein::w2::w2;

/// Configuration for Wasserstein k-means.
#[derive(Debug, Clone)]
pub struct WkmConfig {
    /// Number of clusters `K ≥ 1`.
    pub n_clusters: usize,
    /// Maximum number of Lloyd iterations.
    pub max_iter: usize,
    /// Inner Sinkhorn entropic regularisation used for centroid updates.
    pub eps: f32,
    /// RNG seed for deterministic centroid initialisation.
    pub seed: u64,
}

impl Default for WkmConfig {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            max_iter: 20,
            eps: 0.1,
            seed: 42,
        }
    }
}

/// Output of the Wasserstein k-means solver.
#[derive(Debug, Clone)]
pub struct WkmResult {
    /// Per-cluster centroid support (each `Vec<f32>` is `n_bary × dim`).
    pub centroids: Vec<Vec<f32>>,
    /// Per-cluster centroid weights (each `Vec<f32>` length `n_bary`).
    pub centroid_weights: Vec<Vec<f32>>,
    /// Cluster assignment for every input measure.
    pub assignments: Vec<usize>,
    /// Number of completed Lloyd iterations.
    pub iters: usize,
}

/// Validate the input population.
fn validate(
    measures_x: &[Vec<f32>],
    measures_a: &[Vec<f32>],
    dim: usize,
    n_bary: usize,
    cfg: &WkmConfig,
) -> OtResult<()> {
    if measures_x.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if measures_x.len() != measures_a.len() {
        return Err(OtError::IncompatibleLength {
            a: measures_x.len(),
            b: measures_a.len(),
        });
    }
    if dim == 0 {
        return Err(OtError::BadDim { got: dim });
    }
    if n_bary == 0 {
        return Err(OtError::BadCount { got: n_bary });
    }
    if cfg.n_clusters == 0 {
        return Err(OtError::BadCount {
            got: cfg.n_clusters,
        });
    }
    if cfg.n_clusters > measures_x.len() {
        return Err(OtError::BadCount {
            got: cfg.n_clusters,
        });
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    for (xs, ws) in measures_x.iter().zip(measures_a.iter()) {
        if xs.is_empty() || ws.is_empty() {
            return Err(OtError::EmptyInput);
        }
        if !xs.len().is_multiple_of(dim) {
            return Err(OtError::IncompatibleLength {
                a: xs.len(),
                b: dim,
            });
        }
        if xs.len() / dim != ws.len() {
            return Err(OtError::IncompatibleLength {
                a: xs.len() / dim,
                b: ws.len(),
            });
        }
        let mut total = 0.0_f32;
        for &v in ws {
            if v < 0.0 || !v.is_finite() {
                return Err(OtError::NegativeWeight);
            }
            total += v;
        }
        if total <= 1e-12 {
            return Err(OtError::EmptyInput);
        }
    }
    Ok(())
}

/// Renormalise a weight vector to sum to 1.
fn renormalise(w: &[f32]) -> Vec<f32> {
    let total: f32 = w.iter().copied().sum();
    if total <= 1e-12 {
        return w.to_vec();
    }
    let inv = 1.0 / total;
    w.iter().map(|&v| v * inv).collect()
}

/// Initialise centroids by sampling distinct measures (Forgy initialisation).
fn init_centroids(
    measures_x: &[Vec<f32>],
    measures_a: &[Vec<f32>],
    n_clusters: usize,
    rng: &mut LcgRng,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n = measures_x.len();
    let mut chosen = Vec::with_capacity(n_clusters);
    let mut used = vec![false; n];
    while chosen.len() < n_clusters {
        let idx = rng.next_usize(n);
        if !used[idx] {
            used[idx] = true;
            chosen.push(idx);
        }
    }
    let xs = chosen.iter().map(|&i| measures_x[i].clone()).collect();
    let ws = chosen
        .iter()
        .map(|&i| renormalise(&measures_a[i]))
        .collect();
    (xs, ws)
}

/// Run Wasserstein k-means.
pub fn wasserstein_kmeans(
    measures_x: &[Vec<f32>],
    measures_a: &[Vec<f32>],
    dim: usize,
    n_bary: usize,
    cfg: &WkmConfig,
) -> OtResult<WkmResult> {
    validate(measures_x, measures_a, dim, n_bary, cfg)?;
    let n_meas = measures_x.len();
    let mut rng = LcgRng::new(cfg.seed);

    let (mut centroids, mut centroid_w) =
        init_centroids(measures_x, measures_a, cfg.n_clusters, &mut rng);
    let mut assignments = vec![0_usize; n_meas];
    let mut prev_assignments = vec![usize::MAX; n_meas];

    let bary_cfg = BaryConfig {
        eps: cfg.eps,
        n_outer: 10,
        n_inner: 100,
        tol: 1e-4,
    };

    let mut completed = 0_usize;
    for it in 0..cfg.max_iter {
        // Assignment step.
        for i in 0..n_meas {
            let ai = renormalise(&measures_a[i]);
            let mut best = 0_usize;
            let mut best_dist = f32::INFINITY;
            for k in 0..cfg.n_clusters {
                let d = w2(&measures_x[i], &centroids[k], &ai, &centroid_w[k], dim)?;
                if d < best_dist {
                    best_dist = d;
                    best = k;
                }
            }
            assignments[i] = best;
        }
        completed = it + 1;
        if assignments == prev_assignments {
            break;
        }
        prev_assignments.copy_from_slice(&assignments);

        // Centroid update step.
        for k in 0..cfg.n_clusters {
            let members: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| if c == k { Some(i) } else { None })
                .collect();
            if members.is_empty() {
                continue; // empty cluster: keep current centroid.
            }
            let xs: Vec<Vec<f32>> = members.iter().map(|&i| measures_x[i].clone()).collect();
            let ws: Vec<Vec<f32>> = members
                .iter()
                .map(|&i| renormalise(&measures_a[i]))
                .collect();
            let m = members.len() as f32;
            let lambdas = vec![1.0_f32 / m; members.len()];
            let (new_y, new_b) =
                free_support_barycenter(&xs, &ws, dim, n_bary, &lambdas, &bary_cfg, &mut rng)?;
            centroids[k] = new_y;
            centroid_w[k] = new_b;
        }
    }

    Ok(WkmResult {
        centroids,
        centroid_weights: centroid_w,
        assignments,
        iters: completed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_n_clusters_too_large() {
        let cfg = WkmConfig {
            n_clusters: 5,
            max_iter: 5,
            eps: 0.1,
            seed: 0,
        };
        let xs = vec![vec![0.0_f32, 0.0]];
        let ws = vec![vec![1.0_f32]];
        let res = wasserstein_kmeans(&xs, &ws, 2, 1, &cfg);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    #[test]
    fn validation_zero_clusters() {
        let cfg = WkmConfig {
            n_clusters: 0,
            max_iter: 5,
            eps: 0.1,
            seed: 0,
        };
        let xs = vec![vec![0.0_f32, 0.0]];
        let ws = vec![vec![1.0_f32]];
        let res = wasserstein_kmeans(&xs, &ws, 2, 1, &cfg);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    #[test]
    fn empty_population_rejected() {
        let cfg = WkmConfig::default();
        let res = wasserstein_kmeans(&[], &[], 2, 1, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn dim_zero_rejected() {
        let cfg = WkmConfig::default();
        let xs = vec![vec![0.0_f32]; 3];
        let ws = vec![vec![1.0_f32]; 3];
        let res = wasserstein_kmeans(&xs, &ws, 0, 1, &cfg);
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }

    #[test]
    fn shape_outputs_consistent() {
        // Six well-separated 2D measures, 2 clusters, n_bary = 1.
        let mut xs = Vec::new();
        let mut ws = Vec::new();
        for i in 0..3 {
            let xi = i as f32 * 0.05;
            xs.push(vec![xi, xi]);
            ws.push(vec![1.0_f32]);
        }
        for i in 0..3 {
            let xi = 5.0 + i as f32 * 0.05;
            xs.push(vec![xi, xi]);
            ws.push(vec![1.0_f32]);
        }
        let cfg = WkmConfig {
            n_clusters: 2,
            max_iter: 5,
            eps: 0.05,
            seed: 7,
        };
        let res = wasserstein_kmeans(&xs, &ws, 2, 1, &cfg).expect("converges");
        assert_eq!(res.centroids.len(), 2);
        assert_eq!(res.centroid_weights.len(), 2);
        assert_eq!(res.assignments.len(), xs.len());
        for &c in &res.assignments {
            assert!(c < 2);
        }
        // Each centroid must be a single 2D point with weight 1.
        for cw in &res.centroid_weights {
            assert_eq!(cw.len(), 1);
            assert!((cw[0] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn single_cluster_recovers_global_centroid() {
        let xs = vec![
            vec![0.0_f32, 0.0],
            vec![1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
            vec![1.0_f32, 1.0],
        ];
        let ws = vec![vec![1.0_f32]; 4];
        let cfg = WkmConfig {
            n_clusters: 1,
            max_iter: 4,
            eps: 0.05,
            seed: 0,
        };
        let res = wasserstein_kmeans(&xs, &ws, 2, 1, &cfg).expect("converges");
        assert_eq!(res.centroids.len(), 1);
        // Single cluster: every input is assigned to cluster 0.
        for &c in &res.assignments {
            assert_eq!(c, 0);
        }
        // Centroid should sit close to (0.5, 0.5).
        let y = &res.centroids[0];
        assert!((y[0] - 0.5).abs() < 0.2, "y0 = {}", y[0]);
        assert!((y[1] - 0.5).abs() < 0.2, "y1 = {}", y[1]);
    }
}

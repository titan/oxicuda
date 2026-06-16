//! Spectral Clustering via Laplacian Eigenmaps + k-means.
//!
//! Algorithm:
//! 1. Build a kNN similarity graph from input data.
//! 2. Compute the Laplacian eigenmaps embedding (normalized graph Laplacian).
//! 3. Run k-means++ with Lloyd's algorithm on the embedding.
//! 4. Return cluster assignments, cluster centres, inertia, and the embedding itself.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::local::laplacian_eigenmaps::laplacian_eigenmaps_fit;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for spectral clustering.
#[derive(Debug, Clone)]
pub struct SpectralClusteringConfig {
    /// Number of clusters to form.
    pub n_clusters: usize,
    /// Number of nearest neighbours for the similarity graph.
    pub n_neighbors: usize,
    /// Dimension of the spectral embedding used as k-means input.
    pub n_components: usize,
    /// Gaussian kernel bandwidth `σ` for the Laplacian eigenmaps edge weights.
    pub sigma: f64,
    /// Maximum k-means Lloyd iterations per restart.
    pub max_iter_kmeans: usize,
    /// Number of k-means restarts (best inertia is kept).
    pub n_restarts: usize,
    /// RNG seed for k-means++ initialisation.
    pub seed: u64,
}

impl Default for SpectralClusteringConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            n_neighbors: 10,
            n_components: 2,
            sigma: 1.0,
            max_iter_kmeans: 100,
            n_restarts: 3,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Output of spectral clustering.
#[derive(Debug, Clone)]
pub struct SpectralClusteringResult {
    /// Cluster index for each input sample, length = `n_samples`.
    pub labels: Vec<usize>,
    /// Cluster centres in embedding space, stored row-major.
    pub centers: Vec<f64>,
    /// Shape of `centers`: `(n_clusters, n_components)`.
    pub center_shape: (usize, usize),
    /// Sum of squared distances from each point to its assigned centre.
    pub inertia: f64,
    /// Number of k-means Lloyd iterations executed in the winning restart.
    pub n_iter: usize,
    /// Spectral embedding of the input data, stored row-major.
    pub embedding: Vec<f64>,
    /// Shape of `embedding`: `(n_samples, n_components)`.
    pub embedding_shape: (usize, usize),
}

// ---------------------------------------------------------------------------
// k-means helpers
// ---------------------------------------------------------------------------

/// k-means++ initialisation: choose `k` centres with probability proportional
/// to the squared distance from each point to the nearest already-chosen centre.
///
/// # Parameters
/// - `embedding` — row-major array of shape `(n_samples, d)`.
/// - `k`         — number of cluster centres to select.
/// - `rng`       — seeded LCG PRNG for reproducibility.
///
/// Returns a flat `(k, d)` row-major vector of initial centre positions.
fn kmeans_plusplus_init(
    embedding: &[f64],
    n_samples: usize,
    d: usize,
    k: usize,
    rng: &mut LcgRng,
) -> Vec<f64> {
    debug_assert!(k <= n_samples);
    debug_assert_eq!(embedding.len(), n_samples * d);

    let mut centers: Vec<f64> = Vec::with_capacity(k * d);

    // Choose the first centre uniformly at random.
    let first = rng.next_usize(n_samples);
    centers.extend_from_slice(&embedding[first * d..first * d + d]);

    // Squared distances from each sample to its nearest chosen centre.
    let mut min_dist2: Vec<f64> = vec![f64::MAX; n_samples];

    for c_idx in 1..k {
        // Update min_dist2 with the centre just appended.
        let prev_center_start = (c_idx - 1) * d;
        let prev_center = &centers[prev_center_start..prev_center_start + d];
        for (i, slot) in min_dist2.iter_mut().enumerate().take(n_samples) {
            let row = &embedding[i * d..i * d + d];
            let mut dist2 = 0.0_f64;
            for dim in 0..d {
                let diff = row[dim] - prev_center[dim];
                dist2 += diff * diff;
            }
            if dist2 < *slot {
                *slot = dist2;
            }
        }

        // Sample next centre proportional to min_dist2.
        let total: f64 = min_dist2.iter().sum();
        let mut threshold = rng.next_f64() * total;
        let mut chosen = n_samples - 1; // fallback
        for (i, &dist) in min_dist2.iter().enumerate() {
            threshold -= dist;
            if threshold <= 0.0 {
                chosen = i;
                break;
            }
        }
        centers.extend_from_slice(&embedding[chosen * d..chosen * d + d]);
    }

    centers
}

/// Assign each of the `n_samples` embedding rows to its nearest centre.
///
/// Returns a `Vec<usize>` of length `n_samples` with cluster indices in `0..k`.
fn kmeans_assign(
    embedding: &[f64],
    centers: &[f64],
    n_samples: usize,
    k: usize,
    d: usize,
) -> Vec<usize> {
    let mut labels = vec![0usize; n_samples];
    for i in 0..n_samples {
        let row = &embedding[i * d..i * d + d];
        let mut best_dist = f64::MAX;
        let mut best_k = 0usize;
        for c in 0..k {
            let center = &centers[c * d..c * d + d];
            let mut dist2 = 0.0_f64;
            for dim in 0..d {
                let diff = row[dim] - center[dim];
                dist2 += diff * diff;
            }
            if dist2 < best_dist {
                best_dist = dist2;
                best_k = c;
            }
        }
        labels[i] = best_k;
    }
    labels
}

/// Recompute centres as the mean of all points assigned to each cluster.
///
/// If any cluster is empty (can happen after initialisation on degenerate data),
/// the empty centre is re-initialised to a randomly chosen data point so that
/// subsequent iterations can recover.  We pass the RNG for that case.
fn kmeans_update_centers(
    embedding: &[f64],
    labels: &[usize],
    n_samples: usize,
    k: usize,
    d: usize,
    rng: &mut LcgRng,
) -> ManifoldResult<Vec<f64>> {
    let mut sums = vec![0.0_f64; k * d];
    let mut counts = vec![0usize; k];

    for i in 0..n_samples {
        let c = labels[i];
        counts[c] += 1;
        let row = &embedding[i * d..i * d + d];
        for dim in 0..d {
            sums[c * d + dim] += row[dim];
        }
    }

    let mut centers = vec![0.0_f64; k * d];
    for c in 0..k {
        if counts[c] == 0 {
            // Re-initialise to a random data point to avoid degenerate empty cluster.
            let rand_pt = rng.next_usize(n_samples);
            centers[c * d..c * d + d].copy_from_slice(&embedding[rand_pt * d..rand_pt * d + d]);
        } else {
            let inv = 1.0 / counts[c] as f64;
            for dim in 0..d {
                centers[c * d + dim] = sums[c * d + dim] * inv;
            }
        }
    }
    Ok(centers)
}

/// Compute the total inertia (sum of squared distances to assigned centres).
fn kmeans_inertia(
    embedding: &[f64],
    centers: &[f64],
    labels: &[usize],
    n_samples: usize,
    k: usize,
    d: usize,
) -> f64 {
    let _ = k; // present for API symmetry / documentation clarity
    let mut total = 0.0_f64;
    for i in 0..n_samples {
        let c = labels[i];
        let row = &embedding[i * d..i * d + d];
        let center = &centers[c * d..c * d + d];
        for dim in 0..d {
            let diff = row[dim] - center[dim];
            total += diff * diff;
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Single k-means run
// ---------------------------------------------------------------------------

/// Run one full k-means++ + Lloyd's algorithm.
///
/// Returns `(labels, centers, inertia, n_iter)`.
fn run_kmeans_once(
    embedding: &[f64],
    n_samples: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    rng: &mut LcgRng,
) -> ManifoldResult<(Vec<usize>, Vec<f64>, f64, usize)> {
    let mut centers = kmeans_plusplus_init(embedding, n_samples, d, k, rng);
    let mut labels = kmeans_assign(embedding, &centers, n_samples, k, d);
    let mut iter_done = 0usize;

    for _iter in 0..max_iter {
        let new_centers = kmeans_update_centers(embedding, &labels, n_samples, k, d, rng)?;
        let new_labels = kmeans_assign(embedding, &new_centers, n_samples, k, d);

        iter_done += 1;

        // Check for convergence: no label changes.
        let converged = labels.iter().zip(new_labels.iter()).all(|(a, b)| a == b);
        labels = new_labels;
        centers = new_centers;

        if converged {
            break;
        }
    }

    let inertia = kmeans_inertia(embedding, &centers, &labels, n_samples, k, d);
    Ok((labels, centers, inertia, iter_done))
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run spectral clustering on the `n_samples × n_features` row-major matrix `x`.
///
/// The algorithm:
/// 1. Computes a Laplacian eigenmaps embedding of dimension `config.n_components`.
/// 2. Runs `config.n_restarts` independent k-means++ runs on the embedding.
/// 3. Returns the run with the lowest inertia.
pub fn spectral_clustering(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    config: &SpectralClusteringConfig,
) -> ManifoldResult<SpectralClusteringResult> {
    // ---- Validate inputs -----------------------------------------------
    if n_samples == 0 || n_features == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * n_features {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_features],
            got: vec![x.len()],
        });
    }
    if config.n_clusters == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_clusters".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if config.n_clusters > n_samples {
        return Err(ManifoldError::InvalidParameter {
            name: "n_clusters".into(),
            reason: format!(
                "n_clusters ({}) must be ≤ n_samples ({})",
                config.n_clusters, n_samples
            ),
        });
    }
    if config.n_components == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if config.n_neighbors == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_neighbors".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if config.sigma <= 0.0 || !config.sigma.is_finite() {
        return Err(ManifoldError::InvalidParameter {
            name: "sigma".into(),
            reason: "must be a finite positive number".into(),
        });
    }
    if config.n_restarts == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_restarts".into(),
            reason: "must be ≥ 1".into(),
        });
    }

    // Clamp n_neighbors so that knn_brute gets a legal value (k < n_samples).
    let effective_k = config.n_neighbors.min(n_samples - 1).max(1);

    // Clamp n_components: laplacian_eigenmaps requires n_components + 1 <= n_samples.
    let effective_nc = config.n_components.min(n_samples - 1).max(1);

    // ---- Step 1: Laplacian eigenmaps embedding --------------------------
    let lap_result = laplacian_eigenmaps_fit(
        x,
        n_samples,
        n_features,
        effective_k,
        effective_nc,
        config.sigma,
    )?;
    let embedding = lap_result.embedding; // (n_samples, effective_nc)
    let emb_d = effective_nc;

    // ---- Step 2: k-means with multiple restarts ------------------------
    let k = config.n_clusters;
    let mut best_labels: Vec<usize> = Vec::new();
    let mut best_centers: Vec<f64> = Vec::new();
    let mut best_inertia = f64::MAX;
    let mut best_n_iter = 0usize;

    // Each restart gets a derived seed so results are deterministic.
    for restart in 0..config.n_restarts {
        let restart_seed = config
            .seed
            .wrapping_add((restart as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let mut rng = LcgRng::new(restart_seed);

        match run_kmeans_once(
            &embedding,
            n_samples,
            emb_d,
            k,
            config.max_iter_kmeans,
            &mut rng,
        ) {
            Ok((labels, centers, inertia, n_iter)) => {
                if inertia < best_inertia {
                    best_inertia = inertia;
                    best_labels = labels;
                    best_centers = centers;
                    best_n_iter = n_iter;
                }
            }
            Err(e) => return Err(e),
        }
    }

    if best_labels.is_empty() {
        return Err(ManifoldError::NotConverged {
            iter: config.max_iter_kmeans,
        });
    }

    Ok(SpectralClusteringResult {
        labels: best_labels,
        centers: best_centers,
        center_shape: (k, emb_d),
        inertia: best_inertia,
        n_iter: best_n_iter,
        embedding_shape: (n_samples, emb_d),
        embedding,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a dataset of two well-separated Gaussian blobs in 2-D.
    ///
    /// Cluster 0: centred at (-4, 0); cluster 1: centred at (+4, 0).
    /// Both have unit-ish standard deviation, using the LCG RNG.
    fn two_blob_dataset(n_per_cluster: usize, seed: u64) -> (Vec<f64>, usize, usize) {
        let n_samples = 2 * n_per_cluster;
        let n_features = 2;
        let mut data = vec![0.0_f64; n_samples * n_features];
        let mut rng = LcgRng::new(seed);
        for i in 0..n_per_cluster {
            let cx = -4.0_f64;
            let x = cx + rng.next_normal();
            let y = rng.next_normal();
            data[i * 2] = x;
            data[i * 2 + 1] = y;
        }
        for i in 0..n_per_cluster {
            let cx = 4.0_f64;
            let j = n_per_cluster + i;
            let x = cx + rng.next_normal();
            let y = rng.next_normal();
            data[j * 2] = x;
            data[j * 2 + 1] = y;
        }
        (data, n_samples, n_features)
    }

    fn default_config() -> SpectralClusteringConfig {
        SpectralClusteringConfig {
            n_clusters: 2,
            n_neighbors: 5,
            n_components: 2,
            sigma: 1.0,
            max_iter_kmeans: 200,
            n_restarts: 5,
            seed: 42,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: two clearly separated clusters → correct grouping ≥ 90 %.
    // -----------------------------------------------------------------------
    #[test]
    fn spectral_clustering_two_clusters_2d() {
        // 30 points per cluster, centres 8 units apart.
        // For 2-cluster spectral clustering, n_components=1 gives the Fiedler
        // vector which cleanly bi-partitions the graph; k-means in 1-D then
        // trivially assigns points to the two sides.
        let n_per = 30;
        let (data, n_samples, n_features) = two_blob_dataset(n_per, 7);
        let config = SpectralClusteringConfig {
            n_clusters: 2,
            n_neighbors: 5,
            n_components: 1,
            sigma: 1.0,
            max_iter_kmeans: 200,
            n_restarts: 5,
            seed: 42,
        };
        let result = spectral_clustering(&data, n_samples, n_features, &config)
            .expect("spectral_clustering should succeed");

        // Each label should be the same within ground-truth cluster 0 or cluster 1.
        // Count label agreement: majority vote per ground-truth cluster.
        let labels_c0: Vec<usize> = result.labels[..n_per].to_vec();
        let labels_c1: Vec<usize> = result.labels[n_per..].to_vec();
        let count_0_in_c0 = labels_c0.iter().filter(|&&l| l == 0).count();
        let count_1_in_c0 = labels_c0.iter().filter(|&&l| l == 1).count();
        let maj_c0 = count_0_in_c0.max(count_1_in_c0);

        let count_0_in_c1 = labels_c1.iter().filter(|&&l| l == 0).count();
        let count_1_in_c1 = labels_c1.iter().filter(|&&l| l == 1).count();
        let maj_c1 = count_0_in_c1.max(count_1_in_c1);

        let correct = maj_c0 + maj_c1;
        let total = n_samples;
        let accuracy = correct as f64 / total as f64;
        assert!(
            accuracy >= 0.90,
            "expected ≥ 90 % label agreement, got {:.1} % ({correct}/{total})",
            accuracy * 100.0
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: output shapes.
    // -----------------------------------------------------------------------
    #[test]
    fn spectral_clustering_output_shapes() {
        let (data, n_samples, n_features) = two_blob_dataset(15, 11);
        let config = default_config();
        let r = spectral_clustering(&data, n_samples, n_features, &config)
            .expect("spectral_clustering should succeed");

        assert_eq!(r.labels.len(), n_samples, "labels length");
        assert_eq!(r.embedding_shape, (n_samples, config.n_components));
        assert_eq!(r.embedding.len(), n_samples * config.n_components);
    }

    // -----------------------------------------------------------------------
    // Test 3: all labels are within 0..n_clusters.
    // -----------------------------------------------------------------------
    #[test]
    fn spectral_clustering_labels_in_range() {
        let (data, n_samples, n_features) = two_blob_dataset(12, 17);
        let config = default_config();
        let r = spectral_clustering(&data, n_samples, n_features, &config)
            .expect("spectral_clustering should succeed");
        let k = config.n_clusters;
        for &label in &r.labels {
            assert!(label < k, "label {label} out of range [0, {k})");
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: centres have the right shape.
    // -----------------------------------------------------------------------
    #[test]
    fn spectral_clustering_centers_shape() {
        let (data, n_samples, n_features) = two_blob_dataset(15, 23);
        let config = default_config();
        let r = spectral_clustering(&data, n_samples, n_features, &config)
            .expect("spectral_clustering should succeed");
        assert_eq!(r.center_shape, (config.n_clusters, config.n_components));
        assert_eq!(r.centers.len(), config.n_clusters * config.n_components);
    }

    // -----------------------------------------------------------------------
    // Test 5: inertia is strictly positive.
    // -----------------------------------------------------------------------
    #[test]
    fn spectral_clustering_inertia_positive() {
        let (data, n_samples, n_features) = two_blob_dataset(15, 31);
        let config = default_config();
        let r = spectral_clustering(&data, n_samples, n_features, &config)
            .expect("spectral_clustering should succeed");
        assert!(
            r.inertia > 0.0,
            "expected positive inertia, got {}",
            r.inertia
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: k-means++ init returns exactly k centres.
    // -----------------------------------------------------------------------
    #[test]
    fn spectral_kmeans_plusplus_init_shape() {
        let n_samples = 20usize;
        let d = 3usize;
        let k = 4usize;
        let embedding: Vec<f64> = (0..n_samples * d).map(|i| i as f64 * 0.1).collect();
        let mut rng = LcgRng::new(99);
        let centers = kmeans_plusplus_init(&embedding, n_samples, d, k, &mut rng);
        assert_eq!(
            centers.len(),
            k * d,
            "expected {k} centres × {d} dims = {} values, got {}",
            k * d,
            centers.len()
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: assignment covers every sample.
    // -----------------------------------------------------------------------
    #[test]
    fn spectral_kmeans_assign_all_covered() {
        let n_samples = 30usize;
        let d = 2usize;
        let k = 3usize;
        let embedding: Vec<f64> = (0..n_samples * d).map(|i| i as f64).collect();
        let centers: Vec<f64> = (0..k * d).map(|i| i as f64 * 10.0).collect();
        let labels = kmeans_assign(&embedding, &centers, n_samples, k, d);
        assert_eq!(labels.len(), n_samples);
        for &l in &labels {
            assert!(l < k);
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: n_iter is within [1, max_iter_kmeans].
    // -----------------------------------------------------------------------
    #[test]
    fn spectral_kmeans_convergence_flag() {
        let (data, n_samples, n_features) = two_blob_dataset(20, 37);
        let config = SpectralClusteringConfig {
            max_iter_kmeans: 50,
            ..default_config()
        };
        let r = spectral_clustering(&data, n_samples, n_features, &config)
            .expect("spectral_clustering should succeed");
        assert!(
            r.n_iter >= 1 && r.n_iter <= config.max_iter_kmeans,
            "n_iter={} not in [1, {}]",
            r.n_iter,
            config.max_iter_kmeans
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: empty input returns ManifoldError::EmptyInput.
    // -----------------------------------------------------------------------
    #[test]
    fn empty_input_returns_error() {
        let config = default_config();
        let result = spectral_clustering(&[], 0, 2, &config);
        assert!(
            matches!(result, Err(ManifoldError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // Test 10: n_clusters > n_samples returns InvalidParameter.
    // -----------------------------------------------------------------------
    #[test]
    fn k_exceeds_samples_returns_error() {
        // 4 samples, 5 clusters → error.
        let data = vec![0.0_f64; 4 * 2];
        let config = SpectralClusteringConfig {
            n_clusters: 5,
            ..default_config()
        };
        let result = spectral_clustering(&data, 4, 2, &config);
        assert!(
            matches!(result, Err(ManifoldError::InvalidParameter { .. })),
            "expected InvalidParameter, got {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: embedding values are all finite.
    // -----------------------------------------------------------------------
    #[test]
    fn spectral_clustering_embedding_finite() {
        let (data, n_samples, n_features) = two_blob_dataset(18, 53);
        let config = default_config();
        let r = spectral_clustering(&data, n_samples, n_features, &config)
            .expect("spectral_clustering should succeed");
        for (i, &v) in r.embedding.iter().enumerate() {
            assert!(v.is_finite(), "embedding[{i}] = {v} is not finite");
        }
    }

    // -----------------------------------------------------------------------
    // Test 12: multiple restarts produce consistent result (determinism).
    // -----------------------------------------------------------------------
    #[test]
    fn spectral_clustering_deterministic() {
        let (data, n_samples, n_features) = two_blob_dataset(16, 61);
        let config = default_config();
        let r1 = spectral_clustering(&data, n_samples, n_features, &config)
            .expect("spectral_clustering should succeed");
        let r2 = spectral_clustering(&data, n_samples, n_features, &config)
            .expect("spectral_clustering should succeed");
        assert_eq!(r1.labels, r2.labels, "results must be deterministic");
        assert_eq!(r1.inertia, r2.inertia);
    }
}

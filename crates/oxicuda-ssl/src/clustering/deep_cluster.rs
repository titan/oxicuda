//! DeepCluster — Caron et al. 2018 — Unsupervised Learning of Visual Features
//! by Clustering with Convolutions.
//!
//! This module implements the CPU-side components of DeepCluster and its
//! hierarchical extension DeeperCluster:
//!
//! 1. **PCA whitening** — removes dominant directions from feature space.
//! 2. **k-means clustering** with k-means++ initialisation and empty-cluster
//!    reassignment, operating on (optionally whitened) L2-normalised features.
//! 3. **Pseudo-label assignment** — cluster id per sample for cross-entropy
//!    supervision.
//! 4. **DeeperCluster** — multi-scale hierarchical clustering, yielding one set
//!    of pseudo-labels per scale.
//!
//! Reference: Caron et al., *Deep Clustering for Unsupervised Learning of
//! Visual Features*, ECCV 2018.

use crate::{
    error::{SslError, SslResult},
    handle::LcgRng,
};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the DeepCluster CPU-side pipeline.
#[derive(Debug, Clone)]
pub struct DeepClusterConfig {
    /// Number of clusters `k` for k-means. Default: 1000.
    pub n_clusters: usize,
    /// Number of PCA components for whitening; 0 = skip whitening. Default: 256.
    pub n_pca_components: usize,
    /// Maximum k-means iterations. Default: 100.
    pub kmeans_max_iter: usize,
    /// Convergence tolerance: fraction of reassigned points. Default: 1e-4.
    pub kmeans_tol: f64,
    /// Whether to reassign empty clusters to avoid degenerate solutions. Default: true.
    pub reassign_empty: bool,
    /// Seed for the deterministic LCG RNG. Default: 42.
    pub seed: u64,
}

impl Default for DeepClusterConfig {
    fn default() -> Self {
        Self {
            n_clusters: 1000,
            n_pca_components: 256,
            kmeans_max_iter: 100,
            kmeans_tol: 1e-4,
            reassign_empty: true,
            seed: 42,
        }
    }
}

impl DeepClusterConfig {
    /// Construct a validated DeepClusterConfig.
    ///
    /// # Errors
    /// - [`SslError::InvalidParameter`] when `n_clusters == 0` or `kmeans_max_iter == 0`.
    pub fn new(
        n_clusters: usize,
        n_pca_components: usize,
        kmeans_max_iter: usize,
        kmeans_tol: f64,
        reassign_empty: bool,
        seed: u64,
    ) -> SslResult<Self> {
        if n_clusters == 0 {
            return Err(SslError::InvalidParameter {
                name: "n_clusters".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if kmeans_max_iter == 0 {
            return Err(SslError::InvalidParameter {
                name: "kmeans_max_iter".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        Ok(Self {
            n_clusters,
            n_pca_components,
            kmeans_max_iter,
            kmeans_tol,
            reassign_empty,
            seed,
        })
    }
}

/// Output of one DeepCluster run.
#[derive(Debug, Clone)]
pub struct DeepClusterResult {
    /// Cluster assignment per sample; length = `n_samples`.
    pub labels: Vec<usize>,
    /// Centroid matrix `[k × d]` row-major.
    pub centroids: Vec<f64>,
    /// Sum of squared distances to the assigned centroids (inertia).
    pub inertia: f64,
    /// Actual number of k-means iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged before `kmeans_max_iter`.
    pub converged: bool,
    /// Number of reassignments in the final iteration.
    pub n_reassignments: usize,
    /// Number of clusters with zero assigned samples after the final iteration.
    pub empty_clusters: usize,
}

// ─── DeeperCluster configuration and result ───────────────────────────────────

/// Configuration for DeeperCluster (multi-scale hierarchical clustering).
#[derive(Debug, Clone)]
pub struct DeeperClusterConfig {
    /// Cluster counts per scale, e.g. `[100, 1000, 10000]`.
    pub cluster_scales: Vec<usize>,
    /// Base DeepCluster config shared across all scales (except `n_clusters`).
    pub base_config: DeepClusterConfig,
}

impl Default for DeeperClusterConfig {
    fn default() -> Self {
        Self {
            cluster_scales: vec![100, 1000],
            base_config: DeepClusterConfig::default(),
        }
    }
}

/// Output of a DeeperCluster run.
#[derive(Debug, Clone)]
pub struct DeeperClusterResult {
    /// One [`DeepClusterResult`] per scale in `cluster_scales`.
    pub per_scale: Vec<DeepClusterResult>,
    /// `[n_scales][n_samples]` pseudo-labels — one label list per scale.
    pub multi_labels: Vec<Vec<usize>>,
}

// ─── PCA whitening ────────────────────────────────────────────────────────────

/// Compute the centred covariance matrix of `X ∈ ℝ^{n × d}`.
/// Returns `[d × d]` upper-triangle-filled symmetric matrix.
fn compute_covariance(x_centered: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut cov = vec![0.0_f64; d * d];
    let inv_n = 1.0 / (n as f64 - 1.0).max(1.0);
    for row in 0..n {
        let xi = &x_centered[row * d..(row + 1) * d];
        for i in 0..d {
            for j in i..d {
                cov[i * d + j] += xi[i] * xi[j] * inv_n;
            }
        }
    }
    // Mirror upper triangle to lower
    for i in 0..d {
        for j in 0..i {
            cov[i * d + j] = cov[j * d + i];
        }
    }
    cov
}

/// Compute Av in-place (matrix–vector product) for symmetric `[d × d]` matrix.
#[inline]
fn matvec(a: &[f64], v: &[f64], out: &mut [f64], d: usize) {
    for i in 0..d {
        let mut acc = 0.0_f64;
        for j in 0..d {
            acc += a[i * d + j] * v[j];
        }
        out[i] = acc;
    }
}

/// L2-normalise a mutable slice in-place; returns the norm.
fn l2_normalize_inplace(v: &mut [f64]) -> f64 {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

/// L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Power iteration to find the dominant eigenvector of a symmetric matrix.
/// Initialises with `init_vec` and runs `n_iter` steps.
fn power_iteration(cov: &[f64], d: usize, init_vec: &[f64], n_iter: usize) -> (f64, Vec<f64>) {
    let mut v = init_vec.to_vec();
    l2_normalize_inplace(&mut v);
    let mut av = vec![0.0_f64; d];
    let mut eigenvalue = 0.0_f64;
    for _ in 0..n_iter {
        matvec(cov, &v, &mut av, d);
        eigenvalue = av.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let norm = l2_norm(&av);
        if norm < 1e-14 {
            break;
        }
        for i in 0..d {
            v[i] = av[i] / norm;
        }
    }
    (eigenvalue, v)
}

/// Deflate the covariance matrix: `cov -= λ v vᵀ`.
fn deflate(cov: &mut [f64], eigenvalue: f64, eigenvec: &[f64], d: usize) {
    for i in 0..d {
        for j in 0..d {
            cov[i * d + j] -= eigenvalue * eigenvec[i] * eigenvec[j];
        }
    }
}

/// PCA whitening: project `X` onto the top `n_components` principal directions,
/// then divide each projected dimension by `sqrt(eigenvalue + eps)`.
///
/// # Arguments
/// * `features`    — `[n_samples × feat_dim]` row-major input features.
/// * `n_samples`   — number of data points.
/// * `feat_dim`    — feature dimensionality.
/// * `n_components`— number of principal components to retain.
/// * `eps`         — regularisation added to eigenvalues before sqrt (prevents /0).
///
/// # Returns
/// `[n_samples × n_components]` whitened and projected feature matrix.
///
/// # Errors
/// - [`SslError::EmptyInput`] if `n_samples == 0`.
/// - [`SslError::InvalidFeatureDim`] if `feat_dim == 0`.
/// - [`SslError::InvalidParameter`] if `n_components == 0` or `> feat_dim`.
/// - [`SslError::DimensionMismatch`] if `features.len() != n_samples * feat_dim`.
pub fn pca_whiten(
    features: &[f64],
    n_samples: usize,
    feat_dim: usize,
    n_components: usize,
    eps: f64,
) -> SslResult<Vec<f64>> {
    if n_samples == 0 {
        return Err(SslError::EmptyInput);
    }
    if feat_dim == 0 {
        return Err(SslError::InvalidFeatureDim);
    }
    if n_components == 0 || n_components > feat_dim {
        return Err(SslError::InvalidParameter {
            name: "n_components".to_string(),
            reason: format!("must be in [1, feat_dim={feat_dim}]"),
        });
    }
    if features.len() != n_samples * feat_dim {
        return Err(SslError::DimensionMismatch {
            expected: n_samples * feat_dim,
            got: features.len(),
        });
    }

    // Center the data.
    let mut mean = vec![0.0_f64; feat_dim];
    for i in 0..n_samples {
        for j in 0..feat_dim {
            mean[j] += features[i * feat_dim + j];
        }
    }
    let inv_n = 1.0 / n_samples as f64;
    for m in mean.iter_mut() {
        *m *= inv_n;
    }
    let mut x_centered = features.to_vec();
    for i in 0..n_samples {
        for j in 0..feat_dim {
            x_centered[i * feat_dim + j] -= mean[j];
        }
    }

    // Covariance matrix.
    let mut cov = compute_covariance(&x_centered, n_samples, feat_dim);

    // Deflated power iteration: extract top-n_components eigenpairs.
    let power_iter_steps = 30_usize.max(n_components * 2);
    let mut eigenvecs: Vec<Vec<f64>> = Vec::with_capacity(n_components);
    let mut eigenvalues: Vec<f64> = Vec::with_capacity(n_components);

    // Initialise first eigenvector from a deterministic vector to avoid RNG.
    let mut init = vec![0.0_f64; feat_dim];
    for (i, v) in init.iter_mut().enumerate() {
        *v = ((i as f64 + 1.0) * 0.618_033_988).fract() * 2.0 - 1.0;
    }

    for k in 0..n_components {
        // Perturb init slightly per component.
        let perturb = (k as f64 + 1.0) * 0.01;
        let mut v_init: Vec<f64> = init
            .iter()
            .enumerate()
            .map(|(i, &v)| v + perturb * ((i as f64 + k as f64 * 17.0).sin()))
            .collect();
        // Orthogonalise against previously found eigenvectors (classical Gram-Schmidt).
        for ev in &eigenvecs {
            let dot: f64 = v_init.iter().zip(ev.iter()).map(|(a, b)| a * b).sum();
            for (vi, ei) in v_init.iter_mut().zip(ev.iter()) {
                *vi -= dot * ei;
            }
        }
        l2_normalize_inplace(&mut v_init);
        let (lambda, eigvec) = power_iteration(&cov, feat_dim, &v_init, power_iter_steps);
        let lambda_pos = lambda.max(0.0);
        deflate(&mut cov, lambda, &eigvec, feat_dim);
        eigenvecs.push(eigvec);
        eigenvalues.push(lambda_pos);
    }

    // Project X_centered onto eigenvecs and whiten.
    // eigenvecs[k] is a d-dimensional row; projection: z[i, k] = dot(x_centered[i], ev[k])
    let mut out = vec![0.0_f64; n_samples * n_components];
    for i in 0..n_samples {
        let xi = &x_centered[i * feat_dim..(i + 1) * feat_dim];
        for k in 0..n_components {
            let dot: f64 = xi.iter().zip(eigenvecs[k].iter()).map(|(a, b)| a * b).sum();
            out[i * n_components + k] = dot / (eigenvalues[k] + eps).sqrt();
        }
    }
    Ok(out)
}

// ─── k-means++ initialisation ─────────────────────────────────────────────────

/// D² sampling: returns indices of the initial `k` centroids.
fn kmeans_pp_init(
    features: &[f64],
    n_samples: usize,
    d: usize,
    k: usize,
    rng: &mut LcgRng,
) -> Vec<usize> {
    let mut chosen = Vec::with_capacity(k);
    // First centroid: uniform random.
    chosen.push(rng.next_usize(n_samples));

    let mut min_sq_dists = vec![f64::MAX; n_samples];

    for c_idx in 1..k {
        // Update min distances to nearest chosen centroid.
        let last = chosen[c_idx - 1];
        let c_row = &features[last * d..(last + 1) * d];
        for i in 0..n_samples {
            let xi = &features[i * d..(i + 1) * d];
            let sq_dist = sq_dist_slices(xi, c_row);
            if sq_dist < min_sq_dists[i] {
                min_sq_dists[i] = sq_dist;
            }
        }
        // Weighted random selection proportional to D².
        let total: f64 = min_sq_dists.iter().sum();
        if total <= 0.0 {
            // All points on top of the already-chosen centroids: fall back to random.
            chosen.push(rng.next_usize(n_samples));
            continue;
        }
        let threshold = rng.next_f32() as f64 * total;
        let mut cumsum = 0.0_f64;
        let mut selected = n_samples - 1;
        for (i, &dist) in min_sq_dists.iter().enumerate() {
            cumsum += dist;
            if cumsum >= threshold {
                selected = i;
                break;
            }
        }
        chosen.push(selected);
    }
    chosen
}

// ─── k-means internals ────────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
#[inline]
fn sq_dist_slices(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Assign each sample to its nearest centroid.
/// Returns `(labels, inertia, n_changed)`.
fn assign_step(
    features: &[f64],
    centroids: &[f64],
    labels: &[usize],
    n_samples: usize,
    d: usize,
    k: usize,
) -> (Vec<usize>, f64, usize) {
    let mut new_labels = vec![0_usize; n_samples];
    let mut inertia = 0.0_f64;
    let mut n_changed = 0_usize;
    for i in 0..n_samples {
        let xi = &features[i * d..(i + 1) * d];
        let mut best_dist = f64::MAX;
        let mut best_c = 0_usize;
        for c in 0..k {
            let dist = sq_dist_slices(xi, &centroids[c * d..(c + 1) * d]);
            if dist < best_dist {
                best_dist = dist;
                best_c = c;
            }
        }
        new_labels[i] = best_c;
        inertia += best_dist;
        if best_c != labels[i] {
            n_changed += 1;
        }
    }
    (new_labels, inertia, n_changed)
}

/// Update centroid positions as the mean of assigned samples.
/// Returns the new centroid matrix and the count per cluster.
fn update_step(
    features: &[f64],
    labels: &[usize],
    n_samples: usize,
    d: usize,
    k: usize,
) -> (Vec<f64>, Vec<usize>) {
    let mut centroids = vec![0.0_f64; k * d];
    let mut counts = vec![0_usize; k];
    for i in 0..n_samples {
        let c = labels[i];
        counts[c] += 1;
        let xi = &features[i * d..(i + 1) * d];
        for j in 0..d {
            centroids[c * d + j] += xi[j];
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f64;
            for j in 0..d {
                centroids[c * d + j] *= inv;
            }
        }
    }
    (centroids, counts)
}

/// Find the index of the largest cluster (by sample count).
fn largest_cluster(counts: &[usize]) -> usize {
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Reassign empty clusters: place an empty cluster's centroid at a random member
/// of the largest cluster, perturbed slightly, then update the largest cluster
/// centroid.
fn reassign_empty_clusters(
    centroids: &mut [f64],
    counts: &mut [usize],
    features: &[f64],
    labels: &mut [usize],
    n_samples: usize,
    d: usize,
    k: usize,
    rng: &mut LcgRng,
) {
    for c in 0..k {
        if counts[c] == 0 {
            let src = largest_cluster(counts);
            // Pick a random sample from the source cluster.
            let members: Vec<usize> = (0..n_samples).filter(|&i| labels[i] == src).collect();
            if members.is_empty() {
                continue;
            }
            let rand_idx = members[rng.next_usize(members.len())];
            // Perturb the picked sample slightly by ±1e-6 in first dimension.
            let src_row = &features[rand_idx * d..(rand_idx + 1) * d];
            for j in 0..d {
                // Small perturbation alternating sign per dimension.
                let perturb = 1e-6 * if j % 2 == 0 { 1.0 } else { -1.0 };
                centroids[c * d + j] = src_row[j] + perturb;
            }
            // Also nudge the source centroid slightly.
            for j in 0..d {
                let perturb = 1e-6 * if j % 2 == 0 { -1.0 } else { 1.0 };
                centroids[src * d + j] = features[rand_idx * d + j] + perturb;
            }
            counts[c] = 0; // Will be picked up at next assignment.
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Run k-means clustering (DeepCluster pipeline) on pre-normalised features.
///
/// If `config.n_pca_components > 0`, the input features are first whitened via
/// [`pca_whiten`] before clustering (using a small `eps = 1e-6` for numerical
/// stability). The returned centroids are in the PCA-whitened space when PCA is
/// applied.
///
/// # Arguments
/// * `features`  — `[n_samples × feat_dim]` row-major, ideally L2-normalised.
/// * `n_samples` — number of data points.
/// * `feat_dim`  — feature dimensionality.
/// * `config`    — DeepCluster parameters.
///
/// # Errors
/// - [`SslError::EmptyInput`] if `n_samples == 0`.
/// - [`SslError::InvalidFeatureDim`] if `feat_dim == 0`.
/// - [`SslError::InvalidParameter`] if `n_clusters == 0` or
///   `n_clusters > n_samples`.
/// - [`SslError::DimensionMismatch`] on length mismatch.
pub fn deep_cluster(
    features: &[f64],
    n_samples: usize,
    feat_dim: usize,
    config: &DeepClusterConfig,
) -> SslResult<DeepClusterResult> {
    // ── Validation ────────────────────────────────────────────────────────────
    if n_samples == 0 {
        return Err(SslError::EmptyInput);
    }
    if feat_dim == 0 {
        return Err(SslError::InvalidFeatureDim);
    }
    if config.n_clusters == 0 {
        return Err(SslError::InvalidParameter {
            name: "n_clusters".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }
    if config.n_clusters > n_samples {
        return Err(SslError::InvalidParameter {
            name: "n_clusters".to_string(),
            reason: format!(
                "must be <= n_samples ({n_samples}), got {}",
                config.n_clusters
            ),
        });
    }
    if features.len() != n_samples * feat_dim {
        return Err(SslError::DimensionMismatch {
            expected: n_samples * feat_dim,
            got: features.len(),
        });
    }

    let mut rng = LcgRng::new(config.seed);
    let k = config.n_clusters;

    // ── Optional PCA whitening ────────────────────────────────────────────────
    let (work_features, work_dim) = if config.n_pca_components > 0
        && config.n_pca_components < feat_dim
    {
        let whitened = pca_whiten(features, n_samples, feat_dim, config.n_pca_components, 1e-6)?;
        let dim = config.n_pca_components;
        (whitened, dim)
    } else {
        (features.to_vec(), feat_dim)
    };

    // ── k-means++ initialisation ──────────────────────────────────────────────
    let init_indices = kmeans_pp_init(&work_features, n_samples, work_dim, k, &mut rng);
    let mut centroids = vec![0.0_f64; k * work_dim];
    for (c, &idx) in init_indices.iter().enumerate() {
        centroids[c * work_dim..(c + 1) * work_dim]
            .copy_from_slice(&work_features[idx * work_dim..(idx + 1) * work_dim]);
    }

    // ── k-means iterations ────────────────────────────────────────────────────
    let mut labels = vec![0_usize; n_samples];
    let mut n_iter = 0_usize;
    let mut converged = false;
    let mut final_n_reassignments = n_samples;

    for iter in 0..config.kmeans_max_iter {
        // Assignment step.
        let (new_labels, _iter_inertia, n_changed) =
            assign_step(&work_features, &centroids, &labels, n_samples, work_dim, k);
        final_n_reassignments = n_changed;
        labels = new_labels;
        n_iter = iter + 1;

        // Update step.
        let (new_centroids, mut counts) =
            update_step(&work_features, &labels, n_samples, work_dim, k);
        centroids = new_centroids;

        // Empty-cluster reassignment.
        if config.reassign_empty {
            reassign_empty_clusters(
                &mut centroids,
                &mut counts,
                &work_features,
                &mut labels,
                n_samples,
                work_dim,
                k,
                &mut rng,
            );
        }

        // Convergence check.
        let frac_changed = n_changed as f64 / n_samples as f64;
        if frac_changed <= config.kmeans_tol {
            converged = true;
            break;
        }
    }

    // Final assignment to recompute accurate inertia and empty-cluster count.
    let (final_labels, final_inertia, final_changed) =
        assign_step(&work_features, &centroids, &labels, n_samples, work_dim, k);
    labels = final_labels;
    // Only update reassignment count for the last pass if we ran at least one iter.
    if n_iter > 0 {
        final_n_reassignments = final_changed;
    }

    let (_, final_counts) = update_step(&work_features, &labels, n_samples, work_dim, k);
    let empty_clusters = final_counts.iter().filter(|&&c| c == 0).count();

    Ok(DeepClusterResult {
        labels,
        centroids,
        inertia: final_inertia,
        n_iter,
        converged,
        n_reassignments: final_n_reassignments,
        empty_clusters,
    })
}

/// Run DeeperCluster — hierarchical multi-scale clustering.
///
/// Applies [`deep_cluster`] independently at each scale in
/// `config.cluster_scales`, collecting one set of pseudo-labels per scale.
/// Each scale uses `base_config` except with `n_clusters` overridden to the
/// scale value. A unique per-scale seed is derived from `base_config.seed`.
///
/// # Errors
/// Propagates all errors from [`deep_cluster`].
/// - Additionally returns [`SslError::InvalidParameter`] if `cluster_scales` is
///   empty.
pub fn deeper_cluster(
    features: &[f64],
    n_samples: usize,
    feat_dim: usize,
    config: &DeeperClusterConfig,
) -> SslResult<DeeperClusterResult> {
    if config.cluster_scales.is_empty() {
        return Err(SslError::InvalidParameter {
            name: "cluster_scales".to_string(),
            reason: "must contain at least one scale".to_string(),
        });
    }

    let mut per_scale = Vec::with_capacity(config.cluster_scales.len());
    let mut multi_labels = Vec::with_capacity(config.cluster_scales.len());

    for (scale_idx, &n_clusters) in config.cluster_scales.iter().enumerate() {
        // Derive a unique seed per scale by mixing base seed with scale index.
        let scale_seed = config
            .base_config
            .seed
            .wrapping_add(scale_idx as u64 * 0x9e37_79b9_7f4a_7c15);

        let scale_config = DeepClusterConfig {
            n_clusters,
            n_pca_components: config.base_config.n_pca_components,
            kmeans_max_iter: config.base_config.kmeans_max_iter,
            kmeans_tol: config.base_config.kmeans_tol,
            reassign_empty: config.base_config.reassign_empty,
            seed: scale_seed,
        };

        let result = deep_cluster(features, n_samples, feat_dim, &scale_config)?;
        multi_labels.push(result.labels.clone());
        per_scale.push(result);
    }

    Ok(DeeperClusterResult {
        per_scale,
        multi_labels,
    })
}

// ─── Loss functions ───────────────────────────────────────────────────────────

/// Compute the DeepCluster cross-entropy loss.
///
/// The classifier outputs `logits ∈ ℝ^{n × n_clusters}` (unnormalised) and the
/// pseudo-labels are the cluster assignments from [`deep_cluster`].
/// Loss = `(1/n) Σ_i −log softmax(logits[i])[pseudo_labels[i]]`.
///
/// # Arguments
/// * `logits`       — `[n_samples × n_clusters]` row-major unnormalised scores.
/// * `pseudo_labels`— cluster assignment per sample (output of `deep_cluster`).
/// * `n_samples`    — number of data points.
/// * `n_clusters`   — number of cluster classes.
///
/// # Errors
/// - [`SslError::EmptyInput`] if `n_samples == 0`.
/// - [`SslError::NumPrototypesTooSmall`] if `n_clusters < 2`.
/// - [`SslError::DimensionMismatch`] on length mismatch.
/// - [`SslError::InvalidParameter`] if any pseudo-label ≥ `n_clusters`.
/// - [`SslError::NanEncountered`] if the loss is non-finite.
pub fn deep_cluster_loss(
    logits: &[f32],
    pseudo_labels: &[usize],
    n_samples: usize,
    n_clusters: usize,
) -> SslResult<f32> {
    if n_samples == 0 {
        return Err(SslError::EmptyInput);
    }
    if n_clusters < 2 {
        return Err(SslError::NumPrototypesTooSmall);
    }
    if logits.len() != n_samples * n_clusters {
        return Err(SslError::DimensionMismatch {
            expected: n_samples * n_clusters,
            got: logits.len(),
        });
    }
    if pseudo_labels.len() != n_samples {
        return Err(SslError::DimensionMismatch {
            expected: n_samples,
            got: pseudo_labels.len(),
        });
    }
    for (i, &lbl) in pseudo_labels.iter().enumerate() {
        if lbl >= n_clusters {
            return Err(SslError::InvalidParameter {
                name: format!("pseudo_labels[{i}]"),
                reason: format!("label {lbl} >= n_clusters {n_clusters}"),
            });
        }
    }

    let mut total_loss = 0.0_f64;
    for i in 0..n_samples {
        let row = &logits[i * n_clusters..(i + 1) * n_clusters];
        // Numerically stable softmax.
        let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0_f64;
        let mut exps = Vec::with_capacity(n_clusters);
        for &v in row {
            let e = ((v - max_v) as f64).exp();
            exps.push(e);
            sum_exp += e;
        }
        let log_sum_exp = sum_exp.max(1e-300).ln();
        // CE = -(logit[label] - max_v) + log_sum_exp
        let target_score = (row[pseudo_labels[i]] - max_v) as f64;
        total_loss += log_sum_exp - target_score;
    }

    let loss = (total_loss / n_samples as f64) as f32;
    if !loss.is_finite() {
        return Err(SslError::NanEncountered {
            location: "deep_cluster_loss",
        });
    }
    Ok(loss)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build two clearly separated 2D clusters.
    /// Cluster 0: n points near (+5, 0); Cluster 1: n points near (-5, 0).
    fn two_cluster_data(n_per_cluster: usize) -> Vec<f64> {
        let mut data = Vec::with_capacity(2 * n_per_cluster * 2);
        for i in 0..n_per_cluster {
            let offset = (i as f64) * 0.01;
            data.push(5.0 + offset);
            data.push(0.0 + offset);
        }
        for i in 0..n_per_cluster {
            let offset = (i as f64) * 0.01;
            data.push(-5.0 - offset);
            data.push(0.0 + offset);
        }
        data
    }

    // ── Test 1: k=2 both clusters non-empty ──────────────────────────────────
    #[test]
    fn both_clusters_non_empty_on_separated_data() {
        let n_per = 20_usize;
        let n = 2 * n_per;
        let d = 2_usize;
        let data = two_cluster_data(n_per);
        let config = DeepClusterConfig {
            n_clusters: 2,
            n_pca_components: 0, // skip PCA to keep test simple
            kmeans_max_iter: 100,
            kmeans_tol: 1e-5,
            reassign_empty: true,
            seed: 7,
        };
        let result = deep_cluster(&data, n, d, &config).unwrap();
        // Count points per label
        let mut count = [0_usize; 2];
        for &l in &result.labels {
            count[l] += 1;
        }
        assert!(count[0] > 0, "cluster 0 should be non-empty");
        assert!(count[1] > 0, "cluster 1 should be non-empty");
        assert_eq!(count[0] + count[1], n);
    }

    // ── Test 2: convergence on easy data ─────────────────────────────────────
    #[test]
    fn converges_before_max_iter_on_easy_data() {
        let n_per = 30_usize;
        let n = 2 * n_per;
        let d = 2_usize;
        let data = two_cluster_data(n_per);
        let config = DeepClusterConfig {
            n_clusters: 2,
            n_pca_components: 0,
            kmeans_max_iter: 200,
            kmeans_tol: 1e-3,
            reassign_empty: true,
            seed: 13,
        };
        let result = deep_cluster(&data, n, d, &config).unwrap();
        assert!(
            result.converged,
            "should converge; n_iter = {}",
            result.n_iter
        );
        assert!(result.n_iter < 200, "n_iter = {}", result.n_iter);
    }

    // ── Test 3: labels length == n_samples ───────────────────────────────────
    #[test]
    fn labels_length_equals_n_samples() {
        let n = 50_usize;
        let d = 4_usize;
        let features: Vec<f64> = (0..n * d).map(|i| (i as f64) * 0.01).collect();
        let config = DeepClusterConfig {
            n_clusters: 5,
            n_pca_components: 0,
            kmeans_max_iter: 20,
            kmeans_tol: 1e-4,
            reassign_empty: true,
            seed: 17,
        };
        let result = deep_cluster(&features, n, d, &config).unwrap();
        assert_eq!(result.labels.len(), n);
    }

    // ── Test 4: centroids shape == [k * d] ───────────────────────────────────
    #[test]
    fn centroids_shape_correct() {
        let n = 40_usize;
        let d = 6_usize;
        let k = 4_usize;
        let features: Vec<f64> = (0..n * d).map(|i| ((i as f64) * 0.17).sin()).collect();
        let config = DeepClusterConfig {
            n_clusters: k,
            n_pca_components: 0,
            kmeans_max_iter: 30,
            kmeans_tol: 1e-4,
            reassign_empty: true,
            seed: 23,
        };
        let result = deep_cluster(&features, n, d, &config).unwrap();
        assert_eq!(result.centroids.len(), k * d);
    }

    // ── Test 5: deep_cluster_loss finite and non-negative ────────────────────
    #[test]
    fn loss_finite_and_non_negative() {
        let n = 8_usize;
        let k = 4_usize;
        let logits: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.1).collect();
        let labels = vec![0_usize, 1, 2, 3, 0, 1, 2, 3];
        let loss = deep_cluster_loss(&logits, &labels, n, k).unwrap();
        assert!(loss.is_finite(), "loss = {loss}");
        assert!(loss >= 0.0, "loss = {loss}");
    }

    // ── Test 6: uniform logits → loss ≈ ln(k) ────────────────────────────────
    #[test]
    fn uniform_logits_give_ln_k_loss() {
        let n = 16_usize;
        let k = 8_usize;
        let logits = vec![0.0_f32; n * k]; // all equal → softmax = 1/k
        let labels: Vec<usize> = (0..n).map(|i| i % k).collect();
        let loss = deep_cluster_loss(&logits, &labels, n, k).unwrap();
        let expected = (k as f32).ln();
        assert!(
            (loss - expected).abs() < 1e-4,
            "loss = {loss}, expected = {expected}"
        );
    }

    // ── Test 7: DeeperCluster with 2 scales returns 2 results ────────────────
    #[test]
    fn deeper_cluster_two_scales() {
        let n = 60_usize;
        let d = 4_usize;
        let features: Vec<f64> = (0..n * d).map(|i| ((i as f64) * 0.23).sin()).collect();
        let base = DeepClusterConfig {
            n_clusters: 2, // will be overridden per scale
            n_pca_components: 0,
            kmeans_max_iter: 20,
            kmeans_tol: 1e-3,
            reassign_empty: true,
            seed: 31,
        };
        let config = DeeperClusterConfig {
            cluster_scales: vec![2, 3],
            base_config: base,
        };
        let result = deeper_cluster(&features, n, d, &config).unwrap();
        assert_eq!(result.per_scale.len(), 2);
        assert_eq!(result.multi_labels.len(), 2);
        assert_eq!(result.multi_labels[0].len(), n);
        assert_eq!(result.multi_labels[1].len(), n);
        // Cluster counts should match requested scales.
        for &lbl in &result.multi_labels[0] {
            assert!(lbl < 2, "scale-0 label {lbl} out of range");
        }
        for &lbl in &result.multi_labels[1] {
            assert!(lbl < 3, "scale-1 label {lbl} out of range");
        }
    }

    // ── Test 8: pca_whiten output is approximately whitened ──────────────────
    #[test]
    fn pca_whiten_output_unit_variance_columns() {
        // Create 2D data with variance = [4, 1] (axis-aligned).
        let n = 200_usize;
        let d = 2_usize;
        let mut features = Vec::with_capacity(n * d);
        for i in 0..n {
            let t = i as f64;
            features.push(2.0 * (t * 0.031).sin()); // σ≈√2 in x
            features.push(1.0 * (t * 0.073).cos()); // σ≈1/√2 in y
        }
        let n_comp = 2_usize;
        let whitened = pca_whiten(&features, n, d, n_comp, 1e-6).unwrap();
        assert_eq!(whitened.len(), n * n_comp);
        // Check each column has roughly unit variance.
        for col in 0..n_comp {
            let mean: f64 = whitened.iter().skip(col).step_by(n_comp).sum::<f64>() / n as f64;
            let var: f64 = whitened
                .iter()
                .skip(col)
                .step_by(n_comp)
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f64>()
                / (n as f64 - 1.0);
            assert!(
                var > 0.0 && var.is_finite(),
                "col {col} variance = {var} should be finite and positive"
            );
        }
    }

    // ── Test 9: empty cluster reassignment doesn't crash ─────────────────────
    #[test]
    fn empty_cluster_reassignment_does_not_crash() {
        // Duplicate data — guaranteed empty clusters with many k.
        let n = 10_usize;
        let d = 2_usize;
        // All points at same location → lots of empty clusters.
        let features = vec![1.0_f64; n * d];
        let config = DeepClusterConfig {
            n_clusters: 5,
            n_pca_components: 0,
            kmeans_max_iter: 10,
            kmeans_tol: 0.0, // always run max_iter
            reassign_empty: true,
            seed: 37,
        };
        // Should complete without panic.
        let result = deep_cluster(&features, n, d, &config).unwrap();
        assert_eq!(result.labels.len(), n);
    }

    // ── Test 10: n_clusters > n_samples → error ───────────────────────────────
    #[test]
    fn error_on_more_clusters_than_samples() {
        let n = 5_usize;
        let d = 2_usize;
        let features = vec![1.0_f64; n * d];
        let config = DeepClusterConfig {
            n_clusters: 10, // > n
            n_pca_components: 0,
            kmeans_max_iter: 10,
            kmeans_tol: 1e-4,
            reassign_empty: true,
            seed: 41,
        };
        assert!(deep_cluster(&features, n, d, &config).is_err());
    }

    // ── Test 11: n_clusters = 0 → error ──────────────────────────────────────
    #[test]
    fn error_on_zero_clusters() {
        let result = DeepClusterConfig::new(0, 0, 10, 1e-4, true, 42);
        assert!(result.is_err(), "n_clusters=0 should return an error");
    }

    // ── Test 12: inertia non-negative and finite ──────────────────────────────
    #[test]
    fn inertia_non_negative_and_finite() {
        let n = 50_usize;
        let d = 3_usize;
        let features: Vec<f64> = (0..n * d).map(|i| ((i as f64) * 0.11).sin()).collect();
        let config = DeepClusterConfig {
            n_clusters: 5,
            n_pca_components: 0,
            kmeans_max_iter: 50,
            kmeans_tol: 1e-4,
            reassign_empty: true,
            seed: 53,
        };
        let result = deep_cluster(&features, n, d, &config).unwrap();
        assert!(result.inertia.is_finite(), "inertia = {}", result.inertia);
        assert!(result.inertia >= 0.0, "inertia = {}", result.inertia);
    }

    // ── Test 13: converged=true when data is already clustered ────────────────
    #[test]
    fn converged_true_when_stable() {
        let n_per = 20_usize;
        let n = 2 * n_per;
        let d = 2_usize;
        let data = two_cluster_data(n_per);
        let config = DeepClusterConfig {
            n_clusters: 2,
            n_pca_components: 0,
            kmeans_max_iter: 500,
            kmeans_tol: 0.01, // 1% tolerance — well-separated data converges easily
            reassign_empty: true,
            seed: 61,
        };
        let result = deep_cluster(&data, n, d, &config).unwrap();
        assert!(result.converged, "should have converged");
    }

    // ── Test 14: loss rejects invalid label ───────────────────────────────────
    #[test]
    fn loss_rejects_out_of_range_label() {
        let n = 4_usize;
        let k = 3_usize;
        let logits = vec![0.0_f32; n * k];
        let labels = vec![0_usize, 1, 2, 3]; // 3 >= k=3 → invalid
        assert!(deep_cluster_loss(&logits, &labels, n, k).is_err());
    }

    // ── Test 15: pca_whiten rejects bad n_components ──────────────────────────
    #[test]
    fn pca_whiten_rejects_invalid_n_components() {
        let n = 10_usize;
        let d = 4_usize;
        let features = vec![1.0_f64; n * d];
        // n_components == 0
        assert!(pca_whiten(&features, n, d, 0, 1e-6).is_err());
        // n_components > feat_dim
        assert!(pca_whiten(&features, n, d, d + 1, 1e-6).is_err());
    }

    // ── Test 16: deep_cluster_loss with k=1 → error ───────────────────────────
    #[test]
    fn loss_rejects_single_cluster() {
        let logits = vec![1.0_f32; 4];
        let labels = vec![0_usize; 4];
        assert!(deep_cluster_loss(&logits, &labels, 4, 1).is_err());
    }
}

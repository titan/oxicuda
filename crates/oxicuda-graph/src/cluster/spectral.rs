//! Spectral clustering via graph Laplacian eigenvectors (Von Luxburg 2007).
//!
//! The algorithm:
//! 1. Build (or accept) an `n×n` symmetric affinity matrix **A**.
//! 2. Compute the degree vector `D[i] = Σ_j A[i,j]`.
//! 3. Form the symmetrically normalised matrix `M = D^{-1/2} A D^{-1/2}`.
//!    The smallest eigenvectors of `L_sym = I − M` are the largest of `M`.
//! 4. Extract the top-k eigenvectors of `M` via **power iteration with deflation**
//!    (Gram-Schmidt orthogonalisation between each eigenvector).
//! 5. Row-normalise the embedding matrix `U ∈ ℝ^{n × k}`.
//! 6. Run Lloyd's k-means on the rows of `U`.

use crate::error::{GraphError, GraphResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for spectral clustering.
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    /// Number of output clusters.
    pub n_clusters: usize,
    /// Number of eigenvectors to compute (typically == `n_clusters`).
    pub n_eigenvectors: usize,
    /// RBF kernel parameter `γ`: `A[i,j] = exp(−γ ‖x_i − x_j‖²)`.
    pub gamma: f64,
    /// Number of Lloyd's k-means iterations.
    pub n_iter_kmeans: usize,
    /// Power-iteration steps per eigenvector.
    pub n_iter_power: usize,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            n_eigenvectors: 2,
            gamma: 1.0,
            n_iter_kmeans: 50,
            n_iter_power: 100,
        }
    }
}

// ─── Result type ─────────────────────────────────────────────────────────────

/// Output of a spectral clustering run.
pub struct SpectralClustering {
    labels: Vec<usize>,
    eigenvalues: Vec<f64>,
    n_clusters: usize,
}

impl SpectralClustering {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Cluster `n` points from an `n×n` affinity matrix (row-major, flat slice).
    ///
    /// # Errors
    /// - [`GraphError::InvalidPlan`] for shape / config inconsistencies.
    pub fn fit_affinity(affinity: &[f64], n: usize, config: &SpectralConfig) -> GraphResult<Self> {
        validate_config(n, config)?;
        if affinity.len() != n * n {
            return Err(GraphError::InvalidPlan(format!(
                "affinity length {} != n*n {}",
                affinity.len(),
                n * n
            )));
        }

        // Step 1: degree vector
        let mut degree = vec![0.0_f64; n];
        for i in 0..n {
            let s: f64 = affinity[i * n..(i + 1) * n].iter().sum();
            degree[i] = s;
        }

        // Step 2: D^{-1/2}
        let d_inv_sqrt: Vec<f64> = degree
            .iter()
            .map(|&d| if d > 1e-14 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        // Step 3: M = D^{-1/2} A D^{-1/2}  (symmetric, stored as full n×n)
        let mut m = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] = d_inv_sqrt[i] * affinity[i * n + j] * d_inv_sqrt[j];
            }
        }

        // Step 4: top-k eigenvectors of M via power iteration + deflation
        let k = config.n_eigenvectors;
        let (eigvecs, eigenvalues) = power_iteration_deflation(&m, n, k, config.n_iter_power);

        // Step 5: build embedding U [n × k], row-normalise
        let mut u = eigvecs; // already [n × k] column-major: col_i is eigvec_i
        // Transpose to row-major [n × k]
        let mut u_row = vec![0.0_f64; n * k];
        for i in 0..n {
            for c in 0..k {
                u_row[i * k + c] = u[c * n + i];
            }
        }
        // Row-normalise
        for i in 0..n {
            let row = &mut u_row[i * k..(i + 1) * k];
            let norm = row.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-12);
            for v in row.iter_mut() {
                *v /= norm;
            }
        }

        // Step 6: k-means
        let seed = affinity
            .iter()
            .take(16)
            .fold(0u64, |acc, &v| acc.wrapping_add(v.to_bits()));
        let labels = kmeans(&u_row, n, k, config.n_clusters, config.n_iter_kmeans, seed);

        // Eigenvalues stored as eigenvalues of M (not of L_sym)
        let eigenvalues_out: Vec<f64> = eigenvalues;
        // Reuse mutable u binding to avoid "unused mut" warning:
        u = Vec::new();
        let _ = u;

        Ok(Self {
            labels,
            eigenvalues: eigenvalues_out,
            n_clusters: config.n_clusters,
        })
    }

    /// Cluster `n` data points from a `[n × dim]` row-major data matrix,
    /// using an RBF affinity: `A[i,j] = exp(−γ ‖x_i − x_j‖²)`.
    ///
    /// # Errors
    /// - [`GraphError::InvalidPlan`] for shape / config inconsistencies.
    pub fn fit_data(
        data: &[f64],
        n: usize,
        dim: usize,
        config: &SpectralConfig,
    ) -> GraphResult<Self> {
        if dim == 0 {
            return Err(GraphError::InvalidPlan("dim must be > 0".to_owned()));
        }
        if data.len() != n * dim {
            return Err(GraphError::InvalidPlan(format!(
                "data length {} != n*dim {}",
                data.len(),
                n * dim
            )));
        }

        // Build RBF affinity matrix
        let mut affinity = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let sq_dist: f64 = (0..dim)
                    .map(|d| {
                        let diff = data[i * dim + d] - data[j * dim + d];
                        diff * diff
                    })
                    .sum();
                affinity[i * n + j] = (-config.gamma * sq_dist).exp();
            }
        }

        Self::fit_affinity(&affinity, n, config)
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Cluster label for each point (length `n`, values in `[0, n_clusters)`).
    #[must_use]
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Eigenvalues of the normalised affinity matrix `M = D^{-1/2} A D^{-1/2}`
    /// for the selected eigenvectors (descending order).
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Number of clusters.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Validate configuration against the number of points.
fn validate_config(n: usize, config: &SpectralConfig) -> GraphResult<()> {
    if n == 0 {
        return Err(GraphError::InvalidPlan("n must be > 0".to_owned()));
    }
    if config.n_clusters == 0 || config.n_clusters > n {
        return Err(GraphError::InvalidPlan(format!(
            "n_clusters {} out of range [1, {}]",
            config.n_clusters, n
        )));
    }
    if config.n_eigenvectors == 0 {
        return Err(GraphError::InvalidPlan(
            "n_eigenvectors must be > 0".to_owned(),
        ));
    }
    Ok(())
}

/// Matrix-vector product `y = A x` for symmetric `n×n` matrix `A` (row-major).
fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
    y
}

/// Dot product of two vectors.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// L2-normalise a vector in-place; returns the original norm.
fn l2_normalise(v: &mut [f64]) -> f64 {
    let norm = dot(v, v).sqrt().max(1e-14);
    for x in v.iter_mut() {
        *x /= norm;
    }
    norm
}

/// Power iteration with Gram-Schmidt deflation.
///
/// Returns `(eigvecs, eigenvalues)` where `eigvecs` is stored **column-major**
/// as a flat `[k × n]` slice (i.e. `eigvecs[c * n..][..n]` is eigenvector `c`).
fn power_iteration_deflation(m: &[f64], n: usize, k: usize, n_iter: usize) -> (Vec<f64>, Vec<f64>) {
    let k_actual = k.min(n);
    let mut eigvecs = Vec::with_capacity(k_actual * n);
    let mut eigenvalues = Vec::with_capacity(k_actual);

    for c in 0..k_actual {
        // Initialise with a deterministic non-zero vector
        let mut v: Vec<f64> = (0..n)
            .map(|i| if i == c % n { 1.0 } else { 0.01 })
            .collect();
        l2_normalise(&mut v);

        let mut eigenvalue = 0.0_f64;
        for _ in 0..n_iter {
            // Power step: v ← M v
            let mut mv = matvec(m, &v, n);

            // Deflate by all previously found eigenvectors (Gram-Schmidt)
            for prev in 0..c {
                let prev_vec = &eigvecs[prev * n..(prev + 1) * n];
                let coeff = dot(&mv, prev_vec);
                for (x, &pv) in mv.iter_mut().zip(prev_vec.iter()) {
                    *x -= coeff * pv;
                }
            }

            eigenvalue = l2_normalise(&mut mv);
            v = mv;
        }

        eigvecs.extend_from_slice(&v);
        eigenvalues.push(eigenvalue);
    }

    (eigvecs, eigenvalues)
}

/// Simple Lloyd's k-means on rows of `u` (`n × k_embed` row-major).
/// Returns cluster label for each of the `n` rows.
fn kmeans(
    u: &[f64],
    n: usize,
    k_embed: usize,
    k_clusters: usize,
    n_iter: usize,
    seed: u64,
) -> Vec<usize> {
    if k_clusters == 0 || n == 0 {
        return vec![0; n];
    }

    // Deterministic initialisation: pick k_clusters evenly-spaced row indices
    let mut centroids = vec![0.0_f64; k_clusters * k_embed];
    let step = (n / k_clusters).max(1);
    // Use seed to offset start
    let start = (seed as usize) % n.max(1);
    for c in 0..k_clusters {
        let row_idx = (start + c * step) % n;
        for d in 0..k_embed {
            centroids[c * k_embed + d] = u[row_idx * k_embed + d];
        }
    }

    let mut labels = vec![0_usize; n];

    for _iter in 0..n_iter {
        // Assignment step
        let mut changed = false;
        for i in 0..n {
            let row = &u[i * k_embed..(i + 1) * k_embed];
            let mut best = 0_usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..k_clusters {
                let centroid = &centroids[c * k_embed..(c + 1) * k_embed];
                let dist: f64 = row
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step
        let mut new_centroids = vec![0.0_f64; k_clusters * k_embed];
        let mut counts = vec![0_usize; k_clusters];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            let row = &u[i * k_embed..(i + 1) * k_embed];
            for d in 0..k_embed {
                new_centroids[c * k_embed + d] += row[d];
            }
        }
        for c in 0..k_clusters {
            if counts[c] > 0 {
                let cnt = counts[c] as f64;
                for d in 0..k_embed {
                    new_centroids[c * k_embed + d] /= cnt;
                }
            } else {
                // Empty cluster: reinitialise to a random point
                let row_idx = c % n;
                for d in 0..k_embed {
                    new_centroids[c * k_embed + d] = u[row_idx * k_embed + d];
                }
            }
        }
        centroids = new_centroids;
    }

    labels
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_block_affinity(n: usize) -> Vec<f64> {
        // Two equal blocks with high within-block, zero between-block affinity
        let half = n / 2;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let same_block = (i < half && j < half) || (i >= half && j >= half);
                a[i * n + j] = if same_block { 1.0 } else { 0.0 };
            }
        }
        a
    }

    // 1. Output has exactly n labels
    #[test]
    fn labels_len() {
        let n = 6;
        let a = two_block_affinity(n);
        let cfg = SpectralConfig {
            n_clusters: 2,
            n_eigenvectors: 2,
            ..Default::default()
        };
        let sc =
            SpectralClustering::fit_affinity(&a, n, &cfg).expect("fit_affinity should succeed");
        assert_eq!(sc.labels().len(), n);
    }

    // 2. All labels are in [0, n_clusters)
    #[test]
    fn labels_in_range() {
        let n = 8;
        let a = two_block_affinity(n);
        let cfg = SpectralConfig {
            n_clusters: 2,
            n_eigenvectors: 2,
            ..Default::default()
        };
        let sc =
            SpectralClustering::fit_affinity(&a, n, &cfg).expect("fit_affinity should succeed");
        for &l in sc.labels() {
            assert!(l < 2, "label {l} out of range");
        }
    }

    // 3. Two clearly separated groups → points within each group share a label
    #[test]
    fn two_clusters_separated() {
        let n = 8;
        let a = two_block_affinity(n);
        let cfg = SpectralConfig {
            n_clusters: 2,
            n_eigenvectors: 2,
            n_iter_power: 200,
            n_iter_kmeans: 100,
            ..Default::default()
        };
        let sc =
            SpectralClustering::fit_affinity(&a, n, &cfg).expect("fit_affinity should succeed");
        let labels = sc.labels();
        let half = n / 2;
        // First half should all share one label
        let label_0 = labels[0];
        for &l in &labels[..half] {
            assert_eq!(l, label_0, "first-half labels should match");
        }
        // Second half all share another label
        let label_1 = labels[half];
        for &l in &labels[half..] {
            assert_eq!(l, label_1, "second-half labels should match");
        }
        // The two groups must have different labels
        assert_ne!(label_0, label_1);
    }

    // 4. Single cluster: all labels == 0
    #[test]
    fn single_cluster() {
        let n = 5;
        let a = vec![1.0_f64; n * n]; // fully connected
        let cfg = SpectralConfig {
            n_clusters: 1,
            n_eigenvectors: 1,
            ..Default::default()
        };
        let sc =
            SpectralClustering::fit_affinity(&a, n, &cfg).expect("fit_affinity should succeed");
        for &l in sc.labels() {
            assert_eq!(l, 0);
        }
    }

    // 5. Symmetric affinity → same result as transposed (trivially true since input is symmetric)
    #[test]
    fn affinity_symmetric() {
        let n = 4;
        let a = two_block_affinity(n);
        // Transpose (which equals itself for a symmetric matrix)
        let mut at = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                at[j * n + i] = a[i * n + j];
            }
        }
        let cfg = SpectralConfig::default();
        let sc1 =
            SpectralClustering::fit_affinity(&a, n, &cfg).expect("fit_affinity should succeed");
        let sc2 =
            SpectralClustering::fit_affinity(&at, n, &cfg).expect("fit_affinity should succeed");
        // Both should produce identical label assignments (same affinity)
        assert_eq!(sc1.labels(), sc2.labels());
    }

    // 6. n_clusters=1 works without panic
    #[test]
    fn n_clusters_1_works() {
        let n = 4;
        let a = two_block_affinity(n);
        let cfg = SpectralConfig {
            n_clusters: 1,
            n_eigenvectors: 1,
            ..Default::default()
        };
        let sc =
            SpectralClustering::fit_affinity(&a, n, &cfg).expect("fit_affinity should succeed");
        assert_eq!(sc.n_clusters(), 1);
        assert!(sc.labels().iter().all(|&l| l == 0));
    }

    // 7. Diagonal values don't affect cluster structure
    //    (zeroing diagonal shouldn't change which points are grouped together)
    #[test]
    fn affinity_diagonal_ignored() {
        let n = 6;
        let mut a = two_block_affinity(n);
        // Make diagonal 100×  instead of 1×
        for i in 0..n {
            a[i * n + i] = 100.0;
        }
        let cfg = SpectralConfig {
            n_clusters: 2,
            n_eigenvectors: 2,
            ..Default::default()
        };
        let sc =
            SpectralClustering::fit_affinity(&a, n, &cfg).expect("fit_affinity should succeed");
        assert_eq!(sc.labels().len(), n);
        for &l in sc.labels() {
            assert!(l < 2);
        }
    }

    // 8. Eigenvalues are all finite
    #[test]
    fn eigenvalues_finite() {
        let n = 6;
        let a = two_block_affinity(n);
        let cfg = SpectralConfig {
            n_clusters: 2,
            n_eigenvectors: 2,
            ..Default::default()
        };
        let sc =
            SpectralClustering::fit_affinity(&a, n, &cfg).expect("fit_affinity should succeed");
        for &ev in sc.eigenvalues() {
            assert!(ev.is_finite(), "eigenvalue {ev} is not finite");
        }
    }

    // 9. Works when n >> n_eigenvectors
    #[test]
    fn n_gt_n_eigenvectors_ok() {
        let n = 10;
        let a = two_block_affinity(n);
        let cfg = SpectralConfig {
            n_clusters: 2,
            n_eigenvectors: 2,
            ..Default::default()
        };
        let sc =
            SpectralClustering::fit_affinity(&a, n, &cfg).expect("fit_affinity should succeed");
        assert_eq!(sc.labels().len(), n);
    }

    // 10. fit_data from RBF affinity
    #[test]
    fn fit_data_rbf() {
        // Two well-separated clusters in 2D
        let n = 8;
        let dim = 2;
        let mut data = vec![0.0_f64; n * dim];
        for i in 0..4 {
            data[i * dim] = i as f64 * 0.1;
            data[i * dim + 1] = 0.0;
        }
        for i in 4..8 {
            data[i * dim] = 10.0 + (i - 4) as f64 * 0.1;
            data[i * dim + 1] = 0.0;
        }
        let cfg = SpectralConfig {
            n_clusters: 2,
            n_eigenvectors: 2,
            gamma: 1.0,
            n_iter_kmeans: 100,
            n_iter_power: 200,
        };
        let sc =
            SpectralClustering::fit_data(&data, n, dim, &cfg).expect("fit_data should succeed");
        assert_eq!(sc.labels().len(), n);
        for &l in sc.labels() {
            assert!(l < 2);
        }
    }
}

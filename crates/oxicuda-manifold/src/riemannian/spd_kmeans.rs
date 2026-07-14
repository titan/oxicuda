//! Riemannian k-means and Fréchet mean on Symmetric Positive-Definite (SPD) matrices.
//!
//! Implements:
//! - **Fréchet mean** (Riemannian centroid) via Karcher flow / gradient descent on
//!   the affine-invariant metric.  Reference: Moakher (2005), Bhatia & Holbrook (2006).
//! - **Riemannian k-means** on SPD(n) with k-means++ initialisation and multiple
//!   restarts.  Reference: Fletcher et al. (2004), Pennec (2006).
//!
//! All matrices are stored **row-major** as flat `Vec<f64>` / slices.
//! A batch of `k` matrices each of size `n×n` is stored flat as
//! `matrices[i * n*n .. (i+1) * n*n]` for `i in 0..k`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::riemannian::spd::{spd_distance, spd_exp, spd_log};

// ────────────────────────────────────────────────────────────────────────────
// Fréchet Mean
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for iterative Fréchet (Karcher) mean computation.
#[derive(Debug, Clone)]
pub struct FrechetMeanConfig {
    /// Maximum number of Karcher-flow gradient steps.
    pub max_iter: usize,
    /// Convergence tolerance: iteration stops when ||mean log map||_F < tol.
    pub tol: f64,
    /// Step size for the exponential-map update (1.0 = full Karcher step).
    pub step_size: f64,
}

impl Default for FrechetMeanConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
            step_size: 1.0,
        }
    }
}

/// Output of [`spd_frechet_mean`].
#[derive(Debug, Clone)]
pub struct FrechetMeanResult {
    /// Fréchet mean matrix, stored row-major as `[n × n]`.
    pub mean: Vec<f64>,
    /// Number of gradient steps performed.
    pub n_iter: usize,
    /// Whether the iteration converged within tolerance.
    pub converged: bool,
    /// Frobenius norm of the mean log-map at the last iteration (gradient proxy).
    pub final_grad_norm: f64,
}

/// Compute the Fréchet mean of `k` SPD matrices of size `n×n`.
///
/// # Arguments
/// * `matrices` — flat `[k × n × n]` row-major; `matrices[i*n*n..(i+1)*n*n]` is the i-th SPD matrix.
/// * `k`        — number of input matrices.
/// * `n`        — matrix dimension.
/// * `config`   — algorithm hyper-parameters.
///
/// # Algorithm (Karcher flow)
/// Initialise `μ` as the arithmetic mean (guaranteed SPD for SPD inputs).  Then:
/// 1. Compute `V_i = log_μ(P_i)` for each i.
/// 2. Take the element-wise mean of the tangent vectors: `V̄ = (1/k) Σ V_i`.
/// 3. Measure the gradient norm `‖V̄‖_F`.
/// 4. Update `μ ← exp_μ(step_size · V̄)`.
/// 5. Repeat until `‖V̄‖_F < tol` or `max_iter` is reached.
pub fn spd_frechet_mean(
    matrices: &[f64],
    k: usize,
    n: usize,
    config: &FrechetMeanConfig,
) -> ManifoldResult<FrechetMeanResult> {
    // ── Validate inputs ────────────────────────────────────────────────────
    if k == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    let expected_len = k * n * n;
    if matrices.len() != expected_len {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![k, n, n],
            got: vec![matrices.len()],
        });
    }
    if n == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n".into(),
            reason: "matrix dimension must be >= 1".into(),
        });
    }
    if config.max_iter == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "max_iter".into(),
            reason: "must be >= 1".into(),
        });
    }

    // ── Special case: single matrix ─────────────────────────────────────────
    if k == 1 {
        return Ok(FrechetMeanResult {
            mean: matrices[..n * n].to_vec(),
            n_iter: 0,
            converged: true,
            final_grad_norm: 0.0,
        });
    }

    // ── Initialise as arithmetic mean ────────────────────────────────────────
    let mut mu = spd_arithmetic_mean(matrices, k, n);

    let mut n_iter = 0usize;
    let mut converged = false;
    let mut final_grad_norm = f64::INFINITY;

    // ── Karcher flow iteration ────────────────────────────────────────────────
    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // Accumulate log maps into a mean tangent vector.
        let mut v_mean = vec![0.0f64; n * n];
        let mut valid_count = 0usize;

        for i in 0..k {
            let pi = &matrices[i * n * n..(i + 1) * n * n];
            match spd_log(&mu, pi, n) {
                Ok(vi) => {
                    for (acc, val) in v_mean.iter_mut().zip(vi.iter()) {
                        *acc += val;
                    }
                    valid_count += 1;
                }
                Err(_) => {
                    // Skip numerically problematic log maps gracefully.
                }
            }
        }

        if valid_count == 0 {
            return Err(ManifoldError::NumericalInstability(
                "all log maps failed at current Fréchet iterate".into(),
            ));
        }

        // Normalise: V̄ = (1/valid_count) Σ V_i
        let inv_count = 1.0 / valid_count as f64;
        for v in v_mean.iter_mut() {
            *v *= inv_count;
        }

        // Scale by step size.
        if (config.step_size - 1.0).abs() > f64::EPSILON {
            for v in v_mean.iter_mut() {
                *v *= config.step_size;
            }
        }

        // Gradient norm (Frobenius of mean tangent).
        let grad_norm = mat_frobenius_norm(&v_mean, n);
        final_grad_norm = grad_norm;

        // Convergence check *before* the update so we detect an already-converged μ.
        if grad_norm < config.tol {
            converged = true;
            break;
        }

        // μ ← exp_μ(V̄)
        match spd_exp(&mu, &v_mean, n) {
            Ok(mu_new) => mu = mu_new,
            Err(e) => {
                // Propagate hard failures (not just near-singular intermediates).
                return Err(ManifoldError::NumericalInstability(format!(
                    "exp map failed during Fréchet mean iteration {iter}: {e}"
                )));
            }
        }
    }

    Ok(FrechetMeanResult {
        mean: mu,
        n_iter,
        converged,
        final_grad_norm,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Riemannian k-means on SPD
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for [`spd_kmeans`].
#[derive(Debug, Clone)]
pub struct SpdKmeansConfig {
    /// Number of clusters k.
    pub n_clusters: usize,
    /// Maximum number of EM (assignment + centroid update) iterations.
    pub max_iter: usize,
    /// Convergence threshold on the change of total inertia between iterations.
    pub tol: f64,
    /// Configuration passed to the Fréchet mean solver for centroid updates.
    pub frechet_config: FrechetMeanConfig,
    /// Seed for the LCG RNG used in k-means++ initialisation.
    pub seed: u64,
    /// Number of independent random restarts; the run with lowest inertia is returned.
    pub n_restarts: usize,
}

impl Default for SpdKmeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            max_iter: 50,
            tol: 1e-6,
            frechet_config: FrechetMeanConfig::default(),
            seed: 42,
            n_restarts: 3,
        }
    }
}

/// Output of [`spd_kmeans`].
#[derive(Debug, Clone)]
pub struct SpdKmeansResult {
    /// Cluster label for each data point; `labels[i] in 0..n_clusters`.
    pub labels: Vec<usize>,
    /// Centroids stored flat as `[n_clusters × n × n]` row-major.
    pub centroids: Vec<f64>,
    /// Sum of squared affine-invariant distances from each point to its centroid.
    pub inertia: f64,
    /// Number of EM iterations actually performed (best restart).
    pub n_iter: usize,
    /// Whether the best run converged within `tol` before `max_iter`.
    pub converged: bool,
}

/// Riemannian k-means clustering of SPD matrices using the affine-invariant metric.
///
/// # Arguments
/// * `data`       — flat `[n_matrices × n × n]` row-major SPD matrices.
/// * `n_matrices` — number of data points.
/// * `n`          — matrix dimension (each matrix is `n×n`).
/// * `config`     — algorithm hyper-parameters.
///
/// # Algorithm
/// Uses **k-means++** initialisation on the SPD manifold (distances via
/// [`spd_distance`]) followed by iterative **EM steps**:
/// 1. **Assign**: each point is assigned to its nearest centroid.
/// 2. **Update**: each centroid is updated to the Fréchet mean of its cluster.
/// 3. **Converge**: stop when |inertia_old − inertia_new| < tol.
///
/// `n_restarts` independent runs are performed; the best (lowest inertia) is returned.
pub fn spd_kmeans(
    data: &[f64],
    n_matrices: usize,
    n: usize,
    config: &SpdKmeansConfig,
) -> ManifoldResult<SpdKmeansResult> {
    // ── Validate inputs ────────────────────────────────────────────────────
    if n_matrices == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if n == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n".into(),
            reason: "matrix dimension must be >= 1".into(),
        });
    }
    let expected_len = n_matrices * n * n;
    if data.len() != expected_len {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_matrices, n, n],
            got: vec![data.len()],
        });
    }
    if config.n_clusters == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_clusters".into(),
            reason: "must be >= 1".into(),
        });
    }
    if config.n_clusters > n_matrices {
        return Err(ManifoldError::InvalidParameter {
            name: "n_clusters".into(),
            reason: format!(
                "n_clusters ({}) > n_matrices ({})",
                config.n_clusters, n_matrices
            ),
        });
    }
    if config.n_restarts == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_restarts".into(),
            reason: "must be >= 1".into(),
        });
    }

    // Each restart gets a distinct seed derived from the base seed.
    let restart_seed = |restart: usize| {
        config
            .seed
            .wrapping_add((restart as u64).wrapping_mul(0xDEAD_BEEF_CAFE_BABE))
    };

    // `config.n_restarts >= 1` is guaranteed by the validation above, so restart 0
    // always runs and seeds `best` directly — no `Option<SpdKmeansResult>` sentinel
    // (and no end-of-function unwrap) is needed to track "has a restart completed".
    let mut rng0 = LcgRng::new(restart_seed(0));
    let mut best = run_single_kmeans(data, n_matrices, n, config, &mut rng0)?;

    for restart in 1..config.n_restarts {
        let mut rng = LcgRng::new(restart_seed(restart));
        let result = run_single_kmeans(data, n_matrices, n, config, &mut rng)?;
        if result.inertia < best.inertia {
            best = result;
        }
    }

    Ok(best)
}

// ────────────────────────────────────────────────────────────────────────────
// Internal: single k-means run
// ────────────────────────────────────────────────────────────────────────────

fn run_single_kmeans(
    data: &[f64],
    n_matrices: usize,
    n: usize,
    config: &SpdKmeansConfig,
    rng: &mut LcgRng,
) -> ManifoldResult<SpdKmeansResult> {
    let k = config.n_clusters;
    let nn = n * n;

    // ── k-means++ initialisation ─────────────────────────────────────────────
    let mut centroids = kmeans_plus_plus_init(data, n_matrices, n, k, rng)?;

    let (mut labels, mut inertia) = assign_labels(data, &centroids, n_matrices, k, n);

    let mut n_iter = 0usize;
    let mut converged = false;

    // ── EM iterations ─────────────────────────────────────────────────────────
    for _iter in 0..config.max_iter {
        n_iter += 1;

        // ── Centroid update (Fréchet mean per cluster) ────────────────────
        let mut new_centroids = vec![0.0f64; k * nn];

        for c in 0..k {
            // Collect indices assigned to cluster c.
            let cluster_indices: Vec<usize> = (0..n_matrices).filter(|&i| labels[i] == c).collect();

            if cluster_indices.is_empty() {
                // Empty cluster: reinitialise to a random data point.
                let random_idx = rng.next_usize(n_matrices);
                new_centroids[c * nn..(c + 1) * nn]
                    .copy_from_slice(&data[random_idx * nn..(random_idx + 1) * nn]);
                continue;
            }

            if cluster_indices.len() == 1 {
                // Singleton cluster: centroid = the single member.
                let idx = cluster_indices[0];
                new_centroids[c * nn..(c + 1) * nn]
                    .copy_from_slice(&data[idx * nn..(idx + 1) * nn]);
                continue;
            }

            // Build flat batch for Fréchet mean.
            let cluster_mats: Vec<f64> = cluster_indices
                .iter()
                .flat_map(|&idx| data[idx * nn..(idx + 1) * nn].iter().copied())
                .collect();

            let freq_result = spd_frechet_mean(
                &cluster_mats,
                cluster_indices.len(),
                n,
                &config.frechet_config,
            );

            match freq_result {
                Ok(fr) => {
                    new_centroids[c * nn..(c + 1) * nn].copy_from_slice(&fr.mean);
                }
                Err(_) => {
                    // Fall back to arithmetic mean if Fréchet mean fails.
                    let arith = spd_arithmetic_mean(&cluster_mats, cluster_indices.len(), n);
                    new_centroids[c * nn..(c + 1) * nn].copy_from_slice(&arith);
                }
            }
        }

        centroids = new_centroids;

        // ── Assignment step ──────────────────────────────────────────────────
        let (new_labels, new_inertia) = assign_labels(data, &centroids, n_matrices, k, n);

        // ── Convergence check ────────────────────────────────────────────────
        let delta = (inertia - new_inertia).abs();
        labels = new_labels;
        inertia = new_inertia;

        if delta < config.tol {
            converged = true;
            break;
        }
    }

    Ok(SpdKmeansResult {
        labels,
        centroids,
        inertia,
        n_iter,
        converged,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// k-means++ initialisation on the SPD manifold
// ────────────────────────────────────────────────────────────────────────────

/// Initialise k cluster centroids using the k-means++ strategy.
///
/// 1. Sample the first centroid uniformly at random.
/// 2. For each subsequent centroid, sample from data with probability
///    proportional to `d²(xᵢ, nearest centroid so far)`.
fn kmeans_plus_plus_init(
    data: &[f64],
    n_matrices: usize,
    n: usize,
    k: usize,
    rng: &mut LcgRng,
) -> ManifoldResult<Vec<f64>> {
    let nn = n * n;
    let mut centroids = Vec::with_capacity(k * nn);

    // Choose first centroid uniformly at random.
    let first_idx = rng.next_usize(n_matrices);
    centroids.extend_from_slice(&data[first_idx * nn..(first_idx + 1) * nn]);

    // Maintain squared distances from each point to its nearest centroid so far.
    let mut min_sq_dists = vec![f64::INFINITY; n_matrices];

    for c_idx in 1..k {
        // The centroid just added is at position c_idx-1 in `centroids`.
        let new_centroid_offset = (c_idx - 1) * nn;

        // Update min_sq_dists with distances to the newly added centroid.
        let ci = &centroids[new_centroid_offset..new_centroid_offset + nn];
        for (i, slot) in min_sq_dists.iter_mut().enumerate() {
            let pi = &data[i * nn..(i + 1) * nn];
            let d_sq = match spd_distance(pi, ci, n) {
                Ok(d) => d * d,
                Err(_) => 0.0, // treat failed distance as 0 to avoid selection
            };
            if d_sq < *slot {
                *slot = d_sq;
            }
        }

        // Sample next centroid with probability ∝ min_sq_dists.
        let total: f64 = min_sq_dists.iter().sum();

        let chosen_idx = if total <= 0.0 {
            // Degenerate: all distances zero (identical matrices); pick randomly.
            rng.next_usize(n_matrices)
        } else {
            let threshold = rng.next_f64() * total;
            let mut cumulative = 0.0;
            let mut chosen = n_matrices - 1; // fallback
            for (i, &d) in min_sq_dists.iter().enumerate() {
                cumulative += d;
                if cumulative >= threshold {
                    chosen = i;
                    break;
                }
            }
            chosen
        };

        centroids.extend_from_slice(&data[chosen_idx * nn..(chosen_idx + 1) * nn]);
    }

    Ok(centroids)
}

// ────────────────────────────────────────────────────────────────────────────
// Private helpers
// ────────────────────────────────────────────────────────────────────────────

/// Compute the element-wise arithmetic mean of `k` matrices of size `n×n`.
fn spd_arithmetic_mean(matrices: &[f64], k: usize, n: usize) -> Vec<f64> {
    let nn = n * n;
    let mut mean = vec![0.0f64; nn];
    for i in 0..k {
        let m = &matrices[i * nn..(i + 1) * nn];
        for (acc, &val) in mean.iter_mut().zip(m.iter()) {
            *acc += val;
        }
    }
    let inv_k = 1.0 / k as f64;
    for v in mean.iter_mut() {
        *v *= inv_k;
    }
    mean
}

/// Assign each data point to its nearest centroid.
///
/// Returns `(labels, inertia)` where `inertia = Σᵢ d²(pᵢ, centroid_{label_i})`.
fn assign_labels(
    data: &[f64],
    centroids: &[f64],
    n_matrices: usize,
    n_centroids: usize,
    n: usize,
) -> (Vec<usize>, f64) {
    let nn = n * n;
    let mut labels = vec![0usize; n_matrices];
    let mut inertia = 0.0f64;

    for i in 0..n_matrices {
        let pi = &data[i * nn..(i + 1) * nn];
        let mut best_label = 0usize;
        let mut best_d_sq = f64::INFINITY;

        for c in 0..n_centroids {
            let ci = &centroids[c * nn..(c + 1) * nn];
            let d_sq = match spd_distance(pi, ci, n) {
                Ok(d) => d * d,
                Err(_) => f64::INFINITY,
            };
            if d_sq < best_d_sq {
                best_d_sq = d_sq;
                best_label = c;
            }
        }

        labels[i] = best_label;
        if best_d_sq.is_finite() {
            inertia += best_d_sq;
        }
    }

    (labels, inertia)
}

/// Frobenius norm of a flat `n×n` matrix.
fn mat_frobenius_norm(v: &[f64], _n: usize) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Squared Frobenius norm of a flat `n×n` matrix.
#[allow(dead_code)]
fn mat_frobenius_norm_sq(v: &[f64], _n: usize) -> f64 {
    v.iter().map(|&x| x * x).sum()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: diagonal SPD matrix [[a,0],[0,b]] ──────────────────────────
    fn diag2(a: f64, b: f64) -> Vec<f64> {
        vec![a, 0.0, 0.0, b]
    }

    // ── Helper: 3×3 diagonal SPD matrix ──────────────────────────────────────
    fn diag3(a: f64, b: f64, c: f64) -> Vec<f64> {
        vec![
            a, 0.0, 0.0, //
            0.0, b, 0.0, //
            0.0, 0.0, c, //
        ]
    }

    // ── Helper: identity n×n ──────────────────────────────────────────────────
    fn identity(n: usize) -> Vec<f64> {
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    // ── Helper: check matrix is symmetric and all eigenvalues positive ─────────
    fn is_spd(m: &[f64], n: usize) -> bool {
        use crate::linalg::jacobi_eig::jacobi_eigh;
        // Symmetry check.
        for i in 0..n {
            for j in 0..n {
                if (m[i * n + j] - m[j * n + i]).abs() > 1e-6 {
                    return false;
                }
            }
        }
        // Positive definiteness.
        match jacobi_eigh(m, n) {
            Ok((w, _)) => w.iter().all(|&ev| ev > 1e-10),
            Err(_) => false,
        }
    }

    // ── Test 1: Fréchet mean of a single matrix equals itself ──────────────────
    #[test]
    fn frechet_mean_single_matrix() {
        let p = diag2(2.0, 3.0);
        let config = FrechetMeanConfig::default();
        let res = spd_frechet_mean(&p, 1, 2, &config).expect("ok");
        assert!(res.converged);
        assert_eq!(res.n_iter, 0);
        for (a, b) in res.mean.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-10, "mean != input: {a} vs {b}");
        }
    }

    // ── Test 2: Fréchet mean of two identical matrices equals the original ──────
    #[test]
    fn frechet_mean_two_identical() {
        let p = diag2(4.0, 5.0);
        let mut matrices = Vec::new();
        matrices.extend_from_slice(&p);
        matrices.extend_from_slice(&p);
        let config = FrechetMeanConfig::default();
        let res = spd_frechet_mean(&matrices, 2, 2, &config).expect("ok");
        for (a, b) in res.mean.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-6, "mean != input: {a} vs {b}");
        }
    }

    // ── Test 3: Fréchet mean of k identity matrices is the identity ────────────
    #[test]
    fn frechet_mean_identity_matrices() {
        let n = 3;
        let ident = identity(n);
        let k = 5;
        let matrices: Vec<f64> = (0..k).flat_map(|_| ident.iter().copied()).collect();
        let config = FrechetMeanConfig::default();
        let res = spd_frechet_mean(&matrices, k, n, &config).expect("ok");
        for (i, (&a, &b)) in res.mean.iter().zip(ident.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "element {i}: mean={a}, expected={b}");
        }
    }

    // ── Test 4: Fréchet mean converges for 3 generic 2×2 SPD matrices ──────────
    #[test]
    fn frechet_mean_converges() {
        let p1 = diag2(1.0, 2.0);
        let p2 = diag2(2.0, 3.0);
        let p3 = diag2(3.0, 5.0);
        let mut matrices = Vec::new();
        matrices.extend_from_slice(&p1);
        matrices.extend_from_slice(&p2);
        matrices.extend_from_slice(&p3);
        let config = FrechetMeanConfig::default();
        let res = spd_frechet_mean(&matrices, 3, 2, &config).expect("ok");
        assert!(res.converged, "Karcher flow did not converge");
        assert!(
            res.final_grad_norm < 1e-7,
            "grad norm too large: {}",
            res.final_grad_norm
        );
    }

    // ── Test 5: Fréchet mean result is a valid SPD matrix ──────────────────────
    #[test]
    fn frechet_mean_result_is_spd() {
        let n = 3;
        // Use non-trivially diagonal 3×3 SPD matrices (diagonals only, still SPD).
        let p1 = diag3(1.0, 2.0, 3.0);
        let p2 = diag3(2.0, 4.0, 1.0);
        let p3 = diag3(3.0, 1.0, 2.0);
        let p4 = diag3(1.5, 2.5, 4.0);
        let k = 4;
        let mut matrices = Vec::new();
        for p in [&p1, &p2, &p3, &p4] {
            matrices.extend_from_slice(p);
        }
        let config = FrechetMeanConfig::default();
        let res = spd_frechet_mean(&matrices, k, n, &config).expect("ok");
        assert!(
            is_spd(&res.mean, n),
            "Fréchet mean is not SPD; mean={:?}",
            res.mean
        );
    }

    // ── Test 6: Fréchet mean of two matrices is the geodesic midpoint ──────────
    #[test]
    fn frechet_mean_is_midpoint() {
        // For diagonal SPD, the affine-invariant geodesic midpoint of diag(a) and diag(b)
        // is diag(sqrt(a*b)) component-wise.
        let p = diag2(1.0, 1.0); // identity
        let q = diag2(4.0, 9.0);
        let mut matrices = Vec::new();
        matrices.extend_from_slice(&p);
        matrices.extend_from_slice(&q);
        let config = FrechetMeanConfig {
            tol: 1e-10,
            ..Default::default()
        };
        let res = spd_frechet_mean(&matrices, 2, 2, &config).expect("ok");
        // Geodesic midpoint: diag(sqrt(1*4), sqrt(1*9)) = diag(2, 3)
        let expected = diag2(2.0, 3.0);
        for (a, b) in res.mean.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "midpoint element: got {a}, expected {b}"
            );
        }
    }

    // ── Test 7: k-means with single cluster ─────────────────────────────────────
    #[test]
    fn kmeans_single_cluster() {
        let n = 2;
        let mats: Vec<f64> = [diag2(1.0, 2.0), diag2(2.0, 3.0), diag2(3.0, 4.0)]
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        let config = SpdKmeansConfig {
            n_clusters: 1,
            ..Default::default()
        };
        let res = spd_kmeans(&mats, 3, n, &config).expect("ok");
        assert_eq!(res.labels.len(), 3);
        assert!(res.labels.iter().all(|&l| l == 0));
        // n_clusters=1 -> single centroid stored as flattened n×n matrix.
        assert_eq!(res.centroids.len(), n * n);

        // The single centroid should be close to the Fréchet mean of all 3.
        let freq = spd_frechet_mean(&mats, 3, n, &FrechetMeanConfig::default()).expect("ok");
        for (a, b) in res.centroids.iter().zip(freq.mean.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "centroid element {a} vs frechet mean {b}"
            );
        }
    }

    // ── Test 8: k-means on two well-separated clusters ──────────────────────────
    #[test]
    fn kmeans_two_well_separated_clusters() {
        let n = 2;
        // Group A: small matrices diag(0.1, 0.1), ...
        // Group B: large matrices diag(100.0, 100.0), ...
        let group_a: Vec<Vec<f64>> = vec![
            diag2(0.1, 0.1),
            diag2(0.2, 0.15),
            diag2(0.15, 0.2),
            diag2(0.12, 0.18),
            diag2(0.18, 0.12),
        ];
        let group_b: Vec<Vec<f64>> = vec![
            diag2(100.0, 100.0),
            diag2(120.0, 80.0),
            diag2(80.0, 120.0),
            diag2(110.0, 90.0),
            diag2(90.0, 110.0),
        ];
        let all_mats: Vec<f64> = group_a
            .iter()
            .chain(group_b.iter())
            .flat_map(|m| m.iter().copied())
            .collect();
        let n_matrices = 10;
        let config = SpdKmeansConfig {
            n_clusters: 2,
            n_restarts: 5,
            ..Default::default()
        };
        let res = spd_kmeans(&all_mats, n_matrices, n, &config).expect("ok");

        // All points in group A should share the same label, same for group B,
        // and the two labels should differ.
        let label_a = res.labels[0];
        let label_b = res.labels[5];
        assert_ne!(label_a, label_b, "groups must be in distinct clusters");
        for &l in &res.labels[..5] {
            assert_eq!(l, label_a, "group A point misclassified");
        }
        for &l in &res.labels[5..] {
            assert_eq!(l, label_b, "group B point misclassified");
        }
    }

    // ── Test 9: labels length matches n_matrices ─────────────────────────────────
    #[test]
    fn kmeans_labels_length() {
        let n = 2;
        let mats: Vec<f64> = (1..=7)
            .flat_map(|v| diag2(v as f64, (v + 1) as f64))
            .collect();
        let config = SpdKmeansConfig {
            n_clusters: 3,
            ..Default::default()
        };
        let res = spd_kmeans(&mats, 7, n, &config).expect("ok");
        assert_eq!(res.labels.len(), 7);
    }

    // ── Test 10: inertia is non-negative ─────────────────────────────────────────
    #[test]
    fn kmeans_inertia_nonneg() {
        let n = 2;
        let mats: Vec<f64> = (1..=6)
            .flat_map(|v| diag2(v as f64, (v + 2) as f64))
            .collect();
        let config = SpdKmeansConfig {
            n_clusters: 2,
            ..Default::default()
        };
        let res = spd_kmeans(&mats, 6, n, &config).expect("ok");
        assert!(res.inertia >= 0.0, "inertia is negative: {}", res.inertia);
    }

    // ── Test 11: centroids have correct shape ─────────────────────────────────────
    #[test]
    fn kmeans_centroids_shape() {
        let n = 3;
        let k_data = 9;
        let n_clusters = 3;
        let mats: Vec<f64> = (1..=k_data)
            .flat_map(|v| diag3(v as f64, (v + 1) as f64, (v + 2) as f64))
            .collect();
        let config = SpdKmeansConfig {
            n_clusters,
            ..Default::default()
        };
        let res = spd_kmeans(&mats, k_data, n, &config).expect("ok");
        assert_eq!(
            res.centroids.len(),
            n_clusters * n * n,
            "centroids length mismatch"
        );
    }

    // ── Test 12: n_clusters > n_matrices yields an error ─────────────────────────
    #[test]
    fn kmeans_invalid_n_clusters_error() {
        let n = 2;
        let mats = diag2(1.0, 2.0);
        let config = SpdKmeansConfig {
            n_clusters: 5, // > 1 data point
            ..Default::default()
        };
        let err = spd_kmeans(&mats, 1, n, &config);
        assert!(err.is_err(), "should error when n_clusters > n_matrices");
    }

    // ── Test 13: n_clusters = 0 yields an error ───────────────────────────────────
    #[test]
    fn kmeans_n_clusters_zero_error() {
        let n = 2;
        let mats: Vec<f64> = diag2(1.0, 2.0).into_iter().chain(diag2(3.0, 4.0)).collect();
        let config = SpdKmeansConfig {
            n_clusters: 0,
            ..Default::default()
        };
        let err = spd_kmeans(&mats, 2, n, &config);
        assert!(err.is_err(), "should error when n_clusters = 0");
    }

    // ── Test 14: more restarts give inertia <= fewer restarts ─────────────────────
    #[test]
    fn kmeans_multiple_restarts_improves() {
        let n = 2;
        // 8 matrices in 2 clusters, slightly noisy.
        let mats: Vec<f64> = [
            diag2(1.0, 1.0),
            diag2(1.1, 0.9),
            diag2(0.9, 1.1),
            diag2(1.05, 0.95),
            diag2(10.0, 10.0),
            diag2(11.0, 9.0),
            diag2(9.0, 11.0),
            diag2(10.5, 9.5),
        ]
        .iter()
        .flat_map(|m| m.iter().copied())
        .collect();

        let base_config = SpdKmeansConfig {
            n_clusters: 2,
            seed: 123,
            ..Default::default()
        };

        let config_1 = SpdKmeansConfig {
            n_restarts: 1,
            ..base_config.clone()
        };
        let config_5 = SpdKmeansConfig {
            n_restarts: 5,
            ..base_config.clone()
        };

        let res_1 = spd_kmeans(&mats, 8, n, &config_1).expect("ok");
        let res_5 = spd_kmeans(&mats, 8, n, &config_5).expect("ok");

        // 5 restarts must produce inertia <= 1 restart (more chances to find optimum).
        assert!(
            res_5.inertia <= res_1.inertia + 1e-8,
            "5-restart inertia ({}) > 1-restart inertia ({})",
            res_5.inertia,
            res_1.inertia
        );
    }
}

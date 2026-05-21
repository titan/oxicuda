//! Trimap dimensionality reduction (Wang et al., 2021).
//!
//! # Algorithm Overview
//!
//! Trimap preserves both local *and* global structure by constructing a set of ordered
//! triplets `(anchor i, near j, far k)` from the original high-dimensional space and then
//! optimising an embedding `Y ∈ ℝ^{n × d}` so that each triplet's constraint is satisfied:
//!
//! ```text
//! d_Y(i, j) ≪ d_Y(i, k)     for all (i, j, k)
//! ```
//!
//! ## Pipeline
//!
//! 1. **kNN** — for each anchor `i` find its `n_inliers` nearest neighbours.
//! 2. **Far sampling** — for each anchor sample `n_outliers` points outside the kNN set.
//! 3. **Triplet formation** — combine each `(i, near_j)` with a random far point `far_k`.
//! 4. **Distance-based weights** — `w = sqrt(d_orig(i,j)) + 1` for inlier triplets;
//!    `w = 1` for outlier / random triplets.
//! 5. **Logistic loss** — `l = 1 / (1 + exp(γ (d_ij − d_ik)))`, minimising pushes
//!    `d_ij ≪ d_ik`.
//! 6. **Momentum SGD** with cosine-annealing learning rate, optional PCA warm start.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Initialisation strategy for the Trimap embedding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrimapInit {
    /// Draw each coordinate i.i.d. from N(0, 1e-4).
    Random,
    /// Use the top-`n_components` PCA projection (scaled to unit variance) as the
    /// starting point.  Falls back to [`TrimapInit::Random`] if `n_components > dim`.
    Pca,
}

/// Hyper-parameter bundle for [`trimap`].
#[derive(Debug, Clone)]
pub struct TrimapConfig {
    /// Embedding dimensionality (default: 2).
    pub n_components: usize,
    /// Number of inlier (near) neighbours per anchor (default: 12).
    pub n_inliers: usize,
    /// Number of outlier (far) samples per anchor (default: 4).
    pub n_outliers: usize,
    /// Additional random triplets per anchor (default: 3).
    pub n_random: usize,
    /// Total SGD iterations (default: 400).
    pub n_iter: usize,
    /// Initial learning rate (default: 1.0).
    pub lr: f64,
    /// Loss sharpness γ — controls the steepness of the logistic curve (default: 1.0).
    pub gamma: f64,
    /// Enable distance-weighted triplet importance (default: `true`).
    pub weight_adj: bool,
    /// Initialisation strategy.
    pub init: TrimapInit,
    /// RNG seed.
    pub seed: u64,
}

impl Default for TrimapConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_inliers: 12,
            n_outliers: 4,
            n_random: 3,
            n_iter: 400,
            lr: 1.0,
            gamma: 1.0,
            weight_adj: true,
            init: TrimapInit::Pca,
            seed: 0,
        }
    }
}

/// Outcome of a [`trimap`] call.
pub struct TrimapResult {
    /// Row-major embedding of shape `[n_samples × n_components]`.
    pub embedding: Vec<f64>,
    /// Number of input samples.
    pub n_samples: usize,
    /// Embedding dimensionality.
    pub n_components: usize,
    /// Total number of triplets used during optimisation.
    pub n_triplets: usize,
    /// Weighted loss on the final embedding.
    pub final_loss: f64,
    /// Loss snapshot recorded every 50 iterations (plus iteration 0 and the last).
    pub loss_history: Vec<f64>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal types
// ──────────────────────────────────────────────────────────────────────────────

/// A single weighted triplet `(anchor, near, far, weight)`.
#[derive(Clone, Copy)]
struct Triplet {
    anchor: usize,
    near: usize,
    far: usize,
    weight: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Fit a Trimap embedding.
///
/// # Parameters
/// - `data` — row-major input matrix of shape `[n_samples × dim]`.
/// - `n_samples` — number of rows.
/// - `dim` — number of input dimensions.
/// - `config` — algorithm hyper-parameters.
///
/// # Errors
/// Returns [`ManifoldError`] for invalid inputs or degenerate configurations.
pub fn trimap(
    data: &[f64],
    n_samples: usize,
    dim: usize,
    config: &TrimapConfig,
) -> ManifoldResult<TrimapResult> {
    // ── 1. Input validation ──────────────────────────────────────────────────
    validate_inputs(data, n_samples, dim, config)?;

    let n = n_samples;
    let d_in = dim;
    let d_out = config.n_components;

    // Clamp n_inliers so we always have at least 1 near neighbour and never
    // request more neighbours than exist.
    let n_inliers = config.n_inliers.min(n - 1).max(1);
    let n_outliers = config.n_outliers.max(1);
    let n_random = config.n_random;

    let mut rng = LcgRng::new(config.seed);

    // ── 2. Pairwise squared distances (input space) ──────────────────────────
    let sq_dists = compute_sq_distance_matrix(data, n, d_in);

    // ── 3. kNN indices & distances for each anchor ───────────────────────────
    let (knn_idx, knn_dist) = brute_knn(&sq_dists, n, n_inliers);

    // ── 4. Build triplet list ────────────────────────────────────────────────
    let triplets = build_triplets(
        &sq_dists,
        &knn_idx,
        &knn_dist,
        n,
        n_inliers,
        n_outliers,
        n_random,
        config.weight_adj,
        &mut rng,
    );

    let n_triplets = triplets.len();
    if n_triplets == 0 {
        return Err(ManifoldError::InvalidConfiguration(
            "no triplets formed — increase n_inliers or n_samples".into(),
        ));
    }

    // ── 5. Initialise embedding ───────────────────────────────────────────────
    let mut y = initialise_embedding(data, n, d_in, d_out, &config.init, &mut rng)?;

    // ── 6. Momentum SGD with cosine-annealing LR ─────────────────────────────
    let (final_loss, loss_history) = optimise(
        &mut y,
        &triplets,
        n,
        d_out,
        config.n_iter,
        config.lr,
        config.gamma,
        &mut rng,
    );

    Ok(TrimapResult {
        embedding: y,
        n_samples: n,
        n_components: d_out,
        n_triplets,
        final_loss,
        loss_history,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Validation
// ──────────────────────────────────────────────────────────────────────────────

fn validate_inputs(
    data: &[f64],
    n_samples: usize,
    dim: usize,
    config: &TrimapConfig,
) -> ManifoldResult<()> {
    if n_samples < 3 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_samples".into(),
            reason: format!("need ≥ 3 samples to form triplets, got {n_samples}"),
        });
    }
    if dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if data.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![data.len()],
        });
    }
    if config.n_components == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if config.lr <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "lr".into(),
            reason: "must be positive".into(),
        });
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Distance utilities
// ──────────────────────────────────────────────────────────────────────────────

/// Compute a symmetric `n × n` squared-Euclidean distance matrix (row-major).
fn compute_sq_distance_matrix(x: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut d = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0_f64;
            for k in 0..dim {
                let diff = x[i * dim + k] - x[j * dim + k];
                s += diff * diff;
            }
            d[i * n + j] = s;
            d[j * n + i] = s;
        }
    }
    d
}

/// Squared Euclidean distance between rows `i` and `j` of the embedding `y`.
#[inline]
fn embedding_sq_dist(y: &[f64], i: usize, j: usize, dim: usize) -> f64 {
    let mut s = 0.0_f64;
    for k in 0..dim {
        let diff = y[i * dim + k] - y[j * dim + k];
        s += diff * diff;
    }
    s
}

// ──────────────────────────────────────────────────────────────────────────────
// brute-force kNN on a precomputed distance matrix
// ──────────────────────────────────────────────────────────────────────────────

/// Returns `(knn_indices, knn_sq_dists)` both of shape `[n × k]`.
fn brute_knn(sq_dists: &[f64], n: usize, k: usize) -> (Vec<usize>, Vec<f64>) {
    let mut idx = vec![0usize; n * k];
    let mut dist = vec![0.0_f64; n * k];
    let mut buf: Vec<(f64, usize)> = Vec::with_capacity(n - 1);

    for i in 0..n {
        buf.clear();
        for j in 0..n {
            if j == i {
                continue;
            }
            buf.push((sq_dists[i * n + j], j));
        }
        buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let effective_k = k.min(buf.len());
        for kk in 0..effective_k {
            idx[i * k + kk] = buf[kk].1;
            dist[i * k + kk] = buf[kk].0;
        }
    }
    (idx, dist)
}

// ──────────────────────────────────────────────────────────────────────────────
// Triplet construction
// ──────────────────────────────────────────────────────────────────────────────

fn build_triplets(
    sq_dists: &[f64],
    knn_idx: &[usize],
    knn_dist: &[f64],
    n: usize,
    n_inliers: usize,
    n_outliers: usize,
    n_random: usize,
    weight_adj: bool,
    rng: &mut LcgRng,
) -> Vec<Triplet> {
    let mut triplets: Vec<Triplet> = Vec::with_capacity(n * (n_inliers * n_outliers + n_random));

    // Build a set lookup for the kNN of each anchor to exclude them from "far" sampling.
    // We use a sorted small vec for O(k) membership test (k ≤ 64 in practice).
    let mut knn_set: Vec<usize> = Vec::with_capacity(n_inliers);

    for anchor in 0..n {
        // Collect kNN set for fast exclusion
        knn_set.clear();
        for kk in 0..n_inliers {
            knn_set.push(knn_idx[anchor * n_inliers + kk]);
        }
        knn_set.sort_unstable();

        // ── Inlier × outlier triplets ────────────────────────────────────────
        for ni in 0..n_inliers {
            let near = knn_idx[anchor * n_inliers + ni];
            let near_sq_d = knn_dist[anchor * n_inliers + ni];

            // Weight: sqrt of original distance + 1 (inlier importance).
            let w_inlier = if weight_adj {
                near_sq_d.sqrt() + 1.0
            } else {
                1.0
            };

            for _ in 0..n_outliers {
                let far = sample_far_point(anchor, &knn_set, n, rng);
                triplets.push(Triplet {
                    anchor,
                    near,
                    far,
                    weight: w_inlier,
                });
            }
        }

        // ── Additional random triplets ────────────────────────────────────────
        for _ in 0..n_random {
            // Near: pick uniformly from the kNN
            let ni = rng.next_usize(n_inliers);
            let near = knn_idx[anchor * n_inliers + ni];

            // Far: any point not equal to anchor (may coincide with kNN — OK for random tripltes)
            let far = sample_random_not_self(anchor, n, rng);

            // Weight for random triplets: based on original distance to the near point
            let near_sq_d = sq_dists[anchor * n + near];
            let w_rand = if weight_adj {
                near_sq_d.sqrt() + 1.0
            } else {
                1.0
            };

            triplets.push(Triplet {
                anchor,
                near,
                far,
                weight: w_rand,
            });
        }
    }

    triplets
}

/// Sample a point that is not `anchor` and not in `knn_set` (sorted).
/// Falls back to any non-anchor point if no far point can be found after a bounded number of tries.
fn sample_far_point(anchor: usize, knn_set: &[usize], n: usize, rng: &mut LcgRng) -> usize {
    // Maximum attempts before falling back to a non-knn non-self point via scan
    const MAX_TRIES: usize = 32;
    for _ in 0..MAX_TRIES {
        let candidate = rng.next_usize(n);
        if candidate != anchor && knn_set.binary_search(&candidate).is_err() {
            return candidate;
        }
    }
    // Deterministic fallback: find the first point that is neither anchor nor kNN
    for candidate in 0..n {
        if candidate != anchor && knn_set.binary_search(&candidate).is_err() {
            return candidate;
        }
    }
    // Last resort: any non-self point (happens only when n ≤ n_inliers + 1)
    sample_random_not_self(anchor, n, rng)
}

/// Sample any point index in `[0, n)` that is not `anchor`.
fn sample_random_not_self(anchor: usize, n: usize, rng: &mut LcgRng) -> usize {
    loop {
        let c = rng.next_usize(n);
        if c != anchor {
            return c;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Embedding initialisation
// ──────────────────────────────────────────────────────────────────────────────

fn initialise_embedding(
    data: &[f64],
    n: usize,
    d_in: usize,
    d_out: usize,
    init: &TrimapInit,
    rng: &mut LcgRng,
) -> ManifoldResult<Vec<f64>> {
    match init {
        TrimapInit::Random => Ok(random_init(n, d_out, rng)),
        TrimapInit::Pca => {
            if d_out > d_in {
                // PCA cannot produce more components than input dims: fall back to random
                Ok(random_init(n, d_out, rng))
            } else {
                pca_init(data, n, d_in, d_out, rng)
            }
        }
    }
}

fn random_init(n: usize, d_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut y = vec![0.0_f64; n * d_out];
    for v in &mut y {
        *v = rng.next_normal() * 0.01;
    }
    y
}

/// Minimal PCA-based initialisation (power-iteration covariance eigen, top-k projection).
fn pca_init(
    data: &[f64],
    n: usize,
    d_in: usize,
    d_out: usize,
    rng: &mut LcgRng,
) -> ManifoldResult<Vec<f64>> {
    // Column means
    let mut mean = vec![0.0_f64; d_in];
    for i in 0..n {
        for j in 0..d_in {
            mean[j] += data[i * d_in + j];
        }
    }
    let n_f = n as f64;
    for m in &mut mean {
        *m /= n_f;
    }

    // Centered data
    let mut centered = vec![0.0_f64; n * d_in];
    for i in 0..n {
        for j in 0..d_in {
            centered[i * d_in + j] = data[i * d_in + j] - mean[j];
        }
    }

    // Compute top-d_out principal components via deflated power iteration
    let mut components: Vec<Vec<f64>> = Vec::with_capacity(d_out);
    let mut residual = centered.clone();

    for _c in 0..d_out {
        // Random unit vector
        let mut v: Vec<f64> = (0..d_in).map(|_| rng.next_normal()).collect();
        normalise_vec(&mut v);

        // Power iteration: v ← X^T (X v) / ||X^T (X v)||
        for _ in 0..40 {
            // Xv
            let mut xv = vec![0.0_f64; n];
            for i in 0..n {
                for j in 0..d_in {
                    xv[i] += residual[i * d_in + j] * v[j];
                }
            }
            // X^T (Xv)
            let mut xtxv = vec![0.0_f64; d_in];
            for i in 0..n {
                for j in 0..d_in {
                    xtxv[j] += residual[i * d_in + j] * xv[i];
                }
            }
            let norm: f64 = xtxv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-14 {
                break;
            }
            for (vi, xi) in v.iter_mut().zip(&xtxv) {
                *vi = xi / norm;
            }
        }

        // Deflate: residual ← residual - (residual v) v^T
        let mut proj = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..d_in {
                proj[i] += residual[i * d_in + j] * v[j];
            }
        }
        for i in 0..n {
            for j in 0..d_in {
                residual[i * d_in + j] -= proj[i] * v[j];
            }
        }
        components.push(v);
    }

    // Project original (centered) data onto the first d_out components
    let mut y = vec![0.0_f64; n * d_out];
    for i in 0..n {
        for c in 0..d_out {
            let mut acc = 0.0_f64;
            for j in 0..d_in {
                acc += centered[i * d_in + j] * components[c][j];
            }
            y[i * d_out + c] = acc;
        }
    }

    // Scale so each component has std ≈ 1e-2 to match the random init scale
    for c in 0..d_out {
        let mut var = 0.0_f64;
        for i in 0..n {
            var += y[i * d_out + c].powi(2);
        }
        var /= n_f;
        let std = var.sqrt().max(1e-14);
        for i in 0..n {
            y[i * d_out + c] = y[i * d_out + c] / std * 0.01;
        }
    }

    Ok(y)
}

fn normalise_vec(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-14 {
        for vi in v.iter_mut() {
            *vi /= norm;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Loss & gradient
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluate the total weighted logistic loss on the current embedding.
fn evaluate_loss(y: &[f64], triplets: &[Triplet], d_out: usize, gamma: f64) -> f64 {
    let mut total = 0.0_f64;
    for t in triplets {
        let d_ij = embedding_sq_dist(y, t.anchor, t.near, d_out);
        let d_ik = embedding_sq_dist(y, t.anchor, t.far, d_out);
        // l = 1 / (1 + exp(γ (d_ik − d_ij)))   (0 when satisfied, 1 when violated)
        let exponent = gamma * (d_ij - d_ik);
        let l = logistic(exponent);
        total += t.weight * l;
    }
    total
}

/// Numerically stable logistic / sigmoid function: `σ(x) = 1 / (1 + exp(−x))`.
/// We call it with `x = γ(d_ij − d_ik)` so we minimise the loss toward 0.
#[inline]
fn logistic(x: f64) -> f64 {
    // Clamp to avoid overflow in exp
    let x = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-x).exp())
}

// ──────────────────────────────────────────────────────────────────────────────
// Optimiser (momentum SGD + cosine LR annealing)
// ──────────────────────────────────────────────────────────────────────────────

fn optimise(
    y: &mut [f64],
    triplets: &[Triplet],
    n: usize,
    d_out: usize,
    n_iter: usize,
    lr_init: f64,
    gamma: f64,
    rng: &mut LcgRng,
) -> (f64, Vec<f64>) {
    let n_triplets = triplets.len();
    let mut velocity = vec![0.0_f64; n * d_out];
    let momentum = 0.9_f64;
    let lr_min = lr_init / 100.0;
    // Normalise full-batch gradient by triplet count so that the learning rate
    // is independent of dataset size. Without this, lr behaves like
    // `lr * n_triplets`, which causes momentum-driven divergence on larger
    // problems (loss oscillates / embedding blows up).
    let inv_n_triplets = 1.0 / (n_triplets.max(1) as f64);

    let mut loss_history: Vec<f64> = Vec::new();

    // Record initial loss
    let initial_loss = evaluate_loss(y, triplets, d_out, gamma);
    loss_history.push(initial_loss);

    if n_iter == 0 {
        return (initial_loss, loss_history);
    }

    // Shuffle buffer for triplet indices
    let mut order: Vec<usize> = (0..n_triplets).collect();

    let mut final_loss = initial_loss;

    for iter in 0..n_iter {
        // Cosine annealing LR schedule
        let progress = iter as f64 / n_iter as f64;
        let lr =
            lr_min + 0.5 * (lr_init - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos());

        // Shuffle triplet order (Fisher-Yates with LcgRng)
        fisher_yates_shuffle(&mut order, rng);

        // Gradient accumulator (zeroed each iteration)
        let mut grad = vec![0.0_f64; n * d_out];

        // Accumulate gradients for all triplets in this iteration
        for &t_idx in &order {
            let t = triplets[t_idx];
            accumulate_triplet_grad(y, &mut grad, &t, d_out, gamma);
        }

        // Momentum update (gradient is averaged over triplets to keep the
        // effective step size dataset-independent).
        for i in 0..n * d_out {
            velocity[i] = momentum * velocity[i] - lr * grad[i] * inv_n_triplets;
            y[i] += velocity[i];
        }

        // Re-centre each output dimension to zero mean (prevents drift)
        centre_embedding(y, n, d_out);

        // Record loss every 50 iterations and at the final step
        if (iter + 1) % 50 == 0 || iter + 1 == n_iter {
            let loss = evaluate_loss(y, triplets, d_out, gamma);
            loss_history.push(loss);
            if iter + 1 == n_iter {
                final_loss = loss;
            }
        }
    }

    (final_loss, loss_history)
}

/// Add the gradient contribution of one triplet to `grad`.
///
/// Loss per triplet: `l = σ(γ(d_ij − d_ik))`  where `σ` is the logistic function.
///
/// Derivatives:
/// ```text
/// ∂l/∂d_ij = γ σ(1−σ)
/// ∂l/∂d_ik = −γ σ(1−σ)
/// ∂d_ij/∂y_i = 2(y_i − y_j),  ∂d_ij/∂y_j = −2(y_i − y_j)
/// ∂d_ik/∂y_i = 2(y_i − y_k),  ∂d_ik/∂y_k = −2(y_i − y_k)
/// ```
fn accumulate_triplet_grad(y: &[f64], grad: &mut [f64], t: &Triplet, d_out: usize, gamma: f64) {
    let a = t.anchor;
    let j = t.near;
    let k = t.far;
    let w = t.weight;

    let d_ij = embedding_sq_dist(y, a, j, d_out);
    let d_ik = embedding_sq_dist(y, a, k, d_out);

    let exponent = (gamma * (d_ij - d_ik)).clamp(-500.0, 500.0);
    let sig = 1.0 / (1.0 + (-exponent).exp());
    let sig_deriv = sig * (1.0 - sig); // σ(1−σ)

    // ∂l/∂d_ij = γ w σ(1−σ)
    let dl_ddij = w * gamma * sig_deriv;
    // ∂l/∂d_ik = −γ w σ(1−σ)
    let dl_ddik = -dl_ddij;

    // Gradient for each output dimension
    for kk in 0..d_out {
        let ya = y[a * d_out + kk];
        let yj = y[j * d_out + kk];
        let yk = y[k * d_out + kk];

        let diff_ij = ya - yj;
        let diff_ik = ya - yk;

        // ∂l/∂y_a = dl_ddij * 2 diff_ij + dl_ddik * 2 diff_ik
        grad[a * d_out + kk] += dl_ddij * 2.0 * diff_ij + dl_ddik * 2.0 * diff_ik;
        // ∂l/∂y_j = dl_ddij * (-2 diff_ij)
        grad[j * d_out + kk] += dl_ddij * (-2.0) * diff_ij;
        // ∂l/∂y_k = dl_ddik * (-2 diff_ik)
        grad[k * d_out + kk] += dl_ddik * (-2.0) * diff_ik;
    }
}

/// Remove the per-dimension mean from `y` to prevent unbounded drift.
fn centre_embedding(y: &mut [f64], n: usize, d_out: usize) {
    for c in 0..d_out {
        let mut mean = 0.0_f64;
        for i in 0..n {
            mean += y[i * d_out + c];
        }
        mean /= n as f64;
        for i in 0..n {
            y[i * d_out + c] -= mean;
        }
    }
}

/// In-place Fisher-Yates shuffle using `LcgRng`.
fn fisher_yates_shuffle(v: &mut [usize], rng: &mut LcgRng) {
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        v.swap(i, j);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_two_cluster_data(n_per_cluster: usize, dim: usize, seed: u64) -> Vec<f64> {
        let n = n_per_cluster * 2;
        let mut data = vec![0.0_f64; n * dim];
        let mut rng = LcgRng::new(seed);
        for i in 0..n {
            let centre = if i < n_per_cluster { 10.0 } else { -10.0 };
            for d in 0..dim {
                data[i * dim + d] = centre + 0.3 * rng.next_normal();
            }
        }
        data
    }

    fn cluster_centre(embedding: &[f64], start: usize, end: usize, d_out: usize) -> Vec<f64> {
        let count = (end - start) as f64;
        let mut centre = vec![0.0_f64; d_out];
        for i in start..end {
            for k in 0..d_out {
                centre[k] += embedding[i * d_out + k];
            }
        }
        for c in &mut centre {
            *c /= count;
        }
        centre
    }

    fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    // ── Test 1: output shape is correct ───────────────────────────────────────
    #[test]
    fn test_embedding_shape() {
        let n = 20;
        let dim = 4;
        let data = make_two_cluster_data(10, dim, 1);
        let cfg = TrimapConfig {
            n_iter: 10,
            n_inliers: 4,
            n_outliers: 2,
            n_random: 1,
            ..TrimapConfig::default()
        };
        let result = trimap(&data, n, dim, &cfg).expect("trimap should succeed");
        assert_eq!(result.embedding.len(), n * cfg.n_components);
        assert_eq!(result.n_samples, n);
        assert_eq!(result.n_components, cfg.n_components);
    }

    // ── Test 2: final loss is finite and non-negative ─────────────────────────
    #[test]
    fn test_loss_finite_nonneg() {
        let n = 15;
        let dim = 3;
        let data = make_two_cluster_data(8, dim, 2);
        let cfg = TrimapConfig {
            n_iter: 30,
            n_inliers: 3,
            n_outliers: 2,
            n_random: 1,
            ..TrimapConfig::default()
        };
        let result = trimap(&data[..n * dim], n, dim, &cfg).expect("ok");
        assert!(result.final_loss.is_finite(), "final loss must be finite");
        assert!(result.final_loss >= 0.0, "loss must be non-negative");
    }

    // ── Test 3: no NaN or infinite values in the embedding ───────────────────
    #[test]
    fn test_embedding_all_finite() {
        let n = 20;
        let dim = 5;
        let data = make_two_cluster_data(10, dim, 3);
        let cfg = TrimapConfig {
            n_iter: 50,
            n_inliers: 4,
            n_outliers: 2,
            n_random: 2,
            ..TrimapConfig::default()
        };
        let result = trimap(&data, n, dim, &cfg).expect("ok");
        for (i, v) in result.embedding.iter().enumerate() {
            assert!(v.is_finite(), "embedding[{i}] = {v} is not finite");
        }
    }

    // ── Test 4: loss_history is non-empty ─────────────────────────────────────
    #[test]
    fn test_loss_history_nonempty() {
        let n = 12;
        let dim = 3;
        let data = make_two_cluster_data(6, dim, 4);
        let cfg = TrimapConfig {
            n_iter: 100,
            n_inliers: 3,
            n_outliers: 2,
            n_random: 1,
            ..TrimapConfig::default()
        };
        let result = trimap(&data, n, dim, &cfg).expect("ok");
        assert!(
            !result.loss_history.is_empty(),
            "loss_history must not be empty"
        );
    }

    // ── Test 5: n_triplets > 0 ───────────────────────────────────────────────
    #[test]
    fn test_triplet_count_positive() {
        let n = 12;
        let dim = 3;
        let data = make_two_cluster_data(6, dim, 5);
        let cfg = TrimapConfig {
            n_iter: 10,
            n_inliers: 3,
            n_outliers: 2,
            n_random: 1,
            ..TrimapConfig::default()
        };
        let result = trimap(&data, n, dim, &cfg).expect("ok");
        assert!(result.n_triplets > 0, "must have at least one triplet");
    }

    // ── Test 6: two separated clusters end up separated in the embedding ──────
    #[test]
    fn test_cluster_separation() {
        let n_per = 20;
        let n = n_per * 2;
        let dim = 5;
        let data = make_two_cluster_data(n_per, dim, 6);
        let cfg = TrimapConfig {
            n_iter: 200,
            n_inliers: 8,
            n_outliers: 3,
            n_random: 2,
            lr: 0.5,
            seed: 6,
            ..TrimapConfig::default()
        };
        let result = trimap(&data, n, dim, &cfg).expect("ok");
        let d_out = cfg.n_components;

        let centre_a = cluster_centre(&result.embedding, 0, n_per, d_out);
        let centre_b = cluster_centre(&result.embedding, n_per, n, d_out);
        let inter = l2_distance(&centre_a, &centre_b);

        // Mean intra-cluster distance
        let mut intra = 0.0_f64;
        let mut count = 0usize;
        for i in 0..n_per {
            for j in (i + 1)..n_per {
                intra += l2_distance(
                    &result.embedding[i * d_out..(i + 1) * d_out],
                    &result.embedding[j * d_out..(j + 1) * d_out],
                );
                count += 1;
            }
        }
        intra /= count.max(1) as f64;

        assert!(
            inter > intra,
            "inter-cluster distance {inter:.4} should exceed intra-cluster {intra:.4}"
        );
    }

    // ── Test 7: n_components=3 works ──────────────────────────────────────────
    #[test]
    fn test_three_components() {
        let n = 20;
        let dim = 6;
        let data = make_two_cluster_data(10, dim, 7);
        let cfg = TrimapConfig {
            n_components: 3,
            n_iter: 30,
            n_inliers: 4,
            n_outliers: 2,
            n_random: 1,
            ..TrimapConfig::default()
        };
        let result = trimap(&data, n, dim, &cfg).expect("ok");
        assert_eq!(result.embedding.len(), n * 3);
        assert_eq!(result.n_components, 3);
        assert!(result.embedding.iter().all(|v| v.is_finite()));
    }

    // ── Test 8: n_inliers clamped when >= n_samples ───────────────────────────
    #[test]
    fn test_large_n_inliers_clamped() {
        let n = 8;
        let dim = 3;
        let data = make_two_cluster_data(4, dim, 8);
        // n_inliers = 100 >> n = 8 → should be clamped, not error
        let cfg = TrimapConfig {
            n_inliers: 100,
            n_iter: 10,
            n_outliers: 2,
            n_random: 1,
            ..TrimapConfig::default()
        };
        let result = trimap(&data, n, dim, &cfg);
        assert!(
            result.is_ok(),
            "large n_inliers should be clamped, not fail"
        );
    }

    // ── Test 9: n_samples < 3 → error ────────────────────────────────────────
    #[test]
    fn test_too_few_samples() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 × 3
        let cfg = TrimapConfig::default();
        let err = trimap(&data, 2, 3, &cfg);
        assert!(err.is_err(), "n_samples < 3 must return an error");
    }

    // ── Test 10: PCA init produces correct embedding shape ────────────────────
    #[test]
    fn test_pca_init_shape() {
        let n = 20;
        let dim = 6;
        let data = make_two_cluster_data(10, dim, 10);
        let cfg = TrimapConfig {
            n_iter: 20,
            n_inliers: 4,
            n_outliers: 2,
            n_random: 1,
            init: TrimapInit::Pca,
            ..TrimapConfig::default()
        };
        let result = trimap(&data, n, dim, &cfg).expect("PCA init should succeed");
        assert_eq!(result.embedding.len(), n * cfg.n_components);
        assert!(result.embedding.iter().all(|v| v.is_finite()));
    }

    // ── Test 11: n_iter=0 returns initial embedding with populated loss_history
    #[test]
    fn test_zero_iterations() {
        let n = 12;
        let dim = 3;
        let data = make_two_cluster_data(6, dim, 11);
        let cfg = TrimapConfig {
            n_iter: 0,
            n_inliers: 3,
            n_outliers: 2,
            n_random: 1,
            ..TrimapConfig::default()
        };
        let result = trimap(&data, n, dim, &cfg).expect("zero iter should succeed");
        assert_eq!(result.embedding.len(), n * cfg.n_components);
        // loss_history contains at least the initial loss
        assert!(!result.loss_history.is_empty());
        assert!(result.final_loss.is_finite());
    }

    // ── Test 12: weight_adj=false produces a valid embedding ─────────────────
    #[test]
    fn test_no_weight_adj() {
        let n = 16;
        let dim = 4;
        let data = make_two_cluster_data(8, dim, 12);
        let cfg = TrimapConfig {
            n_iter: 30,
            n_inliers: 4,
            n_outliers: 2,
            n_random: 1,
            weight_adj: false,
            ..TrimapConfig::default()
        };
        let result = trimap(&data, n, dim, &cfg).expect("weight_adj=false should succeed");
        assert_eq!(result.embedding.len(), n * cfg.n_components);
        assert!(result.embedding.iter().all(|v| v.is_finite()));
        assert!(result.final_loss.is_finite() && result.final_loss >= 0.0);
    }

    // ── Test 13: random vs PCA init both converge to a finite embedding ───────
    #[test]
    fn test_init_strategies_converge() {
        let n = 20;
        let dim = 5;
        let data = make_two_cluster_data(10, dim, 13);
        let base = TrimapConfig {
            n_iter: 50,
            n_inliers: 4,
            n_outliers: 2,
            n_random: 1,
            ..TrimapConfig::default()
        };

        let r_rnd = trimap(
            &data,
            n,
            dim,
            &TrimapConfig {
                init: TrimapInit::Random,
                ..base.clone()
            },
        )
        .expect("random init");
        let r_pca = trimap(
            &data,
            n,
            dim,
            &TrimapConfig {
                init: TrimapInit::Pca,
                ..base.clone()
            },
        )
        .expect("pca init");

        assert_eq!(r_rnd.embedding.len(), r_pca.embedding.len());
        assert!(r_rnd.embedding.iter().all(|v| v.is_finite()));
        assert!(r_pca.embedding.iter().all(|v| v.is_finite()));
    }

    // ── Test 14: logistic loss values are in [0, 1] ───────────────────────────
    #[test]
    fn test_logistic_bounds() {
        // logistic(-large) ≈ 0, logistic(0) = 0.5, logistic(large) ≈ 1
        assert!((logistic(-1000.0) - 0.0).abs() < 1e-6);
        assert!((logistic(0.0) - 0.5).abs() < 1e-9);
        assert!((logistic(1000.0) - 1.0).abs() < 1e-6);
        // All values must be in (0, 1)
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0_f64] {
            let v = logistic(x);
            assert!(v > 0.0 && v < 1.0, "logistic({x}) = {v} out of (0,1)");
        }
    }

    // ── Test 15: determinism with same seed ───────────────────────────────────
    #[test]
    fn test_determinism() {
        let n = 16;
        let dim = 4;
        let data = make_two_cluster_data(8, dim, 15);
        let cfg = TrimapConfig {
            n_iter: 40,
            n_inliers: 4,
            n_outliers: 2,
            n_random: 1,
            seed: 42,
            ..TrimapConfig::default()
        };
        let r1 = trimap(&data, n, dim, &cfg).expect("run 1");
        let r2 = trimap(&data, n, dim, &cfg).expect("run 2");
        assert_eq!(
            r1.embedding, r2.embedding,
            "same seed must give identical output"
        );
    }
}

//! PaCMAP — Pairwise Controlled Manifold Approximation Projection (Wang et al., 2021).
//!
//! # Algorithm Overview
//!
//! PaCMAP preserves both local *and* global structure by maintaining three types of
//! pairwise relationships and optimising them through three distinct phases:
//!
//! ## Pair Types
//!
//! 1. **Near pairs (NP)**: the k nearest neighbours of each anchor — provide local structure.
//! 2. **Mid-near pairs (MNP)**: neighbours in the 6th–50th rank range — provide global structure.
//! 3. **Far pairs (FP)**: random non-neighbour pairs — prevent collapse (repulsion).
//!
//! ## Loss Function
//!
//! ```text
//! L = w1 * L_NP + w2 * L_MNP + w3 * L_FP
//!
//! L_NP  = Σ w_ij · d_ij / (d_ij + 1)          (attractive, d_ij = ||y_i − y_j||²)
//! L_MNP = Σ w_ij · d_ij / (d_ij + 1)          (attractive)
//! L_FP  = Σ w_ik · 1   / (d_ik + 1)          (repulsive — smaller d_ik raises loss)
//! ```
//!
//! ## Phase Schedule
//!
//! | Phase | Iterations      | w1  | w2  | w3 |
//! |-------|-----------------|-----|-----|----|
//! | 1     | [0,   n/3)      | 2   | 500 | 1  |
//! | 2     | [n/3, 2n/3)     | 3   | 500 | 1  |
//! | 3     | [2n/3, n)       | 1   | 0   | 1  |
//!
//! ## Optimiser
//!
//! Adam (β1=0.9, β2=0.999, ε=1e-8) with bias correction, one step per full-gradient epoch.
//! The embedding is re-centred after every update to prevent drift.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Initialisation strategy for the PaCMAP embedding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaCMapInit {
    /// Draw each coordinate i.i.d. from N(0, 1e-4).
    Random,
    /// Use the top-`n_components` PCA projection (scaled) as the starting point.
    /// Falls back to [`PaCMapInit::Random`] if `n_components > dim`.
    Pca,
}

/// Hyper-parameter bundle for [`pacmap`].
#[derive(Debug, Clone)]
pub struct PaCMapConfig {
    /// Embedding dimensionality (default: 2).
    pub n_components: usize,
    /// Number of near (kNN) pairs per anchor (default: 10).
    pub n_neighbors: usize,
    /// Number of mid-near pairs per anchor (default: 5).
    pub n_mid_near: usize,
    /// Number of far pairs per anchor (default: 20).
    pub n_far: usize,
    /// Learning rate for Adam (default: 1.0).
    pub lr: f64,
    /// Total SGD iterations (default: 450).
    pub n_iter: usize,
    /// Initialisation strategy (default: [`PaCMapInit::Pca`]).
    pub init: PaCMapInit,
    /// RNG seed.
    pub seed: u64,
}

impl Default for PaCMapConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 10,
            n_mid_near: 5,
            n_far: 20,
            lr: 1.0,
            n_iter: 450,
            init: PaCMapInit::Pca,
            seed: 0,
        }
    }
}

/// Outcome of a [`pacmap`] call.
pub struct PaCMapResult {
    /// Row-major embedding of shape `[n_samples × n_components]`.
    pub embedding: Vec<f64>,
    /// Number of input samples.
    pub n_samples: usize,
    /// Embedding dimensionality.
    pub n_components: usize,
    /// Loss snapshot recorded every 50 iterations (and after the final step).
    pub loss_history: Vec<f64>,
    /// Weighted loss on the final embedding.
    pub final_loss: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal pair types
// ──────────────────────────────────────────────────────────────────────────────

/// A weighted near or mid-near pair (anchor, other, weight).
#[derive(Clone, Copy)]
struct AttractPair {
    anchor: usize,
    other: usize,
    /// Input-space distance weight: 1 / (||x_i − x_j||² + 1).
    weight: f64,
}

/// A far pair (anchor, far, weight=1.0 by default for repulsion).
#[derive(Clone, Copy)]
struct FarPair {
    anchor: usize,
    far: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Fit a PaCMAP embedding.
///
/// # Parameters
/// - `data`      — row-major input matrix `[n_samples × dim]`.
/// - `n_samples` — number of rows.
/// - `dim`       — number of input dimensions.
/// - `config`    — algorithm hyper-parameters.
///
/// # Errors
/// Returns [`ManifoldError`] for invalid inputs or degenerate configurations.
pub fn pacmap(
    data: &[f64],
    n_samples: usize,
    dim: usize,
    config: &PaCMapConfig,
) -> ManifoldResult<PaCMapResult> {
    // ── 1. Validate ──────────────────────────────────────────────────────────
    validate_inputs(data, n_samples, dim, config)?;

    let n = n_samples;
    let d_in = dim;
    let d_out = config.n_components;

    let mut rng = LcgRng::new(config.seed);

    // ── 2. Pairwise squared distances (input space) ──────────────────────────
    let sq_dists = compute_sq_distance_matrix(data, n, d_in);

    // Clamp n_neighbors to at most n-1.
    let k = config.n_neighbors.min(n - 1).max(1);

    // ── 3. Brute-force kNN (sorted ascending) ────────────────────────────────
    // sorted_neighbors[i] = [(dist, j), ...] sorted ascending for point i
    // We keep up to min(n-1, 50) neighbours for mid-near selection.
    let mid_range_end = (50usize).min(n - 1);
    let knn_width = mid_range_end.max(k);
    let (knn_idx, knn_sq_dist) = brute_knn_sorted(&sq_dists, n, knn_width);

    // ── 4. Build near pairs ──────────────────────────────────────────────────
    let near_pairs = build_near_pairs(&knn_idx, &knn_sq_dist, n, k);

    // ── 5. Build mid-near pairs ───────────────────────────────────────────────
    let mnp_count = config.n_mid_near;
    let mid_near_pairs = build_mid_near_pairs(
        &knn_idx,
        &knn_sq_dist,
        n,
        k,
        mnp_count,
        mid_range_end,
        &mut rng,
    );

    // ── 6. Build far pairs ────────────────────────────────────────────────────
    let fp_count = config.n_far;
    let far_pairs = build_far_pairs(&knn_idx, n, k, fp_count, &mut rng);

    // ── 7. Initialise embedding ───────────────────────────────────────────────
    let mut y = init_embedding(data, n, d_in, d_out, &config.init, &mut rng)?;

    // ── 8. Adam optimisation with phase schedule ──────────────────────────────
    let (final_loss, loss_history) = optimise_adam(
        &mut y,
        &near_pairs,
        &mid_near_pairs,
        &far_pairs,
        n,
        d_out,
        config.n_iter,
        config.lr,
    );

    Ok(PaCMapResult {
        embedding: y,
        n_samples: n,
        n_components: d_out,
        loss_history,
        final_loss,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Validation
// ──────────────────────────────────────────────────────────────────────────────

fn validate_inputs(
    data: &[f64],
    n_samples: usize,
    dim: usize,
    config: &PaCMapConfig,
) -> ManifoldResult<()> {
    if n_samples < 3 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_samples".into(),
            reason: format!("need ≥ 3 samples, got {n_samples}"),
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

/// Squared Euclidean distance between rows `i` and `j` of embedding `y`.
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
// Brute-force kNN (returns sorted ascending up to `width` neighbours)
// ──────────────────────────────────────────────────────────────────────────────

/// Returns `(knn_indices, knn_sq_dists)` both of shape `[n × width]`, sorted ascending.
///
/// `width` is clamped to `n − 1` automatically.
fn brute_knn_sorted(sq_dists: &[f64], n: usize, width: usize) -> (Vec<usize>, Vec<f64>) {
    let effective_width = width.min(n - 1);
    let mut idx = vec![0usize; n * effective_width];
    let mut dist = vec![0.0_f64; n * effective_width];
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
        let take = effective_width.min(buf.len());
        for kk in 0..take {
            idx[i * effective_width + kk] = buf[kk].1;
            dist[i * effective_width + kk] = buf[kk].0;
        }
    }
    (idx, dist)
}

// ──────────────────────────────────────────────────────────────────────────────
// Pair construction
// ──────────────────────────────────────────────────────────────────────────────

/// Build near pairs from the first `k` neighbours (indices 0..k).
///
/// Weight: `w_ij = 1 / (||x_i − x_j||² + 1)` (input-space distance).
fn build_near_pairs(
    knn_idx: &[usize],
    knn_sq_dist: &[f64],
    n: usize,
    k: usize,
) -> Vec<AttractPair> {
    let width = knn_sq_dist.len() / n;
    let mut pairs = Vec::with_capacity(n * k);
    for anchor in 0..n {
        for ni in 0..k {
            if ni >= width {
                break;
            }
            let other = knn_idx[anchor * width + ni];
            let d2 = knn_sq_dist[anchor * width + ni];
            let weight = 1.0 / (d2 + 1.0);
            pairs.push(AttractPair {
                anchor,
                other,
                weight,
            });
        }
    }
    pairs
}

/// Build mid-near pairs by sampling from the rank range `[k, mid_range_end)`.
///
/// If the range contains fewer than `count` candidates the effective sample
/// count is clamped to the available range size. For very small datasets where
/// the mid-near range is empty (all neighbours are used as near pairs), an
/// empty vector is returned gracefully.
///
/// Weight: `w_ij = 1 / (||x_i − x_j||² + 1)`.
fn build_mid_near_pairs(
    knn_idx: &[usize],
    knn_sq_dist: &[f64],
    n: usize,
    k: usize,
    count: usize,
    mid_range_end: usize,
    rng: &mut LcgRng,
) -> Vec<AttractPair> {
    let width = knn_sq_dist.len() / n;
    let mut pairs = Vec::with_capacity(n * count);

    for anchor in 0..n {
        // Available mid-near candidates: indices [k, min(mid_range_end, width))
        let lo = k;
        let hi = mid_range_end.min(width);
        if lo >= hi {
            // No mid-near range available (dataset too small or k already covers all).
            continue;
        }
        let range_len = hi - lo;
        let effective_count = count.min(range_len);

        // Sample `effective_count` distinct positions from [lo, hi)
        // For small ranges use a simple reservoir; for larger do repeated random picks
        // with rejection (bounded attempts).
        if effective_count == range_len {
            // Take all
            for pos in lo..hi {
                let other = knn_idx[anchor * width + pos];
                let d2 = knn_sq_dist[anchor * width + pos];
                let weight = 1.0 / (d2 + 1.0);
                pairs.push(AttractPair {
                    anchor,
                    other,
                    weight,
                });
            }
        } else {
            // Fisher-Yates partial shuffle to sample without replacement.
            let mut indices: Vec<usize> = (lo..hi).collect();
            for i in (1..range_len).rev() {
                let j = rng.next_usize(i + 1);
                indices.swap(i, j);
            }
            for pos in indices.into_iter().take(effective_count) {
                let other = knn_idx[anchor * width + pos];
                let d2 = knn_sq_dist[anchor * width + pos];
                let weight = 1.0 / (d2 + 1.0);
                pairs.push(AttractPair {
                    anchor,
                    other,
                    weight,
                });
            }
        }
    }
    pairs
}

/// Build far pairs by sampling random non-kNN points for each anchor.
///
/// Far pair weights are uniform (1.0); only the repulsion gradient uses them.
fn build_far_pairs(
    knn_idx: &[usize],
    n: usize,
    k: usize,
    count: usize,
    rng: &mut LcgRng,
) -> Vec<FarPair> {
    let width = knn_idx.len() / n;
    let mut pairs = Vec::with_capacity(n * count);
    // Sorted kNN set for fast membership test (binary search).
    let mut knn_set: Vec<usize> = Vec::with_capacity(k);

    for anchor in 0..n {
        knn_set.clear();
        let effective_k = k.min(width);
        for ni in 0..effective_k {
            knn_set.push(knn_idx[anchor * width + ni]);
        }
        knn_set.sort_unstable();

        let mut added = 0;
        let max_tries = count * 16;
        let mut tries = 0;
        while added < count && tries < max_tries {
            tries += 1;
            let candidate = rng.next_usize(n);
            if candidate != anchor && knn_set.binary_search(&candidate).is_err() {
                pairs.push(FarPair {
                    anchor,
                    far: candidate,
                });
                added += 1;
            }
        }
        // Deterministic fallback if random sampling exhausted attempts.
        if added < count {
            for candidate in 0..n {
                if added >= count {
                    break;
                }
                if candidate != anchor && knn_set.binary_search(&candidate).is_err() {
                    pairs.push(FarPair {
                        anchor,
                        far: candidate,
                    });
                    added += 1;
                }
            }
        }
    }
    pairs
}

// ──────────────────────────────────────────────────────────────────────────────
// Embedding initialisation
// ──────────────────────────────────────────────────────────────────────────────

fn init_embedding(
    data: &[f64],
    n: usize,
    d_in: usize,
    d_out: usize,
    init: &PaCMapInit,
    rng: &mut LcgRng,
) -> ManifoldResult<Vec<f64>> {
    match init {
        PaCMapInit::Random => Ok(random_embedding(n, d_out, rng)),
        PaCMapInit::Pca => {
            if d_out > d_in {
                Ok(random_embedding(n, d_out, rng))
            } else {
                pca_embedding(data, n, d_in, d_out, rng)
            }
        }
    }
}

fn random_embedding(n: usize, d_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut y = vec![0.0_f64; n * d_out];
    for v in &mut y {
        *v = rng.next_normal() * 0.01;
    }
    y
}

/// PCA initialisation via deflated power iteration.
///
/// The projected coordinates are scaled to std ≈ 0.01 to match the random init.
fn pca_embedding(
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
    let nf = n as f64;
    for m in &mut mean {
        *m /= nf;
    }

    // Centered data
    let mut centered = vec![0.0_f64; n * d_in];
    for i in 0..n {
        for j in 0..d_in {
            centered[i * d_in + j] = data[i * d_in + j] - mean[j];
        }
    }

    // Deflated power iteration for top-d_out eigenvectors.
    let mut components: Vec<Vec<f64>> = Vec::with_capacity(d_out);
    let mut residual = centered.clone();

    for _ in 0..d_out {
        let mut v: Vec<f64> = (0..d_in).map(|_| rng.next_normal()).collect();
        normalise_vec(&mut v);

        for _ in 0..60 {
            // Xv  (n-vector)
            let mut xv = vec![0.0_f64; n];
            for i in 0..n {
                for j in 0..d_in {
                    xv[i] += residual[i * d_in + j] * v[j];
                }
            }
            // X^T(Xv)  (d_in-vector)
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

        // Deflate residual
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

    // Project and scale
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

    for c in 0..d_out {
        let mut var = 0.0_f64;
        for i in 0..n {
            var += y[i * d_out + c].powi(2);
        }
        var /= nf;
        let std_dev = var.sqrt().max(1e-14);
        for i in 0..n {
            y[i * d_out + c] = y[i * d_out + c] / std_dev * 0.01;
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
// Loss evaluation
// ──────────────────────────────────────────────────────────────────────────────

/// Phase weights `(w1, w2, w3)` = `(near, mid_near, far)`.
#[derive(Clone, Copy)]
struct PhaseWeights {
    near: f64,
    mid_near: f64,
    far: f64,
}

impl PhaseWeights {
    fn for_iter(iter: usize, n_iter: usize) -> Self {
        let phase1_end = n_iter / 3;
        let phase2_end = 2 * n_iter / 3;
        if iter < phase1_end {
            // Phase 1: heavy mid-near to get global structure
            PhaseWeights {
                near: 2.0,
                mid_near: 500.0,
                far: 1.0,
            }
        } else if iter < phase2_end {
            // Phase 2: increase near attraction
            PhaseWeights {
                near: 3.0,
                mid_near: 500.0,
                far: 1.0,
            }
        } else {
            // Phase 3: fine-tune local structure
            PhaseWeights {
                near: 1.0,
                mid_near: 0.0,
                far: 1.0,
            }
        }
    }
}

/// Evaluate the total PaCMAP loss at the current embedding.
fn evaluate_loss(
    y: &[f64],
    near_pairs: &[AttractPair],
    mid_near_pairs: &[AttractPair],
    far_pairs: &[FarPair],
    d_out: usize,
    pw: PhaseWeights,
) -> f64 {
    let mut l_np = 0.0_f64;
    for p in near_pairs {
        let d = embedding_sq_dist(y, p.anchor, p.other, d_out);
        // f(d) = d / (d + 1)
        l_np += p.weight * d / (d + 1.0);
    }

    let mut l_mnp = 0.0_f64;
    for p in mid_near_pairs {
        let d = embedding_sq_dist(y, p.anchor, p.other, d_out);
        l_mnp += p.weight * d / (d + 1.0);
    }

    let mut l_fp = 0.0_f64;
    for p in far_pairs {
        let d = embedding_sq_dist(y, p.anchor, p.far, d_out);
        // g(d) = 1 / (d + 1)  — repulsive: small d → loss ≈ 1, large d → loss → 0
        l_fp += 1.0 / (d + 1.0);
    }

    pw.near * l_np + pw.mid_near * l_mnp + pw.far * l_fp
}

// ──────────────────────────────────────────────────────────────────────────────
// Gradient accumulation
// ──────────────────────────────────────────────────────────────────────────────

/// Accumulate gradients from all near and mid-near pairs (attractive).
///
/// Loss per pair: `f(d) = w * d / (d + 1)`.
/// Gradient: `∂f/∂y_i = w * 2(y_i − y_j) / (d + 1)²`.
#[inline]
fn accumulate_attract_grad(
    y: &[f64],
    grad: &mut [f64],
    pairs: &[AttractPair],
    d_out: usize,
    phase_weight: f64,
) {
    for p in pairs {
        let a = p.anchor;
        let b = p.other;
        let d = embedding_sq_dist(y, a, b, d_out);
        let denom = (d + 1.0).powi(2);
        // ∂f/∂y_a = phase_weight * p.weight * 2(y_a − y_b) / (d + 1)²
        let coeff = phase_weight * p.weight * 2.0 / denom;
        for k in 0..d_out {
            let diff = y[a * d_out + k] - y[b * d_out + k];
            grad[a * d_out + k] += coeff * diff;
            grad[b * d_out + k] -= coeff * diff;
        }
    }
}

/// Accumulate gradients from all far pairs (repulsive).
///
/// Loss per pair: `g(d) = 1 / (d + 1)`.
/// Gradient: `∂g/∂y_i = −2(y_i − y_k) / (d + 1)²`.
#[inline]
fn accumulate_repulse_grad(
    y: &[f64],
    grad: &mut [f64],
    pairs: &[FarPair],
    d_out: usize,
    phase_weight: f64,
) {
    for p in pairs {
        let a = p.anchor;
        let b = p.far;
        let d = embedding_sq_dist(y, a, b, d_out);
        let denom = (d + 1.0).powi(2);
        // ∂g/∂y_a = phase_weight * (−2(y_a − y_b) / (d + 1)²)
        let coeff = phase_weight * (-2.0) / denom;
        for k in 0..d_out {
            let diff = y[a * d_out + k] - y[b * d_out + k];
            grad[a * d_out + k] += coeff * diff;
            // The far point also receives a gradient: −(∂g/∂y_a) = +2(y_a - y_b)/(d+1)²
            // This symmetry prevents collapse toward far points.
            grad[b * d_out + k] -= coeff * diff;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Adam optimiser with phase schedule
// ──────────────────────────────────────────────────────────────────────────────

/// Adam hyper-parameters (fixed, per PaCMAP paper recommendations).
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-8;

/// Run the full Adam-based PaCMAP optimisation loop.
///
/// Returns `(final_loss, loss_history)`.
fn optimise_adam(
    y: &mut [f64],
    near_pairs: &[AttractPair],
    mid_near_pairs: &[AttractPair],
    far_pairs: &[FarPair],
    n: usize,
    d_out: usize,
    n_iter: usize,
    lr: f64,
) -> (f64, Vec<f64>) {
    let param_dim = n * d_out;
    let mut m = vec![0.0_f64; param_dim]; // first moment
    let mut v_adam = vec![0.0_f64; param_dim]; // second moment
    let mut loss_history: Vec<f64> = Vec::new();

    if n_iter == 0 {
        let pw = PhaseWeights::for_iter(0, 1);
        let loss = evaluate_loss(y, near_pairs, mid_near_pairs, far_pairs, d_out, pw);
        loss_history.push(loss);
        return (loss, loss_history);
    }

    let initial_pw = PhaseWeights::for_iter(0, n_iter);
    let init_loss = evaluate_loss(y, near_pairs, mid_near_pairs, far_pairs, d_out, initial_pw);
    loss_history.push(init_loss);

    let mut final_loss = init_loss;

    for iter in 0..n_iter {
        let pw = PhaseWeights::for_iter(iter, n_iter);
        let t = (iter + 1) as f64; // Adam time step (1-indexed)

        // Accumulate full gradients
        let mut grad = vec![0.0_f64; param_dim];
        accumulate_attract_grad(y, &mut grad, near_pairs, d_out, pw.near);
        if pw.mid_near != 0.0 {
            accumulate_attract_grad(y, &mut grad, mid_near_pairs, d_out, pw.mid_near);
        }
        if pw.far != 0.0 {
            accumulate_repulse_grad(y, &mut grad, far_pairs, d_out, pw.far);
        }

        // Adam update with bias correction
        let bc1 = 1.0 - ADAM_BETA1.powf(t);
        let bc2 = 1.0 - ADAM_BETA2.powf(t);
        let lr_t = lr * bc2.sqrt() / bc1;

        for i in 0..param_dim {
            m[i] = ADAM_BETA1 * m[i] + (1.0 - ADAM_BETA1) * grad[i];
            v_adam[i] = ADAM_BETA2 * v_adam[i] + (1.0 - ADAM_BETA2) * grad[i] * grad[i];
            y[i] -= lr_t * m[i] / (v_adam[i].sqrt() + ADAM_EPS);
        }

        // Re-centre each output dimension to zero mean — prevents drift.
        centre_embedding(y, n, d_out);

        // Record loss every 50 iterations and at the final step.
        if (iter + 1) % 50 == 0 || iter + 1 == n_iter {
            let loss = evaluate_loss(y, near_pairs, mid_near_pairs, far_pairs, d_out, pw);
            loss_history.push(loss);
            if iter + 1 == n_iter {
                final_loss = loss;
            }
        }
    }

    (final_loss, loss_history)
}

/// Re-centre each output dimension of `y` to zero mean.
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

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Generate two tight clusters offset by ±10 along each dimension.
    fn make_two_cluster_data(n_per: usize, dim: usize, seed: u64) -> Vec<f64> {
        let n = n_per * 2;
        let mut data = vec![0.0_f64; n * dim];
        let mut rng = LcgRng::new(seed);
        for i in 0..n {
            let centre = if i < n_per { 10.0 } else { -10.0 };
            for d in 0..dim {
                data[i * dim + d] = centre + 0.3 * rng.next_normal();
            }
        }
        data
    }

    fn cluster_centre(emb: &[f64], start: usize, end: usize, d_out: usize) -> Vec<f64> {
        let count = (end - start) as f64;
        let mut c = vec![0.0_f64; d_out];
        for i in start..end {
            for k in 0..d_out {
                c[k] += emb[i * d_out + k];
            }
        }
        for x in &mut c {
            *x /= count;
        }
        c
    }

    fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    // ── Test 1: embedding shape is correct ────────────────────────────────────
    #[test]
    fn test_embedding_shape() {
        let n = 20;
        let dim = 4;
        let data = make_two_cluster_data(10, dim, 1);
        let cfg = PaCMapConfig {
            n_iter: 10,
            n_neighbors: 4,
            n_mid_near: 2,
            n_far: 5,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data, n, dim, &cfg).expect("pacmap should succeed");
        assert_eq!(result.embedding.len(), n * cfg.n_components);
        assert_eq!(result.n_samples, n);
        assert_eq!(result.n_components, cfg.n_components);
    }

    // ── Test 2: all embedding values are finite (no NaN / inf) ───────────────
    #[test]
    fn test_embedding_all_finite() {
        let n = 24;
        let dim = 5;
        let data = make_two_cluster_data(12, dim, 2);
        let cfg = PaCMapConfig {
            n_iter: 50,
            n_neighbors: 5,
            n_mid_near: 3,
            n_far: 8,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data, n, dim, &cfg).expect("ok");
        for (i, v) in result.embedding.iter().enumerate() {
            assert!(v.is_finite(), "embedding[{i}] = {v} is not finite");
        }
    }

    // ── Test 3: loss_history is non-empty ────────────────────────────────────
    #[test]
    fn test_loss_history_nonempty() {
        let n = 12;
        let dim = 3;
        let data = make_two_cluster_data(6, dim, 3);
        let cfg = PaCMapConfig {
            n_iter: 100,
            n_neighbors: 3,
            n_mid_near: 2,
            n_far: 5,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data, n, dim, &cfg).expect("ok");
        assert!(
            !result.loss_history.is_empty(),
            "loss_history must not be empty"
        );
    }

    // ── Test 4: final_loss is finite and non-negative ─────────────────────────
    #[test]
    fn test_final_loss_finite_nonneg() {
        let n = 15;
        let dim = 4;
        let data = make_two_cluster_data(8, dim, 4);
        let cfg = PaCMapConfig {
            n_iter: 30,
            n_neighbors: 4,
            n_mid_near: 2,
            n_far: 6,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data[..n * dim], n, dim, &cfg).expect("ok");
        assert!(result.final_loss.is_finite(), "final_loss must be finite");
        assert!(result.final_loss >= 0.0, "final_loss must be non-negative");
    }

    // ── Test 5: two-cluster separation in the embedding ──────────────────────
    #[test]
    fn test_cluster_separation() {
        let n_per = 20;
        let n = n_per * 2;
        let dim = 6;
        let data = make_two_cluster_data(n_per, dim, 5);
        let cfg = PaCMapConfig {
            n_iter: 450,
            n_neighbors: 8,
            n_mid_near: 5,
            n_far: 15,
            lr: 1.0,
            seed: 5,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data, n, dim, &cfg).expect("ok");
        let d_out = cfg.n_components;

        let ca = cluster_centre(&result.embedding, 0, n_per, d_out);
        let cb = cluster_centre(&result.embedding, n_per, n, d_out);
        let inter = l2_dist(&ca, &cb);

        // Mean intra-cluster distance (within cluster A)
        let mut intra = 0.0_f64;
        let mut cnt = 0usize;
        for i in 0..n_per {
            for j in (i + 1)..n_per {
                intra += l2_dist(
                    &result.embedding[i * d_out..(i + 1) * d_out],
                    &result.embedding[j * d_out..(j + 1) * d_out],
                );
                cnt += 1;
            }
        }
        intra /= cnt.max(1) as f64;

        assert!(
            inter > intra,
            "inter-cluster dist {inter:.4} should exceed intra-cluster {intra:.4}"
        );
    }

    // ── Test 6: n_components=3 produces correct shape ─────────────────────────
    #[test]
    fn test_three_components() {
        let n = 20;
        let dim = 6;
        let data = make_two_cluster_data(10, dim, 6);
        let cfg = PaCMapConfig {
            n_components: 3,
            n_iter: 30,
            n_neighbors: 4,
            n_mid_near: 2,
            n_far: 5,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data, n, dim, &cfg).expect("ok");
        assert_eq!(result.embedding.len(), n * 3);
        assert_eq!(result.n_components, 3);
        assert!(result.embedding.iter().all(|v| v.is_finite()));
    }

    // ── Test 7: n_samples < 3 → error ────────────────────────────────────────
    #[test]
    fn test_too_few_samples_error() {
        let data = vec![1.0_f64; 2 * 4]; // 2 samples × 4 dims
        let cfg = PaCMapConfig::default();
        let err = pacmap(&data, 2, 4, &cfg);
        assert!(err.is_err(), "n_samples < 3 must return an error");
    }

    // ── Test 8: n_neighbors ≥ n_samples → clamped gracefully ─────────────────
    #[test]
    fn test_large_n_neighbors_clamped() {
        let n = 8;
        let dim = 3;
        let data = make_two_cluster_data(4, dim, 8);
        // n_neighbors = 200 >> n = 8 → should be clamped, not panic
        let cfg = PaCMapConfig {
            n_neighbors: 200,
            n_iter: 10,
            n_mid_near: 2,
            n_far: 3,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data, n, dim, &cfg);
        assert!(
            result.is_ok(),
            "large n_neighbors should be clamped, not fail"
        );
    }

    // ── Test 9: PaCMapInit::Pca produces correct embedding shape ─────────────
    #[test]
    fn test_pca_init_shape() {
        let n = 20;
        let dim = 6;
        let data = make_two_cluster_data(10, dim, 9);
        let cfg = PaCMapConfig {
            n_iter: 20,
            n_neighbors: 4,
            n_mid_near: 2,
            n_far: 5,
            init: PaCMapInit::Pca,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data, n, dim, &cfg).expect("PCA init should succeed");
        assert_eq!(result.embedding.len(), n * cfg.n_components);
        assert!(result.embedding.iter().all(|v| v.is_finite()));
    }

    // ── Test 10: loss_history length ≈ n_iter/50 ─────────────────────────────
    #[test]
    fn test_loss_history_length() {
        let n = 16;
        let dim = 4;
        let data = make_two_cluster_data(8, dim, 10);
        let cfg = PaCMapConfig {
            n_iter: 200,
            n_neighbors: 4,
            n_mid_near: 2,
            n_far: 5,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data, n, dim, &cfg).expect("ok");
        // Expect: 1 initial + ceil(200/50) = 4 snapshots = 5 total
        let expected_snapshots = 1 + cfg.n_iter / 50;
        let actual = result.loss_history.len();
        assert!(
            actual >= expected_snapshots.saturating_sub(1) && actual <= expected_snapshots + 2,
            "expected ≈{expected_snapshots} history entries, got {actual}"
        );
    }

    // ── Test 11: deterministic with same seed ─────────────────────────────────
    #[test]
    fn test_determinism_same_seed() {
        let n = 16;
        let dim = 4;
        let data = make_two_cluster_data(8, dim, 11);
        let cfg = PaCMapConfig {
            n_iter: 40,
            n_neighbors: 4,
            n_mid_near: 2,
            n_far: 5,
            seed: 42,
            ..PaCMapConfig::default()
        };
        let r1 = pacmap(&data, n, dim, &cfg).expect("run 1");
        let r2 = pacmap(&data, n, dim, &cfg).expect("run 2");
        assert_eq!(
            r1.embedding, r2.embedding,
            "same seed must give identical output"
        );
    }

    // ── Test 12: PCA vs Random init — both converge to valid embeddings ───────
    #[test]
    fn test_pca_vs_random_init_both_valid() {
        let n = 20;
        let dim = 5;
        let data = make_two_cluster_data(10, dim, 12);
        let base = PaCMapConfig {
            n_iter: 50,
            n_neighbors: 4,
            n_mid_near: 2,
            n_far: 5,
            ..PaCMapConfig::default()
        };
        let r_rnd = pacmap(
            &data,
            n,
            dim,
            &PaCMapConfig {
                init: PaCMapInit::Random,
                seed: 0,
                ..base.clone()
            },
        )
        .expect("random init");
        let r_pca = pacmap(
            &data,
            n,
            dim,
            &PaCMapConfig {
                init: PaCMapInit::Pca,
                seed: 0,
                ..base.clone()
            },
        )
        .expect("pca init");

        // Both produce same-shaped, finite embeddings
        assert_eq!(r_rnd.embedding.len(), r_pca.embedding.len());
        assert!(r_rnd.embedding.iter().all(|v| v.is_finite()));
        assert!(r_pca.embedding.iter().all(|v| v.is_finite()));

        // PCA init produces a different initialisation (different first history entry)
        // but both should converge to sensible values
        assert!(r_pca.final_loss.is_finite() && r_pca.final_loss >= 0.0);
        assert!(r_rnd.final_loss.is_finite() && r_rnd.final_loss >= 0.0);
    }

    // ── Test 13: phase weight schedule changes loss objective ─────────────────
    #[test]
    fn test_phase_weights_schedule() {
        // Verify PhaseWeights::for_iter returns the correct constants.
        let n_iter = 450;
        // Phase 1: iter < 150
        let pw1 = PhaseWeights::for_iter(0, n_iter);
        assert_eq!(pw1.near, 2.0);
        assert_eq!(pw1.mid_near, 500.0);
        assert_eq!(pw1.far, 1.0);

        // Phase 2: 150 ≤ iter < 300
        let pw2 = PhaseWeights::for_iter(200, n_iter);
        assert_eq!(pw2.near, 3.0);
        assert_eq!(pw2.mid_near, 500.0);
        assert_eq!(pw2.far, 1.0);

        // Phase 3: iter ≥ 300
        let pw3 = PhaseWeights::for_iter(350, n_iter);
        assert_eq!(pw3.near, 1.0);
        assert_eq!(pw3.mid_near, 0.0);
        assert_eq!(pw3.far, 1.0);
    }

    // ── Test 14: n_iter=0 returns a valid result with initial loss ────────────
    #[test]
    fn test_zero_iterations() {
        let n = 12;
        let dim = 3;
        let data = make_two_cluster_data(6, dim, 14);
        let cfg = PaCMapConfig {
            n_iter: 0,
            n_neighbors: 3,
            n_mid_near: 2,
            n_far: 4,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data, n, dim, &cfg).expect("zero iter should succeed");
        assert_eq!(result.embedding.len(), n * cfg.n_components);
        assert!(!result.loss_history.is_empty());
        assert!(result.final_loss.is_finite());
    }

    // ── Test 15: input-space weights are positive ─────────────────────────────
    #[test]
    fn test_near_pair_weights_positive() {
        let n = 10;
        let dim = 3;
        let data = make_two_cluster_data(5, dim, 15);
        let sq_dists = compute_sq_distance_matrix(&data, n, dim);
        let width = 4usize.min(n - 1);
        let (knn_idx, knn_sq) = brute_knn_sorted(&sq_dists, n, width);
        let pairs = build_near_pairs(&knn_idx, &knn_sq, n, width);
        for p in &pairs {
            assert!(
                p.weight > 0.0,
                "near-pair weight must be positive, got {}",
                p.weight
            );
        }
    }

    // ── Test 16: large n_mid_near clipped to available range ─────────────────
    #[test]
    fn test_mid_near_large_count_clipped() {
        // With n=6 and k=3, only positions [3,4] are available for mid-near.
        let n = 6;
        let dim = 3;
        let data = make_two_cluster_data(3, dim, 16);
        let cfg = PaCMapConfig {
            n_neighbors: 3,
            n_mid_near: 999, // much larger than available range
            n_far: 2,
            n_iter: 5,
            ..PaCMapConfig::default()
        };
        let result = pacmap(&data, n, dim, &cfg);
        // Should succeed — clipped gracefully
        assert!(
            result.is_ok(),
            "large n_mid_near should clip, not fail: {:?}",
            result.err()
        );
    }
}

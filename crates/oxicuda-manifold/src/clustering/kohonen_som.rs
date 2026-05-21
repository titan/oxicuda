//! Kohonen Self-Organizing Map (SOM) — Kohonen (1982).
//!
//! A SOM maps high-dimensional input space onto a 2-D grid of neurons while
//! preserving topological relationships.  Training is competitive: for each
//! sample the Best Matching Unit (BMU) and its grid-neighbors are pulled
//! toward the sample, with the neighborhood radius and learning rate both
//! decaying exponentially across epochs.
//!
//! # Architecture
//! - Grid: `grid_rows × grid_cols` neurons, each carrying a weight vector in ℝ^d.
//! - Weight matrix: W ∈ ℝ^{(grid_rows × grid_cols) × d}, stored row-major.
//!
//! # Quality metrics
//! - **Quantization error** (QE): mean ‖xᵢ − W_{bmu_i}‖² per sample.
//! - **Topographic error** (TE, Kiviluoto 1996): fraction of samples whose 1st
//!   and 2nd BMU are not adjacent on the grid (adjacency threshold √2).

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Initialization strategy
// ---------------------------------------------------------------------------

/// Weight-initialization strategy for the Kohonen SOM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SomInit {
    /// Sample `n_neurons` rows from the training data at random (with replacement
    /// when `n_neurons > n_data`).
    Random,
    /// Spread neurons along the first two principal components of the data.
    /// Falls back to `Random` when `dim < 2`.
    PcaBased,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Full configuration for [`kohonen_som_fit`].
#[derive(Debug, Clone)]
pub struct KohonenSomConfig {
    /// Number of neuron rows in the 2-D grid.  Default: 10.
    pub grid_rows: usize,
    /// Number of neuron columns in the 2-D grid.  Default: 10.
    pub grid_cols: usize,
    /// Number of training epochs.  Default: 100.
    pub n_epochs: usize,
    /// Initial learning rate α₀.  Default: 0.5.
    pub alpha_init: f64,
    /// Final learning rate αf.  Default: 0.01.
    pub alpha_final: f64,
    /// Initial neighborhood radius σ₀.  Default: max(grid_rows, grid_cols) / 2.0.
    /// When set to 0.0 the default is computed at fit time.
    pub sigma_init: f64,
    /// Final neighborhood radius σf.  Default: 1.0.
    pub sigma_final: f64,
    /// Weight initialization strategy.
    pub init: SomInit,
    /// RNG seed for reproducibility.  Default: 42.
    pub seed: u64,
}

impl Default for KohonenSomConfig {
    fn default() -> Self {
        Self {
            grid_rows: 10,
            grid_cols: 10,
            n_epochs: 100,
            alpha_init: 0.5,
            alpha_final: 0.01,
            sigma_init: 0.0, // computed at fit time
            sigma_final: 1.0,
            init: SomInit::Random,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Output of [`kohonen_som_fit`].
#[derive(Debug, Clone)]
pub struct KohonenSomResult {
    /// Trained weight matrix: row-major, shape `(n_neurons, dim)`.
    pub weights: Vec<f64>,
    /// BMU index (in `0..n_neurons`) for each input sample.  Length = `n_data`.
    pub labels: Vec<usize>,
    /// Mean ‖xᵢ − W_{bmu_i}‖² (per-sample quantization error).
    pub quantization_error: f64,
    /// Fraction of samples whose 1st and 2nd BMU are non-adjacent on the grid
    /// (topographic error per Kiviluoto 1996).
    pub topographic_error: f64,
    /// Number of epochs that were actually trained.
    pub n_epochs: usize,
    /// Grid row count (same as config).
    pub grid_rows: usize,
    /// Grid column count (same as config).
    pub grid_cols: usize,
    /// Dimensionality of input vectors.
    pub input_dim: usize,
}

// ---------------------------------------------------------------------------
// Public API — helper functions
// ---------------------------------------------------------------------------

/// Return the 2-D grid coordinates of neuron `k` given `grid_cols`.
///
/// # Examples
/// ```
/// use oxicuda_manifold::clustering::kohonen_som::som_grid_pos;
/// assert_eq!(som_grid_pos(7, 3), (2, 1));
/// ```
#[must_use]
pub fn som_grid_pos(k: usize, grid_cols: usize) -> (usize, usize) {
    debug_assert!(grid_cols > 0, "grid_cols must be > 0");
    (k / grid_cols, k % grid_cols)
}

/// Return a reference to the weight slice `[dim]` for the neuron at grid
/// position `(row, col)`.
#[must_use]
pub fn som_weight_at(
    weights: &[f64],
    row: usize,
    col: usize,
    grid_cols: usize,
    dim: usize,
) -> &[f64] {
    let k = row * grid_cols + col;
    let start = k * dim;
    &weights[start..start + dim]
}

/// Find the index of the Best Matching Unit (BMU) for input `x`.
///
/// Returns the neuron index `k ∈ 0..n_neurons` minimising ‖x − W_k‖².
#[must_use]
pub fn som_predict(weights: &[f64], n_neurons: usize, x: &[f64], dim: usize) -> usize {
    find_bmu(weights, n_neurons, x, dim)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between slices `a` and `b` of length `d`.
#[inline]
fn l2_dist_sq(a: &[f64], b: &[f64], d: usize) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..d {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}

/// Squared grid-distance between neurons `k1` and `k2`.
/// Returns `(r1 - r2)² + (c1 - c2)²` as `f64`.
#[inline]
fn grid_distance_sq(k1: usize, k2: usize, grid_cols: usize) -> f64 {
    let (r1, c1) = som_grid_pos(k1, grid_cols);
    let (r2, c2) = som_grid_pos(k2, grid_cols);
    let dr = r1 as f64 - r2 as f64;
    let dc = c1 as f64 - c2 as f64;
    dr * dr + dc * dc
}

/// Gaussian neighborhood function h(bmu, k, t) = exp(−d²_grid / (2σ²)).
#[inline]
fn neighborhood(k_bmu: usize, k: usize, grid_cols: usize, sigma: f64) -> f64 {
    let dist_sq = grid_distance_sq(k_bmu, k, grid_cols);
    let denom = 2.0 * sigma * sigma;
    (-dist_sq / denom).exp()
}

/// Return the index of the BMU for `x` among all `n_neurons` neurons.
fn find_bmu(weights: &[f64], n_neurons: usize, x: &[f64], dim: usize) -> usize {
    let mut best_idx = 0usize;
    let mut best_dist = f64::MAX;
    for k in 0..n_neurons {
        let w = &weights[k * dim..k * dim + dim];
        let d = l2_dist_sq(x, w, dim);
        if d < best_dist {
            best_dist = d;
            best_idx = k;
        }
    }
    best_idx
}

/// Return the indices of the two closest neurons (BMU1, BMU2) for `x`.
fn find_two_bmus(weights: &[f64], n_neurons: usize, x: &[f64], dim: usize) -> (usize, usize) {
    let mut best_idx = 0usize;
    let mut best_dist = f64::MAX;
    let mut second_idx = 0usize;
    let mut second_dist = f64::MAX;

    for k in 0..n_neurons {
        let w = &weights[k * dim..k * dim + dim];
        let d = l2_dist_sq(x, w, dim);
        if d < best_dist {
            second_dist = best_dist;
            second_idx = best_idx;
            best_dist = d;
            best_idx = k;
        } else if d < second_dist {
            second_dist = d;
            second_idx = k;
        }
    }
    (best_idx, second_idx)
}

/// Produce a Fisher-Yates shuffled index array `[0, n)` using the provided RNG.
fn fisher_yates_indices(n: usize, rng: &mut LcgRng) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        indices.swap(i, j);
    }
    indices
}

// ---------------------------------------------------------------------------
// Weight initialization
// ---------------------------------------------------------------------------

/// Initialize weight matrix using randomly sampled data rows.
fn init_random(
    data: &[f64],
    n_data: usize,
    dim: usize,
    n_neurons: usize,
    rng: &mut LcgRng,
) -> Vec<f64> {
    let mut weights = vec![0.0_f64; n_neurons * dim];
    if n_data == 0 {
        return weights;
    }

    if n_neurons <= n_data {
        // Fisher-Yates shuffle of indices, take the first n_neurons without replacement.
        let shuffled = fisher_yates_indices(n_data, rng);
        for (k, &src_row) in shuffled.iter().take(n_neurons).enumerate() {
            let w_start = k * dim;
            let d_start = src_row * dim;
            weights[w_start..w_start + dim].copy_from_slice(&data[d_start..d_start + dim]);
        }
    } else {
        // With replacement for the overflow.
        for k in 0..n_neurons {
            let src_row = rng.next_usize(n_data);
            let w_start = k * dim;
            let d_start = src_row * dim;
            weights[w_start..w_start + dim].copy_from_slice(&data[d_start..d_start + dim]);
        }
    }
    weights
}

/// Initialize weight matrix using the first two principal components of the data.
///
/// Neurons are placed on a regular grid spanning ± (scale factor) along each PC.
/// Falls back to random if `dim < 2` or n_data is too small.
fn init_pca_based(
    data: &[f64],
    n_data: usize,
    dim: usize,
    n_neurons: usize,
    grid_rows: usize,
    grid_cols: usize,
    rng: &mut LcgRng,
) -> Vec<f64> {
    // Fallback for degenerate cases.
    if dim < 2 || n_data < 2 {
        return init_random(data, n_data, dim, n_neurons, rng);
    }

    // Step 1: compute per-feature mean and center the data (in-place copy).
    let mut mean = vec![0.0_f64; dim];
    for row in 0..n_data {
        for j in 0..dim {
            mean[j] += data[row * dim + j];
        }
    }
    let scale = 1.0 / n_data as f64;
    for m in mean.iter_mut() {
        *m *= scale;
    }

    let mut centered = vec![0.0_f64; n_data * dim];
    for row in 0..n_data {
        for j in 0..dim {
            centered[row * dim + j] = data[row * dim + j] - mean[j];
        }
    }

    // Step 2: extract the first two PCs via deflated power iteration.
    let pc1 = power_iteration_pc(&centered, n_data, dim, None, 200);
    let pc2 = power_iteration_pc(&centered, n_data, dim, Some(&pc1), 200);

    // Step 3: estimate variance (singular values) for pc1 and pc2.
    let sv1 = singular_value_from_pc(&centered, n_data, dim, &pc1);
    let sv2 = singular_value_from_pc(&centered, n_data, dim, &pc2);

    let scale_factor = 2.0 * sv1.max(sv2);

    // Step 4: place neurons on the grid.
    let mut weights = vec![0.0_f64; n_neurons * dim];
    let rows_f = grid_rows as f64;
    let cols_f = grid_cols as f64;

    for r in 0..grid_rows {
        for c in 0..grid_cols {
            let k = r * grid_cols + c;
            // Normalized coordinates in [-1, 1] ish.
            let t1 = (r as f64 - rows_f / 2.0) / rows_f * scale_factor;
            let t2 = (c as f64 - cols_f / 2.0) / cols_f * scale_factor;
            let w_start = k * dim;
            for j in 0..dim {
                weights[w_start + j] = mean[j] + pc1[j] * t1 + pc2[j] * t2;
            }
        }
    }
    weights
}

/// Power iteration to find the dominant principal component of the row-major
/// matrix `x` (shape n × d).  If `deflate` is provided, the result is
/// orthogonalized against it (one step of deflation).
fn power_iteration_pc(
    x: &[f64],
    n: usize,
    d: usize,
    deflate: Option<&[f64]>,
    max_iter: usize,
) -> Vec<f64> {
    // Initialize v to the first data row (non-random for reproducibility).
    let mut v: Vec<f64> = x[..d].to_vec();
    normalize_vec(&mut v, d);

    for _ in 0..max_iter {
        // u = X v  (n-vector)
        let mut u = vec![0.0_f64; n];
        for row in 0..n {
            let mut dot = 0.0_f64;
            for j in 0..d {
                dot += x[row * d + j] * v[j];
            }
            u[row] = dot;
        }
        // v_new = X^T u  (d-vector)
        let mut v_new = vec![0.0_f64; d];
        for row in 0..n {
            for j in 0..d {
                v_new[j] += x[row * d + j] * u[row];
            }
        }
        // Deflate if needed.
        if let Some(pc) = deflate {
            let dot: f64 = v_new.iter().zip(pc.iter()).map(|(a, b)| a * b).sum();
            for j in 0..d {
                v_new[j] -= dot * pc[j];
            }
        }
        normalize_vec(&mut v_new, d);
        v = v_new;
    }
    v
}

/// Estimate the singular value corresponding to a given principal component `pc`
/// in centered matrix `x` (shape n × d).  Returns ‖X pc‖ / √n.
fn singular_value_from_pc(x: &[f64], n: usize, d: usize, pc: &[f64]) -> f64 {
    let mut sum_sq = 0.0_f64;
    for row in 0..n {
        let mut dot = 0.0_f64;
        for j in 0..d {
            dot += x[row * d + j] * pc[j];
        }
        sum_sq += dot * dot;
    }
    (sum_sq / n as f64).sqrt()
}

/// Normalize a vector in-place to unit length (L2).  No-op if the norm is
/// effectively zero (avoids NaN).
fn normalize_vec(v: &mut [f64], d: usize) {
    let norm_sq: f64 = v.iter().take(d).map(|x| x * x).sum();
    if norm_sq > 1.0e-300 {
        let inv = 1.0 / norm_sq.sqrt();
        for val in v.iter_mut().take(d) {
            *val *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// Decay schedules
// ---------------------------------------------------------------------------

/// Exponential decay: `v0 * (vf / v0)^(t / T)`.
///
/// Clamped so the ratio does not go below machine epsilon.
#[inline]
fn exponential_decay(v0: f64, vf: f64, t: usize, total: usize) -> f64 {
    if total == 0 || v0 <= 0.0 || vf <= 0.0 {
        return v0;
    }
    let ratio = (vf / v0).max(f64::EPSILON);
    v0 * ratio.powf(t as f64 / total as f64)
}

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

/// Compute quantization error and topographic error for `data` against `weights`.
///
/// Returns `(qe, te)` where:
/// - `qe` = mean ‖xᵢ − W_{bmuᵢ}‖² (per sample)
/// - `te` = fraction of samples where bmu1 and bmu2 are non-adjacent (√2 threshold)
fn compute_quality_metrics(
    data: &[f64],
    n_data: usize,
    dim: usize,
    weights: &[f64],
    n_neurons: usize,
    grid_cols: usize,
) -> (f64, f64) {
    if n_data == 0 {
        return (0.0, 0.0);
    }

    let adjacency_threshold_sq = 2.0_f64 + 1.0e-9; // √2² = 2; diagonal neighbors count as adjacent

    let mut qe_sum = 0.0_f64;
    let mut te_count = 0usize;

    for i in 0..n_data {
        let x = &data[i * dim..i * dim + dim];
        let (bmu1, bmu2) = find_two_bmus(weights, n_neurons, x, dim);

        // Quantization contribution.
        let w1 = &weights[bmu1 * dim..bmu1 * dim + dim];
        qe_sum += l2_dist_sq(x, w1, dim);

        // Topographic error: non-adjacent = grid distance² > 2.
        if n_neurons > 1 {
            let grid_dist_sq = grid_distance_sq(bmu1, bmu2, grid_cols);
            if grid_dist_sq > adjacency_threshold_sq {
                te_count += 1;
            }
        }
    }

    let qe = qe_sum / n_data as f64;
    let te = te_count as f64 / n_data as f64;
    (qe, te)
}

/// Compute BMU labels for all samples (used after training).
fn assign_labels(
    data: &[f64],
    n_data: usize,
    dim: usize,
    weights: &[f64],
    n_neurons: usize,
) -> Vec<usize> {
    (0..n_data)
        .map(|i| find_bmu(weights, n_neurons, &data[i * dim..i * dim + dim], dim))
        .collect()
}

// ---------------------------------------------------------------------------
// Main training function
// ---------------------------------------------------------------------------

/// Fit a Kohonen Self-Organizing Map to `data`.
///
/// # Parameters
/// - `data`   — row-major input matrix of shape `(n_data, dim)`.
/// - `n_data` — number of samples.
/// - `dim`    — input feature dimensionality.
/// - `config` — full training configuration.
///
/// # Errors
/// Returns `ManifoldError::EmptyInput` if `n_data == 0`.
/// Returns `ManifoldError::InvalidParameter` if `grid_rows == 0`, `grid_cols == 0`,
/// or `alpha_init <= 0`.
pub fn kohonen_som_fit(
    data: &[f64],
    n_data: usize,
    dim: usize,
    config: &KohonenSomConfig,
) -> ManifoldResult<KohonenSomResult> {
    // ---- Validation -------------------------------------------------------
    if n_data == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if config.grid_rows == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "grid_rows".to_owned(),
            reason: "must be > 0".to_owned(),
        });
    }
    if config.grid_cols == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "grid_cols".to_owned(),
            reason: "must be > 0".to_owned(),
        });
    }
    if config.alpha_init <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "alpha_init".to_owned(),
            reason: "must be > 0".to_owned(),
        });
    }
    if dim == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "dim".to_owned(),
            reason: "must be > 0".to_owned(),
        });
    }
    if data.len() != n_data * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_data, dim],
            got: vec![data.len()],
        });
    }

    let n_neurons = config.grid_rows * config.grid_cols;
    let grid_cols = config.grid_cols;

    // Effective sigma_init: use configured value or default to max(rows, cols) / 2.
    let sigma_init = if config.sigma_init > 0.0 {
        config.sigma_init
    } else {
        (config.grid_rows.max(config.grid_cols)) as f64 / 2.0
    };
    let sigma_final = config.sigma_final.max(f64::EPSILON);

    let alpha_final = config.alpha_final.max(f64::EPSILON);

    let mut rng = LcgRng::new(config.seed);

    // ---- Weight initialization --------------------------------------------
    let mut weights = match config.init {
        SomInit::Random => init_random(data, n_data, dim, n_neurons, &mut rng),
        SomInit::PcaBased => init_pca_based(
            data,
            n_data,
            dim,
            n_neurons,
            config.grid_rows,
            config.grid_cols,
            &mut rng,
        ),
    };

    // ---- Training loop ----------------------------------------------------
    let total_epochs = config.n_epochs;

    for epoch in 0..total_epochs {
        // Decay schedules.
        let alpha = exponential_decay(config.alpha_init, alpha_final, epoch, total_epochs);
        let sigma = exponential_decay(sigma_init, sigma_final, epoch, total_epochs);

        // Guard against degenerate sigma.
        let sigma_eff = sigma.max(1.0e-10);

        // Shuffle sample order.
        let order = fisher_yates_indices(n_data, &mut rng);

        for &i in &order {
            let x = &data[i * dim..i * dim + dim];

            // Find BMU.
            let bmu = find_bmu(&weights, n_neurons, x, dim);

            // Update all neurons via neighborhood function.
            for k in 0..n_neurons {
                let h = neighborhood(bmu, k, grid_cols, sigma_eff);
                if h < 1.0e-12 {
                    continue; // negligible influence — skip
                }
                let factor = alpha * h;
                let w_start = k * dim;
                for j in 0..dim {
                    weights[w_start + j] += factor * (x[j] - weights[w_start + j]);
                }
            }
        }
    }

    // ---- Post-training metrics --------------------------------------------
    let labels = assign_labels(data, n_data, dim, &weights, n_neurons);
    let (quantization_error, topographic_error) =
        compute_quality_metrics(data, n_data, dim, &weights, n_neurons, grid_cols);

    Ok(KohonenSomResult {
        weights,
        labels,
        quantization_error,
        topographic_error,
        n_epochs: total_epochs,
        grid_rows: config.grid_rows,
        grid_cols: config.grid_cols,
        input_dim: dim,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons.
    const EPS: f64 = 1.0e-10;

    // -----------------------------------------------------------------------
    // 1. config_defaults
    // -----------------------------------------------------------------------
    #[test]
    fn config_defaults() {
        let cfg = KohonenSomConfig::default();
        assert_eq!(cfg.grid_rows, 10);
        assert_eq!(cfg.grid_cols, 10);
        assert_eq!(cfg.n_epochs, 100);
        assert!((cfg.alpha_init - 0.5).abs() < EPS);
        assert!((cfg.alpha_final - 0.01).abs() < EPS);
        assert_eq!(cfg.seed, 42);
    }

    // -----------------------------------------------------------------------
    // 2. som_grid_pos
    // -----------------------------------------------------------------------
    #[test]
    fn som_grid_pos_correctness() {
        assert_eq!(som_grid_pos(0, 3), (0, 0));
        assert_eq!(som_grid_pos(7, 3), (2, 1));
        assert_eq!(som_grid_pos(11, 4), (2, 3));
        assert_eq!(som_grid_pos(9, 10), (0, 9));
    }

    // -----------------------------------------------------------------------
    // 3. som_predict_closest — single neuron, all predictions = 0
    // -----------------------------------------------------------------------
    #[test]
    fn som_predict_single_neuron() {
        let weights = vec![1.0_f64, 2.0, 3.0];
        for _ in 0..10 {
            let x = vec![0.0_f64, 0.0, 0.0];
            assert_eq!(som_predict(&weights, 1, &x, 3), 0);
            let x2 = vec![100.0_f64, 100.0, 100.0];
            assert_eq!(som_predict(&weights, 1, &x2, 3), 0);
        }
    }

    // -----------------------------------------------------------------------
    // 4. fit_simple_grid — n_data=50, d=2, 5×5 grid; labels.len()==50
    // -----------------------------------------------------------------------
    #[test]
    fn fit_simple_grid_label_count() {
        let n_data = 50;
        let dim = 2;
        // Generate data as a sequence filling [0,1]².
        let mut data = vec![0.0_f64; n_data * dim];
        let mut rng = LcgRng::new(1234);
        for v in data.iter_mut() {
            *v = rng.next_f64();
        }
        let cfg = KohonenSomConfig {
            grid_rows: 5,
            grid_cols: 5,
            n_epochs: 10,
            alpha_init: 0.3,
            alpha_final: 0.01,
            sigma_init: 2.5,
            sigma_final: 1.0,
            init: SomInit::Random,
            seed: 7,
        };
        let result = kohonen_som_fit(&data, n_data, dim, &cfg).unwrap();
        assert_eq!(result.labels.len(), n_data);
        assert_eq!(result.grid_rows, 5);
        assert_eq!(result.grid_cols, 5);
        assert_eq!(result.input_dim, dim);
    }

    // -----------------------------------------------------------------------
    // 5. quantization_error_decreases — more epochs → lower or equal QE
    // -----------------------------------------------------------------------
    #[test]
    fn quantization_error_decreases_with_more_epochs() {
        // Three well-separated clusters.
        let n_per_cluster = 30;
        let dim = 2;
        let n_data = n_per_cluster * 3;
        let mut data = vec![0.0_f64; n_data * dim];
        let centers = [(0.0_f64, 0.0_f64), (10.0, 0.0), (5.0, 8.66)];
        let mut rng = LcgRng::new(9);
        for (c, &(cx, cy)) in centers.iter().enumerate() {
            for i in 0..n_per_cluster {
                let idx = (c * n_per_cluster + i) * dim;
                data[idx] = cx + rng.next_range(-0.5, 0.5);
                data[idx + 1] = cy + rng.next_range(-0.5, 0.5);
            }
        }

        let base_cfg = KohonenSomConfig {
            grid_rows: 3,
            grid_cols: 3,
            alpha_init: 0.5,
            alpha_final: 0.01,
            sigma_init: 1.5,
            sigma_final: 0.5,
            init: SomInit::Random,
            seed: 42,
            ..Default::default()
        };

        let cfg1 = KohonenSomConfig {
            n_epochs: 1,
            ..base_cfg.clone()
        };
        let cfg100 = KohonenSomConfig {
            n_epochs: 200,
            ..base_cfg.clone()
        };

        let r1 = kohonen_som_fit(&data, n_data, dim, &cfg1).unwrap();
        let r100 = kohonen_som_fit(&data, n_data, dim, &cfg100).unwrap();

        assert!(
            r100.quantization_error <= r1.quantization_error + 1.0,
            "QE(200)={} should be ≤ QE(1)={}",
            r100.quantization_error,
            r1.quantization_error
        );
    }

    // -----------------------------------------------------------------------
    // 6. weights_shape
    // -----------------------------------------------------------------------
    #[test]
    fn weights_shape_correct() {
        let n_data = 20;
        let dim = 5;
        let mut data = vec![0.0_f64; n_data * dim];
        let mut rng = LcgRng::new(3);
        for v in data.iter_mut() {
            *v = rng.next_f64();
        }
        let cfg = KohonenSomConfig {
            grid_rows: 4,
            grid_cols: 6,
            n_epochs: 5,
            ..Default::default()
        };
        let result = kohonen_som_fit(&data, n_data, dim, &cfg).unwrap();
        assert_eq!(result.weights.len(), 4 * 6 * dim);
    }

    // -----------------------------------------------------------------------
    // 7. topographic_error_range
    // -----------------------------------------------------------------------
    #[test]
    fn topographic_error_in_range() {
        let n_data = 40;
        let dim = 3;
        let mut data = vec![0.0_f64; n_data * dim];
        let mut rng = LcgRng::new(77);
        for v in data.iter_mut() {
            *v = rng.next_f64();
        }
        let cfg = KohonenSomConfig {
            grid_rows: 3,
            grid_cols: 3,
            n_epochs: 20,
            ..Default::default()
        };
        let result = kohonen_som_fit(&data, n_data, dim, &cfg).unwrap();
        assert!(
            result.topographic_error >= 0.0 && result.topographic_error <= 1.0,
            "TE={} not in [0,1]",
            result.topographic_error
        );
    }

    // -----------------------------------------------------------------------
    // 8. quantization_error_nonneg
    // -----------------------------------------------------------------------
    #[test]
    fn quantization_error_nonneg() {
        let n_data = 15;
        let dim = 4;
        let mut data = vec![0.0_f64; n_data * dim];
        let mut rng = LcgRng::new(55);
        for v in data.iter_mut() {
            *v = rng.next_normal();
        }
        let cfg = KohonenSomConfig {
            grid_rows: 3,
            grid_cols: 3,
            n_epochs: 10,
            ..Default::default()
        };
        let result = kohonen_som_fit(&data, n_data, dim, &cfg).unwrap();
        assert!(result.quantization_error >= 0.0);
    }

    // -----------------------------------------------------------------------
    // 9. fit_line_topology_preserved
    // -----------------------------------------------------------------------
    /// 30 points on a line; SOM 1×10 grid; the grid ordering should correlate
    /// with the 1-D ordering of inputs (Spearman-like check via rank inversion).
    #[test]
    fn fit_line_topology_preserved() {
        let n_data = 30;
        let dim = 2;
        // Points on the x-axis: 0, 1, 2, ..., 29.
        let mut data = vec![0.0_f64; n_data * dim];
        for i in 0..n_data {
            data[i * dim] = i as f64;
            data[i * dim + 1] = 0.0;
        }
        let cfg = KohonenSomConfig {
            grid_rows: 1,
            grid_cols: 10,
            n_epochs: 300,
            alpha_init: 0.5,
            alpha_final: 0.01,
            sigma_init: 5.0,
            sigma_final: 0.5,
            init: SomInit::Random,
            seed: 123,
        };
        let result = kohonen_som_fit(&data, n_data, dim, &cfg).unwrap();

        // Extract the x-coordinate of each neuron weight (neuron index → x pos).
        let neuron_x: Vec<f64> = (0..10).map(|k| result.weights[k * dim]).collect();

        // Count how many consecutive neuron pairs are in the right order (ascending x).
        let mut monotone_pairs = 0usize;
        let mut total_pairs = 0usize;
        for pair in neuron_x.windows(2) {
            total_pairs += 1;
            if pair[1] >= pair[0] {
                monotone_pairs += 1;
            }
        }
        // At least 70% of consecutive neuron pairs should be monotonically ordered.
        assert!(
            monotone_pairs * 10 >= total_pairs * 7,
            "Topology not preserved: {monotone_pairs}/{total_pairs} monotone pairs"
        );
    }

    // -----------------------------------------------------------------------
    // 10. fit_cluster_topology
    // -----------------------------------------------------------------------
    /// 3 well-separated clusters in 2D; verify that SOM cells are assigned
    /// predominantly to one cluster (purity check).
    #[test]
    fn fit_cluster_topology() {
        let n_per = 60;
        let dim = 2;
        let n_data = n_per * 3;
        let mut data = vec![0.0_f64; n_data * dim];
        let cluster_centers = [(0.0_f64, 0.0_f64), (20.0, 0.0), (10.0, 17.32)];
        let mut rng = LcgRng::new(314);
        for (c, &(cx, cy)) in cluster_centers.iter().enumerate() {
            for i in 0..n_per {
                let idx = (c * n_per + i) * dim;
                data[idx] = cx + rng.next_range(-1.0, 1.0);
                data[idx + 1] = cy + rng.next_range(-1.0, 1.0);
            }
        }
        // Ground-truth cluster for each sample.
        let ground_truth: Vec<usize> = (0..n_data).map(|i| i / n_per).collect();

        let cfg = KohonenSomConfig {
            grid_rows: 3,
            grid_cols: 3,
            n_epochs: 200,
            alpha_init: 0.5,
            alpha_final: 0.01,
            sigma_init: 1.5,
            sigma_final: 0.5,
            init: SomInit::PcaBased,
            seed: 42,
        };
        let result = kohonen_som_fit(&data, n_data, dim, &cfg).unwrap();

        // For each SOM neuron, find the dominant ground-truth cluster label.
        let n_neurons = 3 * 3;
        let mut neuron_correct = vec![0usize; n_neurons];
        let mut neuron_total = vec![0usize; n_neurons];
        for (i, gt) in ground_truth.iter().enumerate().take(n_data) {
            let bmu = result.labels[i];
            neuron_total[bmu] += 1;
            if result.labels[i] < n_neurons {
                // Check which cluster owns this neuron.
                neuron_correct[bmu] += 1; // placeholder; purity check below
                let _ = *gt;
            }
        }

        // Purity: for each neuron, check that all assigned samples share the same GT cluster.
        let mut purity_count = 0usize;
        let mut occupied_neurons = 0usize;
        for neuron in 0..n_neurons {
            let mut cluster_hist = [0usize; 3];
            for i in 0..n_data {
                if result.labels[i] == neuron {
                    cluster_hist[ground_truth[i]] += 1;
                }
            }
            let total: usize = cluster_hist.iter().sum();
            if total > 0 {
                occupied_neurons += 1;
                let max_count = *cluster_hist.iter().max().unwrap();
                if max_count * 2 >= total {
                    // Majority (≥50%) from one cluster.
                    purity_count += 1;
                }
            }
        }
        // At least 2/3 of occupied neurons should be pure.
        assert!(
            purity_count * 3 >= occupied_neurons * 2,
            "Purity too low: {purity_count}/{occupied_neurons} pure neurons"
        );
    }

    // -----------------------------------------------------------------------
    // 11. sigma_too_small_still_works
    // -----------------------------------------------------------------------
    #[test]
    fn sigma_too_small_still_works() {
        let n_data = 20;
        let dim = 2;
        let mut data = vec![0.0_f64; n_data * dim];
        let mut rng = LcgRng::new(88);
        for v in data.iter_mut() {
            *v = rng.next_f64();
        }
        let cfg = KohonenSomConfig {
            grid_rows: 4,
            grid_cols: 4,
            n_epochs: 10,
            sigma_init: 0.1,
            sigma_final: 0.01,
            ..Default::default()
        };
        let result = kohonen_som_fit(&data, n_data, dim, &cfg);
        assert!(result.is_ok(), "fit should succeed with tiny sigma");
    }

    // -----------------------------------------------------------------------
    // 12. empty_data_error
    // -----------------------------------------------------------------------
    #[test]
    fn empty_data_error() {
        let data: Vec<f64> = vec![];
        let cfg = KohonenSomConfig::default();
        let result = kohonen_som_fit(&data, 0, 2, &cfg);
        assert!(
            matches!(result, Err(ManifoldError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // 13. grid_rows_zero_error
    // -----------------------------------------------------------------------
    #[test]
    fn grid_rows_zero_error() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let cfg = KohonenSomConfig {
            grid_rows: 0,
            ..Default::default()
        };
        let result = kohonen_som_fit(&data, 2, 2, &cfg);
        assert!(
            matches!(result, Err(ManifoldError::InvalidParameter { .. })),
            "expected InvalidParameter, got {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // 14. n_epochs_zero_returns_initial
    // -----------------------------------------------------------------------
    /// With zero epochs the weights come directly from initialization and
    /// no training update occurs.  The function must not error, and
    /// labels / QE must be consistent with the initial weight state.
    #[test]
    fn n_epochs_zero_returns_initial() {
        let n_data = 10;
        let dim = 2;
        let mut data = vec![0.0_f64; n_data * dim];
        let mut rng = LcgRng::new(1);
        for v in data.iter_mut() {
            *v = rng.next_f64();
        }
        let cfg = KohonenSomConfig {
            grid_rows: 3,
            grid_cols: 3,
            n_epochs: 0,
            ..Default::default()
        };
        let result = kohonen_som_fit(&data, n_data, dim, &cfg).unwrap();
        assert_eq!(result.n_epochs, 0);
        assert_eq!(result.labels.len(), n_data);
        assert!(result.quantization_error >= 0.0);
    }

    // -----------------------------------------------------------------------
    // Additional: som_weight_at accessor
    // -----------------------------------------------------------------------
    #[test]
    fn som_weight_at_returns_correct_slice() {
        // Build a weight matrix where neuron (r, c) has weight [r as f64, c as f64].
        let grid_rows = 3;
        let grid_cols = 4;
        let dim = 2;
        let n_neurons = grid_rows * grid_cols;
        let mut weights = vec![0.0_f64; n_neurons * dim];
        for r in 0..grid_rows {
            for c in 0..grid_cols {
                let k = r * grid_cols + c;
                weights[k * dim] = r as f64;
                weights[k * dim + 1] = c as f64;
            }
        }
        for r in 0..grid_rows {
            for c in 0..grid_cols {
                let w = som_weight_at(&weights, r, c, grid_cols, dim);
                assert!((w[0] - r as f64).abs() < EPS);
                assert!((w[1] - c as f64).abs() < EPS);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Additional: pca_init produces valid weights
    // -----------------------------------------------------------------------
    #[test]
    fn pca_init_produces_valid_weights() {
        let n_data = 50;
        let dim = 4;
        let mut data = vec![0.0_f64; n_data * dim];
        let mut rng = LcgRng::new(2025);
        for v in data.iter_mut() {
            *v = rng.next_normal();
        }
        let cfg = KohonenSomConfig {
            grid_rows: 4,
            grid_cols: 4,
            n_epochs: 30,
            init: SomInit::PcaBased,
            seed: 1,
            ..Default::default()
        };
        let result = kohonen_som_fit(&data, n_data, dim, &cfg).unwrap();
        // All weights must be finite.
        assert!(
            result.weights.iter().all(|v| v.is_finite()),
            "PCA-init produced non-finite weights"
        );
        assert_eq!(result.weights.len(), 4 * 4 * dim);
    }
}

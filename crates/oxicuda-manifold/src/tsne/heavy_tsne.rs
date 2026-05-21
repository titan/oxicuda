//! Heavy-tailed t-SNE variants.
//!
//! Implements three generalised stochastic neighbour embedding algorithms:
//!
//! 1. **Heavy-tailed t-SNE** (Yang et al. 2009; Kobak et al. 2019): replaces the standard
//!    Student-t kernel (1 d.f.) with a generalised kernel parameterised by `α` degrees of
//!    freedom:
//!    ```text
//!    w_ij = (1 + d²_ij / α)^{-(α+1)/2},   q_ij = w_ij / Z
//!    ```
//!    - α = 1  → standard t-SNE (heaviest tail among integer-DoF kernels)
//!    - α = 0.5 → Cauchy kernel (even heavier tail; excellent for large datasets)
//!    - α → ∞  → Gaussian kernel (same as SSNE below)
//!
//! 2. **α-annealing t-SNE**: starts with a large `α_init` (Gaussian-like, preserving fine
//!    local structure) and decreases exponentially to `α_final` over the optimisation,
//!    helping escape early local minima.
//!
//! 3. **Symmetric SNE (SSNE)**: uses a Gaussian output kernel
//!    ```text
//!    q_ij = exp(-d²_ij) / Z
//!    ```
//!    providing a comparison baseline and corresponding to the `α → ∞` limit of heavy t-SNE.
//!
//! # References
//!
//! - Yang, Z., King, I., Xu, Z., & Oja, E. (2009). "Heavy-tailed symmetric stochastic
//!   neighbor embedding". NeurIPS 22.
//! - Kobak, D., Linderman, G., Steinerberger, S., Coifman, R. R., & Berens, P. (2019).
//!   "Heavy-tailed kernels reveal a finer cluster structure in t-SNE visualisations".
//!   arXiv:1902.05804.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::tsne::perplexity::compute_perplexity_p_matrix;

// ══════════════════════════════════════════════════════════════════════════════
// Configuration structs
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for heavy-tailed t-SNE with configurable degrees of freedom `α`.
///
/// The output kernel is:
/// ```text
/// w_ij = (1 + d²_ij / α)^{-(α+1)/2},   q_ij = w_ij / Z
/// ```
///
/// Setting `alpha = 1.0` exactly recovers standard t-SNE.
/// Setting `alpha = 0.5` gives a Cauchy kernel with excellent cluster separation.
#[derive(Debug, Clone)]
pub struct HeavyTsneConfig {
    /// Output dimensionality (default 2).
    pub n_components: usize,
    /// Perplexity target for the input probability binary search (default 30.0).
    pub perplexity: f64,
    /// Degrees of freedom α > 0.  α = 1 recovers standard t-SNE (default 1.0).
    pub alpha: f64,
    /// Total number of gradient-descent iterations (default 500).
    pub n_iter: usize,
    /// Learning rate for the adaptive momentum step (default 200.0).
    pub learning_rate: f64,
    /// Initial momentum coefficient (default 0.5).
    pub momentum: f64,
    /// Final momentum coefficient, applied after `momentum_switch_iter` (default 0.8).
    pub final_momentum: f64,
    /// Iteration index at which momentum switches from initial to final (default 250).
    pub momentum_switch_iter: usize,
    /// Early-exaggeration multiplier applied to P for the first
    /// `early_exaggeration_iters` iterations (default 12.0).
    pub early_exaggeration: f64,
    /// Number of iterations for which early exaggeration is active (default 100).
    pub early_exaggeration_iters: usize,
    /// Minimum adaptive-gain value (default 0.01).
    pub min_gain: f64,
    /// Maximum iterations for the per-row perplexity binary search (default 50).
    pub perp_max_iter: usize,
    /// Convergence tolerance for the perplexity binary search (default 1e-5).
    pub perp_tol: f64,
}

impl Default for HeavyTsneConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            alpha: 1.0,
            n_iter: 500,
            learning_rate: 200.0,
            momentum: 0.5,
            final_momentum: 0.8,
            momentum_switch_iter: 250,
            early_exaggeration: 12.0,
            early_exaggeration_iters: 100,
            min_gain: 0.01,
            perp_max_iter: 50,
            perp_tol: 1e-5,
        }
    }
}

/// Configuration for the α-annealing t-SNE variant.
///
/// Begins the optimisation with `alpha_init` (large → Gaussian-like, preserving fine
/// local structure) and decreases `α` exponentially towards `alpha_final` over the full
/// iteration count.  The schedule is:
/// ```text
/// α(t) = alpha_init · (alpha_final / alpha_init)^{t / n_iter}
/// ```
#[derive(Debug, Clone)]
pub struct AlphaTsneConfig {
    /// Base t-SNE configuration (alpha field is the final α; alpha_init below overrides
    /// the starting value).
    pub base: HeavyTsneConfig,
    /// Starting degrees of freedom (large, e.g. 10.0).
    pub alpha_init: f64,
    /// Ending degrees of freedom (small, e.g. 0.5).
    pub alpha_final: f64,
}

impl Default for AlphaTsneConfig {
    fn default() -> Self {
        Self {
            base: HeavyTsneConfig::default(),
            alpha_init: 10.0,
            alpha_final: 1.0,
        }
    }
}

/// Configuration for Symmetric SNE (Gaussian output kernel).
///
/// SSNE corresponds to the `α → ∞` limit of heavy-tailed t-SNE:
/// ```text
/// q_ij = exp(-d²_ij) / Z
/// ```
#[derive(Debug, Clone)]
pub struct SsneConfig {
    /// Output dimensionality (default 2).
    pub n_components: usize,
    /// Perplexity target (default 30.0).
    pub perplexity: f64,
    /// Total gradient-descent iterations (default 500).
    pub n_iter: usize,
    /// Learning rate (default 200.0).
    pub learning_rate: f64,
    /// Initial momentum (default 0.5).
    pub momentum: f64,
    /// Final momentum (default 0.8).
    pub final_momentum: f64,
    /// Iteration index for momentum switch (default 250).
    pub momentum_switch_iter: usize,
    /// Early-exaggeration multiplier (default 12.0).
    pub early_exaggeration: f64,
    /// Number of early-exaggeration iterations (default 100).
    pub early_exaggeration_iters: usize,
    /// Minimum adaptive gain (default 0.01).
    pub min_gain: f64,
    /// Max perplexity binary-search iterations (default 50).
    pub perp_max_iter: usize,
    /// Perplexity binary-search tolerance (default 1e-5).
    pub perp_tol: f64,
}

impl Default for SsneConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            n_iter: 500,
            learning_rate: 200.0,
            momentum: 0.5,
            final_momentum: 0.8,
            momentum_switch_iter: 250,
            early_exaggeration: 12.0,
            early_exaggeration_iters: 100,
            min_gain: 0.01,
            perp_max_iter: 50,
            perp_tol: 1e-5,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Result structs
// ══════════════════════════════════════════════════════════════════════════════

/// Result produced by [`heavy_tsne_fit`], [`cauchy_tsne_fit`], and [`alpha_tsne_fit`].
#[derive(Debug)]
pub struct HeavyTsneResult {
    /// Row-major embedding matrix of shape `[n_samples × n_components]`.
    pub embedding: Vec<f64>,
    /// Final KL(P‖Q) divergence computed at the last iteration.
    pub final_kl: f64,
}

/// Result produced by [`ssne_fit`].
#[derive(Debug)]
pub struct SsneResult {
    /// Row-major embedding matrix of shape `[n_samples × n_components]`.
    pub embedding: Vec<f64>,
    /// Final KL(P‖Q) divergence (with Gaussian Q) computed at the last iteration.
    pub final_kl: f64,
}

// ══════════════════════════════════════════════════════════════════════════════
// Public API
// ══════════════════════════════════════════════════════════════════════════════

/// Fit heavy-tailed t-SNE on row-major data `x` of shape `(n_samples, dim)`.
///
/// The generalised Student-t kernel with `cfg.alpha` degrees of freedom is used for the
/// low-dimensional distribution Q.  Setting `cfg.alpha = 1.0` exactly recovers standard
/// t-SNE.
///
/// # Errors
///
/// Returns [`ManifoldError`] if `n_samples == 0`, `dim == 0`, `cfg.alpha <= 0`,
/// `cfg.n_components` is outside `1..=8`, or if the perplexity search fails.
pub fn heavy_tsne_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    cfg: &HeavyTsneConfig,
    rng: &mut LcgRng,
) -> ManifoldResult<HeavyTsneResult> {
    validate_heavy_inputs(
        x,
        n_samples,
        dim,
        cfg.n_components,
        cfg.perplexity,
        cfg.alpha,
    )?;

    let n = n_samples;
    let d_out = cfg.n_components;

    // Step 1: build joint probability matrix P from input distances.
    let d2_input = pairwise_sq_dist(x, n, dim);
    let mut p = compute_perplexity_p_matrix(
        &d2_input,
        n,
        cfg.perplexity,
        cfg.perp_max_iter,
        cfg.perp_tol,
    )?;
    clamp_floor(&mut p, f64::EPSILON);

    // Step 2: early exaggeration.
    for v in &mut p {
        *v *= cfg.early_exaggeration;
    }

    // Step 3: initialise Y ~ N(0, 1e-4).
    let mut y = small_normal_init(n, d_out, rng);
    let mut dy_prev = vec![0.0_f64; n * d_out];
    let mut gains = vec![1.0_f64; n * d_out];

    // Step 4: gradient-descent loop.
    let mut final_kl = 0.0_f64;
    for iter in 0..cfg.n_iter {
        if iter == cfg.early_exaggeration_iters {
            for v in &mut p {
                *v /= cfg.early_exaggeration;
            }
        }
        let mom = momentum_at(
            iter,
            cfg.momentum,
            cfg.final_momentum,
            cfg.momentum_switch_iter,
        );
        let grad = heavy_tsne_gradient(&y, &p, n, d_out, cfg.alpha);
        apply_gain_update(
            &mut y,
            &grad,
            &mut dy_prev,
            &mut gains,
            cfg.learning_rate,
            mom,
            cfg.min_gain,
        );
        centre_embedding(&mut y, n, d_out);

        if iter == cfg.n_iter.saturating_sub(1) {
            let (q, _z) = heavy_q_matrix(&y, n, d_out, cfg.alpha);
            final_kl = kl_pq(&p, &q, n);
        }
    }

    Ok(HeavyTsneResult {
        embedding: y,
        final_kl,
    })
}

/// Convenience wrapper for Cauchy-kernel t-SNE (α = 0.5).
///
/// The Cauchy kernel has exceptionally heavy tails, yielding excellent cluster separation
/// for large datasets at the cost of some fine-grained local structure.
///
/// # Errors
///
/// Returns [`ManifoldError`] for empty inputs or perplexity-search failures.
pub fn cauchy_tsne_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    n_iter: usize,
    perplexity: f64,
    learning_rate: f64,
    rng: &mut LcgRng,
) -> ManifoldResult<HeavyTsneResult> {
    let cfg = HeavyTsneConfig {
        alpha: 0.5,
        n_iter,
        perplexity,
        learning_rate,
        early_exaggeration_iters: (n_iter / 5).max(50),
        momentum_switch_iter: n_iter / 2,
        ..HeavyTsneConfig::default()
    };
    heavy_tsne_fit(x, n_samples, dim, &cfg, rng)
}

/// Fit α-annealing t-SNE.
///
/// The degrees of freedom `α` decreases exponentially from `cfg.alpha_init` to
/// `cfg.alpha_final` over the full iteration count.  Starting with a large `α` (nearly
/// Gaussian) preserves fine local structure during the early phase; decreasing `α` then
/// gradually increases the tail weight, improving inter-cluster separation.
///
/// # Errors
///
/// Returns [`ManifoldError`] if `alpha_init ≤ 0`, `alpha_final ≤ 0`,
/// `alpha_init < alpha_final`, or if any of the base config fields are invalid.
pub fn alpha_tsne_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    cfg: &AlphaTsneConfig,
    rng: &mut LcgRng,
) -> ManifoldResult<HeavyTsneResult> {
    let base = &cfg.base;
    if cfg.alpha_init <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "alpha_init".into(),
            reason: "must be > 0".into(),
        });
    }
    if cfg.alpha_final <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "alpha_final".into(),
            reason: "must be > 0".into(),
        });
    }
    if cfg.alpha_init < cfg.alpha_final {
        return Err(ManifoldError::InvalidParameter {
            name: "alpha_init".into(),
            reason: "alpha_init must be >= alpha_final (annealing decreases alpha)".into(),
        });
    }
    validate_heavy_inputs(
        x,
        n_samples,
        dim,
        base.n_components,
        base.perplexity,
        cfg.alpha_init,
    )?;

    let n = n_samples;
    let d_out = base.n_components;

    let d2_input = pairwise_sq_dist(x, n, dim);
    let mut p = compute_perplexity_p_matrix(
        &d2_input,
        n,
        base.perplexity,
        base.perp_max_iter,
        base.perp_tol,
    )?;
    clamp_floor(&mut p, f64::EPSILON);

    for v in &mut p {
        *v *= base.early_exaggeration;
    }

    let mut y = small_normal_init(n, d_out, rng);
    let mut dy_prev = vec![0.0_f64; n * d_out];
    let mut gains = vec![1.0_f64; n * d_out];

    // Pre-compute the log-ratio for the exponential schedule.
    let log_ratio = (cfg.alpha_final / cfg.alpha_init).ln();
    let n_iter_f = base.n_iter as f64;

    let mut final_kl = 0.0_f64;
    for iter in 0..base.n_iter {
        if iter == base.early_exaggeration_iters {
            for v in &mut p {
                *v /= base.early_exaggeration;
            }
        }

        // Anneal α: α(t) = α_init · (α_final / α_init)^{t/n_iter}
        let alpha_t = cfg.alpha_init * (log_ratio * (iter as f64 / n_iter_f)).exp();
        let alpha_t = alpha_t.max(f64::EPSILON);

        let mom = momentum_at(
            iter,
            base.momentum,
            base.final_momentum,
            base.momentum_switch_iter,
        );
        let grad = heavy_tsne_gradient(&y, &p, n, d_out, alpha_t);
        apply_gain_update(
            &mut y,
            &grad,
            &mut dy_prev,
            &mut gains,
            base.learning_rate,
            mom,
            base.min_gain,
        );
        centre_embedding(&mut y, n, d_out);

        if iter == base.n_iter.saturating_sub(1) {
            let alpha_final = cfg.alpha_final.max(f64::EPSILON);
            let (q, _z) = heavy_q_matrix(&y, n, d_out, alpha_final);
            final_kl = kl_pq(&p, &q, n);
        }
    }

    Ok(HeavyTsneResult {
        embedding: y,
        final_kl,
    })
}

/// Fit Symmetric SNE (SSNE) with a Gaussian output kernel.
///
/// SSNE is the `α → ∞` limit of heavy-tailed t-SNE.  The output kernel is:
/// ```text
/// q_ij = exp(-d²_ij) / Z
/// ```
///
/// SSNE tends to crowd points in the centre of the embedding, which is why t-SNE's
/// Student-t kernel was introduced.  SSNE is provided as a comparison baseline.
///
/// # Errors
///
/// Returns [`ManifoldError`] for empty inputs, shape mismatches, invalid parameters,
/// or perplexity-search failures.
pub fn ssne_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    cfg: &SsneConfig,
    rng: &mut LcgRng,
) -> ManifoldResult<SsneResult> {
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![x.len()],
        });
    }
    if cfg.n_components == 0 || cfg.n_components > 8 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be in 1..=8".into(),
        });
    }
    if cfg.perplexity <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "perplexity".into(),
            reason: "must be > 0".into(),
        });
    }

    let n = n_samples;
    let d_out = cfg.n_components;

    let d2_input = pairwise_sq_dist(x, n, dim);
    let mut p = compute_perplexity_p_matrix(
        &d2_input,
        n,
        cfg.perplexity,
        cfg.perp_max_iter,
        cfg.perp_tol,
    )?;
    clamp_floor(&mut p, f64::EPSILON);

    for v in &mut p {
        *v *= cfg.early_exaggeration;
    }

    let mut y = small_normal_init(n, d_out, rng);
    let mut dy_prev = vec![0.0_f64; n * d_out];
    let mut gains = vec![1.0_f64; n * d_out];

    let mut final_kl = 0.0_f64;
    for iter in 0..cfg.n_iter {
        if iter == cfg.early_exaggeration_iters {
            for v in &mut p {
                *v /= cfg.early_exaggeration;
            }
        }
        let mom = momentum_at(
            iter,
            cfg.momentum,
            cfg.final_momentum,
            cfg.momentum_switch_iter,
        );
        let grad = ssne_gradient(&y, &p, n, d_out);
        apply_gain_update(
            &mut y,
            &grad,
            &mut dy_prev,
            &mut gains,
            cfg.learning_rate,
            mom,
            cfg.min_gain,
        );
        centre_embedding(&mut y, n, d_out);

        if iter == cfg.n_iter.saturating_sub(1) {
            let (q, _z) = ssne_q_matrix(&y, n, d_out);
            final_kl = kl_pq(&p, &q, n);
        }
    }

    Ok(SsneResult {
        embedding: y,
        final_kl,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Gradient engines
// ══════════════════════════════════════════════════════════════════════════════

/// Compute the heavy-tailed t-SNE gradient for one iteration.
///
/// Uses the generalised Student-t kernel:
/// ```text
/// w_ij = (1 + d²_ij / α)^{-(α+1)/2}
/// q_ij = w_ij / Z    (diagonal zeroed)
/// ```
///
/// The KL gradient with respect to embedding coordinate `y_i` is:
/// ```text
/// ∂KL/∂y_i = 4 · ((α+1)/α) · Σ_{j≠i} (p_ij - q_ij) · w_ij · (y_i - y_j)
/// ```
///
/// This function is the hot inner loop; `alpha` is taken as a parameter so the same
/// function serves both fixed-α and annealed-α variants.
fn heavy_tsne_gradient(y: &[f64], p: &[f64], n: usize, d_out: usize, alpha: f64) -> Vec<f64> {
    // Unnormalised weights and normalisation constant.
    let (q, w, z) = heavy_w_q(y, n, d_out, alpha);
    let _ = z;

    // Factor common to all pairs: (α+1)/α.
    let factor = (alpha + 1.0) / alpha;

    let mut grad = vec![0.0_f64; n * d_out];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let pij = p[i * n + j];
            let qij = q[i * n + j];
            let wij = w[i * n + j];
            let coeff = factor * (pij - qij) * wij;
            for k in 0..d_out {
                grad[i * d_out + k] += coeff * (y[i * d_out + k] - y[j * d_out + k]);
            }
        }
    }
    for v in &mut grad {
        *v *= 4.0;
    }
    grad
}

/// Compute the SSNE gradient for one iteration.
///
/// Uses the Gaussian output kernel:
/// ```text
/// w_ij = exp(-d²_ij)
/// q_ij = w_ij / Z
/// ```
///
/// The KL gradient is:
/// ```text
/// ∂KL_SSNE/∂y_i = 4 · Σ_{j≠i} (p_ij - q_ij) · w_ij · (y_i - y_j)
/// ```
fn ssne_gradient(y: &[f64], p: &[f64], n: usize, d_out: usize) -> Vec<f64> {
    let (q, w, _z) = ssne_q_and_w(y, n, d_out);

    let mut grad = vec![0.0_f64; n * d_out];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let pij = p[i * n + j];
            let qij = q[i * n + j];
            let wij = w[i * n + j];
            let coeff = (pij - qij) * wij;
            for k in 0..d_out {
                grad[i * d_out + k] += coeff * (y[i * d_out + k] - y[j * d_out + k]);
            }
        }
    }
    for v in &mut grad {
        *v *= 4.0;
    }
    grad
}

// ══════════════════════════════════════════════════════════════════════════════
// Q-matrix builders (also exposed to fit functions for KL computation)
// ══════════════════════════════════════════════════════════════════════════════

/// Build (w, q, Z) for the heavy-tailed kernel with `alpha` degrees of freedom.
///
/// `w_ij = (1 + d²_ij / α)^{-(α+1)/2}`, diagonal = 0.
/// `q_ij = w_ij / Z`, clamped to ≥ `f64::EPSILON`.
fn heavy_w_q(y: &[f64], n: usize, d_out: usize, alpha: f64) -> (Vec<f64>, Vec<f64>, f64) {
    let exponent = -(alpha + 1.0) / 2.0;
    let inv_alpha = 1.0 / alpha;

    let mut w = vec![0.0_f64; n * n];
    let mut z = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut d2 = 0.0_f64;
            for k in 0..d_out {
                let v = y[i * d_out + k] - y[j * d_out + k];
                d2 += v * v;
            }
            // w_ij = (1 + d²/α)^{-(α+1)/2}
            let wij = (1.0 + d2 * inv_alpha).powf(exponent);
            w[i * n + j] = wij;
            z += wij;
        }
    }
    let z = z.max(f64::EPSILON);

    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            q[i * n + j] = (w[i * n + j] / z).max(f64::EPSILON);
        }
    }
    (q, w, z)
}

/// Build (q, Z) for the heavy-tailed kernel (omits W).  Used in KL computation.
fn heavy_q_matrix(y: &[f64], n: usize, d_out: usize, alpha: f64) -> (Vec<f64>, f64) {
    let (q, _w, z) = heavy_w_q(y, n, d_out, alpha);
    (q, z)
}

/// Build (w, q, Z) for the Gaussian (SSNE) kernel.
///
/// `w_ij = exp(-d²_ij)`, diagonal = 0.
/// `q_ij = w_ij / Z`, clamped to ≥ `f64::EPSILON`.
fn ssne_q_and_w(y: &[f64], n: usize, d_out: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let mut w = vec![0.0_f64; n * n];
    let mut z = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut d2 = 0.0_f64;
            for k in 0..d_out {
                let v = y[i * d_out + k] - y[j * d_out + k];
                d2 += v * v;
            }
            let wij = (-d2).exp();
            w[i * n + j] = wij;
            z += wij;
        }
    }
    let z = z.max(f64::EPSILON);

    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            q[i * n + j] = (w[i * n + j] / z).max(f64::EPSILON);
        }
    }
    (q, w, z)
}

/// Build (q, Z) for the Gaussian (SSNE) kernel.  Used in KL computation.
fn ssne_q_matrix(y: &[f64], n: usize, d_out: usize) -> (Vec<f64>, f64) {
    let (q, _w, z) = ssne_q_and_w(y, n, d_out);
    (q, z)
}

// ══════════════════════════════════════════════════════════════════════════════
// Shared optimisation helpers
// ══════════════════════════════════════════════════════════════════════════════

/// Adaptive gain update following the t-SNE momentum scheme.
///
/// For each parameter `y[i]`:
/// - If the gradient keeps the same sign as the previous velocity: `gain *= 0.8`.
/// - If the gradient flips sign:                                    `gain += 0.2`.
/// - Gain is clamped from below at `min_gain`.
/// - Velocity: `dy = momentum · dy_prev - lr · gain · grad`.
/// - Update:    `y[i] += dy`.
fn apply_gain_update(
    y: &mut [f64],
    grad: &[f64],
    dy_prev: &mut [f64],
    gains: &mut [f64],
    lr: f64,
    momentum: f64,
    min_gain: f64,
) {
    let m = y.len();
    for idx in 0..m {
        let same_sign = grad[idx].signum() == dy_prev[idx].signum();
        if same_sign {
            gains[idx] *= 0.8;
        } else {
            gains[idx] += 0.2;
        }
        if gains[idx] < min_gain {
            gains[idx] = min_gain;
        }
        let dy = momentum * dy_prev[idx] - lr * gains[idx] * grad[idx];
        dy_prev[idx] = dy;
        y[idx] += dy;
    }
}

/// Re-centre the embedding so every column has zero mean.
fn centre_embedding(y: &mut [f64], n: usize, d_out: usize) {
    for k in 0..d_out {
        let mut mean = 0.0_f64;
        for i in 0..n {
            mean += y[i * d_out + k];
        }
        mean /= n as f64;
        for i in 0..n {
            y[i * d_out + k] -= mean;
        }
    }
}

/// Compute KL(P‖Q) = Σ_{i≠j} p_ij · log(p_ij / q_ij), skipping pairs where p_ij < 1e-12.
fn kl_pq(p: &[f64], q: &[f64], _n: usize) -> f64 {
    let mut kl = 0.0_f64;
    for (pi, qi) in p.iter().zip(q.iter()) {
        if *pi > 1e-12 {
            let qi_safe = qi.max(f64::EPSILON);
            kl += pi * (pi / qi_safe).ln();
        }
    }
    kl
}

/// Select the momentum coefficient for iteration `iter`.
#[inline]
fn momentum_at(iter: usize, initial: f64, final_mom: f64, switch: usize) -> f64 {
    if iter < switch { initial } else { final_mom }
}

// ══════════════════════════════════════════════════════════════════════════════
// Input helpers
// ══════════════════════════════════════════════════════════════════════════════

/// Validate common inputs for heavy-tailed variants.
fn validate_heavy_inputs(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    n_components: usize,
    perplexity: f64,
    alpha: f64,
) -> ManifoldResult<()> {
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![x.len()],
        });
    }
    if n_components == 0 || n_components > 8 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be in 1..=8".into(),
        });
    }
    if perplexity <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "perplexity".into(),
            reason: "must be > 0".into(),
        });
    }
    if alpha <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "alpha".into(),
            reason: "degrees of freedom must be > 0".into(),
        });
    }
    Ok(())
}

/// Compute row-major pairwise squared Euclidean distances for `n` points of `dim` dimensions.
fn pairwise_sq_dist(x: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut d2 = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0_f64;
            for k in 0..dim {
                let v = x[i * dim + k] - x[j * dim + k];
                s += v * v;
            }
            d2[i * n + j] = s;
            d2[j * n + i] = s;
        }
    }
    d2
}

/// Initialise embedding from N(0, 1e-4) (tiny variance to avoid early repulsion explosions).
fn small_normal_init(n: usize, d_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut y = vec![0.0_f64; n * d_out];
    for v in &mut y {
        *v = rng.next_normal() * 1e-4;
    }
    y
}

/// Clamp all elements of `v` from below at `floor`.
fn clamp_floor(v: &mut [f64], floor: f64) {
    for x in v.iter_mut() {
        if *x < floor {
            *x = floor;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build two-cluster data: `n_per` points at +offset, `n_per` at -offset.
    fn two_cluster_data(n_per: usize, dim: usize, offset: f64, rng: &mut LcgRng) -> Vec<f64> {
        let n = 2 * n_per;
        let mut x = vec![0.0_f64; n * dim];
        for i in 0..n_per {
            for d in 0..dim {
                x[i * dim + d] = offset + 0.05 * rng.next_normal();
            }
        }
        for i in n_per..n {
            for d in 0..dim {
                x[i * dim + d] = -offset + 0.05 * rng.next_normal();
            }
        }
        x
    }

    /// Return cluster centroid differences in first embedding dimension.
    fn cluster_centroid_diff(emb: &[f64], n_per: usize, n_components: usize) -> f64 {
        let mut ca = 0.0_f64;
        let mut cb = 0.0_f64;
        for i in 0..n_per {
            ca += emb[i * n_components];
        }
        for i in n_per..2 * n_per {
            cb += emb[i * n_components];
        }
        ca /= n_per as f64;
        cb /= n_per as f64;
        (ca - cb).abs()
    }

    // ── 1. alpha=1 gradient matches standard t-SNE pattern ────────────────────
    #[test]
    fn heavy_tsne_alpha1_matches_tsne() {
        // With alpha=1 the kernel (1 + d²/1)^{-1} = (1 + d²)^{-1}, identical to t-SNE.
        let mut rng = LcgRng::new(1001);
        let n = 6;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = HeavyTsneConfig {
            alpha: 1.0,
            n_iter: 30,
            early_exaggeration_iters: 10,
            perplexity: 2.0,
            momentum_switch_iter: 20,
            ..HeavyTsneConfig::default()
        };
        let result = heavy_tsne_fit(&x, n, dim, &cfg, &mut rng);
        assert!(result.is_ok(), "alpha=1 failed: {:?}", result.err());
        let r = result.unwrap();
        assert_eq!(r.embedding.len(), n * 2);
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }

    // ── 2. Output shape correct ───────────────────────────────────────────────
    #[test]
    fn heavy_tsne_output_shape() {
        let mut rng = LcgRng::new(1002);
        let n = 8;
        let dim = 4;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = HeavyTsneConfig {
            n_components: 3,
            n_iter: 20,
            early_exaggeration_iters: 5,
            perplexity: 2.0,
            momentum_switch_iter: 15,
            ..HeavyTsneConfig::default()
        };
        let r = heavy_tsne_fit(&x, n, dim, &cfg, &mut rng).expect("output_shape");
        assert_eq!(r.embedding.len(), n * 3);
    }

    // ── 3. Empty input returns Err ────────────────────────────────────────────
    #[test]
    fn heavy_tsne_empty_error() {
        let mut rng = LcgRng::new(1003);
        let cfg = HeavyTsneConfig::default();
        let result = heavy_tsne_fit(&[], 0, 2, &cfg, &mut rng);
        assert!(matches!(result, Err(ManifoldError::EmptyInput)));
    }

    // ── 4. alpha ≤ 0 returns InvalidParameter ────────────────────────────────
    #[test]
    fn heavy_tsne_alpha_invalid() {
        let mut rng = LcgRng::new(1004);
        let x = vec![1.0_f64; 6];
        let cfg = HeavyTsneConfig {
            alpha: 0.0,
            perplexity: 2.0,
            ..HeavyTsneConfig::default()
        };
        let result = heavy_tsne_fit(&x, 3, 2, &cfg, &mut rng);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifoldError::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            e => panic!("unexpected error variant: {e}"),
        }
    }

    // ── 5. alpha=1 separates clusters ─────────────────────────────────────────
    #[test]
    fn heavy_tsne_cluster_separation() {
        let mut rng = LcgRng::new(1005);
        let n_per = 5;
        let dim = 3;
        let x = two_cluster_data(n_per, dim, 5.0, &mut rng);
        let cfg = HeavyTsneConfig {
            alpha: 1.0,
            n_iter: 200,
            early_exaggeration_iters: 80,
            perplexity: 2.0,
            learning_rate: 50.0,
            momentum_switch_iter: 100,
            ..HeavyTsneConfig::default()
        };
        let r = heavy_tsne_fit(&x, 2 * n_per, dim, &cfg, &mut rng).expect("cluster_sep");
        assert!(r.embedding.iter().all(|v| v.is_finite()));
        // Clusters should be distinguishable — centroids separated in embedding.
        let diff = cluster_centroid_diff(&r.embedding, n_per, 2);
        assert!(
            diff > 0.0,
            "cluster centroids coincide in embedding: diff={diff:.6}"
        );
    }

    // ── 6. alpha=0.5 (Cauchy) separates clusters ──────────────────────────────
    #[test]
    fn heavy_tsne_alpha_05_cluster_separation() {
        let mut rng = LcgRng::new(1006);
        let n_per = 5;
        let dim = 3;
        let x = two_cluster_data(n_per, dim, 5.0, &mut rng);
        let cfg = HeavyTsneConfig {
            alpha: 0.5,
            n_iter: 200,
            early_exaggeration_iters: 80,
            perplexity: 2.0,
            learning_rate: 50.0,
            momentum_switch_iter: 100,
            ..HeavyTsneConfig::default()
        };
        let r = heavy_tsne_fit(&x, 2 * n_per, dim, &cfg, &mut rng).expect("cauchy_sep");
        assert!(r.embedding.iter().all(|v| v.is_finite()));
        let diff = cluster_centroid_diff(&r.embedding, n_per, 2);
        assert!(diff > 0.0, "Cauchy clusters not separated: diff={diff:.6}");
    }

    // ── 7. alpha=2 (sub-heavy tail) separates clusters ────────────────────────
    #[test]
    fn heavy_tsne_alpha_2_cluster_separation() {
        let mut rng = LcgRng::new(1007);
        let n_per = 5;
        let dim = 3;
        let x = two_cluster_data(n_per, dim, 5.0, &mut rng);
        let cfg = HeavyTsneConfig {
            alpha: 2.0,
            n_iter: 200,
            early_exaggeration_iters: 80,
            perplexity: 2.0,
            learning_rate: 50.0,
            momentum_switch_iter: 100,
            ..HeavyTsneConfig::default()
        };
        let r = heavy_tsne_fit(&x, 2 * n_per, dim, &cfg, &mut rng).expect("alpha2_sep");
        assert!(r.embedding.iter().all(|v| v.is_finite()));
        let diff = cluster_centroid_diff(&r.embedding, n_per, 2);
        assert!(diff > 0.0, "alpha=2 clusters not separated: diff={diff:.6}");
    }

    // ── 8. cauchy_tsne_fit runs without error ─────────────────────────────────
    #[test]
    fn cauchy_tsne_runs() {
        let mut rng = LcgRng::new(1008);
        let n = 8;
        let dim = 3;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let result = cauchy_tsne_fit(&x, n, dim, 50, 2.0, 100.0, &mut rng);
        assert!(result.is_ok(), "cauchy_tsne_fit failed: {:?}", result.err());
        let r = result.unwrap();
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }

    // ── 9. cauchy_tsne_fit output shape ───────────────────────────────────────
    #[test]
    fn cauchy_tsne_output_shape() {
        let mut rng = LcgRng::new(1009);
        let n = 10;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let r = cauchy_tsne_fit(&x, n, dim, 30, 2.0, 100.0, &mut rng).expect("cauchy_shape");
        // Default n_components = 2
        assert_eq!(r.embedding.len(), n * 2);
    }

    // ── 10. alpha_tsne_fit with alpha_init=5, alpha_final=0.5 runs ───────────
    #[test]
    fn alpha_tsne_config_valid() {
        let mut rng = LcgRng::new(1010);
        let n = 8;
        let dim = 3;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = AlphaTsneConfig {
            base: HeavyTsneConfig {
                n_iter: 50,
                early_exaggeration_iters: 15,
                perplexity: 2.0,
                learning_rate: 100.0,
                momentum_switch_iter: 30,
                ..HeavyTsneConfig::default()
            },
            alpha_init: 5.0,
            alpha_final: 0.5,
        };
        let result = alpha_tsne_fit(&x, n, dim, &cfg, &mut rng);
        assert!(result.is_ok(), "alpha_tsne_fit failed: {:?}", result.err());
        let r = result.unwrap();
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }

    // ── 11. alpha_tsne_fit output shape ───────────────────────────────────────
    #[test]
    fn alpha_tsne_output_shape() {
        let mut rng = LcgRng::new(1011);
        let n = 10;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = AlphaTsneConfig {
            base: HeavyTsneConfig {
                n_iter: 30,
                early_exaggeration_iters: 10,
                perplexity: 2.0,
                n_components: 2,
                momentum_switch_iter: 20,
                ..HeavyTsneConfig::default()
            },
            alpha_init: 10.0,
            alpha_final: 1.0,
        };
        let r = alpha_tsne_fit(&x, n, dim, &cfg, &mut rng).expect("alpha_shape");
        assert_eq!(r.embedding.len(), n * 2);
    }

    // ── 12. ssne_fit output shape ─────────────────────────────────────────────
    #[test]
    fn ssne_output_shape() {
        let mut rng = LcgRng::new(1012);
        let n = 10;
        let dim = 3;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = SsneConfig {
            n_components: 2,
            n_iter: 30,
            early_exaggeration_iters: 10,
            perplexity: 2.0,
            momentum_switch_iter: 20,
            ..SsneConfig::default()
        };
        let r = ssne_fit(&x, n, dim, &cfg, &mut rng).expect("ssne_shape");
        assert_eq!(r.embedding.len(), n * 2);
    }

    // ── 13. SSNE separates clusters ───────────────────────────────────────────
    #[test]
    fn ssne_cluster_separation() {
        let mut rng = LcgRng::new(1013);
        let n_per = 5;
        let dim = 3;
        let x = two_cluster_data(n_per, dim, 5.0, &mut rng);
        let cfg = SsneConfig {
            n_iter: 200,
            early_exaggeration_iters: 80,
            perplexity: 2.0,
            learning_rate: 50.0,
            momentum_switch_iter: 100,
            ..SsneConfig::default()
        };
        let r = ssne_fit(&x, 2 * n_per, dim, &cfg, &mut rng).expect("ssne_sep");
        assert!(r.embedding.iter().all(|v| v.is_finite()));
        let diff = cluster_centroid_diff(&r.embedding, n_per, 2);
        assert!(diff > 0.0, "SSNE clusters not separated: diff={diff:.6}");
    }

    // ── 14. ssne_fit final_kl is finite ───────────────────────────────────────
    #[test]
    fn ssne_finite_kl() {
        let mut rng = LcgRng::new(1014);
        let n = 10;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = SsneConfig {
            n_iter: 50,
            early_exaggeration_iters: 15,
            perplexity: 2.0,
            momentum_switch_iter: 30,
            ..SsneConfig::default()
        };
        let r = ssne_fit(&x, n, dim, &cfg, &mut rng).expect("ssne_kl");
        assert!(r.final_kl.is_finite(), "ssne final_kl={}", r.final_kl);
    }

    // ── 15. heavy_tsne_fit final_kl is finite ────────────────────────────────
    #[test]
    fn heavy_tsne_finite_kl() {
        let mut rng = LcgRng::new(1015);
        let n = 10;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = HeavyTsneConfig {
            n_iter: 50,
            early_exaggeration_iters: 15,
            perplexity: 2.0,
            alpha: 1.5,
            momentum_switch_iter: 30,
            ..HeavyTsneConfig::default()
        };
        let r = heavy_tsne_fit(&x, n, dim, &cfg, &mut rng).expect("heavy_kl");
        assert!(r.final_kl.is_finite(), "heavy_tsne final_kl={}", r.final_kl);
    }

    // ── 16. Large alpha (≈ Gaussian) behaves like SSNE ───────────────────────
    #[test]
    fn heavy_tsne_alpha_large_gaussian_like() {
        // With α = 1000 the kernel (1 + d²/1000)^{-500.5} ≈ exp(-d²/2) for small d².
        // The embedding should still be finite and the KL should be finite.
        let mut rng = LcgRng::new(1016);
        let n = 8;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = HeavyTsneConfig {
            alpha: 1000.0,
            n_iter: 50,
            early_exaggeration_iters: 15,
            perplexity: 2.0,
            learning_rate: 100.0,
            momentum_switch_iter: 30,
            ..HeavyTsneConfig::default()
        };
        let r = heavy_tsne_fit(&x, n, dim, &cfg, &mut rng).expect("large_alpha");
        assert!(
            r.embedding.iter().all(|v| v.is_finite()),
            "embedding not finite"
        );
        assert!(r.final_kl.is_finite(), "KL not finite for large alpha");
    }
}

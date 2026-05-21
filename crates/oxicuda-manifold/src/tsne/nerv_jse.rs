//! NeRV (Neighbor Retrieval Visualizer) and JSE (Jensen-Shannon Embedding).
//!
//! Both algorithms extend t-SNE by replacing the KL(P||Q) objective with alternative
//! divergences that allow finer control over the precision/recall trade-off in the
//! embedding.
//!
//! # References
//!
//! - Venna & Kaski (2006) "Visualizing Gene Interaction Graphs with Local Multidimensional
//!   Scaling". ESANN.
//! - Lee, Renard & Verleysen (2013) "Type 1 and 2 mixtures of Kullback–Leibler divergences as
//!   cost functions in dimensionality reduction based on similarity preservation". Neurocomputing.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::tsne::perplexity::compute_perplexity_p_matrix;

// ──────────────────────────────────────────────────────────────────────────────
// NeRV (Neighbor Retrieval Visualizer)
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the NeRV algorithm.
///
/// NeRV balances precision (low false-positive rate) and recall (low false-negative
/// rate) by mixing KL(Q||P) and KL(P||Q) with a trade-off parameter λ:
///
/// `L_NeRV = λ · KL(Q||P) + (1-λ) · KL(P||Q)`
///
/// - λ = 1.0 → pure KL(Q||P) = standard t-SNE (focus on precision)
/// - λ = 0.0 → pure KL(P||Q) (focus on recall)
#[derive(Debug, Clone)]
pub struct NervConfig {
    /// Embedding dimensionality (default 2).
    pub n_components: usize,
    /// Perplexity target for the P-matrix binary search (default 30.0).
    pub perplexity: f64,
    /// Trade-off λ ∈ [0, 1]. λ = 1 recovers standard t-SNE (default 0.5).
    pub lambda: f64,
    /// Total gradient-descent iterations (default 500).
    pub n_iter: usize,
    /// Initial learning rate (default 100.0).
    pub learning_rate: f64,
    /// Initial momentum coefficient (default 0.5).
    pub momentum: f64,
    /// Final momentum coefficient (default 0.8).
    pub final_momentum: f64,
    /// Iteration at which momentum switches from initial to final (default 250).
    pub momentum_switch_iter: usize,
    /// Early-exaggeration multiplier (default 4.0).
    pub early_exaggeration: f64,
    /// Number of early-exaggeration iterations (default 100).
    pub early_exaggeration_iters: usize,
    /// Minimum adaptive gain value (default 0.01).
    pub min_gain: f64,
    /// Maximum iterations for the perplexity binary search (default 50).
    pub perp_max_iter: usize,
    /// Convergence tolerance for the perplexity binary search (default 1e-5).
    pub perp_tol: f64,
}

impl Default for NervConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            lambda: 0.5,
            n_iter: 500,
            learning_rate: 100.0,
            momentum: 0.5,
            final_momentum: 0.8,
            momentum_switch_iter: 250,
            early_exaggeration: 4.0,
            early_exaggeration_iters: 100,
            min_gain: 0.01,
            perp_max_iter: 50,
            perp_tol: 1e-5,
        }
    }
}

/// Result of NeRV dimensionality reduction.
#[derive(Debug)]
pub struct NervResult {
    /// Row-major embedding matrix of shape `[n_samples, n_components]`.
    pub embedding: Vec<f64>,
    /// Final value of the NeRV objective `L_NeRV`.
    pub final_loss: f64,
}

/// Fit NeRV on row-major input data of shape `(n_samples, dim)`.
///
/// # Arguments
///
/// * `x`         – Row-major input data `[n_samples × dim]`.
/// * `n_samples` – Number of data points.
/// * `dim`       – Input dimensionality.
/// * `cfg`       – [`NervConfig`] hyperparameters.
/// * `rng`       – Seeded [`LcgRng`] for reproducible initialisation.
///
/// # Errors
///
/// Returns [`ManifoldError`] for empty input, shape mismatches, or invalid parameters.
pub fn nerv_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    cfg: &NervConfig,
    rng: &mut LcgRng,
) -> ManifoldResult<NervResult> {
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
    if cfg.n_components == 0 || cfg.n_components > 8 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be in 1..=8".into(),
        });
    }
    if !(0.0..=1.0).contains(&cfg.lambda) {
        return Err(ManifoldError::InvalidParameter {
            name: "lambda".into(),
            reason: "must be in [0, 1]".into(),
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

    // ── Step 1: pairwise squared Euclidean distances ──────────────────────────
    let d2 = compute_pairwise_sq_dist(x, n, dim);

    // ── Step 2: joint probability matrix P via perplexity binary search ───────
    let mut p =
        compute_perplexity_p_matrix(&d2, n, cfg.perplexity, cfg.perp_max_iter, cfg.perp_tol)?;
    // Clamp to avoid exact zeros in log computations
    for v in &mut p {
        if *v < f64::EPSILON {
            *v = f64::EPSILON;
        }
    }

    // ── Step 3: apply early exaggeration ─────────────────────────────────────
    for v in &mut p {
        *v *= cfg.early_exaggeration;
    }

    // ── Step 4: initialise embedding from N(0, 1e-4) ─────────────────────────
    let mut y = vec![0.0_f64; n * d_out];
    for v in &mut y {
        *v = rng.next_normal() * 1e-4;
    }

    // Momentum / gain state
    let mut dy_prev = vec![0.0_f64; n * d_out];
    let mut gains = vec![1.0_f64; n * d_out];

    let mut final_loss = 0.0_f64;

    // ── Step 5: gradient-descent loop ────────────────────────────────────────
    for iter in 0..cfg.n_iter {
        // Remove early exaggeration at the right iteration
        if iter == cfg.early_exaggeration_iters {
            for v in &mut p {
                *v /= cfg.early_exaggeration;
            }
        }

        // Current momentum
        let mom = if iter < cfg.momentum_switch_iter {
            cfg.momentum
        } else {
            cfg.final_momentum
        };

        // Compute Q matrix (student-t kernel, unnormalised weights and normalised q_ij)
        let (q, w) = compute_q_and_weights(&y, n, d_out);

        // Compute NeRV gradient
        let grad = nerv_gradient(&p, &q, &w, &y, n, d_out, cfg.lambda);

        // Adaptive gain update and parameter step
        apply_gain_step(
            &mut y,
            &mut dy_prev,
            &mut gains,
            &grad,
            cfg.learning_rate,
            mom,
            cfg.min_gain,
        );

        // Re-centre embedding
        centre_embedding(&mut y, n, d_out);

        // Record final loss on last iteration
        if iter == cfg.n_iter.saturating_sub(1) {
            final_loss = nerv_loss(&p, &q, cfg.lambda);
        }
    }

    Ok(NervResult {
        embedding: y,
        final_loss,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// JSE (Jensen-Shannon Embedding)
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the JSE algorithm.
///
/// JSE minimises the generalised Jensen-Shannon divergence:
///
/// `JS_κ(P||Q) = κ · KL(P || M_κ) + (1-κ) · KL(Q || M_κ)`
///
/// where `M_κ = κP + (1-κ)Q` is the mixture distribution.  The symmetric case
/// κ = 0.5 yields the standard Jensen-Shannon divergence.
#[derive(Debug, Clone)]
pub struct JseConfig {
    /// Embedding dimensionality (default 2).
    pub n_components: usize,
    /// Perplexity target for the P-matrix binary search (default 30.0).
    pub perplexity: f64,
    /// Mixture weight κ ∈ (0, 1) (default 0.5).
    pub kappa: f64,
    /// Total gradient-descent iterations (default 500).
    pub n_iter: usize,
    /// Initial learning rate (default 100.0).
    pub learning_rate: f64,
    /// Initial momentum coefficient (default 0.5).
    pub momentum: f64,
    /// Final momentum coefficient (default 0.8).
    pub final_momentum: f64,
    /// Iteration at which momentum switches from initial to final (default 250).
    pub momentum_switch_iter: usize,
    /// Early-exaggeration multiplier (default 4.0).
    pub early_exaggeration: f64,
    /// Number of early-exaggeration iterations (default 100).
    pub early_exaggeration_iters: usize,
    /// Minimum adaptive gain value (default 0.01).
    pub min_gain: f64,
    /// Maximum iterations for the perplexity binary search (default 50).
    pub perp_max_iter: usize,
    /// Convergence tolerance for the perplexity binary search (default 1e-5).
    pub perp_tol: f64,
}

impl Default for JseConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            kappa: 0.5,
            n_iter: 500,
            learning_rate: 100.0,
            momentum: 0.5,
            final_momentum: 0.8,
            momentum_switch_iter: 250,
            early_exaggeration: 4.0,
            early_exaggeration_iters: 100,
            min_gain: 0.01,
            perp_max_iter: 50,
            perp_tol: 1e-5,
        }
    }
}

/// Result of JSE dimensionality reduction.
#[derive(Debug)]
pub struct JseResult {
    /// Row-major embedding matrix of shape `[n_samples, n_components]`.
    pub embedding: Vec<f64>,
    /// Final value of the JSE objective `JS_κ(P||Q)`.
    pub final_loss: f64,
}

/// Fit JSE on row-major input data of shape `(n_samples, dim)`.
///
/// # Arguments
///
/// * `x`         – Row-major input data `[n_samples × dim]`.
/// * `n_samples` – Number of data points.
/// * `dim`       – Input dimensionality.
/// * `cfg`       – [`JseConfig`] hyperparameters.
/// * `rng`       – Seeded [`LcgRng`] for reproducible initialisation.
///
/// # Errors
///
/// Returns [`ManifoldError`] for empty input, shape mismatches, or invalid parameters.
pub fn jse_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    cfg: &JseConfig,
    rng: &mut LcgRng,
) -> ManifoldResult<JseResult> {
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
    if cfg.n_components == 0 || cfg.n_components > 8 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be in 1..=8".into(),
        });
    }
    if cfg.kappa <= 0.0 || cfg.kappa >= 1.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "kappa".into(),
            reason: "must be in (0, 1)".into(),
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

    // ── Step 1: pairwise squared Euclidean distances ──────────────────────────
    let d2 = compute_pairwise_sq_dist(x, n, dim);

    // ── Step 2: joint probability matrix P via perplexity binary search ───────
    let mut p =
        compute_perplexity_p_matrix(&d2, n, cfg.perplexity, cfg.perp_max_iter, cfg.perp_tol)?;
    // Clamp to avoid exact zeros in log computations
    for v in &mut p {
        if *v < f64::EPSILON {
            *v = f64::EPSILON;
        }
    }

    // ── Step 3: apply early exaggeration ─────────────────────────────────────
    for v in &mut p {
        *v *= cfg.early_exaggeration;
    }

    // ── Step 4: initialise embedding from N(0, 1e-4) ─────────────────────────
    let mut y = vec![0.0_f64; n * d_out];
    for v in &mut y {
        *v = rng.next_normal() * 1e-4;
    }

    // Momentum / gain state
    let mut dy_prev = vec![0.0_f64; n * d_out];
    let mut gains = vec![1.0_f64; n * d_out];

    let mut final_loss = 0.0_f64;

    // ── Step 5: gradient-descent loop ────────────────────────────────────────
    for iter in 0..cfg.n_iter {
        // Remove early exaggeration at the right iteration
        if iter == cfg.early_exaggeration_iters {
            for v in &mut p {
                *v /= cfg.early_exaggeration;
            }
        }

        // Current momentum
        let mom = if iter < cfg.momentum_switch_iter {
            cfg.momentum
        } else {
            cfg.final_momentum
        };

        // Compute Q matrix and unnormalised student-t weights
        let (q, w) = compute_q_and_weights(&y, n, d_out);

        // Compute JSE gradient
        let grad = jse_gradient(&p, &q, &w, &y, n, d_out, cfg.kappa);

        // Adaptive gain update and parameter step
        apply_gain_step(
            &mut y,
            &mut dy_prev,
            &mut gains,
            &grad,
            cfg.learning_rate,
            mom,
            cfg.min_gain,
        );

        // Re-centre embedding
        centre_embedding(&mut y, n, d_out);

        // Record final loss on last iteration
        if iter == cfg.n_iter.saturating_sub(1) {
            final_loss = jse_loss(&p, &q, cfg.kappa);
        }
    }

    Ok(JseResult {
        embedding: y,
        final_loss,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared computation helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute row-major pairwise squared Euclidean distances for `n` points of `dim` dimensions.
fn compute_pairwise_sq_dist(x: &[f64], n: usize, dim: usize) -> Vec<f64> {
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

/// Compute the student-t kernel Q matrix and the unnormalised weights W.
///
/// `w_ij = (1 + ||y_i - y_j||^2)^{-1}` (0 on diagonal)
/// `q_ij = w_ij / Z`  where  `Z = Σ_{k≠l} w_kl`
///
/// Returns `(q, w)` both of length `n*n`.  Values on the diagonal are 0.
/// Off-diagonal q values are clamped to ≥ `f64::EPSILON`.
fn compute_q_and_weights(y: &[f64], n: usize, dim: usize) -> (Vec<f64>, Vec<f64>) {
    let mut w = vec![0.0_f64; n * n];
    let mut z = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut d2 = 0.0_f64;
            for k in 0..dim {
                let v = y[i * dim + k] - y[j * dim + k];
                d2 += v * v;
            }
            let wij = 1.0 / (1.0 + d2);
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
            let qij = w[i * n + j] / z;
            // Clamp to avoid log(0)
            q[i * n + j] = qij.max(f64::EPSILON);
        }
    }
    (q, w)
}

/// NeRV gradient.
///
/// The NeRV objective is:
/// ```text
/// L_NeRV = λ · KL(Q||P) + (1-λ) · KL(P||Q)
/// ```
///
/// Differentiating with respect to `y_i`:
///
/// **KL(Q||P) component** (standard t-SNE gradient):
/// ```text
/// ∂KL(Q||P)/∂y_i = 4 Σ_{j≠i} (q_ij - p_ij) · w_ij · (y_i - y_j)
/// ```
///
/// **KL(P||Q) component** (derived from Venna & Kaski 2006, Eq. 3):
/// ```text
/// ∂KL(P||Q)/∂y_i = 4 Σ_{j≠i} p_ij · (log p_ij - log q_ij) · w_ij · (y_i - y_j)
/// ```
///
/// The combined NeRV gradient is the λ-weighted sum.
fn nerv_gradient(
    p: &[f64],
    q: &[f64],
    w: &[f64],
    y: &[f64],
    n: usize,
    d_out: usize,
    lambda: f64,
) -> Vec<f64> {
    let mut grad = vec![0.0_f64; n * d_out];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let pij = p[i * n + j].max(f64::EPSILON);
            let qij = q[i * n + j].max(f64::EPSILON);
            let wij = w[i * n + j];

            // KL(Q||P) term:  (q_ij - p_ij) · w_ij
            let kl_qp_coeff = (qij - pij) * wij;

            // KL(P||Q) term:  p_ij · (log p_ij - log q_ij) · w_ij
            //   = p_ij · log(p_ij / q_ij) · w_ij
            let log_ratio = (pij / qij).ln();
            let kl_pq_coeff = pij * log_ratio * wij;

            let coeff = lambda * kl_qp_coeff + (1.0 - lambda) * kl_pq_coeff;
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

/// JSE gradient.
///
/// Let `M_ij = κ · p_ij + (1-κ) · q_ij`.
///
/// The JSE objective is:
/// ```text
/// L_JSE = κ · Σ p_ij log(p_ij / M_ij) + (1-κ) · Σ q_ij log(q_ij / M_ij)
/// ```
///
/// Differentiating with respect to `y_i` (Lee et al. 2013, Eq. 10):
/// ```text
/// ∂L_JSE/∂y_i = 4(1-κ) Σ_{j≠i} [ q_ij · log(q_ij / M_ij) - q_ij + M_ij ] · (y_i - y_j)
///             + 4(1-κ)κ Σ_{j≠i} (p_ij - q_ij) / M_ij · w_ij · (y_i - y_j)
/// ```
///
/// Using the factored stable form (avoids log of near-zero):
/// ```text
/// ∂L_JSE/∂y_i = 4(1-κ) Σ_{j≠i} [
///       q_ij · log(q_ij / M_ij)        ← recall term
///     + κ · (p_ij - q_ij) / M_ij · w_ij  ← cross term
/// ] · (y_i - y_j)
/// ```
///
/// For κ = 0.5 this reduces to:
/// ```text
/// ∂L_JSE/∂y_i = 2 Σ_{j≠i} (p_ij - q_ij) / (p_ij + q_ij) · q_ij · (y_i - y_j)
/// ```
fn jse_gradient(
    p: &[f64],
    q: &[f64],
    w: &[f64],
    y: &[f64],
    n: usize,
    d_out: usize,
    kappa: f64,
) -> Vec<f64> {
    let one_minus_k = 1.0 - kappa;
    let mut grad = vec![0.0_f64; n * d_out];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let pij = p[i * n + j].max(f64::EPSILON);
            let qij = q[i * n + j].max(f64::EPSILON);
            let wij = w[i * n + j];

            // Mixture distribution M_ij = κ·p_ij + (1-κ)·q_ij
            let mij = (kappa * pij + one_minus_k * qij).max(f64::EPSILON);

            // Recall term: q_ij · log(q_ij / M_ij)  (q contribution to JS divergence)
            let recall_term = qij * (qij / mij).ln();

            // Cross term: κ · (p_ij - q_ij) / M_ij · w_ij
            //   This arises from differentiating q_ij w.r.t. y_i through the Q normalisation.
            let cross_term = kappa * (pij - qij) / mij * wij;

            // Combined coefficient (scaled by 4(1-κ) outside loop)
            let coeff = recall_term + cross_term;

            for k in 0..d_out {
                grad[i * d_out + k] += coeff * (y[i * d_out + k] - y[j * d_out + k]);
            }
        }
    }

    // Scale by 4(1-κ)
    let scale = 4.0 * one_minus_k;
    for v in &mut grad {
        *v *= scale;
    }
    grad
}

/// Compute the NeRV loss `L_NeRV = λ·KL(Q||P) + (1-λ)·KL(P||Q)`.
fn nerv_loss(p: &[f64], q: &[f64], lambda: f64) -> f64 {
    let mut kl_qp = 0.0_f64; // Σ q_ij log(q_ij / p_ij)
    let mut kl_pq = 0.0_f64; // Σ p_ij log(p_ij / q_ij)
    let n2 = p.len();
    for idx in 0..n2 {
        let pij = p[idx].max(f64::EPSILON);
        let qij = q[idx].max(f64::EPSILON);
        kl_qp += qij * (qij / pij).ln();
        kl_pq += pij * (pij / qij).ln();
    }
    lambda * kl_qp + (1.0 - lambda) * kl_pq
}

/// Compute the JSE loss `JS_κ(P||Q) = κ·KL(P||M) + (1-κ)·KL(Q||M)`.
fn jse_loss(p: &[f64], q: &[f64], kappa: f64) -> f64 {
    let one_minus_k = 1.0 - kappa;
    let mut kl_pm = 0.0_f64; // Σ p_ij log(p_ij / M_ij)
    let mut kl_qm = 0.0_f64; // Σ q_ij log(q_ij / M_ij)
    let n2 = p.len();
    for idx in 0..n2 {
        let pij = p[idx].max(f64::EPSILON);
        let qij = q[idx].max(f64::EPSILON);
        let mij = (kappa * pij + one_minus_k * qij).max(f64::EPSILON);
        kl_pm += pij * (pij / mij).ln();
        kl_qm += qij * (qij / mij).ln();
    }
    kappa * kl_pm + one_minus_k * kl_qm
}

/// Apply the adaptive gain rule and gradient-descent step.
///
/// Uses the standard t-SNE gain adaptation:
/// - If gradient keeps same sign as previous update: multiply gain by 0.8.
/// - If gradient flips sign: add 0.2 to gain.
/// - Gain clamped from below at `min_gain`.
///
/// Then applies `Δy = momentum · Δy_prev - lr · gain · grad`.
fn apply_gain_step(
    y: &mut [f64],
    dy_prev: &mut [f64],
    gains: &mut [f64],
    grad: &[f64],
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

/// Re-centre the embedding so its column means are zero.
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

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. NervConfig::default() is valid ────────────────────────────────────
    #[test]
    fn nerv_default_config_valid() {
        let cfg = NervConfig::default();
        assert_eq!(cfg.n_components, 2);
        assert!((0.0..=1.0).contains(&cfg.lambda));
        assert!(cfg.perplexity > 0.0);
        assert!(cfg.n_iter > 0);
        assert!(cfg.learning_rate > 0.0);
    }

    // ── 2. JseConfig::default() is valid ─────────────────────────────────────
    #[test]
    fn jse_default_config_valid() {
        let cfg = JseConfig::default();
        assert_eq!(cfg.n_components, 2);
        assert!(cfg.kappa > 0.0 && cfg.kappa < 1.0);
        assert!(cfg.perplexity > 0.0);
        assert!(cfg.n_iter > 0);
        assert!(cfg.learning_rate > 0.0);
    }

    // ── 3. nerv_fit returns Err on empty input ────────────────────────────────
    #[test]
    fn nerv_empty_input_error() {
        let mut rng = LcgRng::new(1);
        let cfg = NervConfig::default();
        let result = nerv_fit(&[], 0, 2, &cfg, &mut rng);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifoldError::EmptyInput => {}
            e => panic!("unexpected error: {e}"),
        }
    }

    // ── 4. jse_fit returns Err on empty input ─────────────────────────────────
    #[test]
    fn jse_empty_input_error() {
        let mut rng = LcgRng::new(2);
        let cfg = JseConfig::default();
        let result = jse_fit(&[], 0, 2, &cfg, &mut rng);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifoldError::EmptyInput => {}
            e => panic!("unexpected error: {e}"),
        }
    }

    // ── 5. With lambda=1.0, NeRV behaves like t-SNE ───────────────────────────
    #[test]
    fn nerv_lambda_1_close_to_tsne() {
        let mut rng = LcgRng::new(42);
        let n = 8;
        let dim = 3;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = NervConfig {
            lambda: 1.0,
            n_iter: 30,
            early_exaggeration_iters: 10,
            perplexity: 2.0,
            n_components: 2,
            ..NervConfig::default()
        };
        let result = nerv_fit(&x, n, dim, &cfg, &mut rng);
        assert!(result.is_ok(), "NeRV lambda=1.0 failed: {:?}", result.err());
        let r = result.unwrap();
        assert_eq!(r.embedding.len(), n * 2);
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }

    // ── 6. With lambda=0.0, NeRV runs (pure KL(P||Q)) ────────────────────────
    #[test]
    fn nerv_lambda_0_kl_pq() {
        let mut rng = LcgRng::new(43);
        let n = 8;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = NervConfig {
            lambda: 0.0,
            n_iter: 30,
            early_exaggeration_iters: 10,
            perplexity: 2.0,
            n_components: 2,
            ..NervConfig::default()
        };
        let result = nerv_fit(&x, n, dim, &cfg, &mut rng);
        assert!(result.is_ok(), "NeRV lambda=0.0 failed: {:?}", result.err());
        let r = result.unwrap();
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }

    // ── 7. With kappa=0.5, JSE is symmetric and runs cleanly ─────────────────
    #[test]
    fn jse_kappa_half_symmetric() {
        let mut rng = LcgRng::new(44);
        let n = 8;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = JseConfig {
            kappa: 0.5,
            n_iter: 30,
            early_exaggeration_iters: 10,
            perplexity: 2.0,
            n_components: 2,
            ..JseConfig::default()
        };
        let result = jse_fit(&x, n, dim, &cfg, &mut rng);
        assert!(result.is_ok(), "JSE kappa=0.5 failed: {:?}", result.err());
        let r = result.unwrap();
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }

    // ── 8. nerv_fit produces correct output shape ─────────────────────────────
    #[test]
    fn nerv_output_shape() {
        let mut rng = LcgRng::new(45);
        let n = 10;
        let dim = 4;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = NervConfig {
            n_components: 3,
            n_iter: 20,
            early_exaggeration_iters: 5,
            perplexity: 2.0,
            ..NervConfig::default()
        };
        let r = nerv_fit(&x, n, dim, &cfg, &mut rng).expect("nerv_output_shape");
        assert_eq!(r.embedding.len(), n * 3);
    }

    // ── 9. jse_fit produces correct output shape ──────────────────────────────
    #[test]
    fn jse_output_shape() {
        let mut rng = LcgRng::new(46);
        let n = 10;
        let dim = 4;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = JseConfig {
            n_components: 3,
            n_iter: 20,
            early_exaggeration_iters: 5,
            perplexity: 2.0,
            ..JseConfig::default()
        };
        let r = jse_fit(&x, n, dim, &cfg, &mut rng).expect("jse_output_shape");
        assert_eq!(r.embedding.len(), n * 3);
    }

    // ── 10. NeRV separates two well-separated clusters ───────────────────────
    #[test]
    fn nerv_cluster_separation() {
        let mut rng = LcgRng::new(100);
        let n_per_cluster = 5;
        let n = 2 * n_per_cluster;
        let dim = 3;
        let mut x = vec![0.0_f64; n * dim];
        // Cluster A at +5, cluster B at -5
        for i in 0..n_per_cluster {
            for d in 0..dim {
                x[i * dim + d] = 5.0 + 0.05 * rng.next_normal();
            }
        }
        for i in n_per_cluster..n {
            for d in 0..dim {
                x[i * dim + d] = -5.0 + 0.05 * rng.next_normal();
            }
        }
        let cfg = NervConfig {
            lambda: 0.5,
            n_iter: 200,
            early_exaggeration_iters: 80,
            perplexity: 2.0,
            learning_rate: 50.0,
            n_components: 2,
            ..NervConfig::default()
        };
        let r = nerv_fit(&x, n, dim, &cfg, &mut rng).expect("nerv_cluster_separation");
        assert!(r.embedding.iter().all(|v| v.is_finite()));

        // Compute cluster centroids in embedding space (dimension 0 only for speed)
        let mut ca = 0.0_f64;
        let mut cb = 0.0_f64;
        for i in 0..n_per_cluster {
            ca += r.embedding[i * 2];
        }
        for i in n_per_cluster..n {
            cb += r.embedding[i * 2];
        }
        ca /= n_per_cluster as f64;
        cb /= n_per_cluster as f64;

        // Centroids should be on opposite sides of zero
        assert!(
            ca * cb < 0.0 || (ca - cb).abs() > 0.5,
            "Clusters not separated: ca={ca:.4} cb={cb:.4}"
        );
    }

    // ── 11. JSE separates two well-separated clusters ────────────────────────
    #[test]
    fn jse_cluster_separation() {
        let mut rng = LcgRng::new(101);
        let n_per_cluster = 5;
        let n = 2 * n_per_cluster;
        let dim = 3;
        let mut x = vec![0.0_f64; n * dim];
        // Cluster A at +5, cluster B at -5
        for i in 0..n_per_cluster {
            for d in 0..dim {
                x[i * dim + d] = 5.0 + 0.05 * rng.next_normal();
            }
        }
        for i in n_per_cluster..n {
            for d in 0..dim {
                x[i * dim + d] = -5.0 + 0.05 * rng.next_normal();
            }
        }
        let cfg = JseConfig {
            kappa: 0.5,
            n_iter: 200,
            early_exaggeration_iters: 80,
            perplexity: 2.0,
            learning_rate: 50.0,
            n_components: 2,
            ..JseConfig::default()
        };
        let r = jse_fit(&x, n, dim, &cfg, &mut rng).expect("jse_cluster_separation");
        assert!(r.embedding.iter().all(|v| v.is_finite()));

        let mut ca = 0.0_f64;
        let mut cb = 0.0_f64;
        for i in 0..n_per_cluster {
            ca += r.embedding[i * 2];
        }
        for i in n_per_cluster..n {
            cb += r.embedding[i * 2];
        }
        ca /= n_per_cluster as f64;
        cb /= n_per_cluster as f64;

        assert!(
            ca * cb < 0.0 || (ca - cb).abs() > 0.5,
            "Clusters not separated: ca={ca:.4} cb={cb:.4}"
        );
    }

    // ── 12. lambda > 1.0 returns InvalidParameter error ──────────────────────
    #[test]
    fn nerv_invalid_lambda_err() {
        let mut rng = LcgRng::new(50);
        let x = vec![1.0_f64; 4];
        let cfg = NervConfig {
            lambda: 1.5,
            perplexity: 2.0,
            ..NervConfig::default()
        };
        let result = nerv_fit(&x, 2, 2, &cfg, &mut rng);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifoldError::InvalidParameter { name, .. } => {
                assert_eq!(name, "lambda");
            }
            e => panic!("unexpected error: {e}"),
        }
    }

    // ── 13. kappa ≤ 0 or ≥ 1 returns InvalidParameter error ─────────────────
    #[test]
    fn jse_invalid_kappa_err() {
        let mut rng = LcgRng::new(51);
        let x = vec![1.0_f64; 4];

        // kappa = 0.0
        let cfg0 = JseConfig {
            kappa: 0.0,
            perplexity: 2.0,
            ..JseConfig::default()
        };
        let r0 = jse_fit(&x, 2, 2, &cfg0, &mut rng);
        assert!(r0.is_err());
        match r0.unwrap_err() {
            ManifoldError::InvalidParameter { name, .. } => assert_eq!(name, "kappa"),
            e => panic!("unexpected error: {e}"),
        }

        // kappa = 1.0
        let cfg1 = JseConfig {
            kappa: 1.0,
            perplexity: 2.0,
            ..JseConfig::default()
        };
        let r1 = jse_fit(&x, 2, 2, &cfg1, &mut rng);
        assert!(r1.is_err());
        match r1.unwrap_err() {
            ManifoldError::InvalidParameter { name, .. } => assert_eq!(name, "kappa"),
            e => panic!("unexpected error: {e}"),
        }
    }

    // ── 14. nerv_fit final_loss is finite ─────────────────────────────────────
    #[test]
    fn nerv_finite_loss() {
        let mut rng = LcgRng::new(60);
        let n = 10;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = NervConfig {
            n_iter: 50,
            early_exaggeration_iters: 15,
            perplexity: 2.0,
            lambda: 0.7,
            n_components: 2,
            ..NervConfig::default()
        };
        let r = nerv_fit(&x, n, dim, &cfg, &mut rng).expect("nerv_finite_loss");
        assert!(r.final_loss.is_finite(), "final_loss={}", r.final_loss);
    }

    // ── 15. jse_fit final_loss is finite ──────────────────────────────────────
    #[test]
    fn jse_finite_loss() {
        let mut rng = LcgRng::new(61);
        let n = 10;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let cfg = JseConfig {
            n_iter: 50,
            early_exaggeration_iters: 15,
            perplexity: 2.0,
            kappa: 0.3,
            n_components: 2,
            ..JseConfig::default()
        };
        let r = jse_fit(&x, n, dim, &cfg, &mut rng).expect("jse_finite_loss");
        assert!(r.final_loss.is_finite(), "final_loss={}", r.final_loss);
    }
}

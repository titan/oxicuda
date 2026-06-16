//! Parametric t-SNE with MLP encoder.
//!
//! Reference: van der Maaten, L. (2009). "Learning a Parametric Embedding by Preserving Local
//! Structure." *Proceedings of AISTATS*, pp. 384–391.
//!
//! Trains a multi-layer perceptron (MLP) as a parametric embedding function by minimising the
//! KL divergence KL(P||Q) between the high-dimensional affinity matrix P and the low-dimensional
//! Student-t affinities Q. Enables out-of-sample generalisation via the learned encoder.
//!
//! # Architecture
//! - Encoder: `d → h₁ → … → hₖ → n_components` (ReLU hidden layers, linear output)
//! - Weight initialisation: Kaiming uniform (He init) for ReLU activations
//! - Optimiser: Adam (adaptive moment estimation) with bias correction
//! - P matrix: joint symmetrised affinities via per-row perplexity binary search
//! - Early exaggeration: P is multiplied by `early_exaggeration` for the first
//!   `early_exaggeration_epochs` epochs to encourage tight initial clusters

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

/// Forward-pass cache entry: `(pre_activations, activations)`.
///
/// - `pre_activations[l]` is the raw linear output (before activation) of layer l.
/// - `activations[l]` is the post-activation output; `activations[0]` is the input.
type MlpCache = (Vec<Vec<f64>>, Vec<Vec<f64>>);

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for parametric t-SNE.
#[derive(Debug, Clone)]
pub struct ParametricTsneConfig {
    /// Embedding dimensionality (typically 2).
    pub n_components: usize,
    /// Perplexity target for the joint probability matrix P.
    pub perplexity: f64,
    /// Widths of hidden layers (between input and output layers).
    pub hidden_dims: Vec<usize>,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Total number of training epochs.
    pub n_epochs: usize,
    /// Mini-batch size for stochastic gradient updates.
    pub batch_size: usize,
    /// Early exaggeration multiplier applied to P during the first epochs.
    pub early_exaggeration: f64,
    /// Number of epochs during which early exaggeration is active.
    pub early_exaggeration_epochs: usize,
    /// Adam β₁ (first-moment decay).
    pub beta1: f64,
    /// Adam β₂ (second-moment decay).
    pub beta2: f64,
    /// Adam ε (numerical stability).
    pub adam_eps: f64,
    /// Seed for the LCG RNG (weight init and minibatch shuffling).
    pub seed: u64,
}

impl Default for ParametricTsneConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            hidden_dims: vec![500, 500],
            learning_rate: 1e-3,
            n_epochs: 1000,
            batch_size: 64,
            early_exaggeration: 12.0,
            early_exaggeration_epochs: 250,
            beta1: 0.9,
            beta2: 0.999,
            adam_eps: 1e-8,
            seed: 42,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────────────────────────────────────

/// A trained parametric t-SNE model (MLP encoder).
///
/// Layer dimensions: `layer_dims[0]` = input dim, `layer_dims.last()` = output dim.
/// `weights[l]` is the weight matrix for layer l stored row-major as `[out × in]`.
/// `biases[l]` is the bias vector of length `layer_dims[l+1]`.
#[derive(Debug, Clone)]
pub struct ParametricTsneModel {
    /// Weight tensors, one per layer transition. `weights[l].len() = out_l × in_l`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vectors, one per layer transition.
    pub biases: Vec<Vec<f64>>,
    /// Layer widths `[n_features, h1, …, n_components]`.
    pub layer_dims: Vec<usize>,
    /// Configuration used to fit this model.
    pub config: ParametricTsneConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP forward pass helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Perform a forward pass of the MLP on a single input vector `x` (length `n_features`).
///
/// Returns the output embedding vector of length `n_components`.
/// Hidden layers use ReLU activation; the final layer is linear.
#[must_use]
pub fn parametric_tsne_forward(model: &ParametricTsneModel, x: &[f64]) -> Vec<f64> {
    let n_layers = model.weights.len();
    let mut activation: Vec<f64> = x.to_vec();
    for l in 0..n_layers {
        let in_dim = model.layer_dims[l];
        let out_dim = model.layer_dims[l + 1];
        let w = &model.weights[l];
        let b = &model.biases[l];
        let mut next = vec![0.0f64; out_dim];
        for o in 0..out_dim {
            let mut acc = b[o];
            for i in 0..in_dim {
                acc += w[o * in_dim + i] * activation[i];
            }
            // ReLU on all hidden layers; linear on the final layer
            next[o] = if l + 1 < n_layers { acc.max(0.0) } else { acc };
        }
        activation = next;
    }
    activation
}

/// Forward pass that also returns pre-activations and post-activations for each hidden layer.
///
/// Returns `(pre_activations, activations)` where `activations[0]` is the input, and
/// `activations[i+1]` is after layer i (post-activation). `pre_activations[l]` is the raw
/// linear output of layer l (before activation).
fn mlp_forward_with_cache(
    model: &ParametricTsneModel,
    x: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_layers = model.weights.len();
    let mut activations: Vec<Vec<f64>> = Vec::with_capacity(n_layers + 1);
    let mut pre_acts: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
    activations.push(x.to_vec());
    for l in 0..n_layers {
        let in_dim = model.layer_dims[l];
        let out_dim = model.layer_dims[l + 1];
        let w = &model.weights[l];
        let b = &model.biases[l];
        let prev = &activations[l];
        let mut z = vec![0.0f64; out_dim];
        for o in 0..out_dim {
            let mut acc = b[o];
            for i in 0..in_dim {
                acc += w[o * in_dim + i] * prev[i];
            }
            z[o] = acc;
        }
        pre_acts.push(z.clone());
        // Apply activation: ReLU on all hidden layers, linear on final
        let a: Vec<f64> = if l + 1 < n_layers {
            z.iter().map(|&v| v.max(0.0)).collect()
        } else {
            z
        };
        activations.push(a);
    }
    (pre_acts, activations)
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight initialisation (Kaiming/He uniform for ReLU)
// ─────────────────────────────────────────────────────────────────────────────

/// Initialise a weight matrix with Kaiming uniform: U(-√(6/fan_in), +√(6/fan_in)).
fn kaiming_uniform_init(out_dim: usize, in_dim: usize, rng: &mut LcgRng) -> Vec<f64> {
    let fan_in = in_dim.max(1);
    let bound = (6.0_f64 / fan_in as f64).sqrt();
    (0..out_dim * in_dim)
        .map(|_| rng.next_range(-bound, bound))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// P matrix computation (inline, self-contained)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute conditional P(j|i) for a single row i via perplexity binary search.
///
/// `dist_sq_row` is the row of squared distances from point i to all n points.
/// Returns the conditional probability row of length n (with p[i] = 0).
fn p_row_binary_search(
    dist_sq_row: &[f64],
    i: usize,
    perplexity: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = dist_sq_row.len();
    let log_perp = perplexity.ln();
    let mut beta = 1.0_f64;
    let mut beta_min = f64::NEG_INFINITY;
    let mut beta_max = f64::INFINITY;
    let mut p_row = vec![0.0f64; n];

    for _ in 0..max_iter {
        let mut z = 0.0_f64;
        let mut h_num = 0.0_f64;
        for (j, &d) in dist_sq_row.iter().enumerate() {
            if j == i {
                continue;
            }
            let e = (-d * beta).exp();
            p_row[j] = e;
            z += e;
            h_num += e * (-d * beta);
        }
        // Handle near-zero partition function
        if z < 1e-300 {
            beta /= 2.0;
            continue;
        }
        // Normalise
        for (j, pv) in p_row.iter_mut().enumerate() {
            if j == i {
                *pv = 0.0;
            } else {
                *pv /= z;
            }
        }
        // Entropy: H = log(Z) - H_num/Z = log Z + β * Σ d*p
        let h = z.ln() - h_num / z;
        let diff = h - log_perp;
        if diff.abs() < tol {
            return p_row;
        }
        if diff > 0.0 {
            beta_min = beta;
            beta = if beta_max.is_infinite() {
                beta * 2.0
            } else {
                (beta + beta_max) / 2.0
            };
        } else {
            beta_max = beta;
            beta = if beta_min.is_infinite() {
                beta / 2.0
            } else {
                (beta + beta_min) / 2.0
            };
        }
    }
    p_row
}

/// Build the symmetric joint affinity matrix P from row-major input data.
///
/// Returns flat n×n matrix where `p[i*n+j] = (p(j|i) + p(i|j)) / (2n)`,
/// clamped to at least 1e-12 for numerical stability.
fn compute_p_matrix(x: &[f64], n: usize, n_features: usize, perplexity: f64) -> Vec<f64> {
    // Compute pairwise squared Euclidean distances
    let mut dist_sq = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let mut s = 0.0;
            for k in 0..n_features {
                let v = x[i * n_features + k] - x[j * n_features + k];
                s += v * v;
            }
            dist_sq[i * n + j] = s;
            dist_sq[j * n + i] = s;
        }
    }

    // Compute conditional probabilities for each row
    let mut p_cond = vec![0.0f64; n * n];
    for i in 0..n {
        let row_sq = &dist_sq[i * n..(i + 1) * n];
        let p_row = p_row_binary_search(row_sq, i, perplexity, 200, 1e-5);
        p_cond[i * n..(i + 1) * n].copy_from_slice(&p_row);
    }

    // Symmetrise: p_ij = (p(j|i) + p(i|j)) / (2n), clamped to >= 1e-12
    let denom = 2.0 * n as f64;
    let mut p = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let val = (p_cond[i * n + j] + p_cond[j * n + i]) / denom;
            p[i * n + j] = val.max(1e-12);
        }
    }
    p
}

// ─────────────────────────────────────────────────────────────────────────────
// Training
// ─────────────────────────────────────────────────────────────────────────────

/// Fisher-Yates shuffle of index array using LcgRng.
fn shuffle_indices(indices: &mut [usize], rng: &mut LcgRng) {
    let n = indices.len();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        indices.swap(i, j);
    }
}

/// Compute the batch-local KL gradient with respect to the embedding outputs `y_batch`
/// (shape `batch × n_components`). Returns gradient w.r.t. each y_i of the same shape.
///
/// Uses P re-normalised within the batch and Q batch-local Student-t affinities.
fn batch_kl_gradient(
    y_batch: &[f64],
    p_full: &[f64],
    batch_indices: &[usize],
    n_total: usize,
    n_components: usize,
) -> Vec<f64> {
    let batch = batch_indices.len();
    // Build Q numerators: (1 + ||y_i - y_j||^2)^-1 and sum for normalisation
    let mut q_num = vec![0.0f64; batch * batch];
    let mut q_denom = 0.0f64;
    for i in 0..batch {
        for j in 0..batch {
            if i == j {
                continue;
            }
            let mut sq_dist = 0.0;
            for k in 0..n_components {
                let v = y_batch[i * n_components + k] - y_batch[j * n_components + k];
                sq_dist += v * v;
            }
            let qij = 1.0 / (1.0 + sq_dist);
            q_num[i * batch + j] = qij;
            q_denom += qij;
        }
    }
    if q_denom < 1e-300 {
        q_denom = 1.0;
    }

    // Build batch-local P re-normalisation: sum p_ij for all i!=j in batch
    let mut p_batch_sum = 0.0f64;
    for i in 0..batch {
        for j in 0..batch {
            if i != j {
                let gi = batch_indices[i];
                let gj = batch_indices[j];
                p_batch_sum += p_full[gi * n_total + gj];
            }
        }
    }
    if p_batch_sum < 1e-300 {
        p_batch_sum = 1.0;
    }

    // Gradient: dC/dy_i = 4 Σ_{j≠i} (p_ij_batch - q_ij) (y_i - y_j) q_num_ij
    let mut grad = vec![0.0f64; batch * n_components];
    for i in 0..batch {
        for j in 0..batch {
            if i == j {
                continue;
            }
            let gi = batch_indices[i];
            let gj = batch_indices[j];
            let p_ij_batch = p_full[gi * n_total + gj] / p_batch_sum;
            let q_ij = q_num[i * batch + j] / q_denom;
            let factor = 4.0 * (p_ij_batch - q_ij) * q_num[i * batch + j];
            for k in 0..n_components {
                let dy = y_batch[i * n_components + k] - y_batch[j * n_components + k];
                grad[i * n_components + k] += factor * dy;
            }
        }
    }
    grad
}

/// Backpropagate the output gradient through the MLP to obtain weight/bias gradients.
///
/// Returns `(weight_grads, bias_grads)` matching the shapes of `model.weights/biases`.
fn mlp_backward(
    model: &ParametricTsneModel,
    pre_acts: &[Vec<f64>],
    activations: &[Vec<f64>],
    output_grad: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_layers = model.weights.len();
    let mut w_grads: Vec<Vec<f64>> = model
        .weights
        .iter()
        .map(|w| vec![0.0f64; w.len()])
        .collect();
    let mut b_grads: Vec<Vec<f64>> = model.biases.iter().map(|b| vec![0.0f64; b.len()]).collect();

    // delta = gradient flowing into this layer's output (before activation)
    let mut delta = output_grad.to_vec();

    for l in (0..n_layers).rev() {
        let in_dim = model.layer_dims[l];
        let out_dim = model.layer_dims[l + 1];
        let pre = &pre_acts[l];
        let act_in = &activations[l]; // input activations to layer l

        // Apply activation derivative for hidden layers (ReLU: 1 if z>0 else 0)
        // The final layer is linear, so delta passes through unchanged
        if l + 1 < n_layers {
            for (dv, &z) in delta.iter_mut().zip(pre.iter()) {
                *dv *= if z > 0.0 { 1.0 } else { 0.0 };
            }
        }

        // Accumulate weight gradients: dL/dW[o,i] = delta[o] * act_in[i]
        for o in 0..out_dim {
            b_grads[l][o] += delta[o];
            for i in 0..in_dim {
                w_grads[l][o * in_dim + i] += delta[o] * act_in[i];
            }
        }

        // Propagate delta to previous layer: delta_prev[i] = Σ_o W[o,i] * delta[o]
        if l > 0 {
            let mut delta_prev = vec![0.0f64; in_dim];
            let w = &model.weights[l];
            for i in 0..in_dim {
                let mut acc = 0.0;
                for o in 0..out_dim {
                    acc += w[o * in_dim + i] * delta[o];
                }
                delta_prev[i] = acc;
            }
            delta = delta_prev;
        }
    }
    (w_grads, b_grads)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a parametric t-SNE model on input data.
///
/// # Arguments
/// * `x`          — row-major input data of shape `[n_samples × n_features]`
/// * `n_samples`  — number of data points
/// * `n_features` — input dimensionality
/// * `config`     — training configuration
///
/// # Errors
/// Returns `InvalidParameter` if validation constraints are violated.
pub fn parametric_tsne_fit(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    config: &ParametricTsneConfig,
) -> ManifoldResult<ParametricTsneModel> {
    // ── Input validation ─────────────────────────────────────────────────────
    if n_samples < 10 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_samples".into(),
            reason: "must be >= 10 for meaningful t-SNE".into(),
        });
    }
    if config.hidden_dims.is_empty() {
        return Err(ManifoldError::InvalidParameter {
            name: "hidden_dims".into(),
            reason: "must have at least one hidden layer".into(),
        });
    }
    if config.n_components == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be >= 1".into(),
        });
    }
    if config.perplexity >= n_samples as f64 {
        return Err(ManifoldError::InvalidParameter {
            name: "perplexity".into(),
            reason: format!(
                "perplexity ({}) must be < n_samples ({})",
                config.perplexity, n_samples
            ),
        });
    }
    if x.len() != n_samples * n_features {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_features],
            got: vec![x.len()],
        });
    }

    let mut rng = LcgRng::new(config.seed);

    // ── Build layer dimension array ──────────────────────────────────────────
    let mut layer_dims = Vec::with_capacity(config.hidden_dims.len() + 2);
    layer_dims.push(n_features);
    layer_dims.extend_from_slice(&config.hidden_dims);
    layer_dims.push(config.n_components);
    let n_layers = layer_dims.len() - 1;

    // ── Kaiming-uniform weight initialisation ────────────────────────────────
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
    let mut biases: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let in_dim = layer_dims[l];
        let out_dim = layer_dims[l + 1];
        weights.push(kaiming_uniform_init(out_dim, in_dim, &mut rng));
        biases.push(vec![0.0f64; out_dim]);
    }

    // ── Compute the global joint affinity matrix P ───────────────────────────
    let p_global = compute_p_matrix(x, n_samples, n_features, config.perplexity);

    // ── Adam moment buffers ──────────────────────────────────────────────────
    let mut m_w: Vec<Vec<f64>> = weights.iter().map(|w| vec![0.0f64; w.len()]).collect();
    let mut v_w: Vec<Vec<f64>> = weights.iter().map(|w| vec![0.0f64; w.len()]).collect();
    let mut m_b: Vec<Vec<f64>> = biases.iter().map(|b| vec![0.0f64; b.len()]).collect();
    let mut v_b: Vec<Vec<f64>> = biases.iter().map(|b| vec![0.0f64; b.len()]).collect();

    let beta1 = config.beta1;
    let beta2 = config.beta2;
    let eps = config.adam_eps;
    let lr = config.learning_rate;

    // Build mutable model for training (update in-place)
    let mut model = ParametricTsneModel {
        weights,
        biases,
        layer_dims: layer_dims.clone(),
        config: config.clone(),
    };

    // Effective batch size: clamp to n_samples, at least 2
    let batch_size = config.batch_size.min(n_samples).max(2);
    let mut adam_t = 0u64;

    // Index array for shuffling
    let mut indices: Vec<usize> = (0..n_samples).collect();

    for epoch in 0..config.n_epochs {
        // Early exaggeration: multiply P entries by factor for first epochs
        let exag = if epoch < config.early_exaggeration_epochs {
            config.early_exaggeration
        } else {
            1.0
        };

        // Shuffle data indices each epoch
        shuffle_indices(&mut indices, &mut rng);

        // Process minibatches
        let mut batch_start = 0;
        while batch_start < n_samples {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let actual_batch = batch_end - batch_start;
            if actual_batch < 2 {
                // Need at least 2 points for meaningful KL gradient
                batch_start = batch_end;
                continue;
            }

            let batch_indices: Vec<usize> = indices[batch_start..batch_end].to_vec();

            // Forward pass for each sample in batch
            let mut y_batch = vec![0.0f64; actual_batch * config.n_components];
            let mut caches: Vec<MlpCache> = Vec::with_capacity(actual_batch);

            for (bi, &gi) in batch_indices.iter().enumerate() {
                let xi = &x[gi * n_features..(gi + 1) * n_features];
                let (pre_acts, activations) = mlp_forward_with_cache(&model, xi);
                // Last activation is the output embedding
                if let Some(y_i) = activations.last() {
                    y_batch[bi * config.n_components..(bi + 1) * config.n_components]
                        .copy_from_slice(y_i);
                }
                caches.push((pre_acts, activations));
            }

            // Compute gradient of KL w.r.t. each y_i (batch-local)
            // Apply exaggeration by scaling relevant P entries
            let grad_y_batch = if (exag - 1.0).abs() > 1e-10 {
                let mut p_exag = p_global.clone();
                for ii in &batch_indices {
                    for jj in &batch_indices {
                        let idx = ii * n_samples + jj;
                        p_exag[idx] *= exag;
                    }
                }
                batch_kl_gradient(
                    &y_batch,
                    &p_exag,
                    &batch_indices,
                    n_samples,
                    config.n_components,
                )
            } else {
                batch_kl_gradient(
                    &y_batch,
                    &p_global,
                    &batch_indices,
                    n_samples,
                    config.n_components,
                )
            };

            adam_t += 1;
            let t = adam_t as f64;
            let bias_corr1 = 1.0 - beta1.powf(t);
            let bias_corr2 = 1.0 - beta2.powf(t);

            // Accumulate gradients across batch samples via backprop
            let mut agg_w_grads: Vec<Vec<f64>> = model
                .weights
                .iter()
                .map(|w| vec![0.0f64; w.len()])
                .collect();
            let mut agg_b_grads: Vec<Vec<f64>> =
                model.biases.iter().map(|b| vec![0.0f64; b.len()]).collect();

            for (bi, (pre_acts, activations)) in caches.iter().enumerate() {
                let output_grad =
                    &grad_y_batch[bi * config.n_components..(bi + 1) * config.n_components];
                let (wg, bg) = mlp_backward(&model, pre_acts, activations, output_grad);
                for l in 0..n_layers {
                    for (ag, g) in agg_w_grads[l].iter_mut().zip(wg[l].iter()) {
                        *ag += g;
                    }
                    for (ag, g) in agg_b_grads[l].iter_mut().zip(bg[l].iter()) {
                        *ag += g;
                    }
                }
            }

            // Scale gradients by 1/batch_size
            let inv_batch = 1.0 / actual_batch as f64;
            for l in 0..n_layers {
                for g in agg_w_grads[l].iter_mut() {
                    *g *= inv_batch;
                }
                for g in agg_b_grads[l].iter_mut() {
                    *g *= inv_batch;
                }
            }

            // Adam update for weights
            for l in 0..n_layers {
                let n_w = model.weights[l].len();
                for idx in 0..n_w {
                    let g = agg_w_grads[l][idx];
                    m_w[l][idx] = beta1 * m_w[l][idx] + (1.0 - beta1) * g;
                    v_w[l][idx] = beta2 * v_w[l][idx] + (1.0 - beta2) * g * g;
                    let m_hat = m_w[l][idx] / bias_corr1;
                    let v_hat = v_w[l][idx] / bias_corr2;
                    model.weights[l][idx] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }

            // Adam update for biases
            for l in 0..n_layers {
                let n_b = model.biases[l].len();
                for idx in 0..n_b {
                    let g = agg_b_grads[l][idx];
                    m_b[l][idx] = beta1 * m_b[l][idx] + (1.0 - beta1) * g;
                    v_b[l][idx] = beta2 * v_b[l][idx] + (1.0 - beta2) * g * g;
                    let m_hat = m_b[l][idx] / bias_corr1;
                    let v_hat = v_b[l][idx] / bias_corr2;
                    model.biases[l][idx] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }

            batch_start = batch_end;
        }
    }

    Ok(model)
}

/// Apply a trained parametric t-SNE model to new data.
///
/// # Arguments
/// * `model`     — trained model from [`parametric_tsne_fit`]
/// * `x`         — row-major data of shape `[n_samples × n_features]`
/// * `n_samples` — number of data points to transform
///
/// # Returns
/// Embedding of shape `[n_samples × n_components]`, row-major.
pub fn parametric_tsne_transform(
    model: &ParametricTsneModel,
    x: &[f64],
    n_samples: usize,
) -> ManifoldResult<Vec<f64>> {
    let n_features = model.layer_dims[0];
    if x.len() != n_samples * n_features {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_features],
            got: vec![x.len()],
        });
    }
    let n_out = model.config.n_components;
    let mut out = vec![0.0f64; n_samples * n_out];
    for i in 0..n_samples {
        let xi = &x[i * n_features..(i + 1) * n_features];
        let yi = parametric_tsne_forward(model, xi);
        out[i * n_out..(i + 1) * n_out].copy_from_slice(&yi);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate Gaussian cluster data with LcgRng.
    /// Returns `n_clusters * pts_per_cluster` points in d dimensions.
    fn make_clusters(
        n_clusters: usize,
        pts_per_cluster: usize,
        d: usize,
        scale: f64,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        let n = n_clusters * pts_per_cluster;
        let mut data = vec![0.0f64; n * d];
        for c in 0..n_clusters {
            // Cluster centre: each cluster shifted far apart
            let center: Vec<f64> = (0..d).map(|_k| c as f64 * 10.0).collect();
            for p in 0..pts_per_cluster {
                let idx = (c * pts_per_cluster + p) * d;
                for k in 0..d {
                    data[idx + k] = center[k] + rng.next_normal() * scale;
                }
            }
        }
        data
    }

    /// Compute within-cluster variance of the embedding.
    fn within_cluster_var(embed: &[f64], n_samples: usize, n_out: usize, n_clusters: usize) -> f64 {
        let pts_per = n_samples / n_clusters;
        let mut total_var = 0.0;
        for c in 0..n_clusters {
            let mut center = vec![0.0f64; n_out];
            for p in 0..pts_per {
                let idx = (c * pts_per + p) * n_out;
                for k in 0..n_out {
                    center[k] += embed[idx + k];
                }
            }
            for cv in center.iter_mut() {
                *cv /= pts_per as f64;
            }
            for p in 0..pts_per {
                let idx = (c * pts_per + p) * n_out;
                for k in 0..n_out {
                    let v = embed[idx + k] - center[k];
                    total_var += v * v;
                }
            }
        }
        total_var / n_samples as f64
    }

    /// Compute between-cluster variance of the embedding.
    fn between_cluster_var(
        embed: &[f64],
        n_samples: usize,
        n_out: usize,
        n_clusters: usize,
    ) -> f64 {
        let pts_per = n_samples / n_clusters;
        let mut centers = vec![0.0f64; n_clusters * n_out];
        for c in 0..n_clusters {
            for p in 0..pts_per {
                let idx = (c * pts_per + p) * n_out;
                for k in 0..n_out {
                    centers[c * n_out + k] += embed[idx + k];
                }
            }
            for k in 0..n_out {
                centers[c * n_out + k] /= pts_per as f64;
            }
        }
        let mut global = vec![0.0f64; n_out];
        for c in 0..n_clusters {
            for k in 0..n_out {
                global[k] += centers[c * n_out + k];
            }
        }
        for gv in global.iter_mut() {
            *gv /= n_clusters as f64;
        }
        let mut bvar = 0.0;
        for c in 0..n_clusters {
            for k in 0..n_out {
                let v = centers[c * n_out + k] - global[k];
                bvar += v * v * pts_per as f64;
            }
        }
        bvar / n_samples as f64
    }

    /// Small config for fast tests.
    fn fast_config(seed: u64) -> ParametricTsneConfig {
        ParametricTsneConfig {
            n_components: 2,
            perplexity: 5.0,
            hidden_dims: vec![32, 16],
            learning_rate: 1e-3,
            n_epochs: 30,
            batch_size: 20,
            early_exaggeration: 4.0,
            early_exaggeration_epochs: 10,
            beta1: 0.9,
            beta2: 0.999,
            adam_eps: 1e-8,
            seed,
        }
    }

    // ── Test 1: output shape is correct ─────────────────────────────────────
    #[test]
    fn transform_output_shape() {
        let n = 20;
        let d = 4;
        let data = make_clusters(2, 10, d, 0.5, 1);
        let cfg = fast_config(1);
        let model = parametric_tsne_fit(&data, n, d, &cfg).expect("fit ok");
        let emb = parametric_tsne_transform(&model, &data, n).expect("transform ok");
        assert_eq!(emb.len(), n * cfg.n_components);
    }

    // ── Test 2: forward output dimension matches n_components ────────────────
    #[test]
    fn forward_output_dim() {
        let d = 4;
        let data = make_clusters(2, 10, d, 0.5, 2);
        let cfg = fast_config(2);
        let model = parametric_tsne_fit(&data, 20, d, &cfg).expect("fit ok");
        let x0 = &data[..d];
        let y = parametric_tsne_forward(&model, x0);
        assert_eq!(y.len(), cfg.n_components);
    }

    // ── Test 3: seed reproducibility ─────────────────────────────────────────
    #[test]
    fn seed_reproducibility() {
        let n = 20;
        let d = 4;
        let data = make_clusters(2, 10, d, 0.5, 42);
        let cfg = fast_config(99);
        let m1 = parametric_tsne_fit(&data, n, d, &cfg).expect("fit1 ok");
        let m2 = parametric_tsne_fit(&data, n, d, &cfg).expect("fit2 ok");
        assert_eq!(
            m1.weights[0][0], m2.weights[0][0],
            "Seed reproducibility: first weight element must match"
        );
        assert_eq!(
            m1.weights[0].len(),
            m2.weights[0].len(),
            "Weight shapes must match"
        );
    }

    // ── Test 4: cluster separation (between/within variance ratio > 1) ────────
    #[test]
    fn cluster_separation() {
        let n_clusters = 3;
        let pts = 20;
        let n = n_clusters * pts;
        let d = 4;
        let data = make_clusters(n_clusters, pts, d, 0.3, 7);
        let cfg = ParametricTsneConfig {
            n_components: 2,
            perplexity: 5.0,
            hidden_dims: vec![32],
            learning_rate: 1e-3,
            n_epochs: 40,
            batch_size: 20,
            early_exaggeration: 4.0,
            early_exaggeration_epochs: 10,
            beta1: 0.9,
            beta2: 0.999,
            adam_eps: 1e-8,
            seed: 77,
        };
        let model = parametric_tsne_fit(&data, n, d, &cfg).expect("fit ok");
        let emb = parametric_tsne_transform(&model, &data, n).expect("transform ok");
        let bv = between_cluster_var(&emb, n, 2, n_clusters);
        let wv = within_cluster_var(&emb, n, 2, n_clusters);
        assert!(
            bv > wv,
            "Cluster separation failed: between_var={bv:.4}, within_var={wv:.4}"
        );
    }

    // ── Test 5: validation — n_samples < 10 → InvalidParameter ──────────────
    #[test]
    fn validation_n_samples_too_small() {
        let data = vec![1.0f64; 9 * 4];
        let cfg = fast_config(1);
        let res = parametric_tsne_fit(&data, 9, 4, &cfg);
        assert!(res.is_err());
        match res {
            Err(ManifoldError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "n_samples");
            }
            other => panic!("Expected InvalidParameter, got {other:?}"),
        }
    }

    // ── Test 6: validation — empty hidden_dims → InvalidParameter ────────────
    #[test]
    fn validation_empty_hidden_dims() {
        let data = vec![1.0f64; 20 * 4];
        let cfg = ParametricTsneConfig {
            hidden_dims: vec![],
            ..fast_config(1)
        };
        let res = parametric_tsne_fit(&data, 20, 4, &cfg);
        assert!(res.is_err());
        match res {
            Err(ManifoldError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "hidden_dims");
            }
            other => panic!("Expected InvalidParameter, got {other:?}"),
        }
    }

    // ── Test 7: validation — n_components=0 → InvalidParameter ──────────────
    #[test]
    fn validation_n_components_zero() {
        let data = vec![1.0f64; 20 * 4];
        let cfg = ParametricTsneConfig {
            n_components: 0,
            ..fast_config(1)
        };
        let res = parametric_tsne_fit(&data, 20, 4, &cfg);
        assert!(res.is_err());
        match res {
            Err(ManifoldError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "n_components");
            }
            other => panic!("Expected InvalidParameter, got {other:?}"),
        }
    }

    // ── Test 8: validation — perplexity >= n_samples → InvalidParameter ──────
    #[test]
    fn validation_perplexity_too_large() {
        let data = vec![1.0f64; 20 * 4];
        let cfg = ParametricTsneConfig {
            perplexity: 20.0,
            ..fast_config(1)
        };
        let res = parametric_tsne_fit(&data, 20, 4, &cfg);
        assert!(res.is_err());
        match res {
            Err(ManifoldError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "perplexity");
            }
            other => panic!("Expected InvalidParameter, got {other:?}"),
        }
    }

    // ── Test 9: model layer_dims are correct ─────────────────────────────────
    #[test]
    fn layer_dims_correct() {
        let n = 20;
        let d = 6;
        let data = make_clusters(2, 10, d, 0.5, 3);
        let cfg = ParametricTsneConfig {
            n_components: 3,
            hidden_dims: vec![64, 32],
            perplexity: 5.0,
            ..fast_config(3)
        };
        let model = parametric_tsne_fit(&data, n, d, &cfg).expect("fit ok");
        assert_eq!(model.layer_dims, vec![6, 64, 32, 3]);
    }

    // ── Test 10: weight/bias shapes match layer_dims ─────────────────────────
    #[test]
    fn weight_bias_shapes() {
        let n = 20;
        let d = 4;
        let data = make_clusters(2, 10, d, 0.5, 4);
        let cfg = ParametricTsneConfig {
            hidden_dims: vec![16, 8],
            n_components: 2,
            ..fast_config(4)
        };
        let model = parametric_tsne_fit(&data, n, d, &cfg).expect("fit ok");
        assert_eq!(model.weights[0].len(), 16 * 4);
        assert_eq!(model.biases[0].len(), 16);
        assert_eq!(model.weights[1].len(), 8 * 16);
        assert_eq!(model.biases[1].len(), 8);
        assert_eq!(model.weights[2].len(), 2 * 8);
        assert_eq!(model.biases[2].len(), 2);
    }

    // ── Test 11: transform with wrong feature count returns error ────────────
    #[test]
    fn transform_shape_mismatch() {
        let n = 20;
        let d = 4;
        let data = make_clusters(2, 10, d, 0.5, 5);
        let cfg = fast_config(5);
        let model = parametric_tsne_fit(&data, n, d, &cfg).expect("fit ok");
        let wrong_data = vec![1.0f64; 20 * 3]; // wrong d=3
        let res = parametric_tsne_transform(&model, &wrong_data, 20);
        assert!(res.is_err());
    }

    // ── Test 12: P matrix is symmetric ────────────────────────────────────────
    #[test]
    fn p_matrix_symmetric() {
        let n = 15;
        let d = 3;
        let data = make_clusters(3, 5, d, 0.5, 6);
        let p = compute_p_matrix(&data, n, d, 4.0);
        for i in 0..n {
            for j in 0..n {
                let diff = (p[i * n + j] - p[j * n + i]).abs();
                assert!(diff < 1e-10, "P not symmetric at ({i},{j}): {diff}");
            }
        }
    }

    // ── Test 13: P matrix entries are non-negative ────────────────────────────
    #[test]
    fn p_matrix_nonneg() {
        let n = 12;
        let d = 3;
        let data = make_clusters(3, 4, d, 0.5, 13);
        let p = compute_p_matrix(&data, n, d, 3.0);
        for &v in &p {
            assert!(v >= 1e-12, "P entry below minimum: {v}");
        }
    }

    // ── Test 14: forward pass is deterministic ────────────────────────────────
    #[test]
    fn forward_deterministic() {
        let n = 20;
        let d = 4;
        let data = make_clusters(2, 10, d, 0.5, 14);
        let cfg = fast_config(14);
        let model = parametric_tsne_fit(&data, n, d, &cfg).expect("fit ok");
        let x0 = &data[..d];
        let y1 = parametric_tsne_forward(&model, x0);
        let y2 = parametric_tsne_forward(&model, x0);
        for (a, b) in y1.iter().zip(y2.iter()) {
            assert_eq!(*a, *b, "Forward pass not deterministic");
        }
    }

    // ── Test 15: different seeds produce different initial weights ────────────
    #[test]
    fn different_seeds_different_weights() {
        let n = 20;
        let d = 4;
        let data = make_clusters(2, 10, d, 0.5, 99);
        let cfg_a = fast_config(1);
        let cfg_b = fast_config(2);
        let m_a = parametric_tsne_fit(&data, n, d, &cfg_a).expect("ok");
        let m_b = parametric_tsne_fit(&data, n, d, &cfg_b).expect("ok");
        let diff: f64 = m_a.weights[0]
            .iter()
            .zip(m_b.weights[0].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-10, "Different seeds yielded identical weights");
    }

    // ── Test 16: biases update after training ─────────────────────────────────
    #[test]
    fn biases_update_after_training() {
        let n = 20;
        let d = 4;
        let data = make_clusters(2, 10, d, 0.5, 16);
        let cfg = ParametricTsneConfig {
            n_epochs: 5,
            ..fast_config(16)
        };
        let model = parametric_tsne_fit(&data, n, d, &cfg).expect("fit ok");
        let any_nonzero = model
            .biases
            .iter()
            .any(|b| b.iter().any(|&v| v.abs() > 1e-15));
        assert!(any_nonzero, "All biases remain zero after training");
    }

    // ── Test 17: Kaiming uniform bound is respected ───────────────────────────
    #[test]
    fn kaiming_bounds_respected() {
        let out_dim = 16;
        let in_dim = 32;
        let mut rng = LcgRng::new(17);
        let w = kaiming_uniform_init(out_dim, in_dim, &mut rng);
        let bound = (6.0_f64 / in_dim as f64).sqrt();
        for &v in &w {
            assert!(
                v >= -bound - 1e-12 && v <= bound + 1e-12,
                "Weight {v} outside Kaiming bounds ±{bound}"
            );
        }
    }

    // ── Test 18: no hidden layer is rejected ──────────────────────────────────
    #[test]
    fn no_hidden_layer_rejected() {
        let data = vec![1.0f64; 20 * 4];
        let cfg = ParametricTsneConfig {
            hidden_dims: vec![],
            ..fast_config(1)
        };
        assert!(parametric_tsne_fit(&data, 20, 4, &cfg).is_err());
    }

    // ── Test 19: batch KL gradient shape is correct ───────────────────────────
    #[test]
    fn batch_kl_gradient_shape() {
        let n_total = 20;
        let n_comp = 2;
        let batch_sz = 5;
        let mut rng = LcgRng::new(19);
        let y: Vec<f64> = (0..batch_sz * n_comp).map(|_| rng.next_normal()).collect();
        let p: Vec<f64> = (0..n_total * n_total)
            .map(|_| rng.next_f64() / (n_total * n_total) as f64)
            .collect();
        let batch_indices: Vec<usize> = (0..batch_sz).collect();
        let grad = batch_kl_gradient(&y, &p, &batch_indices, n_total, n_comp);
        assert_eq!(grad.len(), batch_sz * n_comp);
    }

    // ── Test 20: transform output is finite ───────────────────────────────────
    #[test]
    fn transform_output_finite() {
        let n = 20;
        let d = 4;
        let data = make_clusters(2, 10, d, 0.5, 20);
        let cfg = fast_config(20);
        let model = parametric_tsne_fit(&data, n, d, &cfg).expect("fit ok");
        let emb = parametric_tsne_transform(&model, &data, n).expect("transform ok");
        for &v in &emb {
            assert!(v.is_finite(), "Non-finite value in embedding: {v}");
        }
    }

    // ── Test 21: small perplexity still trains without panic ─────────────────
    #[test]
    fn small_perplexity_trains() {
        let n = 12;
        let d = 3;
        let data = make_clusters(2, 6, d, 0.5, 21);
        let cfg = ParametricTsneConfig {
            perplexity: 2.0,
            n_epochs: 5,
            ..fast_config(21)
        };
        let res = parametric_tsne_fit(&data, n, d, &cfg);
        assert!(
            res.is_ok(),
            "Training with small perplexity failed: {res:?}"
        );
    }

    // ── Test 22: model stored config matches input config ─────────────────────
    #[test]
    fn stored_config_matches() {
        let n = 20;
        let d = 4;
        let data = make_clusters(2, 10, d, 0.5, 22);
        let cfg = ParametricTsneConfig {
            n_epochs: 7,
            batch_size: 10,
            perplexity: 5.0,
            hidden_dims: vec![24, 12],
            ..Default::default()
        };
        let model = parametric_tsne_fit(&data, n, d, &cfg).expect("fit ok");
        assert_eq!(model.config.n_epochs, 7);
        assert_eq!(model.config.batch_size, 10);
        assert_eq!(model.config.hidden_dims, vec![24, 12]);
    }
}

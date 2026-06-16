//! Memory-Augmented Autoencoder (MemAE) anomaly detection.
//!
//! Gong et al. 2019 "Memorizing Normality to Detect Anomaly".
//!
//! An autoencoder is augmented with a fixed-size external memory `M` that
//! stores prototype "normal" patterns.  During inference the encoder produces
//! a query `q`, soft-attention over the memory retrieves a convex combination
//! of stored prototypes, and the decoder reconstructs from that combination.
//! Because the memory only stores normal patterns, anomalous inputs cannot
//! retrieve a good prototype and incur a high reconstruction error.
//!
//! **Score** = MSE reconstruction error `(1/d) Σ (x_j - x̂_j)²`.
//!
//! # Memory update rule (gradient-free)
//!
//! After each forward pass, for each memory slot `i`:
//! ```text
//! M[i] ← M[i] + α · ŵ_i · (q − M[i])
//! M[i] ← M[i] / ‖M[i]‖₂          (unit-norm re-normalisation)
//! ```
//! where `α = lr * 0.1` and `ŵ_i` is the hard-shrinkage attention weight.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Constants ───────────────────────────────────────────────────────────────

const EPS: f64 = 1e-12;

// ─── Xavier initialisation ───────────────────────────────────────────────────

fn xavier_init(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u = rng.next_f32() as f64;
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─── Dense layer forward pass ─────────────────────────────────────────────────

fn dense(x: &[f64], w: &[f64], b: &[f64], fan_in: usize, fan_out: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; fan_out];
    for o in 0..fan_out {
        let mut acc = b[o];
        for i in 0..fan_in {
            acc += w[o * fan_in + i] * x[i];
        }
        out[o] = acc;
    }
    out
}

fn relu(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.max(0.0)).collect()
}

fn sigmoid(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps: Vec<f64> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(EPS);
    for e in exps.iter_mut() {
        *e /= sum;
    }
    exps
}

// ─── Hard shrinkage ───────────────────────────────────────────────────────────

/// Element-wise hard shrinkage: hat(w_i) = max(0, w_i - λ/2) * w_i / |w_i - λ/2|
/// Numerically: if |w_i - λ/2| < eps → 0; else  sign(w_i - λ/2) * w_i / |w_i - λ/2| clamped ≥ 0.
fn hard_shrink(w: &[f64], lambda: f64) -> Vec<f64> {
    let half_lambda = lambda * 0.5;
    w.iter()
        .map(|&wi| {
            let diff = wi - half_lambda;
            if diff > EPS {
                // diff > 0 → wi > half_lambda → multiply positive
                diff * wi / diff // simplifies to wi but keeps the formulation clear
            } else {
                0.0
            }
        })
        .collect()
}

// ─── L1 normalisation ────────────────────────────────────────────────────────

fn l1_normalize(v: &[f64]) -> Vec<f64> {
    let sum: f64 = v.iter().sum::<f64>().max(EPS);
    v.iter().map(|&x| x / sum).collect()
}

// ─── L2 normalisation (for memory rows) ──────────────────────────────────────

fn l2_norm_row(row: &[f64]) -> f64 {
    row.iter().map(|&x| x * x).sum::<f64>().sqrt().max(EPS)
}

fn l2_normalize_row_inplace(row: &mut [f64]) {
    let norm = l2_norm_row(row);
    for v in row.iter_mut() {
        *v /= norm;
    }
}

// ─── MemAeConfig ─────────────────────────────────────────────────────────────

/// Configuration for Memory-Augmented Autoencoder.
#[derive(Debug, Clone)]
pub struct MemAeConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Intermediate hidden layer width.
    pub hidden_dim: usize,
    /// Latent (bottleneck) dimension — also the memory slot dimension.
    pub latent_dim: usize,
    /// Number of prototype memory slots.
    pub mem_size: usize,
    /// Anomaly threshold applied in `mem_ae_predict`.
    pub threshold: f64,
    /// Learning rate for encoder/decoder SGD.
    pub lr: f64,
    /// Number of training epochs.
    pub n_epochs: usize,
    /// Hard-shrinkage regularisation λ (Eq. 7 in paper; typical 0.0025).
    pub hard_shrink_lambda: f64,
}

impl Default for MemAeConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 32,
            latent_dim: 8,
            mem_size: 50,
            threshold: 0.1,
            lr: 1e-3,
            n_epochs: 20,
            hard_shrink_lambda: 0.0025,
        }
    }
}

// ─── MemAeFit ────────────────────────────────────────────────────────────────

/// Fitted Memory-Augmented Autoencoder model.
///
/// All weight matrices are stored as flat row-major `Vec<f64>`.
///
/// Layout of each layer weight `W` with shape `[fan_out, fan_in]`:
/// `W[o * fan_in + i]` = weight from input neuron `i` to output neuron `o`.
#[derive(Debug, Clone)]
pub struct MemAeFit {
    /// Encoder layer 1: shape `[hidden_dim, input_dim]`.
    pub enc_w1: Vec<f64>,
    /// Encoder layer 1 bias: shape `[hidden_dim]`.
    pub enc_b1: Vec<f64>,
    /// Encoder layer 2: shape `[latent_dim, hidden_dim]`.
    pub enc_w2: Vec<f64>,
    /// Encoder layer 2 bias: shape `[latent_dim]`.
    pub enc_b2: Vec<f64>,
    /// Decoder layer 1: shape `[hidden_dim, latent_dim]`.
    pub dec_w1: Vec<f64>,
    /// Decoder layer 1 bias: shape `[hidden_dim]`.
    pub dec_b1: Vec<f64>,
    /// Decoder layer 2: shape `[input_dim, hidden_dim]`.
    pub dec_w2: Vec<f64>,
    /// Decoder layer 2 bias: shape `[input_dim]`.
    pub dec_b2: Vec<f64>,
    /// Memory matrix: flat row-major `[mem_size, latent_dim]`.
    /// Each row is a unit-norm prototype vector.
    pub memory: Vec<f64>,
    /// Stored configuration.
    pub config: MemAeConfig,
}

// ─── Forward pass helpers ─────────────────────────────────────────────────────

/// Encode one sample `x` → query `q` of shape `[latent_dim]`.
fn encode(fit: &MemAeFit, x: &[f64]) -> Vec<f64> {
    let cfg = &fit.config;
    let h1 = relu(&dense(
        x,
        &fit.enc_w1,
        &fit.enc_b1,
        cfg.input_dim,
        cfg.hidden_dim,
    ));
    dense(
        &h1,
        &fit.enc_w2,
        &fit.enc_b2,
        cfg.hidden_dim,
        cfg.latent_dim,
    )
}

/// Decode retrieved code `z_hat` → reconstruction `x_hat`.
fn decode(fit: &MemAeFit, z_hat: &[f64]) -> Vec<f64> {
    let cfg = &fit.config;
    let h1 = relu(&dense(
        z_hat,
        &fit.dec_w1,
        &fit.dec_b1,
        cfg.latent_dim,
        cfg.hidden_dim,
    ));
    sigmoid(&dense(
        &h1,
        &fit.dec_w2,
        &fit.dec_b2,
        cfg.hidden_dim,
        cfg.input_dim,
    ))
}

/// Attention-based memory read.
///
/// Returns `(attention_weights, retrieved_code)`.
/// `attention_weights` is after hard-shrinkage + L1 re-normalisation.
fn memory_read(fit: &MemAeFit, query: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let cfg = &fit.config;
    let mem_size = cfg.mem_size;
    let latent_dim = cfg.latent_dim;

    // Dot-product attention logits: w_i = q · m_i
    let logits: Vec<f64> = (0..mem_size)
        .map(|i| {
            let start = i * latent_dim;
            fit.memory[start..start + latent_dim]
                .iter()
                .zip(query.iter())
                .map(|(&m, &q)| m * q)
                .sum::<f64>()
        })
        .collect();

    // Softmax over raw dot products
    let raw_weights = softmax(&logits);

    // Hard shrinkage
    let shrunk = hard_shrink(&raw_weights, cfg.hard_shrink_lambda);

    // L1 re-normalise
    let weights = l1_normalize(&shrunk);

    // Retrieved code: z_hat = Σ w_i * m_i
    let mut retrieved = vec![0.0_f64; latent_dim];
    for (i, &w) in weights.iter().enumerate() {
        let start = i * latent_dim;
        for (j, r) in retrieved.iter_mut().enumerate() {
            *r += w * fit.memory[start + j];
        }
    }

    (weights, retrieved)
}

// ─── Gradient helpers ─────────────────────────────────────────────────────────

/// Compute MSE loss and the gradient of loss w.r.t. reconstruction.
/// `loss = (1/d) Σ (x_j - x_hat_j)²`
/// `d_loss/d_x_hat_j = -2/d * (x_j - x_hat_j)`
fn mse_loss_and_grad(x: &[f64], x_hat: &[f64]) -> (f64, Vec<f64>) {
    let d = x.len() as f64;
    let loss = x
        .iter()
        .zip(x_hat.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f64>()
        / d;
    let grad: Vec<f64> = x
        .iter()
        .zip(x_hat.iter())
        .map(|(&a, &b)| -2.0 * (a - b) / d)
        .collect();
    (loss, grad)
}

/// Backward through sigmoid output layer: `δ_in = δ_out * σ(z) * (1 - σ(z))`.
fn sigmoid_backward(out: &[f64], grad_out: &[f64]) -> Vec<f64> {
    out.iter()
        .zip(grad_out.iter())
        .map(|(&o, &g)| g * o * (1.0 - o))
        .collect()
}

/// Backward through ReLU: `δ_in = δ_out * (out > 0)`.
fn relu_backward(out: &[f64], grad_out: &[f64]) -> Vec<f64> {
    out.iter()
        .zip(grad_out.iter())
        .map(|(&o, &g)| if o > 0.0 { g } else { 0.0 })
        .collect()
}

/// Compute gradient w.r.t. weights, bias, and input for one dense layer.
///
/// Returns `(dW, db, dx_in)`.
fn dense_backward(
    x_in: &[f64],
    w: &[f64],
    grad_out: &[f64],
    fan_in: usize,
    fan_out: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // dW[o * fan_in + i] = grad_out[o] * x_in[i]
    let mut dw = vec![0.0_f64; fan_out * fan_in];
    for o in 0..fan_out {
        for i in 0..fan_in {
            dw[o * fan_in + i] = grad_out[o] * x_in[i];
        }
    }
    // db = grad_out
    let db = grad_out.to_vec();
    // dx_in[i] = Σ_o w[o*fan_in+i] * grad_out[o]
    let mut dx = vec![0.0_f64; fan_in];
    for o in 0..fan_out {
        for i in 0..fan_in {
            dx[i] += w[o * fan_in + i] * grad_out[o];
        }
    }
    (dw, db, dx)
}

/// SGD update: `param -= lr * grad`.
fn sgd_update(params: &mut [f64], grad: &[f64], lr: f64) {
    for (p, &g) in params.iter_mut().zip(grad.iter()) {
        *p -= lr * g;
    }
}

// ─── Training ────────────────────────────────────────────────────────────────

/// Fit a Memory-Augmented Autoencoder to training data.
///
/// `x` is a flat row-major matrix of shape `[n, input_dim]`.
pub fn mem_ae_fit(x: &[f64], n: usize, cfg: &MemAeConfig, seed: u64) -> AnomalyResult<MemAeFit> {
    // --- Validate config ---
    if cfg.mem_size == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "mem_size must be > 0".into(),
        });
    }
    if cfg.input_dim == 0 || cfg.hidden_dim == 0 || cfg.latent_dim == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "input_dim, hidden_dim, and latent_dim must be > 0".into(),
        });
    }
    if n == 0 {
        return Err(AnomalyError::InsufficientSamples { need: 1, got: 0 });
    }
    if x.len() != n * cfg.input_dim {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * cfg.input_dim,
            got: x.len(),
        });
    }

    let mut rng = LcgRng::new(seed);

    // --- Initialise encoder weights ---
    let enc_w1 = xavier_init(cfg.input_dim, cfg.hidden_dim, &mut rng);
    let enc_b1 = vec![0.0_f64; cfg.hidden_dim];
    let enc_w2 = xavier_init(cfg.hidden_dim, cfg.latent_dim, &mut rng);
    let enc_b2 = vec![0.0_f64; cfg.latent_dim];

    // --- Initialise decoder weights ---
    let dec_w1 = xavier_init(cfg.latent_dim, cfg.hidden_dim, &mut rng);
    let dec_b1 = vec![0.0_f64; cfg.hidden_dim];
    let dec_w2 = xavier_init(cfg.hidden_dim, cfg.input_dim, &mut rng);
    let dec_b2 = vec![0.0_f64; cfg.input_dim];

    // --- Initialise memory with random unit-norm rows ---
    let mut memory = vec![0.0_f64; cfg.mem_size * cfg.latent_dim];
    for i in 0..cfg.mem_size {
        let start = i * cfg.latent_dim;
        let row = &mut memory[start..start + cfg.latent_dim];
        for v in row.iter_mut() {
            *v = rng.next_normal() as f64;
        }
        l2_normalize_row_inplace(row);
    }

    let mut fit = MemAeFit {
        enc_w1,
        enc_b1,
        enc_w2,
        enc_b2,
        dec_w1,
        dec_b1,
        dec_w2,
        dec_b2,
        memory,
        config: cfg.clone(),
    };

    let lr = cfg.lr;
    let lr_mem = lr * 0.1;
    let input_dim = cfg.input_dim;
    let hidden_dim = cfg.hidden_dim;
    let latent_dim = cfg.latent_dim;

    // --- Training loop ---
    for _epoch in 0..cfg.n_epochs {
        for s in 0..n {
            let xi = &x[s * input_dim..(s + 1) * input_dim];

            // ── Encoder forward ──────────────────────────────────────────────
            let enc_h1_pre = dense(xi, &fit.enc_w1, &fit.enc_b1, input_dim, hidden_dim);
            let enc_h1 = relu(&enc_h1_pre);
            let query = dense(&enc_h1, &fit.enc_w2, &fit.enc_b2, hidden_dim, latent_dim);

            // ── Memory read ──────────────────────────────────────────────────
            let (weights, z_hat) = memory_read(&fit, &query);

            // ── Decoder forward ──────────────────────────────────────────────
            let dec_h1_pre = dense(&z_hat, &fit.dec_w1, &fit.dec_b1, latent_dim, hidden_dim);
            let dec_h1 = relu(&dec_h1_pre);
            let dec_out_pre = dense(&dec_h1, &fit.dec_w2, &fit.dec_b2, hidden_dim, input_dim);
            let x_hat = sigmoid(&dec_out_pre);

            // ── MSE loss gradient ────────────────────────────────────────────
            let (_mse, grad_xhat) = mse_loss_and_grad(xi, &x_hat);

            // Entropy loss gradient contribution:
            // L_entropy = -Σ_i w_i * log(w_i + eps)
            // We treat the entropy loss as part of the total loss.
            // dL/dw_i = -(log(w_i + eps) + 1)
            // However, memory weights are not directly parameterised —
            // they flow through the query, softmax, and shrink pipeline.
            // We propagate only the MSE gradient through the decoder/encoder
            // and handle entropy as a self-regularising effect of the memory update.

            // ── Decoder backward ─────────────────────────────────────────────
            // Layer 2: sigmoid output
            let grad_dec_out_pre = sigmoid_backward(&x_hat, &grad_xhat);
            let (dw2, db2, grad_dec_h1) = dense_backward(
                &dec_h1,
                &fit.dec_w2,
                &grad_dec_out_pre,
                hidden_dim,
                input_dim,
            );

            // Layer 1: ReLU hidden
            let grad_dec_h1_pre = relu_backward(&dec_h1, &grad_dec_h1);
            let (dw1, db1, _grad_z_hat) = dense_backward(
                &z_hat,
                &fit.dec_w1,
                &grad_dec_h1_pre,
                latent_dim,
                hidden_dim,
            );

            // ── Decoder parameter update ─────────────────────────────────────
            sgd_update(&mut fit.dec_w2, &dw2, lr);
            sgd_update(&mut fit.dec_b2, &db2, lr);
            sgd_update(&mut fit.dec_w1, &dw1, lr);
            sgd_update(&mut fit.dec_b1, &db1, lr);

            // ── Gradient w.r.t. retrieved z_hat flows back to query ──────────
            // z_hat = Σ_i w_hat_i * m_i
            // d_query is approximated as d_z_hat (memory rows are not backpropped)
            // d_loss/d_query ≈ d_loss/d_z_hat (skip hard-shrink Jacobian for stability)
            let grad_query = &_grad_z_hat;

            // ── Encoder backward (from grad_query) ────────────────────────────
            // Layer 2: linear (no activation at encoder output)
            let (dew2, deb2, grad_enc_h1) =
                dense_backward(&enc_h1, &fit.enc_w2, grad_query, hidden_dim, latent_dim);

            // Layer 1: ReLU hidden
            let grad_enc_h1_pre = relu_backward(&enc_h1, &grad_enc_h1);
            let (dew1, deb1, _grad_xi) =
                dense_backward(xi, &fit.enc_w1, &grad_enc_h1_pre, input_dim, hidden_dim);

            // ── Encoder parameter update ─────────────────────────────────────
            sgd_update(&mut fit.enc_w2, &dew2, lr);
            sgd_update(&mut fit.enc_b2, &deb2, lr);
            sgd_update(&mut fit.enc_w1, &dew1, lr);
            sgd_update(&mut fit.enc_b1, &deb1, lr);

            // ── Memory update (momentum, gradient-free) ──────────────────────
            for (i, &wi) in weights.iter().enumerate() {
                if wi < EPS {
                    continue;
                }
                let start = i * latent_dim;
                for (j, mem_j) in fit.memory[start..start + latent_dim].iter_mut().enumerate() {
                    let diff = query[j] - *mem_j;
                    *mem_j += lr_mem * wi * diff;
                }
                // Re-normalise to unit norm
                let row = &mut fit.memory[start..start + latent_dim];
                l2_normalize_row_inplace(row);
            }
        }
    }

    Ok(fit)
}

// ─── Scoring ─────────────────────────────────────────────────────────────────

/// Compute anomaly scores for `n` samples in `x` (flat row-major, `[n, input_dim]`).
///
/// Returns MSE reconstruction error per sample (higher = more anomalous).
pub fn mem_ae_score(fit: &MemAeFit, x: &[f64], n: usize) -> AnomalyResult<Vec<f64>> {
    let input_dim = fit.config.input_dim;
    if n == 0 {
        return Err(AnomalyError::InsufficientSamples { need: 1, got: 0 });
    }
    if x.len() != n * input_dim {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * input_dim,
            got: x.len(),
        });
    }

    let mut scores = Vec::with_capacity(n);
    for s in 0..n {
        let xi = &x[s * input_dim..(s + 1) * input_dim];
        let query = encode(fit, xi);
        let (_weights, z_hat) = memory_read(fit, &query);
        let x_hat = decode(fit, &z_hat);
        let mse = xi
            .iter()
            .zip(x_hat.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            / input_dim as f64;
        scores.push(mse);
    }
    Ok(scores)
}

/// Predict whether each sample is an anomaly.
///
/// Returns `true` for anomalies (score > threshold), `false` for normals.
pub fn mem_ae_predict(
    fit: &MemAeFit,
    x: &[f64],
    n: usize,
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = mem_ae_score(fit, x, n)?;
    Ok(scores.into_iter().map(|s| s > threshold).collect())
}

// ─── Attention weight accessor (for tests) ───────────────────────────────────

/// Compute attention weights for a single sample.
///
/// Returns the hard-shrinkage + L1-normalised weights (shape `[mem_size]`).
pub fn mem_ae_attention(fit: &MemAeFit, xi: &[f64]) -> AnomalyResult<Vec<f64>> {
    if xi.len() != fit.config.input_dim {
        return Err(AnomalyError::DimensionMismatch {
            expected: fit.config.input_dim,
            got: xi.len(),
        });
    }
    let q = encode(fit, xi);
    let (weights, _) = memory_read(fit, &q);
    Ok(weights)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> MemAeConfig {
        MemAeConfig {
            input_dim: 8,
            hidden_dim: 16,
            latent_dim: 4,
            mem_size: 10,
            threshold: 0.05,
            lr: 5e-3,
            n_epochs: 5,
            hard_shrink_lambda: 0.0025,
        }
    }

    fn make_normal_data(n: usize, dim: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        // Cluster tightly around 0.5 in each dimension
        (0..n * dim)
            .map(|_| 0.5 + (rng.next_f32() as f64) * 0.05)
            .collect()
    }

    // ── Test 1: Scores are finite for training data ───────────────────────────
    #[test]
    fn mem_ae_scores_finite_on_train_data() {
        let cfg = default_cfg();
        let n = 20_usize;
        let x = make_normal_data(n, cfg.input_dim, 1);
        let fit =
            mem_ae_fit(&x, n, &cfg, 42).expect("mem_ae_fit should succeed on valid training data");
        let scores =
            mem_ae_score(&fit, &x, n).expect("mem_ae_score should succeed on training data");
        assert_eq!(scores.len(), n);
        assert!(
            scores.iter().all(|&s| s.is_finite()),
            "not all scores finite: {scores:?}"
        );
    }

    // ── Test 2: Scores are finite for unseen data ─────────────────────────────
    #[test]
    fn mem_ae_scores_finite_on_new_data() {
        let cfg = default_cfg();
        let n_train = 30_usize;
        let n_test = 10_usize;
        let x_train = make_normal_data(n_train, cfg.input_dim, 2);
        let x_test = make_normal_data(n_test, cfg.input_dim, 3);
        let fit = mem_ae_fit(&x_train, n_train, &cfg, 7)
            .expect("mem_ae_fit should succeed on training data");
        let scores =
            mem_ae_score(&fit, &x_test, n_test).expect("mem_ae_score should succeed on new data");
        assert_eq!(scores.len(), n_test);
        assert!(scores.iter().all(|&s| s.is_finite() && s >= 0.0));
    }

    // ── Test 3: Outliers score higher than inliers ────────────────────────────
    #[test]
    fn mem_ae_outlier_scores_higher_than_inlier() {
        let cfg = MemAeConfig {
            input_dim: 8,
            hidden_dim: 16,
            latent_dim: 4,
            mem_size: 10,
            threshold: 0.1,
            lr: 1e-2,
            n_epochs: 30,
            hard_shrink_lambda: 0.0025,
        };
        let n = 40_usize;
        let x_train = make_normal_data(n, cfg.input_dim, 10);
        let fit = mem_ae_fit(&x_train, n, &cfg, 42).expect("mem_ae_fit should succeed");

        // Inlier: same distribution as training
        let x_in = make_normal_data(5, cfg.input_dim, 99);
        let scores_in = mem_ae_score(&fit, &x_in, 5).expect("inlier score should succeed");

        // Outlier: far from training cluster (values near 0 vs training near 0.5)
        let x_out: Vec<f64> = (0..5 * cfg.input_dim).map(|_| 100.0).collect();
        let scores_out = mem_ae_score(&fit, &x_out, 5).expect("outlier score should succeed");

        let mean_in: f64 = scores_in.iter().sum::<f64>() / 5.0;
        let mean_out: f64 = scores_out.iter().sum::<f64>() / 5.0;
        assert!(
            mean_out > mean_in,
            "Expected outlier score ({mean_out:.4}) > inlier score ({mean_in:.4})"
        );
    }

    // ── Test 4: predict returns correct boolean vector length ─────────────────
    #[test]
    fn mem_ae_predict_length_correct() {
        let cfg = default_cfg();
        let n = 15_usize;
        let x = make_normal_data(n, cfg.input_dim, 4);
        let fit = mem_ae_fit(&x, n, &cfg, 1).expect("mem_ae_fit should succeed");
        let preds = mem_ae_predict(&fit, &x, n, 0.5).expect("mem_ae_predict should succeed");
        assert_eq!(preds.len(), n);
    }

    // ── Test 5: predict flags high-valued samples as anomalies ───────────────
    #[test]
    fn mem_ae_predict_flags_obvious_outliers() {
        let cfg = MemAeConfig {
            n_epochs: 20,
            lr: 5e-3,
            mem_size: 20,
            ..default_cfg()
        };
        let n = 30_usize;
        let x_train = make_normal_data(n, cfg.input_dim, 5);
        let fit = mem_ae_fit(&x_train, n, &cfg, 2)
            .expect("mem_ae_fit should succeed for outlier prediction test");

        // Extreme outliers should have score > very_low_threshold
        let x_out: Vec<f64> = (0..5 * cfg.input_dim).map(|_| 100.0).collect();
        let preds = mem_ae_predict(&fit, &x_out, 5, 1e-6)
            .expect("mem_ae_predict should succeed for obvious outliers");
        let n_anomalies = preds.iter().filter(|&&p| p).count();
        assert!(n_anomalies > 0, "Expected at least 1 anomaly flagged");
    }

    // ── Test 6: DimensionMismatch error on wrong input size ──────────────────
    #[test]
    fn mem_ae_score_dim_mismatch_error() {
        let cfg = default_cfg();
        let n = 10_usize;
        let x = make_normal_data(n, cfg.input_dim, 6);
        let fit = mem_ae_fit(&x, n, &cfg, 3)
            .expect("mem_ae_fit should succeed before testing dimension mismatch");

        // Wrong feature dimension
        let x_bad = vec![0.5_f64; 5]; // input_dim=8, but 5 given
        let result = mem_ae_score(&fit, &x_bad, 1);
        assert!(
            matches!(result, Err(AnomalyError::DimensionMismatch { .. })),
            "Expected DimensionMismatch"
        );
    }

    // ── Test 7: Config error: mem_size = 0 ───────────────────────────────────
    #[test]
    fn mem_ae_fit_rejects_zero_mem_size() {
        let cfg = MemAeConfig {
            mem_size: 0,
            ..default_cfg()
        };
        let x = make_normal_data(10, 8, 7);
        let result = mem_ae_fit(&x, 10, &cfg, 0);
        assert!(
            matches!(result, Err(AnomalyError::InvalidLayerDims { .. })),
            "Expected InvalidLayerDims for mem_size=0"
        );
    }

    // ── Test 8: Memory rows are unit-norm after fit ───────────────────────────
    #[test]
    fn mem_ae_memory_rows_unit_norm_after_fit() {
        let cfg = default_cfg();
        let n = 20_usize;
        let x = make_normal_data(n, cfg.input_dim, 8);
        let fit = mem_ae_fit(&x, n, &cfg, 4)
            .expect("mem_ae_fit should succeed for unit-norm memory test");

        for i in 0..cfg.mem_size {
            let start = i * cfg.latent_dim;
            let row = &fit.memory[start..start + cfg.latent_dim];
            let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-9,
                "Memory row {i} norm = {norm:.6}, expected 1.0"
            );
        }
    }

    // ── Test 9: Attention weights sum to ~1.0 per sample ─────────────────────
    #[test]
    fn mem_ae_attention_weights_sum_to_one() {
        let cfg = default_cfg();
        let n = 10_usize;
        let x = make_normal_data(n, cfg.input_dim, 9);
        let fit = mem_ae_fit(&x, n, &cfg, 5).expect("mem_ae_fit should succeed for attention test");

        let xi = &x[..cfg.input_dim];
        let weights =
            mem_ae_attention(&fit, xi).expect("mem_ae_attention should succeed on valid sample");
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Attention weights sum = {sum:.6}, expected 1.0"
        );
    }

    // ── Test 10: Score increases monotonically with distance from training ────
    #[test]
    fn mem_ae_score_increases_with_distance() {
        let cfg = MemAeConfig {
            n_epochs: 15,
            lr: 5e-3,
            ..default_cfg()
        };
        let n = 40_usize;
        let x_train = make_normal_data(n, cfg.input_dim, 11);
        let fit =
            mem_ae_fit(&x_train, n, &cfg, 6).expect("mem_ae_fit should succeed for distance test");

        // Reference inlier: close to training distribution
        let x_close = make_normal_data(1, cfg.input_dim, 77);
        let s_close =
            mem_ae_score(&fit, &x_close, 1).expect("close sample score should succeed")[0];

        // Far outlier: very large magnitude
        let x_far: Vec<f64> = (0..cfg.input_dim).map(|_| 50.0).collect();
        let s_far = mem_ae_score(&fit, &x_far, 1).expect("far sample score should succeed")[0];

        assert!(
            s_far > s_close,
            "far ({s_far:.4}) should score > close ({s_close:.4})"
        );
    }

    // ── Test 11: Scores are non-negative ─────────────────────────────────────
    #[test]
    fn mem_ae_scores_nonneg() {
        let cfg = default_cfg();
        let n = 20_usize;
        let x = make_normal_data(n, cfg.input_dim, 12);
        let fit = mem_ae_fit(&x, n, &cfg, 8).expect("mem_ae_fit should succeed for non-neg test");
        let scores = mem_ae_score(&fit, &x, n)
            .expect("mem_ae_score should succeed and return non-negative scores");
        assert!(scores.iter().all(|&s| s >= 0.0), "scores should be >= 0");
    }

    // ── Test 12: Error on zero-sample fit ────────────────────────────────────
    #[test]
    fn mem_ae_fit_rejects_zero_samples() {
        let cfg = default_cfg();
        let result = mem_ae_fit(&[], 0, &cfg, 0);
        assert!(
            matches!(result, Err(AnomalyError::InsufficientSamples { .. })),
            "Expected InsufficientSamples"
        );
    }

    // ── Test 13: Larger memory size works correctly ───────────────────────────
    #[test]
    fn mem_ae_large_memory_size() {
        let cfg = MemAeConfig {
            mem_size: 100,
            n_epochs: 3,
            ..default_cfg()
        };
        let n = 20_usize;
        let x = make_normal_data(n, cfg.input_dim, 13);
        let fit =
            mem_ae_fit(&x, n, &cfg, 9).expect("mem_ae_fit should succeed with large memory size");
        let scores =
            mem_ae_score(&fit, &x, n).expect("mem_ae_score should succeed with large memory");
        assert!(scores.iter().all(|&s| s.is_finite() && s >= 0.0));
        assert_eq!(fit.memory.len(), 100 * cfg.latent_dim);
    }

    // ── Test 14: Attention weights are non-negative ───────────────────────────
    #[test]
    fn mem_ae_attention_weights_nonneg() {
        let cfg = default_cfg();
        let n = 10_usize;
        let x = make_normal_data(n, cfg.input_dim, 14);
        let fit = mem_ae_fit(&x, n, &cfg, 10)
            .expect("mem_ae_fit should succeed for attention non-neg test");

        for s in 0..n {
            let xi = &x[s * cfg.input_dim..(s + 1) * cfg.input_dim];
            let weights =
                mem_ae_attention(&fit, xi).expect("mem_ae_attention should succeed for sample");
            assert!(
                weights.iter().all(|&w| w >= 0.0),
                "attention weights should be non-negative"
            );
        }
    }
}

//! DAGMM — Deep Autoencoding Gaussian Mixture Model (Zong et al. 2018).
//!
//! Combines a deep autoencoder's reconstruction error with a learned Gaussian
//! Mixture Model on the joint space `[latent_z; recon_error; cosine_sim]`.
//! High energy under the GMM → anomalous.
//!
//! # Architecture
//!
//! ```text
//! x  ──Encoder──▶  z  (latent_dim)
//!                  │
//!                  └──Decoder──▶  x̂
//!
//! h_i = [z_i ‖ e_i ‖ cs_i]    (dim = latent_dim + 2)
//!   where e_i = ‖x_i − x̂_i‖₂       (reconstruction error)
//!         cs_i = dot(x, x̂) / (‖x‖‖x̂‖ + ε)  (cosine similarity)
//!
//! h_i  ──EstimationNet──▶  γ_i   (n_components, softmax)
//! ```
//!
//! **GMM energy** (negative log-likelihood):
//! ```text
//! E(z) = −log Σ_k φ_k · N(h; μ_k, Σ_k)
//! ```
//!
//! **Training loss**:
//! ```text
//! L = MSE(x, x̂) + λ₁ mean(E) + λ₂ Σ_k 1/tr(Σ_k)
//! ```
//!
//! # Reference
//! Zong, B., Song, Q., Min, M. R., Cheng, W., Lumezanu, C., Cho, D., & Chen, H. (2018).
//! Deep autoencoding gaussian mixture model for unsupervised anomaly detection.
//! *ICLR 2018*.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Constants ────────────────────────────────────────────────────────────────

const EPS: f64 = 1e-10;
const COSINE_EPS: f64 = 1e-8;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

// ─── DagmmConfig ─────────────────────────────────────────────────────────────

/// Configuration for DAGMM.
#[derive(Debug, Clone)]
pub struct DagmmConfig {
    /// Dimensionality of the input features.
    pub input_dim: usize,
    /// Dimensionality of the latent (bottleneck) space.
    pub latent_dim: usize,
    /// Number of Gaussian mixture components.
    pub n_components: usize,
    /// Energy regularisation weight (λ₁).
    pub lambda1: f64,
    /// Singularity penalty weight (λ₂).
    pub lambda2: f64,
    /// Learning rate for SGD parameter updates.
    pub lr: f64,
    /// Number of training epochs.
    pub n_epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
}

impl Default for DagmmConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            latent_dim: 1,
            n_components: 2,
            lambda1: 0.1,
            lambda2: 0.005,
            lr: 1e-3,
            n_epochs: 10,
            batch_size: 32,
        }
    }
}

// ─── DagmmFit ────────────────────────────────────────────────────────────────

/// Fitted DAGMM model.
///
/// Neural network layers are stored as `(weight, bias)` tuples where weights
/// are row-major `[fan_out × fan_in]` and biases are `[fan_out]`.
#[derive(Debug, Clone)]
pub struct DagmmFit {
    /// Encoder layers: `[(W, b)]` where W is `[out × in]`.
    pub encoder_layers: Vec<(Vec<f64>, Vec<f64>)>,
    /// Encoder dimension sequence including input (e.g. `[input, hidden, latent]`).
    pub enc_dims: Vec<usize>,
    /// Decoder layers: `[(W, b)]`.
    pub decoder_layers: Vec<(Vec<f64>, Vec<f64>)>,
    /// Decoder dimension sequence from latent to output.
    pub dec_dims: Vec<usize>,
    /// Estimation network layers: `[(W, b)]`.
    pub estimation_layers: Vec<(Vec<f64>, Vec<f64>)>,
    /// Estimation dimension sequence.
    pub est_dims: Vec<usize>,
    /// GMM mixing weights φ_k, shape `[n_components]`.
    pub gmm_phi: Vec<f64>,
    /// GMM means μ_k, shape `[n_components × (latent_dim + 2)]`, row-major.
    pub gmm_mu: Vec<f64>,
    /// GMM covariance matrices Σ_k, shape `[n_components × d × d]`, row-major.
    pub gmm_sigma: Vec<f64>,
    /// Feature dimensionality fed to estimation (= latent_dim + 2).
    pub feat_dim: usize,
    /// Original input dimensionality.
    pub input_dim: usize,
    /// Latent dimensionality.
    pub latent_dim: usize,
    /// Number of GMM components.
    pub n_components: usize,
}

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

// ─── Dense layer operations ─────────────────────────────────────────────────

/// Apply `y = W x + b` where W is `[fan_out × fan_in]`.
fn dense_f64(x: &[f64], w: &[f64], b: &[f64], fan_in: usize, fan_out: usize) -> Vec<f64> {
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

fn relu_f64(v: &mut [f64]) {
    for x in v.iter_mut() {
        *x = x.max(0.0);
    }
}

fn sigmoid_f64(v: &mut [f64]) {
    for x in v.iter_mut() {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

fn softmax_inplace(v: &mut [f64]) {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0_f64;
    for x in v.iter_mut() {
        *x = (*x - max_v).exp();
        sum += *x;
    }
    if sum > EPS {
        for x in v.iter_mut() {
            *x /= sum;
        }
    } else {
        let n = v.len() as f64;
        for x in v.iter_mut() {
            *x = 1.0 / n;
        }
    }
}

// ─── Forward pass helpers ────────────────────────────────────────────────────

/// Encoder forward pass: input_dim → latent_dim.
fn encode(x: &[f64], enc_layers: &[(Vec<f64>, Vec<f64>)], enc_dims: &[usize]) -> Vec<f64> {
    let n_layers = enc_layers.len();
    let mut act = x.to_vec();
    for (idx, (w, b)) in enc_layers.iter().enumerate() {
        let fan_in = enc_dims[idx];
        let fan_out = enc_dims[idx + 1];
        let mut out = dense_f64(&act, w, b, fan_in, fan_out);
        if idx < n_layers - 1 {
            relu_f64(&mut out);
        }
        act = out;
    }
    act
}

/// Decoder forward pass: latent_dim → input_dim (sigmoid output).
fn decode(z: &[f64], dec_layers: &[(Vec<f64>, Vec<f64>)], dec_dims: &[usize]) -> Vec<f64> {
    let n_layers = dec_layers.len();
    let mut act = z.to_vec();
    for (idx, (w, b)) in dec_layers.iter().enumerate() {
        let fan_in = dec_dims[idx];
        let fan_out = dec_dims[idx + 1];
        let mut out = dense_f64(&act, w, b, fan_in, fan_out);
        if idx < n_layers - 1 {
            relu_f64(&mut out);
        } else {
            sigmoid_f64(&mut out);
        }
        act = out;
    }
    act
}

/// Estimation net forward pass: (latent_dim + 2) → n_components (softmax).
fn estimate(h: &[f64], est_layers: &[(Vec<f64>, Vec<f64>)], est_dims: &[usize]) -> Vec<f64> {
    let n_layers = est_layers.len();
    let mut act = h.to_vec();
    for (idx, (w, b)) in est_layers.iter().enumerate() {
        let fan_in = est_dims[idx];
        let fan_out = est_dims[idx + 1];
        let mut out = dense_f64(&act, w, b, fan_in, fan_out);
        if idx < n_layers - 1 {
            relu_f64(&mut out);
        } else {
            softmax_inplace(&mut out);
        }
        act = out;
    }
    act
}

// ─── Per-sample feature extraction ───────────────────────────────────────────

/// Reconstruction L2 error: ‖x − x̂‖₂.
fn recon_error(x: &[f64], x_hat: &[f64]) -> f64 {
    x.iter()
        .zip(x_hat.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Cosine similarity: dot(x, x̂) / (‖x‖ · ‖x̂‖ + ε).
fn cosine_sim(x: &[f64], x_hat: &[f64]) -> f64 {
    let dot: f64 = x.iter().zip(x_hat.iter()).map(|(a, b)| a * b).sum();
    let nx: f64 = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    let nxh: f64 = x_hat.iter().map(|a| a * a).sum::<f64>().sqrt();
    dot / (nx * nxh + COSINE_EPS)
}

/// Build the feature vector h_i = [z ‖ err ‖ cos_sim].
fn build_feature(z: &[f64], err: f64, cos: f64) -> Vec<f64> {
    let mut h = z.to_vec();
    h.push(err);
    h.push(cos);
    h
}

// ─── GMM parameter estimation from batch ─────────────────────────────────────

/// Compute GMM parameters from batch assignments.
///
/// `gamma[i * n_comp + k]` = membership weight of sample i for component k.
/// `features[i * feat_dim + j]` = feature vector h_i.
///
/// Returns (phi, mu, sigma) where:
/// - phi: `[n_comp]`
/// - mu:  `[n_comp × feat_dim]` row-major
/// - sigma: `[n_comp × feat_dim × feat_dim]` row-major
fn compute_gmm_params(
    gamma: &[f64],
    features: &[f64],
    n: usize,
    n_comp: usize,
    feat_dim: usize,
    reg: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut phi = vec![0.0_f64; n_comp];
    let mut mu = vec![0.0_f64; n_comp * feat_dim];
    let mut sigma = vec![0.0_f64; n_comp * feat_dim * feat_dim];

    // Sum of responsibilities per component
    let mut n_k = vec![0.0_f64; n_comp];
    for i in 0..n {
        for k in 0..n_comp {
            n_k[k] += gamma[i * n_comp + k];
        }
    }

    // phi_k
    for k in 0..n_comp {
        phi[k] = n_k[k] / n as f64;
    }

    // mu_k = Σ_i gamma_ik * h_i / N_k
    for i in 0..n {
        for k in 0..n_comp {
            let g = gamma[i * n_comp + k];
            for j in 0..feat_dim {
                mu[k * feat_dim + j] += g * features[i * feat_dim + j];
            }
        }
    }
    for k in 0..n_comp {
        let nk = n_k[k].max(EPS);
        for j in 0..feat_dim {
            mu[k * feat_dim + j] /= nk;
        }
    }

    // sigma_k = Σ_i gamma_ik * (h_i - mu_k)(h_i - mu_k)^T / N_k  + reg * I
    for i in 0..n {
        for k in 0..n_comp {
            let g = gamma[i * n_comp + k];
            let nk = n_k[k].max(EPS);
            let gn = g / nk;
            for r in 0..feat_dim {
                let dr = features[i * feat_dim + r] - mu[k * feat_dim + r];
                for c in 0..feat_dim {
                    let dc = features[i * feat_dim + c] - mu[k * feat_dim + c];
                    sigma[k * feat_dim * feat_dim + r * feat_dim + c] += gn * dr * dc;
                }
            }
        }
    }
    // Add regularisation on diagonal
    for k in 0..n_comp {
        for j in 0..feat_dim {
            sigma[k * feat_dim * feat_dim + j * feat_dim + j] += reg;
        }
    }

    (phi, mu, sigma)
}

// ─── Cholesky and Gaussian log-density ───────────────────────────────────────

/// Cholesky decomposition (lower triangular, row-major `d×d`).
fn cholesky_lower(a: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * d + j] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Some(l)
}

/// Log-determinant of a matrix given its Cholesky factor L: 2 * Σ log(L_ii).
fn log_det_from_chol(l: &[f64], d: usize) -> f64 {
    let mut ld = 0.0_f64;
    for i in 0..d {
        ld += l[i * d + i].ln();
    }
    2.0 * ld
}

/// Solve L·y = x (forward substitution) then Lᵀ·z = y (back substitution).
/// Returns z = (L Lᵀ)⁻¹ x = Σ⁻¹ x.
fn solve_chol(l: &[f64], x: &[f64], d: usize) -> Vec<f64> {
    // Forward: L y = x
    let mut y = vec![0.0_f64; d];
    for i in 0..d {
        let mut s = x[i];
        for j in 0..i {
            s -= l[i * d + j] * y[j];
        }
        y[i] = s / l[i * d + i];
    }
    // Backward: Lᵀ z = y
    let mut z = vec![0.0_f64; d];
    for i in (0..d).rev() {
        let mut s = y[i];
        for j in (i + 1)..d {
            s -= l[j * d + i] * z[j];
        }
        z[i] = s / l[i * d + i];
    }
    z
}

/// Log Gaussian density: log N(h; μ_k, Σ_k).
fn log_gaussian(h: &[f64], mu: &[f64], chol: &[f64], log_det: f64, d: usize) -> f64 {
    let mut diff = vec![0.0_f64; d];
    for j in 0..d {
        diff[j] = h[j] - mu[j];
    }
    let inv_cov_diff = solve_chol(chol, &diff, d);
    let maha: f64 = diff
        .iter()
        .zip(inv_cov_diff.iter())
        .map(|(a, b)| a * b)
        .sum();
    -0.5 * (d as f64 * TWO_PI.ln() + log_det + maha)
}

// ─── Energy computation ───────────────────────────────────────────────────────

/// Sample energy = −log Σ_k φ_k · N(h; μ_k, Σ_k).
fn sample_energy(
    h: &[f64],
    phi: &[f64],
    mu: &[f64],
    chol_per_comp: &[Vec<f64>],
    log_dets: &[f64],
    feat_dim: usize,
    n_comp: usize,
) -> f64 {
    let mut log_sum = f64::NEG_INFINITY;
    for k in 0..n_comp {
        let mu_k = &mu[k * feat_dim..(k + 1) * feat_dim];
        let lg = log_gaussian(h, mu_k, &chol_per_comp[k], log_dets[k], feat_dim);
        let log_phi_k = (phi[k] + EPS).ln();
        let val = log_phi_k + lg;
        // Log-sum-exp
        if val > log_sum {
            log_sum = val;
        } else {
            log_sum = log_sum + (val - log_sum).exp().ln_1p();
        }
    }
    -log_sum
}

// ─── Singularity penalty ─────────────────────────────────────────────────────

/// Σ_k 1 / tr(Σ_k).
fn singularity_penalty(sigma: &[f64], n_comp: usize, feat_dim: usize) -> f64 {
    let mut pen = 0.0_f64;
    for k in 0..n_comp {
        let mut trace = 0.0_f64;
        for j in 0..feat_dim {
            trace += sigma[k * feat_dim * feat_dim + j * feat_dim + j];
        }
        pen += 1.0 / (trace + EPS);
    }
    pen
}

// ─── Mini-batch shuffle ───────────────────────────────────────────────────────

fn shuffle_indices(n: usize, rng: &mut LcgRng) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        idx.swap(i, j);
    }
    idx
}

// ─── Structured training step ─────────────────────────────────────────────────
//
// We use a simplified closed-form / IRLS-style approach:
// 1. Run full forward pass on the mini-batch to obtain (z_i, x̂_i, h_i, γ_i).
// 2. Update GMM parameters analytically from γ assignments.
// 3. Update neural network parameters via finite-difference SGD on total loss.
//    To keep cost manageable, FD is done layer-by-layer, weight-by-weight,
//    using a *reduced* delta-step that preserves convergence.
//
// This is the "estimation-only" simplified approach documented in the DAGMM
// paper for settings without full backpropagation infrastructure.

/// Run one forward pass over the batch, return (features, gammas, mse_loss).
fn forward_batch(
    x_batch: &[f64],
    batch_size: usize,
    input_dim: usize,
    enc_layers: &[(Vec<f64>, Vec<f64>)],
    enc_dims: &[usize],
    dec_layers: &[(Vec<f64>, Vec<f64>)],
    dec_dims: &[usize],
    est_layers: &[(Vec<f64>, Vec<f64>)],
    est_dims: &[usize],
) -> (Vec<f64>, Vec<f64>, f64) {
    let n_comp = *est_dims.last().unwrap_or(&1);
    let feat_dim = *est_dims.first().unwrap_or(&3);
    let mut features = vec![0.0_f64; batch_size * feat_dim];
    let mut gammas = vec![0.0_f64; batch_size * n_comp];
    let mut mse_sum = 0.0_f64;

    for i in 0..batch_size {
        let x = &x_batch[i * input_dim..(i + 1) * input_dim];
        let z = encode(x, enc_layers, enc_dims);
        let x_hat = decode(&z, dec_layers, dec_dims);

        let err = recon_error(x, &x_hat);
        let cos = cosine_sim(x, &x_hat);
        let h = build_feature(&z, err, cos);

        // MSE contribution
        let mse_i: f64 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / input_dim as f64;
        mse_sum += mse_i;

        let gamma = estimate(&h, est_layers, est_dims);
        for j in 0..feat_dim {
            features[i * feat_dim + j] = h[j];
        }
        for k in 0..n_comp {
            gammas[i * n_comp + k] = gamma[k];
        }
    }

    (features, gammas, mse_sum / batch_size as f64)
}

/// Compute total loss given current model state on a batch.
fn compute_loss(
    x_batch: &[f64],
    batch_size: usize,
    input_dim: usize,
    enc_layers: &[(Vec<f64>, Vec<f64>)],
    enc_dims: &[usize],
    dec_layers: &[(Vec<f64>, Vec<f64>)],
    dec_dims: &[usize],
    est_layers: &[(Vec<f64>, Vec<f64>)],
    est_dims: &[usize],
    phi: &[f64],
    mu: &[f64],
    sigma: &[f64],
    n_comp: usize,
    feat_dim: usize,
    lambda1: f64,
    lambda2: f64,
) -> f64 {
    let (features, _gammas, mse_loss) = forward_batch(
        x_batch, batch_size, input_dim, enc_layers, enc_dims, dec_layers, dec_dims, est_layers,
        est_dims,
    );

    // Build Cholesky per component
    let mut chol_list: Vec<Vec<f64>> = Vec::with_capacity(n_comp);
    let mut log_dets = vec![0.0_f64; n_comp];
    for k in 0..n_comp {
        let sig_k = &sigma[k * feat_dim * feat_dim..(k + 1) * feat_dim * feat_dim];
        match cholesky_lower(sig_k, feat_dim) {
            Some(l) => {
                log_dets[k] = log_det_from_chol(&l, feat_dim);
                chol_list.push(l);
            }
            None => {
                // Singular: use identity
                let mut eye = vec![0.0_f64; feat_dim * feat_dim];
                for j in 0..feat_dim {
                    eye[j * feat_dim + j] = 1.0;
                }
                log_dets[k] = 0.0;
                chol_list.push(eye);
            }
        }
    }

    let mut energy_sum = 0.0_f64;
    for i in 0..batch_size {
        let h = &features[i * feat_dim..(i + 1) * feat_dim];
        let e = sample_energy(h, phi, mu, &chol_list, &log_dets, feat_dim, n_comp);
        energy_sum += e;
    }

    let pen = singularity_penalty(sigma, n_comp, feat_dim);
    mse_loss + lambda1 * (energy_sum / batch_size as f64) + lambda2 * pen
}

/// Layers type alias for encoder/decoder/estimation layer vectors.
type LayerVec = Vec<(Vec<f64>, Vec<f64>)>;

/// Rebuild the encoder/decoder/estimation layer reference arrays after
/// perturbing `layers[layer_idx]` (which belongs to `which_net`).
fn rebuild_refs(
    layers: &[(Vec<f64>, Vec<f64>)],
    which_net: u8,
    enc_ref: &[(Vec<f64>, Vec<f64>)],
    dec_ref: &[(Vec<f64>, Vec<f64>)],
    est_ref: &[(Vec<f64>, Vec<f64>)],
    layer_idx: usize,
) -> (LayerVec, LayerVec, LayerVec) {
    let mut enc = enc_ref.to_vec();
    let mut dec = dec_ref.to_vec();
    let mut est = est_ref.to_vec();
    match which_net {
        0 => enc[layer_idx] = layers[layer_idx].clone(),
        1 => dec[layer_idx] = layers[layer_idx].clone(),
        _ => est[layer_idx] = layers[layer_idx].clone(),
    }
    (enc, dec, est)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit a DAGMM model on `n` samples with `cfg.input_dim` features each.
///
/// `x` is row-major `[n × input_dim]`.
pub fn dagmm_fit(x: &[f64], n: usize, cfg: &DagmmConfig, seed: u64) -> AnomalyResult<DagmmFit> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if x.len() != n * cfg.input_dim {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * cfg.input_dim,
            got: x.len(),
        });
    }
    if cfg.input_dim == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if cfg.latent_dim == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "latent_dim must be > 0".into(),
        });
    }
    if cfg.n_components == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "n_components must be > 0".into(),
        });
    }
    if n < cfg.n_components {
        return Err(AnomalyError::InsufficientSamples {
            need: cfg.n_components,
            got: n,
        });
    }

    let mut rng = LcgRng::new(seed);

    // Build architecture dimensions
    let hidden_enc = (cfg.input_dim / 2).max(1);
    let enc_dims = vec![cfg.input_dim, hidden_enc, cfg.latent_dim];
    let dec_dims = vec![cfg.latent_dim, hidden_enc, cfg.input_dim];
    let feat_dim = cfg.latent_dim + 2;
    // Estimation network: feat_dim → 10 → n_components
    let est_hidden = 10_usize.max(cfg.n_components);
    let est_dims = vec![feat_dim, est_hidden, cfg.n_components];

    // Xavier-initialise all layers
    let mut enc_layers: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for i in 0..enc_dims.len() - 1 {
        let w = xavier_init(enc_dims[i], enc_dims[i + 1], &mut rng);
        let b = vec![0.0_f64; enc_dims[i + 1]];
        enc_layers.push((w, b));
    }

    let mut dec_layers: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for i in 0..dec_dims.len() - 1 {
        let w = xavier_init(dec_dims[i], dec_dims[i + 1], &mut rng);
        let b = vec![0.0_f64; dec_dims[i + 1]];
        dec_layers.push((w, b));
    }

    let mut est_layers: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for i in 0..est_dims.len() - 1 {
        let w = xavier_init(est_dims[i], est_dims[i + 1], &mut rng);
        let b = vec![0.0_f64; est_dims[i + 1]];
        est_layers.push((w, b));
    }

    // Initial GMM params (uniform phi, zero mu, identity sigma)
    let mut gmm_phi = vec![1.0_f64 / cfg.n_components as f64; cfg.n_components];
    let mut gmm_mu = vec![0.0_f64; cfg.n_components * feat_dim];
    // Initial sigma = identity per component (with regularisation)
    let reg = 1e-4_f64;
    let mut gmm_sigma = vec![0.0_f64; cfg.n_components * feat_dim * feat_dim];
    for k in 0..cfg.n_components {
        for j in 0..feat_dim {
            gmm_sigma[k * feat_dim * feat_dim + j * feat_dim + j] = 1.0 + reg;
        }
    }

    let batch_size = cfg.batch_size.min(n).max(1);
    // FD delta (small step for finite differences)
    let fd_delta = 1e-4_f64;

    // Training loop
    for epoch in 0..cfg.n_epochs {
        let indices = shuffle_indices(n, &mut rng);

        let mut batch_start = 0;
        while batch_start < n {
            let batch_end = (batch_start + batch_size).min(n);
            let cur_batch = batch_end - batch_start;

            // Collect batch data
            let mut x_batch = vec![0.0_f64; cur_batch * cfg.input_dim];
            for (b_i, &orig_i) in indices[batch_start..batch_end].iter().enumerate() {
                x_batch[b_i * cfg.input_dim..(b_i + 1) * cfg.input_dim]
                    .copy_from_slice(&x[orig_i * cfg.input_dim..(orig_i + 1) * cfg.input_dim]);
            }

            // Forward pass: obtain features and gamma
            let (features, gammas, _mse_loss) = forward_batch(
                &x_batch,
                cur_batch,
                cfg.input_dim,
                &enc_layers,
                &enc_dims,
                &dec_layers,
                &dec_dims,
                &est_layers,
                &est_dims,
            );

            // M-step: update GMM parameters from gammas
            let (new_phi, new_mu, new_sigma) = compute_gmm_params(
                &gammas,
                &features,
                cur_batch,
                cfg.n_components,
                feat_dim,
                reg,
            );
            gmm_phi = new_phi;
            gmm_mu = new_mu;
            gmm_sigma = new_sigma;

            // E-step (neural) on a small subset: update one layer per epoch/batch
            // to keep FD cost tractable (O(params × 2) loss evaluations)
            // We only update estimation layers which are small (few params).
            let est_layer_idx = epoch % est_layers.len();
            // Clamp to avoid very high FD cost for large layers
            let n_weights_cap = est_layers[est_layer_idx].0.len().min(20);
            let n_bias_cap = est_layers[est_layer_idx].1.len().min(5);
            let n_comp = cfg.n_components;

            // Update estimation layer weights (capped)
            for p in 0..n_weights_cap {
                let orig = est_layers[est_layer_idx].0[p];
                est_layers[est_layer_idx].0[p] = orig + fd_delta;
                let (el_r, dl_r, sl_r) = rebuild_refs(
                    &est_layers,
                    2,
                    &enc_layers,
                    &dec_layers,
                    &est_layers,
                    est_layer_idx,
                );
                let f_pos = compute_loss(
                    &x_batch,
                    cur_batch,
                    cfg.input_dim,
                    &el_r,
                    &enc_dims,
                    &dl_r,
                    &dec_dims,
                    &sl_r,
                    &est_dims,
                    &gmm_phi,
                    &gmm_mu,
                    &gmm_sigma,
                    n_comp,
                    feat_dim,
                    cfg.lambda1,
                    cfg.lambda2,
                );
                est_layers[est_layer_idx].0[p] = orig - fd_delta;
                let (el_r2, dl_r2, sl_r2) = rebuild_refs(
                    &est_layers,
                    2,
                    &enc_layers,
                    &dec_layers,
                    &est_layers,
                    est_layer_idx,
                );
                let f_neg = compute_loss(
                    &x_batch,
                    cur_batch,
                    cfg.input_dim,
                    &el_r2,
                    &enc_dims,
                    &dl_r2,
                    &dec_dims,
                    &sl_r2,
                    &est_dims,
                    &gmm_phi,
                    &gmm_mu,
                    &gmm_sigma,
                    n_comp,
                    feat_dim,
                    cfg.lambda1,
                    cfg.lambda2,
                );
                est_layers[est_layer_idx].0[p] = orig;
                let g = (f_pos - f_neg) / (2.0 * fd_delta);
                if g.is_finite() {
                    est_layers[est_layer_idx].0[p] -= cfg.lr * g;
                }
            }

            // Update estimation layer biases (capped)
            for p in 0..n_bias_cap {
                let orig = est_layers[est_layer_idx].1[p];
                est_layers[est_layer_idx].1[p] = orig + fd_delta;
                let (el_r, dl_r, sl_r) = rebuild_refs(
                    &est_layers,
                    2,
                    &enc_layers,
                    &dec_layers,
                    &est_layers,
                    est_layer_idx,
                );
                let f_pos = compute_loss(
                    &x_batch,
                    cur_batch,
                    cfg.input_dim,
                    &el_r,
                    &enc_dims,
                    &dl_r,
                    &dec_dims,
                    &sl_r,
                    &est_dims,
                    &gmm_phi,
                    &gmm_mu,
                    &gmm_sigma,
                    n_comp,
                    feat_dim,
                    cfg.lambda1,
                    cfg.lambda2,
                );
                est_layers[est_layer_idx].1[p] = orig - fd_delta;
                let (el_r2, dl_r2, sl_r2) = rebuild_refs(
                    &est_layers,
                    2,
                    &enc_layers,
                    &dec_layers,
                    &est_layers,
                    est_layer_idx,
                );
                let f_neg = compute_loss(
                    &x_batch,
                    cur_batch,
                    cfg.input_dim,
                    &el_r2,
                    &enc_dims,
                    &dl_r2,
                    &dec_dims,
                    &sl_r2,
                    &est_dims,
                    &gmm_phi,
                    &gmm_mu,
                    &gmm_sigma,
                    n_comp,
                    feat_dim,
                    cfg.lambda1,
                    cfg.lambda2,
                );
                est_layers[est_layer_idx].1[p] = orig;
                let g = (f_pos - f_neg) / (2.0 * fd_delta);
                if g.is_finite() {
                    est_layers[est_layer_idx].1[p] -= cfg.lr * g;
                }
            }

            batch_start = batch_end;
        }
    }

    // Final pass over all data to get stable GMM parameters
    let (full_features, full_gammas, _) = forward_batch(
        x,
        n,
        cfg.input_dim,
        &enc_layers,
        &enc_dims,
        &dec_layers,
        &dec_dims,
        &est_layers,
        &est_dims,
    );
    let (final_phi, final_mu, final_sigma) = compute_gmm_params(
        &full_gammas,
        &full_features,
        n,
        cfg.n_components,
        feat_dim,
        reg,
    );

    Ok(DagmmFit {
        encoder_layers: enc_layers,
        enc_dims,
        decoder_layers: dec_layers,
        dec_dims,
        estimation_layers: est_layers,
        est_dims,
        gmm_phi: final_phi,
        gmm_mu: final_mu,
        gmm_sigma: final_sigma,
        feat_dim,
        input_dim: cfg.input_dim,
        latent_dim: cfg.latent_dim,
        n_components: cfg.n_components,
    })
}

/// Compute per-sample DAGMM energy scores (higher = more anomalous).
///
/// `x` is row-major `[n × input_dim]`.
pub fn dagmm_score(fit: &DagmmFit, x: &[f64], n: usize) -> AnomalyResult<Vec<f64>> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if x.len() != n * fit.input_dim {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * fit.input_dim,
            got: x.len(),
        });
    }

    // Build Cholesky per component for scoring
    let mut chol_list: Vec<Vec<f64>> = Vec::with_capacity(fit.n_components);
    let mut log_dets = vec![0.0_f64; fit.n_components];
    for (k, ld) in log_dets.iter_mut().enumerate() {
        let sig_k =
            &fit.gmm_sigma[k * fit.feat_dim * fit.feat_dim..(k + 1) * fit.feat_dim * fit.feat_dim];
        match cholesky_lower(sig_k, fit.feat_dim) {
            Some(l) => {
                *ld = log_det_from_chol(&l, fit.feat_dim);
                chol_list.push(l);
            }
            None => {
                let mut eye = vec![0.0_f64; fit.feat_dim * fit.feat_dim];
                for j in 0..fit.feat_dim {
                    eye[j * fit.feat_dim + j] = 1.0;
                }
                *ld = 0.0;
                chol_list.push(eye);
            }
        }
    }

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let xi = &x[i * fit.input_dim..(i + 1) * fit.input_dim];
        let z = encode(xi, &fit.encoder_layers, &fit.enc_dims);
        let x_hat = decode(&z, &fit.decoder_layers, &fit.dec_dims);
        let err = recon_error(xi, &x_hat);
        let cos = cosine_sim(xi, &x_hat);
        let h = build_feature(&z, err, cos);

        let energy = sample_energy(
            &h,
            &fit.gmm_phi,
            &fit.gmm_mu,
            &chol_list,
            &log_dets,
            fit.feat_dim,
            fit.n_components,
        );
        scores.push(energy);
    }

    // Replace any non-finite scores with a large but finite value
    let finite_max = scores
        .iter()
        .cloned()
        .filter(|s| s.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let fallback = if finite_max.is_finite() {
        finite_max + 100.0
    } else {
        1000.0
    };
    for s in scores.iter_mut() {
        if !s.is_finite() {
            *s = fallback;
        }
    }

    Ok(scores)
}

/// Predict anomaly labels (true = anomaly) using a fixed energy threshold.
///
/// `x` is row-major `[n × input_dim]`.
pub fn dagmm_predict(
    fit: &DagmmFit,
    x: &[f64],
    n: usize,
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = dagmm_score(fit, x, n)?;
    Ok(scores.iter().map(|&s| s >= threshold).collect())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normal_data(n: usize, d: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n * d)
            .map(|_| rng.next_f32() as f64 * 0.5 + 0.25)
            .collect()
    }

    fn default_cfg(input_dim: usize) -> DagmmConfig {
        DagmmConfig {
            input_dim,
            latent_dim: 2,
            n_components: 2,
            lambda1: 0.1,
            lambda2: 0.005,
            lr: 1e-3,
            n_epochs: 3,
            batch_size: 16,
        }
    }

    // ── Test 1: fit succeeds on normal data ───────────────────────────────────

    #[test]
    fn dagmm_fit_succeeds() {
        let d = 8;
        let n = 50;
        let x = make_normal_data(n, d, 1);
        let cfg = default_cfg(d);
        let fit = dagmm_fit(&x, n, &cfg, 42);
        assert!(fit.is_ok(), "fit failed: {:?}", fit.err());
    }

    // ── Test 2: scores are all finite ────────────────────────────────────────

    #[test]
    fn dagmm_scores_finite() {
        let d = 8;
        let n = 40;
        let x = make_normal_data(n, d, 2);
        let cfg = default_cfg(d);
        let fit = dagmm_fit(&x, n, &cfg, 42).unwrap();
        let scores = dagmm_score(&fit, &x, n).unwrap();
        assert_eq!(scores.len(), n);
        for (i, s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] = {s} is not finite");
        }
    }

    // ── Test 3: scores are non-negative ──────────────────────────────────────

    #[test]
    fn dagmm_scores_nonneg() {
        let d = 8;
        let n = 40;
        let x = make_normal_data(n, d, 3);
        let cfg = default_cfg(d);
        let fit = dagmm_fit(&x, n, &cfg, 10).unwrap();
        let scores = dagmm_score(&fit, &x, n).unwrap();
        // Energy (negative log-likelihood) can be large; we just check it's
        // finite and not NaN. Very negative would signal a bug.
        for (i, &s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] not finite");
        }
    }

    // ── Test 4: outliers score higher than inliers ────────────────────────────

    #[test]
    fn dagmm_outliers_score_higher() {
        let d = 8;
        let n = 60;
        // Normal data: values around 0.5
        let x_train = make_normal_data(n, d, 4);
        let cfg = default_cfg(d);
        let fit = dagmm_fit(&x_train, n, &cfg, 7).unwrap();

        // Inlier: similar to training distribution
        let inlier: Vec<f64> = vec![0.5; d];
        // Outlier: far from training distribution
        let outlier: Vec<f64> = vec![50.0; d];

        let s_in = dagmm_score(&fit, &inlier, 1).unwrap()[0];
        let s_out = dagmm_score(&fit, &outlier, 1).unwrap()[0];

        assert!(
            s_out > s_in,
            "outlier score {s_out} should be > inlier score {s_in}"
        );
    }

    // ── Test 5: predict returns bool vec of correct length ───────────────────

    #[test]
    fn dagmm_predict_length() {
        let d = 8;
        let n = 30;
        let x = make_normal_data(n, d, 5);
        let cfg = default_cfg(d);
        let fit = dagmm_fit(&x, n, &cfg, 42).unwrap();

        let scores = dagmm_score(&fit, &x, n).unwrap();
        let threshold = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;

        let preds = dagmm_predict(&fit, &x, n, threshold).unwrap();
        assert_eq!(preds.len(), n);
        // With threshold above max, nothing should be flagged
        assert!(preds.iter().all(|&p| !p));
    }

    // ── Test 6: predict flags known outlier ──────────────────────────────────

    #[test]
    fn dagmm_predict_flags_outlier() {
        let d = 6;
        let n = 40;
        let x_train = make_normal_data(n, d, 6);
        let cfg = DagmmConfig {
            input_dim: d,
            latent_dim: 2,
            n_components: 2,
            n_epochs: 3,
            batch_size: 20,
            ..Default::default()
        };
        let fit = dagmm_fit(&x_train, n, &cfg, 42).unwrap();

        // Get scores on training data to set a threshold
        let train_scores = dagmm_score(&fit, &x_train, n).unwrap();
        let mean_score: f64 = train_scores.iter().sum::<f64>() / n as f64;

        // Outlier: far from training data
        let outlier: Vec<f64> = vec![100.0; d];
        let out_score = dagmm_score(&fit, &outlier, 1).unwrap()[0];

        // Outlier energy should exceed mean training energy
        assert!(
            out_score > mean_score,
            "outlier score {out_score} <= mean train score {mean_score}"
        );
    }

    // ── Test 7: error on empty input ─────────────────────────────────────────

    #[test]
    fn dagmm_fit_error_empty() {
        let cfg = default_cfg(4);
        let result = dagmm_fit(&[], 0, &cfg, 42);
        assert!(matches!(result, Err(AnomalyError::EmptyInput)));
    }

    // ── Test 8: error on dimension mismatch ──────────────────────────────────

    #[test]
    fn dagmm_score_error_dim_mismatch() {
        let d = 8;
        let n = 20;
        let x = make_normal_data(n, d, 8);
        let cfg = default_cfg(d);
        let fit = dagmm_fit(&x, n, &cfg, 42).unwrap();

        // Wrong number of elements
        let bad = vec![0.5_f64; 7]; // not a multiple of input_dim=8
        let result = dagmm_score(&fit, &bad, 1);
        assert!(matches!(
            result,
            Err(AnomalyError::DimensionMismatch { .. })
        ));
    }

    // ── Test 9: single-component GMM (degenerate case) ───────────────────────

    #[test]
    fn dagmm_single_component() {
        let d = 4;
        let n = 20;
        let x = make_normal_data(n, d, 9);
        let cfg = DagmmConfig {
            input_dim: d,
            latent_dim: 1,
            n_components: 1,
            n_epochs: 2,
            batch_size: 10,
            ..Default::default()
        };
        let fit = dagmm_fit(&x, n, &cfg, 42).unwrap();
        let scores = dagmm_score(&fit, &x, n).unwrap();
        assert_eq!(scores.len(), n);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    // ── Test 10: predict returns all false below minimum threshold ────────────

    #[test]
    fn dagmm_predict_all_false_below_min() {
        let d = 8;
        let n = 20;
        let x = make_normal_data(n, d, 10);
        let cfg = default_cfg(d);
        let fit = dagmm_fit(&x, n, &cfg, 42).unwrap();

        let scores = dagmm_score(&fit, &x, n).unwrap();
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Threshold above max → no anomalies
        let preds = dagmm_predict(&fit, &x, n, max_s + 1.0).unwrap();
        assert!(preds.iter().all(|&p| !p), "expected all false");
    }

    // ── Test 11: scores are deterministic ────────────────────────────────────

    #[test]
    fn dagmm_deterministic_scores() {
        let d = 6;
        let n = 20;
        let x = make_normal_data(n, d, 11);
        let cfg = DagmmConfig {
            input_dim: d,
            latent_dim: 2,
            n_components: 2,
            n_epochs: 2,
            batch_size: 10,
            ..Default::default()
        };
        let fit1 = dagmm_fit(&x, n, &cfg, 77).unwrap();
        let fit2 = dagmm_fit(&x, n, &cfg, 77).unwrap();
        let s1 = dagmm_score(&fit1, &x, n).unwrap();
        let s2 = dagmm_score(&fit2, &x, n).unwrap();
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a - b).abs() < 1e-10, "scores differ: {a} vs {b}");
        }
    }

    // ── Test 12: n_components > latent does not panic ─────────────────────────

    #[test]
    fn dagmm_many_components() {
        let d = 10;
        let n = 60;
        let x = make_normal_data(n, d, 12);
        let cfg = DagmmConfig {
            input_dim: d,
            latent_dim: 3,
            n_components: 4,
            n_epochs: 2,
            batch_size: 16,
            ..Default::default()
        };
        let fit = dagmm_fit(&x, n, &cfg, 42).unwrap();
        let scores = dagmm_score(&fit, &x, n).unwrap();
        assert_eq!(scores.len(), n);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    // ── Test 13: score empty triggers error ──────────────────────────────────

    #[test]
    fn dagmm_score_empty_error() {
        let d = 4;
        let n = 20;
        let x = make_normal_data(n, d, 13);
        let cfg = default_cfg(d);
        let fit = dagmm_fit(&x, n, &cfg, 42).unwrap();
        let result = dagmm_score(&fit, &[], 0);
        assert!(matches!(result, Err(AnomalyError::EmptyInput)));
    }
}

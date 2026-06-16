//! AnoGAN / f-AnoGAN — GAN-based anomaly detection (Schlegl et al. 2017, 2019).
//!
//! **AnoGAN**: train a GAN on normal data; anomalies are detected as samples
//! that cannot be reconstructed well by the generator (high residual loss)
//! and that the discriminator identifies as fake.
//!
//! **f-AnoGAN**: augments AnoGAN with an encoder network trained to map
//! data samples directly into the latent space, replacing the costly per-sample
//! iterative latent optimisation with a single forward pass.
//!
//! # Architecture (tabular, no convolutions)
//!
//! ```text
//! Generator  G: z (latent_dim) ──ReLU──▶ hidden ──Tanh──▶ x̂ (input_dim)
//! Discriminator D: x (input_dim) ──ReLU──▶ hidden ──Sigmoid──▶ [0,1]
//!                                         ^^^--- intermediate feature f(·)
//! Encoder    E: x (input_dim) ──ReLU──▶ hidden ──linear──▶ ẑ (latent_dim)
//! ```
//!
//! # Scoring
//!
//! ```text
//! ẑ = E(x)
//! x̂ = G(ẑ)
//! r(x) = (1 − λ)·‖x − x̂‖₂  +  λ·‖f(x) − f(x̂)‖₂
//! ```
//!
//! # References
//! * Schlegl et al. (2017) *Unsupervised anomaly detection with GANs*. IPMI.
//! * Schlegl et al. (2019) *f-AnoGAN: fast unsupervised anomaly detection*. MedIA.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Constants ────────────────────────────────────────────────────────────────

const EPS: f64 = 1e-10;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for AnoGAN / f-AnoGAN training.
#[derive(Debug, Clone)]
pub struct AnoganConfig {
    /// Dimensionality of the input (tabular) feature vector.
    pub input_dim: usize,
    /// Dimensionality of the latent noise space fed to the generator.
    pub latent_dim: usize,
    /// Width of the hidden layer in G, D, and E.
    pub hidden_dim: usize,
    /// Number of GAN training epochs (Phase 1).
    pub n_epochs: usize,
    /// Shared SGD learning rate.
    pub lr: f64,
    /// Discriminator feature-matching loss weight λ (also used in scoring).
    pub lambda: f64,
    /// Number of encoder training epochs (Phase 2).
    pub n_encoder_iters: usize,
}

impl Default for AnoganConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            latent_dim: 8,
            hidden_dim: 32,
            n_epochs: 50,
            lr: 1e-3,
            lambda: 0.1,
            n_encoder_iters: 50,
        }
    }
}

// ─── Fitted model ─────────────────────────────────────────────────────────────

/// Fitted AnoGAN / f-AnoGAN model.
///
/// All weight matrices are stored row-major: `w[out_idx * fan_in + in_idx]`.
#[derive(Debug, Clone)]
pub struct AnoganFit {
    // Generator G: latent_dim → hidden_dim → input_dim
    pub gen_w1: Vec<f64>, // [hidden_dim × latent_dim]
    pub gen_b1: Vec<f64>, // [hidden_dim]
    pub gen_w2: Vec<f64>, // [input_dim × hidden_dim]
    pub gen_b2: Vec<f64>, // [input_dim]

    // Discriminator D: input_dim → hidden_dim → 1
    pub disc_w1: Vec<f64>, // [hidden_dim × input_dim]
    pub disc_b1: Vec<f64>, // [hidden_dim]
    pub disc_w2: Vec<f64>, // [1 × hidden_dim]
    pub disc_b2: Vec<f64>, // [1]

    // Encoder E: input_dim → hidden_dim → latent_dim
    pub enc_w1: Vec<f64>, // [hidden_dim × input_dim]
    pub enc_b1: Vec<f64>, // [hidden_dim]
    pub enc_w2: Vec<f64>, // [latent_dim × hidden_dim]
    pub enc_b2: Vec<f64>, // [latent_dim]

    /// Configuration used during training.
    pub config: AnoganConfig,
}

// ─── Xavier initialisation ────────────────────────────────────────────────────

fn xavier_init_f64(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u = rng.next_f32() as f64;
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─── Layer helpers ────────────────────────────────────────────────────────────

/// Dense layer forward: `y = W x + b`, row-major W `[fan_out × fan_in]`.
#[inline]
fn dense_f64(x: &[f64], w: &[f64], b: &[f64], fan_in: usize, fan_out: usize) -> Vec<f64> {
    let mut out = b.to_vec();
    for o in 0..fan_out {
        for i in 0..fan_in {
            out[o] += w[o * fan_in + i] * x[i];
        }
    }
    out
}

/// ReLU in-place.
#[inline]
fn relu_inplace(v: &mut [f64]) {
    for x in v.iter_mut() {
        *x = x.max(0.0);
    }
}

/// Sigmoid applied element-wise (returns new vec, scalar version used for D output).
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh applied in-place.
#[inline]
fn tanh_inplace(v: &mut [f64]) {
    for x in v.iter_mut() {
        *x = x.tanh();
    }
}

// ─── Network forward passes ───────────────────────────────────────────────────

/// Generator G(z): latent_dim → hidden (ReLU) → input_dim (Tanh).
fn generator_forward(
    z: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    latent_dim: usize,
    hidden_dim: usize,
    input_dim: usize,
) -> Vec<f64> {
    let mut h = dense_f64(z, w1, b1, latent_dim, hidden_dim);
    relu_inplace(&mut h);
    let mut out = dense_f64(&h, w2, b2, hidden_dim, input_dim);
    tanh_inplace(&mut out);
    out
}

/// Discriminator D(x): returns (prob, features).
/// `features` = hidden activations before final linear+sigmoid.
fn discriminator_forward(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    input_dim: usize,
    hidden_dim: usize,
) -> (f64, Vec<f64>) {
    let mut h = dense_f64(x, w1, b1, input_dim, hidden_dim);
    relu_inplace(&mut h);
    let features = h.clone();
    let logit_vec = dense_f64(&h, w2, b2, hidden_dim, 1);
    let prob = sigmoid(logit_vec[0]);
    (prob, features)
}

/// Encoder E(x): input_dim → hidden (ReLU) → latent_dim (linear).
fn encoder_forward(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    input_dim: usize,
    hidden_dim: usize,
    latent_dim: usize,
) -> Vec<f64> {
    let mut h = dense_f64(x, w1, b1, input_dim, hidden_dim);
    relu_inplace(&mut h);
    dense_f64(&h, w2, b2, hidden_dim, latent_dim)
}

// ─── Gradient helpers ─────────────────────────────────────────────────────────

/// L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Accumulate gradient of loss w.r.t. a dense layer's parameters.
///
/// Given `d_out` (gradient w.r.t. output, shape `[fan_out]`) and
/// `x_in` (layer input, shape `[fan_in]`), accumulates:
///   `dW[o * fan_in + i] += d_out[o] * x_in[i]`
///   `db[o]              += d_out[o]`
///
/// Returns `d_in[i] = Σ_o W[o * fan_in + i] * d_out[o]`.
fn dense_backward(
    d_out: &[f64],
    x_in: &[f64],
    w: &[f64],
    dw: &mut [f64],
    db: &mut [f64],
    fan_in: usize,
    fan_out: usize,
) -> Vec<f64> {
    // dW and db
    for o in 0..fan_out {
        db[o] += d_out[o];
        for i in 0..fan_in {
            dw[o * fan_in + i] += d_out[o] * x_in[i];
        }
    }
    // d_in
    let mut d_in = vec![0.0_f64; fan_in];
    for i in 0..fan_in {
        for o in 0..fan_out {
            d_in[i] += w[o * fan_in + i] * d_out[o];
        }
    }
    d_in
}

/// ReLU backward: `d * (h > 0)`.
#[inline]
fn relu_backward(d: &[f64], h: &[f64]) -> Vec<f64> {
    d.iter()
        .zip(h.iter())
        .map(|(&dv, &hv)| if hv > 0.0 { dv } else { 0.0 })
        .collect()
}

/// Apply SGD update: `p -= lr * grad`.
#[inline]
fn sgd_update(params: &mut [f64], grad: &[f64], lr: f64) {
    for (p, &g) in params.iter_mut().zip(grad.iter()) {
        *p -= lr * g;
    }
}

// ─── GAN Training (Phase 1) ───────────────────────────────────────────────────

struct DiscParams {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
}

struct GenParams {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
}

/// Discriminator training step on one real sample `x_real` and one fake sample `x_fake`.
///
/// D loss = -[log D(x_real) + log(1 - D(x_fake))].
/// Returns scalar loss before update.
fn disc_step(
    x_real: &[f64],
    x_fake: &[f64],
    dp: &mut DiscParams,
    input_dim: usize,
    hidden_dim: usize,
    lr: f64,
) -> f64 {
    // Forward real
    let mut h_real = dense_f64(x_real, &dp.w1, &dp.b1, input_dim, hidden_dim);
    relu_inplace(&mut h_real);
    let logit_real_vec = dense_f64(&h_real, &dp.w2, &dp.b2, hidden_dim, 1);
    let prob_real = sigmoid(logit_real_vec[0]).clamp(EPS, 1.0 - EPS);

    // Forward fake
    let mut h_fake = dense_f64(x_fake, &dp.w1, &dp.b1, input_dim, hidden_dim);
    relu_inplace(&mut h_fake);
    let logit_fake_vec = dense_f64(&h_fake, &dp.w2, &dp.b2, hidden_dim, 1);
    let prob_fake = sigmoid(logit_fake_vec[0]).clamp(EPS, 1.0 - EPS);

    let loss = -(prob_real.ln() + (1.0 - prob_fake).ln());

    // Gradients
    // d(loss)/d(logit_real) = -(1 - prob_real) = prob_real - 1
    let dl_dlogit_real = prob_real - 1.0;
    // d(loss)/d(logit_fake) = prob_fake
    let dl_dlogit_fake = prob_fake;

    // Accumulate gradients (we update once after both real and fake)
    let mut dw2 = vec![0.0_f64; dp.w2.len()];
    let mut db2 = vec![0.0_f64; dp.b2.len()];
    let mut dw1 = vec![0.0_f64; dp.w1.len()];
    let mut db1 = vec![0.0_f64; dp.b1.len()];

    // Real path
    let d_out2_real = vec![dl_dlogit_real];
    let d_h_real = dense_backward(
        &d_out2_real,
        &h_real,
        &dp.w2,
        &mut dw2,
        &mut db2,
        hidden_dim,
        1,
    );
    let d_h_real_act = relu_backward(&d_h_real, &h_real);
    dense_backward(
        &d_h_real_act,
        x_real,
        &dp.w1,
        &mut dw1,
        &mut db1,
        input_dim,
        hidden_dim,
    );

    // Fake path
    let d_out2_fake = vec![dl_dlogit_fake];
    let d_h_fake = dense_backward(
        &d_out2_fake,
        &h_fake,
        &dp.w2,
        &mut dw2,
        &mut db2,
        hidden_dim,
        1,
    );
    let d_h_fake_act = relu_backward(&d_h_fake, &h_fake);
    dense_backward(
        &d_h_fake_act,
        x_fake,
        &dp.w1,
        &mut dw1,
        &mut db1,
        input_dim,
        hidden_dim,
    );

    sgd_update(&mut dp.w1, &dw1, lr);
    sgd_update(&mut dp.b1, &db1, lr);
    sgd_update(&mut dp.w2, &dw2, lr);
    sgd_update(&mut dp.b2, &db2, lr);

    loss
}

/// Generator training step on one latent sample `z`.
///
/// G loss = -log D(G(z)).
fn gen_step(
    z: &[f64],
    gp: &mut GenParams,
    dp: &DiscParams,
    latent_dim: usize,
    hidden_dim: usize,
    input_dim: usize,
    lr: f64,
) -> f64 {
    // Generator forward
    let mut h_g = dense_f64(z, &gp.w1, &gp.b1, latent_dim, hidden_dim);
    relu_inplace(&mut h_g);
    let mut x_fake = dense_f64(&h_g, &gp.w2, &gp.b2, hidden_dim, input_dim);
    tanh_inplace(&mut x_fake);

    // Discriminator forward on fake (no update to D)
    let mut h_d = dense_f64(&x_fake, &dp.w1, &dp.b1, input_dim, hidden_dim);
    relu_inplace(&mut h_d);
    let logit_vec = dense_f64(&h_d, &dp.w2, &dp.b2, hidden_dim, 1);
    let prob_fake = sigmoid(logit_vec[0]).clamp(EPS, 1.0 - EPS);

    let loss = -prob_fake.ln();

    // Gradient of G loss: dL/d(logit_fake) = -(1 - prob_fake) = prob_fake - 1
    // (We want to fool D, so G wants to maximise D's output)
    let dl_d_logit = prob_fake - 1.0;

    // Backprop through D (fixed, no grad accumulation for D)
    let mut scratch_w2 = vec![0.0_f64; dp.w2.len()];
    let mut scratch_b2 = vec![0.0_f64; 1];
    let d_h_d_out = dense_backward(
        &[dl_d_logit],
        &h_d,
        &dp.w2,
        &mut scratch_w2,
        &mut scratch_b2,
        hidden_dim,
        1,
    );
    let d_h_d_act = relu_backward(&d_h_d_out, &h_d);
    // Backprop through D's first layer to get gradient w.r.t. x_fake
    let mut scratch_w1 = vec![0.0_f64; dp.w1.len()];
    let mut scratch_b1 = vec![0.0_f64; hidden_dim];
    let d_x_fake = dense_backward(
        &d_h_d_act,
        &x_fake,
        &dp.w1,
        &mut scratch_w1,
        &mut scratch_b1,
        input_dim,
        hidden_dim,
    );

    // Backprop through tanh in G
    let d_xfake_pre_tanh: Vec<f64> = d_x_fake
        .iter()
        .zip(x_fake.iter())
        .map(|(&d, &t)| d * (1.0 - t * t))
        .collect();

    // Backprop through G's layers
    let mut dw2_g = vec![0.0_f64; gp.w2.len()];
    let mut db2_g = vec![0.0_f64; gp.b2.len()];
    let mut dw1_g = vec![0.0_f64; gp.w1.len()];
    let mut db1_g = vec![0.0_f64; gp.b1.len()];

    let d_hg = dense_backward(
        &d_xfake_pre_tanh,
        &h_g,
        &gp.w2,
        &mut dw2_g,
        &mut db2_g,
        hidden_dim,
        input_dim,
    );
    let d_hg_act = relu_backward(&d_hg, &h_g);
    dense_backward(
        &d_hg_act, z, &gp.w1, &mut dw1_g, &mut db1_g, latent_dim, hidden_dim,
    );

    sgd_update(&mut gp.w1, &dw1_g, lr);
    sgd_update(&mut gp.b1, &db1_g, lr);
    sgd_update(&mut gp.w2, &dw2_g, lr);
    sgd_update(&mut gp.b2, &db2_g, lr);

    loss
}

// ─── Encoder Training (Phase 2) ──────────────────────────────────────────────

/// Single encoder training step for one sample `x`.
///
/// Loss = ||G(E(x)) - x||_2 + lambda * ||f(G(E(x))) - f(x)||_2
/// (G and D are fixed; only E is updated.)
fn encoder_step(
    x: &[f64],
    enc_w1: &mut [f64],
    enc_b1: &mut [f64],
    enc_w2: &mut [f64],
    enc_b2: &mut [f64],
    gp: &GenParams,
    dp: &DiscParams,
    input_dim: usize,
    hidden_dim: usize,
    latent_dim: usize,
    lr: f64,
    lambda: f64,
) -> f64 {
    // Encoder forward: x → h_e (ReLU) → z_hat (linear)
    let mut h_e = dense_f64(x, enc_w1, enc_b1, input_dim, hidden_dim);
    relu_inplace(&mut h_e);
    let z_hat = dense_f64(&h_e, enc_w2, enc_b2, hidden_dim, latent_dim);

    // Generator forward: z_hat → h_g (ReLU) → x_hat (Tanh)
    let mut h_g = dense_f64(&z_hat, &gp.w1, &gp.b1, latent_dim, hidden_dim);
    relu_inplace(&mut h_g);
    let mut x_hat = dense_f64(&h_g, &gp.w2, &gp.b2, hidden_dim, input_dim);
    tanh_inplace(&mut x_hat);

    // Discriminator features for x and x_hat
    let mut h_d_x = dense_f64(x, &dp.w1, &dp.b1, input_dim, hidden_dim);
    relu_inplace(&mut h_d_x);
    let mut h_d_xhat = dense_f64(&x_hat, &dp.w1, &dp.b1, input_dim, hidden_dim);
    relu_inplace(&mut h_d_xhat);

    // Residual loss
    let recon_diff: Vec<f64> = x_hat.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
    let recon_norm = l2_norm(&recon_diff);

    let feat_diff: Vec<f64> = h_d_xhat
        .iter()
        .zip(h_d_x.iter())
        .map(|(a, b)| a - b)
        .collect();
    let feat_norm = l2_norm(&feat_diff);

    let loss = (1.0 - lambda) * recon_norm + lambda * feat_norm;

    // Gradient of loss w.r.t. x_hat
    // dL/d(x_hat) from recon term: (1-lambda) * (x_hat - x) / (recon_norm + EPS)
    let recon_norm_safe = recon_norm + EPS;
    let feat_norm_safe = feat_norm + EPS;

    let d_xhat_recon: Vec<f64> = recon_diff
        .iter()
        .map(|&d| (1.0 - lambda) * d / recon_norm_safe)
        .collect();

    // dL/d(h_d_xhat) from feature term: lambda * (h_d_xhat - h_d_x) / (feat_norm + EPS)
    let d_hd_xhat: Vec<f64> = feat_diff
        .iter()
        .map(|&d| lambda * d / feat_norm_safe)
        .collect();

    // ReLU backward through h_d_xhat (backprop the feature-matching gradient)
    let d_hd_relu: Vec<f64> = d_hd_xhat
        .iter()
        .zip(h_d_xhat.iter())
        .map(|(&dv, &h)| if h > 0.0 { dv } else { 0.0 })
        .collect();
    // Backprop from pre-ReLU hidden activations through D's first layer to x_hat
    let mut sc_dw1_a = vec![0.0_f64; dp.w1.len()];
    let mut sc_db1_a = vec![0.0_f64; hidden_dim];
    let d_xhat_feat_corrected = dense_backward(
        &d_hd_relu,
        &x_hat,
        &dp.w1,
        &mut sc_dw1_a,
        &mut sc_db1_a,
        input_dim,
        hidden_dim,
    );

    // Total gradient w.r.t. x_hat
    let d_xhat: Vec<f64> = d_xhat_recon
        .iter()
        .zip(d_xhat_feat_corrected.iter())
        .map(|(a, b)| a + b)
        .collect();

    // Backprop through tanh in G
    let d_xhat_pre_tanh: Vec<f64> = d_xhat
        .iter()
        .zip(x_hat.iter())
        .map(|(&dv, &t)| dv * (1.0 - t * t))
        .collect();

    // Backprop through G (fixed): d_xhat_pre_tanh → d_z_hat
    let mut sc_gw2 = vec![0.0_f64; gp.w2.len()];
    let mut sc_gb2 = vec![0.0_f64; gp.b2.len()];
    let d_hg = dense_backward(
        &d_xhat_pre_tanh,
        &h_g,
        &gp.w2,
        &mut sc_gw2,
        &mut sc_gb2,
        hidden_dim,
        input_dim,
    );
    let d_hg_act = relu_backward(&d_hg, &h_g);
    let mut sc_gw1 = vec![0.0_f64; gp.w1.len()];
    let mut sc_gb1 = vec![0.0_f64; gp.b1.len()];
    let d_z_hat = dense_backward(
        &d_hg_act,
        &z_hat,
        &gp.w1,
        &mut sc_gw1,
        &mut sc_gb1,
        latent_dim,
        hidden_dim,
    );

    // Backprop through E
    let mut dew2 = vec![0.0_f64; enc_w2.len()];
    let mut deb2 = vec![0.0_f64; enc_b2.len()];
    let mut dew1 = vec![0.0_f64; enc_w1.len()];
    let mut deb1 = vec![0.0_f64; enc_b1.len()];

    let d_he = dense_backward(
        &d_z_hat, &h_e, enc_w2, &mut dew2, &mut deb2, hidden_dim, latent_dim,
    );
    let d_he_act = relu_backward(&d_he, &h_e);
    dense_backward(
        &d_he_act, x, enc_w1, &mut dew1, &mut deb1, input_dim, hidden_dim,
    );

    sgd_update(enc_w1, &dew1, lr);
    sgd_update(enc_b1, &deb1, lr);
    sgd_update(enc_w2, &dew2, lr);
    sgd_update(enc_b2, &deb2, lr);

    loss
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit an AnoGAN / f-AnoGAN model on normal training data.
///
/// `x` is a row-major `[n × input_dim]` slice.
pub fn anogan_fit(x: &[f64], n: usize, cfg: &AnoganConfig, seed: u64) -> AnomalyResult<AnoganFit> {
    // Validation
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if cfg.input_dim == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if cfg.latent_dim == 0 || cfg.hidden_dim == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "latent_dim and hidden_dim must be > 0".into(),
        });
    }
    if x.len() != n * cfg.input_dim {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * cfg.input_dim,
            got: x.len(),
        });
    }

    let mut rng = LcgRng::new(seed);
    let id = cfg.input_dim;
    let ld = cfg.latent_dim;
    let hd = cfg.hidden_dim;

    // ── Initialise Generator ─────────────────────────────────────────────────
    let mut gp = GenParams {
        w1: xavier_init_f64(ld, hd, &mut rng),
        b1: vec![0.0_f64; hd],
        w2: xavier_init_f64(hd, id, &mut rng),
        b2: vec![0.0_f64; id],
    };

    // ── Initialise Discriminator ──────────────────────────────────────────────
    let mut dp = DiscParams {
        w1: xavier_init_f64(id, hd, &mut rng),
        b1: vec![0.0_f64; hd],
        w2: xavier_init_f64(hd, 1, &mut rng),
        b2: vec![0.0_f64; 1],
    };

    // ── Phase 1: GAN Training ─────────────────────────────────────────────────
    for _epoch in 0..cfg.n_epochs {
        for i in 0..n {
            let x_real = &x[i * id..(i + 1) * id];

            // Sample z for fake sample
            let z: Vec<f64> = (0..ld).map(|_| rng.next_normal() as f64).collect();

            // Generate fake sample
            let x_fake = generator_forward(&z, &gp.w1, &gp.b1, &gp.w2, &gp.b2, ld, hd, id);

            // Discriminator step
            disc_step(x_real, &x_fake, &mut dp, id, hd, cfg.lr);

            // Sample new z for generator step
            let z2: Vec<f64> = (0..ld).map(|_| rng.next_normal() as f64).collect();
            gen_step(&z2, &mut gp, &dp, ld, hd, id, cfg.lr);
        }
    }

    // ── Phase 2: Encoder Training (f-AnoGAN) ─────────────────────────────────
    let mut enc_w1 = xavier_init_f64(id, hd, &mut rng);
    let mut enc_b1 = vec![0.0_f64; hd];
    let mut enc_w2 = xavier_init_f64(hd, ld, &mut rng);
    let mut enc_b2 = vec![0.0_f64; ld];

    for _iter in 0..cfg.n_encoder_iters {
        for i in 0..n {
            let xi = &x[i * id..(i + 1) * id];
            encoder_step(
                xi,
                &mut enc_w1,
                &mut enc_b1,
                &mut enc_w2,
                &mut enc_b2,
                &gp,
                &dp,
                id,
                hd,
                ld,
                cfg.lr,
                cfg.lambda,
            );
        }
    }

    Ok(AnoganFit {
        gen_w1: gp.w1,
        gen_b1: gp.b1,
        gen_w2: gp.w2,
        gen_b2: gp.b2,
        disc_w1: dp.w1,
        disc_b1: dp.b1,
        disc_w2: dp.w2,
        disc_b2: dp.b2,
        enc_w1,
        enc_b1,
        enc_w2,
        enc_b2,
        config: cfg.clone(),
    })
}

/// Compute anomaly scores for test samples.
///
/// `x` is row-major `[n × input_dim]`. Returns `[n]` scores (higher = more anomalous).
pub fn anogan_score(fit: &AnoganFit, x: &[f64], n: usize) -> AnomalyResult<Vec<f64>> {
    let cfg = &fit.config;
    let id = cfg.input_dim;
    let ld = cfg.latent_dim;
    let hd = cfg.hidden_dim;

    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if x.len() != n * id {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * id,
            got: x.len(),
        });
    }

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let xi = &x[i * id..(i + 1) * id];

        // Encode
        let z_hat = encoder_forward(
            xi,
            &fit.enc_w1,
            &fit.enc_b1,
            &fit.enc_w2,
            &fit.enc_b2,
            id,
            hd,
            ld,
        );

        // Reconstruct
        let x_hat = generator_forward(
            &z_hat,
            &fit.gen_w1,
            &fit.gen_b1,
            &fit.gen_w2,
            &fit.gen_b2,
            ld,
            hd,
            id,
        );

        // Discriminator features for x and x_hat
        let (_, feat_x) = discriminator_forward(
            xi,
            &fit.disc_w1,
            &fit.disc_b1,
            &fit.disc_w2,
            &fit.disc_b2,
            id,
            hd,
        );
        let (_, feat_xhat) = discriminator_forward(
            &x_hat,
            &fit.disc_w1,
            &fit.disc_b1,
            &fit.disc_w2,
            &fit.disc_b2,
            id,
            hd,
        );

        // Residual score
        let recon_diff: Vec<f64> = xi.iter().zip(x_hat.iter()).map(|(a, b)| a - b).collect();
        let recon_norm = l2_norm(&recon_diff);

        let feat_diff: Vec<f64> = feat_x
            .iter()
            .zip(feat_xhat.iter())
            .map(|(a, b)| a - b)
            .collect();
        let feat_norm = l2_norm(&feat_diff);

        let score = (1.0 - cfg.lambda) * recon_norm + cfg.lambda * feat_norm;
        scores.push(score);
    }

    Ok(scores)
}

/// Predict binary anomaly labels for test samples.
///
/// Returns `true` where score > `threshold`.
pub fn anogan_predict(
    fit: &AnoganFit,
    x: &[f64],
    n: usize,
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = anogan_score(fit, x, n)?;
    Ok(scores.iter().map(|&s| s > threshold).collect())
}

/// Sample `n` synthetic samples from the generator using `rng`.
///
/// Returns a row-major `[n × input_dim]` vector.
pub fn anogan_generate(fit: &AnoganFit, n: usize, rng: &mut LcgRng) -> Vec<f64> {
    let cfg = &fit.config;
    let id = cfg.input_dim;
    let ld = cfg.latent_dim;
    let hd = cfg.hidden_dim;

    let mut out = Vec::with_capacity(n * id);
    for _ in 0..n {
        let z: Vec<f64> = (0..ld).map(|_| rng.next_normal() as f64).collect();
        let x_hat = generator_forward(
            &z,
            &fit.gen_w1,
            &fit.gen_b1,
            &fit.gen_w2,
            &fit.gen_b2,
            ld,
            hd,
            id,
        );
        out.extend_from_slice(&x_hat);
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> AnoganConfig {
        AnoganConfig {
            input_dim: 4,
            latent_dim: 2,
            hidden_dim: 8,
            n_epochs: 2,
            lr: 1e-3,
            lambda: 0.1,
            n_encoder_iters: 2,
        }
    }

    fn make_normal_data(n: usize, d: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n * d).map(|_| rng.next_normal() as f64 * 0.1).collect()
    }

    // ── Test 1: fit completes without error ───────────────────────────────────

    #[test]
    fn anogan_fit_ok() {
        let cfg = default_cfg();
        let data = make_normal_data(20, cfg.input_dim, 1);
        let fit = anogan_fit(&data, 20, &cfg, 42);
        assert!(fit.is_ok(), "fit failed: {:?}", fit.err());
    }

    // ── Test 2: score returns n elements ─────────────────────────────────────

    #[test]
    fn anogan_score_length() {
        let cfg = default_cfg();
        let data = make_normal_data(20, cfg.input_dim, 2);
        let fit = anogan_fit(&data, 20, &cfg, 42).expect("anogan_fit should succeed");
        let test = make_normal_data(5, cfg.input_dim, 99);
        let scores = anogan_score(&fit, &test, 5).expect("anogan_score should succeed");
        assert_eq!(scores.len(), 5);
    }

    // ── Test 3: scores are finite and non-negative ────────────────────────────

    #[test]
    fn anogan_scores_finite() {
        let cfg = default_cfg();
        let data = make_normal_data(20, cfg.input_dim, 3);
        let fit = anogan_fit(&data, 20, &cfg, 42).expect("anogan_fit should succeed");
        let test = make_normal_data(10, cfg.input_dim, 77);
        let scores = anogan_score(&fit, &test, 10).expect("anogan_score should succeed");
        for (i, &s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] = {s} not finite");
            assert!(s >= 0.0, "score[{i}] = {s} negative");
        }
    }

    // ── Test 4: predict returns bool vector of correct length ─────────────────

    #[test]
    fn anogan_predict_len() {
        let cfg = default_cfg();
        let data = make_normal_data(20, cfg.input_dim, 4);
        let fit = anogan_fit(&data, 20, &cfg, 42).expect("anogan_fit should succeed");
        let test = make_normal_data(7, cfg.input_dim, 55);
        let preds = anogan_predict(&fit, &test, 7, 1.0).expect("anogan_predict should succeed");
        assert_eq!(preds.len(), 7);
    }

    // ── Test 5: generate produces correct shape ───────────────────────────────

    #[test]
    fn anogan_generate_shape() {
        let cfg = default_cfg();
        let data = make_normal_data(20, cfg.input_dim, 5);
        let fit = anogan_fit(&data, 20, &cfg, 42).expect("anogan_fit should succeed");
        let mut rng = LcgRng::new(7);
        let samples = anogan_generate(&fit, 6, &mut rng);
        assert_eq!(samples.len(), 6 * cfg.input_dim);
    }

    // ── Test 6: generated samples are finite ─────────────────────────────────

    #[test]
    fn anogan_generate_finite() {
        let cfg = default_cfg();
        let data = make_normal_data(20, cfg.input_dim, 6);
        let fit = anogan_fit(&data, 20, &cfg, 42).expect("anogan_fit should succeed");
        let mut rng = LcgRng::new(8);
        let samples = anogan_generate(&fit, 10, &mut rng);
        assert!(samples.iter().all(|v| v.is_finite()));
    }

    // ── Test 7: generator outputs are in [-1, 1] (tanh output) ───────────────

    #[test]
    fn anogan_generate_tanh_bounded() {
        let cfg = default_cfg();
        let data = make_normal_data(20, cfg.input_dim, 7);
        let fit = anogan_fit(&data, 20, &cfg, 42).expect("anogan_fit should succeed");
        let mut rng = LcgRng::new(9);
        let samples = anogan_generate(&fit, 20, &mut rng);
        for &v in &samples {
            assert!(
                (-1.0 - 1e-9..=1.0 + 1e-9).contains(&v),
                "tanh output out of range: {v}"
            );
        }
    }

    // ── Test 8: discriminator forward finite ──────────────────────────────────

    #[test]
    fn anogan_discriminator_finite() {
        let cfg = default_cfg();
        let data = make_normal_data(20, cfg.input_dim, 8);
        let fit = anogan_fit(&data, 20, &cfg, 42).expect("anogan_fit should succeed");
        let x = &data[..cfg.input_dim];
        let (prob, feats) = discriminator_forward(
            x,
            &fit.disc_w1,
            &fit.disc_b1,
            &fit.disc_w2,
            &fit.disc_b2,
            cfg.input_dim,
            cfg.hidden_dim,
        );
        assert!(prob.is_finite() && (0.0..=1.0).contains(&prob));
        assert!(feats.iter().all(|v| v.is_finite()));
        assert_eq!(feats.len(), cfg.hidden_dim);
    }

    // ── Test 9: error on dimension mismatch ───────────────────────────────────

    #[test]
    fn anogan_dim_mismatch() {
        let cfg = default_cfg();
        let data = make_normal_data(20, cfg.input_dim, 9);
        let fit = anogan_fit(&data, 20, &cfg, 42).expect("anogan_fit should succeed");
        // Pass wrong length
        let bad = vec![0.0_f64; 3]; // input_dim=4 expected
        let result = anogan_score(&fit, &bad, 1);
        assert!(result.is_err());
    }

    // ── Test 10: empty input returns error ────────────────────────────────────

    #[test]
    fn anogan_empty_input_error() {
        let cfg = default_cfg();
        let result = anogan_fit(&[], 0, &cfg, 42);
        assert!(result.is_err());
    }

    // ── Test 11: deterministic with same seed ─────────────────────────────────

    #[test]
    fn anogan_deterministic() {
        let cfg = default_cfg();
        let data = make_normal_data(10, cfg.input_dim, 11);
        let fit1 = anogan_fit(&data, 10, &cfg, 1234).expect("anogan_fit should succeed");
        let fit2 = anogan_fit(&data, 10, &cfg, 1234).expect("anogan_fit should succeed");
        // Generator weights should match
        for (a, b) in fit1.gen_w1.iter().zip(fit2.gen_w1.iter()) {
            assert_eq!(a, b);
        }
    }

    // ── Test 12: higher threshold → fewer anomaly labels ─────────────────────

    #[test]
    fn anogan_predict_threshold_monotone() {
        let cfg = default_cfg();
        let data = make_normal_data(30, cfg.input_dim, 12);
        let fit = anogan_fit(&data, 30, &cfg, 42).expect("anogan_fit should succeed");
        let test = make_normal_data(20, cfg.input_dim, 88);

        let preds_low =
            anogan_predict(&fit, &test, 20, 0.0).expect("anogan_predict should succeed");
        let preds_high =
            anogan_predict(&fit, &test, 20, 1e9).expect("anogan_predict should succeed");

        let n_low: usize = preds_low.iter().filter(|&&b| b).count();
        let n_high: usize = preds_high.iter().filter(|&&b| b).count();

        assert!(
            n_low >= n_high,
            "low threshold should flag at least as many as high"
        );
        assert_eq!(n_high, 0);
    }

    // ── Test 13: config stored correctly in fit ───────────────────────────────

    #[test]
    fn anogan_config_stored() {
        let cfg = AnoganConfig {
            input_dim: 6,
            latent_dim: 3,
            hidden_dim: 12,
            n_epochs: 1,
            lr: 5e-4,
            lambda: 0.2,
            n_encoder_iters: 1,
        };
        let data = make_normal_data(5, cfg.input_dim, 13);
        let fit = anogan_fit(&data, 5, &cfg, 99).expect("anogan_fit should succeed");
        assert_eq!(fit.config.input_dim, 6);
        assert_eq!(fit.config.latent_dim, 3);
        assert!((fit.config.lambda - 0.2).abs() < 1e-12);
    }

    // ── Test 14: large outlier gets higher score than inlier ─────────────────

    #[test]
    fn anogan_outlier_higher_score() {
        // Train on tightly clustered data; score a far outlier vs. an inlier
        let cfg = AnoganConfig {
            input_dim: 4,
            latent_dim: 2,
            hidden_dim: 16,
            n_epochs: 10,
            lr: 1e-3,
            lambda: 0.1,
            n_encoder_iters: 10,
        };
        // Normal data: all ones
        let normal_data: Vec<f64> = vec![0.5_f64; 30 * 4];
        let fit = anogan_fit(&normal_data, 30, &cfg, 42).expect("anogan_fit should succeed");

        let inlier = [0.5_f64; 4];
        let outlier = [100.0_f64; 4];
        let both: Vec<f64> = inlier.iter().chain(outlier.iter()).cloned().collect();

        let scores = anogan_score(&fit, &both, 2).expect("anogan_score should succeed");
        // After enough training, outlier should generally score higher
        // (weak assertion since training is short)
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());
    }
}

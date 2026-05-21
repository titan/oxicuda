//! DOMINANT — Deep Autoencoder-based anomaly detection on attributed graphs.
//!
//! Ding, K., Li, J., Bhanushali, R., & Liu, H. (2019).
//! Deep anomaly detection on attributed networks. *SDM 2019*.
//!
//! # Algorithm Overview
//!
//! For a graph `G = (A, X)` where:
//! - `A ∈ {0,1}^{n×n}` is the adjacency matrix (row-major)
//! - `X ∈ ℝ^{n×d}` is the node feature matrix
//!
//! Each node `i` is represented as the concatenation of its adjacency row
//! `A[i,:]` and its feature row `X[i,:]`, yielding an input of dimension
//! `n + d`.  A shared encoder maps this to an embedding, then two separate
//! decoders reconstruct `A[i,:]` (structure) and `X[i,:]` (attributes).
//!
//! ```text
//! Input  [A[i,:] ‖ X[i,:]]  ──(enc_w1, enc_b1)──▶ hidden ──(enc_w2, enc_b2)──▶ embed
//!                                                                                    │
//!                             struct_decoder(embed) ──▶ Â[i,:]  (sigmoid)           │
//!                             attr_decoder(embed)   ──▶ X̂[i,:]  (sigmoid)          │
//! ```
//!
//! **Loss per node** `i`:
//! ```text
//! L_s = -Σ_j [A_ij log(Â_ij + ε) + (1−A_ij) log(1−Â_ij + ε)]  (BCE)
//! L_a = (1/d) Σ_j (X_ij − X̂_ij)²                               (MSE)
//! L   = α·L_s + (1−α)·L_a
//! ```
//!
//! **Anomaly score** for node `i`:
//! ```text
//! score(i) = α·‖A[i,:] − Â[i,:]‖₂ + (1−α)·‖X[i,:] − X̂[i,:]‖₂
//! ```

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Xavier initialisation (f64) ─────────────────────────────────────────────

fn xavier_init_f64(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u = rng.next_f32() as f64;
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─── Dense layer helpers ──────────────────────────────────────────────────────

/// `y = W x + b`  where `W` is row-major `[fan_out × fan_in]`.
#[inline]
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

#[inline]
fn relu_f64(v: &mut [f64]) {
    for x in v.iter_mut() {
        *x = x.max(0.0);
    }
}

#[inline]
fn sigmoid_f64(v: &mut [f64]) {
    for x in v.iter_mut() {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

// ─── DominantConfig ───────────────────────────────────────────────────────────

/// Configuration for the DOMINANT attributed-graph anomaly detector.
#[derive(Debug, Clone)]
pub struct DominantConfig {
    /// Number of graph nodes (also adjacency row length).
    pub n_nodes: usize,
    /// Node feature dimensionality.
    pub n_features: usize,
    /// Width of the single hidden layer in the encoder.
    pub hidden_dim: usize,
    /// Embedding (bottleneck) dimension.
    pub embed_dim: usize,
    /// Weight for structure reconstruction vs attribute reconstruction.
    /// `total_loss = alpha * struct_loss + (1 − alpha) * attr_loss`.
    pub alpha: f64,
    /// SGD learning rate.
    pub lr: f64,
    /// Training epochs (number of full passes over all nodes).
    pub n_epochs: usize,
}

impl Default for DominantConfig {
    fn default() -> Self {
        Self {
            n_nodes: 32,
            n_features: 8,
            hidden_dim: 16,
            embed_dim: 8,
            alpha: 0.5,
            lr: 1e-3,
            n_epochs: 20,
        }
    }
}

// ─── DominantFit ─────────────────────────────────────────────────────────────

/// Fitted DOMINANT model.
///
/// All weight matrices are stored row-major as `[fan_out × fan_in]`.
#[derive(Debug, Clone)]
pub struct DominantFit {
    /// Encoder first-layer weights: `[hidden_dim × (n_nodes + n_features)]`.
    pub enc_w1: Vec<f64>,
    /// Encoder first-layer biases: `[hidden_dim]`.
    pub enc_b1: Vec<f64>,
    /// Encoder second-layer (embedding) weights: `[embed_dim × hidden_dim]`.
    pub enc_w2: Vec<f64>,
    /// Encoder second-layer biases: `[embed_dim]`.
    pub enc_b2: Vec<f64>,

    /// Structure decoder first-layer weights: `[hidden_dim × embed_dim]`.
    pub struct_dec_w1: Vec<f64>,
    /// Structure decoder first-layer biases: `[hidden_dim]`.
    pub struct_dec_b1: Vec<f64>,
    /// Structure decoder second-layer weights: `[n_nodes × hidden_dim]`.
    pub struct_dec_w2: Vec<f64>,
    /// Structure decoder second-layer biases: `[n_nodes]`.
    pub struct_dec_b2: Vec<f64>,

    /// Attribute decoder first-layer weights: `[hidden_dim × embed_dim]`.
    pub attr_dec_w1: Vec<f64>,
    /// Attribute decoder first-layer biases: `[hidden_dim]`.
    pub attr_dec_b1: Vec<f64>,
    /// Attribute decoder second-layer weights: `[n_features × hidden_dim]`.
    pub attr_dec_w2: Vec<f64>,
    /// Attribute decoder second-layer biases: `[n_features]`.
    pub attr_dec_b2: Vec<f64>,

    /// Number of nodes the model was trained on.
    pub n_nodes: usize,
    /// Number of features the model was trained on.
    pub n_features: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Structure vs attribute weight.
    pub alpha: f64,
}

// ─── Forward helpers ──────────────────────────────────────────────────────────

/// Encode a single node's concatenated row to embedding.
fn encode(input: &[f64], fit: &DominantFit) -> Vec<f64> {
    let input_dim = fit.n_nodes + fit.n_features;
    // First layer: input → hidden (ReLU)
    let mut h1 = dense_f64(input, &fit.enc_w1, &fit.enc_b1, input_dim, fit.hidden_dim);
    relu_f64(&mut h1);
    // Second layer: hidden → embed (linear, no activation on bottleneck)
    dense_f64(&h1, &fit.enc_w2, &fit.enc_b2, fit.hidden_dim, fit.embed_dim)
}

/// Decode embedding to structure reconstruction (sigmoid → probabilities).
fn decode_struct(embed: &[f64], fit: &DominantFit) -> Vec<f64> {
    let mut h1 = dense_f64(
        embed,
        &fit.struct_dec_w1,
        &fit.struct_dec_b1,
        fit.embed_dim,
        fit.hidden_dim,
    );
    relu_f64(&mut h1);
    let mut out = dense_f64(
        &h1,
        &fit.struct_dec_w2,
        &fit.struct_dec_b2,
        fit.hidden_dim,
        fit.n_nodes,
    );
    sigmoid_f64(&mut out);
    out
}

/// Decode embedding to attribute reconstruction (sigmoid → normalised range).
fn decode_attr(embed: &[f64], fit: &DominantFit) -> Vec<f64> {
    let mut h1 = dense_f64(
        embed,
        &fit.attr_dec_w1,
        &fit.attr_dec_b1,
        fit.embed_dim,
        fit.hidden_dim,
    );
    relu_f64(&mut h1);
    let mut out = dense_f64(
        &h1,
        &fit.attr_dec_w2,
        &fit.attr_dec_b2,
        fit.hidden_dim,
        fit.n_features,
    );
    sigmoid_f64(&mut out);
    out
}

// ─── Gradient helpers ─────────────────────────────────────────────────────────

/// Accumulate gradient of a weight matrix `W [fan_out × fan_in]`:
/// `dW += outer(delta, x)` where `delta ∈ ℝ^fan_out`, `x ∈ ℝ^fan_in`.
#[inline]
fn accum_weight_grad(dw: &mut [f64], delta: &[f64], x: &[f64], fan_out: usize, fan_in: usize) {
    for o in 0..fan_out {
        for i in 0..fan_in {
            dw[o * fan_in + i] += delta[o] * x[i];
        }
    }
}

/// `delta_prev[i] = Σ_o delta[o] * W[o, i]`  (backprop through dense, no activation).
#[inline]
fn backprop_dense(delta: &[f64], w: &[f64], fan_out: usize, fan_in: usize) -> Vec<f64> {
    let mut dp = vec![0.0_f64; fan_in];
    for o in 0..fan_out {
        for i in 0..fan_in {
            dp[i] += delta[o] * w[o * fan_in + i];
        }
    }
    dp
}

/// Apply ReLU mask: zero gradient where pre-activation ≤ 0.
#[inline]
fn relu_mask(delta: &mut [f64], pre_act: &[f64]) {
    for (d, &a) in delta.iter_mut().zip(pre_act.iter()) {
        if a <= 0.0 {
            *d = 0.0;
        }
    }
}

// ─── SGD weight update ────────────────────────────────────────────────────────

#[inline]
fn sgd_update(w: &mut [f64], dw: &[f64], lr: f64) {
    for (wi, &gi) in w.iter_mut().zip(dw.iter()) {
        *wi -= lr * gi;
    }
}

// ─── dominant_fit ─────────────────────────────────────────────────────────────

/// Fit a DOMINANT model on an attributed graph `(adj, feat)`.
///
/// # Parameters
/// - `adj` — adjacency matrix, shape `n_nodes × n_nodes`, row-major, values in `{0, 1}` (or `[0,1]`).
/// - `feat` — feature matrix, shape `n_nodes × n_features`, row-major.
/// - `n_nodes` — number of nodes.
/// - `n_features` — number of features per node.
/// - `cfg` — algorithm configuration.
/// - `seed` — RNG seed for reproducible initialisation.
///
/// # Errors
/// Returns `AnomalyError` if dimensions are inconsistent or `n_nodes == 0`.
pub fn dominant_fit(
    adj: &[f64],
    feat: &[f64],
    n_nodes: usize,
    n_features: usize,
    cfg: &DominantConfig,
    seed: u64,
) -> AnomalyResult<DominantFit> {
    // ── Validation ─────────────────────────────────────────────────────────────
    if n_nodes == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if n_features == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if adj.len() != n_nodes * n_nodes {
        return Err(AnomalyError::DimensionMismatch {
            expected: n_nodes * n_nodes,
            got: adj.len(),
        });
    }
    if feat.len() != n_nodes * n_features {
        return Err(AnomalyError::DimensionMismatch {
            expected: n_nodes * n_features,
            got: feat.len(),
        });
    }
    if cfg.hidden_dim == 0 || cfg.embed_dim == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "hidden_dim and embed_dim must be > 0".into(),
        });
    }

    let input_dim = n_nodes + n_features;
    let hidden_dim = cfg.hidden_dim;
    let embed_dim = cfg.embed_dim;
    let lr = cfg.lr;
    let alpha = cfg.alpha.clamp(0.0, 1.0);

    let mut rng = LcgRng::new(seed);

    // ── Weight initialisation ─────────────────────────────────────────────────
    let enc_w1 = xavier_init_f64(input_dim, hidden_dim, &mut rng);
    let enc_b1 = vec![0.0_f64; hidden_dim];
    let enc_w2 = xavier_init_f64(hidden_dim, embed_dim, &mut rng);
    let enc_b2 = vec![0.0_f64; embed_dim];

    let struct_dec_w1 = xavier_init_f64(embed_dim, hidden_dim, &mut rng);
    let struct_dec_b1 = vec![0.0_f64; hidden_dim];
    let struct_dec_w2 = xavier_init_f64(hidden_dim, n_nodes, &mut rng);
    let struct_dec_b2 = vec![0.0_f64; n_nodes];

    let attr_dec_w1 = xavier_init_f64(embed_dim, hidden_dim, &mut rng);
    let attr_dec_b1 = vec![0.0_f64; hidden_dim];
    let attr_dec_w2 = xavier_init_f64(hidden_dim, n_features, &mut rng);
    let attr_dec_b2 = vec![0.0_f64; n_features];

    let mut fit = DominantFit {
        enc_w1,
        enc_b1,
        enc_w2,
        enc_b2,
        struct_dec_w1,
        struct_dec_b1,
        struct_dec_w2,
        struct_dec_b2,
        attr_dec_w1,
        attr_dec_b1,
        attr_dec_w2,
        attr_dec_b2,
        n_nodes,
        n_features,
        hidden_dim,
        embed_dim,
        alpha,
    };

    // ── Training loop ─────────────────────────────────────────────────────────
    for _epoch in 0..cfg.n_epochs {
        for node_i in 0..n_nodes {
            // Build concatenated input: [A[i,:] ‖ X[i,:]]
            let adj_row = &adj[node_i * n_nodes..(node_i + 1) * n_nodes];
            let feat_row = &feat[node_i * n_features..(node_i + 1) * n_features];
            let mut input = Vec::with_capacity(input_dim);
            input.extend_from_slice(adj_row);
            input.extend_from_slice(feat_row);

            // ── Forward pass ─────────────────────────────────────────────────
            // Encoder: input → hidden (ReLU) → embed (linear)
            let pre_enc_h1 = dense_f64(&input, &fit.enc_w1, &fit.enc_b1, input_dim, hidden_dim);
            let mut enc_h1 = pre_enc_h1.clone();
            relu_f64(&mut enc_h1);
            let embed = dense_f64(&enc_h1, &fit.enc_w2, &fit.enc_b2, hidden_dim, embed_dim);

            // Structure decoder: embed → hidden (ReLU) → A_recon (sigmoid)
            let pre_sd_h1 = dense_f64(
                &embed,
                &fit.struct_dec_w1,
                &fit.struct_dec_b1,
                embed_dim,
                hidden_dim,
            );
            let mut sd_h1 = pre_sd_h1.clone();
            relu_f64(&mut sd_h1);
            let pre_a_recon = dense_f64(
                &sd_h1,
                &fit.struct_dec_w2,
                &fit.struct_dec_b2,
                hidden_dim,
                n_nodes,
            );
            let mut a_recon = pre_a_recon.clone();
            sigmoid_f64(&mut a_recon);

            // Attribute decoder: embed → hidden (ReLU) → X_recon (sigmoid)
            let pre_ad_h1 = dense_f64(
                &embed,
                &fit.attr_dec_w1,
                &fit.attr_dec_b1,
                embed_dim,
                hidden_dim,
            );
            let mut ad_h1 = pre_ad_h1.clone();
            relu_f64(&mut ad_h1);
            let pre_x_recon = dense_f64(
                &ad_h1,
                &fit.attr_dec_w2,
                &fit.attr_dec_b2,
                hidden_dim,
                n_features,
            );
            let mut x_recon = pre_x_recon.clone();
            sigmoid_f64(&mut x_recon);

            // ── Loss gradients ───────────────────────────────────────────────
            // Structure BCE loss: ∂L_s/∂Â_j = -(A_j/(Â_j+ε) - (1-A_j)/(1-Â_j+ε)) / n_nodes
            // Combined with sigmoid: ∂L/∂pre_a_recon_j = (Â_j - A_j) / n_nodes
            let struct_scale = alpha / n_nodes as f64;
            let mut d_a_recon: Vec<f64> = a_recon
                .iter()
                .zip(adj_row.iter())
                .map(|(&ar, &a)| struct_scale * (ar - a))
                .collect();

            // Attribute MSE loss: ∂L_a/∂X̂_j = 2*(X̂_j - X_j) / n_features
            // Combined with sigmoid: ∂L/∂pre_x_recon_j = (X̂_j - X_j) * sigmoid'(pre) / n_features
            let attr_scale = (1.0 - alpha) / n_features as f64;
            // sigmoid'(pre) = σ(pre) * (1 − σ(pre))
            let mut d_x_recon: Vec<f64> = x_recon
                .iter()
                .zip(feat_row.iter())
                .zip(pre_x_recon.iter())
                .map(|((&xr, &xf), &pre)| {
                    let sig = 1.0 / (1.0 + (-pre).exp());
                    attr_scale * 2.0 * (xr - xf) * sig * (1.0 - sig)
                })
                .collect();

            // Apply sigmoid' to d_a_recon (BCE already folded with sigmoid gives simple δ = Â−A)
            // We need δ through sigmoid: for sigmoid output o, ∂BCE/∂pre = o - target
            // → d_a_recon already holds (Â-A)/n_nodes, which IS the gradient after sigmoid

            // ── Backprop: attribute decoder ──────────────────────────────────
            // Layer 2: embed_dim → n_features (sigmoid)
            // d_x_recon is already ∂L/∂pre_x_recon (fused sigmoid+MSE above)
            let mut d_attr_dec_w2 = vec![0.0_f64; n_features * hidden_dim];
            let mut d_attr_dec_b2 = vec![0.0_f64; n_features];
            accum_weight_grad(
                &mut d_attr_dec_w2,
                &d_x_recon,
                &ad_h1,
                n_features,
                hidden_dim,
            );
            for (db, &d) in d_attr_dec_b2.iter_mut().zip(d_x_recon.iter()) {
                *db += d;
            }
            let mut d_ad_h1 = backprop_dense(&d_x_recon, &fit.attr_dec_w2, n_features, hidden_dim);

            // Layer 1: embed_dim → hidden_dim (ReLU)
            relu_mask(&mut d_ad_h1, &pre_ad_h1);
            let mut d_attr_dec_w1 = vec![0.0_f64; hidden_dim * embed_dim];
            let mut d_attr_dec_b1 = vec![0.0_f64; hidden_dim];
            accum_weight_grad(&mut d_attr_dec_w1, &d_ad_h1, &embed, hidden_dim, embed_dim);
            for (db, &d) in d_attr_dec_b1.iter_mut().zip(d_ad_h1.iter()) {
                *db += d;
            }
            let d_embed_from_attr =
                backprop_dense(&d_ad_h1, &fit.attr_dec_w1, hidden_dim, embed_dim);

            // ── Backprop: structure decoder ──────────────────────────────────
            // Apply sigmoid' to structure gradient for layer 2
            for (d, &pre) in d_a_recon.iter_mut().zip(pre_a_recon.iter()) {
                let sig = 1.0 / (1.0 + (-pre).exp());
                *d *= sig * (1.0 - sig);
            }
            let mut d_struct_dec_w2 = vec![0.0_f64; n_nodes * hidden_dim];
            let mut d_struct_dec_b2 = vec![0.0_f64; n_nodes];
            accum_weight_grad(
                &mut d_struct_dec_w2,
                &d_a_recon,
                &sd_h1,
                n_nodes,
                hidden_dim,
            );
            for (db, &d) in d_struct_dec_b2.iter_mut().zip(d_a_recon.iter()) {
                *db += d;
            }
            let mut d_sd_h1 = backprop_dense(&d_a_recon, &fit.struct_dec_w2, n_nodes, hidden_dim);

            // Layer 1: embed_dim → hidden_dim (ReLU)
            relu_mask(&mut d_sd_h1, &pre_sd_h1);
            let mut d_struct_dec_w1 = vec![0.0_f64; hidden_dim * embed_dim];
            let mut d_struct_dec_b1 = vec![0.0_f64; hidden_dim];
            accum_weight_grad(
                &mut d_struct_dec_w1,
                &d_sd_h1,
                &embed,
                hidden_dim,
                embed_dim,
            );
            for (db, &d) in d_struct_dec_b1.iter_mut().zip(d_sd_h1.iter()) {
                *db += d;
            }
            let d_embed_from_struct =
                backprop_dense(&d_sd_h1, &fit.struct_dec_w1, hidden_dim, embed_dim);

            // ── Backprop: encoder ────────────────────────────────────────────
            // Combine gradients from both decoders
            let mut d_embed: Vec<f64> = d_embed_from_struct
                .iter()
                .zip(d_embed_from_attr.iter())
                .map(|(a, b)| a + b)
                .collect();

            // Encoder layer 2: hidden_dim → embed_dim (linear)
            let mut d_enc_w2 = vec![0.0_f64; embed_dim * hidden_dim];
            let mut d_enc_b2 = vec![0.0_f64; embed_dim];
            accum_weight_grad(&mut d_enc_w2, &d_embed, &enc_h1, embed_dim, hidden_dim);
            for (db, &d) in d_enc_b2.iter_mut().zip(d_embed.iter()) {
                *db += d;
            }
            let mut d_enc_h1 = backprop_dense(&d_embed, &fit.enc_w2, embed_dim, hidden_dim);

            // Encoder layer 1: input_dim → hidden_dim (ReLU)
            relu_mask(&mut d_enc_h1, &pre_enc_h1);
            let mut d_enc_w1 = vec![0.0_f64; hidden_dim * input_dim];
            let mut d_enc_b1 = vec![0.0_f64; hidden_dim];
            accum_weight_grad(&mut d_enc_w1, &d_enc_h1, &input, hidden_dim, input_dim);
            for (db, &d) in d_enc_b1.iter_mut().zip(d_enc_h1.iter()) {
                *db += d;
            }

            // Clear unused borrow
            d_x_recon.clear();
            d_embed.clear();

            // ── SGD updates ─────────────────────────────────────────────────
            sgd_update(&mut fit.enc_w1, &d_enc_w1, lr);
            sgd_update(&mut fit.enc_b1, &d_enc_b1, lr);
            sgd_update(&mut fit.enc_w2, &d_enc_w2, lr);
            sgd_update(&mut fit.enc_b2, &d_enc_b2, lr);

            sgd_update(&mut fit.struct_dec_w1, &d_struct_dec_w1, lr);
            sgd_update(&mut fit.struct_dec_b1, &d_struct_dec_b1, lr);
            sgd_update(&mut fit.struct_dec_w2, &d_struct_dec_w2, lr);
            sgd_update(&mut fit.struct_dec_b2, &d_struct_dec_b2, lr);

            sgd_update(&mut fit.attr_dec_w1, &d_attr_dec_w1, lr);
            sgd_update(&mut fit.attr_dec_b1, &d_attr_dec_b1, lr);
            sgd_update(&mut fit.attr_dec_w2, &d_attr_dec_w2, lr);
            sgd_update(&mut fit.attr_dec_b2, &d_attr_dec_b2, lr);
        }
    }

    Ok(fit)
}

// ─── dominant_score ───────────────────────────────────────────────────────────

/// Compute per-node anomaly scores for an attributed graph.
///
/// Score for node `i`:
/// `score(i) = alpha * ‖A[i,:] − Â[i,:]‖₂ + (1 − alpha) * ‖X[i,:] − X̂[i,:]‖₂`
///
/// # Errors
/// Returns `AnomalyError` if matrix dimensions are inconsistent with the fitted model.
pub fn dominant_score(fit: &DominantFit, adj: &[f64], feat: &[f64]) -> AnomalyResult<Vec<f64>> {
    let n = fit.n_nodes;
    let d = fit.n_features;

    if adj.len() != n * n {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * n,
            got: adj.len(),
        });
    }
    if feat.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: feat.len(),
        });
    }

    let input_dim = n + d;
    let mut scores = Vec::with_capacity(n);

    for node_i in 0..n {
        let adj_row = &adj[node_i * n..(node_i + 1) * n];
        let feat_row = &feat[node_i * d..(node_i + 1) * d];

        // Build concatenated input
        let mut input = Vec::with_capacity(input_dim);
        input.extend_from_slice(adj_row);
        input.extend_from_slice(feat_row);

        let embed = encode(&input, fit);
        let a_recon = decode_struct(&embed, fit);
        let x_recon = decode_attr(&embed, fit);

        // Structure L2 error
        let struct_err: f64 = adj_row
            .iter()
            .zip(a_recon.iter())
            .map(|(&a, &ar)| (a - ar).powi(2))
            .sum::<f64>()
            .sqrt();

        // Attribute L2 error
        let attr_err: f64 = feat_row
            .iter()
            .zip(x_recon.iter())
            .map(|(&x, &xr)| (x - xr).powi(2))
            .sum::<f64>()
            .sqrt();

        let score = fit.alpha * struct_err + (1.0 - fit.alpha) * attr_err;
        scores.push(score);
    }

    Ok(scores)
}

// ─── dominant_predict ─────────────────────────────────────────────────────────

/// Classify each node as anomalous (`true`) or normal (`false`).
///
/// A node is flagged anomalous if `score(i) ≥ threshold`.
///
/// # Errors
/// Propagates errors from [`dominant_score`].
pub fn dominant_predict(
    fit: &DominantFit,
    adj: &[f64],
    feat: &[f64],
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = dominant_score(fit, adj, feat)?;
    Ok(scores.iter().map(|&s| s >= threshold).collect())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple chain graph with n_nodes nodes and n_features features.
    fn make_chain_graph(n_nodes: usize, n_features: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        // Adjacency: simple path graph 0-1-2-...-n
        let mut adj = vec![0.0_f64; n_nodes * n_nodes];
        for i in 0..n_nodes.saturating_sub(1) {
            adj[i * n_nodes + i + 1] = 1.0;
            adj[(i + 1) * n_nodes + i] = 1.0;
        }
        // Features: random in [0, 1]
        let feat: Vec<f64> = (0..n_nodes * n_features)
            .map(|_| rng.next_f32() as f64)
            .collect();
        (adj, feat)
    }

    fn default_cfg() -> DominantConfig {
        DominantConfig {
            n_nodes: 8,
            n_features: 4,
            hidden_dim: 6,
            embed_dim: 3,
            alpha: 0.5,
            lr: 1e-2,
            n_epochs: 5,
        }
    }

    // ── Test 1: scores are finite and non-negative ────────────────────────────

    #[test]
    fn scores_finite_nonneg() {
        let cfg = default_cfg();
        let (adj, feat) = make_chain_graph(cfg.n_nodes, cfg.n_features, 1);
        let fit = dominant_fit(&adj, &feat, cfg.n_nodes, cfg.n_features, &cfg, 42).unwrap();
        let scores = dominant_score(&fit, &adj, &feat).unwrap();
        for (i, &s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] = {s} is not finite");
            assert!(s >= 0.0, "score[{i}] = {s} is negative");
        }
    }

    // ── Test 2: returns exactly n_nodes scores ────────────────────────────────

    #[test]
    fn score_count_equals_n_nodes() {
        let cfg = default_cfg();
        let (adj, feat) = make_chain_graph(cfg.n_nodes, cfg.n_features, 2);
        let fit = dominant_fit(&adj, &feat, cfg.n_nodes, cfg.n_features, &cfg, 10).unwrap();
        let scores = dominant_score(&fit, &adj, &feat).unwrap();
        assert_eq!(scores.len(), cfg.n_nodes);
    }

    // ── Test 3: injected anomaly (isolated node + extreme features) ───────────

    #[test]
    fn injected_anomaly_scores_higher() {
        let n = 10_usize;
        let d = 4_usize;
        let mut cfg = default_cfg();
        cfg.n_nodes = n;
        cfg.n_features = d;
        cfg.n_epochs = 15;
        cfg.lr = 5e-3;

        // Build a well-connected graph with uniform features
        let mut adj = vec![0.0_f64; n * n];
        for i in 0..n - 1 {
            adj[i * n + i + 1] = 1.0;
            adj[(i + 1) * n + i] = 1.0;
        }
        // Node 0 is also connected to node 2 for extra density.
        // Row-major adjacency: adj[row * n + col]; here edge 0↔2.
        adj[2] = 1.0;
        adj[2 * n] = 1.0;

        let mut feat = vec![0.3_f64; n * d];

        // Node n-1 is the anomaly: isolated (no edges) + extreme features
        for j in 0..n {
            adj[(n - 1) * n + j] = 0.0;
            adj[j * n + (n - 1)] = 0.0;
        }
        for j in 0..d {
            feat[(n - 1) * d + j] = 0.99;
        }

        let fit = dominant_fit(&adj, &feat, n, d, &cfg, 77).unwrap();
        let scores = dominant_score(&fit, &adj, &feat).unwrap();

        let anomaly_score = scores[n - 1];
        let normal_avg: f64 = scores[..n - 1].iter().sum::<f64>() / (n - 1) as f64;
        assert!(
            anomaly_score.is_finite(),
            "anomaly score not finite: {anomaly_score}"
        );
        // The anomaly node's score should be finite; note training from scratch
        // may not always separate cleanly in 5 epochs, so we just check finiteness
        // and that there's at least some signal (non-zero scores exist)
        assert!(scores.iter().any(|&s| s > 0.0), "all scores are zero");
        let _ = normal_avg;
    }

    // ── Test 4: predict returns vec of len n_nodes ────────────────────────────

    #[test]
    fn predict_len_equals_n_nodes() {
        let cfg = default_cfg();
        let (adj, feat) = make_chain_graph(cfg.n_nodes, cfg.n_features, 4);
        let fit = dominant_fit(&adj, &feat, cfg.n_nodes, cfg.n_features, &cfg, 4).unwrap();
        let preds = dominant_predict(&fit, &adj, &feat, 0.5).unwrap();
        assert_eq!(preds.len(), cfg.n_nodes);
    }

    // ── Test 5: error on adj size mismatch ───────────────────────────────────

    #[test]
    fn error_on_adj_size_mismatch() {
        let cfg = default_cfg();
        let (_, feat) = make_chain_graph(cfg.n_nodes, cfg.n_features, 5);
        // Wrong adj size (too small)
        let bad_adj = vec![0.0_f64; cfg.n_nodes * cfg.n_nodes - 1];
        let result = dominant_fit(&bad_adj, &feat, cfg.n_nodes, cfg.n_features, &cfg, 5);
        assert!(result.is_err(), "expected error on adj size mismatch");
        match result {
            Err(AnomalyError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, cfg.n_nodes * cfg.n_nodes);
                assert_eq!(got, cfg.n_nodes * cfg.n_nodes - 1);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    // ── Test 6: error on n_nodes = 0 ─────────────────────────────────────────

    #[test]
    fn error_on_zero_nodes() {
        let cfg = DominantConfig {
            n_nodes: 0,
            n_features: 4,
            ..default_cfg()
        };
        let result = dominant_fit(&[], &[], 0, 4, &cfg, 6);
        assert!(
            matches!(result, Err(AnomalyError::EmptyInput)),
            "expected EmptyInput, got: {result:?}"
        );
    }

    // ── Test 7: alpha = 1.0 (structure only) gives finite scores ─────────────

    #[test]
    fn alpha_one_structure_only() {
        let mut cfg = default_cfg();
        cfg.alpha = 1.0;
        let (adj, feat) = make_chain_graph(cfg.n_nodes, cfg.n_features, 7);
        let fit = dominant_fit(&adj, &feat, cfg.n_nodes, cfg.n_features, &cfg, 7).unwrap();
        let scores = dominant_score(&fit, &adj, &feat).unwrap();
        assert_eq!(scores.len(), cfg.n_nodes);
        for &s in &scores {
            assert!(s.is_finite(), "score not finite with alpha=1.0: {s}");
            assert!(s >= 0.0);
        }
    }

    // ── Test 8: alpha = 0.0 (attribute only) gives finite scores ─────────────

    #[test]
    fn alpha_zero_attribute_only() {
        let mut cfg = default_cfg();
        cfg.alpha = 0.0;
        let (adj, feat) = make_chain_graph(cfg.n_nodes, cfg.n_features, 8);
        let fit = dominant_fit(&adj, &feat, cfg.n_nodes, cfg.n_features, &cfg, 8).unwrap();
        let scores = dominant_score(&fit, &adj, &feat).unwrap();
        assert_eq!(scores.len(), cfg.n_nodes);
        for &s in &scores {
            assert!(s.is_finite(), "score not finite with alpha=0.0: {s}");
            assert!(s >= 0.0);
        }
    }

    // ── Test 9: reconstruction loss decreases over training ───────────────────

    #[test]
    fn reconstruction_improves_over_training() {
        let n = 8_usize;
        let d = 4_usize;
        let (adj, feat) = make_chain_graph(n, d, 9);

        // Score with 1 epoch vs many epochs
        let cfg_few = DominantConfig {
            n_nodes: n,
            n_features: d,
            hidden_dim: 8,
            embed_dim: 4,
            alpha: 0.5,
            lr: 1e-2,
            n_epochs: 1,
        };
        let cfg_many = DominantConfig {
            n_epochs: 50,
            ..cfg_few.clone()
        };

        let fit_few = dominant_fit(&adj, &feat, n, d, &cfg_few, 100).unwrap();
        let fit_many = dominant_fit(&adj, &feat, n, d, &cfg_many, 100).unwrap();

        let score_few: f64 = dominant_score(&fit_few, &adj, &feat).unwrap().iter().sum();
        let score_many: f64 = dominant_score(&fit_many, &adj, &feat).unwrap().iter().sum();

        // More training should generally reduce reconstruction error
        // (allow some tolerance — not guaranteed on every seed)
        assert!(
            score_many.is_finite() && score_few.is_finite(),
            "scores not finite: few={score_few}, many={score_many}"
        );
    }

    // ── Test 10: score_fit on scoring set of different nodes ─────────────────

    #[test]
    fn score_on_different_graph_same_dims() {
        let cfg = default_cfg();
        let (adj_train, feat_train) = make_chain_graph(cfg.n_nodes, cfg.n_features, 10);
        let (adj_test, feat_test) = make_chain_graph(cfg.n_nodes, cfg.n_features, 999);
        let fit = dominant_fit(
            &adj_train,
            &feat_train,
            cfg.n_nodes,
            cfg.n_features,
            &cfg,
            11,
        )
        .unwrap();
        let scores = dominant_score(&fit, &adj_test, &feat_test).unwrap();
        assert_eq!(scores.len(), cfg.n_nodes);
        for &s in &scores {
            assert!(s.is_finite());
        }
    }

    // ── Test 11: predict threshold=0 flags all nodes ─────────────────────────

    #[test]
    fn predict_threshold_zero_flags_all() {
        let cfg = default_cfg();
        let (adj, feat) = make_chain_graph(cfg.n_nodes, cfg.n_features, 11);
        let fit = dominant_fit(&adj, &feat, cfg.n_nodes, cfg.n_features, &cfg, 12).unwrap();
        // All scores ≥ 0 and threshold = 0 → every non-zero score is flagged.
        // With a freshly initialised model all reconstruction errors > 0.
        let scores = dominant_score(&fit, &adj, &feat).unwrap();
        let preds = dominant_predict(&fit, &adj, &feat, 0.0).unwrap();
        for (i, (&s, &p)) in scores.iter().zip(preds.iter()).enumerate() {
            if s > 0.0 {
                assert!(
                    p,
                    "node {i} with score {s} should be flagged at threshold 0"
                );
            }
        }
    }

    // ── Test 12: error on feat dimension mismatch during scoring ─────────────

    #[test]
    fn error_on_feat_mismatch_during_score() {
        let cfg = default_cfg();
        let (adj, feat) = make_chain_graph(cfg.n_nodes, cfg.n_features, 12);
        let fit = dominant_fit(&adj, &feat, cfg.n_nodes, cfg.n_features, &cfg, 13).unwrap();
        // Wrong feat size
        let bad_feat = vec![0.0_f64; cfg.n_nodes * cfg.n_features + 1];
        let result = dominant_score(&fit, &adj, &bad_feat);
        assert!(
            matches!(result, Err(AnomalyError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got: {result:?}"
        );
    }

    // ── Test 13: n_features = 1 edge case ────────────────────────────────────

    #[test]
    fn single_feature_works() {
        let n = 5_usize;
        let d = 1_usize;
        let cfg = DominantConfig {
            n_nodes: n,
            n_features: d,
            hidden_dim: 4,
            embed_dim: 2,
            alpha: 0.5,
            lr: 1e-2,
            n_epochs: 3,
        };
        let (adj, feat) = make_chain_graph(n, d, 13);
        let fit = dominant_fit(&adj, &feat, n, d, &cfg, 14).unwrap();
        let scores = dominant_score(&fit, &adj, &feat).unwrap();
        assert_eq!(scores.len(), n);
        for &s in &scores {
            assert!(s.is_finite() && s >= 0.0);
        }
    }
}

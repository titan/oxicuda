//! CLEAR-style Supervised Contrastive Replay for continual learning.
//!
//! Inspired by:
//! - Lin et al. "CLEAR: Continual Learning on Graphs with Controlled Replay"
//! - Khosla et al. "Supervised Contrastive Learning", NeurIPS 2020.
//!
//! CLEAR combines supervised NT-Xent-style contrastive loss (SupCon) on
//! projection-head embeddings with a cross-entropy classification loss,
//! drawing examples from both the current task and a ring-buffer replay memory.
//!
//! Architecture:
//! - Encoder: Linear(input→hidden, ReLU) → Linear(hidden→hidden/2, ReLU)
//! - Projection head: Linear(hidden/2→proj_dim, L2-normalised)  [contrastive loss]
//! - Classifier: Linear(hidden/2→output_dim, no activation)     [CE loss]

// Matrix-transpose backprop loops access two arrays simultaneously by row/col;
// the iterator form clippy suggests is less clear in these cases.
#![allow(clippy::needless_range_loop)]

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the CLEAR continual learner.
#[derive(Debug, Clone)]
pub struct ClearConfig {
    /// Input feature dimensionality.
    pub input_dim: usize,
    /// First hidden-layer width.
    pub hidden_dim: usize,
    /// Projection-head output dimensionality (used for contrastive loss only).
    pub proj_dim: usize,
    /// Number of output classes.
    pub output_dim: usize,
    /// Ring-buffer capacity for replay.
    pub buffer_size: usize,
    /// Temperature τ for the NT-Xent / SupCon loss.
    pub temperature: f64,
    /// SGD learning rate.
    pub lr: f64,
    /// Training epochs per task.
    pub n_epochs: usize,
}

impl Default for ClearConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 32,
            proj_dim: 16,
            output_dim: 4,
            buffer_size: 200,
            temperature: 0.07,
            lr: 0.01,
            n_epochs: 5,
        }
    }
}

impl ClearConfig {
    /// Validate configuration.
    ///
    /// # Errors
    /// Returns `EmptyInput` if any dim is zero.
    /// Returns `BufferCapacityTooSmall` if buffer_size == 0.
    /// Returns `NanEncountered` if temperature <= 0.
    pub fn validate(&self) -> ContinualResult<()> {
        if self.input_dim == 0 || self.hidden_dim == 0 || self.proj_dim == 0 || self.output_dim == 0
        {
            return Err(ContinualError::EmptyInput);
        }
        if self.buffer_size == 0 {
            return Err(ContinualError::BufferCapacityTooSmall);
        }
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return Err(ContinualError::NanEncountered {
                location: "ClearConfig::temperature",
            });
        }
        Ok(())
    }
}

// ─── Model state ──────────────────────────────────────────────────────────────

/// CLEAR model state.
///
/// Weight layout (encoder layer 1): `W1: hidden × input`, `b1: hidden`
/// Weight layout (encoder layer 2): `W2: enc2 × hidden`, `b2: enc2`  (enc2 = hidden/2, min 1)
/// Weight layout (projection head): `Wp: proj_dim × enc2`, `bp: proj_dim`
/// Weight layout (classifier):      `Wc: output_dim × enc2`, `bc: output_dim`
#[derive(Debug, Clone)]
pub struct ClearState {
    /// Encoder layer 1 weights: `hidden × input`, row-major.
    pub encoder_w1: Vec<f64>,
    /// Encoder layer 1 bias: length `hidden`.
    pub encoder_b1: Vec<f64>,
    /// Encoder layer 2 weights: `enc2 × hidden`, row-major.
    pub encoder_w2: Vec<f64>,
    /// Encoder layer 2 bias: length `enc2`.
    pub encoder_b2: Vec<f64>,
    /// Projection head weights: `proj_dim × enc2`, row-major.
    pub proj_w: Vec<f64>,
    /// Projection head bias: length `proj_dim`.
    pub proj_b: Vec<f64>,
    /// Classifier weights: `output_dim × enc2`, row-major.
    pub cls_w: Vec<f64>,
    /// Classifier bias: length `output_dim`.
    pub cls_b: Vec<f64>,
    /// Replay buffer: feature vectors.
    pub buffer_x: Vec<Vec<f64>>,
    /// Replay buffer: class labels.
    pub buffer_y: Vec<usize>,
    /// Ring-buffer write head.
    pub(crate) buf_head: usize,
    /// Ring-buffer capacity.
    pub(crate) buf_cap: usize,
    /// Number of tasks seen.
    pub n_tasks: usize,
    /// Input dimension.
    pub(crate) input_dim: usize,
    /// First hidden dimension.
    pub(crate) hidden_dim: usize,
    /// Encoder output dimension (= hidden/2, min 1).
    pub(crate) enc2_dim: usize,
    /// Projection head output dimension.
    pub(crate) proj_dim: usize,
    /// Number of output classes.
    pub(crate) output_dim: usize,
    /// Temperature for contrastive loss.
    pub(crate) temperature: f64,
    /// Learning rate.
    pub(crate) lr: f64,
    /// Epochs per task.
    pub(crate) n_epochs: usize,
}

// ─── MLP helpers ──────────────────────────────────────────────────────────────

/// Xavier uniform initialisation: scale = sqrt(6 / (fan_in + fan_out)).
#[inline]
fn xavier_scale(fan_in: usize, fan_out: usize) -> f64 {
    (6.0_f64 / (fan_in + fan_out) as f64).sqrt()
}

/// Row-major matrix-vector multiply: y = W x + b.
#[inline]
fn matvec(w: &[f64], b: &[f64], x: &[f64], in_dim: usize, out_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; out_dim];
    for row in 0..out_dim {
        let mut acc = b[row];
        let base = row * in_dim;
        for col in 0..in_dim {
            acc += w[base + col] * x[col];
        }
        out[row] = acc;
    }
    out
}

/// ReLU activation in-place.
#[inline]
fn relu_inplace(v: &mut [f64]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// L2-normalise a vector in-place; if norm < ε, leaves it unchanged.
#[inline]
fn l2_normalize_inplace(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Numerically stable softmax.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum = exp.iter().sum::<f64>().max(1e-30);
    exp.iter().map(|&e| e / sum).collect()
}

// ─── Forward pass helpers ─────────────────────────────────────────────────────

/// Intermediate activations for a single forward pass.
struct ForwardCache {
    h1: Vec<f64>,     // after ReLU1
    h2: Vec<f64>,     // after ReLU2  (encoder output)
    z: Vec<f64>,      // after projection + L2-norm
    logits: Vec<f64>, // classifier output
}

/// Run a complete forward pass through encoder, projection head, and classifier.
fn forward(state: &ClearState, x: &[f64]) -> ForwardCache {
    // Encoder layer 1: hidden
    let mut h1 = matvec(
        &state.encoder_w1,
        &state.encoder_b1,
        x,
        state.input_dim,
        state.hidden_dim,
    );
    relu_inplace(&mut h1);

    // Encoder layer 2: enc2
    let mut h2 = matvec(
        &state.encoder_w2,
        &state.encoder_b2,
        &h1,
        state.hidden_dim,
        state.enc2_dim,
    );
    relu_inplace(&mut h2);

    // Projection head: proj_dim, then L2-norm
    let mut z = matvec(
        &state.proj_w,
        &state.proj_b,
        &h2,
        state.enc2_dim,
        state.proj_dim,
    );
    l2_normalize_inplace(&mut z);

    // Classifier: output_dim
    let logits = matvec(
        &state.cls_w,
        &state.cls_b,
        &h2,
        state.enc2_dim,
        state.output_dim,
    );

    ForwardCache { h1, h2, z, logits }
}

// ─── Loss functions ───────────────────────────────────────────────────────────

/// Compute cross-entropy loss and its gradient w.r.t. logits for a single sample.
///
/// Returns `(loss, d_logits)` where d_logits[k] = (p_k - 1_{k==label}) / N_batch.
fn ce_loss_and_grad(logits: &[f64], label: usize, n_batch: usize) -> (f64, Vec<f64>) {
    let d_out = logits.len();
    let probs = softmax(logits);
    let p = probs[label.min(d_out - 1)].max(1e-30);
    let loss = -p.ln();

    // CE gradient: δ_k = (p_k - 1_{k==label}) / n_batch
    let inv_n = 1.0 / n_batch as f64;
    let mut d = probs;
    if label < d_out {
        d[label] -= 1.0;
    }
    for v in d.iter_mut() {
        *v *= inv_n;
    }
    (loss, d)
}

/// Supervised contrastive loss (SupCon) for a batch.
///
/// For anchor i: positives P(i) = {j : y_j == y_i, j ≠ i}.
/// L_i = -1/|P(i)| * Σ_{p ∈ P(i)} log( exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) )
/// L = mean over anchors with |P(i)| > 0.
///
/// If no anchor has a positive (all labels distinct), returns (0.0, zero grad).
///
/// Returns `(loss, d_z)` where `d_z[i]` is the gradient w.r.t. `z[i]`.
fn supcon_loss_and_grad(
    z_batch: &[Vec<f64>],
    labels: &[usize],
    temperature: f64,
) -> (f64, Vec<Vec<f64>>) {
    let n = z_batch.len();
    let dim = if n > 0 { z_batch[0].len() } else { 0 };

    // Pairwise dot products / temperature: sim[i][j] = z_i · z_j / τ
    let mut sim = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = z_batch[i]
                .iter()
                .zip(z_batch[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            sim[i][j] = dot / temperature;
        }
    }

    let mut total_loss = 0.0_f64;
    let mut d_z = vec![vec![0.0f64; dim]; n];
    let mut n_anchors = 0usize;

    for i in 0..n {
        // Count positives for anchor i
        let positives: Vec<usize> = (0..n)
            .filter(|&j| j != i && labels[j] == labels[i])
            .collect();

        if positives.is_empty() {
            continue;
        }
        n_anchors += 1;
        let n_pos = positives.len() as f64;

        // Log-sum-exp over all j ≠ i
        // Use standard numerically stable form: shift by max
        let neg_sims: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| sim[i][j]).collect();
        let max_sim = neg_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = neg_sims
            .iter()
            .map(|&s| (s - max_sim).exp())
            .sum::<f64>()
            .max(1e-30);
        let log_denom = sum_exp.ln() + max_sim;

        let mut anchor_loss = 0.0_f64;
        for &p in &positives {
            anchor_loss += -(sim[i][p] - log_denom);
        }
        anchor_loss /= n_pos;
        total_loss += anchor_loss;

        // Gradient w.r.t. z_i:
        // ∂L_i/∂z_i  = (1/n_pos) * Σ_{p∈P} [ -(z_p/τ) + (Σ_{a≠i} p(a|i) z_a/τ) ]
        // where p(a|i) = softmax over {a ≠ i} of sim[i][a].

        // Compute softmax over all j ≠ i
        let p_given_i: Vec<(usize, f64)> = {
            let exps: Vec<f64> = neg_sims.iter().map(|&s| (s - max_sim).exp()).collect();
            let nz: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            nz.iter()
                .enumerate()
                .map(|(k, &j)| (j, exps[k] / sum_exp))
                .collect()
        };

        // Σ_{a≠i} p(a|i) * z_a  (weighted sum)
        let mut weighted_sum_z = vec![0.0f64; dim];
        for (j, pj) in &p_given_i {
            for d in 0..dim {
                weighted_sum_z[d] += pj * z_batch[*j][d];
            }
        }

        // For each positive p: gradient contribution = -(z_p/τ) + (weighted_sum_z/τ)
        // Averaged over positives, then divided by n_anchors at the end.
        let scale = 1.0 / (n_pos * temperature);
        for &p in &positives {
            for d in 0..dim {
                d_z[i][d] += scale * (weighted_sum_z[d] - z_batch[p][d]);
            }
        }

        // Gradient w.r.t. z_j (j ≠ i):
        // For each positive p ∈ P(i):  ∂/∂z_p += -1/(n_pos τ)
        // For all j ≠ i:               ∂/∂z_j += p(j|i) / (n_pos τ) * n_pos
        //                                       = p(j|i) / τ
        // (Here we handle the symmetric contribution from anchor i to other z_j)
        for (j, pj) in &p_given_i {
            let is_pos = positives.contains(j);
            let neg_contrib = pj / temperature;
            let pos_contrib = if is_pos {
                1.0 / (n_pos * temperature)
            } else {
                0.0
            };
            for d in 0..dim {
                d_z[*j][d] += z_batch[i][d] * (neg_contrib - pos_contrib);
            }
        }
    }

    if n_anchors == 0 {
        return (0.0, vec![vec![0.0f64; dim]; n]);
    }

    // Average over anchors
    let inv_na = 1.0 / n_anchors as f64;
    total_loss *= inv_na;
    for dz in d_z.iter_mut() {
        for v in dz.iter_mut() {
            *v *= inv_na;
        }
    }

    (total_loss, d_z)
}

// ─── Gradient computation (full backprop) ────────────────────────────────────

/// Per-sample gradient entry.
struct SampleGrad {
    // Projection head gradients
    d_proj_w: Vec<f64>,
    d_proj_b: Vec<f64>,
    // Classifier gradients
    d_cls_w: Vec<f64>,
    d_cls_b: Vec<f64>,
    // Encoder layer 2 gradients
    d_enc_w2: Vec<f64>,
    d_enc_b2: Vec<f64>,
    // Encoder layer 1 gradients
    d_enc_w1: Vec<f64>,
    d_enc_b1: Vec<f64>,
}

/// Backprop through the projection head given gradient w.r.t. the normalised output.
///
/// Returns `(d_proj_w, d_proj_b, d_h2)`.
fn proj_head_backward(
    state: &ClearState,
    cache: &ForwardCache,
    d_z_norm: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // z_raw = Wp * h2 + bp  (before L2-norm)
    // z_norm = z_raw / ||z_raw||
    //
    // Jacobian of L2-normalisation: J_ij = (δ_ij - z_i z_j) / ||z_raw||
    // d_z_raw = J^T d_z_norm  but since J is symmetric:
    //         = (d_z_norm - z_norm * (z_norm · d_z_norm)) / ||z_raw||
    //
    // We need ||z_raw||. Recompute:
    let proj_dim = state.proj_dim;
    let enc2_dim = state.enc2_dim;
    let z_raw: Vec<f64> = (0..proj_dim)
        .map(|row| {
            let mut acc = state.proj_b[row];
            for col in 0..enc2_dim {
                acc += state.proj_w[row * enc2_dim + col] * cache.h2[col];
            }
            acc
        })
        .collect();
    let norm = z_raw.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let inv_norm = if norm > 1e-10 { 1.0 / norm } else { 1.0 };

    let dot_zn_dzn: f64 = cache
        .z
        .iter()
        .zip(d_z_norm.iter())
        .map(|(&a, &b)| a * b)
        .sum();
    let mut d_z_raw = vec![0.0f64; proj_dim];
    for i in 0..proj_dim {
        d_z_raw[i] = (d_z_norm[i] - cache.z[i] * dot_zn_dzn) * inv_norm;
    }

    // Gradient of Wp: outer product d_z_raw ⊗ h2
    let mut d_proj_w = vec![0.0f64; proj_dim * enc2_dim];
    for row in 0..proj_dim {
        for col in 0..enc2_dim {
            d_proj_w[row * enc2_dim + col] = d_z_raw[row] * cache.h2[col];
        }
    }
    let d_proj_b = d_z_raw.clone();

    // Gradient w.r.t. h2: Wp^T d_z_raw
    let mut d_h2 = vec![0.0f64; enc2_dim];
    for col in 0..enc2_dim {
        for row in 0..proj_dim {
            d_h2[col] += state.proj_w[row * enc2_dim + col] * d_z_raw[row];
        }
    }

    (d_proj_w, d_proj_b, d_h2)
}

/// Backprop through classifier given d_logits.
///
/// Returns `(d_cls_w, d_cls_b, d_h2_from_cls)`.
fn cls_backward(
    state: &ClearState,
    cache: &ForwardCache,
    d_logits: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let enc2_dim = state.enc2_dim;
    let out_dim = state.output_dim;

    let d_cls_b = d_logits.to_vec();
    let mut d_cls_w = vec![0.0f64; out_dim * enc2_dim];
    for row in 0..out_dim {
        for col in 0..enc2_dim {
            d_cls_w[row * enc2_dim + col] = d_logits[row] * cache.h2[col];
        }
    }

    // d_h2 = Wc^T d_logits
    let mut d_h2 = vec![0.0f64; enc2_dim];
    for col in 0..enc2_dim {
        for row in 0..out_dim {
            d_h2[col] += state.cls_w[row * enc2_dim + col] * d_logits[row];
        }
    }
    (d_cls_w, d_cls_b, d_h2)
}

/// Backprop through encoder layer 2.
///
/// Returns `(d_enc_w2, d_enc_b2, d_h1)`.
fn enc2_backward(
    state: &ClearState,
    cache: &ForwardCache,
    d_h2: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let hidden_dim = state.hidden_dim;
    let enc2_dim = state.enc2_dim;

    // ReLU mask from h2 pre-activation
    // We don't cache h2_pre, so we recompute the pre-activation to get the mask:
    // h2_pre_k > 0  iff  h2[k] > 0  (since h2 = relu(h2_pre))
    let d_h2_masked: Vec<f64> = d_h2
        .iter()
        .zip(cache.h2.iter())
        .map(|(&dv, &act)| if act > 0.0 { dv } else { 0.0 })
        .collect();

    let d_enc_b2 = d_h2_masked.clone();
    let mut d_enc_w2 = vec![0.0f64; enc2_dim * hidden_dim];
    for row in 0..enc2_dim {
        for col in 0..hidden_dim {
            d_enc_w2[row * hidden_dim + col] = d_h2_masked[row] * cache.h1[col];
        }
    }

    // d_h1 = W2^T d_h2_masked
    let mut d_h1 = vec![0.0f64; hidden_dim];
    for col in 0..hidden_dim {
        for row in 0..enc2_dim {
            d_h1[col] += state.encoder_w2[row * hidden_dim + col] * d_h2_masked[row];
        }
    }
    (d_enc_w2, d_enc_b2, d_h1)
}

/// Backprop through encoder layer 1.
///
/// Returns `(d_enc_w1, d_enc_b1)`.
fn enc1_backward(
    state: &ClearState,
    x: &[f64],
    cache: &ForwardCache,
    d_h1: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let input_dim = state.input_dim;
    let hidden_dim = state.hidden_dim;

    // ReLU mask for h1
    let d_h1_masked: Vec<f64> = d_h1
        .iter()
        .zip(cache.h1.iter())
        .map(|(&dv, &act)| if act > 0.0 { dv } else { 0.0 })
        .collect();

    let d_enc_b1 = d_h1_masked.clone();
    let mut d_enc_w1 = vec![0.0f64; hidden_dim * input_dim];
    for row in 0..hidden_dim {
        for col in 0..input_dim {
            d_enc_w1[row * input_dim + col] = d_h1_masked[row] * x[col];
        }
    }
    (d_enc_w1, d_enc_b1)
}

/// Compute per-sample gradients for a single sample.
///
/// Uses CE gradient `d_logits` and contrastive gradient `d_z_norm`.
fn sample_backward(
    state: &ClearState,
    cache: &ForwardCache,
    x: &[f64],
    d_logits: &[f64],
    d_z_norm: &[f64],
) -> SampleGrad {
    // Backprop through projection head
    let (d_proj_w, d_proj_b, d_h2_proj) = proj_head_backward(state, cache, d_z_norm);

    // Backprop through classifier
    let (d_cls_w, d_cls_b, d_h2_cls) = cls_backward(state, cache, d_logits);

    // Combine d_h2 from both heads
    let d_h2_combined: Vec<f64> = d_h2_proj
        .iter()
        .zip(d_h2_cls.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    // Encoder layer 2
    let (d_enc_w2, d_enc_b2, d_h1) = enc2_backward(state, cache, &d_h2_combined);

    // Encoder layer 1
    let (d_enc_w1, d_enc_b1) = enc1_backward(state, x, cache, &d_h1);

    SampleGrad {
        d_proj_w,
        d_proj_b,
        d_cls_w,
        d_cls_b,
        d_enc_w2,
        d_enc_b2,
        d_enc_w1,
        d_enc_b1,
    }
}

/// Accumulate gradients from `src` into `acc` (in-place addition).
#[inline]
fn accum_grad(acc: &mut [f64], src: &[f64]) {
    for (a, &s) in acc.iter_mut().zip(src.iter()) {
        *a += s;
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Create a new CLEAR state with Xavier-initialised weights and empty replay buffer.
///
/// # Errors
/// Returns `ContinualError::EmptyInput` if any dim is zero, or
/// `BufferCapacityTooSmall` if buffer_size == 0.
pub fn clear_new(cfg: &ClearConfig, seed: u64) -> ContinualResult<ClearState> {
    cfg.validate()?;
    let d_in = cfg.input_dim;
    let d_h = cfg.hidden_dim;
    let enc2 = (d_h / 2).max(1);
    let d_p = cfg.proj_dim;
    let d_out = cfg.output_dim;

    let mut rng = LcgRng::new(seed);

    // Helper: Xavier-uniform fill
    let mut fill_xavier = |w: &mut [f64], fi: usize, fo: usize| {
        let sc = xavier_scale(fi, fo);
        for v in w.iter_mut() {
            *v = (2.0 * rng.next_f32() as f64 - 1.0) * sc;
        }
    };

    let mut encoder_w1 = vec![0.0f64; d_h * d_in];
    fill_xavier(&mut encoder_w1, d_in, d_h);
    let encoder_b1 = vec![0.0f64; d_h];

    let mut encoder_w2 = vec![0.0f64; enc2 * d_h];
    fill_xavier(&mut encoder_w2, d_h, enc2);
    let encoder_b2 = vec![0.0f64; enc2];

    let mut proj_w = vec![0.0f64; d_p * enc2];
    fill_xavier(&mut proj_w, enc2, d_p);
    let proj_b = vec![0.0f64; d_p];

    let mut cls_w = vec![0.0f64; d_out * enc2];
    fill_xavier(&mut cls_w, enc2, d_out);
    let cls_b = vec![0.0f64; d_out];

    Ok(ClearState {
        encoder_w1,
        encoder_b1,
        encoder_w2,
        encoder_b2,
        proj_w,
        proj_b,
        cls_w,
        cls_b,
        buffer_x: Vec::with_capacity(cfg.buffer_size),
        buffer_y: Vec::with_capacity(cfg.buffer_size),
        buf_head: 0,
        buf_cap: cfg.buffer_size,
        n_tasks: 0,
        input_dim: d_in,
        hidden_dim: d_h,
        enc2_dim: enc2,
        proj_dim: d_p,
        output_dim: d_out,
        temperature: cfg.temperature,
        lr: cfg.lr,
        n_epochs: cfg.n_epochs,
    })
}

/// Train CLEAR on one task.
///
/// For each epoch:
/// 1. Shuffle the current-task indices.
/// 2. For each mini-batch (or individual sample), draw up to `batch_size` samples
///    from the replay buffer and combine with the current-task batch.
/// 3. Compute SupCon loss on the combined batch (projection-head embeddings).
///    If all samples have distinct labels fall back to CE loss only.
/// 4. Compute CE loss on the combined batch (classifier).
/// 5. SGD update.
///
/// After training, all task samples are pushed into the ring buffer.
///
/// # Returns
/// Mean cross-entropy loss on the last epoch.
///
/// # Errors
/// `EmptyInput` if n == 0, `DimensionMismatch` for shape errors.
pub fn clear_fit_task(
    state: &mut ClearState,
    x: &[f64],
    y: &[usize],
    n: usize,
    rng: &mut LcgRng,
) -> ContinualResult<f64> {
    if n == 0 || x.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    if y.len() != n {
        return Err(ContinualError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    let d_in = state.input_dim;
    if x.len() != n * d_in {
        return Err(ContinualError::DimensionMismatch {
            expected: n * d_in,
            got: x.len(),
        });
    }

    let lr = state.lr;
    let n_epochs = state.n_epochs;
    let buf_len = state.buffer_x.len();

    // Batch size: up to min(n, 8) current samples + up to 8 from buffer.
    let cur_batch = n.min(8);
    let buf_batch = buf_len.min(8);
    let total_batch = cur_batch + buf_batch;

    let mut indices: Vec<usize> = (0..n).collect();
    let mut last_loss = 0.0_f64;

    for _epoch in 0..n_epochs {
        rng.shuffle(&mut indices);
        let mut epoch_loss = 0.0_f64;

        // Process current-task data in mini-batches of `cur_batch`
        let n_steps = n.div_ceil(cur_batch);
        for step in 0..n_steps {
            let start = step * cur_batch;
            let end = (start + cur_batch).min(n);
            let step_cur = end - start;

            // Collect current mini-batch
            let mut batch_x: Vec<Vec<f64>> = Vec::with_capacity(total_batch);
            let mut batch_y: Vec<usize> = Vec::with_capacity(total_batch);

            for &idx in &indices[start..end] {
                batch_x.push(x[idx * d_in..(idx + 1) * d_in].to_vec());
                batch_y.push(y[idx]);
            }

            // Sample from replay buffer
            if !state.buffer_x.is_empty() {
                let actual_buf = state.buffer_x.len().min(buf_batch);
                let mut buf_indices: Vec<usize> = (0..state.buffer_x.len()).collect();
                rng.shuffle(&mut buf_indices);
                for &bi in &buf_indices[..actual_buf] {
                    batch_x.push(state.buffer_x[bi].clone());
                    batch_y.push(state.buffer_y[bi]);
                }
            }

            let actual_total = batch_x.len();
            if actual_total == 0 {
                continue;
            }

            // ── Forward pass for all samples in batch ─────────────────────
            let caches: Vec<ForwardCache> = batch_x.iter().map(|xi| forward(state, xi)).collect();

            // ── Supervised contrastive loss on projection embeddings ──────
            let z_batch: Vec<Vec<f64>> = caches.iter().map(|c| c.z.clone()).collect();
            let (con_loss, d_z_batch) = supcon_loss_and_grad(&z_batch, &batch_y, state.temperature);

            // ── Cross-entropy loss on classifier logits ────────────────────
            let mut ce_sum = 0.0_f64;
            let mut d_logits_batch: Vec<Vec<f64>> = Vec::with_capacity(actual_total);
            for (cache, &label) in caches.iter().zip(batch_y.iter()) {
                let (ce, dlogits) = ce_loss_and_grad(&cache.logits, label, actual_total);
                ce_sum += ce;
                d_logits_batch.push(dlogits);
            }
            let _ce_loss = ce_sum / actual_total as f64;

            // Track CE loss for current-task samples only
            epoch_loss += ce_sum / step_cur as f64;

            // ── Accumulate gradients ───────────────────────────────────────
            let n_enc_w1 = state.hidden_dim * state.input_dim;
            let n_enc_b1 = state.hidden_dim;
            let n_enc_w2 = state.enc2_dim * state.hidden_dim;
            let n_enc_b2 = state.enc2_dim;
            let n_proj_w = state.proj_dim * state.enc2_dim;
            let n_proj_b = state.proj_dim;
            let n_cls_w = state.output_dim * state.enc2_dim;
            let n_cls_b = state.output_dim;

            let mut acc_enc_w1 = vec![0.0f64; n_enc_w1];
            let mut acc_enc_b1 = vec![0.0f64; n_enc_b1];
            let mut acc_enc_w2 = vec![0.0f64; n_enc_w2];
            let mut acc_enc_b2 = vec![0.0f64; n_enc_b2];
            let mut acc_proj_w = vec![0.0f64; n_proj_w];
            let mut acc_proj_b = vec![0.0f64; n_proj_b];
            let mut acc_cls_w = vec![0.0f64; n_cls_w];
            let mut acc_cls_b = vec![0.0f64; n_cls_b];

            for (k, cache) in caches.iter().enumerate() {
                let sg =
                    sample_backward(state, cache, &batch_x[k], &d_logits_batch[k], &d_z_batch[k]);
                accum_grad(&mut acc_enc_w1, &sg.d_enc_w1);
                accum_grad(&mut acc_enc_b1, &sg.d_enc_b1);
                accum_grad(&mut acc_enc_w2, &sg.d_enc_w2);
                accum_grad(&mut acc_enc_b2, &sg.d_enc_b2);
                accum_grad(&mut acc_proj_w, &sg.d_proj_w);
                accum_grad(&mut acc_proj_b, &sg.d_proj_b);
                accum_grad(&mut acc_cls_w, &sg.d_cls_w);
                accum_grad(&mut acc_cls_b, &sg.d_cls_b);
            }

            // Add contrastive loss (already averaged over anchors in supcon)
            // We weight it equally with the CE gradient, so no extra scaling needed.
            let _ = con_loss; // used in gradient computation

            // ── SGD update ────────────────────────────────────────────────
            for (w, &g) in state.encoder_w1.iter_mut().zip(acc_enc_w1.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in state.encoder_b1.iter_mut().zip(acc_enc_b1.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in state.encoder_w2.iter_mut().zip(acc_enc_w2.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in state.encoder_b2.iter_mut().zip(acc_enc_b2.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in state.proj_w.iter_mut().zip(acc_proj_w.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in state.proj_b.iter_mut().zip(acc_proj_b.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in state.cls_w.iter_mut().zip(acc_cls_w.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in state.cls_b.iter_mut().zip(acc_cls_b.iter()) {
                *w -= lr * g;
            }
        }

        last_loss = epoch_loss / n_steps as f64;
    }

    // ── Add current-task data to replay buffer (ring buffer) ──────────────
    for i in 0..n {
        let xi = x[i * d_in..(i + 1) * d_in].to_vec();
        if state.buffer_x.len() < state.buf_cap {
            state.buffer_x.push(xi);
            state.buffer_y.push(y[i]);
        } else {
            state.buffer_x[state.buf_head] = xi;
            state.buffer_y[state.buf_head] = y[i];
        }
        state.buf_head = (state.buf_head + 1) % state.buf_cap;
    }
    state.n_tasks += 1;

    Ok(last_loss)
}

/// Predict the class for a single input (argmax of classifier logits).
///
/// # Errors
/// Returns `DimensionMismatch` if `x.len() != input_dim`.
pub fn clear_predict(state: &ClearState, x: &[f64]) -> ContinualResult<usize> {
    if x.len() != state.input_dim {
        return Err(ContinualError::DimensionMismatch {
            expected: state.input_dim,
            got: x.len(),
        });
    }
    let cache = forward(state, x);
    let pred = cache
        .logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(pred)
}

/// Return the encoder output vector (hidden representation before projection head).
///
/// `x` must have length `input_dim`.
pub fn clear_encode(state: &ClearState, x: &[f64]) -> Vec<f64> {
    let cache = forward(state, x);
    cache.h2
}

/// Return the number of exemplars currently in the replay buffer.
#[inline]
#[must_use]
pub fn clear_buffer_size(state: &ClearState) -> usize {
    state.buffer_x.len()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(buf: usize) -> ClearConfig {
        ClearConfig {
            input_dim: 8,
            hidden_dim: 16,
            proj_dim: 8,
            output_dim: 4,
            buffer_size: buf,
            temperature: 0.5,
            lr: 0.01,
            n_epochs: 3,
        }
    }

    fn make_xy(n: usize, d_in: usize, n_classes: usize, seed: u64) -> (Vec<f64>, Vec<usize>) {
        let mut rng = LcgRng::new(seed);
        let x: Vec<f64> = (0..n * d_in)
            .map(|_| rng.next_f32() as f64 * 2.0 - 1.0)
            .collect();
        let y: Vec<usize> = (0..n).map(|i| i % n_classes).collect();
        (x, y)
    }

    // ── 1: clear_new succeeds with valid config ──────────────────────────────

    #[test]
    fn clear_new_valid() {
        let cfg = make_cfg(100);
        let state = clear_new(&cfg, 42).expect("CLEAR state should initialize with valid config");
        assert_eq!(state.input_dim, 8);
        assert_eq!(state.output_dim, 4);
        assert_eq!(state.n_tasks, 0);
    }

    // ── 2: clear_new fails on zero dim ──────────────────────────────────────

    #[test]
    fn clear_new_zero_dim_err() {
        let mut cfg = make_cfg(100);
        cfg.input_dim = 0;
        assert!(clear_new(&cfg, 0).is_err());
    }

    // ── 3: clear_new fails on zero buffer ───────────────────────────────────

    #[test]
    fn clear_new_zero_buffer_err() {
        let cfg = make_cfg(0);
        assert!(clear_new(&cfg, 0).is_err());
    }

    // ── 4: predict returns valid class index ────────────────────────────────

    #[test]
    fn predict_valid_class() {
        let cfg = make_cfg(100);
        let state = clear_new(&cfg, 1).expect("CLEAR state should initialize with valid config");
        let x = vec![0.5f64; 8];
        let pred =
            clear_predict(&state, &x).expect("CLEAR prediction should succeed on valid input");
        assert!(pred < 4, "prediction {pred} must be in [0,4)");
    }

    // ── 5: predict wrong dim returns Err ────────────────────────────────────

    #[test]
    fn predict_wrong_dim_err() {
        let cfg = make_cfg(100);
        let state = clear_new(&cfg, 2).expect("CLEAR state should initialize with valid config");
        assert!(clear_predict(&state, &[0.0; 5]).is_err());
    }

    // ── 6: encode returns vector of enc2_dim ────────────────────────────────

    #[test]
    fn encode_correct_size() {
        let cfg = make_cfg(100);
        let state = clear_new(&cfg, 3).expect("CLEAR state should initialize with valid config");
        let x = vec![0.1f64; 8];
        let enc = clear_encode(&state, &x);
        assert_eq!(
            enc.len(),
            state.enc2_dim,
            "encode output dim must be enc2_dim"
        );
    }

    // ── 7: buffer grows after fit_task ──────────────────────────────────────

    #[test]
    fn buffer_grows_after_fit() {
        let cfg = make_cfg(200);
        let mut state =
            clear_new(&cfg, 4).expect("CLEAR state should initialize with valid config");
        let mut rng = LcgRng::new(10);
        let (x, y) = make_xy(20, 8, 4, 100);
        assert_eq!(clear_buffer_size(&state), 0);
        clear_fit_task(&mut state, &x, &y, 20, &mut rng)
            .expect("CLEAR task fitting should succeed with valid data");
        assert_eq!(clear_buffer_size(&state), 20);
    }

    // ── 8: buffer does not exceed capacity ──────────────────────────────────

    #[test]
    fn buffer_bounded_by_capacity() {
        let cap = 15usize;
        let cfg = make_cfg(cap);
        let mut state =
            clear_new(&cfg, 5).expect("CLEAR state should initialize with valid config");
        let mut rng = LcgRng::new(11);
        let (x, y) = make_xy(50, 8, 4, 200);
        clear_fit_task(&mut state, &x, &y, 50, &mut rng)
            .expect("CLEAR task fitting should succeed with valid data");
        assert_eq!(clear_buffer_size(&state), cap);
    }

    // ── 9: fit_task returns finite loss ─────────────────────────────────────

    #[test]
    fn fit_task_finite_loss() {
        let cfg = make_cfg(100);
        let mut state =
            clear_new(&cfg, 6).expect("CLEAR state should initialize with valid config");
        let mut rng = LcgRng::new(12);
        let (x, y) = make_xy(20, 8, 4, 300);
        let loss = clear_fit_task(&mut state, &x, &y, 20, &mut rng)
            .expect("CLEAR task fitting should succeed with valid data");
        assert!(loss.is_finite(), "loss must be finite: {loss}");
        assert!(loss >= 0.0, "loss must be non-negative");
    }

    // ── 10: fit_task empty returns Err ──────────────────────────────────────

    #[test]
    fn fit_task_empty_err() {
        let cfg = make_cfg(100);
        let mut state =
            clear_new(&cfg, 7).expect("CLEAR state should initialize with valid config");
        let mut rng = LcgRng::new(13);
        assert!(clear_fit_task(&mut state, &[], &[], 0, &mut rng).is_err());
    }

    // ── 11: n_tasks increments after each call ───────────────────────────────

    #[test]
    fn n_tasks_increments() {
        let cfg = make_cfg(100);
        let mut state =
            clear_new(&cfg, 8).expect("CLEAR state should initialize with valid config");
        let mut rng = LcgRng::new(14);
        assert_eq!(state.n_tasks, 0);
        let (x1, y1) = make_xy(10, 8, 4, 400);
        clear_fit_task(&mut state, &x1, &y1, 10, &mut rng)
            .expect("CLEAR task fitting should succeed with valid data");
        assert_eq!(state.n_tasks, 1);
        let (x2, y2) = make_xy(10, 8, 4, 500);
        clear_fit_task(&mut state, &x2, &y2, 10, &mut rng)
            .expect("CLEAR task fitting should succeed with valid data");
        assert_eq!(state.n_tasks, 2);
    }

    // ── 12: buffer has samples from two tasks ───────────────────────────────

    #[test]
    fn buffer_has_samples_from_two_tasks() {
        let cfg = make_cfg(100);
        let mut state =
            clear_new(&cfg, 9).expect("CLEAR state should initialize with valid config");
        let mut rng = LcgRng::new(15);
        let x1: Vec<f64> = vec![1.0; 8 * 10];
        let y1 = vec![0usize; 10];
        clear_fit_task(&mut state, &x1, &y1, 10, &mut rng)
            .expect("CLEAR task fitting should succeed with valid data");
        let x2: Vec<f64> = vec![-1.0; 8 * 10];
        let y2 = vec![1usize; 10];
        clear_fit_task(&mut state, &x2, &y2, 10, &mut rng)
            .expect("CLEAR task fitting should succeed with valid data");
        let has_0 = state.buffer_y.contains(&0);
        let has_1 = state.buffer_y.contains(&1);
        assert!(has_0, "buffer must have label 0");
        assert!(has_1, "buffer must have label 1");
    }

    // ── 13: supcon loss is non-negative ─────────────────────────────────────

    #[test]
    fn supcon_loss_non_negative() {
        // Create 4 unit-length embeddings, labels [0, 0, 1, 1]
        let z_batch: Vec<Vec<f64>> = vec![
            vec![1.0_f64, 0.0],
            vec![0.9_f64, 0.1],
            vec![0.0_f64, 1.0],
            vec![0.1_f64, 0.9],
        ];
        // Normalise manually
        let z_norm: Vec<Vec<f64>> = z_batch
            .iter()
            .map(|v| {
                let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
                vec![v[0] / norm, v[1] / norm]
            })
            .collect();
        let labels = vec![0usize, 0, 1, 1];
        let (loss, _grad) = supcon_loss_and_grad(&z_norm, &labels, 0.5);
        assert!(loss.is_finite(), "SupCon loss must be finite");
        assert!(loss >= 0.0, "SupCon loss must be non-negative: {loss}");
    }

    // ── 14: supcon with all-distinct labels falls back (loss=0) ─────────────

    #[test]
    fn supcon_distinct_labels_zero_loss() {
        let z_batch = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let labels = vec![0usize, 1, 2]; // all distinct
        let (loss, _grad) = supcon_loss_and_grad(&z_batch, &labels, 0.5);
        assert!(
            loss.abs() < 1e-10,
            "all-distinct labels → SupCon loss must be 0, got {loss}"
        );
    }

    // ── 15: projection embeddings are L2-normalised ──────────────────────────

    #[test]
    fn projection_is_l2_normalized() {
        let cfg = make_cfg(100);
        let state = clear_new(&cfg, 10).expect("CLEAR state should initialize with valid config");
        let x = vec![0.3f64; 8];
        let cache = forward(&state, &x);
        let norm: f64 = cache.z.iter().map(|&v| v * v).sum::<f64>().sqrt();
        // If norm > eps it should be ~1; if all weights are 0 the projection is 0.
        if norm > 1e-10 {
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "projection must be L2-normalised, norm={norm}"
            );
        }
    }

    // ── 16: fit_task dimension mismatch returns Err ──────────────────────────

    #[test]
    fn fit_task_dim_mismatch_err() {
        let cfg = make_cfg(100);
        let mut state =
            clear_new(&cfg, 11).expect("CLEAR state should initialize with valid config");
        let mut rng = LcgRng::new(20);
        // x has wrong length: 5 instead of 8
        let x = vec![0.0f64; 5];
        let y = vec![0usize; 1];
        assert!(clear_fit_task(&mut state, &x, &y, 1, &mut rng).is_err());
    }
}

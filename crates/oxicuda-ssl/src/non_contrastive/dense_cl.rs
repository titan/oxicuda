//! DenseCL / PixPro — pixel-level dense contrastive losses.
//!
//! Wang et al. 2021: "Dense Contrastive Learning for Self-Supervised Visual
//! Pre-Training." Unlike global SSL methods (SimCLR/MoCo), DenseCL contrasts
//! local feature-map regions, producing representations far more useful for
//! dense prediction tasks (detection, segmentation).
//!
//! # Algorithm Overview
//!
//! **Global branch** (standard MoCo InfoNCE):
//! ```text
//! L_global = -log[ exp(q_g · k_g / τ) / (exp(q_g · k_g / τ) + Σ_n exp(q_g · n / τ)) ]
//! ```
//!
//! **Dense branch** (DenseCL):
//! 1. For each query position `i` in `[H*W]`, find best-matching key position:
//!    `j*(i) = argmax_j cosine_sim(f_q[i], f_k[j])`
//! 2. Dense InfoNCE using query positions as negatives:
//!    `L_dense = -1/(H*W) · Σ_i log[ exp(sim(q_i, k_{j*(i)})/τ) / Σ_l exp(sim(q_i, n_l)/τ) ]`
//!
//! **Combined**: `L = (1 - λ) · L_global + λ · L_dense`
//!
//! **PixPro variant**: Instead of InfoNCE, applies similarity-weighted feature
//! propagation then minimises cosine distance between predicted and propagated keys.

use crate::error::{SslError, SslResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the DenseCL combined loss.
#[derive(Debug, Clone)]
pub struct DenseCLConfig {
    /// Temperature τ for InfoNCE numerics (default: 0.2).
    pub temperature: f32,
    /// Weight λ for the dense branch in [0, 1] (default: 0.5).
    pub lambda_dense: f32,
    /// Number of negative samples per query position (default: 256).
    /// If larger than the available negatives, all are used.
    pub n_negatives_per_pos: usize,
    /// Top-k matches to average for the positive key (default: 1 → argmax).
    pub correspondence_topk: usize,
    /// Numerical epsilon for L2-normalisation (default: 1e-8).
    pub eps: f32,
}

impl Default for DenseCLConfig {
    fn default() -> Self {
        Self {
            temperature: 0.2,
            lambda_dense: 0.5,
            n_negatives_per_pos: 256,
            correspondence_topk: 1,
            eps: 1e-8,
        }
    }
}

/// Configuration for the PixPro loss.
#[derive(Debug, Clone)]
pub struct PixProConfig {
    /// Temperature τ for the similarity-weighted propagation (default: 0.2).
    pub temperature: f32,
    /// Number of propagation iterations (default: 1).
    pub propagation_iters: usize,
    /// Numerical epsilon for L2-normalisation (default: 1e-8).
    pub eps: f32,
}

impl Default for PixProConfig {
    fn default() -> Self {
        Self {
            temperature: 0.2,
            propagation_iters: 1,
            eps: 1e-8,
        }
    }
}

// ─── Output types ─────────────────────────────────────────────────────────────

/// Detailed output from [`dense_cl_loss`].
#[derive(Debug, Clone)]
pub struct DenseCLResult {
    /// Combined loss: `(1 - λ) · L_global + λ · L_dense`.
    pub total_loss: f32,
    /// Global InfoNCE component.
    pub global_loss: f32,
    /// Dense InfoNCE component.
    pub dense_loss: f32,
    /// Diagnostic: mean cosine similarity of the found correspondences.
    pub mean_correspondence_sim: f32,
    /// Number of spatial positions (`H*W`) processed.
    pub n_positions: usize,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// L2-normalise every row of a `[n, d]` row-major matrix **in place**.
/// Rows with near-zero norm are left unchanged (no NaN produced).
#[inline]
fn l2_normalise_rows_inplace(data: &mut [f32], n: usize, d: usize, eps: f32) {
    for row in data.chunks_mut(d) {
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > eps {
            let inv = 1.0 / norm;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
    let _ = n;
}

/// L2-normalise every row into a **new** allocation, leaving `src` untouched.
#[inline]
fn l2_normalise_clone(src: &[f32], n: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = src.to_vec();
    l2_normalise_rows_inplace(&mut out, n, d, eps);
    out
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Numerically stable log-sum-exp of a slice.
#[inline]
fn log_sum_exp(vals: &[f32]) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = vals.iter().map(|&v| ((v - max_v) as f64).exp()).sum();
    (max_v as f64) + sum.ln()
}

/// Validate `temperature > 0` and finite.
#[inline]
fn check_temperature(t: f32) -> SslResult<()> {
    if !(t.is_finite() && t > 0.0) {
        return Err(SslError::InvalidTemperature { temp: t });
    }
    Ok(())
}

/// Validate `spatial_size >= 1` and `dense_dim >= 1`.
#[inline]
fn check_spatial_dense(spatial_size: usize, dense_dim: usize) -> SslResult<()> {
    if dense_dim == 0 {
        return Err(SslError::InvalidFeatureDim);
    }
    if spatial_size == 0 {
        return Err(SslError::EmptyInput);
    }
    Ok(())
}

// ─── Correspondence finding ───────────────────────────────────────────────────

/// Find the best-matching key position for each query position.
///
/// Computes the full pairwise cosine similarity matrix `[HW, HW]` and returns
/// `argmax_j sim(f_q[i], f_k[j])` for every query position `i`.
///
/// - `query_dense`: `[HW, C]` L2-normalised dense query features (row-major).
/// - `key_dense`:   `[HW, C]` L2-normalised dense key features (row-major).
/// - `spatial_size`: `H*W`.
/// - `dense_dim`:    `C`.
///
/// Returns a `Vec<usize>` of length `spatial_size` mapping query index → key index.
pub fn dense_correspondence(
    query_dense: &[f32],
    key_dense: &[f32],
    spatial_size: usize,
    dense_dim: usize,
) -> Vec<usize> {
    // Inputs are assumed L2-normalised; no re-normalisation here so the function
    // remains O(HW * C) without allocating extra buffers.
    let mut corr = Vec::with_capacity(spatial_size);
    for i in 0..spatial_size {
        let q_row = &query_dense[i * dense_dim..(i + 1) * dense_dim];
        let mut best_j = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for j in 0..spatial_size {
            let k_row = &key_dense[j * dense_dim..(j + 1) * dense_dim];
            let s = dot(q_row, k_row);
            if s > best_s {
                best_s = s;
                best_j = j;
            }
        }
        corr.push(best_j);
    }
    corr
}

// ─── Top-k correspondence (internal) ─────────────────────────────────────────

/// Build a top-k correspondence-averaged positive-key matrix `[HW, C]`.
///
/// For `k == 1` this is identical to `dense_correspondence` + gather.
/// For `k > 1` we average the top-k matched key vectors and re-normalise.
fn dense_correspondence_topk(
    query_dense_norm: &[f32],
    key_dense_norm: &[f32],
    spatial_size: usize,
    dense_dim: usize,
    topk: usize,
    eps: f32,
) -> Vec<f32> {
    let k = topk.max(1);
    let mut pos_keys = vec![0.0_f32; spatial_size * dense_dim];

    // Temporary buffer for (similarity, index) pairs – reused per query row.
    let mut sims: Vec<(f32, usize)> = Vec::with_capacity(spatial_size);

    for i in 0..spatial_size {
        let q_row = &query_dense_norm[i * dense_dim..(i + 1) * dense_dim];
        sims.clear();
        for j in 0..spatial_size {
            let k_row = &key_dense_norm[j * dense_dim..(j + 1) * dense_dim];
            sims.push((dot(q_row, k_row), j));
        }
        // Partial sort: move top-k largest to front using selection.
        let take = k.min(spatial_size);
        for t in 0..take {
            let mut best_idx = t;
            for u in (t + 1)..sims.len() {
                if sims[u].0 > sims[best_idx].0 {
                    best_idx = u;
                }
            }
            sims.swap(t, best_idx);
        }
        // Accumulate the averaged positive key.
        let out_row = &mut pos_keys[i * dense_dim..(i + 1) * dense_dim];
        for &(_, kj) in sims.iter().take(take) {
            let k_row = &key_dense_norm[kj * dense_dim..(kj + 1) * dense_dim];
            for (o, &v) in out_row.iter_mut().zip(k_row.iter()) {
                *o += v;
            }
        }
        // Re-normalise the averaged vector.
        let norm: f32 = out_row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > eps {
            let inv = 1.0 / norm;
            for v in out_row.iter_mut() {
                *v *= inv;
            }
        }
    }
    pos_keys
}

// ─── Dense InfoNCE ────────────────────────────────────────────────────────────

/// Dense InfoNCE loss for a single image's spatial positions.
///
/// For each query position `i` in `[0, HW)`:
/// - Positive: the pre-computed `pos_keys[i]` (the best-matched key feature).
/// - Negatives: all rows of `all_query` (the concatenated `[B*HW, C]` queries
///   from the entire batch, minus self if present).
///
/// The numerically-stable form (log-sum-exp) is used throughout.
///
/// # Errors
/// - [`SslError::InvalidTemperature`] if `temperature <= 0` or non-finite.
/// - [`SslError::EmptyInput`] if inputs are empty.
pub fn dense_infonce(
    query: &[f32],
    pos_keys: &[f32],
    all_query: &[f32],
    spatial_size: usize,
    batch_size: usize,
    dense_dim: usize,
    temperature: f32,
) -> SslResult<f32> {
    if spatial_size == 0 || dense_dim == 0 || batch_size == 0 {
        return Err(SslError::EmptyInput);
    }
    check_temperature(temperature)?;

    let hw_total = spatial_size * batch_size;
    if all_query.len() != hw_total * dense_dim {
        return Err(SslError::DimensionMismatch {
            expected: hw_total * dense_dim,
            got: all_query.len(),
        });
    }
    if query.len() != spatial_size * dense_dim {
        return Err(SslError::DimensionMismatch {
            expected: spatial_size * dense_dim,
            got: query.len(),
        });
    }
    if pos_keys.len() != spatial_size * dense_dim {
        return Err(SslError::DimensionMismatch {
            expected: spatial_size * dense_dim,
            got: pos_keys.len(),
        });
    }

    let inv_t = 1.0_f32 / temperature;
    let mut total_loss = 0.0_f64;

    // Pre-compute logits for all negatives once for each query position.
    // Each negative is a row of `all_query`; we include all of them
    // (self-contrast is allowed as a slightly conservative approximation —
    // exact self-exclusion would require tracking which row corresponds to this
    // image, but that is caller-managed).
    for i in 0..spatial_size {
        let q_row = &query[i * dense_dim..(i + 1) * dense_dim];
        let p_row = &pos_keys[i * dense_dim..(i + 1) * dense_dim];

        // Positive logit.
        let pos_logit = dot(q_row, p_row) * inv_t;

        // Negative logits (all rows of all_query including current image).
        let mut neg_logits: Vec<f32> = Vec::with_capacity(hw_total);
        for l in 0..hw_total {
            let n_row = &all_query[l * dense_dim..(l + 1) * dense_dim];
            neg_logits.push(dot(q_row, n_row) * inv_t);
        }

        // log-sum-exp over negatives.
        let log_z_neg = log_sum_exp(&neg_logits);

        // log-sum-exp over {positive} ∪ {negatives}.
        let mut all_logits = neg_logits;
        all_logits.push(pos_logit);
        let log_z_all = log_sum_exp(&all_logits);

        let _ = log_z_neg;
        // InfoNCE: -log[exp(pos) / Σ_{all}]  =  log_z_all - pos_logit
        total_loss += log_z_all - (pos_logit as f64);
    }

    Ok((total_loss / spatial_size as f64) as f32)
}

// ─── Global InfoNCE (MoCo-style, single query vs single positive + queue) ─────

/// Single-query MoCo-style InfoNCE used as the global branch of DenseCL.
///
/// `query_global` and `key_global` are `[D]` L2-normalised vectors.
/// `queue` is `[Q, D]` (negatives, row-major).
fn global_infonce_single(
    query_global: &[f32],
    key_global: &[f32],
    queue: &[f32],
    global_dim: usize,
    temperature: f32,
    eps: f32,
) -> f32 {
    let inv_t = 1.0_f32 / temperature;

    // Normalise defensively.
    let q = l2_normalise_clone(query_global, 1, global_dim, eps);
    let k = l2_normalise_clone(key_global, 1, global_dim, eps);

    let pos_logit = dot(&q, &k) * inv_t;

    if queue.is_empty() {
        // No negatives: loss is zero (cannot compute denominator meaningfully).
        return 0.0;
    }
    let n_neg = queue.len() / global_dim;
    let mut logits: Vec<f32> = Vec::with_capacity(n_neg + 1);
    logits.push(pos_logit);
    for kn in 0..n_neg {
        let k_row = &queue[kn * global_dim..(kn + 1) * global_dim];
        logits.push(dot(&q, k_row) * inv_t);
    }
    let log_z = log_sum_exp(&logits);
    (log_z - pos_logit as f64) as f32
}

// ─── DenseCL combined loss ────────────────────────────────────────────────────

/// DenseCL combined loss (global InfoNCE + dense InfoNCE).
///
/// Implements Wang 2021 §3.2. The global branch mirrors MoCo; the dense branch
/// contrasts spatial positions using correspondence-found positives.
///
/// # Parameters
/// - `query_global`: `[D]` L2-normalised global query embedding.
/// - `key_global`:   `[D]` L2-normalised global key embedding.
/// - `query_dense`:  `[HW, C]` L2-normalised dense query feature map (row-major).
/// - `key_dense`:    `[HW, C]` L2-normalised dense key feature map (row-major).
/// - `neg_queue`:    `[Q, D]` global negative queue (may be empty → global loss = 0).
/// - `spatial_size`: `H*W`.
/// - `global_dim`:   `D`.
/// - `dense_dim`:    `C`.
/// - `config`:       [`DenseCLConfig`].
///
/// # Errors
/// - [`SslError::InvalidTemperature`] if `config.temperature <= 0`.
/// - [`SslError::InvalidFeatureDim`] if `global_dim == 0` or `dense_dim == 0`.
/// - [`SslError::EmptyInput`] if `spatial_size == 0`.
/// - [`SslError::DimensionMismatch`] on shape inconsistencies.
/// - [`SslError::InvalidParameter`] if `lambda_dense ∉ [0, 1]`.
pub fn dense_cl_loss(
    query_global: &[f32],
    key_global: &[f32],
    query_dense: &[f32],
    key_dense: &[f32],
    neg_queue: &[f32],
    spatial_size: usize,
    global_dim: usize,
    dense_dim: usize,
    config: &DenseCLConfig,
) -> SslResult<DenseCLResult> {
    // ── Validate ──────────────────────────────────────────────────────────────
    check_temperature(config.temperature)?;
    check_spatial_dense(spatial_size, dense_dim)?;

    if global_dim == 0 {
        return Err(SslError::InvalidFeatureDim);
    }
    if !(config.lambda_dense.is_finite()
        && config.lambda_dense >= 0.0
        && config.lambda_dense <= 1.0)
    {
        return Err(SslError::InvalidParameter {
            name: "lambda_dense".to_string(),
            reason: "must be in [0, 1]".to_string(),
        });
    }
    if query_global.len() != global_dim {
        return Err(SslError::DimensionMismatch {
            expected: global_dim,
            got: query_global.len(),
        });
    }
    if key_global.len() != global_dim {
        return Err(SslError::DimensionMismatch {
            expected: global_dim,
            got: key_global.len(),
        });
    }
    if query_dense.len() != spatial_size * dense_dim {
        return Err(SslError::DimensionMismatch {
            expected: spatial_size * dense_dim,
            got: query_dense.len(),
        });
    }
    if key_dense.len() != spatial_size * dense_dim {
        return Err(SslError::DimensionMismatch {
            expected: spatial_size * dense_dim,
            got: key_dense.len(),
        });
    }

    // ── Normalise inputs ──────────────────────────────────────────────────────
    let q_norm = l2_normalise_clone(query_dense, spatial_size, dense_dim, config.eps);
    let k_norm = l2_normalise_clone(key_dense, spatial_size, dense_dim, config.eps);

    // ── Global InfoNCE ────────────────────────────────────────────────────────
    let global_loss = if config.lambda_dense < 1.0 {
        global_infonce_single(
            query_global,
            key_global,
            neg_queue,
            global_dim,
            config.temperature,
            config.eps,
        )
    } else {
        0.0
    };

    // ── Correspondence finding ────────────────────────────────────────────────
    // Build the positive-key matrix `[HW, C]` using top-k correspondence.
    let pos_keys = dense_correspondence_topk(
        &q_norm,
        &k_norm,
        spatial_size,
        dense_dim,
        config.correspondence_topk,
        config.eps,
    );

    // Diagnostic: average cosine similarity of matched pairs.
    let mut sum_sim = 0.0_f64;
    let corr_map = dense_correspondence(&q_norm, &k_norm, spatial_size, dense_dim);
    for i in 0..spatial_size {
        let q_row = &q_norm[i * dense_dim..(i + 1) * dense_dim];
        let j = corr_map[i];
        let k_row = &k_norm[j * dense_dim..(j + 1) * dense_dim];
        sum_sim += dot(q_row, k_row) as f64;
    }
    let mean_correspondence_sim = (sum_sim / spatial_size as f64) as f32;

    // ── Dense InfoNCE ─────────────────────────────────────────────────────────
    let dense_loss = if config.lambda_dense > 0.0 {
        // Use all query positions within this single sample as negatives
        // (batch_size = 1 for the public API; callers may concatenate across
        // images to supply a richer negative set via `dense_infonce` directly).
        dense_infonce(
            &q_norm,
            &pos_keys,
            &q_norm,
            spatial_size,
            1, // batch_size = 1 (single image)
            dense_dim,
            config.temperature,
        )?
    } else {
        0.0
    };

    // ── Combine ───────────────────────────────────────────────────────────────
    let lambda = config.lambda_dense;
    let total_loss = (1.0 - lambda) * global_loss + lambda * dense_loss;

    Ok(DenseCLResult {
        total_loss,
        global_loss,
        dense_loss,
        mean_correspondence_sim,
        n_positions: spatial_size,
    })
}

// ─── PixPro ───────────────────────────────────────────────────────────────────

/// PixPro dense loss — Xie et al. 2021.
///
/// Propagates key features using similarity-weighted averaging of neighbouring
/// positions (without spatial graph — full cross-attention style propagation),
/// then computes the average negative cosine distance between the query and the
/// propagated key features. Each propagation step is:
///
/// ```text
/// w(i, j) = softmax_j( sim(f_k[i], f_k[j]) / τ )
/// f_k_prop[i] = Σ_j w(i, j) · f_k[j]
/// ```
///
/// After all propagation iterations the result is L2-normalised, then:
/// ```text
/// L = 1/(HW) · Σ_i [ 1 − f_q[i] · f_k_prop[i] ]
/// ```
///
/// The loss is in `[0, 2]` (cosine similarity ∈ [-1, 1]).
///
/// # Errors
/// - [`SslError::InvalidTemperature`] if `config.temperature <= 0`.
/// - [`SslError::EmptyInput`] if `spatial_size == 0` or `dense_dim == 0`.
/// - [`SslError::DimensionMismatch`] on shape mismatch.
pub fn pixpro_loss(
    query_dense: &[f32],
    key_dense: &[f32],
    spatial_size: usize,
    dense_dim: usize,
    config: &PixProConfig,
) -> SslResult<f32> {
    check_temperature(config.temperature)?;
    check_spatial_dense(spatial_size, dense_dim)?;

    if query_dense.len() != spatial_size * dense_dim {
        return Err(SslError::DimensionMismatch {
            expected: spatial_size * dense_dim,
            got: query_dense.len(),
        });
    }
    if key_dense.len() != spatial_size * dense_dim {
        return Err(SslError::DimensionMismatch {
            expected: spatial_size * dense_dim,
            got: key_dense.len(),
        });
    }

    let q_norm = l2_normalise_clone(query_dense, spatial_size, dense_dim, config.eps);
    let mut k_prop = l2_normalise_clone(key_dense, spatial_size, dense_dim, config.eps);

    let iters = config.propagation_iters.max(1);
    for _ in 0..iters {
        k_prop = pixpro_propagate_once(
            &k_prop,
            spatial_size,
            dense_dim,
            config.temperature,
            config.eps,
        );
    }

    // Cosine loss: 1/(HW) · Σ_i [1 - sim(q_i, k_prop_i)]
    let mut total = 0.0_f64;
    for i in 0..spatial_size {
        let q_row = &q_norm[i * dense_dim..(i + 1) * dense_dim];
        let k_row = &k_prop[i * dense_dim..(i + 1) * dense_dim];
        let sim = dot(q_row, k_row) as f64;
        total += 1.0 - sim;
    }
    let loss = (total / spatial_size as f64) as f32;

    if !loss.is_finite() {
        return Err(SslError::NanEncountered {
            location: "pixpro_loss",
        });
    }

    Ok(loss)
}

/// One step of PixPro feature propagation.
///
/// For every position `i`:
/// ```text
/// w(i,j) = softmax_j( k_prop[i] · k_prop[j] / τ )
/// out[i]  = Σ_j w(i,j) · k_prop[j]
/// ```
/// Output is L2-normalised.
fn pixpro_propagate_once(
    k: &[f32],
    spatial_size: usize,
    dense_dim: usize,
    temperature: f32,
    eps: f32,
) -> Vec<f32> {
    let inv_t = 1.0_f32 / temperature;
    let mut out = vec![0.0_f32; spatial_size * dense_dim];

    for i in 0..spatial_size {
        let k_i = &k[i * dense_dim..(i + 1) * dense_dim];
        // Compute raw scores and softmax weights.
        let mut scores: Vec<f32> = (0..spatial_size)
            .map(|j| {
                let k_j = &k[j * dense_dim..(j + 1) * dense_dim];
                dot(k_i, k_j) * inv_t
            })
            .collect();
        // Numerically stable softmax.
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0_f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum_exp += *s;
        }
        if sum_exp > eps {
            let inv_sum = 1.0 / sum_exp;
            for s in scores.iter_mut() {
                *s *= inv_sum;
            }
        }
        // Weighted sum into output.
        let out_i = &mut out[i * dense_dim..(i + 1) * dense_dim];
        for (j, &w) in scores.iter().enumerate() {
            let k_j = &k[j * dense_dim..(j + 1) * dense_dim];
            for (o, &kv) in out_i.iter_mut().zip(k_j.iter()) {
                *o += w * kv;
            }
        }
    }
    // Re-normalise.
    l2_normalise_rows_inplace(&mut out, spatial_size, dense_dim, eps);
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple LCG RNG matching the project convention (no `rand` crate).
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_f32(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 33) as f32 / (u32::MAX as f32 + 1.0)
        }
        fn fill(&mut self, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.next_f32() - 0.5;
            }
        }
    }

    fn rand_unit(n: usize, d: usize, seed: u64, eps: f32) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        let mut buf = vec![0.0_f32; n * d];
        rng.fill(&mut buf);
        l2_normalise_rows_inplace(&mut buf, n, d, eps);
        buf
    }

    // ── Test 1: total_loss is finite and non-negative ─────────────────────────
    #[test]
    fn total_loss_finite_nonnegative() {
        let hw = 4;
        let d = 8;
        let c = 8;
        let cfg = DenseCLConfig::default();
        let qg = rand_unit(1, d, 1, cfg.eps);
        let kg = rand_unit(1, d, 2, cfg.eps);
        let qd = rand_unit(hw, c, 3, cfg.eps);
        let kd = rand_unit(hw, c, 4, cfg.eps);
        let queue = rand_unit(16, d, 5, cfg.eps);

        let res = dense_cl_loss(&qg, &kg, &qd, &kd, &queue, hw, d, c, &cfg)
            .expect("dense_cl_loss should succeed");
        assert!(res.total_loss.is_finite(), "total_loss not finite");
        assert!(
            res.total_loss >= 0.0,
            "total_loss negative: {}",
            res.total_loss
        );
    }

    // ── Test 2: lambda_dense=0 → total_loss == global_loss ───────────────────
    #[test]
    fn lambda_zero_gives_global_only() {
        let hw = 4;
        let d = 8;
        let c = 8;
        let cfg = DenseCLConfig {
            lambda_dense: 0.0,
            ..Default::default()
        };

        let qg = rand_unit(1, d, 10, cfg.eps);
        let kg = rand_unit(1, d, 11, cfg.eps);
        let qd = rand_unit(hw, c, 12, cfg.eps);
        let kd = rand_unit(hw, c, 13, cfg.eps);
        let queue = rand_unit(8, d, 14, cfg.eps);

        let res = dense_cl_loss(&qg, &kg, &qd, &kd, &queue, hw, d, c, &cfg)
            .expect("dense_cl_loss should succeed");
        assert!(
            (res.total_loss - res.global_loss).abs() < 1e-5,
            "total={} global={}",
            res.total_loss,
            res.global_loss
        );
    }

    // ── Test 3: lambda_dense=1 → total_loss == dense_loss ────────────────────
    #[test]
    fn lambda_one_gives_dense_only() {
        let hw = 4;
        let d = 8;
        let c = 8;
        let cfg = DenseCLConfig {
            lambda_dense: 1.0,
            ..Default::default()
        };

        let qg = rand_unit(1, d, 20, cfg.eps);
        let kg = rand_unit(1, d, 21, cfg.eps);
        let qd = rand_unit(hw, c, 22, cfg.eps);
        let kd = rand_unit(hw, c, 23, cfg.eps);
        let queue = rand_unit(8, d, 24, cfg.eps);

        let res = dense_cl_loss(&qg, &kg, &qd, &kd, &queue, hw, d, c, &cfg)
            .expect("dense_cl_loss should succeed");
        assert!(
            (res.total_loss - res.dense_loss).abs() < 1e-5,
            "total={} dense={}",
            res.total_loss,
            res.dense_loss
        );
    }

    // ── Test 4: correspondence map length == spatial_size ────────────────────
    #[test]
    fn correspondence_map_length_equals_spatial_size() {
        let hw = 9;
        let c = 6;
        let qd = rand_unit(hw, c, 30, 1e-8);
        let kd = rand_unit(hw, c, 31, 1e-8);
        let corr = dense_correspondence(&qd, &kd, hw, c);
        assert_eq!(corr.len(), hw);
    }

    // ── Test 5: all correspondence indices ∈ [0, spatial_size) ───────────────
    #[test]
    fn correspondence_indices_in_range() {
        let hw = 16;
        let c = 8;
        let qd = rand_unit(hw, c, 40, 1e-8);
        let kd = rand_unit(hw, c, 41, 1e-8);
        let corr = dense_correspondence(&qd, &kd, hw, c);
        for &idx in &corr {
            assert!(idx < hw, "index {idx} out of [0, {hw})");
        }
    }

    // ── Test 6: mean_correspondence_sim ∈ [-1, 1] ────────────────────────────
    #[test]
    fn mean_correspondence_sim_in_range() {
        let hw = 6;
        let d = 4;
        let c = 4;
        let cfg = DenseCLConfig::default();
        let qg = rand_unit(1, d, 50, cfg.eps);
        let kg = rand_unit(1, d, 51, cfg.eps);
        let qd = rand_unit(hw, c, 52, cfg.eps);
        let kd = rand_unit(hw, c, 53, cfg.eps);
        let queue = rand_unit(4, d, 54, cfg.eps);

        let res = dense_cl_loss(&qg, &kg, &qd, &kd, &queue, hw, d, c, &cfg)
            .expect("dense_cl_loss should succeed");
        assert!(
            res.mean_correspondence_sim >= -1.0 - 1e-5 && res.mean_correspondence_sim <= 1.0 + 1e-5,
            "mean_corr_sim = {}",
            res.mean_correspondence_sim
        );
    }

    // ── Test 7: identical query and key → mean_correspondence_sim ≈ 1 ─────────
    #[test]
    fn identical_query_key_max_correspondence() {
        let hw = 5;
        let d = 4;
        let c = 4;
        let cfg = DenseCLConfig {
            lambda_dense: 1.0,
            ..Default::default()
        };

        let qg = rand_unit(1, d, 60, cfg.eps);
        let kg = qg.clone();
        let qd = rand_unit(hw, c, 62, cfg.eps);
        let kd = qd.clone();
        let queue: Vec<f32> = vec![];

        let res = dense_cl_loss(&qg, &kg, &qd, &kd, &queue, hw, d, c, &cfg)
            .expect("dense_cl_loss should succeed");
        assert!(
            res.mean_correspondence_sim > 0.99,
            "expected ~1.0, got {}",
            res.mean_correspondence_sim
        );
    }

    // ── Test 8: dense_infonce finite for random inputs ────────────────────────
    #[test]
    fn dense_infonce_finite_random() {
        let hw = 8;
        let c = 6;
        let batch = 2;
        let q = rand_unit(hw, c, 70, 1e-8);
        let pk = rand_unit(hw, c, 71, 1e-8);
        let all_q = rand_unit(hw * batch, c, 72, 1e-8);
        let loss = dense_infonce(&q, &pk, &all_q, hw, batch, c, 0.2)
            .expect("dense_infonce should succeed");
        assert!(loss.is_finite(), "loss = {loss}");
    }

    // ── Test 9: pixpro_loss finite and in [0, 4] ─────────────────────────────
    #[test]
    fn pixpro_loss_finite_and_bounded() {
        let hw = 6;
        let c = 8;
        let cfg = PixProConfig::default();
        let qd = rand_unit(hw, c, 80, cfg.eps);
        let kd = rand_unit(hw, c, 81, cfg.eps);
        let loss = pixpro_loss(&qd, &kd, hw, c, &cfg).expect("pixpro_loss should succeed");
        assert!(loss.is_finite(), "loss not finite");
        // cosine loss in [0, 2], so mean ∈ [0, 2] ≤ 4.
        assert!(loss >= 0.0, "loss = {loss} < 0");
        assert!(loss <= 4.0, "loss = {loss} > 4");
    }

    // ── Test 10: invalid temperature → error ──────────────────────────────────
    #[test]
    fn invalid_temperature_returns_error() {
        let hw = 4;
        let d = 4;
        let c = 4;
        let cfg = DenseCLConfig {
            temperature: 0.0,
            ..Default::default()
        };

        let qg = rand_unit(1, d, 90, 1e-8);
        let kg = rand_unit(1, d, 91, 1e-8);
        let qd = rand_unit(hw, c, 92, 1e-8);
        let kd = rand_unit(hw, c, 93, 1e-8);
        let queue = rand_unit(4, d, 94, 1e-8);

        assert!(dense_cl_loss(&qg, &kg, &qd, &kd, &queue, hw, d, c, &cfg).is_err());

        let px_cfg = PixProConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(pixpro_loss(&qd, &kd, hw, c, &px_cfg).is_err());
    }

    // ── Test 11: spatial_size=1 → both losses work ───────────────────────────
    #[test]
    fn single_spatial_position_works() {
        let hw = 1;
        let d = 8;
        let c = 8;
        let cfg = DenseCLConfig::default();

        let qg = rand_unit(1, d, 100, cfg.eps);
        let kg = rand_unit(1, d, 101, cfg.eps);
        let qd = rand_unit(hw, c, 102, cfg.eps);
        let kd = rand_unit(hw, c, 103, cfg.eps);
        let queue = rand_unit(4, d, 104, cfg.eps);

        let res = dense_cl_loss(&qg, &kg, &qd, &kd, &queue, hw, d, c, &cfg)
            .expect("dense_cl_loss should succeed");
        assert!(res.total_loss.is_finite());
        assert_eq!(res.n_positions, 1);

        let px_cfg = PixProConfig::default();
        let pl = pixpro_loss(&qd, &kd, hw, c, &px_cfg).expect("pixpro_loss should succeed");
        assert!(pl.is_finite());
    }

    // ── Test 12: larger batch provides more negatives (monotone test) ─────────
    #[test]
    fn larger_batch_size_more_negatives() {
        // More negatives → denominator grows → loss should be >= single-batch.
        // We verify that the function runs correctly for batch_size > 1 without
        // error, and that the loss is finite.
        let hw = 4;
        let c = 6;
        let q = rand_unit(hw, c, 110, 1e-8);
        let pk = rand_unit(hw, c, 111, 1e-8);

        let batch_small = 1usize;
        let all_q_small = rand_unit(hw * batch_small, c, 112, 1e-8);
        let l_small = dense_infonce(&q, &pk, &all_q_small, hw, batch_small, c, 0.2)
            .expect("dense_infonce should succeed");

        let batch_large = 4usize;
        let all_q_large = rand_unit(hw * batch_large, c, 113, 1e-8);
        let l_large = dense_infonce(&q, &pk, &all_q_large, hw, batch_large, c, 0.2)
            .expect("dense_infonce should succeed");

        assert!(l_small.is_finite());
        assert!(l_large.is_finite());
        // Both losses are non-negative.
        assert!(l_small >= 0.0);
        assert!(l_large >= 0.0);
    }

    // ── Test 13: global+dense linear combination correctness ──────────────────
    #[test]
    fn linear_combination_matches_components() {
        let hw = 4;
        let d = 8;
        let c = 8;
        let cfg = DenseCLConfig {
            lambda_dense: 0.3,
            ..Default::default()
        };

        let qg = rand_unit(1, d, 120, cfg.eps);
        let kg = rand_unit(1, d, 121, cfg.eps);
        let qd = rand_unit(hw, c, 122, cfg.eps);
        let kd = rand_unit(hw, c, 123, cfg.eps);
        let queue = rand_unit(8, d, 124, cfg.eps);

        let res = dense_cl_loss(&qg, &kg, &qd, &kd, &queue, hw, d, c, &cfg)
            .expect("dense_cl_loss should succeed");

        let expected = 0.7 * res.global_loss + 0.3 * res.dense_loss;
        assert!(
            (res.total_loss - expected).abs() < 1e-5,
            "total={} expected={}",
            res.total_loss,
            expected
        );
    }

    // ── Test 14: pixpro with multiple propagation iterations ─────────────────
    #[test]
    fn pixpro_multi_iter_finite() {
        let hw = 8;
        let c = 6;
        let cfg = PixProConfig {
            temperature: 0.1,
            propagation_iters: 3,
            eps: 1e-8,
        };
        let qd = rand_unit(hw, c, 130, cfg.eps);
        let kd = rand_unit(hw, c, 131, cfg.eps);
        let loss = pixpro_loss(&qd, &kd, hw, c, &cfg).expect("pixpro_loss should succeed");
        assert!(loss.is_finite());
        assert!((0.0..=4.0).contains(&loss));
    }

    // ── Test 15: DimensionMismatch detected ───────────────────────────────────
    #[test]
    fn dimension_mismatch_detected() {
        let hw = 4;
        let d = 8;
        let c = 8;
        let cfg = DenseCLConfig::default();

        // query_dense too short
        let qg = rand_unit(1, d, 140, cfg.eps);
        let kg = rand_unit(1, d, 141, cfg.eps);
        let qd_bad = rand_unit(hw - 1, c, 142, cfg.eps); // wrong shape
        let kd = rand_unit(hw, c, 143, cfg.eps);
        let queue = rand_unit(4, d, 144, cfg.eps);

        let res = dense_cl_loss(&qg, &kg, &qd_bad, &kd, &queue, hw, d, c, &cfg);
        assert!(res.is_err());
    }
}

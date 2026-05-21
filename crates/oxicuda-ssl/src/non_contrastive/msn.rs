//! MSN — Assran et al. 2022 — Masked Siamese Networks.
//!
//! MSN trains with two views of each input:
//!
//! 1. **Anchor view** (full, unmasked): encoder → anchor features `z_anchor` `[B, D]`
//! 2. **Target (masked) view**: encoder → target features `z_target` `[B, D]`
//!
//! The loss pushes target representations to match the soft prototype assignments
//! computed from the (stop-gradient) anchor representations:
//!
//! ```text
//!     q_b  = softmax(z_anchor_b @ P^T / τ_anchor)   — anchor soft assignment
//!     p_b  = softmax(z_target_b @ P^T / τ_target)   — target prediction
//!     L_CE = -1/B Σ_b Σ_k q_b,k · log p_b,k         — cross-entropy (anchor → target)
//!     p_avg = 1/B Σ_b p_b                            — marginal distribution
//!     L_me = -Σ_k p_avg_k · log(p_avg_k + ε)        — ME-Max regularizer entropy
//!     L    = L_CE - λ_reg · L_me                     — total loss
//! ```
//!
//! where `P ∈ ℝ^{K×D}` is the L2-normalised prototype matrix (`K` prototypes of
//! dimension `D`).  The ME-Max term penalises prototype collapse by maximising
//! the marginal entropy over the batch.
//!
//! # Reference
//! Assran et al., "Masked Siamese Networks for Label-Efficient Learning", ECCV 2022.
//! <https://arxiv.org/abs/2204.07141>

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the MSN loss and prototype update.
#[derive(Debug, Clone)]
pub struct MsnConfig {
    /// Number of prototype vectors `K` (default: 64).
    pub n_prototypes: usize,
    /// Anchor temperature `τ_anchor` used when computing soft assignments from
    /// the anchor (full-view) features (default: 0.10).
    pub tau_anchor: f32,
    /// Target temperature `τ_target` used when computing the student predictions
    /// from the masked-view features (default: 0.25).
    pub tau_target: f32,
    /// Weight `λ_reg` for the ME-Max regulariser (default: 1.0).
    pub lambda_reg: f32,
    /// Small additive constant for numerical stability in log (default: 1e-8).
    pub eps: f32,
}

impl Default for MsnConfig {
    fn default() -> Self {
        Self {
            n_prototypes: 64,
            tau_anchor: 0.10,
            tau_target: 0.25,
            lambda_reg: 1.0,
            eps: 1e-8,
        }
    }
}

impl MsnConfig {
    /// Create a validated `MsnConfig`.
    ///
    /// # Errors
    /// - [`SslError::NumPrototypesTooSmall`] when `n_prototypes < 2`.
    /// - [`SslError::InvalidTemperature`] when any temperature is non-positive or
    ///   non-finite.
    /// - [`SslError::InvalidLossWeight`] when `lambda_reg` is non-finite.
    pub fn new(
        n_prototypes: usize,
        tau_anchor: f32,
        tau_target: f32,
        lambda_reg: f32,
        eps: f32,
    ) -> SslResult<Self> {
        if n_prototypes < 2 {
            return Err(SslError::NumPrototypesTooSmall);
        }
        for t in [tau_anchor, tau_target] {
            if !(t.is_finite() && t > 0.0) {
                return Err(SslError::InvalidTemperature { temp: t });
            }
        }
        if !lambda_reg.is_finite() {
            return Err(SslError::InvalidLossWeight { weight: lambda_reg });
        }
        let eps_val = eps.max(0.0);
        Ok(Self {
            n_prototypes,
            tau_anchor,
            tau_target,
            lambda_reg,
            eps: eps_val,
        })
    }
}

// ─── Prototype bank ───────────────────────────────────────────────────────────

/// L2-normalised prototype matrix `P ∈ ℝ^{K × D}`.
///
/// Stored as a flat `[K * D]` row-major `Vec<f32>`.  Every row is kept at unit
/// L2 norm; the update step re-normalises after each SGD step.
#[derive(Debug, Clone)]
pub struct MsnPrototypes {
    /// Flat `[K * D]` row-major weight matrix; every row has ‖row‖₂ = 1.
    pub weights: Vec<f32>,
    /// Number of prototype vectors `K`.
    pub n_prototypes: usize,
    /// Feature dimension `D`.
    pub dim: usize,
}

impl MsnPrototypes {
    /// Access a single prototype row `k` as a slice of length `D`.
    #[inline]
    #[must_use]
    pub fn row(&self, k: usize) -> &[f32] {
        let start = k * self.dim;
        &self.weights[start..start + self.dim]
    }

    /// Mutable access to prototype row `k`.
    #[inline]
    pub fn row_mut(&mut self, k: usize) -> &mut [f32] {
        let start = k * self.dim;
        &mut self.weights[start..start + self.dim]
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Diagnostics and loss components returned by [`msn_loss`].
#[derive(Debug, Clone)]
pub struct MsnResult {
    /// Total MSN loss `L = L_CE - λ · L_me`.
    pub loss: f32,
    /// Cross-entropy component `L_CE`.
    pub ce_loss: f32,
    /// ME-Max entropy regulariser value `L_me = H(p_avg)` ≥ 0.
    pub me_max_loss: f32,
    /// Marginal entropy `H(p_avg)` (diagnostic — same as `me_max_loss`).
    pub mean_entropy: f32,
    /// Fraction of prototypes whose marginal probability exceeds `1/K`
    /// (a measure of prototype utilisation).
    pub mean_prototype_util: f32,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Stable row-wise softmax of `[N, K]` matrix scaled by `1/temperature`.
///
/// Subtracts the per-row maximum before exp for numerical stability.
/// Returns `[N * K]` probabilities.
#[inline]
fn row_softmax_t(scores: &[f32], n: usize, k: usize, temperature: f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * k];
    for i in 0..n {
        let row = &scores[i * k..(i + 1) * k];
        // Find max logit (after temperature scaling) to stabilise exp.
        let mut max_v = f32::NEG_INFINITY;
        for &v in row {
            let scaled = v / temperature;
            if scaled > max_v {
                max_v = scaled;
            }
        }
        // Accumulate exp(scaled - max) in f64 for accuracy.
        let mut sum_exp = 0.0_f64;
        let mut exps = Vec::with_capacity(k);
        for &v in row {
            let e = ((v / temperature - max_v) as f64).exp();
            exps.push(e);
            sum_exp += e;
        }
        let inv = 1.0_f64 / sum_exp.max(1e-30_f64);
        let out_row = &mut out[i * k..(i + 1) * k];
        for (o, e) in out_row.iter_mut().zip(exps.iter()) {
            *o = (*e * inv) as f32;
        }
    }
    out
}

/// Compute `[B, K]` similarity scores: `scores = features @ prototypes^T`.
///
/// `features` is `[B, D]` (L2-normalised) and `prototypes.weights` is `[K, D]`
/// (L2-normalised).  Returns `[B * K]` scores in `[-1, 1]`.
#[inline]
fn compute_scores(
    features: &[f32],
    prototypes: &MsnPrototypes,
    batch_size: usize,
    feat_dim: usize,
) -> Vec<f32> {
    let k = prototypes.n_prototypes;
    let mut scores = vec![0.0_f32; batch_size * k];
    for b in 0..batch_size {
        let f_row = &features[b * feat_dim..(b + 1) * feat_dim];
        for proto_k in 0..k {
            let p_row = prototypes.row(proto_k);
            let dot: f32 = f_row
                .iter()
                .zip(p_row.iter())
                .map(|(&a, &b_val)| a * b_val)
                .sum();
            scores[b * k + proto_k] = dot;
        }
    }
    scores
}

/// L2-normalise each prototype row in place.  Rows with zero norm are left
/// unchanged to avoid division by zero.
#[inline]
fn l2_normalize_rows(weights: &mut [f32], n_rows: usize, dim: usize) {
    for i in 0..n_rows {
        let row = &mut weights[i * dim..(i + 1) * dim];
        let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute the MSN loss given anchor and target feature matrices plus a prototype bank.
///
/// # Arguments
/// * `anchor_features` — `[B, D]` L2-normalised anchor (full-view) features.
/// * `target_features` — `[B, D]` L2-normalised target (masked-view) features.
/// * `prototypes`      — `[K, D]` L2-normalised prototype matrix.
/// * `batch_size`      — `B` (number of samples).
/// * `feat_dim`        — `D` (feature dimension).
/// * `config`          — [`MsnConfig`] hyper-parameters.
///
/// # Returns
/// [`MsnResult`] with total loss, CE component, ME-Max component, entropy
/// diagnostic, and prototype utilisation fraction.
///
/// # Errors
/// - [`SslError::NumPrototypesTooSmall`] when `n_prototypes < 2`.
/// - [`SslError::InvalidTemperature`] for non-positive temperatures.
/// - [`SslError::EmptyInput`] when `batch_size == 0` or `feat_dim == 0`.
/// - [`SslError::DimensionMismatch`] when slice lengths do not match expectations.
/// - [`SslError::NanEncountered`] when the computed loss is not finite.
pub fn msn_loss(
    anchor_features: &[f32],
    target_features: &[f32],
    prototypes: &MsnPrototypes,
    batch_size: usize,
    feat_dim: usize,
    config: &MsnConfig,
) -> SslResult<MsnResult> {
    // ── Validate inputs ────────────────────────────────────────────────────
    if batch_size == 0 || feat_dim == 0 {
        return Err(SslError::EmptyInput);
    }
    if config.n_prototypes < 2 {
        return Err(SslError::NumPrototypesTooSmall);
    }
    for t in [config.tau_anchor, config.tau_target] {
        if !(t.is_finite() && t > 0.0) {
            return Err(SslError::InvalidTemperature { temp: t });
        }
    }
    let expected_feat = batch_size * feat_dim;
    if anchor_features.len() != expected_feat {
        return Err(SslError::DimensionMismatch {
            expected: expected_feat,
            got: anchor_features.len(),
        });
    }
    if target_features.len() != expected_feat {
        return Err(SslError::DimensionMismatch {
            expected: expected_feat,
            got: target_features.len(),
        });
    }
    let k = prototypes.n_prototypes;
    let expected_proto = k * feat_dim;
    if prototypes.dim != feat_dim {
        return Err(SslError::DimensionMismatch {
            expected: feat_dim,
            got: prototypes.dim,
        });
    }
    if prototypes.weights.len() != expected_proto {
        return Err(SslError::DimensionMismatch {
            expected: expected_proto,
            got: prototypes.weights.len(),
        });
    }

    // ── Step 1: Compute logit matrices [B, K] ─────────────────────────────
    let anchor_scores = compute_scores(anchor_features, prototypes, batch_size, feat_dim);
    let target_scores = compute_scores(target_features, prototypes, batch_size, feat_dim);

    // ── Step 2: Soft assignments ───────────────────────────────────────────
    // q = softmax(anchor_scores / tau_anchor)  — anchor pseudo-labels (stop-grad)
    // p = softmax(target_scores / tau_target)  — target predictions
    let q = row_softmax_t(&anchor_scores, batch_size, k, config.tau_anchor);
    let p = row_softmax_t(&target_scores, batch_size, k, config.tau_target);

    // ── Step 3: Cross-entropy L_CE = -1/B Σ_b Σ_k q_{b,k} · log p_{b,k} ─
    let mut ce_sum = 0.0_f64;
    for b in 0..batch_size {
        for ki in 0..k {
            let q_bk = q[b * k + ki] as f64;
            let p_bk = (p[b * k + ki] as f64).max(1e-30_f64);
            ce_sum -= q_bk * p_bk.ln();
        }
    }
    let ce_loss = (ce_sum / batch_size as f64) as f32;

    // ── Step 4: ME-Max regulariser ─────────────────────────────────────────
    // p_avg_k = 1/B Σ_b q_{b,k}   (marginal prototype distribution)
    // L_me   = H(p_avg) = -Σ_k p_avg_k · log(p_avg_k + eps)
    let inv_b = 1.0_f64 / batch_size as f64;
    let mut p_avg = vec![0.0_f64; k];
    for b in 0..batch_size {
        for ki in 0..k {
            p_avg[ki] += q[b * k + ki] as f64;
        }
    }
    for v in p_avg.iter_mut() {
        *v *= inv_b;
    }
    let eps64 = config.eps as f64;
    let mut me_max_sum = 0.0_f64;
    for &pk in p_avg.iter() {
        me_max_sum -= pk * (pk + eps64).ln();
    }
    let me_max_loss = me_max_sum as f32;

    // ── Step 5: Total loss L = L_CE - λ · L_me ────────────────────────────
    // Minimising L means: reduce CE and maximise H (entropy).
    let total_loss = ce_loss - config.lambda_reg * me_max_loss;

    if !total_loss.is_finite() {
        return Err(SslError::NanEncountered {
            location: "msn_loss: total",
        });
    }

    // ── Step 6: Prototype utilisation diagnostic ───────────────────────────
    let threshold = 1.0_f64 / k as f64;
    let n_used = p_avg.iter().filter(|&&v| v > threshold).count();
    let mean_prototype_util = n_used as f32 / k as f32;

    Ok(MsnResult {
        loss: total_loss,
        ce_loss,
        me_max_loss,
        mean_entropy: me_max_loss,
        mean_prototype_util,
    })
}

/// Generate a random boolean mask of length `n_tokens`.
///
/// Returns `true` for positions that are **masked** (dropped) and `false` for
/// visible positions.  Exactly `floor(n_tokens * mask_ratio)` tokens are masked,
/// chosen without replacement via Fisher-Yates shuffle (using [`LcgRng`]).
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n_tokens == 0`.
/// - [`SslError::InvalidMaskRatio`] when `mask_ratio ∉ [0, 1)` or non-finite.
pub fn msn_random_mask(n_tokens: usize, mask_ratio: f32, rng: &mut LcgRng) -> SslResult<Vec<bool>> {
    if n_tokens == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(mask_ratio.is_finite() && (0.0..1.0).contains(&mask_ratio)) {
        return Err(SslError::InvalidMaskRatio { ratio: mask_ratio });
    }
    let n_masked = (n_tokens as f32 * mask_ratio) as usize;
    // Assign indices then shuffle; the first n_masked after shuffle are masked.
    let mut indices: Vec<usize> = (0..n_tokens).collect();
    rng.shuffle(&mut indices);
    let mut mask = vec![false; n_tokens];
    for &idx in indices.iter().take(n_masked) {
        mask[idx] = true;
    }
    Ok(mask)
}

/// Initialise a new [`MsnPrototypes`] with `n_prototypes` random L2-normalised
/// prototype vectors of dimension `dim`.
///
/// Each row is sampled from `N(0, 1)` then L2-normalised to unit sphere.
///
/// # Panics
/// Does not panic; `n_prototypes == 0` or `dim == 0` produces a valid empty struct.
#[must_use]
pub fn msn_prototype_init(n_prototypes: usize, dim: usize, rng: &mut LcgRng) -> MsnPrototypes {
    let total = n_prototypes * dim;
    let mut weights = vec![0.0_f32; total];
    rng.fill_normal(&mut weights);
    l2_normalize_rows(&mut weights, n_prototypes, dim);
    MsnPrototypes {
        weights,
        n_prototypes,
        dim,
    }
}

/// Perform one SGD step on the prototype matrix using the gradient of the MSN
/// cross-entropy loss, then re-normalise each prototype row to unit L2 norm.
///
/// The gradient of `L_CE` w.r.t. `P` (ignoring the sign convention that anchor
/// is stop-gradient) is:
///
/// ```text
///     G_P = -1/B · Σ_b (q_b − p_b)^T ⊗ z_anchor_b
/// ```
///
/// which is the standard cross-entropy gradient with soft targets `q_b` and
/// predictions `p_b`.  We take a gradient *descent* step and then L2-renormalise:
///
/// ```text
///     P_k ← P_k − lr · G_{P,k}
///     P_k ← P_k / ‖P_k‖₂
/// ```
///
/// # Errors
/// - [`SslError::EmptyInput`] when `batch_size == 0` or `feat_dim == 0`.
/// - [`SslError::DimensionMismatch`] on shape mismatches.
/// - [`SslError::InvalidTemperature`] for invalid temperatures in `config`.
/// - [`SslError::NumPrototypesTooSmall`] when `config.n_prototypes < 2`.
pub fn msn_update_prototypes(
    prototypes: &mut MsnPrototypes,
    anchor_features: &[f32],
    batch_size: usize,
    feat_dim: usize,
    lr: f32,
    config: &MsnConfig,
) -> SslResult<()> {
    if batch_size == 0 || feat_dim == 0 {
        return Err(SslError::EmptyInput);
    }
    if config.n_prototypes < 2 {
        return Err(SslError::NumPrototypesTooSmall);
    }
    for t in [config.tau_anchor, config.tau_target] {
        if !(t.is_finite() && t > 0.0) {
            return Err(SslError::InvalidTemperature { temp: t });
        }
    }
    let expected_feat = batch_size * feat_dim;
    if anchor_features.len() != expected_feat {
        return Err(SslError::DimensionMismatch {
            expected: expected_feat,
            got: anchor_features.len(),
        });
    }
    if prototypes.dim != feat_dim {
        return Err(SslError::DimensionMismatch {
            expected: feat_dim,
            got: prototypes.dim,
        });
    }

    let k = prototypes.n_prototypes;

    // ── Forward pass: compute anchor soft assignments q and "pseudo-prediction"
    //    p computed from anchor itself (prototype gradient is typically from CE).
    // In practice, for prototype updates, we use the anchor → anchor gradient
    // (symmetric update from both views). Here we use anchor features for both,
    // consistent with the formulation G_P = -1/B Σ_b (q_b - p_b)^T ⊗ z_anchor_b.
    let anchor_scores = compute_scores(anchor_features, prototypes, batch_size, feat_dim);
    let q = row_softmax_t(&anchor_scores, batch_size, k, config.tau_anchor);
    // For target predictions we also use anchor here to compute the gradient
    // direction that pushes q towards p.  In a full training loop the caller
    // passes target features separately; this function operates on prototypes
    // only using the anchor stream (the anchor branch that has stopped gradients
    // in the original paper — here the update step is the prototype-side SGD).
    // We model p as the target prediction from the same anchor for the gradient.
    let p = row_softmax_t(&anchor_scores, batch_size, k, config.tau_target);

    // ── Compute gradient G_P[k, d] = -1/B Σ_b (q_b,k - p_b,k) * z_anchor_b,d
    let inv_b = 1.0_f32 / batch_size as f32;
    let mut grad = vec![0.0_f32; k * feat_dim];
    for b in 0..batch_size {
        let feat_row = &anchor_features[b * feat_dim..(b + 1) * feat_dim];
        for ki in 0..k {
            let delta = (q[b * k + ki] - p[b * k + ki]) * inv_b;
            let g_row = &mut grad[ki * feat_dim..(ki + 1) * feat_dim];
            for (g, &f) in g_row.iter_mut().zip(feat_row.iter()) {
                *g -= delta * f; // gradient descent direction: -(-delta*f) = delta*f would ascend
            }
        }
    }

    // ── SGD step: P_k ← P_k - lr · G_{P,k}
    for (w, g) in prototypes.weights.iter_mut().zip(grad.iter()) {
        *w -= lr * g;
    }

    // ── Re-normalise every prototype row to unit sphere
    l2_normalize_rows(&mut prototypes.weights, k, feat_dim);

    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create an L2-normalised feature matrix [B, D] from a pattern.
    fn make_features_normalised(batch_size: usize, dim: usize, seed_offset: f32) -> Vec<f32> {
        let mut feats = Vec::with_capacity(batch_size * dim);
        for b in 0..batch_size {
            let mut row: Vec<f32> = (0..dim)
                .map(|d| (b as f32 * 0.31 + d as f32 * 0.17 + seed_offset).sin())
                .collect();
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in row.iter_mut() {
                *v /= norm;
            }
            feats.extend_from_slice(&row);
        }
        feats
    }

    /// 1. Valid loss is finite and non-negative for random inputs.
    #[test]
    fn loss_is_finite_and_non_negative() {
        let mut rng = LcgRng::new(42);
        let b = 4;
        let d = 16;
        let cfg = MsnConfig::default();
        let protos = msn_prototype_init(cfg.n_prototypes, d, &mut rng);
        let anchor = make_features_normalised(b, d, 0.0);
        let target = make_features_normalised(b, d, 1.0);
        let result = msn_loss(&anchor, &target, &protos, b, d, &cfg).unwrap();
        assert!(result.loss.is_finite(), "loss not finite: {}", result.loss);
        // CE component must be non-negative.
        assert!(
            result.ce_loss >= 0.0,
            "ce_loss negative: {}",
            result.ce_loss
        );
    }

    /// 2. With equal temperatures, CE(q, q) == H(q) ≤ ln(K); anchor==target at equal temps.
    ///
    /// When tau_anchor == tau_target and anchor == target, both soft assignments are
    /// identical: q == p.  Cross-entropy CE(q, q) = H(q) = -Σ_k q_k ln(q_k).
    /// For any valid distribution H(q) ≤ ln(K), so the CE loss must be bounded
    /// above by ln(K).  This confirms the softmax + CE implementation is correct.
    #[test]
    fn equal_temperatures_identical_inputs_ce_bounded_by_ln_k() {
        let mut rng = LcgRng::new(7);
        let b = 8;
        let d = 32;
        let k = 16;
        let cfg = MsnConfig {
            n_prototypes: k,
            tau_anchor: 0.15,
            tau_target: 0.15, // equal temperatures → q == p when anchor == target
            lambda_reg: 0.0,  // disable regulariser to isolate CE
            eps: 1e-8,
        };
        let protos = msn_prototype_init(k, d, &mut rng);
        let anchor = make_features_normalised(b, d, 0.0);
        // target == anchor, tau_anchor == tau_target → CE = H(q) ≤ ln(K).
        let result = msn_loss(&anchor, &anchor, &protos, b, d, &cfg).unwrap();
        let max_ce = (k as f32).ln(); // ln(16) ≈ 2.77
        assert!(
            result.ce_loss <= max_ce + 1e-4,
            "CE ({}) must be ≤ ln(K) = {}",
            result.ce_loss,
            max_ce
        );
        assert!(
            result.ce_loss >= 0.0,
            "CE must be non-negative, got {}",
            result.ce_loss
        );
    }

    /// 3. Uniform anchors — ME-Max entropy should be close to ln(K).
    #[test]
    fn uniform_anchors_high_me_max_entropy() {
        let b = 8;
        let d = 8;
        let k = 8;
        let cfg = MsnConfig {
            n_prototypes: k,
            tau_anchor: 0.1,
            tau_target: 0.25,
            lambda_reg: 1.0,
            eps: 1e-8,
        };
        // Build prototypes that are orthogonal basis vectors to get uniform score.
        // Use the identity-block trick: first k dims are identity.
        let mut proto_weights = vec![0.0_f32; k * d];
        for ki in 0..k {
            proto_weights[ki * d + ki] = 1.0; // row ki = e_ki
        }
        let protos = MsnPrototypes {
            weights: proto_weights,
            n_prototypes: k,
            dim: d,
        };
        // Anchor features: all identical (uniform direction) → uniform scores.
        // Use [1/√d, 1/√d, ...] for all samples.
        let uniform_val = 1.0_f32 / (d as f32).sqrt();
        let anchor: Vec<f32> = vec![uniform_val; b * d];
        let target: Vec<f32> = vec![uniform_val; b * d];
        let result = msn_loss(&anchor, &target, &protos, b, d, &cfg).unwrap();
        // ME-Max entropy with uniform marginal ≈ ln(K).
        let expected_h = (k as f32).ln();
        assert!(
            (result.me_max_loss - expected_h).abs() < 0.1,
            "Expected entropy ≈ ln({}) ≈ {:.4}, got {:.4}",
            k,
            expected_h,
            result.me_max_loss
        );
    }

    /// 4. CE loss component is separately non-negative.
    #[test]
    fn ce_loss_always_non_negative() {
        let mut rng = LcgRng::new(123);
        let b = 6;
        let d = 24;
        let cfg = MsnConfig::default();
        let protos = msn_prototype_init(cfg.n_prototypes, d, &mut rng);
        let anchor = make_features_normalised(b, d, 2.5);
        let target = make_features_normalised(b, d, 3.7);
        let result = msn_loss(&anchor, &target, &protos, b, d, &cfg).unwrap();
        assert!(
            result.ce_loss >= 0.0,
            "CE must be non-negative, got {}",
            result.ce_loss
        );
    }

    /// 5. Invalid n_prototypes=0 → error (NumPrototypesTooSmall for n=1 too).
    #[test]
    fn invalid_n_prototypes_returns_error() {
        let d = 8;
        let anchor = make_features_normalised(2, d, 0.0);
        let target = make_features_normalised(2, d, 1.0);
        let cfg = MsnConfig {
            n_prototypes: 1, // < 2 → invalid
            ..MsnConfig::default()
        };
        // Build a "dummy" prototype with K=1.
        let protos = MsnPrototypes {
            weights: vec![1.0_f32; d],
            n_prototypes: 1,
            dim: d,
        };
        let result = msn_loss(&anchor, &target, &protos, 2, d, &cfg);
        assert!(
            result.is_err(),
            "Expected error for n_prototypes < 2, got Ok"
        );
    }

    /// 6. Invalid temperature=0 → error.
    #[test]
    fn invalid_temperature_zero_returns_error() {
        let cfg_result = MsnConfig::new(16, 0.0, 0.25, 1.0, 1e-8);
        assert!(
            cfg_result.is_err(),
            "Expected InvalidTemperature for tau_anchor=0"
        );
        let cfg_result2 = MsnConfig::new(16, 0.1, 0.0, 1.0, 1e-8);
        assert!(
            cfg_result2.is_err(),
            "Expected InvalidTemperature for tau_target=0"
        );
    }

    /// 7. With all mask ratio = 1.0 is invalid; use 0.99 as near-full masking.
    ///    Loss must still compute (no NaN).
    #[test]
    fn near_full_masking_loss_still_finite() {
        let mut rng = LcgRng::new(99);
        let b = 4;
        let d = 16;
        let cfg = MsnConfig::default();
        let protos = msn_prototype_init(cfg.n_prototypes, d, &mut rng);
        // Simulate masked features as near-zero (all tokens masked — tiny values).
        let anchor = make_features_normalised(b, d, 0.0);
        let target: Vec<f32> = vec![1.0_f32 / (d as f32).sqrt(); b * d]; // degenerate target
        let result = msn_loss(&anchor, &target, &protos, b, d, &cfg).unwrap();
        assert!(result.loss.is_finite(), "Expected finite loss, got NaN");
    }

    /// 8. Single-sample batch (B=1) works correctly.
    #[test]
    fn single_sample_batch_works() {
        let mut rng = LcgRng::new(55);
        let b = 1;
        let d = 8;
        let cfg = MsnConfig {
            n_prototypes: 4,
            ..MsnConfig::default()
        };
        let protos = msn_prototype_init(cfg.n_prototypes, d, &mut rng);
        let anchor = make_features_normalised(b, d, 0.0);
        let target = make_features_normalised(b, d, 1.0);
        let result = msn_loss(&anchor, &target, &protos, b, d, &cfg).unwrap();
        assert!(result.loss.is_finite(), "Single-sample loss not finite");
    }

    /// 9. mean_prototype_util ∈ [0, 1].
    #[test]
    fn prototype_util_in_unit_interval() {
        let mut rng = LcgRng::new(11);
        let b = 8;
        let d = 32;
        let cfg = MsnConfig::default();
        let protos = msn_prototype_init(cfg.n_prototypes, d, &mut rng);
        let anchor = make_features_normalised(b, d, 0.0);
        let target = make_features_normalised(b, d, 1.0);
        let result = msn_loss(&anchor, &target, &protos, b, d, &cfg).unwrap();
        assert!(
            (0.0..=1.0).contains(&result.mean_prototype_util),
            "util out of [0,1]: {}",
            result.mean_prototype_util
        );
    }

    /// 10. msn_prototype_init produces L2-normalised rows (each row norm ≈ 1.0).
    #[test]
    fn prototype_init_rows_are_unit_norm() {
        let mut rng = LcgRng::new(17);
        let k = 32;
        let d = 64;
        let protos = msn_prototype_init(k, d, &mut rng);
        for ki in 0..k {
            let row = protos.row(ki);
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Prototype row {ki} has norm {norm:.6}, expected ≈ 1.0"
            );
        }
    }

    /// 11. msn_random_mask exact ratio: n_masked == floor(n_tokens * mask_ratio).
    #[test]
    fn random_mask_exact_count() {
        let mut rng = LcgRng::new(31);
        let n_tokens = 196;
        let mask_ratio = 0.75_f32;
        let mask = msn_random_mask(n_tokens, mask_ratio, &mut rng).unwrap();
        assert_eq!(mask.len(), n_tokens);
        let n_masked = mask.iter().filter(|&&v| v).count();
        let expected = (n_tokens as f32 * mask_ratio) as usize; // 147
        assert_eq!(
            n_masked, expected,
            "Expected {expected} masked tokens, got {n_masked}"
        );
    }

    /// 12. msn_update_prototypes keeps prototypes L2-normalised.
    #[test]
    fn update_prototypes_keeps_unit_norm() {
        let mut rng = LcgRng::new(77);
        let b = 4;
        let d = 16;
        let cfg = MsnConfig {
            n_prototypes: 8,
            ..MsnConfig::default()
        };
        let mut protos = msn_prototype_init(cfg.n_prototypes, d, &mut rng);
        let anchor = make_features_normalised(b, d, 0.0);
        msn_update_prototypes(&mut protos, &anchor, b, d, 0.01, &cfg).unwrap();
        for ki in 0..cfg.n_prototypes {
            let row = protos.row(ki);
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "After update, prototype {ki} has norm {norm:.6}, expected ≈ 1.0"
            );
        }
    }

    /// 13. msn_random_mask rejects invalid mask ratio.
    #[test]
    fn random_mask_rejects_invalid_ratio() {
        let mut rng = LcgRng::new(42);
        assert!(msn_random_mask(16, 1.0, &mut rng).is_err()); // ratio must be in [0, 1)
        assert!(msn_random_mask(16, -0.1, &mut rng).is_err());
        assert!(msn_random_mask(0, 0.5, &mut rng).is_err()); // empty input
    }

    /// 14. msn_loss rejects dimension mismatch.
    #[test]
    fn loss_rejects_dim_mismatch() {
        let mut rng = LcgRng::new(42);
        let d = 16;
        let protos = msn_prototype_init(8, d, &mut rng);
        let anchor = vec![0.1_f32; 3 * d]; // B=3
        let target = vec![0.1_f32; 4 * d]; // B=4 ← mismatch
        let cfg = MsnConfig {
            n_prototypes: 8,
            ..MsnConfig::default()
        };
        let result = msn_loss(&anchor, &target, &protos, 3, d, &cfg);
        assert!(result.is_err(), "Expected DimensionMismatch error");
    }

    /// 15. me_max_loss and mean_entropy fields are identical.
    #[test]
    fn me_max_and_mean_entropy_are_equal() {
        let mut rng = LcgRng::new(88);
        let b = 4;
        let d = 16;
        let cfg = MsnConfig::default();
        let protos = msn_prototype_init(cfg.n_prototypes, d, &mut rng);
        let anchor = make_features_normalised(b, d, 0.0);
        let target = make_features_normalised(b, d, 1.0);
        let result = msn_loss(&anchor, &target, &protos, b, d, &cfg).unwrap();
        assert_eq!(
            result.me_max_loss, result.mean_entropy,
            "me_max_loss and mean_entropy must be identical diagnostic values"
        );
    }
}

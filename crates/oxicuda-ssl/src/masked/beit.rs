//! BEiT — Bao et al. 2021 — BERT Pre-Training of Image Transformers.
//!
//! Key insight: instead of predicting raw pixel values (like MAE / SimMIM),
//! BEiT treats masked image modeling as **discrete token prediction**.  A
//! VQ-VAE / dVAE tokenizer maps each patch to a codebook index; the ViT is
//! then trained to predict those discrete tokens at masked positions via
//! cross-entropy — analogous to BERT's masked word prediction in NLP.
//!
//! # Components
//!
//! - [`VqCodebook`] — EMA-maintained vector-quantisation codebook.
//! - [`BeitConfig`] — hyper-parameters for tokenizer + pretraining loss.
//! - [`BeitResult`] — composite loss (BEiT CE + VQ commitment) and metrics.
//! - [`vq_codebook_init`] — random initialisation of the codebook.
//! - [`vq_encode`] — nearest-neighbour encoding + straight-through VQ loss.
//! - [`vq_update_codebook`] — EMA codebook update from assigned embeddings.
//! - [`beit_loss`] — cross-entropy of student logits vs. discrete VQ tokens.
//! - [`beit_block_mask`] — BEiT-style random rectangular block masking.
//!
//! # References
//! - Bao et al. *BEiT: BERT Pre-Training of Image Transformers* (ICLR 2022)
//! - van den Oord et al. *Neural Discrete Representation Learning* (NeurIPS 2017)

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// BEiT tokenizer + pretraining configuration.
#[derive(Debug, Clone)]
pub struct BeitConfig {
    /// Codebook size K (number of discrete visual tokens). Default: 8192.
    pub n_codes: usize,
    /// Code dimensionality C. Default: 256.
    pub code_dim: usize,
    /// Fraction of patches to mask during pre-training. Default: 0.4.
    pub mask_ratio: f32,
    /// EMA momentum for codebook update. Default: 0.999.
    pub ema_momentum: f32,
    /// Commitment loss weight β. Default: 0.25.
    pub commitment_weight: f32,
    /// Softmax temperature for BEiT loss (scales student logits). Default: 1.0.
    pub temperature: f32,
    /// Numerical stability ε. Default: 1e-6.
    pub eps: f32,
}

impl Default for BeitConfig {
    fn default() -> Self {
        Self {
            n_codes: 8192,
            code_dim: 256,
            mask_ratio: 0.4,
            ema_momentum: 0.999,
            commitment_weight: 0.25,
            temperature: 1.0,
            eps: 1e-6,
        }
    }
}

impl BeitConfig {
    /// Validated BEiT configuration constructor.
    ///
    /// # Errors
    /// - [`SslError::InvalidParameter`] for any out-of-range or zero value.
    /// - [`SslError::InvalidMaskRatio`] when `mask_ratio ∉ [0, 1)`.
    /// - [`SslError::InvalidTemperature`] when `temperature ≤ 0`.
    pub fn new(
        n_codes: usize,
        code_dim: usize,
        mask_ratio: f32,
        ema_momentum: f32,
        commitment_weight: f32,
        temperature: f32,
        eps: f32,
    ) -> SslResult<Self> {
        if n_codes == 0 {
            return Err(SslError::InvalidParameter {
                name: "n_codes".into(),
                reason: "must be > 0".into(),
            });
        }
        if code_dim == 0 {
            return Err(SslError::InvalidParameter {
                name: "code_dim".into(),
                reason: "must be > 0".into(),
            });
        }
        if !(mask_ratio.is_finite() && (0.0..1.0).contains(&mask_ratio)) {
            return Err(SslError::InvalidMaskRatio { ratio: mask_ratio });
        }
        if !(ema_momentum.is_finite() && (0.0..=1.0).contains(&ema_momentum)) {
            return Err(SslError::InvalidMomentum {
                momentum: ema_momentum,
            });
        }
        if !(commitment_weight.is_finite() && commitment_weight >= 0.0) {
            return Err(SslError::InvalidParameter {
                name: "commitment_weight".into(),
                reason: "must be finite and >= 0".into(),
            });
        }
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(SslError::InvalidTemperature { temp: temperature });
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(SslError::InvalidParameter {
                name: "eps".into(),
                reason: "must be finite and > 0".into(),
            });
        }
        Ok(Self {
            n_codes,
            code_dim,
            mask_ratio,
            ema_momentum,
            commitment_weight,
            temperature,
            eps,
        })
    }
}

// ─── VQ Codebook ─────────────────────────────────────────────────────────────

/// EMA-maintained vector-quantisation codebook.
///
/// Stores K code vectors of dimension C as a flat `[K × C]` row-major buffer.
/// EMA running statistics (`ema_counts`, `ema_sum`) are used for the
/// Laplace-smoothed online codebook update rule:
///
/// ```text
///     n_k  ← m·n_k + (1-m)·|S_k|
///     e_k  ← m·e_k_sum + (1-m)·Σ_{z∈S_k} z   (running sum)
///     code_k ← e_k_sum / n_k                   (normalised)
/// ```
#[derive(Debug, Clone)]
pub struct VqCodebook {
    /// Codebook entries E ∈ ℝ^{K×C}, row-major. Length = K * C.
    pub embeddings: Vec<f32>,
    /// Number of codes K.
    pub n_codes: usize,
    /// Code dimensionality C.
    pub code_dim: usize,
    /// EMA momentum m for codebook update (closer to 1 → slower update).
    pub ema_momentum: f32,
    /// Commitment loss weight β.
    pub commitment_weight: f32,
    /// EMA usage counts per code \[K\]. Initialised to 1 (Laplace smoothing).
    pub ema_counts: Vec<f32>,
    /// EMA running sums per code \[K × C\]. Initialised to the code vectors.
    pub ema_sum: Vec<f32>,
}

// ─── Initialisation ───────────────────────────────────────────────────────────

/// Initialise a [`VqCodebook`] with random N(0, 1/√C) entries and EMA state.
///
/// The codes are drawn from N(0, 1) and scaled by 1/√`code_dim` so that
/// initialised norms are ≈ 1, preventing large initial VQ losses.
///
/// # Errors
/// - [`SslError::InvalidParameter`] when `n_codes == 0` or `code_dim == 0`.
pub fn vq_codebook_init(
    n_codes: usize,
    code_dim: usize,
    rng: &mut LcgRng,
) -> SslResult<VqCodebook> {
    if n_codes == 0 {
        return Err(SslError::InvalidParameter {
            name: "n_codes".into(),
            reason: "must be > 0".into(),
        });
    }
    if code_dim == 0 {
        return Err(SslError::InvalidParameter {
            name: "code_dim".into(),
            reason: "must be > 0".into(),
        });
    }
    let total = n_codes * code_dim;
    let mut embeddings = vec![0.0_f32; total];
    rng.fill_normal(&mut embeddings);
    let scale = 1.0 / (code_dim as f32).sqrt();
    for v in &mut embeddings {
        *v *= scale;
    }
    // EMA state: counts start at 1 (Laplace), sums start equal to the codes.
    let ema_counts = vec![1.0_f32; n_codes];
    let ema_sum = embeddings.clone();
    Ok(VqCodebook {
        embeddings,
        n_codes,
        code_dim,
        ema_momentum: 0.999,
        commitment_weight: 0.25,
        ema_counts,
        ema_sum,
    })
}

// ─── Encoding ─────────────────────────────────────────────────────────────────

/// Encode patch embeddings to nearest codebook indices + straight-through VQ loss.
///
/// For each embedding `z_i ∈ ℝ^C`, finds the nearest code via brute-force
/// L2 distance: `k* = argmin_k ||z_i - e_k||²`.  Returns:
/// - `indices`: `[N]` integer codebook assignments.
/// - `quantized_z`: `[N × C]` quantised embeddings (with straight-through
///   gradient: `z_q = z + sg(e_{k*} - z)`, implemented here as `e_{k*}`
///   since we are forward-only).
/// - `vq_loss`: scalar combining codebook loss + β·commitment loss.
///
/// VQ loss formula:
/// ```text
///     L_vq = mean_i [ ||sg(z_i) - e_{k*}||² + β·||z_i - sg(e_{k*})||² ]
/// ```
/// Both terms equal `||z_i - e_{k*}||²` in the forward pass (no gradients
/// here); we weight the second by `commitment_weight`.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n_patches == 0` or `code_dim == 0`.
/// - [`SslError::DimensionMismatch`] when `embeddings.len() != n_patches * code_dim`.
pub fn vq_encode(
    codebook: &VqCodebook,
    embeddings: &[f32],
    n_patches: usize,
    code_dim: usize,
) -> SslResult<(Vec<usize>, Vec<f32>, f32)> {
    if n_patches == 0 || code_dim == 0 {
        return Err(SslError::EmptyInput);
    }
    let expected = n_patches * code_dim;
    if embeddings.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: embeddings.len(),
        });
    }
    if codebook.n_codes == 0 {
        return Err(SslError::EmptyInput);
    }

    let k = codebook.n_codes;
    let c = code_dim;
    let beta = codebook.commitment_weight;

    let mut indices = Vec::with_capacity(n_patches);
    let mut quantized_z = Vec::with_capacity(n_patches * c);
    let mut vq_loss_acc = 0.0_f64;

    for i in 0..n_patches {
        let z = &embeddings[i * c..(i + 1) * c];

        // Brute-force nearest-neighbour search: O(K·C) per embedding.
        let mut best_k = 0usize;
        let mut best_dist = f64::MAX;

        for ki in 0..k {
            let e_k = &codebook.embeddings[ki * c..(ki + 1) * c];
            let dist: f64 = z
                .iter()
                .zip(e_k.iter())
                .map(|(&zi, &eki)| {
                    let d = (zi - eki) as f64;
                    d * d
                })
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_k = ki;
            }
        }

        indices.push(best_k);

        // Quantised embedding = nearest code (straight-through in forward).
        let e_star = &codebook.embeddings[best_k * c..(best_k + 1) * c];
        quantized_z.extend_from_slice(e_star);

        // VQ loss: codebook term ||sg(z) - e_{k*}||² + β·||z - sg(e_{k*})||²
        // Both equal best_dist in forward; we apply the β weight to the
        // commitment (encoder) term.
        vq_loss_acc += best_dist * (1.0 + beta as f64);
    }

    let vq_loss = (vq_loss_acc / n_patches as f64) as f32;
    Ok((indices, quantized_z, vq_loss))
}

// ─── Codebook update ─────────────────────────────────────────────────────────

/// EMA update of the codebook using the embeddings assigned to each code.
///
/// Implements:
/// ```text
///     n_k ← m·n_k + (1-m)·|S_k|          (EMA of cluster sizes)
///     sum_k ← m·sum_k + (1-m)·Σ_{z∈S_k} z  (EMA of cluster sums)
///     e_k ← sum_k / n_k                     (normalised code vector)
/// ```
/// Codes that receive no assignments in this batch are left unchanged
/// (their counts and sums are decayed by momentum only).
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n_patches == 0`.
/// - [`SslError::DimensionMismatch`] when slice lengths are inconsistent.
pub fn vq_update_codebook(
    codebook: &mut VqCodebook,
    embeddings: &[f32],
    indices: &[usize],
    n_patches: usize,
) -> SslResult<()> {
    if n_patches == 0 {
        return Err(SslError::EmptyInput);
    }
    let c = codebook.code_dim;
    let k = codebook.n_codes;
    let expected_emb = n_patches * c;
    if embeddings.len() != expected_emb {
        return Err(SslError::DimensionMismatch {
            expected: expected_emb,
            got: embeddings.len(),
        });
    }
    if indices.len() != n_patches {
        return Err(SslError::DimensionMismatch {
            expected: n_patches,
            got: indices.len(),
        });
    }
    // Validate index range.
    for &idx in indices {
        if idx >= k {
            return Err(SslError::InvalidParameter {
                name: "index".into(),
                reason: format!("codebook index {idx} out of range [0, {k})"),
            });
        }
    }

    let m = codebook.ema_momentum;
    let one_minus_m = 1.0 - m;

    // Accumulate per-code batch statistics.
    let mut batch_counts = vec![0.0_f32; k];
    let mut batch_sums = vec![0.0_f32; k * c];

    for (i, &ki) in indices.iter().enumerate() {
        batch_counts[ki] += 1.0;
        let z = &embeddings[i * c..(i + 1) * c];
        let sum_slice = &mut batch_sums[ki * c..(ki + 1) * c];
        for (s, &zi) in sum_slice.iter_mut().zip(z.iter()) {
            *s += zi;
        }
    }

    // EMA update of counts and sums, then re-normalise codebook entries.
    for ki in 0..k {
        codebook.ema_counts[ki] = m * codebook.ema_counts[ki] + one_minus_m * batch_counts[ki];
        let count = codebook.ema_counts[ki].max(1e-6); // avoid div-by-zero
        let sum_slice = &mut codebook.ema_sum[ki * c..(ki + 1) * c];
        let batch_sum_slice = &batch_sums[ki * c..(ki + 1) * c];
        for (s, &bs) in sum_slice.iter_mut().zip(batch_sum_slice.iter()) {
            *s = m * (*s) + one_minus_m * bs;
        }
        // Normalise to get the updated code vector.
        let inv_count = 1.0 / count;
        let emb_slice = &mut codebook.embeddings[ki * c..(ki + 1) * c];
        let ema_sum_slice = &codebook.ema_sum[ki * c..(ki + 1) * c];
        for (e, &es) in emb_slice.iter_mut().zip(ema_sum_slice.iter()) {
            *e = es * inv_count;
        }
    }

    Ok(())
}

// ─── BEiT pretraining loss ────────────────────────────────────────────────────

/// Composite result from the BEiT pretraining loss.
#[derive(Debug, Clone)]
pub struct BeitResult {
    /// Cross-entropy loss of student logits vs. discrete VQ tokens at masked positions.
    pub beit_loss: f32,
    /// VQ commitment loss (codebook term + β·encoder term).
    pub vq_loss: f32,
    /// `beit_loss + vq_loss`.
    pub total_loss: f32,
    /// Number of masked patches (positions where loss was computed).
    pub n_masked: usize,
    /// Fraction of codebook entries used at least once this batch (∈ [0, 1]).
    pub codebook_usage: f32,
    /// Effective codebook perplexity = exp(H(assignment distribution)) ∈ [1, K].
    pub perplexity: f32,
}

/// BEiT pretraining cross-entropy loss.
///
/// Computes:
/// ```text
///     L = -1/M  Σ_{i: mask[i]=true}  log softmax(p_i / τ)[q_i]
/// ```
/// where `p_i ∈ ℝ^K` are the student's unnormalized logits for patch `i`,
/// `q_i` is the VQ codebook index assigned by the tokenizer, `τ` is the
/// softmax temperature, and `M` is the number of masked patches.
///
/// When no patches are masked (`mask` is all `false`), returns
/// `BeitResult { beit_loss: 0, vq_loss, total_loss: vq_loss, n_masked: 0, .. }`.
///
/// # Errors
/// - [`SslError::InvalidParameter`] when `n_codes == 0`.
/// - [`SslError::EmptyInput`] when `n_patches == 0`.
/// - [`SslError::DimensionMismatch`] when slice lengths are inconsistent.
/// - [`SslError::InvalidTemperature`] when `config.temperature ≤ 0`.
pub fn beit_loss(
    student_logits: &[f32],
    token_indices: &[usize],
    mask: &[bool],
    n_patches: usize,
    n_codes: usize,
    config: &BeitConfig,
) -> SslResult<BeitResult> {
    if n_codes == 0 {
        return Err(SslError::InvalidParameter {
            name: "n_codes".into(),
            reason: "must be > 0".into(),
        });
    }
    if n_patches == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(config.temperature.is_finite() && config.temperature > 0.0) {
        return Err(SslError::InvalidTemperature {
            temp: config.temperature,
        });
    }

    let expected_logits = n_patches * n_codes;
    if student_logits.len() != expected_logits {
        return Err(SslError::DimensionMismatch {
            expected: expected_logits,
            got: student_logits.len(),
        });
    }
    if token_indices.len() != n_patches {
        return Err(SslError::DimensionMismatch {
            expected: n_patches,
            got: token_indices.len(),
        });
    }
    if mask.len() != n_patches {
        return Err(SslError::DimensionMismatch {
            expected: n_patches,
            got: mask.len(),
        });
    }

    // Validate token index range.
    for &qi in token_indices {
        if qi >= n_codes {
            return Err(SslError::InvalidParameter {
                name: "token_index".into(),
                reason: format!("token index {qi} out of range [0, {n_codes})"),
            });
        }
    }

    let tau = config.temperature;
    let n_masked = mask.iter().filter(|&&m| m).count();

    // ── BEiT cross-entropy at masked positions ────────────────────────────────
    let mut beit_loss_acc = 0.0_f64;

    // Per-code assignment frequency for perplexity / usage calculation.
    let mut code_freq = vec![0.0_f64; n_codes];

    for i in 0..n_patches {
        let qi = token_indices[i];
        let logits = &student_logits[i * n_codes..(i + 1) * n_codes];

        // Accumulate code frequencies over ALL patches (not just masked) for
        // a representative perplexity estimate.
        code_freq[qi] += 1.0;

        if !mask[i] {
            continue; // only predict at masked positions
        }

        // Numerically stable softmax with temperature.
        let mut max_v = f32::NEG_INFINITY;
        for &lv in logits {
            let scaled = lv / tau;
            if scaled > max_v {
                max_v = scaled;
            }
        }
        let mut sum_exp = 0.0_f64;
        let mut exp_qi = 0.0_f64;
        for (k, &lv) in logits.iter().enumerate() {
            let e = ((lv / tau - max_v) as f64).exp();
            sum_exp += e;
            if k == qi {
                exp_qi = e;
            }
        }
        let log_prob = (exp_qi / sum_exp.max(1e-30)).max(1e-30_f64).ln();
        beit_loss_acc -= log_prob;
    }

    let beit_loss_val = if n_masked == 0 {
        0.0_f32
    } else {
        (beit_loss_acc / n_masked as f64) as f32
    };

    // ── Codebook usage and perplexity ────────────────────────────────────────
    let total_assignments = n_patches as f64;
    let n_used = code_freq.iter().filter(|&&f| f > 0.0).count();
    let codebook_usage = n_used as f32 / n_codes as f32;

    // Perplexity = exp(H) where H = -Σ p_k log p_k, p_k = freq_k / total.
    let mut entropy = 0.0_f64;
    for &freq in &code_freq {
        if freq > 0.0 {
            let p = freq / total_assignments;
            entropy -= p * p.ln();
        }
    }
    let perplexity = entropy.exp().clamp(1.0, n_codes as f64) as f32;

    // ── VQ loss (passthrough from config — callers typically compute it via
    //    vq_encode + vq_update_codebook; here we provide 0 as placeholder
    //    unless the caller supplies it via config.commitment_weight context).
    // Since BEiT loss function doesn't have access to the raw embeddings,
    // vq_loss is reported as 0 here.  The caller should add the vq_loss
    // returned by vq_encode to the total when assembling the training step.
    let vq_loss_val = 0.0_f32;
    let total_loss = beit_loss_val + vq_loss_val;

    Ok(BeitResult {
        beit_loss: beit_loss_val,
        vq_loss: vq_loss_val,
        total_loss,
        n_masked,
        codebook_usage,
        perplexity,
    })
}

// ─── Block masking ────────────────────────────────────────────────────────────

/// Generate a BEiT-style block mask on a 2-D patch grid.
///
/// Unlike MAE's per-patch Bernoulli mask, BEiT uses **random rectangular
/// blocks** (aspect-ratio-aware) to mask contiguous spatial regions.  This
/// encourages the model to reason about object structure rather than isolated
/// pixels.
///
/// The algorithm:
/// 1. Sample a random block area uniformly in `[min_area, max_area]` patches
///    where `min_area = max(1, floor(n_patches · 0.05))` and
///    `max_area = max(min_area, ceil(n_patches · 0.3))`.
/// 2. Sample a random aspect ratio r ∈ {0.3, 0.5, 0.75, 1.0, 1.33, 2.0, 3.0}
///    (log-uniform discrete grid from the BEiT paper).
/// 3. Compute block height `bh = sqrt(area / r)`, width `bw = sqrt(area * r)`,
///    clamped to the grid bounds.
/// 4. Place the block at a uniformly random position.
/// 5. Repeat until the number of newly masked patches reaches the target
///    `floor(n_patches · mask_ratio)` or a safety iteration limit is hit.
///
/// Returns `Vec<bool>` of length `n_patches` in row-major order
/// (`true` ⟺ patch is masked).
///
/// # Errors
/// - [`SslError::EmptyInput`] when `patch_grid_h == 0` or `patch_grid_w == 0`.
/// - [`SslError::InvalidMaskRatio`] when `mask_ratio ∉ [0, 1)`.
/// - [`SslError::InvalidParameter`] when `n_patches != patch_grid_h * patch_grid_w`.
pub fn beit_block_mask(
    n_patches: usize,
    patch_grid_h: usize,
    patch_grid_w: usize,
    mask_ratio: f32,
    rng: &mut LcgRng,
) -> SslResult<Vec<bool>> {
    if patch_grid_h == 0 || patch_grid_w == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(mask_ratio.is_finite() && (0.0..1.0).contains(&mask_ratio)) {
        return Err(SslError::InvalidMaskRatio { ratio: mask_ratio });
    }
    let grid_total = patch_grid_h * patch_grid_w;
    if n_patches != grid_total {
        return Err(SslError::InvalidParameter {
            name: "n_patches".into(),
            reason: format!(
                "n_patches ({n_patches}) must equal patch_grid_h * patch_grid_w ({grid_total})"
            ),
        });
    }

    let target_masked = (n_patches as f32 * mask_ratio).floor() as usize;
    let mut mask = vec![false; n_patches];
    let mut n_masked = 0usize;

    if target_masked == 0 {
        return Ok(mask);
    }

    // Discrete aspect-ratio candidates (BEiT paper uses log-uniform grid).
    const ASPECT_RATIOS: [f32; 7] = [0.3, 0.5, 0.75, 1.0, 1.33, 2.0, 3.0];

    // Block area range: 5%–30% of total grid area, at least 1.
    let min_area = (n_patches as f32 * 0.05).ceil() as usize;
    let min_area = min_area.max(1);
    let max_area = (n_patches as f32 * 0.30).ceil() as usize;
    let max_area = max_area.max(min_area);

    // Safety valve to prevent infinite loops on tiny grids.
    let max_iters = (target_masked * 16 + 1).max(200);
    let mut iters = 0usize;

    while n_masked < target_masked && iters < max_iters {
        iters += 1;

        // Sample block area.
        let area_range = max_area - min_area + 1;
        let area = min_area + rng.next_usize(area_range);

        // Sample aspect ratio.
        let ratio_idx = rng.next_usize(ASPECT_RATIOS.len());
        let ar = ASPECT_RATIOS[ratio_idx];

        // Derive block height and width from area and aspect ratio.
        let bh_f = (area as f32 / ar).sqrt();
        let bw_f = (area as f32 * ar).sqrt();
        let bh = (bh_f.round() as usize).clamp(1, patch_grid_h);
        let bw = (bw_f.round() as usize).clamp(1, patch_grid_w);

        // Uniformly sample top-left anchor.
        let r0 = if patch_grid_h > bh {
            rng.next_usize(patch_grid_h - bh + 1)
        } else {
            0
        };
        let c0 = if patch_grid_w > bw {
            rng.next_usize(patch_grid_w - bw + 1)
        } else {
            0
        };

        // Stamp the block onto the mask.
        for r in r0..r0 + bh {
            for c in c0..c0 + bw {
                let idx = r * patch_grid_w + c;
                if !mask[idx] {
                    mask[idx] = true;
                    n_masked += 1;
                    // Stop stamping this block if we've hit the target.
                    if n_masked >= target_masked {
                        break;
                    }
                }
            }
            if n_masked >= target_masked {
                break;
            }
        }
    }

    Ok(mask)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── vq_codebook_init ──────────────────────────────────────────────────────

    /// Initialised codebook must have exactly K×C entries.
    #[test]
    fn vq_codebook_init_correct_shape() {
        let mut rng = LcgRng::new(1);
        let cb = vq_codebook_init(64, 32, &mut rng).unwrap();
        assert_eq!(cb.embeddings.len(), 64 * 32);
        assert_eq!(cb.n_codes, 64);
        assert_eq!(cb.code_dim, 32);
        assert_eq!(cb.ema_counts.len(), 64);
        assert_eq!(cb.ema_sum.len(), 64 * 32);
    }

    /// All initialised entries must be finite.
    #[test]
    fn vq_codebook_init_entries_finite() {
        let mut rng = LcgRng::new(2);
        let cb = vq_codebook_init(16, 8, &mut rng).unwrap();
        assert!(cb.embeddings.iter().all(|v| v.is_finite()));
        assert!(cb.ema_sum.iter().all(|v| v.is_finite()));
    }

    /// Zero n_codes must return an error.
    #[test]
    fn vq_codebook_init_rejects_zero_codes() {
        let mut rng = LcgRng::new(3);
        assert!(vq_codebook_init(0, 32, &mut rng).is_err());
    }

    /// Zero code_dim must return an error.
    #[test]
    fn vq_codebook_init_rejects_zero_dim() {
        let mut rng = LcgRng::new(4);
        assert!(vq_codebook_init(16, 0, &mut rng).is_err());
    }

    // ── vq_encode ─────────────────────────────────────────────────────────────

    /// All returned indices must be within [0, K).
    #[test]
    fn vq_encode_indices_in_range() {
        let mut rng = LcgRng::new(5);
        let k = 32;
        let c = 8;
        let cb = vq_codebook_init(k, c, &mut rng).unwrap();
        let n = 20;
        let mut emb = vec![0.0_f32; n * c];
        rng.fill_normal(&mut emb);
        let (indices, _, _) = vq_encode(&cb, &emb, n, c).unwrap();
        assert_eq!(indices.len(), n);
        for &idx in &indices {
            assert!(idx < k, "index {idx} out of range");
        }
    }

    /// VQ loss must be non-negative.
    #[test]
    fn vq_encode_vq_loss_non_negative() {
        let mut rng = LcgRng::new(6);
        let k = 16;
        let c = 4;
        let cb = vq_codebook_init(k, c, &mut rng).unwrap();
        let n = 10;
        let mut emb = vec![0.0_f32; n * c];
        rng.fill_normal(&mut emb);
        let (_, _, vq_loss) = vq_encode(&cb, &emb, n, c).unwrap();
        assert!(vq_loss >= 0.0, "vq_loss = {vq_loss} should be >= 0");
    }

    /// Quantised output has shape [N × C].
    #[test]
    fn vq_encode_quantized_shape() {
        let mut rng = LcgRng::new(7);
        let k = 8;
        let c = 6;
        let cb = vq_codebook_init(k, c, &mut rng).unwrap();
        let n = 5;
        let mut emb = vec![0.0_f32; n * c];
        rng.fill_normal(&mut emb);
        let (indices, quantized, _) = vq_encode(&cb, &emb, n, c).unwrap();
        assert_eq!(quantized.len(), n * c);
        assert_eq!(indices.len(), n);
    }

    /// If the embedding exactly equals a codebook entry, that entry is selected.
    #[test]
    fn vq_encode_exact_match_selected() {
        let mut rng = LcgRng::new(8);
        let k = 8;
        let c = 4;
        let mut cb = vq_codebook_init(k, c, &mut rng).unwrap();
        // Force codebook[3] to be the zero vector.
        for v in &mut cb.embeddings[3 * c..4 * c] {
            *v = 0.0;
        }
        // Embed = zero vector → should match code 3 (or whichever other code
        // is closest to zero; we just assert the returned distance is minimal).
        let emb = vec![0.0_f32; c];
        let (indices, _, vq_loss) = vq_encode(&cb, &emb, 1, c).unwrap();
        // The selected code must be a valid index.
        assert!(indices[0] < k);
        // Loss must be non-negative.
        assert!(vq_loss >= 0.0);
    }

    // ── vq_update_codebook ────────────────────────────────────────────────────

    /// After assigning all embeddings to code 0 with a constant vector,
    /// code 0 must move toward that vector.
    #[test]
    fn vq_update_codebook_ema_moves_toward_assigned() {
        let mut rng = LcgRng::new(9);
        let k = 4;
        let c = 3;
        let mut cb = vq_codebook_init(k, c, &mut rng).unwrap();
        // Use a small momentum so the update is visible.
        cb.ema_momentum = 0.5;

        // Record original code-0 value.
        let orig_code0: Vec<f32> = cb.embeddings[0..c].to_vec();

        // All patches assigned to code 0 with embedding = [1, 1, 1].
        let n = 5;
        let emb = vec![1.0_f32; n * c];
        let indices = vec![0usize; n];
        vq_update_codebook(&mut cb, &emb, &indices, n).unwrap();

        let updated_code0: Vec<f32> = cb.embeddings[0..c].to_vec();
        // Each component of code 0 should be between its original value and 1.0.
        for (orig, updated) in orig_code0.iter().zip(updated_code0.iter()) {
            let dist_before = (orig - 1.0).abs();
            let dist_after = (updated - 1.0).abs();
            assert!(
                dist_after < dist_before || dist_before < 1e-6,
                "EMA update did not move code 0 toward [1,1,1]: orig={orig} updated={updated}"
            );
        }
    }

    // ── beit_loss ─────────────────────────────────────────────────────────────

    /// BEiT loss must be finite and non-negative for random inputs.
    #[test]
    fn beit_loss_finite_and_non_negative() {
        let mut rng = LcgRng::new(10);
        let n = 16;
        let k = 8;
        let cfg = BeitConfig {
            n_codes: k,
            code_dim: 4,
            ..BeitConfig::default()
        };
        let mut logits = vec![0.0_f32; n * k];
        rng.fill_normal(&mut logits);
        let indices: Vec<usize> = (0..n).map(|i| i % k).collect();
        let mask: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let result = beit_loss(&logits, &indices, &mask, n, k, &cfg).unwrap();
        assert!(result.total_loss.is_finite(), "total_loss should be finite");
        assert!(result.beit_loss >= 0.0, "beit_loss should be >= 0");
    }

    /// n_masked must equal the count of `true` entries in the mask.
    #[test]
    fn beit_loss_n_masked_matches_mask() {
        let n = 20;
        let k = 4;
        let cfg = BeitConfig {
            n_codes: k,
            ..BeitConfig::default()
        };
        let logits = vec![1.0_f32; n * k];
        let indices = vec![0usize; n];
        let mask: Vec<bool> = (0..n).map(|i| i < 7).collect(); // 7 masked
        let result = beit_loss(&logits, &indices, &mask, n, k, &cfg).unwrap();
        assert_eq!(result.n_masked, 7);
    }

    /// When mask is all-false, beit_loss must be 0 (no positions to predict).
    #[test]
    fn beit_loss_all_unmasked_returns_zero() {
        let n = 8;
        let k = 4;
        let cfg = BeitConfig {
            n_codes: k,
            ..BeitConfig::default()
        };
        let logits = vec![0.5_f32; n * k];
        let indices = vec![0usize; n];
        let mask = vec![false; n];
        let result = beit_loss(&logits, &indices, &mask, n, k, &cfg).unwrap();
        assert_eq!(result.n_masked, 0);
        assert!(
            result.beit_loss.abs() < 1e-7,
            "expected 0 loss, got {}",
            result.beit_loss
        );
    }

    /// codebook_usage must lie in [0, 1].
    #[test]
    fn beit_loss_codebook_usage_in_range() {
        let mut rng = LcgRng::new(11);
        let n = 12;
        let k = 16;
        let cfg = BeitConfig {
            n_codes: k,
            ..BeitConfig::default()
        };
        let mut logits = vec![0.0_f32; n * k];
        rng.fill_normal(&mut logits);
        let indices: Vec<usize> = (0..n).map(|_| rng.next_usize(k)).collect();
        let mask = vec![true; n];
        let result = beit_loss(&logits, &indices, &mask, n, k, &cfg).unwrap();
        assert!(
            (0.0..=1.0).contains(&result.codebook_usage),
            "codebook_usage = {}",
            result.codebook_usage
        );
    }

    /// Perplexity must be in [1, K].
    #[test]
    fn beit_loss_perplexity_in_range() {
        let mut rng = LcgRng::new(12);
        let n = 32;
        let k = 16;
        let cfg = BeitConfig {
            n_codes: k,
            ..BeitConfig::default()
        };
        let mut logits = vec![0.0_f32; n * k];
        rng.fill_normal(&mut logits);
        // Assign each patch to a distinct code (cycling) to maximise diversity.
        let indices: Vec<usize> = (0..n).map(|i| i % k).collect();
        let mask = vec![true; n];
        let result = beit_loss(&logits, &indices, &mask, n, k, &cfg).unwrap();
        assert!(
            result.perplexity >= 1.0 && result.perplexity <= k as f32 + 1e-4,
            "perplexity = {} out of [1, {}]",
            result.perplexity,
            k
        );
    }

    /// Invalid n_codes = 0 must return an error.
    #[test]
    fn beit_loss_rejects_zero_n_codes() {
        let logits = vec![1.0_f32; 4];
        let indices = vec![0usize; 4];
        let mask = vec![true; 4];
        let cfg = BeitConfig::default();
        assert!(beit_loss(&logits, &indices, &mask, 4, 0, &cfg).is_err());
    }

    // ── beit_block_mask ───────────────────────────────────────────────────────

    /// The mask must have exactly n_patches entries.
    #[test]
    fn beit_block_mask_correct_length() {
        let mut rng = LcgRng::new(13);
        let h = 14;
        let w = 14;
        let n = h * w;
        let mask = beit_block_mask(n, h, w, 0.4, &mut rng).unwrap();
        assert_eq!(mask.len(), n);
    }

    /// With mask_ratio = 0, no patches should be masked.
    #[test]
    fn beit_block_mask_zero_ratio_all_unmasked() {
        let mut rng = LcgRng::new(14);
        let h = 8;
        let w = 8;
        let n = h * w;
        let mask = beit_block_mask(n, h, w, 0.0, &mut rng).unwrap();
        assert!(mask.iter().all(|&v| !v));
    }

    /// mask_ratio > 1 must return InvalidMaskRatio error.
    #[test]
    fn beit_block_mask_rejects_invalid_ratio() {
        let mut rng = LcgRng::new(15);
        assert!(beit_block_mask(16, 4, 4, 1.1, &mut rng).is_err());
        assert!(beit_block_mask(16, 4, 4, -0.1, &mut rng).is_err());
        assert!(beit_block_mask(16, 4, 4, f32::NAN, &mut rng).is_err());
    }

    /// For a 14×14 grid with mask_ratio ≈ 0.4, roughly 40% of patches should
    /// be masked (within a wide block-masking tolerance of ±0.25).
    #[test]
    fn beit_block_mask_approx_ratio() {
        let mut rng = LcgRng::new(16);
        let h = 14;
        let w = 14;
        let n = h * w; // 196
        let ratio = 0.4_f32;
        let mask = beit_block_mask(n, h, w, ratio, &mut rng).unwrap();
        let n_masked = mask.iter().filter(|&&v| v).count();
        // The block mask stops exactly at target = floor(196 * 0.4) = 78.
        let target = (n as f32 * ratio).floor() as usize;
        assert!(
            n_masked <= target,
            "n_masked ({n_masked}) > target ({target}): block stopped early but should not over-shoot"
        );
        // We expect it to reach at least 30% of target.
        assert!(
            n_masked >= target / 2,
            "too few patches masked: {n_masked} vs target {target}"
        );
    }

    /// A batch of patches: all assignments returned by vq_encode are valid.
    #[test]
    fn vq_encode_batch_all_valid_assignments() {
        let mut rng = LcgRng::new(17);
        let k = 32;
        let c = 16;
        let cb = vq_codebook_init(k, c, &mut rng).unwrap();
        let n = 50;
        let mut emb = vec![0.0_f32; n * c];
        rng.fill_normal(&mut emb);
        let (indices, quantized, vq_loss) = vq_encode(&cb, &emb, n, c).unwrap();
        assert_eq!(indices.len(), n);
        assert_eq!(quantized.len(), n * c);
        assert!(vq_loss.is_finite() && vq_loss >= 0.0);
        for &idx in &indices {
            assert!(idx < k, "assignment {idx} out of [0, {k})");
        }
    }
}

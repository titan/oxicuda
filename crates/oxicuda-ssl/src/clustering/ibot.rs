//! iBOT — Image BERT pre-training with Online Tokenizer (Zhou et al. 2021).
//!
//! iBOT extends DINO by adding a Masked Image Modeling (MIM) objective on top
//! of the standard DINO CLS-token loss. The student processes masked patch
//! tokens and must predict what the teacher would output for those same
//! positions, using a shared set of online prototypes (the "online tokenizer").
//!
//! # Loss decomposition
//!
//! ```text
//!   L_total = λ_cls · L_cls  +  λ_patch · L_mim
//!
//!   L_cls   = CE(softmax((t_cls − c_cls) / τ_t), softmax(s_cls / τ_s))
//!             averaged over the batch                                      (DINO)
//!
//!   L_mim   = CE(softmax((t_patch − c_patch) / τ_t),
//!                softmax(s_patch / τ_s))
//!             averaged over all (batch × masked_token) pairs              (MIM)
//! ```
//!
//! Both teacher outputs are stop-gradient and centred with a running EMA of
//! the batch mean (one centre per prototype dimension for CLS, one for patches).
//!
//! # Reference
//! Zhou et al. "iBOT: Image BERT Pre-Training with Online Tokenizer."
//! ICLR 2022. <https://arxiv.org/abs/2111.07832>

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// iBOT configuration parameters.
#[derive(Debug, Clone)]
pub struct IBotConfig {
    /// Prototype vocabulary size (default: 8192).
    pub n_prototypes: usize,
    /// Student temperature τ_s (default: 0.1).
    pub tau_student: f32,
    /// Teacher temperature τ_t (default: 0.04).
    pub tau_teacher: f32,
    /// EMA momentum for centre update (default: 0.9).
    pub center_momentum: f32,
    /// Weight for the CLS-level DINO loss (default: 1.0).
    pub lambda_cls: f32,
    /// Weight for the patch MIM loss (default: 1.0).
    pub lambda_patch: f32,
    /// Numerical stability floor for log (default: 1e-6).
    pub eps: f32,
}

impl Default for IBotConfig {
    fn default() -> Self {
        Self {
            n_prototypes: 8192,
            tau_student: 0.1,
            tau_teacher: 0.04,
            center_momentum: 0.9,
            lambda_cls: 1.0,
            lambda_patch: 1.0,
            eps: 1e-6,
        }
    }
}

impl IBotConfig {
    /// Construct a validated `IBotConfig`.
    ///
    /// # Errors
    /// - [`SslError::NumPrototypesTooSmall`] if `n_prototypes < 2`.
    /// - [`SslError::InvalidTemperature`] if either temperature ≤ 0 or non-finite.
    /// - [`SslError::InvalidMomentum`] if `center_momentum ∉ [0, 1]`.
    /// - [`SslError::InvalidLossWeight`] if `lambda_cls` or `lambda_patch` is non-finite.
    /// - [`SslError::InvalidParameter`] if `eps ≤ 0`.
    pub fn new(
        n_prototypes: usize,
        tau_student: f32,
        tau_teacher: f32,
        center_momentum: f32,
        lambda_cls: f32,
        lambda_patch: f32,
        eps: f32,
    ) -> SslResult<Self> {
        if n_prototypes < 2 {
            return Err(SslError::NumPrototypesTooSmall);
        }
        for t in [tau_student, tau_teacher] {
            if !(t.is_finite() && t > 0.0) {
                return Err(SslError::InvalidTemperature { temp: t });
            }
        }
        if !(center_momentum.is_finite() && (0.0..=1.0).contains(&center_momentum)) {
            return Err(SslError::InvalidMomentum {
                momentum: center_momentum,
            });
        }
        for w in [lambda_cls, lambda_patch] {
            if !w.is_finite() {
                return Err(SslError::InvalidLossWeight { weight: w });
            }
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(SslError::InvalidParameter {
                name: "eps".to_string(),
                reason: "must be finite and > 0".to_string(),
            });
        }
        Ok(Self {
            n_prototypes,
            tau_student,
            tau_teacher,
            center_momentum,
            lambda_cls,
            lambda_patch,
            eps,
        })
    }
}

// ─── Online centre buffers ────────────────────────────────────────────────────

/// Running EMA centres for the CLS and patch prototype outputs.
///
/// Each centre is `[K]` where `K = n_prototypes`. Both are initialised to
/// zero (see [`ibot_centers_init`]) and updated via [`ibot_update_centers`].
#[derive(Debug, Clone)]
pub struct IBotCenters {
    /// Running EMA centre for teacher CLS logits. Shape: `[n_prototypes]`.
    pub cls_center: Vec<f32>,
    /// Running EMA centre for teacher patch logits. Shape: `[n_prototypes]`.
    pub patch_center: Vec<f32>,
}

/// Aggregate result returned by [`ibot_loss`].
#[derive(Debug, Clone)]
pub struct IBotResult {
    /// Weighted sum `λ_cls · L_cls + λ_patch · L_mim`.
    pub total_loss: f32,
    /// Unweighted CLS cross-entropy loss.
    pub cls_loss: f32,
    /// Unweighted MIM (patch) cross-entropy loss.
    pub mim_loss: f32,
    /// Total number of (batch × masked_token) pairs used in the MIM term.
    pub n_masked_patches: usize,
    /// Mean Shannon entropy of the teacher CLS distribution over the batch.
    pub mean_teacher_entropy: f32,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Stable per-row softmax of a `[rows × k]` matrix scaled by temperature `t`.
///
/// Uses f64 accumulator for numerical stability.
fn row_softmax_t(scores: &[f32], rows: usize, k: usize, t: f32, eps: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * k);
    for i in 0..rows {
        let row = &scores[i * k..(i + 1) * k];
        // Find per-row maximum in the scaled space to avoid exp overflow.
        let mut max_v = f32::NEG_INFINITY;
        for &v in row {
            let scaled = v / t;
            if scaled > max_v {
                max_v = scaled;
            }
        }
        let mut s = 0.0_f64;
        let mut tmp = Vec::with_capacity(k);
        for &v in row {
            let e = ((v / t - max_v) as f64).exp();
            tmp.push(e);
            s += e;
        }
        let inv = 1.0_f64 / s.max(eps as f64);
        for e in &tmp {
            out.push((*e * inv) as f32);
        }
    }
    out
}

/// Cross-entropy `−Σ_k q[k] · log p[k]` summed over all rows, averaged by
/// `n_total` (caller-specified normaliser, e.g. batch size or batch×tokens).
///
/// Uses f64 accumulation for precision.
fn cross_entropy_sum(q: &[f32], p: &[f32], rows: usize, k: usize, eps: f32) -> f32 {
    let mut total = 0.0_f64;
    for i in 0..rows {
        for j in 0..k {
            let log_pj = (p[i * k + j].max(eps) as f64).ln();
            total -= (q[i * k + j] as f64) * log_pj;
        }
    }
    total as f32
}

/// Compute the mean Shannon entropy `H = −Σ_k p_k · log p_k` of rows
/// in a probability matrix `[rows × k]`, averaged over rows.
fn mean_entropy(probs: &[f32], rows: usize, k: usize, eps: f32) -> f32 {
    if rows == 0 {
        return 0.0;
    }
    let mut total = 0.0_f64;
    for i in 0..rows {
        for j in 0..k {
            let p = probs[i * k + j].max(eps) as f64;
            total -= p * p.ln();
        }
    }
    (total / rows as f64) as f32
}

/// Subtract a `[k]` centre vector from every row of a `[rows × k]` matrix,
/// returning the result as a new allocation.
fn subtract_center(logits: &[f32], rows: usize, k: usize, center: &[f32]) -> Vec<f32> {
    let mut out = logits.to_vec();
    for i in 0..rows {
        for j in 0..k {
            out[i * k + j] -= center[j];
        }
    }
    out
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Initialise iBOT centres to all zeros.
///
/// This is the canonical starting point before the first forward pass.
#[must_use]
pub fn ibot_centers_init(n_prototypes: usize) -> IBotCenters {
    IBotCenters {
        cls_center: vec![0.0_f32; n_prototypes],
        patch_center: vec![0.0_f32; n_prototypes],
    }
}

/// CLS-token iBOT loss (identical to the DINO loss).
///
/// # Arguments
/// - `student_cls`: `[B × K]` row-major student CLS logits.
/// - `teacher_cls`: `[B × K]` row-major teacher CLS logits.
/// - `centers`: current EMA centres; only `cls_center` is used.
/// - `batch_size`: `B`.
/// - `n_prototypes`: `K`.
/// - `config`: iBOT hyper-parameters.
///
/// # Returns
/// Mean CE loss (scalar, in nats).
///
/// # Errors
/// - [`SslError::EmptyInput`] if `B == 0` or `K == 0`.
/// - [`SslError::NumPrototypesTooSmall`] if `K < 2`.
/// - [`SslError::DimensionMismatch`] for shape mismatches.
pub fn ibot_cls_loss(
    student_cls: &[f32],
    teacher_cls: &[f32],
    centers: &IBotCenters,
    batch_size: usize,
    n_prototypes: usize,
    config: &IBotConfig,
) -> SslResult<f32> {
    if batch_size == 0 {
        return Err(SslError::EmptyInput);
    }
    if n_prototypes < 2 {
        return Err(SslError::NumPrototypesTooSmall);
    }
    let expected = batch_size * n_prototypes;
    if student_cls.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: student_cls.len(),
        });
    }
    if teacher_cls.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: teacher_cls.len(),
        });
    }
    if centers.cls_center.len() != n_prototypes {
        return Err(SslError::DimensionMismatch {
            expected: n_prototypes,
            got: centers.cls_center.len(),
        });
    }

    // Centre-subtract teacher logits then apply teacher sharpening.
    let t_centred = subtract_center(teacher_cls, batch_size, n_prototypes, &centers.cls_center);
    let p_teacher = row_softmax_t(
        &t_centred,
        batch_size,
        n_prototypes,
        config.tau_teacher,
        config.eps,
    );
    // Student logits: no centering, apply student temperature.
    let p_student = row_softmax_t(
        student_cls,
        batch_size,
        n_prototypes,
        config.tau_student,
        config.eps,
    );

    let ce_sum = cross_entropy_sum(&p_teacher, &p_student, batch_size, n_prototypes, config.eps);
    Ok(ce_sum / batch_size as f32)
}

/// MIM (Masked Image Modeling) patch-level iBOT loss.
///
/// # Arguments
/// - `student_patches`: `[B × M × K]` student logits at masked patch positions.
/// - `teacher_patches`: `[B × M × K]` teacher logits at the same positions.
/// - `batch_size`: `B`.
/// - `n_masked`: `M` (number of masked tokens per sample; 0 is allowed).
/// - `n_prototypes`: `K`.
/// - `config`: iBOT hyper-parameters.
///
/// # Returns
/// Mean CE loss averaged over `B × M` pairs, or `0.0` when `M == 0`.
///
/// # Errors
/// - [`SslError::EmptyInput`] if `B == 0`.
/// - [`SslError::NumPrototypesTooSmall`] if `K < 2`.
/// - [`SslError::DimensionMismatch`] for shape mismatches.
pub fn ibot_mim_loss(
    student_patches: &[f32],
    teacher_patches: &[f32],
    batch_size: usize,
    n_masked: usize,
    n_prototypes: usize,
    config: &IBotConfig,
) -> SslResult<f32> {
    if batch_size == 0 {
        return Err(SslError::EmptyInput);
    }
    if n_prototypes < 2 {
        return Err(SslError::NumPrototypesTooSmall);
    }
    // M == 0 is valid: no masked tokens → MIM loss is zero by definition.
    if n_masked == 0 {
        return Ok(0.0);
    }
    let expected = batch_size * n_masked * n_prototypes;
    if student_patches.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: student_patches.len(),
        });
    }
    if teacher_patches.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: teacher_patches.len(),
        });
    }

    // Treat the [B × M] token positions as a flat [B*M] rows × K matrix.
    let total_tokens = batch_size * n_masked;

    // Teacher: no explicit centering for the patch loss — the patch_center is
    // applied by the caller (ibot_loss) or can be passed zero if unused.
    // Per the spec, the MIM teacher logits are NOT centre-subtracted here
    // (the caller is responsible). We apply only temperature sharpening.
    //
    // NOTE: `ibot_mim_loss` is a low-level primitive. For correct iBOT behaviour,
    // call `ibot_loss` which handles centering before passing to this function,
    // or pre-subtract the patch_center before calling this function.
    let p_teacher = row_softmax_t(
        teacher_patches,
        total_tokens,
        n_prototypes,
        config.tau_teacher,
        config.eps,
    );
    let p_student = row_softmax_t(
        student_patches,
        total_tokens,
        n_prototypes,
        config.tau_student,
        config.eps,
    );

    let ce_sum = cross_entropy_sum(
        &p_teacher,
        &p_student,
        total_tokens,
        n_prototypes,
        config.eps,
    );
    Ok(ce_sum / total_tokens as f32)
}

/// Combined iBOT loss: CLS (DINO) + patch MIM, with automatic centre update.
///
/// This is the primary entry point for iBOT training. It:
/// 1. Subtracts `cls_center` from teacher CLS logits.
/// 2. Subtracts `patch_center` from teacher patch logits.
/// 3. Computes `L_cls` and `L_mim`.
/// 4. Updates `centers` via EMA.
/// 5. Returns `IBotResult` with `total = λ_cls·L_cls + λ_patch·L_mim`.
///
/// # Arguments
/// - `student_cls`: `[B × K]` student CLS projection logits.
/// - `teacher_cls`: `[B × K]` teacher CLS projection logits.
/// - `student_patches`: `[B × M × K]` student masked-patch logits.
/// - `teacher_patches`: `[B × M × K]` teacher masked-patch logits.
/// - `centers`: mutable EMA centres (updated in place after loss is computed).
/// - `batch_size`: `B`.
/// - `n_masked`: `M` masked tokens per sample.
/// - `n_prototypes`: `K`.
/// - `config`: iBOT configuration.
///
/// # Errors
/// Propagates dimension and parameter errors from component functions.
pub fn ibot_loss(
    student_cls: &[f32],
    teacher_cls: &[f32],
    student_patches: &[f32],
    teacher_patches: &[f32],
    centers: &mut IBotCenters,
    batch_size: usize,
    n_masked: usize,
    n_prototypes: usize,
    config: &IBotConfig,
) -> SslResult<IBotResult> {
    if batch_size == 0 {
        return Err(SslError::EmptyInput);
    }
    if n_prototypes < 2 {
        return Err(SslError::NumPrototypesTooSmall);
    }
    let cls_expected = batch_size * n_prototypes;
    if student_cls.len() != cls_expected {
        return Err(SslError::DimensionMismatch {
            expected: cls_expected,
            got: student_cls.len(),
        });
    }
    if teacher_cls.len() != cls_expected {
        return Err(SslError::DimensionMismatch {
            expected: cls_expected,
            got: teacher_cls.len(),
        });
    }
    if centers.cls_center.len() != n_prototypes {
        return Err(SslError::DimensionMismatch {
            expected: n_prototypes,
            got: centers.cls_center.len(),
        });
    }
    if centers.patch_center.len() != n_prototypes {
        return Err(SslError::DimensionMismatch {
            expected: n_prototypes,
            got: centers.patch_center.len(),
        });
    }

    // ── CLS loss (DINO-style) ────────────────────────────────────────────────
    let t_cls_centred = subtract_center(teacher_cls, batch_size, n_prototypes, &centers.cls_center);
    let p_teacher_cls = row_softmax_t(
        &t_cls_centred,
        batch_size,
        n_prototypes,
        config.tau_teacher,
        config.eps,
    );
    let p_student_cls = row_softmax_t(
        student_cls,
        batch_size,
        n_prototypes,
        config.tau_student,
        config.eps,
    );
    let cls_ce_sum = cross_entropy_sum(
        &p_teacher_cls,
        &p_student_cls,
        batch_size,
        n_prototypes,
        config.eps,
    );
    let cls_loss = cls_ce_sum / batch_size as f32;

    // ── Teacher entropy for diagnostics ─────────────────────────────────────
    let mean_teacher_entropy = mean_entropy(&p_teacher_cls, batch_size, n_prototypes, config.eps);

    // ── MIM loss ─────────────────────────────────────────────────────────────
    let (mim_loss, n_masked_patches) = if n_masked == 0 {
        (0.0_f32, 0_usize)
    } else {
        let patch_expected = batch_size * n_masked * n_prototypes;
        if student_patches.len() != patch_expected {
            return Err(SslError::DimensionMismatch {
                expected: patch_expected,
                got: student_patches.len(),
            });
        }
        if teacher_patches.len() != patch_expected {
            return Err(SslError::DimensionMismatch {
                expected: patch_expected,
                got: teacher_patches.len(),
            });
        }
        let total_tokens = batch_size * n_masked;
        // Centre-subtract patch teacher logits.
        let t_patch_centred = subtract_center(
            teacher_patches,
            total_tokens,
            n_prototypes,
            &centers.patch_center,
        );
        let p_teacher_patch = row_softmax_t(
            &t_patch_centred,
            total_tokens,
            n_prototypes,
            config.tau_teacher,
            config.eps,
        );
        let p_student_patch = row_softmax_t(
            student_patches,
            total_tokens,
            n_prototypes,
            config.tau_student,
            config.eps,
        );
        let mim_ce_sum = cross_entropy_sum(
            &p_teacher_patch,
            &p_student_patch,
            total_tokens,
            n_prototypes,
            config.eps,
        );
        (mim_ce_sum / total_tokens as f32, total_tokens)
    };

    // ── Update centres via EMA ───────────────────────────────────────────────
    ibot_update_centers(
        centers,
        teacher_cls,
        teacher_patches,
        batch_size,
        n_masked,
        n_prototypes,
        config.center_momentum,
    )?;

    let total_loss = config.lambda_cls * cls_loss + config.lambda_patch * mim_loss;
    Ok(IBotResult {
        total_loss,
        cls_loss,
        mim_loss,
        n_masked_patches,
        mean_teacher_entropy,
    })
}

/// Update the EMA centres from the current batch of teacher outputs.
///
/// ```text
///   c ← m · c + (1 − m) · mean_batch(teacher_logits)
/// ```
///
/// Both CLS and patch centres are updated independently.  When `n_masked == 0`
/// the patch centre is left unchanged.
///
/// # Errors
/// - [`SslError::InvalidMomentum`] if `momentum ∉ [0, 1]`.
/// - [`SslError::DimensionMismatch`] for shape mismatches.
/// - [`SslError::EmptyInput`] if `batch_size == 0`.
pub fn ibot_update_centers(
    centers: &mut IBotCenters,
    teacher_cls: &[f32],
    teacher_patches: &[f32],
    batch_size: usize,
    n_masked: usize,
    n_prototypes: usize,
    momentum: f32,
) -> SslResult<()> {
    if !(momentum.is_finite() && (0.0..=1.0).contains(&momentum)) {
        return Err(SslError::InvalidMomentum { momentum });
    }
    if batch_size == 0 {
        return Err(SslError::EmptyInput);
    }
    if centers.cls_center.len() != n_prototypes {
        return Err(SslError::DimensionMismatch {
            expected: n_prototypes,
            got: centers.cls_center.len(),
        });
    }
    if teacher_cls.len() != batch_size * n_prototypes {
        return Err(SslError::DimensionMismatch {
            expected: batch_size * n_prototypes,
            got: teacher_cls.len(),
        });
    }

    // ── CLS centre ────────────────────────────────────────────────────────────
    let inv_b = 1.0_f32 / batch_size as f32;
    for j in 0..n_prototypes {
        let mut mean_j = 0.0_f32;
        for i in 0..batch_size {
            mean_j += teacher_cls[i * n_prototypes + j];
        }
        mean_j *= inv_b;
        centers.cls_center[j] = momentum * centers.cls_center[j] + (1.0 - momentum) * mean_j;
    }

    // ── Patch centre (only when there are masked tokens) ─────────────────────
    if n_masked > 0 {
        let total_tokens = batch_size * n_masked;
        if centers.patch_center.len() != n_prototypes {
            return Err(SslError::DimensionMismatch {
                expected: n_prototypes,
                got: centers.patch_center.len(),
            });
        }
        if teacher_patches.len() != total_tokens * n_prototypes {
            return Err(SslError::DimensionMismatch {
                expected: total_tokens * n_prototypes,
                got: teacher_patches.len(),
            });
        }
        let inv_t = 1.0_f32 / total_tokens as f32;
        for j in 0..n_prototypes {
            let mut mean_j = 0.0_f32;
            for i in 0..total_tokens {
                mean_j += teacher_patches[i * n_prototypes + j];
            }
            mean_j *= inv_t;
            centers.patch_center[j] =
                momentum * centers.patch_center[j] + (1.0 - momentum) * mean_j;
        }
    }

    Ok(())
}

/// Generate a random binary patch mask using the Fisher-Yates-based scheme.
///
/// Returns a `Vec<bool>` of length `n_patches` where `true` means *masked*.
/// Exactly `floor(n_patches × mask_ratio)` positions are set to `true`.
///
/// # Errors
/// - [`SslError::InvalidMaskRatio`] if `mask_ratio ∉ [0, 1)`.
/// - [`SslError::EmptyInput`] if `n_patches == 0`.
pub fn ibot_random_patch_mask(
    n_patches: usize,
    mask_ratio: f32,
    rng: &mut LcgRng,
) -> SslResult<Vec<bool>> {
    if n_patches == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(mask_ratio.is_finite() && (0.0..1.0).contains(&mask_ratio)) {
        return Err(SslError::InvalidMaskRatio { ratio: mask_ratio });
    }
    let n_masked = (n_patches as f32 * mask_ratio) as usize;
    // Initialise indices 0..n_patches, shuffle, first n_masked → masked.
    let mut indices: Vec<usize> = (0..n_patches).collect();
    rng.shuffle(&mut indices);
    let mut mask = vec![false; n_patches];
    for &idx in &indices[..n_masked] {
        mask[idx] = true;
    }
    Ok(mask)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_logits(rows: usize, k: usize, scale: f32) -> Vec<f32> {
        (0..rows * k).map(|i| (i as f32 * scale).sin()).collect()
    }

    fn default_config() -> IBotConfig {
        IBotConfig::default()
    }

    fn default_centers(k: usize) -> IBotCenters {
        ibot_centers_init(k)
    }

    // ── test 1: CLS loss is finite and non-negative ───────────────────────────

    #[test]
    fn cls_loss_finite_nonneg() {
        let b = 4;
        let k = 16;
        let cfg = default_config();
        let s = make_logits(b, k, 0.013);
        let t = make_logits(b, k, 0.027);
        let centers = default_centers(k);
        let loss = ibot_cls_loss(&s, &t, &centers, b, k, &cfg).unwrap();
        assert!(loss.is_finite(), "CLS loss should be finite, got {loss}");
        assert!(loss >= 0.0, "CLS loss should be non-negative, got {loss}");
    }

    // ── test 2: MIM loss is finite and non-negative ───────────────────────────

    #[test]
    fn mim_loss_finite_nonneg() {
        let b = 4;
        let m = 8;
        let k = 16;
        let cfg = default_config();
        let s = make_logits(b * m, k, 0.013);
        let t = make_logits(b * m, k, 0.029);
        let loss = ibot_mim_loss(&s, &t, b, m, k, &cfg).unwrap();
        assert!(loss.is_finite(), "MIM loss should be finite, got {loss}");
        assert!(loss >= 0.0, "MIM loss should be non-negative, got {loss}");
    }

    // ── test 3: total loss == lambda_cls * L_cls + lambda_patch * L_mim ──────

    #[test]
    fn total_loss_is_weighted_sum() {
        let b = 4;
        let m = 6;
        let k = 8;
        let mut cfg = default_config();
        cfg.n_prototypes = k;
        cfg.lambda_cls = 0.5;
        cfg.lambda_patch = 2.0;
        let mut centers = default_centers(k);
        let s_cls = make_logits(b, k, 0.011);
        let t_cls = make_logits(b, k, 0.023);
        let s_patch = make_logits(b * m, k, 0.031);
        let t_patch = make_logits(b * m, k, 0.041);
        let result = ibot_loss(
            &s_cls,
            &t_cls,
            &s_patch,
            &t_patch,
            &mut centers,
            b,
            m,
            k,
            &cfg,
        )
        .unwrap();
        let expected = cfg.lambda_cls * result.cls_loss + cfg.lambda_patch * result.mim_loss;
        assert!(
            (result.total_loss - expected).abs() < 1e-4,
            "total_loss = {}, expected {expected}",
            result.total_loss
        );
    }

    // ── test 4: lambda_patch = 0 → total_loss = lambda_cls * cls_loss ─────────

    #[test]
    fn lambda_patch_zero_suppresses_mim() {
        let b = 4;
        let m = 5;
        let k = 8;
        let mut cfg = default_config();
        cfg.n_prototypes = k;
        cfg.lambda_patch = 0.0;
        let mut centers = default_centers(k);
        let s_cls = make_logits(b, k, 0.017);
        let t_cls = make_logits(b, k, 0.019);
        let s_patch = make_logits(b * m, k, 0.021);
        let t_patch = make_logits(b * m, k, 0.023);
        let result = ibot_loss(
            &s_cls,
            &t_cls,
            &s_patch,
            &t_patch,
            &mut centers,
            b,
            m,
            k,
            &cfg,
        )
        .unwrap();
        let expected = cfg.lambda_cls * result.cls_loss;
        assert!(
            (result.total_loss - expected).abs() < 1e-5,
            "total_loss = {}, expected {expected}",
            result.total_loss
        );
    }

    // ── test 5: n_masked == 0 → mim_loss == 0, no NaN ────────────────────────

    #[test]
    fn zero_masked_tokens_gives_zero_mim_loss() {
        let b = 4;
        let m = 0;
        let k = 8;
        let mut cfg = default_config();
        cfg.n_prototypes = k;
        let mut centers = default_centers(k);
        let s_cls = make_logits(b, k, 0.017);
        let t_cls = make_logits(b, k, 0.019);
        // Empty slices for patches when m == 0.
        let s_patch: Vec<f32> = vec![];
        let t_patch: Vec<f32> = vec![];
        let result = ibot_loss(
            &s_cls,
            &t_cls,
            &s_patch,
            &t_patch,
            &mut centers,
            b,
            m,
            k,
            &cfg,
        )
        .unwrap();
        assert_eq!(
            result.mim_loss, 0.0,
            "MIM loss must be 0 when n_masked == 0"
        );
        assert!(result.total_loss.is_finite());
    }

    // ── test 6: centre update is correct EMA ─────────────────────────────────

    #[test]
    fn center_update_ema_correctness() {
        let b = 2;
        let m = 0;
        let k = 4;
        let momentum = 0.9;
        let mut centers = IBotCenters {
            cls_center: vec![1.0_f32; k],
            patch_center: vec![0.0_f32; k],
        };
        // Constant teacher CLS logits = 5.0 everywhere → batch mean = 5.0
        let t_cls = vec![5.0_f32; b * k];
        let t_patch: Vec<f32> = vec![];
        ibot_update_centers(&mut centers, &t_cls, &t_patch, b, m, k, momentum).unwrap();
        // expected: 0.9 * 1.0 + 0.1 * 5.0 = 0.9 + 0.5 = 1.4
        for &v in &centers.cls_center {
            assert!(
                (v - 1.4).abs() < 1e-5,
                "EMA centre wrong: expected 1.4, got {v}"
            );
        }
    }

    // ── test 7: random patch mask — exact count ───────────────────────────────

    #[test]
    fn random_patch_mask_exact_count() {
        let mut rng = LcgRng::new(42);
        let n = 196;
        let ratio = 0.75;
        let mask = ibot_random_patch_mask(n, ratio, &mut rng).unwrap();
        let expected_masked = (n as f32 * ratio) as usize; // 147
        let actual_masked = mask.iter().filter(|&&v| v).count();
        assert_eq!(
            actual_masked, expected_masked,
            "expected {expected_masked} masked, got {actual_masked}"
        );
        assert_eq!(mask.len(), n);
    }

    // ── test 8: n_prototypes == 0 (or 1) → error ─────────────────────────────

    #[test]
    fn zero_prototypes_returns_error() {
        let b = 2;
        let k = 0; // invalid
        let cfg = IBotConfig {
            n_prototypes: 8192,
            ..default_config()
        };
        let s = vec![0.0_f32; b]; // wrong size, but we expect error on n_prototypes check
        let t = vec![0.0_f32; b];
        let centers = IBotCenters {
            cls_center: vec![],
            patch_center: vec![],
        };
        let result = ibot_cls_loss(&s, &t, &centers, b, k, &cfg);
        assert!(result.is_err(), "Should error for n_prototypes < 2");
    }

    // ── test 9: tau_student = 0 → error from config validation ───────────────

    #[test]
    fn tau_student_zero_rejected_by_config() {
        let result = IBotConfig::new(8, 0.0, 0.04, 0.9, 1.0, 1.0, 1e-6);
        assert!(result.is_err(), "tau_student = 0 must be rejected");
    }

    // ── test 10: batch_size = 1 works ────────────────────────────────────────

    #[test]
    fn batch_size_one_works() {
        let b = 1;
        let m = 3;
        let k = 4;
        let cfg = IBotConfig {
            n_prototypes: k,
            ..default_config()
        };
        let mut centers = default_centers(k);
        let s_cls = make_logits(b, k, 0.031);
        let t_cls = make_logits(b, k, 0.037);
        let s_patch = make_logits(b * m, k, 0.041);
        let t_patch = make_logits(b * m, k, 0.043);
        let result = ibot_loss(
            &s_cls,
            &t_cls,
            &s_patch,
            &t_patch,
            &mut centers,
            b,
            m,
            k,
            &cfg,
        )
        .unwrap();
        assert!(result.total_loss.is_finite());
        assert!(result.cls_loss.is_finite());
        assert!(result.mim_loss.is_finite());
    }

    // ── test 11: ibot_centers_init produces zero centres ─────────────────────

    #[test]
    fn centers_init_all_zeros() {
        let k = 64;
        let c = ibot_centers_init(k);
        assert_eq!(c.cls_center.len(), k);
        assert_eq!(c.patch_center.len(), k);
        assert!(c.cls_center.iter().all(|&v| v == 0.0));
        assert!(c.patch_center.iter().all(|&v| v == 0.0));
    }

    // ── test 12: mean_teacher_entropy ∈ [0, ln(K)] ───────────────────────────

    #[test]
    fn teacher_entropy_in_valid_range() {
        let b = 8;
        let m = 4;
        let k = 16;
        let cfg = IBotConfig {
            n_prototypes: k,
            ..default_config()
        };
        let mut centers = default_centers(k);
        let s_cls = make_logits(b, k, 0.011);
        let t_cls = make_logits(b, k, 0.013);
        let s_patch = make_logits(b * m, k, 0.015);
        let t_patch = make_logits(b * m, k, 0.017);
        let result = ibot_loss(
            &s_cls,
            &t_cls,
            &s_patch,
            &t_patch,
            &mut centers,
            b,
            m,
            k,
            &cfg,
        )
        .unwrap();
        let max_entropy = (k as f32).ln();
        assert!(
            result.mean_teacher_entropy >= 0.0,
            "entropy must be >= 0, got {}",
            result.mean_teacher_entropy
        );
        assert!(
            result.mean_teacher_entropy <= max_entropy + 1e-4,
            "entropy must be <= ln(K)={max_entropy}, got {}",
            result.mean_teacher_entropy
        );
    }

    // ── test 13: identical student and teacher → low CLS loss ─────────────────

    #[test]
    fn identical_student_teacher_low_cls_loss() {
        let b = 4;
        let k = 8;
        let mut cfg = default_config();
        // Make the two temperatures equal so the ratio is 1 and the
        // cross-entropy equals the entropy of the distribution.
        cfg.tau_student = 0.04;
        cfg.tau_teacher = 0.04;
        let logits: Vec<f32> = (0..b * k).map(|i| (i as f32) * 0.1).collect();
        let centers = default_centers(k);
        let loss = ibot_cls_loss(&logits, &logits, &centers, b, k, &cfg).unwrap();
        // CE(p, p) = H(p) >= 0. For non-uniform p, H is a positive finite value.
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
        // CE cannot exceed ln(K) for a valid probability distribution.
        assert!(loss <= (k as f32).ln() + 1e-3, "loss = {loss}");
    }

    // ── test 14: n_masked_patches count in IBotResult ─────────────────────────

    #[test]
    fn n_masked_patches_count_is_batch_times_m() {
        let b = 3;
        let m = 7;
        let k = 4;
        let cfg = IBotConfig {
            n_prototypes: k,
            ..default_config()
        };
        let mut centers = default_centers(k);
        let s_cls = make_logits(b, k, 0.051);
        let t_cls = make_logits(b, k, 0.053);
        let s_patch = make_logits(b * m, k, 0.057);
        let t_patch = make_logits(b * m, k, 0.059);
        let result = ibot_loss(
            &s_cls,
            &t_cls,
            &s_patch,
            &t_patch,
            &mut centers,
            b,
            m,
            k,
            &cfg,
        )
        .unwrap();
        assert_eq!(
            result.n_masked_patches,
            b * m,
            "expected n_masked_patches = {}, got {}",
            b * m,
            result.n_masked_patches
        );
    }

    // ── test 15: mask_ratio = 0 → no masked patches ───────────────────────────

    #[test]
    fn mask_ratio_zero_produces_no_masked_patches() {
        let mut rng = LcgRng::new(7);
        let mask = ibot_random_patch_mask(100, 0.0, &mut rng).unwrap();
        assert!(
            mask.iter().all(|&v| !v),
            "ratio=0 should produce no masked patches"
        );
    }

    // ── test 16: invalid mask ratio → error ──────────────────────────────────

    #[test]
    fn invalid_mask_ratio_rejected() {
        let mut rng = LcgRng::new(99);
        assert!(ibot_random_patch_mask(64, 1.0, &mut rng).is_err());
        assert!(ibot_random_patch_mask(64, 1.5, &mut rng).is_err());
        assert!(ibot_random_patch_mask(64, -0.1, &mut rng).is_err());
    }
}

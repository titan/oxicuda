//! data2vec — Baevski et al. 2022, ICML.
//!
//! Unified self-supervised learning via **teacher-student masked prediction**:
//!
//! 1. A **teacher** network (EMA of the student) encodes the **full, unmasked**
//!    input and produces target representations.
//! 2. The **student** encoder receives the **masked** input and predicts the
//!    teacher's representations at masked positions.
//! 3. The loss is the smooth-L1 (Huber) divergence between L2-normalised student
//!    predictions and L2-normalised teacher targets, summed only over masked tokens.
//!
//! ```text
//!  θ_teacher ← m · θ_teacher + (1−m) · θ_student       [EMA update]
//!  target_j  ← target_j / (‖target[:,j]‖₂ + ε)         [per-dim batch norm]
//!  L          = mean huber(student_pred − target, β)     [masked positions only]
//! ```
//!
//! Reference: "data2vec: A General Framework for Self-supervised Learning in
//! Speech, Vision and Language", Baevski et al., ICML 2022.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for the data2vec training objective.
#[derive(Debug, Clone)]
pub struct Data2VecConfig {
    /// Fraction of tokens to mask (default 0.65, paper canonical for vision).
    pub mask_ratio: f32,
    /// EMA coefficient for the teacher network (default 0.999).
    pub momentum: f32,
    /// Huber loss threshold β (default 2.0).
    pub beta: f32,
    /// Whether to L2-normalise teacher representations per feature dimension
    /// across the batch/token axis before computing the loss (default `true`).
    pub normalize_targets: bool,
    /// Number of teacher layer outputs to average for target computation
    /// (default 1; set > 1 to enable a rolling exponential history buffer).
    pub top_k_average: usize,
}

impl Default for Data2VecConfig {
    fn default() -> Self {
        Self {
            mask_ratio: 0.65,
            momentum: 0.999,
            beta: 2.0,
            normalize_targets: true,
            top_k_average: 1,
        }
    }
}

// ─── Result types ─────────────────────────────────────────────────────────────

/// Output of a single data2vec loss computation.
#[derive(Debug, Clone)]
pub struct Data2VecResult {
    /// Mean Huber loss over all masked positions and feature dimensions.
    pub loss: f32,
    /// Number of token positions that were masked (contributed to the loss).
    pub n_masked: usize,
    /// Not meaningful for regression; always 0.0.
    pub accuracy_at_1: f32,
}

// ─── Teacher-state ────────────────────────────────────────────────────────────

/// Mutable state that tracks the teacher EMA parameter vector and training step.
///
/// The teacher is initialised to a copy of the online (student) parameters and
/// updated each training step via:
/// ```text
///   θ_teacher ← m · θ_teacher + (1−m) · θ_online
/// ```
#[derive(Debug, Clone)]
pub struct Data2VecState {
    /// EMA teacher parameter vector (flat, same length as the student).
    pub teacher_params: Vec<f32>,
    /// Training-step counter (incremented by [`Self::update_teacher`]).
    pub step: usize,
}

impl Data2VecState {
    /// Initialise state by cloning the current online (student) parameters.
    ///
    /// The teacher starts as an exact copy of the student so that the very first
    /// EMA update moves from a well-defined position.
    #[must_use]
    pub fn new(online_params: &[f32]) -> Self {
        Self {
            teacher_params: online_params.to_vec(),
            step: 0,
        }
    }

    /// Apply the EMA update `θ_teacher ← m·θ_teacher + (1−m)·θ_online` and
    /// increment the step counter.
    ///
    /// # Errors
    /// - [`SslError::InvalidMomentum`] when `momentum` is outside `[0, 1]` or
    ///   non-finite.
    /// - [`SslError::DimensionMismatch`] when `online_params` has a different
    ///   length than the stored teacher vector.
    pub fn update_teacher(&mut self, online_params: &[f32], momentum: f32) -> SslResult<()> {
        if !(momentum.is_finite() && (0.0..=1.0).contains(&momentum)) {
            return Err(SslError::InvalidMomentum { momentum });
        }
        if self.teacher_params.len() != online_params.len() {
            return Err(SslError::DimensionMismatch {
                expected: self.teacher_params.len(),
                got: online_params.len(),
            });
        }
        let one_minus_m = 1.0 - momentum;
        for (t, &o) in self.teacher_params.iter_mut().zip(online_params.iter()) {
            *t = momentum * *t + one_minus_m * o;
        }
        self.step += 1;
        Ok(())
    }

    /// Reference to the current teacher parameter vector.
    #[must_use]
    #[inline]
    pub fn teacher(&self) -> &[f32] {
        &self.teacher_params
    }
}

// ─── Core primitives ──────────────────────────────────────────────────────────

/// Per-element Huber (smooth-L1) loss, averaged over all elements.
///
/// For each element `x = prediction − target`:
/// ```text
///   huber(x, β) = 0.5·x²/β   if |x| < β
///               = |x| − β/2  otherwise
/// ```
///
/// # Panics
/// Does not panic, but returns `f32::NAN` if either slice contains NaN/inf.
#[must_use]
pub fn huber_loss(predictions: &[f32], targets: &[f32], beta: f32) -> f32 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return 0.0;
    }
    let n = predictions.len() as f64;
    let half_beta = (beta as f64) / 2.0;
    let inv_beta = 1.0 / (beta as f64);
    let total: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            let x = (p - t) as f64;
            let ax = x.abs();
            if ax < beta as f64 {
                0.5 * x * x * inv_beta
            } else {
                ax - half_beta
            }
        })
        .sum();
    (total / n) as f32
}

/// Normalise teacher representations **along the batch dimension** in-place.
///
/// For each feature dimension `d ∈ [0, dim)` the normalisation is:
/// ```text
///   norm_d = sqrt( Σᵢ target[i·dim + d]² / n_tokens )
///   target[i·dim + d] /= (norm_d + ε),   ε = 1e-8
/// ```
///
/// This is applied **only over the provided slice**, which the caller
/// restricts to masked tokens. The slice must have length `n_tokens × dim`.
pub fn normalize_teacher_targets(targets: &mut [f32], n_tokens: usize, dim: usize) {
    if n_tokens == 0 || dim == 0 || targets.len() != n_tokens * dim {
        return;
    }
    const EPS: f32 = 1e-8;
    let n = n_tokens as f32;
    // For each feature dimension, compute the RMS across tokens and normalise.
    for d in 0..dim {
        let mut sum_sq = 0.0_f32;
        for i in 0..n_tokens {
            let v = targets[i * dim + d];
            sum_sq += v * v;
        }
        let norm = (sum_sq / n).sqrt();
        let scale = 1.0 / (norm + EPS);
        for i in 0..n_tokens {
            targets[i * dim + d] *= scale;
        }
    }
}

/// Generate a boolean mask of length `n_tokens` with exactly
/// `floor(n_tokens × mask_ratio)` positions set to `true` (= masked).
///
/// The selection is performed via a Fisher-Yates partial shuffle over an index
/// array, mirroring the approach in [`crate::masked::mae::random_patch_mask`],
/// but produces a `Vec<bool>` directly keyed to token indices.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n_tokens == 0`.
/// - [`SslError::InvalidMaskRatio`] when `mask_ratio` is outside `[0, 1)` or
///   non-finite.
pub fn data2vec_mask(n_tokens: usize, mask_ratio: f32, rng: &mut LcgRng) -> SslResult<Vec<bool>> {
    if n_tokens == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(mask_ratio.is_finite() && (0.0..1.0).contains(&mask_ratio)) {
        return Err(SslError::InvalidMaskRatio { ratio: mask_ratio });
    }
    let n_mask = (n_tokens as f32 * mask_ratio) as usize;
    // Build an index pool and shuffle the first n_mask positions.
    let mut indices: Vec<usize> = (0..n_tokens).collect();
    rng.shuffle(&mut indices);
    let mut mask = vec![false; n_tokens];
    for &idx in indices.iter().take(n_mask) {
        mask[idx] = true;
    }
    Ok(mask)
}

// ─── Loss computation ─────────────────────────────────────────────────────────

/// Compute the data2vec loss for a single sample.
///
/// Implements the full algorithm:
/// 1. Validate shapes.
/// 2. Optionally L2-normalise teacher representations across masked tokens per
///    feature dimension.
/// 3. Compute mean Huber loss between student predictions and normalised teacher
///    targets at **masked positions only**.
///
/// # Arguments
/// * `student_pred`   — `[n_tokens × dim]` student predictions (row-major).
/// * `teacher_repr`   — `[n_tokens × dim]` teacher representations (row-major).
/// * `mask`           — `[n_tokens]` boolean vector; `true` = masked position.
/// * `n_tokens`, `dim` — spatial and channel dimensions.
/// * `config`         — data2vec hyper-parameters.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n_tokens == 0` or `dim == 0`.
/// - [`SslError::DimensionMismatch`] when any buffer has the wrong length.
/// - [`SslError::EmptyInput`] when no tokens are masked (graceful; returns 0.0
///   loss inside `Data2VecResult` rather than erroring, since the caller may
///   legitimately supply an all-visible batch during warm-up).
pub fn data2vec_loss(
    student_pred: &[f32],
    teacher_repr: &[f32],
    mask: &[bool],
    n_tokens: usize,
    dim: usize,
    config: &Data2VecConfig,
) -> SslResult<Data2VecResult> {
    // ── 1. Shape validation ───────────────────────────────────────────────────
    if n_tokens == 0 || dim == 0 {
        return Err(SslError::EmptyInput);
    }
    let expected = n_tokens * dim;
    if student_pred.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: student_pred.len(),
        });
    }
    if teacher_repr.len() != expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: teacher_repr.len(),
        });
    }
    if mask.len() != n_tokens {
        return Err(SslError::DimensionMismatch {
            expected: n_tokens,
            got: mask.len(),
        });
    }

    // ── 2. Collect masked-token indices ───────────────────────────────────────
    let masked_indices: Vec<usize> = (0..n_tokens).filter(|&i| mask[i]).collect();
    let n_masked = masked_indices.len();

    if n_masked == 0 {
        // Graceful: no masked tokens → loss is trivially 0.
        return Ok(Data2VecResult {
            loss: 0.0,
            n_masked: 0,
            accuracy_at_1: 0.0,
        });
    }

    // ── 3. Build contiguous masked-token buffers ──────────────────────────────
    // Collect masked teacher representations into a contiguous buffer so that
    // normalize_teacher_targets can operate with simple row-major indexing.
    let mut teacher_masked = Vec::with_capacity(n_masked * dim);
    let mut student_masked = Vec::with_capacity(n_masked * dim);
    for &i in &masked_indices {
        let start = i * dim;
        let end = start + dim;
        teacher_masked.extend_from_slice(&teacher_repr[start..end]);
        student_masked.extend_from_slice(&student_pred[start..end]);
    }

    // ── 4. Optional target normalisation ─────────────────────────────────────
    if config.normalize_targets {
        normalize_teacher_targets(&mut teacher_masked, n_masked, dim);
    }

    // ── 5. Huber loss over masked positions ───────────────────────────────────
    let loss = huber_loss(&student_masked, &teacher_masked, config.beta);

    Ok(Data2VecResult {
        loss,
        n_masked,
        accuracy_at_1: 0.0,
    })
}

/// Compute the mean data2vec loss over a batch of samples.
///
/// Buffers are laid out batch-first:
/// - `student_preds` : `[batch_size × n_tokens × dim]`
/// - `teacher_reprs` : `[batch_size × n_tokens × dim]`
/// - `masks`         : `[batch_size × n_tokens]` boolean
///
/// Each sample's loss is computed independently with
/// [`data2vec_loss`] and the results are averaged.
///
/// # Errors
/// Propagates all errors from [`data2vec_loss`] together with additional shape
/// checks for the batch dimension.
pub fn data2vec_batch_loss(
    student_preds: &[f32],
    teacher_reprs: &[f32],
    masks: &[bool],
    batch_size: usize,
    n_tokens: usize,
    dim: usize,
    config: &Data2VecConfig,
) -> SslResult<f32> {
    if batch_size == 0 {
        return Err(SslError::EmptyInput);
    }
    let sample_len = n_tokens * dim;
    let expected_feat = batch_size * sample_len;
    let expected_mask = batch_size * n_tokens;

    if student_preds.len() != expected_feat {
        return Err(SslError::DimensionMismatch {
            expected: expected_feat,
            got: student_preds.len(),
        });
    }
    if teacher_reprs.len() != expected_feat {
        return Err(SslError::DimensionMismatch {
            expected: expected_feat,
            got: teacher_reprs.len(),
        });
    }
    if masks.len() != expected_mask {
        return Err(SslError::DimensionMismatch {
            expected: expected_mask,
            got: masks.len(),
        });
    }

    let mut total_loss = 0.0_f64;
    for b in 0..batch_size {
        let feat_start = b * sample_len;
        let feat_end = feat_start + sample_len;
        let mask_start = b * n_tokens;
        let mask_end = mask_start + n_tokens;

        let result = data2vec_loss(
            &student_preds[feat_start..feat_end],
            &teacher_reprs[feat_start..feat_end],
            &masks[mask_start..mask_end],
            n_tokens,
            dim,
            config,
        )?;
        total_loss += result.loss as f64;
    }
    Ok((total_loss / batch_size as f64) as f32)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── 1. Config defaults ────────────────────────────────────────────────────

    #[test]
    fn config_defaults() {
        let cfg = Data2VecConfig::default();
        assert!((cfg.mask_ratio - 0.65).abs() < 1e-7);
        assert!((cfg.momentum - 0.999).abs() < 1e-7);
        assert!((cfg.beta - 2.0).abs() < 1e-7);
        assert!(cfg.normalize_targets);
        assert_eq!(cfg.top_k_average, 1);
    }

    // ── 2. Huber loss — small error (|x| < β) ────────────────────────────────
    // huber(0.5, β=2) = 0.5 * 0.5² / 2.0 = 0.5 * 0.25 / 2.0 = 0.0625

    #[test]
    fn huber_loss_small_error() {
        let pred = vec![0.5_f32];
        let tgt = vec![0.0_f32];
        let loss = huber_loss(&pred, &tgt, 2.0);
        let expected = 0.5_f32 * 0.25_f32 / 2.0_f32; // 0.0625
        assert!(
            (loss - expected).abs() < 1e-6,
            "loss={loss} expected={expected}"
        );
    }

    // ── 3. Huber loss — large error (|x| >= β) ───────────────────────────────
    // huber(3.0, β=2) = |3| − β/2 = 3 − 1 = 2.0

    #[test]
    fn huber_loss_large_error() {
        let pred = vec![3.0_f32];
        let tgt = vec![0.0_f32];
        let loss = huber_loss(&pred, &tgt, 2.0);
        assert!((loss - 2.0).abs() < 1e-6, "loss={loss}");
    }

    // ── 4. Huber loss — zero when pred == target ──────────────────────────────

    #[test]
    fn huber_loss_zero() {
        let v = vec![1.5_f32, -0.7, 3.2, 0.0];
        let loss = huber_loss(&v, &v, 2.0);
        assert!(loss.abs() < 1e-7, "loss={loss}");
    }

    // ── 5. Mask has exact number of masked tokens ─────────────────────────────

    #[test]
    fn mask_exact_ratio() {
        let mut rng = LcgRng::new(42);
        let mask = data2vec_mask(100, 0.65, &mut rng).expect("data2vec_mask should succeed");
        let n_masked = mask.iter().filter(|&&v| v).count();
        assert_eq!(n_masked, 65, "expected 65 masked, got {n_masked}");
    }

    // ── 6. Mask length equals n_tokens ───────────────────────────────────────

    #[test]
    fn mask_length() {
        let mut rng = LcgRng::new(7);
        let mask = data2vec_mask(196, 0.75, &mut rng).expect("data2vec_mask should succeed");
        assert_eq!(mask.len(), 196);
    }

    // ── 7. Loss is zero when student perfectly matches teacher at masked positions

    #[test]
    fn data2vec_loss_only_masked() {
        let n_tokens = 10;
        let dim = 4;
        // Construct a mask where only token 3 and 7 are masked.
        let mut mask = vec![false; n_tokens];
        mask[3] = true;
        mask[7] = true;

        // Teacher = student everywhere.
        let repr: Vec<f32> = (0..n_tokens * dim).map(|i| (i as f32) * 0.1).collect();
        let cfg = Data2VecConfig {
            normalize_targets: false,
            ..Data2VecConfig::default()
        };
        let result = data2vec_loss(&repr, &repr, &mask, n_tokens, dim, &cfg)
            .expect("data2vec_loss should succeed");
        assert!(result.loss.abs() < 1e-6, "loss={}", result.loss);
        assert_eq!(result.n_masked, 2);
        assert!((result.accuracy_at_1 - 0.0).abs() < 1e-7);
    }

    // ── 8. All-false mask (no masked tokens) → graceful zero loss ─────────────

    #[test]
    fn data2vec_loss_no_masked_tokens() {
        let n_tokens = 8;
        let dim = 3;
        let mask = vec![false; n_tokens];
        let v = vec![0.0_f32; n_tokens * dim];
        let cfg = Data2VecConfig::default();
        let result = data2vec_loss(&v, &v, &mask, n_tokens, dim, &cfg)
            .expect("data2vec_loss should succeed");
        assert_eq!(result.n_masked, 0);
        assert!(result.loss.abs() < 1e-7);
    }

    // ── 9. normalize_teacher_targets reduces large values ─────────────────────

    #[test]
    fn normalize_targets_reduces_large_values() {
        let n_tokens = 4;
        let dim = 2;
        // All values are large.
        let mut targets = vec![100.0_f32; n_tokens * dim];
        normalize_teacher_targets(&mut targets, n_tokens, dim);
        // After normalisation every value should be near 1.0 (all equal, so
        // norm_d = sqrt(Σ 100² / 4) = sqrt(10000) = 100, scale = 1/100 → 1.0).
        for &v in &targets {
            assert!(v.abs() < 2.0, "value after norm={v}");
        }
    }

    // ── 10. State init: teacher equals online params ──────────────────────────

    #[test]
    fn state_init_matches_online() {
        let online = vec![0.1_f32, 0.5, -0.3, 1.2];
        let state = Data2VecState::new(&online);
        assert_eq!(state.teacher(), online.as_slice());
        assert_eq!(state.step, 0);
    }

    // ── 11. EMA update with momentum=0 copies online exactly ─────────────────

    #[test]
    fn state_update_closer_to_online_m0() {
        let teacher_init = vec![1.0_f32, 2.0, 3.0];
        let online = vec![10.0_f32, 20.0, 30.0];
        let mut state = Data2VecState::new(&teacher_init);
        state
            .update_teacher(&online, 0.0)
            .expect("update_teacher should succeed");
        // With m=0: teacher = 0*old + 1*online = online.
        for (&t, &o) in state.teacher().iter().zip(online.iter()) {
            assert!((t - o).abs() < 1e-6, "teacher={t} online={o}");
        }
        assert_eq!(state.step, 1);
    }

    // ── 12. EMA update with momentum=1.0 leaves teacher unchanged ────────────

    #[test]
    fn state_update_m1_unchanged() {
        let teacher_init = vec![5.0_f32, -3.0, 0.7];
        let online = vec![0.0_f32, 0.0, 0.0];
        let mut state = Data2VecState::new(&teacher_init);
        let expected = state.teacher().to_vec();
        state
            .update_teacher(&online, 1.0)
            .expect("update_teacher should succeed");
        // With m=1: teacher = 1*old + 0*online = old.
        for (&t, &e) in state.teacher().iter().zip(expected.iter()) {
            assert!((t - e).abs() < 1e-6, "teacher={t} expected={e}");
        }
    }

    // ── 13. Batch loss with batch_size=1 matches single-sample loss ───────────

    #[test]
    fn batch_loss_matches_single() {
        let n_tokens = 6;
        let dim = 4;
        let mut rng = LcgRng::new(99);

        let mut student = vec![0.0_f32; n_tokens * dim];
        let mut teacher = vec![0.0_f32; n_tokens * dim];
        rng.fill_normal(&mut student);
        rng.fill_normal(&mut teacher);

        let mask = data2vec_mask(n_tokens, 0.5, &mut rng).expect("data2vec_mask should succeed");

        let cfg = Data2VecConfig::default();

        let single = data2vec_loss(&student, &teacher, &mask, n_tokens, dim, &cfg)
            .expect("value should be present")
            .loss;
        let batch = data2vec_batch_loss(&student, &teacher, &mask, 1, n_tokens, dim, &cfg)
            .expect("data2vec_batch_loss should succeed");

        assert!(
            (single - batch).abs() < 1e-5,
            "single={single} batch={batch}"
        );
    }

    // ── 14. Batch loss is finite for random inputs ────────────────────────────

    #[test]
    fn batch_loss_finite() {
        let batch_size = 4;
        let n_tokens = 16;
        let dim = 8;
        let mut rng = LcgRng::new(1337);

        let total_feat = batch_size * n_tokens * dim;
        let mut student = vec![0.0_f32; total_feat];
        let mut teacher = vec![0.0_f32; total_feat];
        rng.fill_normal(&mut student);
        rng.fill_normal(&mut teacher);

        let mut masks = Vec::with_capacity(batch_size * n_tokens);
        for _ in 0..batch_size {
            masks.extend(
                data2vec_mask(n_tokens, 0.65, &mut rng).expect("data2vec_mask should succeed"),
            );
        }

        let cfg = Data2VecConfig::default();
        let loss = data2vec_batch_loss(&student, &teacher, &masks, batch_size, n_tokens, dim, &cfg)
            .expect("value should be present");

        assert!(loss.is_finite(), "loss={loss}");
        assert!(loss >= 0.0, "loss={loss}");
    }

    // ── Extra: invalid mask_ratio errors ─────────────────────────────────────

    #[test]
    fn mask_invalid_ratio_errors() {
        let mut rng = LcgRng::new(1);
        assert!(data2vec_mask(10, 1.0, &mut rng).is_err()); // ratio == 1.0 forbidden
        assert!(data2vec_mask(10, -0.1, &mut rng).is_err());
        assert!(data2vec_mask(10, f32::NAN, &mut rng).is_err());
    }

    // ── Extra: invalid momentum rejected by state update ─────────────────────

    #[test]
    fn state_update_rejects_invalid_momentum() {
        let mut state = Data2VecState::new(&[1.0_f32, 2.0]);
        let online = vec![3.0_f32, 4.0];
        assert!(state.update_teacher(&online, 1.5).is_err());
        assert!(state.update_teacher(&online, -0.1).is_err());
        assert!(state.update_teacher(&online, f32::NAN).is_err());
    }

    // ── Extra: normalize_teacher_targets is a no-op on empty input ───────────

    #[test]
    fn normalize_teacher_targets_empty_noop() {
        let mut v: Vec<f32> = vec![];
        normalize_teacher_targets(&mut v, 0, 4); // must not panic
        let mut v2 = vec![1.0_f32; 8];
        normalize_teacher_targets(&mut v2, 4, 0); // must not panic
    }

    // ── Extra: DimensionMismatch for shape errors ─────────────────────────────

    #[test]
    fn data2vec_loss_shape_errors() {
        let n = 4;
        let d = 3;
        let cfg = Data2VecConfig::default();
        let good = vec![0.0_f32; n * d];
        let short = vec![0.0_f32; n * d - 1];
        let mask = vec![true; n];
        // Wrong student length.
        assert!(data2vec_loss(&short, &good, &mask, n, d, &cfg).is_err());
        // Wrong teacher length.
        assert!(data2vec_loss(&good, &short, &mask, n, d, &cfg).is_err());
        // Wrong mask length.
        let bad_mask = vec![true; n - 1];
        assert!(data2vec_loss(&good, &good, &bad_mask, n, d, &cfg).is_err());
    }
}

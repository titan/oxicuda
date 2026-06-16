//! Learning without Forgetting (LwF) regularization.
//!
//! Implements the method from:
//! Li & Hoiem. "Learning without Forgetting." TPAMI 2017.
//!
//! LwF uses knowledge distillation to prevent catastrophic forgetting.
//! Before training on a new task, the old model's predictions on new-task data
//! are recorded as soft targets. During training, a distillation loss keeps the
//! student's outputs aligned with those teacher soft targets.
//!
//! **Temperature-scaled softmax:**
//! `p_i = exp(z_i / T) / Σ_j exp(z_j / T)`
//!
//! **KL distillation loss (single sample):**
//! `L_KL = T² · Σ_i p_i^{teacher} · log(p_i^{teacher} / (p_i^{student} + ε))`
//!
//! (The T² factor compensates for the 1/T² gradient scaling introduced by the
//! temperature-scaled softmax during back-propagation.)
//!
//! **Combined loss:**
//! `L_total = α · L_KL + (1 − α) · L_task`

use crate::error::{ContinualError, ContinualResult};

/// Small epsilon added inside the logarithm to avoid log(0).
const KL_EPS: f32 = 1e-8;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for LwF (Learning without Forgetting).
#[derive(Debug, Clone)]
pub struct LwfConfig {
    /// Temperature T for soft-target softmax (> 0). Default 2.0.
    pub temperature: f32,
    /// Weight of the distillation loss (α ∈ [0, 1]). Default 0.5.
    pub alpha: f32,
}

impl Default for LwfConfig {
    fn default() -> Self {
        Self {
            temperature: 2.0,
            alpha: 0.5,
        }
    }
}

impl LwfConfig {
    /// Validate the configuration fields.
    pub fn validate(&self) -> ContinualResult<()> {
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return Err(ContinualError::InvalidLambda {
                lambda: self.temperature,
            });
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(ContinualError::Internal("alpha must be in [0,1]".into()));
        }
        Ok(())
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Apply temperature-scaled softmax in-place to `out`.
///
/// Uses the max-subtraction trick for numerical stability.
fn temperature_softmax(logits: &[f32], temperature: f32, out: &mut [f32]) {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits
        .iter()
        .map(|&l| ((l - max_l) / temperature).exp())
        .sum();
    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        *o = ((l - max_l) / temperature).exp() / exp_sum;
    }
}

/// Validate that neither teacher nor student logit slices contain NaN.
fn check_logits_nan(teacher: &[f32], student: &[f32]) -> ContinualResult<()> {
    if teacher.iter().any(|v| v.is_nan()) {
        return Err(ContinualError::NanEncountered {
            location: "teacher_logits",
        });
    }
    if student.iter().any(|v| v.is_nan()) {
        return Err(ContinualError::NanEncountered {
            location: "student_logits",
        });
    }
    Ok(())
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute LwF distillation loss for a single sample.
///
/// Applies temperature-scaled softmax to both `teacher_logits` and
/// `student_logits`, then computes:
/// `T² · KL(teacher_soft ‖ student_soft)`
///
/// # Arguments
/// * `teacher_logits` — old model's logits (length = n_classes)
/// * `student_logits` — current model's logits (same length)
/// * `config` — LwF configuration
///
/// # Errors
/// Returns an error on empty input, mismatched lengths, invalid config, or
/// NaN values in the logits.
pub fn lwf_distill_loss(
    teacher_logits: &[f32],
    student_logits: &[f32],
    config: &LwfConfig,
) -> ContinualResult<f32> {
    config.validate()?;

    if teacher_logits.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    if teacher_logits.len() != student_logits.len() {
        return Err(ContinualError::DimensionMismatch {
            expected: teacher_logits.len(),
            got: student_logits.len(),
        });
    }

    check_logits_nan(teacher_logits, student_logits)?;

    let n = teacher_logits.len();
    let t = config.temperature;

    let mut p_teacher = vec![0.0_f32; n];
    let mut p_student = vec![0.0_f32; n];
    temperature_softmax(teacher_logits, t, &mut p_teacher);
    temperature_softmax(student_logits, t, &mut p_student);

    // KL(teacher ‖ student) = Σ_i p_t · log(p_t / (p_s + ε))
    let kl: f32 = p_teacher
        .iter()
        .zip(p_student.iter())
        .map(|(&pt, &ps)| pt * (pt / (ps + KL_EPS)).ln())
        .sum();

    let loss = t * t * kl;
    if !loss.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "lwf_distill_loss",
        });
    }
    Ok(loss)
}

/// Full LwF combined loss for one sample.
///
/// `L_total = α · L_distill + (1 − α) · task_loss`
///
/// # Arguments
/// * `teacher_logits` — teacher logits on old classes
/// * `student_logits` — student logits on old classes
/// * `task_loss`      — new-task cross-entropy (precomputed, must be finite)
/// * `config`         — LwF configuration
///
/// # Errors
/// Propagates all errors from [`lwf_distill_loss`], and additionally returns
/// an error if `task_loss` is NaN.
pub fn lwf_combined_loss(
    teacher_logits: &[f32],
    student_logits: &[f32],
    task_loss: f32,
    config: &LwfConfig,
) -> ContinualResult<f32> {
    if task_loss.is_nan() {
        return Err(ContinualError::NanEncountered {
            location: "task_loss",
        });
    }
    let distill = lwf_distill_loss(teacher_logits, student_logits, config)?;
    let alpha = config.alpha;
    let combined = alpha * distill + (1.0 - alpha) * task_loss;
    if !combined.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "lwf_combined_loss",
        });
    }
    Ok(combined)
}

/// Batch version: average LwF distillation loss over a batch.
///
/// Both `teacher` and `student` are row-major flat buffers of shape
/// `[n_samples × n_classes]`.
///
/// # Errors
/// Returns an error on empty batch, wrong buffer sizes, or any per-sample
/// error from [`lwf_distill_loss`].
pub fn lwf_distill_loss_batch(
    teacher: &[f32],
    student: &[f32],
    n_samples: usize,
    n_classes: usize,
    config: &LwfConfig,
) -> ContinualResult<f32> {
    config.validate()?;

    if n_samples == 0 {
        return Err(ContinualError::EmptyInput);
    }

    let expected = n_samples * n_classes;
    if teacher.len() != expected {
        return Err(ContinualError::DimensionMismatch {
            expected,
            got: teacher.len(),
        });
    }
    if student.len() != expected {
        return Err(ContinualError::DimensionMismatch {
            expected,
            got: student.len(),
        });
    }

    let mut acc = 0.0_f32;
    for i in 0..n_samples {
        let start = i * n_classes;
        let end = start + n_classes;
        let t_row = &teacher[start..end];
        let s_row = &student[start..end];
        acc += lwf_distill_loss(t_row, s_row, config)?;
    }
    Ok(acc / n_samples as f32)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> LwfConfig {
        LwfConfig::default()
    }

    // 1. Identical logits → distillation loss ≈ 0
    #[test]
    fn distill_loss_identical_logits_is_zero() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let loss = lwf_distill_loss(&logits, &logits, &default_cfg())
            .expect("LwF distillation loss should compute with valid logits");
        assert!(
            loss.abs() < 1e-5,
            "expected ~0 for identical logits, got {loss}"
        );
    }

    // 2. Very different logits → loss > 0
    #[test]
    fn distill_loss_different_logits_is_positive() {
        let teacher = vec![10.0_f32, -10.0, -10.0];
        let student = vec![-10.0_f32, 10.0, -10.0];
        let loss = lwf_distill_loss(&teacher, &student, &default_cfg())
            .expect("LwF distillation loss should compute with valid logits");
        assert!(loss > 0.0, "expected positive KL loss, got {loss}");
    }

    // 3. Greater divergence between teacher/student → higher distillation loss
    #[test]
    fn distill_loss_larger_divergence_gives_higher_loss() {
        // Teacher and student with small logit difference vs large difference.
        let teacher = vec![1.0_f32, 0.0, -1.0];
        let close_student = vec![0.9_f32, 0.05, -0.95]; // very similar to teacher
        let far_student = vec![-1.0_f32, 0.0, 1.0]; // opposite polarity

        let cfg = LwfConfig {
            temperature: 2.0,
            alpha: 0.5,
        };

        let loss_close = lwf_distill_loss(&teacher, &close_student, &cfg)
            .expect("LwF distillation loss should compute with valid logits");
        let loss_far = lwf_distill_loss(&teacher, &far_student, &cfg)
            .expect("LwF distillation loss should compute with valid logits");

        assert!(
            loss_far > loss_close,
            "larger divergence must give higher loss: close={loss_close}, far={loss_far}"
        );
        // Identical teacher and close student → close-to-zero loss
        let loss_identical = lwf_distill_loss(&teacher, &teacher, &cfg)
            .expect("LwF distillation loss should compute with valid logits");
        assert!(
            loss_identical < loss_close,
            "identical logits must give smaller loss than close logits: identical={loss_identical}, close={loss_close}"
        );
    }

    // 4. alpha = 0 → combined == task_loss
    #[test]
    fn combined_loss_alpha_zero_equals_task_loss() {
        let logits = vec![1.0_f32, 0.0, -1.0];
        let task_loss = 0.42_f32;
        let cfg = LwfConfig {
            temperature: 2.0,
            alpha: 0.0,
        };
        let combined = lwf_combined_loss(&logits, &logits, task_loss, &cfg)
            .expect("LwF combined loss should compute with valid inputs");
        assert!(
            (combined - task_loss).abs() < 1e-6,
            "alpha=0 should return task_loss unchanged, got {combined}"
        );
    }

    // 5. alpha = 1 → combined == distillation loss only
    #[test]
    fn combined_loss_alpha_one_equals_distill() {
        let teacher = vec![3.0_f32, 1.0, -1.0];
        let student = vec![2.0_f32, 0.0, 0.0];
        let task_loss = 99.0_f32; // should be ignored
        let cfg = LwfConfig {
            temperature: 2.0,
            alpha: 1.0,
        };
        let combined = lwf_combined_loss(&teacher, &student, task_loss, &cfg)
            .expect("LwF combined loss should compute with valid inputs");
        let distill_only = lwf_distill_loss(&teacher, &student, &cfg)
            .expect("LwF distillation loss should compute with valid logits");
        assert!(
            (combined - distill_only).abs() < 1e-5,
            "alpha=1 should equal distill loss: combined={combined}, distill={distill_only}"
        );
    }

    // 6. Batch length-1 matches single-sample
    #[test]
    fn batch_length_one_matches_single_sample() {
        let teacher = vec![1.0_f32, 2.0, -1.0];
        let student = vec![0.5_f32, 1.5, 0.0];
        let cfg = default_cfg();

        let single = lwf_distill_loss(&teacher, &student, &cfg)
            .expect("LwF distillation loss should compute with valid logits");
        let batch = lwf_distill_loss_batch(&teacher, &student, 1, 3, &cfg)
            .expect("LwF batch distillation loss should compute");

        assert!(
            (single - batch).abs() < 1e-6,
            "batch n=1 should match single sample: single={single}, batch={batch}"
        );
    }

    // 7. Empty teacher logits → EmptyInput
    #[test]
    fn distill_loss_empty_teacher_returns_empty_input() {
        let result = lwf_distill_loss(&[], &[], &default_cfg());
        assert_eq!(result, Err(ContinualError::EmptyInput));
    }

    // 8. Mismatched lengths → DimensionMismatch
    #[test]
    fn distill_loss_length_mismatch_returns_error() {
        let teacher = vec![1.0_f32, 2.0, 3.0];
        let student = vec![1.0_f32, 2.0];
        let result = lwf_distill_loss(&teacher, &student, &default_cfg());
        assert!(
            matches!(
                result,
                Err(ContinualError::DimensionMismatch {
                    expected: 3,
                    got: 2
                })
            ),
            "expected DimensionMismatch(3, 2), got {result:?}"
        );
    }

    // 9. Temperature = 0 → error
    #[test]
    fn distill_loss_zero_temperature_returns_error() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let cfg = LwfConfig {
            temperature: 0.0,
            alpha: 0.5,
        };
        let result = lwf_distill_loss(&logits, &logits, &cfg);
        assert!(
            matches!(result, Err(ContinualError::InvalidLambda { .. })),
            "expected InvalidLambda for T=0, got {result:?}"
        );
    }

    // 10. Batch with 0 samples → EmptyInput
    #[test]
    fn batch_zero_samples_returns_empty_input() {
        let result = lwf_distill_loss_batch(&[], &[], 0, 3, &default_cfg());
        assert_eq!(result, Err(ContinualError::EmptyInput));
    }

    // 11. task_loss NaN → NanEncountered
    #[test]
    fn combined_loss_nan_task_loss_returns_error() {
        let logits = vec![1.0_f32, 0.0, -1.0];
        let result = lwf_combined_loss(&logits, &logits, f32::NAN, &default_cfg());
        assert!(
            matches!(
                result,
                Err(ContinualError::NanEncountered {
                    location: "task_loss"
                })
            ),
            "expected NanEncountered(task_loss), got {result:?}"
        );
    }

    // 12. NaN in teacher logits → NanEncountered
    #[test]
    fn distill_loss_nan_in_teacher_returns_error() {
        let teacher = vec![f32::NAN, 1.0, 2.0];
        let student = vec![0.0_f32, 1.0, 2.0];
        let result = lwf_distill_loss(&teacher, &student, &default_cfg());
        assert!(
            matches!(
                result,
                Err(ContinualError::NanEncountered {
                    location: "teacher_logits"
                })
            ),
            "expected NanEncountered(teacher_logits), got {result:?}"
        );
    }

    // 13. Batch wrong teacher buffer size → DimensionMismatch
    #[test]
    fn batch_wrong_teacher_size_returns_error() {
        // n_samples=2, n_classes=3 → expected 6 elements; give 5
        let teacher = vec![1.0_f32; 5];
        let student = vec![1.0_f32; 6];
        let result = lwf_distill_loss_batch(&teacher, &student, 2, 3, &default_cfg());
        assert!(
            matches!(
                result,
                Err(ContinualError::DimensionMismatch {
                    expected: 6,
                    got: 5
                })
            ),
            "expected DimensionMismatch(6, 5), got {result:?}"
        );
    }

    // 14. negative alpha → Internal error
    #[test]
    fn config_negative_alpha_returns_error() {
        let cfg = LwfConfig {
            temperature: 2.0,
            alpha: -0.1,
        };
        assert!(
            matches!(cfg.validate(), Err(ContinualError::Internal(_))),
            "expected Internal error for alpha < 0"
        );
    }
}

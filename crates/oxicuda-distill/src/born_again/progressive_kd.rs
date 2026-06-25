//! Progressive Knowledge Distillation (Wang et al. 2021).
//!
//! Standard distillation supervises the student with the *final* converged teacher only.
//! When the capacity gap is large, the fully-trained teacher's sharp, high-confidence
//! predictions are hard for a small student to imitate — the optimisation target is too far
//! from the student's own (near-random) starting point. Progressive KD instead builds a
//! **curriculum of intermediate teacher checkpoints** saved along the teacher's training
//! trajectory, and presents them to the student from *easiest to hardest*:
//!
//! 1. Order the checkpoints by training progress (the checkpoint closest to random
//!    initialisation comes first, the final converged teacher last).
//! 2. Allocate the student's training budget across stages.
//! 3. At each stage, distil from the corresponding checkpoint with a standard temperature-
//!    scaled KD objective; advance to the next, harder checkpoint when the stage budget is
//!    exhausted.
//!
//! Because consecutive checkpoints differ only incrementally, each stage poses a small
//! additional gap, smoothing the optimisation landscape the student must traverse. This
//! module provides the scheduling logic, a per-stage curriculum loss, and a soft
//! checkpoint-blending variant that linearly interpolates between adjacent checkpoints to
//! avoid abrupt target shifts at stage boundaries.

use crate::error::{DistillError, DistillResult};
use crate::logit::hinton_kd::{cross_entropy, kl_divergence, softmax_with_temp};

/// A single teacher checkpoint along the training trajectory.
#[derive(Debug, Clone)]
pub struct TeacherCheckpoint {
    /// Training-progress fraction `∈ [0, 1]` (0 = near random init, 1 = converged).
    pub progress: f32,
    /// Logits this checkpoint produces on a reference example (or a flattened batch).
    pub logits: Vec<f32>,
}

impl TeacherCheckpoint {
    /// Construct a checkpoint, validating `progress ∈ [0, 1]` and non-empty logits.
    pub fn new(progress: f32, logits: Vec<f32>) -> DistillResult<Self> {
        if !(0.0..=1.0).contains(&progress) {
            return Err(DistillError::InvalidConfig {
                msg: format!("progress must be in [0, 1], got {progress}"),
            });
        }
        if logits.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        Ok(Self { progress, logits })
    }
}

/// Curriculum schedule over teacher checkpoints.
#[derive(Debug, Clone)]
pub struct ProgressiveKdSchedule {
    /// Checkpoints sorted ascending by `progress` (easiest first).
    pub checkpoints: Vec<TeacherCheckpoint>,
    /// Distillation temperature `T > 0`.
    pub temperature: f32,
    /// Soft-target weight `alpha ∈ [0, 1]`.
    pub alpha: f32,
    /// Total number of training steps spread across stages.
    pub total_steps: usize,
}

impl ProgressiveKdSchedule {
    /// Build a schedule, sorting the provided checkpoints from easiest (lowest progress)
    /// to hardest (highest progress) and validating the hyper-parameters.
    pub fn new(
        mut checkpoints: Vec<TeacherCheckpoint>,
        temperature: f32,
        alpha: f32,
        total_steps: usize,
    ) -> DistillResult<Self> {
        if checkpoints.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(DistillError::InvalidConfig {
                msg: format!("temperature must be finite and > 0, got {temperature}"),
            });
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(DistillError::InvalidConfig {
                msg: format!("alpha must be in [0, 1], got {alpha}"),
            });
        }
        if total_steps == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "total_steps must be non-zero".into(),
            });
        }
        // Stable sort by ascending progress (easiest first).
        checkpoints.sort_by(|a, b| {
            a.progress
                .partial_cmp(&b.progress)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // All checkpoints must agree on logit dimensionality.
        let dim = checkpoints[0].logits.len();
        for c in &checkpoints {
            if c.logits.len() != dim {
                return Err(DistillError::DimensionMismatch {
                    expected: dim,
                    got: c.logits.len(),
                });
            }
        }
        Ok(Self {
            checkpoints,
            temperature,
            alpha,
            total_steps,
        })
    }

    /// Number of curriculum stages (one per checkpoint).
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.checkpoints.len()
    }

    /// Map a global training step to the active stage index.
    ///
    /// The budget is split as evenly as possible; remainder steps are absorbed by the final
    /// (hardest) stage so the student spends the most time on the converged teacher.
    #[must_use]
    pub fn stage_for_step(&self, step: usize) -> usize {
        let stages = self.n_stages();
        if stages <= 1 {
            return 0;
        }
        let per_stage = self.total_steps / stages;
        if per_stage == 0 {
            // More stages than steps: advance one stage per step, clamp at the last.
            return step.min(stages - 1);
        }
        let stage = step / per_stage;
        stage.min(stages - 1)
    }

    /// The teacher logits the student should imitate at `step`, accounting for optional
    /// soft blending across the stage boundary.
    ///
    /// When `blend` is true the target is a linear interpolation between the current and next
    /// checkpoint, ramping from the current checkpoint at the stage start to the next one at
    /// the stage end, so the supervision signal shifts smoothly instead of jumping.
    pub fn target_logits(&self, step: usize, blend: bool) -> DistillResult<Vec<f32>> {
        let stage = self.stage_for_step(step);
        let cur = &self.checkpoints[stage].logits;
        if !blend || stage + 1 >= self.n_stages() {
            return Ok(cur.clone());
        }
        let next = &self.checkpoints[stage + 1].logits;
        let stages = self.n_stages();
        let per_stage = (self.total_steps / stages).max(1);
        let within = (step % per_stage) as f32 / per_stage as f32; // ∈ [0, 1)
        let blended: Vec<f32> = cur
            .iter()
            .zip(next.iter())
            .map(|(&a, &b)| (1.0 - within) * a + within * b)
            .collect();
        Ok(blended)
    }

    /// Curriculum distillation loss at a given training step for one student example.
    ///
    /// `= alpha · T² · KL(p_target || p_student) + (1 − alpha) · CE(student, label)`,
    /// where `p_target` comes from the (optionally blended) active checkpoint.
    pub fn curriculum_loss(
        &self,
        student_logits: &[f32],
        label: usize,
        step: usize,
        blend: bool,
    ) -> DistillResult<f32> {
        if student_logits.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let target = self.target_logits(step, blend)?;
        if student_logits.len() != target.len() {
            return Err(DistillError::DimensionMismatch {
                expected: target.len(),
                got: student_logits.len(),
            });
        }
        if label >= student_logits.len() {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "label {label} out of range for {} classes",
                    student_logits.len()
                ),
            });
        }
        let t = self.temperature;
        let p_t = softmax_with_temp(&target, t);
        let p_s = softmax_with_temp(student_logits, t);
        let soft = kl_divergence(&p_t, &p_s) * t * t;
        let hard = cross_entropy(student_logits, label);
        Ok(self.alpha * soft + (1.0 - self.alpha) * hard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schedule(blend_steps: usize) -> ProgressiveKdSchedule {
        // Three checkpoints supplied out of order; should sort easiest-first.
        let ckpts = vec![
            TeacherCheckpoint::new(1.0, vec![4.0, 1.0, 0.5]).expect("c"),
            TeacherCheckpoint::new(0.0, vec![1.0, 1.0, 1.0]).expect("c"),
            TeacherCheckpoint::new(0.5, vec![2.0, 1.5, 1.0]).expect("c"),
        ];
        ProgressiveKdSchedule::new(ckpts, 4.0, 0.7, blend_steps).expect("sched")
    }

    #[test]
    fn checkpoints_sorted_easiest_first() {
        let s = make_schedule(30);
        assert_eq!(s.checkpoints[0].progress, 0.0);
        assert_eq!(s.checkpoints[1].progress, 0.5);
        assert_eq!(s.checkpoints[2].progress, 1.0);
    }

    #[test]
    fn stage_advances_then_saturates() {
        let s = make_schedule(30); // 3 stages, 10 steps each
        assert_eq!(s.stage_for_step(0), 0);
        assert_eq!(s.stage_for_step(9), 0);
        assert_eq!(s.stage_for_step(10), 1);
        assert_eq!(s.stage_for_step(20), 2);
        // Beyond the budget the final stage absorbs everything.
        assert_eq!(s.stage_for_step(100), 2);
    }

    #[test]
    fn more_stages_than_steps_advances_per_step() {
        let ckpts = vec![
            TeacherCheckpoint::new(0.0, vec![1.0, 0.0]).expect("c"),
            TeacherCheckpoint::new(0.5, vec![0.0, 1.0]).expect("c"),
            TeacherCheckpoint::new(1.0, vec![2.0, 0.0]).expect("c"),
        ];
        let s = ProgressiveKdSchedule::new(ckpts, 2.0, 0.5, 2).expect("sched");
        assert_eq!(s.stage_for_step(0), 0);
        assert_eq!(s.stage_for_step(1), 1);
        assert_eq!(s.stage_for_step(2), 2);
        assert_eq!(s.stage_for_step(99), 2);
    }

    #[test]
    fn unblended_target_is_current_checkpoint() {
        let s = make_schedule(30);
        let target = s.target_logits(5, false).expect("t");
        assert_eq!(target, s.checkpoints[0].logits);
    }

    #[test]
    fn blended_target_interpolates() {
        let s = make_schedule(30); // 10 steps/stage
        // At the very start of stage 0, blend weight ~0 → ~current checkpoint.
        let start = s.target_logits(0, true).expect("t");
        for (a, b) in start.iter().zip(s.checkpoints[0].logits.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        // Near the end of stage 0 (step 9), target should lean toward checkpoint 1.
        let near_end = s.target_logits(9, true).expect("t");
        let cur = &s.checkpoints[0].logits;
        let next = &s.checkpoints[1].logits;
        for i in 0..3 {
            let lo = cur[i].min(next[i]) - 1e-4;
            let hi = cur[i].max(next[i]) + 1e-4;
            assert!(near_end[i] >= lo && near_end[i] <= hi);
        }
    }

    #[test]
    fn blend_clamped_in_last_stage() {
        let s = make_schedule(30);
        // In the final stage there is no next checkpoint → returns current unchanged.
        let target = s.target_logits(25, true).expect("t");
        assert_eq!(target, s.checkpoints[2].logits);
    }

    #[test]
    fn curriculum_loss_finite_across_steps() {
        let s = make_schedule(30);
        let student = vec![0.5_f32, 1.0, 0.8];
        for step in [0usize, 5, 10, 15, 20, 29] {
            let l = s.curriculum_loss(&student, 0, step, true).expect("loss");
            assert!(l.is_finite() && l >= 0.0, "step {step} loss {l}");
        }
    }

    #[test]
    fn curriculum_loss_alpha_zero_is_ce() {
        let ckpts = vec![TeacherCheckpoint::new(1.0, vec![3.0, 1.0, 0.0]).expect("c")];
        let s = ProgressiveKdSchedule::new(ckpts, 2.0, 0.0, 10).expect("sched");
        let student = vec![0.2_f32, 1.1, 0.5];
        let loss = s.curriculum_loss(&student, 1, 0, false).expect("loss");
        let ce = cross_entropy(&student, 1);
        assert!((loss - ce).abs() < 1e-5, "loss {loss} ce {ce}");
    }

    #[test]
    fn rejects_inconsistent_logit_dims() {
        let ckpts = vec![
            TeacherCheckpoint::new(0.0, vec![1.0, 2.0]).expect("c"),
            TeacherCheckpoint::new(1.0, vec![1.0, 2.0, 3.0]).expect("c"),
        ];
        assert!(ProgressiveKdSchedule::new(ckpts, 2.0, 0.5, 10).is_err());
    }

    #[test]
    fn rejects_bad_hyperparams() {
        let ok = vec![TeacherCheckpoint::new(0.0, vec![1.0, 2.0]).expect("c")];
        assert!(ProgressiveKdSchedule::new(ok.clone(), 0.0, 0.5, 10).is_err());
        assert!(ProgressiveKdSchedule::new(ok.clone(), 2.0, 1.5, 10).is_err());
        assert!(ProgressiveKdSchedule::new(ok, 2.0, 0.5, 0).is_err());
        assert!(TeacherCheckpoint::new(1.5, vec![1.0]).is_err());
        assert!(TeacherCheckpoint::new(0.5, vec![]).is_err());
    }
}

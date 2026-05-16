//! Progressive distillation — halve inference steps each generation (Salimans & Ho 2022).

use crate::error::{DistillError, DistillResult};

/// Configuration tracking progression state.
#[derive(Debug, Clone, Copy)]
pub struct ProgressiveConfig {
    /// Number of steps used by the original (teacher) model.
    pub initial_steps: usize,
    /// Number of steps used in the current generation.
    pub current_steps: usize,
    /// Total number of distillation generations planned.
    pub total_generations: usize,
}

impl ProgressiveConfig {
    /// Advance to the next generation by halving current steps (minimum 1).
    #[must_use]
    pub fn next_generation(&self) -> Self {
        let next_steps = (self.current_steps / 2).max(1);
        Self {
            initial_steps: self.initial_steps,
            current_steps: next_steps,
            total_generations: self.total_generations,
        }
    }

    /// Steps at a given generation index: `initial_steps / 2^gen`, minimum 1.
    #[must_use]
    pub fn steps_for_generation(&self, generation: usize) -> usize {
        (self.initial_steps >> generation).max(1)
    }
}

/// Trajectory consistency MSE between student and teacher outputs.
#[must_use]
pub fn consistency_loss(student_out: &[f32], teacher_out: &[f32]) -> f32 {
    if student_out.is_empty() {
        return 0.0;
    }
    let n = student_out.len() as f32;
    student_out
        .iter()
        .zip(teacher_out.iter())
        .map(|(&s, &t)| (s - t).powi(2))
        .sum::<f32>()
        / n
}

/// One progressive distillation step: consistency loss scaled by step fraction.
pub fn progressive_distill_step(
    student_pred: &[f32],
    teacher_pred: &[f32],
    cfg: &ProgressiveConfig,
) -> DistillResult<f32> {
    if student_pred.is_empty() || teacher_pred.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if cfg.initial_steps == 0 {
        return Err(DistillError::InvalidConfig {
            msg: "initial_steps must be > 0".into(),
        });
    }
    let scale = cfg.current_steps as f32 / cfg.initial_steps as f32;
    Ok(consistency_loss(student_pred, teacher_pred) * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_generation_halves() {
        let cfg = ProgressiveConfig {
            initial_steps: 1000,
            current_steps: 500,
            total_generations: 5,
        };
        let next = cfg.next_generation();
        assert_eq!(next.current_steps, 250);
    }

    #[test]
    fn steps_for_generation_correct() {
        let cfg = ProgressiveConfig {
            initial_steps: 1024,
            current_steps: 1024,
            total_generations: 10,
        };
        assert_eq!(cfg.steps_for_generation(0_usize), 1024);
        assert_eq!(cfg.steps_for_generation(1_usize), 512);
        assert_eq!(cfg.steps_for_generation(10_usize), 1);
    }

    #[test]
    fn progressive_distill_step_finite() {
        let cfg = ProgressiveConfig {
            initial_steps: 1000,
            current_steps: 500,
            total_generations: 5,
        };
        let s = vec![1.0_f32, 0.5, 0.0];
        let t = vec![0.9_f32, 0.6, 0.1];
        let loss = progressive_distill_step(&s, &t, &cfg).unwrap();
        assert!(loss.is_finite() && loss >= 0.0);
    }
}

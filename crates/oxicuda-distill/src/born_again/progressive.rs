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

    /// Non-uniform geometric step schedule from `initial_steps` down to `final_steps`.
    ///
    /// Standard progressive distillation halves the step count every generation (ratio = 2),
    /// reaching `initial_steps / 2^G`. A non-uniform schedule instead picks a *per-generation
    /// ratio* `r = (initial / final)^(1 / G)` so the trajectory lands exactly on a chosen
    /// `final_steps` after `total_generations` generations, regardless of whether that ratio
    /// is an integer. The returned vector has length `total_generations + 1`, is monotonically
    /// non-increasing, starts at `initial_steps`, and ends at `final_steps` (clamped to ≥ 1).
    /// This lets the practitioner aim for, say, 1000 → 8 steps over 4 generations
    /// (ratio ≈ 3.16) rather than being constrained to powers of two.
    pub fn non_uniform_schedule(&self, final_steps: usize) -> DistillResult<Vec<usize>> {
        if self.initial_steps == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "initial_steps must be > 0".into(),
            });
        }
        if self.total_generations == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "total_generations must be > 0".into(),
            });
        }
        let target = final_steps.max(1).min(self.initial_steps);
        let g = self.total_generations;
        let ratio = (self.initial_steps as f64 / target as f64).powf(1.0 / g as f64);
        let mut schedule = Vec::with_capacity(g + 1);
        schedule.push(self.initial_steps);
        let mut prev = self.initial_steps;
        for i in 1..=g {
            let raw = (self.initial_steps as f64 / ratio.powi(i as i32)).round() as usize;
            // Enforce monotone non-increasing and clamp to the [target, initial] band.
            let clamped = raw.clamp(target, self.initial_steps).min(prev);
            schedule.push(clamped);
            prev = clamped;
        }
        // Pin the final entry exactly on the requested target.
        if let Some(last) = schedule.last_mut() {
            *last = target;
        }
        Ok(schedule)
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
    fn non_uniform_schedule_endpoints_and_monotone() {
        let cfg = ProgressiveConfig {
            initial_steps: 1000,
            current_steps: 1000,
            total_generations: 4,
        };
        let sched = cfg.non_uniform_schedule(8).expect("schedule");
        assert_eq!(sched.len(), 5);
        assert_eq!(sched[0], 1000);
        assert_eq!(*sched.last().expect("last"), 8);
        // Monotonically non-increasing.
        for w in sched.windows(2) {
            assert!(w[1] <= w[0], "not monotone: {} -> {}", w[0], w[1]);
        }
        // Genuinely non-uniform: the per-step ratio is not the power-of-two halving.
        // 1000 -> 8 over 4 gens implies ratio ~3.16, so the second entry is well below 500.
        assert!(
            sched[1] < 500,
            "expected non-binary ratio, got {}",
            sched[1]
        );
    }

    #[test]
    fn non_uniform_schedule_matches_geometric() {
        let cfg = ProgressiveConfig {
            initial_steps: 64,
            current_steps: 64,
            total_generations: 3,
        };
        // 64 -> 8 over 3 generations has an integer ratio of 2 (64/8 = 8 = 2^3),
        // so this degenerates to the exact halving schedule 64,32,16,8.
        let sched = cfg.non_uniform_schedule(8).expect("schedule");
        assert_eq!(sched, vec![64, 32, 16, 8]);
    }

    #[test]
    fn non_uniform_schedule_rejects_zero_generations() {
        let cfg = ProgressiveConfig {
            initial_steps: 100,
            current_steps: 100,
            total_generations: 0,
        };
        assert!(cfg.non_uniform_schedule(4).is_err());
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
        let loss = progressive_distill_step(&s, &t, &cfg)
            .expect("progressive_distill_step should succeed");
        assert!(loss.is_finite() && loss >= 0.0);
    }
}

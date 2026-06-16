//! TAS — Teacher-Assistant distillation (Mirzadeh et al. 2020) — capacity-bridging via assistant.

use crate::error::DistillResult;
use crate::logit::hinton_kd::{cross_entropy, kl_divergence, softmax_with_temp};

/// Capacity gap between teacher and student models.
#[derive(Debug, Clone, Copy)]
pub struct CapacityGap {
    /// Ratio of teacher parameters to student parameters.
    pub ratio: f32,
}

impl CapacityGap {
    /// Compute the capacity gap ratio.
    #[must_use]
    pub fn compute(teacher_params: usize, student_params: usize) -> Self {
        let ratio = if student_params == 0 {
            f32::INFINITY
        } else {
            teacher_params as f32 / student_params as f32
        };
        Self { ratio }
    }

    /// Heuristic: an intermediate assistant is beneficial when the ratio exceeds 10.
    #[must_use]
    pub fn needs_assistant(&self) -> bool {
        self.ratio > 10.0
    }

    /// Geometric mean of teacher and student parameter counts as the optimal assistant size.
    #[must_use]
    pub fn optimal_assistant_size(&self, teacher_params: usize, student_params: usize) -> usize {
        ((teacher_params as f64 * student_params as f64).sqrt()) as usize
    }
}

/// Configuration holding the three-level size plan.
#[derive(Debug, Clone, Copy)]
pub struct TasConfig {
    /// Number of teacher parameters (for planning purposes).
    pub teacher_size: usize,
    /// Number of assistant parameters.
    pub assistant_size: usize,
    /// Number of student parameters.
    pub student_size: usize,
}

/// TAS loss: student distils from the assistant, not the teacher directly.
///
/// `= T² · KL(softmax(assistant/T) ‖ softmax(student/T)) + CE(student, label)`
pub fn tas_loss(
    student_logits: &[f32],
    assistant_logits: &[f32],
    label: usize,
    temp: f32,
) -> DistillResult<f32> {
    let p_s = softmax_with_temp(student_logits, temp);
    let p_a = softmax_with_temp(assistant_logits, temp);
    let soft = temp * temp * kl_divergence(&p_a, &p_s);
    let hard = cross_entropy(student_logits, label);
    Ok(soft + hard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_gap_needs_assistant() {
        let gap = CapacityGap::compute(1_000_000, 50_000);
        assert!(gap.needs_assistant());
    }

    #[test]
    fn capacity_gap_no_assistant() {
        let gap = CapacityGap::compute(1_000, 500);
        assert!(!gap.needs_assistant());
    }

    #[test]
    fn tas_loss_finite() {
        let s = vec![1.0_f32, 2.0, 3.0];
        let a = vec![1.5_f32, 1.8, 2.7];
        let loss = tas_loss(&s, &a, 2, 4.0).expect("tas_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn optimal_assistant_geometric_mean() {
        let gap = CapacityGap::compute(1_000_000, 10_000);
        let asst = gap.optimal_assistant_size(1_000_000, 10_000);
        // sqrt(1e6 * 1e4) = sqrt(1e10) = 1e5
        assert!((asst as i64 - 100_000).abs() < 5);
    }
}

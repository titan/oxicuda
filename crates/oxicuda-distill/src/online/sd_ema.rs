//! EMA self-distillation — mean-teacher style (Tarvainen & Valpola 2017).

use crate::error::DistillResult;
use crate::online::dml::{kl_divergence, softmax};

/// An exponential moving average teacher that tracks student parameter updates.
#[derive(Debug, Clone)]
pub struct EmaTeacher {
    /// Teacher parameter buffer (EMA of student parameters).
    pub params: Vec<f32>,
    /// EMA momentum ∈ [0, 1].  θ_t ← m·θ_t + (1−m)·θ_s.
    pub momentum: f32,
}

impl EmaTeacher {
    /// Initialise the EMA teacher by copying student `params`.
    #[must_use]
    pub fn new(params: &[f32], momentum: f32) -> Self {
        Self {
            params: params.to_vec(),
            momentum,
        }
    }

    /// Update the EMA teacher: `θ_t ← m·θ_t + (1−m)·θ_s`.
    pub fn update(&mut self, student_params: &[f32]) {
        let m = self.momentum;
        for (tp, &sp) in self.params.iter_mut().zip(student_params.iter()) {
            *tp = m * *tp + (1.0 - m) * sp;
        }
    }

    /// Compute the EMA distillation loss.
    ///
    /// `soft_loss = temp² · KL(softmax(teacher/T) ‖ softmax(student/T))`
    /// `hard_loss = CE(student_logits, label)`
    /// `= alpha · soft_loss + (1 − alpha) · hard_loss`
    pub fn ema_loss(
        teacher_logits: &[f32],
        student_logits: &[f32],
        label: usize,
        alpha: f32,
        temp: f32,
    ) -> DistillResult<f32> {
        use crate::online::dml::cross_entropy_from_probs;
        let t_safe = temp.max(1e-12);
        let p_t = softmax(
            &teacher_logits
                .iter()
                .map(|&x| x / t_safe)
                .collect::<Vec<_>>(),
        );
        let p_s = softmax(
            &student_logits
                .iter()
                .map(|&x| x / t_safe)
                .collect::<Vec<_>>(),
        );
        let soft_loss = temp * temp * kl_divergence(&p_t, &p_s);
        let hard_loss = cross_entropy_from_probs(student_logits, label);
        Ok(alpha * soft_loss + (1.0 - alpha) * hard_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_update_interpolates() {
        let mut ema = EmaTeacher::new(&[0.0_f32, 0.0], 0.9);
        let student = vec![1.0_f32, 1.0];
        ema.update(&student);
        // After one step: θ = 0.9*0 + 0.1*1 = 0.1
        assert!((ema.params[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn ema_loss_finite() {
        let t_logits = vec![1.0_f32, 2.0, 3.0];
        let s_logits = vec![0.9_f32, 2.1, 3.0];
        let loss = EmaTeacher::ema_loss(&t_logits, &s_logits, 2, 0.5, 4.0)
            .expect("ema_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0);
    }
}

//! BAN — Born-Again Networks (Furlanello et al. 2018) — iterative distillation across generations.

use crate::error::DistillResult;
use crate::logit::hinton_kd::{cross_entropy, kl_divergence, softmax_with_temp};

/// A single BAN generation tracking parameter snapshots.
#[derive(Debug, Clone)]
pub struct BanGeneration {
    /// Generation index (0 = original teacher).
    pub generation: usize,
    /// Model parameter snapshot for this generation.
    pub params: Vec<f32>,
}

impl BanGeneration {
    /// Create a new generation record.
    #[must_use]
    pub fn new(generation: usize, params: &[f32]) -> Self {
        Self {
            generation,
            params: params.to_vec(),
        }
    }

    /// BAN distillation loss: `T² · KL(teacher_soft ‖ student_soft) + CE(student, label)`.
    pub fn ban_loss(
        student_logits: &[f32],
        teacher_logits: &[f32],
        label: usize,
        temp: f32,
    ) -> DistillResult<f32> {
        let p_s = softmax_with_temp(student_logits, temp);
        let p_t = softmax_with_temp(teacher_logits, temp);
        let soft = temp * temp * kl_divergence(&p_t, &p_s);
        let hard = cross_entropy(student_logits, label);
        Ok(0.5 * soft + 0.5 * hard)
    }

    /// Ensemble logits by averaging across all generations.
    #[must_use]
    pub fn ensemble_logits(generations: &[Vec<f32>]) -> Vec<f32> {
        if generations.is_empty() {
            return vec![];
        }
        let n = generations.len() as f32;
        let d = generations[0].len();
        let mut out = vec![0.0_f32; d];
        for logits in generations {
            for (o, &v) in out.iter_mut().zip(logits.iter()) {
                *o += v;
            }
        }
        for o in out.iter_mut() {
            *o /= n;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ban_loss_finite() {
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![0.8_f32, 2.2, 3.0];
        let l = BanGeneration::ban_loss(&s, &t, 2, 4.0).unwrap();
        assert!(l.is_finite() && l >= 0.0);
    }

    #[test]
    fn ensemble_mean() {
        let gens = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let ens = BanGeneration::ensemble_logits(&gens);
        assert!((ens[0] - 2.0).abs() < 1e-5);
        assert!((ens[1] - 3.0).abs() < 1e-5);
    }
}

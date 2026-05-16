//! Hinton et al. 2015 — soft-label knowledge distillation via temperature scaling.

use crate::error::{DistillError, DistillResult};

/// Configuration for Hinton-style knowledge distillation.
#[derive(Debug, Clone)]
pub struct HintonKdConfig {
    /// Temperature T (> 1.0 softens distributions, == 1.0 recovers hard CE).
    pub temperature: f32,
    /// Weighting of soft loss ∈ [0, 1]; hard loss weight = 1 − alpha.
    pub alpha: f32,
}

/// Compute numerically-stable softmax after dividing logits by temperature `t`.
///
/// Returns a probability vector of the same length as `logits`.
#[must_use]
pub fn softmax_with_temp(logits: &[f32], t: f32) -> Vec<f32> {
    let t_safe = if t.abs() < 1e-12 { 1e-12 } else { t };
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .map(|&x| ((x / t_safe) - (max_val / t_safe)).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    let sum_safe = if sum < 1e-30 { 1e-30 } else { sum };
    exps.iter().map(|&e| e / sum_safe).collect()
}

/// Kullback-Leibler divergence KL(p || q) = Σ p_i * ln(p_i / (q_i + ε)).
///
/// Terms where `p_i == 0` are skipped (0 * ln(0) = 0 by convention).
#[must_use]
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    const EPS: f32 = 1e-10;
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi <= 0.0 {
                0.0
            } else {
                pi * (pi / (qi + EPS)).ln()
            }
        })
        .sum()
}

/// Cross-entropy loss for the student at the hard label `label`.
///
/// CE = −log(softmax(logits)`[label]`)
#[must_use]
pub fn cross_entropy(logits: &[f32], label: usize) -> f32 {
    const EPS: f32 = 1e-10;
    let p = softmax_with_temp(logits, 1.0);
    let p_label = if label < p.len() { p[label] } else { EPS };
    -(p_label + EPS).ln()
}

/// Compute the combined Hinton KD loss for a single sample.
///
/// `soft = alpha * T² * KL(softmax(s/T) ‖ softmax(t/T))`
/// `hard = (1 − alpha) * CE(student_logits, label)`
/// `return soft + hard`
pub fn kd_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    label: usize,
    cfg: &HintonKdConfig,
) -> DistillResult<f32> {
    if student_logits.is_empty() || teacher_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if student_logits.len() != teacher_logits.len() {
        return Err(DistillError::DimensionMismatch {
            expected: student_logits.len(),
            got: teacher_logits.len(),
        });
    }
    if label >= student_logits.len() {
        return Err(DistillError::InvalidConfig {
            msg: format!(
                "label {} out of range for {} classes",
                label,
                student_logits.len()
            ),
        });
    }
    let t = cfg.temperature;
    let p_s = softmax_with_temp(student_logits, t);
    let p_t = softmax_with_temp(teacher_logits, t);
    let soft = cfg.alpha * t * t * kl_divergence(&p_t, &p_s);
    let hard = (1.0 - cfg.alpha) * cross_entropy(student_logits, label);
    Ok(soft + hard)
}

/// Mean KD loss over a batch of samples.
pub fn kd_loss_batch(
    s_batch: &[Vec<f32>],
    t_batch: &[Vec<f32>],
    labels: &[usize],
    cfg: &HintonKdConfig,
) -> DistillResult<f32> {
    if s_batch.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_batch.len() != t_batch.len() || s_batch.len() != labels.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_batch.len(),
            got: t_batch.len().min(labels.len()),
        });
    }
    let mut total = 0.0_f32;
    for ((s, t), &lbl) in s_batch.iter().zip(t_batch.iter()).zip(labels.iter()) {
        total += kd_loss(s, t, lbl, cfg)?;
    }
    Ok(total / s_batch.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let p = softmax_with_temp(&logits, 1.0);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn kl_identical_is_zero() {
        let p = vec![0.2_f32, 0.5, 0.3];
        assert!(kl_divergence(&p, &p) < 1e-5);
    }

    #[test]
    fn kd_loss_finite() {
        let cfg = HintonKdConfig {
            temperature: 4.0,
            alpha: 0.5,
        };
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![1.1_f32, 2.1, 2.9];
        let loss = kd_loss(&s, &t, 2, &cfg).unwrap();
        assert!(loss.is_finite() && loss >= 0.0);
    }
}

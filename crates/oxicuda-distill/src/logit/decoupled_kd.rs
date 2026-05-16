//! DKD — Decoupled Knowledge Distillation (Zhao et al. 2022).
//!
//! Separates the distillation signal into target-class (TCKD) and non-target-class (NCKD)
//! components, allowing independent weighting of each.

use crate::error::{DistillError, DistillResult};
use crate::logit::hinton_kd::{kl_divergence, softmax_with_temp};

const EPS: f32 = 1e-10;

/// Target-Class Knowledge Distillation loss.
///
/// Measures how well the student matches the teacher's confidence on the ground-truth class.
/// `tckd = −p_t^t · log(p_t^s + ε)` where p^s/t = softmax(logits)`[label]`.
pub fn tckd_loss(s_logits: &[f32], t_logits: &[f32], label: usize, t: f32) -> DistillResult<f32> {
    if s_logits.is_empty() || t_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_logits.len() != t_logits.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_logits.len(),
            got: t_logits.len(),
        });
    }
    if label >= s_logits.len() {
        return Err(DistillError::InvalidConfig {
            msg: format!("label {} >= num_classes {}", label, s_logits.len()),
        });
    }
    let p_s = softmax_with_temp(s_logits, t);
    let p_t = softmax_with_temp(t_logits, t);
    let loss = -p_t[label] * (p_s[label] + EPS).ln();
    Ok(loss)
}

/// Non-Target-Class Knowledge Distillation loss.
///
/// Removes the target class from both distributions, re-normalises with temperature, then
/// computes `T² · KL(t_hat ‖ s_hat)`.
pub fn nckd_loss(s_logits: &[f32], t_logits: &[f32], label: usize, t: f32) -> DistillResult<f32> {
    if s_logits.is_empty() || t_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_logits.len() != t_logits.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_logits.len(),
            got: t_logits.len(),
        });
    }
    if label >= s_logits.len() {
        return Err(DistillError::InvalidConfig {
            msg: format!("label {} >= num_classes {}", label, s_logits.len()),
        });
    }
    // Build masked (non-target) logit vectors.
    let s_masked: Vec<f32> = s_logits
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if i != label { Some(v) } else { None })
        .collect();
    let t_masked: Vec<f32> = t_logits
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if i != label { Some(v) } else { None })
        .collect();
    if s_masked.is_empty() {
        // Only one class — NCKD is zero by definition.
        return Ok(0.0);
    }
    let s_hat = softmax_with_temp(&s_masked, t);
    let t_hat = softmax_with_temp(&t_masked, t);
    Ok(t * t * kl_divergence(&t_hat, &s_hat))
}

/// Combined DKD loss: `alpha · TCKD + beta · NCKD`.
pub fn dkd_loss(
    s_logits: &[f32],
    t_logits: &[f32],
    label: usize,
    alpha: f32,
    beta: f32,
    t: f32,
) -> DistillResult<f32> {
    let tc = tckd_loss(s_logits, t_logits, label, t)?;
    let nc = nckd_loss(s_logits, t_logits, label, t)?;
    Ok(alpha * tc + beta * nc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tckd_nonneg() {
        let s = vec![1.0_f32, 3.0, 2.0];
        let t = vec![0.5_f32, 3.5, 2.0];
        let v = tckd_loss(&s, &t, 1, 4.0).unwrap();
        assert!(v >= 0.0 && v.is_finite());
    }

    #[test]
    fn nckd_nonneg() {
        let s = vec![1.0_f32, 3.0, 2.0];
        let t = vec![0.5_f32, 3.5, 2.0];
        let v = nckd_loss(&s, &t, 1, 4.0).unwrap();
        assert!(v >= 0.0 && v.is_finite());
    }

    #[test]
    fn dkd_loss_finite() {
        let s = vec![1.0_f32, 2.0, 3.0, 4.0];
        let t = vec![1.5_f32, 2.5, 2.5, 3.5];
        let l = dkd_loss(&s, &t, 3, 1.0, 1.0, 4.0).unwrap();
        assert!(l.is_finite());
    }
}

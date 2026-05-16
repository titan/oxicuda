//! Transformer attention weight distillation.

use crate::error::{DistillError, DistillResult};

/// Element-wise MSE between two attention weight maps.
#[must_use]
pub fn attn_loss(s_attn: &[f32], t_attn: &[f32]) -> f32 {
    if s_attn.is_empty() {
        return 0.0;
    }
    let n = s_attn.len() as f32;
    s_attn
        .iter()
        .zip(t_attn.iter())
        .map(|(&s, &t)| (s - t).powi(2))
        .sum::<f32>()
        / n
}

/// Mean attention loss across multiple heads.
pub fn multi_head_attn_loss(s_attns: &[Vec<f32>], t_attns: &[Vec<f32>]) -> DistillResult<f32> {
    if s_attns.is_empty() || t_attns.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_attns.len() != t_attns.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_attns.len(),
            got: t_attns.len(),
        });
    }
    let total: f32 = s_attns
        .iter()
        .zip(t_attns.iter())
        .map(|(s, t)| attn_loss(s, t))
        .sum();
    Ok(total / s_attns.len() as f32)
}

/// Mean attention loss across layers, each layer having multiple heads.
///
/// `s_layers`: `Vec[layer][head][attn_weights_flat]`.
pub fn multi_layer_attn_loss(
    s_layers: &[Vec<Vec<f32>>],
    t_layers: &[Vec<Vec<f32>>],
) -> DistillResult<f32> {
    if s_layers.is_empty() || t_layers.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_layers.len() != t_layers.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_layers.len(),
            got: t_layers.len(),
        });
    }
    let mut total = 0.0_f32;
    for (s_layer, t_layer) in s_layers.iter().zip(t_layers.iter()) {
        total += multi_head_attn_loss(s_layer, t_layer)?;
    }
    Ok(total / s_layers.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attn_loss_identical_is_zero() {
        let v = vec![0.1_f32, 0.5, 0.4];
        assert!(attn_loss(&v, &v) < 1e-10);
    }

    #[test]
    fn multi_head_finite() {
        let s: Vec<Vec<f32>> = vec![vec![0.3_f32, 0.7], vec![0.6, 0.4]];
        let t: Vec<Vec<f32>> = vec![vec![0.25_f32, 0.75], vec![0.55, 0.45]];
        let loss = multi_head_attn_loss(&s, &t).unwrap();
        assert!(loss.is_finite() && loss >= 0.0);
    }
}

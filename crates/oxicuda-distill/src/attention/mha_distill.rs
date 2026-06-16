//! Multi-head attention distillation with optional 1-D Wasserstein transport.

use crate::error::{DistillError, DistillResult};

/// Head-level attention MSE (identical to [`crate::attention::attn_distill::attn_loss`]).
#[must_use]
pub fn head_attn_mse(s_attn: &[f32], t_attn: &[f32]) -> f32 {
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

/// 1-D Wasserstein-1 distance between two equal-length 1-D distributions.
///
/// Both inputs are sorted; the distance equals the mean absolute CDF difference,
/// which for 1-D reduces to mean |s_sorted`[i]` − t_sorted`[i]`|.
#[must_use]
pub fn wasserstein_1d(s_dist: &[f32], t_dist: &[f32]) -> f32 {
    if s_dist.is_empty() || t_dist.is_empty() {
        return 0.0;
    }
    let mut s_sorted = s_dist.to_vec();
    let mut t_sorted = t_dist.to_vec();
    s_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    t_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s_sorted.len().min(t_sorted.len()) as f32;
    s_sorted
        .iter()
        .zip(t_sorted.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>()
        / n
}

/// MHA distillation loss across head pairs.
///
/// When `use_wasserstein` is `true`, uses 1-D Wasserstein distance instead of MSE.
pub fn mha_distill_loss(
    s_attns: &[Vec<f32>],
    t_attns: &[Vec<f32>],
    use_wasserstein: bool,
) -> DistillResult<f32> {
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
        .map(|(s, t)| {
            if use_wasserstein {
                wasserstein_1d(s, t)
            } else {
                head_attn_mse(s, t)
            }
        })
        .sum();
    Ok(total / s_attns.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasserstein_identical_is_zero() {
        let v = vec![0.5_f32, 0.3, 0.2];
        assert!(wasserstein_1d(&v, &v) < 1e-10);
    }

    #[test]
    fn mha_mse_nonneg() {
        let s = vec![vec![0.4_f32, 0.6], vec![0.3, 0.7]];
        let t = vec![vec![0.45_f32, 0.55], vec![0.35, 0.65]];
        let loss = mha_distill_loss(&s, &t, false).expect("mha_distill_loss should succeed");
        assert!(loss >= 0.0 && loss.is_finite());
    }

    #[test]
    fn mha_wasserstein_nonneg() {
        let s = vec![vec![0.4_f32, 0.6], vec![0.3, 0.7]];
        let t = vec![vec![0.45_f32, 0.55], vec![0.35, 0.65]];
        let loss = mha_distill_loss(&s, &t, true).expect("mha_distill_loss should succeed");
        assert!(loss >= 0.0 && loss.is_finite());
    }
}

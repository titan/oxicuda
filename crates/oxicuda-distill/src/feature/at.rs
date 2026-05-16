//! Attention Transfer (Zagoruyko & Komodakis 2017) — spatial activation-map distillation.

use crate::error::{DistillError, DistillResult};

/// Compute the spatial AT map by summing |F`[c,h,w]`|^p over channels.
///
/// `feature_map` is stored as `[channels × height × width]` (channel-major, flat).
/// Output shape: `height × width` (flat).
#[must_use]
pub fn at_map(
    feature_map: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    p: f32,
) -> Vec<f32> {
    let hw = height * width;
    let mut out = vec![0.0_f32; hw];
    for c in 0..channels {
        for (hw_idx, slot) in out.iter_mut().enumerate().take(hw) {
            let feat_idx = c * hw + hw_idx;
            let val = if feat_idx < feature_map.len() {
                feature_map[feat_idx]
            } else {
                0.0
            };
            *slot += val.abs().powf(p);
        }
    }
    out
}

/// L2-normalise a vector (safe against zero-norm vectors via ε = 1e-8).
#[must_use]
pub fn l2_normalize(x: &[f32]) -> Vec<f32> {
    const EPS: f32 = 1e-8;
    let norm: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let norm_safe = norm.max(EPS);
    x.iter().map(|&v| v / norm_safe).collect()
}

/// AT loss: squared L2 distance between normalised student and teacher attention maps.
///
/// `= ‖q_s − q_t‖₂²`
pub fn at_loss(
    s_feat: &[f32],
    t_feat: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    p: f32,
) -> DistillResult<f32> {
    if s_feat.is_empty() || t_feat.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let q_s = l2_normalize(&at_map(s_feat, channels, h, w, p));
    let q_t = l2_normalize(&at_map(t_feat, channels, h, w, p));
    let loss: f32 = q_s
        .iter()
        .zip(q_t.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();
    Ok(loss)
}

/// Mean AT loss over a batch of feature maps.
pub fn at_loss_batch(
    s_batch: &[Vec<f32>],
    t_batch: &[Vec<f32>],
    ch: usize,
    h: usize,
    w: usize,
    p: f32,
) -> DistillResult<f32> {
    if s_batch.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_batch.len() != t_batch.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_batch.len(),
            got: t_batch.len(),
        });
    }
    let mut total = 0.0_f32;
    for (s, t) in s_batch.iter().zip(t_batch.iter()) {
        total += at_loss(s, t, ch, h, w, p)?;
    }
    Ok(total / s_batch.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_map_output_length() {
        let feat: Vec<f32> = (0..24).map(|i| i as f32).collect(); // 2ch × 3h × 4w
        let map = at_map(&feat, 2, 3, 4, 2.0);
        assert_eq!(map.len(), 12);
    }

    #[test]
    fn l2_normalize_unit() {
        let v = vec![3.0_f32, 4.0];
        let n = l2_normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}

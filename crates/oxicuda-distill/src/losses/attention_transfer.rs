//! Attention Transfer (AT) loss — Zagoruyko & Komodakis, 2016.
//!
//! Computes per-layer L2-normalised attention maps from CNN feature maps, then
//! penalises the squared Frobenius distance between the student's and teacher's
//! attention maps across all selected layers.

use crate::error::{DistillError, DistillResult};

/// Compute a spatial attention map from a flat feature-map array.
///
/// `feature_map` must have length `height × width × channels` in
/// row-major, channel-last layout: index `(h, w, c)` maps to
/// `(h * width + w) * channels + c`.
///
/// The attention value at spatial position `(h, w)` is the sum of squared
/// channel activations (p = 2 pooling):
///
/// ```text
/// A[h, w] = Σ_c  feature_map[(h·W + w)·C + c]²
/// ```
///
/// The resulting map is then L2-normalised to unit Frobenius norm.  If the
/// map is all zeros the function returns a zero vector (no division by zero).
///
/// # Errors
/// Returns [`DistillError::DimensionMismatch`] when
/// `feature_map.len() != height * width * channels`.
pub fn attention_map(
    feature_map: &[f32],
    height: usize,
    width: usize,
    channels: usize,
) -> DistillResult<Vec<f32>> {
    let expected = height * width * channels;
    if feature_map.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: feature_map.len(),
        });
    }

    let hw = height * width;
    let mut attn = vec![0.0_f32; hw];

    for (hw_idx, cell) in attn.iter_mut().enumerate() {
        let base = hw_idx * channels;
        let mut acc = 0.0_f32;
        for c in 0..channels {
            let v = feature_map[base + c];
            acc += v * v;
        }
        *cell = acc;
    }

    // L2-normalise: divide every element by ||attn||_2.
    let norm: f32 = attn.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for a in &mut attn {
            *a /= norm;
        }
    }

    Ok(attn)
}

/// Attention Transfer distillation loss summed over multiple layers.
///
/// `student_features` and `teacher_features` are flat arrays where layer `l`
/// occupies the slice `l * stride .. (l+1) * stride` with
/// `stride = height * width * channels`.
///
/// For each layer the function:
/// 1. Extracts the attention map for both student and teacher via [`attention_map`].
/// 2. Accumulates the squared Frobenius norm of the difference:
///    `loss += Σ_i (A_s[i] - A_t[i])²`.
///
/// Both attention maps are already L2-normalised by [`attention_map`].
///
/// # Errors
/// - [`DistillError::DimensionMismatch`] if either feature array has unexpected length.
pub fn attention_transfer_loss(
    student_features: &[f32],
    teacher_features: &[f32],
    n_layers: usize,
    height: usize,
    width: usize,
    channels: usize,
) -> DistillResult<f32> {
    let stride = height * width * channels;
    let expected_len = n_layers * stride;

    if student_features.len() != expected_len {
        return Err(DistillError::DimensionMismatch {
            expected: expected_len,
            got: student_features.len(),
        });
    }
    if teacher_features.len() != expected_len {
        return Err(DistillError::DimensionMismatch {
            expected: expected_len,
            got: teacher_features.len(),
        });
    }

    let mut total_loss = 0.0_f32;

    for l in 0..n_layers {
        let s_slice = &student_features[l * stride..(l + 1) * stride];
        let t_slice = &teacher_features[l * stride..(l + 1) * stride];

        let a_s = attention_map(s_slice, height, width, channels)?;
        let a_t = attention_map(t_slice, height, width, channels)?;

        let layer_loss: f32 = a_s
            .iter()
            .zip(a_t.iter())
            .map(|(&s, &t)| (s - t) * (s - t))
            .sum();
        total_loss += layer_loss;
    }

    Ok(total_loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1 ──────────────────────────────────────────────────────────
    /// Output length of `attention_map` must equal `height * width`.
    #[test]
    fn attention_map_shape() {
        let h = 4_usize;
        let w = 5_usize;
        let c = 3_usize;
        let feat: Vec<f32> = (0..h * w * c).map(|i| i as f32 + 1.0).collect();
        let map = attention_map(&feat, h, w, c).expect("ok");
        assert_eq!(map.len(), h * w);
    }

    // ── Test 2 ──────────────────────────────────────────────────────────
    /// All attention-map values must be non-negative (sum of squares >= 0).
    #[test]
    fn attention_map_nonneg() {
        let feat: Vec<f32> = (0..48).map(|i| (i as f32) - 24.0).collect();
        let map = attention_map(&feat, 4, 4, 3).expect("ok");
        for &v in &map {
            assert!(v >= 0.0, "negative value: {v}");
        }
    }

    // ── Test 3 ──────────────────────────────────────────────────────────
    /// After L2 normalisation the output must have unit Frobenius norm.
    #[test]
    fn attention_map_normalized() {
        let feat: Vec<f32> = (0..36).map(|i| i as f32 + 1.0).collect();
        let map = attention_map(&feat, 3, 4, 3).expect("ok");
        let norm: f32 = map.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    // ── Test 4 ──────────────────────────────────────────────────────────
    /// `attention_transfer_loss` must return a finite value.
    #[test]
    fn loss_finite() {
        let (n_layers, h, w, c) = (2, 3, 3, 4);
        let stride = h * w * c;
        let s: Vec<f32> = (0..n_layers * stride).map(|i| i as f32 * 0.1).collect();
        let t: Vec<f32> = (0..n_layers * stride).map(|i| i as f32 * 0.09).collect();
        let loss = attention_transfer_loss(&s, &t, n_layers, h, w, c).expect("ok");
        assert!(loss.is_finite(), "loss is not finite: {loss}");
    }

    // ── Test 5 ──────────────────────────────────────────────────────────
    /// When student and teacher are identical the loss must be zero.
    #[test]
    fn loss_zero_same_features() {
        let (n_layers, h, w, c) = (3, 4, 4, 8);
        let stride = h * w * c;
        let feats: Vec<f32> = (0..n_layers * stride)
            .map(|i| (i % 7) as f32 + 0.5)
            .collect();
        let loss = attention_transfer_loss(&feats, &feats, n_layers, h, w, c).expect("ok");
        assert!(
            loss < 1e-5,
            "identical features must give ~0 loss, got {loss}"
        );
    }

    // ── Test 6 ──────────────────────────────────────────────────────────
    /// Different student and teacher features must produce a positive loss.
    #[test]
    fn loss_increases_for_different() {
        let (n_layers, h, w, c) = (2, 3, 3, 4);
        let stride = h * w * c;
        let s: Vec<f32> = (0..n_layers * stride).map(|i| i as f32 + 1.0).collect();
        let t: Vec<f32> = (0..n_layers * stride).map(|_| 1.0).collect();
        let loss = attention_transfer_loss(&s, &t, n_layers, h, w, c).expect("ok");
        assert!(
            loss > 0.0,
            "different features must give positive loss, got {loss}"
        );
    }

    // ── Test 7 ──────────────────────────────────────────────────────────
    /// `n_layers = 1` must work without error.
    #[test]
    fn single_layer() {
        let (n_layers, h, w, c) = (1, 2, 2, 4);
        let stride = h * w * c;
        let s: Vec<f32> = (0..stride).map(|i| i as f32 + 0.1).collect();
        let t: Vec<f32> = (0..stride).map(|i| i as f32 + 0.2).collect();
        let loss = attention_transfer_loss(&s, &t, n_layers, h, w, c).expect("ok");
        assert!(loss.is_finite());
    }

    // ── Test 8 ──────────────────────────────────────────────────────────
    /// A feature map whose length does not match `h * w * c` must return an error.
    #[test]
    fn channels_mismatch_error() {
        // feature_map length is wrong (1 element too few).
        let result = attention_map(&[1.0_f32; 11], 2, 2, 3); // expected 12
        assert!(
            result.is_err(),
            "wrong length must produce DimensionMismatch"
        );
    }

    // ── Test 9 ──────────────────────────────────────────────────────────
    /// Wrong student length in `attention_transfer_loss` must return an error.
    #[test]
    fn student_len_mismatch_error() {
        let (n_layers, h, w, c) = (2, 3, 3, 4);
        let stride = h * w * c;
        let s: Vec<f32> = vec![0.0; n_layers * stride - 1]; // too short
        let t: Vec<f32> = vec![0.0; n_layers * stride];
        assert!(attention_transfer_loss(&s, &t, n_layers, h, w, c).is_err());
    }
}

//! Rotation-prediction self-supervised pretext head (S2M2 / RotNet style).
//!
//! Following Gidaris et al. 2018 ("Unsupervised Representation Learning by
//! Predicting Image Rotations") and its use as the auxiliary pretext task in
//! Mangla et al. 2020 ("S2M2: Charting the Right Manifold for Few-Shot
//! Learning"), this module rotates an image by one of the four canonical
//! quarter-turns `{0°, 90°, 180°, 270°}` and trains a small linear head on the
//! backbone embedding to predict *which* rotation was applied.
//!
//! The module owns:
//!
//! * [`rotate_chw`] — an exact `k·90°` clockwise rotation of a row-major
//!   `(C × H × W)` tensor;
//! * [`RotationHead`] — a `k`-way linear softmax classifier over an embedding,
//!   with an analytic single-step trainer;
//! * [`rotation_pretext_loss`] — the end-to-end auxiliary loss that feeds the
//!   four rotated views of a (square) image through a [`Conv4Backbone`] and
//!   averages the cross-entropy of predicting their rotation labels.

use crate::error::{MetaError, MetaResult};
use crate::gradient::inner_loop::cross_entropy_loss;
use crate::handle::LcgRng;
use crate::network::conv4_backbone::Conv4Backbone;

/// The number of rotations used by the canonical pretext task.
pub const NUM_ROTATIONS: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Rotation transform
// ─────────────────────────────────────────────────────────────────────────────

/// Rotate a single `(C × H × W)` channel-major tensor 90° clockwise, returning
/// the `(C × W × H)` result.
fn rot90_cw(img: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let new_h = w;
    let new_w = h;
    let mut out = vec![0.0_f32; channels * new_h * new_w];
    for ch in 0..channels {
        let in_base = ch * h * w;
        let out_base = ch * new_h * new_w;
        for r in 0..new_h {
            for c in 0..new_w {
                // (r, c) in the rotated image reads from (h-1-c, r) in the source.
                out[out_base + r * new_w + c] = img[in_base + (h - 1 - c) * w + r];
            }
        }
    }
    out
}

/// Rotate a row-major `(C × H × W)` tensor by `quarter_turns · 90°` clockwise.
///
/// Returns the rotated buffer together with its new `(height, width)` (which are
/// swapped for odd numbers of quarter-turns).  For square inputs the shape is
/// preserved, so all four rotations can feed the same backbone.
///
/// # Errors
/// `InvalidEpisodeConfig` if any dimension is zero, or `DimensionMismatch` if
/// `img.len() != channels · h · w`.
pub fn rotate_chw(
    img: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    quarter_turns: usize,
) -> MetaResult<(Vec<f32>, usize, usize)> {
    if channels == 0 || h == 0 || w == 0 {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "channels, h and w must all be > 0".into(),
        });
    }
    if img.len() != channels * h * w {
        return Err(MetaError::DimensionMismatch {
            expected: channels * h * w,
            got: img.len(),
        });
    }
    let k = quarter_turns % NUM_ROTATIONS;
    let mut cur = img.to_vec();
    let mut cur_h = h;
    let mut cur_w = w;
    for _ in 0..k {
        cur = rot90_cw(&cur, channels, cur_h, cur_w);
        core::mem::swap(&mut cur_h, &mut cur_w);
    }
    Ok((cur, cur_h, cur_w))
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration + head
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`RotationHead`].
#[derive(Debug, Clone)]
pub struct RotationConfig {
    /// Dimensionality of the backbone embedding fed to the head.
    pub embed_dim: usize,
    /// Number of rotation classes (typically [`NUM_ROTATIONS`]).
    pub n_rotations: usize,
}

/// A `n_rotations`-way linear softmax head predicting the applied rotation from
/// a backbone embedding.
pub struct RotationHead {
    w: Vec<f32>,
    b: Vec<f32>,
    embed_dim: usize,
    n_rot: usize,
}

impl RotationHead {
    /// Construct a Xavier-initialised rotation head (zero biases).
    ///
    /// # Errors
    /// `InvalidFeatDim` if `embed_dim == 0`, `InvalidNWay` if `n_rotations < 2`.
    pub fn new(config: RotationConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        if config.embed_dim == 0 {
            return Err(MetaError::InvalidFeatDim {
                dim: config.embed_dim,
            });
        }
        if config.n_rotations < 2 {
            return Err(MetaError::InvalidNWay {
                n_way: config.n_rotations,
            });
        }
        let limit = (6.0_f32 / (config.embed_dim + config.n_rotations) as f32).sqrt();
        let mut w = vec![0.0_f32; config.n_rotations * config.embed_dim];
        for v in w.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * limit;
        }
        Ok(Self {
            w,
            b: vec![0.0_f32; config.n_rotations],
            embed_dim: config.embed_dim,
            n_rot: config.n_rotations,
        })
    }

    /// Number of rotation classes.
    pub fn n_rotations(&self) -> usize {
        self.n_rot
    }

    /// Expected embedding dimensionality.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Rotation logits for a single embedding.
    ///
    /// # Errors
    /// `DimensionMismatch` if `embed.len() != embed_dim`.
    pub fn forward(&self, embed: &[f32]) -> MetaResult<Vec<f32>> {
        if embed.len() != self.embed_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.embed_dim,
                got: embed.len(),
            });
        }
        let mut logits = self.b.clone();
        for (c, logit) in logits.iter_mut().enumerate() {
            let row = &self.w[c * self.embed_dim..(c + 1) * self.embed_dim];
            *logit += row
                .iter()
                .zip(embed.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>();
        }
        Ok(logits)
    }

    /// Predicted rotation class (argmax of the logits) for an embedding.
    ///
    /// # Errors
    /// Propagates [`Self::forward`] errors.
    pub fn predict(&self, embed: &[f32]) -> MetaResult<usize> {
        let logits = self.forward(embed)?;
        let mut best = 0;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        Ok(best)
    }

    /// Cross-entropy loss of predicting `label` from `embed`.
    ///
    /// # Errors
    /// `InvalidNWay` if `label >= n_rotations`, else propagates `forward`.
    pub fn loss(&self, embed: &[f32], label: usize) -> MetaResult<f32> {
        if label >= self.n_rot {
            return Err(MetaError::InvalidNWay { n_way: self.n_rot });
        }
        let logits = self.forward(embed)?;
        cross_entropy_loss(&logits, &[label as u32], self.n_rot)
    }

    /// Analytic single-sample SGD step on the linear softmax head.
    /// Returns the loss measured *before* the update.
    ///
    /// # Errors
    /// `InvalidLr` for a non-positive `lr`, `InvalidNWay` for a bad label, else
    /// propagates `forward`.
    pub fn train_step(&mut self, embed: &[f32], label: usize, lr: f32) -> MetaResult<f32> {
        if lr <= 0.0 || !lr.is_finite() {
            return Err(MetaError::InvalidLr { lr });
        }
        if label >= self.n_rot {
            return Err(MetaError::InvalidNWay { n_way: self.n_rot });
        }
        let logits = self.forward(embed)?;
        let loss = cross_entropy_loss(&logits, &[label as u32], self.n_rot)?;

        // Softmax probabilities.
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f32> = logits.iter().map(|&z| (z - max).exp()).collect();
        let sum: f32 = p.iter().sum();
        if sum > 0.0 {
            for v in p.iter_mut() {
                *v /= sum;
            }
        }
        // dL/dlogits = softmax − onehot.
        p[label] -= 1.0;

        for (c, &dl) in p.iter().enumerate() {
            self.b[c] -= lr * dl;
            let row = &mut self.w[c * self.embed_dim..(c + 1) * self.embed_dim];
            for (wv, &xi) in row.iter_mut().zip(embed.iter()) {
                *wv -= lr * dl * xi;
            }
        }
        Ok(loss)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// End-to-end pretext loss over a Conv-4 backbone
// ─────────────────────────────────────────────────────────────────────────────

/// Average rotation-prediction cross-entropy over the four rotated views of a
/// single square image, embedded with `backbone` and classified by `head`.
///
/// This is the S2M2 auxiliary self-supervised loss: for each `k ∈ {0,1,2,3}`
/// the image is rotated `k·90°`, embedded, and the head must recover `k`.
///
/// # Errors
/// * `InvalidEpisodeConfig` if the image is not square (90° rotations would
///   change the backbone input shape) or the head is not 4-way;
/// * `DimensionMismatch` if the backbone embedding width differs from the
///   head's `embed_dim`;
/// * any backbone / head forward error.
pub fn rotation_pretext_loss(
    backbone: &Conv4Backbone,
    head: &RotationHead,
    img: &[f32],
    channels: usize,
    h: usize,
    w: usize,
) -> MetaResult<f32> {
    if h != w {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "rotation pretext requires a square image (h == w)".into(),
        });
    }
    if head.n_rotations() != NUM_ROTATIONS {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "rotation pretext head must be 4-way".into(),
        });
    }
    if backbone.output_dim() != head.embed_dim() {
        return Err(MetaError::DimensionMismatch {
            expected: head.embed_dim(),
            got: backbone.output_dim(),
        });
    }

    let mut total = 0.0_f32;
    for k in 0..NUM_ROTATIONS {
        let (rotated, _, _) = rotate_chw(img, channels, h, w, k)?;
        let embed = backbone.forward(&rotated)?;
        let logits = head.forward(&embed)?;
        total += cross_entropy_loss(&logits, &[k as u32], NUM_ROTATIONS)?;
    }
    Ok(total / NUM_ROTATIONS as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::conv4_backbone::Conv4Config;

    // ── rotation transform correctness ──────────────────────────────────────

    #[test]
    fn rotate_zero_is_identity() {
        let img: Vec<f32> = (0..2 * 3 * 3).map(|i| i as f32).collect();
        let (out, oh, ow) = rotate_chw(&img, 2, 3, 3, 0).expect("rotate_chw should succeed");
        assert_eq!((oh, ow), (3, 3));
        assert_eq!(out, img);
    }

    #[test]
    fn rotate_90_known_values() {
        // Single channel 2×3:
        //   1 2 3
        //   4 5 6
        // 90° clockwise → 3×2:
        //   4 1
        //   5 2
        //   6 3
        let img = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, oh, ow) = rotate_chw(&img, 1, 2, 3, 1).expect("rotate_chw should succeed");
        assert_eq!((oh, ow), (3, 2));
        assert_eq!(out, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
    }

    #[test]
    fn rotate_four_times_is_identity() {
        let img: Vec<f32> = (0..3 * 4 * 4).map(|i| (i as f32) * 0.5).collect();
        let mut cur = img.clone();
        let (mut h, mut w) = (4_usize, 4_usize);
        for _ in 0..4 {
            let (o, oh, ow) = rotate_chw(&cur, 3, h, w, 1).expect("rotate_chw should succeed");
            cur = o;
            h = oh;
            w = ow;
        }
        assert_eq!(cur, img);
    }

    #[test]
    fn rotate_180_is_double_flip() {
        let img: Vec<f32> = (0..4 * 4).map(|i| i as f32).collect();
        let (out, _, _) = rotate_chw(&img, 1, 4, 4, 2).expect("rotate_chw should succeed");
        // 180° maps (i, j) → (h-1-i, w-1-j); for a 4×4 that is a full reversal.
        let mut expected = img.clone();
        expected.reverse();
        assert_eq!(out, expected);
    }

    #[test]
    fn rotate_90_then_270_is_identity() {
        let img: Vec<f32> = (0..2 * 5 * 5).map(|i| i as f32 * 0.25).collect();
        let (r1, h1, w1) = rotate_chw(&img, 2, 5, 5, 1).expect("rotate_chw should succeed");
        let (r2, _, _) = rotate_chw(&r1, 2, h1, w1, 3).expect("rotate_chw should succeed");
        assert_eq!(r2, img);
    }

    #[test]
    fn rotate_wrong_length_errors() {
        assert!(matches!(
            rotate_chw(&[0.0; 5], 1, 3, 3, 1),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── head behaviour ──────────────────────────────────────────────────────

    fn head(embed_dim: usize) -> RotationHead {
        let mut rng = LcgRng::new(2026);
        RotationHead::new(
            RotationConfig {
                embed_dim,
                n_rotations: NUM_ROTATIONS,
            },
            &mut rng,
        )
        .expect("value should be present")
    }

    #[test]
    fn head_rejects_bad_config() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            RotationHead::new(
                RotationConfig {
                    embed_dim: 0,
                    n_rotations: 4
                },
                &mut rng
            ),
            Err(MetaError::InvalidFeatDim { .. })
        ));
        assert!(matches!(
            RotationHead::new(
                RotationConfig {
                    embed_dim: 8,
                    n_rotations: 1
                },
                &mut rng
            ),
            Err(MetaError::InvalidNWay { .. })
        ));
    }

    #[test]
    fn head_forward_four_logits() {
        let h = head(6);
        let logits = h
            .forward(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("forward should succeed");
        assert_eq!(logits.len(), NUM_ROTATIONS);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn head_forward_wrong_dim_errors() {
        let h = head(6);
        assert!(matches!(
            h.forward(&[0.0; 3]),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn head_loss_finite_and_nonneg() {
        let h = head(5);
        let embed = vec![0.3_f32; 5];
        for k in 0..NUM_ROTATIONS {
            let l = h.loss(&embed, k).expect("loss should succeed");
            assert!(l.is_finite() && l >= 0.0);
        }
        assert!(matches!(
            h.loss(&embed, NUM_ROTATIONS),
            Err(MetaError::InvalidNWay { .. })
        ));
    }

    #[test]
    fn head_train_step_reduces_loss() {
        let mut h = head(6);
        let embed = vec![0.2_f32, -0.4, 0.6, 0.1, -0.2, 0.3];
        let label = 2;
        let before = h.loss(&embed, label).expect("loss should succeed");
        for _ in 0..25 {
            h.train_step(&embed, label, 0.2)
                .expect("train_step should succeed");
        }
        let after = h.loss(&embed, label).expect("loss should succeed");
        assert!(
            after < before,
            "training must reduce loss: {after} !< {before}"
        );
        assert_eq!(h.predict(&embed).expect("predict should succeed"), label);
    }

    #[test]
    fn head_deterministic_under_seed() {
        let a = head(7);
        let b = head(7);
        let embed = vec![0.5_f32; 7];
        assert_eq!(
            a.forward(&embed).expect("forward should succeed"),
            b.forward(&embed).expect("forward should succeed")
        );
    }

    // ── end-to-end pretext loss ─────────────────────────────────────────────

    fn make_backbone() -> Conv4Backbone {
        let mut rng = LcgRng::new(99);
        Conv4Backbone::new(
            Conv4Config {
                in_channels: 1,
                width: 4,
                input_h: 16,
                input_w: 16,
            },
            &mut rng,
        )
        .expect("value should be present")
    }

    #[test]
    fn pretext_loss_finite() {
        let backbone = make_backbone();
        let h = head(backbone.output_dim());
        let img: Vec<f32> = (0..16 * 16).map(|i| (i as f32 * 0.01).sin()).collect();
        let loss = rotation_pretext_loss(&backbone, &h, &img, 1, 16, 16)
            .expect("rotation_pretext_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn pretext_loss_requires_square() {
        let backbone = make_backbone();
        let h = head(backbone.output_dim());
        let img = vec![0.0_f32; 16 * 16];
        // 16×16 backbone fed a non-square shape request.
        assert!(matches!(
            rotation_pretext_loss(&backbone, &h, &img, 1, 16, 32),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn pretext_loss_dim_mismatch_errors() {
        let backbone = make_backbone();
        let wrong = head(backbone.output_dim() + 1);
        let img = vec![0.0_f32; 16 * 16];
        assert!(matches!(
            rotation_pretext_loss(&backbone, &wrong, &img, 1, 16, 16),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }
}

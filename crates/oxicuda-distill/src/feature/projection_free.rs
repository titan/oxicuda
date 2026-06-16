//! Projection-Free Feature Distillation (RA-DKD style).
//!
//! Traditional feature distillation (FitNets, AT, PKT) often requires a learnable projection
//! layer to bridge teacher–student channel-count mismatches. This module avoids those extra
//! parameters by:
//!   1. Global average pooling to collapse spatial dimensions → per-channel scalars.
//!   2. L2 (or L1) normalisation to make features from different architectures comparable.
//!   3. Decoupled foreground / background channel alignment with independent loss weights.
//!
//! Particularly valuable for edge / mobile deployment where projection parameters are costly.

use crate::error::{DistillError, DistillResult};

/// Normalisation mode applied to the globally-pooled channel vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjFreeNorm {
    /// L2-normalise: v / (‖v‖₂ + ε).
    L2,
    /// L1-normalise: v / (‖v‖₁ + ε).
    L1,
    /// No normalisation — use raw pooled values.
    None,
}

/// Choice of element-wise loss applied after pooling and normalisation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjFreeLossType {
    /// Mean squared error: (1/C) Σ (t_i − s_i)².
    Mse,
    /// Cosine distance: 1 − (t · s) / (‖t‖ · ‖s‖ + ε), range [0, 2].
    CosineDist,
    /// Smooth-L1 / Huber (δ = 1.0): mean over f(t_i − s_i).
    SmoothL1,
}

/// Configuration for projection-free feature distillation.
pub struct ProjFreeConfig {
    /// Normalisation applied to pooled channel vectors.
    pub norm: ProjFreeNorm,
    /// Loss type applied between normalised teacher and student vectors.
    pub loss_type: ProjFreeLossType,
    /// Weight for foreground (high-activation) channels.
    pub alpha_fg: f32,
    /// Weight for background (low-activation) channels.
    pub alpha_bg: f32,
    /// Fraction of channels treated as foreground (top-k fraction), ∈ (0, 1].
    pub fg_threshold: f32,
    /// Softmax temperature for channel-importance weighting (1.0 = uniform).
    pub temperature: f32,
}

impl Default for ProjFreeConfig {
    fn default() -> Self {
        Self {
            norm: ProjFreeNorm::L2,
            loss_type: ProjFreeLossType::Mse,
            alpha_fg: 1.0,
            alpha_bg: 0.5,
            fg_threshold: 0.5,
            temperature: 1.0,
        }
    }
}

/// Stateless helper containing all projection-free distillation operations.
pub struct ProjFreeDistiller;

impl ProjFreeDistiller {
    /// Global average pool a [C × S] flat feature map to a length-C vector.
    ///
    /// `feature` must have exactly `channels * spatial_size` elements.
    /// If `spatial_size == 1` the input is returned unchanged (no copy needed; a clone
    /// is still returned for a consistent owned return type).
    pub fn global_avg_pool(
        feature: &[f32],
        channels: usize,
        spatial_size: usize,
    ) -> DistillResult<Vec<f32>> {
        if feature.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let expected =
            channels
                .checked_mul(spatial_size)
                .ok_or_else(|| DistillError::InvalidConfig {
                    msg: "channels * spatial_size overflows usize".to_owned(),
                })?;
        if feature.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: feature.len(),
            });
        }
        if spatial_size == 1 {
            return Ok(feature.to_vec());
        }
        let inv_s = 1.0_f32 / spatial_size as f32;
        let mut out = vec![0.0_f32; channels];
        for (c, slot) in out.iter_mut().enumerate() {
            let base = c * spatial_size;
            let sum: f32 = feature[base..base + spatial_size].iter().sum();
            *slot = sum * inv_s;
        }
        Ok(out)
    }

    /// L2-normalise a vector: v / (‖v‖₂ + ε).
    ///
    /// Returns the zero vector if ‖v‖₂ < ε.
    #[must_use]
    pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
        const EPS: f32 = 1e-12;
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm < EPS {
            return vec![0.0_f32; v.len()];
        }
        v.iter().map(|&x| x / (norm + EPS)).collect()
    }

    /// L1-normalise a vector: v / (‖v‖₁ + ε).
    ///
    /// Returns the zero vector if ‖v‖₁ < ε.
    #[must_use]
    pub fn l1_normalize(v: &[f32]) -> Vec<f32> {
        const EPS: f32 = 1e-12;
        let norm: f32 = v.iter().map(|&x| x.abs()).sum();
        if norm < EPS {
            return vec![0.0_f32; v.len()];
        }
        v.iter().map(|&x| x / (norm + EPS)).collect()
    }

    /// Apply the requested normalisation mode to a pooled channel vector.
    #[must_use]
    pub fn normalize(v: &[f32], norm: ProjFreeNorm) -> Vec<f32> {
        match norm {
            ProjFreeNorm::L2 => Self::l2_normalize(v),
            ProjFreeNorm::L1 => Self::l1_normalize(v),
            ProjFreeNorm::None => v.to_vec(),
        }
    }

    /// Mean squared error: (1/n) Σ (t_i − s_i)².
    pub fn mse_loss(teacher: &[f32], student: &[f32]) -> DistillResult<f32> {
        if teacher.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if teacher.len() != student.len() {
            return Err(DistillError::DimensionMismatch {
                expected: teacher.len(),
                got: student.len(),
            });
        }
        let mse: f32 = teacher
            .iter()
            .zip(student.iter())
            .map(|(&t, &s)| {
                let d = t - s;
                d * d
            })
            .sum::<f32>()
            / teacher.len() as f32;
        Self::check_finite(mse, "mse_loss")
    }

    /// Cosine distance: 1 − (t · s) / (‖t‖₂ · ‖s‖₂ + ε), range [0, 2].
    pub fn cosine_dist(teacher: &[f32], student: &[f32]) -> DistillResult<f32> {
        if teacher.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if teacher.len() != student.len() {
            return Err(DistillError::DimensionMismatch {
                expected: teacher.len(),
                got: student.len(),
            });
        }
        const EPS: f32 = 1e-12;
        let dot: f32 = teacher
            .iter()
            .zip(student.iter())
            .map(|(&t, &s)| t * s)
            .sum();
        let nt: f32 = teacher.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let ns: f32 = student.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (nt * ns + EPS);
        // Clamp to [-1, 1] for numerical safety before subtracting from 1.
        let cos_sim_clamped = cos_sim.clamp(-1.0_f32, 1.0_f32);
        Self::check_finite(1.0 - cos_sim_clamped, "cosine_dist")
    }

    /// Smooth-L1 (Huber, δ = 1.0) loss: mean over f(t_i − s_i).
    ///
    /// f(x) = 0.5 x² if |x| < δ, else |x| − 0.5.
    pub fn smooth_l1_loss(teacher: &[f32], student: &[f32]) -> DistillResult<f32> {
        if teacher.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if teacher.len() != student.len() {
            return Err(DistillError::DimensionMismatch {
                expected: teacher.len(),
                got: student.len(),
            });
        }
        const DELTA: f32 = 1.0;
        let sum: f32 = teacher
            .iter()
            .zip(student.iter())
            .map(|(&t, &s)| {
                let x = (t - s).abs();
                if x < DELTA { 0.5 * x * x } else { x - 0.5 }
            })
            .sum();
        Self::check_finite(sum / teacher.len() as f32, "smooth_l1_loss")
    }

    /// Compute per-channel importance weights from the teacher's pooled feature.
    ///
    /// Algorithm:
    ///   1. Take absolute values of pooled teacher channels.
    ///   2. Softmax over those values divided by `temperature`.
    ///   3. Scale by C so that the weights sum to C (uniform → each weight = 1).
    ///
    /// Returns a `Vec<f32>` of length `pooled_teacher.len()`.
    #[must_use]
    pub fn channel_weights(pooled_teacher: &[f32], temperature: f32) -> Vec<f32> {
        let c = pooled_teacher.len();
        if c == 0 {
            return Vec::new();
        }
        let t_safe = if temperature.abs() < 1e-12 {
            1e-12_f32
        } else {
            temperature
        };
        let abs_vals: Vec<f32> = pooled_teacher.iter().map(|&v| v.abs() / t_safe).collect();
        let max_val = abs_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = abs_vals.iter().map(|&a| (a - max_val).exp()).collect();
        let sum_exp: f32 = exps.iter().sum::<f32>().max(1e-30_f32);
        // Scale by C so uniform weights equal 1.0 each.
        exps.iter().map(|&e| e / sum_exp * c as f32).collect()
    }

    /// Split channel indices into foreground (top-k by absolute magnitude) and background.
    ///
    /// `fg_fraction` ∈ (0, 1]: fraction of channels considered foreground.
    /// `k = max(1, ceil(fg_fraction * C))`.
    ///
    /// Returns `(fg_indices, bg_indices)` with fg sorted by decreasing `|pooled_teacher[i]|`.
    #[must_use]
    pub fn fg_bg_split(pooled_teacher: &[f32], fg_fraction: f32) -> (Vec<usize>, Vec<usize>) {
        let c = pooled_teacher.len();
        if c == 0 {
            return (Vec::new(), Vec::new());
        }
        // k >= 1, bounded by c.
        let raw_k = (fg_fraction * c as f32).ceil() as usize;
        let k = raw_k.clamp(1, c);

        // Sort indices by absolute value descending.
        let mut order: Vec<usize> = (0..c).collect();
        order.sort_by(|&a, &b| {
            pooled_teacher[b]
                .abs()
                .partial_cmp(&pooled_teacher[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let fg_indices = order[..k].to_vec();
        let bg_indices = order[k..].to_vec();
        (fg_indices, bg_indices)
    }

    /// Main projection-free distillation loss.
    ///
    /// Steps:
    ///   1. Pool each feature map to per-channel scalars.
    ///   2. Align channels: `min_c = min(C_t, C_s)` (truncate the longer vector).
    ///   3. Normalise both aligned vectors.
    ///   4. Decoupled split: compute foreground and background losses separately.
    ///   5. Return `α_fg · loss_fg + α_bg · loss_bg`.
    ///
    /// If `bg_indices` is empty, the background term is omitted.
    pub fn loss(
        teacher_feat: &[f32],
        teacher_channels: usize,
        teacher_spatial: usize,
        student_feat: &[f32],
        student_channels: usize,
        student_spatial: usize,
        cfg: &ProjFreeConfig,
    ) -> DistillResult<f32> {
        if teacher_feat.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if student_feat.is_empty() {
            return Err(DistillError::EmptyInput);
        }

        // --- Step 1: Global average pool ---
        let t_pooled = Self::global_avg_pool(teacher_feat, teacher_channels, teacher_spatial)?;
        let s_pooled = Self::global_avg_pool(student_feat, student_channels, student_spatial)?;

        // --- Step 2: Align channels ---
        let min_c = t_pooled.len().min(s_pooled.len());
        if min_c == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "channel count is zero after alignment".to_owned(),
            });
        }
        let t_aligned = &t_pooled[..min_c];
        let s_aligned = &s_pooled[..min_c];

        // --- Step 3: Normalise ---
        let t_norm = Self::normalize(t_aligned, cfg.norm);
        let s_norm = Self::normalize(s_aligned, cfg.norm);

        // --- Step 4: FG / BG split based on normalised teacher ---
        let (fg_idx, bg_idx) = Self::fg_bg_split(&t_norm, cfg.fg_threshold);

        // Helper: extract elements at the given indices.
        let extract =
            |v: &[f32], idx: &[usize]| -> Vec<f32> { idx.iter().map(|&i| v[i]).collect() };

        let t_fg = extract(&t_norm, &fg_idx);
        let s_fg = extract(&s_norm, &fg_idx);

        // --- Step 5: Loss dispatcher ---
        let loss_fn = |t: &[f32], s: &[f32]| -> DistillResult<f32> {
            match cfg.loss_type {
                ProjFreeLossType::Mse => Self::mse_loss(t, s),
                ProjFreeLossType::CosineDist => Self::cosine_dist(t, s),
                ProjFreeLossType::SmoothL1 => Self::smooth_l1_loss(t, s),
            }
        };

        let fg_loss = loss_fn(&t_fg, &s_fg)?;
        let mut total = cfg.alpha_fg * fg_loss;

        if !bg_idx.is_empty() {
            let t_bg = extract(&t_norm, &bg_idx);
            let s_bg = extract(&s_norm, &bg_idx);
            let bg_loss = loss_fn(&t_bg, &s_bg)?;
            total += cfg.alpha_bg * bg_loss;
        }

        Self::check_finite(total, "proj_free::loss")
    }

    // ── internal helper ──────────────────────────────────────────────────────

    fn check_finite(v: f32, ctx: &str) -> DistillResult<f32> {
        if v.is_finite() {
            Ok(v)
        } else {
            Err(DistillError::NumericalError {
                msg: format!("{ctx}: non-finite value {v}"),
            })
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── global_avg_pool ───────────────────────────────────────────────────────

    #[test]
    fn gap_spatial_one_returns_unchanged() {
        let feat = vec![1.0_f32, 2.0, 3.0];
        let result = ProjFreeDistiller::global_avg_pool(&feat, 3, 1)
            .expect("global_avg_pool should succeed");
        assert_eq!(result, feat);
    }

    #[test]
    fn gap_spatial_four_computes_mean() {
        // 1 channel, 4 spatial positions [1,2,3,4] → mean = 2.5
        let feat = vec![1.0_f32, 2.0, 3.0, 4.0];
        let result = ProjFreeDistiller::global_avg_pool(&feat, 1, 4)
            .expect("global_avg_pool should succeed");
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.5_f32).abs() < 1e-6);
    }

    #[test]
    fn gap_two_channels_two_spatial() {
        // channel 0: [1, 3] → 2.0; channel 1: [2, 4] → 3.0
        let feat = vec![1.0_f32, 3.0, 2.0, 4.0]; // row-major: ch0 row then ch1 row
        let result = ProjFreeDistiller::global_avg_pool(&feat, 2, 2)
            .expect("global_avg_pool should succeed");
        assert!((result[0] - 2.0_f32).abs() < 1e-6);
        assert!((result[1] - 3.0_f32).abs() < 1e-6);
    }

    #[test]
    fn gap_empty_returns_err() {
        let result = ProjFreeDistiller::global_avg_pool(&[], 0, 1);
        assert!(matches!(result, Err(DistillError::EmptyInput)));
    }

    #[test]
    fn gap_dimension_mismatch_returns_err() {
        let feat = vec![1.0_f32, 2.0, 3.0];
        // 2 channels × 3 spatial = 6, but feat has 3 → mismatch
        let result = ProjFreeDistiller::global_avg_pool(&feat, 2, 3);
        assert!(matches!(
            result,
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    // ── l2_normalize ──────────────────────────────────────────────────────────

    #[test]
    fn l2_normalize_known_vector() {
        let v = vec![3.0_f32, 4.0];
        let n = ProjFreeDistiller::l2_normalize(&v);
        assert!((n[0] - 0.6_f32).abs() < 1e-5);
        assert!((n[1] - 0.8_f32).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let v = vec![0.0_f32, 0.0, 0.0];
        let n = ProjFreeDistiller::l2_normalize(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    // ── l1_normalize ──────────────────────────────────────────────────────────

    #[test]
    fn l1_normalize_known_vector() {
        let v = vec![1.0_f32, 2.0, 3.0]; // sum = 6
        let n = ProjFreeDistiller::l1_normalize(&v);
        assert!((n[0] - 1.0 / 6.0_f32).abs() < 1e-5);
        assert!((n[1] - 2.0 / 6.0_f32).abs() < 1e-5);
        assert!((n[2] - 3.0 / 6.0_f32).abs() < 1e-5);
    }

    // ── mse_loss ──────────────────────────────────────────────────────────────

    #[test]
    fn mse_identical_is_zero() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let loss = ProjFreeDistiller::mse_loss(&v, &v).expect("mse_loss should succeed");
        assert!(loss.abs() < 1e-7);
    }

    #[test]
    fn mse_dimension_mismatch_returns_err() {
        let t = vec![1.0_f32, 2.0];
        let s = vec![1.0_f32];
        let result = ProjFreeDistiller::mse_loss(&t, &s);
        assert!(matches!(
            result,
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    // ── cosine_dist ───────────────────────────────────────────────────────────

    #[test]
    fn cosine_dist_identical_is_zero() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let dist = ProjFreeDistiller::cosine_dist(&v, &v).expect("cosine_dist should succeed");
        assert!(dist.abs() < 1e-5);
    }

    #[test]
    fn cosine_dist_orthogonal_is_one() {
        let t = vec![1.0_f32, 0.0];
        let s = vec![0.0_f32, 1.0];
        let dist = ProjFreeDistiller::cosine_dist(&t, &s).expect("cosine_dist should succeed");
        assert!((dist - 1.0_f32).abs() < 1e-5);
    }

    // ── smooth_l1_loss ────────────────────────────────────────────────────────

    #[test]
    fn smooth_l1_identical_is_zero() {
        let v = vec![5.0_f32, -3.0, 0.0];
        let loss =
            ProjFreeDistiller::smooth_l1_loss(&v, &v).expect("smooth_l1_loss should succeed");
        assert!(loss.abs() < 1e-7);
    }

    #[test]
    fn smooth_l1_large_diff_uses_linear_branch() {
        // |2 - 0| = 2 >= delta=1 → 2 - 0.5 = 1.5
        let t = vec![2.0_f32];
        let s = vec![0.0_f32];
        let loss =
            ProjFreeDistiller::smooth_l1_loss(&t, &s).expect("smooth_l1_loss should succeed");
        assert!((loss - 1.5_f32).abs() < 1e-5);
    }

    #[test]
    fn smooth_l1_small_diff_uses_quadratic_branch() {
        // |0.5| < 1 → 0.5 * 0.25 = 0.125
        let t = vec![0.5_f32];
        let s = vec![0.0_f32];
        let loss =
            ProjFreeDistiller::smooth_l1_loss(&t, &s).expect("smooth_l1_loss should succeed");
        assert!((loss - 0.125_f32).abs() < 1e-5);
    }

    // ── channel_weights ───────────────────────────────────────────────────────

    #[test]
    fn channel_weights_sum_to_c() {
        let pooled = vec![0.5_f32, -1.0, 2.0, -0.3];
        let w = ProjFreeDistiller::channel_weights(&pooled, 1.0);
        let sum: f32 = w.iter().sum();
        assert!((sum - pooled.len() as f32).abs() < 1e-4);
    }

    #[test]
    fn channel_weights_uniform_for_equal_abs() {
        let pooled = vec![1.0_f32, 1.0, 1.0]; // all same abs → uniform
        let w = ProjFreeDistiller::channel_weights(&pooled, 1.0);
        for &wi in &w {
            assert!((wi - 1.0_f32).abs() < 1e-5);
        }
    }

    // ── fg_bg_split ───────────────────────────────────────────────────────────

    #[test]
    fn fg_bg_split_fraction_one_all_fg() {
        let pooled = vec![0.1_f32, 0.5, -0.3, 0.8];
        let (fg, bg) = ProjFreeDistiller::fg_bg_split(&pooled, 1.0);
        assert_eq!(fg.len(), 4);
        assert!(bg.is_empty());
    }

    #[test]
    fn fg_bg_split_half() {
        let pooled = vec![1.0_f32, -2.0, 0.5, -0.1];
        let (fg, bg) = ProjFreeDistiller::fg_bg_split(&pooled, 0.5);
        assert_eq!(fg.len(), 2);
        assert_eq!(bg.len(), 2);
        // Largest absolute values: index 1 (|-2.0|) and index 0 (|1.0|)
        assert!(fg.contains(&1));
        assert!(fg.contains(&0));
    }

    // ── loss ──────────────────────────────────────────────────────────────────

    #[test]
    fn loss_identical_teacher_student_near_zero() {
        let feat: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect(); // 2ch × 4 spatial
        let cfg = ProjFreeConfig::default();
        let l =
            ProjFreeDistiller::loss(&feat, 2, 4, &feat, 2, 4, &cfg).expect("loss should succeed");
        assert!(
            l.abs() < 1e-5,
            "loss should be ~0 for identical inputs, got {l}"
        );
    }

    #[test]
    fn loss_handles_channel_mismatch() {
        // Teacher: 4ch × 2 spatial, Student: 2ch × 2 spatial → aligned to 2 ch
        let t_feat: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let s_feat: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let cfg = ProjFreeConfig::default();
        let result = ProjFreeDistiller::loss(&t_feat, 4, 2, &s_feat, 2, 2, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn loss_empty_teacher_returns_err() {
        let cfg = ProjFreeConfig::default();
        let result = ProjFreeDistiller::loss(&[], 0, 1, &[1.0], 1, 1, &cfg);
        assert!(matches!(result, Err(DistillError::EmptyInput)));
    }

    #[test]
    fn loss_cosine_dist_type() {
        let feat: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = ProjFreeConfig {
            loss_type: ProjFreeLossType::CosineDist,
            ..Default::default()
        };
        let l =
            ProjFreeDistiller::loss(&feat, 2, 2, &feat, 2, 2, &cfg).expect("loss should succeed");
        assert!(l.abs() < 1e-5);
    }

    #[test]
    fn loss_smooth_l1_type() {
        let feat: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = ProjFreeConfig {
            loss_type: ProjFreeLossType::SmoothL1,
            ..Default::default()
        };
        let l =
            ProjFreeDistiller::loss(&feat, 2, 2, &feat, 2, 2, &cfg).expect("loss should succeed");
        assert!(l.abs() < 1e-5);
    }

    #[test]
    fn default_config_has_expected_values() {
        let cfg = ProjFreeConfig::default();
        assert_eq!(cfg.norm, ProjFreeNorm::L2);
        assert_eq!(cfg.loss_type, ProjFreeLossType::Mse);
        assert!((cfg.alpha_fg - 1.0_f32).abs() < 1e-7);
        assert!((cfg.alpha_bg - 0.5_f32).abs() < 1e-7);
        assert!((cfg.fg_threshold - 0.5_f32).abs() < 1e-7);
        assert!((cfg.temperature - 1.0_f32).abs() < 1e-7);
    }
}

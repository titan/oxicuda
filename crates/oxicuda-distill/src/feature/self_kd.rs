//! Self-Knowledge Distillation via MixUp and CutMix consistency.
//!
//! Self-KD leverages the student's own predictions on augmented or mixed inputs as
//! soft targets — no teacher network is required.  The library operates on logits
//! supplied by the caller (who performs the actual pixel-level mixing).
//!
//! Supported mixing strategies:
//! - **MixUp** (Zhang et al. 2018): linear interpolation of two samples
//! - **CutMix** (Yun et al. 2019): rectangular patch replacement (same loss formula)
//! - **Feature consistency**: MSE between augmented and original feature maps
//!
//! Reference: Yuan et al. 2020 "Revisiting Knowledge Distillation via Label Smoothing."

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;
use crate::logit::hinton_kd::{kl_divergence, softmax_with_temp};

const EPS: f32 = 1e-10;

/// A single element of a batch passed to [`SelfKd::mixup_loss_batch`].
///
/// Fields: `(logits_mix, logits_i, logits_j, lambda, label_i, label_j)`.
pub type MixupBatchElement = (Vec<f32>, Vec<f32>, Vec<f32>, f32, usize, usize);

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for self-knowledge distillation.
#[derive(Debug, Clone)]
pub struct SelfKdConfig {
    /// Temperature for soft labels (default: 1.0, must be > 0).
    pub temperature: f32,
    /// Weight of the self-KD loss component ∈ `[0, 1]` (default: 0.5).
    pub alpha: f32,
    /// Label smoothing ε ∈ `[0, 1)` applied to hard CE targets (default: 0.0).
    pub label_smoothing: f32,
}

impl Default for SelfKdConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            alpha: 0.5,
            label_smoothing: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SelfKd implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Self-knowledge distillation via MixUp / CutMix soft labels.
pub struct SelfKd;

impl SelfKd {
    // ── Internal helper ──────────────────────────────────────────────────────

    /// Validate that `temperature > 0` and `lambda ∈ [0, 1]`.
    fn validate_config_and_lambda(cfg: &SelfKdConfig, lambda: f32) -> DistillResult<()> {
        if cfg.temperature <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("temperature must be > 0, got {}", cfg.temperature),
            });
        }
        if !(0.0..=1.0).contains(&lambda) {
            return Err(DistillError::InvalidConfig {
                msg: format!("lambda must be in [0, 1], got {lambda}"),
            });
        }
        Ok(())
    }

    /// Core MixUp / CutMix loss (shared formula).
    ///
    /// `p_mix_target = λ·softmax(l_i/T) + (1-λ)·softmax(l_j/T)`
    /// `kd = T² · KL(softmax(l_mix/T) ‖ p_mix_target)`
    /// `ce = mixed_ce_smooth(l_mix, label_i, label_j, λ, ε)`
    /// `return α·kd + (1-α)·ce`
    fn mixing_loss_core(
        logits_mix: &[f32],
        logits_i: &[f32],
        logits_j: &[f32],
        lambda: f32,
        label_i: usize,
        label_j: usize,
        cfg: &SelfKdConfig,
    ) -> DistillResult<f32> {
        // Shape validation.
        if logits_mix.is_empty() || logits_i.is_empty() || logits_j.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let n = logits_mix.len();
        if logits_i.len() != n {
            return Err(DistillError::DimensionMismatch {
                expected: n,
                got: logits_i.len(),
            });
        }
        if logits_j.len() != n {
            return Err(DistillError::DimensionMismatch {
                expected: n,
                got: logits_j.len(),
            });
        }
        if label_i >= n {
            return Err(DistillError::InvalidConfig {
                msg: format!("label_i={label_i} out of range for {n} classes"),
            });
        }
        if label_j >= n {
            return Err(DistillError::InvalidConfig {
                msg: format!("label_j={label_j} out of range for {n} classes"),
            });
        }

        let t = cfg.temperature;
        let p_i = softmax_with_temp(logits_i, t);
        let p_j = softmax_with_temp(logits_j, t);

        // Mixed soft target: λ·p_i + (1−λ)·p_j.
        let p_mix_target: Vec<f32> = p_i
            .iter()
            .zip(p_j.iter())
            .map(|(&pi, &pj)| lambda * pi + (1.0 - lambda) * pj)
            .collect();

        let p_mix_student = softmax_with_temp(logits_mix, t);
        let kd = t * t * kl_divergence(&p_mix_student, &p_mix_target);

        let ce = Self::mixed_ce_smooth(logits_mix, label_i, label_j, lambda, cfg.label_smoothing)?;

        Ok(cfg.alpha * kd + (1.0 - cfg.alpha) * ce)
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// MixUp self-KD loss for a single sample pair.
    ///
    /// Given the student's logits for:
    /// - `logits_mix`: the mixed input `x_mix = λ·x_i + (1−λ)·x_j`
    /// - `logits_i`: original sample `x_i`
    /// - `logits_j`: original sample `x_j`
    ///
    /// Mixed soft target: `p_mix = λ·softmax(logits_i/T) + (1−λ)·softmax(logits_j/T)`
    pub fn mixup_loss(
        logits_mix: &[f32],
        logits_i: &[f32],
        logits_j: &[f32],
        lambda: f32,
        label_i: usize,
        label_j: usize,
        cfg: &SelfKdConfig,
    ) -> DistillResult<f32> {
        Self::validate_config_and_lambda(cfg, lambda)?;
        Self::mixing_loss_core(
            logits_mix, logits_i, logits_j, lambda, label_i, label_j, cfg,
        )
    }

    /// CutMix self-KD loss (identical formula to MixUp; λ = area ratio of patch from j).
    ///
    /// The caller computes `logits_mix` by forwarding the CutMix-augmented image through
    /// the student. `lambda` is the fraction of pixels taken from sample j.
    pub fn cutmix_loss(
        logits_mix: &[f32],
        logits_i: &[f32],
        logits_j: &[f32],
        lambda: f32,
        label_i: usize,
        label_j: usize,
        cfg: &SelfKdConfig,
    ) -> DistillResult<f32> {
        Self::validate_config_and_lambda(cfg, lambda)?;
        Self::mixing_loss_core(
            logits_mix, logits_i, logits_j, lambda, label_i, label_j, cfg,
        )
    }

    /// Batch MixUp self-KD loss (mean over elements).
    ///
    /// Each element of `batch` is a [`MixupBatchElement`]:
    /// `(logits_mix, logits_i, logits_j, lambda, label_i, label_j)`.
    pub fn mixup_loss_batch(batch: &[MixupBatchElement], cfg: &SelfKdConfig) -> DistillResult<f32> {
        if batch.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if cfg.temperature <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("temperature must be > 0, got {}", cfg.temperature),
            });
        }
        let mut total = 0.0_f32;
        for (logits_mix, logits_i, logits_j, lambda, label_i, label_j) in batch.iter() {
            total += Self::mixup_loss(
                logits_mix, logits_i, logits_j, *lambda, *label_i, *label_j, cfg,
            )?;
        }
        Ok(total / batch.len() as f32)
    }

    /// Sample a MixUp λ from a folded uniform distribution approximating `Beta(alpha, alpha)`.
    ///
    /// The returned λ is in `[0.5, 1.0]` following common MixUp practice
    /// (`lambda = max(u, 1-u)` for `u ~ Uniform(0,1)`).
    pub fn sample_lambda(_alpha: f32, rng: &mut LcgRng) -> f32 {
        let u = rng.next_f32();
        // Fold into [0.5, 1.0]: `max(u, 1−u)` ensures λ ≥ 0.5.
        if u >= 0.5 { u } else { 1.0 - u }
    }

    /// Feature-level self-consistency loss: mean squared error between augmented and
    /// original feature vectors.
    ///
    /// `L = (1/D) · ‖features_aug − features_orig‖²`
    pub fn feature_consistency_loss(
        features_aug: &[f32],
        features_orig: &[f32],
    ) -> DistillResult<f32> {
        if features_aug.is_empty() || features_orig.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if features_aug.len() != features_orig.len() {
            return Err(DistillError::DimensionMismatch {
                expected: features_orig.len(),
                got: features_aug.len(),
            });
        }
        let n = features_aug.len() as f32;
        let mse: f32 = features_aug
            .iter()
            .zip(features_orig.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / n;
        Ok(mse)
    }

    /// Mixed CE loss with label smoothing.
    ///
    /// Smoothed one-hot for class `y` over `n` classes:
    /// `q_c = (1−ε)·I[c=y] + ε/n`
    ///
    /// `CE_smooth(s, y) = −Σ_c q_c · log(softmax(s)_c + ε)`
    ///
    /// Returns `λ · CE_smooth(logits, label_i) + (1−λ) · CE_smooth(logits, label_j)`.
    pub fn mixed_ce_smooth(
        logits: &[f32],
        label_i: usize,
        label_j: usize,
        lambda: f32,
        epsilon: f32,
    ) -> DistillResult<f32> {
        if logits.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let n = logits.len();
        if label_i >= n {
            return Err(DistillError::InvalidConfig {
                msg: format!("label_i={label_i} out of range for {n} classes"),
            });
        }
        if label_j >= n {
            return Err(DistillError::InvalidConfig {
                msg: format!("label_j={label_j} out of range for {n} classes"),
            });
        }
        let p = softmax_with_temp(logits, 1.0);
        let n_f = n as f32;

        // Cross-entropy with label smoothing for a single target class.
        let ce_smooth = |label: usize| -> f32 {
            p.iter()
                .enumerate()
                .map(|(c, &pc)| {
                    let q_c = if c == label {
                        (1.0 - epsilon) + epsilon / n_f
                    } else {
                        epsilon / n_f
                    };
                    -q_c * (pc + EPS).ln()
                })
                .sum::<f32>()
        };

        let ce_i = ce_smooth(label_i);
        let ce_j = ce_smooth(label_j);
        Ok(lambda * ce_i + (1.0 - lambda) * ce_j)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> SelfKdConfig {
        SelfKdConfig::default()
    }

    // ── 1. self_kd_mixup_loss_finite ─────────────────────────────────────────

    #[test]
    fn self_kd_mixup_loss_finite() {
        let logits_mix = vec![1.0_f32, 2.0, 3.0];
        let logits_i = vec![1.5_f32, 1.5, 3.0];
        let logits_j = vec![0.5_f32, 2.5, 2.0];
        let loss = SelfKd::mixup_loss(&logits_mix, &logits_i, &logits_j, 0.6, 2, 2, &default_cfg())
            .expect("value should be present");
        assert!(
            loss.is_finite() && loss >= 0.0,
            "MixUp loss should be finite non-negative, got {loss}"
        );
    }

    // ── 2. self_kd_mixup_loss_lambda0_is_j_distill ───────────────────────────

    #[test]
    fn self_kd_mixup_loss_lambda0_is_j_distill() {
        // λ=0 → soft target = softmax(logits_j/T), CE target = label_j.
        let logits_mix = vec![0.5_f32, 2.5, 2.0];
        let logits_i = vec![1.5_f32, 1.5, 3.0];
        let logits_j = vec![0.5_f32, 2.5, 2.0];
        let loss = SelfKd::mixup_loss(&logits_mix, &logits_i, &logits_j, 0.0, 0, 1, &default_cfg())
            .expect("value should be present");
        assert!(loss.is_finite(), "lambda=0 loss should be finite");
    }

    // ── 3. self_kd_mixup_loss_lambda1_is_i_distill ───────────────────────────

    #[test]
    fn self_kd_mixup_loss_lambda1_is_i_distill() {
        // λ=1 → soft target = softmax(logits_i/T), CE target = label_i.
        let logits_mix = vec![1.5_f32, 1.5, 3.0];
        let logits_i = vec![1.5_f32, 1.5, 3.0];
        let logits_j = vec![0.5_f32, 2.5, 2.0];
        let loss = SelfKd::mixup_loss(&logits_mix, &logits_i, &logits_j, 1.0, 2, 1, &default_cfg())
            .expect("value should be present");
        assert!(loss.is_finite(), "lambda=1 loss should be finite");
    }

    // ── 4. self_kd_mixup_loss_symmetric ──────────────────────────────────────

    #[test]
    fn self_kd_mixup_loss_symmetric() {
        // Swap (i, j) and (λ, 1-λ): result should be equal when labels are the same.
        let logits_mix = vec![1.0_f32, 2.0, 3.0];
        let logits_i = vec![1.5_f32, 1.5, 3.0];
        let logits_j = vec![0.5_f32, 2.5, 2.0];
        let lam = 0.4_f32;
        let label = 2_usize;
        let loss_ab = SelfKd::mixup_loss(
            &logits_mix,
            &logits_i,
            &logits_j,
            lam,
            label,
            label,
            &default_cfg(),
        )
        .expect("value should be present");
        let loss_ba = SelfKd::mixup_loss(
            &logits_mix,
            &logits_j,
            &logits_i,
            1.0 - lam,
            label,
            label,
            &default_cfg(),
        )
        .expect("value should be present");
        assert!(
            (loss_ab - loss_ba).abs() < 1e-4,
            "symmetric: loss_ab={loss_ab} loss_ba={loss_ba}"
        );
    }

    // ── 5. self_kd_cutmix_matches_mixup ──────────────────────────────────────

    #[test]
    fn self_kd_cutmix_matches_mixup() {
        // CutMix and MixUp use the same loss formula.
        let logits_mix = vec![1.0_f32, 2.0, 3.0];
        let logits_i = vec![1.5_f32, 1.5, 3.0];
        let logits_j = vec![0.5_f32, 2.5, 2.0];
        let lam = 0.7_f32;
        let cfg = default_cfg();
        let mixup = SelfKd::mixup_loss(&logits_mix, &logits_i, &logits_j, lam, 2, 1, &cfg)
            .expect("mixup_loss should succeed");
        let cutmix = SelfKd::cutmix_loss(&logits_mix, &logits_i, &logits_j, lam, 2, 1, &cfg)
            .expect("cutmix_loss should succeed");
        assert!(
            (mixup - cutmix).abs() < 1e-6,
            "cutmix and mixup should be identical: mixup={mixup} cutmix={cutmix}"
        );
    }

    // ── 6. self_kd_identical_logits_kd_zero ──────────────────────────────────

    #[test]
    fn self_kd_identical_logits_kd_zero() {
        // When logits_mix == logits_i == logits_j, KL divergence = 0.
        let logits = vec![1.0_f32, 2.0, 3.0];
        let cfg = SelfKdConfig {
            alpha: 1.0,
            ..default_cfg()
        }; // pure KD loss
        let loss = SelfKd::mixup_loss(&logits, &logits, &logits, 0.5, 2, 2, &cfg)
            .expect("mixup_loss should succeed");
        assert!(
            loss.abs() < 1e-5,
            "identical logits with alpha=1 → KD term = 0, got {loss}"
        );
    }

    // ── 7. self_kd_batch_loss_matches_mean ───────────────────────────────────

    #[test]
    fn self_kd_batch_loss_matches_mean() {
        let cfg = default_cfg();
        let e1 = (
            vec![1.0_f32, 2.0, 3.0],
            vec![1.5_f32, 1.5, 3.0],
            vec![0.5_f32, 2.5, 2.0],
            0.6_f32,
            2_usize,
            1_usize,
        );
        let e2 = (
            vec![0.5_f32, 2.5, 2.0],
            vec![1.0_f32, 2.0, 3.0],
            vec![1.5_f32, 1.5, 3.0],
            0.4_f32,
            1_usize,
            2_usize,
        );
        let batch_loss = SelfKd::mixup_loss_batch(&[e1.clone(), e2.clone()], &cfg)
            .expect("value should be present");
        let l1 = SelfKd::mixup_loss(&e1.0, &e1.1, &e1.2, e1.3, e1.4, e1.5, &cfg)
            .expect("mixup_loss should succeed");
        let l2 = SelfKd::mixup_loss(&e2.0, &e2.1, &e2.2, e2.3, e2.4, e2.5, &cfg)
            .expect("mixup_loss should succeed");
        let expected = (l1 + l2) / 2.0;
        assert!(
            (batch_loss - expected).abs() < 1e-5,
            "batch loss={batch_loss} expected mean={expected}"
        );
    }

    // ── 8. self_kd_batch_empty_error ─────────────────────────────────────────

    #[test]
    fn self_kd_batch_empty_error() {
        let result = SelfKd::mixup_loss_batch(&[], &default_cfg());
        assert!(
            matches!(result, Err(DistillError::EmptyInput)),
            "empty batch should yield EmptyInput"
        );
    }

    // ── 9. self_kd_sample_lambda_in_range ────────────────────────────────────

    #[test]
    fn self_kd_sample_lambda_in_range() {
        let mut rng = LcgRng::new(42);
        for _ in 0..1_000 {
            let lam = SelfKd::sample_lambda(0.4, &mut rng);
            assert!(
                (0.0..=1.0).contains(&lam),
                "sample_lambda must be in [0,1], got {lam}"
            );
        }
    }

    // ── 10. self_kd_sample_lambda_ge_half ────────────────────────────────────

    #[test]
    fn self_kd_sample_lambda_ge_half() {
        let mut rng = LcgRng::new(99);
        for _ in 0..1_000 {
            let lam = SelfKd::sample_lambda(0.4, &mut rng);
            assert!(lam >= 0.5, "max(u, 1-u) should always be >= 0.5, got {lam}");
        }
    }

    // ── 11. self_kd_feature_consistency_zero_diff ────────────────────────────

    #[test]
    fn self_kd_feature_consistency_zero_diff() {
        let f = vec![1.0_f32, 2.0, 3.0, 4.0];
        let loss = SelfKd::feature_consistency_loss(&f, &f)
            .expect("feature_consistency_loss should succeed");
        assert!(
            loss.abs() < 1e-10,
            "identical features → MSE = 0, got {loss}"
        );
    }

    // ── 12. self_kd_feature_consistency_shape_mismatch ───────────────────────

    #[test]
    fn self_kd_feature_consistency_shape_mismatch() {
        let aug = vec![1.0_f32, 2.0, 3.0];
        let orig = vec![1.0_f32, 2.0];
        let result = SelfKd::feature_consistency_loss(&aug, &orig);
        assert!(
            matches!(result, Err(DistillError::DimensionMismatch { .. })),
            "different lengths should yield DimensionMismatch"
        );
    }

    // ── 13. self_kd_feature_consistency_finite ───────────────────────────────

    #[test]
    fn self_kd_feature_consistency_finite() {
        let aug = vec![1.0_f32, 3.0, -1.0, 2.5];
        let orig = vec![0.5_f32, 2.5, 0.5, 1.5];
        let loss = SelfKd::feature_consistency_loss(&aug, &orig)
            .expect("feature_consistency_loss should succeed");
        assert!(
            loss.is_finite() && loss >= 0.0,
            "MSE should be finite non-negative, got {loss}"
        );
    }

    // ── 14. self_kd_mixed_ce_smooth_lambda0 ──────────────────────────────────

    #[test]
    fn self_kd_mixed_ce_smooth_lambda0() {
        // λ=0 → pure CE on label_j.
        let logits = vec![1.0_f32, 2.0, 3.0];
        let ce_j = SelfKd::mixed_ce_smooth(&logits, 0, 2, 0.0, 0.0)
            .expect("mixed_ce_smooth should succeed");
        // Compute CE(logits, label_j=2) manually via softmax + log.
        let p = softmax_with_temp(&logits, 1.0);
        let expected = -(p[2] + EPS).ln();
        assert!(
            (ce_j - expected).abs() < 1e-5,
            "lambda=0: mixed CE should equal CE(label_j), got {ce_j} expected {expected}"
        );
    }

    // ── 15. self_kd_mixed_ce_smooth_lambda1 ──────────────────────────────────

    #[test]
    fn self_kd_mixed_ce_smooth_lambda1() {
        // λ=1 → pure CE on label_i.
        let logits = vec![1.0_f32, 2.0, 3.0];
        let ce_i = SelfKd::mixed_ce_smooth(&logits, 2, 0, 1.0, 0.0)
            .expect("mixed_ce_smooth should succeed");
        let p = softmax_with_temp(&logits, 1.0);
        let expected = -(p[2] + EPS).ln();
        assert!(
            (ce_i - expected).abs() < 1e-5,
            "lambda=1: mixed CE should equal CE(label_i), got {ce_i} expected {expected}"
        );
    }

    // ── 16. self_kd_mixed_ce_smooth_uniform_logits ───────────────────────────

    #[test]
    fn self_kd_mixed_ce_smooth_uniform_logits() {
        // Uniform logits, no label smoothing → CE ≈ ln(n_classes) for any label.
        let n = 4_usize;
        let logits = vec![0.0_f32; n];
        let loss = SelfKd::mixed_ce_smooth(&logits, 0, 1, 0.5, 0.0)
            .expect("mixed_ce_smooth should succeed");
        let expected = (n as f32).ln();
        // Allow generous tolerance due to EPS in log.
        assert!(
            (loss - expected).abs() < 0.01,
            "uniform loss should be ≈ ln(4)={expected:.4}, got {loss}"
        );
    }

    // ── 17. self_kd_label_out_of_bounds_i ────────────────────────────────────

    #[test]
    fn self_kd_label_out_of_bounds_i() {
        let logits_mix = vec![1.0_f32, 2.0, 3.0];
        let logits_i = vec![1.0_f32, 2.0, 3.0];
        let logits_j = vec![1.0_f32, 2.0, 3.0];
        let result = SelfKd::mixup_loss(
            &logits_mix,
            &logits_i,
            &logits_j,
            0.5,
            10,
            1,
            &default_cfg(),
        );
        assert!(
            matches!(result, Err(DistillError::InvalidConfig { .. })),
            "label_i out of bounds should yield InvalidConfig"
        );
    }

    // ── 18. self_kd_label_out_of_bounds_j ────────────────────────────────────

    #[test]
    fn self_kd_label_out_of_bounds_j() {
        let logits_mix = vec![1.0_f32, 2.0, 3.0];
        let logits_i = vec![1.0_f32, 2.0, 3.0];
        let logits_j = vec![1.0_f32, 2.0, 3.0];
        let result = SelfKd::mixup_loss(
            &logits_mix,
            &logits_i,
            &logits_j,
            0.5,
            1,
            10,
            &default_cfg(),
        );
        assert!(
            matches!(result, Err(DistillError::InvalidConfig { .. })),
            "label_j out of bounds should yield InvalidConfig"
        );
    }

    // ── 19. self_kd_invalid_lambda ────────────────────────────────────────────

    #[test]
    fn self_kd_invalid_lambda() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let cfg = default_cfg();

        let result_neg = SelfKd::mixup_loss(&logits, &logits, &logits, -0.1, 0, 0, &cfg);
        assert!(
            matches!(result_neg, Err(DistillError::InvalidConfig { .. })),
            "lambda < 0 should yield InvalidConfig"
        );

        let result_gt1 = SelfKd::mixup_loss(&logits, &logits, &logits, 1.1, 0, 0, &cfg);
        assert!(
            matches!(result_gt1, Err(DistillError::InvalidConfig { .. })),
            "lambda > 1 should yield InvalidConfig"
        );
    }

    // ── 20. self_kd_config_default ───────────────────────────────────────────

    #[test]
    fn self_kd_config_default() {
        let cfg = SelfKdConfig::default();
        assert!(
            (cfg.temperature - 1.0).abs() < f32::EPSILON,
            "default T should be 1.0"
        );
        assert!(
            (cfg.alpha - 0.5).abs() < f32::EPSILON,
            "default alpha should be 0.5"
        );
        assert!(
            cfg.label_smoothing.abs() < f32::EPSILON,
            "default label_smoothing should be 0.0"
        );
    }

    // ── 21. self_kd_config_invalid_temp ──────────────────────────────────────

    #[test]
    fn self_kd_config_invalid_temp() {
        let cfg_bad = SelfKdConfig {
            temperature: -1.0,
            ..default_cfg()
        };
        let logits = vec![1.0_f32, 2.0, 3.0];
        let result = SelfKd::mixup_loss(&logits, &logits, &logits, 0.5, 0, 0, &cfg_bad);
        assert!(
            matches!(result, Err(DistillError::InvalidConfig { .. })),
            "T <= 0 should yield InvalidConfig"
        );
    }

    // ── 22. self_kd_feature_consistency_empty ────────────────────────────────

    #[test]
    fn self_kd_feature_consistency_empty() {
        let result = SelfKd::feature_consistency_loss(&[], &[]);
        assert!(
            matches!(result, Err(DistillError::EmptyInput)),
            "empty features should yield EmptyInput"
        );
    }
}

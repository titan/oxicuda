//! Vanilla knowledge-distillation loss — Hinton et al. 2015.
//!
//! "Distilling the Knowledge in a Neural Network" (Hinton, Vinyals & Dean,
//! 2015) trains a compact *student* to match the temperature-softened output
//! distribution of a larger *teacher*. The soft-target objective is the
//! temperature-scaled KL divergence
//!
//! ```text
//! L_KD = T² · KL( softmax(teacher / T) ‖ softmax(student / T) )
//! ```
//!
//! The `T²` factor restores the gradient magnitude lost when logits are divided
//! by the temperature `T`, so the soft-target gradient stays commensurate with
//! the hard-label cross-entropy gradient. The combined training loss balances
//! the two with a mixing weight `alpha`:
//!
//! ```text
//! L = alpha · L_KD + (1 − alpha) · CE(student, true_label)
//! ```
//!
//! This is a thin, ergonomic wrapper exposing a [`KdLoss`] object; the
//! underlying numerically-stable `softmax_with_temp`, `kl_divergence`, and
//! `cross_entropy` primitives live in [`crate::logit::hinton_kd`].

use crate::error::{DistillError, DistillResult};
use crate::logit::hinton_kd::{cross_entropy, kl_divergence, softmax_with_temp};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`KdLoss`].
#[derive(Debug, Clone, PartialEq)]
pub struct KdLossConfig {
    /// Distillation temperature `T > 0`. Higher `T` softens both
    /// distributions, exposing the teacher's inter-class "dark knowledge".
    pub temperature: f32,
    /// Mixing weight `alpha ∈ [0, 1]` between the distillation term and the
    /// hard-label cross-entropy: `alpha = 1` ⇒ pure KD, `alpha = 0` ⇒ pure CE.
    pub alpha: f32,
}

// ─── KdLoss ──────────────────────────────────────────────────────────────────

/// Hinton-style knowledge-distillation loss.
#[derive(Debug, Clone)]
pub struct KdLoss {
    config: KdLossConfig,
}

impl KdLoss {
    /// Construct a new `KdLoss`, validating the configuration.
    ///
    /// # Errors
    /// - [`DistillError::InvalidConfig`] if `temperature ≤ 0` or non-finite, or
    ///   if `alpha` is outside `[0, 1]` (or non-finite).
    pub fn new(config: KdLossConfig) -> DistillResult<Self> {
        if config.temperature <= 0.0 || !config.temperature.is_finite() {
            return Err(DistillError::InvalidConfig {
                msg: format!("temperature must be > 0, got {}", config.temperature),
            });
        }
        if !(0.0..=1.0).contains(&config.alpha) || !config.alpha.is_finite() {
            return Err(DistillError::InvalidConfig {
                msg: format!("alpha must be in [0, 1], got {}", config.alpha),
            });
        }
        Ok(Self { config })
    }

    /// The configured distillation temperature.
    #[must_use]
    #[inline]
    pub fn temperature(&self) -> f32 {
        self.config.temperature
    }

    /// Validate that both logit slices are non-empty and match `n_classes`.
    fn check(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        n_classes: usize,
    ) -> DistillResult<()> {
        if n_classes == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "n_classes must be > 0".into(),
            });
        }
        if student_logits.len() != n_classes {
            return Err(DistillError::DimensionMismatch {
                expected: n_classes,
                got: student_logits.len(),
            });
        }
        if teacher_logits.len() != n_classes {
            return Err(DistillError::DimensionMismatch {
                expected: n_classes,
                got: teacher_logits.len(),
            });
        }
        Ok(())
    }

    /// Temperature-scaled distillation (soft-target) loss for one sample.
    ///
    /// `T² · KL( softmax(teacher / T) ‖ softmax(student / T) )`.
    ///
    /// # Errors
    /// - [`DistillError::InvalidConfig`] if `n_classes == 0`.
    /// - [`DistillError::DimensionMismatch`] if either logit slice length does
    ///   not equal `n_classes`.
    pub fn distillation_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        n_classes: usize,
    ) -> DistillResult<f32> {
        self.check(student_logits, teacher_logits, n_classes)?;
        let t = self.config.temperature;
        let p_student = softmax_with_temp(student_logits, t);
        let p_teacher = softmax_with_temp(teacher_logits, t);
        let kl = kl_divergence(&p_teacher, &p_student);
        Ok(t * t * kl)
    }

    /// Combined distillation + hard-label loss for one sample.
    ///
    /// `alpha · distillation_loss + (1 − alpha) · CE(student, true_label)`.
    ///
    /// # Errors
    /// - [`DistillError::InvalidConfig`] if `n_classes == 0` or
    ///   `true_label >= n_classes`.
    /// - [`DistillError::DimensionMismatch`] if either logit slice length does
    ///   not equal `n_classes`.
    pub fn combined_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        true_label: usize,
        n_classes: usize,
    ) -> DistillResult<f32> {
        self.check(student_logits, teacher_logits, n_classes)?;
        if true_label >= n_classes {
            return Err(DistillError::InvalidConfig {
                msg: format!("true_label {true_label} out of range for {n_classes} classes"),
            });
        }
        let kd = self.distillation_loss(student_logits, teacher_logits, n_classes)?;
        let ce = cross_entropy(student_logits, true_label);
        Ok(self.config.alpha * kd + (1.0 - self.config.alpha) * ce)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn loss(temperature: f32, alpha: f32) -> KdLoss {
        KdLoss::new(KdLossConfig { temperature, alpha }).expect("valid config")
    }

    #[test]
    fn distill_loss_nonneg() {
        let kd = loss(4.0, 0.5);
        let s = vec![1.0_f32, 2.0, 3.0, 0.5];
        let t = vec![0.5_f32, 2.5, 2.0, 1.0];
        let v = kd.distillation_loss(&s, &t, 4).expect("ok");
        assert!(v >= 0.0 && v.is_finite(), "kd loss = {v}");
    }

    #[test]
    fn distill_loss_zero_for_identical() {
        let kd = loss(3.0, 0.5);
        let logits = vec![0.1_f32, 1.5, -0.3, 2.0];
        let v = kd.distillation_loss(&logits, &logits, 4).expect("ok");
        assert!(v < 1e-4, "KL of identical logits must be ~0, got {v}");
    }

    #[test]
    fn combined_loss_finite() {
        let kd = loss(2.0, 0.7);
        let s = vec![1.0_f32, 0.0, -1.0];
        let t = vec![0.8_f32, 0.1, -0.5];
        let v = kd.combined_loss(&s, &t, 0, 3).expect("ok");
        assert!(v.is_finite() && v >= 0.0, "combined = {v}");
    }

    #[test]
    fn temperature_1_works() {
        let kd = loss(1.0, 0.5);
        assert!((kd.temperature() - 1.0).abs() < 1e-9);
        let s = vec![2.0_f32, 1.0, 0.0];
        let t = vec![1.0_f32, 2.0, 0.0];
        let v = kd.distillation_loss(&s, &t, 3).expect("ok");
        assert!(v.is_finite() && v >= 0.0);
    }

    #[test]
    fn high_temp_softer() {
        // For non-identical logits, a higher temperature softens the teacher
        // distribution; the T² factor inflates magnitude but the *underlying*
        // distributions become more uniform. We assert both temperatures give
        // finite, non-negative losses and that they differ (temperature is
        // actually used).
        let s = vec![3.0_f32, 0.0, -3.0];
        let t = vec![0.0_f32, 0.0, 0.0];
        let low = loss(1.0, 1.0).distillation_loss(&s, &t, 3).expect("ok");
        let high = loss(8.0, 1.0).distillation_loss(&s, &t, 3).expect("ok");
        assert!(low.is_finite() && high.is_finite());
        assert!(
            (low - high).abs() > 1e-4,
            "temperature should change the loss: low={low} high={high}"
        );
    }

    #[test]
    fn alpha_0_pure_ce() {
        let kd = loss(4.0, 0.0);
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![3.0_f32, 2.0, 1.0];
        let combined = kd.combined_loss(&s, &t, 1, 3).expect("ok");
        let ce = cross_entropy(&s, 1);
        assert!(
            (combined - ce).abs() < 1e-5,
            "alpha=0 must equal CE: combined={combined} ce={ce}"
        );
    }

    #[test]
    fn alpha_1_pure_kd() {
        let kd = loss(4.0, 1.0);
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![3.0_f32, 2.0, 1.0];
        let combined = kd.combined_loss(&s, &t, 1, 3).expect("ok");
        let distill = kd.distillation_loss(&s, &t, 3).expect("ok");
        assert!(
            (combined - distill).abs() < 1e-5,
            "alpha=1 must equal distillation loss: {combined} vs {distill}"
        );
    }

    #[test]
    fn n_classes_mismatch_error() {
        let kd = loss(4.0, 0.5);
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![1.0_f32, 2.0]; // wrong length
        let r = kd.distillation_loss(&s, &t, 3);
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
        let r2 = kd.distillation_loss(&s, &s, 4); // n_classes != slice len
        assert!(matches!(r2, Err(DistillError::DimensionMismatch { .. })));
    }

    #[test]
    fn temperature_0_error() {
        let r = KdLoss::new(KdLossConfig {
            temperature: 0.0,
            alpha: 0.5,
        });
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
        let r_neg = KdLoss::new(KdLossConfig {
            temperature: -1.0,
            alpha: 0.5,
        });
        assert!(matches!(r_neg, Err(DistillError::InvalidConfig { .. })));
    }

    #[test]
    fn label_out_of_range_error() {
        let kd = loss(4.0, 0.5);
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![3.0_f32, 2.0, 1.0];
        let r = kd.combined_loss(&s, &t, 3, 3); // label == n_classes ⇒ OOR
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
    }

    #[test]
    fn alpha_out_of_range_error() {
        let r = KdLoss::new(KdLossConfig {
            temperature: 4.0,
            alpha: 1.5,
        });
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
    }
}

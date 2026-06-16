//! Logit Standardization in Knowledge Distillation (Sun et al. 2024).
//!
//! Reference: Sun, S., Ren, W., Li, J., Wang, R., & Cao, X. (2024).
//! *Logit Standardization in Knowledge Distillation*. CVPR 2024.
//! <https://arxiv.org/abs/2403.01427>
//!
//! Vanilla Hinton KD forces the student to match the *exact magnitude* of the teacher's
//! logits, but the student and teacher generally operate at different logit scales (and
//! "logit shifts"). Sun et al. show that the softmax is invariant to a per-sample affine
//! transform of the logits, so the only distributional information that matters is the
//! *relative* structure. They therefore apply a **Z-score standardisation** to both the
//! teacher and student logits *per sample* before the temperature-scaled softmax:
//!
//! ```text
//!   z(v)_k = (v_k − μ(v)) / (σ(v) + ε)        with  μ = mean(v),  σ = std(v)
//! ```
//!
//! The standardised logits are then divided by the temperature and softmaxed as usual. This
//! decouples the student from the teacher's absolute logit range and consistently improves
//! distillation across teacher/student capacity gaps.
//!
//! The loss is the temperature-scaled KL divergence on the standardised distributions:
//!
//! ```text
//!   L = T² · KL( p_t ‖ p_s )      where  p = softmax( z(v) / T )
//! ```
//!
//! optionally combined with a hard cross-entropy term weighted by `alpha`.

use crate::error::{DistillError, DistillResult};
use crate::logit::hinton_kd::{cross_entropy, kl_divergence};

const EPS: f32 = 1e-7;

/// Configuration for logit-standardisation KD.
#[derive(Debug, Clone)]
pub struct LogitStdConfig {
    /// Temperature `T` applied after standardisation (> 0).
    pub temperature: f32,
    /// Weight of the soft (KD) term ∈ `[0, 1]`; the hard CE weight is `1 − alpha`.
    pub alpha: f32,
}

impl LogitStdConfig {
    fn validate(&self) -> DistillResult<()> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "temperature must be finite and > 0, got {}",
                    self.temperature
                ),
            });
        }
        if !self.alpha.is_finite() || !(0.0..=1.0).contains(&self.alpha) {
            return Err(DistillError::InvalidConfig {
                msg: format!("alpha must be in [0, 1], got {}", self.alpha),
            });
        }
        Ok(())
    }
}

/// Z-score standardise a logit vector: `(v_k − μ) / (σ + ε)`.
///
/// `μ` is the mean and `σ` the population standard deviation (divide by `n`, not `n − 1`).
/// For a constant vector (`σ = 0`) the result is all zeros.
#[must_use]
pub fn standardize(logits: &[f32]) -> Vec<f32> {
    let n = logits.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = logits.iter().sum::<f32>() / n as f32;
    let var = logits.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    logits.iter().map(|&v| (v - mean) / (std + EPS)).collect()
}

/// Temperature-scaled softmax of an already-standardised logit vector.
///
/// Standardisation is performed internally; the input is the *raw* logits.
#[must_use]
pub fn standardized_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let t = if temperature.abs() < EPS {
        EPS
    } else {
        temperature
    };
    let z = standardize(logits);
    let max_val = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = z.iter().map(|&x| ((x - max_val) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let sum_safe = if sum < EPS { EPS } else { sum };
    exps.iter().map(|&e| e / sum_safe).collect()
}

/// Logit-standardisation KD loss for a single sample.
///
/// `soft = alpha · T² · KL(p_t ‖ p_s)` with standardised, temperature-scaled distributions,
/// plus `hard = (1 − alpha) · CE(student_logits, label)` on the *raw* student logits.
///
/// # Errors
/// - [`DistillError::EmptyInput`] if either logit slice is empty.
/// - [`DistillError::InvalidConfig`] if the config is invalid or `label` is out of range.
/// - [`DistillError::DimensionMismatch`] if the logit slices differ in length.
pub fn logit_std_kd_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    label: usize,
    cfg: &LogitStdConfig,
) -> DistillResult<f32> {
    cfg.validate()?;
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
    let p_s = standardized_softmax(student_logits, t);
    let p_t = standardized_softmax(teacher_logits, t);
    let soft = cfg.alpha * t * t * kl_divergence(&p_t, &p_s);
    let hard = (1.0 - cfg.alpha) * cross_entropy(student_logits, label);
    let loss = soft + hard;
    if !loss.is_finite() {
        return Err(DistillError::NumericalError {
            msg: "logit-std KD loss is not finite".into(),
        });
    }
    Ok(loss)
}

/// Mean logit-standardisation KD loss over a batch.
///
/// # Errors
/// - [`DistillError::EmptyInput`] if `s_batch` is empty.
/// - [`DistillError::DimensionMismatch`] if batch sizes disagree.
/// - Propagates per-sample errors from [`logit_std_kd_loss`].
pub fn logit_std_kd_loss_batch(
    s_batch: &[Vec<f32>],
    t_batch: &[Vec<f32>],
    labels: &[usize],
    cfg: &LogitStdConfig,
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
        total += logit_std_kd_loss(s, t, lbl, cfg)?;
    }
    Ok(total / s_batch.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardize_zero_mean_unit_var() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let z = standardize(&v);
        let mean: f32 = z.iter().sum::<f32>() / z.len() as f32;
        let var: f32 = z.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / z.len() as f32;
        assert!(
            mean.abs() < 1e-4,
            "standardized mean must be ~0, got {mean}"
        );
        assert!(
            (var - 1.0).abs() < 1e-3,
            "standardized var must be ~1, got {var}"
        );
    }

    #[test]
    fn standardize_constant_is_zeros() {
        let v = vec![3.0_f32; 5];
        let z = standardize(&v);
        for &x in &z {
            assert!(x.abs() < 1e-5, "constant input → zeros, got {x}");
        }
    }

    #[test]
    fn standardize_empty_is_empty() {
        let z = standardize(&[]);
        assert!(z.is_empty());
    }

    #[test]
    fn standardize_shift_invariant() {
        // Adding a constant to all logits must not change the standardised output.
        let a = vec![1.0_f32, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|&x| x + 100.0).collect();
        let za = standardize(&a);
        let zb = standardize(&b);
        for (x, y) in za.iter().zip(zb.iter()) {
            assert!(
                (x - y).abs() < 1e-3,
                "shift should be invariant: {x} vs {y}"
            );
        }
    }

    #[test]
    fn standardized_softmax_sums_to_one() {
        let v = vec![1.0_f32, 5.0, 2.0, 0.5];
        let p = standardized_softmax(&v, 2.0);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
    }

    #[test]
    fn standardized_softmax_scale_invariant() {
        // Scaling logits by a positive factor leaves the standardised softmax unchanged.
        let a = vec![1.0_f32, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|&x| x * 10.0).collect();
        let pa = standardized_softmax(&a, 1.0);
        let pb = standardized_softmax(&b, 1.0);
        for (x, y) in pa.iter().zip(pb.iter()) {
            assert!(
                (x - y).abs() < 1e-3,
                "scale should be invariant: {x} vs {y}"
            );
        }
    }

    #[test]
    fn loss_identical_logits_equals_hard_only() {
        // Identical student/teacher → KL ~ 0 → loss ≈ (1-alpha)·CE.
        let cfg = LogitStdConfig {
            temperature: 2.0,
            alpha: 0.7,
        };
        let v = vec![1.0_f32, 2.0, 3.0, 0.5];
        let loss = logit_std_kd_loss(&v, &v, 2, &cfg).expect("logit_std_kd_loss should succeed");
        let hard = (1.0 - cfg.alpha) * cross_entropy(&v, 2);
        assert!((loss - hard).abs() < 1e-4, "loss={loss}, hard={hard}");
    }

    #[test]
    fn loss_finite_and_nonneg() {
        let cfg = LogitStdConfig {
            temperature: 4.0,
            alpha: 0.5,
        };
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![10.0_f32, 21.0, 29.0]; // different scale
        let loss = logit_std_kd_loss(&s, &t, 1, &cfg).expect("logit_std_kd_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn loss_robust_to_teacher_scale() {
        // The whole point of logit standardisation: loss should be (nearly) invariant to a
        // pure scaling of the teacher logits when alpha=1 (pure soft loss).
        let cfg = LogitStdConfig {
            temperature: 2.0,
            alpha: 1.0,
        };
        let s = vec![1.0_f32, 2.0, 3.0, 1.5];
        let t1 = vec![0.5_f32, 1.5, 2.5, 1.0];
        let t2: Vec<f32> = t1.iter().map(|&x| x * 5.0).collect();
        let l1 = logit_std_kd_loss(&s, &t1, 2, &cfg).expect("logit_std_kd_loss should succeed");
        let l2 = logit_std_kd_loss(&s, &t2, 2, &cfg).expect("logit_std_kd_loss should succeed");
        assert!(
            (l1 - l2).abs() < 1e-2,
            "loss must be ~scale-invariant: {l1} vs {l2}"
        );
    }

    #[test]
    fn loss_empty_input_errors() {
        let cfg = LogitStdConfig {
            temperature: 2.0,
            alpha: 0.5,
        };
        assert!(matches!(
            logit_std_kd_loss(&[], &[1.0, 2.0], 0, &cfg),
            Err(DistillError::EmptyInput)
        ));
    }

    #[test]
    fn loss_dimension_mismatch_errors() {
        let cfg = LogitStdConfig {
            temperature: 2.0,
            alpha: 0.5,
        };
        let s = vec![1.0_f32, 2.0];
        let t = vec![1.0_f32, 2.0, 3.0];
        assert!(matches!(
            logit_std_kd_loss(&s, &t, 0, &cfg),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn loss_invalid_temperature_errors() {
        let cfg = LogitStdConfig {
            temperature: 0.0,
            alpha: 0.5,
        };
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!(matches!(
            logit_std_kd_loss(&v, &v, 0, &cfg),
            Err(DistillError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn loss_invalid_alpha_errors() {
        let cfg = LogitStdConfig {
            temperature: 2.0,
            alpha: 1.5,
        };
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!(matches!(
            logit_std_kd_loss(&v, &v, 0, &cfg),
            Err(DistillError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn loss_label_out_of_range_errors() {
        let cfg = LogitStdConfig {
            temperature: 2.0,
            alpha: 0.5,
        };
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!(matches!(
            logit_std_kd_loss(&v, &v, 5, &cfg),
            Err(DistillError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn batch_loss_is_mean() {
        let cfg = LogitStdConfig {
            temperature: 2.0,
            alpha: 0.5,
        };
        let s = vec![vec![1.0_f32, 2.0, 3.0], vec![0.5_f32, 1.5, 2.5]];
        let t = vec![vec![1.1_f32, 2.1, 2.9], vec![0.6_f32, 1.4, 2.6]];
        let labels = vec![2, 1];
        let l0 = logit_std_kd_loss(&s[0], &t[0], labels[0], &cfg)
            .expect("logit_std_kd_loss should succeed");
        let l1 = logit_std_kd_loss(&s[1], &t[1], labels[1], &cfg)
            .expect("logit_std_kd_loss should succeed");
        let mean = logit_std_kd_loss_batch(&s, &t, &labels, &cfg)
            .expect("logit_std_kd_loss_batch should succeed");
        assert!(
            (mean - (l0 + l1) / 2.0).abs() < 1e-5,
            "mean mismatch: {mean}"
        );
    }

    #[test]
    fn batch_loss_empty_errors() {
        let cfg = LogitStdConfig {
            temperature: 2.0,
            alpha: 0.5,
        };
        let s: Vec<Vec<f32>> = vec![];
        let t: Vec<Vec<f32>> = vec![];
        let labels: Vec<usize> = vec![];
        assert!(matches!(
            logit_std_kd_loss_batch(&s, &t, &labels, &cfg),
            Err(DistillError::EmptyInput)
        ));
    }
}

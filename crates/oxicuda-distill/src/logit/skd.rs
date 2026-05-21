//! Switchable Knowledge Distillation (SKD) with student-dependent adaptive gating.
//!
//! SKD adapts the KD weight α based on the student's current prediction confidence
//! or entropy. When the student is uncertain (high entropy) it relies more on the
//! teacher; when confident it falls back to ground-truth labels. A multi-teacher
//! variant is also provided via differentiable gating over cosine similarity.

use crate::error::{DistillError, DistillResult};
use crate::logit::hinton_kd::{cross_entropy, kl_divergence, softmax_with_temp};

const EPS: f32 = 1e-10;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Mode for adaptive KD weighting.
#[derive(Debug, Clone)]
pub enum SkdGatingMode {
    /// Gate based on entropy of student logits.
    ///
    /// High entropy → high KD weight (confused student needs more teacher guidance).
    /// `gate(H) = σ(β · (H − H_threshold))` where `H = −Σ p·log(p)`.
    EntropyBased {
        /// Sharpness of sigmoid gate (default: 5.0).
        beta: f32,
        /// Entropy threshold for switching (default: 0.5).
        entropy_threshold: f32,
    },
    /// Gate based on student top-1 confidence.
    ///
    /// Low max probability → high KD weight.
    /// `gate(conf) = clamp(alpha_max − (alpha_max − alpha_min) · max_p, alpha_min, alpha_max)`
    ConfidenceBased {
        /// Minimum KD weight (default: 0.0).
        alpha_min: f32,
        /// Maximum KD weight (default: 1.0).
        alpha_max: f32,
    },
    /// Fixed gate — identical to standard Hinton KD (baseline comparison).
    Fixed {
        /// Fixed KD weight in `[0, 1]`.
        alpha: f32,
    },
}

impl Default for SkdGatingMode {
    fn default() -> Self {
        Self::EntropyBased {
            beta: 5.0,
            entropy_threshold: 0.5,
        }
    }
}

/// Configuration for Switchable KD.
#[derive(Debug, Clone)]
pub struct SkdConfig {
    /// Distillation temperature T (must be > 0).
    pub temperature: f32,
    /// Gating mechanism to compute adaptive α.
    pub gating_mode: SkdGatingMode,
}

impl SkdConfig {
    /// Construct a new [`SkdConfig`] with the given temperature and default gating mode.
    ///
    /// Returns [`DistillError::InvalidConfig`] if `temperature ≤ 0`.
    pub fn new(temperature: f32) -> DistillResult<Self> {
        if temperature <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("temperature must be > 0, got {temperature}"),
            });
        }
        Ok(Self {
            temperature,
            gating_mode: SkdGatingMode::default(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Skd implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Switchable Knowledge Distillation — single-teacher and multi-teacher variants.
pub struct Skd;

impl Skd {
    // ── Helper: sigmoid ──────────────────────────────────────────────────────

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Compute the Shannon entropy of the distribution implied by `logits`.
    ///
    /// `H = −Σ p_i · log(p_i + ε)`  where `p = softmax(logits)` and `ε = 1e-10`.
    pub fn entropy(logits: &[f32]) -> DistillResult<f32> {
        if logits.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let p = softmax_with_temp(logits, 1.0);
        let h: f32 = p.iter().map(|&pi| -pi * (pi + EPS).ln()).sum();
        Ok(h)
    }

    /// Compute the adaptive KD weight α for a single sample based on `student_logits`.
    ///
    /// The returned value is in `[0, 1]`.
    pub fn adaptive_alpha(student_logits: &[f32], mode: &SkdGatingMode) -> DistillResult<f32> {
        if student_logits.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let alpha = match mode {
            SkdGatingMode::EntropyBased {
                beta,
                entropy_threshold,
            } => {
                let h = Self::entropy(student_logits)?;
                Self::sigmoid(beta * (h - entropy_threshold))
            }
            SkdGatingMode::ConfidenceBased {
                alpha_min,
                alpha_max,
            } => {
                let p = softmax_with_temp(student_logits, 1.0);
                let max_conf = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let raw = alpha_max - (alpha_max - alpha_min) * max_conf;
                raw.clamp(*alpha_min, *alpha_max)
            }
            SkdGatingMode::Fixed { alpha } => *alpha,
        };
        Ok(alpha)
    }

    /// SKD loss for one sample.
    ///
    /// `loss = α · T² · KL(softmax(s/T) ‖ softmax(t/T)) + (1−α) · CE(s, label)`
    ///
    /// where `α = adaptive_alpha(student_logits, mode)`.
    pub fn skd_loss(
        student_logits: &[f32],
        teacher_logits: &[f32],
        label: usize,
        cfg: &SkdConfig,
    ) -> DistillResult<f32> {
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
        let alpha = Self::adaptive_alpha(student_logits, &cfg.gating_mode)?;
        let soft_student = softmax_with_temp(student_logits, t);
        let soft_teacher = softmax_with_temp(teacher_logits, t);
        let kd = t * t * kl_divergence(&soft_student, &soft_teacher);
        let hard = cross_entropy(student_logits, label);
        Ok(alpha * kd + (1.0 - alpha) * hard)
    }

    /// SKD loss over a batch — average of per-sample losses.
    pub fn skd_loss_batch(
        student_logits: &[Vec<f32>],
        teacher_logits: &[Vec<f32>],
        labels: &[usize],
        cfg: &SkdConfig,
    ) -> DistillResult<f32> {
        if student_logits.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if student_logits.len() != teacher_logits.len() || student_logits.len() != labels.len() {
            return Err(DistillError::DimensionMismatch {
                expected: student_logits.len(),
                got: teacher_logits.len().min(labels.len()),
            });
        }
        let mut total = 0.0_f32;
        for ((s, t), &lbl) in student_logits
            .iter()
            .zip(teacher_logits.iter())
            .zip(labels.iter())
        {
            total += Self::skd_loss(s, t, lbl, cfg)?;
        }
        Ok(total / student_logits.len() as f32)
    }

    /// Cosine similarity between two probability vectors (normalised dot product).
    ///
    /// Returns a value in `[−1, 1]`.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> DistillResult<f32> {
        if a.is_empty() || b.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if a.len() != b.len() {
            return Err(DistillError::DimensionMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
        let norm_a: f32 = a.iter().map(|&ai| ai * ai).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|&bi| bi * bi).sum::<f32>().sqrt();
        Ok(dot / (norm_a * norm_b + EPS))
    }

    /// Multi-teacher SKD: differentiable mixture of multiple teacher predictions.
    ///
    /// Gate scores for teacher `i`:
    /// `g_i = gate_beta · cosine_similarity(softmax(s/T), softmax(t_i/T))`
    ///
    /// Gate weights: `w = softmax(g, T=1)`
    ///
    /// Mixed soft target: `y_mix = Σ_i w_i · softmax(t_i / T)`
    ///
    /// Loss: `α · T² · KL(softmax(s/T) ‖ y_mix) + (1−α) · CE(s, label)`
    pub fn multi_teacher_loss(
        student_logits: &[f32],
        teacher_logits_list: &[Vec<f32>],
        label: usize,
        cfg: &SkdConfig,
        gate_beta: f32,
    ) -> DistillResult<f32> {
        if student_logits.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if teacher_logits_list.is_empty() {
            return Err(DistillError::EmptyInput);
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
        let n_classes = student_logits.len();
        let t = cfg.temperature;

        // Validate all teacher logit lengths.
        for (i, t_logits) in teacher_logits_list.iter().enumerate() {
            if t_logits.len() != n_classes {
                return Err(DistillError::DimensionMismatch {
                    expected: n_classes,
                    got: t_logits.len(),
                });
            }
            // Require non-empty (already checked via n_classes > 0 from student guard above,
            // but be explicit for each teacher slice)
            if t_logits.is_empty() {
                return Err(DistillError::InvalidConfig {
                    msg: format!("teacher_logits_list[{i}] is empty"),
                });
            }
        }

        let student_p = softmax_with_temp(student_logits, t);

        // Compute gate score for each teacher via cosine similarity.
        let gate_scores: Vec<f32> = teacher_logits_list
            .iter()
            .map(|t_logits| {
                let teacher_p = softmax_with_temp(t_logits, t);
                let sim = Self::cosine_similarity(&student_p, &teacher_p)?;
                Ok(gate_beta * sim)
            })
            .collect::<DistillResult<Vec<f32>>>()?;

        // Softmax over gate scores (T=1.0) to get mixture weights.
        let gate_weights = softmax_with_temp(&gate_scores, 1.0);

        // Build mixed soft target: Σ_i w_i * softmax(teacher_i / T).
        let mut mixed_target = vec![0.0_f32; n_classes];
        for (w, t_logits) in gate_weights.iter().zip(teacher_logits_list.iter()) {
            let teacher_p = softmax_with_temp(t_logits, t);
            for (m, &tp) in mixed_target.iter_mut().zip(teacher_p.iter()) {
                *m += w * tp;
            }
        }

        // KL divergence loss.
        let kd = t * t * kl_divergence(&student_p, &mixed_target);

        // CE + adaptive alpha.
        let hard = cross_entropy(student_logits, label);
        let alpha = Self::adaptive_alpha(student_logits, &cfg.gating_mode)?;

        Ok(alpha * kd + (1.0 - alpha) * hard)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. skd_entropy_uniform ───────────────────────────────────────────────

    #[test]
    fn skd_entropy_uniform() {
        // Uniform logits → max entropy = ln(n_classes).
        let n = 4_usize;
        let logits = vec![0.0_f32; n];
        let h = Skd::entropy(&logits).unwrap();
        let expected = (n as f32).ln();
        assert!(
            (h - expected).abs() < 1e-5,
            "uniform entropy should be ln({n})={expected}, got {h}"
        );
    }

    // ── 2. skd_entropy_peaked ────────────────────────────────────────────────

    #[test]
    fn skd_entropy_peaked() {
        // Highly peaked distribution → entropy close to 0.
        let logits = vec![100.0_f32, 0.0, 0.0, 0.0];
        let h = Skd::entropy(&logits).unwrap();
        assert!(
            h < 0.01,
            "peaked distribution should have low entropy, got {h}"
        );
    }

    // ── 3. skd_entropy_empty_error ───────────────────────────────────────────

    #[test]
    fn skd_entropy_empty_error() {
        let result = Skd::entropy(&[]);
        assert!(
            matches!(result, Err(DistillError::EmptyInput)),
            "empty logits should yield EmptyInput"
        );
    }

    // ── 4. skd_adaptive_alpha_entropy_high_gives_high_weight ─────────────────

    #[test]
    fn skd_adaptive_alpha_entropy_high_gives_high_weight() {
        // Uniform logits → high entropy → alpha near 1.
        let logits = vec![0.0_f32; 10];
        let mode = SkdGatingMode::EntropyBased {
            beta: 5.0,
            entropy_threshold: 0.5,
        };
        let alpha = Skd::adaptive_alpha(&logits, &mode).unwrap();
        assert!(
            alpha > 0.7,
            "high-entropy student should yield alpha > 0.7, got {alpha}"
        );
    }

    // ── 5. skd_adaptive_alpha_entropy_low_gives_low_weight ───────────────────

    #[test]
    fn skd_adaptive_alpha_entropy_low_gives_low_weight() {
        // Peaked logits → low entropy → alpha near 0.
        let logits = vec![100.0_f32, 0.0, 0.0, 0.0];
        let mode = SkdGatingMode::EntropyBased {
            beta: 5.0,
            entropy_threshold: 0.5,
        };
        let alpha = Skd::adaptive_alpha(&logits, &mode).unwrap();
        assert!(
            alpha < 0.3,
            "low-entropy student should yield alpha < 0.3, got {alpha}"
        );
    }

    // ── 6. skd_adaptive_alpha_confidence_mode ────────────────────────────────

    #[test]
    fn skd_adaptive_alpha_confidence_mode() {
        // Uniform softmax (max_conf = 1/n) → alpha near alpha_max.
        let logits = vec![0.0_f32; 4];
        let mode = SkdGatingMode::ConfidenceBased {
            alpha_min: 0.0,
            alpha_max: 1.0,
        };
        let alpha = Skd::adaptive_alpha(&logits, &mode).unwrap();
        // max_conf ≈ 0.25 → alpha ≈ 1.0 - 1.0 * 0.25 = 0.75
        assert!(
            (alpha - 0.75).abs() < 0.01,
            "uniform confidence mode alpha should be ≈ 0.75, got {alpha}"
        );

        // Peaked distribution → max_conf ≈ 1 → alpha ≈ 0.
        let logits_peaked = vec![100.0_f32, 0.0, 0.0, 0.0];
        let alpha_peaked = Skd::adaptive_alpha(&logits_peaked, &mode).unwrap();
        assert!(
            alpha_peaked < 0.01,
            "high-confidence student → alpha ≈ 0, got {alpha_peaked}"
        );
    }

    // ── 7. skd_adaptive_alpha_fixed_mode ─────────────────────────────────────

    #[test]
    fn skd_adaptive_alpha_fixed_mode() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let mode = SkdGatingMode::Fixed { alpha: 0.42 };
        let alpha = Skd::adaptive_alpha(&logits, &mode).unwrap();
        assert!(
            (alpha - 0.42).abs() < 1e-6,
            "fixed mode must return exact alpha, got {alpha}"
        );
    }

    // ── 8. skd_loss_single_sample_finite ─────────────────────────────────────

    #[test]
    fn skd_loss_single_sample_finite() {
        let cfg = SkdConfig::new(4.0).unwrap();
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![1.1_f32, 2.1, 2.9];
        let loss = Skd::skd_loss(&s, &t, 2, &cfg).unwrap();
        assert!(
            loss.is_finite() && loss >= 0.0,
            "loss should be finite non-negative, got {loss}"
        );
    }

    // ── 9. skd_loss_teacher_equals_student ───────────────────────────────────

    #[test]
    fn skd_loss_teacher_equals_student() {
        // When student == teacher logits, KL = 0, loss = (1-alpha) * CE.
        let logits = vec![1.0_f32, 2.0, 3.0];
        let mode = SkdGatingMode::Fixed { alpha: 0.5 };
        let cfg = SkdConfig {
            temperature: 2.0,
            gating_mode: mode,
        };
        let loss = Skd::skd_loss(&logits, &logits, 1, &cfg).unwrap();
        // Expected: 0.5 * 0 + 0.5 * CE = 0.5 * CE
        let ce = cross_entropy(&logits, 1);
        let expected = 0.5 * ce;
        assert!(
            (loss - expected).abs() < 1e-4,
            "identical logits: loss={loss} expected 0.5·CE={expected}"
        );
    }

    // ── 10. skd_loss_batch_shape ──────────────────────────────────────────────

    #[test]
    fn skd_loss_batch_shape() {
        let mode = SkdGatingMode::Fixed { alpha: 0.5 };
        let cfg = SkdConfig {
            temperature: 2.0,
            gating_mode: mode,
        };
        let s1 = vec![1.0_f32, 2.0, 3.0];
        let s2 = vec![0.5_f32, 1.5, 2.5];
        let t1 = vec![1.1_f32, 2.1, 2.9];
        let t2 = vec![0.4_f32, 1.6, 2.4];

        let batch_loss = Skd::skd_loss_batch(
            &[s1.clone(), s2.clone()],
            &[t1.clone(), t2.clone()],
            &[2, 2],
            &cfg,
        )
        .unwrap();

        let l1 = Skd::skd_loss(&s1, &t1, 2, &cfg).unwrap();
        let l2 = Skd::skd_loss(&s2, &t2, 2, &cfg).unwrap();
        let expected = (l1 + l2) / 2.0;

        assert!(
            (batch_loss - expected).abs() < 1e-5,
            "batch loss={batch_loss} expected mean={expected}"
        );
    }

    // ── 11. skd_loss_batch_empty_error ───────────────────────────────────────

    #[test]
    fn skd_loss_batch_empty_error() {
        let cfg = SkdConfig::new(2.0).unwrap();
        let result = Skd::skd_loss_batch(&[], &[], &[], &cfg);
        assert!(
            matches!(result, Err(DistillError::EmptyInput)),
            "empty batch should yield EmptyInput"
        );
    }

    // ── 12. skd_loss_label_out_of_bounds ─────────────────────────────────────

    #[test]
    fn skd_loss_label_out_of_bounds() {
        let cfg = SkdConfig::new(2.0).unwrap();
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![1.0_f32, 2.0, 3.0];
        let result = Skd::skd_loss(&s, &t, 5, &cfg); // label=5 >= n_classes=3
        assert!(
            matches!(result, Err(DistillError::InvalidConfig { .. })),
            "label out-of-bounds should yield InvalidConfig"
        );
    }

    // ── 13. skd_temp_1_fixed_alpha0_equals_ce ────────────────────────────────

    #[test]
    fn skd_temp_1_fixed_alpha0_equals_ce() {
        // T=1, alpha=0 → pure CE (Hinton baseline with no KD).
        let mode = SkdGatingMode::Fixed { alpha: 0.0 };
        let cfg = SkdConfig {
            temperature: 1.0,
            gating_mode: mode,
        };
        let s = vec![1.0_f32, 2.0, 3.0, 0.5];
        let t = vec![2.0_f32, 1.0, 3.0, 0.5];
        let loss = Skd::skd_loss(&s, &t, 2, &cfg).unwrap();
        let ce = cross_entropy(&s, 2);
        assert!(
            (loss - ce).abs() < 1e-4,
            "T=1, alpha=0 → loss={loss} should equal CE={ce}"
        );
    }

    // ── 14. skd_multi_teacher_shape ──────────────────────────────────────────

    #[test]
    fn skd_multi_teacher_shape() {
        let cfg = SkdConfig::new(2.0).unwrap();
        let s = vec![1.0_f32, 2.0, 3.0];
        let t1 = vec![1.5_f32, 1.5, 3.0];
        let t2 = vec![0.5_f32, 2.5, 2.0];
        let loss = Skd::multi_teacher_loss(&s, &[t1, t2], 2, &cfg, 2.0).unwrap();
        assert!(
            loss.is_finite() && loss >= 0.0,
            "multi-teacher loss should be finite non-negative"
        );
    }

    // ── 15. skd_multi_teacher_uniform_teachers ───────────────────────────────

    #[test]
    fn skd_multi_teacher_uniform_teachers() {
        // Identical teacher logits → same result regardless of gating (both get weight 0.5).
        let mode = SkdGatingMode::Fixed { alpha: 0.5 };
        let cfg = SkdConfig {
            temperature: 2.0,
            gating_mode: mode,
        };
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![1.5_f32, 2.5, 2.0];
        // Single teacher vs two identical teachers.
        let loss_single = Skd::skd_loss(&s, &t, 2, &cfg).unwrap();
        let loss_multi =
            Skd::multi_teacher_loss(&s, &[t.clone(), t.clone()], 2, &cfg, 1.0).unwrap();
        // Both should be finite.
        assert!(loss_single.is_finite());
        assert!(loss_multi.is_finite());
        // Multi-teacher with two identical teachers should match single-teacher KD loss closely.
        assert!(
            (loss_single - loss_multi).abs() < 1e-4,
            "identical teachers: single={loss_single} multi={loss_multi}"
        );
    }

    // ── 16. skd_cosine_similarity_same_vector ────────────────────────────────

    #[test]
    fn skd_cosine_similarity_same_vector() {
        let v = vec![0.3_f32, 0.5, 0.2];
        let sim = Skd::cosine_similarity(&v, &v).unwrap();
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "cos_sim(v, v) should be 1.0, got {sim}"
        );
    }

    // ── 17. skd_cosine_similarity_orthogonal ─────────────────────────────────

    #[test]
    fn skd_cosine_similarity_orthogonal() {
        // Strict orthogonal vectors.
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = Skd::cosine_similarity(&a, &b).unwrap();
        // Orthogonal vectors have dot product = 0 → cosine similarity ≈ 0.
        assert!(
            sim.abs() < 1e-5,
            "orthogonal vectors should have cos_sim ≈ 0, got {sim}"
        );
    }

    // ── 18. skd_cosine_similarity_empty_error ────────────────────────────────

    #[test]
    fn skd_cosine_similarity_empty_error() {
        let result = Skd::cosine_similarity(&[], &[]);
        assert!(
            matches!(result, Err(DistillError::EmptyInput)),
            "empty inputs should yield EmptyInput"
        );
    }

    // ── 19. skd_config_invalid_temp ──────────────────────────────────────────

    #[test]
    fn skd_config_invalid_temp() {
        assert!(
            matches!(SkdConfig::new(0.0), Err(DistillError::InvalidConfig { .. })),
            "temperature=0 should yield InvalidConfig"
        );
        assert!(
            matches!(
                SkdConfig::new(-1.0),
                Err(DistillError::InvalidConfig { .. })
            ),
            "temperature=-1 should yield InvalidConfig"
        );
    }

    // ── 20. skd_batch_size_mismatch_error ────────────────────────────────────

    #[test]
    fn skd_batch_size_mismatch_error() {
        let cfg = SkdConfig::new(2.0).unwrap();
        let s = vec![vec![1.0_f32, 2.0, 3.0], vec![1.0_f32, 2.0, 3.0]];
        let t = vec![vec![1.0_f32, 2.0, 3.0]]; // length mismatch
        let result = Skd::skd_loss_batch(&s, &t, &[0, 0], &cfg);
        assert!(
            matches!(result, Err(DistillError::DimensionMismatch { .. })),
            "mismatched batch sizes should yield DimensionMismatch"
        );
    }

    // ── 21. skd_entropy_single_class ─────────────────────────────────────────

    #[test]
    fn skd_entropy_single_class() {
        // Single-class distribution: softmax = [1.0] → H ≈ 0.
        let logits = vec![5.0_f32];
        let h = Skd::entropy(&logits).unwrap();
        // p=[1.0], H = -1.0 * ln(1+eps) ≈ 0
        assert!(
            h.abs() < 1e-4,
            "single-class entropy should be ≈ 0, got {h}"
        );
    }
}

//! Robust accuracy and certified accuracy.
//!
//! * **Robust accuracy** — fraction of samples that the model classifies
//!   correctly under an attack.
//! * **Certified accuracy** — fraction of samples that are *both* correctly
//!   classified by the smoothed/base classifier *and* whose certified radius
//!   meets or exceeds a target radius.

use crate::error::{AdvError, AdvResult};

/// Robust accuracy on a batch of `(adv_pred, label)` pairs.
///
/// # Errors
/// * [`AdvError::EmptyInput`]        — empty inputs.
/// * [`AdvError::DimensionMismatch`] — length mismatch.
pub fn robust_accuracy(adv_pred: &[usize], labels: &[usize]) -> AdvResult<f32> {
    if adv_pred.is_empty() || labels.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if adv_pred.len() != labels.len() {
        return Err(AdvError::DimensionMismatch {
            expected: labels.len(),
            got: adv_pred.len(),
        });
    }
    let n = adv_pred.len();
    let correct = adv_pred
        .iter()
        .zip(labels.iter())
        .filter(|(p, y)| p == y)
        .count();
    Ok(correct as f32 / n as f32)
}

/// Certified accuracy at a target radius `r_target`.
///
/// `pred[i]` is the (smoothed) classifier's prediction, `labels[i]` is the
/// ground truth, and `radii[i]` is the certified L2 radius (or any other
/// pre-chosen Lp radius). A sample is "certified" iff the prediction is
/// correct and `radii[i] >= r_target`.
///
/// # Errors
/// * [`AdvError::EmptyInput`]        — empty inputs.
/// * [`AdvError::DimensionMismatch`] — any length mismatch.
/// * [`AdvError::InvalidEpsilon`]    — `r_target` is negative or non-finite.
/// * [`AdvError::NanEncountered`]    — non-finite radius.
pub fn certified_accuracy(
    pred: &[usize],
    labels: &[usize],
    radii: &[f32],
    r_target: f32,
) -> AdvResult<f32> {
    if pred.is_empty() || labels.is_empty() || radii.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if pred.len() != labels.len() {
        return Err(AdvError::DimensionMismatch {
            expected: labels.len(),
            got: pred.len(),
        });
    }
    if radii.len() != pred.len() {
        return Err(AdvError::DimensionMismatch {
            expected: pred.len(),
            got: radii.len(),
        });
    }
    if !(r_target.is_finite() && r_target >= 0.0) {
        return Err(AdvError::InvalidEpsilon { eps: r_target });
    }
    if radii.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "certified_accuracy:radii",
        });
    }
    let n = pred.len();
    let cert = pred
        .iter()
        .zip(labels.iter())
        .zip(radii.iter())
        .filter(|((p, y), r)| p == y && **r >= r_target)
        .count();
    Ok(cert as f32 / n as f32)
}

// ─── Per-class robust accuracy report ────────────────────────────────────────

/// Configuration for per-class robust accuracy computation.
#[derive(Debug, Clone, Copy)]
pub struct RobustAccConfig {
    /// Number of output classes (must be ≥ 1).
    pub n_classes: usize,
    /// L∞ perturbation radius used to generate adversarial examples.
    pub epsilon: f32,
}

/// Per-class accuracy summary produced by [`robust_accuracy_report`].
#[derive(Debug, Clone)]
pub struct ClassResult {
    /// Class index (0-based).
    pub class: usize,
    /// Fraction of *clean* predictions correct for this class.
    pub clean_acc: f32,
    /// Fraction of *adversarial* predictions correct for this class.
    pub robust_acc: f32,
    /// Number of samples belonging to this class.
    pub n_samples: usize,
}

/// Compute per-class clean and robust accuracy from pre-computed logit arrays.
///
/// `clean_logits` and `adv_logits` are flat row-major matrices of shape
/// `[n_samples × n_classes]`.  A sample is "clean-correct" when the argmax
/// of its clean logit row equals its label; similarly for "robust-correct".
///
/// # Errors
/// * [`AdvError::EmptyInput`]         — `n_samples == 0`.
/// * [`AdvError::Internal`]           — `n_classes == 0`.
/// * [`AdvError::DimensionMismatch`]  — logit buffer length ≠ `n_samples * n_classes`,
///   or `labels.len() ≠ n_samples`.
pub fn robust_accuracy_report(
    clean_logits: &[f32],
    adv_logits: &[f32],
    labels: &[usize],
    n_samples: usize,
    n_classes: usize,
) -> AdvResult<Vec<ClassResult>> {
    if n_samples == 0 {
        return Err(AdvError::EmptyInput);
    }
    if n_classes == 0 {
        return Err(AdvError::Internal("n_classes must be > 0".into()));
    }
    let expected_len = n_samples
        .checked_mul(n_classes)
        .ok_or_else(|| AdvError::Internal("n_samples * n_classes overflows usize".into()))?;
    if clean_logits.len() != expected_len {
        return Err(AdvError::DimensionMismatch {
            expected: expected_len,
            got: clean_logits.len(),
        });
    }
    if adv_logits.len() != expected_len {
        return Err(AdvError::DimensionMismatch {
            expected: expected_len,
            got: adv_logits.len(),
        });
    }
    if labels.len() != n_samples {
        return Err(AdvError::DimensionMismatch {
            expected: n_samples,
            got: labels.len(),
        });
    }

    // Helper: argmax over a logit row.
    let argmax = |row: &[f32]| -> usize {
        row.iter()
            .enumerate()
            .fold(0, |best, (i, &v)| if v > row[best] { i } else { best })
    };

    let mut results = Vec::with_capacity(n_classes);
    for c in 0..n_classes {
        let mut n_class = 0_usize;
        let mut clean_correct = 0_usize;
        let mut robust_correct = 0_usize;

        for (i, &label) in labels.iter().enumerate() {
            if label != c {
                continue;
            }
            n_class += 1;
            let base = i * n_classes;
            let clean_row = &clean_logits[base..base + n_classes];
            let adv_row = &adv_logits[base..base + n_classes];
            if argmax(clean_row) == c {
                clean_correct += 1;
            }
            if argmax(adv_row) == c {
                robust_correct += 1;
            }
        }

        let (clean_acc, robust_acc) = if n_class > 0 {
            (
                clean_correct as f32 / n_class as f32,
                robust_correct as f32 / n_class as f32,
            )
        } else {
            (0.0_f32, 0.0_f32)
        };

        results.push(ClassResult {
            class: c,
            clean_acc,
            robust_acc,
            n_samples: n_class,
        });
    }
    Ok(results)
}

/// Return the minimum robust accuracy across all classes.
///
/// Returns `1.0` if `results` is empty (vacuously perfect robustness).
pub fn worst_class_robust_acc(results: &[ClassResult]) -> f32 {
    results.iter().map(|r| r.robust_acc).fold(1.0_f32, f32::min)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_acc_perfect() {
        let p = vec![0_usize, 1, 2];
        let y = p.clone();
        assert!(
            (robust_accuracy(&p, &y).expect("robust_accuracy should succeed") - 1.0).abs() < 1e-6
        );
    }

    #[test]
    fn robust_acc_zero() {
        let p = vec![0_usize, 1, 2];
        let y = vec![1_usize, 2, 0];
        assert!((robust_accuracy(&p, &y).expect("robust_accuracy should succeed")).abs() < 1e-6);
    }

    #[test]
    fn robust_acc_dim_mismatch() {
        assert!(matches!(
            robust_accuracy(&[0_usize, 1], &[0_usize]).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn robust_acc_empty() {
        assert!(matches!(
            robust_accuracy(&[], &[]).unwrap_err(),
            AdvError::EmptyInput
        ));
    }

    #[test]
    fn certified_acc_basic() {
        // 3 samples; predictions all correct; radii [0.1, 0.5, 1.0].
        // r_target = 0.4 ⇒ samples 1 and 2 qualify ⇒ 2/3.
        let p = vec![0_usize, 1, 2];
        let y = p.clone();
        let r = vec![0.1_f32, 0.5, 1.0];
        let acc = certified_accuracy(&p, &y, &r, 0.4).expect("certified_accuracy should succeed");
        assert!((acc - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn certified_acc_wrong_pred_excluded() {
        // Sample 0 wrong → never certified regardless of radius.
        let p = vec![1_usize, 1];
        let y = vec![0_usize, 1];
        let r = vec![10.0_f32, 0.5];
        let acc = certified_accuracy(&p, &y, &r, 0.1).expect("certified_accuracy should succeed");
        assert!((acc - 0.5).abs() < 1e-6);
    }

    #[test]
    fn certified_acc_invalid_target() {
        let p = vec![0_usize];
        let y = vec![0_usize];
        let r = vec![0.5_f32];
        assert!(matches!(
            certified_accuracy(&p, &y, &r, -0.1).unwrap_err(),
            AdvError::InvalidEpsilon { .. }
        ));
        assert!(matches!(
            certified_accuracy(&p, &y, &r, f32::NAN).unwrap_err(),
            AdvError::InvalidEpsilon { .. }
        ));
    }

    #[test]
    fn certified_acc_nan_radius() {
        let p = vec![0_usize];
        let y = vec![0_usize];
        let r = vec![f32::NAN];
        assert!(matches!(
            certified_accuracy(&p, &y, &r, 0.1).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    // ── robust_accuracy_report tests ─────────────────────────────────────────

    /// Build a minimal logit flat array for `n_samples` × `n_classes`.
    /// `preds[i]` is the class that sample `i` should predict (highest logit).
    fn make_logits(n_samples: usize, n_classes: usize, preds: &[usize]) -> Vec<f32> {
        let mut v = vec![0.0_f32; n_samples * n_classes];
        for (i, &p) in preds.iter().enumerate() {
            // Make pred class logit = 1.0, others = 0.0.
            v[i * n_classes + p] = 1.0;
        }
        v
    }

    // ── 1. report_len_equals_n_classes ───────────────────────────────────────
    #[test]
    fn report_len_equals_n_classes() {
        let n_samples = 6;
        let n_classes = 3;
        let labels = vec![0_usize, 0, 1, 1, 2, 2];
        let clean = make_logits(n_samples, n_classes, &labels);
        let adv = make_logits(n_samples, n_classes, &labels);
        let res = robust_accuracy_report(&clean, &adv, &labels, n_samples, n_classes)
            .expect("robust_accuracy_report should succeed");
        assert_eq!(res.len(), n_classes);
    }

    // ── 2. acc_in_range ──────────────────────────────────────────────────────
    #[test]
    fn acc_in_range() {
        let n_samples = 6;
        let n_classes = 3;
        let labels = vec![0_usize, 0, 1, 1, 2, 2];
        let clean = make_logits(n_samples, n_classes, &labels);
        let adv_preds = vec![0_usize, 1, 1, 0, 2, 1]; // some wrong
        let adv = make_logits(n_samples, n_classes, &adv_preds);
        let res = robust_accuracy_report(&clean, &adv, &labels, n_samples, n_classes)
            .expect("robust_accuracy_report should succeed");
        for r in &res {
            assert!(
                (0.0..=1.0).contains(&r.clean_acc),
                "clean_acc {} out of range",
                r.clean_acc
            );
            assert!(
                (0.0..=1.0).contains(&r.robust_acc),
                "robust_acc {} out of range",
                r.robust_acc
            );
        }
    }

    // ── 3. worst_class_min ───────────────────────────────────────────────────
    #[test]
    fn worst_class_min() {
        let results = vec![
            ClassResult {
                class: 0,
                clean_acc: 1.0,
                robust_acc: 0.8,
                n_samples: 2,
            },
            ClassResult {
                class: 1,
                clean_acc: 0.9,
                robust_acc: 0.5,
                n_samples: 2,
            },
            ClassResult {
                class: 2,
                clean_acc: 1.0,
                robust_acc: 0.6,
                n_samples: 2,
            },
        ];
        let worst = worst_class_robust_acc(&results);
        assert!((worst - 0.5).abs() < 1e-6);
    }

    // ── 4. perfect_clean_acc ─────────────────────────────────────────────────
    #[test]
    fn perfect_clean_acc() {
        let n_samples = 4;
        let n_classes = 2;
        let labels = vec![0_usize, 0, 1, 1];
        let clean = make_logits(n_samples, n_classes, &labels);
        let adv = make_logits(n_samples, n_classes, &labels);
        let res = robust_accuracy_report(&clean, &adv, &labels, n_samples, n_classes)
            .expect("robust_accuracy_report should succeed");
        for r in &res {
            assert!(
                (r.clean_acc - 1.0).abs() < 1e-6,
                "class {} clean_acc {}",
                r.class,
                r.clean_acc
            );
        }
    }

    // ── 5. all_wrong_zero_acc ────────────────────────────────────────────────
    #[test]
    fn all_wrong_zero_acc() {
        let n_samples = 4;
        let n_classes = 2;
        let labels = vec![0_usize, 0, 1, 1];
        let clean = make_logits(n_samples, n_classes, &labels);
        // adv always predicts the *other* class.
        let adv_preds = vec![1_usize, 1, 0, 0];
        let adv = make_logits(n_samples, n_classes, &adv_preds);
        let res = robust_accuracy_report(&clean, &adv, &labels, n_samples, n_classes)
            .expect("robust_accuracy_report should succeed");
        for r in &res {
            assert!(
                r.robust_acc.abs() < 1e-6,
                "class {} robust_acc {}",
                r.class,
                r.robust_acc
            );
        }
    }

    // ── 6. robust_le_clean ───────────────────────────────────────────────────
    #[test]
    fn robust_le_clean() {
        let n_samples = 6;
        let n_classes = 3;
        let labels = vec![0_usize, 0, 1, 1, 2, 2];
        let clean = make_logits(n_samples, n_classes, &labels); // all correct
        // adv: first sample of each class is wrong.
        let adv_preds = vec![1_usize, 0, 0, 1, 1, 2];
        let adv = make_logits(n_samples, n_classes, &adv_preds);
        let res = robust_accuracy_report(&clean, &adv, &labels, n_samples, n_classes)
            .expect("robust_accuracy_report should succeed");
        for r in &res {
            assert!(
                r.robust_acc <= r.clean_acc + 1e-5,
                "class {}: robust_acc {} > clean_acc {}",
                r.class,
                r.robust_acc,
                r.clean_acc
            );
        }
    }

    // ── 7. empty_input_error ─────────────────────────────────────────────────
    #[test]
    fn empty_input_error() {
        let result = robust_accuracy_report(&[], &[], &[], 0, 3);
        assert!(matches!(result.unwrap_err(), AdvError::EmptyInput));
    }

    // ── 8. single_class ──────────────────────────────────────────────────────
    #[test]
    fn single_class() {
        let n_samples = 3;
        let n_classes = 1;
        let labels = vec![0_usize, 0, 0];
        // Single-class logit: always predicts class 0.
        let logits = vec![1.0_f32; n_samples]; // [n_samples × 1]
        let res = robust_accuracy_report(&logits, &logits, &labels, n_samples, n_classes)
            .expect("robust_accuracy_report should succeed");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].class, 0);
        assert!((res[0].clean_acc - 1.0).abs() < 1e-6);
        assert!((res[0].robust_acc - 1.0).abs() < 1e-6);
    }

    // ── 9. robust_acc_finite ─────────────────────────────────────────────────
    #[test]
    fn robust_acc_finite() {
        let n_samples = 6;
        let n_classes = 3;
        let labels = vec![0_usize, 0, 1, 1, 2, 2];
        let clean = make_logits(n_samples, n_classes, &labels);
        let adv = make_logits(n_samples, n_classes, &labels);
        let res = robust_accuracy_report(&clean, &adv, &labels, n_samples, n_classes)
            .expect("robust_accuracy_report should succeed");
        for r in &res {
            assert!(
                r.robust_acc.is_finite(),
                "non-finite robust_acc for class {}",
                r.class
            );
        }
    }
}

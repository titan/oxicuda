//! Stratified robust-accuracy reporting.
//!
//! When a model is attacked, an aggregate (micro) robust accuracy can hide
//! sharp per-class disparities — a defence may keep an overall robust accuracy
//! of 60 % while one class collapses to 0 %. This module computes, from
//! post-attack predictions and ground-truth labels:
//!
//! * **per-class robust accuracy** — fraction of each class's samples still
//!   classified correctly under attack;
//! * **worst-class robust accuracy** — the minimum over *populated* classes
//!   (the standard fairness-flavoured robustness summary);
//! * **macro-average robust accuracy** — the unweighted mean over populated
//!   classes (every class counts equally regardless of size);
//! * **micro-average robust accuracy** — the sample-weighted overall accuracy.
//!
//! Classes that have **no** samples in `labels` are reported (with
//! `n_samples == 0`, `robust_acc == 0`) but **excluded** from the worst-class
//! and macro-average aggregates so that absent classes neither dominate the
//! worst case nor dilute the macro mean.
//!
//! Unlike [`crate::metrics::robust_accuracy::robust_accuracy_report`] (which
//! consumes raw logit matrices), this reporter consumes the already-decoded
//! predicted classes, and additionally returns macro/micro aggregates.

use crate::error::{AdvError, AdvResult};

/// Per-class robust-accuracy entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassRobustness {
    /// Class index (0-based).
    pub class: usize,
    /// Number of samples whose ground-truth label is this class.
    pub n_samples: usize,
    /// Samples of this class still classified correctly under attack.
    pub correct: usize,
    /// `correct / n_samples`, or `0.0` when the class has no samples.
    pub robust_acc: f32,
}

impl ClassRobustness {
    /// Whether this class has at least one sample.
    #[must_use]
    pub fn is_populated(&self) -> bool {
        self.n_samples > 0
    }
}

/// Full stratified robust-accuracy report.
#[derive(Debug, Clone, PartialEq)]
pub struct StratifiedReport {
    /// Number of classes (length of [`StratifiedReport::per_class`]).
    pub n_classes: usize,
    /// One [`ClassRobustness`] per class index `0..n_classes`.
    pub per_class: Vec<ClassRobustness>,
    /// Minimum robust accuracy over *populated* classes; `1.0` if no class is
    /// populated (vacuously perfect).
    pub worst_class: f32,
    /// Index of the worst populated class; `usize::MAX` if none is populated.
    pub worst_class_idx: usize,
    /// Unweighted mean robust accuracy over *populated* classes; `0.0` if none.
    pub macro_avg: f32,
    /// Sample-weighted overall robust accuracy (`total_correct / n_samples`).
    pub micro_avg: f32,
}

/// Compute a stratified robust-accuracy report from post-attack predictions.
///
/// * `pred[i]`   — predicted class of sample `i` under attack.
/// * `labels[i]` — ground-truth class of sample `i` (must be `< n_classes`).
/// * `n_classes` — number of classes (`>= 1`).
///
/// # Errors
/// * [`AdvError::EmptyInput`]         — empty `pred`/`labels`.
/// * [`AdvError::Internal`]           — `n_classes == 0`.
/// * [`AdvError::DimensionMismatch`]  — `pred.len() != labels.len()`, or a
///   label is `>= n_classes`.
pub fn stratified_robust_accuracy(
    pred: &[usize],
    labels: &[usize],
    n_classes: usize,
) -> AdvResult<StratifiedReport> {
    if pred.is_empty() || labels.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if n_classes == 0 {
        return Err(AdvError::Internal("n_classes must be > 0".into()));
    }
    if pred.len() != labels.len() {
        return Err(AdvError::DimensionMismatch {
            expected: labels.len(),
            got: pred.len(),
        });
    }
    if let Some(&bad) = labels.iter().find(|&&y| y >= n_classes) {
        return Err(AdvError::DimensionMismatch {
            expected: n_classes,
            got: bad + 1,
        });
    }

    let mut n_per = vec![0_usize; n_classes];
    let mut correct_per = vec![0_usize; n_classes];
    let mut total_correct = 0_usize;
    for (&p, &y) in pred.iter().zip(labels.iter()) {
        n_per[y] += 1;
        if p == y {
            correct_per[y] += 1;
            total_correct += 1;
        }
    }

    let mut per_class = Vec::with_capacity(n_classes);
    let mut worst_class = 1.0_f32;
    let mut worst_class_idx = usize::MAX;
    let mut macro_sum = 0.0_f32;
    let mut populated = 0_usize;

    for c in 0..n_classes {
        let n_c = n_per[c];
        let correct = correct_per[c];
        let robust_acc = if n_c > 0 {
            correct as f32 / n_c as f32
        } else {
            0.0
        };
        per_class.push(ClassRobustness {
            class: c,
            n_samples: n_c,
            correct,
            robust_acc,
        });
        if n_c > 0 {
            populated += 1;
            macro_sum += robust_acc;
            if worst_class_idx == usize::MAX || robust_acc < worst_class {
                worst_class = robust_acc;
                worst_class_idx = c;
            }
        }
    }

    let macro_avg = if populated > 0 {
        macro_sum / populated as f32
    } else {
        0.0
    };
    let micro_avg = total_correct as f32 / pred.len() as f32;
    // No populated class ⇒ vacuously perfect worst case (mirrors
    // `worst_class_robust_acc`'s empty convention).
    if worst_class_idx == usize::MAX {
        worst_class = 1.0;
    }

    Ok(StratifiedReport {
        n_classes,
        per_class,
        worst_class,
        worst_class_idx,
        macro_avg,
        micro_avg,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    // ── per-class rates on a hand-built example ─────────────────────────────

    #[test]
    fn per_class_rates_hand_built() {
        // 3 classes.
        // class 0: samples at idx 0,1,2 → preds 0,0,1 → 2/3 correct.
        // class 1: samples at idx 3,4   → preds 1,2   → 1/2 correct.
        // class 2: samples at idx 5,6,7 → preds 2,2,2 → 3/3 correct.
        let labels = vec![0_usize, 0, 0, 1, 1, 2, 2, 2];
        let pred = vec![0_usize, 0, 1, 1, 2, 2, 2, 2];
        let r = stratified_robust_accuracy(&pred, &labels, 3)
            .expect("stratified_robust_accuracy should succeed");
        assert_eq!(r.per_class.len(), 3);
        assert!(approx(r.per_class[0].robust_acc, 2.0 / 3.0));
        assert!(approx(r.per_class[1].robust_acc, 0.5));
        assert!(approx(r.per_class[2].robust_acc, 1.0));
        assert_eq!(r.per_class[0].n_samples, 3);
        assert_eq!(r.per_class[0].correct, 2);
    }

    // ── worst-class == min of per-class ─────────────────────────────────────

    #[test]
    fn worst_class_is_min_over_populated() {
        let labels = vec![0_usize, 0, 0, 1, 1, 2, 2, 2];
        let pred = vec![0_usize, 0, 1, 1, 2, 2, 2, 2];
        let r = stratified_robust_accuracy(&pred, &labels, 3)
            .expect("stratified_robust_accuracy should succeed");
        // min(2/3, 1/2, 1) = 1/2 at class 1.
        assert!(approx(r.worst_class, 0.5));
        assert_eq!(r.worst_class_idx, 1);
        let min_pc = r
            .per_class
            .iter()
            .filter(|c| c.is_populated())
            .map(|c| c.robust_acc)
            .fold(f32::INFINITY, f32::min);
        assert!(approx(r.worst_class, min_pc));
    }

    // ── macro-average correctness ───────────────────────────────────────────

    #[test]
    fn macro_average_correct() {
        let labels = vec![0_usize, 0, 0, 1, 1, 2, 2, 2];
        let pred = vec![0_usize, 0, 1, 1, 2, 2, 2, 2];
        let r = stratified_robust_accuracy(&pred, &labels, 3)
            .expect("stratified_robust_accuracy should succeed");
        // (2/3 + 1/2 + 1) / 3.
        let expect = (2.0 / 3.0 + 0.5 + 1.0) / 3.0;
        assert!(approx(r.macro_avg, expect), "macro={}", r.macro_avg);
    }

    // ── micro-average correctness ───────────────────────────────────────────

    #[test]
    fn micro_average_is_overall_accuracy() {
        let labels = vec![0_usize, 0, 0, 1, 1, 2, 2, 2];
        let pred = vec![0_usize, 0, 1, 1, 2, 2, 2, 2];
        let r = stratified_robust_accuracy(&pred, &labels, 3)
            .expect("stratified_robust_accuracy should succeed");
        // correct: idx 0,1 (cls0), idx 3 (cls1), idx 5,6,7 (cls2) = 6 of 8.
        assert!(approx(r.micro_avg, 6.0 / 8.0), "micro={}", r.micro_avg);
    }

    // ── empty class is skipped from aggregates ──────────────────────────────

    #[test]
    fn empty_class_skipped_from_aggregates() {
        // 4 classes declared but class 3 never appears.
        let labels = vec![0_usize, 1, 1, 2];
        let pred = vec![1_usize, 1, 1, 2]; // cls0 wrong; cls1,2 perfect.
        let r = stratified_robust_accuracy(&pred, &labels, 4)
            .expect("stratified_robust_accuracy should succeed");
        assert_eq!(r.per_class.len(), 4);
        // class 3 present but unpopulated.
        assert_eq!(r.per_class[3].n_samples, 0);
        assert!(!r.per_class[3].is_populated());
        assert_eq!(r.per_class[3].robust_acc, 0.0);
        // worst class is class 0 (0.0), NOT the empty class 3.
        assert!(approx(r.worst_class, 0.0));
        assert_eq!(r.worst_class_idx, 0);
        // macro over populated {0,1,2}: (0 + 1 + 1)/3.
        assert!(approx(r.macro_avg, 2.0 / 3.0));
    }

    // ── rates in [0, 1] ─────────────────────────────────────────────────────

    #[test]
    fn all_rates_in_unit_interval() {
        let labels = vec![0_usize, 1, 2, 0, 1, 2];
        let pred = vec![0_usize, 2, 2, 1, 1, 0];
        let r = stratified_robust_accuracy(&pred, &labels, 3)
            .expect("stratified_robust_accuracy should succeed");
        for c in &r.per_class {
            assert!((0.0..=1.0).contains(&c.robust_acc));
        }
        assert!((0.0..=1.0).contains(&r.worst_class));
        assert!((0.0..=1.0).contains(&r.macro_avg));
        assert!((0.0..=1.0).contains(&r.micro_avg));
    }

    // ── perfect & zero extremes ─────────────────────────────────────────────

    #[test]
    fn perfect_defence_all_one() {
        let labels = vec![0_usize, 1, 2, 1];
        let pred = labels.clone();
        let r = stratified_robust_accuracy(&pred, &labels, 3)
            .expect("stratified_robust_accuracy should succeed");
        assert!(approx(r.worst_class, 1.0));
        assert!(approx(r.macro_avg, 1.0));
        assert!(approx(r.micro_avg, 1.0));
    }

    #[test]
    fn broken_defence_all_zero() {
        let labels = vec![0_usize, 1, 2];
        let pred = vec![1_usize, 2, 0];
        let r = stratified_robust_accuracy(&pred, &labels, 3)
            .expect("stratified_robust_accuracy should succeed");
        assert!(approx(r.worst_class, 0.0));
        assert!(approx(r.macro_avg, 0.0));
        assert!(approx(r.micro_avg, 0.0));
    }

    // ── error handling ──────────────────────────────────────────────────────

    #[test]
    fn empty_inputs_rejected() {
        assert!(matches!(
            stratified_robust_accuracy(&[], &[], 3).unwrap_err(),
            AdvError::EmptyInput
        ));
    }

    #[test]
    fn zero_classes_rejected() {
        assert!(matches!(
            stratified_robust_accuracy(&[0_usize], &[0_usize], 0).unwrap_err(),
            AdvError::Internal(_)
        ));
    }

    #[test]
    fn length_mismatch_rejected() {
        assert!(matches!(
            stratified_robust_accuracy(&[0_usize, 1], &[0_usize], 2).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn label_out_of_range_rejected() {
        // label 5 but only 3 classes.
        assert!(matches!(
            stratified_robust_accuracy(&[0_usize, 1], &[0_usize, 5], 3).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }
}

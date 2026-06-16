//! Adaptive Prediction Sets (APS) for multi-class conformal prediction.
//!
//! Romano, Sesia & Candès (2020): "Classification with Valid and Adaptive
//! Coverage", NeurIPS 2020. Score for sample `i` = cumulative probability mass
//! in descending-probability order up to and including the true class. The
//! threshold is the `(1 − alpha)` quantile of calibration scores. At test
//! time, classes are added in descending order until the cumulative mass
//! reaches the threshold.

use crate::error::{TabularError, TabularResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`ApsConformal`].
#[derive(Debug, Clone, Copy)]
pub struct ApsConformalConfig {
    /// Target miscoverage rate `alpha ∈ (0, 1)`.
    /// Coverage target is `1 − alpha`.
    pub alpha: f32,
}

// ─── ApsConformal ─────────────────────────────────────────────────────────────

/// Adaptive Prediction Sets (APS) conformal classifier.
///
/// Provides distribution-free marginal coverage of `≥ 1 − alpha` under
/// exchangeability of calibration and test data.
pub struct ApsConformal {
    threshold: f32,
    n_cal: usize,
    config: ApsConformalConfig,
}

impl ApsConformal {
    /// Calibrate from softmax probability vectors and true labels.
    ///
    /// `cal_probs` is a row-major `[n_cal × n_classes]` matrix; each row
    /// should sum to approximately 1.
    ///
    /// # Errors
    /// - [`TabularError::EmptyInput`] when `n_cal == 0`.
    /// - [`TabularError::DimensionMismatch`] on shape mismatch.
    /// - [`TabularError::LabelOutOfRange`] when any label `≥ n_classes`.
    pub fn calibrate(
        cal_probs: &[f32],
        cal_labels: &[usize],
        n_cal: usize,
        n_classes: usize,
        config: ApsConformalConfig,
    ) -> TabularResult<Self> {
        if n_cal == 0 {
            return Err(TabularError::EmptyInput);
        }
        let expected = n_cal * n_classes;
        if cal_probs.len() != expected {
            return Err(TabularError::DimensionMismatch {
                expected,
                got: cal_probs.len(),
            });
        }
        if cal_labels.len() != n_cal {
            return Err(TabularError::DimensionMismatch {
                expected: n_cal,
                got: cal_labels.len(),
            });
        }
        for &label in cal_labels {
            if label >= n_classes {
                return Err(TabularError::LabelOutOfRange { label, n_classes });
            }
        }

        // Compute APS score for each calibration sample.
        let mut scores = Vec::with_capacity(n_cal);
        for i in 0..n_cal {
            let row = &cal_probs[i * n_classes..(i + 1) * n_classes];
            let sorted_idx = argsort_desc(row);
            let mut cumulative = 0.0_f32;
            let mut score = 1.0_f32;
            for &c in &sorted_idx {
                cumulative += row[c];
                if c == cal_labels[i] {
                    score = cumulative;
                    break;
                }
            }
            scores.push(score);
        }

        // Finite-sample (n + 1) correction for marginal coverage guarantee.
        let q_level = ((n_cal as f32 + 1.0) * (1.0 - config.alpha) / n_cal as f32).min(1.0);
        let threshold = empirical_quantile_aps(&mut scores, q_level);

        Ok(Self {
            threshold,
            n_cal,
            config,
        })
    }

    /// Predict the set of classes for a single test sample.
    ///
    /// Classes are added in descending probability order until the cumulative
    /// mass reaches `threshold`. The set is never empty (at minimum the argmax
    /// class is included).
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] when `probs.len() != n_classes`.
    pub fn predict_set(&self, probs: &[f32], n_classes: usize) -> TabularResult<Vec<usize>> {
        if probs.len() != n_classes {
            return Err(TabularError::DimensionMismatch {
                expected: n_classes,
                got: probs.len(),
            });
        }
        let sorted_idx = argsort_desc(probs);
        let mut cumulative = 0.0_f32;
        let mut set = Vec::new();
        for &c in &sorted_idx {
            set.push(c);
            cumulative += probs[c];
            if cumulative >= self.threshold {
                break;
            }
        }
        set.sort_unstable();
        Ok(set)
    }

    /// Fraction of test samples where the true label is in the prediction set.
    ///
    /// # Errors
    /// - [`TabularError::EmptyInput`] when `n_test == 0`.
    /// - [`TabularError::DimensionMismatch`] on shape mismatch.
    /// - [`TabularError::LabelOutOfRange`] on invalid labels.
    pub fn coverage_rate(
        &self,
        test_probs: &[f32],
        test_labels: &[usize],
        n_test: usize,
        n_classes: usize,
    ) -> TabularResult<f32> {
        if n_test == 0 {
            return Err(TabularError::EmptyInput);
        }
        let expected = n_test * n_classes;
        if test_probs.len() != expected {
            return Err(TabularError::DimensionMismatch {
                expected,
                got: test_probs.len(),
            });
        }
        if test_labels.len() != n_test {
            return Err(TabularError::DimensionMismatch {
                expected: n_test,
                got: test_labels.len(),
            });
        }
        for &label in test_labels {
            if label >= n_classes {
                return Err(TabularError::LabelOutOfRange { label, n_classes });
            }
        }

        let mut covered = 0usize;
        for i in 0..n_test {
            let row = &test_probs[i * n_classes..(i + 1) * n_classes];
            let set = self.predict_set(row, n_classes)?;
            if set.contains(&test_labels[i]) {
                covered += 1;
            }
        }
        Ok(covered as f32 / n_test as f32)
    }

    /// Average size of prediction sets on test data.
    ///
    /// # Errors
    /// - [`TabularError::EmptyInput`] when `n_test == 0`.
    /// - [`TabularError::DimensionMismatch`] on shape mismatch.
    pub fn average_set_size(
        &self,
        test_probs: &[f32],
        n_test: usize,
        n_classes: usize,
    ) -> TabularResult<f32> {
        if n_test == 0 {
            return Err(TabularError::EmptyInput);
        }
        let expected = n_test * n_classes;
        if test_probs.len() != expected {
            return Err(TabularError::DimensionMismatch {
                expected,
                got: test_probs.len(),
            });
        }
        let mut total_size = 0usize;
        for i in 0..n_test {
            let row = &test_probs[i * n_classes..(i + 1) * n_classes];
            let set = self.predict_set(row, n_classes)?;
            total_size += set.len();
        }
        Ok(total_size as f32 / n_test as f32)
    }

    /// The calibrated threshold.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Number of calibration samples.
    #[must_use]
    pub fn n_cal(&self) -> usize {
        self.n_cal
    }

    /// The configured miscoverage rate `alpha`.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.config.alpha
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Argsort descending: returns indices that sort `v` in decreasing order.
/// Ties are broken by smaller index first (deterministic).
pub(crate) fn argsort_desc(v: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| {
        v[b].partial_cmp(&v[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

/// Empirical quantile at level `q ∈ [0, 1]` of `scores` (mutates to sort).
///
/// Returns `1.0` on empty input (conservative fallback).
pub(crate) fn empirical_quantile_aps(scores: &mut [f32], q: f32) -> f32 {
    if scores.is_empty() {
        return 1.0;
    }
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = scores.len();
    let idx = ((q * n as f32).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    scores[idx]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Build a one-hot probabilities matrix where probs[true_label] = 1.0.
    fn one_hot_probs(n: usize, n_classes: usize) -> (Vec<f32>, Vec<usize>) {
        let mut probs = vec![0.0_f32; n * n_classes];
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let label = i % n_classes;
            labels.push(label);
            probs[i * n_classes + label] = 1.0;
        }
        (probs, labels)
    }

    // ── 1. Perfect predictor: calibration scores = 1.0, sets are singletons ─
    #[test]
    fn perfect_predictor_singletons() {
        let n_cal = 10;
        let n_classes = 3;
        let (probs, labels) = one_hot_probs(n_cal, n_classes);
        let cfg = ApsConformalConfig { alpha: 0.1 };
        let aps = ApsConformal::calibrate(&probs, &labels, n_cal, n_classes, cfg)
            .expect("calibrate should succeed");
        // All calibration scores = 1.0, threshold ≈ 1.0.
        assert!(
            aps.threshold() > 0.99,
            "threshold should be ≈ 1.0, got {}",
            aps.threshold()
        );
        // Prediction sets on perfect probs should be singletons.
        let test_row = vec![0.0_f32, 1.0, 0.0];
        let set = aps
            .predict_set(&test_row, n_classes)
            .expect("predict_set should succeed");
        assert_eq!(set.len(), 1, "perfect predictor → singleton set");
        assert_eq!(set[0], 1);
    }

    // ── 2. Coverage on calibration set ≥ 1 - alpha ──────────────────────────
    #[test]
    fn coverage_on_cal_set() {
        let n_cal = 100;
        let n_classes = 4;
        let mut probs = vec![0.0_f32; n_cal * n_classes];
        let mut labels = vec![0usize; n_cal];
        for i in 0..n_cal {
            let label = i % n_classes;
            labels[i] = label;
            for c in 0..n_classes {
                probs[i * n_classes + c] = if c == label { 0.7 } else { 0.1 };
            }
        }
        let cfg = ApsConformalConfig { alpha: 0.1 };
        let aps = ApsConformal::calibrate(&probs, &labels, n_cal, n_classes, cfg)
            .expect("calibrate should succeed");
        let cov = aps
            .coverage_rate(&probs, &labels, n_cal, n_classes)
            .expect("value should be present");
        assert!(
            cov >= 1.0 - cfg.alpha - 0.02,
            "coverage on cal set should be ≥ 1-alpha, got {cov}"
        );
    }

    // ── 3. Perfect predictor set always contains true class ──────────────────
    #[test]
    fn set_contains_true_class() {
        let n_cal = 12;
        let n_classes = 3;
        let (probs, labels) = one_hot_probs(n_cal, n_classes);
        let cfg = ApsConformalConfig { alpha: 0.1 };
        let aps = ApsConformal::calibrate(&probs, &labels, n_cal, n_classes, cfg)
            .expect("calibrate should succeed");
        for i in 0..n_cal {
            let row = &probs[i * n_classes..(i + 1) * n_classes];
            let set = aps
                .predict_set(row, n_classes)
                .expect("predict_set should succeed");
            assert!(
                set.contains(&labels[i]),
                "set must contain true class for perfect predictor"
            );
        }
    }

    // ── 4. Uniform probs → larger sets ──────────────────────────────────────
    #[test]
    fn uniform_probs_larger_sets() {
        let n_cal = 20;
        let n_classes = 5;
        let (probs, labels) = one_hot_probs(n_cal, n_classes);
        let cfg = ApsConformalConfig { alpha: 0.1 };
        // Calibrate on perfect predictor so threshold = 1.0.
        let aps = ApsConformal::calibrate(&probs, &labels, n_cal, n_classes, cfg)
            .expect("calibrate should succeed");
        // Uniform test row: equal probability for all classes.
        let uniform: Vec<f32> = vec![1.0 / n_classes as f32; n_classes];
        let set = aps
            .predict_set(&uniform, n_classes)
            .expect("predict_set should succeed");
        // With uniform probs and threshold = 1.0, all classes must be included.
        assert_eq!(
            set.len(),
            n_classes,
            "uniform probs should produce full set, got len={}",
            set.len()
        );
    }

    // ── 5. alpha = 0.0 → threshold very high → all classes included ──────────
    #[test]
    fn alpha_zero_all_classes() {
        let n_cal = 10;
        let n_classes = 4;
        let (probs, labels) = one_hot_probs(n_cal, n_classes);
        let cfg = ApsConformalConfig { alpha: 0.0 };
        // alpha = 0 means q_level = (n_cal+1)/n_cal * 1.0 which is clamped to 1.0,
        // giving threshold = the maximum score = 1.0.
        let aps = ApsConformal::calibrate(&probs, &labels, n_cal, n_classes, cfg)
            .expect("calibrate should succeed");
        // Any test row should include all classes when threshold = 1.0.
        let uniform: Vec<f32> = vec![1.0 / n_classes as f32; n_classes];
        let set = aps
            .predict_set(&uniform, n_classes)
            .expect("predict_set should succeed");
        assert_eq!(set.len(), n_classes, "alpha=0: all classes must be in set");
    }

    // ── 6. argsort_desc correctness ──────────────────────────────────────────
    #[test]
    fn sorted_classes_first() {
        let v = vec![0.1_f32, 0.5, 0.4];
        let idx = argsort_desc(&v);
        assert_eq!(idx, vec![1, 2, 0], "argsort_desc([0.1,0.5,0.4]) → [1,2,0]");
    }

    // ── 7. Threshold always in [0, 1] ────────────────────────────────────────
    #[test]
    fn threshold_in_zero_one() {
        let n_cal = 30;
        let n_classes = 3;
        let (probs, labels) = one_hot_probs(n_cal, n_classes);
        for alpha in [0.05_f32, 0.1, 0.2, 0.3, 0.5] {
            let cfg = ApsConformalConfig { alpha };
            let aps = ApsConformal::calibrate(&probs, &labels, n_cal, n_classes, cfg)
                .expect("calibrate should succeed");
            assert!(
                (0.0..=1.0).contains(&aps.threshold()),
                "threshold out of [0,1]: {}",
                aps.threshold()
            );
        }
    }

    // ── 8. average_set_size is finite and ≥ 1 ───────────────────────────────
    #[test]
    fn average_set_size_finite() {
        let n_cal = 20;
        let n_classes = 3;
        let (probs, labels) = one_hot_probs(n_cal, n_classes);
        let cfg = ApsConformalConfig { alpha: 0.1 };
        let aps = ApsConformal::calibrate(&probs, &labels, n_cal, n_classes, cfg)
            .expect("calibrate should succeed");
        let avg = aps
            .average_set_size(&probs, n_cal, n_classes)
            .expect("average_set_size should succeed");
        assert!(avg.is_finite() && avg >= 1.0, "avg set size must be ≥ 1.0");
    }

    // ── 9. Binary classification: set size ≤ 2 ──────────────────────────────
    #[test]
    fn n_classes_2_binary() {
        let n_cal = 20;
        let n_classes = 2;
        let (probs, labels) = one_hot_probs(n_cal, n_classes);
        let cfg = ApsConformalConfig { alpha: 0.1 };
        let aps = ApsConformal::calibrate(&probs, &labels, n_cal, n_classes, cfg)
            .expect("calibrate should succeed");
        let test_row = vec![0.6_f32, 0.4];
        let set = aps
            .predict_set(&test_row, n_classes)
            .expect("predict_set should succeed");
        assert!(set.len() <= 2, "binary: set must have ≤ 2 elements");
    }

    // ── 10. n_cal = 0 → EmptyInput error ────────────────────────────────────
    #[test]
    fn empty_cal_error() {
        let cfg = ApsConformalConfig { alpha: 0.1 };
        let result = ApsConformal::calibrate(&[], &[], 0, 3, cfg);
        assert!(
            matches!(result, Err(TabularError::EmptyInput)),
            "expected EmptyInput"
        );
    }

    // ── 11. Label out of range → error ───────────────────────────────────────
    #[test]
    fn label_out_of_range_error() {
        let n_classes = 3;
        let probs = vec![0.5_f32, 0.3, 0.2];
        let labels = vec![5usize]; // 5 >= 3
        let cfg = ApsConformalConfig { alpha: 0.1 };
        let result = ApsConformal::calibrate(&probs, &labels, 1, n_classes, cfg);
        assert!(
            matches!(result, Err(TabularError::LabelOutOfRange { .. })),
            "expected LabelOutOfRange"
        );
    }
}

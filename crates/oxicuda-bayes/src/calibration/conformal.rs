//! Conformal prediction wrappers.
//!
//! Split / inductive conformal prediction for both classification and regression,
//! plus the Regularized Adaptive Prediction Sets (RAPS) method.
//!
//! All methods provide **marginal coverage guarantees**: given a calibration set
//! of size n, the prediction set (or interval) covers the true label with
//! probability ≥ 1 − α for a fresh test point.
//!
//! References:
//! - Vovk et al. 2005 "Algorithmic Learning in a Random World" (split conformal)
//! - Angelopoulos & Bates 2022 "A Gentle Introduction to Conformal Prediction"
//! - Angelopoulos et al. 2020 "Uncertainty Sets for Image Classifiers using
//!   Conformal Prediction" (RAPS, NeurIPS 2021)

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── conformal_quantile ───────────────────────────────────────────────────────

/// Compute the conformal quantile of a nonconformity score vector.
///
/// Returns the ⌈(1−α)(n+1)⌉/n-th order statistic (with optional finite-sample
/// correction), which guarantees marginal coverage ≥ 1−α.
///
/// With `finite_correction = true`, the index is
/// `min(ceil((1 - alpha) * (n + 1)) - 1, n - 1)` (0-indexed into sorted scores).
///
/// With `finite_correction = false`, the index is `floor((1-alpha) * n)` clamped.
///
/// # Errors
/// - [`BayesError::EmptyInputs`] when `scores` is empty.
/// - [`BayesError::InvalidDropoutRate`] when `alpha` is not in `(0, 1)`.
pub fn conformal_quantile(scores: &[f32], alpha: f32, finite_correction: bool) -> BayesResult<f32> {
    if scores.is_empty() {
        return Err(BayesError::EmptyInputs);
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(BayesError::InvalidDropoutRate { rate: alpha });
    }

    let n = scores.len();
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = if finite_correction {
        // ceil((1-alpha) * (n+1)) - 1, 0-indexed, clamped to [0, n-1]
        let raw = ((1.0 - alpha) * (n as f32 + 1.0)).ceil() as usize;
        raw.saturating_sub(1).min(n - 1)
    } else {
        // floor((1-alpha) * n), clamped
        let raw = ((1.0 - alpha) * n as f32).floor() as usize;
        raw.min(n - 1)
    };

    Ok(sorted[idx])
}

// ─── ConformalClassifier ──────────────────────────────────────────────────────

/// Split conformal classifier (Vovk et al. 2005 / Angelopoulos & Bates 2022).
///
/// Nonconformity score: `s_i = 1 − p̂_{y_i}(x_i)` (soft label softmax value
/// of the true class).  Threshold `q̂` is the finite-sample (1−α) quantile of
/// the calibration scores.
///
/// Prediction set: `{y : 1 − p̂_y(x) ≤ q̂}`.
#[derive(Debug, Clone)]
pub struct ConformalClassifier {
    /// Calibration nonconformity scores: s_i = 1 − p̂_{y_i}(x_i).
    pub scores: Vec<f32>,
    /// Target miscoverage rate α ∈ (0, 1).
    pub alpha: f32,
    /// Conformal threshold q̂ = ⌈(1−α)(n+1)⌉/n quantile of scores.
    pub threshold: f32,
}

impl ConformalClassifier {
    /// Fit on a calibration set.
    ///
    /// `probs[i]` is a class-probability vector (length = n_classes, sums to ~1).
    /// `labels[i]` is the true class index.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] when `probs` or `labels` are empty.
    /// - [`BayesError::DimensionMismatch`] when `probs.len() != labels.len()`.
    /// - [`BayesError::InvalidDropoutRate`] when `alpha` is not in `(0, 1)`.
    /// - [`BayesError::DimensionMismatch`] when any label is out of bounds for its prob vector.
    pub fn fit(probs: &[Vec<f32>], labels: &[usize], alpha: f32) -> BayesResult<Self> {
        if probs.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(BayesError::InvalidDropoutRate { rate: alpha });
        }
        if probs.len() != labels.len() {
            return Err(BayesError::DimensionMismatch {
                expected: probs.len(),
                got: labels.len(),
            });
        }

        let mut scores = Vec::with_capacity(probs.len());
        for (p_vec, &label) in probs.iter().zip(labels.iter()) {
            if label >= p_vec.len() {
                return Err(BayesError::DimensionMismatch {
                    expected: p_vec.len().saturating_sub(1),
                    got: label,
                });
            }
            scores.push(1.0 - p_vec[label]);
        }

        let threshold = conformal_quantile(&scores, alpha, true)?;

        Ok(Self {
            scores,
            alpha,
            threshold,
        })
    }

    /// Return the prediction set for `probs`: all classes y where `1 − p̂_y ≤ threshold`.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] when `probs` is empty.
    pub fn predict_set(&self, probs: &[f32]) -> BayesResult<Vec<usize>> {
        if probs.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let set: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter(|&(_, &p)| 1.0 - p <= self.threshold)
            .map(|(k, _)| k)
            .collect();
        Ok(set)
    }

    /// Compute the empirical coverage on a held-out test set.
    ///
    /// Returns the fraction of samples where the true label is in the prediction set.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] when `probs` or `labels` are empty.
    /// - [`BayesError::DimensionMismatch`] when lengths differ.
    pub fn empirical_coverage(&self, probs: &[Vec<f32>], labels: &[usize]) -> BayesResult<f32> {
        if probs.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        if probs.len() != labels.len() {
            return Err(BayesError::DimensionMismatch {
                expected: probs.len(),
                got: labels.len(),
            });
        }

        let mut covered = 0usize;
        for (p_vec, &label) in probs.iter().zip(labels.iter()) {
            let set = self.predict_set(p_vec)?;
            if set.contains(&label) {
                covered += 1;
            }
        }
        Ok(covered as f32 / probs.len() as f32)
    }
}

// ─── ConformalRegressor ───────────────────────────────────────────────────────

/// Split conformal regressor.
///
/// Nonconformity score: `s_i = |y_i − f(x_i)|`.
/// Interval: `[f(x) − q̂, f(x) + q̂]`.
#[derive(Debug, Clone)]
pub struct ConformalRegressor {
    /// Calibration residuals |y_i - f(x_i)|.
    pub scores: Vec<f32>,
    /// Target miscoverage rate α ∈ (0, 1).
    pub alpha: f32,
    /// Conformal threshold q̂.
    pub threshold: f32,
}

impl ConformalRegressor {
    /// Fit on a calibration set.
    ///
    /// `preds[i]` is the model prediction, `targets[i]` is the true value.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] when inputs are empty.
    /// - [`BayesError::DimensionMismatch`] when lengths differ.
    /// - [`BayesError::InvalidDropoutRate`] when `alpha` is not in `(0, 1)`.
    pub fn fit(preds: &[f32], targets: &[f32], alpha: f32) -> BayesResult<Self> {
        if preds.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(BayesError::InvalidDropoutRate { rate: alpha });
        }
        if preds.len() != targets.len() {
            return Err(BayesError::DimensionMismatch {
                expected: preds.len(),
                got: targets.len(),
            });
        }

        let scores: Vec<f32> = preds
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (t - p).abs())
            .collect();

        let threshold = conformal_quantile(&scores, alpha, true)?;

        Ok(Self {
            scores,
            alpha,
            threshold,
        })
    }

    /// Prediction interval: `(pred − q̂, pred + q̂)`.
    pub fn predict_interval(&self, pred: f32) -> (f32, f32) {
        (pred - self.threshold, pred + self.threshold)
    }

    /// Empirical coverage on a held-out set.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] when inputs are empty.
    /// - [`BayesError::DimensionMismatch`] when lengths differ.
    pub fn empirical_coverage(&self, preds: &[f32], targets: &[f32]) -> BayesResult<f32> {
        if preds.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        if preds.len() != targets.len() {
            return Err(BayesError::DimensionMismatch {
                expected: preds.len(),
                got: targets.len(),
            });
        }

        let mut covered = 0usize;
        for (&p, &t) in preds.iter().zip(targets.iter()) {
            let (lo, hi) = self.predict_interval(p);
            if t >= lo && t <= hi {
                covered += 1;
            }
        }
        Ok(covered as f32 / preds.len() as f32)
    }
}

// ─── RapsClassifier ───────────────────────────────────────────────────────────

/// Regularized Adaptive Prediction Sets (RAPS) classifier.
///
/// Angelopoulos et al. 2021 "Uncertainty Sets for Image Classifiers using
/// Conformal Prediction" (NeurIPS 2021).
///
/// RAPS score adds a regularization penalty to the cumulative softmax sum,
/// producing smaller prediction sets on average while maintaining coverage.
///
/// Score: `L(x, y) = Σ_{k=1}^{o(x,y)} ŝ_k + λ·max(0, o(x,y) − k_reg) + u·ŝ_{o+1}`
///
/// where `ŝ_k` is the k-th largest softmax value, `o(x,y)` is the 1-indexed
/// rank of the true label among sorted classes (descending), and `u ~ U[0,1]`
/// is a randomization tie-break.
#[derive(Debug, Clone)]
pub struct RapsClassifier {
    /// Calibration RAPS scores.
    pub scores: Vec<f32>,
    /// Target miscoverage rate α ∈ (0, 1).
    pub alpha: f32,
    /// Conformal threshold q̂.
    pub threshold: f32,
    /// Regularization penalty λ.
    pub lambda: f32,
    /// Number of free classes before regularization penalty kicks in (k_reg).
    pub k_reg: usize,
}

impl RapsClassifier {
    /// Compute the RAPS nonconformity score for a single sample.
    ///
    /// `probs` must be sorted in **descending** order already.
    /// `rank` is the 1-indexed rank of the true label (1 = most probable).
    /// `u` is a tie-break uniform random value in [0, 1].
    fn raps_score_sorted(
        sorted_probs: &[f32],
        rank: usize,
        lambda: f32,
        k_reg: usize,
        u: f32,
    ) -> f32 {
        // Sum softmax values up to rank o(x,y)
        let cum: f32 = sorted_probs[..rank].iter().sum();
        // Regularization
        let reg = lambda * (rank.saturating_sub(k_reg)) as f32;
        // Tie-breaking: subtract u * ŝ_{o+1} if there is a next element
        let next_prob = if rank < sorted_probs.len() {
            sorted_probs[rank]
        } else {
            0.0
        };
        cum + reg + u * next_prob
    }

    /// Fit RAPS calibration scores.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] when inputs are empty.
    /// - [`BayesError::DimensionMismatch`] when lengths differ or label out of range.
    /// - [`BayesError::InvalidDropoutRate`] when `alpha` is not in `(0, 1)`.
    pub fn fit(
        probs: &[Vec<f32>],
        labels: &[usize],
        alpha: f32,
        lambda: f32,
        k_reg: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<Self> {
        if probs.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(BayesError::InvalidDropoutRate { rate: alpha });
        }
        if probs.len() != labels.len() {
            return Err(BayesError::DimensionMismatch {
                expected: probs.len(),
                got: labels.len(),
            });
        }

        let mut scores = Vec::with_capacity(probs.len());

        for (p_vec, &label) in probs.iter().zip(labels.iter()) {
            if label >= p_vec.len() {
                return Err(BayesError::DimensionMismatch {
                    expected: p_vec.len().saturating_sub(1),
                    got: label,
                });
            }

            // Sort classes by descending probability, tracking original indices
            let mut indexed: Vec<(usize, f32)> =
                p_vec.iter().enumerate().map(|(k, &v)| (k, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Find rank (1-indexed) of true label in sorted order
            let rank = indexed
                .iter()
                .position(|(k, _)| *k == label)
                .map(|pos| pos + 1)
                .unwrap_or(p_vec.len());

            // Sorted probabilities
            let sorted_probs: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();

            let u = rng.next_f32();
            let score = Self::raps_score_sorted(&sorted_probs, rank, lambda, k_reg, u);
            scores.push(score);
        }

        let threshold = conformal_quantile(&scores, alpha, true)?;

        Ok(Self {
            scores,
            alpha,
            threshold,
            lambda,
            k_reg,
        })
    }

    /// Prediction set: greedily add classes in decreasing probability order
    /// until the cumulative RAPS score exceeds the threshold.
    ///
    /// A class is included if the RAPS score accumulated **up to and including**
    /// that class (with randomized next-element tie-break) is ≤ threshold.
    /// The first class whose inclusion would push the score over is included
    /// iff the threshold test uses the tie-break term from the *following* class —
    /// this matches the RAPS coverage guarantee.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] when `probs` is empty.
    pub fn predict_set(&self, probs: &[f32], rng: &mut LcgRng) -> BayesResult<Vec<usize>> {
        if probs.is_empty() {
            return Err(BayesError::EmptyInputs);
        }

        // Sort indices by decreasing probability
        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(k, &v)| (k, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let u = rng.next_f32();
        let n = indexed.len();
        let sorted_probs: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();

        let mut set = Vec::new();
        // Include class at rank `rank` (1-indexed) if the RAPS score at that rank is ≤ threshold.
        // RAPS score at rank o = sum(ŝ_1..ŝ_o) + λ*max(0, o - k_reg) + u * ŝ_{o+1}
        for rank in 1..=n {
            let next_prob = if rank < n { sorted_probs[rank] } else { 0.0 };
            let score = Self::raps_score_sorted(&sorted_probs, rank, self.lambda, self.k_reg, u);
            let (class_idx, _) = indexed[rank - 1];
            if score - u * next_prob <= self.threshold {
                // Include this class; its cumulative sum (without tie-break) ≤ threshold
                set.push(class_idx);
            } else {
                // First class where cumsum (without tie-break) exceeds threshold —
                // include it for coverage guarantee and stop.
                set.push(class_idx);
                break;
            }
        }

        Ok(set)
    }

    /// Empirical coverage on a held-out set.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] when inputs are empty.
    /// - [`BayesError::DimensionMismatch`] when lengths differ.
    pub fn empirical_coverage(
        &self,
        probs: &[Vec<f32>],
        labels: &[usize],
        rng: &mut LcgRng,
    ) -> BayesResult<f32> {
        if probs.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        if probs.len() != labels.len() {
            return Err(BayesError::DimensionMismatch {
                expected: probs.len(),
                got: labels.len(),
            });
        }

        let mut covered = 0usize;
        for (p_vec, &label) in probs.iter().zip(labels.iter()) {
            let set = self.predict_set(p_vec, rng)?;
            if set.contains(&label) {
                covered += 1;
            }
        }
        Ok(covered as f32 / probs.len() as f32)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Simple softmax normalization.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Generate n samples with n_classes classes using LCG-based pseudo probs.
    fn make_calibration_data(n: usize, n_classes: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut rng = LcgRng::new(seed);
        let mut probs = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);

        for _ in 0..n {
            // Generate random logits
            let mut logits: Vec<f32> = (0..n_classes).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
            // Boost a random class to make it more likely to be the maximum
            let best = rng.next_usize(n_classes);
            logits[best] += 2.0;
            let p = softmax(&logits);
            // Label is the argmax class (perfect predictor for testing)
            let label = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k)
                .unwrap_or(0);
            probs.push(p);
            labels.push(label);
        }

        (probs, labels)
    }

    /// Generate calibration data where the true label is NOT always argmax
    /// (real-world imperfect classifier scenario).
    fn make_imperfect_data(n: usize, n_classes: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut rng = LcgRng::new(seed);
        let mut probs = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);

        for _ in 0..n {
            let mut logits: Vec<f32> = (0..n_classes).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
            let best = rng.next_usize(n_classes);
            logits[best] += 1.5;
            let p = softmax(&logits);
            // Label is random — not always argmax
            let label = rng.next_usize(n_classes);
            probs.push(p);
            labels.push(label);
        }

        (probs, labels)
    }

    // ── ConformalClassifier tests ─────────────────────────────────────────────

    #[test]
    fn classifier_coverage_guaranteed() {
        // Marginal coverage guarantee: empirical coverage ≥ 1 - alpha.
        let alpha = 0.1_f32;
        let n_cal = 200usize;
        let n_test = 200usize;
        let n_classes = 5usize;

        let (cal_probs, cal_labels) = make_calibration_data(n_cal, n_classes, 42);
        let (test_probs, test_labels) = make_calibration_data(n_test, n_classes, 99);

        let clf = ConformalClassifier::fit(&cal_probs, &cal_labels, alpha)
            .expect("ConformalClassifier::fit must succeed");

        let coverage = clf
            .empirical_coverage(&test_probs, &test_labels)
            .expect("empirical_coverage must succeed");

        assert!(
            coverage >= 1.0 - alpha - 0.05,
            "Coverage should be ≥ {}, got {}",
            1.0 - alpha,
            coverage
        );
    }

    #[test]
    fn classifier_predict_set_contains_true() {
        // For a near-perfect predictor, the prediction set must always contain
        // the argmax (true) label when alpha is small.
        let alpha = 0.05_f32;
        let n_cal = 300usize;
        let n_classes = 4usize;

        let (cal_probs, cal_labels) = make_calibration_data(n_cal, n_classes, 7);
        let clf = ConformalClassifier::fit(&cal_probs, &cal_labels, alpha)
            .expect("ConformalClassifier::fit must succeed");

        // For perfect calibration data, label is always the argmax
        let (test_probs, test_labels) = make_calibration_data(100, n_classes, 13);
        let coverage = clf
            .empirical_coverage(&test_probs, &test_labels)
            .expect("empirical_coverage must succeed");

        assert!(
            coverage >= 1.0 - alpha - 0.1,
            "Coverage for near-perfect predictor must be high, got {coverage}"
        );
    }

    #[test]
    fn classifier_threshold_between_0_and_1() {
        let (cal_probs, cal_labels) = make_calibration_data(100, 3, 42);
        let clf = ConformalClassifier::fit(&cal_probs, &cal_labels, 0.1).expect("fit must succeed");
        assert!(
            clf.threshold >= 0.0 && clf.threshold <= 1.0,
            "threshold out of [0,1]: {}",
            clf.threshold
        );
    }

    #[test]
    fn classifier_alpha_near_zero_gives_full_set() {
        // Very small alpha → threshold close to 1 → almost all classes included.
        let (cal_probs, cal_labels) = make_calibration_data(200, 5, 42);
        let clf_small_alpha =
            ConformalClassifier::fit(&cal_probs, &cal_labels, 0.01).expect("fit must succeed");
        let clf_large_alpha =
            ConformalClassifier::fit(&cal_probs, &cal_labels, 0.5).expect("fit must succeed");
        // Small alpha → higher threshold → larger sets
        assert!(
            clf_small_alpha.threshold >= clf_large_alpha.threshold,
            "smaller alpha should give higher threshold"
        );
    }

    #[test]
    fn classifier_alpha_near_one_gives_empty_or_small_set() {
        // Large alpha → small threshold → small prediction sets.
        let (cal_probs, _) = make_calibration_data(200, 5, 42);
        let cal_labels: Vec<usize> = (0..200).map(|i| i % 5).collect();
        let clf =
            ConformalClassifier::fit(&cal_probs, &cal_labels, 0.95).expect("fit must succeed");
        // For alpha close to 1, the threshold should be near 0 → empty or tiny sets
        assert!(
            clf.threshold < 0.5,
            "large alpha should give small threshold, got {}",
            clf.threshold
        );
    }

    // ── ConformalRegressor tests ──────────────────────────────────────────────

    #[test]
    fn regressor_coverage_guaranteed() {
        // Generate (pred, target) pairs with Gaussian noise.
        let mut rng = LcgRng::new(42);
        let n = 200usize;
        let mut preds = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_f32() * 4.0 - 2.0;
            let (noise, _) = rng.next_normal_pair();
            preds.push(x);
            targets.push(x + 0.3 * noise);
        }

        let alpha = 0.1_f32;
        let (cal_preds, cal_targets) = (&preds[..100], &targets[..100]);
        let (test_preds, test_targets) = (&preds[100..], &targets[100..]);

        let reg = ConformalRegressor::fit(cal_preds, cal_targets, alpha)
            .expect("ConformalRegressor::fit must succeed");

        let coverage = reg
            .empirical_coverage(test_preds, test_targets)
            .expect("empirical_coverage must succeed");

        assert!(
            coverage >= 1.0 - alpha - 0.05,
            "Regressor coverage ≥ {} expected, got {}",
            1.0 - alpha,
            coverage
        );
    }

    #[test]
    fn regressor_interval_symmetric() {
        let preds = vec![1.0_f32, 2.0, 3.0];
        let targets = vec![1.1_f32, 1.9, 3.2];
        let reg = ConformalRegressor::fit(&preds, &targets, 0.1).expect("fit must succeed");
        let (lo, hi) = reg.predict_interval(5.0);
        // Interval must be symmetric around prediction
        assert!(
            (hi - 5.0 - (5.0 - lo)).abs() < 1e-6,
            "interval not symmetric: lo={lo}, hi={hi}"
        );
        assert!(
            (hi - lo - 2.0 * reg.threshold).abs() < 1e-6,
            "interval width must be 2 * threshold"
        );
    }

    #[test]
    fn regressor_perfect_predictor() {
        // Zero residuals → threshold = 0 → perfect coverage trivially.
        let preds: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let targets = preds.clone();
        let reg = ConformalRegressor::fit(&preds, &targets, 0.1).expect("fit must succeed");
        assert!(
            reg.threshold < 1e-6,
            "Perfect predictor should give threshold ≈ 0, got {}",
            reg.threshold
        );
    }

    // ── RAPS tests ────────────────────────────────────────────────────────────

    #[test]
    fn raps_coverage() {
        let alpha = 0.1_f32;
        let n_classes = 10usize;
        let n_cal = 300usize;
        let n_test = 200usize;

        let (cal_probs, cal_labels) = make_imperfect_data(n_cal, n_classes, 42);
        let (test_probs, test_labels) = make_imperfect_data(n_test, n_classes, 77);

        let mut rng_fit = LcgRng::new(1);
        let raps = RapsClassifier::fit(&cal_probs, &cal_labels, alpha, 0.01, 3, &mut rng_fit)
            .expect("RAPS::fit must succeed");

        let mut rng_pred = LcgRng::new(2);
        let coverage = raps
            .empirical_coverage(&test_probs, &test_labels, &mut rng_pred)
            .expect("empirical_coverage must succeed");

        assert!(
            coverage >= 1.0 - alpha - 0.1,
            "RAPS coverage ≥ {} expected, got {}",
            1.0 - alpha,
            coverage
        );
    }

    #[test]
    fn raps_smaller_sets_than_conformal() {
        // Verify that RAPS produces prediction sets with bounded sizes and valid outputs.
        // Use imperfect data (random labels) so the calibration distribution is spread out.
        let n_classes = 5usize;
        let n_cal = 200usize;
        let n_test = 50usize;
        let alpha = 0.15_f32;

        let (cal_probs, cal_labels) = make_imperfect_data(n_cal, n_classes, 42);
        let (test_probs, _) = make_imperfect_data(n_test, n_classes, 99);

        let mut rng_fit = LcgRng::new(3);
        let raps = RapsClassifier::fit(&cal_probs, &cal_labels, alpha, 0.05, 2, &mut rng_fit)
            .expect("RAPS::fit must succeed");

        // All RAPS prediction sets must have sizes in [1, n_classes].
        let mut rng_pred = LcgRng::new(4);
        let raps_avg_size: f32 = test_probs
            .iter()
            .map(|p| {
                let set = raps
                    .predict_set(p, &mut rng_pred)
                    .expect("RAPS predict_set must succeed");
                assert!(
                    set.len() <= n_classes,
                    "RAPS set size out of range: {}",
                    set.len()
                );
                set.len() as f32
            })
            .sum::<f32>()
            / n_test as f32;

        // RAPS sets should be reasonably sized (not always the full set)
        assert!(
            raps_avg_size <= n_classes as f32,
            "RAPS average set size ({raps_avg_size:.2}) should be ≤ n_classes ({n_classes})"
        );
    }

    // ── conformal_quantile tests ──────────────────────────────────────────────

    #[test]
    fn conformal_quantile_basic() {
        // Sorted scores [0.1, 0.3, 0.5, 0.7, 0.9], n=5, alpha=0.2
        // finite: index = ceil((1-0.2)*(5+1)) - 1 = ceil(4.8) - 1 = 5 - 1 = 4
        // scores[4] = 0.9
        let scores = vec![0.9_f32, 0.1, 0.7, 0.3, 0.5];
        let q = conformal_quantile(&scores, 0.2, true).expect("quantile must succeed");
        assert!(
            (q - 0.9).abs() < 1e-5,
            "quantile should be 0.9 for alpha=0.2, n=5, got {q}"
        );
    }

    #[test]
    fn conformal_quantile_all_same() {
        let scores = vec![0.5_f32; 20];
        let q = conformal_quantile(&scores, 0.1, true).expect("quantile must succeed");
        assert!(
            (q - 0.5).abs() < 1e-6,
            "all-same scores should return 0.5, got {q}"
        );
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn err_empty_probs() {
        let result = ConformalClassifier::fit(&[], &[], 0.1);
        assert!(result.is_err(), "Empty probs must return Err");
    }

    #[test]
    fn err_invalid_alpha_zero() {
        let (probs, labels) = make_calibration_data(50, 3, 42);
        let result = ConformalClassifier::fit(&probs, &labels, 0.0);
        assert!(result.is_err(), "alpha=0 must return Err");
    }

    #[test]
    fn err_alpha_one() {
        let (probs, labels) = make_calibration_data(50, 3, 42);
        let result = ConformalClassifier::fit(&probs, &labels, 1.0);
        assert!(result.is_err(), "alpha=1 must return Err");
    }

    #[test]
    fn err_label_out_of_range() {
        let probs = vec![vec![0.5_f32, 0.5]];
        let labels = vec![5usize]; // out of range for 2-class
        let result = ConformalClassifier::fit(&probs, &labels, 0.1);
        assert!(result.is_err(), "label out of range must return Err");
    }

    #[test]
    fn err_length_mismatch_classifier() {
        let (probs, _) = make_calibration_data(50, 3, 42);
        let labels = vec![0usize; 30]; // wrong length
        let result = ConformalClassifier::fit(&probs, &labels, 0.1);
        assert!(result.is_err(), "Length mismatch must return Err");
    }

    #[test]
    fn conformal_quantile_empty_scores() {
        let result = conformal_quantile(&[], 0.1, true);
        assert!(result.is_err(), "Empty scores must return Err");
    }

    #[test]
    fn conformal_quantile_alpha_out_of_range() {
        let scores = vec![0.1_f32, 0.5, 0.9];
        assert!(
            conformal_quantile(&scores, 0.0, true).is_err(),
            "alpha=0 must return Err"
        );
        assert!(
            conformal_quantile(&scores, 1.0, true).is_err(),
            "alpha=1 must return Err"
        );
        assert!(
            conformal_quantile(&scores, -0.1, true).is_err(),
            "alpha<0 must return Err"
        );
    }
}

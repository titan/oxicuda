//! ECOD — Empirical-CDF-based Outlier Detection (Li et al. 2022, arXiv:2201.00382).
//!
//! Uses per-feature empirical CDFs with skewness-aware tail weighting.
//!
//! **Fit phase:**
//! For each feature `j`, sort the training values (ECDF lookup) and compute
//! the Fisher–Pearson skewness `skew_j = (1/n) Σ_i (x_{ij} − μ_j)³ / σ_j³`.
//!
//! **Score phase (per feature `j`):**
//! * Left-tail:  `p_L = F̂_j(x_j)` — fraction of training values ≤ `x_j`
//! * Right-tail: `p_R = 1 − F̂_j(x_j)` — fraction of training values > `x_j`
//! * Weights driven by skewness:
//!   `w_L = 0.5 + 0.5 · tanh(skew_j)`,  `w_R = 1 − w_L`
//! * Per-feature score: `o_j(x) = w_L · (−ln(p_L + ε)) + w_R · (−ln(p_R + ε))`
//! * Final ECOD score:  `(1/d) Σ_j o_j(x)` — mean over all `d` features
//!
//! For symmetric distributions (`skew ≈ 0`) the weights are both `≈ 0.5`,
//! making ECOD equivalent to two-tailed COPOD per feature.
//! ε = 1e-10 to avoid `log(0)`.

use crate::error::{AnomalyError, AnomalyResult};

const LOG_EPS: f32 = 1e-10;

// ─── Ecod ─────────────────────────────────────────────────────────────────────

/// ECOD anomaly detector.
///
/// Non-parametric, no hyperparameters required. Fit once then score any
/// point using empirical marginal CDFs with skewness-aware tail weights.
pub struct Ecod {
    /// `sorted_features[j]` = sorted training values for feature `j`.
    sorted_features: Vec<Vec<f32>>,
    /// Per-feature Fisher–Pearson skewness computed during fit.
    skewness: Vec<f32>,
    n_train: usize,
    n_features: usize,
}

impl Ecod {
    /// Create an unfitted ECOD detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sorted_features: Vec::new(),
            skewness: Vec::new(),
            n_train: 0,
            n_features: 0,
        }
    }

    /// Fit on row-major `data` with shape `[n_samples × n_features]`.
    ///
    /// Stores a sorted copy of each feature column and pre-computes
    /// the Fisher–Pearson skewness for skewness-aware tail weighting.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }

        self.n_train = n_samples;
        self.n_features = n_features;
        self.sorted_features = Vec::with_capacity(n_features);
        self.skewness = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect feature column j.
            let mut col: Vec<f32> = (0..n_samples).map(|i| data[i * n_features + j]).collect();
            col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Fisher–Pearson skewness on the sorted (same values) column.
            let n = n_samples as f32;
            let mean = col.iter().sum::<f32>() / n;
            let var = col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let std_dev = var.sqrt();
            let skew = if std_dev > 1e-8_f32 {
                col.iter()
                    .map(|&v| {
                        let z = (v - mean) / std_dev;
                        z * z * z
                    })
                    .sum::<f32>()
                    / n
            } else {
                0.0_f32
            };

            self.skewness.push(skew);
            self.sorted_features.push(col);
        }

        Ok(())
    }

    /// Empirical CDF for feature `feat` at value `v`.
    ///
    /// Returns the fraction of training samples ≤ `v` (binary search, O(log n)).
    fn ecdf(&self, feat: usize, v: f32) -> f32 {
        let col = &self.sorted_features[feat];
        let count = col.partition_point(|&x| x <= v);
        count as f32 / self.n_train as f32
    }

    /// ECOD anomaly score for a single sample `x`.
    ///
    /// Returns the mean over features of the skewness-weighted negative
    /// log-probability in the more anomalous tail.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        if self.n_features == 0 {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }

        let mut acc = 0.0_f32;
        for (j, &xi) in x.iter().enumerate().take(self.n_features) {
            let p_left = self.ecdf(j, xi);
            let p_right = 1.0_f32 - p_left;

            // Skewness-aware weights: tanh maps skew ∈ ℝ → (−1, 1).
            // w_left ∈ (0, 1); w_right = 1 − w_left.
            let skew = self.skewness[j];
            let w_left = 0.5_f32 + 0.5_f32 * skew.tanh();
            let w_right = 1.0_f32 - w_left;

            let score_left = -(p_left + LOG_EPS).ln();
            let score_right = -(p_right + LOG_EPS).ln();

            acc += w_left * score_left + w_right * score_right;
        }

        Ok(acc / self.n_features as f32)
    }

    /// Batch scoring for `n` samples stored row-major in `x` (`n × n_features`).
    ///
    /// Returns a `Vec<f32>` of length `n`.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if self.n_features == 0 {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != n * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.n_features,
                got: x.len(),
            });
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let sample = &x[i * self.n_features..(i + 1) * self.n_features];
            scores.push(self.score(sample)?);
        }
        Ok(scores)
    }

    /// Return the per-feature skewness values computed during fit.
    #[must_use]
    pub fn skewness(&self) -> &[f32] {
        &self.skewness
    }

    /// Number of features the detector was fitted on (0 if not yet fitted).
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of training samples (0 if not yet fitted).
    #[must_use]
    pub fn n_train(&self) -> usize {
        self.n_train
    }
}

impl Default for Ecod {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform 1-D training set: 0.0, 1.0, …, 19.0.
    fn make_1d(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    /// 2-D row-major data: feature 0 = i, feature 1 = 2i, for i in 0..n.
    fn make_2d(n: usize) -> Vec<f32> {
        (0..n).flat_map(|i| [i as f32, (i * 2) as f32]).collect()
    }

    // 1. Basic fit + score returns a finite non-negative value.
    #[test]
    fn ecod_1d_fit_score_finite() {
        let data = make_1d(20);
        let mut det = Ecod::new();
        det.fit(&data, 20, 1).unwrap();
        let s = det.score(&[10.0_f32]).unwrap();
        assert!(s.is_finite() && s >= 0.0, "s={s}");
    }

    // 2. Scoring before fit returns NotFitted.
    #[test]
    fn ecod_not_fitted_error() {
        let det = Ecod::new();
        let err = det.score(&[1.0_f32]).unwrap_err();
        assert!(
            matches!(err, AnomalyError::NotFitted),
            "expected NotFitted, got {err:?}"
        );
    }

    // 3. Extreme outlier scores higher than inlier.
    #[test]
    fn ecod_outlier_higher_than_inlier() {
        let data = make_1d(40);
        let mut det = Ecod::new();
        det.fit(&data, 40, 1).unwrap();
        let s_normal = det.score(&[20.0_f32]).unwrap();
        let s_outlier = det.score(&[1000.0_f32]).unwrap();
        assert!(
            s_outlier > s_normal,
            "outlier={s_outlier} should > inlier={s_normal}"
        );
    }

    // 4. score_batch returns a Vec with the correct length.
    #[test]
    fn ecod_batch_correct_length() {
        let data = make_2d(20);
        let mut det = Ecod::new();
        det.fit(&data, 20, 2).unwrap();
        let queries = make_2d(5);
        let scores = det.score_batch(&queries, 5).unwrap();
        assert_eq!(scores.len(), 5);
        assert!(scores.iter().all(|s| s.is_finite()), "all scores finite");
    }

    // 5. Empty input at fit returns EmptyInput.
    #[test]
    fn ecod_empty_input_error() {
        let mut det = Ecod::new();
        let err = det.fit(&[], 0, 3).unwrap_err();
        assert!(matches!(err, AnomalyError::EmptyInput), "got {err:?}");
    }

    // 6. Dimension mismatch at fit returns DimensionMismatch.
    #[test]
    fn ecod_fit_dimension_mismatch() {
        let mut det = Ecod::new();
        // 5 rows × 2 features = 10 elements, but we pass 9.
        let err = det.fit(&[0.0_f32; 9], 5, 2).unwrap_err();
        assert!(
            matches!(err, AnomalyError::DimensionMismatch { .. }),
            "got {err:?}"
        );
    }

    // 7. FeatureCountMismatch when score receives wrong-length slice.
    #[test]
    fn ecod_feature_count_mismatch_at_score() {
        let data = make_2d(20);
        let mut det = Ecod::new();
        det.fit(&data, 20, 2).unwrap();
        // Pass a 3-element slice to a 2-feature model.
        let err = det.score(&[1.0_f32, 2.0, 3.0]).unwrap_err();
        assert!(
            matches!(
                err,
                AnomalyError::FeatureCountMismatch {
                    expected: 2,
                    got: 3
                }
            ),
            "got {err:?}"
        );
    }

    // 8. Symmetric feature has left/right weights both ≈ 0.5 (skew ≈ 0 ⇒ tanh(0)=0).
    #[test]
    fn ecod_symmetric_skew_weights_balanced() {
        // Perfectly symmetric data: -n, -(n-1), …, 0, …, (n-1), n
        let n = 21_usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32 - 10.0_f32).collect();
        let mut det = Ecod::new();
        det.fit(&data, n, 1).unwrap();

        let skew = det.skewness()[0];
        let w_left = 0.5_f32 + 0.5_f32 * skew.tanh();
        let w_right = 1.0_f32 - w_left;

        assert!(
            (w_left - 0.5_f32).abs() < 0.05_f32,
            "w_left should be ≈0.5, got {w_left}"
        );
        assert!(
            (w_right - 0.5_f32).abs() < 0.05_f32,
            "w_right should be ≈0.5, got {w_right}"
        );
    }

    // 9. Multi-feature data: all batch scores are finite and positive.
    #[test]
    fn ecod_multi_feature_batch_finite() {
        let data = make_2d(30);
        let mut det = Ecod::new();
        det.fit(&data, 30, 2).unwrap();

        // Five diverse query points.
        #[rustfmt::skip]
        let queries: Vec<f32> = vec![
             5.0,  10.0,   // inlier
            29.0,  58.0,   // boundary
            -1.0,  -2.0,   // below min
           500.0, 1000.0,  // extreme outlier
            15.0,  30.0,   // mid-range
        ];
        let scores = det.score_batch(&queries, 5).unwrap();
        assert_eq!(scores.len(), 5);
        for (idx, &s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{idx}] is not finite: {s}");
            assert!(s >= 0.0, "score[{idx}] should be non-negative, got {s}");
        }
        // Extreme outlier (idx 3) should beat mid-range (idx 4).
        assert!(
            scores[3] > scores[4],
            "outlier score {} should exceed mid-range {}",
            scores[3],
            scores[4]
        );
    }
}

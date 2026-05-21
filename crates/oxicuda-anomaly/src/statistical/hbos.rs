//! Histogram-Based Outlier Score (Goldstein & Dengel 2012).
//!
//! ```text
//! HBOS(x) = Σ_j  -log( density_j(x_j) + ε )
//! ```
//!
//! where `density_j(v)` is the empirical probability density of feature `j`
//! evaluated at value `v`, estimated via an equi-width histogram built during
//! `fit`.  Higher score → more anomalous.
//!
//! **Fit** (per feature `j`):
//! 1. Compute `[min_j, max_j]` over training data.
//! 2. Divide the range into `n_bins` equi-width bins of width
//!    `w_j = (max_j − min_j) / n_bins`.
//! 3. Count samples per bin, then normalize:
//!    `density[bin] = count / (n_samples · w_j)`.
//! 4. If `max_j == min_j` (constant feature) the single bin has density `1`.
//!
//! **Score**:
//! `HBOS(x) = Σ_j  -log( density_j(x_j) + ε )`,  `ε = 1e-10`.

use crate::error::{AnomalyError, AnomalyResult};

// ─── Constants ────────────────────────────────────────────────────────────────

const EPSILON: f32 = 1e-10;

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for [`Hbos`].
#[derive(Debug, Clone)]
pub struct HbosConfig {
    /// Number of equi-width bins per feature histogram (default `10`).
    pub n_bins: usize,
}

impl Default for HbosConfig {
    fn default() -> Self {
        Self { n_bins: 10 }
    }
}

// ─── Hbos ─────────────────────────────────────────────────────────────────────

/// Histogram-Based Outlier Score detector.
///
/// Call [`Hbos::new`] (or `Hbos::default()`), then [`Hbos::fit`], then
/// [`Hbos::score`] or [`Hbos::score_batch`].
pub struct Hbos {
    config: HbosConfig,
    /// Minimum value per feature (`[n_features]`).
    min_vals: Vec<f32>,
    /// Maximum value per feature (`[n_features]`).
    max_vals: Vec<f32>,
    /// Bin width per feature (`[n_features]`).
    bin_widths: Vec<f32>,
    /// `densities[j][bin]` = probability density of `bin` in feature `j`.
    densities: Vec<Vec<f32>>,
    n_features: usize,
    n_samples: usize,
}

impl Default for Hbos {
    fn default() -> Self {
        Self::new(HbosConfig::default())
    }
}

impl Hbos {
    /// Create an unfitted HBOS detector from `config`.
    #[must_use]
    pub fn new(config: HbosConfig) -> Self {
        Self {
            config,
            min_vals: Vec::new(),
            max_vals: Vec::new(),
            bin_widths: Vec::new(),
            densities: Vec::new(),
            n_features: 0,
            n_samples: 0,
        }
    }

    // ── Fit ───────────────────────────────────────────────────────────────────

    /// Build per-feature histograms from `data` (row-major, shape `[n_samples × n_features]`).
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
        if self.config.n_bins == 0 {
            return Err(AnomalyError::Internal {
                msg: "n_bins must be >= 1".into(),
            });
        }

        let n_bins = self.config.n_bins;

        // ── Per-feature min / max ─────────────────────────────────────────────
        let mut min_vals = vec![f32::INFINITY; n_features];
        let mut max_vals = vec![f32::NEG_INFINITY; n_features];

        for s in 0..n_samples {
            for j in 0..n_features {
                let v = data[s * n_features + j];
                if v < min_vals[j] {
                    min_vals[j] = v;
                }
                if v > max_vals[j] {
                    max_vals[j] = v;
                }
            }
        }

        // ── Build histograms ──────────────────────────────────────────────────
        let mut bin_widths = vec![0.0_f32; n_features];
        let mut densities: Vec<Vec<f32>> = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let range = max_vals[j] - min_vals[j];

            if range <= 0.0 {
                // Constant feature: single bin with density 1.
                bin_widths[j] = 1.0; // sentinel; unused in lookup
                densities.push(vec![1.0_f32]);
                continue;
            }

            let w = range / n_bins as f32;
            bin_widths[j] = w;

            let mut counts = vec![0_u64; n_bins];
            for s in 0..n_samples {
                let v = data[s * n_features + j];
                let bin = Self::bin_index(v, min_vals[j], w, n_bins);
                counts[bin] += 1;
            }

            // Normalize to density: count / (n_samples * w)
            let denom = n_samples as f32 * w;
            let dens: Vec<f32> = counts.iter().map(|&c| c as f32 / denom).collect();
            densities.push(dens);
        }

        self.min_vals = min_vals;
        self.max_vals = max_vals;
        self.bin_widths = bin_widths;
        self.densities = densities;
        self.n_features = n_features;
        self.n_samples = n_samples;

        Ok(())
    }

    // ── Score ─────────────────────────────────────────────────────────────────

    /// Compute the HBOS anomaly score for a single sample `x` (length `n_features`).
    ///
    /// Higher values indicate higher anomaly likelihood.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        if self.n_samples == 0 {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }

        let mut hbos = 0.0_f32;

        for (j, &v) in x.iter().enumerate() {
            let density = self.lookup_density(j, v);
            hbos += -(density + EPSILON).ln();
        }

        Ok(hbos)
    }

    /// Batch HBOS scoring; `x` is row-major `[n × n_features]`; returns `[n]` scores.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if self.n_samples == 0 {
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

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Map a value `v` in feature `j` to its bin density.
    ///
    /// Values outside the training range return `0.0` (zero density), giving the
    /// maximum possible anomaly contribution `-log(ε)`.
    #[inline]
    fn lookup_density(&self, j: usize, v: f32) -> f32 {
        let feature_dens = &self.densities[j];

        // Constant feature (max == min): all values equal the training constant score 1,
        // anything else is a zero-density outlier.
        if (self.max_vals[j] - self.min_vals[j]).abs() < 1e-8 {
            if (v - self.min_vals[j]).abs() < 1e-8 {
                return feature_dens[0];
            }
            return 0.0;
        }

        // Out-of-range: return zero density for maximum anomaly contribution.
        if v < self.min_vals[j] || v > self.max_vals[j] {
            return 0.0;
        }

        let n_bins = feature_dens.len();
        let bin = Self::bin_index(v, self.min_vals[j], self.bin_widths[j], n_bins);
        feature_dens[bin]
    }

    /// Map `v` to a clamped bin index `[0, n_bins)`.
    #[inline]
    fn bin_index(v: f32, min_val: f32, bin_width: f32, n_bins: usize) -> usize {
        let raw = ((v - min_val) / bin_width).floor() as isize;
        // Clamp: values at exactly max_val fall into the last bin.
        raw.max(0).min(n_bins as isize - 1) as usize
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple 1-D dataset: 0.0, 0.1, …, 1.9  (20 points).
    fn simple_1d() -> Vec<f32> {
        (0..20).map(|i| i as f32 * 0.1).collect()
    }

    #[test]
    fn fit_and_score_finite() {
        let data = simple_1d();
        let mut hbos = Hbos::default();
        hbos.fit(&data, 20, 1).unwrap();
        let s = hbos.score(&[0.9_f32]).unwrap();
        assert!(s.is_finite(), "score should be finite, got {s}");
    }

    #[test]
    fn unfitted_returns_not_fitted_error() {
        let hbos = Hbos::default();
        let err = hbos.score(&[0.5_f32]).unwrap_err();
        assert!(
            matches!(err, AnomalyError::NotFitted),
            "expected NotFitted, got {err:?}"
        );
    }

    #[test]
    fn outlier_scores_higher_than_inlier() {
        // Data tightly clustered around 0.5.
        let mut data: Vec<f32> = (0..50).map(|i| 0.4 + i as f32 * 0.004).collect();
        // Append a clear outlier far from the cluster.
        data.push(10.0_f32);

        // Fit on cluster-only data (exclude the outlier for training)
        let train = &data[..50];
        let mut hbos = Hbos::new(HbosConfig { n_bins: 10 });
        hbos.fit(train, 50, 1).unwrap();

        let inlier_score = hbos.score(&[0.5_f32]).unwrap();
        let outlier_score = hbos.score(&[10.0_f32]).unwrap();

        assert!(
            outlier_score > inlier_score,
            "outlier_score={outlier_score} should exceed inlier_score={inlier_score}"
        );
    }

    #[test]
    fn empty_input_returns_error() {
        let mut hbos = Hbos::default();
        let err = hbos.fit(&[], 0, 1).unwrap_err();
        assert!(
            matches!(err, AnomalyError::EmptyInput),
            "expected EmptyInput, got {err:?}"
        );
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let mut hbos = Hbos::default();
        // Claim 10 samples × 2 features but only supply 10 floats (not 20).
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let err = hbos.fit(&data, 10, 2).unwrap_err();
        assert!(
            matches!(err, AnomalyError::DimensionMismatch { .. }),
            "expected DimensionMismatch, got {err:?}"
        );
    }

    #[test]
    fn score_batch_returns_correct_length() {
        let data = simple_1d();
        let mut hbos = Hbos::default();
        hbos.fit(&data, 20, 1).unwrap();

        let queries: Vec<f32> = (0..5).map(|i| i as f32 * 0.3).collect();
        let scores = hbos.score_batch(&queries, 5).unwrap();
        assert_eq!(scores.len(), 5, "batch output should have 5 scores");
        assert!(
            scores.iter().all(|s| s.is_finite()),
            "all scores must be finite"
        );
    }

    #[test]
    fn constant_feature_no_panic() {
        // All values identical → constant feature edge case.
        const CONSTANT_VALUE: f32 = 2.5;
        let data = vec![CONSTANT_VALUE; 30];
        let mut hbos = Hbos::default();
        // Should not panic or return Err.
        hbos.fit(&data, 30, 1).unwrap();
        // Scoring the constant value itself must return a finite result.
        let s = hbos.score(&[CONSTANT_VALUE]).unwrap();
        assert!(
            s.is_finite(),
            "score on constant feature must be finite, got {s}"
        );
    }

    #[test]
    fn n_bins_one_all_inliers_same_score() {
        // With n_bins=1 every point lands in the single bin → all get the same score.
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let mut hbos = Hbos::new(HbosConfig { n_bins: 1 });
        hbos.fit(&data, 20, 1).unwrap();

        let s0 = hbos.score(&[0.0_f32]).unwrap();
        let s1 = hbos.score(&[10.0_f32]).unwrap();
        let s2 = hbos.score(&[19.0_f32]).unwrap();

        assert!(
            (s0 - s1).abs() < 1e-5 && (s0 - s2).abs() < 1e-5,
            "all inliers should share the same score: {s0}, {s1}, {s2}"
        );
    }

    #[test]
    fn zero_n_bins_returns_error() {
        let mut hbos = Hbos::new(HbosConfig { n_bins: 0 });
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let err = hbos.fit(&data, 10, 1).unwrap_err();
        assert!(
            matches!(err, AnomalyError::Internal { .. }),
            "expected Internal error for n_bins=0, got {err:?}"
        );
    }

    #[test]
    fn feature_count_mismatch_on_score() {
        let data = simple_1d();
        let mut hbos = Hbos::default();
        hbos.fit(&data, 20, 1).unwrap();

        // Provide 2 features when model expects 1.
        let err = hbos.score(&[0.5_f32, 0.5_f32]).unwrap_err();
        assert!(
            matches!(
                err,
                AnomalyError::FeatureCountMismatch {
                    expected: 1,
                    got: 2
                }
            ),
            "expected FeatureCountMismatch, got {err:?}"
        );
    }

    #[test]
    fn multivariate_fit_and_score() {
        // 2-feature dataset: feature 0 in [0,1], feature 1 in [10,11].
        let data: Vec<f32> = (0..30)
            .flat_map(|i| {
                let f0 = i as f32 / 30.0;
                let f1 = 10.0 + i as f32 / 30.0;
                [f0, f1]
            })
            .collect();

        let mut hbos = Hbos::new(HbosConfig { n_bins: 5 });
        hbos.fit(&data, 30, 2).unwrap();

        // Point well within both feature ranges.
        let s_in = hbos.score(&[0.5_f32, 10.5_f32]).unwrap();
        // Point outside both feature ranges.
        let s_out = hbos.score(&[5.0_f32, 0.0_f32]).unwrap();

        assert!(s_in.is_finite(), "inlier score must be finite");
        assert!(s_out.is_finite(), "outlier score must be finite");
        assert!(
            s_out > s_in,
            "out-of-range point should score higher: {s_out} vs {s_in}"
        );
    }
}

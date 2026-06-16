//! COPOD — Copula-Based Outlier Detection (Li et al. 2020).
//!
//! Uses empirical marginal CDFs per feature.
//!
//! * Left-tail:  `OPZL(x) = −Σ_j log(F̂_j(x_j) + ε)`
//! * Right-tail: `OPZR(x) = −Σ_j log(1 − F̂_j(x_j) + ε)`
//! * Score:      `(OPZL + OPZR) / 2`
//!
//! Both tails are checked because anomalies can be extreme in either direction.
//! ε = 1e-10 to avoid log(0).

use crate::error::{AnomalyError, AnomalyResult};

const LOG_EPS: f32 = 1e-10;

// ─── Copod ────────────────────────────────────────────────────────────────────

/// COPOD anomaly detector.
pub struct Copod {
    /// `sorted_features[j]` = sorted training values for feature `j`.
    sorted_features: Vec<Vec<f32>>,
    n_train: usize,
    n_features: usize,
}

impl Copod {
    /// Create an unfitted COPOD detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sorted_features: Vec::new(),
            n_train: 0,
            n_features: 0,
        }
    }

    /// Fit by sorting each feature's training values.
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

        for j in 0..n_features {
            let mut col: Vec<f32> = (0..n_samples).map(|i| data[i * n_features + j]).collect();
            col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.sorted_features.push(col);
        }

        Ok(())
    }

    /// Empirical CDF for feature `feat` at value `v`.
    ///
    /// Returns the fraction of training samples ≤ `v` (binary search).
    fn ecdf(&self, feat: usize, v: f32) -> f32 {
        let col = &self.sorted_features[feat];
        // upper_bound: count of values ≤ v
        let count = col.partition_point(|&x| x <= v);
        count as f32 / self.n_train as f32
    }

    /// Anomaly score: `(OPZL + OPZR) / 2`.
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

        let mut opzl = 0.0_f32;
        let mut opzr = 0.0_f32;
        for (j, &xi) in x.iter().enumerate().take(self.n_features) {
            let cdf = self.ecdf(j, xi);
            opzl -= (cdf + LOG_EPS).ln();
            opzr -= (1.0 - cdf + LOG_EPS).ln();
        }
        Ok((opzl + opzr) * 0.5)
    }

    /// Batch scoring; `x` is `[n * n_features]`; returns `[n]`.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
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

    /// Skewness-adjusted variant: computes COPOD score with tail weights based on
    /// the empirical skewness of each feature.
    ///
    /// For positively skewed features the right tail contributes more;
    /// for negatively skewed features the left tail contributes more.
    pub fn score_skew_adjusted(&self, x: &[f32]) -> AnomalyResult<f32> {
        if self.n_features == 0 {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }

        let mut score_acc = 0.0_f32;
        for (j, &xi) in x.iter().enumerate().take(self.n_features) {
            let col = &self.sorted_features[j];
            // Compute mean and std for skewness
            let mean = col.iter().sum::<f32>() / col.len() as f32;
            let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / col.len() as f32;
            let std = var.sqrt();
            // Fisher-Pearson skewness
            let skew = if std > 1e-8 {
                col.iter().map(|v| ((v - mean) / std).powi(3)).sum::<f32>() / col.len() as f32
            } else {
                0.0
            };

            let cdf = self.ecdf(j, xi);
            let opzl = -(cdf + LOG_EPS).ln();
            let opzr = -(1.0 - cdf + LOG_EPS).ln();

            // Weight right tail higher for positive skew, left tail for negative skew
            let w_r = (0.5 + skew.clamp(-1.0, 1.0) * 0.5).clamp(0.0, 1.0);
            let w_l = 1.0 - w_r;
            score_acc += w_l * opzl + w_r * opzr;
        }
        Ok(score_acc)
    }
}

impl Default for Copod {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data() -> Vec<f32> {
        (0..20_usize)
            .flat_map(|i| vec![i as f32, (i * 2) as f32])
            .collect()
    }

    #[test]
    fn copod_fit_score() {
        let data = make_data();
        let mut det = Copod::new();
        det.fit(&data, 20, 2)
            .expect("fit should succeed with valid 20-sample 2-feature data");
        let s = det
            .score(&[5.0_f32, 10.0])
            .expect("score should succeed after successful fit");
        assert!(s.is_finite() && s >= 0.0, "s={s}");
    }

    #[test]
    fn copod_extreme_outlier_higher() {
        let data = make_data();
        let mut det = Copod::new();
        det.fit(&data, 20, 2)
            .expect("fit should succeed with valid 20-sample 2-feature data");
        let s_normal = det
            .score(&[10.0_f32, 20.0])
            .expect("score should succeed for inlier point");
        let s_outlier = det
            .score(&[1000.0_f32, 2000.0])
            .expect("score should succeed for outlier point");
        assert!(s_outlier > s_normal, "{s_outlier} should > {s_normal}");
    }

    #[test]
    fn copod_skew_adjusted_finite() {
        let data = make_data();
        let mut det = Copod::new();
        det.fit(&data, 20, 2)
            .expect("fit should succeed with valid 20-sample 2-feature data");
        let s = det
            .score_skew_adjusted(&[5.0_f32, 10.0])
            .expect("skew-adjusted score should succeed after fit");
        assert!(s.is_finite(), "s={s}");
    }
}

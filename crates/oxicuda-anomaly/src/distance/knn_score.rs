//! k-NN distance anomaly scorer.
//!
//! Score = average distance to the `k` nearest training neighbours.
//! Simple and effective as a baseline anomaly detector.

use crate::error::{AnomalyError, AnomalyResult};

/// k-NN distance-based anomaly scorer.
pub struct KnnAnomalyScorer {
    data: Vec<f32>,
    n_samples: usize,
    n_features: usize,
    k: usize,
}

impl KnnAnomalyScorer {
    /// Create an unfitted scorer with neighbour count `k`.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            data: Vec::new(),
            n_samples: 0,
            n_features: 0,
            k,
        }
    }

    /// Store training data for later scoring.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if self.k == 0 {
            return Err(AnomalyError::InvalidK { k: 0 });
        }
        if n_samples < self.k {
            return Err(AnomalyError::InsufficientSamples {
                need: self.k,
                got: n_samples,
            });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }
        self.data = data.to_vec();
        self.n_samples = n_samples;
        self.n_features = n_features;
        Ok(())
    }

    /// Average distance to `k` nearest training neighbours.
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

        // Compute distances to all training points
        let mut dists: Vec<f32> = (0..self.n_samples)
            .map(|i| {
                let row = &self.data[i * self.n_features..(i + 1) * self.n_features];
                row.iter()
                    .zip(x.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();

        dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let avg = dists[..self.k].iter().sum::<f32>() / self.k as f32;
        Ok(avg)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_score_basic() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut scorer = KnnAnomalyScorer::new(3);
        scorer.fit(&data, 10, 1).unwrap();
        let s = scorer.score(&[5.0_f32]).unwrap();
        assert!(s.is_finite() && s >= 0.0, "score={s}");
    }

    #[test]
    fn knn_score_zero_for_training_point() {
        let data = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let mut scorer = KnnAnomalyScorer::new(1);
        scorer.fit(&data, 5, 1).unwrap();
        let s = scorer.score(&[2.0_f32]).unwrap();
        assert!(s < 1e-5, "score={s}"); // nearest is itself at distance 0
    }
}

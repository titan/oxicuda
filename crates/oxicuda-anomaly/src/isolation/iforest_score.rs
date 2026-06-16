//! Isolation Forest scoring (Liu et al. 2008).
//!
//! Anomaly score formula:
//! ```text
//! s(x) = 2^{−E[h(x)] / c(n)}
//! ```
//! where `c(n) = 2·H(n−1) − 2·(n−1)/n` and `H(k) ≈ ln(k) + γ`.
//!
//! Rather than building full isolation trees (which require tree data structures),
//! this module provides a lightweight **random-projection estimator** for the
//! expected path length.  For each projection direction `w_t`, we project the
//! training set and use the resulting rank statistics as a proxy for isolation.
//!
//! Path-length contribution per projection:
//! ```text
//! path_contrib = −log2(|2 · rank_fraction − 1| + ε)
//! ```
//! where `rank_fraction = fraction of training projections < p_x`.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

/// Euler–Mascheroni constant.
const EULER_MASCHERONI: f32 = 0.577_215_6_f32;

// ─── c_factor ────────────────────────────────────────────────────────────────

/// Average path length of an unsuccessful BST search: `c(n)`.
///
/// `c(n) = 2·(ln(n−1) + γ) − 2·(n−1)/n` for `n ≥ 2`.
///
/// Special cases: `c(1) = 0`, `c(2) = 1`.
#[must_use]
pub fn c_factor(n: usize) -> f32 {
    match n {
        0 | 1 => 0.0,
        2 => 1.0,
        _ => {
            let nm1 = (n - 1) as f32;
            2.0 * (nm1.ln() + EULER_MASCHERONI) - 2.0 * nm1 / n as f32
        }
    }
}

// ─── isolation_score_from_path ────────────────────────────────────────────────

/// Convert average path length to isolation score.
///
/// `s = 2^{−avg_path / c(n)}`
///
/// * `s > 0.5` → anomaly
/// * `s < 0.5` → normal
/// * `s ≈ 0.5` → neutral
#[must_use]
pub fn isolation_score_from_path(avg_path_length: f32, n: usize) -> f32 {
    let c = c_factor(n);
    if c < 1e-8 {
        return 0.5;
    }
    let exp = -avg_path_length / c;
    (2.0_f32).powf(exp)
}

// ─── IsolationScorer ─────────────────────────────────────────────────────────

/// Lightweight isolation-based anomaly scorer using random 1-D projections.
pub struct IsolationScorer {
    /// Random unit projection directions: `[n_estimators * n_features]`.
    projections: Vec<Vec<f32>>,
    /// Projected training values per estimator: `[n_estimators][n_train]` sorted.
    proj_training: Vec<Vec<f32>>,
    n_estimators: usize,
    n_features: usize,
    n_train: usize,
}

impl IsolationScorer {
    /// Create scorer with `n_estimators` random projection directions.
    pub fn new(n_estimators: usize, rng: &mut LcgRng) -> Self {
        let mut projections = Vec::with_capacity(n_estimators);
        // Placeholder; actual feature count set in fit()
        // We store direction vecs of size 0 until fit() is called
        for _ in 0..n_estimators {
            let _ = rng.next_u32(); // consume entropy
            projections.push(Vec::new());
        }
        Self {
            projections,
            proj_training: Vec::new(),
            n_estimators,
            n_features: 0,
            n_train: 0,
        }
    }

    /// Project training data onto `n_estimators` random unit vectors.
    pub fn fit(
        &mut self,
        data: &[f32],
        n_samples: usize,
        n_features: usize,
        rng: &mut LcgRng,
    ) -> AnomalyResult<()> {
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

        self.n_features = n_features;
        self.n_train = n_samples;

        // Generate n_estimators random unit projection directions
        self.projections = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            let mut dir: Vec<f32> = (0..n_features).map(|_| rng.next_normal()).collect();
            // Normalise to unit vector
            let norm = dir.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for v in &mut dir {
                    *v /= norm;
                }
            } else {
                // Degenerate: use first-axis
                dir = vec![0.0_f32; n_features];
                dir[0] = 1.0;
            }
            self.projections.push(dir);
        }

        // Project training data onto each direction and sort
        self.proj_training = Vec::with_capacity(self.n_estimators);
        for t in 0..self.n_estimators {
            let dir = &self.projections[t];
            let mut projs: Vec<f32> = (0..n_samples)
                .map(|i| {
                    let row = &data[i * n_features..(i + 1) * n_features];
                    row.iter().zip(dir.iter()).map(|(x, w)| x * w).sum()
                })
                .collect();
            projs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.proj_training.push(projs);
        }

        Ok(())
    }

    /// Estimate the expected isolation path length for query `x`.
    pub fn path_length(&self, x: &[f32]) -> AnomalyResult<f32> {
        if self.n_features == 0 {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }

        let mut total_path = 0.0_f32;
        for t in 0..self.n_estimators {
            let dir = &self.projections[t];
            let p_x: f32 = x.iter().zip(dir.iter()).map(|(xi, wi)| xi * wi).sum();
            let sorted = &self.proj_training[t];
            // rank_fraction = fraction of training projections < p_x
            let rank = sorted.partition_point(|&v| v < p_x);
            let rank_frac = rank as f32 / self.n_train as f32;
            // Path contribution: isolating a point far from the median is cheaper
            let path_contrib = -(((2.0 * rank_frac - 1.0).abs()) + 1e-8).log2();
            total_path += path_contrib;
        }
        Ok(total_path / self.n_estimators as f32)
    }

    /// Isolation-based anomaly score ∈ (0, 1).
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        let path = self.path_length(x)?;
        Ok(isolation_score_from_path(path, self.n_train))
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
    fn c_factor_values() {
        assert!((c_factor(0)).abs() < 1e-6);
        assert!((c_factor(1)).abs() < 1e-6);
        assert!((c_factor(2) - 1.0).abs() < 1e-6);
        assert!(c_factor(100) > 0.0);
    }

    #[test]
    fn isolation_score_in_range() {
        let s = isolation_score_from_path(5.0, 256);
        assert!((0.0..=1.0).contains(&s), "s={s}");
    }

    #[test]
    fn iforest_fit_score() {
        let mut rng = LcgRng::new(42);
        let mut scorer = IsolationScorer::new(50, &mut rng);
        let data: Vec<f32> = (0..100_usize)
            .flat_map(|i| vec![i as f32 * 0.1, i as f32 * 0.05])
            .collect();
        scorer
            .fit(&data, 100, 2, &mut rng)
            .expect("isolation scorer fit should succeed");
        let s = scorer
            .score(&[5.0_f32, 2.5])
            .expect("isolation scorer score should succeed");
        assert!((0.0..=1.0).contains(&s), "s={s}");
    }
}

//! INNE — Isolation using Nearest-Neighbour Ensembles (Bandaragoda et al. 2018).
//!
//! INNE builds an ensemble of `n_estimators` models. Each model draws a random
//! sub-sample `D` of `sample_size` (ψ) points without replacement. Within the
//! sub-sample, every reference point `c` defines a **hypersphere** `B(c)`
//! centred at `c` whose radius `τ(c)` equals the distance from `c` to its
//! nearest neighbour inside `D`:
//!
//! ```text
//! τ(c) = min_{c' ∈ D, c' ≠ c} ‖c − c'‖
//! ```
//!
//! For a query `x` the **isolation score** of one model is the relative radius
//! of the smallest hypersphere that contains `x`:
//!
//! ```text
//! cnn(x) = argmin_{ c ∈ D : ‖x − c‖ ≤ τ(c) } τ(c)
//! I(x)   = 1 − τ(η(cnn(x))) / τ(cnn(x))        if x is covered by some ball
//!        = 1                                    if x is covered by no ball
//! ```
//!
//! where `η(c)` is the nearest neighbour of `c` inside `D` (so `τ(η(c))` is the
//! radius of that neighbour's ball). The final anomaly score averages `I(x)`
//! over the ensemble. Because every per-model term is `≤ 1`, the averaged score
//! is also `≤ 1`; larger values indicate stronger isolation (anomaly).
//!
//! # Reference
//! Bandaragoda, T. R., Ting, K. M., Albrecht, D., Liu, F. T., Zhu, Y., &
//! Wells, J. R. (2018). *Isolation-based anomaly detection using
//! nearest-neighbour ensembles*. Computational Intelligence, 34(4), 968–998.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

/// Numerical floor used to keep radii strictly positive.
const RADIUS_FLOOR: f32 = 1e-12;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for the INNE detector.
#[derive(Debug, Clone)]
pub struct InneConfig {
    /// Number of ensemble members `t` (default 200).
    pub n_estimators: usize,
    /// Sub-sample size `ψ` per estimator (default 16). Must be `≥ 2`.
    pub sample_size: usize,
    /// RNG seed for reproducible sub-sampling (default 42).
    pub seed: u64,
}

impl Default for InneConfig {
    fn default() -> Self {
        Self {
            n_estimators: 200,
            sample_size: 16,
            seed: 42,
        }
    }
}

// ─── Per-estimator hypersphere set ────────────────────────────────────────────

/// One fitted INNE estimator: the sub-sample centroids together with each
/// centroid's hypersphere radius and the precomputed radius ratio.
#[derive(Debug, Clone)]
struct InneEstimator {
    /// Sub-sample centroids, row-major `[sample_size * n_features]`.
    centroids: Vec<f32>,
    /// Hypersphere radius `τ(c)` per centroid `[sample_size]`.
    radii: Vec<f32>,
    /// Precomputed `τ(η(c)) / τ(c)` per centroid `[sample_size]`.
    ratio: Vec<f32>,
}

impl InneEstimator {
    /// Isolation score `I(x)` for one query against this estimator.
    fn isolation_score(&self, x: &[f32], n_features: usize) -> f32 {
        let mut best_radius = f32::INFINITY;
        let mut covered_ratio: Option<f32> = None;
        for (c, centroid) in self.centroids.chunks_exact(n_features).enumerate() {
            let dist = euclidean(x, centroid);
            // `x` falls inside ball B(c) when dist ≤ radius. Among all covering
            // balls keep the one with the smallest radius (the "nearest" ball).
            if dist <= self.radii[c] && self.radii[c] < best_radius {
                best_radius = self.radii[c];
                covered_ratio = Some(self.ratio[c]);
            }
        }
        match covered_ratio {
            Some(ratio) => 1.0 - ratio,
            None => 1.0,
        }
    }
}

// ─── Euclidean distance (internal, equal length assumed) ──────────────────────

/// Euclidean distance between two equal-length slices.
#[inline]
fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

// ─── InneDetector ─────────────────────────────────────────────────────────────

/// Isolation using Nearest-Neighbour Ensembles anomaly detector.
///
/// # Usage
///
/// ```rust,ignore
/// let mut det = InneDetector::new(InneConfig::default());
/// det.fit(&train, n_samples, n_features)?;
/// let s = det.score(&query)?;
/// ```
#[derive(Debug, Clone)]
pub struct InneDetector {
    config: InneConfig,
    estimators: Vec<InneEstimator>,
    n_features: usize,
    fitted: bool,
}

impl InneDetector {
    /// Create an unfitted detector from the supplied configuration.
    #[must_use]
    pub fn new(config: InneConfig) -> Self {
        Self {
            config,
            estimators: Vec::new(),
            n_features: 0,
            fitted: false,
        }
    }

    /// Fit the ensemble on `data` (`n_samples × n_features`, row-major).
    ///
    /// # Errors
    /// * [`AnomalyError::EmptyInput`] if `n_samples == 0`.
    /// * [`AnomalyError::InvalidFeatureCount`] if `n_features == 0`.
    /// * [`AnomalyError::DimensionMismatch`] if `data.len() != n_samples * n_features`.
    /// * [`AnomalyError::InsufficientSamples`] if `sample_size < 2` or
    ///   `sample_size > n_samples`.
    /// * [`AnomalyError::Internal`] if `n_estimators == 0`.
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
        if self.config.n_estimators == 0 {
            return Err(AnomalyError::Internal {
                msg: "n_estimators must be > 0".into(),
            });
        }
        let psi = self.config.sample_size;
        if psi < 2 {
            return Err(AnomalyError::InsufficientSamples { need: 2, got: psi });
        }
        if psi > n_samples {
            return Err(AnomalyError::InsufficientSamples {
                need: psi,
                got: n_samples,
            });
        }

        let mut rng = LcgRng::new(self.config.seed);
        let mut estimators = Vec::with_capacity(self.config.n_estimators);
        let mut index_pool: Vec<usize> = (0..n_samples).collect();

        for _ in 0..self.config.n_estimators {
            // Partial Fisher–Yates: select `psi` distinct indices without replacement.
            for k in 0..psi {
                let j = k + rng.next_usize(n_samples - k);
                index_pool.swap(k, j);
            }

            // Gather the sub-sample centroids.
            let mut centroids = vec![0.0_f32; psi * n_features];
            for (slot, &src) in index_pool[..psi].iter().enumerate() {
                let dst = &mut centroids[slot * n_features..(slot + 1) * n_features];
                dst.copy_from_slice(&data[src * n_features..(src + 1) * n_features]);
            }

            // Radius τ(c) = NN distance within the sub-sample; record NN index.
            let radius_nn: Vec<(f32, usize)> = (0..psi)
                .map(|c| {
                    let row_c = &centroids[c * n_features..(c + 1) * n_features];
                    let mut best = f32::INFINITY;
                    let mut best_j = c;
                    for d in 0..psi {
                        if d == c {
                            continue;
                        }
                        let dist =
                            euclidean(row_c, &centroids[d * n_features..(d + 1) * n_features]);
                        if dist < best {
                            best = dist;
                            best_j = d;
                        }
                    }
                    (best, best_j)
                })
                .collect();

            let radii: Vec<f32> = radius_nn.iter().map(|&(r, _)| r).collect();
            // ratio(c) = τ(η(c)) / τ(c); floor the denominator to stay finite.
            let ratio: Vec<f32> = radius_nn
                .iter()
                .map(|&(r, nn)| radii[nn] / r.max(RADIUS_FLOOR))
                .collect();

            estimators.push(InneEstimator {
                centroids,
                radii,
                ratio,
            });
        }

        self.estimators = estimators;
        self.n_features = n_features;
        self.fitted = true;
        Ok(())
    }

    /// Anomaly score for one query `x` (averaged isolation score, `≤ 1`).
    ///
    /// Higher values indicate stronger isolation (more anomalous).
    ///
    /// # Errors
    /// * [`AnomalyError::NotFitted`] if [`InneDetector::fit`] has not been called.
    /// * [`AnomalyError::FeatureCountMismatch`] if `x.len() != n_features`.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let sum: f32 = self
            .estimators
            .iter()
            .map(|est| est.isolation_score(x, self.n_features))
            .sum();
        Ok(sum / self.estimators.len() as f32)
    }

    /// Batch scoring; `x` is `[n × n_features]` row-major, returns `[n]`.
    ///
    /// # Errors
    /// * [`AnomalyError::NotFitted`] if the detector is unfitted.
    /// * [`AnomalyError::DimensionMismatch`] if `x.len() != n * n_features`.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if !self.fitted {
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

    /// Number of features the detector was fitted on (0 if unfitted).
    #[inline]
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of fitted ensemble members.
    #[inline]
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense 2-D cluster of `n` points near the origin (deterministic jitter).
    fn dense_cluster(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut data = Vec::with_capacity(n * 2);
        for _ in 0..n {
            data.push(rng.next_f32() * 0.2);
            data.push(rng.next_f32() * 0.2);
        }
        data
    }

    // ── Test (a): a clear outlier scores higher than inliers ──────────────────
    #[test]
    fn outlier_scores_higher_than_inliers() {
        let n = 40_usize;
        let data = dense_cluster(n, 1);
        let cfg = InneConfig {
            n_estimators: 100,
            sample_size: 8,
            seed: 7,
        };
        let mut det = InneDetector::new(cfg);
        det.fit(&data, n, 2).expect("fit");

        let outlier = det.score(&[50.0_f32, 50.0]).expect("outlier");
        // A handful of inliers sampled from the cluster region.
        let inliers = [[0.05_f32, 0.1], [0.1, 0.05], [0.15, 0.15], [0.08, 0.12]];
        for inlier in &inliers {
            let s_in = det.score(inlier).expect("inlier");
            assert!(
                outlier > s_in,
                "outlier {outlier} should exceed inlier {s_in}"
            );
        }
    }

    // ── Test (b): scores are finite and bounded above by 1 ────────────────────
    #[test]
    fn scores_finite_and_bounded() {
        let n = 30_usize;
        let data = dense_cluster(n, 2);
        let mut det = InneDetector::new(InneConfig {
            n_estimators: 64,
            sample_size: 8,
            seed: 11,
        });
        det.fit(&data, n, 2).expect("fit");

        let queries = [
            [0.1_f32, 0.1],
            [0.5, 0.2],
            [10.0, -10.0],
            [0.0, 0.0],
            [-3.0, 4.0],
        ];
        for q in &queries {
            let s = det.score(q).expect("score");
            assert!(s.is_finite(), "score must be finite, got {s}");
            assert!(s <= 1.0 + 1e-5, "score {s} must be ≤ 1");
            assert!(s >= -5.0, "score {s} unexpectedly small");
        }
    }

    // ── Test (c): larger ensemble → lower score variance across seeds ──────────
    #[test]
    fn larger_ensemble_reduces_variance() {
        let n = 36_usize;
        let data = dense_cluster(n, 3);
        let query = [0.12_f32, 0.09]; // an inlier-ish point with a non-trivial score

        let variance_for = |n_estimators: usize| -> f32 {
            let mut scores = Vec::new();
            for seed in 0..8_u64 {
                let mut det = InneDetector::new(InneConfig {
                    n_estimators,
                    sample_size: 6,
                    seed,
                });
                det.fit(&data, n, 2).expect("fit");
                scores.push(det.score(&query).expect("score"));
            }
            let mean = scores.iter().sum::<f32>() / scores.len() as f32;
            scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32
        };

        let var_small = variance_for(2);
        let var_large = variance_for(64);
        assert!(
            var_large < var_small,
            "variance should shrink with more estimators: small={var_small}, large={var_large}"
        );
    }

    // ── Test (d): invalid sample_size / empty data → error ────────────────────
    #[test]
    fn invalid_sample_size_and_empty_errors() {
        let n = 10_usize;
        let data = dense_cluster(n, 4);

        // sample_size > n
        let mut det = InneDetector::new(InneConfig {
            n_estimators: 10,
            sample_size: 50,
            seed: 1,
        });
        assert!(matches!(
            det.fit(&data, n, 2),
            Err(AnomalyError::InsufficientSamples { .. })
        ));

        // n = 0 → EmptyInput
        let mut det2 = InneDetector::new(InneConfig::default());
        assert!(matches!(det2.fit(&[], 0, 2), Err(AnomalyError::EmptyInput)));

        // sample_size < 2 → InsufficientSamples
        let mut det3 = InneDetector::new(InneConfig {
            n_estimators: 10,
            sample_size: 1,
            seed: 1,
        });
        assert!(matches!(
            det3.fit(&data, n, 2),
            Err(AnomalyError::InsufficientSamples { need: 2, got: 1 })
        ));

        // score before fit → NotFitted
        let det4 = InneDetector::new(InneConfig::default());
        assert!(matches!(
            det4.score(&[0.0_f32, 0.0]),
            Err(AnomalyError::NotFitted)
        ));
    }

    // ── Test (e): determinism with a fixed seed ───────────────────────────────
    #[test]
    fn deterministic_with_fixed_seed() {
        let n = 30_usize;
        let data = dense_cluster(n, 5);
        let cfg = InneConfig {
            n_estimators: 50,
            sample_size: 8,
            seed: 123,
        };
        let mut det_a = InneDetector::new(cfg.clone());
        let mut det_b = InneDetector::new(cfg);
        det_a.fit(&data, n, 2).expect("fit a");
        det_b.fit(&data, n, 2).expect("fit b");

        for q in &[[0.1_f32, 0.1], [5.0, 5.0], [0.3, 0.0]] {
            let sa = det_a.score(q).expect("score a");
            let sb = det_b.score(q).expect("score b");
            assert!((sa - sb).abs() < 1e-6, "scores differ: {sa} vs {sb}");
        }
    }

    // ── Extra: dimension mismatch + batch scoring ─────────────────────────────
    #[test]
    fn feature_mismatch_and_batch() {
        let n = 20_usize;
        let data = dense_cluster(n, 6);
        let mut det = InneDetector::new(InneConfig {
            n_estimators: 20,
            sample_size: 5,
            seed: 9,
        });
        det.fit(&data, n, 2).expect("fit");

        assert!(matches!(
            det.score(&[0.1_f32, 0.2, 0.3]),
            Err(AnomalyError::FeatureCountMismatch {
                expected: 2,
                got: 3
            })
        ));

        let batch = det.score_batch(&data, n).expect("batch");
        assert_eq!(batch.len(), n);
        assert!(batch.iter().all(|s| s.is_finite()));
    }
}

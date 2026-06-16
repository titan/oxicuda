//! FastABOD: Approximate Angle-Based Outlier Detection.
//!
//! Kriegel, Schubert & Zimek (2008), §4: "Angle-Based Outlier Detection in
//! High-Dimensional Data", KDD 2008.
//!
//! # Key idea
//!
//! Exact ABOD has O(n²·d) complexity per query — intractable for large n.
//! FastABOD restricts the angle-variance computation to the k-nearest
//! neighbours of the query point, reducing the cost to O(k·n·d):
//!
//! ```text
//! ABOF_fast(p) = Var[⟨pa, pb⟩ / (‖pa‖² · ‖pb‖²)]
//!              evaluated over the k-NN set of p
//! ```
//!
//! Outliers on the periphery see their k-NN concentrated in one direction →
//! low ABOF.  Inliers see their k-NN spread around them → high ABOF.
//!
//! [`AbodApprox::score`] returns `1 / (ABOF_fast + ε)` so that
//! **high score = more anomalous**.

use crate::error::{AnomalyError, AnomalyResult};

// ─── AbodApprox ───────────────────────────────────────────────────────────────

/// FastABOD approximate angle-based outlier detector.
///
/// Uses k-nearest-neighbour restriction to bring O(n²·d) → O(k·n·d).
#[derive(Debug, Clone)]
pub struct AbodApprox {
    /// Number of neighbours to use for the angle variance approximation.
    pub k: usize,
    /// Training data (row-major, `[n_samples × n_features]`).
    data: Vec<f32>,
    n_samples: usize,
    n_features: usize,
    fitted: bool,
}

impl AbodApprox {
    /// Create a new FastABOD detector with `k` neighbours.
    ///
    /// # Errors
    /// Returns `AnomalyError::InvalidK` if `k == 0`.
    pub fn new(k: usize) -> AnomalyResult<Self> {
        if k == 0 {
            return Err(AnomalyError::InvalidK { k });
        }
        Ok(Self {
            k,
            data: Vec::new(),
            n_samples: 0,
            n_features: 0,
            fitted: false,
        })
    }

    /// Fit: store training data.
    ///
    /// Requires at least `k + 1` training samples (need `k` neighbours beyond
    /// the query point itself).
    ///
    /// # Errors
    /// - `AnomalyError::InsufficientSamples` if `n_samples < k + 1`.
    /// - `AnomalyError::InvalidFeatureCount` if `n_features == 0`.
    /// - `AnomalyError::DimensionMismatch` if `data.len() != n_samples * n_features`.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if n_samples < self.k + 1 {
            return Err(AnomalyError::InsufficientSamples {
                need: self.k + 1,
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
        self.fitted = true;
        Ok(())
    }

    /// Compute FastABOD score for a single query point `p`.
    ///
    /// Returns `1.0 / (ABOF + ε)` — **higher = more anomalous**.
    ///
    /// # Errors
    /// - `AnomalyError::NotFitted` if `fit` has not been called.
    /// - `AnomalyError::DimensionMismatch` if `p.len() != n_features`.
    pub fn score(&self, p: &[f32]) -> AnomalyResult<f32> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if p.len() != self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.n_features,
                got: p.len(),
            });
        }
        // Step 1: find k nearest neighbours of p.
        let mut dists: Vec<(f32, usize)> = (0..self.n_samples)
            .map(|i| {
                let row = &self.data[i * self.n_features..(i + 1) * self.n_features];
                (euclidean_sq(p, row), i)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // Take k nearest (skip i=0 in case p coincides with a training point).
        let knn_indices: Vec<usize> = dists.iter().take(self.k).map(|&(_, i)| i).collect();

        // Step 2: compute angle variance over k-NN pairs.
        let k = knn_indices.len();
        if k < 2 {
            return Ok(1.0); // Not enough neighbours for a variance.
        }

        let mut sum_f = 0.0_f64;
        let mut sum_f2 = 0.0_f64;
        let mut count = 0_u64;

        for i in 0..k {
            for j in (i + 1)..k {
                let a = &self.data
                    [knn_indices[i] * self.n_features..(knn_indices[i] + 1) * self.n_features];
                let b = &self.data
                    [knn_indices[j] * self.n_features..(knn_indices[j] + 1) * self.n_features];

                // pa = a - p,  pb = b - p.
                let pa: Vec<f64> = p
                    .iter()
                    .zip(a.iter())
                    .map(|(&pi, &ai)| (ai - pi) as f64)
                    .collect();
                let pb: Vec<f64> = p
                    .iter()
                    .zip(b.iter())
                    .map(|(&pi, &bi)| (bi - pi) as f64)
                    .collect();

                let dot: f64 = pa.iter().zip(pb.iter()).map(|(x, y)| x * y).sum();
                let norm_pa_sq: f64 = pa.iter().map(|x| x * x).sum();
                let norm_pb_sq: f64 = pb.iter().map(|x| x * x).sum();
                let denom = norm_pa_sq * norm_pb_sq;
                if denom < 1e-30 {
                    continue;
                }
                let f = dot / denom;
                sum_f += f;
                sum_f2 += f * f;
                count += 1;
            }
        }

        if count == 0 {
            return Ok(1.0);
        }
        let n = count as f64;
        let mean_f = sum_f / n;
        let abof = (sum_f2 / n - mean_f * mean_f).max(0.0);
        Ok((1.0 / (abof + 1e-10)) as f32)
    }

    /// Score a batch of query points (row-major `[n_queries × n_features]`).
    ///
    /// # Errors
    /// Propagates errors from [`Self::score`].
    pub fn score_batch(&self, queries: &[f32], n_queries: usize) -> AnomalyResult<Vec<f32>> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if queries.len() != n_queries * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_queries * self.n_features,
                got: queries.len(),
            });
        }
        (0..n_queries)
            .map(|i| {
                let row = &queries[i * self.n_features..(i + 1) * self.n_features];
                self.score(row)
            })
            .collect()
    }
}

// ─── Helper ───────────────────────────────────────────────────────────────────

#[inline]
fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 2D dataset: 10 points in a tight cluster near origin,
    /// plus 1 outlier at (50, 50).
    fn cluster_with_outlier() -> (Vec<f32>, usize) {
        let mut data = Vec::new();
        for i in 0..10_usize {
            data.push(i as f32 * 0.1);
            data.push(i as f32 * 0.1);
        }
        // Outlier
        data.push(50.0_f32);
        data.push(50.0_f32);
        (data, 11)
    }

    // ── 1. New: k=0 returns error ─────────────────────────────────────────────
    #[test]
    fn new_k_zero_error() {
        assert!(AbodApprox::new(0).is_err());
    }

    // ── 2. Fit: too few samples returns error ─────────────────────────────────
    #[test]
    fn fit_too_few_samples_error() {
        let mut det = AbodApprox::new(5).expect("k=5 is a valid neighbour count");
        let data = vec![0.0_f32; 4 * 2]; // only 4 samples, need 6
        assert!(det.fit(&data, 4, 2).is_err());
    }

    // ── 3. Fit: dimension mismatch error ──────────────────────────────────────
    #[test]
    fn fit_dim_mismatch_error() {
        let mut det = AbodApprox::new(2).expect("k=2 is a valid neighbour count");
        let data = vec![0.0_f32; 10]; // claims 3 samples × 2 features but wrong
        assert!(det.fit(&data, 3, 2).is_err()); // 3*2=6 ≠ 10
    }

    // ── 4. score: NotFitted returns error ─────────────────────────────────────
    #[test]
    fn score_not_fitted_error() {
        let det = AbodApprox::new(3).expect("AbodApprox with k=3 should be created");
        assert!(det.score(&[0.0, 0.0]).is_err());
    }

    // ── 5. score: dimension mismatch returns error ────────────────────────────
    #[test]
    fn score_dim_mismatch_error() {
        let mut det = AbodApprox::new(3).expect("AbodApprox with k=3 should be created");
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        det.fit(&data, 10, 2)
            .expect("fit with 10 samples and 2 features should succeed");
        // wrong query dim
        assert!(det.score(&[0.0, 0.0, 0.0]).is_err());
    }

    // ── 6. Outlier scores higher than inlier ─────────────────────────────────
    #[test]
    fn outlier_scores_higher_than_inlier() {
        let (data, n) = cluster_with_outlier();
        let mut det = AbodApprox::new(5).expect("k=5 is a valid neighbour count");
        det.fit(&data, n, 2)
            .expect("fit on cluster_with_outlier data should succeed");
        let s_inlier = det
            .score(&[0.3_f32, 0.3])
            .expect("scoring an inlier point should succeed");
        let s_outlier = det
            .score(&[50.0_f32, 50.0])
            .expect("scoring an outlier point should succeed");
        assert!(
            s_outlier > s_inlier,
            "outlier score {s_outlier} should > inlier {s_inlier}"
        );
    }

    // ── 7. Score is finite and positive ──────────────────────────────────────
    #[test]
    fn score_finite_positive() {
        let (data, n) = cluster_with_outlier();
        let mut det = AbodApprox::new(3).expect("AbodApprox with k=3 should be created");
        det.fit(&data, n, 2)
            .expect("fit on cluster_with_outlier data should succeed");
        let s = det
            .score(&[0.1_f32, 0.1])
            .expect("scoring a valid inlier point should succeed");
        assert!(s.is_finite() && s > 0.0, "score={s}");
    }

    // ── 8. Batch scoring matches individual scores ────────────────────────────
    #[test]
    fn batch_scores_match_individual() {
        let (data, n) = cluster_with_outlier();
        let mut det = AbodApprox::new(4).expect("k=4 is a valid neighbour count");
        det.fit(&data, n, 2)
            .expect("fit on cluster_with_outlier data should succeed");
        let queries = vec![0.2_f32, 0.2, 0.5, 0.5, 50.0, 50.0];
        let batch = det
            .score_batch(&queries, 3)
            .expect("batch scoring 3 valid queries should succeed");
        for (i, &bs) in batch.iter().enumerate() {
            let q = &queries[i * 2..(i + 1) * 2];
            let s = det
                .score(q)
                .expect("individual scoring of batch query should succeed");
            assert!((bs - s).abs() < 1e-5, "batch[{i}]={bs} vs individual={s}");
        }
    }

    // ── 9. Works for high-dimensional data ────────────────────────────────────
    #[test]
    fn works_high_dimensional() {
        let n = 20_usize;
        let d = 16_usize;
        let data: Vec<f32> = (0..n * d).map(|i| (i % 10) as f32 * 0.1).collect();
        let mut det = AbodApprox::new(5).expect("k=5 is a valid neighbour count");
        det.fit(&data, n, d)
            .expect("fit on high-dimensional data should succeed");
        let q = vec![0.5_f32; d];
        let s = det
            .score(&q)
            .expect("scoring a high-dimensional point should succeed");
        assert!(s.is_finite() && s > 0.0);
    }

    // ── 10. Large k (k = n-1) still works ────────────────────────────────────
    #[test]
    fn large_k_works() {
        let (data, n) = cluster_with_outlier(); // n=11
        let mut det = AbodApprox::new(10).expect("k=10 is valid (n-1 for n=11)"); // k = n-1
        det.fit(&data, n, 2).expect("fit with k=n-1 should succeed");
        let s = det
            .score(&[0.0_f32, 0.0])
            .expect("scoring with k=n-1 should succeed");
        assert!(s.is_finite());
    }
}

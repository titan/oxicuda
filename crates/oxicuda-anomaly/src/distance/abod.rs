//! Angle-Based Outlier Detection (ABOD).
//!
//! Kriegel, Schubert, Zimek. "Angle-based outlier detection in high-dimensional data". KDD 2008.
//!
//! # Key idea
//!
//! For a query point `p`, the Angle-Based Outlier Factor (ABOF) is the variance of the
//! weighted cosine kernel computed over all unordered pairs `{a, b}` of training points:
//!
//! ```text
//! f(a, b) = ⟨p-a, p-b⟩ / (‖p-a‖² · ‖p-b‖²)
//! ABOF(p) = Var[f(a,b)] = E[f²] − E[f]²
//! ```
//!
//! Inliers that lie *inside* a cluster see their neighbors spread in all directions →
//! large angular variance → large ABOF.  Outliers on the periphery see all training
//! data from roughly one direction → small ABOF.
//!
//! [`Abod::score`] returns `1.0 / (ABOF + ε)` so that **high score = more anomalous**.

use crate::error::{AnomalyError, AnomalyResult};

// ─── Abod ─────────────────────────────────────────────────────────────────────

/// Exact Angle-Based Outlier Detector.
///
/// Complexity: O(n²·d) per query, where n = #training samples, d = #features.
#[derive(Debug, Clone)]
pub struct Abod {
    data: Vec<f32>,
    n_samples: usize,
    n_features: usize,
    fitted: bool,
}

impl Abod {
    /// Create a new, unfitted ABOD detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            n_samples: 0,
            n_features: 0,
            fitted: false,
        }
    }

    /// Fit: store training data for later scoring.
    ///
    /// Requires at least 2 training samples (need ≥1 pair).
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_samples < 2 {
            return Err(AnomalyError::InsufficientSamples {
                need: 2,
                got: n_samples,
            });
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
        self.data = data.to_vec();
        self.n_samples = n_samples;
        self.n_features = n_features;
        self.fitted = true;
        Ok(())
    }

    /// Compute the ABOD-based anomaly score for query point `x`.
    ///
    /// Returns `1.0 / (ABOF + 1e-10)` — higher score means more anomalous.
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
        let abof = compute_abof(x, &self.data, self.n_samples, self.n_features)?;
        Ok(1.0 / (abof + 1e-10))
    }

    /// Batch scoring over `n_samples` rows packed into `x` (row-major).
    pub fn score_batch(&self, x: &[f32], n_samples: usize) -> AnomalyResult<Vec<f32>> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != n_samples * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * self.n_features,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let row = &x[i * self.n_features..(i + 1) * self.n_features];
            out.push(self.score(row)?);
        }
        Ok(out)
    }
}

impl Default for Abod {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Core computation ─────────────────────────────────────────────────────────

/// Compute ABOF(p) for query `p` against `data` (n × d, row-major).
///
/// Accumulates `sum_f` and `sum_f2` in `f64` to reduce numerical drift.
/// Returns `Var[f(a,b)]` using the computational-variance identity.
fn compute_abof(
    p: &[f32],
    data: &[f32],
    n_samples: usize,
    n_features: usize,
) -> AnomalyResult<f32> {
    let d = n_features;

    // Precompute pa = p - a vectors and ‖pa‖² for all training points.
    let mut pa_vecs: Vec<Vec<f32>> = Vec::with_capacity(n_samples);
    let mut pa_sq: Vec<f32> = Vec::with_capacity(n_samples);
    let mut valid: Vec<bool> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let a = &data[i * d..(i + 1) * d];
        let pa: Vec<f32> = p.iter().zip(a.iter()).map(|(pi, ai)| pi - ai).collect();
        let sq: f32 = pa.iter().map(|v| v * v).sum();
        let is_valid = sq > 1e-20; // skip coincident points
        pa_vecs.push(pa);
        pa_sq.push(sq);
        valid.push(is_valid);
    }

    let mut sum_f = 0.0_f64;
    let mut sum_f2 = 0.0_f64;
    let mut count = 0u64;

    for i in 0..n_samples {
        if !valid[i] {
            continue;
        }
        for j in (i + 1)..n_samples {
            if !valid[j] {
                continue;
            }
            // dot(pa_i, pa_j)
            let dot: f32 = pa_vecs[i]
                .iter()
                .zip(pa_vecs[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            // f = dot / (‖pa_i‖² · ‖pa_j‖²)
            let denom = pa_sq[i] * pa_sq[j];
            let f = (dot as f64) / (denom as f64);
            sum_f += f;
            sum_f2 += f * f;
            count += 1;
        }
    }

    if count == 0 {
        return Err(AnomalyError::InsufficientSamples {
            need: 2,
            got: n_samples,
        });
    }

    let n = count as f64;
    let mean_f = sum_f / n;
    let mean_f2 = sum_f2 / n;
    // Var[f] = E[f²] - E[f]²  (population variance, non-negative by construction)
    let variance = (mean_f2 - mean_f * mean_f).max(0.0);
    Ok(variance as f32)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster_2d(n: usize, cx: f32, cy: f32, r: f32, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut data = Vec::with_capacity(n * 2);
        for _ in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let x = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 * r - r + cx;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let y = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 * r - r + cy;
            data.push(x);
            data.push(y);
        }
        data
    }

    #[test]
    fn test_score_is_finite() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut abod = Abod::new();
        abod.fit(&data, 3, 2)
            .expect("fit should succeed with 3 samples, 2 features");
        let s = abod
            .score(&[0.5, 0.5])
            .expect("score should succeed for interior point");
        assert!(s.is_finite(), "score={s}");
        assert!(s > 0.0, "score must be positive, got {s}");
    }

    #[test]
    fn test_outlier_scores_higher_than_inlier() {
        // 10 inliers near origin, 1 outlier far away
        let mut data = cluster_2d(10, 0.0, 0.0, 0.5, 42);
        data.extend_from_slice(&[20.0_f32, 20.0]);
        let n = 11;
        let mut abod = Abod::new();
        abod.fit(&data, n, 2).expect("fit should succeed");
        let s_inlier = abod
            .score(&[0.1_f32, 0.1])
            .expect("inlier score should succeed");
        let s_outlier = abod
            .score(&[20.0_f32, 20.0])
            .expect("outlier score should succeed");
        assert!(
            s_outlier > s_inlier,
            "outlier({s_outlier}) should score higher than inlier({s_inlier})"
        );
    }

    #[test]
    fn test_cluster_center_has_lowest_score() {
        // Dense 2d cluster; center should have the lowest score (most inlier-like = highest ABOF)
        let data = cluster_2d(20, 0.0, 0.0, 1.0, 7);
        let mut abod = Abod::new();
        abod.fit(&data, 20, 2)
            .expect("fit should succeed with 20 samples");
        let s_center = abod
            .score(&[0.0_f32, 0.0])
            .expect("center score should succeed");
        let s_edge = abod
            .score(&[3.0_f32, 3.0])
            .expect("edge score should succeed");
        // Score of true center < score of far-away point
        assert!(
            s_center < s_edge,
            "center({s_center}) should score lower than far edge({s_edge})"
        );
    }

    #[test]
    fn test_two_inliers_one_outlier() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut abod = Abod::new();
        abod.fit(&data, 4, 2)
            .expect("fit should succeed with 4 samples");
        let s_inlier = abod
            .score(&[0.5_f32, 0.5])
            .expect("inlier score should succeed"); // inside cluster
        let s_outlier = abod
            .score(&[10.0_f32, 10.0])
            .expect("outlier score should succeed"); // far outside
        assert!(
            s_outlier > s_inlier,
            "outlier({s_outlier}) > inlier({s_inlier})"
        );
    }

    #[test]
    fn test_score_batch_consistent() {
        let data = cluster_2d(15, 0.0, 0.0, 1.0, 13);
        let mut abod = Abod::new();
        abod.fit(&data, 15, 2)
            .expect("fit should succeed with 15 samples");

        let queries: Vec<f32> = vec![0.0, 0.0, 5.0, 5.0, -5.0, -5.0];
        let batch_scores = abod
            .score_batch(&queries, 3)
            .expect("batch score should succeed");
        for i in 0..3 {
            let single = abod
                .score(&queries[i * 2..(i + 1) * 2])
                .expect("single score should succeed");
            assert!(
                (batch_scores[i] - single).abs() < 1e-6,
                "batch[{i}]={} vs single={}",
                batch_scores[i],
                single
            );
        }
    }

    #[test]
    fn test_fit_requires_at_least_2_samples() {
        let data: Vec<f32> = vec![1.0, 2.0];
        let mut abod = Abod::new();
        let result = abod.fit(&data, 1, 2);
        assert!(matches!(
            result,
            Err(AnomalyError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_score_before_fit_error() {
        let abod = Abod::new();
        let result = abod.score(&[0.0_f32, 0.0]);
        assert!(matches!(result, Err(AnomalyError::NotFitted)));
    }

    #[test]
    fn test_empty_features_error() {
        let mut abod = Abod::new();
        let result = abod.fit(&[], 3, 0);
        assert!(matches!(
            result,
            Err(AnomalyError::InvalidFeatureCount { .. })
        ));
    }

    #[test]
    fn test_dimension_mismatch_score() {
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut abod = Abod::new();
        abod.fit(&data, 3, 2)
            .expect("fit should succeed with 3 samples and 2 features");
        // query with wrong dimension
        let result = abod.score(&[0.0_f32, 0.0, 0.0]); // 3 features, expected 2
        assert!(matches!(
            result,
            Err(AnomalyError::FeatureCountMismatch { .. })
        ));
    }

    #[test]
    fn test_score_deterministic() {
        let data = cluster_2d(10, 0.0, 0.0, 1.0, 99);
        let mut abod = Abod::new();
        abod.fit(&data, 10, 2)
            .expect("fit should succeed with 10 samples");
        let s1 = abod
            .score(&[2.0_f32, 0.0])
            .expect("first score should be deterministic");
        let s2 = abod
            .score(&[2.0_f32, 0.0])
            .expect("second score should be deterministic");
        assert_eq!(s1, s2, "score should be deterministic");
    }

    #[test]
    fn test_collinear_training_data_handled() {
        // All training data on a line; test that scoring doesn't crash
        let data: Vec<f32> = (0..10).flat_map(|i| vec![i as f32, 0.0_f32]).collect();
        let mut abod = Abod::new();
        abod.fit(&data, 10, 2)
            .expect("fit should succeed for collinear data");
        let s = abod
            .score(&[5.0_f32, 0.0])
            .expect("score on collinear data should not crash");
        assert!(s.is_finite(), "collinear data score={s} should be finite");
    }

    #[test]
    fn test_high_dimensional_abod() {
        // d=10, n=20 — verify scores are finite
        let d = 10_usize;
        let n = 20_usize;
        let mut state = 55u64;
        let data: Vec<f32> = (0..n * d)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32) / (u32::MAX as f32)
            })
            .collect();
        let mut abod = Abod::new();
        abod.fit(&data, n, d)
            .expect("fit should succeed for high-dimensional data");
        let query: Vec<f32> = vec![0.5; d];
        let s = abod
            .score(&query)
            .expect("score should succeed for high-dimensional query");
        assert!(s.is_finite() && s > 0.0, "hd score={s}");
    }

    #[test]
    fn test_inlier_lower_than_outlier_2d_cluster() {
        // 20 inliers in unit square, 1 outlier at (5,5)
        let mut data = cluster_2d(20, 0.5, 0.5, 0.5, 111);
        data.extend_from_slice(&[5.0_f32, 5.0]);
        let n = 21;
        let mut abod = Abod::new();
        abod.fit(&data, n, 2)
            .expect("fit should succeed with 21 samples");

        // Score every inlier
        let max_inlier_score: f32 = (0..20)
            .map(|i| {
                abod.score(&data[i * 2..(i + 1) * 2])
                    .expect("inlier score should succeed")
            })
            .fold(f32::NEG_INFINITY, f32::max);
        let outlier_score = abod
            .score(&[5.0_f32, 5.0])
            .expect("outlier score should succeed");

        assert!(
            outlier_score > max_inlier_score,
            "outlier({outlier_score}) must beat max inlier({max_inlier_score})"
        );
    }

    #[test]
    fn test_default_impl() {
        let abod = Abod::default();
        assert!(!abod.fitted, "fresh ABOD should not be fitted");
    }

    #[test]
    fn test_data_dimension_mismatch_fit() {
        let mut abod = Abod::new();
        // claim 3 samples × 2 features = 6, but only provide 5 values
        let result = abod.fit(&[1.0, 2.0, 3.0, 4.0, 5.0], 3, 2);
        assert!(matches!(
            result,
            Err(AnomalyError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_score_batch_dimension_mismatch() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut abod = Abod::new();
        abod.fit(&data, 3, 2)
            .expect("fit should succeed before testing batch mismatch");
        // Claim 2 samples × 2 features = 4 values, but provide 3
        let result = abod.score_batch(&[0.0_f32, 0.0, 1.0], 2);
        assert!(matches!(
            result,
            Err(AnomalyError::DimensionMismatch { .. })
        ));
    }
}

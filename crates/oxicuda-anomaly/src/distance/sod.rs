//! SOD: Subspace Outlier Degree (Kriegel et al. 2009).
//!
//! Kriegel, H.-P., Kröger, P., Schubert, E. & Zimek, A. (2009).
//! "Outlier Detection in Axis-Parallel Subspaces of High Dimensional Data."
//! PAKDD 2009.
//!
//! # Key idea
//!
//! In high-dimensional spaces, the Euclidean distance becomes less meaningful
//! due to the *concentration of measure* phenomenon.  SOD finds, for each
//! query point `p`, a **shared-nearest-neighbour (SNN) subspace** —
//! the dimensions where the neighbours of `p` have low variance — and then
//! computes the outlier degree as the distance from `p` to the mean of its
//! SNN set *in that subspace*, normalised by the subspace variance:
//!
//! ```text
//! SOD(p) = d(p, μ_SNN) / sqrt(Var_SNN_subspace)
//! ```
//!
//! Dimensions are selected as those where the variance among the SNN members
//! falls below a threshold `α · global_var_j` (relative threshold per feature).
//!
//! # Complexity
//!
//! O(n · k · d) per query (k-NN search + SNN variance computation).

use crate::error::{AnomalyError, AnomalyResult};

// ─── SodConfig ────────────────────────────────────────────────────────────────

/// Configuration for SOD.
#[derive(Debug, Clone, Copy)]
pub struct SodConfig {
    /// Number of nearest neighbours to use for the SNN subspace.
    pub k: usize,
    /// Relative variance threshold for subspace selection: a dimension `j` is
    /// included if `Var_SNN_j < alpha * Var_all_j`.  Typical: 0.5.
    pub alpha: f32,
}

impl Default for SodConfig {
    fn default() -> Self {
        Self { k: 10, alpha: 0.8 }
    }
}

// ─── Sod ──────────────────────────────────────────────────────────────────────

/// Subspace Outlier Degree detector.
#[derive(Debug, Clone)]
pub struct Sod {
    config: SodConfig,
    data: Vec<f32>,
    n_samples: usize,
    n_features: usize,
    /// Per-feature global variance (computed on fit data).
    global_var: Vec<f32>,
    fitted: bool,
}

impl Sod {
    /// Create a new, unfitted SOD detector.
    ///
    /// # Errors
    /// - `AnomalyError::InvalidK` if `k == 0`.
    pub fn new(config: SodConfig) -> AnomalyResult<Self> {
        if config.k == 0 {
            return Err(AnomalyError::InvalidK { k: 0 });
        }
        Ok(Self {
            config,
            data: Vec::new(),
            n_samples: 0,
            n_features: 0,
            global_var: Vec::new(),
            fitted: false,
        })
    }

    /// Fit: store training data and compute per-feature global variances.
    ///
    /// # Errors
    /// - `AnomalyError::InsufficientSamples` if `n_samples < k + 1`.
    /// - `AnomalyError::InvalidFeatureCount` if `n_features == 0`.
    /// - `AnomalyError::DimensionMismatch` if `data.len() != n_samples * n_features`.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if n_samples < self.config.k + 1 {
            return Err(AnomalyError::InsufficientSamples {
                need: self.config.k + 1,
                got: n_samples,
            });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }
        // Compute per-feature global variance.
        let mut global_mean = vec![0.0_f64; n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                global_mean[j] += data[i * n_features + j] as f64;
            }
        }
        for m in &mut global_mean {
            *m /= n_samples as f64;
        }
        let mut global_var = vec![0.0_f32; n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                let d = data[i * n_features + j] as f64 - global_mean[j];
                global_var[j] += (d * d) as f32;
            }
        }
        for v in &mut global_var {
            *v /= n_samples as f32;
        }
        self.data = data.to_vec();
        self.n_samples = n_samples;
        self.n_features = n_features;
        self.global_var = global_var;
        self.fitted = true;
        Ok(())
    }

    /// Compute the SOD score for a single query point `p`.
    ///
    /// Returns the normalised subspace distance; **higher = more anomalous**.
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
        let k = self.config.k;
        let d = self.n_features;

        // Step 1: find k nearest neighbours of p in the full space.
        let mut dists: Vec<(f32, usize)> = (0..self.n_samples)
            .map(|i| {
                let row = &self.data[i * d..(i + 1) * d];
                let sq: f32 = p
                    .iter()
                    .zip(row.iter())
                    .map(|(&a, &b)| {
                        let diff = a - b;
                        diff * diff
                    })
                    .sum();
                (sq, i)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let knn: Vec<usize> = dists.iter().take(k).map(|&(_, i)| i).collect();

        // Step 2: compute per-feature mean and variance over the SNN set.
        let mut snn_mean = vec![0.0_f64; d];
        for &idx in &knn {
            for (j, m) in snn_mean.iter_mut().enumerate() {
                *m += self.data[idx * d + j] as f64;
            }
        }
        for m in &mut snn_mean {
            *m /= k as f64;
        }
        let mut snn_var = vec![0.0_f32; d];
        for &idx in &knn {
            for (j, v) in snn_var.iter_mut().enumerate() {
                let diff = self.data[idx * d + j] as f64 - snn_mean[j];
                *v += (diff * diff) as f32;
            }
        }
        for v in &mut snn_var {
            *v /= k as f32;
        }

        // Step 3: select subspace dimensions where snn_var[j] < alpha * global_var[j].
        let alpha = self.config.alpha;
        let subspace: Vec<usize> = (0..d)
            .filter(|&j| {
                let gv = self.global_var[j];
                if gv < 1e-12 {
                    false
                } else {
                    snn_var[j] < alpha * gv
                }
            })
            .collect();

        if subspace.is_empty() {
            // No discriminative subspace found; fall back to full-space distance
            // to the SNN mean, normalised by total global variance.
            let total_var: f32 = self.global_var.iter().sum();
            let dist_sq: f32 = (0..d)
                .map(|j| {
                    let diff = p[j] as f64 - snn_mean[j];
                    (diff * diff) as f32
                })
                .sum();
            let normaliser = total_var.sqrt().max(1e-10);
            return Ok((dist_sq.sqrt() / normaliser).max(0.0));
        }

        // Step 4: distance from p to SNN mean in the selected subspace,
        //         normalised by the sqrt of the sum of snn variances.
        let mut dist_sq = 0.0_f32;
        let mut var_sum = 0.0_f32;
        for &j in &subspace {
            let diff = p[j] as f64 - snn_mean[j];
            dist_sq += (diff * diff) as f32;
            var_sum += snn_var[j];
        }
        let normaliser = var_sum.sqrt().max(1e-10);
        Ok((dist_sq.sqrt() / normaliser).max(0.0))
    }

    /// Score a full `[n_queries × n_features]` row-major batch.
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

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data() -> (Vec<f32>, usize, usize) {
        // 15 points tightly clustered in 2D, plus 1 outlier far away.
        let mut data = Vec::new();
        for i in 0..15_usize {
            data.push((i % 5) as f32 * 0.1);
            data.push((i / 5) as f32 * 0.1);
        }
        data.push(100.0_f32);
        data.push(100.0_f32);
        (data, 16, 2)
    }

    // ── 1. new: k=0 returns error ─────────────────────────────────────────────
    #[test]
    fn new_k_zero_error() {
        assert!(Sod::new(SodConfig { k: 0, alpha: 0.8 }).is_err());
    }

    // ── 2. fit: insufficient samples error ───────────────────────────────────
    #[test]
    fn fit_insufficient_samples() {
        let mut det = Sod::new(SodConfig { k: 10, alpha: 0.8 })
            .expect("Sod::new with valid config should succeed");
        let data = vec![0.0_f32; 5 * 2]; // only 5 samples, need 11
        assert!(det.fit(&data, 5, 2).is_err());
    }

    // ── 3. fit: dimension mismatch error ─────────────────────────────────────
    #[test]
    fn fit_dim_mismatch() {
        let mut det =
            Sod::new(SodConfig::default()).expect("Sod::new with default config should succeed");
        let data = vec![0.0_f32; 11]; // claims 5 × 2 but wrong
        assert!(det.fit(&data, 5, 2).is_err());
    }

    // ── 4. score: NotFitted error ─────────────────────────────────────────────
    #[test]
    fn score_not_fitted_error() {
        let det =
            Sod::new(SodConfig::default()).expect("Sod::new with default config should succeed");
        assert!(det.score(&[0.0, 0.0]).is_err());
    }

    // ── 5. score: dimension mismatch error ───────────────────────────────────
    #[test]
    fn score_dim_mismatch_error() {
        let (data, n, d) = make_data();
        let mut det = Sod::new(SodConfig { k: 5, alpha: 0.8 })
            .expect("Sod::new with valid config should succeed");
        det.fit(&data, n, d).expect("SOD fit should succeed");
        assert!(det.score(&[0.0_f32]).is_err()); // wrong dim
    }

    // ── 6. Outlier scores higher than inlier ─────────────────────────────────
    #[test]
    fn outlier_scores_higher() {
        let (data, n, d) = make_data();
        let mut det = Sod::new(SodConfig { k: 5, alpha: 0.8 })
            .expect("Sod::new with valid config should succeed");
        det.fit(&data, n, d).expect("SOD fit should succeed");
        let s_in = det
            .score(&[0.2_f32, 0.1])
            .expect("inlier score should succeed");
        let s_out = det
            .score(&[100.0_f32, 100.0])
            .expect("outlier score should succeed");
        assert!(s_out > s_in, "outlier={s_out} should > inlier={s_in}");
    }

    // ── 7. Score is non-negative and finite ───────────────────────────────────
    #[test]
    fn score_non_negative_finite() {
        let (data, n, d) = make_data();
        let mut det = Sod::new(SodConfig { k: 4, alpha: 0.7 })
            .expect("Sod::new with valid config should succeed");
        det.fit(&data, n, d).expect("SOD fit should succeed");
        for i in 0..n {
            let p = &data[i * d..(i + 1) * d];
            let s = det
                .score(p)
                .expect("score should succeed for training point");
            assert!(s.is_finite() && s >= 0.0, "s={s} for point {i}");
        }
    }

    // ── 8. Batch scoring is consistent with individual scoring ────────────────
    #[test]
    fn batch_consistent_with_individual() {
        let (data, n, d) = make_data();
        let mut det = Sod::new(SodConfig { k: 3, alpha: 0.8 })
            .expect("Sod::new with valid config should succeed");
        det.fit(&data, n, d).expect("SOD fit should succeed");
        let batch = det
            .score_batch(&data, n)
            .expect("batch score should succeed after fit");
        for (i, &bs) in batch.iter().enumerate() {
            let p = &data[i * d..(i + 1) * d];
            let s = det
                .score(p)
                .expect("individual score should match batch score");
            assert!((bs - s).abs() < 1e-5, "batch[{i}]={bs} vs {s}");
        }
    }

    // ── 9. Works in high dimensions ────────────────────────────────────────────
    #[test]
    fn works_high_dimensional() {
        let n = 25_usize;
        let d = 20_usize;
        let mut data: Vec<f32> = (0..n * d).map(|i| (i % 7) as f32 * 0.1).collect();
        // Make last point an outlier.
        for v in &mut data[(n - 1) * d..] {
            *v = 50.0;
        }
        let mut det = Sod::new(SodConfig { k: 5, alpha: 0.9 })
            .expect("Sod::new with valid config should succeed");
        det.fit(&data, n, d).expect("SOD fit should succeed");
        let s_normal = det
            .score(&data[..d])
            .expect("score on normal point should succeed");
        let s_outlier = det
            .score(&data[(n - 1) * d..])
            .expect("score on outlier point should succeed");
        assert!(
            s_outlier >= s_normal,
            "outlier={s_outlier} should >= inlier={s_normal}"
        );
    }

    // ── 10. Alpha = 0 (no subspace): falls back to full distance ──────────────
    #[test]
    fn alpha_zero_fallback() {
        let (data, n, d) = make_data();
        let mut det =
            Sod::new(SodConfig { k: 4, alpha: 0.0 }).expect("Sod::new with alpha=0 should succeed");
        det.fit(&data, n, d).expect("SOD fit should succeed");
        let s = det
            .score(&[0.2_f32, 0.1])
            .expect("alpha=0 fallback score should succeed");
        assert!(s.is_finite() && s >= 0.0, "s={s}");
    }
}

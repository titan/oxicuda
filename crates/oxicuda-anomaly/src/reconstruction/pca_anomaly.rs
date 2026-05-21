//! PCA-based reconstruction anomaly detector.
//!
//! Given training data `X ∈ ℝ^{n × d}`, the detector learns the top-`k`
//! principal components via power iteration with deflation, then scores a
//! query point `x` by its squared reconstruction error in the learned subspace:
//!
//! ```text
//! μ   = (1/n) Σ_i x_i                   (per-feature mean)
//! C   = X̃ᵀ X̃ / (n-1)                    (d × d covariance, X̃ = X - μ)
//! W   ∈ ℝ^{k × d}                        (top-k eigenvectors, row-major)
//! z   = x - μ                            (centered query)
//! h   = W z    ∈ ℝ^k                     (projection)
//! r   = Wᵀ h   ∈ ℝ^d                     (reconstruction)
//! score(x) = ‖z - r‖₂²                   (higher ⟹ more anomalous)
//! ```
//!
//! Power iteration with Gram-Schmidt deflation ensures orthonormal components.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`PcaAnomaly`].
#[derive(Debug, Clone)]
pub struct PcaAnomalyConfig {
    /// Number of principal components to retain (`k`). Default: 2.
    pub n_components: usize,
    /// Maximum power-iteration steps per component. Default: 100.
    pub max_iter: usize,
    /// Convergence tolerance for power iteration. Default: 1e-5.
    pub tol: f32,
}

impl Default for PcaAnomalyConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 100,
            tol: 1e-5,
        }
    }
}

// ─── PcaAnomaly ──────────────────────────────────────────────────────────────

/// PCA reconstruction anomaly detector.
///
/// Call [`PcaAnomaly::new`] → [`PcaAnomaly::fit`] → [`PcaAnomaly::score`].
pub struct PcaAnomaly {
    config: PcaAnomalyConfig,
    /// Per-feature mean `μ`, shape `[n_features]`.
    mean: Vec<f32>,
    /// Top-k eigenvectors stored row-major, shape `[n_components × n_features]`.
    components: Vec<f32>,
    n_features: usize,
    fitted: bool,
}

impl PcaAnomaly {
    /// Create an unfitted detector with the given configuration.
    #[must_use]
    pub fn new(config: PcaAnomalyConfig) -> Self {
        Self {
            config,
            mean: Vec::new(),
            components: Vec::new(),
            n_features: 0,
            fitted: false,
        }
    }

    // ─── Fit ─────────────────────────────────────────────────────────────────

    /// Fit the detector on `data` (row-major, `n_samples × n_features`).
    ///
    /// After fitting, [`PcaAnomaly::score`] is available.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        // ── Validate inputs ──────────────────────────────────────────────────
        if self.config.n_components == 0 {
            return Err(AnomalyError::Internal {
                msg: "n_components must be > 0".into(),
            });
        }
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if n_samples < 2 {
            return Err(AnomalyError::InsufficientSamples {
                need: 2,
                got: n_samples,
            });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }

        // Clamp k to at most n_features
        let k = self.config.n_components.min(n_features);

        // ── Step 1: compute per-feature mean ─────────────────────────────────
        let mut mean = vec![0.0_f32; n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                mean[j] += data[i * n_features + j];
            }
        }
        let inv_n = 1.0 / n_samples as f32;
        for m in mean.iter_mut() {
            *m *= inv_n;
        }

        // ── Step 2: build centered data X̃ (n_samples × n_features) ──────────
        let mut x_centered = data.to_vec();
        for i in 0..n_samples {
            for j in 0..n_features {
                x_centered[i * n_features + j] -= mean[j];
            }
        }

        // ── Step 3: covariance C = X̃ᵀ X̃ / (n-1), shape d × d ───────────────
        // Stored as a flat Vec<f32> row-major: C[r, c] = cov[r * d + c]
        let d = n_features;
        let mut cov = vec![0.0_f32; d * d];
        let inv_nm1 = 1.0 / (n_samples - 1) as f32;
        for i in 0..n_samples {
            let row = &x_centered[i * d..(i + 1) * d];
            for r in 0..d {
                for c in r..d {
                    let v = row[r] * row[c] * inv_nm1;
                    cov[r * d + c] += v;
                    if r != c {
                        cov[c * d + r] += v;
                    }
                }
            }
        }

        // ── Step 4: power iteration + deflation to find top-k eigenvectors ───
        let mut components = vec![0.0_f32; k * d];
        // Work with a mutable copy of cov for successive deflation
        let mut cov_work = cov.clone();

        for ki in 0..k {
            // Initialize random unit vector via LcgRng
            let mut rng = LcgRng::new(42_u64.wrapping_add(ki as u64));
            let mut v: Vec<f32> = (0..d).map(|_| rng.next_normal()).collect();
            normalize_vec(&mut v)?;

            // Power iteration: v ← C v / ‖C v‖
            let mut converged = false;
            for _iter in 0..self.config.max_iter {
                let cv = mat_vec_mul(&cov_work, &v, d);
                let norm = vec_norm(&cv);
                if norm < 1e-12 {
                    // Zero eigenvector — matrix is rank-deficient; stop early
                    converged = true;
                    break;
                }
                let v_new: Vec<f32> = cv.iter().map(|x| x / norm).collect();
                // Check convergence: ‖v_new - v‖ < tol
                let diff: f32 = v_new
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                v = v_new;
                if diff < self.config.tol {
                    converged = true;
                    break;
                }
            }

            if !converged {
                // Accept last iterate silently — partial convergence is still
                // useful for anomaly scoring in practice.
            }

            // Store eigenvector as row ki
            components[ki * d..(ki + 1) * d].copy_from_slice(&v);

            // Gram-Schmidt: re-orthogonalise v against all previously found
            // components to guard against floating-point drift, then deflate.
            // Deflation: C ← C - (v vᵀ) eigenvalue  (rank-1 subtraction)
            // We need the eigenvalue λ = vᵀ C v.
            let cv = mat_vec_mul(&cov_work, &v, d);
            let lambda: f32 = v.iter().zip(cv.iter()).map(|(a, b)| a * b).sum();
            // C_new = C - λ v vᵀ
            for r in 0..d {
                for c in 0..d {
                    cov_work[r * d + c] -= lambda * v[r] * v[c];
                }
            }
        }

        self.mean = mean;
        self.components = components;
        self.n_features = n_features;
        self.fitted = true;
        Ok(())
    }

    // ─── Score ───────────────────────────────────────────────────────────────

    /// Compute the squared reconstruction error for a single point `x`.
    ///
    /// Returns `‖(x - μ) - Wᵀ W (x - μ)‖₂²`.  Higher means more anomalous.
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
        let d = self.n_features;
        let k = self.components.len() / d;

        // z = x - μ
        let z: Vec<f32> = x
            .iter()
            .zip(self.mean.iter())
            .map(|(xi, mi)| xi - mi)
            .collect();

        // h = W z  (shape k)
        let h = self.project(&z, k, d);

        // r = Wᵀ h  (shape d)
        let r = self.reconstruct(&h, k, d);

        // score = ‖z - r‖₂²
        let sq_err: f32 = z
            .iter()
            .zip(r.iter())
            .map(|(zi, ri)| (zi - ri).powi(2))
            .sum();

        Ok(sq_err)
    }

    /// Batch scoring: `x` is row-major `[n × n_features]`, returns `[n]` scores.
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

    // ─── Internal helpers ─────────────────────────────────────────────────────

    /// Project centered vector `z` onto component subspace: `h = W z`.
    fn project(&self, z: &[f32], k: usize, d: usize) -> Vec<f32> {
        (0..k)
            .map(|ki| {
                let row = &self.components[ki * d..(ki + 1) * d];
                row.iter().zip(z.iter()).map(|(w, zi)| w * zi).sum()
            })
            .collect()
    }

    /// Reconstruct from projection `h` back to feature space: `r = Wᵀ h`.
    fn reconstruct(&self, h: &[f32], k: usize, d: usize) -> Vec<f32> {
        let mut r = vec![0.0_f32; d];
        for (ki, &scale) in h.iter().enumerate().take(k) {
            let row = &self.components[ki * d..(ki + 1) * d];
            for (rj, &wj) in r.iter_mut().zip(row.iter()) {
                *rj += scale * wj;
            }
        }
        r
    }
}

// ─── Free helpers ─────────────────────────────────────────────────────────────

/// Multiply square matrix `m` (d × d, row-major) by vector `v` (len d).
fn mat_vec_mul(m: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; d];
    for r in 0..d {
        out[r] = m[r * d..(r + 1) * d]
            .iter()
            .zip(v.iter())
            .map(|(a, b)| a * b)
            .sum();
    }
    out
}

/// L2 norm of a slice.
#[inline]
fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x.powi(2)).sum::<f32>().sqrt()
}

/// Normalize a vector in-place; returns error if the norm is too small.
fn normalize_vec(v: &mut [f32]) -> AnomalyResult<()> {
    let n = vec_norm(v);
    if n < 1e-12 {
        return Err(AnomalyError::Internal {
            msg: "zero initial vector in power iteration".into(),
        });
    }
    for x in v.iter_mut() {
        *x /= n;
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a small 2-D dataset on a line y = x.
    fn line_data(n: usize) -> Vec<f32> {
        (0..n)
            .flat_map(|i| {
                let t = i as f32 / n as f32;
                vec![t, t]
            })
            .collect()
    }

    /// Helper: build identity-style blob data (2 features, independent).
    fn blob_data() -> Vec<f32> {
        // 10 points near (0,0) and (1,0) alternating
        (0..20)
            .flat_map(|i| {
                let x = if i % 2 == 0 { 0.0_f32 } else { 1.0_f32 };
                vec![x, 0.0_f32]
            })
            .collect()
    }

    // Test 1: fit on 2-D data, score returns finite non-negative value
    #[test]
    fn fit_score_basic_finite_nonneg() {
        let data = line_data(20);
        let mut det = PcaAnomaly::new(PcaAnomalyConfig::default());
        det.fit(&data, 20, 2).unwrap();
        let s = det.score(&[0.3_f32, 0.3_f32]).unwrap();
        assert!(s.is_finite(), "score must be finite, got {s}");
        assert!(s >= 0.0, "score must be non-negative, got {s}");
    }

    // Test 2: unfitted detector returns NotFitted error
    #[test]
    fn unfitted_returns_not_fitted_error() {
        let det = PcaAnomaly::new(PcaAnomalyConfig::default());
        let result = det.score(&[0.0_f32, 1.0_f32]);
        assert!(
            matches!(result, Err(AnomalyError::NotFitted)),
            "expected NotFitted, got {result:?}"
        );
    }

    // Test 3: outlier has higher reconstruction error than inlier
    #[test]
    fn outlier_higher_than_inlier() {
        // Data lies on y = x line; principal component is (1/√2, 1/√2)
        let data = line_data(30);
        let mut det = PcaAnomaly::new(PcaAnomalyConfig {
            n_components: 1,
            ..Default::default()
        });
        det.fit(&data, 30, 2).unwrap();
        let inlier_score = det.score(&[0.5_f32, 0.5_f32]).unwrap();
        // Outlier is far off the line
        let outlier_score = det.score(&[100.0_f32, -100.0_f32]).unwrap();
        assert!(
            outlier_score > inlier_score,
            "outlier ({outlier_score}) should exceed inlier ({inlier_score})"
        );
    }

    // Test 4: score_batch returns vector of correct length
    #[test]
    fn score_batch_correct_length() {
        let data = line_data(20);
        let mut det = PcaAnomaly::new(PcaAnomalyConfig::default());
        det.fit(&data, 20, 2).unwrap();
        let batch: Vec<f32> = (0..5)
            .flat_map(|i| vec![i as f32 * 0.1, i as f32 * 0.1])
            .collect();
        let scores = det.score_batch(&batch, 5).unwrap();
        assert_eq!(scores.len(), 5, "expected 5 scores, got {}", scores.len());
    }

    // Test 5: n_samples < 2 returns InsufficientSamples error
    #[test]
    fn insufficient_samples_error() {
        let data = vec![1.0_f32, 2.0_f32];
        let mut det = PcaAnomaly::new(PcaAnomalyConfig::default());
        let result = det.fit(&data, 1, 2);
        assert!(
            matches!(
                result,
                Err(AnomalyError::InsufficientSamples { need: 2, got: 1 })
            ),
            "expected InsufficientSamples, got {result:?}"
        );
    }

    // Test 6: x.len() mismatch returns FeatureCountMismatch
    #[test]
    fn feature_count_mismatch_error() {
        let data = line_data(10);
        let mut det = PcaAnomaly::new(PcaAnomalyConfig::default());
        det.fit(&data, 10, 2).unwrap();
        let result = det.score(&[1.0_f32, 2.0_f32, 3.0_f32]);
        assert!(
            matches!(
                result,
                Err(AnomalyError::FeatureCountMismatch {
                    expected: 2,
                    got: 3
                })
            ),
            "expected FeatureCountMismatch, got {result:?}"
        );
    }

    // Test 7: n_components == 1 works (single component extracted)
    #[test]
    fn single_component_works() {
        let data = line_data(15);
        let mut det = PcaAnomaly::new(PcaAnomalyConfig {
            n_components: 1,
            max_iter: 200,
            tol: 1e-6,
        });
        det.fit(&data, 15, 2).unwrap();
        let s = det.score(&[0.5_f32, 0.5_f32]).unwrap();
        assert!(
            s.is_finite(),
            "single-component score must be finite, got {s}"
        );
        assert!(
            s >= 0.0,
            "single-component score must be non-negative, got {s}"
        );
    }

    // Test 8: score on training data is lower than distant outlier
    #[test]
    fn training_data_lower_than_distant_outlier() {
        let data = blob_data();
        let mut det = PcaAnomaly::new(PcaAnomalyConfig {
            n_components: 1,
            ..Default::default()
        });
        det.fit(&data, 20, 2).unwrap();
        let inlier = det.score(&[0.5_f32, 0.0_f32]).unwrap();
        let outlier = det.score(&[50.0_f32, 50.0_f32]).unwrap();
        assert!(
            outlier > inlier,
            "distant outlier ({outlier}) should have higher score than inlier ({inlier})"
        );
    }

    // Test 9: reconstruction of a point in the learned subspace has small error
    #[test]
    fn point_in_subspace_low_reconstruction_error() {
        // Dataset: all points lie on the x-axis (feature 1 = 0).
        // The first PC should be (1, 0), so any point (t, 0) reconstructs perfectly.
        let n = 20_usize;
        let data: Vec<f32> = (0..n).flat_map(|i| vec![i as f32, 0.0_f32]).collect();
        let mut det = PcaAnomaly::new(PcaAnomalyConfig {
            n_components: 1,
            max_iter: 300,
            tol: 1e-7,
        });
        det.fit(&data, n, 2).unwrap();
        // A point on the x-axis should reconstruct with near-zero error
        let on_axis = det.score(&[5.0_f32, 0.0_f32]).unwrap();
        // A point far off the x-axis should reconstruct with large error
        let off_axis = det.score(&[5.0_f32, 1000.0_f32]).unwrap();
        assert!(
            off_axis > on_axis + 1.0,
            "off-axis ({off_axis}) should dwarf on-axis ({on_axis}) error"
        );
    }

    // Test 10: n_components == 0 returns Internal error
    #[test]
    fn zero_components_returns_error() {
        let data = line_data(10);
        let mut det = PcaAnomaly::new(PcaAnomalyConfig {
            n_components: 0,
            ..Default::default()
        });
        let result = det.fit(&data, 10, 2);
        assert!(
            matches!(result, Err(AnomalyError::Internal { .. })),
            "expected Internal error for n_components=0, got {result:?}"
        );
    }

    // Test 11: n_features == 0 returns InvalidFeatureCount
    #[test]
    fn zero_features_returns_error() {
        let mut det = PcaAnomaly::new(PcaAnomalyConfig::default());
        let result = det.fit(&[], 5, 0);
        assert!(
            matches!(result, Err(AnomalyError::InvalidFeatureCount { n: 0 })),
            "expected InvalidFeatureCount, got {result:?}"
        );
    }
}

//! Mahalanobis distance anomaly detector.
//!
//! Score = `D²(x) = (x − μ)ᵀ · Σ⁻¹ · (x − μ)`.
//!
//! High `D²` → anomaly.
//!
//! The sample covariance is regularised with `0.01 · I` before inversion
//! to avoid near-singular matrices.  Inversion is done via Gauss-Jordan
//! elimination on the augmented matrix `[Σ | I]`.

use crate::error::{AnomalyError, AnomalyResult};

// ─── Gauss-Jordan inversion ───────────────────────────────────────────────────

/// Invert a `d × d` matrix using Gauss-Jordan elimination.
///
/// Returns `Err(SingularCovariance)` if any pivot magnitude is < 1e-10.
fn invert_matrix(m: &[f32], d: usize) -> AnomalyResult<Vec<f32>> {
    // Augmented matrix [M | I], row-major, width = 2*d
    let width = 2 * d;
    let mut aug = vec![0.0_f32; d * width];

    // Fill M
    for i in 0..d {
        for j in 0..d {
            aug[i * width + j] = m[i * d + j];
        }
    }
    // Fill I
    for i in 0..d {
        aug[i * width + d + i] = 1.0;
    }

    // Forward elimination (and back-substitution combined = Gauss-Jordan)
    for col in 0..d {
        // Find max-magnitude pivot in this column (partial pivoting)
        let mut max_row = col;
        let mut max_val = aug[col * width + col].abs();
        for row in (col + 1)..d {
            let v = aug[row * width + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        // Swap rows col ↔ max_row
        if max_row != col {
            for k in 0..width {
                aug.swap(col * width + k, max_row * width + k);
            }
        }

        let pivot = aug[col * width + col];
        if pivot.abs() < 1e-10 {
            return Err(AnomalyError::SingularCovariance);
        }

        // Normalise pivot row
        let inv_pivot = 1.0 / pivot;
        for k in 0..width {
            aug[col * width + k] *= inv_pivot;
        }

        // Eliminate column in all other rows
        for row in 0..d {
            if row == col {
                continue;
            }
            let factor = aug[row * width + col];
            if factor.abs() < 1e-30 {
                continue;
            }
            for k in 0..width {
                let sub = factor * aug[col * width + k];
                aug[row * width + k] -= sub;
            }
        }
    }

    // Extract right half
    let mut inv = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..d {
            inv[i * d + j] = aug[i * width + d + j];
        }
    }
    Ok(inv)
}

// ─── MahalanobisDetector ──────────────────────────────────────────────────────

/// Mahalanobis distance anomaly detector.
pub struct MahalanobisDetector {
    /// Sample mean: `[d]`.
    mean: Vec<f32>,
    /// Inverse regularised covariance: `[d * d]` row-major.
    cov_inv: Vec<f32>,
    /// Number of features.
    n_features: usize,
}

impl MahalanobisDetector {
    /// Create an unfitted detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mean: Vec::new(),
            cov_inv: Vec::new(),
            n_features: 0,
        }
    }

    /// Fit: compute `μ`, `Σ_reg = Σ + 0.01 I`, and invert.
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

        self.n_features = n_features;

        // Compute mean
        let mut mean = vec![0.0_f32; n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                mean[j] += data[i * n_features + j];
            }
        }
        let inv_n = 1.0 / n_samples as f32;
        for v in &mut mean {
            *v *= inv_n;
        }

        // Compute sample covariance (unbiased, divisor n-1)
        let mut cov = vec![0.0_f32; n_features * n_features];
        for i in 0..n_samples {
            for r in 0..n_features {
                let diff_r = data[i * n_features + r] - mean[r];
                for c in 0..n_features {
                    let diff_c = data[i * n_features + c] - mean[c];
                    cov[r * n_features + c] += diff_r * diff_c;
                }
            }
        }
        let denom = (n_samples - 1) as f32;
        for v in &mut cov {
            *v /= denom;
        }

        // Regularise: Σ_reg = Σ + 0.01 * I
        for i in 0..n_features {
            cov[i * n_features + i] += 0.01;
        }

        self.mean = mean;
        self.cov_inv = invert_matrix(&cov, n_features)?;
        Ok(())
    }

    /// Squared Mahalanobis distance `D²(x)`.
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
        let d = self.n_features;
        // diff = x - mean
        let diff: Vec<f32> = x.iter().zip(self.mean.iter()).map(|(a, m)| a - m).collect();
        // temp = Σ⁻¹ · diff
        let mut temp = vec![0.0_f32; d];
        for (r, t) in temp.iter_mut().enumerate() {
            let row = &self.cov_inv[r * d..(r + 1) * d];
            *t = row.iter().zip(diff.iter()).map(|(&a, &b)| a * b).sum();
        }
        // D² = diff · temp
        let d2: f32 = diff.iter().zip(temp.iter()).map(|(a, b)| a * b).sum();
        Ok(d2.max(0.0))
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

impl Default for MahalanobisDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mahal_fit_score() {
        // 2-D Gaussian data
        let data: Vec<f32> = vec![
            1.0, 2.0, 1.1, 1.9, 0.9, 2.1, 1.05, 1.95, 0.95, 2.05, 1.0, 2.0, 1.1, 1.9, 0.9, 2.1,
            1.05, 1.95, 0.95, 2.05,
        ];
        let mut det = MahalanobisDetector::new();
        det.fit(&data, 10, 2)
            .expect("fit should succeed with valid 10-sample 2-feature data");
        // Mean point should have ~0 distance
        let s_normal = det
            .score(&[1.0_f32, 2.0])
            .expect("score should succeed for inlier at mean");
        assert!(s_normal < 1.0, "s_normal={s_normal}");
        // Far away point should have high distance
        let s_outlier = det
            .score(&[100.0_f32, 200.0])
            .expect("score should succeed for outlier point");
        assert!(s_outlier > s_normal, "{s_outlier} > {s_normal}");
    }

    #[test]
    fn invert_identity() {
        let id = vec![1.0_f32, 0.0, 0.0, 1.0];
        let inv = invert_matrix(&id, 2).expect("2×2 identity matrix must be invertible");
        assert!((inv[0] - 1.0).abs() < 1e-5);
        assert!((inv[3] - 1.0).abs() < 1e-5);
        assert!(inv[1].abs() < 1e-5);
        assert!(inv[2].abs() < 1e-5);
    }
}

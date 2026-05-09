//! Accuracy surrogate predictor for NAS.
//!
//! Implements two complementary models:
//! - [`KnnAccuracyPredictor`] — k-nearest-neighbour regression on
//!   architecture feature vectors. Useful with small training sets and
//!   strong feature engineering.
//! - [`RbfAccuracyPredictor`] — radial-basis-function (RBF) regression
//!   `f(x) = Σ_i α_i · exp(-‖x − x_i‖² / (2σ²))`. Equivalent to a Gaussian
//!   kernel ridge regressor with closed-form fit.
//!
//! The accuracy returned is in `[0, 1]` (clamped at predict time).

use crate::error::{NasError, NasResult};
use crate::predictor::predictor_io::{ArchFeatures, LayerSpec};

/// k-nearest-neighbour accuracy regressor.
#[derive(Debug, Clone)]
pub struct KnnAccuracyPredictor {
    /// Stored architecture feature vectors.
    pub samples: Vec<Vec<f32>>,
    /// Stored accuracies, parallel to `samples`.
    pub accuracies: Vec<f32>,
    /// Number of neighbours.
    pub k: usize,
}

impl KnnAccuracyPredictor {
    /// New empty predictor.
    ///
    /// # Errors
    /// [`NasError::InvalidNumOps`] if `k == 0`.
    pub fn new(k: usize) -> NasResult<Self> {
        if k == 0 {
            return Err(NasError::InvalidNumOps);
        }
        Ok(Self {
            samples: Vec::new(),
            accuracies: Vec::new(),
            k,
        })
    }

    /// Add a training sample.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `features.is_empty()`.
    /// - [`NasError::DimensionMismatch`] if a stored sample exists with a
    ///   different feature length.
    /// - [`NasError::NanInArchParams`] if `accuracy` is non-finite or outside `[0, 1]`.
    pub fn add(&mut self, features: Vec<f32>, accuracy: f32) -> NasResult<()> {
        if features.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        if let Some(s) = self.samples.first() {
            if s.len() != features.len() {
                return Err(NasError::DimensionMismatch {
                    expected: s.len(),
                    got: features.len(),
                });
            }
        }
        if !accuracy.is_finite() || !(0.0..=1.0).contains(&accuracy) {
            return Err(NasError::NanInArchParams);
        }
        self.samples.push(features);
        self.accuracies.push(accuracy);
        Ok(())
    }

    /// Number of stored training samples.
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Predict accuracy of a candidate architecture.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if no training samples were added.
    /// - [`NasError::DimensionMismatch`] if features don't match training dim.
    pub fn predict(&self, layers: &[LayerSpec]) -> NasResult<f32> {
        if self.samples.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        let f = ArchFeatures::from_layers(layers)?;
        if f.dim() != self.samples[0].len() {
            return Err(NasError::DimensionMismatch {
                expected: self.samples[0].len(),
                got: f.dim(),
            });
        }
        // Compute distances and keep the k smallest.
        let mut idx_dist: Vec<(usize, f32)> = self
            .samples
            .iter()
            .enumerate()
            .map(|(i, s)| (i, l2_distance(&f.data, s)))
            .collect();
        idx_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.k.min(idx_dist.len());
        let mut sum_w = 0.0_f32;
        let mut sum_acc = 0.0_f32;
        for &(i, d) in &idx_dist[..k] {
            // Inverse-distance weighting with epsilon to avoid div-by-zero.
            let w = 1.0_f32 / (d + 1e-6);
            sum_w += w;
            sum_acc += w * self.accuracies[i];
        }
        Ok((sum_acc / sum_w).clamp(0.0, 1.0))
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Radial-basis-function accuracy regressor `f(x) = Σ_i α_i · exp(-‖x − x_i‖² / (2σ²))`.
#[derive(Debug, Clone)]
pub struct RbfAccuracyPredictor {
    /// Centres (training feature vectors).
    pub centres: Vec<Vec<f32>>,
    /// RBF coefficients `α_i`.
    pub alpha: Vec<f32>,
    /// Bandwidth `σ²` denominator multiplier.
    pub bandwidth: f32,
    /// Optional bias term.
    pub bias: f32,
}

impl RbfAccuracyPredictor {
    /// Fit an RBF regressor using a simple ridge-regularised normal equation
    /// `(K + λI) α = y`. Solves the normal equation via Gauss-Jordan.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if no samples.
    /// - [`NasError::DimensionMismatch`] if features have inconsistent length.
    /// - [`NasError::Internal`] if the linear solve becomes singular.
    pub fn fit(samples: &[(Vec<f32>, f32)], bandwidth: f32, ridge: f32) -> NasResult<Self> {
        if samples.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        if !(bandwidth.is_finite() && bandwidth > 0.0) {
            return Err(NasError::NanInArchParams);
        }
        let n = samples.len();
        let dim = samples[0].0.len();
        for (x, _) in samples {
            if x.len() != dim {
                return Err(NasError::DimensionMismatch {
                    expected: dim,
                    got: x.len(),
                });
            }
        }
        // Build kernel matrix K (n × n) with regularisation on the diagonal.
        let mut k_mat = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let d2: f32 = samples[i]
                    .0
                    .iter()
                    .zip(samples[j].0.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                let k = (-d2 / (2.0 * bandwidth)).exp();
                k_mat[i * n + j] = k;
            }
            k_mat[i * n + i] += ridge;
        }
        let y: Vec<f32> = samples.iter().map(|(_, t)| *t).collect();
        let alpha = solve_linear(&mut k_mat, &y, n)?;
        let centres: Vec<Vec<f32>> = samples.iter().map(|(x, _)| x.clone()).collect();
        Ok(Self {
            centres,
            alpha,
            bandwidth,
            bias: 0.0,
        })
    }

    /// Predict accuracy. Output is clamped to `[0, 1]`.
    ///
    /// # Errors
    /// - [`NasError::DimensionMismatch`] if the feature length disagrees.
    pub fn predict(&self, layers: &[LayerSpec]) -> NasResult<f32> {
        if self.centres.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        let f = ArchFeatures::from_layers(layers)?;
        if f.dim() != self.centres[0].len() {
            return Err(NasError::DimensionMismatch {
                expected: self.centres[0].len(),
                got: f.dim(),
            });
        }
        let mut y = self.bias;
        for (centre, &alpha) in self.centres.iter().zip(self.alpha.iter()) {
            let d2: f32 = f
                .data
                .iter()
                .zip(centre.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            y += alpha * (-d2 / (2.0 * self.bandwidth)).exp();
        }
        Ok(y.clamp(0.0, 1.0))
    }
}

/// Solve `A·x = b` for `x` where `A` is square row-major `[n × n]`. Mutates `A`.
/// Uses Gauss-Jordan with partial pivoting.
fn solve_linear(a: &mut [f32], b: &[f32], n: usize) -> NasResult<Vec<f32>> {
    if a.len() != n * n || b.len() != n {
        return Err(NasError::DimensionMismatch {
            expected: n * n,
            got: a.len(),
        });
    }
    let mut x = b.to_vec();
    for i in 0..n {
        // Pivot
        let mut max_row = i;
        let mut max_val = a[i * n + i].abs();
        for r in (i + 1)..n {
            let v = a[r * n + i].abs();
            if v > max_val {
                max_val = v;
                max_row = r;
            }
        }
        if max_val < 1e-12 {
            return Err(NasError::Internal("RBF kernel matrix singular".into()));
        }
        if max_row != i {
            for col in 0..n {
                a.swap(i * n + col, max_row * n + col);
            }
            x.swap(i, max_row);
        }
        // Normalise pivot row.
        let pivot = a[i * n + i];
        let inv_pivot = 1.0 / pivot;
        for col in 0..n {
            a[i * n + col] *= inv_pivot;
        }
        x[i] *= inv_pivot;
        // Eliminate other rows.
        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r * n + i];
            if factor == 0.0 {
                continue;
            }
            for col in 0..n {
                let v = a[i * n + col];
                a[r * n + col] -= factor * v;
            }
            x[r] -= factor * x[i];
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::OpKind;

    #[test]
    fn knn_predicts_constant_dataset() {
        let mut p = KnnAccuracyPredictor::new(3).unwrap();
        let f =
            ArchFeatures::from_layers(&[LayerSpec::new(OpKind::SepConv3x3, 4, 4, 8, 8)]).unwrap();
        for _ in 0..5 {
            p.add(f.data.clone(), 0.7).unwrap();
        }
        let q = p
            .predict(&[LayerSpec::new(OpKind::SepConv3x3, 4, 4, 8, 8)])
            .unwrap();
        assert!((q - 0.7).abs() < 1e-4);
    }

    #[test]
    fn knn_rejects_empty_predict() {
        let p = KnnAccuracyPredictor::new(2).unwrap();
        assert!(
            p.predict(&[LayerSpec::new(OpKind::Identity, 4, 4, 8, 8)])
                .is_err()
        );
    }

    #[test]
    fn knn_rejects_invalid_accuracy() {
        let mut p = KnnAccuracyPredictor::new(1).unwrap();
        let f = ArchFeatures::from_layers(&[LayerSpec::new(OpKind::Identity, 4, 4, 8, 8)]).unwrap();
        assert!(p.add(f.data.clone(), 1.5).is_err());
        assert!(p.add(f.data, f32::NAN).is_err());
    }

    #[test]
    fn knn_zero_k_rejected() {
        assert!(KnnAccuracyPredictor::new(0).is_err());
    }

    #[test]
    fn rbf_fit_zero_target_zero_alpha() {
        let f = ArchFeatures::from_layers(&[LayerSpec::new(OpKind::Identity, 4, 4, 8, 8)]).unwrap();
        let samples = vec![(f.data.clone(), 0.0_f32); 4];
        let p = RbfAccuracyPredictor::fit(&samples, 1.0, 0.1).unwrap();
        let q = p
            .predict(&[LayerSpec::new(OpKind::Identity, 4, 4, 8, 8)])
            .unwrap();
        assert!(q.abs() < 1e-4);
    }

    #[test]
    fn rbf_fit_recovers_constant_target() {
        let f = ArchFeatures::from_layers(&[LayerSpec::new(OpKind::Identity, 4, 4, 8, 8)]).unwrap();
        let samples = vec![(f.data.clone(), 0.5_f32); 3];
        let p = RbfAccuracyPredictor::fit(&samples, 1.0, 1e-4).unwrap();
        let q = p
            .predict(&[LayerSpec::new(OpKind::Identity, 4, 4, 8, 8)])
            .unwrap();
        assert!((q - 0.5).abs() < 1e-2, "q = {q}");
    }

    #[test]
    fn rbf_rejects_invalid_bandwidth() {
        let f = ArchFeatures::from_layers(&[LayerSpec::new(OpKind::Identity, 4, 4, 8, 8)]).unwrap();
        let samples = vec![(f.data, 0.5_f32)];
        let r = RbfAccuracyPredictor::fit(&samples, 0.0, 1e-4);
        assert!(r.is_err());
    }

    #[test]
    fn rbf_rejects_dim_mismatch() {
        let samples = vec![(vec![0.0_f32, 1.0], 0.5), (vec![0.0_f32, 1.0, 2.0], 0.6)];
        let r = RbfAccuracyPredictor::fit(&samples, 1.0, 1e-4);
        assert!(r.is_err());
    }

    #[test]
    fn rbf_rejects_empty() {
        let r = RbfAccuracyPredictor::fit(&[], 1.0, 1e-4);
        assert!(r.is_err());
    }
}

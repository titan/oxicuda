//! Last-layer Laplace approximation (MacKay 1992; Daxberger et al. 2021).
//!
//! Given a deterministic feature extractor `φ: x → ℝ^D` and a final linear
//! layer `f(x) = w·φ(x) + b` trained to MAP, the Laplace approximation
//! constructs a Gaussian posterior over `(w, b)` from the curvature of the
//! loss at the MAP estimate. For a binary cross-entropy loss with `L2`
//! prior (precision `α`), the Hessian for sample `i` with feature `φ_i` and
//! softmax probability `p_i` is `H_i = p_i (1−p_i) φ_i φ_iᵀ` plus a
//! constant `α·I` from the prior.
//!
//! This module implements the diagonal-Hessian last-layer Laplace which is
//! the simplest and most common variant — it matches the off-the-shelf
//! `laplace-torch` `last_layer / diag` configuration.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

/// Last-layer Laplace approximation over weights `w ∈ ℝ^D` (no bias).
#[derive(Debug, Clone, PartialEq)]
pub struct LastLayerLaplace {
    /// MAP weight estimate `w̄` (length `D`).
    pub map_weights: Vec<f32>,
    /// Posterior diagonal precision `H_diag + α·1` (length `D`).
    pub precision_diag: Vec<f32>,
    /// Prior precision `α`.
    pub prior_precision: f32,
}

impl LastLayerLaplace {
    /// Fit the diagonal Hessian Laplace approximation for binary logistic regression
    /// with MAP weights `w̄`, training features `φ ∈ ℝ^{N×D}`, and labels `y ∈ {0,1}^N`.
    ///
    /// `precision_diag[d] = α + Σ_i p_i(1−p_i) φ_{i,d}²`.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] if `phi.is_empty()` / dim 0.
    /// - [`BayesError::DimensionMismatch`] for shape mismatches.
    /// - [`BayesError::InvalidPriorVariance`] when `prior_precision <= 0`.
    pub fn fit_binary_logistic(
        map_weights: &[f32],
        phi: &[f32],
        labels: &[u8],
        prior_precision: f32,
    ) -> BayesResult<Self> {
        let d = map_weights.len();
        if d == 0 || phi.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        if !(prior_precision.is_finite() && prior_precision > 0.0) {
            return Err(BayesError::InvalidPriorVariance);
        }
        if phi.len() % d != 0 {
            return Err(BayesError::DimensionMismatch {
                expected: d,
                got: phi.len() % d,
            });
        }
        let n = phi.len() / d;
        if labels.len() != n {
            return Err(BayesError::DimensionMismatch {
                expected: n,
                got: labels.len(),
            });
        }
        for &y in labels {
            if y > 1 {
                return Err(BayesError::DimensionMismatch {
                    expected: 1,
                    got: usize::from(y),
                });
            }
        }

        let mut precision_diag = vec![prior_precision; d];
        for i in 0..n {
            let row = &phi[i * d..(i + 1) * d];
            let mut z = 0.0_f32;
            for (w, &x) in map_weights.iter().zip(row.iter()) {
                z += w * x;
            }
            // stable sigmoid
            let p = if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let e = z.exp();
                e / (1.0 + e)
            };
            let pq = p * (1.0 - p);
            for (h, &x) in precision_diag.iter_mut().zip(row.iter()) {
                *h += pq * x * x;
            }
        }
        Ok(Self {
            map_weights: map_weights.to_vec(),
            precision_diag,
            prior_precision,
        })
    }

    /// Posterior standard deviation per dimension `1/√precision_diag`.
    #[must_use]
    pub fn std_dev(&self) -> Vec<f32> {
        self.precision_diag
            .iter()
            .map(|p| 1.0 / p.max(1e-30).sqrt())
            .collect()
    }

    /// Sample a weight vector `w̃ ~ N(w̄, diag(1/H_diag))`.
    ///
    /// # Errors
    /// Should not fail for a properly-fit posterior; defensive checks.
    pub fn sample_weights(&self, rng: &mut LcgRng) -> BayesResult<Vec<f32>> {
        let d = self.map_weights.len();
        let std = self.std_dev();
        let mut z = vec![0.0_f32; d];
        rng.fill_normal(&mut z);
        let sample = self
            .map_weights
            .iter()
            .zip(std.iter())
            .zip(z.iter())
            .map(|((mu, sigma), zi)| *mu + *sigma * *zi)
            .collect();
        Ok(sample)
    }

    /// Predictive logit for a feature vector using the closed-form linearisation
    /// `μ_z = w̄·φ`, `σ_z² = Σ_d φ_d² / H_diag[d]`.
    /// Returns `(mean, variance)` of the pre-sigmoid logit.
    ///
    /// # Errors
    /// [`BayesError::DimensionMismatch`] if `phi.len() != self.map_weights.len()`.
    pub fn predictive_logit(&self, phi: &[f32]) -> BayesResult<(f32, f32)> {
        let d = self.map_weights.len();
        if phi.len() != d {
            return Err(BayesError::DimensionMismatch {
                expected: d,
                got: phi.len(),
            });
        }
        let mut mean = 0.0_f32;
        let mut variance = 0.0_f32;
        for ((w, &x), &h) in self
            .map_weights
            .iter()
            .zip(phi.iter())
            .zip(self.precision_diag.iter())
        {
            mean += w * x;
            variance += x * x / h.max(1e-30);
        }
        let _ = d;
        Ok((mean, variance))
    }

    /// Probit-approximated marginal probability `σ(μ / √(1 + (π/8)σ²))`
    /// — a well-known closed-form approximation to ∫ σ(z) N(z;μ,σ²) dz.
    ///
    /// # Errors
    /// Propagated from [`Self::predictive_logit`].
    pub fn predictive_probability(&self, phi: &[f32]) -> BayesResult<f32> {
        let (mu, var) = self.predictive_logit(phi)?;
        let kappa = 1.0 / (1.0 + std::f32::consts::PI * var / 8.0).sqrt();
        let z = mu * kappa;
        // stable sigmoid
        Ok(if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let e = z.exp();
            e / (1.0 + e)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laplace_fit_basic_shape() {
        let map = vec![0.5_f32, -0.5];
        let phi = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let labels = vec![1_u8, 0, 1];
        let l = LastLayerLaplace::fit_binary_logistic(&map, &phi, &labels, 1.0)
            .expect("fit_binary_logistic should succeed");
        assert_eq!(l.map_weights, map);
        assert_eq!(l.precision_diag.len(), 2);
        // precision starts at α=1 then has positive contributions; should be > 1
        assert!(l.precision_diag.iter().all(|&v| v >= 1.0));
    }

    #[test]
    fn laplace_predictive_logit_matches_dot_product() {
        let map = vec![0.5_f32, -0.5];
        let phi = vec![1.0_f32, 0.0];
        let l = LastLayerLaplace {
            map_weights: map.clone(),
            precision_diag: vec![1.0, 1.0],
            prior_precision: 1.0,
        };
        let (mu, var) = l
            .predictive_logit(&phi)
            .expect("predictive_logit should succeed");
        assert!((mu - 0.5).abs() < 1e-6);
        // var = 1²/1 + 0²/1 = 1.0
        assert!((var - 1.0).abs() < 1e-6);
    }

    #[test]
    fn laplace_predictive_probability_in_range() {
        let l = LastLayerLaplace {
            map_weights: vec![0.0_f32, 0.0],
            precision_diag: vec![10.0, 10.0],
            prior_precision: 1.0,
        };
        let p = l
            .predictive_probability(&[0.0_f32, 0.0])
            .expect("predictive_probability should succeed");
        assert!((p - 0.5).abs() < 1e-5);
        let p2 = l
            .predictive_probability(&[1.0_f32, 1.0])
            .expect("predictive_probability should succeed");
        assert!((0.0..=1.0).contains(&p2));
    }

    #[test]
    fn laplace_predictive_probability_more_uncertainty_pulls_to_half() {
        let mut l = LastLayerLaplace {
            map_weights: vec![1.0_f32],
            precision_diag: vec![1.0],
            prior_precision: 1.0,
        };
        let p_certain = l
            .predictive_probability(&[3.0_f32])
            .expect("predictive_probability should succeed");
        // Reduce precision to inflate variance and the prediction should move
        // toward 0.5.
        l.precision_diag = vec![0.01];
        let p_uncertain = l
            .predictive_probability(&[3.0_f32])
            .expect("predictive_probability should succeed");
        assert!(p_uncertain < p_certain);
        assert!(p_uncertain > 0.5);
    }

    #[test]
    fn laplace_sample_dimension_correct() {
        let mut rng = LcgRng::new(0);
        let l = LastLayerLaplace {
            map_weights: vec![0.0_f32; 3],
            precision_diag: vec![1.0; 3],
            prior_precision: 1.0,
        };
        let s = l
            .sample_weights(&mut rng)
            .expect("sample_weights should succeed");
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn laplace_rejects_invalid_prior() {
        let r = LastLayerLaplace::fit_binary_logistic(&[0.0_f32], &[0.0_f32], &[0_u8], -1.0);
        assert!(r.is_err());
    }

    #[test]
    fn laplace_rejects_dim_mismatch() {
        let r = LastLayerLaplace::fit_binary_logistic(
            &[0.0_f32, 0.0],
            &[0.0_f32, 0.0, 0.0],
            &[0_u8],
            1.0,
        );
        // phi.len()=3 not divisible by d=2
        assert!(r.is_err());
    }

    #[test]
    fn laplace_rejects_label_count_mismatch() {
        let r = LastLayerLaplace::fit_binary_logistic(
            &[0.0_f32, 0.0],
            &[0.0_f32, 0.0],
            &[0_u8, 1, 0],
            1.0,
        );
        assert!(r.is_err());
    }

    #[test]
    fn laplace_predictive_logit_dim_mismatch() {
        let l = LastLayerLaplace {
            map_weights: vec![0.0_f32, 0.0],
            precision_diag: vec![1.0, 1.0],
            prior_precision: 1.0,
        };
        assert!(l.predictive_logit(&[1.0_f32, 1.0, 1.0]).is_err());
    }
}

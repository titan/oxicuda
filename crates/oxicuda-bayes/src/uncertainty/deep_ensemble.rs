//! Deep Ensembles for predictive uncertainty (Lakshminarayanan et al. 2017).
//!
//! Train `M` independent neural networks with different random seeds (and
//! optionally different shuffles / data subsets), then average their
//! probabilistic predictions:
//!
//! - predictive mean `μ̄ = (1/M) Σ_m μ_m` (or softmax-averaged probabilities)
//! - predictive variance `σ̄² = (1/M) Σ_m σ_m² + (1/M) Σ_m (μ_m − μ̄)²`
//!   (the law of total variance: aleatoric + epistemic).
//!
//! For classification we provide [`DeepEnsemble::aggregate_probabilities`]
//! which returns the mean probability and disagreement-style variance.

use crate::error::{BayesError, BayesResult};

/// Aggregated statistics from an ensemble of `M` predictions.
#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleStats {
    /// Mean prediction across the ensemble (length = output dim).
    pub mean: Vec<f32>,
    /// Variance across the ensemble (sample variance, length = output dim).
    pub variance: Vec<f32>,
    /// Number of ensemble members.
    pub n_members: usize,
}

impl EnsembleStats {
    /// Maximum disagreement (largest variance component).
    #[must_use]
    pub fn max_variance(&self) -> f32 {
        self.variance.iter().copied().fold(0.0_f32, f32::max)
    }
}

/// Deep ensemble container holding `M` independent predictions with shape `[K]` each.
///
/// `predictions[m]` is the output (e.g. softmax probabilities) of the `m`-th
/// ensemble member.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepEnsemble {
    /// Per-member outputs, all of equal length `K`.
    pub predictions: Vec<Vec<f32>>,
}

impl DeepEnsemble {
    /// New ensemble from a list of member predictions; validates lengths.
    ///
    /// # Errors
    /// - [`BayesError::InsufficientEnsembleMembers`] if `predictions.len() < 2`.
    /// - [`BayesError::EmptyInputs`] if the first member is empty.
    /// - [`BayesError::DimensionMismatch`] when member shapes differ.
    pub fn new(predictions: Vec<Vec<f32>>) -> BayesResult<Self> {
        if predictions.len() < 2 {
            return Err(BayesError::InsufficientEnsembleMembers {
                min: 2,
                got: predictions.len(),
            });
        }
        let k = predictions[0].len();
        if k == 0 {
            return Err(BayesError::EmptyInputs);
        }
        for (m, p) in predictions.iter().enumerate().skip(1) {
            if p.len() != k {
                return Err(BayesError::DimensionMismatch {
                    expected: k,
                    got: p.len(),
                });
            }
            // forbid NaNs early
            for &v in p {
                if !v.is_finite() {
                    return Err(BayesError::NanEncountered {
                        location: "DeepEnsemble::new",
                    });
                }
            }
            let _ = m;
        }
        Ok(Self { predictions })
    }

    /// Number of members.
    #[must_use]
    pub fn n_members(&self) -> usize {
        self.predictions.len()
    }

    /// Output dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.predictions.first().map(Vec::len).unwrap_or(0)
    }

    /// Mean and variance across members (sample variance with `n - 1` denominator
    /// when `M ≥ 2`, otherwise zero variance).
    #[must_use]
    pub fn aggregate(&self) -> EnsembleStats {
        let m = self.n_members();
        let k = self.dim();
        let mut mean = vec![0.0_f64; k];
        for p in &self.predictions {
            for (acc, &v) in mean.iter_mut().zip(p.iter()) {
                *acc += v as f64;
            }
        }
        let inv_m = 1.0_f64 / m as f64;
        for v in mean.iter_mut() {
            *v *= inv_m;
        }
        let mut var = vec![0.0_f64; k];
        for p in &self.predictions {
            for (acc, (&v, &mu)) in var.iter_mut().zip(p.iter().zip(mean.iter())) {
                let d = v as f64 - mu;
                *acc += d * d;
            }
        }
        let denom = if m >= 2 { (m - 1) as f64 } else { 1.0 };
        let mean_f32: Vec<f32> = mean.iter().map(|v| *v as f32).collect();
        let var_f32: Vec<f32> = var.iter().map(|v| (*v / denom) as f32).collect();
        EnsembleStats {
            mean: mean_f32,
            variance: var_f32,
            n_members: m,
        }
    }

    /// Aggregate probability vectors specifically: clamp each member to a valid
    /// simplex first by re-normalising, then return the mean over members and
    /// the per-class disagreement variance.
    ///
    /// # Errors
    /// Returns [`BayesError::NanEncountered`] if any member has total mass ≤ 0.
    pub fn aggregate_probabilities(&self) -> BayesResult<EnsembleStats> {
        let m = self.n_members();
        let k = self.dim();
        let mut probs: Vec<Vec<f32>> = Vec::with_capacity(m);
        for p in &self.predictions {
            let s: f32 = p.iter().sum();
            if !(s.is_finite() && s > 0.0) {
                return Err(BayesError::NanEncountered {
                    location: "DeepEnsemble::aggregate_probabilities: invalid mass",
                });
            }
            let inv = 1.0 / s;
            let mut norm = Vec::with_capacity(k);
            for &v in p.iter() {
                norm.push(v * inv);
            }
            probs.push(norm);
        }
        let renorm = DeepEnsemble { predictions: probs };
        Ok(renorm.aggregate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensemble_aggregate_two_member_mean() {
        let preds = vec![vec![0.0_f32, 1.0, 2.0], vec![2.0_f32, 1.0, 0.0]];
        let e = DeepEnsemble::new(preds).unwrap();
        let s = e.aggregate();
        assert_eq!(s.n_members, 2);
        assert!((s.mean[0] - 1.0).abs() < 1e-6);
        assert!((s.mean[1] - 1.0).abs() < 1e-6);
        assert!((s.mean[2] - 1.0).abs() < 1e-6);
        // Sample var with n=2: ((0-1)^2 + (2-1)^2) / 1 = 2 for index 0/2; 0 for index 1.
        assert!((s.variance[0] - 2.0).abs() < 1e-5);
        assert!(s.variance[1].abs() < 1e-5);
        assert!((s.variance[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn ensemble_aggregate_identical_members_zero_variance() {
        let p = vec![0.1_f32, 0.5, 0.4];
        let e = DeepEnsemble::new(vec![p.clone(), p.clone(), p]).unwrap();
        let s = e.aggregate();
        for v in &s.variance {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn ensemble_aggregate_probabilities_renormalises() {
        // Member 1 sums to 2.0; should be re-normalised
        let preds = vec![vec![0.4_f32, 0.6], vec![0.5_f32, 1.5]];
        let e = DeepEnsemble::new(preds).unwrap();
        let s = e.aggregate_probabilities().unwrap();
        // First member normalises to (0.4, 0.6); second to (0.25, 0.75); mean (0.325, 0.675)
        assert!((s.mean[0] - 0.325).abs() < 1e-5);
        assert!((s.mean[1] - 0.675).abs() < 1e-5);
        // sums to 1
        assert!((s.mean.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ensemble_rejects_too_few_members() {
        let r = DeepEnsemble::new(vec![vec![0.5_f32]]);
        assert!(r.is_err());
    }

    #[test]
    fn ensemble_rejects_empty_member() {
        let r = DeepEnsemble::new(vec![vec![], vec![1.0_f32]]);
        assert!(r.is_err());
    }

    #[test]
    fn ensemble_rejects_shape_mismatch() {
        let r = DeepEnsemble::new(vec![vec![0.5_f32], vec![0.4_f32, 0.6_f32]]);
        assert!(r.is_err());
    }

    #[test]
    fn ensemble_rejects_nan_member() {
        let r = DeepEnsemble::new(vec![vec![0.5_f32, 0.5], vec![f32::NAN, 1.0]]);
        assert!(r.is_err());
    }

    #[test]
    fn ensemble_max_variance_finds_largest() {
        let preds = vec![vec![0.0_f32, 0.0, 0.0], vec![1.0_f32, 0.0, 5.0]];
        let s = DeepEnsemble::new(preds).unwrap().aggregate();
        assert!((s.max_variance() - 12.5).abs() < 1e-5);
    }

    #[test]
    fn ensemble_aggregate_probabilities_rejects_zero_mass() {
        let preds = vec![vec![0.0_f32, 0.0], vec![0.5_f32, 0.5]];
        let r = DeepEnsemble::new(preds).unwrap().aggregate_probabilities();
        assert!(r.is_err());
    }
}

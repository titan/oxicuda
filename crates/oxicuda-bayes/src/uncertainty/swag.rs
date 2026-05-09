//! SWAG: Stochastic Weight Averaging Gaussian (Maddox et al. 2019).
//!
//! Approximates the posterior over network weights by a Gaussian whose
//! mean and covariance are estimated from SGD iterates `θ_1, θ_2, …, θ_T`:
//!
//! - mean `μ̄ = (1/T) Σ_t θ_t`
//! - diagonal `σ²_diag = (1/T) Σ_t θ_t² − μ̄²`
//! - low-rank deviation columns `D_t = θ_t − μ̄_t` (last `K` iterates) form
//!   a `[d × K]` matrix whose `(1/(K-1)) D Dᵀ` is the rank-`K` covariance
//!   contribution.
//!
//! The full SWAG covariance is
//! `Σ = ½·diag(σ²_diag) + (1/(2(K−1)))·D·Dᵀ`,
//! which corresponds to combining the diagonal and low-rank pieces with weight ½ each.
//!
//! Sampling: `θ̃ = μ̄ + (1/√2)·σ_diag ⊙ z₁ + (1/√(2(K−1)))·D·z₂`,
//! `z₁ ~ N(0, I_d), z₂ ~ N(0, I_K)`.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

/// SWAG posterior holding running statistics from SGD iterates.
#[derive(Debug, Clone, PartialEq)]
pub struct SwagPosterior {
    /// Parameter dimensionality `d`.
    pub dim: usize,
    /// Maximum number of low-rank deviation columns to retain (`K`).
    pub max_rank: usize,
    /// Number of iterates absorbed so far.
    pub n_iterates: usize,
    /// Running first moment `μ̄_t` (length `d`).
    pub mean: Vec<f32>,
    /// Running second moment `Σ_t θ_t² / t` (length `d`).
    pub second_moment: Vec<f32>,
    /// FIFO queue of deviation columns `θ_t − μ̄_t` (length `≤ max_rank`).
    pub deviations: Vec<Vec<f32>>,
}

impl SwagPosterior {
    /// Empty posterior with given parameter dim and rank.
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] if `dim == 0`.
    /// - [`BayesError::InsufficientSamples`] if `max_rank == 0`.
    pub fn new(dim: usize, max_rank: usize) -> BayesResult<Self> {
        if dim == 0 {
            return Err(BayesError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if max_rank == 0 {
            return Err(BayesError::InsufficientSamples { min: 1, got: 0 });
        }
        Ok(Self {
            dim,
            max_rank,
            n_iterates: 0,
            mean: vec![0.0_f32; dim],
            second_moment: vec![0.0_f32; dim],
            deviations: Vec::with_capacity(max_rank),
        })
    }

    /// Absorb a single SGD iterate `θ_t`.
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] if `iterate.len() != self.dim`.
    /// - [`BayesError::NanEncountered`] if `iterate` contains a non-finite value.
    pub fn update(&mut self, iterate: &[f32]) -> BayesResult<()> {
        if iterate.len() != self.dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.dim,
                got: iterate.len(),
            });
        }
        for &v in iterate {
            if !v.is_finite() {
                return Err(BayesError::NanEncountered {
                    location: "SwagPosterior::update",
                });
            }
        }
        self.n_iterates += 1;
        let t = self.n_iterates as f32;
        let inv = 1.0 / t;
        for ((mu, m2), &x) in self
            .mean
            .iter_mut()
            .zip(self.second_moment.iter_mut())
            .zip(iterate.iter())
        {
            *mu = (*mu) * (t - 1.0) * inv + x * inv;
            *m2 = (*m2) * (t - 1.0) * inv + x * x * inv;
        }
        // Compute deviation column θ_t − μ̄_t and push (FIFO).
        let deviation: Vec<f32> = iterate
            .iter()
            .zip(self.mean.iter())
            .map(|(&x, &mu)| x - mu)
            .collect();
        if self.deviations.len() == self.max_rank {
            self.deviations.remove(0);
        }
        self.deviations.push(deviation);
        Ok(())
    }

    /// Diagonal `σ²_diag = E[θ²] − E[θ]²`, clamped to ≥ 0.
    #[must_use]
    pub fn diagonal_variance(&self) -> Vec<f32> {
        self.mean
            .iter()
            .zip(self.second_moment.iter())
            .map(|(&mu, &m2)| (m2 - mu * mu).max(0.0))
            .collect()
    }

    /// Sample a parameter vector `θ̃ ~ N(μ, ½·diag(σ²) + (1/(2(K−1)))·D·Dᵀ)`.
    ///
    /// # Errors
    /// - [`BayesError::InsufficientSamples`] if fewer than 2 iterates have
    ///   been absorbed (need at least 2 for variance / low-rank).
    pub fn sample(&self, rng: &mut LcgRng) -> BayesResult<Vec<f32>> {
        let k = self.deviations.len();
        if self.n_iterates < 2 || k < 2 {
            return Err(BayesError::InsufficientSamples {
                min: 2,
                got: self.n_iterates,
            });
        }
        let diag_var = self.diagonal_variance();
        let inv_sqrt2 = 1.0_f32 / std::f32::consts::SQRT_2;
        let inv_sqrt_2km1 = 1.0_f32 / (2.0 * (k as f32 - 1.0)).sqrt();

        let mut z1 = vec![0.0_f32; self.dim];
        rng.fill_normal(&mut z1);
        let mut z2 = vec![0.0_f32; k];
        rng.fill_normal(&mut z2);

        let mut sample = self.mean.clone();
        for ((s, var), zi) in sample.iter_mut().zip(diag_var.iter()).zip(z1.iter()) {
            *s += inv_sqrt2 * var.sqrt() * *zi;
        }
        // Low-rank: D z2 / sqrt(2(K-1))
        for (col_idx, dev) in self.deviations.iter().enumerate() {
            let coef = inv_sqrt_2km1 * z2[col_idx];
            for (s, &d) in sample.iter_mut().zip(dev.iter()) {
                *s += coef * d;
            }
        }
        Ok(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swag_initialises_with_zero_mean() {
        let s = SwagPosterior::new(4, 3).unwrap();
        assert_eq!(s.mean, vec![0.0_f32; 4]);
        assert_eq!(s.second_moment, vec![0.0_f32; 4]);
        assert!(s.deviations.is_empty());
        assert_eq!(s.n_iterates, 0);
    }

    #[test]
    fn swag_update_running_mean_correct() {
        let mut s = SwagPosterior::new(2, 5).unwrap();
        s.update(&[1.0_f32, 2.0]).unwrap();
        s.update(&[3.0_f32, 4.0]).unwrap();
        assert!((s.mean[0] - 2.0).abs() < 1e-5);
        assert!((s.mean[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn swag_diagonal_variance_correct() {
        let mut s = SwagPosterior::new(1, 5).unwrap();
        s.update(&[1.0_f32]).unwrap();
        s.update(&[3.0_f32]).unwrap();
        // E[X] = 2, E[X^2] = 5, var = 1
        let v = s.diagonal_variance();
        assert!((v[0] - 1.0).abs() < 1e-5, "v={v:?}");
    }

    #[test]
    fn swag_low_rank_buffer_caps_at_max_rank() {
        let mut s = SwagPosterior::new(1, 2).unwrap();
        for i in 0..5 {
            s.update(&[i as f32]).unwrap();
        }
        assert_eq!(s.deviations.len(), 2);
        assert_eq!(s.n_iterates, 5);
    }

    #[test]
    fn swag_sample_returns_correct_dim() {
        let mut rng = LcgRng::new(42);
        let mut s = SwagPosterior::new(3, 4).unwrap();
        for _ in 0..6 {
            let mut v = vec![0.0_f32; 3];
            rng.fill_normal(&mut v);
            s.update(&v).unwrap();
        }
        let theta = s.sample(&mut rng).unwrap();
        assert_eq!(theta.len(), 3);
        assert!(theta.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn swag_rejects_zero_dim() {
        let r = SwagPosterior::new(0, 2);
        assert!(r.is_err());
    }

    #[test]
    fn swag_rejects_zero_rank() {
        let r = SwagPosterior::new(2, 0);
        assert!(r.is_err());
    }

    #[test]
    fn swag_update_rejects_dim_mismatch() {
        let mut s = SwagPosterior::new(3, 2).unwrap();
        let r = s.update(&[1.0_f32, 2.0]);
        assert!(r.is_err());
    }

    #[test]
    fn swag_update_rejects_nan() {
        let mut s = SwagPosterior::new(2, 2).unwrap();
        let r = s.update(&[1.0_f32, f32::NAN]);
        assert!(r.is_err());
    }

    #[test]
    fn swag_sample_requires_two_iterates() {
        let mut rng = LcgRng::new(0);
        let mut s = SwagPosterior::new(2, 2).unwrap();
        s.update(&[1.0_f32, 2.0]).unwrap();
        assert!(s.sample(&mut rng).is_err());
    }
}

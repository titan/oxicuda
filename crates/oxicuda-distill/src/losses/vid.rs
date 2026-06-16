//! VID — Variational Information Distillation (Ahn et al. 2019).
//!
//! Reference: Ahn, S., Hu, S. X., Damianou, A., Lawrence, N. D., & Dai, Z. (2019).
//! *Variational Information Distillation for Knowledge Transfer*. CVPR 2019.
//! <https://arxiv.org/abs/1904.05835>
//!
//! VID maximises a **variational lower bound on the mutual information** `I(t; s)`
//! between a teacher feature `t` and a student feature `s`. The intractable
//! conditional `p(t | s)` is replaced by a learnable variational Gaussian whose mean
//! is a regressor `μ(s)` on the student features and whose variance `σ²` is a
//! learnable *per-channel* parameter:
//!
//! ```text
//!   −log q(t | s) = Σ_c  ½ [ log σ²_c + (t_c − μ_c(s))² / σ²_c ]  + const .
//! ```
//!
//! Minimising this negative log-likelihood (equivalently, maximising the MI bound)
//! drives `μ(s)` toward `t` while the learnable variance down-weights channels the
//! student cannot predict. To keep `σ²` strictly positive it is parametrised through
//! a softplus of an unconstrained parameter `α`:
//!
//! ```text
//!   σ²_c = softplus(α_c) + ε = log(1 + exp(α_c)) + ε .
//! ```
//!
//! [`VidRegressor`] holds the `1×1` linear mean predictor `μ(s) = W·s + b` (a
//! per-channel affine map, matching the paper's channel-wise regressor for already
//! spatially-pooled features) together with the variance parameters `α`.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

const EPS: f32 = 1e-6;

/// Numerically stable softplus `log(1 + eˣ)`.
#[inline]
#[must_use]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Variational regressor `q(t | s)` for VID: a per-channel affine mean predictor plus
/// learnable log-variance parameters.
#[derive(Debug, Clone)]
pub struct VidRegressor {
    /// Number of feature channels.
    pub channels: usize,
    /// Per-channel weight of the mean predictor `μ_c = w_c · s_c + b_c`.
    pub weight: Vec<f32>,
    /// Per-channel bias of the mean predictor.
    pub bias: Vec<f32>,
    /// Unconstrained variance parameters `α_c` (variance is `softplus(α_c) + ε`).
    pub log_var: Vec<f32>,
}

impl VidRegressor {
    /// Build a regressor with identity mean (`w = 1`, `b = 0`) and unit-ish variance.
    ///
    /// The initial `α` is chosen so that `softplus(α) ≈ 1`, i.e. `α = log(e − 1)`.
    ///
    /// # Errors
    ///
    /// [`DistillError::EmptyInput`] if `channels == 0`.
    pub fn new(channels: usize) -> DistillResult<Self> {
        if channels == 0 {
            return Err(DistillError::EmptyInput);
        }
        let alpha_unit = (std::f32::consts::E - 1.0).ln(); // softplus(α)=1
        Ok(Self {
            channels,
            weight: vec![1.0_f32; channels],
            bias: vec![0.0_f32; channels],
            log_var: vec![alpha_unit; channels],
        })
    }

    /// Build a regressor with He-initialised mean weights and unit variance.
    ///
    /// # Errors
    ///
    /// [`DistillError::EmptyInput`] if `channels == 0`.
    pub fn with_random_mean(channels: usize, rng: &mut LcgRng) -> DistillResult<Self> {
        let mut reg = Self::new(channels)?;
        let scale = (2.0_f32 / channels as f32).sqrt();
        for w in reg.weight.iter_mut() {
            *w = rng.next_normal() * scale;
        }
        Ok(reg)
    }

    /// Predicted mean `μ(s)` for a student feature vector `s` of length `channels`.
    ///
    /// # Errors
    ///
    /// [`DistillError::DimensionMismatch`] if `s.len() != channels`.
    pub fn mean(&self, s: &[f32]) -> DistillResult<Vec<f32>> {
        if s.len() != self.channels {
            return Err(DistillError::DimensionMismatch {
                expected: self.channels,
                got: s.len(),
            });
        }
        Ok(s.iter()
            .zip(self.weight.iter())
            .zip(self.bias.iter())
            .map(|((&si, &wi), &bi)| wi * si + bi)
            .collect())
    }

    /// Per-channel variance `σ²_c = softplus(α_c) + ε`.
    #[must_use]
    pub fn variance(&self) -> Vec<f32> {
        self.log_var.iter().map(|&a| softplus(a) + EPS).collect()
    }

    /// VID negative-log-likelihood loss for one (student, teacher) feature pair.
    ///
    /// `L = Σ_c ½ [ log σ²_c + (t_c − μ_c(s))² / σ²_c ]`.
    ///
    /// # Errors
    ///
    /// - [`DistillError::DimensionMismatch`] if `s.len() != channels` or
    ///   `t.len() != channels`.
    /// - [`DistillError::NumericalError`] if the result is non-finite.
    pub fn loss(&self, s: &[f32], t: &[f32]) -> DistillResult<f32> {
        if t.len() != self.channels {
            return Err(DistillError::DimensionMismatch {
                expected: self.channels,
                got: t.len(),
            });
        }
        let mu = self.mean(s)?;
        let var = self.variance();
        let mut total = 0.0_f32;
        for c in 0..self.channels {
            let diff = t[c] - mu[c];
            total += 0.5 * (var[c].ln() + diff * diff / var[c]);
        }
        if !total.is_finite() {
            return Err(DistillError::NumericalError {
                msg: "VID loss produced a non-finite value".into(),
            });
        }
        Ok(total)
    }

    /// Mean VID loss over a batch of feature pairs.
    ///
    /// `student` / `teacher` are `batch × channels` row-major matrices.
    ///
    /// # Errors
    ///
    /// - [`DistillError::EmptyInput`] if `batch == 0`.
    /// - [`DistillError::DimensionMismatch`] if either slice length disagrees with
    ///   `batch · channels`.
    /// - Propagates [`DistillError::NumericalError`] from [`VidRegressor::loss`].
    pub fn batch_loss(&self, student: &[f32], teacher: &[f32], batch: usize) -> DistillResult<f32> {
        if batch == 0 {
            return Err(DistillError::EmptyInput);
        }
        if student.len() != batch * self.channels {
            return Err(DistillError::DimensionMismatch {
                expected: batch * self.channels,
                got: student.len(),
            });
        }
        if teacher.len() != batch * self.channels {
            return Err(DistillError::DimensionMismatch {
                expected: batch * self.channels,
                got: teacher.len(),
            });
        }
        let mut total = 0.0_f32;
        for b in 0..batch {
            let s = &student[b * self.channels..(b + 1) * self.channels];
            let t = &teacher[b * self.channels..(b + 1) * self.channels];
            total += self.loss(s, t)?;
        }
        Ok(total / batch as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_is_positive_and_monotone() {
        assert!(softplus(-10.0) > 0.0);
        assert!(softplus(0.0) > softplus(-1.0));
        assert!(softplus(5.0) > softplus(0.0));
        // Large-x branch returns x.
        assert!((softplus(50.0) - 50.0).abs() < 1e-3);
    }

    #[test]
    fn new_rejects_zero_channels() {
        assert!(matches!(
            VidRegressor::new(0),
            Err(DistillError::EmptyInput)
        ));
    }

    #[test]
    fn init_variance_is_unit() {
        let reg = VidRegressor::new(4).expect("new should succeed");
        for v in reg.variance() {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "initial variance should be ~1, got {v}"
            );
        }
    }

    #[test]
    fn identity_mean_returns_input() {
        let reg = VidRegressor::new(3).expect("new should succeed");
        let s = vec![1.0_f32, -2.0, 3.5];
        let mu = reg.mean(&s).expect("mean should succeed");
        for (a, b) in mu.iter().zip(s.iter()) {
            assert!((a - b).abs() < 1e-6, "identity mean must echo input");
        }
    }

    #[test]
    fn mean_dim_mismatch_errors() {
        let reg = VidRegressor::new(3).expect("new should succeed");
        assert!(matches!(
            reg.mean(&[1.0, 2.0]),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn variance_strictly_positive_for_extreme_alpha() {
        let mut reg = VidRegressor::new(2).expect("new should succeed");
        reg.log_var = vec![-50.0, 50.0];
        for v in reg.variance() {
            assert!(
                v > 0.0 && v.is_finite(),
                "variance must stay positive, got {v}"
            );
        }
    }

    #[test]
    fn loss_minimized_when_mean_matches_teacher() {
        // With identity mean, loss is smallest when s == t.
        let reg = VidRegressor::new(3).expect("new should succeed");
        let t = vec![0.5_f32, -1.0, 2.0];
        let matched = reg.loss(&t, &t).expect("loss should succeed");
        let mismatched = reg.loss(&[0.0, 0.0, 0.0], &t).expect("loss should succeed");
        assert!(
            matched < mismatched,
            "matched features must give lower loss"
        );
    }

    #[test]
    fn loss_equals_half_logvar_when_matched() {
        // s == t ⇒ residual 0 ⇒ loss = Σ_c ½·log σ²_c. With unit variance log 1 ≈ 0.
        let reg = VidRegressor::new(4).expect("new should succeed");
        let t = vec![1.0_f32, 2.0, 3.0, 4.0];
        let loss = reg.loss(&t, &t).expect("loss should succeed");
        // σ² ≈ 1 + ε ⇒ ½·log(1+ε)·4 ≈ tiny positive.
        assert!(
            loss.abs() < 1e-2,
            "matched unit-variance loss should be ~0, got {loss}"
        );
    }

    #[test]
    fn larger_variance_down_weights_residual() {
        // Increasing σ² reduces the quadratic penalty contribution from a residual.
        let t = vec![5.0_f32];
        let s = vec![0.0_f32]; // residual 5
        let mut small = VidRegressor::new(1).expect("new should succeed");
        small.log_var = vec![(std::f32::consts::E - 1.0).ln()]; // σ²≈1
        let mut big = VidRegressor::new(1).expect("new should succeed");
        big.log_var = vec![10.0]; // σ² large
        let l_small = small.loss(&s, &t).expect("loss should succeed");
        let l_big = big.loss(&s, &t).expect("loss should succeed");
        assert!(
            l_big < l_small,
            "larger variance must reduce the residual penalty"
        );
    }

    #[test]
    fn loss_dim_mismatch_errors() {
        let reg = VidRegressor::new(3).expect("new should succeed");
        assert!(matches!(
            reg.loss(&[1.0, 2.0, 3.0], &[1.0, 2.0]),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn batch_loss_is_mean() {
        let reg = VidRegressor::new(2).expect("new should succeed");
        let student = vec![1.0_f32, 1.0, 0.0, 0.0]; // 2 samples × 2 ch
        let teacher = vec![1.0_f32, 1.0, 5.0, 5.0];
        let l0 = reg
            .loss(&student[0..2], &teacher[0..2])
            .expect("loss should succeed");
        let l1 = reg
            .loss(&student[2..4], &teacher[2..4])
            .expect("loss should succeed");
        let batch = reg
            .batch_loss(&student, &teacher, 2)
            .expect("batch_loss should succeed");
        assert!((batch - (l0 + l1) / 2.0).abs() < 1e-5);
    }

    #[test]
    fn batch_loss_dim_mismatch_errors() {
        let reg = VidRegressor::new(2).expect("new should succeed");
        let student = vec![1.0_f32; 3]; // not 2*2
        let teacher = vec![1.0_f32; 4];
        assert!(matches!(
            reg.batch_loss(&student, &teacher, 2),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn batch_loss_zero_batch_errors() {
        let reg = VidRegressor::new(2).expect("new should succeed");
        assert!(matches!(
            reg.batch_loss(&[], &[], 0),
            Err(DistillError::EmptyInput)
        ));
    }

    #[test]
    fn random_mean_is_deterministic_for_seed() {
        let mut r1 = LcgRng::new(5);
        let mut r2 = LcgRng::new(5);
        let a =
            VidRegressor::with_random_mean(8, &mut r1).expect("with_random_mean should succeed");
        let b =
            VidRegressor::with_random_mean(8, &mut r2).expect("with_random_mean should succeed");
        assert_eq!(a.weight, b.weight, "same seed must produce same weights");
    }
}

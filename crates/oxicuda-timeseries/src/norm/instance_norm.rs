//! Standard Instance Normalisation over the time axis.
//!
//! Normalises each `(batch, variate)` independently:
//! `y[t] = (x[t] - μ) / (σ + ε)` with optional learnable affine `γ, β`.
//!
//! Unlike RevIN this does not include a reverse step; it is a simple
//! normalisation layer used inside encoder blocks.

use crate::error::{TsError, TsResult};

/// Instance normalisation over the time axis of a `[T, C]` tensor.
#[derive(Debug, Clone)]
pub struct InstanceNorm1d {
    /// Number of channels.
    pub c: usize,
    /// Numerical stability constant.
    pub eps: f32,
    /// Optional per-channel affine weight `[C]` (None = no affine).
    pub gamma: Option<Vec<f32>>,
    /// Optional per-channel affine bias `[C]`.
    pub beta: Option<Vec<f32>>,
}

impl InstanceNorm1d {
    /// Create instance norm without learnable affine parameters.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::InvalidNumVariates`] when `c == 0`.
    pub fn new(c: usize) -> TsResult<Self> {
        if c == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        Ok(Self {
            c,
            eps: 1e-5,
            gamma: None,
            beta: None,
        })
    }

    /// Create instance norm with learnable affine parameters (gamma=1, beta=0).
    ///
    /// # Errors
    ///
    /// Returns [`TsError::InvalidNumVariates`] when `c == 0`.
    pub fn with_affine(c: usize) -> TsResult<Self> {
        if c == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        Ok(Self {
            c,
            eps: 1e-5,
            gamma: Some(vec![1.0_f32; c]),
            beta: Some(vec![0.0_f32; c]),
        })
    }

    /// Normalise a `[T, C]` tensor in-place.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `t == 0`.
    /// - [`TsError::DimensionMismatch`] when `features.len() != t * self.c`.
    pub fn forward(&self, features: &mut [f32], t: usize) -> TsResult<()> {
        if t == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        let expected = t * self.c;
        if features.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: features.len(),
            });
        }

        for ci in 0..self.c {
            // Compute mean over T
            let mut sum = 0.0_f32;
            for ti in 0..t {
                sum += features[ti * self.c + ci];
            }
            let mean = sum / t as f32;

            // Compute variance
            let mut var = 0.0_f32;
            for ti in 0..t {
                let d = features[ti * self.c + ci] - mean;
                var += d * d;
            }
            let inv_std = 1.0 / (var / t as f32 + self.eps).sqrt();

            let gamma = self.gamma.as_ref().map(|g| g[ci]).unwrap_or(1.0);
            let beta = self.beta.as_ref().map(|b| b[ci]).unwrap_or(0.0);

            for ti in 0..t {
                let x = features[ti * self.c + ci];
                features[ti * self.c + ci] = (x - mean) * inv_std * gamma + beta;
            }
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_norm_new_ok() {
        let ln = InstanceNorm1d::new(8).expect("ok");
        assert_eq!(ln.c, 8);
        assert!(ln.gamma.is_none());
    }

    #[test]
    fn instance_norm_zero_c_error() {
        assert!(matches!(
            InstanceNorm1d::new(0).unwrap_err(),
            TsError::InvalidNumVariates(0)
        ));
    }

    #[test]
    fn instance_norm_with_affine() {
        let ln = InstanceNorm1d::with_affine(4).expect("ok");
        assert!(ln.gamma.is_some());
        assert!(ln.beta.is_some());
    }

    #[test]
    fn instance_norm_forward_zero_mean() {
        let ln = InstanceNorm1d::new(4).expect("ok");
        let mut features: Vec<f32> = (0..20 * 4).map(|i| i as f32).collect();
        ln.forward(&mut features, 20).expect("ok");
        for ci in 0..4 {
            let s: f32 = (0..20).map(|ti| features[ti * 4 + ci]).sum::<f32>() / 20.0;
            assert!(s.abs() < 1e-4, "channel {ci} mean={s}");
        }
    }

    #[test]
    fn instance_norm_forward_unit_variance() {
        let ln = InstanceNorm1d::new(2).expect("ok");
        let mut features: Vec<f32> = (0..100 * 2).map(|i| (i as f32) * 0.3 - 15.0).collect();
        ln.forward(&mut features, 100).expect("ok");
        for ci in 0..2 {
            let mean: f32 = (0..100).map(|ti| features[ti * 2 + ci]).sum::<f32>() / 100.0;
            let var: f32 = (0..100)
                .map(|ti| {
                    let d = features[ti * 2 + ci] - mean;
                    d * d
                })
                .sum::<f32>()
                / 100.0;
            assert!((var - 1.0).abs() < 0.01, "channel {ci} var={var}");
        }
    }

    #[test]
    fn instance_norm_zero_t_error() {
        let ln = InstanceNorm1d::new(4).expect("ok");
        let mut f = vec![];
        assert!(matches!(
            ln.forward(&mut f, 0).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }

    #[test]
    fn instance_norm_dim_mismatch() {
        let ln = InstanceNorm1d::new(4).expect("ok");
        let mut f = vec![0.0_f32; 5];
        assert!(matches!(
            ln.forward(&mut f, 2).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn instance_norm_affine_identity() {
        // gamma=1, beta=0 should give same result as no-affine
        let ln_plain = InstanceNorm1d::new(4).expect("ok");
        let ln_affine = InstanceNorm1d::with_affine(4).expect("ok");
        let raw: Vec<f32> = (0..16 * 4).map(|i| i as f32 * 0.1 - 0.8).collect();
        let mut a = raw.clone();
        let mut b = raw.clone();
        ln_plain.forward(&mut a, 16).expect("ok");
        ln_affine.forward(&mut b, 16).expect("ok");
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!((ai - bi).abs() < 1e-5, "idx={i} plain={ai} affine={bi}");
        }
    }
}

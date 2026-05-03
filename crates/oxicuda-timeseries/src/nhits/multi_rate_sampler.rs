//! Multi-rate average-pooling sampler for NHiTS.
//!
//! Each NHiTS stack operates at a different temporal resolution.  This module
//! provides average-pooling downsampling (`forward`) and nearest-neighbour
//! upsampling (`upsample`) that together implement the resolution changes
//! described in Challu et al. 2022.

use crate::error::{TsError, TsResult};

/// Average-pooling / nearest-neighbour-upsampling operator.
///
/// `pool_size` controls how many input timesteps are collapsed into one output
/// timestep.  Pool size 1 is the identity (no downsampling).
#[derive(Debug, Clone)]
pub struct MultiRateSampler {
    /// Number of consecutive timesteps averaged per output timestep.
    pub pool_size: usize,
}

impl MultiRateSampler {
    /// Construct a sampler.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::InvalidPoolSize`] when `pool_size == 0`.
    pub fn new(pool_size: usize) -> TsResult<Self> {
        if pool_size == 0 {
            return Err(TsError::InvalidPoolSize(0));
        }
        Ok(Self { pool_size })
    }

    /// Number of timesteps produced by pooling a `[T, C]` input.
    ///
    /// Returns `T / pool_size` using integer (floor) division.
    #[must_use]
    pub fn output_len(&self, t: usize) -> usize {
        t / self.pool_size
    }

    /// Average-pool a `[T, C]` tensor to `[T_out, C]` where `T_out = T / pool_size`.
    ///
    /// Each output timestep is the arithmetic mean of `pool_size` consecutive
    /// input timesteps (non-overlapping, floor-aligned):
    /// `y[j, c] = (1/P) * Σ_{k=0}^{P-1} x[j*P + k, c]`
    ///
    /// When `T < pool_size` the output has 0 rows (empty `Vec`).
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != t * c`.
    pub fn forward(&self, x: &[f32], t: usize, c: usize) -> TsResult<Vec<f32>> {
        if x.len() != t * c {
            return Err(TsError::DimensionMismatch {
                expected: t * c,
                got: x.len(),
            });
        }

        let t_out = self.output_len(t);
        let mut out = vec![0.0_f32; t_out * c];
        let inv_p = 1.0_f32 / self.pool_size as f32;

        for j in 0..t_out {
            for k in 0..self.pool_size {
                let src_t = j * self.pool_size + k;
                for ci in 0..c {
                    out[j * c + ci] += x[src_t * c + ci];
                }
            }
            for ci in 0..c {
                out[j * c + ci] *= inv_p;
            }
        }
        Ok(out)
    }

    /// Nearest-neighbour upsample `[T_in, C]` → `[T_out, C]`.
    ///
    /// Each input timestep `j` is repeated for all output timesteps
    /// `floor(i * T_in / T_out)` to achieve the desired output length, i.e.
    /// `y[i, c] = x[floor(i * T_in / T_out), c]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != t_in * c`.
    pub fn upsample(&self, x: &[f32], t_in: usize, t_out: usize, c: usize) -> TsResult<Vec<f32>> {
        if x.len() != t_in * c {
            return Err(TsError::DimensionMismatch {
                expected: t_in * c,
                got: x.len(),
            });
        }

        let mut out = vec![0.0_f32; t_out * c];
        if t_in == 0 || t_out == 0 {
            return Ok(out);
        }

        for i in 0..t_out {
            // Nearest-neighbour source index: floor(i * t_in / t_out)
            let src = (i * t_in) / t_out;
            let src_clamped = src.min(t_in - 1);
            for ci in 0..c {
                out[i * c + ci] = x[src_clamped * c + ci];
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_rate_sampler_zero_pool_error() {
        assert!(matches!(
            MultiRateSampler::new(0).unwrap_err(),
            TsError::InvalidPoolSize(0)
        ));
    }

    #[test]
    fn multi_rate_sampler_output_len() {
        let s = MultiRateSampler::new(4).expect("ok");
        assert_eq!(s.output_len(12), 3);
        assert_eq!(s.output_len(13), 3); // floor division
        assert_eq!(s.output_len(3), 0);
    }

    #[test]
    fn multi_rate_sampler_forward_shape() {
        let s = MultiRateSampler::new(2).expect("ok");
        let t = 20;
        let c = 4;
        let x = vec![1.0_f32; t * c];
        let out = s.forward(&x, t, c).expect("ok");
        assert_eq!(out.len(), s.output_len(t) * c);
        assert_eq!(out.len(), 10 * 4);
    }

    #[test]
    fn multi_rate_sampler_pool1_identity() {
        let s = MultiRateSampler::new(1).expect("ok");
        let t = 8;
        let c = 3;
        let x: Vec<f32> = (0..t * c).map(|i| i as f32).collect();
        let out = s.forward(&x, t, c).expect("ok");
        assert_eq!(out, x);
    }

    #[test]
    fn multi_rate_sampler_forward_constant_input() {
        // Averaging a constant field must yield the same constant.
        let s = MultiRateSampler::new(3).expect("ok");
        let x = vec![7.5_f32; 12 * 2];
        let out = s.forward(&x, 12, 2).expect("ok");
        for &v in &out {
            assert!((v - 7.5).abs() < 1e-6, "expected 7.5, got {v}");
        }
    }

    #[test]
    fn multi_rate_sampler_upsample_shape() {
        let s = MultiRateSampler::new(2).expect("ok");
        let t_in = 5;
        let t_out = 10;
        let c = 3;
        let x: Vec<f32> = (0..t_in * c).map(|i| i as f32).collect();
        let out = s.upsample(&x, t_in, t_out, c).expect("ok");
        assert_eq!(out.len(), t_out * c);
    }

    #[test]
    fn multi_rate_sampler_upsample_dim_mismatch() {
        let s = MultiRateSampler::new(2).expect("ok");
        let x = vec![0.0_f32; 5]; // wrong
        assert!(matches!(
            s.upsample(&x, 3, 6, 2).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn multi_rate_sampler_forward_dim_mismatch() {
        let s = MultiRateSampler::new(2).expect("ok");
        let x = vec![0.0_f32; 7]; // wrong
        assert!(matches!(
            s.forward(&x, 4, 2).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }
}

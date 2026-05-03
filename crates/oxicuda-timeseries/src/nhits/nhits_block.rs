//! Single NHiTS block: pool → MLP → backcast/forecast heads.
//!
//! Implements the building block described in Challu et al. 2022:
//! 1. Average-pool the `[T, C]` input to `[T_pool, C]`.
//! 2. Flatten to `[T_pool * C]` and pass through a 2-layer MLP with ReLU.
//! 3. Two independent linear heads project to backcast `[T * C]` and
//!    forecast `[horizon * C]`.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;
use crate::nhits::multi_rate_sampler::MultiRateSampler;

#[inline]
fn relu_vec(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// Apply a dense linear layer: `out[j] = Σ_i w[j, i] * x[i] + b[j]`.
///
/// `w` has shape `[out_dim, in_dim]` (row-major), `b` has length `out_dim`.
fn linear(x: &[f32], w: &[f32], b: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = b.to_vec();
    for j in 0..out_dim {
        let row = &w[j * in_dim..(j + 1) * in_dim];
        out[j] += row
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>();
    }
    out
}

/// Xavier-uniform init scale: `sqrt(6 / (fan_in + fan_out))`.
fn xavier_fill(buf: &mut [f32], fan_in: usize, fan_out: usize, rng: &mut LcgRng) {
    let scale = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    rng.fill_normal(buf);
    for v in buf.iter_mut() {
        *v *= scale;
    }
}

/// One NHiTS block.
///
/// Processes `[T, C]` input through average pooling, a 2-layer MLP, then
/// two linear heads that produce the backcast and forecast components.
#[derive(Debug, Clone)]
pub struct NHitsBlock {
    /// Pooling operator.
    pub pool: MultiRateSampler,
    /// MLP first-layer weight `[mlp_units, T_pool * C]`.
    pub mlp_w1: Vec<f32>,
    /// MLP first-layer bias `[mlp_units]`.
    pub mlp_b1: Vec<f32>,
    /// MLP second-layer weight `[mlp_units, mlp_units]`.
    pub mlp_w2: Vec<f32>,
    /// MLP second-layer bias `[mlp_units]`.
    pub mlp_b2: Vec<f32>,
    /// Backcast head weight `[T * C, mlp_units]`.
    pub backcast_w: Vec<f32>,
    /// Backcast head bias `[T * C]`.
    pub backcast_b: Vec<f32>,
    /// Forecast head weight `[horizon * C, mlp_units]`.
    pub forecast_w: Vec<f32>,
    /// Forecast head bias `[horizon * C]`.
    pub forecast_b: Vec<f32>,
    /// Input sequence length.
    pub t: usize,
    /// Number of channels.
    pub c: usize,
    /// Forecast horizon length.
    pub horizon: usize,
    /// MLP hidden units.
    pub mlp_units: usize,
    /// Pooling stride.
    pub pool_size: usize,
}

impl NHitsBlock {
    /// Construct a `NHitsBlock` with Xavier-initialised weights.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidPoolSize`] when `pool_size == 0` (propagated from
    ///   [`MultiRateSampler::new`]).
    /// - [`TsError::InvalidHorizon`] when `horizon == 0`.
    pub fn new(
        t: usize,
        c: usize,
        horizon: usize,
        pool_size: usize,
        mlp_units: usize,
        rng: &mut LcgRng,
    ) -> TsResult<Self> {
        if horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }

        let pool = MultiRateSampler::new(pool_size)?;
        let t_pool = pool.output_len(t).max(1); // at least 1 to avoid zero-dim MLP
        let flat_in = t_pool * c;

        let mut mlp_w1 = vec![0.0_f32; mlp_units * flat_in];
        xavier_fill(&mut mlp_w1, flat_in, mlp_units, rng);
        let mlp_b1 = vec![0.0_f32; mlp_units];

        let mut mlp_w2 = vec![0.0_f32; mlp_units * mlp_units];
        xavier_fill(&mut mlp_w2, mlp_units, mlp_units, rng);
        let mlp_b2 = vec![0.0_f32; mlp_units];

        let mut backcast_w = vec![0.0_f32; t * c * mlp_units];
        xavier_fill(&mut backcast_w, mlp_units, t * c, rng);
        let backcast_b = vec![0.0_f32; t * c];

        let mut forecast_w = vec![0.0_f32; horizon * c * mlp_units];
        xavier_fill(&mut forecast_w, mlp_units, horizon * c, rng);
        let forecast_b = vec![0.0_f32; horizon * c];

        Ok(Self {
            pool,
            mlp_w1,
            mlp_b1,
            mlp_w2,
            mlp_b2,
            backcast_w,
            backcast_b,
            forecast_w,
            forecast_b,
            t,
            c,
            horizon,
            mlp_units,
            pool_size,
        })
    }

    /// Forward pass.
    ///
    /// # Returns
    ///
    /// `(backcast, forecast)` where backcast has shape `[T, C]` and forecast
    /// has shape `[horizon, C]` (both flat row-major).
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != self.t * self.c`.
    pub fn forward(&self, x: &[f32]) -> TsResult<(Vec<f32>, Vec<f32>)> {
        let expected = self.t * self.c;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // Pool the input to reduce temporal resolution.
        let pooled = self.pool.forward(x, self.t, self.c)?;
        let t_pool_actual = self.pool.output_len(self.t).max(1);

        // When pool produces 0 rows (T < pool_size), use the first timestep as a
        // degenerate 1-row stand-in so the MLP always has non-empty input.
        let flat_in = t_pool_actual * self.c;
        let flat: Vec<f32> = if pooled.is_empty() {
            x[..self.c.min(x.len())].to_vec()
        } else {
            let actual_flat = pooled.len().min(flat_in);
            let mut f = vec![0.0_f32; flat_in];
            f[..actual_flat].copy_from_slice(&pooled[..actual_flat]);
            f
        };

        // MLP layer 1
        let mut h1 = linear(&flat, &self.mlp_w1, &self.mlp_b1, flat_in, self.mlp_units);
        relu_vec(&mut h1);

        // MLP layer 2
        let mut h2 = linear(
            &h1,
            &self.mlp_w2,
            &self.mlp_b2,
            self.mlp_units,
            self.mlp_units,
        );
        relu_vec(&mut h2);

        // Backcast head
        let backcast = linear(
            &h2,
            &self.backcast_w,
            &self.backcast_b,
            self.mlp_units,
            self.t * self.c,
        );

        // Forecast head
        let forecast = linear(
            &h2,
            &self.forecast_w,
            &self.forecast_b,
            self.mlp_units,
            self.horizon * self.c,
        );

        Ok((backcast, forecast))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn nhits_block_backcast_shape() {
        let mut rng = make_rng();
        let block = NHitsBlock::new(24, 3, 8, 2, 64, &mut rng).expect("ok");
        let x = vec![0.5_f32; 24 * 3];
        let (bc, _fc) = block.forward(&x).expect("ok");
        assert_eq!(bc.len(), 24 * 3);
    }

    #[test]
    fn nhits_block_forecast_shape() {
        let mut rng = make_rng();
        let block = NHitsBlock::new(24, 3, 8, 2, 64, &mut rng).expect("ok");
        let x = vec![0.5_f32; 24 * 3];
        let (_bc, fc) = block.forward(&x).expect("ok");
        assert_eq!(fc.len(), 8 * 3);
    }

    #[test]
    fn nhits_block_output_finite() {
        let mut rng = make_rng();
        let block = NHitsBlock::new(16, 4, 4, 2, 32, &mut rng).expect("ok");
        let mut x = vec![0.0_f32; 16 * 4];
        rng.fill_normal(&mut x);
        let (bc, fc) = block.forward(&x).expect("ok");
        assert!(bc.iter().all(|v| v.is_finite()), "backcast non-finite");
        assert!(fc.iter().all(|v| v.is_finite()), "forecast non-finite");
    }

    #[test]
    fn nhits_block_zero_horizon_error() {
        let mut rng = make_rng();
        assert!(matches!(
            NHitsBlock::new(16, 4, 0, 2, 32, &mut rng).unwrap_err(),
            TsError::InvalidHorizon(0)
        ));
    }

    #[test]
    fn nhits_block_zero_pool_error() {
        let mut rng = make_rng();
        assert!(matches!(
            NHitsBlock::new(16, 4, 8, 0, 32, &mut rng).unwrap_err(),
            TsError::InvalidPoolSize(0)
        ));
    }

    #[test]
    fn nhits_block_dim_mismatch_error() {
        let mut rng = make_rng();
        let block = NHitsBlock::new(16, 4, 8, 2, 32, &mut rng).expect("ok");
        let x = vec![0.0_f32; 5]; // wrong
        assert!(matches!(
            block.forward(&x).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn nhits_block_pool_size_1() {
        // pool_size=1 is identity — block should still forward correctly
        let mut rng = make_rng();
        let block = NHitsBlock::new(12, 2, 6, 1, 32, &mut rng).expect("ok");
        let x = vec![1.0_f32; 12 * 2];
        let (bc, fc) = block.forward(&x).expect("ok");
        assert_eq!(bc.len(), 12 * 2);
        assert_eq!(fc.len(), 6 * 2);
    }

    #[test]
    fn nhits_block_large_pool() {
        // pool_size > T: output_len is 0, block should handle gracefully
        let mut rng = make_rng();
        let block = NHitsBlock::new(4, 2, 4, 8, 16, &mut rng).expect("ok");
        let x = vec![1.0_f32; 4 * 2];
        let (bc, fc) = block.forward(&x).expect("ok");
        assert_eq!(bc.len(), 4 * 2);
        assert_eq!(fc.len(), 4 * 2);
    }
}

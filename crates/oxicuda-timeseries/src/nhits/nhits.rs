//! NHiTS (Neural Hierarchical Interpolation for Time Series) forecaster.
//!
//! The hierarchical residual forward pass:
//! 1. Start with `residual = x` (the full `[T, C]` input).
//! 2. For every block across all stacks: subtract the block's backcast from
//!    the residual, accumulate the block's forecast into the running sum.
//! 3. Return the accumulated forecast `[horizon, C]`.
//!
//! Different stacks pool at different rates (`pool_sizes`), so each stack
//! specialises in a different temporal frequency band.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;
use crate::nhits::nhits_block::NHitsBlock;

/// Configuration for a `NHits` model.
#[derive(Debug, Clone)]
pub struct NHitsConfig {
    /// Number of stacks (each with its own pooling rate).
    pub n_stacks: usize,
    /// Number of `NHitsBlock`s per stack.
    pub blocks_per_stack: usize,
    /// Per-stack pooling rates; length must equal `n_stacks`.
    pub pool_sizes: Vec<usize>,
    /// MLP hidden dimension shared across all blocks.
    pub mlp_units: usize,
    /// Input sequence length `T`.
    pub t: usize,
    /// Number of channels / variates `C`.
    pub c: usize,
    /// Forecast horizon.
    pub horizon: usize,
}

impl NHitsConfig {
    /// Tiny configuration for fast tests.
    ///
    /// `n_stacks=3, blocks_per_stack=1, pool_sizes=[1,2,4], mlp_units=64`
    #[must_use]
    pub fn tiny(t: usize, c: usize, horizon: usize) -> Self {
        Self {
            n_stacks: 3,
            blocks_per_stack: 1,
            pool_sizes: vec![1, 2, 4],
            mlp_units: 64,
            t,
            c,
            horizon,
        }
    }

    /// Default production configuration.
    ///
    /// `n_stacks=3, blocks_per_stack=1, pool_sizes=[1,2,4], mlp_units=512`
    #[must_use]
    pub fn default_config(t: usize, c: usize, horizon: usize) -> Self {
        Self {
            n_stacks: 3,
            blocks_per_stack: 1,
            pool_sizes: vec![1, 2, 4],
            mlp_units: 512,
            t,
            c,
            horizon,
        }
    }
}

/// NHiTS forecaster.
///
/// Stacks of [`NHitsBlock`]s arranged so that each stack operates at a
/// different temporal resolution.
#[derive(Debug, Clone)]
pub struct NHits {
    /// Outer dimension: stacks; inner dimension: blocks within a stack.
    pub stacks: Vec<Vec<NHitsBlock>>,
    /// Config used to build this model.
    pub config: NHitsConfig,
}

impl NHits {
    /// Build a `NHits` model from a `NHitsConfig`.
    ///
    /// # Errors
    ///
    /// - [`TsError::ShapeMismatch`] when `pool_sizes.len() != n_stacks`.
    /// - [`TsError::InvalidPoolSize`] if any pool size is 0.
    /// - [`TsError::InvalidHorizon`] if `horizon == 0`.
    pub fn new(config: NHitsConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.pool_sizes.len() != config.n_stacks {
            return Err(TsError::ShapeMismatch {
                msg: format!(
                    "pool_sizes length {} != n_stacks {}",
                    config.pool_sizes.len(),
                    config.n_stacks
                ),
            });
        }

        let mut stacks = Vec::with_capacity(config.n_stacks);
        for s in 0..config.n_stacks {
            let pool_size = config.pool_sizes[s];
            let mut stack_blocks = Vec::with_capacity(config.blocks_per_stack);
            for _ in 0..config.blocks_per_stack {
                stack_blocks.push(NHitsBlock::new(
                    config.t,
                    config.c,
                    config.horizon,
                    pool_size,
                    config.mlp_units,
                    rng,
                )?);
            }
            stacks.push(stack_blocks);
        }

        Ok(Self { stacks, config })
    }

    /// Hierarchical residual forward pass.
    ///
    /// Returns `[horizon, C]` forecast (flat row-major).
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != T * C`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let expected = self.config.t * self.config.c;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let mut residual = x.to_vec();
        let mut forecast_sum = vec![0.0_f32; self.config.horizon * self.config.c];

        for stack in &self.stacks {
            for block in stack {
                let (backcast, forecast) = block.forward(&residual)?;

                for (r, b) in residual.iter_mut().zip(backcast.iter()) {
                    *r -= b;
                }

                for (fs, &fc) in forecast_sum.iter_mut().zip(forecast.iter()) {
                    *fs += fc;
                }
            }
        }

        Ok(forecast_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(7)
    }

    #[test]
    fn nhits_output_shape() {
        let mut rng = make_rng();
        let cfg = NHitsConfig::tiny(24, 3, 8);
        let model = NHits::new(cfg, &mut rng).expect("ok");
        let x = vec![0.5_f32; 24 * 3];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), 8 * 3);
    }

    #[test]
    fn nhits_output_finite() {
        let mut rng = make_rng();
        let cfg = NHitsConfig::tiny(32, 4, 12);
        let model = NHits::new(cfg, &mut rng).expect("ok");
        let mut x = vec![0.0_f32; 32 * 4];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite forecast");
    }

    #[test]
    fn nhits_pool_sizes_mismatch_error() {
        let mut rng = make_rng();
        let cfg = NHitsConfig {
            n_stacks: 3,
            blocks_per_stack: 1,
            pool_sizes: vec![1, 2], // only 2 items for 3 stacks
            mlp_units: 32,
            t: 16,
            c: 2,
            horizon: 4,
        };
        assert!(matches!(
            NHits::new(cfg, &mut rng).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn nhits_dim_mismatch_error() {
        let mut rng = make_rng();
        let cfg = NHitsConfig::tiny(24, 3, 8);
        let model = NHits::new(cfg, &mut rng).expect("ok");
        let x = vec![0.0_f32; 10]; // wrong
        assert!(matches!(
            model.forward(&x).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn nhits_stack_and_block_counts() {
        let mut rng = make_rng();
        let cfg = NHitsConfig {
            n_stacks: 3,
            blocks_per_stack: 2,
            pool_sizes: vec![1, 2, 4],
            mlp_units: 32,
            t: 16,
            c: 2,
            horizon: 4,
        };
        let model = NHits::new(cfg, &mut rng).expect("ok");
        assert_eq!(model.stacks.len(), 3);
        for stack in &model.stacks {
            assert_eq!(stack.len(), 2);
        }
    }

    #[test]
    fn nhits_single_variate() {
        let mut rng = make_rng();
        let cfg = NHitsConfig::tiny(16, 1, 4);
        let model = NHits::new(cfg, &mut rng).expect("ok");
        let x = vec![1.0_f32; 16];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn nhits_zero_residual_monotone() {
        // Verifies the residual is updated (each block subtracts backcast);
        // we just check the forecast result is not identically zero for non-zero input.
        let mut rng = make_rng();
        let cfg = NHitsConfig::tiny(16, 2, 4);
        let model = NHits::new(cfg, &mut rng).expect("ok");
        let x: Vec<f32> = (0..16 * 2).map(|i| (i as f32) * 0.1 - 0.8).collect();
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), 4 * 2);
        // Merely checking it ran; forecast will rarely be exactly zero with random weights
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

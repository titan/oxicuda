//! N-BEATS forecaster: hierarchical stacks of basis-expansion blocks.
//!
//! Implements the double-residual stacking scheme from Oreshkin et al. (ICLR 2020).
//! Each block produces a **backcast** (subtracted from the running residual) and a
//! **forecast** (accumulated into the output). The final forecast is the sum of all
//! block forecasts.

use super::nbeats_block::{NBeatsBlock, NBeatsBlockType};
use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── NBeatsConfig ─────────────────────────────────────────────────────────────

/// Configuration for an N-BEATS model.
#[derive(Debug, Clone)]
pub struct NBeatsConfig {
    /// Input sequence length `T`.
    pub t: usize,
    /// Number of channels / variates `C`.
    pub c: usize,
    /// Forecast horizon `H`.
    pub horizon: usize,
    /// Stack layout: outer vec = stacks, inner vec = block types within a stack.
    ///
    /// E.g., `[[Trend, Trend], [Seasonality, Seasonality], [Generic, Generic]]`
    /// gives a 3-stack interpretable N-BEATS.
    pub stacks: Vec<Vec<NBeatsBlockType>>,
}

impl NBeatsConfig {
    /// Tiny config for fast tests: 1 stack with 1 Generic block.
    #[must_use]
    pub fn tiny(t: usize, c: usize, horizon: usize) -> Self {
        Self {
            t,
            c,
            horizon,
            stacks: vec![vec![NBeatsBlockType::Generic {
                hidden_units: 32,
                theta_dim: 8,
            }]],
        }
    }

    /// Interpretable N-BEATS config: Trend stack + Seasonality stack + Generic stack.
    #[must_use]
    pub fn interpretable(t: usize, c: usize, horizon: usize) -> Self {
        Self {
            t,
            c,
            horizon,
            stacks: vec![
                vec![
                    NBeatsBlockType::Trend {
                        hidden_units: 256,
                        degree: 2,
                    },
                    NBeatsBlockType::Trend {
                        hidden_units: 256,
                        degree: 2,
                    },
                ],
                vec![
                    NBeatsBlockType::Seasonality {
                        hidden_units: 256,
                        n_harmonics: 4,
                    },
                    NBeatsBlockType::Seasonality {
                        hidden_units: 256,
                        n_harmonics: 4,
                    },
                ],
                vec![
                    NBeatsBlockType::Generic {
                        hidden_units: 256,
                        theta_dim: 32,
                    },
                    NBeatsBlockType::Generic {
                        hidden_units: 256,
                        theta_dim: 32,
                    },
                ],
            ],
        }
    }
}

// ─── NBeats ───────────────────────────────────────────────────────────────────

/// N-BEATS forecaster.
///
/// Hierarchical stacks of [`NBeatsBlock`]s arranged so that each block subtracts
/// its backcast from the running residual and adds its forecast to the running
/// accumulator. The returned forecast `[H × C]` is the total accumulated sum.
#[derive(Debug, Clone)]
pub struct NBeats {
    /// `[n_stacks][blocks_per_stack]` array of N-BEATS blocks.
    pub stacks: Vec<Vec<NBeatsBlock>>,
    /// Config used to build this model.
    pub config: NBeatsConfig,
}

impl NBeats {
    /// Build an `NBeats` model from the supplied `NBeatsConfig`.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `t == 0`.
    /// - [`TsError::InvalidHorizon`] when `horizon == 0`.
    /// - [`TsError::InvalidNumVariates`] when `c == 0`.
    /// - [`TsError::EmptyInput`] when `stacks` is empty.
    /// - Any block-construction error propagated from [`NBeatsBlock::new`].
    pub fn new(config: NBeatsConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.t == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if config.horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if config.c == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        if config.stacks.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "stacks must not be empty".to_string(),
            });
        }

        let mut stacks = Vec::with_capacity(config.stacks.len());
        for stack_cfg in &config.stacks {
            let mut stack_blocks = Vec::with_capacity(stack_cfg.len());
            for block_type in stack_cfg {
                stack_blocks.push(NBeatsBlock::new(
                    config.t,
                    config.c,
                    config.horizon,
                    block_type.clone(),
                    rng,
                )?);
            }
            stacks.push(stack_blocks);
        }

        Ok(Self { stacks, config })
    }

    /// Hierarchical residual forward pass.
    ///
    /// Returns `[H × C]` forecast (flat row-major).
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
        let mut total_forecast = vec![0.0_f32; self.config.horizon * self.config.c];

        for stack in &self.stacks {
            for block in stack {
                let (backcast, forecast) = block.forward(&residual)?;

                for (r, b) in residual.iter_mut().zip(backcast.iter()) {
                    *r -= b;
                }
                for (tf, &fc) in total_forecast.iter_mut().zip(forecast.iter()) {
                    *tf += fc;
                }
            }
        }

        Ok(total_forecast)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(7)
    }

    #[test]
    fn nbeats_forward_shape() {
        // 3-stack interpretable model.
        let mut rng = make_rng();
        let t = 24;
        let c = 2;
        let horizon = 8;
        let cfg = NBeatsConfig {
            t,
            c,
            horizon,
            stacks: vec![
                vec![NBeatsBlockType::Trend {
                    hidden_units: 32,
                    degree: 2,
                }],
                vec![NBeatsBlockType::Seasonality {
                    hidden_units: 32,
                    n_harmonics: 3,
                }],
                vec![NBeatsBlockType::Generic {
                    hidden_units: 32,
                    theta_dim: 8,
                }],
            ],
        };
        let model = NBeats::new(cfg, &mut rng).expect("ok");
        let x = vec![0.1_f32; t * c];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), horizon * c);
    }

    #[test]
    fn nbeats_forward_single_stack() {
        let mut rng = make_rng();
        let t = 16;
        let c = 1;
        let horizon = 4;
        let cfg = NBeatsConfig::tiny(t, c, horizon);
        let model = NBeats::new(cfg, &mut rng).expect("ok");
        let x = vec![0.5_f32; t * c];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), horizon * c);
    }

    #[test]
    fn nbeats_residual_decreases() {
        // After one block the residual has changed from the original input.
        // We verify this indirectly: two differently seeded models produce different forecasts.
        let t = 16;
        let c = 2;
        let horizon = 4;
        let cfg = NBeatsConfig::tiny(t, c, horizon);

        let mut rng_a = LcgRng::new(10);
        let model_a = NBeats::new(cfg.clone(), &mut rng_a).expect("ok");

        let mut rng_b = LcgRng::new(20);
        let model_b = NBeats::new(cfg, &mut rng_b).expect("ok");

        let x = vec![1.0_f32; t * c];
        let out_a = model_a.forward(&x).expect("ok");
        let out_b = model_b.forward(&x).expect("ok");

        // With different weights the two forecasts will differ.
        let same = out_a
            .iter()
            .zip(out_b.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(
            !same,
            "two models with different seeds produced identical forecasts"
        );
    }

    #[test]
    fn nbeats_forward_finite() {
        let mut rng = make_rng();
        let t = 24;
        let c = 3;
        let horizon = 8;
        // Interpretable N-BEATS with small hidden units for speed.
        let cfg = NBeatsConfig {
            t,
            c,
            horizon,
            stacks: vec![
                vec![
                    NBeatsBlockType::Trend {
                        hidden_units: 32,
                        degree: 2,
                    },
                    NBeatsBlockType::Trend {
                        hidden_units: 32,
                        degree: 2,
                    },
                ],
                vec![
                    NBeatsBlockType::Seasonality {
                        hidden_units: 32,
                        n_harmonics: 4,
                    },
                    NBeatsBlockType::Seasonality {
                        hidden_units: 32,
                        n_harmonics: 4,
                    },
                ],
                vec![
                    NBeatsBlockType::Generic {
                        hidden_units: 32,
                        theta_dim: 8,
                    },
                    NBeatsBlockType::Generic {
                        hidden_units: 32,
                        theta_dim: 8,
                    },
                ],
            ],
        };
        let model = NBeats::new(cfg, &mut rng).expect("ok");
        let mut x = vec![0.0_f32; t * c];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite forecast");
    }

    // ── Error cases ────────────────────────────────────────────────────────────

    #[test]
    fn nbeats_err_invalid_horizon() {
        let mut rng = make_rng();
        let cfg = NBeatsConfig {
            t: 16,
            c: 2,
            horizon: 0,
            stacks: vec![vec![NBeatsBlockType::Generic {
                hidden_units: 16,
                theta_dim: 4,
            }]],
        };
        assert!(matches!(
            NBeats::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidHorizon(0)
        ));
    }

    #[test]
    fn nbeats_err_empty_stacks() {
        let mut rng = make_rng();
        let cfg = NBeatsConfig {
            t: 16,
            c: 2,
            horizon: 4,
            stacks: vec![],
        };
        assert!(matches!(
            NBeats::new(cfg, &mut rng).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }

    #[test]
    fn nbeats_err_dimension_mismatch() {
        let mut rng = make_rng();
        let cfg = NBeatsConfig::tiny(16, 2, 4);
        let model = NBeats::new(cfg, &mut rng).expect("ok");
        assert!(matches!(
            model.forward(&[0.0_f32; 5]).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn nbeats_err_invalid_c() {
        let mut rng = make_rng();
        let cfg = NBeatsConfig {
            t: 16,
            c: 0,
            horizon: 4,
            stacks: vec![vec![NBeatsBlockType::Generic {
                hidden_units: 16,
                theta_dim: 4,
            }]],
        };
        assert!(matches!(
            NBeats::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidNumVariates(0)
        ));
    }

    #[test]
    fn nbeats_err_invalid_sequence_length() {
        let mut rng = make_rng();
        let cfg = NBeatsConfig {
            t: 0,
            c: 2,
            horizon: 4,
            stacks: vec![vec![NBeatsBlockType::Generic {
                hidden_units: 16,
                theta_dim: 4,
            }]],
        };
        assert!(matches!(
            NBeats::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }
}

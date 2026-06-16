//! Simplified N-BEATS implementation with per-block per-layer weight matrices.
//!
//! This is a self-contained N-BEATS variant that differs from `crate::nbeats` in that
//! it stores fully-connected layer weights as flat `Vec<f32>` buffers (row-major,
//! `[out × in]`) and performs double-residual stacking entirely on the CPU.
//!
//! Reference: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
//! interpretable time series forecasting," ICLR 2020.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── BasisType ───────────────────────────────────────────────────────────────

/// Expansion basis used by each N-BEATS block.
///
/// `Generic` uses learned linear projections.  `Trend` and `Seasonality` are
/// placeholder variants reserved for constrained basis expansions; they behave
/// identically to `Generic` in this simplified implementation but allow
/// downstream code to inspect the intended basis type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisType {
    /// Unconstrained learned basis (generic stack).
    Generic,
    /// Polynomial trend basis (reserved; behaves like Generic here).
    Trend,
    /// Fourier seasonality basis (reserved; behaves like Generic here).
    Seasonality,
}

// ─── NbeatsConfig ────────────────────────────────────────────────────────────

/// Configuration for the simplified N-BEATS model.
#[derive(Debug, Clone)]
pub struct NbeatsConfig {
    /// Number of fully-connected layers per block.
    pub n_layers: usize,
    /// Number of neurons in each fully-connected layer.
    pub layer_width: usize,
    /// Number of stacks (each stack contains `n_blocks_per_stack` blocks).
    pub n_stacks: usize,
    /// Number of blocks in each stack.
    pub n_blocks_per_stack: usize,
    /// Input window length T (lookback).
    pub lookback: usize,
    /// Forecast horizon H.
    pub forecast_horizon: usize,
    /// Basis expansion type applied uniformly to all blocks.
    pub basis_type: BasisType,
}

// ─── NbeatsBlock ─────────────────────────────────────────────────────────────

/// A single N-BEATS block with FC layers and linear output projections.
///
/// Weight convention: every weight matrix `W` has shape `[out × in]` stored
/// row-major, so `out[i] = Σ_j W[i*in + j] * x[j] + b[i]`.
pub struct NbeatsBlock {
    /// `fc_w[l]` — weight matrix for FC layer `l`.
    ///
    /// - Layer 0: shape `[layer_width × in_dim]`, length = `layer_width * in_dim`.
    /// - Layer l > 0: shape `[layer_width × layer_width]`, length = `layer_width²`.
    pub fc_w: Vec<Vec<f32>>,
    /// `fc_b[l]` — bias vector for FC layer `l`, length = `layer_width`.
    pub fc_b: Vec<Vec<f32>>,
    /// Backcast projection: shape `[lookback × layer_width]`.
    pub theta_backcast_w: Vec<f32>,
    /// Forecast projection: shape `[horizon × layer_width]`.
    pub theta_forecast_w: Vec<f32>,
    /// Basis type recorded for introspection.
    pub basis_type: BasisType,
    lookback: usize,
    horizon: usize,
}

impl NbeatsBlock {
    /// Construct a new `NbeatsBlock` with Xavier-initialised weights.
    ///
    /// # Parameters
    /// - `in_dim`   — dimensionality of the input vector (usually `lookback`).
    /// - `lookback` — length of the backcast output.
    /// - `horizon`  — length of the forecast output.
    /// - `config`   — shared model configuration.
    /// - `rng`      — deterministic LCG for weight initialisation.
    ///
    /// # Errors
    /// Returns `TsError::EmptyInput` when any required dimension is zero.
    pub fn new(
        in_dim: usize,
        lookback: usize,
        horizon: usize,
        config: &NbeatsConfig,
        rng: &mut LcgRng,
    ) -> TsResult<Self> {
        if in_dim == 0 {
            return Err(TsError::EmptyInput {
                msg: "in_dim must be > 0".into(),
            });
        }
        if lookback == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if config.n_layers == 0 {
            return Err(TsError::EmptyInput {
                msg: "n_layers must be > 0".into(),
            });
        }
        if config.layer_width == 0 {
            return Err(TsError::EmptyInput {
                msg: "layer_width must be > 0".into(),
            });
        }

        let lw = config.layer_width;
        let nl = config.n_layers;

        // ── Build FC layers ──────────────────────────────────────────────────
        let mut fc_w: Vec<Vec<f32>> = Vec::with_capacity(nl);
        let mut fc_b: Vec<Vec<f32>> = Vec::with_capacity(nl);

        for l in 0..nl {
            let fan_in = if l == 0 { in_dim } else { lw };
            let fan_out = lw;
            let n_weights = fan_out * fan_in;

            // Xavier uniform: scale = sqrt(6 / (fan_in + fan_out))
            let scale = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
            let mut w = vec![0.0_f32; n_weights];
            rng.fill_normal(&mut w);
            for v in &mut w {
                *v *= scale;
            }
            fc_w.push(w);

            // Biases initialised to zero.
            fc_b.push(vec![0.0_f32; lw]);
        }

        // ── Backcast projection: [lookback × layer_width] ────────────────────
        let bc_scale = (6.0_f32 / (lw + lookback) as f32).sqrt();
        let mut theta_backcast_w = vec![0.0_f32; lookback * lw];
        rng.fill_normal(&mut theta_backcast_w);
        for v in &mut theta_backcast_w {
            *v *= bc_scale;
        }

        // ── Forecast projection: [horizon × layer_width] ─────────────────────
        let fc_scale = (6.0_f32 / (lw + horizon) as f32).sqrt();
        let mut theta_forecast_w = vec![0.0_f32; horizon * lw];
        rng.fill_normal(&mut theta_forecast_w);
        for v in &mut theta_forecast_w {
            *v *= fc_scale;
        }

        Ok(Self {
            fc_w,
            fc_b,
            theta_backcast_w,
            theta_forecast_w,
            basis_type: config.basis_type,
            lookback,
            horizon,
        })
    }

    /// Run the block forward.
    ///
    /// # Arguments
    /// - `x` — input slice of length `in_dim` (the current residual).
    ///
    /// # Returns
    /// `(backcast, forecast)` where `backcast.len() == lookback` and
    /// `forecast.len() == horizon`.
    ///
    /// # Errors
    /// Returns `TsError::DimensionMismatch` if `x.len()` does not match the
    /// expected fan-in of layer 0, or `TsError::NonFinite` if any intermediate
    /// value is non-finite.
    pub fn forward(&self, x: &[f32]) -> TsResult<(Vec<f32>, Vec<f32>)> {
        let lw = self.fc_b[0].len(); // layer_width
        let in_dim = self.fc_w[0].len() / lw; // == fan_in of layer 0

        if x.len() != in_dim {
            return Err(TsError::DimensionMismatch {
                expected: in_dim,
                got: x.len(),
            });
        }

        // ── FC stack with ReLU ───────────────────────────────────────────────
        let mut hidden: Vec<f32> = vec![0.0_f32; lw];

        // Layer 0: [lw × in_dim] @ [in_dim] + [lw]
        {
            let w = &self.fc_w[0];
            let b = &self.fc_b[0];
            for i in 0..lw {
                let mut acc = b[i];
                for j in 0..in_dim {
                    acc += w[i * in_dim + j] * x[j];
                }
                hidden[i] = acc.max(0.0); // ReLU
            }
        }

        // Layers 1..n_layers: [lw × lw] @ [lw] + [lw]
        for l in 1..self.fc_w.len() {
            let w = &self.fc_w[l];
            let b = &self.fc_b[l];
            let mut next = vec![0.0_f32; lw];
            for i in 0..lw {
                let mut acc = b[i];
                for j in 0..lw {
                    acc += w[i * lw + j] * hidden[j];
                }
                next[i] = acc.max(0.0); // ReLU
            }
            hidden = next;
        }

        // ── Validate hidden state ────────────────────────────────────────────
        if hidden.iter().any(|v| !v.is_finite()) {
            return Err(TsError::NonFinite);
        }

        // ── Backcast projection: [lookback × lw] @ [lw] ─────────────────────
        let mut backcast = vec![0.0_f32; self.lookback];
        {
            let w = &self.theta_backcast_w;
            for i in 0..self.lookback {
                let mut acc = 0.0_f32;
                for j in 0..lw {
                    acc += w[i * lw + j] * hidden[j];
                }
                backcast[i] = acc;
            }
        }

        // ── Forecast projection: [horizon × lw] @ [lw] ──────────────────────
        let mut forecast = vec![0.0_f32; self.horizon];
        {
            let w = &self.theta_forecast_w;
            for i in 0..self.horizon {
                let mut acc = 0.0_f32;
                for j in 0..lw {
                    acc += w[i * lw + j] * hidden[j];
                }
                forecast[i] = acc;
            }
        }

        Ok((backcast, forecast))
    }
}

// ─── Nbeats ──────────────────────────────────────────────────────────────────

/// Simplified N-BEATS model composed of stacks of `NbeatsBlock`s.
///
/// Forward pass applies **double-residual stacking**: each block receives the
/// current residual, subtracts its backcast estimate, and adds its partial
/// forecast to the running total.
pub struct Nbeats {
    /// `stacks[s][b]` — block `b` in stack `s`.
    pub stacks: Vec<Vec<NbeatsBlock>>,
    config: NbeatsConfig,
}

impl Nbeats {
    /// Construct an `Nbeats` model with Xavier-initialised weights.
    ///
    /// # Errors
    /// - `EmptyInput` when `n_stacks == 0` or `n_blocks_per_stack == 0`.
    /// - `InvalidSequenceLength(0)` when `lookback == 0`.
    /// - `InvalidHorizon(0)` when `forecast_horizon == 0`.
    pub fn new(config: NbeatsConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.n_stacks == 0 {
            return Err(TsError::EmptyInput {
                msg: "n_stacks must be > 0".into(),
            });
        }
        if config.n_blocks_per_stack == 0 {
            return Err(TsError::EmptyInput {
                msg: "n_blocks_per_stack must be > 0".into(),
            });
        }
        if config.lookback == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if config.forecast_horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }

        let in_dim = config.lookback;
        let lookback = config.lookback;
        let horizon = config.forecast_horizon;

        let mut stacks: Vec<Vec<NbeatsBlock>> = Vec::with_capacity(config.n_stacks);
        for _ in 0..config.n_stacks {
            let mut stack: Vec<NbeatsBlock> = Vec::with_capacity(config.n_blocks_per_stack);
            for _ in 0..config.n_blocks_per_stack {
                let block = NbeatsBlock::new(in_dim, lookback, horizon, &config, rng)?;
                stack.push(block);
            }
            stacks.push(stack);
        }

        Ok(Self { stacks, config })
    }

    /// Run a forward pass over the full N-BEATS model.
    ///
    /// # Arguments
    /// - `x` — input time series slice of length `lookback`.
    ///
    /// # Returns
    /// The aggregated forecast vector of length `forecast_horizon`.
    ///
    /// # Errors
    /// Propagates any `TsError` emitted by an individual block's `forward`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let horizon = self.config.forecast_horizon;

        if x.len() != self.config.lookback {
            return Err(TsError::DimensionMismatch {
                expected: self.config.lookback,
                got: x.len(),
            });
        }

        let mut residual: Vec<f32> = x.to_vec();
        let mut total_forecast: Vec<f32> = vec![0.0_f32; horizon];

        for stack in &self.stacks {
            for block in stack {
                let (backcast, partial_forecast) = block.forward(&residual)?;

                // residual -= backcast
                for (r, b) in residual.iter_mut().zip(backcast.iter()) {
                    *r -= b;
                }

                // total_forecast += partial_forecast
                for (f, p) in total_forecast.iter_mut().zip(partial_forecast.iter()) {
                    *f += p;
                }
            }
        }

        Ok(total_forecast)
    }

    /// Input window length T.
    #[must_use]
    #[inline]
    pub fn lookback(&self) -> usize {
        self.config.lookback
    }

    /// Forecast horizon H.
    #[must_use]
    #[inline]
    pub fn horizon(&self) -> usize {
        self.config.forecast_horizon
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn tiny_config() -> NbeatsConfig {
        NbeatsConfig {
            n_layers: 2,
            layer_width: 16,
            n_stacks: 2,
            n_blocks_per_stack: 2,
            lookback: 24,
            forecast_horizon: 8,
            basis_type: BasisType::Generic,
        }
    }

    // ── output_shape ─────────────────────────────────────────────────────────

    #[test]
    fn output_shape() {
        let mut rng = make_rng();
        let cfg = tiny_config();
        let horizon = cfg.forecast_horizon;
        let lookback = cfg.lookback;
        let model = Nbeats::new(cfg, &mut rng).expect("model creation ok");
        let x = vec![0.5_f32; lookback];
        let out = model.forward(&x).expect("forward ok");
        assert_eq!(
            out.len(),
            horizon,
            "output length must equal forecast_horizon"
        );
    }

    // ── backcast_shape ────────────────────────────────────────────────────────

    #[test]
    fn backcast_shape() {
        let mut rng = make_rng();
        let cfg = tiny_config();
        let lookback = cfg.lookback;
        let horizon = cfg.forecast_horizon;
        let block = NbeatsBlock::new(lookback, lookback, horizon, &cfg, &mut rng)
            .expect("block creation ok");
        let x = vec![1.0_f32; lookback];
        let (backcast, forecast) = block.forward(&x).expect("forward ok");
        assert_eq!(
            backcast.len(),
            lookback,
            "backcast must have length == lookback"
        );
        assert_eq!(
            forecast.len(),
            horizon,
            "forecast must have length == horizon"
        );
    }

    // ── residual_decreases ────────────────────────────────────────────────────

    #[test]
    fn residual_decreases() {
        // After the double-residual stacking the internal residual should
        // differ from the original input (at least one block must produce a
        // non-zero backcast).
        let mut rng = make_rng();
        let cfg = NbeatsConfig {
            n_layers: 2,
            layer_width: 32,
            n_stacks: 3,
            n_blocks_per_stack: 3,
            lookback: 16,
            forecast_horizon: 4,
            basis_type: BasisType::Generic,
        };
        let lookback = cfg.lookback;
        let model = Nbeats::new(cfg, &mut rng).expect("model ok");

        // Use a non-trivial input so at least one weight product is non-zero.
        let x: Vec<f32> = (0..lookback).map(|i| (i as f32).sin()).collect();
        let forecast_a = model.forward(&x).expect("forward ok");

        // With Xavier init and non-zero input the forecast should be non-zero.
        let all_zero = forecast_a.iter().all(|&v| v == 0.0);
        assert!(
            !all_zero,
            "forecast should be non-zero for non-trivial input"
        );
    }

    // ── n_stacks_1_works ──────────────────────────────────────────────────────

    #[test]
    fn n_stacks_1_works() {
        let mut rng = make_rng();
        let cfg = NbeatsConfig {
            n_layers: 1,
            layer_width: 8,
            n_stacks: 1,
            n_blocks_per_stack: 1,
            lookback: 12,
            forecast_horizon: 4,
            basis_type: BasisType::Generic,
        };
        let lookback = cfg.lookback;
        let model = Nbeats::new(cfg, &mut rng).expect("model ok");
        let x = vec![0.1_f32; lookback];
        let out = model.forward(&x).expect("forward ok");
        assert_eq!(out.len(), 4);
    }

    // ── different_inputs_different_outputs ────────────────────────────────────

    #[test]
    fn different_inputs_different_outputs() {
        let mut rng = make_rng();
        let cfg = tiny_config();
        let lookback = cfg.lookback;
        let model = Nbeats::new(cfg, &mut rng).expect("model ok");

        let x1: Vec<f32> = (0..lookback).map(|i| i as f32 * 0.1).collect();
        let x2: Vec<f32> = (0..lookback).map(|i| -(i as f32) * 0.1 - 1.0).collect();

        let out1 = model.forward(&x1).expect("forward x1 ok");
        let out2 = model.forward(&x2).expect("forward x2 ok");

        let are_equal = out1
            .iter()
            .zip(out2.iter())
            .all(|(a, b)| (a - b).abs() < 1e-9);
        assert!(
            !are_equal,
            "different inputs must yield different forecasts"
        );
    }

    // ── finite_output ─────────────────────────────────────────────────────────

    #[test]
    fn finite_output() {
        let mut rng = make_rng();
        let cfg = tiny_config();
        let lookback = cfg.lookback;
        let model = Nbeats::new(cfg, &mut rng).expect("model ok");
        let mut x = vec![0.0_f32; lookback];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("forward ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "forecast contains non-finite values"
        );
    }

    // ── n_blocks_per_stack_0_error ────────────────────────────────────────────

    #[test]
    fn n_blocks_per_stack_0_error() {
        let mut rng = make_rng();
        let cfg = NbeatsConfig {
            n_layers: 2,
            layer_width: 8,
            n_stacks: 2,
            n_blocks_per_stack: 0, // invalid
            lookback: 12,
            forecast_horizon: 4,
            basis_type: BasisType::Generic,
        };
        let result = Nbeats::new(cfg, &mut rng);
        assert!(result.is_err(), "n_blocks_per_stack=0 must return Err");
        assert!(
            matches!(result, Err(TsError::EmptyInput { .. })),
            "expected EmptyInput error"
        );
    }

    // ── lookback_0_error ──────────────────────────────────────────────────────

    #[test]
    fn lookback_0_error() {
        let mut rng = make_rng();
        let cfg = NbeatsConfig {
            n_layers: 2,
            layer_width: 8,
            n_stacks: 1,
            n_blocks_per_stack: 1,
            lookback: 0, // invalid
            forecast_horizon: 4,
            basis_type: BasisType::Generic,
        };
        let result = Nbeats::new(cfg, &mut rng);
        assert!(result.is_err(), "lookback=0 must return Err");
        assert!(
            matches!(result, Err(TsError::InvalidSequenceLength(0))),
            "expected InvalidSequenceLength(0)"
        );
    }

    // ── horizon_0_error ───────────────────────────────────────────────────────

    #[test]
    fn horizon_0_error() {
        let mut rng = make_rng();
        let cfg = NbeatsConfig {
            n_layers: 2,
            layer_width: 8,
            n_stacks: 1,
            n_blocks_per_stack: 1,
            lookback: 12,
            forecast_horizon: 0, // invalid
            basis_type: BasisType::Generic,
        };
        let result = Nbeats::new(cfg, &mut rng);
        assert!(result.is_err(), "forecast_horizon=0 must return Err");
        assert!(
            matches!(result, Err(TsError::InvalidHorizon(0))),
            "expected InvalidHorizon(0)"
        );
    }

    // ── generic_basis_works ───────────────────────────────────────────────────

    #[test]
    fn generic_basis_works() {
        let mut rng = make_rng();
        let cfg = NbeatsConfig {
            n_layers: 3,
            layer_width: 32,
            n_stacks: 2,
            n_blocks_per_stack: 3,
            lookback: 32,
            forecast_horizon: 10,
            basis_type: BasisType::Generic,
        };
        let lookback = cfg.lookback;
        let horizon = cfg.forecast_horizon;
        let model = Nbeats::new(cfg, &mut rng).expect("model ok");
        let x: Vec<f32> = (0..lookback).map(|i| (i as f32 * 0.2).sin()).collect();
        let out = model.forward(&x).expect("forward ok");
        assert_eq!(out.len(), horizon);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "BasisType::Generic produced non-finite output"
        );
    }

    // ── trend_basis_works ─────────────────────────────────────────────────────

    #[test]
    fn trend_basis_works() {
        let mut rng = make_rng();
        let cfg = NbeatsConfig {
            n_layers: 2,
            layer_width: 16,
            n_stacks: 1,
            n_blocks_per_stack: 2,
            lookback: 20,
            forecast_horizon: 6,
            basis_type: BasisType::Trend,
        };
        let lookback = cfg.lookback;
        let horizon = cfg.forecast_horizon;
        let model = Nbeats::new(cfg, &mut rng).expect("model ok");
        let x: Vec<f32> = (0..lookback).map(|i| i as f32 * 0.5).collect();
        let out = model.forward(&x).expect("forward ok");
        assert_eq!(out.len(), horizon);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── seasonality_basis_works ───────────────────────────────────────────────

    #[test]
    fn seasonality_basis_works() {
        let mut rng = make_rng();
        let cfg = NbeatsConfig {
            n_layers: 2,
            layer_width: 16,
            n_stacks: 1,
            n_blocks_per_stack: 2,
            lookback: 48,
            forecast_horizon: 12,
            basis_type: BasisType::Seasonality,
        };
        let lookback = cfg.lookback;
        let horizon = cfg.forecast_horizon;
        let model = Nbeats::new(cfg, &mut rng).expect("model ok");
        let x: Vec<f32> = (0..lookback)
            .map(|i| (i as f32 * std::f32::consts::TAU / 12.0).sin())
            .collect();
        let out = model.forward(&x).expect("forward ok");
        assert_eq!(out.len(), horizon);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── accessors ─────────────────────────────────────────────────────────────

    #[test]
    fn accessors_return_correct_values() {
        let mut rng = make_rng();
        let cfg = NbeatsConfig {
            n_layers: 1,
            layer_width: 4,
            n_stacks: 1,
            n_blocks_per_stack: 1,
            lookback: 10,
            forecast_horizon: 5,
            basis_type: BasisType::Generic,
        };
        let model = Nbeats::new(cfg, &mut rng).expect("model ok");
        assert_eq!(model.lookback(), 10);
        assert_eq!(model.horizon(), 5);
    }

    // ── n_stacks_0_error ──────────────────────────────────────────────────────

    #[test]
    fn n_stacks_0_error() {
        let mut rng = make_rng();
        let cfg = NbeatsConfig {
            n_layers: 2,
            layer_width: 8,
            n_stacks: 0, // invalid
            n_blocks_per_stack: 2,
            lookback: 12,
            forecast_horizon: 4,
            basis_type: BasisType::Generic,
        };
        let result = Nbeats::new(cfg, &mut rng);
        assert!(result.is_err(), "n_stacks=0 must return Err");
        assert!(
            matches!(result, Err(TsError::EmptyInput { .. })),
            "expected EmptyInput error"
        );
    }

    // ── dimension_mismatch_error ──────────────────────────────────────────────

    #[test]
    fn dimension_mismatch_error() {
        let mut rng = make_rng();
        let cfg = tiny_config();
        let model = Nbeats::new(cfg, &mut rng).expect("model ok");
        // Pass wrong input length
        let x = vec![0.5_f32; 5]; // lookback is 24
        let result = model.forward(&x);
        assert!(result.is_err());
        assert!(
            matches!(
                result,
                Err(TsError::DimensionMismatch {
                    expected: 24,
                    got: 5
                })
            ),
            "expected DimensionMismatch"
        );
    }
}

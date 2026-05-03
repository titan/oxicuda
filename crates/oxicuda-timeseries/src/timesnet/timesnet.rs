//! TimesNet: full model with input projection, stacked TimesBlocks, and
//! prediction head.
//!
//! Architecture:
//!   1. Input projection: `[T, C] → [T, d_model]` (linear, per-timestep).
//!   2. N TimesBlocks: each `[T, d_model] → [T, d_model]`.
//!   3. Flatten `[T * d_model]` → linear → `[horizon * C]`.
//!   4. Reshape `[horizon, C]`.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;
use crate::timesnet::times_block::TimesBlock;

// ─── TimesNetConfig ──────────────────────────────────────────────────────────

/// Configuration for a `TimesNet` model.
#[derive(Debug, Clone)]
pub struct TimesNetConfig {
    /// Number of input variates (channels).
    pub c: usize,
    /// Input sequence length.
    pub t: usize,
    /// Forecast horizon (output timesteps).
    pub horizon: usize,
    /// Internal projection dimensionality.
    pub d_model: usize,
    /// Number of stacked `TimesBlock`s.
    pub n_layers: usize,
    /// Top-k periods per block.
    pub top_k: usize,
}

impl TimesNetConfig {
    /// Tiny configuration: `d_model=32, n_layers=2, top_k=3`.
    ///
    /// Suitable for fast unit tests.
    #[must_use]
    pub fn tiny(c: usize, t: usize, horizon: usize) -> Self {
        Self {
            c,
            t,
            horizon,
            d_model: 32,
            n_layers: 2,
            top_k: 3,
        }
    }

    /// Default configuration: `d_model=64, n_layers=2, top_k=5`.
    #[must_use]
    pub fn default_config(c: usize, t: usize, horizon: usize) -> Self {
        Self {
            c,
            t,
            horizon,
            d_model: 64,
            n_layers: 2,
            top_k: 5,
        }
    }
}

// ─── TimesNet ────────────────────────────────────────────────────────────────

/// Full TimesNet model for multivariate time-series forecasting.
///
/// Operates on time-major `[T, C]` tensors and produces `[horizon, C]` forecasts.
#[derive(Debug, Clone)]
pub struct TimesNet {
    /// Input projection weight `[d_model, C]`.
    pub input_proj_w: Vec<f32>,
    /// Input projection bias `[d_model]`.
    pub input_proj_b: Vec<f32>,
    /// Stack of `TimesBlock`s.
    pub blocks: Vec<TimesBlock>,
    /// Unused per-step head weight `[horizon, d_model]` (kept for API completeness).
    pub head_w: Vec<f32>,
    /// Unused per-step head bias `[horizon]`.
    pub head_b: Vec<f32>,
    /// Final linear weight `[horizon * C, T * d_model]`.
    pub final_w: Vec<f32>,
    /// Final linear bias `[horizon * C]`.
    pub final_b: Vec<f32>,
    /// Configuration used to build this model.
    pub config: TimesNetConfig,
}

impl TimesNet {
    /// Construct a `TimesNet` model.
    ///
    /// All weights are initialised with Xavier / Kaiming He scaling.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidNumVariates`]`(0)` when `c == 0`.
    /// - [`TsError::InvalidHorizon`]`(0)` when `horizon == 0`.
    /// - [`TsError::InvalidSequenceLength`]`(t)` when `t < 2`.
    pub fn new(config: TimesNetConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.c == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        if config.horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if config.t < 2 {
            return Err(TsError::InvalidSequenceLength(config.t));
        }

        let c = config.c;
        let d = config.d_model;
        let t = config.t;
        let h = config.horizon;

        // ── Input projection [d_model, C]: Xavier std = sqrt(2 / (C + d_model))
        let proj_std = (2.0_f32 / (c + d) as f32).sqrt();
        let mut input_proj_w = vec![0.0_f32; d * c];
        rng.fill_normal(&mut input_proj_w);
        for v in &mut input_proj_w {
            *v *= proj_std;
        }
        let input_proj_b = vec![0.0_f32; d];

        // ── TimesBlocks (operate on d_model channels)
        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(TimesBlock::new(d, config.top_k, rng)?);
        }

        // ── Per-step head [horizon, d_model] (stored but superseded by final_w)
        let head_std = (2.0_f32 / (d + h) as f32).sqrt();
        let mut head_w = vec![0.0_f32; h * d];
        rng.fill_normal(&mut head_w);
        for v in &mut head_w {
            *v *= head_std;
        }
        let head_b = vec![0.0_f32; h];

        // ── Final linear [horizon*C, T*d_model]: Kaiming He std = sqrt(2 / (T*d))
        let final_in = t * d;
        let final_out = h * c;
        let final_std = (2.0_f32 / final_in as f32).sqrt();
        let mut final_w = vec![0.0_f32; final_out * final_in];
        rng.fill_normal(&mut final_w);
        for v in &mut final_w {
            *v *= final_std;
        }
        let final_b = vec![0.0_f32; final_out];

        Ok(Self {
            input_proj_w,
            input_proj_b,
            blocks,
            head_w,
            head_b,
            final_w,
            final_b,
            config,
        })
    }

    /// Forward pass: `[T, C] → [horizon, C]`.
    ///
    /// # Pipeline
    ///
    /// 1. Project input `[T, C] → [T, d_model]`.
    /// 2. Run through `n_layers` `TimesBlock`s.
    /// 3. Flatten to `[T * d_model]`, apply `final_w` / `final_b` → `[horizon * C]`.
    /// 4. Reshape to `[horizon, C]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != T * C`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let c = self.config.c;
        let t = self.config.t;
        let d = self.config.d_model;
        let h = self.config.horizon;

        let expected = t * c;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // ── 1. Input projection: [T, C] → [T, d_model] ───────────────────────
        // proj_out[ti, di] = Σ_ci x[ti, ci] * w[di, ci] + b[di]
        let mut proj = vec![0.0_f32; t * d];
        for ti in 0..t {
            for di in 0..d {
                let mut acc = self.input_proj_b[di];
                for ci in 0..c {
                    acc += x[ti * c + ci] * self.input_proj_w[di * c + ci];
                }
                proj[ti * d + di] = acc;
            }
        }

        // ── 2. TimesBlocks ────────────────────────────────────────────────────
        let mut h_state = proj;
        for block in &self.blocks {
            h_state = block.forward(&h_state, t)?;
        }

        // ── 3. Flatten and final linear: [T*d_model] → [horizon*C] ───────────
        // final_out[oi] = Σ_j flat[j] * final_w[oi, j] + final_b[oi]
        let final_in = t * d;
        let final_out_len = h * c;
        let mut pred = vec![0.0_f32; final_out_len];
        for (oi, pv) in pred.iter_mut().enumerate() {
            let mut acc = self.final_b[oi];
            let w_row = &self.final_w[oi * final_in..(oi + 1) * final_in];
            for (&flat_j, &wj) in h_state.iter().zip(w_row.iter()) {
                acc += flat_j * wj;
            }
            *pv = acc;
        }

        // ── 4. Output is already [horizon * C] ───────────────────────────────
        Ok(pred)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(77)
    }

    #[test]
    fn timesnet_new_ok() {
        let mut rng = make_rng();
        let cfg = TimesNetConfig::tiny(4, 24, 8);
        let model = TimesNet::new(cfg, &mut rng).expect("should construct");
        assert_eq!(model.config.c, 4);
        assert_eq!(model.config.t, 24);
        assert_eq!(model.config.horizon, 8);
    }

    #[test]
    fn timesnet_forward_shape() {
        let mut rng = make_rng();
        let cfg = TimesNetConfig::tiny(4, 24, 8);
        let model = TimesNet::new(cfg, &mut rng).expect("ok");
        let x = vec![0.1_f32; 24 * 4];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), 8 * 4, "output must be [horizon * C]");
    }

    #[test]
    fn timesnet_forward_finite() {
        let mut rng = make_rng();
        let cfg = TimesNetConfig::tiny(4, 16, 4);
        let model = TimesNet::new(cfg, &mut rng).expect("ok");
        let mut x = vec![0.0_f32; 16 * 4];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output contains non-finite values"
        );
    }

    #[test]
    fn timesnet_zero_c_error() {
        let mut rng = make_rng();
        let cfg = TimesNetConfig::tiny(0, 16, 4);
        assert!(matches!(
            TimesNet::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidNumVariates(0)
        ));
    }

    #[test]
    fn timesnet_zero_horizon_error() {
        let mut rng = make_rng();
        let cfg = TimesNetConfig::tiny(4, 16, 0);
        assert!(matches!(
            TimesNet::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidHorizon(0)
        ));
    }

    #[test]
    fn timesnet_short_seq_error() {
        let mut rng = make_rng();
        let cfg = TimesNetConfig::tiny(4, 1, 4);
        assert!(matches!(
            TimesNet::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidSequenceLength(1)
        ));
    }

    #[test]
    fn timesnet_forward_dim_mismatch() {
        let mut rng = make_rng();
        let cfg = TimesNetConfig::tiny(4, 16, 4);
        let model = TimesNet::new(cfg, &mut rng).expect("ok");
        // Provide wrong input size
        let x = vec![0.0_f32; 10]; // expected 16*4=64
        assert!(matches!(
            model.forward(&x).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn timesnet_default_config_output_shape() {
        let mut rng = make_rng();
        let cfg = TimesNetConfig::default_config(7, 24, 12);
        let model = TimesNet::new(cfg, &mut rng).expect("ok");
        let x = vec![0.05_f32; 24 * 7];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), 12 * 7);
    }
}

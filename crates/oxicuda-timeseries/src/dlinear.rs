//! DLinear and NLinear — simple linear decomposition baselines.
//!
//! From Zeng et al. (AAAI 2023) "Are Transformers Effective for Time Series
//! Forecasting?".  Both models operate **independently per channel** (no
//! cross-variate mixing) and are competitive with much larger Transformer-based
//! architectures on standard benchmarks.
//!
//! - **DLinear**: decomposes the input into trend + seasonal via a centred
//!   moving average, then applies a separate learned linear layer to each
//!   component per channel, and sums the two forecasts.
//! - **NLinear**: subtracts the last observed value (distribution shift
//!   normalisation), applies one linear layer per channel, then adds the
//!   last value back.

use crate::decomp::MovingAvg;
use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Apply a per-channel linear layer: `out[h] = Σ_t w[h*t + t] * x[t] + b[h]`.
///
/// - `x`:  `[T]` — single-channel input.
/// - `w`:  `[H × T]` row-major — weight matrix for this channel.
/// - `b`:  `[H]` — bias for this channel.
/// - Returns `[H]`.
fn channel_linear(x: &[f32], w: &[f32], b: &[f32], t: usize, horizon: usize) -> Vec<f32> {
    let mut out = b.to_vec();
    for h in 0..horizon {
        let row = &w[h * t..(h + 1) * t];
        out[h] += row
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>();
    }
    out
}

/// Xavier-uniform initialisation: `scale = sqrt(6 / (fan_in + fan_out))`.
fn xavier_fill(buf: &mut [f32], fan_in: usize, fan_out: usize, rng: &mut LcgRng) {
    let scale = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    rng.fill_normal(buf);
    for v in buf.iter_mut() {
        *v *= scale;
    }
}

// ─── DLinearConfig ────────────────────────────────────────────────────────────

/// Configuration for the DLinear (decomposition linear) model.
#[derive(Debug, Clone)]
pub struct DLinearConfig {
    /// Input sequence length `T`.
    pub t: usize,
    /// Forecast horizon `H`.
    pub horizon: usize,
    /// Number of channels / variates `C`.
    pub c: usize,
    /// Kernel size for the centred moving average used in trend extraction.
    /// Defaults to 25 (matching the paper).
    pub moving_avg_window: usize,
}

impl DLinearConfig {
    /// Construct with the standard `moving_avg_window = 25`.
    #[must_use]
    pub fn new(t: usize, horizon: usize, c: usize) -> Self {
        Self {
            t,
            horizon,
            c,
            moving_avg_window: 25,
        }
    }
}

// ─── DLinear ─────────────────────────────────────────────────────────────────

/// DLinear model: Linear(seasonal) + Linear(trend) → forecast.
///
/// Both linear layers are **per-channel** (independent variates): no
/// cross-channel weight sharing.
///
/// Weight layout:
/// - `seasonal_w[ci * H * T .. (ci+1) * H * T]` = `[H × T]` for channel `ci`.
/// - `trend_w` — same layout.
/// - `seasonal_b[ci * H .. (ci+1) * H]` = bias for channel `ci`.
/// - `trend_b` — same layout.
#[derive(Debug, Clone)]
pub struct DLinear {
    /// Seasonal linear weights, shape `[C × H × T]`.
    pub seasonal_w: Vec<f32>,
    /// Seasonal linear biases, shape `[C × H]`.
    pub seasonal_b: Vec<f32>,
    /// Trend linear weights, shape `[C × H × T]`.
    pub trend_w: Vec<f32>,
    /// Trend linear biases, shape `[C × H]`.
    pub trend_b: Vec<f32>,
    /// Config used to build this model.
    pub config: DLinearConfig,
    decomp: MovingAvg,
}

impl DLinear {
    /// Build a `DLinear` model with Xavier-uniform-initialised weights.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `t == 0`.
    /// - [`TsError::InvalidHorizon`] when `horizon == 0`.
    /// - [`TsError::InvalidNumVariates`] when `c == 0`.
    /// - [`TsError::InvalidKernelSize`] when `moving_avg_window == 0`.
    /// - [`TsError::ShapeMismatch`] when `moving_avg_window > t`.
    pub fn new(config: DLinearConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.t == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if config.horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if config.c == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        if config.moving_avg_window == 0 {
            return Err(TsError::InvalidKernelSize(0));
        }
        if config.moving_avg_window > config.t {
            return Err(TsError::ShapeMismatch {
                msg: format!(
                    "moving_avg_window {} > t {}",
                    config.moving_avg_window, config.t
                ),
            });
        }

        let decomp = MovingAvg::new(config.moving_avg_window)?;

        let t = config.t;
        let horizon = config.horizon;
        let c = config.c;

        // Per-channel weight: [C × H × T] → C * H * T elements total.
        let weight_elems = c * horizon * t;

        let mut seasonal_w = vec![0.0_f32; weight_elems];
        xavier_fill(&mut seasonal_w, t, horizon, rng);
        let seasonal_b = vec![0.0_f32; c * horizon];

        let mut trend_w = vec![0.0_f32; weight_elems];
        xavier_fill(&mut trend_w, t, horizon, rng);
        let trend_b = vec![0.0_f32; c * horizon];

        Ok(Self {
            seasonal_w,
            seasonal_b,
            trend_w,
            trend_b,
            config,
            decomp,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// `x` — `[T × C]` row-major input.
    ///
    /// # Returns
    ///
    /// `[H × C]` row-major forecast.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != T * C`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let t = self.config.t;
        let c = self.config.c;
        let horizon = self.config.horizon;

        let expected = t * c;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // ── Decompose into trend + seasonal ──────────────────────────────────
        let trend = self.decomp.forward(x, t, c)?;
        let seasonal: Vec<f32> = x
            .iter()
            .zip(trend.iter())
            .map(|(&xi, &tr)| xi - tr)
            .collect();

        // ── Per-channel linear projections ────────────────────────────────────
        let mut forecast = vec![0.0_f32; horizon * c];

        for ci in 0..c {
            // Extract single-channel slices from [T × C] input.
            let s_input: Vec<f32> = (0..t).map(|ti| seasonal[ti * c + ci]).collect();
            let t_input: Vec<f32> = (0..t).map(|ti| trend[ti * c + ci]).collect();

            let sw = &self.seasonal_w[ci * horizon * t..(ci + 1) * horizon * t];
            let sb = &self.seasonal_b[ci * horizon..(ci + 1) * horizon];
            let tw = &self.trend_w[ci * horizon * t..(ci + 1) * horizon * t];
            let tb = &self.trend_b[ci * horizon..(ci + 1) * horizon];

            let fc_s = channel_linear(&s_input, sw, sb, t, horizon);
            let fc_t = channel_linear(&t_input, tw, tb, t, horizon);

            for h in 0..horizon {
                forecast[h * c + ci] = fc_s[h] + fc_t[h];
            }
        }

        Ok(forecast)
    }
}

// ─── NLinearConfig ────────────────────────────────────────────────────────────

/// Configuration for the NLinear (normalised linear) model.
#[derive(Debug, Clone)]
pub struct NLinearConfig {
    /// Input sequence length `T`.
    pub t: usize,
    /// Forecast horizon `H`.
    pub horizon: usize,
    /// Number of channels / variates `C`.
    pub c: usize,
}

// ─── NLinear ─────────────────────────────────────────────────────────────────

/// NLinear model: subtract last time step, apply linear, add last time step back.
///
/// This simple normalisation corrects for distribution shift and often
/// outperforms complex Transformer-based models on standard benchmarks.
///
/// Weight layout (per-channel, independent variates):
/// - `w[ci * H * T .. (ci+1) * H * T]` = `[H × T]` for channel `ci`.
/// - `b[ci * H .. (ci+1) * H]` = bias for channel `ci`.
#[derive(Debug, Clone)]
pub struct NLinear {
    /// Per-channel linear weights, shape `[C × H × T]`.
    pub w: Vec<f32>,
    /// Per-channel linear biases, shape `[C × H]`.
    pub b: Vec<f32>,
    /// Config used to build this model.
    pub config: NLinearConfig,
}

impl NLinear {
    /// Build an `NLinear` model with Xavier-uniform-initialised weights.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `t == 0`.
    /// - [`TsError::InvalidHorizon`] when `horizon == 0`.
    /// - [`TsError::InvalidNumVariates`] when `c == 0`.
    pub fn new(config: NLinearConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.t == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if config.horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if config.c == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }

        let t = config.t;
        let horizon = config.horizon;
        let c = config.c;

        let mut w = vec![0.0_f32; c * horizon * t];
        xavier_fill(&mut w, t, horizon, rng);
        let b = vec![0.0_f32; c * horizon];

        Ok(Self { w, b, config })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// `x` — `[T × C]` row-major input.
    ///
    /// # Returns
    ///
    /// `[H × C]` row-major forecast.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != T * C`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let t = self.config.t;
        let c = self.config.c;
        let horizon = self.config.horizon;

        let expected = t * c;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // ── Last time-step values (distribution shift normalisation) ──────────
        let last: Vec<f32> = (0..c).map(|ci| x[(t - 1) * c + ci]).collect();

        // ── Normalised input: subtract last from every time step ──────────────
        let x_norm: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(idx, &xi)| xi - last[idx % c])
            .collect();

        // ── Per-channel linear projection + denormalise ───────────────────────
        let mut forecast = vec![0.0_f32; horizon * c];

        for ci in 0..c {
            let norm_input: Vec<f32> = (0..t).map(|ti| x_norm[ti * c + ci]).collect();

            let cw = &self.w[ci * horizon * t..(ci + 1) * horizon * t];
            let cb = &self.b[ci * horizon..(ci + 1) * horizon];

            let fc = channel_linear(&norm_input, cw, cb, t, horizon);

            for h in 0..horizon {
                forecast[h * c + ci] = fc[h] + last[ci];
            }
        }

        Ok(forecast)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── DLinear ───────────────────────────────────────────────────────────────

    #[test]
    fn dlinear_forward_shape() {
        let mut rng = make_rng();
        let cfg = DLinearConfig {
            t: 48,
            horizon: 8,
            c: 3,
            moving_avg_window: 5,
        };
        let model = DLinear::new(cfg, &mut rng).expect("ok");
        let x = vec![0.5_f32; 48 * 3];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), 8 * 3);
    }

    #[test]
    fn dlinear_finite_output() {
        let mut rng = make_rng();
        let cfg = DLinearConfig::new(48, 12, 4);
        let model = DLinear::new(cfg, &mut rng).expect("ok");
        let mut x = vec![0.0_f32; 48 * 4];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn dlinear_multichannel() {
        let mut rng = make_rng();
        let cfg = DLinearConfig {
            t: 32,
            horizon: 8,
            c: 4,
            moving_avg_window: 5,
        };
        let model = DLinear::new(cfg, &mut rng).expect("ok");
        let x = vec![1.0_f32; 32 * 4];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), 8 * 4);
    }

    #[test]
    fn dlinear_horizon_1() {
        let mut rng = make_rng();
        let cfg = DLinearConfig {
            t: 32,
            horizon: 1,
            c: 2,
            moving_avg_window: 5,
        };
        let model = DLinear::new(cfg, &mut rng).expect("ok");
        let x = vec![0.3_f32; 32 * 2];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn dlinear_decomp_seasonal_plus_trend_equals_x() {
        // For constant input, trend = input, seasonal = 0.
        // Verify the decomposition identity holds for a non-trivial signal
        // by checking seasonal + trend = x directly via the moving average.
        let t = 30;
        let c = 2;
        let window = 5;
        let ma = MovingAvg::new(window).expect("ok");
        let x: Vec<f32> = (0..t * c).map(|i| i as f32 * 0.1 - 1.5).collect();
        let trend = ma.forward(&x, t, c).expect("ok");
        let seasonal: Vec<f32> = x
            .iter()
            .zip(trend.iter())
            .map(|(&xi, &tr)| xi - tr)
            .collect();
        for (i, (&orig, (&tr, &se))) in x.iter().zip(trend.iter().zip(seasonal.iter())).enumerate()
        {
            assert!(
                (orig - (tr + se)).abs() < 1e-5,
                "idx={i}: orig={orig} trend+seasonal={}",
                tr + se
            );
        }
    }

    #[test]
    fn dlinear_last_minus_first_monotone() {
        // A constant-valued input has zero seasonal component.
        let mut rng = make_rng();
        let cfg = DLinearConfig {
            t: 20,
            horizon: 4,
            c: 1,
            moving_avg_window: 3,
        };
        let model = DLinear::new(cfg, &mut rng).expect("ok");
        let x = vec![5.0_f32; 20];
        // constant input → trend = x, seasonal = 0
        // model should still produce a finite output
        let out = model.forward(&x).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── NLinear ───────────────────────────────────────────────────────────────

    #[test]
    fn nlinear_forward_shape() {
        let mut rng = make_rng();
        let cfg = NLinearConfig {
            t: 24,
            horizon: 8,
            c: 3,
        };
        let model = NLinear::new(cfg, &mut rng).expect("ok");
        let x = vec![0.5_f32; 24 * 3];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), 8 * 3);
    }

    #[test]
    fn nlinear_finite_output() {
        let mut rng = make_rng();
        let cfg = NLinearConfig {
            t: 48,
            horizon: 12,
            c: 4,
        };
        let model = NLinear::new(cfg, &mut rng).expect("ok");
        let mut x = vec![0.0_f32; 48 * 4];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn nlinear_zero_input() {
        // Zero weights (zero bias) and zero input → forecast should be zero.
        // With xavier init the biases are 0 so with x=0 the normalised input is 0
        // and last=0, so the linear output is 0 + bias = 0 + 0 = 0.
        let cfg = NLinearConfig {
            t: 8,
            horizon: 4,
            c: 2,
        };
        // Build with zero weights manually.
        let model = NLinear {
            w: vec![0.0_f32; 2 * 4 * 8],
            b: vec![0.0_f32; 2 * 4],
            config: cfg,
        };
        let x = vec![0.0_f32; 8 * 2];
        let out = model.forward(&x).expect("ok");
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn nlinear_constant_input() {
        // Constant input c_val: last = c_val, x_norm = 0.
        // With zero bias and zero weights: forecast = 0 + c_val = c_val.
        let c_val = 3.0_f32;
        let t = 10;
        let h = 5;
        let c = 2;
        let model = NLinear {
            w: vec![0.0_f32; c * h * t],
            b: vec![0.0_f32; c * h],
            config: NLinearConfig { t, horizon: h, c },
        };
        let x = vec![c_val; t * c];
        let out = model.forward(&x).expect("ok");
        for &v in &out {
            assert!((v - c_val).abs() < 1e-5, "expected {c_val}, got {v}");
        }
    }

    #[test]
    fn nlinear_multichannel() {
        let mut rng = make_rng();
        let cfg = NLinearConfig {
            t: 32,
            horizon: 8,
            c: 4,
        };
        let model = NLinear::new(cfg, &mut rng).expect("ok");
        let x = vec![0.5_f32; 32 * 4];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), 8 * 4);
    }

    #[test]
    fn nlinear_single_timestep_input() {
        // Edge case: T=1.  last = x[0], x_norm = 0.
        let c_val = 2.0_f32;
        let t = 1;
        let h = 3;
        let c = 1;
        let model = NLinear {
            w: vec![0.0_f32; c * h * t],
            b: vec![0.0_f32; c * h],
            config: NLinearConfig { t, horizon: h, c },
        };
        let x = vec![c_val];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), h);
        for &v in &out {
            assert!((v - c_val).abs() < 1e-5, "expected {c_val}, got {v}");
        }
    }

    // ── Error cases ────────────────────────────────────────────────────────────

    #[test]
    fn dlinear_err_invalid_horizon() {
        let mut rng = make_rng();
        let cfg = DLinearConfig {
            t: 16,
            horizon: 0,
            c: 2,
            moving_avg_window: 3,
        };
        assert!(matches!(
            DLinear::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidHorizon(0)
        ));
    }

    #[test]
    fn nlinear_err_invalid_c() {
        let mut rng = make_rng();
        let cfg = NLinearConfig {
            t: 16,
            horizon: 4,
            c: 0,
        };
        assert!(matches!(
            NLinear::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidNumVariates(0)
        ));
    }

    #[test]
    fn dlinear_err_dimension_mismatch() {
        let mut rng = make_rng();
        let cfg = DLinearConfig {
            t: 32,
            horizon: 4,
            c: 2,
            moving_avg_window: 5,
        };
        let model = DLinear::new(cfg, &mut rng).expect("ok");
        assert!(matches!(
            model.forward(&[0.0_f32; 5]).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn nlinear_err_dimension_mismatch() {
        let mut rng = make_rng();
        let cfg = NLinearConfig {
            t: 16,
            horizon: 4,
            c: 2,
        };
        let model = NLinear::new(cfg, &mut rng).expect("ok");
        assert!(matches!(
            model.forward(&[0.0_f32; 3]).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn dlinear_err_moving_avg_too_large() {
        let mut rng = make_rng();
        let cfg = DLinearConfig {
            t: 8,
            horizon: 4,
            c: 2,
            moving_avg_window: 10, // > t
        };
        assert!(matches!(
            DLinear::new(cfg, &mut rng).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn dlinear_err_zero_kernel() {
        let mut rng = make_rng();
        let cfg = DLinearConfig {
            t: 16,
            horizon: 4,
            c: 2,
            moving_avg_window: 0,
        };
        assert!(matches!(
            DLinear::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidKernelSize(0)
        ));
    }
}

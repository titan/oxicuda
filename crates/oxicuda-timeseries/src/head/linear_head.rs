//! Linear forecasting head: `[in_features] → [out_features]`.
//!
//! A single weight matrix and bias that maps a flattened encoder output
//! directly to a multi-step forecast.  Weights are initialised with
//! Xavier-uniform scaling.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

/// Single linear projection head.
///
/// Weight layout: `[out_features, in_features]` row-major.
#[derive(Debug, Clone)]
pub struct LinearHead {
    /// Weight matrix `[out_features, in_features]`.
    pub weight: Vec<f32>,
    /// Bias vector `[out_features]`.
    pub bias: Vec<f32>,
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension (equals the forecast horizon when used
    /// per-variate).
    pub out_features: usize,
}

impl LinearHead {
    /// Construct with Xavier-uniform initialisation.
    ///
    /// The gain is `sqrt(6 / (in_features + out_features))`.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidEmbedDim`]`(0)` when `in_features == 0`.
    /// - [`TsError::InvalidHorizon`]`(0)` when `out_features == 0`.
    pub fn new(in_features: usize, out_features: usize, rng: &mut LcgRng) -> TsResult<Self> {
        if in_features == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if out_features == 0 {
            return Err(TsError::InvalidHorizon(0));
        }

        let n_weights = out_features * in_features;
        let scale = (6.0_f32 / (in_features + out_features) as f32).sqrt();

        // Sample U(-1, 1) via Box-Muller normals then re-scale to
        // uniform-like spread.  We draw U[0,1) pairs and fold them into
        // U(-scale, +scale).
        let mut weight = vec![0.0_f32; n_weights];
        // Fill with normal samples, then clip to Xavier range.
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            // Clamp normal draw to ~uniform [-scale, scale] via tanh-like
            // stretching: sign preserved, magnitude capped.
            let clamped = w.clamp(-scale * 3.0, scale * 3.0);
            *w = clamped * (scale / (scale * 3.0));
        }
        let bias = vec![0.0_f32; out_features];

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Apply the linear projection.
    ///
    /// # Arguments
    ///
    /// * `input` — row-major `[N, in_features]` flat slice.
    ///
    /// Returns `[N, out_features]` flat row-major.
    ///
    /// # Errors
    ///
    /// - [`TsError::EmptyInput`] when `input` is empty.
    /// - [`TsError::DimensionMismatch`] when `input.len() % in_features != 0`.
    pub fn forward(&self, input: &[f32]) -> TsResult<Vec<f32>> {
        if input.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "linear_head forward: input is empty".into(),
            });
        }
        if input.len() % self.in_features != 0 {
            return Err(TsError::DimensionMismatch {
                expected: self.in_features,
                got: input.len(),
            });
        }

        let n = input.len() / self.in_features;
        let mut out = vec![0.0_f32; n * self.out_features];

        for row in 0..n {
            let x = &input[row * self.in_features..(row + 1) * self.in_features];
            for o in 0..self.out_features {
                let w_row = &self.weight[o * self.in_features..(o + 1) * self.in_features];
                let dot: f32 = w_row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
                out[row * self.out_features + o] = dot + self.bias[o];
            }
        }
        Ok(out)
    }

    /// Per-variate time-series forward: `[T, C] → [horizon, C]`.
    ///
    /// For each variate `c` the time dimension `[0..T, c]` is treated as an
    /// `in_features`-dimensional vector and projected to `out_features`
    /// (the forecast horizon).  The result is assembled as `[horizon, C]`
    /// row-major.
    ///
    /// This variant requires `in_features == t` and `out_features == horizon`.
    ///
    /// # Arguments
    ///
    /// * `input` — `[T, C]` row-major flat slice.
    /// * `t`     — sequence length.
    /// * `c`     — number of variates.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `input.len() != t * c`.
    /// - [`TsError::WeightShapeMismatch`] when `in_features != t`.
    pub fn forward_ts(&self, input: &[f32], t: usize, c: usize) -> TsResult<Vec<f32>> {
        if input.len() != t * c {
            return Err(TsError::DimensionMismatch {
                expected: t * c,
                got: input.len(),
            });
        }
        if self.in_features != t {
            return Err(TsError::WeightShapeMismatch {
                msg: format!("forward_ts: in_features={} but t={}", self.in_features, t),
            });
        }

        let horizon = self.out_features;
        let mut out = vec![0.0_f32; horizon * c];

        for ci in 0..c {
            // Extract the time-slice for variate ci: [T] column-wise from [T, C].
            let series: Vec<f32> = (0..t).map(|ti| input[ti * c + ci]).collect();

            // Project [T] → [horizon] using a single-row forward call.
            let proj = self.forward(&series)?;

            // Write into [horizon, C] layout.
            for hi in 0..horizon {
                out[hi * c + ci] = proj[hi];
            }
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // 1. Output shape for single-sample forward.
    #[test]
    fn linear_head_forward_shape() {
        let mut rng = make_rng();
        let head = LinearHead::new(64, 24, &mut rng).expect("ok");
        let input = vec![0.1_f32; 64];
        let out = head.forward(&input).expect("ok");
        assert_eq!(out.len(), 24);
    }

    // 2. All outputs are finite.
    #[test]
    fn linear_head_forward_finite() {
        let mut rng = make_rng();
        let head = LinearHead::new(32, 16, &mut rng).expect("ok");
        let mut input = vec![0.0_f32; 32];
        rng.fill_normal(&mut input);
        let out = head.forward(&input).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    // 3. Zero in_features → InvalidEmbedDim(0).
    #[test]
    fn linear_head_zero_in_error() {
        let mut rng = make_rng();
        assert!(matches!(
            LinearHead::new(0, 8, &mut rng).unwrap_err(),
            TsError::InvalidEmbedDim(0)
        ));
    }

    // 4. Zero out_features → InvalidHorizon(0).
    #[test]
    fn linear_head_zero_out_error() {
        let mut rng = make_rng();
        assert!(matches!(
            LinearHead::new(16, 0, &mut rng).unwrap_err(),
            TsError::InvalidHorizon(0)
        ));
    }

    // 5. Batch forward: N=4 samples.
    #[test]
    fn linear_head_batch_shape() {
        let mut rng = make_rng();
        let head = LinearHead::new(8, 4, &mut rng).expect("ok");
        // N=4 samples, each of length 8.
        let input = vec![0.5_f32; 4 * 8];
        let out = head.forward(&input).expect("ok");
        assert_eq!(out.len(), 4 * 4);
    }

    // 6. forward_ts shape: [T, C] → [horizon, C].
    #[test]
    fn linear_head_ts_shape() {
        let t = 48;
        let c = 7;
        let horizon = 24;
        let mut rng = make_rng();
        let head = LinearHead::new(t, horizon, &mut rng).expect("ok");
        let input = vec![0.1_f32; t * c];
        let out = head.forward_ts(&input, t, c).expect("ok");
        assert_eq!(out.len(), horizon * c);
    }

    // 7. forward_ts weight mismatch: in_features != t.
    #[test]
    fn linear_head_ts_weight_mismatch() {
        let mut rng = make_rng();
        // Head built for in_features=32 but we pass t=48.
        let head = LinearHead::new(32, 24, &mut rng).expect("ok");
        let input = vec![0.0_f32; 48 * 7];
        assert!(matches!(
            head.forward_ts(&input, 48, 7).unwrap_err(),
            TsError::WeightShapeMismatch { .. }
        ));
    }

    // 8. Empty input → EmptyInput error.
    #[test]
    fn linear_head_empty_input_error() {
        let mut rng = make_rng();
        let head = LinearHead::new(8, 4, &mut rng).expect("ok");
        assert!(matches!(
            head.forward(&[]).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }
}

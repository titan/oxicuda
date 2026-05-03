//! Two-layer MLP forecasting head: `in → hidden (ReLU) → out`.
//!
//! The first layer uses Kaiming He initialisation (appropriate for the
//! pre-ReLU activation), and the second layer uses Xavier-uniform
//! initialisation.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Activation ──────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// ─── MlpHead ─────────────────────────────────────────────────────────────────

/// Two-layer MLP head with a single hidden layer gated by ReLU.
///
/// Architecture: `in → hidden (ReLU) → out`.
///
/// Weight layouts:
/// - `w1`: `[hidden, in_features]` row-major (Kaiming He init).
/// - `w2`: `[out_features, hidden]` row-major (Xavier-uniform init).
#[derive(Debug, Clone)]
pub struct MlpHead {
    /// First layer weight `[hidden, in_features]`.
    pub w1: Vec<f32>,
    /// First layer bias `[hidden]`.
    pub b1: Vec<f32>,
    /// Second layer weight `[out_features, hidden]`.
    pub w2: Vec<f32>,
    /// Second layer bias `[out_features]`.
    pub b2: Vec<f32>,
    /// Input feature dimension.
    pub in_features: usize,
    /// Hidden layer width.
    pub hidden: usize,
    /// Output feature dimension (forecast horizon when used per-variate).
    pub out_features: usize,
}

impl MlpHead {
    /// Construct a `MlpHead` with Kaiming He init for `w1` and Xavier-uniform
    /// for `w2`.
    ///
    /// Kaiming He scale for a ReLU layer: `sqrt(2 / in_features)`.
    /// Xavier-uniform scale: `sqrt(6 / (hidden + out_features))`.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidEmbedDim`]`(0)` when `in_features == 0` or
    ///   `hidden == 0`.
    /// - [`TsError::InvalidHorizon`]`(0)` when `out_features == 0`.
    pub fn new(
        in_features: usize,
        hidden: usize,
        out_features: usize,
        rng: &mut LcgRng,
    ) -> TsResult<Self> {
        if in_features == 0 || hidden == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if out_features == 0 {
            return Err(TsError::InvalidHorizon(0));
        }

        // Kaiming He: scale = sqrt(2 / fan_in) where fan_in = in_features.
        let kaiming_scale = (2.0_f32 / in_features as f32).sqrt();
        let mut w1 = vec![0.0_f32; hidden * in_features];
        rng.fill_normal(&mut w1);
        for w in &mut w1 {
            *w *= kaiming_scale;
        }
        let b1 = vec![0.0_f32; hidden];

        // Xavier-uniform: scale = sqrt(6 / (hidden + out_features)).
        let xavier_scale = (6.0_f32 / (hidden + out_features) as f32).sqrt();
        let mut w2 = vec![0.0_f32; out_features * hidden];
        rng.fill_normal(&mut w2);
        for w in &mut w2 {
            // Map normal draw to ~uniform [-xavier_scale, xavier_scale].
            let clamped = w.clamp(-xavier_scale * 3.0, xavier_scale * 3.0);
            *w = clamped * (xavier_scale / (xavier_scale * 3.0));
        }
        let b2 = vec![0.0_f32; out_features];

        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            in_features,
            hidden,
            out_features,
        })
    }

    /// Apply the two-layer MLP.
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
                msg: "mlp_head forward: input is empty".into(),
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
        // Scratch buffer for the hidden activations of one sample.
        let mut h = vec![0.0_f32; self.hidden];

        for row in 0..n {
            let x = &input[row * self.in_features..(row + 1) * self.in_features];

            // Layer 1: h = ReLU(W1 x + b1).
            for (hi, hv) in h.iter_mut().enumerate() {
                let w_row = &self.w1[hi * self.in_features..(hi + 1) * self.in_features];
                let dot: f32 = w_row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
                *hv = relu(dot + self.b1[hi]);
            }

            // Layer 2: y = W2 h + b2.
            for o in 0..self.out_features {
                let w_row = &self.w2[o * self.hidden..(o + 1) * self.hidden];
                let dot: f32 = w_row.iter().zip(h.iter()).map(|(&w, &hi)| w * hi).sum();
                out[row * self.out_features + o] = dot + self.b2[o];
            }
        }
        Ok(out)
    }

    /// Per-variate time-series forward: `[T, C] → [horizon, C]`.
    ///
    /// For each variate `c` the time dimension `[0..T, c]` is treated as an
    /// `in_features`-dimensional vector, projected through the MLP to produce
    /// `out_features` (the forecast horizon).  The result is assembled as
    /// `[horizon, C]` row-major.
    ///
    /// Requires `in_features == t` and `out_features == horizon`.
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
                msg: format!(
                    "mlp_head forward_ts: in_features={} but t={}",
                    self.in_features, t
                ),
            });
        }

        let horizon = self.out_features;
        let mut out = vec![0.0_f32; horizon * c];

        for ci in 0..c {
            // Extract [T] column for this variate from [T, C] row-major layout.
            let series: Vec<f32> = (0..t).map(|ti| input[ti * c + ci]).collect();

            // Apply MLP: [T] → [horizon].
            let proj = self.forward(&series)?;

            // Write into [horizon, C] row-major output.
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
        LcgRng::new(77)
    }

    // 1. Output shape for a single sample.
    #[test]
    fn mlp_head_forward_shape() {
        let mut rng = make_rng();
        let head = MlpHead::new(64, 128, 24, &mut rng).expect("ok");
        let input = vec![0.1_f32; 64];
        let out = head.forward(&input).expect("ok");
        assert_eq!(out.len(), 24);
    }

    // 2. All outputs are finite.
    #[test]
    fn mlp_head_forward_finite() {
        let mut rng = make_rng();
        let head = MlpHead::new(32, 64, 16, &mut rng).expect("ok");
        let mut input = vec![0.0_f32; 32];
        rng.fill_normal(&mut input);
        let out = head.forward(&input).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    // 3. Zero in_features → InvalidEmbedDim(0).
    #[test]
    fn mlp_head_zero_in_error() {
        let mut rng = make_rng();
        assert!(matches!(
            MlpHead::new(0, 32, 8, &mut rng).unwrap_err(),
            TsError::InvalidEmbedDim(0)
        ));
    }

    // 4. Zero hidden → InvalidEmbedDim(0).
    #[test]
    fn mlp_head_zero_hidden_error() {
        let mut rng = make_rng();
        assert!(matches!(
            MlpHead::new(16, 0, 8, &mut rng).unwrap_err(),
            TsError::InvalidEmbedDim(0)
        ));
    }

    // 5. Zero out_features → InvalidHorizon(0).
    #[test]
    fn mlp_head_zero_out_error() {
        let mut rng = make_rng();
        assert!(matches!(
            MlpHead::new(16, 32, 0, &mut rng).unwrap_err(),
            TsError::InvalidHorizon(0)
        ));
    }

    // 6. Batch forward: N=4 samples.
    #[test]
    fn mlp_head_batch_shape() {
        let mut rng = make_rng();
        let head = MlpHead::new(8, 16, 4, &mut rng).expect("ok");
        let input = vec![0.5_f32; 4 * 8];
        let out = head.forward(&input).expect("ok");
        assert_eq!(out.len(), 4 * 4);
    }

    // 7. forward_ts shape: [T, C] → [horizon, C].
    #[test]
    fn mlp_head_ts_shape() {
        let t = 48;
        let c = 7;
        let horizon = 24;
        let mut rng = make_rng();
        let head = MlpHead::new(t, 96, horizon, &mut rng).expect("ok");
        let input = vec![0.1_f32; t * c];
        let out = head.forward_ts(&input, t, c).expect("ok");
        assert_eq!(out.len(), horizon * c);
    }

    // 8. Empty input → EmptyInput error.
    #[test]
    fn mlp_head_empty_input_error() {
        let mut rng = make_rng();
        let head = MlpHead::new(8, 16, 4, &mut rng).expect("ok");
        assert!(matches!(
            head.forward(&[]).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }
}

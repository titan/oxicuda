//! 1×1 lateral convolution for Feature Pyramid Network channel reduction.
//!
//! Each `LateralConv1x1` maps a feature map from `in_channels` channels to
//! `out_channels` channels at every spatial position independently. Weights
//! are Xavier-initialised: scale = 1/√in_channels.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

// ─── LateralWeights ───────────────────────────────────────────────────────────

/// Weights for a 1×1 lateral convolution.
///
/// Layout:
/// - `weight`: `[out_channels × in_channels]` row-major.
/// - `bias`:   `[out_channels]`.
pub struct LateralWeights {
    /// Convolution kernel: `[out_channels × in_channels]`.
    pub weight: Vec<f32>,
    /// Bias vector: `[out_channels]`.
    pub bias: Vec<f32>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
}

impl LateralWeights {
    /// Create Xavier-initialised lateral weights.
    ///
    /// Xavier scale = 1/√in_channels.
    ///
    /// # Errors
    /// Returns `InvalidImageSize` if either channel count is zero.
    pub fn new(in_channels: usize, out_channels: usize, rng: &mut LcgRng) -> VisionResult<Self> {
        if in_channels == 0 {
            return Err(VisionError::InvalidImageSize {
                height: 0,
                width: 0,
                channels: in_channels,
            });
        }
        if out_channels == 0 {
            return Err(VisionError::InvalidImageSize {
                height: 0,
                width: 0,
                channels: out_channels,
            });
        }

        let scale = 1.0_f32 / (in_channels as f32).sqrt();
        let n_weights = out_channels * in_channels;
        let mut weight = vec![0.0f32; n_weights];
        rng.fill_normal(&mut weight);
        for v in &mut weight {
            *v *= scale;
        }
        let bias = vec![0.0f32; out_channels];

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
        })
    }
}

// ─── LateralConv1x1 ──────────────────────────────────────────────────────────

/// 1×1 lateral convolution: reduces channels from `in_channels` to `out_channels`.
pub struct LateralConv1x1 {
    /// Learned weights for this lateral connection.
    pub weights: LateralWeights,
}

impl LateralConv1x1 {
    /// Construct a new `LateralConv1x1` with Xavier-initialised weights.
    ///
    /// # Errors
    /// Propagates errors from `LateralWeights::new`.
    pub fn new(in_channels: usize, out_channels: usize, rng: &mut LcgRng) -> VisionResult<Self> {
        let weights = LateralWeights::new(in_channels, out_channels, rng)?;
        Ok(Self { weights })
    }

    /// Apply the 1×1 convolution to a CHW feature map.
    ///
    /// `feat` must have length `in_channels * h * w`.
    /// Returns a new `Vec<f32>` of length `out_channels * h * w`.
    ///
    /// For each spatial position `(i, j)`:
    /// ```text
    /// out[oc, i, j] = bias[oc] + Σ_ic weight[oc, ic] * feat[ic, i, j]
    /// ```
    ///
    /// # Errors
    /// - `DimensionMismatch` if `feat.len() != in_channels * h * w`.
    /// - `EmptyInput` if `h == 0` or `w == 0`.
    pub fn forward(&self, feat: &[f32], h: usize, w: usize) -> VisionResult<Vec<f32>> {
        let ic = self.weights.in_channels;
        let oc = self.weights.out_channels;

        if h == 0 || w == 0 {
            return Err(VisionError::EmptyInput(
                "lateral conv feature map spatial dims",
            ));
        }

        let expected = ic * h * w;
        if feat.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: feat.len(),
            });
        }

        let spatial = h * w;
        let mut out = vec![0.0f32; oc * spatial];

        // For each spatial position, apply the [out_channels × in_channels] linear map.
        for pos in 0..spatial {
            for o in 0..oc {
                let w_row = &self.weights.weight[o * ic..(o + 1) * ic];
                let mut acc = self.weights.bias[o];
                for i in 0..ic {
                    // feat layout: [ic, h, w] → feat[i * spatial + pos]
                    acc += w_row[i] * feat[i * spatial + pos];
                }
                out[o * spatial + pos] = acc;
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

    // ── LateralWeights construction ──────────────────────────────────────────

    #[test]
    fn lateral_weights_valid_shape() {
        let mut rng = make_rng();
        let w = LateralWeights::new(256, 128, &mut rng).expect("valid weights");
        assert_eq!(w.weight.len(), 256 * 128, "weight tensor size");
        assert_eq!(w.bias.len(), 128, "bias size");
        assert_eq!(w.in_channels, 256);
        assert_eq!(w.out_channels, 128);
    }

    #[test]
    fn lateral_weights_zero_in_channels_errors() {
        let mut rng = make_rng();
        let r = LateralWeights::new(0, 64, &mut rng);
        assert!(r.is_err(), "expected error for in_channels=0");
    }

    #[test]
    fn lateral_weights_zero_out_channels_errors() {
        let mut rng = make_rng();
        let r = LateralWeights::new(64, 0, &mut rng);
        assert!(r.is_err(), "expected error for out_channels=0");
    }

    #[test]
    fn lateral_weights_xavier_scale_reasonable() {
        let mut rng = make_rng();
        let ic = 256;
        let w = LateralWeights::new(ic, 128, &mut rng).expect("valid weights");
        // Xavier scale = 1/sqrt(256) ≈ 0.0625; most weights should be < 3 * 0.0625
        let max_abs = w
            .weight
            .iter()
            .cloned()
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1.0,
            "Xavier-scaled weights unexpectedly large: max_abs={max_abs}"
        );
    }

    // ── LateralConv1x1::forward ───────────────────────────────────────────────

    #[test]
    fn forward_output_shape() {
        let mut rng = make_rng();
        let conv = LateralConv1x1::new(32, 16, &mut rng).expect("valid conv");
        let feat = vec![0.5f32; 32 * 8 * 8];
        let out = conv.forward(&feat, 8, 8).expect("forward ok");
        assert_eq!(out.len(), 16 * 8 * 8, "output shape [out_channels, h, w]");
    }

    #[test]
    fn forward_all_zero_input_equals_bias() {
        // With all-zero input, out[oc, i, j] = bias[oc] (which is 0 for default init).
        let mut rng = make_rng();
        let conv = LateralConv1x1::new(8, 4, &mut rng).expect("valid conv");
        let feat = vec![0.0f32; 8 * 3 * 3];
        let out = conv.forward(&feat, 3, 3).expect("forward ok");
        // All biases are 0 → all outputs should be 0.
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.abs() < 1e-7,
                "expected 0 at index {i}, got {v} (bias={:?})",
                conv.weights.bias
            );
        }
    }

    #[test]
    fn forward_wrong_input_size_errors() {
        let mut rng = make_rng();
        let conv = LateralConv1x1::new(16, 8, &mut rng).expect("valid conv");
        // Give wrong-length input (too short)
        let feat = vec![0.0f32; 16 * 4 * 4 - 1];
        let r = conv.forward(&feat, 4, 4);
        assert!(
            matches!(r, Err(VisionError::DimensionMismatch { .. })),
            "expected DimensionMismatch error"
        );
    }

    #[test]
    fn forward_zero_spatial_errors() {
        let mut rng = make_rng();
        let conv = LateralConv1x1::new(16, 8, &mut rng).expect("valid conv");
        let r = conv.forward(&[], 0, 4);
        assert!(r.is_err(), "expected error for h=0");
    }

    #[test]
    fn forward_linearity_check() {
        // Verify that the 1×1 conv is linear: out(a*x) = a*out(x) when bias=0.
        // (bias defaults to 0, so this should hold.)
        let mut rng = LcgRng::new(7);
        let conv = LateralConv1x1::new(4, 2, &mut rng).expect("valid conv");
        let feat: Vec<f32> = (0..4 * 2 * 2).map(|i| i as f32 * 0.1).collect();
        let a = 3.0f32;
        let scaled_feat: Vec<f32> = feat.iter().map(|&v| v * a).collect();
        let out1 = conv.forward(&feat, 2, 2).expect("forward ok");
        let out2 = conv.forward(&scaled_feat, 2, 2).expect("forward ok");
        for (i, (&v1, &v2)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (v2 - a * v1).abs() < 1e-5,
                "linearity violation at {i}: a*out1={}, out2={}",
                a * v1,
                v2
            );
        }
    }

    #[test]
    fn forward_different_h_w_ok() {
        // Non-square spatial dims should work fine.
        let mut rng = make_rng();
        let conv = LateralConv1x1::new(4, 8, &mut rng).expect("valid conv");
        let feat = vec![0.1f32; 4 * 5 * 7];
        let out = conv.forward(&feat, 5, 7).expect("forward non-square ok");
        assert_eq!(out.len(), 8 * 5 * 7);
    }
}

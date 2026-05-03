//! Wav2Vec2 CNN feature encoder.
//!
//! Implements the 7-layer strided 1-D convolution front-end from
//! wav2vec 2.0 (Baevski et al., 2020).  Each layer applies a causal
//! 1-D convolution, group normalisation, and GELU activation to
//! progressively downsample raw waveform (or log-energy) frames into
//! a compact `[T', C]` feature representation.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Numerically stable tanh-approximation GELU activation.
///
/// Formula: `0.5 * x * (1 + tanh(0.797_884_6 * (x + 0.044_715 * x³)))`
#[inline]
fn gelu_exact(x: f32) -> f32 {
    let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Group normalisation over a 1-D channel dimension.
///
/// `x` is logically `[channels]` for a single spatial position,
/// but here we treat the full `[channels * time]` slice that has
/// already been laid out as `[channels, time]`.  The norm is applied
/// channel-wise across the group partition.
///
/// # Errors
///
/// Returns `AudioError::ShapeMismatch` when `channels % n_groups != 0`.
fn group_norm_1d(
    x: &[f32],
    channels: usize,
    time: usize,
    n_groups: usize,
    weight: &[f32],
    bias: &[f32],
) -> AudioResult<Vec<f32>> {
    if channels % n_groups != 0 {
        return Err(AudioError::ShapeMismatch {
            msg: format!("group_norm: channels={channels} not divisible by n_groups={n_groups}"),
        });
    }
    let group_size = channels / n_groups;
    let mut out = vec![0.0_f32; channels * time];

    for g in 0..n_groups {
        let ch_start = g * group_size;
        let ch_end = ch_start + group_size;
        // Compute mean and variance across (group_channels × time).
        let total = group_size * time;
        let mut sum = 0.0_f32;
        for c in ch_start..ch_end {
            for t in 0..time {
                sum += x[c * time + t];
            }
        }
        let mean = sum / total as f32;

        let mut var = 0.0_f32;
        for c in ch_start..ch_end {
            for t in 0..time {
                let diff = x[c * time + t] - mean;
                var += diff * diff;
            }
        }
        let var = var / total as f32;
        let inv_std = 1.0 / (var + 1e-5).sqrt();

        for c in ch_start..ch_end {
            let w = weight[c];
            let b = bias[c];
            for t in 0..time {
                let norm = (x[c * time + t] - mean) * inv_std;
                out[c * time + t] = norm * w + b;
            }
        }
    }
    Ok(out)
}

/// Strided causal 1-D convolution (no padding).
///
/// `input`  — `[in_channels, in_len]` flat row-major.
/// `weight` — `[out_channels, in_channels, kernel_size]` flat.
/// `bias`   — `[out_channels]`.
///
/// Returns `[out_channels, out_len]` where
/// `out_len = (in_len − kernel_size) / stride + 1`.
///
/// # Errors
///
/// Returns `AudioError::ShapeMismatch` when `in_len < kernel_size`.
fn stride_conv1d(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_channels: usize,
    in_len: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
) -> AudioResult<Vec<f32>> {
    if in_len < kernel_size {
        return Err(AudioError::ShapeMismatch {
            msg: format!("stride_conv1d: in_len={in_len} < kernel_size={kernel_size}"),
        });
    }
    let out_len = (in_len - kernel_size) / stride + 1;
    let mut out = vec![0.0_f32; out_channels * out_len];

    for oc in 0..out_channels {
        let b = bias[oc];
        for t_out in 0..out_len {
            let t_in = t_out * stride;
            let mut acc = b;
            for ic in 0..in_channels {
                let w_off = (oc * in_channels + ic) * kernel_size;
                let x_off = ic * in_len + t_in;
                for k in 0..kernel_size {
                    acc += weight[w_off + k] * input[x_off + k];
                }
            }
            out[oc * out_len + t_out] = acc;
        }
    }
    Ok(out)
}

// ─── Public types ────────────────────────────────────────────────────────────

/// A single Wav2Vec2 CNN layer: conv1d + group norm + GELU.
#[derive(Debug)]
pub struct Wav2VecCnnLayer {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Conv kernel width.
    pub kernel_size: usize,
    /// Conv stride.
    pub stride: usize,
    /// Convolution kernel `[out_channels, in_channels, kernel_size]`.
    pub weight: Vec<f32>,
    /// Convolution bias `[out_channels]`.
    pub bias: Vec<f32>,
    /// Group-norm scale `[out_channels]`.
    pub group_norm_weight: Vec<f32>,
    /// Group-norm shift `[out_channels]`.
    pub group_norm_bias: Vec<f32>,
}

/// Construction configuration for [`Wav2VecCnnEncoder`].
#[derive(Debug, Clone)]
pub struct Wav2VecCnnConfig {
    /// Number of input channels (1 for mono waveform).
    pub in_channels: usize,
    /// Output channel width per layer (length == n_layers).
    pub channel_sizes: Vec<usize>,
    /// Kernel sizes per layer.
    pub kernel_sizes: Vec<usize>,
    /// Stride per layer.
    pub strides: Vec<usize>,
    /// Number of groups for group normalisation.
    pub n_groups: usize,
}

impl Wav2VecCnnConfig {
    /// Standard wav2vec 2.0 base configuration (7 layers, 512-ch).
    #[must_use]
    pub fn wav2vec2_base() -> Self {
        Self {
            in_channels: 1,
            channel_sizes: vec![512, 512, 512, 512, 512, 512, 512],
            kernel_sizes: vec![10, 3, 3, 3, 3, 2, 2],
            strides: vec![5, 2, 2, 2, 2, 2, 2],
            n_groups: 1,
        }
    }

    /// Tiny configuration for fast unit tests (3 layers, 32-ch).
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            in_channels: 1,
            channel_sizes: vec![32, 32, 32],
            kernel_sizes: vec![5, 3, 3],
            strides: vec![2, 2, 2],
            n_groups: 1,
        }
    }

    /// Number of layers derived from `channel_sizes`.
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.channel_sizes.len()
    }
}

/// Wav2Vec2 CNN front-end encoder: stack of [`Wav2VecCnnLayer`]s.
#[derive(Debug)]
pub struct Wav2VecCnnEncoder {
    /// Ordered list of conv layers (first = closest to the input).
    pub layers: Vec<Wav2VecCnnLayer>,
    /// Number of groups used for group normalisation in every layer.
    n_groups: usize,
}

impl Wav2VecCnnEncoder {
    /// Construct an encoder from `config`, initialising weights via Xavier uniform.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::ShapeMismatch` when `channel_sizes`, `kernel_sizes`,
    /// and `strides` have different lengths, or `AudioError::InvalidKernelSize`
    /// when any kernel is zero.
    pub fn new(config: &Wav2VecCnnConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        let n = config.n_layers();
        if config.kernel_sizes.len() != n || config.strides.len() != n {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "Wav2VecCnnConfig lengths mismatch: channel_sizes={n}, \
                     kernel_sizes={}, strides={}",
                    config.kernel_sizes.len(),
                    config.strides.len()
                ),
            });
        }
        if n == 0 {
            return Err(AudioError::EmptyInput {
                msg: "Wav2VecCnnConfig has zero layers".into(),
            });
        }
        for &ks in &config.kernel_sizes {
            if ks == 0 {
                return Err(AudioError::InvalidKernelSize(0));
            }
        }

        let mut layers = Vec::with_capacity(n);
        let mut prev_ch = config.in_channels;
        for i in 0..n {
            let out_ch = config.channel_sizes[i];
            let ks = config.kernel_sizes[i];

            // Xavier uniform initialisation.
            let fan_in = prev_ch * ks;
            let fan_out = out_ch * ks;
            let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();

            let w_len = out_ch * prev_ch * ks;
            let mut weight = vec![0.0_f32; w_len];
            for v in weight.iter_mut() {
                *v = (rng.next_f32() * 2.0 - 1.0) * limit;
            }
            let mut bias = vec![0.0_f32; out_ch];
            for v in bias.iter_mut() {
                *v = (rng.next_f32() * 2.0 - 1.0) * limit;
            }

            // Group norm: scale=1, shift=0 at init.
            let gn_weight = vec![1.0_f32; out_ch];
            let gn_bias = vec![0.0_f32; out_ch];

            layers.push(Wav2VecCnnLayer {
                in_channels: prev_ch,
                out_channels: out_ch,
                kernel_size: ks,
                stride: config.strides[i],
                weight,
                bias,
                group_norm_weight: gn_weight,
                group_norm_bias: gn_bias,
            });
            prev_ch = out_ch;
        }

        Ok(Self {
            layers,
            n_groups: config.n_groups,
        })
    }

    /// Compute the output time dimension after all strided convolutions.
    ///
    /// Each layer reduces the length as: `out = (in - k) / s + 1`.
    /// Returns 0 if the input is too short to survive all layers.
    #[must_use]
    pub fn output_len(&self, in_len: usize) -> usize {
        let mut len = in_len;
        for l in &self.layers {
            if len < l.kernel_size {
                return 0;
            }
            len = (len - l.kernel_size) / l.stride + 1;
        }
        len
    }

    /// Run the CNN encoder on a flat `[in_channels, in_len]` input.
    ///
    /// # Returns
    ///
    /// `(output, out_channels, out_len)` where `output` is `[out_channels, out_len]`.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::EmptyInput` when `in_len == 0`,
    /// `AudioError::ShapeMismatch` when the input is too short for the first kernel,
    /// or propagates errors from the internal conv / group-norm helpers.
    pub fn forward(
        &self,
        input: &[f32],
        in_channels: usize,
        in_len: usize,
    ) -> AudioResult<(Vec<f32>, usize, usize)> {
        if in_len == 0 {
            return Err(AudioError::EmptyInput {
                msg: "Wav2VecCnnEncoder: in_len is 0".into(),
            });
        }
        if input.len() != in_channels * in_len {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "input slice length {} != in_channels*in_len={}",
                    input.len(),
                    in_channels * in_len
                ),
            });
        }

        let mut cur_data = input.to_vec();
        let mut cur_ch = in_channels;
        let mut cur_len = in_len;

        for layer in &self.layers {
            // 1. Strided conv.
            let conv_out = stride_conv1d(
                &cur_data,
                &layer.weight,
                &layer.bias,
                cur_ch,
                cur_len,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
            )?;
            let out_len = (cur_len - layer.kernel_size) / layer.stride + 1;

            // 2. Group norm.
            let normed = group_norm_1d(
                &conv_out,
                layer.out_channels,
                out_len,
                self.n_groups,
                &layer.group_norm_weight,
                &layer.group_norm_bias,
            )?;

            // 3. GELU.
            let activated: Vec<f32> = normed.iter().map(|&v| gelu_exact(v)).collect();

            cur_data = activated;
            cur_ch = layer.out_channels;
            cur_len = out_len;
        }

        Ok((cur_data, cur_ch, cur_len))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GELU ─────────────────────────────────────────────────────────────────

    #[test]
    fn gelu_exact_zero() {
        let v = gelu_exact(0.0);
        assert!(v.abs() < 1e-6, "gelu(0) should be ~0, got {v}");
    }

    #[test]
    fn gelu_exact_positive_large() {
        // For large positive x, GELU ≈ x.
        let v = gelu_exact(10.0);
        assert!((v - 10.0).abs() < 0.01, "gelu(10) should be ~10, got {v}");
    }

    #[test]
    fn gelu_exact_negative_large() {
        // For large negative x, GELU ≈ 0.
        let v = gelu_exact(-10.0);
        assert!(v.abs() < 0.01, "gelu(-10) should be ~0, got {v}");
    }

    #[test]
    fn gelu_exact_one() {
        // gelu(1) ≈ 0.8413 (known value).
        let v = gelu_exact(1.0);
        assert!((v - 0.841_3).abs() < 0.002, "gelu(1) off: {v}");
    }

    // ── Group norm ───────────────────────────────────────────────────────────

    #[test]
    fn group_norm_zero_mean() {
        // After norm with scale=1, bias=0, the group mean should be ~0.
        let channels = 4;
        let time = 8;
        let x: Vec<f32> = (0..channels * time).map(|i| i as f32).collect();
        let w = vec![1.0_f32; channels];
        let b = vec![0.0_f32; channels];
        let out = group_norm_1d(&x, channels, time, 1, &w, &b).expect("group_norm_1d failed");
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(
            mean.abs() < 1e-4,
            "mean after group norm should be ~0, got {mean}"
        );
    }

    #[test]
    fn group_norm_unit_variance() {
        // After norm with scale=1, bias=0, variance should be ~1.
        let channels = 4;
        let time = 16;
        let x: Vec<f32> = (0..channels * time).map(|i| i as f32 * 0.1 - 3.2).collect();
        let w = vec![1.0_f32; channels];
        let b = vec![0.0_f32; channels];
        let out = group_norm_1d(&x, channels, time, 1, &w, &b).expect("group_norm_1d failed");
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / out.len() as f32;
        assert!(
            (var - 1.0).abs() < 0.02,
            "variance after group norm should be ~1, got {var}"
        );
    }

    #[test]
    fn group_norm_bad_groups_err() {
        let x = vec![1.0_f32; 6 * 4]; // 6 channels, 4 time
        let w = vec![1.0_f32; 6];
        let b = vec![0.0_f32; 6];
        // 6 channels / 4 groups does not divide evenly.
        let r = group_norm_1d(&x, 6, 4, 4, &w, &b);
        assert!(r.is_err(), "expected ShapeMismatch error");
    }

    // ── Stride conv ──────────────────────────────────────────────────────────

    #[test]
    fn stride_conv1d_output_length() {
        // out_len = (in_len - kernel_size) / stride + 1 = (20 - 5) / 2 + 1 = 8.
        let in_ch = 1;
        let in_len = 20usize;
        let out_ch = 4;
        let ks = 5;
        let stride = 2;
        let weight = vec![0.0_f32; out_ch * in_ch * ks];
        let bias = vec![1.0_f32; out_ch]; // bias-only → output equals bias per channel.
        let input = vec![0.0_f32; in_ch * in_len];
        let out = stride_conv1d(&input, &weight, &bias, in_ch, in_len, out_ch, ks, stride)
            .expect("stride_conv1d failed");
        let expected_out_len = (in_len - ks) / stride + 1;
        assert_eq!(out.len(), out_ch * expected_out_len);
    }

    #[test]
    fn stride_conv1d_too_short_err() {
        let input = vec![1.0_f32; 3]; // in_len=3 < ks=5
        let weight = vec![0.0_f32; 5];
        let bias = vec![0.0_f32; 1];
        let r = stride_conv1d(&input, &weight, &bias, 1, 3, 1, 5, 1);
        assert!(r.is_err());
    }

    #[test]
    fn stride_conv1d_identity_kernel() {
        // Single input channel, kernel=[1.0], stride=1 → output equals input.
        let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0_f32];
        let bias = vec![0.0_f32];
        let out =
            stride_conv1d(&input, &weight, &bias, 1, 5, 1, 1, 1).expect("stride_conv1d failed");
        assert_eq!(out, input);
    }

    // ── Encoder construction & forward ───────────────────────────────────────

    #[test]
    fn wav2vec_tiny_build_ok() {
        let cfg = Wav2VecCnnConfig::tiny();
        let mut rng = LcgRng::new(42);
        let enc = Wav2VecCnnEncoder::new(&cfg, &mut rng);
        assert!(enc.is_ok(), "tiny encoder build failed");
    }

    #[test]
    fn wav2vec_tiny_output_length() {
        // For tiny config: strides [2,2,2], kernels [5,3,3].
        // Manually: l0 = (in-5)/2+1, l1 = (l0-3)/2+1, l2 = (l1-3)/2+1.
        let cfg = Wav2VecCnnConfig::tiny();
        let mut rng = LcgRng::new(1);
        let enc = Wav2VecCnnEncoder::new(&cfg, &mut rng).expect("build");

        let in_len = 200usize;
        let l0 = (in_len - 5) / 2 + 1;
        let l1 = (l0 - 3) / 2 + 1;
        let l2 = (l1 - 3) / 2 + 1;
        let expected = l2;

        assert_eq!(enc.output_len(in_len), expected);
    }

    #[test]
    fn wav2vec_tiny_output_finite() {
        let cfg = Wav2VecCnnConfig::tiny();
        let mut rng = LcgRng::new(7);
        let enc = Wav2VecCnnEncoder::new(&cfg, &mut rng).expect("build");

        let in_len = 128usize;
        let input = vec![0.5_f32; in_len]; // mono, in_channels=1
        let (out, _, _) = enc.forward(&input, 1, in_len).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite in output");
    }

    #[test]
    fn wav2vec_tiny_shape_correct() {
        let cfg = Wav2VecCnnConfig::tiny();
        let mut rng = LcgRng::new(99);
        let enc = Wav2VecCnnEncoder::new(&cfg, &mut rng).expect("build");

        let in_len = 64usize;
        let input = vec![0.1_f32; in_len];
        let (out, out_ch, out_len) = enc.forward(&input, 1, in_len).expect("forward");

        assert_eq!(out_ch, 32, "final channel count should be 32");
        assert_eq!(out.len(), out_ch * out_len);
        assert_eq!(out_len, enc.output_len(in_len));
    }

    #[test]
    fn wav2vec_tiny_empty_input_err() {
        let cfg = Wav2VecCnnConfig::tiny();
        let mut rng = LcgRng::new(5);
        let enc = Wav2VecCnnEncoder::new(&cfg, &mut rng).expect("build");
        let r = enc.forward(&[], 1, 0);
        assert!(r.is_err());
    }

    #[test]
    fn wav2vec_base_layers_count() {
        let cfg = Wav2VecCnnConfig::wav2vec2_base();
        let mut rng = LcgRng::new(0);
        let enc = Wav2VecCnnEncoder::new(&cfg, &mut rng).expect("build");
        assert_eq!(enc.layers.len(), 7);
    }

    #[test]
    fn wav2vec_config_mismatch_err() {
        let cfg = Wav2VecCnnConfig {
            in_channels: 1,
            channel_sizes: vec![32, 32],
            kernel_sizes: vec![5, 3, 3], // length mismatch
            strides: vec![2, 2],
            n_groups: 1,
        };
        let mut rng = LcgRng::new(0);
        let r = Wav2VecCnnEncoder::new(&cfg, &mut rng);
        assert!(r.is_err());
    }

    #[test]
    fn wav2vec_tiny_output_deterministic() {
        let cfg = Wav2VecCnnConfig::tiny();
        let mut rng_a = LcgRng::new(17);
        let mut rng_b = LcgRng::new(17);
        let enc_a = Wav2VecCnnEncoder::new(&cfg, &mut rng_a).expect("build");
        let enc_b = Wav2VecCnnEncoder::new(&cfg, &mut rng_b).expect("build");
        let input = vec![0.3_f32; 80];
        let (out_a, _, _) = enc_a.forward(&input, 1, 80).expect("forward");
        let (out_b, _, _) = enc_b.forward(&input, 1, 80).expect("forward");
        assert_eq!(out_a, out_b);
    }
}

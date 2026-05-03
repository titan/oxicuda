//! Single WaveNet dilated residual block.
//!
//! Each block performs:
//!
//! ```text
//! h              = dilated_causal_conv1d(x, dilation=d, kernel=K)   // [2C, T]
//! filter_out     = tanh_safe(h[0..C, :])
//! gate_out       = sigmoid_stable(h[C..2C, :])
//! activated      = filter_out ⊙ gate_out                            // [C, T]
//! skip           = pointwise_conv(activated, skip_weight, skip_bias) // [skip_C, T]
//! residual       = pointwise_conv(activated, res_weight, res_bias) + x // [C, T]
//! ```
//!
//! The receptive field of a single block is `1 + (kernel_size - 1) * dilation`
//! time steps.  Stacking blocks with exponentially growing dilations gives
//! the full WaveNet receptive field.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private numeric helpers ──────────────────────────────────────────────────

/// Numerically clamped `tanh`: clamps `x` to `[-20, 20]` before evaluation
/// to prevent `f32` overflow in extreme activations.
#[inline]
fn tanh_safe(x: f32) -> f32 {
    x.clamp(-20.0, 20.0).tanh()
}

/// Numerically stable sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid_stable(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─── Private convolution kernels ─────────────────────────────────────────────

/// Left-zero-padded dilated causal 1-D convolution.
///
/// Applies a causal dilated convolution so that output at time `t` depends
/// only on inputs at times `≤ t`.  The input is implicitly left-padded with
/// `(kernel_size - 1) * dilation` zero-value positions.
///
/// Layout conventions:
/// - `input`  — row-major `[in_channels, T]`
/// - `weight` — row-major `[out_channels, in_channels, kernel_size]`
/// - `bias`   — `[out_channels]`
///
/// Returns a row-major `[out_channels, T]` buffer.
fn dilated_causal_conv1d(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_channels: usize,
    out_channels: usize,
    t: usize,
    kernel_size: usize,
    dilation: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_channels * t];

    for oc in 0..out_channels {
        let b = bias[oc];
        for time in 0..t {
            let mut acc = b;
            for k in 0..kernel_size {
                // Dilated offset from the current time step (causal).
                // For kernel position k, the source position (before padding)
                // is: time - (kernel_size - 1 - k) * dilation.
                // Positions that land before t=0 map to the implicit zero pad.
                let lag = (kernel_size - 1 - k) * dilation;
                if time < lag {
                    // Still within the left zero-pad region — contribution is 0.
                    continue;
                }
                let src_t = time - lag;
                for ic in 0..in_channels {
                    let w_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                    acc += weight[w_idx] * input[ic * t + src_t];
                }
            }
            out[oc * t + time] = acc;
        }
    }
    out
}

/// Pointwise (1×1) convolution along the time axis.
///
/// - `input`  — row-major `[in_ch, T]`
/// - `weight` — row-major `[out_ch, in_ch]`
/// - `bias`   — `[out_ch]`
///
/// Returns row-major `[out_ch, T]`.
fn pointwise_conv(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_ch: usize,
    out_ch: usize,
    t: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_ch * t];
    for oc in 0..out_ch {
        let b = bias[oc];
        for time in 0..t {
            let mut acc = b;
            for ic in 0..in_ch {
                acc += weight[oc * in_ch + ic] * input[ic * t + time];
            }
            out[oc * t + time] = acc;
        }
    }
    out
}

// ─── Xavier initialisation ───────────────────────────────────────────────────

/// Fill `buf` with Xavier-uniform values scaled by `sqrt(6 / (fan_in + fan_out))`.
fn xavier_init(rng: &mut LcgRng, fan_in: usize, fan_out: usize, buf: &mut [f32]) {
    let limit = (6.0_f32 / (fan_in + fan_out).max(1) as f32).sqrt();
    for v in buf.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * limit;
    }
}

// ─── WaveNetBlock ─────────────────────────────────────────────────────────────

/// A single WaveNet dilated residual block.
///
/// Stores all learnable parameters on the CPU as contiguous `f32` buffers
/// in row-major order.  Forward evaluation is fully CPU-bound and allocation-free
/// in the sense that all temporaries are freshly allocated (suitable for
/// correctness testing and CPU inference prototyping).
///
/// ## Weight buffer layouts
///
/// | Field              | Shape                                             |
/// |--------------------|---------------------------------------------------|
/// | `conv_weight`      | `[2 * residual_channels, residual_channels, kernel_size]` |
/// | `conv_bias`        | `[2 * residual_channels]`                         |
/// | `skip_weight`      | `[skip_channels, residual_channels]`              |
/// | `skip_bias`        | `[skip_channels]`                                 |
/// | `residual_weight`  | `[residual_channels, residual_channels]`          |
/// | `residual_bias`    | `[residual_channels]`                             |
#[derive(Debug)]
pub struct WaveNetBlock {
    /// Number of residual (hidden) channels `C`.
    pub residual_channels: usize,
    /// Number of skip channels `S`.
    pub skip_channels: usize,
    /// Convolution kernel width.
    pub kernel_size: usize,
    /// Dilation factor for the dilated causal convolution.
    pub dilation: usize,
    /// Dilated conv filter weights: `[2C, C, K]` row-major.
    pub conv_weight: Vec<f32>,
    /// Dilated conv bias: `[2C]`.
    pub conv_bias: Vec<f32>,
    /// Skip 1×1 conv weights: `[S, C]` row-major.
    pub skip_weight: Vec<f32>,
    /// Skip 1×1 conv bias: `[S]`.
    pub skip_bias: Vec<f32>,
    /// Residual 1×1 conv weights: `[C, C]` row-major.
    pub residual_weight: Vec<f32>,
    /// Residual 1×1 conv bias: `[C]`.
    pub residual_bias: Vec<f32>,
}

impl WaveNetBlock {
    /// Construct a `WaveNetBlock` with Xavier-initialised weights.
    ///
    /// # Errors
    ///
    /// - `AudioError::InvalidDilation(0)` when `dilation == 0`.
    /// - `AudioError::InvalidKernelSize(0)` when `kernel_size == 0`.
    /// - `AudioError::InvalidEmbedDim(0)` when `residual_channels == 0`
    ///   or `skip_channels == 0`.
    pub fn new(
        residual_channels: usize,
        skip_channels: usize,
        kernel_size: usize,
        dilation: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        if dilation == 0 {
            return Err(AudioError::InvalidDilation(0));
        }
        if kernel_size == 0 {
            return Err(AudioError::InvalidKernelSize(0));
        }
        if residual_channels == 0 || skip_channels == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }

        let c = residual_channels;
        let s = skip_channels;
        let k = kernel_size;

        // Dilated conv: fan_in = C * K, fan_out = 2C * K (gated).
        let conv_len = 2 * c * c * k;
        let mut conv_weight = vec![0.0_f32; conv_len];
        xavier_init(rng, c * k, 2 * c * k, &mut conv_weight);
        let conv_bias = vec![0.0_f32; 2 * c];

        // Skip 1×1 conv.
        let skip_len = s * c;
        let mut skip_weight = vec![0.0_f32; skip_len];
        xavier_init(rng, c, s, &mut skip_weight);
        let skip_bias = vec![0.0_f32; s];

        // Residual 1×1 conv.
        let res_len = c * c;
        let mut residual_weight = vec![0.0_f32; res_len];
        xavier_init(rng, c, c, &mut residual_weight);
        let residual_bias = vec![0.0_f32; c];

        Ok(Self {
            residual_channels: c,
            skip_channels: s,
            kernel_size: k,
            dilation,
            conv_weight,
            conv_bias,
            skip_weight,
            skip_bias,
            residual_weight,
            residual_bias,
        })
    }

    /// Run the forward pass of this block.
    ///
    /// # Arguments
    ///
    /// * `x` — Input tensor laid out as row-major `[residual_channels, T]`.
    /// * `t` — Time dimension `T` (`x.len()` must equal `residual_channels * T`).
    ///
    /// # Returns
    ///
    /// A tuple `(residual_output, skip_output)` where:
    /// - `residual_output` is `[residual_channels, T]` (input for the next block).
    /// - `skip_output` is `[skip_channels, T]` (accumulated by the stack).
    ///
    /// # Errors
    ///
    /// - `AudioError::DimensionMismatch` when `x.len() != residual_channels * t`.
    /// - `AudioError::NonFinite` when any output value is non-finite.
    pub fn forward(&self, x: &[f32], t: usize) -> AudioResult<(Vec<f32>, Vec<f32>)> {
        let c = self.residual_channels;
        let s = self.skip_channels;

        let expected_len = c * t;
        if x.len() != expected_len {
            return Err(AudioError::DimensionMismatch {
                expected: expected_len,
                got: x.len(),
            });
        }

        // Step 1: dilated causal conv  →  [2C, T].
        let h = dilated_causal_conv1d(
            x,
            &self.conv_weight,
            &self.conv_bias,
            c,
            2 * c,
            t,
            self.kernel_size,
            self.dilation,
        );

        // Step 2: gated activation.
        // filter = tanh(h[0..C, :]), gate = sigmoid(h[C..2C, :]).
        let mut activated = vec![0.0_f32; c * t];
        for ch in 0..c {
            for time in 0..t {
                let filter = tanh_safe(h[ch * t + time]);
                let gate = sigmoid_stable(h[(ch + c) * t + time]);
                activated[ch * t + time] = filter * gate;
            }
        }

        // Step 3: skip projection  →  [S, T].
        let skip = pointwise_conv(&activated, &self.skip_weight, &self.skip_bias, c, s, t);

        // Step 4: residual projection  →  [C, T], then add input x.
        let mut residual = pointwise_conv(
            &activated,
            &self.residual_weight,
            &self.residual_bias,
            c,
            c,
            t,
        );
        for (r, xi) in residual.iter_mut().zip(x.iter()) {
            *r += xi;
        }

        // Guard against non-finite values from degenerate weight configurations.
        let all_finite =
            residual.iter().all(|v| v.is_finite()) && skip.iter().all(|v| v.is_finite());
        if !all_finite {
            return Err(AudioError::NonFinite {
                msg: "WaveNetBlock forward produced non-finite values".into(),
            });
        }

        Ok((residual, skip))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(c: usize, s: usize, k: usize, d: usize, seed: u64) -> WaveNetBlock {
        let mut rng = LcgRng::new(seed);
        WaveNetBlock::new(c, s, k, d, &mut rng).expect("block construction should succeed")
    }

    // ── construction ──────────────────────────────────────────────────────────

    #[test]
    fn wavenet_block_new_ok() {
        let block = make_block(8, 8, 3, 1, 42);
        assert_eq!(block.residual_channels, 8);
        assert_eq!(block.skip_channels, 8);
        assert_eq!(block.kernel_size, 3);
        assert_eq!(block.dilation, 1);
        // conv_weight: [2*8, 8, 3] = 384 elements
        assert_eq!(block.conv_weight.len(), 2 * 8 * 8 * 3);
        // skip_weight: [8, 8] = 64 elements
        assert_eq!(block.skip_weight.len(), 8 * 8);
        // residual_weight: [8, 8] = 64 elements
        assert_eq!(block.residual_weight.len(), 8 * 8);
    }

    #[test]
    fn wavenet_block_dilation_zero_error() {
        let mut rng = LcgRng::new(1);
        let result = WaveNetBlock::new(8, 8, 3, 0, &mut rng);
        assert_eq!(result.unwrap_err(), AudioError::InvalidDilation(0));
    }

    #[test]
    fn wavenet_block_kernel_zero_error() {
        let mut rng = LcgRng::new(2);
        let result = WaveNetBlock::new(8, 8, 0, 1, &mut rng);
        assert_eq!(result.unwrap_err(), AudioError::InvalidKernelSize(0));
    }

    #[test]
    fn wavenet_block_embed_dim_zero_error_residual() {
        let mut rng = LcgRng::new(3);
        let result = WaveNetBlock::new(0, 8, 3, 1, &mut rng);
        assert_eq!(result.unwrap_err(), AudioError::InvalidEmbedDim(0));
    }

    #[test]
    fn wavenet_block_embed_dim_zero_error_skip() {
        let mut rng = LcgRng::new(4);
        let result = WaveNetBlock::new(8, 0, 3, 1, &mut rng);
        assert_eq!(result.unwrap_err(), AudioError::InvalidEmbedDim(0));
    }

    // ── forward shape ─────────────────────────────────────────────────────────

    #[test]
    fn wavenet_block_residual_shape() {
        let block = make_block(8, 16, 3, 1, 10);
        let t = 20;
        let x = vec![0.1_f32; 8 * t];
        let (residual, _skip) = block.forward(&x, t).expect("forward should succeed");
        assert_eq!(residual.len(), 8 * t);
    }

    #[test]
    fn wavenet_block_skip_shape() {
        let block = make_block(8, 16, 3, 1, 11);
        let t = 20;
        let x = vec![0.1_f32; 8 * t];
        let (_residual, skip) = block.forward(&x, t).expect("forward should succeed");
        assert_eq!(skip.len(), 16 * t);
    }

    #[test]
    fn wavenet_block_output_finite() {
        let block = make_block(8, 8, 3, 2, 55);
        let t = 16;
        let mut rng = LcgRng::new(999);
        let mut x = vec![0.0_f32; 8 * t];
        rng.fill_normal(&mut x);
        let (res, skip) = block.forward(&x, t).expect("forward should succeed");
        assert!(
            res.iter().all(|v| v.is_finite()),
            "residual contains non-finite"
        );
        assert!(
            skip.iter().all(|v| v.is_finite()),
            "skip contains non-finite"
        );
    }

    #[test]
    fn wavenet_block_dimension_mismatch_error() {
        let block = make_block(8, 8, 3, 1, 7);
        let t = 10;
        // Provide wrong-length input (one element short).
        let x = vec![0.0_f32; 8 * t - 1];
        let result = block.forward(&x, t);
        assert!(matches!(result, Err(AudioError::DimensionMismatch { .. })));
    }

    // ── numeric helpers ───────────────────────────────────────────────────────

    #[test]
    fn tanh_safe_at_zero() {
        assert!((tanh_safe(0.0) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn tanh_safe_large_positive() {
        // tanh(20) is essentially 1.0; tanh_safe(1e10) must not overflow.
        let v = tanh_safe(1e10);
        assert!((v - 1.0).abs() < 1e-6, "expected ≈1.0, got {v}");
    }

    #[test]
    fn tanh_safe_large_negative() {
        let v = tanh_safe(-1e10);
        assert!((v + 1.0).abs() < 1e-6, "expected ≈-1.0, got {v}");
    }

    #[test]
    fn sigmoid_stable_at_zero() {
        let v = sigmoid_stable(0.0);
        assert!((v - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5, got {v}");
    }

    #[test]
    fn sigmoid_stable_large_positive() {
        let v = sigmoid_stable(100.0);
        assert!(
            (v - 1.0).abs() < 1e-6,
            "sigmoid(100) should be ≈1.0, got {v}"
        );
    }

    #[test]
    fn sigmoid_stable_large_negative() {
        let v = sigmoid_stable(-100.0);
        assert!(v.abs() < 1e-6, "sigmoid(-100) should be ≈0.0, got {v}");
    }

    // ── causality / no future leakage ─────────────────────────────────────────

    #[test]
    fn dilated_causal_no_future_leakage() {
        // Build two inputs that are identical for time steps 0..split but differ
        // for time steps split..t.  Because the layout is [in_channels, T]
        // (channel-first / row-major), element [c, t] lives at index c*T + t.
        //
        // The receptive field for kernel_size=3, dilation=2 is
        // 1 + (3-1)*2 = 5 frames.  Output at time step `s` therefore depends
        // only on input times {s, s-2, s-4} (with zeros for negative indices).
        // We split at time step `split`, so every output at time < split must
        // be identical for both inputs.
        let in_channels = 4;
        let out_channels = 4;
        let kernel_size = 3;
        let dilation = 2;
        let t = 12;
        let split = 6; // both inputs are identical for times 0..split

        let weight_len = out_channels * in_channels * kernel_size;
        let mut rng = LcgRng::new(31415);
        let mut weight = vec![0.0_f32; weight_len];
        rng.fill_normal(&mut weight);
        let bias = vec![0.0_f32; out_channels];

        // Fill all elements, then copy the shared prefix column-by-column so
        // that input_b[c*t + time] == input_a[c*t + time] for all c and time < split.
        let mut input_a = vec![0.0_f32; in_channels * t];
        let mut input_b = vec![0.0_f32; in_channels * t];
        rng.fill_normal(&mut input_a);
        rng.fill_normal(&mut input_b);
        // Overwrite input_b's shared prefix times 0..split for every channel.
        for c in 0..in_channels {
            for time in 0..split {
                input_b[c * t + time] = input_a[c * t + time];
            }
        }

        let out_a = dilated_causal_conv1d(
            &input_a,
            &weight,
            &bias,
            in_channels,
            out_channels,
            t,
            kernel_size,
            dilation,
        );
        let out_b = dilated_causal_conv1d(
            &input_b,
            &weight,
            &bias,
            in_channels,
            out_channels,
            t,
            kernel_size,
            dilation,
        );

        // Every output time strictly before `split` must be identical: it only
        // touches input times 0..split which are the same in both tensors.
        for oc in 0..out_channels {
            for time in 0..split {
                let a = out_a[oc * t + time];
                let b = out_b[oc * t + time];
                assert!(
                    (a - b).abs() < 1e-5,
                    "future leakage at oc={oc} t={time}: {a} vs {b}"
                );
            }
        }
    }

    // ── pointwise_conv ────────────────────────────────────────────────────────

    #[test]
    fn pointwise_conv_shape_correct() {
        let in_ch = 6;
        let out_ch = 10;
        let t = 15;
        let input = vec![1.0_f32; in_ch * t];
        let weight = vec![1.0_f32; out_ch * in_ch];
        let bias = vec![0.0_f32; out_ch];
        let out = pointwise_conv(&input, &weight, &bias, in_ch, out_ch, t);
        assert_eq!(out.len(), out_ch * t);
        // Each output should equal the sum of in_ch (since input=1, weight=1, bias=0).
        for &v in &out {
            assert!((v - in_ch as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn pointwise_conv_bias_applied() {
        let in_ch = 4;
        let out_ch = 4;
        let t = 3;
        let input = vec![0.0_f32; in_ch * t];
        let weight = vec![0.0_f32; out_ch * in_ch];
        let bias = vec![7.0_f32; out_ch];
        let out = pointwise_conv(&input, &weight, &bias, in_ch, out_ch, t);
        assert_eq!(out.len(), out_ch * t);
        for &v in &out {
            assert!((v - 7.0).abs() < 1e-6);
        }
    }
}

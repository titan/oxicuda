//! Full WaveNet dilated residual stack with output head.
//!
//! Assembles `dilation_cycles × layers_per_cycle` [`WaveNetBlock`]s with
//! exponentially-increasing dilations `[1, 2, 4, …, 2^(layers_per_cycle-1)]`
//! repeating for each cycle.
//!
//! The forward pass:
//! 1. Passes the input through every block sequentially (each block's residual
//!    output feeds the next block's input).
//! 2. Accumulates all skip outputs element-wise.
//! 3. Applies `ReLU → head_w1/b1 (1×1 conv) → ReLU → head_w2/b2 (1×1 conv)`
//!    to the summed skip tensor to produce the final `[skip_channels, T]` output.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;
use crate::vocoder::wavenet_block::WaveNetBlock;

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Rectified linear unit.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Fill `buf` with Xavier-uniform values scaled by `sqrt(6 / (fan_in + fan_out))`.
fn xavier_init(rng: &mut LcgRng, fan_in: usize, fan_out: usize, buf: &mut [f32]) {
    let limit = (6.0_f32 / (fan_in + fan_out).max(1) as f32).sqrt();
    for v in buf.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * limit;
    }
}

/// Apply a pointwise (1×1) convolution along the time axis.
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

// ─── WaveNetConfig ────────────────────────────────────────────────────────────

/// Configuration for a [`WaveNetStack`].
///
/// The total number of blocks is `dilation_cycles × layers_per_cycle`.
/// Dilations within each cycle are `1, 2, 4, …, 2^(layers_per_cycle - 1)`.
#[derive(Debug, Clone)]
pub struct WaveNetConfig {
    /// Number of residual (hidden) channels.
    pub residual_channels: usize,
    /// Number of skip-connection channels.
    pub skip_channels: usize,
    /// Dilated convolution kernel width.
    pub kernel_size: usize,
    /// Number of dilation cycles (each cycle repeats the dilation schedule).
    pub dilation_cycles: usize,
    /// Number of layers per cycle; dilations are `1, 2, 4, …, 2^(layers-1)`.
    pub layers_per_cycle: usize,
}

impl WaveNetConfig {
    /// Full-scale default WaveNet configuration (30 blocks, 256 channels).
    ///
    /// Matches the original WaveNet architecture with three cycles of
    /// ten exponentially-dilated layers each, giving a receptive field of
    /// `3 × (1 + 2 + … + 512) × (K-1) + 1` time steps for `K = 3`.
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            residual_channels: 256,
            skip_channels: 256,
            kernel_size: 3,
            dilation_cycles: 3,
            layers_per_cycle: 10,
        }
    }

    /// Tiny configuration for fast unit-testing (8 blocks, 16 channels).
    ///
    /// Two dilation cycles of four layers each
    /// (dilations `1, 2, 4, 8` × 2 cycles = 8 blocks total).
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            residual_channels: 16,
            skip_channels: 16,
            kernel_size: 3,
            dilation_cycles: 2,
            layers_per_cycle: 4,
        }
    }
}

// ─── WaveNetStack ─────────────────────────────────────────────────────────────

/// A complete multi-cycle WaveNet dilated residual stack.
///
/// Holds `dilation_cycles × layers_per_cycle` [`WaveNetBlock`]s and a
/// two-layer 1×1-conv output head that maps the summed skip tensor to the
/// final `[skip_channels, T]` output.
///
/// ## Output head
///
/// ```text
/// skip_sum  →  ReLU  →  head_w1/b1 (1×1 conv)  →  ReLU  →  head_w2/b2 (1×1 conv)
/// ```
pub struct WaveNetStack {
    /// Ordered list of residual blocks.
    pub blocks: Vec<WaveNetBlock>,
    /// Stack configuration.
    pub config: WaveNetConfig,
    /// Output head first-layer weights: `[skip_channels, skip_channels]`.
    pub head_w1: Vec<f32>,
    /// Output head first-layer bias: `[skip_channels]`.
    pub head_b1: Vec<f32>,
    /// Output head second-layer weights: `[skip_channels, skip_channels]`.
    pub head_w2: Vec<f32>,
    /// Output head second-layer bias: `[skip_channels]`.
    pub head_b2: Vec<f32>,
}

impl WaveNetStack {
    /// Build a `WaveNetStack` from the given configuration.
    ///
    /// Blocks are constructed in dilation order
    /// `[1, 2, 4, …, 2^(layers_per_cycle-1)]` repeated `dilation_cycles` times.
    ///
    /// # Errors
    ///
    /// Propagates any `AudioError` returned by [`WaveNetBlock::new`], or
    /// `AudioError::InvalidEmbedDim(0)` when `dilation_cycles == 0` or
    /// `layers_per_cycle == 0`.
    pub fn new(config: WaveNetConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        if config.dilation_cycles == 0 || config.layers_per_cycle == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }

        let c = config.residual_channels;
        let s = config.skip_channels;
        let k = config.kernel_size;

        let total_blocks = config.dilation_cycles * config.layers_per_cycle;
        let mut blocks = Vec::with_capacity(total_blocks);

        for _cycle in 0..config.dilation_cycles {
            for layer in 0..config.layers_per_cycle {
                let dilation = 1usize << layer;
                blocks.push(WaveNetBlock::new(c, s, k, dilation, rng)?);
            }
        }

        // Output head weights — Xavier-initialised 1×1 convs.
        let head_sq = s * s;
        let mut head_w1 = vec![0.0_f32; head_sq];
        xavier_init(rng, s, s, &mut head_w1);
        let head_b1 = vec![0.0_f32; s];

        let mut head_w2 = vec![0.0_f32; head_sq];
        xavier_init(rng, s, s, &mut head_w2);
        let head_b2 = vec![0.0_f32; s];

        Ok(Self {
            blocks,
            config,
            head_w1,
            head_b1,
            head_w2,
            head_b2,
        })
    }

    /// Run the full WaveNet forward pass.
    ///
    /// # Arguments
    ///
    /// * `x` — Input tensor, row-major `[residual_channels, T]`.
    /// * `t` — Time dimension `T`.
    ///
    /// # Returns
    ///
    /// Row-major `[skip_channels, T]` after the output head.
    ///
    /// # Errors
    ///
    /// - `AudioError::DimensionMismatch` when `x.len() != residual_channels * t`.
    /// - `AudioError::NonFinite` if any intermediate or final tensor is non-finite.
    pub fn forward(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        let c = self.config.residual_channels;
        let s = self.config.skip_channels;

        let expected = c * t;
        if x.len() != expected {
            return Err(AudioError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // Initialise skip accumulator to zero.
        let mut skip_sum = vec![0.0_f32; s * t];
        // Running residual — starts as a copy of the input.
        let mut current = x.to_vec();

        for block in &self.blocks {
            let (residual_out, skip_out) = block.forward(&current, t)?;
            // Accumulate skip contribution.
            for (acc, val) in skip_sum.iter_mut().zip(skip_out.iter()) {
                *acc += val;
            }
            current = residual_out;
        }

        // Output head: ReLU → 1×1 conv (w1/b1) → ReLU → 1×1 conv (w2/b2).
        let after_relu1: Vec<f32> = skip_sum.iter().map(|&v| relu(v)).collect();
        let after_conv1 = pointwise_conv(&after_relu1, &self.head_w1, &self.head_b1, s, s, t);
        let after_relu2: Vec<f32> = after_conv1.iter().map(|&v| relu(v)).collect();
        let output = pointwise_conv(&after_relu2, &self.head_w2, &self.head_b2, s, s, t);

        let all_finite = output.iter().all(|v| v.is_finite());
        if !all_finite {
            return Err(AudioError::NonFinite {
                msg: "WaveNetStack output head produced non-finite values".into(),
            });
        }

        Ok(output)
    }

    /// Total number of residual blocks in the stack.
    #[must_use]
    pub fn n_blocks(&self) -> usize {
        self.config.dilation_cycles * self.config.layers_per_cycle
    }

    /// Ordered list of dilation factors across all blocks.
    ///
    /// Returns `[1, 2, 4, …, 2^(layers_per_cycle-1)]` repeated
    /// `dilation_cycles` times.
    #[must_use]
    pub fn dilations(&self) -> Vec<usize> {
        let lpc = self.config.layers_per_cycle;
        let mut out = Vec::with_capacity(self.n_blocks());
        for _cycle in 0..self.config.dilation_cycles {
            for layer in 0..lpc {
                out.push(1usize << layer);
            }
        }
        out
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tiny(seed: u64) -> WaveNetStack {
        let mut rng = LcgRng::new(seed);
        WaveNetStack::new(WaveNetConfig::tiny(), &mut rng)
            .expect("tiny stack construction should succeed")
    }

    // ── construction ──────────────────────────────────────────────────────────

    #[test]
    fn wavenet_stack_tiny_new_ok() {
        let stack = make_tiny(42);
        assert_eq!(stack.blocks.len(), 8); // 2 cycles × 4 layers
        assert_eq!(stack.config.residual_channels, 16);
        assert_eq!(stack.config.skip_channels, 16);
    }

    #[test]
    fn wavenet_stack_n_blocks_correct() {
        let stack = make_tiny(7);
        assert_eq!(
            stack.n_blocks(),
            stack.config.dilation_cycles * stack.config.layers_per_cycle
        );
        assert_eq!(stack.n_blocks(), 8);
    }

    #[test]
    fn wavenet_config_default_dilation_count() {
        let mut rng = LcgRng::new(1);
        let stack = WaveNetStack::new(WaveNetConfig::default_config(), &mut rng)
            .expect("default config construction should succeed");
        assert_eq!(stack.n_blocks(), 30); // 3 × 10
    }

    #[test]
    fn wavenet_config_tiny_n_blocks() {
        let stack = make_tiny(2);
        assert_eq!(stack.n_blocks(), 8); // 2 × 4
    }

    // ── dilation schedule ─────────────────────────────────────────────────────

    #[test]
    fn wavenet_stack_dilations_exponential() {
        let stack = make_tiny(99);
        let d = stack.dilations();
        // Two cycles, each [1, 2, 4, 8].
        assert_eq!(d.len(), 8);
        let expected: Vec<usize> = vec![1, 2, 4, 8, 1, 2, 4, 8];
        assert_eq!(d, expected);
    }

    #[test]
    fn wavenet_stack_dilations_match_blocks() {
        let stack = make_tiny(13);
        let d = stack.dilations();
        for (block, &dil) in stack.blocks.iter().zip(d.iter()) {
            assert_eq!(block.dilation, dil);
        }
    }

    // ── forward shape and correctness ─────────────────────────────────────────

    #[test]
    fn wavenet_stack_forward_output_shape() {
        let stack = make_tiny(55);
        let t = 12;
        let c = stack.config.residual_channels;
        let s = stack.config.skip_channels;
        let x = vec![0.05_f32; c * t];
        let out = stack.forward(&x, t).expect("forward should succeed");
        assert_eq!(out.len(), s * t);
    }

    #[test]
    fn wavenet_stack_forward_finite() {
        let stack = make_tiny(77);
        let t = 8;
        let c = stack.config.residual_channels;
        let mut rng = LcgRng::new(1234);
        let mut x = vec![0.0_f32; c * t];
        rng.fill_normal(&mut x);
        let out = stack.forward(&x, t).expect("forward should succeed");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output contains non-finite values"
        );
    }

    #[test]
    fn wavenet_stack_single_block_shape() {
        let cfg = WaveNetConfig {
            residual_channels: 8,
            skip_channels: 8,
            kernel_size: 3,
            dilation_cycles: 1,
            layers_per_cycle: 1,
        };
        let mut rng = LcgRng::new(5);
        let stack = WaveNetStack::new(cfg, &mut rng)
            .expect("single-block stack construction should succeed");
        assert_eq!(stack.n_blocks(), 1);
        let t = 4;
        let x = vec![0.1_f32; 8 * t];
        let out = stack.forward(&x, t).expect("forward should succeed");
        assert_eq!(out.len(), 8 * t);
    }

    #[test]
    fn wavenet_stack_different_t_values() {
        let stack = make_tiny(11);
        let c = stack.config.residual_channels;
        let s = stack.config.skip_channels;
        for t in [1, 10, 100] {
            let x = vec![0.01_f32; c * t];
            let out = stack
                .forward(&x, t)
                .unwrap_or_else(|e| panic!("forward(t={t}) failed: {e}"));
            assert_eq!(out.len(), s * t, "wrong output len for t={t}");
        }
    }

    // ── pointwise_conv helper ─────────────────────────────────────────────────

    #[test]
    fn pointwise_conv_output_len() {
        let s = 16;
        let t = 5;
        let input = vec![1.0_f32; s * t];
        let weight = vec![1.0_f32; s * s];
        let bias = vec![0.0_f32; s];
        let out = pointwise_conv(&input, &weight, &bias, s, s, t);
        assert_eq!(out.len(), s * t);
    }

    #[test]
    fn relu_zero_and_negative_clamped() {
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(-5.0), 0.0);
        assert!((relu(3.0) - 3.0).abs() < 1e-7);
    }

    // ── error propagation ─────────────────────────────────────────────────────

    #[test]
    fn wavenet_stack_dimension_mismatch_error() {
        let stack = make_tiny(100);
        let t = 8;
        let c = stack.config.residual_channels;
        // Provide an input that's one element too long.
        let x = vec![0.0_f32; c * t + 1];
        let result = stack.forward(&x, t);
        assert!(matches!(result, Err(AudioError::DimensionMismatch { .. })));
    }
}

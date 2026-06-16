//! EfficientNet MBConv (Mobile Inverted Bottleneck Convolution) block.
//!
//! Sandler et al. 2018 "MobileNetV2: Inverted Residuals and Linear Bottlenecks".
//! Tan & Le 2019 "EfficientNet: Rethinking Model Scaling for CNNs".
//!
//! MBConv pipeline:
//! ```text
//! Expand  (1×1, in_ch  → in_ch * expand_ratio, ReLU6)
//! DW-Conv (k×k, depthwise, same groups, ReLU6)
//! SE      (squeeze-excite gate)
//! Project (1×1, in_ch * expand_ratio → out_ch, linear)
//! Skip    (if stride==1 && in_ch==out_ch, add input)
//! ```
//!
//! For CPU unit testing we use a 1-D "spatial" abstraction:
//! the input tensor is treated as `[batch_size × in_channels]` and the output
//! as `[batch_size × out_channels]`.  This preserves all channel-wise
//! computations (expand, SE gate, project, skip) while avoiding the
//! full 2-D convolution that is prohibitively expensive in pure Rust.

use crate::error::{VisionError, VisionResult};
use crate::handle::LcgRng;

/// Type alias for the RNG used by this module.
pub type VisionRng = LcgRng;

// ─── Activation helpers ───────────────────────────────────────────────────────

/// ReLU6: clamp to `[0, 6]`.
#[inline]
fn relu6(x: f32) -> f32 {
    x.clamp(0.0, 6.0)
}

/// Sigmoid: numerically stable.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a single MBConv block.
#[derive(Debug, Clone)]
pub struct MbConvConfig {
    /// Input channel count.
    pub in_channels: usize,
    /// Output channel count.
    pub out_channels: usize,
    /// Expansion ratio for the hidden channel dimension (≥ 1).
    pub expand_ratio: usize,
    /// Stride (1 = same spatial size, 2 = spatial downsampling).
    pub stride: usize,
    /// Depthwise convolution kernel size (3 or 5 in EfficientNet).
    pub kernel_size: usize,
    /// Squeeze-Excite reduction ratio (e.g. 0.25).
    pub se_ratio: f32,
    /// Input spatial height (used for metadata / future spatial convs).
    pub h: usize,
    /// Input spatial width (used for metadata / future spatial convs).
    pub w: usize,
}

impl MbConvConfig {
    /// Expanded channel count.
    #[must_use]
    pub fn expanded_channels(&self) -> usize {
        self.in_channels * self.expand_ratio
    }

    /// Squeeze-Excite hidden channel count: max(1, round(in_ch * se_ratio)).
    #[must_use]
    pub fn se_channels(&self) -> usize {
        let se = (self.in_channels as f32 * self.se_ratio).round() as usize;
        se.max(1)
    }

    /// Whether a skip connection is applicable.
    #[must_use]
    pub fn has_skip(&self) -> bool {
        self.stride == 1 && self.in_channels == self.out_channels
    }
}

// ─── MbConvBlock ─────────────────────────────────────────────────────────────

/// EfficientNet MBConv inverted residual block.
///
/// Weight layout (row-major):
/// - `expand_w`: `[exp_ch × in_ch]`  (1×1 expand conv, bias: `expand_b [exp_ch]`)
/// - `dw_w`    : `[exp_ch × k × k]`  (depthwise weights, bias: `dw_b [exp_ch]`)
/// - `se_fc1_w`: `[se_ch × exp_ch]`  (SE squeeze, bias: `se_fc1_b [se_ch]`)
/// - `se_fc2_w`: `[exp_ch × se_ch]`  (SE excite,  bias: `se_fc2_b [exp_ch]`)
/// - `proj_w`  : `[out_ch × exp_ch]` (1×1 project conv, bias: `proj_b [out_ch]`)
pub struct MbConvBlock {
    expand_w: Vec<f32>,
    expand_b: Vec<f32>,
    dw_w: Vec<f32>,
    dw_b: Vec<f32>,
    se_fc1_w: Vec<f32>,
    se_fc1_b: Vec<f32>,
    se_fc2_w: Vec<f32>,
    se_fc2_b: Vec<f32>,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    config: MbConvConfig,
    has_skip: bool,
}

impl MbConvBlock {
    /// Construct a new MBConv block with Xavier-uniform weight initialization.
    ///
    /// # Errors
    ///
    /// - [`VisionError::InvalidImageSize`] if `in_channels` or `out_channels` is 0.
    /// - [`VisionError::InvalidEmbedDim`] if `expand_ratio` is 0.
    /// - [`VisionError::NonPositiveTemperature`] if `se_ratio ≤ 0`.
    pub fn new(config: MbConvConfig, rng: &mut VisionRng) -> VisionResult<Self> {
        if config.in_channels == 0 || config.out_channels == 0 {
            return Err(VisionError::InvalidImageSize {
                height: config.h,
                width: config.w,
                channels: config.in_channels,
            });
        }
        if config.expand_ratio == 0 {
            return Err(VisionError::InvalidEmbedDim(0));
        }
        if config.se_ratio <= 0.0 {
            return Err(VisionError::NonPositiveTemperature(config.se_ratio));
        }

        let in_ch = config.in_channels;
        let exp_ch = config.expanded_channels();
        let out_ch = config.out_channels;
        let k = config.kernel_size;
        let se_ch = config.se_channels();
        let has_skip = config.has_skip();

        let xavier = |fan_in: usize, fan_out: usize, rng: &mut VisionRng| -> Vec<f32> {
            let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
            let n = fan_out * fan_in;
            (0..n)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * limit)
                .collect()
        };

        // Expand: [exp_ch × in_ch]
        let expand_w = xavier(in_ch, exp_ch, rng);
        let expand_b = vec![0.0_f32; exp_ch];

        // Depthwise: [exp_ch × k × k]
        let dw_w = xavier(k * k, exp_ch, rng);
        let dw_b = vec![0.0_f32; exp_ch];

        // SE fc1: [se_ch × exp_ch]
        let se_fc1_w = xavier(exp_ch, se_ch, rng);
        let se_fc1_b = vec![0.0_f32; se_ch];

        // SE fc2: [exp_ch × se_ch]
        let se_fc2_w = xavier(se_ch, exp_ch, rng);
        let se_fc2_b = vec![0.0_f32; exp_ch];

        // Project: [out_ch × exp_ch]
        let proj_w = xavier(exp_ch, out_ch, rng);
        let proj_b = vec![0.0_f32; out_ch];

        Ok(Self {
            expand_w,
            expand_b,
            dw_w,
            dw_b,
            se_fc1_w,
            se_fc1_b,
            se_fc2_w,
            se_fc2_b,
            proj_w,
            proj_b,
            config,
            has_skip,
        })
    }

    /// Whether this block uses a skip (residual) connection.
    #[must_use]
    pub fn has_skip(&self) -> bool {
        self.has_skip
    }

    /// Forward pass.
    ///
    /// # Input / output layout
    ///
    /// `x`: `[batch_size × in_channels]` row-major.
    /// Returns: `[batch_size × out_channels]` row-major.
    ///
    /// ## Pipeline (per sample)
    ///
    /// 1. **Expand** : `h = ReLU6(W_exp · x + b_exp)`
    ///    output: `[exp_ch]`
    /// 2. **Depthwise** : channel-wise multiplication by `dw_w[c, :]` mean (1D proxy),
    ///    then bias + ReLU6.
    ///    Full 2-D depthwise conv would require H×W spatial input; here we use
    ///    `dw_w[c]` as a per-channel scale (mean of k×k filter weights).
    /// 3. **SE** : global pool → FC1+ReLU → FC2+sigmoid → broadcast multiply.
    /// 4. **Project** : `out = W_proj · h_se + b_proj`  (no activation).
    /// 5. **Skip** : if `has_skip`, `out += x`.
    ///
    /// # Errors
    ///
    /// - [`VisionError::DimensionMismatch`] if `x.len() != batch_size * in_channels`.
    pub fn forward(&self, x: &[f32], batch_size: usize) -> VisionResult<Vec<f32>> {
        let in_ch = self.config.in_channels;
        let exp_ch = self.config.expanded_channels();
        let out_ch = self.config.out_channels;
        let se_ch = self.config.se_channels();
        let k = self.config.kernel_size;

        if x.len() != batch_size * in_ch {
            return Err(VisionError::DimensionMismatch {
                expected: batch_size * in_ch,
                got: x.len(),
            });
        }

        let mut out = vec![0.0_f32; batch_size * out_ch];

        for b in 0..batch_size {
            let x_row = &x[b * in_ch..(b + 1) * in_ch];

            // ── 1. Expand ──────────────────────────────────────────────────
            let h_exp: Vec<f32> = (0..exp_ch)
                .map(|i| {
                    let acc = self.expand_b[i]
                        + x_row
                            .iter()
                            .enumerate()
                            .map(|(j, &xj)| self.expand_w[i * in_ch + j] * xj)
                            .sum::<f32>();
                    relu6(acc)
                })
                .collect();

            // ── 2. Depthwise (1-D proxy) ───────────────────────────────────
            // Per-channel scale = mean of the k×k filter weights.
            let h_dw: Vec<f32> = (0..exp_ch)
                .map(|c| {
                    let w_slice = &self.dw_w[c * k * k..(c + 1) * k * k];
                    let w_mean: f32 = w_slice.iter().sum::<f32>() / (k * k) as f32;
                    relu6(h_exp[c] * w_mean + self.dw_b[c])
                })
                .collect();

            // ── 3. Squeeze-Excite ──────────────────────────────────────────
            // 3a. Global avg pool (1-D: pool = h_dw directly).
            let pooled = &h_dw;

            // 3b. FC1: [se_ch × exp_ch], ReLU
            let se_h1: Vec<f32> = (0..se_ch)
                .map(|i| {
                    let acc = self.se_fc1_b[i]
                        + pooled
                            .iter()
                            .enumerate()
                            .map(|(j, &pj)| self.se_fc1_w[i * exp_ch + j] * pj)
                            .sum::<f32>();
                    acc.max(0.0)
                })
                .collect();

            // 3c. FC2: [exp_ch × se_ch], sigmoid
            let se_gate: Vec<f32> = (0..exp_ch)
                .map(|i| {
                    let acc = self.se_fc2_b[i]
                        + se_h1
                            .iter()
                            .enumerate()
                            .map(|(j, &sj)| self.se_fc2_w[i * se_ch + j] * sj)
                            .sum::<f32>();
                    sigmoid(acc)
                })
                .collect();

            // 3d. Channel-wise gate multiply.
            let h_se: Vec<f32> = h_dw
                .iter()
                .zip(se_gate.iter())
                .map(|(&hd, &sg)| hd * sg)
                .collect();

            // ── 4. Project ─────────────────────────────────────────────────
            let mut y: Vec<f32> = (0..out_ch)
                .map(|i| {
                    self.proj_b[i]
                        + h_se
                            .iter()
                            .enumerate()
                            .map(|(j, &hj)| self.proj_w[i * exp_ch + j] * hj)
                            .sum::<f32>()
                    // no activation
                })
                .collect();

            // ── 5. Skip connection ─────────────────────────────────────────
            if self.has_skip {
                for (yi, &xi) in y.iter_mut().zip(x_row.iter()) {
                    *yi += xi;
                }
            }

            out[b * out_ch..(b + 1) * out_ch].copy_from_slice(&y);
        }

        Ok(out)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn default_config() -> MbConvConfig {
        MbConvConfig {
            in_channels: 16,
            out_channels: 16,
            expand_ratio: 6,
            stride: 1,
            kernel_size: 3,
            se_ratio: 0.25,
            h: 8,
            w: 8,
        }
    }

    fn make_input(batch: usize, channels: usize, seed: u64) -> Vec<f32> {
        let mut r = LcgRng::new(seed);
        (0..batch * channels).map(|_| r.next_f32()).collect()
    }

    // 1 ─ output shape
    #[test]
    fn output_shape() {
        let cfg = default_config();
        let mut r = rng();
        let block = MbConvBlock::new(cfg, &mut r).expect("new should succeed");
        let x = make_input(4, 16, 1);
        let out = block.forward(&x, 4).expect("forward should succeed");
        assert_eq!(out.len(), 4 * 16);
    }

    // 2 ─ output finite
    #[test]
    fn output_finite() {
        let cfg = default_config();
        let mut r = rng();
        let block = MbConvBlock::new(cfg, &mut r).expect("new should succeed");
        let x = make_input(2, 16, 2);
        let out = block.forward(&x, 2).expect("forward should succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}] = {v}");
        }
    }

    // 3 ─ expand_ratio=1 works
    #[test]
    fn expand_ratio_1_works() {
        let cfg = MbConvConfig {
            in_channels: 8,
            out_channels: 8,
            expand_ratio: 1,
            stride: 1,
            kernel_size: 3,
            se_ratio: 0.25,
            h: 4,
            w: 4,
        };
        let mut r = rng();
        let block = MbConvBlock::new(cfg, &mut r).expect("new should succeed");
        let x = make_input(3, 8, 3);
        let out = block.forward(&x, 3).expect("forward should succeed");
        assert_eq!(out.len(), 3 * 8);
    }

    // 4 ─ has_skip correct (stride=1, same channels → skip)
    #[test]
    fn has_skip_correct_same_channels() {
        let cfg = default_config(); // stride=1, in=out=16
        let mut r = rng();
        let block = MbConvBlock::new(cfg, &mut r).expect("new should succeed");
        assert!(block.has_skip());
    }

    // 5 ─ no skip when in_channels != out_channels
    #[test]
    fn no_skip_different_channels() {
        let cfg = MbConvConfig {
            in_channels: 8,
            out_channels: 16,
            expand_ratio: 6,
            stride: 1,
            kernel_size: 3,
            se_ratio: 0.25,
            h: 4,
            w: 4,
        };
        let mut r = rng();
        let block = MbConvBlock::new(cfg, &mut r).expect("new should succeed");
        assert!(!block.has_skip());
    }

    // 6 ─ relu6 clamps at 6
    #[test]
    fn relu6_clamps_at_6() {
        assert!((relu6(10.0) - 6.0).abs() < 1e-7);
        assert!((relu6(-1.0) - 0.0).abs() < 1e-7);
        assert!((relu6(3.0) - 3.0).abs() < 1e-7);
    }

    // 7 ─ batch_size varies
    #[test]
    fn batch_size_varies() {
        let cfg = default_config();
        for &bs in &[1_usize, 2, 8] {
            let mut r = LcgRng::new(bs as u64);
            let block = MbConvBlock::new(cfg.clone(), &mut r).expect("value should be present");
            let x = make_input(bs, 16, bs as u64);
            let out = block.forward(&x, bs).expect("forward should succeed");
            assert_eq!(out.len(), bs * 16);
        }
    }

    // 8 ─ stride=2 config accepted (output shape unchanged in 1-D abstraction)
    #[test]
    fn stride_2_config_accepted() {
        let cfg = MbConvConfig {
            in_channels: 8,
            out_channels: 16,
            expand_ratio: 6,
            stride: 2,
            kernel_size: 5,
            se_ratio: 0.25,
            h: 8,
            w: 8,
        };
        let mut r = rng();
        let block = MbConvBlock::new(cfg, &mut r).expect("new should succeed");
        assert!(!block.has_skip()); // stride=2 → no skip
        let x = make_input(2, 8, 9);
        let out = block.forward(&x, 2).expect("forward should succeed");
        assert_eq!(out.len(), 2 * 16);
    }

    // 9 ─ expand_ratio=0 returns error
    #[test]
    fn expand_ratio_0_error() {
        let cfg = MbConvConfig {
            in_channels: 8,
            out_channels: 8,
            expand_ratio: 0,
            stride: 1,
            kernel_size: 3,
            se_ratio: 0.25,
            h: 4,
            w: 4,
        };
        let mut r = rng();
        let result = MbConvBlock::new(cfg, &mut r);
        assert!(result.is_err());
    }

    // 10 ─ se_ratio=0 returns error
    #[test]
    fn se_ratio_zero_error() {
        let cfg = MbConvConfig {
            in_channels: 8,
            out_channels: 8,
            expand_ratio: 6,
            stride: 1,
            kernel_size: 3,
            se_ratio: 0.0,
            h: 4,
            w: 4,
        };
        let mut r = rng();
        let result = MbConvBlock::new(cfg, &mut r);
        assert!(result.is_err());
    }

    // 11 ─ se_ratio affects param count (different se_ch)
    #[test]
    fn se_ratio_affects_se_channels() {
        let cfg1 = MbConvConfig {
            se_ratio: 0.25,
            ..default_config()
        };
        let cfg2 = MbConvConfig {
            se_ratio: 0.5,
            ..default_config()
        };
        assert!(cfg1.se_channels() < cfg2.se_channels());
    }

    // 12 ─ dimension mismatch error
    #[test]
    fn dimension_mismatch_error() {
        let cfg = default_config();
        let mut r = rng();
        let block = MbConvBlock::new(cfg, &mut r).expect("new should succeed");
        let wrong_x = vec![0.0_f32; 2 * 8]; // wrong: 8 channels not 16
        let result = block.forward(&wrong_x, 2);
        assert!(result.is_err());
    }
}

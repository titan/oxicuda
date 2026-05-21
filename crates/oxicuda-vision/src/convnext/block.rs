//! ConvNeXt block: depthwise 7×7 convolution + channel LayerNorm +
//! inverted bottleneck (1×1 expansion → GELU → 1×1 projection) + layer
//! scale + residual.
//!
//! Reference: Liu et al. 2022 CVPR, *"A ConvNet for the 2020s"*.
//!
//! ## Tensor layout
//! Activations use a channel-major row-major layout: `x` is
//! `(channels, height, width)` flattened so that channel `c`'s pixel `(h, w)`
//! lives at `c·H·W + h·W + w`.
//!
//! ## Forward pass
//! ```text
//! y = depthwise_conv(x)              (C,H,W) same-pad 7×7, per-channel
//! y = channel_layernorm(y)           LN across channels at each (h,w)
//! y = pointwise1(y)                  1×1 conv  C → expansion·C
//! y = GELU(y)
//! y = pointwise2(y)                  1×1 conv  expansion·C → C
//! y = layer_scale ⊙ y                per-channel scale γ
//! out = x + y                        residual
//! ```

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

/// LayerNorm epsilon for the channel-wise normalisation.
const LN_EPS: f32 = 1e-6;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a single ConvNeXt block.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvNextConfig {
    /// Number of feature channels `C` (input == output).
    pub channels: usize,
    /// Spatial height `H` in pixels.
    pub height: usize,
    /// Spatial width `W` in pixels.
    pub width: usize,
    /// Depthwise convolution kernel edge length (must be odd, e.g. 7).
    pub kernel: usize,
    /// Inverted-bottleneck expansion factor (hidden = `expansion · C`).
    pub expansion: usize,
    /// Initial value for the layer-scale gamma (e.g. `1e-6`).
    pub layer_scale_init: f32,
}

impl ConvNextConfig {
    /// Create and validate a `ConvNextConfig`.
    ///
    /// # Errors
    /// - `channels == 0` / `height == 0` / `width == 0` → `InvalidImageSize`
    /// - `kernel == 0` or `kernel` even → `InvalidPatchSize`
    /// - `expansion == 0` → `Internal`
    pub fn new(
        channels: usize,
        height: usize,
        width: usize,
        kernel: usize,
        expansion: usize,
        layer_scale_init: f32,
    ) -> VisionResult<Self> {
        if channels == 0 || height == 0 || width == 0 {
            return Err(VisionError::InvalidImageSize {
                height,
                width,
                channels,
            });
        }
        if kernel == 0 || kernel % 2 == 0 {
            return Err(VisionError::InvalidPatchSize {
                patch_size: kernel,
                img_size: height,
            });
        }
        if expansion == 0 {
            return Err(VisionError::Internal("expansion must be >= 1".to_string()));
        }
        Ok(Self {
            channels,
            height,
            width,
            kernel,
            expansion,
            layer_scale_init,
        })
    }

    /// Number of spatial pixels `H·W`.
    #[must_use]
    #[inline]
    pub fn spatial(&self) -> usize {
        self.height * self.width
    }

    /// Inverted-bottleneck hidden width `expansion · C`.
    #[must_use]
    #[inline]
    pub fn hidden(&self) -> usize {
        self.expansion * self.channels
    }

    /// Symmetric "same" zero padding `(kernel - 1) / 2`.
    #[must_use]
    #[inline]
    pub fn pad(&self) -> usize {
        (self.kernel - 1) / 2
    }
}

// ─── ConvNextBlock ─────────────────────────────────────────────────────────────

/// A single ConvNeXt block.
///
/// All weight tensors are flat row-major `Vec<f32>`.
pub struct ConvNextBlock {
    /// Depthwise kernel `[channels, kernel, kernel]`.
    dw_kernel: Vec<f32>,
    /// Depthwise bias `[channels]`.
    dw_bias: Vec<f32>,
    /// Channel-LayerNorm scale `[channels]` (init 1).
    ln_gamma: Vec<f32>,
    /// Channel-LayerNorm bias `[channels]` (init 0).
    ln_beta: Vec<f32>,
    /// Pointwise-1 kernel `[expansion·C, C]` (1×1 conv).
    pw1_weight: Vec<f32>,
    /// Pointwise-1 bias `[expansion·C]`.
    pw1_bias: Vec<f32>,
    /// Pointwise-2 kernel `[C, expansion·C]` (1×1 conv).
    pw2_weight: Vec<f32>,
    /// Pointwise-2 bias `[C]`.
    pw2_bias: Vec<f32>,
    /// Layer-scale gamma `[channels]` (init `layer_scale_init`).
    layer_scale: Vec<f32>,
    /// Block configuration.
    cfg: ConvNextConfig,
}

impl ConvNextBlock {
    /// Construct a new block with random depthwise / pointwise weights, zeroed
    /// depthwise bias, identity LayerNorm affine, and layer-scale gamma set to
    /// `cfg.layer_scale_init`.
    ///
    /// Kernels and pointwise weights use a Kaiming-ish scaled normal
    /// initialisation drawn from the deterministic LCG RNG.
    ///
    /// # Errors
    /// Propagates configuration validation from [`ConvNextConfig::new`].
    pub fn new(cfg: ConvNextConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let cfg = ConvNextConfig::new(
            cfg.channels,
            cfg.height,
            cfg.width,
            cfg.kernel,
            cfg.expansion,
            cfg.layer_scale_init,
        )?;
        let c = cfg.channels;
        let hidden = cfg.hidden();

        let fill_scaled = |rng: &mut LcgRng, n: usize, sc: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= sc;
            }
            v
        };

        // Depthwise: fan_in = kernel² (one channel per filter).
        let dw_fan_in = cfg.kernel * cfg.kernel;
        let dw_scale = (2.0 / dw_fan_in as f32).sqrt();
        let dw_kernel = fill_scaled(rng, c * dw_fan_in, dw_scale);
        let dw_bias = vec![0.0f32; c];

        let ln_gamma = vec![1.0f32; c];
        let ln_beta = vec![0.0f32; c];

        // Pointwise-1: fan_in = C.
        let pw1_scale = (2.0 / c as f32).sqrt();
        let pw1_weight = fill_scaled(rng, hidden * c, pw1_scale);
        let pw1_bias = vec![0.0f32; hidden];

        // Pointwise-2: fan_in = hidden.
        let pw2_scale = (2.0 / hidden as f32).sqrt();
        let pw2_weight = fill_scaled(rng, c * hidden, pw2_scale);
        let pw2_bias = vec![0.0f32; c];

        let layer_scale = vec![cfg.layer_scale_init; c];

        Ok(Self {
            dw_kernel,
            dw_bias,
            ln_gamma,
            ln_beta,
            pw1_weight,
            pw1_bias,
            pw2_weight,
            pw2_bias,
            layer_scale,
            cfg,
        })
    }

    /// Read-only access to the block configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &ConvNextConfig {
        &self.cfg
    }

    /// Mutable access to the depthwise kernel (used for tests that install an
    /// identity / delta kernel).
    #[inline]
    pub fn dw_kernel_mut(&mut self) -> &mut [f32] {
        &mut self.dw_kernel
    }

    /// Mutable access to the depthwise bias.
    #[inline]
    pub fn dw_bias_mut(&mut self) -> &mut [f32] {
        &mut self.dw_bias
    }

    /// Validate that `x` has the expected `C·H·W` flat length.
    fn check_input_len(&self, x: &[f32]) -> VisionResult<()> {
        let expected = self.cfg.channels * self.cfg.spatial();
        if x.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        Ok(())
    }

    /// Depthwise convolution: each channel `c` is convolved with its own
    /// `kernel × kernel` filter, zero-padded "same" (`pad = (k-1)/2`),
    /// stride 1, plus per-channel bias.
    ///
    /// Input / output are `(C, H, W)` flat.
    ///
    /// # Errors
    /// `DimensionMismatch` if `x.len() != C·H·W`.
    pub fn depthwise_conv(&self, x: &[f32]) -> VisionResult<Vec<f32>> {
        self.check_input_len(x)?;
        let c = self.cfg.channels;
        let h = self.cfg.height;
        let w = self.cfg.width;
        let k = self.cfg.kernel;
        let pad = self.cfg.pad();
        let hw = h * w;
        let k2 = k * k;

        let mut out = vec![0.0f32; c * hw];
        for ch in 0..c {
            let in_base = ch * hw;
            let ker_base = ch * k2;
            let bias = self.dw_bias[ch];
            for oh in 0..h {
                for ow in 0..w {
                    let mut acc = bias;
                    for ki in 0..k {
                        // Signed input row for this kernel tap.
                        let ih = oh as isize + ki as isize - pad as isize;
                        if ih < 0 || ih >= h as isize {
                            continue;
                        }
                        let ih = ih as usize;
                        for kj in 0..k {
                            let iw = ow as isize + kj as isize - pad as isize;
                            if iw < 0 || iw >= w as isize {
                                continue;
                            }
                            let iw = iw as usize;
                            acc +=
                                self.dw_kernel[ker_base + ki * k + kj] * x[in_base + ih * w + iw];
                        }
                    }
                    out[in_base + oh * w + ow] = acc;
                }
            }
        }
        Ok(out)
    }

    /// Channel LayerNorm: at each spatial position `(h, w)`, gather the `C`
    /// channel values, normalise to zero-mean / unit-variance (ε = `1e-6`),
    /// then apply per-channel affine `γ · x̂ + β`.
    ///
    /// Input / output are `(C, H, W)` flat.
    ///
    /// # Errors
    /// `DimensionMismatch` if `x.len() != C·H·W`.
    pub fn channel_layernorm(&self, x: &[f32]) -> VisionResult<Vec<f32>> {
        self.check_input_len(x)?;
        let c = self.cfg.channels;
        let hw = self.cfg.spatial();

        let mut out = vec![0.0f32; c * hw];
        for p in 0..hw {
            // Mean across channels at pixel p.
            let mut mean = 0.0f32;
            for ch in 0..c {
                mean += x[ch * hw + p];
            }
            mean /= c as f32;
            // Variance across channels.
            let mut var = 0.0f32;
            for ch in 0..c {
                let d = x[ch * hw + p] - mean;
                var += d * d;
            }
            var /= c as f32;
            let inv_std = 1.0 / (var + LN_EPS).sqrt();
            for ch in 0..c {
                let norm = (x[ch * hw + p] - mean) * inv_std;
                out[ch * hw + p] = norm * self.ln_gamma[ch] + self.ln_beta[ch];
            }
        }
        Ok(out)
    }

    /// Apply a 1×1 convolution `W·x + b` independently at every spatial
    /// position. `weight` is `[out_c, in_c]` row-major (one filter per output
    /// channel). Input is `(in_c, H, W)`, output is `(out_c, H, W)`.
    fn pointwise(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        in_c: usize,
        out_c: usize,
    ) -> Vec<f32> {
        let hw = self.cfg.spatial();
        let mut out = vec![0.0f32; out_c * hw];
        for p in 0..hw {
            for oc in 0..out_c {
                let wrow = &weight[oc * in_c..(oc + 1) * in_c];
                let mut acc = bias[oc];
                for ic in 0..in_c {
                    acc += wrow[ic] * x[ic * hw + p];
                }
                out[oc * hw + p] = acc;
            }
        }
        out
    }

    /// Forward pass: `(C·H·W) → (C·H·W)`.
    ///
    /// See the module docs for the full pipeline. When `layer_scale_init` is
    /// `0.0`, the residual branch is multiplied by zero so `forward(x) == x`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `x.len() != C·H·W`.
    /// - `NonFinite` if the output contains non-finite values.
    pub fn forward(&self, x: &[f32]) -> VisionResult<Vec<f32>> {
        self.check_input_len(x)?;
        let c = self.cfg.channels;
        let hidden = self.cfg.hidden();
        let hw = self.cfg.spatial();

        // Depthwise conv → channel LayerNorm.
        let y = self.depthwise_conv(x)?;
        let y = self.channel_layernorm(&y)?;

        // Inverted bottleneck: 1×1 expand → GELU → 1×1 project.
        let y = self.pointwise(&y, &self.pw1_weight, &self.pw1_bias, c, hidden);
        let y: Vec<f32> = y.into_iter().map(gelu).collect();
        let mut y = self.pointwise(&y, &self.pw2_weight, &self.pw2_bias, hidden, c);

        // Per-channel layer scale.
        for ch in 0..c {
            let gamma = self.layer_scale[ch];
            for p in 0..hw {
                y[ch * hw + p] *= gamma;
            }
        }

        // Residual.
        let out: Vec<f32> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        if out.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("convnext block output"));
        }
        Ok(out)
    }

    /// Total number of learnable parameters in this block.
    ///
    /// ```text
    /// dw_kernel  : C · k²
    /// dw_bias    : C
    /// ln_gamma   : C
    /// ln_beta    : C
    /// pw1_weight : hidden · C
    /// pw1_bias   : hidden
    /// pw2_weight : C · hidden
    /// pw2_bias   : C
    /// layer_scale: C
    /// ```
    #[must_use]
    pub fn n_params(&self) -> usize {
        let c = self.cfg.channels;
        let hidden = self.cfg.hidden();
        let k2 = self.cfg.kernel * self.cfg.kernel;
        c * k2  // dw_kernel
            + c // dw_bias
            + c // ln_gamma
            + c // ln_beta
            + hidden * c // pw1_weight
            + hidden // pw1_bias
            + c * hidden // pw2_weight
            + c // pw2_bias
            + c // layer_scale
    }
}

// ─── GELU ──────────────────────────────────────────────────────────────────────

/// GELU activation via the tanh approximation:
/// `0.5·v·(1 + tanh(√(2/π)·(v + 0.044715·v³)))`.
#[inline]
fn gelu(v: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
    0.5 * v * (1.0 + inner.tanh())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg() -> ConvNextConfig {
        // C=8, 6×6 spatial, 3×3 depthwise, 4× expansion, layer scale 1e-6.
        ConvNextConfig::new(8, 6, 6, 3, 4, 1e-6).expect("valid config")
    }

    fn random_input(cfg: &ConvNextConfig, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut x = vec![0.0f32; cfg.channels * cfg.spatial()];
        rng.fill_normal(&mut x);
        x
    }

    #[test]
    fn config_derived_quantities() {
        let cfg = make_cfg();
        assert_eq!(cfg.spatial(), 36);
        assert_eq!(cfg.hidden(), 32);
        assert_eq!(cfg.pad(), 1); // (3-1)/2
    }

    #[test]
    fn depthwise_conv_output_length() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(1);
        let block = ConvNextBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 2);
        let y = block.depthwise_conv(&x).expect("dw");
        assert_eq!(y.len(), cfg.channels * cfg.spatial());
    }

    #[test]
    fn depthwise_identity_kernel_is_input() {
        // Center-1 delta kernel + zero bias ⇒ depthwise conv is the identity.
        let cfg = make_cfg();
        let mut rng = LcgRng::new(3);
        let mut block = ConvNextBlock::new(cfg.clone(), &mut rng).expect("block");
        let k = cfg.kernel;
        let k2 = k * k;
        let center = (k / 2) * k + (k / 2);
        {
            let ker = block.dw_kernel_mut();
            for v in ker.iter_mut() {
                *v = 0.0;
            }
            for ch in 0..cfg.channels {
                ker[ch * k2 + center] = 1.0;
            }
        }
        for v in block.dw_bias_mut().iter_mut() {
            *v = 0.0;
        }
        let x = random_input(&cfg, 4);
        let y = block.depthwise_conv(&x).expect("dw");
        for (a, b) in y.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-5, "identity kernel mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn channel_layernorm_zero_mean_unit_var() {
        // With γ=1, β=0 (default init), per-pixel channel stats are ~N(0,1).
        let cfg = make_cfg();
        let mut rng = LcgRng::new(5);
        let block = ConvNextBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 6);
        let y = block.channel_layernorm(&x).expect("ln");
        let c = cfg.channels;
        let hw = cfg.spatial();
        for &p in &[0usize, 7, hw - 1] {
            let mut mean = 0.0f32;
            for ch in 0..c {
                mean += y[ch * hw + p];
            }
            mean /= c as f32;
            let mut var = 0.0f32;
            for ch in 0..c {
                let d = y[ch * hw + p] - mean;
                var += d * d;
            }
            var /= c as f32;
            assert!(mean.abs() < 1e-4, "pixel {p} mean not ~0: {mean}");
            assert!((var - 1.0).abs() < 1e-2, "pixel {p} var not ~1: {var}");
        }
    }

    #[test]
    fn forward_output_length() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(7);
        let block = ConvNextBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 8);
        let out = block.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.channels * cfg.spatial());
    }

    #[test]
    fn forward_finite() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(9);
        let block = ConvNextBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 10);
        let out = block.forward(&x).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn layer_scale_zero_makes_identity() {
        // layer_scale_init = 0 ⇒ residual branch zeroed ⇒ forward(x) == x.
        let cfg = ConvNextConfig::new(8, 6, 6, 3, 4, 0.0).expect("cfg");
        let mut rng = LcgRng::new(11);
        let block = ConvNextBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 12);
        let out = block.forward(&x).expect("forward");
        for (a, b) in out.iter().zip(x.iter()) {
            assert_eq!(a, b, "zero layer scale must be exact identity");
        }
    }

    #[test]
    fn n_params_formula_matches() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(13);
        let block = ConvNextBlock::new(cfg.clone(), &mut rng).expect("block");
        let c = cfg.channels;
        let hidden = cfg.hidden();
        let k2 = cfg.kernel * cfg.kernel;
        let expected = c * k2 + c + c + c + hidden * c + hidden + c * hidden + c + c;
        assert_eq!(block.n_params(), expected);
    }

    #[test]
    fn kernel_one_works() {
        // 1×1 depthwise (kernel=1, pad=0) is valid and behaves as a per-channel
        // scale-plus-bias.
        let cfg = ConvNextConfig::new(4, 5, 5, 1, 2, 1e-6).expect("cfg");
        assert_eq!(cfg.pad(), 0);
        let mut rng = LcgRng::new(14);
        let block = ConvNextBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 15);
        let y = block.depthwise_conv(&x).expect("dw");
        assert_eq!(y.len(), cfg.channels * cfg.spatial());
        let out = block.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.channels * cfg.spatial());
    }

    #[test]
    fn expansion_grows_param_count() {
        let mut rng = LcgRng::new(16);
        let cfg2 = ConvNextConfig::new(8, 4, 4, 3, 2, 1e-6).expect("cfg");
        let cfg4 = ConvNextConfig::new(8, 4, 4, 3, 4, 1e-6).expect("cfg");
        let b2 = ConvNextBlock::new(cfg2, &mut rng).expect("block");
        let b4 = ConvNextBlock::new(cfg4, &mut rng).expect("block");
        assert!(
            b4.n_params() > b2.n_params(),
            "more expansion must mean more params"
        );
    }

    #[test]
    fn gelu_zero_is_zero() {
        assert!(gelu(0.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_large_positive_approx_identity() {
        let v = 10.0f32;
        assert!((gelu(v) - v).abs() < 1e-3, "GELU({v}) = {}", gelu(v));
    }

    #[test]
    fn gelu_large_negative_approx_zero() {
        let v = -10.0f32;
        assert!(gelu(v).abs() < 1e-3, "GELU({v}) = {}", gelu(v));
    }

    #[test]
    fn err_channels_zero() {
        let r = ConvNextConfig::new(0, 6, 6, 3, 4, 1e-6);
        assert!(matches!(r, Err(VisionError::InvalidImageSize { .. })));
    }

    #[test]
    fn err_kernel_even() {
        let r = ConvNextConfig::new(8, 6, 6, 4, 4, 1e-6);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn err_kernel_zero() {
        let r = ConvNextConfig::new(8, 6, 6, 0, 4, 1e-6);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn err_expansion_zero() {
        let r = ConvNextConfig::new(8, 6, 6, 3, 0, 1e-6);
        assert!(matches!(r, Err(VisionError::Internal(_))));
    }

    #[test]
    fn err_height_zero() {
        let r = ConvNextConfig::new(8, 0, 6, 3, 4, 1e-6);
        assert!(matches!(r, Err(VisionError::InvalidImageSize { .. })));
    }

    #[test]
    fn err_forward_wrong_length() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(17);
        let block = ConvNextBlock::new(cfg, &mut rng).expect("block");
        let x = vec![0.0f32; 5]; // wrong
        let r = block.forward(&x);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_depthwise_wrong_length() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(18);
        let block = ConvNextBlock::new(cfg, &mut rng).expect("block");
        let x = vec![0.0f32; 3]; // wrong
        let r = block.depthwise_conv(&x);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn deterministic_given_seed() {
        let cfg = make_cfg();
        let mut rng_a = LcgRng::new(77);
        let mut rng_b = LcgRng::new(77);
        let block_a = ConvNextBlock::new(cfg.clone(), &mut rng_a).expect("block");
        let block_b = ConvNextBlock::new(cfg.clone(), &mut rng_b).expect("block");
        let x = random_input(&cfg, 78);
        let out_a = block_a.forward(&x).expect("forward");
        let out_b = block_b.forward(&x).expect("forward");
        assert_eq!(out_a, out_b, "same seed must give identical output");
    }
}

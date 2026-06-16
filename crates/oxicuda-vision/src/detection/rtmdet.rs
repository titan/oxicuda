//! RTMDet — a compact, faithful CPU reference of the one-stage detector from
//! Lyu et al. 2022, *"RTMDet: An Empirical Study of Designing Real-Time Object
//! Detectors"*.
//!
//! The model is built from three classic pieces, all implemented here with real
//! convolutional arithmetic (no shape-only stubs):
//!
//! 1. **CSPNeXt backbone** — a stem followed by a stack of stages. Each stage
//!    downsamples by a stride-2 convolution and then applies a **CSP layer**:
//!    the channels are *split* into two branches, one branch passes through a
//!    stack of **bottlenecks built from large-kernel depthwise convolutions**,
//!    and the two branches are *concatenated* and fused by a 1×1 convolution.
//!    Successive stages halve the spatial resolution, producing a multi-scale
//!    feature hierarchy.
//! 2. **PAFPN neck** — a Path-Aggregation Feature Pyramid Network performing a
//!    *top-down* pass (upsample coarse features and add to finer ones) followed
//!    by a *bottom-up* pass (downsample fine features and add to coarser ones),
//!    emitting one fused map per input scale with a uniform channel count.
//! 3. **Decoupled head** — a *shared* head with **separate classification and
//!    regression branches**, producing per-location class logits and 4 box
//!    regression values at every scale.
//!
//! It also provides the **SimOTA-lite** dynamic soft-label assignment cost
//! (`cost = cls_cost + λ · iou_cost`) used to match predictions to ground truth.
//!
//! ## Tensor layout
//! All feature maps use the channel-major [`FeatureMap`] layout
//! (`[channels, height, width]`, row-major).

use super::anchor_nms::iou;
use crate::{
    error::{VisionError, VisionResult},
    fpn::top_down::FeatureMap,
    handle::LcgRng,
};

// ─── Activations ───────────────────────────────────────────────────────────────

/// SiLU / swish activation applied in place: `x ← x · sigmoid(x)`.
fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= 1.0 / (1.0 + (-*v).exp());
    }
}

/// Numerically-stable softplus: `log(1 + exp(x))`.
#[inline]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-(x.abs())).exp().ln_1p()
}

// ─── Conv2d ────────────────────────────────────────────────────────────────────

/// A dense 2-D convolution (groups = 1) with `[c_out, c_in, k, k]` weights.
#[derive(Debug, Clone)]
pub struct Conv2d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    c_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl Conv2d {
    /// He-initialised convolution.
    fn new(
        c_in: usize,
        c_out: usize,
        k: usize,
        stride: usize,
        pad: usize,
        rng: &mut LcgRng,
    ) -> Self {
        let fan_in = c_in * k * k;
        let scale = (2.0 / fan_in as f32).sqrt();
        let mut weight = vec![0.0f32; c_out * fan_in];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= scale;
        }
        Self {
            weight,
            bias: vec![0.0f32; c_out],
            c_in,
            c_out,
            k,
            stride,
            pad,
        }
    }

    /// Output channel count.
    #[must_use]
    #[inline]
    pub fn out_channels(&self) -> usize {
        self.c_out
    }

    /// Forward pass over a [`FeatureMap`].
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `x.channels != c_in`.
    /// - [`VisionError::InvalidImageSize`] if the kernel does not fit the
    ///   padded input.
    pub fn forward(&self, x: &FeatureMap) -> VisionResult<FeatureMap> {
        if x.channels != self.c_in {
            return Err(VisionError::DimensionMismatch {
                expected: self.c_in,
                got: x.channels,
            });
        }
        let (h, w) = (x.height, x.width);
        if h + 2 * self.pad < self.k || w + 2 * self.pad < self.k {
            return Err(VisionError::InvalidImageSize {
                height: h,
                width: w,
                channels: x.channels,
            });
        }
        let h_out = (h + 2 * self.pad - self.k) / self.stride + 1;
        let w_out = (w + 2 * self.pad - self.k) / self.stride + 1;
        let mut out = vec![0.0f32; self.c_out * h_out * w_out];
        let k = self.k;
        for oc in 0..self.c_out {
            let oc_w = oc * self.c_in * k * k;
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = self.bias[oc];
                    for ic in 0..self.c_in {
                        let in_base = ic * h * w;
                        let w_base = oc_w + ic * k * k;
                        for ki in 0..k {
                            let ih = oh * self.stride + ki;
                            if ih < self.pad || ih >= h + self.pad {
                                continue;
                            }
                            let ih = ih - self.pad;
                            for kj in 0..k {
                                let iw = ow * self.stride + kj;
                                if iw < self.pad || iw >= w + self.pad {
                                    continue;
                                }
                                let iw = iw - self.pad;
                                acc += self.weight[w_base + ki * k + kj]
                                    * out_in(x, in_base, ih, w, iw);
                            }
                        }
                    }
                    out[(oc * h_out + oh) * w_out + ow] = acc;
                }
            }
        }
        FeatureMap::new(out, self.c_out, h_out, w_out)
    }
}

/// Read `x.data[in_base + ih*w + iw]`. Helper to keep the inner conv loop tidy.
#[inline]
fn out_in(x: &FeatureMap, in_base: usize, ih: usize, w: usize, iw: usize) -> f32 {
    x.data[in_base + ih * w + iw]
}

// ─── DwConv2d ──────────────────────────────────────────────────────────────────

/// A depthwise 2-D convolution (groups = channels) with `[c, k, k]` weights.
///
/// Used for the **large-kernel** depthwise convolutions inside the CSPNeXt
/// bottleneck.
#[derive(Debug, Clone)]
pub struct DwConv2d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    c: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl DwConv2d {
    fn new(c: usize, k: usize, stride: usize, pad: usize, rng: &mut LcgRng) -> Self {
        let fan_in = k * k;
        let scale = (2.0 / fan_in as f32).sqrt();
        let mut weight = vec![0.0f32; c * fan_in];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= scale;
        }
        Self {
            weight,
            bias: vec![0.0f32; c],
            c,
            k,
            stride,
            pad,
        }
    }

    /// Depthwise forward pass.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `x.channels != c`.
    /// - [`VisionError::InvalidImageSize`] if the kernel does not fit.
    pub fn forward(&self, x: &FeatureMap) -> VisionResult<FeatureMap> {
        if x.channels != self.c {
            return Err(VisionError::DimensionMismatch {
                expected: self.c,
                got: x.channels,
            });
        }
        let (h, w) = (x.height, x.width);
        if h + 2 * self.pad < self.k || w + 2 * self.pad < self.k {
            return Err(VisionError::InvalidImageSize {
                height: h,
                width: w,
                channels: x.channels,
            });
        }
        let h_out = (h + 2 * self.pad - self.k) / self.stride + 1;
        let w_out = (w + 2 * self.pad - self.k) / self.stride + 1;
        let k = self.k;
        let mut out = vec![0.0f32; self.c * h_out * w_out];
        for ch in 0..self.c {
            let in_base = ch * h * w;
            let w_base = ch * k * k;
            let bias = self.bias[ch];
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = bias;
                    for ki in 0..k {
                        let ih = oh * self.stride + ki;
                        if ih < self.pad || ih >= h + self.pad {
                            continue;
                        }
                        let ih = ih - self.pad;
                        for kj in 0..k {
                            let iw = ow * self.stride + kj;
                            if iw < self.pad || iw >= w + self.pad {
                                continue;
                            }
                            let iw = iw - self.pad;
                            acc +=
                                self.weight[w_base + ki * k + kj] * x.data[in_base + ih * w + iw];
                        }
                    }
                    out[(ch * h_out + oh) * w_out + ow] = acc;
                }
            }
        }
        FeatureMap::new(out, self.c, h_out, w_out)
    }
}

// ─── Feature-map helpers ───────────────────────────────────────────────────────

/// Concatenate two feature maps along the channel axis (same spatial size).
fn concat_channels(a: &FeatureMap, b: &FeatureMap) -> VisionResult<FeatureMap> {
    if a.height != b.height || a.width != b.width {
        return Err(VisionError::ShapeMismatch {
            lhs: vec![a.channels, a.height, a.width],
            rhs: vec![b.channels, b.height, b.width],
        });
    }
    let mut data = Vec::with_capacity(a.data.len() + b.data.len());
    data.extend_from_slice(&a.data);
    data.extend_from_slice(&b.data);
    Ok(FeatureMap {
        data,
        channels: a.channels + b.channels,
        height: a.height,
        width: a.width,
    })
}

/// Element-wise `dst += src` (shapes must match exactly).
fn add_inplace(dst: &mut FeatureMap, src: &FeatureMap) -> VisionResult<()> {
    if dst.channels != src.channels || dst.height != src.height || dst.width != src.width {
        return Err(VisionError::ShapeMismatch {
            lhs: vec![dst.channels, dst.height, dst.width],
            rhs: vec![src.channels, src.height, src.width],
        });
    }
    for (a, b) in dst.data.iter_mut().zip(src.data.iter()) {
        *a += *b;
    }
    Ok(())
}

/// Nearest-neighbour 2× upsampling.
fn upsample2x(x: &FeatureMap) -> FeatureMap {
    let (c, h, w) = (x.channels, x.height, x.width);
    let (h2, w2) = (h * 2, w * 2);
    let mut out = vec![0.0f32; c * h2 * w2];
    for ch in 0..c {
        for i in 0..h {
            for j in 0..w {
                let v = x.data[(ch * h + i) * w + j];
                let oi = i * 2;
                let oj = j * 2;
                out[(ch * h2 + oi) * w2 + oj] = v;
                out[(ch * h2 + oi) * w2 + oj + 1] = v;
                out[(ch * h2 + oi + 1) * w2 + oj] = v;
                out[(ch * h2 + oi + 1) * w2 + oj + 1] = v;
            }
        }
    }
    FeatureMap {
        data: out,
        channels: c,
        height: h2,
        width: w2,
    }
}

// ─── Bottleneck ────────────────────────────────────────────────────────────────

/// CSPNeXt bottleneck: large-kernel depthwise conv → 1×1 pointwise conv, with a
/// residual connection. Both convolutions are followed by SiLU.
pub struct Bottleneck {
    dw: DwConv2d,
    pw: Conv2d,
}

impl Bottleneck {
    fn new(channels: usize, dw_kernel: usize, rng: &mut LcgRng) -> Self {
        let pad = (dw_kernel - 1) / 2;
        Self {
            dw: DwConv2d::new(channels, dw_kernel, 1, pad, rng),
            pw: Conv2d::new(channels, channels, 1, 1, 0, rng),
        }
    }

    fn forward(&self, x: &FeatureMap) -> VisionResult<FeatureMap> {
        let mut y = self.dw.forward(x)?;
        silu_inplace(&mut y.data);
        let mut y = self.pw.forward(&y)?;
        silu_inplace(&mut y.data);
        add_inplace(&mut y, x)?; // residual
        Ok(y)
    }
}

// ─── CSP layer ─────────────────────────────────────────────────────────────────

/// A Cross-Stage-Partial layer: split into a "main" and a "short" branch, run a
/// stack of bottlenecks on the main branch, concatenate, and fuse with a 1×1
/// convolution. `out_channels` must be even (`mid = out_channels / 2`).
pub struct CspLayer {
    main_conv: Conv2d,
    short_conv: Conv2d,
    blocks: Vec<Bottleneck>,
    final_conv: Conv2d,
}

impl CspLayer {
    fn new(
        in_channels: usize,
        out_channels: usize,
        n_blocks: usize,
        dw_kernel: usize,
        rng: &mut LcgRng,
    ) -> Self {
        let mid = out_channels / 2;
        let main_conv = Conv2d::new(in_channels, mid, 1, 1, 0, rng);
        let short_conv = Conv2d::new(in_channels, mid, 1, 1, 0, rng);
        let blocks = (0..n_blocks)
            .map(|_| Bottleneck::new(mid, dw_kernel, rng))
            .collect();
        // After concat: 2 * mid = out_channels.
        let final_conv = Conv2d::new(2 * mid, out_channels, 1, 1, 0, rng);
        Self {
            main_conv,
            short_conv,
            blocks,
            final_conv,
        }
    }

    fn forward(&self, x: &FeatureMap) -> VisionResult<FeatureMap> {
        let mut short = self.short_conv.forward(x)?;
        silu_inplace(&mut short.data);
        let mut main = self.main_conv.forward(x)?;
        silu_inplace(&mut main.data);
        for b in &self.blocks {
            main = b.forward(&main)?;
        }
        let cat = concat_channels(&main, &short)?;
        let mut out = self.final_conv.forward(&cat)?;
        silu_inplace(&mut out.data);
        Ok(out)
    }
}

// ─── Backbone ──────────────────────────────────────────────────────────────────

struct BackboneStage {
    downsample: Conv2d,
    csp: CspLayer,
}

/// CSPNeXt-style backbone: a stride-2 stem followed by `stage_channels.len()`
/// downsampling stages, each emitting one multi-scale feature map.
pub struct CspNeXtBackbone {
    stem: Conv2d,
    stages: Vec<BackboneStage>,
}

impl CspNeXtBackbone {
    fn new(cfg: &RtmDetConfig, rng: &mut LcgRng) -> Self {
        let stem = Conv2d::new(cfg.in_chans, cfg.stem_channels, 3, 2, 1, rng);
        let mut stages = Vec::with_capacity(cfg.stage_channels.len());
        let mut prev = cfg.stem_channels;
        for &c in &cfg.stage_channels {
            let downsample = Conv2d::new(prev, c, 3, 2, 1, rng);
            let csp = CspLayer::new(c, c, cfg.n_bottlenecks, cfg.dw_kernel, rng);
            stages.push(BackboneStage { downsample, csp });
            prev = c;
        }
        Self { stem, stages }
    }

    /// Forward pass returning one [`FeatureMap`] per stage (finest first).
    ///
    /// # Errors
    /// Propagates convolution shape errors.
    pub fn forward(&self, image: &FeatureMap) -> VisionResult<Vec<FeatureMap>> {
        let mut x = self.stem.forward(image)?;
        silu_inplace(&mut x.data);
        let mut feats = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let mut d = stage.downsample.forward(&x)?;
            silu_inplace(&mut d.data);
            x = stage.csp.forward(&d)?;
            feats.push(x.clone());
        }
        Ok(feats)
    }
}

// ─── PAFPN neck ────────────────────────────────────────────────────────────────

/// Path-Aggregation Feature Pyramid Network.
pub struct Pafpn {
    lateral: Vec<Conv2d>,
    top_down: Vec<Conv2d>,
    downsample: Vec<Conv2d>,
    bottom_up: Vec<Conv2d>,
    n_levels: usize,
}

impl Pafpn {
    fn new(in_channels: &[usize], out_channels: usize, rng: &mut LcgRng) -> Self {
        let n_levels = in_channels.len();
        let lateral = in_channels
            .iter()
            .map(|&c| Conv2d::new(c, out_channels, 1, 1, 0, rng))
            .collect();
        let top_down = (0..n_levels)
            .map(|_| Conv2d::new(out_channels, out_channels, 3, 1, 1, rng))
            .collect();
        let downsample = (0..n_levels.saturating_sub(1))
            .map(|_| Conv2d::new(out_channels, out_channels, 3, 2, 1, rng))
            .collect();
        let bottom_up = (0..n_levels.saturating_sub(1))
            .map(|_| Conv2d::new(out_channels, out_channels, 3, 1, 1, rng))
            .collect();
        Self {
            lateral,
            top_down,
            downsample,
            bottom_up,
            n_levels,
        }
    }

    /// Forward pass. `feats` must be ordered finest→coarsest and contain
    /// `n_levels` maps whose spatial sizes halve between adjacent levels.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `feats.len() != n_levels`.
    /// - [`VisionError::ShapeMismatch`] if adjacent levels are not in a 2:1
    ///   spatial ratio.
    pub fn forward(&self, feats: Vec<FeatureMap>) -> VisionResult<Vec<FeatureMap>> {
        if feats.len() != self.n_levels {
            return Err(VisionError::DimensionMismatch {
                expected: self.n_levels,
                got: feats.len(),
            });
        }
        let l = self.n_levels;

        // Lateral 1×1 to unify channels.
        let mut lat: Vec<FeatureMap> = Vec::with_capacity(l);
        for (f, conv) in feats.iter().zip(self.lateral.iter()) {
            lat.push(conv.forward(f)?);
        }

        // Top-down: coarse → fine.
        for level in (0..l.saturating_sub(1)).rev() {
            let up = upsample2x(&lat[level + 1]);
            add_inplace(&mut lat[level], &up)?;
            let mut fused = self.top_down[level].forward(&lat[level])?;
            silu_inplace(&mut fused.data);
            lat[level] = fused;
        }
        // Smooth the coarsest level too for symmetry.
        if l > 0 {
            let mut fused = self.top_down[l - 1].forward(&lat[l - 1])?;
            silu_inplace(&mut fused.data);
            lat[l - 1] = fused;
        }

        // Bottom-up: fine → coarse.
        let mut outs: Vec<FeatureMap> = Vec::with_capacity(l);
        outs.push(lat[0].clone());
        for level in 1..l {
            let mut down = self.downsample[level - 1].forward(&outs[level - 1])?;
            silu_inplace(&mut down.data);
            let mut merged = lat[level].clone();
            add_inplace(&mut merged, &down)?;
            let mut fused = self.bottom_up[level - 1].forward(&merged)?;
            silu_inplace(&mut fused.data);
            outs.push(fused);
        }
        Ok(outs)
    }
}

// ─── Decoupled head ────────────────────────────────────────────────────────────

/// Shared decoupled head with independent classification and regression
/// branches.
pub struct DecoupledHead {
    cls_conv: Conv2d,
    cls_pred: Conv2d,
    reg_conv: Conv2d,
    reg_pred: Conv2d,
}

impl DecoupledHead {
    fn new(channels: usize, n_classes: usize, rng: &mut LcgRng) -> Self {
        Self {
            cls_conv: Conv2d::new(channels, channels, 3, 1, 1, rng),
            cls_pred: Conv2d::new(channels, n_classes, 1, 1, 0, rng),
            reg_conv: Conv2d::new(channels, channels, 3, 1, 1, rng),
            reg_pred: Conv2d::new(channels, 4, 1, 1, 0, rng),
        }
    }

    /// Apply the head to one feature map, returning `(cls_logits, reg)` where
    /// `cls_logits` has `n_classes` channels and `reg` has 4 channels.
    ///
    /// # Errors
    /// Propagates convolution shape errors.
    pub fn forward_level(&self, x: &FeatureMap) -> VisionResult<(FeatureMap, FeatureMap)> {
        let mut c = self.cls_conv.forward(x)?;
        silu_inplace(&mut c.data);
        let cls = self.cls_pred.forward(&c)?;

        let mut r = self.reg_conv.forward(x)?;
        silu_inplace(&mut r.data);
        let reg = self.reg_pred.forward(&r)?;
        Ok((cls, reg))
    }
}

// ─── Config ────────────────────────────────────────────────────────────────────

/// RTMDet hyper-parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct RtmDetConfig {
    /// Input image channels (e.g. 3).
    pub in_chans: usize,
    /// Square input spatial size.
    pub img_size: usize,
    /// Stem output channels.
    pub stem_channels: usize,
    /// Output channels of each backbone stage (each value must be even).
    pub stage_channels: Vec<usize>,
    /// Bottlenecks per CSP layer.
    pub n_bottlenecks: usize,
    /// Large depthwise kernel size (odd).
    pub dw_kernel: usize,
    /// Uniform PAFPN / head channel count.
    pub neck_channels: usize,
    /// Number of object classes.
    pub n_classes: usize,
}

impl RtmDetConfig {
    /// Create and validate a configuration.
    ///
    /// # Errors
    /// - [`VisionError::InvalidImageSize`] if `in_chans == 0` or `img_size == 0`.
    /// - [`VisionError::EmptyInput`] if `stage_channels` is empty.
    /// - [`VisionError::InvalidNumClasses`] if `n_classes == 0`.
    /// - [`VisionError::InvalidPatchSize`] if `dw_kernel` is even or zero.
    /// - [`VisionError::DimensionMismatch`] if any stage channel count is odd
    ///   or zero, or `neck_channels`/`stem_channels` is zero.
    pub fn new(
        in_chans: usize,
        img_size: usize,
        stem_channels: usize,
        stage_channels: Vec<usize>,
        n_bottlenecks: usize,
        dw_kernel: usize,
        neck_channels: usize,
        n_classes: usize,
    ) -> VisionResult<Self> {
        if in_chans == 0 || img_size == 0 {
            return Err(VisionError::InvalidImageSize {
                height: img_size,
                width: img_size,
                channels: in_chans,
            });
        }
        if stage_channels.is_empty() {
            return Err(VisionError::EmptyInput("rtmdet stage_channels"));
        }
        if n_classes == 0 {
            return Err(VisionError::InvalidNumClasses(n_classes));
        }
        if dw_kernel == 0 || dw_kernel % 2 == 0 {
            return Err(VisionError::InvalidPatchSize {
                patch_size: dw_kernel,
                img_size,
            });
        }
        if stem_channels == 0 || neck_channels == 0 {
            return Err(VisionError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        for &c in &stage_channels {
            if c == 0 || c % 2 != 0 {
                return Err(VisionError::DimensionMismatch {
                    expected: 2,
                    got: c,
                });
            }
        }
        Ok(Self {
            in_chans,
            img_size,
            stem_channels,
            stage_channels,
            n_bottlenecks,
            dw_kernel,
            neck_channels,
            n_classes,
        })
    }

    /// A tiny configuration for unit tests:
    /// 3×32×32 input, stem 8, stages `[8, 16, 16]`, 1 bottleneck, 5×5 depthwise,
    /// neck 8, 4 classes.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            in_chans: 3,
            img_size: 32,
            stem_channels: 8,
            stage_channels: vec![8, 16, 16],
            n_bottlenecks: 1,
            dw_kernel: 5,
            neck_channels: 8,
            n_classes: 4,
        }
    }

    /// Number of pyramid levels (= number of backbone stages).
    #[must_use]
    #[inline]
    pub fn n_levels(&self) -> usize {
        self.stage_channels.len()
    }
}

// ─── Output ────────────────────────────────────────────────────────────────────

/// Multi-scale detector output.
#[derive(Debug, Clone)]
pub struct RtmDetOutput {
    /// Per-level class logits, each `[n_classes, H, W]`.
    pub cls_scores: Vec<FeatureMap>,
    /// Per-level raw box regression, each `[4, H, W]`.
    pub bbox_preds: Vec<FeatureMap>,
    /// Per-level stride (image pixels / feature pixel).
    pub strides: Vec<usize>,
}

// ─── RtmDet ────────────────────────────────────────────────────────────────────

/// The full RTMDet detector.
pub struct RtmDet {
    cfg: RtmDetConfig,
    backbone: CspNeXtBackbone,
    neck: Pafpn,
    head: DecoupledHead,
}

impl RtmDet {
    /// Build the detector with randomly-initialised weights.
    ///
    /// # Errors
    /// Propagates configuration validation.
    pub fn new(cfg: RtmDetConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let cfg = RtmDetConfig::new(
            cfg.in_chans,
            cfg.img_size,
            cfg.stem_channels,
            cfg.stage_channels.clone(),
            cfg.n_bottlenecks,
            cfg.dw_kernel,
            cfg.neck_channels,
            cfg.n_classes,
        )?;
        let backbone = CspNeXtBackbone::new(&cfg, rng);
        let neck = Pafpn::new(&cfg.stage_channels, cfg.neck_channels, rng);
        let head = DecoupledHead::new(cfg.neck_channels, cfg.n_classes, rng);
        Ok(Self {
            cfg,
            backbone,
            neck,
            head,
        })
    }

    /// Read-only access to the configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &RtmDetConfig {
        &self.cfg
    }

    /// Run only the backbone, returning the multi-scale feature hierarchy.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `image.len() != in_chans·H·W`.
    pub fn backbone_features(&self, image: &[f32]) -> VisionResult<Vec<FeatureMap>> {
        let img = self.make_image(image)?;
        self.backbone.forward(&img)
    }

    /// Run backbone + neck, returning the fused pyramid.
    ///
    /// # Errors
    /// Propagates backbone / neck errors.
    pub fn neck_features(&self, image: &[f32]) -> VisionResult<Vec<FeatureMap>> {
        let feats = self.backbone_features(image)?;
        self.neck.forward(feats)
    }

    /// Full forward pass producing per-level class + box predictions.
    ///
    /// # Errors
    /// Propagates backbone / neck / head errors, or [`VisionError::NonFinite`]
    /// if any prediction is non-finite.
    pub fn forward(&self, image: &[f32]) -> VisionResult<RtmDetOutput> {
        let neck = self.neck_features(image)?;
        let mut cls_scores = Vec::with_capacity(neck.len());
        let mut bbox_preds = Vec::with_capacity(neck.len());
        let mut strides = Vec::with_capacity(neck.len());
        for level in &neck {
            let (cls, reg) = self.head.forward_level(level)?;
            if cls
                .data
                .iter()
                .chain(reg.data.iter())
                .any(|v| !v.is_finite())
            {
                return Err(VisionError::NonFinite("rtmdet head output"));
            }
            strides.push(self.cfg.img_size / level.height.max(1));
            cls_scores.push(cls);
            bbox_preds.push(reg);
        }
        Ok(RtmDetOutput {
            cls_scores,
            bbox_preds,
            strides,
        })
    }

    fn make_image(&self, image: &[f32]) -> VisionResult<FeatureMap> {
        FeatureMap::new(
            image.to_vec(),
            self.cfg.in_chans,
            self.cfg.img_size,
            self.cfg.img_size,
        )
    }
}

// ─── Box decoding ──────────────────────────────────────────────────────────────

/// Decode one level's raw predictions into image-space detections.
///
/// Each location `(i, j)` is treated as an anchor point at the cell centre
/// `((j+0.5)·stride, (i+0.5)·stride)`. The 4 regression channels are decoded as
/// non-negative left/top/right/bottom distances (via softplus, scaled by the
/// stride) to give an axis-aligned box `[x1, y1, x2, y2]`. The score is the
/// maximum per-class sigmoid probability and the label its arg-max.
///
/// Returns `(boxes [n·4], scores [n], labels [n])` with `n = H · W`.
///
/// # Errors
/// - [`VisionError::DimensionMismatch`] if `cls.width != reg.width` /
///   `cls.height != reg.height` or `reg.channels != 4`.
pub fn decode_level(
    cls: &FeatureMap,
    reg: &FeatureMap,
    stride: usize,
) -> VisionResult<(Vec<f32>, Vec<f32>, Vec<usize>)> {
    if cls.height != reg.height || cls.width != reg.width {
        return Err(VisionError::ShapeMismatch {
            lhs: vec![cls.channels, cls.height, cls.width],
            rhs: vec![reg.channels, reg.height, reg.width],
        });
    }
    if reg.channels != 4 {
        return Err(VisionError::DimensionMismatch {
            expected: 4,
            got: reg.channels,
        });
    }
    let (h, w, n_cls) = (cls.height, cls.width, cls.channels);
    let s = stride as f32;
    let n = h * w;
    let mut boxes = vec![0.0f32; n * 4];
    let mut scores = vec![0.0f32; n];
    let mut labels = vec![0usize; n];
    for i in 0..h {
        for j in 0..w {
            let loc = i * w + j;
            let cx = (j as f32 + 0.5) * s;
            let cy = (i as f32 + 0.5) * s;
            let l = softplus(reg.at(0, i, j)) * s;
            let t = softplus(reg.at(1, i, j)) * s;
            let r = softplus(reg.at(2, i, j)) * s;
            let b = softplus(reg.at(3, i, j)) * s;
            boxes[loc * 4] = cx - l;
            boxes[loc * 4 + 1] = cy - t;
            boxes[loc * 4 + 2] = cx + r;
            boxes[loc * 4 + 3] = cy + b;

            let mut best = f32::NEG_INFINITY;
            let mut best_c = 0usize;
            for c in 0..n_cls {
                let p = 1.0 / (1.0 + (-cls.at(c, i, j)).exp());
                if p > best {
                    best = p;
                    best_c = c;
                }
            }
            scores[loc] = best;
            labels[loc] = best_c;
        }
    }
    Ok((boxes, scores, labels))
}

// ─── SimOTA-lite cost ──────────────────────────────────────────────────────────

/// SimOTA-lite dynamic soft-label assignment cost.
///
/// For every (ground-truth, prediction) pair the cost is
///
/// ```text
/// cost = cls_cost + λ · iou_cost
/// cls_cost = −log( p_pred(gt_class) )
/// iou_cost = −log( IoU(pred_box, gt_box) )
/// ```
///
/// A prediction that is *confident in the correct class* and *well-localised*
/// (high IoU) therefore receives a **low** cost — exactly the property exploited
/// by the dynamic assignment.
///
/// # Parameters
/// - `pred_cls`: `[n_pred · n_classes]` per-class **probabilities** in `[0, 1]`.
/// - `pred_boxes`: `[n_pred · 4]` predicted boxes `[x1, y1, x2, y2]`.
/// - `gt_labels`: `[n_gt]` ground-truth class indices.
/// - `gt_boxes`: `[n_gt · 4]` ground-truth boxes `[x1, y1, x2, y2]`.
/// - `n_classes`: number of classes.
/// - `lambda_iou`: weight on the IoU cost term.
///
/// # Returns
/// Flat `[n_gt · n_pred]` cost matrix (row-major, `cost[g · n_pred + p]`).
///
/// # Errors
/// - [`VisionError::EmptyInput`] if there are no predictions or no targets.
/// - [`VisionError::DimensionMismatch`] on inconsistent input lengths or an
///   out-of-range ground-truth label.
/// - [`VisionError::NonFinite`] if `lambda_iou` is not finite or a cost is
///   non-finite.
pub fn simota_cost(
    pred_cls: &[f32],
    pred_boxes: &[f32],
    gt_labels: &[usize],
    gt_boxes: &[f32],
    n_classes: usize,
    lambda_iou: f32,
) -> VisionResult<Vec<f32>> {
    if n_classes == 0 {
        return Err(VisionError::InvalidNumClasses(n_classes));
    }
    if !lambda_iou.is_finite() {
        return Err(VisionError::NonFinite("simota lambda_iou"));
    }
    let n_pred = pred_boxes.len() / 4;
    let n_gt = gt_labels.len();
    if n_pred == 0 {
        return Err(VisionError::EmptyInput("simota predictions"));
    }
    if n_gt == 0 {
        return Err(VisionError::EmptyInput("simota targets"));
    }
    if pred_boxes.len() != n_pred * 4 {
        return Err(VisionError::DimensionMismatch {
            expected: n_pred * 4,
            got: pred_boxes.len(),
        });
    }
    if pred_cls.len() != n_pred * n_classes {
        return Err(VisionError::DimensionMismatch {
            expected: n_pred * n_classes,
            got: pred_cls.len(),
        });
    }
    if gt_boxes.len() != n_gt * 4 {
        return Err(VisionError::DimensionMismatch {
            expected: n_gt * 4,
            got: gt_boxes.len(),
        });
    }

    const EPS: f32 = 1e-7;
    let mut cost = vec![0.0f32; n_gt * n_pred];
    for g in 0..n_gt {
        let cls = gt_labels[g];
        if cls >= n_classes {
            return Err(VisionError::DimensionMismatch {
                expected: n_classes,
                got: cls,
            });
        }
        let gbox = [
            gt_boxes[g * 4],
            gt_boxes[g * 4 + 1],
            gt_boxes[g * 4 + 2],
            gt_boxes[g * 4 + 3],
        ];
        for p in 0..n_pred {
            let prob = pred_cls[p * n_classes + cls].clamp(EPS, 1.0);
            let cls_cost = -prob.ln();
            let pbox = [
                pred_boxes[p * 4],
                pred_boxes[p * 4 + 1],
                pred_boxes[p * 4 + 2],
                pred_boxes[p * 4 + 3],
            ];
            let iou_val = iou(&pbox, &gbox);
            let iou_cost = -(iou_val + EPS).ln();
            cost[g * n_pred + p] = cls_cost + lambda_iou * iou_cost;
        }
    }
    if cost.iter().any(|v| !v.is_finite()) {
        return Err(VisionError::NonFinite("simota cost"));
    }
    Ok(cost)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn random_image(cfg: &RtmDetConfig, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut img = vec![0.0f32; cfg.in_chans * cfg.img_size * cfg.img_size];
        rng.fill_normal(&mut img);
        img
    }

    // ── Config validation ─────────────────────────────────────────────────────

    #[test]
    fn config_tiny_valid() {
        let cfg = RtmDetConfig::tiny();
        assert_eq!(cfg.n_levels(), 3);
    }

    #[test]
    fn config_odd_stage_channel_errors() {
        let r = RtmDetConfig::new(3, 32, 8, vec![8, 15], 1, 5, 8, 4);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn config_even_dw_kernel_errors() {
        let r = RtmDetConfig::new(3, 32, 8, vec![8, 16], 1, 4, 8, 4);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn config_zero_classes_errors() {
        let r = RtmDetConfig::new(3, 32, 8, vec![8, 16], 1, 5, 8, 0);
        assert!(matches!(r, Err(VisionError::InvalidNumClasses(0))));
    }

    // ── Conv sanity ───────────────────────────────────────────────────────────

    #[test]
    fn conv2d_stride2_halves_spatial() {
        let mut rng = LcgRng::new(1);
        let conv = Conv2d::new(3, 4, 3, 2, 1, &mut rng);
        let x = FeatureMap::new(vec![0.5f32; 3 * 16 * 16], 3, 16, 16).expect("ok");
        let y = conv.forward(&x).expect("ok");
        assert_eq!((y.channels, y.height, y.width), (4, 8, 8));
    }

    #[test]
    fn dwconv_identity_kernel_is_input() {
        // Centre-delta 3×3 depthwise kernel with zero bias = identity.
        let mut rng = LcgRng::new(2);
        let mut dw = DwConv2d::new(2, 3, 1, 1, &mut rng);
        for v in dw.weight.iter_mut() {
            *v = 0.0;
        }
        // centre of a 3×3 kernel is index 4.
        for ch in 0..2 {
            dw.weight[ch * 9 + 4] = 1.0;
        }
        let mut data = vec![0.0f32; 2 * 4 * 4];
        let mut r2 = LcgRng::new(3);
        r2.fill_normal(&mut data);
        let x = FeatureMap::new(data.clone(), 2, 4, 4).expect("ok");
        let y = dw.forward(&x).expect("ok");
        for (a, b) in y.data.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "identity dw mismatch {a} vs {b}");
        }
    }

    // ── Backbone: multi-scale halving ─────────────────────────────────────────

    #[test]
    fn backbone_multiscale_halving() {
        let cfg = RtmDetConfig::tiny();
        let mut rng = LcgRng::new(10);
        let det = RtmDet::new(cfg.clone(), &mut rng).expect("ok");
        let img = random_image(&cfg, 11);
        let feats = det.backbone_features(&img).expect("ok");
        assert_eq!(feats.len(), 3, "one feature per stage");
        // Stage spatials: 8, 4, 2 — each halves the previous.
        let spatials: Vec<usize> = feats.iter().map(|f| f.height).collect();
        assert_eq!(spatials, vec![8, 4, 2]);
        for w in feats.windows(2) {
            assert_eq!(w[0].height, w[1].height * 2, "each stage halves spatial");
            assert_eq!(w[0].width, w[1].width * 2);
        }
        // Channels match the configured stage widths.
        let chans: Vec<usize> = feats.iter().map(|f| f.channels).collect();
        assert_eq!(chans, cfg.stage_channels);
        assert!(feats.iter().all(|f| f.data.iter().all(|v| v.is_finite())));
    }

    // ── Neck: same #scales, fused channels ────────────────────────────────────

    #[test]
    fn pafpn_uniform_channels_same_scales() {
        let cfg = RtmDetConfig::tiny();
        let mut rng = LcgRng::new(12);
        let det = RtmDet::new(cfg.clone(), &mut rng).expect("ok");
        let img = random_image(&cfg, 13);
        let neck = det.neck_features(&img).expect("ok");
        assert_eq!(neck.len(), cfg.n_levels(), "neck preserves #scales");
        for fm in &neck {
            assert_eq!(fm.channels, cfg.neck_channels, "uniform fused channels");
        }
        // Spatial sizes preserved per level (8, 4, 2).
        let spatials: Vec<usize> = neck.iter().map(|f| f.height).collect();
        assert_eq!(spatials, vec![8, 4, 2]);
        assert!(neck.iter().all(|f| f.data.iter().all(|v| v.is_finite())));
    }

    // ── Decoupled head shapes ─────────────────────────────────────────────────

    #[test]
    fn decoupled_head_shapes() {
        let cfg = RtmDetConfig::tiny();
        let mut rng = LcgRng::new(14);
        let det = RtmDet::new(cfg.clone(), &mut rng).expect("ok");
        let img = random_image(&cfg, 15);
        let out = det.forward(&img).expect("ok");
        assert_eq!(out.cls_scores.len(), 3);
        assert_eq!(out.bbox_preds.len(), 3);
        for (cls, reg) in out.cls_scores.iter().zip(out.bbox_preds.iter()) {
            assert_eq!(cls.channels, cfg.n_classes, "cls has n_classes channels");
            assert_eq!(reg.channels, 4, "reg has 4 channels");
            assert_eq!(cls.height, reg.height);
            assert_eq!(cls.data.len(), cfg.n_classes * cls.height * cls.width);
            assert_eq!(reg.data.len(), 4 * reg.height * reg.width);
        }
        assert_eq!(out.strides, vec![4, 8, 16]);
    }

    #[test]
    fn forward_all_finite() {
        let cfg = RtmDetConfig::tiny();
        let mut rng = LcgRng::new(16);
        let det = RtmDet::new(cfg.clone(), &mut rng).expect("ok");
        let img = random_image(&cfg, 17);
        let out = det.forward(&img).expect("ok");
        for fm in out.cls_scores.iter().chain(out.bbox_preds.iter()) {
            assert!(fm.data.iter().all(|v| v.is_finite()));
        }
    }

    // ── Varying input changes detections ──────────────────────────────────────

    #[test]
    fn varying_input_changes_detections() {
        let cfg = RtmDetConfig::tiny();
        let mut rng = LcgRng::new(18);
        let det = RtmDet::new(cfg.clone(), &mut rng).expect("ok");
        let img_a = random_image(&cfg, 19);
        let img_b = random_image(&cfg, 20);
        let out_a = det.forward(&img_a).expect("ok");
        let out_b = det.forward(&img_b).expect("ok");
        // The finest-level class scores must differ for different inputs.
        let diff: f32 = out_a.cls_scores[0]
            .data
            .iter()
            .zip(out_b.cls_scores[0].data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "detections should change with input, diff={diff}"
        );
    }

    #[test]
    fn decode_level_produces_valid_boxes() {
        let cfg = RtmDetConfig::tiny();
        let mut rng = LcgRng::new(21);
        let det = RtmDet::new(cfg.clone(), &mut rng).expect("ok");
        let img = random_image(&cfg, 22);
        let out = det.forward(&img).expect("ok");
        let (boxes, scores, labels) =
            decode_level(&out.cls_scores[0], &out.bbox_preds[0], out.strides[0]).expect("ok");
        let n = out.cls_scores[0].height * out.cls_scores[0].width;
        assert_eq!(boxes.len(), n * 4);
        assert_eq!(scores.len(), n);
        assert_eq!(labels.len(), n);
        for loc in 0..n {
            // x2 > x1 and y2 > y1 (softplus distances are positive).
            assert!(boxes[loc * 4 + 2] > boxes[loc * 4], "x2 must exceed x1");
            assert!(boxes[loc * 4 + 3] > boxes[loc * 4 + 1], "y2 must exceed y1");
            assert!((0.0..=1.0).contains(&scores[loc]), "score in [0,1]");
            assert!(labels[loc] < cfg.n_classes);
        }
        assert!(boxes.iter().all(|v| v.is_finite()));
    }

    // ── SimOTA-lite cost ──────────────────────────────────────────────────────

    #[test]
    fn simota_lower_cost_for_better_match() {
        // 2 predictions, 1 ground-truth (class 0, box [0,0,10,10]).
        let n_classes = 2;
        // pred 0: confident on class 0, perfectly localised → should be cheap.
        // pred 1: confident on the wrong class, far away → should be expensive.
        let pred_cls = vec![
            0.9f32, 0.1, // pred 0
            0.1, 0.9, // pred 1
        ];
        let pred_boxes = vec![
            0.0f32, 0.0, 10.0, 10.0, // pred 0 (IoU = 1)
            20.0, 20.0, 30.0, 30.0, // pred 1 (IoU = 0)
        ];
        let gt_labels = vec![0usize];
        let gt_boxes = vec![0.0f32, 0.0, 10.0, 10.0];

        let cost = simota_cost(
            &pred_cls,
            &pred_boxes,
            &gt_labels,
            &gt_boxes,
            n_classes,
            3.0,
        )
        .expect("ok");
        assert_eq!(cost.len(), 2, "[n_gt × n_pred]");
        assert!(cost.iter().all(|v| v.is_finite()), "cost must be finite");
        // cost[gt 0, pred 0] must be much lower than cost[gt 0, pred 1].
        assert!(
            cost[0] < cost[1],
            "better cls+iou match must have lower cost: {} vs {}",
            cost[0],
            cost[1]
        );
    }

    #[test]
    fn simota_cost_monotonic_in_iou() {
        // Same class confidence, varying IoU → cost decreases with IoU.
        let n_classes = 1;
        let pred_cls = vec![0.8f32, 0.8, 0.8];
        let pred_boxes = vec![
            0.0f32, 0.0, 10.0, 10.0, // IoU 1.0
            5.0, 0.0, 15.0, 10.0, // IoU 1/3
            50.0, 50.0, 60.0, 60.0, // IoU 0
        ];
        let gt_labels = vec![0usize];
        let gt_boxes = vec![0.0f32, 0.0, 10.0, 10.0];
        let cost = simota_cost(
            &pred_cls,
            &pred_boxes,
            &gt_labels,
            &gt_boxes,
            n_classes,
            2.0,
        )
        .expect("ok");
        assert!(cost[0] < cost[1], "higher IoU → lower cost");
        assert!(cost[1] < cost[2], "higher IoU → lower cost");
    }

    #[test]
    fn simota_errors_on_bad_shapes() {
        // pred_cls length wrong.
        let r = simota_cost(
            &[0.5f32],
            &[0.0, 0.0, 1.0, 1.0],
            &[0],
            &[0.0, 0.0, 1.0, 1.0],
            2,
            1.0,
        );
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
        // empty predictions.
        let r2 = simota_cost(&[], &[], &[0], &[0.0, 0.0, 1.0, 1.0], 1, 1.0);
        assert!(matches!(r2, Err(VisionError::EmptyInput(_))));
    }

    // ── Determinism ───────────────────────────────────────────────────────────

    #[test]
    fn deterministic_same_seed() {
        let cfg = RtmDetConfig::tiny();
        let img = random_image(&cfg, 30);
        let mut ra = LcgRng::new(99);
        let mut rb = LcgRng::new(99);
        let da = RtmDet::new(cfg.clone(), &mut ra).expect("ok");
        let db = RtmDet::new(cfg, &mut rb).expect("ok");
        let oa = da.forward(&img).expect("ok");
        let ob = db.forward(&img).expect("ok");
        for (a, b) in oa.cls_scores.iter().zip(ob.cls_scores.iter()) {
            assert_eq!(a.data, b.data, "same seed → identical output");
        }
    }
}

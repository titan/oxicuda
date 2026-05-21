//! Mask R-CNN segmentation head.
//!
//! Reference: He et al. 2017 — *"Mask R-CNN"* (ICCV).
//!
//! For each Region-of-Interest (RoI), [`RoIAlign`](super::roi_align()) produces a
//! `(in_channels × roi_size × roi_size)` feature volume. This volume is fed
//! through a tiny **fully convolutional network** (FCN) that predicts a
//! per-class binary mask at **twice** the input resolution:
//!
//! ```text
//!     roi_features                                                            mask_logits
//!   (C_in, R, R)  ─┐                                                        ─►  (K, 2R, 2R)
//!                  │  n_conv × { 3×3 same-pad conv → ReLU }                    (per-class)
//!                  ▼            (first maps C_in → conv_dim, rest conv_dim → conv_dim)
//!            (conv_dim, R, R)
//!                  │  2× transposed-convolution (stride 2, kernel 2)
//!                  ▼            (conv_dim → conv_dim, then ReLU)
//!           (conv_dim, 2R, 2R)
//!                  │  1×1 conv to per-class logits
//!                  ▼
//!           (n_classes, 2R, 2R)
//! ```
//!
//! At inference time, [`MaskHead::predict_mask`] applies a sigmoid on the
//! logits of a chosen class to produce a binary mask in `[0, 1]`.
//!
//! The conv stack is **same-padding** (zero), the transposed conv uses
//! `kernel_size = 2`, `stride = 2`, `padding = 0` so that
//! `out_size = 2 · in_size` exactly, matching He 2017.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

// ─── MaskHeadConfig ───────────────────────────────────────────────────────────

/// Hyper-parameters for the Mask R-CNN segmentation head.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskHeadConfig {
    /// Input feature channels (the FPN level channel count).
    pub in_channels: usize,
    /// Spatial side of the RoIAlign output (commonly 14 or 28).
    pub roi_size: usize,
    /// Number of foreground classes.
    pub n_classes: usize,
    /// Number of 3×3 conv layers in the stack (commonly 4 in He 2017).
    pub n_conv: usize,
    /// Hidden channel dimension shared by all conv layers and the deconv.
    pub conv_dim: usize,
}

// ─── ConvLayer ────────────────────────────────────────────────────────────────

/// A single 3×3 same-pad convolution layer.
struct ConvLayer {
    /// Kernel weights `[out_c × in_c × 3 × 3]` row-major.
    weight: Vec<f32>,
    /// Bias `[out_c]`.
    bias: Vec<f32>,
    in_c: usize,
    out_c: usize,
}

impl ConvLayer {
    fn new(in_c: usize, out_c: usize, rng: &mut LcgRng) -> Self {
        // He initialisation: scale = sqrt(2 / (in_c · k²)).
        let fan_in = (in_c * 9) as f32;
        let scale = (2.0 / fan_in).sqrt();
        let n = out_c * in_c * 9;
        let mut weight = vec![0.0_f32; n];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= scale;
        }
        let bias = vec![0.0_f32; out_c];
        Self {
            weight,
            bias,
            in_c,
            out_c,
        }
    }

    /// 3×3 same-pad conv. Input `(in_c, h, w)`, output `(out_c, h, w)`.
    /// Padding is zero, stride is 1.
    fn forward(&self, x: &[f32], h: usize, w: usize) -> VisionResult<Vec<f32>> {
        let expected = self.in_c * h * w;
        if x.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        let hw = h * w;
        let mut out = vec![0.0_f32; self.out_c * hw];

        for oc in 0..self.out_c {
            let kernel_base = oc * self.in_c * 9;
            let bias = self.bias[oc];
            for oh in 0..h {
                for ow in 0..w {
                    let mut acc = bias;
                    for ic in 0..self.in_c {
                        let in_base = ic * hw;
                        let ker = kernel_base + ic * 9;
                        // Iterate the 3×3 kernel taps with pad=1.
                        for ki in 0..3usize {
                            let ih = oh as isize + ki as isize - 1;
                            if ih < 0 || ih >= h as isize {
                                continue;
                            }
                            let ih = ih as usize;
                            for kj in 0..3usize {
                                let iw = ow as isize + kj as isize - 1;
                                if iw < 0 || iw >= w as isize {
                                    continue;
                                }
                                let iw = iw as usize;
                                acc += self.weight[ker + ki * 3 + kj] * x[in_base + ih * w + iw];
                            }
                        }
                    }
                    out[oc * hw + oh * w + ow] = acc;
                }
            }
        }
        Ok(out)
    }

    fn n_params(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

// ─── DeconvLayer (2× transposed convolution) ──────────────────────────────────

/// 2×2 transposed convolution with stride 2.
///
/// Maps `(in_c, h, w)` → `(out_c, 2h, 2w)` by depositing each input pixel into
/// a `2 × 2` patch of the output via the kernel.
struct DeconvLayer {
    /// Kernel `[in_c × out_c × 2 × 2]` row-major.
    weight: Vec<f32>,
    /// Bias `[out_c]`.
    bias: Vec<f32>,
    in_c: usize,
    out_c: usize,
}

impl DeconvLayer {
    fn new(in_c: usize, out_c: usize, rng: &mut LcgRng) -> Self {
        // He init for transposed conv: fan_in = in_c · k².
        let fan_in = (in_c * 4) as f32;
        let scale = (2.0 / fan_in).sqrt();
        let n = in_c * out_c * 4;
        let mut weight = vec![0.0_f32; n];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= scale;
        }
        let bias = vec![0.0_f32; out_c];
        Self {
            weight,
            bias,
            in_c,
            out_c,
        }
    }

    fn forward(&self, x: &[f32], h: usize, w: usize) -> VisionResult<Vec<f32>> {
        let expected = self.in_c * h * w;
        if x.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        let out_h = h * 2;
        let out_w = w * 2;
        let out_hw = out_h * out_w;
        let mut out = vec![0.0_f32; self.out_c * out_hw];

        // Initialise output to bias.
        for oc in 0..self.out_c {
            let b = self.bias[oc];
            let base = oc * out_hw;
            for p in 0..out_hw {
                out[base + p] = b;
            }
        }

        // For each input pixel (ic, ih, iw), scatter its contribution to the
        // 2×2 output patch starting at (2·ih, 2·iw) for every (oc, ki, kj).
        let hw = h * w;
        for ic in 0..self.in_c {
            let in_base = ic * hw;
            let ker_base = ic * self.out_c * 4;
            for ih in 0..h {
                for iw in 0..w {
                    let v = x[in_base + ih * w + iw];
                    if v == 0.0 {
                        continue;
                    }
                    let oh0 = ih * 2;
                    let ow0 = iw * 2;
                    for oc in 0..self.out_c {
                        let kbase = ker_base + oc * 4;
                        let obase = oc * out_hw;
                        // Unrolled 2×2.
                        out[obase + oh0 * out_w + ow0] += v * self.weight[kbase];
                        out[obase + oh0 * out_w + ow0 + 1] += v * self.weight[kbase + 1];
                        out[obase + (oh0 + 1) * out_w + ow0] += v * self.weight[kbase + 2];
                        out[obase + (oh0 + 1) * out_w + ow0 + 1] += v * self.weight[kbase + 3];
                    }
                }
            }
        }

        Ok(out)
    }

    fn n_params(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

// ─── Pointwise (1×1) conv ─────────────────────────────────────────────────────

/// 1×1 convolution: per-position linear map.
struct Pointwise {
    weight: Vec<f32>, // [out_c × in_c]
    bias: Vec<f32>,   // [out_c]
    in_c: usize,
    out_c: usize,
}

impl Pointwise {
    fn new(in_c: usize, out_c: usize, rng: &mut LcgRng) -> Self {
        let fan_in = in_c as f32;
        let scale = (2.0 / fan_in).sqrt();
        let mut weight = vec![0.0_f32; out_c * in_c];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= scale;
        }
        let bias = vec![0.0_f32; out_c];
        Self {
            weight,
            bias,
            in_c,
            out_c,
        }
    }

    fn forward(&self, x: &[f32], h: usize, w: usize) -> VisionResult<Vec<f32>> {
        let expected = self.in_c * h * w;
        if x.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        let hw = h * w;
        let mut out = vec![0.0_f32; self.out_c * hw];
        for oc in 0..self.out_c {
            let wrow = &self.weight[oc * self.in_c..(oc + 1) * self.in_c];
            let b = self.bias[oc];
            for p in 0..hw {
                let mut acc = b;
                for ic in 0..self.in_c {
                    acc += wrow[ic] * x[ic * hw + p];
                }
                out[oc * hw + p] = acc;
            }
        }
        Ok(out)
    }

    fn n_params(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

// ─── MaskHead ─────────────────────────────────────────────────────────────────

/// Mask R-CNN segmentation head.
pub struct MaskHead {
    cfg: MaskHeadConfig,
    convs: Vec<ConvLayer>,
    deconv: DeconvLayer,
    mask_pred: Pointwise,
}

impl MaskHead {
    /// Build a mask head with He-initialised weights from `rng`.
    ///
    /// # Errors
    /// [`VisionError::DimensionMismatch`] if any of `in_channels`,
    /// `roi_size`, `n_classes`, `n_conv`, `conv_dim` is zero.
    pub fn new(cfg: MaskHeadConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        if cfg.in_channels == 0 {
            return Err(VisionError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if cfg.roi_size == 0 {
            return Err(VisionError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if cfg.n_classes == 0 {
            return Err(VisionError::InvalidNumClasses(0));
        }
        if cfg.n_conv == 0 {
            return Err(VisionError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if cfg.conv_dim == 0 {
            return Err(VisionError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        let mut convs = Vec::with_capacity(cfg.n_conv);
        // First conv maps in_channels → conv_dim; the rest are conv_dim → conv_dim.
        convs.push(ConvLayer::new(cfg.in_channels, cfg.conv_dim, rng));
        for _ in 1..cfg.n_conv {
            convs.push(ConvLayer::new(cfg.conv_dim, cfg.conv_dim, rng));
        }
        let deconv = DeconvLayer::new(cfg.conv_dim, cfg.conv_dim, rng);
        let mask_pred = Pointwise::new(cfg.conv_dim, cfg.n_classes, rng);

        Ok(Self {
            cfg,
            convs,
            deconv,
            mask_pred,
        })
    }

    /// Read-only access to the head configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &MaskHeadConfig {
        &self.cfg
    }

    /// Forward the convolution stack only (used as an intermediate testable
    /// API): returns the activation at the input of the deconv layer,
    /// shape `(conv_dim × roi_size × roi_size)`.
    ///
    /// All conv layers use ReLU activation.
    ///
    /// # Errors
    /// [`VisionError::DimensionMismatch`] if `roi_features.len() !=
    /// in_channels · roi_size²`.
    pub fn forward_conv_stack(&self, roi_features: &[f32]) -> VisionResult<Vec<f32>> {
        let r = self.cfg.roi_size;
        let expected = self.cfg.in_channels * r * r;
        if roi_features.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: roi_features.len(),
            });
        }

        let mut feat = roi_features.to_vec();
        for conv in &self.convs {
            let mut next = conv.forward(&feat, r, r)?;
            // In-place ReLU.
            for v in &mut next {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            feat = next;
        }
        Ok(feat)
    }

    /// Forward pass: returns per-class mask logits.
    ///
    /// Input `(in_channels × roi_size × roi_size)` → output
    /// `(n_classes × 2·roi_size × 2·roi_size)` row-major.
    ///
    /// # Errors
    /// [`VisionError::DimensionMismatch`] if `roi_features.len() !=
    /// in_channels · roi_size²`.
    pub fn forward(&self, roi_features: &[f32]) -> VisionResult<Vec<f32>> {
        let r = self.cfg.roi_size;
        // Step 1: same-pad conv stack at resolution R×R.
        let after_convs = self.forward_conv_stack(roi_features)?;
        // Step 2: 2× transposed conv → (conv_dim, 2R, 2R), then ReLU.
        let mut after_deconv = self.deconv.forward(&after_convs, r, r)?;
        for v in &mut after_deconv {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        // Step 3: 1×1 conv → (n_classes, 2R, 2R). No activation: logits.
        let logits = self.mask_pred.forward(&after_deconv, 2 * r, 2 * r)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("mask head logits"));
        }
        Ok(logits)
    }

    /// Predict the sigmoid mask for a single class.
    ///
    /// Output length: `(2 · roi_size)²`, values in `[0, 1]`.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `roi_features` has the wrong
    ///   length.
    /// - [`VisionError::InvalidNumClasses`] if `class_idx >= n_classes`.
    pub fn predict_mask(&self, roi_features: &[f32], class_idx: usize) -> VisionResult<Vec<f32>> {
        if class_idx >= self.cfg.n_classes {
            return Err(VisionError::InvalidNumClasses(class_idx));
        }
        let logits = self.forward(roi_features)?;
        let out_size = 4 * self.cfg.roi_size * self.cfg.roi_size; // (2R)²
        let base = class_idx * out_size;
        let mut mask = Vec::with_capacity(out_size);
        for i in 0..out_size {
            let z = logits[base + i];
            mask.push(sigmoid(z));
        }
        Ok(mask)
    }

    /// Total learnable parameter count.
    #[must_use]
    pub fn n_params(&self) -> usize {
        let conv_params: usize = self.convs.iter().map(ConvLayer::n_params).sum();
        conv_params + self.deconv.n_params() + self.mask_pred.n_params()
    }
}

// ─── Sigmoid ──────────────────────────────────────────────────────────────────

/// Numerically stable sigmoid.
#[inline]
fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> MaskHeadConfig {
        // in=4, roi=4, classes=3, n_conv=2, conv_dim=8
        MaskHeadConfig {
            in_channels: 4,
            roi_size: 4,
            n_classes: 3,
            n_conv: 2,
            conv_dim: 8,
        }
    }

    fn make_head(seed: u64) -> MaskHead {
        let mut rng = LcgRng::new(seed);
        MaskHead::new(small_cfg(), &mut rng).expect("ok")
    }

    fn random_roi(in_c: usize, r: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut x = vec![0.0_f32; in_c * r * r];
        rng.fill_normal(&mut x);
        x
    }

    // ── Output shapes ─────────────────────────────────────────────────────────

    #[test]
    fn forward_output_length() {
        let head = make_head(1);
        let roi = random_roi(4, 4, 2);
        let out = head.forward(&roi).expect("ok");
        assert_eq!(out.len(), 3 * 8 * 8, "n_classes × 2R × 2R");
    }

    #[test]
    fn predict_mask_length_and_range() {
        let head = make_head(3);
        let roi = random_roi(4, 4, 4);
        for k in 0..3 {
            let m = head.predict_mask(&roi, k).expect("ok");
            assert_eq!(m.len(), 8 * 8);
            for &v in &m {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "sigmoid out of [0,1] for class {k}: {v}"
                );
            }
        }
    }

    #[test]
    fn upsample_doubles_spatial_dims() {
        let cfg = MaskHeadConfig {
            in_channels: 2,
            roi_size: 7,
            n_classes: 1,
            n_conv: 1,
            conv_dim: 4,
        };
        let mut rng = LcgRng::new(5);
        let head = MaskHead::new(cfg, &mut rng).expect("ok");
        let roi = vec![0.0_f32; 2 * 7 * 7];
        let out = head.forward(&roi).expect("ok");
        assert_eq!(out.len(), 14 * 14);
    }

    #[test]
    fn conv_stack_preserves_spatial_size() {
        let head = make_head(6);
        let roi = random_roi(4, 4, 7);
        let mid = head.forward_conv_stack(&roi).expect("ok");
        // Should be (conv_dim, R, R) = (8, 4, 4) = 128.
        assert_eq!(mid.len(), 8 * 4 * 4);
    }

    #[test]
    fn conv_stack_relu_non_negative() {
        let head = make_head(8);
        let roi = random_roi(4, 4, 9);
        let mid = head.forward_conv_stack(&roi).expect("ok");
        for &v in &mid {
            assert!(v >= 0.0, "ReLU should clamp to >= 0; got {v}");
        }
    }

    // ── n_params formula ──────────────────────────────────────────────────────

    #[test]
    fn n_params_positive_and_reasonable() {
        let head = make_head(10);
        let cfg = head.config();
        // Conv0 : conv_dim × in_channels × 9 + conv_dim
        let c0 = cfg.conv_dim * cfg.in_channels * 9 + cfg.conv_dim;
        // Conv1..n_conv-1: conv_dim² × 9 + conv_dim
        let ck = (cfg.n_conv - 1) * (cfg.conv_dim * cfg.conv_dim * 9 + cfg.conv_dim);
        // Deconv: conv_dim × conv_dim × 4 + conv_dim
        let dc = cfg.conv_dim * cfg.conv_dim * 4 + cfg.conv_dim;
        // Mask pred: n_classes × conv_dim + n_classes
        let mp = cfg.n_classes * cfg.conv_dim + cfg.n_classes;
        let expected = c0 + ck + dc + mp;
        assert_eq!(head.n_params(), expected);
        assert!(head.n_params() > 0);
    }

    #[test]
    fn deterministic_given_seed() {
        let head_a = make_head(42);
        let head_b = make_head(42);
        let roi = random_roi(4, 4, 99);
        let oa = head_a.forward(&roi).expect("ok");
        let ob = head_b.forward(&roi).expect("ok");
        assert_eq!(oa, ob);
    }

    #[test]
    fn different_input_changes_output() {
        let head = make_head(11);
        let r1 = random_roi(4, 4, 100);
        let r2 = random_roi(4, 4, 101);
        let o1 = head.forward(&r1).expect("ok");
        let o2 = head.forward(&r2).expect("ok");
        assert_ne!(o1, o2);
    }

    #[test]
    fn n_conv_one_works() {
        let cfg = MaskHeadConfig {
            in_channels: 3,
            roi_size: 5,
            n_classes: 2,
            n_conv: 1,
            conv_dim: 6,
        };
        let mut rng = LcgRng::new(12);
        let head = MaskHead::new(cfg, &mut rng).expect("ok");
        let roi = vec![0.1_f32; 3 * 5 * 5];
        let out = head.forward(&roi).expect("ok");
        assert_eq!(out.len(), 2 * 10 * 10);
    }

    #[test]
    fn single_class_works() {
        let cfg = MaskHeadConfig {
            in_channels: 2,
            roi_size: 3,
            n_classes: 1,
            n_conv: 2,
            conv_dim: 4,
        };
        let mut rng = LcgRng::new(13);
        let head = MaskHead::new(cfg, &mut rng).expect("ok");
        let roi = vec![0.2_f32; 2 * 3 * 3];
        let mask = head.predict_mask(&roi, 0).expect("ok");
        assert_eq!(mask.len(), 6 * 6);
    }

    // ── Error paths ───────────────────────────────────────────────────────────

    #[test]
    fn err_roi_features_wrong_length() {
        let head = make_head(14);
        // expected = 4*4*4 = 64, give 60.
        let roi = vec![0.0_f32; 60];
        let r = head.forward(&roi);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_class_idx_out_of_range() {
        let head = make_head(15);
        let roi = random_roi(4, 4, 16);
        let r = head.predict_mask(&roi, 99);
        assert!(matches!(r, Err(VisionError::InvalidNumClasses(_))));
    }

    #[test]
    fn err_in_channels_zero() {
        let mut rng = LcgRng::new(17);
        let cfg = MaskHeadConfig {
            in_channels: 0,
            roi_size: 4,
            n_classes: 3,
            n_conv: 2,
            conv_dim: 8,
        };
        let r = MaskHead::new(cfg, &mut rng);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_roi_size_zero() {
        let mut rng = LcgRng::new(18);
        let cfg = MaskHeadConfig {
            in_channels: 4,
            roi_size: 0,
            n_classes: 3,
            n_conv: 2,
            conv_dim: 8,
        };
        let r = MaskHead::new(cfg, &mut rng);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_n_classes_zero() {
        let mut rng = LcgRng::new(19);
        let cfg = MaskHeadConfig {
            in_channels: 4,
            roi_size: 4,
            n_classes: 0,
            n_conv: 2,
            conv_dim: 8,
        };
        let r = MaskHead::new(cfg, &mut rng);
        assert!(matches!(r, Err(VisionError::InvalidNumClasses(0))));
    }

    #[test]
    fn err_n_conv_zero() {
        let mut rng = LcgRng::new(20);
        let cfg = MaskHeadConfig {
            in_channels: 4,
            roi_size: 4,
            n_classes: 3,
            n_conv: 0,
            conv_dim: 8,
        };
        let r = MaskHead::new(cfg, &mut rng);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_conv_dim_zero() {
        let mut rng = LcgRng::new(21);
        let cfg = MaskHeadConfig {
            in_channels: 4,
            roi_size: 4,
            n_classes: 3,
            n_conv: 2,
            conv_dim: 0,
        };
        let r = MaskHead::new(cfg, &mut rng);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn logits_finite() {
        let head = make_head(22);
        let roi = random_roi(4, 4, 23);
        let out = head.forward(&roi).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

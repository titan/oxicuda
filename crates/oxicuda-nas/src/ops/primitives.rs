//! Primitive operations for Neural Architecture Search.
//!
//! Implements the eight canonical DARTS operations with CPU reference forward passes
//! and FLOPs counting. All operations operate on feature maps in CHW layout
//! (`[C, H, W]` = `[channels * h * w]` contiguous).

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

// ─── OpKind ──────────────────────────────────────────────────────────────────

/// The eight candidate operations used in DARTS / one-shot NAS cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Zero — output is all zeros regardless of input.
    Zero,
    /// Identity — passes input through unchanged (requires in_ch == out_ch).
    Identity,
    /// 3×3 separable convolution (depthwise + pointwise).
    SepConv3x3,
    /// 5×5 separable convolution (depthwise + pointwise).
    SepConv5x5,
    /// 3×3 dilated convolution (dilation = 2).
    DilConv3x3,
    /// 5×5 dilated convolution (dilation = 2).
    DilConv5x5,
    /// 3×3 max pooling.
    MaxPool3x3,
    /// 3×3 average pooling.
    AvgPool3x3,
}

impl OpKind {
    /// Return all 8 operation variants in canonical order.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Zero,
            Self::Identity,
            Self::SepConv3x3,
            Self::SepConv5x5,
            Self::DilConv3x3,
            Self::DilConv5x5,
            Self::MaxPool3x3,
            Self::AvgPool3x3,
        ]
    }

    /// Total number of operations (8).
    #[must_use]
    pub fn n_ops() -> usize {
        8
    }

    /// Human-readable name for the operation.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Identity => "identity",
            Self::SepConv3x3 => "sep_conv_3x3",
            Self::SepConv5x5 => "sep_conv_5x5",
            Self::DilConv3x3 => "dil_conv_3x3",
            Self::DilConv5x5 => "dil_conv_5x5",
            Self::MaxPool3x3 => "max_pool_3x3",
            Self::AvgPool3x3 => "avg_pool_3x3",
        }
    }

    /// CPU reference forward pass.
    ///
    /// Input layout: `[in_ch, h, w]` contiguous (row-major).
    /// Output layout: `[out_ch, h, w]` (same spatial dims, except pooling
    /// preserves spatial dims with padding=1 stride=1 here).
    pub fn forward_cpu(
        self,
        input: &[f32],
        in_ch: usize,
        h: usize,
        w: usize,
        out_ch: usize,
        weights: &OpWeights,
    ) -> NasResult<Vec<f32>> {
        let expected_len = in_ch * h * w;
        if input.len() != expected_len {
            return Err(NasError::DimensionMismatch {
                expected: expected_len,
                got: input.len(),
            });
        }
        match self {
            Self::Zero => Ok(vec![0.0_f32; out_ch * h * w]),
            Self::Identity => forward_identity(input, in_ch, h, w, out_ch),
            Self::SepConv3x3 => forward_sep_conv(input, in_ch, h, w, out_ch, 3, 1, weights),
            Self::SepConv5x5 => forward_sep_conv(input, in_ch, h, w, out_ch, 5, 2, weights),
            Self::DilConv3x3 => forward_dil_conv(input, in_ch, h, w, out_ch, 3, 2, weights),
            Self::DilConv5x5 => forward_dil_conv(input, in_ch, h, w, out_ch, 5, 4, weights),
            Self::MaxPool3x3 => forward_max_pool(input, in_ch, h, w, 3, 1),
            Self::AvgPool3x3 => forward_avg_pool(input, in_ch, h, w, 3, 1),
        }
    }
}

// ─── OpWeights ───────────────────────────────────────────────────────────────

/// Weight tensors sufficient to parameterise all 8 `OpKind` variants.
///
/// Zero, Identity, and pooling ops ignore the weights; convolutional ops use
/// the relevant fields.
#[derive(Debug, Clone)]
pub struct OpWeights {
    /// Standard conv weights: `[out_ch, in_ch, k, k]`.
    pub weight: Vec<f32>,
    /// Bias: `[out_ch]`.
    pub bias: Vec<f32>,
    /// Depthwise conv weight for SepConv: `[in_ch, 1, k, k]`.
    pub dw_weight: Vec<f32>,
    /// Pointwise conv weight for SepConv: `[out_ch, in_ch]`.
    pub pw_weight: Vec<f32>,
}

impl OpWeights {
    /// Allocate zero-initialised weights for a given channel / kernel config.
    #[must_use]
    pub fn zeros(in_ch: usize, out_ch: usize, kernel: usize) -> Self {
        Self {
            weight: vec![0.0_f32; out_ch * in_ch * kernel * kernel],
            bias: vec![0.0_f32; out_ch],
            dw_weight: vec![0.0_f32; in_ch * kernel * kernel],
            pw_weight: vec![0.0_f32; out_ch * in_ch],
        }
    }

    /// Allocate random (N(0, 0.01)) weights using the given LCG RNG.
    #[must_use]
    pub fn random(in_ch: usize, out_ch: usize, kernel: usize, rng: &mut LcgRng) -> Self {
        let scale = 0.01_f32;
        let mut make = |n: usize| -> Vec<f32> {
            let mut buf = vec![0.0_f32; n];
            rng.fill_normal(&mut buf);
            buf.iter_mut().for_each(|v| *v *= scale);
            buf
        };
        Self {
            weight: make(out_ch * in_ch * kernel * kernel),
            bias: make(out_ch),
            dw_weight: make(in_ch * kernel * kernel),
            pw_weight: make(out_ch * in_ch),
        }
    }
}

// ─── Identity ────────────────────────────────────────────────────────────────

fn forward_identity(
    input: &[f32],
    in_ch: usize,
    h: usize,
    w: usize,
    out_ch: usize,
) -> NasResult<Vec<f32>> {
    if in_ch != out_ch {
        return Err(NasError::DimensionMismatch {
            expected: in_ch,
            got: out_ch,
        });
    }
    Ok(input[..in_ch * h * w].to_vec())
}

// ─── Depthwise convolution helper ────────────────────────────────────────────

/// Naive depthwise convolution: `[in_ch, k, k]` weight, same-padding.
fn depthwise_conv(
    input: &[f32],
    in_ch: usize,
    h: usize,
    w: usize,
    dw_weight: &[f32],
    kernel: usize,
    dilation: usize,
) -> Vec<f32> {
    let pad = dilation * (kernel / 2);
    let mut out = vec![0.0_f32; in_ch * h * w];
    for c in 0..in_ch {
        for oy in 0..h {
            for ox in 0..w {
                let mut acc = 0.0_f32;
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        let iy =
                            oy as isize + (ky as isize - (kernel / 2) as isize) * dilation as isize;
                        let ix =
                            ox as isize + (kx as isize - (kernel / 2) as isize) * dilation as isize;
                        // same-padding: skip out-of-bounds
                        if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                            continue;
                        }
                        let iy = iy as usize;
                        let ix = ix as usize;
                        let in_idx = c * h * w + iy * w + ix;
                        let w_idx = c * kernel * kernel + ky * kernel + kx;
                        acc += input[in_idx] * dw_weight[w_idx];
                    }
                }
                let _ = pad; // suppress unused warning — pad is computed but applied via bounds check
                out[c * h * w + oy * w + ox] = acc;
            }
        }
    }
    out
}

// ─── Pointwise convolution helper ────────────────────────────────────────────

/// 1×1 pointwise convolution: `[out_ch, in_ch]` weight.
fn pointwise_conv(
    input: &[f32],
    in_ch: usize,
    h: usize,
    w: usize,
    out_ch: usize,
    pw_weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_ch * h * w];
    for oc in 0..out_ch {
        let b = bias.get(oc).copied().unwrap_or(0.0);
        for y in 0..h {
            for x in 0..w {
                let mut acc = b;
                for ic in 0..in_ch {
                    acc += pw_weight[oc * in_ch + ic] * input[ic * h * w + y * w + x];
                }
                out[oc * h * w + y * w + x] = acc;
            }
        }
    }
    out
}

// ─── Separable convolution ───────────────────────────────────────────────────

fn forward_sep_conv(
    input: &[f32],
    in_ch: usize,
    h: usize,
    w: usize,
    out_ch: usize,
    kernel: usize,
    pad: usize,
    weights: &OpWeights,
) -> NasResult<Vec<f32>> {
    let _ = pad; // padding handled via bounds checking in depthwise_conv
    // Validate weight sizes
    let expected_dw = in_ch * kernel * kernel;
    if weights.dw_weight.len() < expected_dw {
        return Err(NasError::InvalidWeightShape);
    }
    let expected_pw = out_ch * in_ch;
    if weights.pw_weight.len() < expected_pw {
        return Err(NasError::InvalidWeightShape);
    }
    // Depthwise: dilation = 1 for SepConv
    let dw_out = depthwise_conv(input, in_ch, h, w, &weights.dw_weight, kernel, 1);
    // Pointwise
    let pw_out = pointwise_conv(
        &dw_out,
        in_ch,
        h,
        w,
        out_ch,
        &weights.pw_weight,
        &weights.bias,
    );
    Ok(pw_out)
}

// ─── Dilated convolution ─────────────────────────────────────────────────────

fn forward_dil_conv(
    input: &[f32],
    in_ch: usize,
    h: usize,
    w: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    weights: &OpWeights,
) -> NasResult<Vec<f32>> {
    let expected_w = out_ch * in_ch * kernel * kernel;
    if weights.weight.len() < expected_w {
        return Err(NasError::InvalidWeightShape);
    }
    // Naive dilated conv with same-padding
    let kh = kernel;
    let kw = kernel;
    let mut out = vec![0.0_f32; out_ch * h * w];
    for oc in 0..out_ch {
        let b = weights.bias.get(oc).copied().unwrap_or(0.0);
        for oy in 0..h {
            for ox in 0..w {
                let mut acc = b;
                for ic in 0..in_ch {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy =
                                oy as isize + (ky as isize - (kh / 2) as isize) * dilation as isize;
                            let ix =
                                ox as isize + (kx as isize - (kw / 2) as isize) * dilation as isize;
                            if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                                continue;
                            }
                            let iy = iy as usize;
                            let ix = ix as usize;
                            let in_idx = ic * h * w + iy * w + ix;
                            let w_idx = oc * in_ch * kh * kw + ic * kh * kw + ky * kw + kx;
                            acc += input[in_idx] * weights.weight[w_idx];
                        }
                    }
                }
                out[oc * h * w + oy * w + ox] = acc;
            }
        }
    }
    Ok(out)
}

// ─── Max pooling ─────────────────────────────────────────────────────────────

fn forward_max_pool(
    input: &[f32],
    in_ch: usize,
    h: usize,
    w: usize,
    kernel: usize,
    stride: usize,
) -> NasResult<Vec<f32>> {
    // Same-padding, stride=1 → output spatial dims unchanged
    let out_h = h.div_ceil(stride);
    let out_w = w.div_ceil(stride);
    let pad_h = kernel / 2;
    let pad_w = kernel / 2;
    let mut out = vec![f32::NEG_INFINITY; in_ch * out_h * out_w];
    for c in 0..in_ch {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut max_val = f32::NEG_INFINITY;
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        let iy = oy as isize * stride as isize + ky as isize - pad_h as isize;
                        let ix = ox as isize * stride as isize + kx as isize - pad_w as isize;
                        if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                            continue;
                        }
                        let v = input[c * h * w + iy as usize * w + ix as usize];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                if max_val == f32::NEG_INFINITY {
                    max_val = 0.0;
                }
                out[c * out_h * out_w + oy * out_w + ox] = max_val;
            }
        }
    }
    Ok(out)
}

// ─── Average pooling ─────────────────────────────────────────────────────────

fn forward_avg_pool(
    input: &[f32],
    in_ch: usize,
    h: usize,
    w: usize,
    kernel: usize,
    stride: usize,
) -> NasResult<Vec<f32>> {
    let out_h = h.div_ceil(stride);
    let out_w = w.div_ceil(stride);
    let pad_h = kernel / 2;
    let pad_w = kernel / 2;
    let mut out = vec![0.0_f32; in_ch * out_h * out_w];
    for c in 0..in_ch {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut sum = 0.0_f32;
                let mut count = 0u32;
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        let iy = oy as isize * stride as isize + ky as isize - pad_h as isize;
                        let ix = ox as isize * stride as isize + kx as isize - pad_w as isize;
                        if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                            continue;
                        }
                        sum += input[c * h * w + iy as usize * w + ix as usize];
                        count += 1;
                    }
                }
                out[c * out_h * out_w + oy * out_w + ox] =
                    if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }
    }
    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_kind_all_has_8_variants() {
        assert_eq!(OpKind::all().len(), 8);
        assert_eq!(OpKind::n_ops(), 8);
    }

    #[test]
    fn zero_op_gives_zeros() {
        let input = vec![1.0_f32; 3 * 4 * 4];
        let w = OpWeights::zeros(3, 3, 3);
        let out = OpKind::Zero
            .forward_cpu(&input, 3, 4, 4, 3, &w)
            .expect("test invariant: zero op");
        assert!(out.iter().all(|&v| v == 0.0));
        assert_eq!(out.len(), 3 * 4 * 4);
    }

    #[test]
    fn identity_passes_through() {
        let input: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let w = OpWeights::zeros(3, 3, 1);
        let out = OpKind::Identity
            .forward_cpu(&input, 3, 4, 4, 3, &w)
            .expect("test invariant: identity op");
        assert_eq!(out, input);
    }

    #[test]
    fn identity_rejects_channel_mismatch() {
        let input = vec![0.0_f32; 3 * 4 * 4];
        let w = OpWeights::zeros(3, 4, 1);
        let result = OpKind::Identity.forward_cpu(&input, 3, 4, 4, 4, &w);
        assert!(result.is_err());
    }

    #[test]
    fn sep_conv_3x3_output_shape() {
        let mut rng = LcgRng::new(42);
        let input = vec![1.0_f32; 4 * 8 * 8];
        let w = OpWeights::random(4, 8, 3, &mut rng);
        let out = OpKind::SepConv3x3
            .forward_cpu(&input, 4, 8, 8, 8, &w)
            .expect("test invariant: sep conv 3x3");
        assert_eq!(out.len(), 8 * 8 * 8);
    }

    #[test]
    fn max_pool_output_shape() {
        let input = vec![1.0_f32; 4 * 8 * 8];
        let w = OpWeights::zeros(4, 4, 3);
        let out = OpKind::MaxPool3x3
            .forward_cpu(&input, 4, 8, 8, 4, &w)
            .expect("test invariant: max pool");
        assert_eq!(out.len(), 4 * 8 * 8);
    }
}

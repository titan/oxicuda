//! INT8 quantized inference path for Conformer/Transformer linear layers.
//!
//! This module implements an **integer-arithmetic-only** inference path for the
//! dense (linear) projections that dominate Conformer and Transformer encoders:
//! the feed-forward sub-layers and the query/key/value/output attention
//! projections. The goal is to reproduce the floating-point result of
//! `y = x · Wᵀ + b` while performing the inner-product accumulation in INT8/INT32
//! arithmetic, as deployed on integer-only accelerators and tensor cores.
//!
//! # Quantization scheme
//!
//! The scheme is **symmetric** (zero-point-free) and follows the standard
//! integer-only formulation of Jacob et al. (2018) and the per-channel weight
//! recommendation of Krishnamoorthi (2018):
//!
//! * **Weights** are quantized **per output channel** (one scale per output row
//!   of the `[out, in]` weight matrix). This is the accuracy-critical choice: a
//!   single output channel with a large dynamic range no longer forces a coarse
//!   step size onto every other channel.
//! * **Activations** are quantized **per tensor** (a single scale for the whole
//!   activation matrix), which mirrors how activation scales are produced cheaply
//!   at runtime from a running maximum.
//!
//! A real INT8 value `q` and its scale `s` represent the de-quantized real number
//! `r ≈ q · s`. Because the scheme is symmetric we use the **127-level** range
//! `[-127, 127]` rather than `[-128, 127]`, so the quantization grid is
//! symmetric about zero and no zero-point correction term is needed in the
//! matmul. Rounding uses **round-half-to-even** (the behaviour of
//! [`f32::round_ties_even`]) so that no systematic positive bias is introduced
//! across a tensor.
//!
//! # Integer GEMM
//!
//! For a single output element the accumulation is
//!
//! ```text
//! acc_i32 = Σ_k  q_act[i, k] · q_w[o, k]            (exact INT32 accumulation)
//! y[i, o] = acc_i32 · (act_scale · weight_scale[o]) + bias[o]
//! ```
//!
//! The product of two INT8 values fits in INT16, and accumulating `in_features`
//! of them fits comfortably in INT32 for any realistic feature dimension
//! (`127² · in ≤ 2³¹` for `in ≤ 133 000`), so the accumulator never overflows.
//! De-quantization is a single multiply by the fused scale `act_scale ·
//! weight_scale[o]`.
//!
//! # References
//!
//! * B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and
//!   D. Kalenichenko, "Quantization and Training of Neural Networks for Efficient
//!   Integer-Arithmetic-Only Inference," *CVPR*, 2018.
//! * R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient
//!   inference: A whitepaper," *arXiv:1806.08342*, 2018.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Symmetric INT8 quantization primitives ──────────────────────────────────

/// Largest magnitude representable by a symmetric INT8 quantizer.
///
/// We deliberately use 127 (not 128) so that the grid `[-127, 127]` is symmetric
/// about zero, making the scheme zero-point-free.
const INT8_SYMMETRIC_MAX: f32 = 127.0;

/// Compute a symmetric per-tensor INT8 scale for `data`.
///
/// The scale is `max(|data|) / 127`, the step size that maps the largest
/// magnitude in `data` onto the outermost grid point `±127`.
///
/// * If `data` is empty or contains only zeros, there is no dynamic range to
///   encode; the function returns `1.0` so that subsequent quantization maps
///   every value to `0` without a division by zero.
///
/// # Errors
///
/// Returns [`AudioError::NonFinite`] if any element is NaN or infinite.
pub fn compute_scale_symmetric(data: &[f32]) -> AudioResult<f32> {
    let mut max_abs = 0.0_f32;
    for &v in data {
        if !v.is_finite() {
            return Err(AudioError::NonFinite {
                msg: "compute_scale_symmetric: non-finite input value".to_string(),
            });
        }
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs == 0.0 {
        // All-zero (or empty) input: no scale needed, avoid div-by-zero.
        return Ok(1.0);
    }
    Ok(max_abs / INT8_SYMMETRIC_MAX)
}

/// Quantize `data` to symmetric INT8 using `scale`.
///
/// Each value is mapped with `q = clamp(round_ties_even(x / scale), -127, 127)`.
/// Rounding is **round-half-to-even** so the operation introduces no systematic
/// directional bias.
///
/// A non-positive or non-finite `scale` is treated as the "no dynamic range"
/// case and yields an all-zero output, mirroring [`compute_scale_symmetric`].
#[must_use]
pub fn quantize_symmetric(data: &[f32], scale: f32) -> Vec<i8> {
    if scale <= 0.0 || !scale.is_finite() {
        return vec![0_i8; data.len()];
    }
    let inv = 1.0 / scale;
    data.iter()
        .map(|&x| {
            let q = (x * inv).round_ties_even();
            // Clamp into the symmetric INT8 range, then narrow.
            let clamped = q.clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX);
            clamped as i8
        })
        .collect()
}

/// De-quantize symmetric INT8 values back to `f32` via `x ≈ q · scale`.
#[must_use]
pub fn dequantize_symmetric(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|&qi| f32::from(qi) * scale).collect()
}

/// A per-tensor symmetric INT8 quantized matrix.
///
/// Stores the INT8 payload in row-major `[rows, cols]` order together with the
/// single per-tensor `scale` used to de-quantize it.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Flat row-major INT8 payload, length `rows * cols`.
    data: Vec<i8>,
    /// Per-tensor de-quantization scale.
    scale: f32,
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
}

impl QuantizedTensor {
    /// Quantize an `f32` matrix with a single per-tensor scale.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::ShapeMismatch`] if `data.len() != rows * cols`, or
    /// [`AudioError::NonFinite`] (propagated from [`compute_scale_symmetric`]) if
    /// any element is non-finite.
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> AudioResult<Self> {
        if data.len() != rows * cols {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "QuantizedTensor::from_f32: expected {} elements ({rows}×{cols}), got {}",
                    rows * cols,
                    data.len()
                ),
            });
        }
        let scale = compute_scale_symmetric(data)?;
        Ok(Self {
            data: quantize_symmetric(data, scale),
            scale,
            rows,
            cols,
        })
    }

    /// De-quantize back to a flat row-major `f32` matrix.
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_symmetric(&self.data, self.scale)
    }

    /// Per-tensor de-quantization scale.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Borrow the raw INT8 payload (row-major).
    #[must_use]
    pub fn data(&self) -> &[i8] {
        &self.data
    }
}

// ─── Per-channel quantized linear layer ──────────────────────────────────────

/// A linear (dense) layer whose weights are stored in **per-channel** symmetric
/// INT8 and whose forward pass runs as an integer-arithmetic-only GEMM.
///
/// The weight matrix is laid out `[out_features, in_features]` row-major (the
/// PyTorch `nn.Linear` convention), and each output row carries its own scale in
/// `weight_scales[o]`. The forward pass quantizes the activations per tensor,
/// accumulates `Σ q_act · q_w` in INT32, and de-quantizes with the fused scale
/// `act_scale · weight_scales[o]` before adding the optional `bias`.
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Per-channel INT8 weights, row-major `[out_features, in_features]`.
    qweight: Vec<i8>,
    /// Per-output-channel de-quantization scales, length `out_features`.
    weight_scales: Vec<f32>,
    /// Optional bias added in `f32`, length `out_features`.
    bias: Option<Vec<f32>>,
    /// Number of input features (columns of the weight matrix).
    in_features: usize,
    /// Number of output features (rows of the weight matrix).
    out_features: usize,
}

impl QuantizedLinear {
    /// Quantize a floating-point weight matrix into a per-channel INT8 layer.
    ///
    /// `weight` is `[out_features, in_features]` row-major. Each output row is
    /// quantized with its own symmetric scale (per-channel quantization).
    ///
    /// # Errors
    ///
    /// * [`AudioError::InvalidEmbedDim`] if `in_features` or `out_features` is 0.
    /// * [`AudioError::WeightShapeMismatch`] if
    ///   `weight.len() != out_features * in_features`, or if `bias` is present
    ///   with a length other than `out_features`.
    /// * [`AudioError::NonFinite`] if any weight is non-finite.
    pub fn from_f32_weights(
        weight: &[f32],
        bias: Option<&[f32]>,
        out_features: usize,
        in_features: usize,
    ) -> AudioResult<Self> {
        if in_features == 0 {
            return Err(AudioError::InvalidEmbedDim(in_features));
        }
        if out_features == 0 {
            return Err(AudioError::InvalidEmbedDim(out_features));
        }
        if weight.len() != out_features * in_features {
            return Err(AudioError::WeightShapeMismatch {
                msg: format!(
                    "QuantizedLinear: weight has {} elements, expected {} ({out_features}×{in_features})",
                    weight.len(),
                    out_features * in_features
                ),
            });
        }
        if let Some(b) = bias {
            if b.len() != out_features {
                return Err(AudioError::WeightShapeMismatch {
                    msg: format!(
                        "QuantizedLinear: bias has {} elements, expected {out_features}",
                        b.len()
                    ),
                });
            }
            for &v in b {
                if !v.is_finite() {
                    return Err(AudioError::NonFinite {
                        msg: "QuantizedLinear: non-finite bias value".to_string(),
                    });
                }
            }
        }

        let mut qweight = vec![0_i8; out_features * in_features];
        let mut weight_scales = vec![0.0_f32; out_features];
        for o in 0..out_features {
            let row = &weight[o * in_features..(o + 1) * in_features];
            let scale = compute_scale_symmetric(row)?;
            weight_scales[o] = scale;
            let q_row = quantize_symmetric(row, scale);
            qweight[o * in_features..(o + 1) * in_features].copy_from_slice(&q_row);
        }

        Ok(Self {
            qweight,
            weight_scales,
            bias: bias.map(<[f32]>::to_vec),
            in_features,
            out_features,
        })
    }

    /// Number of input features.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Number of output features.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Borrow the per-output-channel weight scales.
    #[must_use]
    pub fn weight_scales(&self) -> &[f32] {
        &self.weight_scales
    }

    /// Integer-arithmetic-only forward pass.
    ///
    /// `x` is `[n_rows, in_features]` row-major. The activations are quantized
    /// per tensor on the fly, the inner products are accumulated in INT32, and
    /// the result is de-quantized with `act_scale · weight_scales[o]` and offset
    /// by the optional bias. Returns `[n_rows, out_features]` row-major.
    ///
    /// # Errors
    ///
    /// * [`AudioError::EmptyInput`] if `n_rows == 0`.
    /// * [`AudioError::DimensionMismatch`] if `x.len() != n_rows * in_features`.
    /// * [`AudioError::NonFinite`] if an activation is non-finite (propagated
    ///   from [`compute_scale_symmetric`]).
    pub fn forward(&self, x: &[f32], n_rows: usize) -> AudioResult<Vec<f32>> {
        if n_rows == 0 {
            return Err(AudioError::EmptyInput {
                msg: "QuantizedLinear::forward: n_rows must be > 0".to_string(),
            });
        }
        if x.len() != n_rows * self.in_features {
            return Err(AudioError::DimensionMismatch {
                expected: n_rows * self.in_features,
                got: x.len(),
            });
        }

        // Per-tensor activation quantization.
        let act_scale = compute_scale_symmetric(x)?;
        let q_act = quantize_symmetric(x, act_scale);

        let mut out = vec![0.0_f32; n_rows * self.out_features];
        for r in 0..n_rows {
            let act_row = &q_act[r * self.in_features..(r + 1) * self.in_features];
            for o in 0..self.out_features {
                let w_row = &self.qweight[o * self.in_features..(o + 1) * self.in_features];
                // Exact INT32 accumulation of INT8 products.
                let mut acc: i32 = 0;
                for (&qa, &qw) in act_row.iter().zip(w_row.iter()) {
                    acc += i32::from(qa) * i32::from(qw);
                }
                // De-quantize with the fused scale, then add bias.
                let mut y = acc as f32 * (act_scale * self.weight_scales[o]);
                if let Some(ref b) = self.bias {
                    y += b[o];
                }
                out[r * self.out_features + o] = y;
            }
        }
        Ok(out)
    }
}

// ─── Private activation ──────────────────────────────────────────────────────

/// Tanh-approximation GELU activation (matching the rest of the encoder crate).
#[inline]
fn gelu_approx(x: f32) -> f32 {
    let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

// ─── Quantized feed-forward network ──────────────────────────────────────────

/// A Conformer-style two-layer feed-forward network whose linear projections run
/// through the INT8 integer-arithmetic-only path.
///
/// The pipeline is `x → Linear₁ → GELU → Linear₂`, with both linear layers
/// quantized per output channel. The intermediate GELU activation is computed in
/// `f32`; only the GEMMs are integerized, exactly as on integer-only inference
/// accelerators that keep element-wise non-linearities in higher precision.
#[derive(Debug, Clone)]
pub struct QuantizedFfn {
    /// First (expanding) projection: `[ffn_dim, embed_dim]`.
    linear1: QuantizedLinear,
    /// Second (contracting) projection: `[embed_dim, ffn_dim]`.
    linear2: QuantizedLinear,
    /// Model (embedding) dimension.
    embed_dim: usize,
    /// Hidden (expanded) dimension.
    ffn_dim: usize,
}

impl QuantizedFfn {
    /// Build a quantized FFN from floating-point weights.
    ///
    /// `w1` is `[ffn_dim, embed_dim]` and `w2` is `[embed_dim, ffn_dim]`, both
    /// row-major. Biases, when present, must match the output dimension of their
    /// layer (`ffn_dim` for `b1`, `embed_dim` for `b2`).
    ///
    /// # Errors
    ///
    /// * [`AudioError::InvalidEmbedDim`] if `embed_dim` or `ffn_dim` is 0.
    /// * Any error propagated from [`QuantizedLinear::from_f32_weights`]
    ///   (shape/bias/finiteness validation).
    pub fn from_f32(
        w1: &[f32],
        b1: Option<&[f32]>,
        w2: &[f32],
        b2: Option<&[f32]>,
        embed_dim: usize,
        ffn_dim: usize,
    ) -> AudioResult<Self> {
        if embed_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(embed_dim));
        }
        if ffn_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(ffn_dim));
        }
        let linear1 = QuantizedLinear::from_f32_weights(w1, b1, ffn_dim, embed_dim)?;
        let linear2 = QuantizedLinear::from_f32_weights(w2, b2, embed_dim, ffn_dim)?;
        Ok(Self {
            linear1,
            linear2,
            embed_dim,
            ffn_dim,
        })
    }

    /// Build a tiny FFN with deterministic LCG-initialised weights for tests.
    ///
    /// Weights are drawn from a Xavier-style uniform range and biases are zero,
    /// using the supplied `seed` for reproducibility.
    ///
    /// # Errors
    ///
    /// Propagates construction errors from [`QuantizedFfn::from_f32`] (none are
    /// expected for valid `embed_dim`/`ffn_dim`).
    pub fn tiny(embed_dim: usize, ffn_dim: usize, seed: u64) -> AudioResult<Self> {
        let mut rng = LcgRng::new(seed);
        let lim1 = (6.0_f32 / (embed_dim + ffn_dim) as f32).sqrt();
        let lim2 = (6.0_f32 / (ffn_dim + embed_dim) as f32).sqrt();

        let mut w1 = vec![0.0_f32; ffn_dim * embed_dim];
        for v in w1.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * lim1;
        }
        let mut w2 = vec![0.0_f32; embed_dim * ffn_dim];
        for v in w2.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * lim2;
        }
        Self::from_f32(&w1, None, &w2, None, embed_dim, ffn_dim)
    }

    /// Model (embedding) dimension.
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Hidden (expanded) dimension.
    #[must_use]
    pub fn ffn_dim(&self) -> usize {
        self.ffn_dim
    }

    /// Run the quantized FFN.
    ///
    /// `x` is `[n_rows, embed_dim]` row-major; returns `[n_rows, embed_dim]`.
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying [`QuantizedLinear::forward`] calls
    /// (empty input, dimension mismatch, non-finite activations).
    pub fn forward(&self, x: &[f32], n_rows: usize) -> AudioResult<Vec<f32>> {
        // Linear₁ → [n_rows, ffn_dim].
        let mut hidden = self.linear1.forward(x, n_rows)?;
        // GELU in f32.
        for v in hidden.iter_mut() {
            *v = gelu_approx(*v);
        }
        // Linear₂ → [n_rows, embed_dim].
        self.linear2.forward(&hidden, n_rows)
    }
}

// ─── Accuracy helper ─────────────────────────────────────────────────────────

/// Root-mean-square error between a reference and a quantized output of equal
/// length.
///
/// This quantifies the accuracy lost to INT8 quantization. Returns `0.0` for two
/// empty inputs.
///
/// # Errors
///
/// Returns [`AudioError::DimensionMismatch`] if the two slices differ in length.
pub fn quantization_error_rms(
    reference_f32_output: &[f32],
    quantized_output: &[f32],
) -> AudioResult<f32> {
    if reference_f32_output.len() != quantized_output.len() {
        return Err(AudioError::DimensionMismatch {
            expected: reference_f32_output.len(),
            got: quantized_output.len(),
        });
    }
    if reference_f32_output.is_empty() {
        return Ok(0.0);
    }
    let mut sum_sq = 0.0_f64;
    for (&r, &q) in reference_f32_output.iter().zip(quantized_output.iter()) {
        let d = f64::from(r) - f64::from(q);
        sum_sq += d * d;
    }
    Ok((sum_sq / reference_f32_output.len() as f64).sqrt() as f32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive f32 reference matmul: `y = x · Wᵀ + b`.
    ///
    /// `x` is `[n_rows, in_f]`, `w` is `[out_f, in_f]` row-major.
    fn reference_linear(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        n_rows: usize,
        out_f: usize,
        in_f: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; n_rows * out_f];
        for r in 0..n_rows {
            for o in 0..out_f {
                let mut acc = bias.map_or(0.0, |b| b[o]);
                for k in 0..in_f {
                    acc += x[r * in_f + k] * w[o * in_f + k];
                }
                out[r * out_f + o] = acc;
            }
        }
        out
    }

    /// Per-tensor INT8 linear forward (single weight scale for the whole matrix),
    /// used only to demonstrate that per-channel quantization is more accurate.
    fn per_tensor_linear_forward(
        weight: &[f32],
        x: &[f32],
        n_rows: usize,
        out_f: usize,
        in_f: usize,
    ) -> Vec<f32> {
        let w_scale = compute_scale_symmetric(weight).expect("finite weight");
        let q_w = quantize_symmetric(weight, w_scale);
        let act_scale = compute_scale_symmetric(x).expect("finite act");
        let q_act = quantize_symmetric(x, act_scale);
        let mut out = vec![0.0_f32; n_rows * out_f];
        for r in 0..n_rows {
            for o in 0..out_f {
                let mut acc: i32 = 0;
                for k in 0..in_f {
                    acc += i32::from(q_act[r * in_f + k]) * i32::from(q_w[o * in_f + k]);
                }
                out[r * out_f + o] = acc as f32 * (act_scale * w_scale);
            }
        }
        out
    }

    #[test]
    fn round_trip_within_half_step() {
        let data = [-1.0_f32, -0.3, 0.0, 0.27, 0.99];
        let scale = compute_scale_symmetric(&data).expect("finite");
        let q = quantize_symmetric(&data, scale);
        let deq = dequantize_symmetric(&q, scale);
        // Quantization error is bounded by half a step (plus tiny float slack).
        for (&orig, &back) in data.iter().zip(deq.iter()) {
            let err = (orig - back).abs();
            assert!(
                err <= 0.5 * scale + 1e-6,
                "round-trip error {err} exceeds half-step {}",
                0.5 * scale
            );
        }
    }

    #[test]
    fn scale_of_all_zeros_is_one_and_quant_is_zero() {
        let zeros = [0.0_f32; 8];
        let scale = compute_scale_symmetric(&zeros).expect("finite");
        assert_eq!(scale, 1.0);
        let q = quantize_symmetric(&zeros, scale);
        assert!(q.iter().all(|&v| v == 0));
        let deq = dequantize_symmetric(&q, scale);
        assert!(deq.iter().all(|&v| v == 0.0 && v.is_finite()));
    }

    #[test]
    fn empty_input_scale_is_one() {
        let empty: [f32; 0] = [];
        assert_eq!(compute_scale_symmetric(&empty).expect("finite"), 1.0);
    }

    #[test]
    fn non_finite_scale_errors() {
        let bad = [1.0_f32, f32::NAN, 2.0];
        assert!(matches!(
            compute_scale_symmetric(&bad),
            Err(AudioError::NonFinite { .. })
        ));
        let inf = [1.0_f32, f32::INFINITY];
        assert!(matches!(
            compute_scale_symmetric(&inf),
            Err(AudioError::NonFinite { .. })
        ));
    }

    #[test]
    fn quantized_tensor_round_trip() {
        let mut rng = LcgRng::new(101);
        let mut data = vec![0.0_f32; 4 * 6];
        rng.fill_normal(&mut data);
        let qt = QuantizedTensor::from_f32(&data, 4, 6).expect("quantize");
        assert_eq!(qt.rows(), 4);
        assert_eq!(qt.cols(), 6);
        assert_eq!(qt.data().len(), 24);
        let back = qt.to_f32();
        for (&orig, &deq) in data.iter().zip(back.iter()) {
            assert!((orig - deq).abs() <= 0.5 * qt.scale() + 1e-6);
        }
    }

    #[test]
    fn quantized_tensor_shape_mismatch() {
        let data = [0.0_f32; 5];
        assert!(matches!(
            QuantizedTensor::from_f32(&data, 2, 3),
            Err(AudioError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn per_channel_linear_low_rms_vs_reference() {
        let out_f = 12;
        let in_f = 16;
        let n_rows = 5;
        let mut rng = LcgRng::new(2024);
        // Unit-scale random weights and activations.
        let mut weight = vec![0.0_f32; out_f * in_f];
        rng.fill_normal(&mut weight);
        let mut x = vec![0.0_f32; n_rows * in_f];
        rng.fill_normal(&mut x);

        let layer = QuantizedLinear::from_f32_weights(&weight, None, out_f, in_f).expect("build");
        let q_out = layer.forward(&x, n_rows).expect("forward");
        let ref_out = reference_linear(&x, &weight, None, n_rows, out_f, in_f);

        let rms = quantization_error_rms(&ref_out, &q_out).expect("rms");
        // 127-level symmetric quantization with unit-scale inputs should be well
        // under 0.05 RMS for this dimensionality.
        assert!(rms < 0.05, "per-channel RMS {rms} too large");
        assert!(q_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn integer_gemm_exact_on_representable_inputs() {
        // Choose weights/acts that quantize EXACTLY: integer multiples of a chosen
        // scale, with max magnitude landing on a grid point so the scale is exact.
        let out_f = 3;
        let in_f = 4;
        let n_rows = 2;

        // Weight integers in {-2,-1,0,1,2}; per-row max magnitude is 2 so the
        // per-channel scale is exactly 2/127, and every weight is on the grid.
        let w_levels: [[i32; 4]; 3] = [[2, -1, 0, 1], [-2, 2, 1, -1], [0, 1, -2, 2]];
        let w_unit = 2.0_f32 / 127.0;
        let mut weight = vec![0.0_f32; out_f * in_f];
        for o in 0..out_f {
            for k in 0..in_f {
                weight[o * in_f + k] = w_levels[o][k] as f32 * w_unit;
            }
        }

        // Activations in {-2,-1,0,1,2}; max magnitude 2 so act scale is 2/127.
        let a_levels: [[i32; 4]; 2] = [[1, -2, 2, 0], [2, 1, -1, -2]];
        let a_unit = 2.0_f32 / 127.0;
        let mut x = vec![0.0_f32; n_rows * in_f];
        for r in 0..n_rows {
            for k in 0..in_f {
                x[r * in_f + k] = a_levels[r][k] as f32 * a_unit;
            }
        }

        let layer = QuantizedLinear::from_f32_weights(&weight, None, out_f, in_f).expect("build");
        let q_out = layer.forward(&x, n_rows).expect("forward");
        let ref_out = reference_linear(&x, &weight, None, n_rows, out_f, in_f);

        // Exact-representable case → integer accumulation reproduces f32 matmul.
        for (&q, &r) in q_out.iter().zip(ref_out.iter()) {
            assert!((q - r).abs() < 1e-4, "quantized {q} vs reference {r}");
        }
    }

    #[test]
    fn per_channel_beats_per_tensor_on_outlier_row() {
        let out_f = 6;
        let in_f = 12;
        let n_rows = 4;
        let mut rng = LcgRng::new(555);

        // Base weights in a modest range; one output row scaled up massively so a
        // single per-tensor scale would be dominated by that row.
        let mut weight = vec![0.0_f32; out_f * in_f];
        for v in weight.iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        let outlier_row = 2;
        for k in 0..in_f {
            weight[outlier_row * in_f + k] *= 50.0;
        }

        let mut x = vec![0.0_f32; n_rows * in_f];
        rng.fill_normal(&mut x);

        let ref_out = reference_linear(&x, &weight, None, n_rows, out_f, in_f);

        let per_channel = QuantizedLinear::from_f32_weights(&weight, None, out_f, in_f)
            .expect("build")
            .forward(&x, n_rows)
            .expect("forward");
        let per_tensor = per_tensor_linear_forward(&weight, &x, n_rows, out_f, in_f);

        let rms_pc = quantization_error_rms(&ref_out, &per_channel).expect("rms pc");
        let rms_pt = quantization_error_rms(&ref_out, &per_tensor).expect("rms pt");

        assert!(
            rms_pc < rms_pt,
            "per-channel RMS {rms_pc} should beat per-tensor RMS {rms_pt}"
        );
    }

    #[test]
    fn quantized_ffn_shape_and_finite() {
        let embed_dim = 8;
        let ffn_dim = 32;
        let n_rows = 6;
        let ffn = QuantizedFfn::tiny(embed_dim, ffn_dim, 7).expect("build");
        assert_eq!(ffn.embed_dim(), embed_dim);
        assert_eq!(ffn.ffn_dim(), ffn_dim);

        let mut rng = LcgRng::new(321);
        let mut x = vec![0.0_f32; n_rows * embed_dim];
        rng.fill_normal(&mut x);

        let out = ffn.forward(&x, n_rows).expect("forward");
        assert_eq!(out.len(), n_rows * embed_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantized_ffn_from_f32_matches_layout() {
        let embed_dim = 4;
        let ffn_dim = 8;
        let n_rows = 3;
        let mut rng = LcgRng::new(909);
        let mut w1 = vec![0.0_f32; ffn_dim * embed_dim];
        rng.fill_normal(&mut w1);
        let mut w2 = vec![0.0_f32; embed_dim * ffn_dim];
        rng.fill_normal(&mut w2);
        let b1 = vec![0.1_f32; ffn_dim];
        let b2 = vec![-0.2_f32; embed_dim];

        let ffn = QuantizedFfn::from_f32(&w1, Some(&b1), &w2, Some(&b2), embed_dim, ffn_dim)
            .expect("build");
        let mut x = vec![0.0_f32; n_rows * embed_dim];
        rng.fill_normal(&mut x);
        let out = ffn.forward(&x, n_rows).expect("forward");
        assert_eq!(out.len(), n_rows * embed_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn wrong_weight_length_errors() {
        let weight = vec![0.0_f32; 10]; // not 3*4
        assert!(matches!(
            QuantizedLinear::from_f32_weights(&weight, None, 3, 4),
            Err(AudioError::WeightShapeMismatch { .. })
        ));
    }

    #[test]
    fn bias_length_mismatch_errors() {
        let weight = vec![0.1_f32; 3 * 4];
        let bias = vec![0.0_f32; 2]; // should be 3
        assert!(matches!(
            QuantizedLinear::from_f32_weights(&weight, Some(&bias), 3, 4),
            Err(AudioError::WeightShapeMismatch { .. })
        ));
    }

    #[test]
    fn zero_features_error() {
        let weight: [f32; 0] = [];
        assert!(matches!(
            QuantizedLinear::from_f32_weights(&weight, None, 0, 4),
            Err(AudioError::InvalidEmbedDim(0))
        ));
        assert!(matches!(
            QuantizedLinear::from_f32_weights(&weight, None, 4, 0),
            Err(AudioError::InvalidEmbedDim(0))
        ));
    }

    #[test]
    fn forward_dimension_mismatch_errors() {
        let weight = vec![0.1_f32; 3 * 4];
        let layer = QuantizedLinear::from_f32_weights(&weight, None, 3, 4).expect("build");
        let x = vec![0.0_f32; 2 * 5]; // wrong inner dim
        assert!(matches!(
            layer.forward(&x, 2),
            Err(AudioError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            layer.forward(&[], 0),
            Err(AudioError::EmptyInput { .. })
        ));
    }

    #[test]
    fn outlier_clamps_to_grid_no_panic() {
        // A value far larger than the rest must clamp to ±127, not overflow.
        let data = [0.01_f32, -0.02, 1.0e6, 0.03];
        let scale = compute_scale_symmetric(&data).expect("finite");
        let q = quantize_symmetric(&data, scale);
        // The outlier defines max magnitude, so it lands exactly on +127.
        assert_eq!(q[2], 127);
        assert!(q.iter().all(|&v| (-127..=127).contains(&i32::from(v))));

        // Now quantize with a deliberately tiny scale so many values saturate.
        let tiny_scale = 1.0e-3_f32;
        let q2 = quantize_symmetric(&data, tiny_scale);
        let deq2 = dequantize_symmetric(&q2, tiny_scale);
        assert_eq!(q2[2], 127);
        assert!((deq2[2] - 127.0 * tiny_scale).abs() < 1e-6);
        assert!(deq2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantize_with_invalid_scale_yields_zero() {
        let data = [1.0_f32, -2.0, 3.0];
        assert!(quantize_symmetric(&data, 0.0).iter().all(|&v| v == 0));
        assert!(quantize_symmetric(&data, -1.0).iter().all(|&v| v == 0));
        assert!(quantize_symmetric(&data, f32::NAN).iter().all(|&v| v == 0));
    }

    #[test]
    fn rms_helper_basics() {
        let a = [1.0_f32, 2.0, 3.0];
        assert_eq!(quantization_error_rms(&a, &a).expect("equal"), 0.0);
        let b = [1.0_f32, 2.0, 5.0];
        // Only the third element differs by 2 → RMS = sqrt(4/3).
        let rms = quantization_error_rms(&a, &b).expect("rms");
        assert!((rms - (4.0_f32 / 3.0).sqrt()).abs() < 1e-6);
        assert!(matches!(
            quantization_error_rms(&a, &[1.0]),
            Err(AudioError::DimensionMismatch { .. })
        ));
        let empty: [f32; 0] = [];
        assert_eq!(quantization_error_rms(&empty, &empty).expect("empty"), 0.0);
    }
}

//! INT8 post-training quantisation for ViT inference.
//!
//! This module implements the *simulated-quantisation* (fake-quant) and the
//! *integer-domain* INT8 inference path used to compress a trained ViT for
//! deployment. The maths is exact CPU `f32`/`i32` arithmetic — there is no GPU
//! tensor-core dependency here; the **FP8 (E4M3) hardware path** advertised in
//! the roadmap remains hardware-gated and is *not* implemented as fake success.
//!
//! ## Quantisation schemes
//! - **Symmetric** (signed, zero-point 0): `q = round(x / s)`, `s = absmax / 127`.
//!   Used for weights and (when activations are roughly zero-centred) for
//!   activations. The representable range is `[-127, 127]` (we reserve `-128`).
//! - **Affine / asymmetric** (unsigned, with zero-point): `q = round(x / s) + z`,
//!   `s = (max − min) / 255`, `z = round(−min / s)`, range `[0, 255]`. Used for
//!   non-negative activations (e.g. post-GELU/ReLU) where a symmetric range
//!   wastes half the codes.
//!
//! ## Per-channel weight quantisation
//! Transformer linear weights `[n_out, n_in]` are quantised **per output
//! channel** (one scale per row), which is the standard recipe that keeps INT8
//! accuracy close to FP32. Activations are quantised **per tensor**.
//!
//! ## Integer matmul
//! [`QuantLinear::forward`] performs `y = (x_q − z_x) Wᵀ_q · (s_x · s_w) + b`
//! by accumulating the integer products in `i32`, then applying the combined
//! `f32` scale per output element — the classic INT8 GEMM dequantisation used by
//! TensorRT / ONNX Runtime / `gemmlowp`.

use crate::error::{VisionError, VisionResult};

/// Smallest signed INT8 magnitude bound used for symmetric quantisation.
///
/// We clamp symmetric codes to `[-127, 127]` (reserving `-128`) so that
/// negation is symmetric and `−q` is always representable.
const SYM_QMAX: i32 = 127;
/// Unsigned affine upper code bound.
const AFFINE_QMAX: i32 = 255;

// ─── Quantisation parameters ───────────────────────────────────────────────────

/// Affine quantisation parameters: real ≈ `scale · (q − zero_point)`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantParams {
    /// Positive real-valued step size.
    pub scale: f32,
    /// Integer zero-point (0 for symmetric quantisation).
    pub zero_point: i32,
    /// Whether the scheme is symmetric (zero-point fixed at 0, signed codes).
    pub symmetric: bool,
}

impl QuantParams {
    /// Derive **symmetric** parameters from a tensor's absolute maximum.
    ///
    /// `scale = absmax / 127`; a fully-zero tensor maps to `scale = 1.0` (an
    /// identity that quantises every value to code 0 and dequantises to 0).
    #[must_use]
    pub fn symmetric_from_absmax(absmax: f32) -> Self {
        let scale = if absmax > 0.0 {
            absmax / SYM_QMAX as f32
        } else {
            1.0
        };
        Self {
            scale,
            zero_point: 0,
            symmetric: true,
        }
    }

    /// Derive **affine** parameters from a tensor's `[min, max]` range.
    ///
    /// The range is widened to include 0 so the zero-point is representable, as
    /// required for correct padding / bias handling.
    #[must_use]
    pub fn affine_from_range(min: f32, max: f32) -> Self {
        let lo = min.min(0.0);
        let hi = max.max(0.0);
        let span = hi - lo;
        let scale = if span > 0.0 {
            span / AFFINE_QMAX as f32
        } else {
            1.0
        };
        // zero_point chosen so that real 0 maps to an integer code in [0, 255].
        let zp = (-lo / scale).round() as i32;
        let zero_point = zp.clamp(0, AFFINE_QMAX);
        Self {
            scale,
            zero_point,
            symmetric: false,
        }
    }

    /// Quantise one real value to its integer code.
    #[must_use]
    #[inline]
    pub fn quantize(&self, x: f32) -> i32 {
        let q = (x / self.scale).round() as i32 + self.zero_point;
        if self.symmetric {
            q.clamp(-SYM_QMAX, SYM_QMAX)
        } else {
            q.clamp(0, AFFINE_QMAX)
        }
    }

    /// Dequantise one integer code back to a real value.
    #[must_use]
    #[inline]
    pub fn dequantize(&self, q: i32) -> f32 {
        (q - self.zero_point) as f32 * self.scale
    }
}

// ─── Tensor-level helpers ──────────────────────────────────────────────────────

/// Compute the absolute maximum of a slice (0 for an empty slice).
#[must_use]
pub fn absmax(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
}

/// Per-tensor **symmetric** quantisation of activations.
///
/// Returns `(codes, params)` where `codes[i] = params.quantize(x[i])`.
///
/// # Errors
/// - [`VisionError::EmptyInput`] if `x` is empty.
/// - [`VisionError::NonFinite`] if `x` contains a non-finite value.
pub fn quantize_tensor_symmetric(x: &[f32]) -> VisionResult<(Vec<i32>, QuantParams)> {
    if x.is_empty() {
        return Err(VisionError::EmptyInput("quantize tensor"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(VisionError::NonFinite("quantize tensor input"));
    }
    let params = QuantParams::symmetric_from_absmax(absmax(x));
    let codes = x.iter().map(|&v| params.quantize(v)).collect();
    Ok((codes, params))
}

/// Per-tensor **affine** quantisation of (typically non-negative) activations.
///
/// # Errors
/// - [`VisionError::EmptyInput`] if `x` is empty.
/// - [`VisionError::NonFinite`] if `x` contains a non-finite value.
pub fn quantize_tensor_affine(x: &[f32]) -> VisionResult<(Vec<i32>, QuantParams)> {
    if x.is_empty() {
        return Err(VisionError::EmptyInput("quantize tensor"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(VisionError::NonFinite("quantize tensor input"));
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in x {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let params = QuantParams::affine_from_range(lo, hi);
    let codes = x.iter().map(|&v| params.quantize(v)).collect();
    Ok((codes, params))
}

/// Dequantise a code vector back to `f32` using shared parameters.
#[must_use]
pub fn dequantize_tensor(codes: &[i32], params: &QuantParams) -> Vec<f32> {
    codes.iter().map(|&q| params.dequantize(q)).collect()
}

/// Fake-quantise (quantise then immediately dequantise) a tensor in `f32`.
///
/// This simulates the precision loss of INT8 inference while staying in the
/// float domain — the standard tool for quantisation-aware evaluation and for
/// measuring per-layer quantisation error before committing to integer kernels.
///
/// # Errors
/// Propagates [`quantize_tensor_symmetric`] validation.
pub fn fake_quantize_symmetric(x: &[f32]) -> VisionResult<Vec<f32>> {
    let (codes, params) = quantize_tensor_symmetric(x)?;
    Ok(dequantize_tensor(&codes, &params))
}

// ─── Quantised weights ─────────────────────────────────────────────────────────

/// INT8 **per-output-channel** symmetric quantisation of a linear weight.
///
/// `weight` is `[n_out, n_in]` row-major; each output row gets its own scale.
#[derive(Debug, Clone)]
pub struct QuantWeight {
    /// Quantised codes `[n_out, n_in]` (values in `[-127, 127]`).
    pub codes: Vec<i32>,
    /// Per-output-channel scale `[n_out]`.
    pub scales: Vec<f32>,
    /// Output channels.
    pub n_out: usize,
    /// Input channels.
    pub n_in: usize,
}

impl QuantWeight {
    /// Quantise a `[n_out, n_in]` weight per output channel (symmetric).
    ///
    /// # Errors
    /// - [`VisionError::EmptyInput`] if `n_out == 0` or `n_in == 0`.
    /// - [`VisionError::DimensionMismatch`] if `weight.len() != n_out * n_in`.
    /// - [`VisionError::NonFinite`] if a weight is non-finite.
    pub fn from_weight(weight: &[f32], n_out: usize, n_in: usize) -> VisionResult<Self> {
        if n_out == 0 || n_in == 0 {
            return Err(VisionError::EmptyInput("quant weight dims"));
        }
        if weight.len() != n_out * n_in {
            return Err(VisionError::DimensionMismatch {
                expected: n_out * n_in,
                got: weight.len(),
            });
        }
        if weight.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("quant weight input"));
        }
        let mut codes = vec![0i32; n_out * n_in];
        let mut scales = vec![1.0f32; n_out];
        for o in 0..n_out {
            let row = &weight[o * n_in..(o + 1) * n_in];
            let params = QuantParams::symmetric_from_absmax(absmax(row));
            scales[o] = params.scale;
            let dst = &mut codes[o * n_in..(o + 1) * n_in];
            for (d, &w) in dst.iter_mut().zip(row.iter()) {
                *d = params.quantize(w);
            }
        }
        Ok(Self {
            codes,
            scales,
            n_out,
            n_in,
        })
    }

    /// Reconstruct the `f32` weight by dequantising every channel.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_out * self.n_in];
        for o in 0..self.n_out {
            let s = self.scales[o];
            let src = &self.codes[o * self.n_in..(o + 1) * self.n_in];
            let dst = &mut out[o * self.n_in..(o + 1) * self.n_in];
            for (d, &q) in dst.iter_mut().zip(src.iter()) {
                *d = q as f32 * s;
            }
        }
        out
    }
}

// ─── Quantised linear layer ────────────────────────────────────────────────────

/// INT8 quantised linear (dense) layer with `f32` bias and dequantised output.
///
/// Weights are stored per-channel quantised at construction; activations are
/// quantised per-tensor (symmetric) at call time. The matmul accumulates integer
/// products in `i32` and applies the fused scale `s_x · s_w[o]` per output.
#[derive(Debug, Clone)]
pub struct QuantLinear {
    weight: QuantWeight,
    bias: Vec<f32>,
}

impl QuantLinear {
    /// Build a quantised linear layer from a float weight `[n_out, n_in]` and an
    /// optional bias `[n_out]` (all-zero if `None`).
    ///
    /// # Errors
    /// - Propagates [`QuantWeight::from_weight`] validation.
    /// - [`VisionError::DimensionMismatch`] if a supplied bias has the wrong
    ///   length.
    pub fn new(
        weight: &[f32],
        bias: Option<&[f32]>,
        n_out: usize,
        n_in: usize,
    ) -> VisionResult<Self> {
        let qw = QuantWeight::from_weight(weight, n_out, n_in)?;
        let bias = match bias {
            Some(b) => {
                if b.len() != n_out {
                    return Err(VisionError::DimensionMismatch {
                        expected: n_out,
                        got: b.len(),
                    });
                }
                b.to_vec()
            }
            None => vec![0.0f32; n_out],
        };
        Ok(Self { weight: qw, bias })
    }

    /// Output dimension.
    #[must_use]
    #[inline]
    pub fn n_out(&self) -> usize {
        self.weight.n_out
    }

    /// Input dimension.
    #[must_use]
    #[inline]
    pub fn n_in(&self) -> usize {
        self.weight.n_in
    }

    /// Borrow the underlying quantised weight.
    #[must_use]
    pub fn weight(&self) -> &QuantWeight {
        &self.weight
    }

    /// Quantised forward pass over `x` = `[batch, n_in]` → `[batch, n_out]`.
    ///
    /// Activations are quantised symmetrically per **row** (per token), which
    /// is the standard dynamic-quantisation scheme: each token gets a fresh
    /// activation scale. The integer dot product is `Σ_k x_q[k] · w_q[o,k]`,
    /// dequantised by `s_x · s_w[o]` and offset by the float bias.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `x.len()` is not a multiple of
    ///   `n_in`.
    /// - [`VisionError::NonFinite`] if an activation is non-finite.
    pub fn forward(&self, x: &[f32]) -> VisionResult<Vec<f32>> {
        let n_in = self.weight.n_in;
        let n_out = self.weight.n_out;
        if n_in == 0 || x.len() % n_in != 0 {
            return Err(VisionError::DimensionMismatch {
                expected: n_in,
                got: x.len(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("quant linear activation"));
        }
        let batch = x.len() / n_in;
        let mut out = vec![0.0f32; batch * n_out];

        let mut x_q = vec![0i32; n_in];
        for bi in 0..batch {
            let row = &x[bi * n_in..(bi + 1) * n_in];
            // Dynamic per-row activation scale (symmetric).
            let a_params = QuantParams::symmetric_from_absmax(absmax(row));
            for (q, &v) in x_q.iter_mut().zip(row.iter()) {
                *q = a_params.quantize(v);
            }
            let s_x = a_params.scale;

            let o_row = &mut out[bi * n_out..(bi + 1) * n_out];
            for (o, o_val) in o_row.iter_mut().enumerate() {
                let w_row = &self.weight.codes[o * n_in..(o + 1) * n_in];
                let mut acc: i32 = 0;
                for (&xq, &wq) in x_q.iter().zip(w_row.iter()) {
                    acc += xq * wq;
                }
                let s = s_x * self.weight.scales[o];
                *o_val = acc as f32 * s + self.bias[o];
            }
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn symmetric_params_roundtrip_zero() {
        let p = QuantParams::symmetric_from_absmax(0.0);
        assert_eq!(p.zero_point, 0);
        assert_eq!(p.quantize(0.0), 0);
        assert!((p.dequantize(0)).abs() < 1e-9);
    }

    #[test]
    fn symmetric_endpoints_map_to_127() {
        let p = QuantParams::symmetric_from_absmax(2.0);
        // absmax maps to ±127.
        assert_eq!(p.quantize(2.0), 127);
        assert_eq!(p.quantize(-2.0), -127);
        // Out-of-range clamps.
        assert_eq!(p.quantize(100.0), 127);
        assert_eq!(p.quantize(-100.0), -127);
    }

    #[test]
    fn affine_zero_is_representable() {
        let p = QuantParams::affine_from_range(-1.0, 3.0);
        // Real 0 should dequantise back to ~0.
        let q0 = p.quantize(0.0);
        assert!(p.dequantize(q0).abs() < p.scale, "zero not representable");
        // Codes stay in [0, 255].
        assert!((0..=255).contains(&p.quantize(3.0)));
        assert!((0..=255).contains(&p.quantize(-1.0)));
    }

    #[test]
    fn symmetric_quantisation_error_bounded() {
        // Round-trip error must be ≤ half a step (scale/2) plus tiny float slack.
        let mut rng = LcgRng::new(1);
        let mut x = vec![0.0f32; 256];
        rng.fill_normal(&mut x);
        let (codes, params) = quantize_tensor_symmetric(&x).expect("ok");
        let deq = dequantize_tensor(&codes, &params);
        let half_step = params.scale * 0.5 + 1e-6;
        for (a, b) in x.iter().zip(deq.iter()) {
            assert!(
                (a - b).abs() <= half_step,
                "quant error {} exceeds half-step {half_step}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn fake_quant_preserves_shape_and_finite() {
        let mut rng = LcgRng::new(2);
        let mut x = vec![0.0f32; 64];
        rng.fill_normal(&mut x);
        let fq = fake_quantize_symmetric(&x).expect("ok");
        assert_eq!(fq.len(), x.len());
        assert!(fq.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantize_rejects_empty_and_nonfinite() {
        assert!(quantize_tensor_symmetric(&[]).is_err());
        assert!(quantize_tensor_symmetric(&[1.0, f32::NAN]).is_err());
        assert!(quantize_tensor_affine(&[f32::INFINITY]).is_err());
    }

    #[test]
    fn per_channel_weight_roundtrip() {
        let n_out = 3;
        let n_in = 4;
        // Channel magnitudes differ widely; per-channel scaling should handle it.
        let weight = vec![
            0.01, -0.02, 0.015, -0.005, // small channel
            10.0, -8.0, 6.0, -4.0, // large channel
            1.0, -1.0, 0.5, -0.5, // medium channel
        ];
        let qw = QuantWeight::from_weight(&weight, n_out, n_in).expect("ok");
        assert_eq!(qw.scales.len(), n_out);
        let deq = qw.dequantize();
        for o in 0..n_out {
            let s = qw.scales[o];
            for k in 0..n_in {
                let orig = weight[o * n_in + k];
                let got = deq[o * n_in + k];
                assert!(
                    (orig - got).abs() <= s * 0.5 + 1e-6,
                    "channel {o} idx {k}: {orig} vs {got}"
                );
            }
        }
        // Large channel has a larger scale than the small channel.
        assert!(qw.scales[1] > qw.scales[0]);
    }

    #[test]
    fn quant_weight_validation() {
        assert!(QuantWeight::from_weight(&[1.0; 4], 0, 4).is_err());
        assert!(QuantWeight::from_weight(&[1.0; 4], 2, 4).is_err()); // 2*4 != 4
        assert!(QuantWeight::from_weight(&[1.0, f32::NAN, 2.0, 3.0], 2, 2).is_err());
    }

    #[test]
    fn quant_linear_close_to_float() {
        // INT8 quantised linear must approximate the float matmul.
        let n_out = 8;
        let n_in = 16;
        let mut rng = LcgRng::new(3);
        let mut weight = vec![0.0f32; n_out * n_in];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= 0.25; // keep magnitudes modest
        }
        let bias: Vec<f32> = (0..n_out).map(|i| 0.1 * i as f32).collect();
        let ql = QuantLinear::new(&weight, Some(&bias), n_out, n_in).expect("ok");

        let batch = 4;
        let mut x = vec![0.0f32; batch * n_in];
        rng.fill_normal(&mut x);

        // Float reference.
        let mut ref_out = vec![0.0f32; batch * n_out];
        for b in 0..batch {
            for o in 0..n_out {
                let mut acc = bias[o];
                for k in 0..n_in {
                    acc += x[b * n_in + k] * weight[o * n_in + k];
                }
                ref_out[b * n_out + o] = acc;
            }
        }

        let q_out = ql.forward(&x).expect("ok");
        assert_eq!(q_out.len(), ref_out.len());
        // Measure aggregate fidelity: RMS error relative to RMS signal. Per-element
        // relative error is meaningless near zero crossings, so we use the
        // energy-normalised error, which is the standard quantisation-SNR metric.
        let mut err_sq = 0.0f64;
        let mut sig_sq = 0.0f64;
        for (q, r) in q_out.iter().zip(ref_out.iter()) {
            err_sq += ((q - r) as f64).powi(2);
            sig_sq += (*r as f64).powi(2);
        }
        let rel_rms = (err_sq / sig_sq.max(1e-12)).sqrt();
        assert!(
            rel_rms < 0.1,
            "INT8 linear normalised RMS error too high: {rel_rms}"
        );
    }

    #[test]
    fn quant_linear_bias_handling() {
        // Zero weight → output equals bias exactly.
        let n_out = 3;
        let n_in = 5;
        let weight = vec![0.0f32; n_out * n_in];
        let bias = vec![1.0f32, -2.0, 3.5];
        let ql = QuantLinear::new(&weight, Some(&bias), n_out, n_in).expect("ok");
        let x = vec![0.5f32; 2 * n_in];
        let out = ql.forward(&x).expect("ok");
        for b in 0..2 {
            for o in 0..n_out {
                assert!((out[b * n_out + o] - bias[o]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn quant_linear_default_bias_and_validation() {
        let n_out = 2;
        let n_in = 3;
        let weight = vec![0.1f32; n_out * n_in];
        // No bias supplied → zeros.
        let ql = QuantLinear::new(&weight, None, n_out, n_in).expect("ok");
        assert_eq!(ql.n_out(), n_out);
        assert_eq!(ql.n_in(), n_in);
        // Wrong bias length.
        assert!(QuantLinear::new(&weight, Some(&[1.0]), n_out, n_in).is_err());
        // Wrong activation length.
        assert!(ql.forward(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn quant_linear_deterministic() {
        let n_out = 4;
        let n_in = 6;
        let mut rng = LcgRng::new(4);
        let mut weight = vec![0.0f32; n_out * n_in];
        rng.fill_normal(&mut weight);
        let ql = QuantLinear::new(&weight, None, n_out, n_in).expect("ok");
        let x = vec![0.3f32; 3 * n_in];
        let a = ql.forward(&x).expect("ok");
        let b = ql.forward(&x).expect("ok");
        assert_eq!(a, b);
    }
}

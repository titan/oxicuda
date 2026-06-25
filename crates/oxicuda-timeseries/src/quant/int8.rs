//! INT8 post-training quantisation inference path for time-series models.
//!
//! Implements **symmetric** linear quantisation for weight matrices and
//! activations, as used to accelerate TCN / PatchTST inference. The scheme is
//! the standard affine-free (zero-point = 0) mapping:
//!
//! ```text
//!   q = clamp(round(x / scale), -127, 127)        (quantise)
//!   x̂ = q · scale                                 (dequantise)
//! ```
//!
//! Two granularities are supported:
//!
//! * **Per-tensor**: a single `scale` for the whole weight matrix.
//! * **Per-output-channel**: one `scale` per output row of a `[out, in]` weight
//!   matrix (the granularity that recovers most accuracy in practice).
//!
//! A quantised linear layer ([`QuantLinear`]) stores INT8 weights plus the
//! float scales and performs an integer-domain dot product, rescaling the
//! accumulator back to float. This is a *faithful* simulation of an INT8 GEMM
//! inference kernel: the dot products are accumulated in `i32`, exactly as an
//! integer tensor-core / DP4A path would, then dequantised.
//!
//! Reference scheme: Jacob et al. 2018, "Quantization and Training of Neural
//! Networks for Efficient Integer-Arithmetic-Only Inference" (symmetric variant).
//!
//! Pure-Rust CPU reference. No external crates.

use crate::error::{TsError, TsResult};

/// Quantisation granularity for a weight matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantGranularity {
    /// One shared scale for the entire tensor.
    PerTensor,
    /// One scale per output channel (row of a `[out, in]` matrix).
    PerChannel,
}

/// Configuration controlling INT8 quantisation.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Granularity of weight quantisation.
    pub granularity: QuantGranularity,
    /// Clamp range bound; INT8 uses 127 (symmetric, reserves -128).
    pub qmax: i32,
}

impl QuantConfig {
    /// Default INT8 per-channel configuration (`qmax = 127`).
    #[must_use]
    pub fn int8_per_channel() -> Self {
        Self {
            granularity: QuantGranularity::PerChannel,
            qmax: 127,
        }
    }

    /// Default INT8 per-tensor configuration (`qmax = 127`).
    #[must_use]
    pub fn int8_per_tensor() -> Self {
        Self {
            granularity: QuantGranularity::PerTensor,
            qmax: 127,
        }
    }
}

/// Compute a symmetric quantisation scale from a max-abs value.
///
/// `scale = max_abs / qmax`, with a floor to avoid a zero scale on all-zero
/// inputs (which would make dequantisation divide-by-zero downstream).
#[must_use]
fn symmetric_scale(max_abs: f32, qmax: i32) -> f32 {
    let s = max_abs / qmax as f32;
    if s <= 0.0 || !s.is_finite() { 1.0 } else { s }
}

/// Quantise one float value to an INT8 code given a scale.
#[inline]
#[must_use]
fn quantise_value(x: f32, scale: f32, qmax: i32) -> i8 {
    let q = (x / scale).round() as i32;
    q.clamp(-qmax, qmax) as i8
}

/// Quantise a contiguous activation/weight slice to INT8 with a single scale.
///
/// Returns the INT8 codes. The scale is computed from the slice's max-abs.
#[must_use]
pub fn quantise_tensor(x: &[f32], qmax: i32) -> (Vec<i8>, f32) {
    let max_abs = x.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let scale = symmetric_scale(max_abs, qmax);
    let q = x.iter().map(|&v| quantise_value(v, scale, qmax)).collect();
    (q, scale)
}

/// Dequantise INT8 codes back to float using a single scale.
#[must_use]
pub fn dequantise_tensor(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|&c| c as f32 * scale).collect()
}

/// An INT8-quantised linear (fully-connected / 1×1 conv) layer.
///
/// Holds INT8 weights for a `[out_dim, in_dim]` row-major matrix together with
/// per-channel (or single per-tensor) scales and a float bias. Inference runs
/// an integer dot product with `i32` accumulation, then dequantises.
#[derive(Debug, Clone)]
pub struct QuantLinear {
    /// INT8 weights `[out_dim, in_dim]` row-major.
    pub q_weight: Vec<i8>,
    /// Weight scales: length `out_dim` (per-channel) or `1` (per-tensor).
    pub w_scales: Vec<f32>,
    /// Float bias `[out_dim]`.
    pub bias: Vec<f32>,
    /// Output dimension.
    pub out_dim: usize,
    /// Input dimension.
    pub in_dim: usize,
    /// Granularity used for weights.
    pub granularity: QuantGranularity,
    /// Clamp bound.
    pub qmax: i32,
}

impl QuantLinear {
    /// Quantise a float weight matrix `[out_dim, in_dim]` + bias into a
    /// [`QuantLinear`] under the given [`QuantConfig`].
    ///
    /// # Errors
    ///
    /// - [`TsError::WeightShapeMismatch`] when `weight.len() != out_dim * in_dim`
    ///   or `bias.len() != out_dim`.
    /// - [`TsError::InvalidEmbedDim`] when `out_dim == 0` or `in_dim == 0`.
    pub fn quantise(
        weight: &[f32],
        bias: &[f32],
        out_dim: usize,
        in_dim: usize,
        config: &QuantConfig,
    ) -> TsResult<Self> {
        if out_dim == 0 || in_dim == 0 {
            return Err(TsError::InvalidEmbedDim(out_dim.min(in_dim)));
        }
        if weight.len() != out_dim * in_dim {
            return Err(TsError::WeightShapeMismatch {
                msg: format!(
                    "weight len {} != out_dim*in_dim {}",
                    weight.len(),
                    out_dim * in_dim
                ),
            });
        }
        if bias.len() != out_dim {
            return Err(TsError::WeightShapeMismatch {
                msg: format!("bias len {} != out_dim {out_dim}", bias.len()),
            });
        }

        let qmax = config.qmax.max(1);
        let mut q_weight = vec![0_i8; out_dim * in_dim];
        let w_scales = match config.granularity {
            QuantGranularity::PerTensor => {
                let (q, s) = quantise_tensor(weight, qmax);
                q_weight.copy_from_slice(&q);
                vec![s]
            }
            QuantGranularity::PerChannel => {
                let mut scales = vec![1.0_f32; out_dim];
                for o in 0..out_dim {
                    let row = &weight[o * in_dim..(o + 1) * in_dim];
                    let max_abs = row.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
                    let s = symmetric_scale(max_abs, qmax);
                    scales[o] = s;
                    for (i, &w) in row.iter().enumerate() {
                        q_weight[o * in_dim + i] = quantise_value(w, s, qmax);
                    }
                }
                scales
            }
        };

        Ok(Self {
            q_weight,
            w_scales,
            bias: bias.to_vec(),
            out_dim,
            in_dim,
            granularity: config.granularity,
            qmax,
        })
    }

    /// Return the per-output-channel weight scale.
    #[inline]
    fn weight_scale(&self, out_channel: usize) -> f32 {
        match self.granularity {
            QuantGranularity::PerTensor => self.w_scales[0],
            QuantGranularity::PerChannel => self.w_scales[out_channel],
        }
    }

    /// Dequantise the stored INT8 weights back to a float `[out_dim, in_dim]`
    /// matrix (useful for measuring quantisation error vs the original).
    #[must_use]
    pub fn dequantise_weights(&self) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.out_dim * self.in_dim];
        for o in 0..self.out_dim {
            let s = self.weight_scale(o);
            for i in 0..self.in_dim {
                out[o * self.in_dim + i] = self.q_weight[o * self.in_dim + i] as f32 * s;
            }
        }
        out
    }

    /// Run the quantised layer on a single float input vector `[in_dim]`,
    /// returning a float `[out_dim]`.
    ///
    /// The input activation is dynamically quantised per-call (per-tensor over
    /// the input vector), then an integer dot product is accumulated in `i32`
    /// and rescaled by `act_scale · weight_scale[o]`. This mirrors a true
    /// INT8 inference kernel.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != in_dim`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        if x.len() != self.in_dim {
            return Err(TsError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        let (q_x, act_scale) = quantise_tensor(x, self.qmax);

        let mut out = vec![0.0_f32; self.out_dim];
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &self.q_weight[o * self.in_dim..(o + 1) * self.in_dim];
            // Integer accumulation in i32 (DP4A-style).
            let mut acc: i32 = 0;
            for (i, &qw) in row.iter().enumerate() {
                acc += qw as i32 * q_x[i] as i32;
            }
            let dequant = acc as f32 * act_scale * self.weight_scale(o);
            *slot = dequant + self.bias[o];
        }
        Ok(out)
    }

    /// Run the quantised layer over a batch of `n` input rows `[n, in_dim]`,
    /// returning `[n, out_dim]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != n * in_dim`.
    pub fn forward_batch(&self, x: &[f32], n: usize) -> TsResult<Vec<f32>> {
        if x.len() != n * self.in_dim {
            return Err(TsError::DimensionMismatch {
                expected: n * self.in_dim,
                got: x.len(),
            });
        }
        let mut out = vec![0.0_f32; n * self.out_dim];
        for r in 0..n {
            let row_in = &x[r * self.in_dim..(r + 1) * self.in_dim];
            let row_out = self.forward(row_in)?;
            out[r * self.out_dim..(r + 1) * self.out_dim].copy_from_slice(&row_out);
        }
        Ok(out)
    }
}

/// Compute the relative quantisation error between an original float weight
/// matrix and the dequantised INT8 reconstruction.
///
/// Returns the **relative Frobenius error** `‖W − Ŵ‖_F / (‖W‖_F + eps)`.
///
/// # Errors
///
/// - [`TsError::DimensionMismatch`] when the two slices differ in length.
pub fn relative_quant_error(original: &[f32], reconstructed: &[f32]) -> TsResult<f32> {
    if original.len() != reconstructed.len() {
        return Err(TsError::DimensionMismatch {
            expected: original.len(),
            got: reconstructed.len(),
        });
    }
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (&a, &b) in original.iter().zip(reconstructed.iter()) {
        let d = (a - b) as f64;
        num += d * d;
        den += (a as f64) * (a as f64);
    }
    Ok((num.sqrt() / (den.sqrt() + 1e-12)) as f32)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn quantise_dequantise_roundtrip_bounded_error() {
        let mut rng = make_rng();
        let mut x = vec![0.0_f32; 256];
        rng.fill_normal(&mut x);
        let (q, scale) = quantise_tensor(&x, 127);
        let recon = dequantise_tensor(&q, scale);
        // Max error per element is at most half a quantisation step = scale/2.
        for (&orig, &rec) in x.iter().zip(recon.iter()) {
            assert!(
                (orig - rec).abs() <= scale * 0.5 + 1e-6,
                "quant error {} exceeds half-step {}",
                (orig - rec).abs(),
                scale * 0.5
            );
        }
    }

    #[test]
    fn quantise_codes_within_int8_range() {
        let x: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.7).collect();
        let (q, _) = quantise_tensor(&x, 127);
        assert!(q.iter().all(|&c| (-127..=127).contains(&(c as i32))));
    }

    #[test]
    fn quantise_all_zeros_safe() {
        let x = vec![0.0_f32; 16];
        let (q, scale) = quantise_tensor(&x, 127);
        assert!(scale > 0.0 && scale.is_finite());
        assert!(q.iter().all(|&c| c == 0));
        let recon = dequantise_tensor(&q, scale);
        assert!(recon.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn quant_linear_matches_float_within_tolerance() {
        let mut rng = make_rng();
        let out_dim = 8;
        let in_dim = 32;
        let mut weight = vec![0.0_f32; out_dim * in_dim];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= 0.2;
        }
        let bias: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.01).collect();

        let ql = QuantLinear::quantise(
            &weight,
            &bias,
            out_dim,
            in_dim,
            &QuantConfig::int8_per_channel(),
        )
        .expect("quantise");

        let mut x = vec![0.0_f32; in_dim];
        rng.fill_normal(&mut x);

        // Float reference.
        let mut reference = vec![0.0_f32; out_dim];
        for o in 0..out_dim {
            let mut acc = bias[o];
            for i in 0..in_dim {
                acc += x[i] * weight[o * in_dim + i];
            }
            reference[o] = acc;
        }

        let quantised = ql.forward(&x).expect("forward");

        // INT8 quantisation of both weights and activations introduces error;
        // require the result to be close in relative terms.
        let mut num = 0.0_f32;
        let mut den = 0.0_f32;
        for o in 0..out_dim {
            num += (reference[o] - quantised[o]).powi(2);
            den += reference[o].powi(2);
        }
        let rel = (num.sqrt()) / (den.sqrt() + 1e-9);
        assert!(rel < 0.1, "INT8 linear relative error too high: {rel}");
    }

    #[test]
    fn quant_linear_per_channel_better_than_per_tensor_on_skewed() {
        // Build a weight matrix where rows have very different magnitudes; per-
        // channel scaling should reconstruct it more accurately than per-tensor.
        let out_dim = 4;
        let in_dim = 16;
        let mut weight = vec![0.0_f32; out_dim * in_dim];
        for o in 0..out_dim {
            let mag = 10.0_f32.powi(o as i32); // 1, 10, 100, 1000
            for i in 0..in_dim {
                weight[o * in_dim + i] = ((i as f32) - 8.0) / 8.0 * mag;
            }
        }
        let bias = vec![0.0_f32; out_dim];

        let pc = QuantLinear::quantise(
            &weight,
            &bias,
            out_dim,
            in_dim,
            &QuantConfig::int8_per_channel(),
        )
        .expect("pc");
        let pt = QuantLinear::quantise(
            &weight,
            &bias,
            out_dim,
            in_dim,
            &QuantConfig::int8_per_tensor(),
        )
        .expect("pt");

        let err_pc = relative_quant_error(&weight, &pc.dequantise_weights()).expect("err pc");
        let err_pt = relative_quant_error(&weight, &pt.dequantise_weights()).expect("err pt");

        assert!(
            err_pc < err_pt,
            "per-channel error {err_pc} should beat per-tensor {err_pt}"
        );
    }

    #[test]
    fn quant_linear_batch_consistent_with_single() {
        let mut rng = make_rng();
        let out_dim = 6;
        let in_dim = 12;
        let mut weight = vec![0.0_f32; out_dim * in_dim];
        rng.fill_normal(&mut weight);
        let bias = vec![0.0_f32; out_dim];
        let ql = QuantLinear::quantise(
            &weight,
            &bias,
            out_dim,
            in_dim,
            &QuantConfig::int8_per_channel(),
        )
        .expect("q");

        let n = 5;
        let mut x = vec![0.0_f32; n * in_dim];
        rng.fill_normal(&mut x);
        let batch = ql.forward_batch(&x, n).expect("batch");

        for r in 0..n {
            let single = ql
                .forward(&x[r * in_dim..(r + 1) * in_dim])
                .expect("single");
            for o in 0..out_dim {
                assert!((batch[r * out_dim + o] - single[o]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn relative_quant_error_zero_for_identity() {
        let w = vec![1.0_f32, -2.0, 3.0, 4.0];
        let e = relative_quant_error(&w, &w).expect("e");
        assert!(e.abs() < 1e-6);
    }

    #[test]
    fn quant_linear_err_bad_weight_shape() {
        let weight = vec![0.0_f32; 10];
        let bias = vec![0.0_f32; 4];
        assert!(matches!(
            QuantLinear::quantise(&weight, &bias, 4, 8, &QuantConfig::int8_per_channel())
                .unwrap_err(),
            TsError::WeightShapeMismatch { .. }
        ));
    }

    #[test]
    fn quant_linear_err_bad_bias_shape() {
        let weight = vec![0.0_f32; 32];
        let bias = vec![0.0_f32; 3];
        assert!(matches!(
            QuantLinear::quantise(&weight, &bias, 4, 8, &QuantConfig::int8_per_channel())
                .unwrap_err(),
            TsError::WeightShapeMismatch { .. }
        ));
    }

    #[test]
    fn quant_linear_err_zero_dim() {
        let weight: Vec<f32> = vec![];
        let bias: Vec<f32> = vec![];
        assert!(matches!(
            QuantLinear::quantise(&weight, &bias, 0, 8, &QuantConfig::int8_per_channel())
                .unwrap_err(),
            TsError::InvalidEmbedDim(_)
        ));
    }

    #[test]
    fn quant_linear_err_bad_input_len() {
        let weight = vec![0.0_f32; 32];
        let bias = vec![0.0_f32; 4];
        let ql = QuantLinear::quantise(&weight, &bias, 4, 8, &QuantConfig::int8_per_channel())
            .expect("q");
        let x = vec![0.0_f32; 5];
        assert!(matches!(
            ql.forward(&x).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn relative_quant_error_err_mismatch() {
        let a = vec![1.0_f32; 4];
        let b = vec![1.0_f32; 3];
        assert!(matches!(
            relative_quant_error(&a, &b).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }
}

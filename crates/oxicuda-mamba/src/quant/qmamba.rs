//! Q-Mamba — symmetric INT8 post-training quantization (PTQ) for SSM weights.
//!
//! # Background
//!
//! Deploying Mamba / SSM models cheaply benefits from **INT8 post-training
//! quantization**: the floating-point projection matrices (`in_proj`,
//! `out_proj`, `x_proj`, …) and the SSM parameters (the diagonal `A`, the
//! per-channel `D` skip, etc.) are stored as 8-bit integers plus a small set of
//! `f32` scales, and dequantized on the fly during inference.
//!
//! This module implements **symmetric** quantization with the integer range
//! `[-127, 127]` (the value `-128` is dropped so the grid is symmetric about
//! zero, which keeps `dequant(0) == 0` exactly).  Two granularities are
//! supported:
//!
//! * [`QuantScheme::PerTensor`] — a single scale for the whole tensor.
//! * [`QuantScheme::PerChannel`] — one scale per output channel (matrix row),
//!   which typically reduces error when rows have very different magnitudes.
//!
//! For a chosen scale `s`, an element `x` is encoded as
//!
//! ```text
//! q = clamp( round(x / s), −127, 127 )       (i8)
//! x̂ = q · s                                  (dequantized estimate)
//! ```
//!
//! with `s = amax / 127` (`amax` = maximum absolute value in the tensor / row).
//! Round-to-nearest bounds the per-element error by `s / 2` — the property the
//! unit tests verify.  A dequantized linear forward
//! `y[i] = s_i · Σ_j q[i, j] · x[j]` reproduces the `f32` matrix-vector product
//! to within that quantization error.
//!
//! All arithmetic is `f32` to match the other kernels in this crate.

use crate::error::{MambaError, MambaResult};

/// Symmetric INT8 magnitude limit (`-128` is intentionally excluded).
pub const Q_MAX: i32 = 127;

// ─── QuantScheme ─────────────────────────────────────────────────────────────

/// Granularity of the quantization scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// One shared scale for the entire tensor.
    PerTensor,
    /// One scale per output channel (matrix row).
    PerChannel,
}

// ─── Scale / element helpers ─────────────────────────────────────────────────

/// Maximum absolute value of a slice (`0` for an empty slice).
#[inline]
fn amax(data: &[f32]) -> f32 {
    data.iter().fold(0.0_f32, |m, &v| m.max(v.abs()))
}

/// Symmetric scale `amax / 127`; falls back to `1.0` when the slice is all-zero
/// so that dequantization is exact (every code is `0`).
#[inline]
fn scale_from_amax(amax_val: f32) -> f32 {
    if amax_val > 0.0 {
        amax_val / Q_MAX as f32
    } else {
        1.0
    }
}

/// Encode a single value with the given (positive) scale.
#[inline]
fn quantize_value(x: f32, scale: f32) -> i8 {
    let q = (x / scale).round();
    let clamped = q.clamp(-(Q_MAX as f32), Q_MAX as f32);
    clamped as i8
}

// ─── QuantizedTensor ─────────────────────────────────────────────────────────

/// An INT8-quantized 2-D tensor (`rows × cols`, row-major) plus its scales.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    /// Quantized codes, row-major `[rows × cols]`, each in `[-127, 127]`.
    pub q: Vec<i8>,
    /// Dequantization scales: length `1` for per-tensor, `rows` for per-channel.
    pub scales: Vec<f32>,
    /// Number of rows (output channels).
    pub rows: usize,
    /// Number of columns (input features).
    pub cols: usize,
    /// Granularity used to produce `scales`.
    pub scheme: QuantScheme,
}

impl QuantizedTensor {
    /// Quantize a row-major `f32` matrix `[rows × cols]`.
    ///
    /// # Errors
    ///
    /// * [`MambaError::EmptyInput`]        — if `data` is empty.
    /// * [`MambaError::DimensionMismatch`] — if `rows · cols != data.len()`.
    pub fn quantize(
        data: &[f32],
        rows: usize,
        cols: usize,
        scheme: QuantScheme,
    ) -> MambaResult<Self> {
        if data.is_empty() {
            return Err(MambaError::EmptyInput("data"));
        }
        if rows == 0 || cols == 0 || rows * cols != data.len() {
            return Err(MambaError::DimensionMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }

        let mut q = vec![0_i8; data.len()];
        let scales = match scheme {
            QuantScheme::PerTensor => {
                let scale = scale_from_amax(amax(data));
                for (qi, &xi) in q.iter_mut().zip(data.iter()) {
                    *qi = quantize_value(xi, scale);
                }
                vec![scale]
            }
            QuantScheme::PerChannel => {
                let mut scales = Vec::with_capacity(rows);
                for r in 0..rows {
                    let row = &data[r * cols..(r + 1) * cols];
                    let scale = scale_from_amax(amax(row));
                    let q_row = &mut q[r * cols..(r + 1) * cols];
                    for (qi, &xi) in q_row.iter_mut().zip(row.iter()) {
                        *qi = quantize_value(xi, scale);
                    }
                    scales.push(scale);
                }
                scales
            }
        };

        Ok(Self {
            q,
            scales,
            rows,
            cols,
            scheme,
        })
    }

    /// The dequantization scale used for output row `row`.
    #[inline]
    fn scale_for_row(&self, row: usize) -> f32 {
        match self.scheme {
            QuantScheme::PerTensor => self.scales[0],
            QuantScheme::PerChannel => self.scales[row],
        }
    }

    /// Reconstruct the `f32` matrix `x̂ = q · s` (row-major `[rows × cols]`).
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.q.len()];
        for r in 0..self.rows {
            let scale = self.scale_for_row(r);
            let base = r * self.cols;
            let dst = &mut out[base..base + self.cols];
            let src = &self.q[base..base + self.cols];
            for (o, &qv) in dst.iter_mut().zip(src.iter()) {
                *o = qv as f32 * scale;
            }
        }
        out
    }

    /// Dequantized linear forward `y = Ŵ · x` where `Ŵ` is this quantized
    /// weight (`rows` outputs, `cols` inputs).
    ///
    /// Computed as `y[i] = s_i · Σ_j q[i, j] · x[j]` (integer accumulation
    /// followed by a single scale multiply per output).
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if `x.len() != cols`.
    pub fn matvec(&self, x: &[f32]) -> MambaResult<Vec<f32>> {
        if x.len() != self.cols {
            return Err(MambaError::DimensionMismatch {
                expected: self.cols,
                got: x.len(),
            });
        }
        let mut y = vec![0.0_f32; self.rows];
        for (r, yi) in y.iter_mut().enumerate() {
            let scale = self.scale_for_row(r);
            let row = &self.q[r * self.cols..(r + 1) * self.cols];
            let acc: f32 = row
                .iter()
                .zip(x.iter())
                .map(|(&qv, &xv)| qv as f32 * xv)
                .sum();
            *yi = scale * acc;
        }
        Ok(y)
    }

    /// Maximum absolute reconstruction error `max |x − x̂|` against the original
    /// `f32` data (useful for diagnostics / tests).
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if `original.len() != rows · cols`.
    pub fn reconstruction_error(&self, original: &[f32]) -> MambaResult<f32> {
        if original.len() != self.q.len() {
            return Err(MambaError::DimensionMismatch {
                expected: self.q.len(),
                got: original.len(),
            });
        }
        let deq = self.dequantize();
        Ok(original
            .iter()
            .zip(deq.iter())
            .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs())))
    }
}

// ─── QMambaQuantizer ─────────────────────────────────────────────────────────

/// Driver for quantizing the weights of a Mamba / SSM model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QMambaQuantizer {
    /// Quantization granularity applied to matrices.
    pub scheme: QuantScheme,
}

impl QMambaQuantizer {
    /// Create a quantizer with the given scheme.
    #[must_use]
    pub fn new(scheme: QuantScheme) -> Self {
        Self { scheme }
    }

    /// Quantize a projection weight matrix `[rows × cols]` with this scheme.
    ///
    /// # Errors
    ///
    /// Propagates [`QuantizedTensor::quantize`] validation errors.
    pub fn quantize_matrix(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> MambaResult<QuantizedTensor> {
        QuantizedTensor::quantize(data, rows, cols, self.scheme)
    }

    /// Quantize a 1-D SSM parameter vector (e.g. the diagonal `A` or the `D`
    /// skip) as a single `1 × n` per-tensor tensor.
    ///
    /// # Errors
    ///
    /// Propagates [`QuantizedTensor::quantize`] validation errors.
    pub fn quantize_vector(&self, data: &[f32]) -> MambaResult<QuantizedTensor> {
        QuantizedTensor::quantize(data, 1, data.len(), QuantScheme::PerTensor)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    // ── INT8 range ────────────────────────────────────────────────────────────

    #[test]
    fn int8_range_respected() {
        let mut rng = LcgRng::new(1);
        // Scale up well beyond the grid to force clamping behaviour.
        let data: Vec<f32> = randn(&mut rng, 64).iter().map(|v| v * 100.0).collect();
        for scheme in [QuantScheme::PerTensor, QuantScheme::PerChannel] {
            let qt = QuantizedTensor::quantize(&data, 8, 8, scheme).expect("quantize");
            for &qv in &qt.q {
                assert!(
                    (qv as i32).abs() <= Q_MAX,
                    "code {qv} exceeds INT8 symmetric range"
                );
            }
        }
    }

    // ── Round-trip error bounded by the step size ─────────────────────────────

    #[test]
    fn round_trip_error_bounded_per_tensor() {
        let mut rng = LcgRng::new(2);
        let data = randn(&mut rng, 100);
        let qt = QuantizedTensor::quantize(&data, 10, 10, QuantScheme::PerTensor).expect("q");
        let scale = qt.scales[0];
        let err = qt.reconstruction_error(&data).expect("err");
        // Round-to-nearest ⇒ |x − x̂| ≤ scale/2.
        assert!(
            err <= 0.5 * scale + 1e-6,
            "max error {err} exceeds scale/2 = {}",
            0.5 * scale
        );
    }

    #[test]
    fn round_trip_error_bounded_per_channel() {
        let mut rng = LcgRng::new(3);
        let rows = 6;
        let cols = 7;
        let data = randn(&mut rng, rows * cols);
        let qt = QuantizedTensor::quantize(&data, rows, cols, QuantScheme::PerChannel).expect("q");
        let deq = qt.dequantize();
        for r in 0..rows {
            let scale = qt.scales[r];
            for c in 0..cols {
                let idx = r * cols + c;
                let e = (data[idx] - deq[idx]).abs();
                assert!(
                    e <= 0.5 * scale + 1e-6,
                    "row {r} col {c}: error {e} > scale/2 = {}",
                    0.5 * scale
                );
            }
        }
    }

    // ── Per-channel scales are correct ────────────────────────────────────────

    #[test]
    fn per_channel_scales_correct() {
        // Row 0 amax = 1.0, row 1 amax = 0.1, row 2 amax = 4.0.
        let data = vec![
            1.0, -0.5, 0.25, // row 0
            0.1, -0.08, 0.02, // row 1
            -4.0, 2.0, 1.0, // row 2
        ];
        let qt = QuantizedTensor::quantize(&data, 3, 3, QuantScheme::PerChannel).expect("q");
        assert_eq!(qt.scales.len(), 3);
        assert!((qt.scales[0] - 1.0 / Q_MAX as f32).abs() < 1e-9);
        assert!((qt.scales[1] - 0.1 / Q_MAX as f32).abs() < 1e-9);
        assert!((qt.scales[2] - 4.0 / Q_MAX as f32).abs() < 1e-9);
    }

    #[test]
    fn per_tensor_scale_is_global_amax() {
        let data = vec![0.5_f32, -2.0, 1.0, 0.25];
        let qt = QuantizedTensor::quantize(&data, 1, 4, QuantScheme::PerTensor).expect("q");
        assert_eq!(qt.scales.len(), 1);
        assert!((qt.scales[0] - 2.0 / Q_MAX as f32).abs() < 1e-9);
        // The max-magnitude element saturates to ±127.
        assert!(qt.q.contains(&(-Q_MAX as i8)));
    }

    // ── Per-channel reduces error vs per-tensor for heterogeneous rows ────────

    #[test]
    fn per_channel_not_worse_than_per_tensor() {
        // One large-magnitude row and one tiny row: per-tensor wastes codes on
        // the tiny row, so per-channel error should be ≤ per-tensor error.
        let data = vec![
            10.0, -8.0, 6.0, // large row
            0.03, -0.02, 0.01, // tiny row
        ];
        let pt = QuantizedTensor::quantize(&data, 2, 3, QuantScheme::PerTensor).expect("pt");
        let pc = QuantizedTensor::quantize(&data, 2, 3, QuantScheme::PerChannel).expect("pc");
        let e_pt = pt.reconstruction_error(&data).expect("e_pt");
        let e_pc = pc.reconstruction_error(&data).expect("e_pc");
        assert!(
            e_pc <= e_pt + 1e-6,
            "per-channel error {e_pc} should not exceed per-tensor {e_pt}"
        );
    }

    // ── Zero tensor round-trips to exactly zero ───────────────────────────────

    #[test]
    fn zero_tensor_roundtrips_zero() {
        let data = vec![0.0_f32; 12];
        let qt = QuantizedTensor::quantize(&data, 3, 4, QuantScheme::PerTensor).expect("q");
        assert!(qt.q.iter().all(|&v| v == 0));
        assert!(
            (qt.scales[0] - 1.0).abs() < 1e-9,
            "zero tensor ⇒ unit scale"
        );
        let deq = qt.dequantize();
        assert!(deq.iter().all(|&v| v == 0.0));
    }

    // ── Dequantized forward: finite & close to fp32 for small magnitudes ──────

    #[test]
    fn quantized_forward_close_to_fp32() {
        let mut rng = LcgRng::new(4);
        let rows = 5;
        let cols = 6;
        // Small magnitudes (×0.1) as the prompt specifies.
        let w: Vec<f32> = randn(&mut rng, rows * cols)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let x: Vec<f32> = randn(&mut rng, cols).iter().map(|v| v * 0.1).collect();

        let qt = QuantizedTensor::quantize(&w, rows, cols, QuantScheme::PerChannel).expect("q");
        let y_q = qt.matvec(&x).expect("matvec");

        // Reference fp32 matrix-vector product.
        let sum_abs_x: f32 = x.iter().map(|v| v.abs()).sum();
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let y_ref: f32 = row.iter().zip(x.iter()).map(|(&wv, &xv)| wv * xv).sum();
            assert!(y_q[r].is_finite(), "y_q[{r}] must be finite");
            // |error| ≤ (scale_r / 2) · Σ_j |x_j|.
            let bound = 0.5 * qt.scales[r] * sum_abs_x + 1e-5;
            assert!(
                (y_q[r] - y_ref).abs() <= bound,
                "row {r}: |{} − {y_ref}| > bound {bound}",
                y_q[r]
            );
        }
    }

    #[test]
    fn matvec_per_tensor_finite_and_shaped() {
        let mut rng = LcgRng::new(5);
        let rows = 4;
        let cols = 3;
        let w = randn(&mut rng, rows * cols);
        let x = randn(&mut rng, cols);
        let qt = QuantizedTensor::quantize(&w, rows, cols, QuantScheme::PerTensor).expect("q");
        let y = qt.matvec(&x).expect("matvec");
        assert_eq!(y.len(), rows);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ── SSM parameter (1-D) quantization ──────────────────────────────────────

    #[test]
    fn ssm_param_vector_round_trip() {
        // A diagonal A vector (negative, S5/HiPPO-LegS style).
        let a_diag: Vec<f32> = (0..8).map(|n| -((n + 1) as f32)).collect();
        let quantizer = QMambaQuantizer::new(QuantScheme::PerTensor);
        let qt = quantizer.quantize_vector(&a_diag).expect("q");
        assert_eq!(qt.rows, 1);
        assert_eq!(qt.cols, 8);
        let scale = qt.scales[0];
        let err = qt.reconstruction_error(&a_diag).expect("err");
        assert!(err <= 0.5 * scale + 1e-6, "A-diag error {err} > scale/2");
        assert!(qt.q.iter().all(|&v| (v as i32).abs() <= Q_MAX));
    }

    // ── Quantizer driver wraps the chosen scheme ──────────────────────────────

    #[test]
    fn quantizer_matrix_uses_scheme() {
        let mut rng = LcgRng::new(6);
        let data = randn(&mut rng, 12);
        let q_pc = QMambaQuantizer::new(QuantScheme::PerChannel)
            .quantize_matrix(&data, 3, 4)
            .expect("pc");
        assert_eq!(q_pc.scheme, QuantScheme::PerChannel);
        assert_eq!(q_pc.scales.len(), 3);
        let q_pt = QMambaQuantizer::new(QuantScheme::PerTensor)
            .quantize_matrix(&data, 3, 4)
            .expect("pt");
        assert_eq!(q_pt.scheme, QuantScheme::PerTensor);
        assert_eq!(q_pt.scales.len(), 1);
    }

    // ── Error handling ────────────────────────────────────────────────────────

    #[test]
    fn quantize_errors() {
        assert!(matches!(
            QuantizedTensor::quantize(&[], 0, 0, QuantScheme::PerTensor),
            Err(MambaError::EmptyInput(_))
        ));
        assert!(matches!(
            QuantizedTensor::quantize(&[1.0, 2.0, 3.0], 2, 2, QuantScheme::PerTensor),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn matvec_wrong_input_len_errors() {
        let data = vec![0.1_f32; 6];
        let qt = QuantizedTensor::quantize(&data, 2, 3, QuantScheme::PerChannel).expect("q");
        assert!(matches!(
            qt.matvec(&[1.0, 2.0]), // cols = 3, given 2
            Err(MambaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn reconstruction_error_wrong_len_errors() {
        let data = vec![0.1_f32; 6];
        let qt = QuantizedTensor::quantize(&data, 2, 3, QuantScheme::PerTensor).expect("q");
        assert!(matches!(
            qt.reconstruction_error(&[0.0; 5]),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }
}

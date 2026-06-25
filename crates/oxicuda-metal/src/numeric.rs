//! Host-side numeric support for the extended-precision and quantised kernels.
//!
//! Two pure-Rust helpers that pair with the MSL generators in
//! [`crate::msl_nn`]:
//!
//! * [`Int8Quantizer`] — per-tensor symmetric / asymmetric INT8 dynamic
//!   quantisation: derive the scale (and optional zero point) from a slice of
//!   `f32`, quantise to `i8`, and dequantise back.  This produces exactly the
//!   `scale`/`zero` constants the [`crate::msl_nn::int8_quant_gemm_msl`] kernel
//!   consumes.
//!
//! * [`DoubleSingle`] — a double-single (`df64`) value carried as two `f32`
//!   limbs, with the same Dekker/Knuth arithmetic the
//!   [`crate::msl_nn::gemm_msl_f64_ds`] kernel performs on the GPU.  Used to
//!   prepare/verify FP64-emulated GEMM operands on the host.

use crate::error::{MetalError, MetalResult};

// ─── INT8 dynamic quantisation ─────────────────────────────────────────────────

/// Result of quantising an `f32` tensor to INT8.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    /// Quantised values in `[-127, 127]` (symmetric) or `[-128, 127]`.
    pub values: Vec<i8>,
    /// Dequantisation scale: `real ≈ (q - zero_point) * scale`.
    pub scale: f32,
    /// Zero point (0 for symmetric quantisation).
    pub zero_point: i32,
}

/// Per-tensor INT8 dynamic quantiser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8Quantizer {
    /// Symmetric: zero point is 0, range maps to `[-127, 127]`.
    Symmetric,
    /// Asymmetric (affine): zero point chosen so the min maps to `-128`.
    Asymmetric,
}

impl Int8Quantizer {
    /// Quantise `data` to INT8, deriving the scale (and zero point) from its
    /// dynamic range.
    ///
    /// Returns [`MetalError::InvalidArgument`] for an empty slice or
    /// non-finite inputs.
    pub fn quantize(self, data: &[f32]) -> MetalResult<QuantizedTensor> {
        if data.is_empty() {
            return Err(MetalError::InvalidArgument(
                "cannot quantise an empty tensor".into(),
            ));
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(MetalError::InvalidArgument(
                "tensor contains non-finite values".into(),
            ));
        }

        match self {
            Self::Symmetric => {
                let absmax = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                // A constant-zero tensor quantises to all zeros with unit scale.
                let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                let inv = 1.0 / scale;
                let values = data.iter().map(|&v| clamp_i8((v * inv).round())).collect();
                Ok(QuantizedTensor {
                    values,
                    scale,
                    zero_point: 0,
                })
            }
            Self::Asymmetric => {
                let mut min = f32::INFINITY;
                let mut max = f32::NEG_INFINITY;
                for &v in data {
                    min = min.min(v);
                    max = max.max(v);
                }
                let range = max - min;
                let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
                // Map min → -128: zero_point = -128 - round(min/scale).
                let zero_point = clamp_i8_int(-128 - (min / scale).round() as i32);
                let inv = 1.0 / scale;
                let values = data
                    .iter()
                    .map(|&v| clamp_i8((v * inv).round() + zero_point as f32))
                    .collect();
                Ok(QuantizedTensor {
                    values,
                    scale,
                    zero_point,
                })
            }
        }
    }
}

impl QuantizedTensor {
    /// Dequantise back to `f32`: `(q - zero_point) * scale`.
    pub fn dequantize(&self) -> Vec<f32> {
        self.values
            .iter()
            .map(|&q| (i32::from(q) - self.zero_point) as f32 * self.scale)
            .collect()
    }

    /// Maximum absolute reconstruction error against an original tensor.
    ///
    /// Returns [`MetalError::InvalidArgument`] on a length mismatch.
    pub fn max_abs_error(&self, original: &[f32]) -> MetalResult<f32> {
        if original.len() != self.values.len() {
            return Err(MetalError::InvalidArgument(
                "length mismatch in max_abs_error".into(),
            ));
        }
        let deq = self.dequantize();
        Ok(deq
            .iter()
            .zip(original)
            .fold(0.0f32, |m, (&d, &o)| m.max((d - o).abs())))
    }
}

#[inline]
fn clamp_i8(v: f32) -> i8 {
    if v >= 127.0 {
        127
    } else if v <= -128.0 {
        -128
    } else {
        v as i8
    }
}

#[inline]
fn clamp_i8_int(v: i32) -> i32 {
    v.clamp(-128, 127)
}

// ─── Double-single (df64) emulated FP64 ────────────────────────────────────────

/// A double-single value: an unevaluated sum of two `f32` limbs (`hi + lo`).
///
/// Mirrors the `df64` struct used by [`crate::msl_nn::gemm_msl_f64_ds`].  Gives
/// roughly 44 bits of mantissa precision using only `f32` storage and ops,
/// which is how the Metal kernel emulates FP64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleSingle {
    /// High-order limb (the rounded value).
    pub hi: f32,
    /// Low-order limb (the rounding error).
    pub lo: f32,
}

impl DoubleSingle {
    /// The additive identity (`0`).
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    /// Construct from a single `f32` (the low limb is zero).
    pub fn from_f32(a: f32) -> Self {
        Self { hi: a, lo: 0.0 }
    }

    /// Split an `f64` into two `f32` limbs (hi = nearest f32, lo = residual).
    pub fn from_f64(a: f64) -> Self {
        let hi = a as f32;
        let lo = (a - hi as f64) as f32;
        Self { hi, lo }
    }

    /// Reconstruct an approximate `f64` from the two limbs.
    pub fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }

    /// Knuth's two-sum: rounded sum plus its exact rounding error.
    fn two_sum(a: f32, b: f32) -> Self {
        let s = a + b;
        let bb = s - a;
        let err = (a - (s - bb)) + (b - bb);
        Self { hi: s, lo: err }
    }

    /// Dekker's two-product using fused multiply-add for the error term.
    fn two_prod(a: f32, b: f32) -> Self {
        let p = a * b;
        let err = a.mul_add(b, -p);
        Self { hi: p, lo: err }
    }
}

impl std::ops::Add for DoubleSingle {
    type Output = Self;

    /// Extended-precision addition.
    fn add(self, other: Self) -> Self {
        let s = Self::two_sum(self.hi, other.hi);
        let lo = s.lo + (self.lo + other.lo);
        Self::two_sum(s.hi, lo)
    }
}

impl std::ops::Mul for DoubleSingle {
    type Output = Self;

    /// Extended-precision multiplication.
    fn mul(self, other: Self) -> Self {
        let p = Self::two_prod(self.hi, other.hi);
        let lo = p.lo + (self.hi * other.lo + self.lo * other.hi);
        Self::two_sum(p.hi, lo)
    }
}

impl std::ops::AddAssign for DoubleSingle {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::MulAssign for DoubleSingle {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

/// Split a slice of `f64` into interleaved `[hi, lo]` `f32` pairs, the storage
/// layout the `gemm_f64_ds` kernel expects for its `float2` buffers.
pub fn pack_df64(data: &[f64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        let ds = DoubleSingle::from_f64(v);
        out.push(ds.hi);
        out.push(ds.lo);
    }
    out
}

/// Reconstruct a slice of `f64` from interleaved `[hi, lo]` `f32` pairs.
///
/// Returns [`MetalError::InvalidArgument`] if the input length is odd.
pub fn unpack_df64(data: &[f32]) -> MetalResult<Vec<f64>> {
    if data.len() % 2 != 0 {
        return Err(MetalError::InvalidArgument(
            "df64 packed buffer length must be even".into(),
        ));
    }
    Ok(data
        .chunks_exact(2)
        .map(|c| DoubleSingle { hi: c[0], lo: c[1] }.to_f64())
        .collect())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── INT8 quantisation ──
    #[test]
    fn symmetric_quantise_roundtrip() {
        let data = [1.0f32, -2.0, 3.0, -4.0, 0.5];
        let q = Int8Quantizer::Symmetric.quantize(&data).expect("quantise");
        assert_eq!(q.zero_point, 0);
        // absmax = 4.0 → scale = 4/127; the -4.0 maps to -127.
        assert_eq!(*q.values.iter().min().unwrap(), -127);
        let err = q.max_abs_error(&data).expect("error");
        // Error is bounded by half a quantisation step (~scale/2).
        assert!(err <= q.scale, "err {err} scale {}", q.scale);
    }

    #[test]
    fn symmetric_constant_tensor() {
        let data = [0.0f32; 8];
        let q = Int8Quantizer::Symmetric.quantize(&data).expect("quantise");
        assert_eq!(q.scale, 1.0);
        assert!(q.values.iter().all(|&v| v == 0));
        assert!(q.dequantize().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn asymmetric_quantise_covers_range() {
        let data = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let q = Int8Quantizer::Asymmetric.quantize(&data).expect("quantise");
        let err = q.max_abs_error(&data).expect("error");
        assert!(err <= q.scale, "err {err} scale {}", q.scale);
    }

    #[test]
    fn quantise_empty_and_nonfinite_error() {
        assert!(Int8Quantizer::Symmetric.quantize(&[]).is_err());
        assert!(Int8Quantizer::Symmetric.quantize(&[1.0, f32::NAN]).is_err());
        assert!(
            Int8Quantizer::Symmetric
                .quantize(&[1.0, f32::INFINITY])
                .is_err()
        );
    }

    #[test]
    fn max_abs_error_length_mismatch() {
        let q = Int8Quantizer::Symmetric.quantize(&[1.0, 2.0]).expect("q");
        assert!(q.max_abs_error(&[1.0]).is_err());
    }

    #[test]
    fn clamp_helpers() {
        assert_eq!(clamp_i8(200.0), 127);
        assert_eq!(clamp_i8(-200.0), -128);
        assert_eq!(clamp_i8(5.0), 5);
        assert_eq!(clamp_i8_int(500), 127);
        assert_eq!(clamp_i8_int(-500), -128);
    }

    // ── Double-single ──
    #[test]
    fn df64_from_to_f32() {
        let d = DoubleSingle::from_f32(3.5);
        assert_eq!(d.hi, 3.5);
        assert_eq!(d.lo, 0.0);
        assert_eq!(DoubleSingle::ZERO.to_f64(), 0.0);
    }

    #[test]
    fn df64_add_more_precise_than_f32() {
        // 1.0 + 1e-8: a single f32 add loses the small term entirely.
        let a = DoubleSingle::from_f64(1.0);
        let b = DoubleSingle::from_f64(1e-8);
        let sum = a + b;
        let err = (sum.to_f64() - (1.0 + 1e-8)).abs();
        // Double-single retains far more than naive f32 (~1e-7 ulp at 1.0).
        assert!(err < 1e-10, "df64 add error too large: {err}");
    }

    #[test]
    fn df64_mul_two_prod_is_exact_for_small_ints() {
        let a = DoubleSingle::from_f32(123.0);
        let b = DoubleSingle::from_f32(456.0);
        let p = a * b;
        // 123 * 456 = 56088, exactly representable.
        assert_eq!(p.to_f64(), 56088.0);
    }

    #[test]
    fn df64_accumulation_beats_f32() {
        // Sum many small values onto a large one; f32 would stagnate.
        let mut acc = DoubleSingle::from_f64(1_000_000.0);
        let inc = DoubleSingle::from_f64(0.1);
        for _ in 0..1000 {
            acc += inc;
        }
        let expected = 1_000_000.0 + 0.1 * 1000.0;
        let err = (acc.to_f64() - expected).abs();
        assert!(err < 1e-3, "accumulation error {err}");
    }

    #[test]
    fn pack_unpack_df64_roundtrip() {
        let data = [1.0f64, 2.5, -3.25, 1e-9];
        let packed = pack_df64(&data);
        assert_eq!(packed.len(), 8); // 2 limbs per value
        let unpacked = unpack_df64(&packed).expect("unpack");
        for (got, want) in unpacked.iter().zip(data.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got} want {want}");
        }
    }

    #[test]
    fn unpack_df64_odd_length_errors() {
        assert!(unpack_df64(&[1.0, 2.0, 3.0]).is_err());
    }
}

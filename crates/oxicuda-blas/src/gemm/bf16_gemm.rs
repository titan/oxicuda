//! Brain Float 16 (BF16) GEMM.
//!
//! BF16 (Brain Float 16, introduced by Google 2019) uses 16 bits with the
//! same exponent field as IEEE 754 `f32` (8 bits) but only 7 mantissa bits
//! instead of 23.  This gives BF16 the same dynamic range as `f32` with
//! roughly half the storage, at the cost of lower precision.
//!
//! ## Software implementation
//!
//! This module provides a *software* BF16 GEMM that:
//! 1. Converts `f32` inputs to BF16 by truncating the lower 16 bits of the
//!    IEEE 754 bit-pattern (round-to-zero / truncation).
//! 2. Converts each BF16 value back to `f32` for arithmetic (zero-extend).
//! 3. Accumulates products in `f32` precision.
//!
//! This faithfully models the precision loss of true BF16 hardware while
//! remaining portable pure Rust.
//!
//! # Reference
//! - Wang et al. (2019) "BFloat16: The secret to high performance on Cloud
//!   TPUs". Google AI Blog.

use crate::error::{BlasError, BlasResult};

// ---------------------------------------------------------------------------
// BF16 type
// ---------------------------------------------------------------------------

/// A Brain Float 16 value stored as its raw `u16` bit pattern.
///
/// The bit layout is: `[sign(1) | exponent(8) | mantissa(7)]`, matching the
/// upper 16 bits of an IEEE 754 `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16(u16);

impl Bf16 {
    /// Convert an `f32` to BF16 by truncating (rounding toward zero).
    ///
    /// The upper 16 bits of the 32-bit float bit-pattern become the BF16.
    /// Special values (NaN, ±∞, ±0) are preserved exactly.
    ///
    /// Note: this is round-to-zero (truncation), not round-to-nearest-even.
    /// For a production implementation, round-to-nearest-even is preferred;
    /// here we prioritise simplicity and transparency of the approximation.
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        // Cast to u32 bits, keep upper 16
        let bits = x.to_bits();
        Bf16((bits >> 16) as u16)
    }

    /// Convert BF16 back to `f32` by zero-extending the lower 16 bits.
    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// The BF16 representation of `+0.0`.
    #[inline]
    pub fn zero() -> Self {
        Bf16(0)
    }

    /// Returns `true` if this value is a NaN.
    ///
    /// BF16 NaN: exponent field all-ones AND mantissa non-zero.
    #[inline]
    pub fn is_nan(self) -> bool {
        // Exponent = bits [14:7], mantissa = bits [6:0]
        let exp = (self.0 >> 7) & 0xFF;
        let mantissa = self.0 & 0x7F;
        exp == 0xFF && mantissa != 0
    }

    /// Returns `true` if this value is ±infinity.
    ///
    /// BF16 infinity: exponent all-ones AND mantissa zero.
    #[inline]
    pub fn is_inf(self) -> bool {
        let exp = (self.0 >> 7) & 0xFF;
        let mantissa = self.0 & 0x7F;
        exp == 0xFF && mantissa == 0
    }
}

// ---------------------------------------------------------------------------
// BF16 GEMM
// ---------------------------------------------------------------------------

/// BF16 GEMM: `C = alpha * A @ B + beta * C`.
///
/// All matrices are provided and returned as `f32` slices (row-major).
/// Internally the inputs `A` and `B` are downcast to BF16 before multiplication;
/// accumulation is done in `f32`.
///
/// # Arguments
///
/// * `m`, `n`, `k` — Matrix dimensions: A is `[m × k]`, B is `[k × n]`, C is `[m × n]`.
/// * `alpha` — Scalar multiplier for `A @ B`.
/// * `a` — Row-major `f32` slice of length `m * k`.
/// * `b` — Row-major `f32` slice of length `k * n`.
/// * `beta` — Scalar multiplier for `C` (applied before adding `alpha * A @ B`).
/// * `c` — Row-major `f32` slice of length `m * n`, updated in place.
///
/// # Errors
///
/// - [`BlasError::InvalidDimension`] if any of `m`, `n`, `k` is zero.
/// - [`BlasError::BufferTooSmall`] if `a`, `b`, or `c` is too short.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_bf16(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> BlasResult<()> {
    // --- Dimension validation -----------------------------------------------
    if m == 0 || n == 0 || k == 0 {
        return Err(BlasError::InvalidDimension(format!(
            "dimensions must be non-zero: m={m}, n={n}, k={k}"
        )));
    }
    if a.len() < m * k {
        return Err(BlasError::BufferTooSmall {
            expected: m * k,
            actual: a.len(),
        });
    }
    if b.len() < k * n {
        return Err(BlasError::BufferTooSmall {
            expected: k * n,
            actual: b.len(),
        });
    }
    if c.len() < m * n {
        return Err(BlasError::BufferTooSmall {
            expected: m * n,
            actual: c.len(),
        });
    }

    // --- Convert A and B to BF16 -------------------------------------------
    let a_bf16: Vec<Bf16> = a[..m * k].iter().map(|&x| Bf16::from_f32(x)).collect();
    let b_bf16: Vec<Bf16> = b[..k * n].iter().map(|&x| Bf16::from_f32(x)).collect();

    // --- GEMM kernel -------------------------------------------------------
    for i in 0..m {
        for j in 0..n {
            // Accumulate dot product in f32
            let mut acc = 0.0_f32;
            for p in 0..k {
                let a_val = a_bf16[i * k + p].to_f32();
                let b_val = b_bf16[p * n + j].to_f32();
                acc += a_val * b_val;
            }
            c[i * n + j] = alpha * acc + beta * c[i * n + j];
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Error measurement helper
// ---------------------------------------------------------------------------

/// Compute the maximum absolute error between BF16 GEMM and an exact `f32`
/// reference GEMM for the problem `C = A @ B` (alpha=1, beta=0).
///
/// This is a diagnostic utility for evaluating the precision loss introduced
/// by BF16 truncation.
///
/// # Arguments
///
/// * `m`, `n`, `k` — Matrix dimensions.
/// * `a` — Row-major `f32` input of shape `[m × k]`.
/// * `b` — Row-major `f32` input of shape `[k × n]`.
///
/// # Returns
///
/// The maximum absolute difference `max_{i,j} |C_bf16[i,j] - C_f32[i,j]|`.
/// Returns `0.0` if the product matrix is empty (`m == 0` or `n == 0`) or the
/// contraction dimension is empty (`k == 0`), since both GEMMs reduce to an
/// all-zero (or zero-length) result in those cases.
///
/// # Panics
///
/// Panics if `a` or `b` are shorter than `m*k` / `k*n` respectively (for
/// non-zero `m`, `n`, `k`).  Call `sgemm_bf16` for validated execution.
pub fn bf16_gemm_error(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> f32 {
    // Degenerate dimensions: the product matrix is empty (m or n is zero) or
    // the contraction is over zero terms (k == 0). Both GEMMs are trivially
    // equal (all-zero or zero-length) in these cases, so the error is 0.0.
    // This also avoids ever calling `sgemm_bf16` with a zero dimension,
    // which is the only way it can fail below (see comment there).
    if m == 0 || n == 0 || k == 0 {
        return 0.0;
    }

    // Reference: f32 GEMM
    let mut c_ref = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c_ref[i * n + j] = acc;
        }
    }

    // BF16 GEMM
    let mut c_bf16 = vec![0.0_f32; m * n];
    if sgemm_bf16(m, n, k, 1.0, a, b, 0.0, &mut c_bf16).is_err() {
        // Unreachable in practice: `m`, `n`, `k` are all non-zero (checked
        // above), so the only remaining failure mode is `BufferTooSmall` —
        // but the `c_ref` loop above already indexed `a` and `b` up to the
        // same bounds `sgemm_bf16` validates, so if either were too short
        // this function would already have panicked on the raw slice index.
        // Fall back to the documented degenerate-case value rather than
        // panicking a second time via `.expect()`.
        return 0.0;
    }

    // Max absolute error
    c_ref
        .iter()
        .zip(c_bf16.iter())
        .map(|(r, b)| (r - b).abs())
        .fold(0.0_f32, f32::max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_from_f32_roundtrip() {
        let values = [
            0.0_f32,
            1.0,
            -1.0,
            std::f32::consts::PI,
            1e-4,
            1e4,
            0.123_456_79_f32,
        ];
        for &v in &values {
            let recovered = Bf16::from_f32(v).to_f32();
            // BF16 has ~2 decimal digits of precision (relative error ≈ 1/128 ≈ 0.78%)
            let rtol = 1e-2_f32;
            let err = (recovered - v).abs();
            let tol = rtol * v.abs().max(1e-6);
            assert!(
                err <= tol || v.abs() < 1e-30,
                "v={v}, recovered={recovered}, err={err}, tol={tol}"
            );
        }
    }

    #[test]
    fn bf16_zero() {
        let z = Bf16::zero();
        assert_eq!(z.to_f32(), 0.0_f32);
        assert_eq!(z.0, 0);
    }

    #[test]
    fn bf16_nan_propagates() {
        let nan = Bf16::from_f32(f32::NAN);
        assert!(nan.is_nan(), "BF16(NaN).is_nan() should be true");
        assert!(nan.to_f32().is_nan(), "BF16(NaN).to_f32() should be NaN");
    }

    #[test]
    fn bf16_inf() {
        let inf = Bf16::from_f32(f32::INFINITY);
        assert!(inf.is_inf(), "BF16(+inf).is_inf() should be true");
        let ninf = Bf16::from_f32(f32::NEG_INFINITY);
        assert!(ninf.is_inf(), "BF16(-inf).is_inf() should be true");
    }

    #[test]
    fn sgemm_bf16_output_shape() {
        let m = 4_usize;
        let n = 5_usize;
        let k = 3_usize;
        let a = vec![1.0_f32; m * k];
        let b = vec![1.0_f32; k * n];
        let mut c = vec![0.0_f32; m * n];
        sgemm_bf16(m, n, k, 1.0, &a, &b, 0.0, &mut c).expect("sgemm_bf16");
        assert_eq!(c.len(), m * n, "output c must have m*n elements");
    }

    #[test]
    fn sgemm_bf16_identity_matrix() {
        // A @ I should ≈ A (within BF16 precision)
        let n = 4_usize;
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        // Identity matrix
        let mut identity = vec![0.0_f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let mut c = vec![0.0_f32; n * n];
        sgemm_bf16(n, n, n, 1.0, &a, &identity, 0.0, &mut c).expect("sgemm_bf16");
        for i in 0..n * n {
            let err = (c[i] - a[i]).abs();
            assert!(
                err < 0.05,
                "A @ I ≈ A: index {i}, c={}, a={}, err={err}",
                c[i],
                a[i]
            );
        }
    }

    #[test]
    fn sgemm_bf16_zero_alpha() {
        let m = 3_usize;
        let n = 3_usize;
        let k = 3_usize;
        let a = vec![2.0_f32; m * k];
        let b = vec![3.0_f32; k * n];
        let orig_c: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
        let mut c = orig_c.clone();
        sgemm_bf16(m, n, k, 0.0, &a, &b, 1.0, &mut c).expect("sgemm_bf16");
        // With alpha=0 and beta=1, c should be unchanged
        for (i, (&got, &expected)) in c.iter().zip(orig_c.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "c[{i}]={got} should equal original {expected}"
            );
        }
    }

    #[test]
    fn sgemm_bf16_beta_zero() {
        let m = 3_usize;
        let n = 3_usize;
        let k = 3_usize;
        let a = vec![1.0_f32; m * k];
        let b = vec![1.0_f32; k * n];
        let mut c = vec![999.0_f32; m * n]; // large initial value
        sgemm_bf16(m, n, k, 1.0, &a, &b, 0.0, &mut c).expect("sgemm_bf16");
        // With beta=0, old c is discarded; result should be A@B
        for &v in &c {
            assert!(
                (v - k as f32).abs() < 0.1,
                "expected c ≈ k={k} (ones @ ones), got {v}"
            );
        }
    }

    #[test]
    fn sgemm_bf16_1x1() {
        let a = vec![3.0_f32];
        let b = vec![4.0_f32];
        let mut c = vec![0.0_f32];
        sgemm_bf16(1, 1, 1, 1.0, &a, &b, 0.0, &mut c).expect("sgemm_bf16");
        // 3.0 × 4.0 = 12.0; BF16 is exact for integers in this range
        assert!(
            (c[0] - 12.0).abs() < 0.1,
            "1×1 result should be ≈12, got {}",
            c[0]
        );
    }

    #[test]
    fn bf16_gemm_error_bounded() {
        // Use small values to keep the error bound reasonable
        let m = 8_usize;
        let n = 8_usize;
        let k = 8_usize;
        let a: Vec<f32> = (0..m * k)
            .map(|i| (i as f32 + 1.0) / (m * k) as f32)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (i as f32 + 1.0) / (k * n) as f32)
            .collect();
        let err = bf16_gemm_error(m, n, k, &a, &b);
        assert!(
            err < 0.05,
            "BF16 GEMM error {err:.6} should be < 0.05 for small-valued inputs"
        );
    }

    #[test]
    fn sgemm_bf16_finite() {
        let m = 6_usize;
        let n = 6_usize;
        let k = 6_usize;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02).cos()).collect();
        let mut c = vec![0.0_f32; m * n];
        sgemm_bf16(m, n, k, 1.0, &a, &b, 0.0, &mut c).expect("sgemm_bf16");
        for (i, &v) in c.iter().enumerate() {
            assert!(v.is_finite(), "c[{i}]={v} must be finite");
        }
    }

    #[test]
    fn sgemm_bf16_dimension_zero_error() {
        let mut c = vec![0.0_f32; 4];
        let result = sgemm_bf16(0, 2, 2, 1.0, &[], &[1.0, 2.0, 3.0, 4.0], 0.0, &mut c);
        assert!(result.is_err(), "m=0 should return error");
    }
}

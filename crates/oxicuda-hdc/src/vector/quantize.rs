//! Quantization and binarization of real-valued hypervectors.
//!
//! Utilities for converting real `Vec<f32>` HVs (HRR vectors, bundled
//! accumulators, normalized reals) into compact forms: sign-binarized ±1 HVs,
//! uniform multi-level integer codes, and ternary `{-1, 0, +1}` (MAP) codes.
//!
//! # Error conventions
//! - Empty input → [`HdcError::EmptyInput`].
//! - `quantize_levels` / `dequantize_levels` require at least two levels; fewer
//!   than two would divide by `n_levels - 1`, so `n_levels < 2` →
//!   [`HdcError::DivisionByZero`].
//! - `ternarize` accepts any finite threshold and uses its magnitude
//!   (`threshold.abs()`), so there is no error path for a negative threshold.

use crate::error::{HdcError, HdcResult};

/// Sign-binarize a real HV to ±1: `x > 0 → +1`, `x < 0 → -1`, `x == 0 → +1`.
///
/// The zero tie maps to `+1` deterministically. An empty input yields an empty
/// output; use [`binarize_checked`] when an empty input should be an error.
#[must_use]
pub fn sign_binarize(hv: &[f32]) -> Vec<i8> {
    hv.iter()
        .map(|&x| if x < 0.0 { -1i8 } else { 1i8 })
        .collect()
}

/// Sign-binarize a real HV to ±1, erroring on empty input.
///
/// # Errors
/// Returns [`HdcError::EmptyInput`] if `hv` is empty.
pub fn binarize_checked(hv: &[f32]) -> HdcResult<Vec<i8>> {
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    Ok(sign_binarize(hv))
}

/// Uniform mid-tread quantization of a real HV into integer codes
/// `0..n_levels-1` over the data's own `[min, max]` range.
///
/// Each value maps via `round((x - min) / (max - min) * (n_levels - 1))`, clamped
/// to `[0, n_levels - 1]`. A constant vector (`max == min`) maps to all-zero
/// codes (avoiding a divide-by-zero).
///
/// # Errors
/// Returns [`HdcError::EmptyInput`] if `hv` is empty and
/// [`HdcError::DivisionByZero`] if `n_levels < 2` (the formula divides by
/// `n_levels - 1`).
pub fn quantize_levels(hv: &[f32], n_levels: usize) -> HdcResult<Vec<i32>> {
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if n_levels < 2 {
        return Err(HdcError::DivisionByZero);
    }
    let mut min_v = hv[0];
    let mut max_v = hv[0];
    for &x in hv.iter().skip(1) {
        if x < min_v {
            min_v = x;
        }
        if x > max_v {
            max_v = x;
        }
    }
    let range = max_v - min_v;
    let max_level = (n_levels - 1) as f32;
    if range.abs() < f32::EPSILON {
        // Constant vector: every value maps to the lowest level.
        return Ok(vec![0i32; hv.len()]);
    }
    let codes: Vec<i32> = hv
        .iter()
        .map(|&x| {
            let scaled = ((x - min_v) / range) * max_level;
            let level = scaled.round() as i32;
            level.clamp(0, n_levels as i32 - 1)
        })
        .collect();
    Ok(codes)
}

/// Inverse of [`quantize_levels`]: map integer codes back to representative real
/// values at the centre of each level over `[min, max]`.
///
/// `value = min + code * (max - min) / (n_levels - 1)`.
///
/// # Errors
/// Returns [`HdcError::DivisionByZero`] if `n_levels < 2`.
pub fn dequantize_levels(
    codes: &[i32],
    n_levels: usize,
    min: f32,
    max: f32,
) -> HdcResult<Vec<f32>> {
    if n_levels < 2 {
        return Err(HdcError::DivisionByZero);
    }
    let step = (max - min) / (n_levels - 1) as f32;
    Ok(codes.iter().map(|&c| min + c as f32 * step).collect())
}

/// Ternarize a real HV into `{-1, 0, +1}` (MAP model) by a magnitude threshold:
/// `x > |t| → +1`, `x < -|t| → -1`, otherwise `0`.
///
/// The threshold's magnitude is used, so a negative `threshold` behaves
/// identically to its positive counterpart.
///
/// # Errors
/// Returns [`HdcError::EmptyInput`] if `hv` is empty.
pub fn ternarize(hv: &[f32], threshold: f32) -> HdcResult<Vec<i32>> {
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let thr = threshold.abs();
    Ok(hv
        .iter()
        .map(|&x| {
            if x > thr {
                1i32
            } else if x < -thr {
                -1i32
            } else {
                0i32
            }
        })
        .collect())
}

/// Mean-squared quantization error between an original HV and its reconstruction.
///
/// # Errors
/// Returns [`HdcError::DimensionMismatch`] if the lengths differ and
/// [`HdcError::EmptyInput`] if either input is empty.
pub fn quantization_error(original: &[f32], reconstructed: &[f32]) -> HdcResult<f32> {
    if original.len() != reconstructed.len() {
        return Err(HdcError::DimensionMismatch {
            expected: original.len(),
            got: reconstructed.len(),
        });
    }
    if original.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let sum_sq: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| {
            let d = (a - b) as f64;
            d * d
        })
        .sum();
    Ok((sum_sq / original.len() as f64) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_binarize_maps_signs() {
        let hv = vec![0.5, -0.5, 0.0, -3.0, 2.0];
        let b = sign_binarize(&hv);
        assert_eq!(b, vec![1, -1, 1, -1, 1]); // zero → +1
    }

    #[test]
    fn binarize_checked_rejects_empty() {
        let hv: Vec<f32> = Vec::new();
        assert!(matches!(binarize_checked(&hv), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn round_trip_within_bin_width() {
        let hv = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 1.5];
        let n_levels = 8;
        let min_v = -2.0f32;
        let max_v = 2.0f32;
        let codes = quantize_levels(&hv, n_levels).expect("quantize");
        let recon = dequantize_levels(&codes, n_levels, min_v, max_v).expect("dequantize");
        let bin_width = (max_v - min_v) / (n_levels - 1) as f32;
        for (orig, r) in hv.iter().zip(recon.iter()) {
            assert!(
                (orig - r).abs() <= bin_width / 2.0 + 1e-5,
                "orig={orig} recon={r} bin_width={bin_width}"
            );
        }
    }

    #[test]
    fn quantize_rejects_too_few_levels() {
        let hv = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            quantize_levels(&hv, 1),
            Err(HdcError::DivisionByZero)
        ));
        assert!(matches!(
            quantize_levels(&hv, 0),
            Err(HdcError::DivisionByZero)
        ));
    }

    #[test]
    fn quantize_constant_vector_all_zero() {
        let hv = vec![3.0f32; 10];
        let codes = quantize_levels(&hv, 16).expect("quantize");
        assert!(codes.iter().all(|&c| c == 0));
        assert!(codes.iter().all(|&c| !c.to_string().contains("NaN")));
    }

    #[test]
    fn codes_within_range() {
        let hv = vec![-5.0, -1.0, 0.0, 1.0, 5.0, 2.3, -3.7];
        let n_levels = 10;
        let codes = quantize_levels(&hv, n_levels).expect("quantize");
        for &c in &codes {
            assert!(c >= 0 && c < n_levels as i32, "code {c} out of range");
        }
    }

    #[test]
    fn dequantize_rejects_too_few_levels() {
        let codes = vec![0, 1, 2];
        assert!(matches!(
            dequantize_levels(&codes, 1, 0.0, 1.0),
            Err(HdcError::DivisionByZero)
        ));
    }

    #[test]
    fn ternarize_splits_into_three() {
        let hv = vec![0.8, -0.8, 0.2, -0.2, 0.0];
        let t = ternarize(&hv, 0.5).expect("ternarize");
        assert_eq!(t, vec![1, -1, 0, 0, 0]);
    }

    #[test]
    fn ternarize_uses_abs_threshold() {
        let hv = vec![0.8, -0.8, 0.2, -0.2];
        let pos = ternarize(&hv, 0.5).expect("pos");
        let neg = ternarize(&hv, -0.5).expect("neg");
        assert_eq!(pos, neg);
    }

    #[test]
    fn ternarize_rejects_empty() {
        let hv: Vec<f32> = Vec::new();
        assert!(matches!(ternarize(&hv, 0.5), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn quantization_error_identical_is_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let err = quantization_error(&a, &a).expect("qerr");
        assert!(err.abs() < 1e-6, "err={err}");
    }

    #[test]
    fn quantization_error_dimension_mismatch_rejected() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        assert!(matches!(
            quantization_error(&a, &b),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn quantization_error_positive_for_differing() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.5, 2.5, 2.0];
        let err = quantization_error(&a, &b).expect("qerr");
        assert!(err > 0.0, "err={err}");
    }

    #[test]
    fn quantization_error_rejects_empty() {
        let a: Vec<f32> = Vec::new();
        let b: Vec<f32> = Vec::new();
        assert!(matches!(
            quantization_error(&a, &b),
            Err(HdcError::EmptyInput)
        ));
    }
}

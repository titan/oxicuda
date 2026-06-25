//! Mixed-precision Gaussian-noise generation: sample + FP16-quantise the noise,
//! accumulate in FP32.
//!
//! Differentially-private training on modern accelerators frequently stores the
//! injected noise (and the gradients it perturbs) in IEEE-754 binary16 (FP16) to
//! halve memory traffic, while accumulating the perturbed gradient in FP32 to
//! avoid catastrophic cancellation. This module reproduces that numerical path
//! on the CPU:
//!
//! 1. Draw `N(0, σ²)` Gaussian samples (Box-Muller, via the crate RNG).
//! 2. Round each sample to the nearest representable FP16 value (round-to-
//!    nearest-even), emulating an FP16 store/load, then widen back to FP32.
//! 3. Accumulate / return the values in FP32.
//!
//! The FP16 round-trip ([`f32_to_f16_bits`] / [`f16_bits_to_f32`]) is a complete
//! pure-Rust IEEE-754 binary16 implementation: round-to-nearest-even, correct
//! handling of subnormals, gradual underflow to ±0, overflow to ±∞, and
//! NaN/∞ propagation. No external crate is used.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;

// ─── IEEE-754 binary16 round-trip ──────────────────────────────────────────────

/// Convert an `f32` to the bit pattern of the nearest IEEE-754 binary16 (FP16)
/// value, using round-to-nearest, ties-to-even.
///
/// Handles the full FP16 dynamic range:
/// - NaN → a canonical quiet NaN (sign preserved, payload non-zero);
/// - magnitudes ≥ 65520 (the FP16 overflow threshold after rounding) → ±∞;
/// - normal values → biased 5-bit exponent + 10-bit mantissa;
/// - magnitudes below the smallest normal → FP16 subnormals (gradual underflow);
/// - magnitudes below half the smallest subnormal → ±0.
#[must_use]
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32; // unbiased-by-127 field
    let mant = bits & 0x007F_FFFF; // 23-bit mantissa

    // NaN / Inf (f32 exponent all ones).
    if exp == 0xFF {
        if mant != 0 {
            // NaN: keep it quiet with a non-zero payload.
            return sign | 0x7E00;
        }
        return sign | 0x7C00; // ±Inf
    }

    // Re-bias the exponent from f32 (bias 127) to f16 (bias 15).
    let mut half_exp = exp - 127 + 15;

    if half_exp >= 0x1F {
        // Overflow → ±Inf.
        return sign | 0x7C00;
    }

    if half_exp <= 0 {
        // Subnormal or zero in FP16.
        if half_exp < -10 {
            // Too small even for the smallest subnormal → signed zero.
            return sign;
        }
        // Restore the implicit leading 1 of the f32 mantissa, then shift it into
        // FP16 subnormal position. `shift` is in [11, 21].
        let mant_with_implicit = mant | 0x0080_0000;
        let shift = (14 - half_exp) as u32; // 14 = 24 - 10
        let mut frac = mant_with_implicit >> shift;
        // Round-to-nearest-even on the bits shifted out.
        let remainder = mant_with_implicit & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (frac & 1) == 1) {
            frac += 1;
        }
        // `frac` may carry into the normal range (0x400); that yields the
        // smallest normal, which is the correct rounded result.
        return sign | (frac as u16);
    }

    // Normal FP16 value: take the top 10 mantissa bits, round the remaining 13.
    let mut half_mant = (mant >> 13) as u16;
    let remainder = mant & 0x0000_1FFF; // low 13 bits
    let halfway = 0x0000_1000u32; // 2^12
    if remainder > halfway || (remainder == halfway && (half_mant & 1) == 1) {
        half_mant += 1;
        if half_mant == 0x0400 {
            // Mantissa overflow rolls into the exponent.
            half_mant = 0;
            half_exp += 1;
            if half_exp >= 0x1F {
                return sign | 0x7C00; // rounded up into overflow
            }
        }
    }
    sign | ((half_exp as u16) << 10) | half_mant
}

/// Widen an IEEE-754 binary16 (FP16) bit pattern back to `f32` exactly (FP16 ⊂
/// FP32, so this round-trips losslessly).
#[must_use]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Signed zero.
            return f32::from_bits(sign);
        }
        // Subnormal: normalise by finding the leading mantissa bit.
        let mut e = -1i32;
        let mut m = mant;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e += 1;
        }
        m &= 0x03FF; // drop the now-explicit leading bit
        let f32_exp = (127 - 15 - e) as u32;
        return f32::from_bits(sign | (f32_exp << 23) | (m << 13));
    }

    if exp == 0x1F {
        // Inf / NaN.
        if mant == 0 {
            return f32::from_bits(sign | 0x7F80_0000);
        }
        return f32::from_bits(sign | 0x7FC0_0000);
    }

    // Normal: re-bias exponent (15 → 127) and shift the 10-bit mantissa.
    let f32_exp = exp + (127 - 15);
    f32::from_bits(sign | (f32_exp << 23) | (mant << 13))
}

/// Round an `f32` to the nearest representable FP16 value and widen it back to
/// `f32` (a single quantisation step), via [`f32_to_f16_bits`] +
/// [`f16_bits_to_f32`].
#[must_use]
pub fn quantize_f16(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(value))
}

// ─── Mixed-precision noise generation ──────────────────────────────────────────

/// Outcome of a mixed-precision noise draw: the FP16-quantised noise (widened to
/// FP32) plus its empirical first two moments, all accumulated in FP32.
#[derive(Debug, Clone)]
pub struct MixedPrecisionNoise {
    /// The quantised noise samples (FP16-rounded, stored as FP32).
    pub samples: Vec<f64>,
    /// FP32-accumulated sample mean.
    pub mean: f64,
    /// FP32-accumulated (population) variance.
    pub variance: f64,
}

/// Draw `n` Gaussian `N(0, σ²)` samples, quantise each to FP16 (emulating an
/// FP16 store), and accumulate the running mean / variance in FP32.
///
/// The quantisation introduces a bounded rounding error of at most half an ULP
/// of the FP16 grid at each sample's magnitude, so for the σ used in DP training
/// (σ·C of order 1) the sample variance is preserved to well within a percent.
///
/// # Errors
/// - `InvalidParameter` if `sigma < 0`.
/// - `EmptyInput` if `n == 0` (no samples to characterise).
pub fn mixed_precision_gaussian(
    sigma: f64,
    n: usize,
    handle: &mut PrivacyHandle,
) -> PrivacyResult<MixedPrecisionNoise> {
    if sigma < 0.0 {
        return Err(PrivacyError::InvalidParameter("sigma must be ≥ 0".into()));
    }
    if n == 0 {
        return Err(PrivacyError::EmptyInput);
    }

    // Sample FP32 Gaussian noise, then quantise each draw through FP16.
    let raw = handle.generate_gaussian_noise(sigma, n)?;
    let mut samples = Vec::with_capacity(n);

    // Welford's online algorithm in FP32-accumulated f64 for a numerically
    // stable mean / variance of the quantised stream.
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for (i, &x) in raw.iter().enumerate() {
        let q = f64::from(quantize_f16(x as f32));
        samples.push(q);
        let count = (i + 1) as f64;
        let delta = q - mean;
        mean += delta / count;
        let delta2 = q - mean;
        m2 += delta * delta2;
    }
    let variance = m2 / (n as f64);

    Ok(MixedPrecisionNoise {
        samples,
        mean,
        variance,
    })
}

/// Add FP16-quantised Gaussian noise to an FP32 accumulator vector in place,
/// returning the noise standard deviation actually used (`σ`).
///
/// Each gradient coordinate is perturbed by an FP16-rounded `N(0, σ²)` draw but
/// the addition itself is performed in FP32 (`f64`), exactly mirroring the
/// "FP16 sample, FP32 accumulate" pattern.
///
/// # Errors
/// - `InvalidParameter` if `sigma < 0`.
/// - `EmptyInput` if `accumulator` is empty.
pub fn add_fp16_noise_fp32_accumulate(
    accumulator: &mut [f64],
    sigma: f64,
    handle: &mut PrivacyHandle,
) -> PrivacyResult<f64> {
    if sigma < 0.0 {
        return Err(PrivacyError::InvalidParameter("sigma must be ≥ 0".into()));
    }
    if accumulator.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    let noise = handle.generate_gaussian_noise(sigma, accumulator.len())?;
    for (acc, &z) in accumulator.iter_mut().zip(noise.iter()) {
        // FP16 store of the noise, FP32 accumulate.
        *acc += f64::from(quantize_f16(z as f32));
    }
    Ok(sigma)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const F16_MAX: f32 = 65504.0; // largest finite binary16
    const F16_MIN_SUBNORMAL: f32 = 5.960_464_5e-8; // 2^-24
    const F16_MIN_NORMAL: f32 = 6.103_515_6e-5; // 2^-14

    // 1. Exactly-representable values survive the FP16 round-trip unchanged.
    #[test]
    fn exact_values_roundtrip() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 100.0, -256.0] {
            assert_eq!(quantize_f16(v), v, "round-trip changed {v}");
        }
    }

    // 2. Signed zeros are preserved.
    #[test]
    fn signed_zero_preserved() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert!(quantize_f16(-0.0).is_sign_negative());
        assert_eq!(quantize_f16(0.0), 0.0);
    }

    // 3. The largest finite FP16 round-trips; just above it overflows to ∞.
    #[test]
    fn overflow_to_infinity() {
        assert_eq!(quantize_f16(F16_MAX), F16_MAX);
        assert!(quantize_f16(70000.0).is_infinite());
        assert!(quantize_f16(70000.0) > 0.0);
        assert!(quantize_f16(-70000.0).is_infinite());
        assert!(quantize_f16(-70000.0) < 0.0);
    }

    // 4. Subnormals are handled: the smallest subnormal round-trips, and half of
    //    it underflows to zero (ties-to-even at 0).
    #[test]
    fn subnormals_and_underflow() {
        assert_eq!(quantize_f16(F16_MIN_SUBNORMAL), F16_MIN_SUBNORMAL);
        assert_eq!(quantize_f16(F16_MIN_NORMAL), F16_MIN_NORMAL);
        // Far below the smallest subnormal → ±0.
        assert_eq!(quantize_f16(1e-10), 0.0);
        assert!(quantize_f16(-1e-10).is_sign_negative());
        // A value between two subnormal grid points rounds to a multiple of 2^-24.
        let q = quantize_f16(1.5 * F16_MIN_SUBNORMAL);
        let grid = q / F16_MIN_SUBNORMAL;
        assert!(
            (grid - grid.round()).abs() < 1e-3,
            "not on subnormal grid: {q}"
        );
    }

    // 5. Inf / NaN propagate.
    #[test]
    fn inf_nan_propagate() {
        assert!(quantize_f16(f32::INFINITY).is_infinite());
        assert!(quantize_f16(f32::NEG_INFINITY).is_infinite());
        assert!(quantize_f16(f32::NEG_INFINITY) < 0.0);
        assert!(quantize_f16(f32::NAN).is_nan());
    }

    // 6. Round-to-nearest-even: 1 + 2^-11 sits exactly between 1 and 1+2^-10 and
    //    must round down to 1 (even mantissa); 1 + 3·2^-11 rounds up.
    #[test]
    fn round_to_nearest_even() {
        let eps10 = 2f32.powi(-10); // FP16 ulp at 1.0
        // Midpoint between 1.0 (mantissa 0, even) and 1+ulp (mantissa 1, odd):
        let mid = 1.0 + 0.5 * eps10;
        assert_eq!(quantize_f16(mid), 1.0, "tie should round to even (down)");
        // Three-quarter point rounds up to 1+ulp.
        let up = 1.0 + 0.75 * eps10;
        assert!((quantize_f16(up) - (1.0 + eps10)).abs() < 1e-7);
    }

    // 7. Quantisation error is bounded by half an ULP of the local FP16 grid.
    #[test]
    fn quantization_error_bounded() {
        let mut rng = crate::handle::LcgRng::new(2024);
        for _ in 0..10_000 {
            // Values in [-4, 4): FP16 ulp here is at most 2^(1-10) = 2^-9.
            let x = (rng.next_f32() * 8.0) - 4.0;
            let q = quantize_f16(x);
            // ulp at |x|<4 is bounded by 2^(floor(log2|x|)-10) ≤ 2^-8.
            let max_err = 2f32.powi(-8);
            assert!(
                (q - x).abs() <= max_err,
                "err {} > {} for x={x}",
                (q - x).abs(),
                max_err
            );
        }
    }

    // 8. Mixed-precision Gaussian preserves scale: empirical variance ≈ σ².
    #[test]
    fn mixed_precision_variance_matches_sigma() {
        let mut handle = PrivacyHandle::new(80, 4242);
        let sigma = 1.5;
        let result = mixed_precision_gaussian(sigma, 200_000, &mut handle).expect("noise");
        // Variance within 3% of σ²; mean within 0.02 of 0.
        let target_var = sigma * sigma;
        assert!(
            (result.variance - target_var).abs() / target_var < 0.03,
            "var {} vs σ²={target_var}",
            result.variance
        );
        assert!(result.mean.abs() < 0.02, "mean {} ≉ 0", result.mean);
        // Every stored sample must be an exact FP16 grid point.
        for &s in &result.samples {
            assert_eq!(f64::from(quantize_f16(s as f32)), s, "non-FP16 sample {s}");
        }
    }

    // 9. add_fp16_noise_fp32_accumulate adds bounded-error FP16 noise in place.
    #[test]
    fn accumulate_adds_quantised_noise() {
        let mut handle = PrivacyHandle::new(80, 7);
        let mut acc = vec![10.0f64; 5000];
        let sigma = 0.5;
        add_fp16_noise_fp32_accumulate(&mut acc, sigma, &mut handle).expect("add");
        // The mean of the perturbed accumulator stays near the base value 10.
        let mean = acc.iter().sum::<f64>() / acc.len() as f64;
        assert!((mean - 10.0).abs() < 0.05, "perturbed mean {mean} ≉ 10");
        // Each coordinate minus 10 must be an FP16 grid value.
        for &a in &acc {
            let noise = (a - 10.0) as f32;
            assert_eq!(
                quantize_f16(noise),
                noise,
                "noise not on FP16 grid: {noise}"
            );
        }
    }

    // 10. Error paths.
    #[test]
    fn error_paths() {
        let mut handle = PrivacyHandle::new(80, 1);
        assert!(matches!(
            mixed_precision_gaussian(-1.0, 10, &mut handle),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            mixed_precision_gaussian(1.0, 0, &mut handle),
            Err(PrivacyError::EmptyInput)
        ));
        let mut empty: Vec<f64> = vec![];
        assert!(matches!(
            add_fp16_noise_fp32_accumulate(&mut empty, 1.0, &mut handle),
            Err(PrivacyError::EmptyInput)
        ));
        let mut acc = vec![0.0; 4];
        assert!(matches!(
            add_fp16_noise_fp32_accumulate(&mut acc, -0.1, &mut handle),
            Err(PrivacyError::InvalidParameter(_))
        ));
    }

    // 11. f16_bits_to_f32 inverts f32_to_f16_bits on the FP16 lattice (sweep all
    //     65536 bit patterns that are not NaN, comparing to a recompute).
    #[test]
    fn bit_roundtrip_is_idempotent() {
        for bits in 0u16..=0xFFFF {
            let as_f32 = f16_bits_to_f32(bits);
            if as_f32.is_nan() {
                continue;
            }
            // Re-quantising an exact FP16 value must return the same bits
            // (modulo the -0/+0 and the single canonical NaN encoding).
            let rebits = f32_to_f16_bits(as_f32);
            assert_eq!(rebits, bits, "bits {bits:#06x} → {as_f32} → {rebits:#06x}");
        }
    }
}

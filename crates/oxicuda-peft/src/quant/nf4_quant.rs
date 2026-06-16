//! NF4 and FP4 quantization for QLoRA-style weight compression.
//!
//! NF4 (Normal Float 4-bit) places 16 quantization levels at the quantiles of N(0,1),
//! giving minimum quantization error for normally-distributed weights.
//! FP4 (Float 4-bit, e2m1 format) uses a floating-point encoding with sign, 2 exponent
//! bits, and 1 mantissa bit, matching common QLoRA implementations.

/// NF4 quantization levels: 16 values placed at quantiles of N(0,1).
///
/// These are pre-computed so that each level covers an equal probability mass
/// under the standard normal distribution, minimising expected quantization error
/// for normally-distributed weights.
pub const NF4_QUANTS: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// FP4 (e2m1) representable values, indexed by their 4-bit code.
///
/// Codes 0-7 are the non-negative values; codes 8-15 are negatives.
/// Layout: code = sign_bit(1) | exponent(2) | mantissa(1)
/// For exp > 0: val = (-1)^sign * (1 + mantissa/2) * 2^(exp-1)
/// For exp = 0: val = (-1)^sign * mantissa * 0.5  (subnormal)
const FP4_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Quantize a slice of `f32` values to NF4 4-bit codes, packing two codes per byte.
///
/// Normalizes each element by `absmax` (clamping to `[-1, 1]`), then assigns the
/// nearest NF4 quantization level (index 0-15). Two 4-bit codes are packed per byte:
/// the first element occupies the low nibble (bits 3:0), the second the high nibble (bits 7:4).
///
/// If `absmax == 0.0`, all elements map to index 7 (value `0.0`).
///
/// # Returns
/// `(packed_bytes, absmax)` where `packed_bytes.len() == (x.len() + 1) / 2`.
#[must_use]
pub fn quantize_nf4(x: &[f32], absmax: f32) -> (Vec<u8>, f32) {
    let n = x.len();
    let n_packed = n.div_ceil(2);
    let mut packed = vec![0u8; n_packed];

    // Safe absmax: if zero, treat all weights as zero (map to NF4 index 7 = 0.0)
    let safe_absmax = if absmax == 0.0 { 1.0 } else { absmax };

    for (i, &val) in x.iter().enumerate() {
        let normalized = (val / safe_absmax).clamp(-1.0, 1.0);
        let code = nearest_nf4(normalized) as u8;
        if i % 2 == 0 {
            // Low nibble
            packed[i / 2] = code;
        } else {
            // High nibble
            packed[i / 2] |= code << 4;
        }
    }

    (packed, absmax)
}

/// Dequantize NF4-packed bytes back to `f32` values.
///
/// Unpacks nibbles (low then high), looks up `NF4_QUANTS[code]`, and scales by `scale`.
/// Only `n_elements` values are produced; the last byte's high nibble is ignored if `n_elements` is odd.
#[must_use]
pub fn dequantize_nf4(packed: &[u8], scale: f32, n_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elements);
    for (i, &byte) in packed.iter().enumerate() {
        let lo_code = (byte & 0x0F) as usize;
        let hi_code = ((byte >> 4) & 0x0F) as usize;
        let idx0 = i * 2;
        let idx1 = i * 2 + 1;
        if idx0 < n_elements {
            out.push(NF4_QUANTS[lo_code] * scale);
        }
        if idx1 < n_elements {
            out.push(NF4_QUANTS[hi_code] * scale);
        }
    }
    out
}

/// Quantize a slice of `f32` values to FP4 (e2m1) 4-bit codes, packing two per byte.
///
/// Normalizes by `absmax`, finds the nearest FP4 representable value, encodes as a
/// 4-bit code and packs two codes per byte (low nibble first, high nibble second).
///
/// If `absmax == 0.0`, all elements map to code 0 (value `0.0`).
///
/// # Returns
/// `(packed_bytes, absmax)` where `packed_bytes.len() == (x.len() + 1) / 2`.
#[must_use]
pub fn quantize_fp4(x: &[f32], absmax: f32) -> (Vec<u8>, f32) {
    let n = x.len();
    let n_packed = n.div_ceil(2);
    let mut packed = vec![0u8; n_packed];

    let safe_absmax = if absmax == 0.0 { 1.0 } else { absmax };

    for (i, &val) in x.iter().enumerate() {
        let normalized = val / safe_absmax;
        let code = nearest_fp4(normalized) as u8;
        if i % 2 == 0 {
            packed[i / 2] = code;
        } else {
            packed[i / 2] |= code << 4;
        }
    }

    (packed, absmax)
}

/// Dequantize FP4-packed bytes back to `f32` values.
///
/// Unpacks nibbles (low then high), looks up `FP4_VALUES[code]`, and scales by `scale`.
/// Only `n_elements` values are produced.
#[must_use]
pub fn dequantize_fp4(packed: &[u8], scale: f32, n_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elements);
    for (i, &byte) in packed.iter().enumerate() {
        let lo_code = (byte & 0x0F) as usize;
        let hi_code = ((byte >> 4) & 0x0F) as usize;
        let idx0 = i * 2;
        let idx1 = i * 2 + 1;
        if idx0 < n_elements {
            out.push(FP4_VALUES[lo_code] * scale);
        }
        if idx1 < n_elements {
            out.push(FP4_VALUES[hi_code] * scale);
        }
    }
    out
}

/// Find the index of the nearest value in `NF4_QUANTS` to `val`.
///
/// Since `NF4_QUANTS` is sorted, we could binary-search, but a linear scan over
/// 16 entries is branchless-friendly and avoids any edge cases.
#[inline]
fn nearest_nf4(val: f32) -> usize {
    NF4_QUANTS
        .iter()
        .enumerate()
        .min_by(|&(_, a), &(_, b)| {
            (val - a)
                .abs()
                .partial_cmp(&(val - b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Find the index of the nearest value in `FP4_VALUES` to `val`.
#[inline]
fn nearest_fp4(val: f32) -> usize {
    FP4_VALUES
        .iter()
        .enumerate()
        .min_by(|&(_, a), &(_, b)| {
            (val - a)
                .abs()
                .partial_cmp(&(val - b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate a deterministic sequence of values for testing.
    fn test_values(n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n as f32 - 1.0) * 2.0 - 1.0; // [-1, 1]
                t * scale
            })
            .collect()
    }

    #[test]
    fn nf4_roundtrip_error_small() {
        // Use absmax=1.0 so all normalized values fall in [-1, 1] (NF4 range).
        let vals = test_values(64, 1.0);
        let absmax = 1.0_f32;
        let (packed, scale) = quantize_nf4(&vals, absmax);
        let dequant = dequantize_nf4(&packed, scale, vals.len());
        assert_eq!(dequant.len(), vals.len());
        let max_err = vals
            .iter()
            .zip(dequant.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // NF4 has 16 non-uniformly-spaced levels over [-1,1]; max gap ≈ 0.28
        // so max roundtrip error ≤ half the largest gap ≈ 0.14; use 0.20 as safe bound.
        assert!(
            max_err < 0.20,
            "NF4 roundtrip max error {max_err} exceeds 0.20 for values in [-1,1]"
        );
    }

    #[test]
    fn fp4_roundtrip_error_small() {
        // FP4 is a coarse 4-bit floating point format.
        // Use absmax=1.0 so that the FP4 values (0,0.5,1,1.5,2,3,4,6,-0.5,-1,-1.5,-2,-3,-4,-6)
        // map to exact FP4 codes. The max gap between consecutive positive FP4 values
        // (in units of absmax) is between 4.0 and 6.0, so worst-case quantization error
        // for arbitrary inputs is at most 1.0 (half of gap 2.0 between 4.0 and 6.0).
        // Test that roundtrip on the exact representable values has near-zero error.
        let vals = vec![0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let absmax = 1.0_f32; // Treat vals as already normalized — FP4 max = 6.0
        // But FP4_VALUES go up to 6.0 in absolute, so with absmax=1.0, vals are in [0,6]
        // and the nearest codes are exact.
        let (packed, scale) = quantize_fp4(&vals, absmax);
        let dequant = dequantize_fp4(&packed, scale, vals.len());
        assert_eq!(dequant.len(), vals.len());
        let max_err = vals
            .iter()
            .zip(dequant.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-4,
            "FP4 roundtrip for exact representable values max error {max_err} exceeds 1e-4"
        );
    }

    #[test]
    fn nf4_packed_len() {
        for n in [1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let vals = vec![0.5_f32; n];
            let (packed, _) = quantize_nf4(&vals, 1.0);
            let expected = n.div_ceil(2);
            assert_eq!(
                packed.len(),
                expected,
                "NF4 packed len wrong for n={n}: got {} expected {expected}",
                packed.len()
            );
        }
    }

    #[test]
    fn fp4_packed_len() {
        for n in [1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let vals = vec![0.5_f32; n];
            let (packed, _) = quantize_fp4(&vals, 1.0);
            let expected = n.div_ceil(2);
            assert_eq!(
                packed.len(),
                expected,
                "FP4 packed len wrong for n={n}: got {} expected {expected}",
                packed.len()
            );
        }
    }

    #[test]
    fn nf4_zero_stays_zero() {
        // 0.0 normalized is 0.0, nearest NF4 is index 7 (value 0.0)
        let vals = vec![0.0_f32; 8];
        let (packed, _scale) = quantize_nf4(&vals, 1.0);
        let dequant = dequantize_nf4(&packed, 1.0, vals.len());
        for &v in &dequant {
            assert!(v.abs() < 1e-9, "NF4 zero should stay zero, got {v}");
        }
    }

    #[test]
    fn fp4_zero_stays_zero() {
        // 0.0 maps to FP4 code 0 (value 0.0)
        let vals = vec![0.0_f32; 8];
        let (packed, _scale) = quantize_fp4(&vals, 1.0);
        // Verify code 0 is chosen: all bytes should be 0x00
        for &b in &packed {
            assert_eq!(b, 0x00, "FP4 zero: expected byte 0x00, got {b:#04x}");
        }
        let dequant = dequantize_fp4(&packed, 1.0, vals.len());
        for &v in &dequant {
            assert!(v.abs() < 1e-9, "FP4 zero should stay zero, got {v}");
        }
    }

    #[test]
    fn nf4_saturation() {
        // Values beyond absmax should saturate to ±absmax in dequantized output
        let absmax = 2.0_f32;
        let vals = vec![-10.0_f32, 10.0, -10.0, 10.0];
        let (packed, scale) = quantize_nf4(&vals, absmax);
        let dequant = dequantize_nf4(&packed, scale, vals.len());
        for &v in &dequant {
            assert!(
                v.abs() <= absmax + 1e-5,
                "NF4 saturated value {v} exceeds absmax {absmax}"
            );
        }
        // The saturated values should be at ±absmax (NF4 extremes are ±1.0 * scale)
        assert!(
            dequant[0] <= -absmax * 0.9,
            "NF4 negative saturation: expected near -{absmax}, got {}",
            dequant[0]
        );
        assert!(
            dequant[1] >= absmax * 0.9,
            "NF4 positive saturation: expected near {absmax}, got {}",
            dequant[1]
        );
    }

    #[test]
    fn fp4_saturation() {
        // FP4 max representable magnitude is 6.0; values beyond absmax saturate
        let absmax = 1.0_f32;
        let vals = vec![-100.0_f32, 100.0, -100.0, 100.0];
        let (packed, scale) = quantize_fp4(&vals, absmax);
        let dequant = dequantize_fp4(&packed, scale, vals.len());
        // Dequantized max is FP4 max (6.0) * absmax in magnitude
        let fp4_max = 6.0_f32 * absmax;
        for &v in &dequant {
            assert!(
                v.abs() <= fp4_max + 1e-5,
                "FP4 saturated value {v} exceeds fp4_max {fp4_max}"
            );
        }
    }

    #[test]
    fn nf4_vs_fp4_different() {
        // For a non-trivial array, NF4 and FP4 dequantized results should differ
        let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.07).collect();
        let absmax = 1.2_f32;
        let (packed_nf4, s4) = quantize_nf4(&vals, absmax);
        let (packed_fp4, sfp4) = quantize_fp4(&vals, absmax);
        let dq_nf4 = dequantize_nf4(&packed_nf4, s4, vals.len());
        let dq_fp4 = dequantize_fp4(&packed_fp4, sfp4, vals.len());
        let total_diff: f32 = dq_nf4
            .iter()
            .zip(dq_fp4.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            total_diff > 1e-4,
            "NF4 and FP4 dequantized results should differ for non-trivial input, total_diff={total_diff}"
        );
    }

    #[test]
    fn nf4_sorted_quants() {
        // NF4_QUANTS must be sorted in ascending order
        for i in 0..15 {
            assert!(
                NF4_QUANTS[i] < NF4_QUANTS[i + 1],
                "NF4_QUANTS not sorted at index {i}: {} >= {}",
                NF4_QUANTS[i],
                NF4_QUANTS[i + 1]
            );
        }
    }

    #[test]
    fn nf4_absmax_zero_all_zero() {
        // With absmax=0, all values map to the zero level (index 7)
        let vals = vec![0.0_f32; 10];
        let (packed, _) = quantize_nf4(&vals, 0.0);
        let dequant = dequantize_nf4(&packed, 0.0, vals.len());
        for &v in &dequant {
            // scale=0 means all output is 0
            assert!(v.abs() < 1e-9, "absmax=0 should yield all zeros, got {v}");
        }
    }

    #[test]
    fn fp4_absmax_zero_all_zero() {
        let vals = vec![0.0_f32; 10];
        let (packed, _) = quantize_fp4(&vals, 0.0);
        let dequant = dequantize_fp4(&packed, 0.0, vals.len());
        for &v in &dequant {
            assert!(v.abs() < 1e-9, "absmax=0 should yield all zeros, got {v}");
        }
    }

    #[test]
    fn nf4_nibble_packing_correct() {
        // Manually verify nibble packing: encode two known values and check byte layout
        // NF4_QUANTS[0]=-1.0 should map to code 0, NF4_QUANTS[15]=1.0 -> code 15
        let vals = vec![-1.0_f32, 1.0];
        let (packed, _) = quantize_nf4(&vals, 1.0);
        assert_eq!(packed.len(), 1);
        let lo = packed[0] & 0x0F; // first element in low nibble
        let hi = (packed[0] >> 4) & 0x0F; // second element in high nibble
        assert_eq!(lo, 0, "code for -1.0 should be 0 (low nibble)");
        assert_eq!(hi, 15, "code for 1.0 should be 15 (high nibble)");
    }

    #[test]
    fn fp4_nibble_packing_correct() {
        // 0.0 -> code 0 (FP4_VALUES[0] = 0.0)
        // 6.0 with absmax=6.0: normalized=1.0; nearest FP4 value to 1.0 is FP4_VALUES[2]=1.0, code=2
        // Separately: 6.0 with absmax=1.0: normalized=6.0 (clamped); nearest FP4 value to 6.0
        // is FP4_VALUES[7]=6.0, code=7
        // Test the 0.0 -> code 0 case first.
        let absmax = 1.0_f32;
        let vals = vec![0.0_f32, 6.0];
        let (packed, _) = quantize_fp4(&vals, absmax);
        assert_eq!(packed.len(), 1);
        let lo = packed[0] & 0x0F;
        let hi = (packed[0] >> 4) & 0x0F;
        assert_eq!(lo, 0, "code for 0.0 should be 0");
        // 6.0/absmax=1.0 = 6.0; nearest FP4 is FP4_VALUES[7]=6.0 → code 7
        assert_eq!(hi, 7, "code for 6.0 with absmax=1.0 should be 7");
    }
}

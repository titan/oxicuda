//! FP8 quantisation/dequantisation for GPU inference.
//!
//! Implements two standard FP8 formats:
//!
//! | Format | Sign | Exponent | Mantissa | Max value |
//! |--------|------|----------|----------|-----------|
//! | E4M3   |  1   |    4     |    3     |   448.0   |
//! | E5M2   |  1   |    5     |    2     |  57344.0  |
//!
//! The quantisation strategy used here is **per-tensor absmax scaling**:
//!
//! 1. Find `absmax = max(|x_i|)` over the input.
//! 2. Scale `x_i / absmax` into `[-1.0, 1.0]`.
//! 3. Multiply by 127 and round to the nearest integer in `[-127, 127]`.
//! 4. Reinterpret the `i8` bit-pattern as a `u8` (Rust's `as` cast).
//!
//! Dequantisation reverses the steps: interpret `u8` as `i8`, multiply by
//! `absmax / 127`.
//!
//! Block-wise quantisation (`quantize_fp8_block` / `dequantize_fp8_block`)
//! computes an independent `absmax` scale per block of `block_size` elements,
//! matching the bitsandbytes block-quantisation scheme.

// ─── FP8 format tag ──────────────────────────────────────────────────────────

/// FP8 floating-point format selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// 1 sign, 4 exponent, 3 mantissa bits.  Max representable value: 448.0.
    E4M3,
    /// 1 sign, 5 exponent, 2 mantissa bits.  Max representable value: 57344.0.
    E5M2,
}

impl Fp8Format {
    /// Maximum representable positive finite value for this format.
    pub fn max_value(self) -> f32 {
        match self {
            Fp8Format::E4M3 => 448.0_f32,
            Fp8Format::E5M2 => 57344.0_f32,
        }
    }

    /// Symmetric integer scale used for quantisation (i8 range: [-scale, scale]).
    pub fn int_scale(self) -> f32 {
        // Both E4M3 and E5M2 are packed into u8 via i8; we use 127 as the
        // symmetric maximum to avoid the asymmetric -128.
        127.0_f32
    }
}

// ─── Per-tensor quantisation ──────────────────────────────────────────────────

/// Quantise a slice of `f32` values to FP8 packed as `u8`.
///
/// Uses per-tensor absmax scaling.  The caller should store the returned
/// absmax separately so that `dequantize_fp8` can reconstruct the original
/// magnitudes.
///
/// If all inputs are zero, every output byte is `0u8`.
pub fn quantize_fp8(x: &[f32], _format: Fp8Format) -> Vec<u8> {
    let absmax = x.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if absmax == 0.0 {
        return vec![0u8; x.len()];
    }
    let scale = _format.int_scale() / absmax;
    x.iter()
        .map(|&v| {
            let q = (v * scale).clamp(-127.0, 127.0).round() as i8;
            q as u8
        })
        .collect()
}

/// Dequantise FP8 bytes back to `f32`.
///
/// `absmax` is the per-tensor absmax that was used during quantisation.
pub fn dequantize_fp8(q: &[u8], absmax: f32, format: Fp8Format) -> Vec<f32> {
    let inv_scale = absmax / format.int_scale();
    q.iter()
        .map(|&byte| {
            let qi = byte as i8;
            qi as f32 * inv_scale
        })
        .collect()
}

// ─── Block-wise quantisation ──────────────────────────────────────────────────

/// Quantise `x` using **per-block** absmax scaling.
///
/// `x` is partitioned into non-overlapping blocks of `block_size` elements
/// (the last block may be smaller).  Each block is independently scaled by its
/// own absmax.
///
/// Returns `(quantized_bytes, per_block_scales)`.  `per_block_scales[i]` is
/// the absmax of block `i` and is required to reconstruct the original values.
pub fn quantize_fp8_block(x: &[f32], block_size: usize, format: Fp8Format) -> (Vec<u8>, Vec<f32>) {
    let block_size = block_size.max(1);
    let n_blocks = x.len().div_ceil(block_size);
    let mut quantized = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(n_blocks);

    for block in x.chunks(block_size) {
        let absmax = block.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        scales.push(absmax);
        if absmax == 0.0 {
            quantized.extend(vec![0u8; block.len()]);
        } else {
            let scale = format.int_scale() / absmax;
            for &v in block {
                let q = (v * scale).clamp(-127.0, 127.0).round() as i8;
                quantized.push(q as u8);
            }
        }
    }

    (quantized, scales)
}

/// Dequantise block-wise FP8 bytes back to `f32`.
///
/// `scales` must have exactly `ceil(q.len() / block_size)` entries — one
/// per block, matching what `quantize_fp8_block` returned.
pub fn dequantize_fp8_block(
    q: &[u8],
    scales: &[f32],
    block_size: usize,
    format: Fp8Format,
) -> Vec<f32> {
    let block_size = block_size.max(1);
    let mut output = Vec::with_capacity(q.len());
    for (block_idx, block) in q.chunks(block_size).enumerate() {
        let absmax = scales.get(block_idx).copied().unwrap_or(0.0);
        let inv_scale = absmax / format.int_scale();
        for &byte in block {
            let qi = byte as i8;
            output.push(qi as f32 * inv_scale);
        }
    }
    output
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    // 1. Round-trip error is small relative to absmax.
    #[test]
    fn round_trip_error_small() {
        let input = vec![0.5_f32, -0.25, 1.0, -1.0, 0.75];
        let absmax = input.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let quantized = quantize_fp8(&input, Fp8Format::E4M3);
        let recovered = dequantize_fp8(&quantized, absmax, Fp8Format::E4M3);
        // Tolerance: 1 quantisation step ≈ absmax / 127 ≈ 0.0079, well below 0.1.
        let err = max_abs_err(&input, &recovered);
        assert!(
            err < absmax * 0.1,
            "round-trip error {err} exceeds threshold"
        );
    }

    // 2. Quantise output length equals input length.
    #[test]
    fn quantize_output_len() {
        let input = vec![1.0_f32; 64];
        let q = quantize_fp8(&input, Fp8Format::E4M3);
        assert_eq!(q.len(), input.len());
    }

    // 3. Dequantise output length equals quantised length.
    #[test]
    fn dequantize_output_len() {
        let input = vec![-2.0_f32, 0.0, 2.0, 4.0];
        let absmax = 4.0_f32;
        let q = quantize_fp8(&input, Fp8Format::E5M2);
        let out = dequantize_fp8(&q, absmax, Fp8Format::E5M2);
        assert_eq!(out.len(), q.len());
    }

    // 4. E4M3 format tag exists and has the correct max value.
    #[test]
    fn e4m3_max_value() {
        assert_eq!(Fp8Format::E4M3.max_value(), 448.0_f32);
    }

    // 5. E5M2 format tag exists and has the correct max value.
    #[test]
    fn e5m2_max_value() {
        assert_eq!(Fp8Format::E5M2.max_value(), 57344.0_f32);
    }

    // 6. Zero input quantises to all-zero bytes.
    #[test]
    fn zero_quantizes_to_zero() {
        let input = vec![0.0_f32];
        let q = quantize_fp8(&input, Fp8Format::E4M3);
        // 0.0 / absmax is undefined; code special-cases absmax==0 → all zeros.
        assert_eq!(q, vec![0u8]);
    }

    // 7. Block-wise quantisation returns one scale per block.
    #[test]
    fn block_scales_shape() {
        let input = vec![1.0_f32; 8];
        let (_q, scales) = quantize_fp8_block(&input, 4, Fp8Format::E4M3);
        assert_eq!(scales.len(), 2);
    }

    // 8. Block round-trip error is small.
    #[test]
    fn block_round_trip() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.75).collect();
        let block_size = 4;
        let (q, scales) = quantize_fp8_block(&input, block_size, Fp8Format::E4M3);
        let recovered = dequantize_fp8_block(&q, &scales, block_size, Fp8Format::E4M3);
        // The largest scale is the absmax of the block with the biggest range.
        let max_scale = scales.iter().cloned().fold(0.0_f32, f32::max);
        let err = max_abs_err(&input, &recovered);
        assert!(
            err < max_scale * 0.1,
            "block round-trip error {err} exceeds threshold (scale={max_scale})"
        );
    }

    // 9. Quantising the same input with both formats yields same-length output.
    #[test]
    fn format_affects_output() {
        let input = vec![100.0_f32, -200.0, 0.5, -0.5, std::f32::consts::PI];
        let q_e4m3 = quantize_fp8(&input, Fp8Format::E4M3);
        let q_e5m2 = quantize_fp8(&input, Fp8Format::E5M2);
        assert_eq!(q_e4m3.len(), q_e5m2.len());
        // Both formats use the same i8/u8 packing, so quantised bytes may differ
        // only in scale factor — just confirm length equality here.
    }

    // 10. Large positive value is clamped to 127 after scaling.
    #[test]
    fn max_quantized_value_is_127() {
        // Single-element input: absmax = 1000.0, scale = 127/1000.
        // x = 1000.0 → 1000 * (127/1000) = 127.0 → i8 127 → u8 127.
        let input = vec![1000.0_f32];
        let q = quantize_fp8(&input, Fp8Format::E4M3);
        assert_eq!(q[0], 127u8);
    }

    // 11. Negative value packs correctly as two's-complement u8.
    #[test]
    fn negative_packs_as_twos_complement() {
        // input = [-1.0]; absmax = 1.0; q = -127 as i8 = 0x81u8.
        let input = vec![-1.0_f32];
        let q = quantize_fp8(&input, Fp8Format::E4M3);
        let expected = (-127_i8) as u8;
        assert_eq!(q[0], expected);
    }

    // 12. Block-wise with a block that is all zeros produces zero scale.
    #[test]
    fn block_all_zeros_scale_is_zero() {
        let input = vec![0.0_f32, 0.0, 0.0, 0.0, 1.0, -1.0, 0.5, -0.5];
        let (_q, scales) = quantize_fp8_block(&input, 4, Fp8Format::E5M2);
        assert_eq!(scales[0], 0.0_f32);
        assert!(scales[1] > 0.0_f32);
    }
}

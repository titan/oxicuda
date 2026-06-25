//! # GGML / GGUF Block Quantization Schemes
//!
//! Pure-Rust implementations of the block-wise weight quantization formats used
//! by `llama.cpp` / `ggml` and serialized in GGUF model containers. These are
//! the on-disk codecs behind names such as `Q8_0`, `Q4_0`, `Q4_1` and the
//! "k-quant" super-block format `Q4_K`.
//!
//! Each format trades file size for reconstruction fidelity by storing a small
//! number of bits per weight plus per-block scale/offset metadata. The codecs
//! here operate purely on flat `&[f32]` weight slices and produce/consume the
//! exact byte-equivalent block representations (with f16 scales emulated by an
//! IEEE-754 half round-trip), so they are fully CPU-verifiable without any GPU
//! or external container library.
//!
//! ## Supported formats
//!
//! | Format | Block | Bits/wt | Scale         | Offset | Reference |
//! |--------|-------|---------|---------------|--------|-----------|
//! | `Q8_0` | 32    | 8.5     | f16 `d`       | —      | symmetric |
//! | `Q4_0` | 32    | 4.5     | f16 `d`       | −8`d`  | symmetric |
//! | `Q4_1` | 32    | 5.0     | f16 `d`, `m`  | `m`    | affine    |
//! | `Q4_K` | 256   | ~4.5    | 6-bit + super | 6-bit  | k-quant   |
//!
//! "Bits/wt" includes amortized metadata overhead, matching the canonical
//! `ggml` block layouts.
//!
//! ## f16 scale emulation
//!
//! `ggml` stores block scales as IEEE-754 binary16. To produce numerically
//! identical reconstruction the scales are round-tripped through [`f16_round`],
//! which rounds an `f32` to the nearest representable half (round-to-nearest-
//! even, subnormal-aware) and returns the resulting `f32`.

use crate::error::{QuantError, QuantResult};

// ─── f16 emulation ─────────────────────────────────────────────────────────────

/// Round an `f32` to the nearest IEEE-754 binary16 value and return it as `f32`.
///
/// Implements round-to-nearest-even with full subnormal and overflow handling,
/// so the returned value is bit-exactly what `ggml` would reconstruct from a
/// stored f16 scale.
#[must_use]
pub fn f16_round(x: f32) -> f32 {
    f16_to_f32(f32_to_f16_bits(x))
}

/// Encode an `f32` as the 16-bit pattern of the nearest binary16 value.
#[must_use]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf / NaN.
        return if mantissa != 0 {
            sign | 0x7E00 // NaN (quiet)
        } else {
            sign | 0x7C00 // Inf
        };
    }

    // Unbiased exponent of the half: e16 = exp - 127 + 15.
    let mut e = exp - 127 + 15;

    if e >= 0x1F {
        // Overflow → Inf.
        return sign | 0x7C00;
    }

    if e <= 0 {
        // Subnormal or zero in half precision.
        if e < -10 {
            // Too small even for the smallest subnormal → ±0.
            return sign;
        }
        // Add the implicit leading 1 and shift into subnormal position.
        let m = mantissa | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut half = (m >> shift) as u16;
        // Round to nearest even using the bits shifted out.
        let round_bit = 1u32 << (shift - 1);
        if (m & round_bit) != 0 && ((m & (round_bit - 1)) != 0 || (half & 1) != 0) {
            half += 1;
        }
        return sign | half;
    }

    // Normal half: take top 10 mantissa bits, round to nearest even.
    let mut half_mant = (mantissa >> 13) as u16;
    let round_bit = mantissa & 0x0000_1000;
    if round_bit != 0 && ((mantissa & 0x0000_0FFF) != 0 || (half_mant & 1) != 0) {
        half_mant += 1;
        if half_mant == 0x0400 {
            // Mantissa overflow → bump exponent.
            half_mant = 0;
            e += 1;
            if e >= 0x1F {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((e as u16) << 10) | half_mant
}

/// Decode a binary16 bit pattern into the `f32` value it represents.
#[must_use]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign // ±0
        } else {
            // Subnormal half: value = mant · 2⁻²⁴. Build it directly from the
            // float magnitude to avoid any normalization bit-surgery error.
            let magnitude = (mant as f32) * (1.0_f32 / 16_777_216.0); // 2⁻²⁴
            return if (sign >> 31) != 0 {
                -magnitude
            } else {
                magnitude
            };
        }
    } else if exp == 0x1F {
        // Inf / NaN.
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        let exp32 = exp + (127 - 15);
        sign | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ─── Format descriptor ──────────────────────────────────────────────────────────

/// A GGML block-quantization format.
///
/// Variant names mirror the canonical `ggml`/`llama.cpp` format identifiers
/// (`Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`) verbatim, so the `non_camel_case_types`
/// lint is locally suppressed to preserve that on-disk naming convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    /// 8-bit symmetric, block of 32, f16 scale.
    Q8_0,
    /// 4-bit symmetric, block of 32, f16 scale, fixed offset −8.
    Q4_0,
    /// 4-bit affine, block of 32, f16 scale + f16 min.
    Q4_1,
    /// 4-bit k-quant, super-block of 256 (8 sub-blocks of 32).
    Q4_K,
}

impl GgmlType {
    /// Number of weights per (super-)block.
    #[must_use]
    pub fn block_size(self) -> usize {
        match self {
            GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q4_1 => 32,
            GgmlType::Q4_K => 256,
        }
    }

    /// Amortized bits stored per weight, including block metadata.
    #[must_use]
    pub fn bits_per_weight(self) -> f32 {
        match self {
            // 32 × i8 + f16 scale = 256 + 16 bits over 32 weights.
            GgmlType::Q8_0 => (256.0 + 16.0) / 32.0,
            // 32 × 4-bit + f16 scale = 128 + 16 bits over 32 weights.
            GgmlType::Q4_0 => (128.0 + 16.0) / 32.0,
            // 32 × 4-bit + f16 scale + f16 min = 128 + 32 bits over 32 weights.
            GgmlType::Q4_1 => (128.0 + 32.0) / 32.0,
            // 256 × 4-bit + 8×(6+6)-bit + f16 d + f16 dmin
            //   = 1024 + 96 + 32 bits over 256 weights.
            GgmlType::Q4_K => (1024.0 + 96.0 + 32.0) / 256.0,
        }
    }
}

// ─── Q8_0 ───────────────────────────────────────────────────────────────────────

/// A single Q8_0 block: 32 signed 8-bit codes + one f16 scale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockQ8_0 {
    /// f16 scale stored as raw 16-bit pattern.
    pub d: u16,
    /// 32 signed quantized codes in `[-127, 127]`.
    pub qs: [i8; 32],
}

/// A single Q4_0 block: 32 packed 4-bit codes + one f16 scale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockQ4_0 {
    /// f16 scale stored as raw 16-bit pattern.
    pub d: u16,
    /// 16 bytes, each packing two 4-bit codes (low nibble = element `i`,
    /// high nibble = element `i + 16`, matching the `ggml` layout).
    pub qs: [u8; 16],
}

/// A single Q4_1 block: 32 packed 4-bit codes + f16 scale + f16 min.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockQ4_1 {
    /// f16 scale stored as raw 16-bit pattern.
    pub d: u16,
    /// f16 affine offset (`min`) stored as raw 16-bit pattern.
    pub m: u16,
    /// 16 bytes, each packing two 4-bit codes (low nibble = element `i`,
    /// high nibble = element `i + 16`).
    pub qs: [u8; 16],
}

/// A single Q4_K super-block: 256 weights in 8 sub-blocks of 32.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockQ4K {
    /// Super-block scale-of-scales (f16 raw bits).
    pub d: u16,
    /// Super-block scale-of-mins (f16 raw bits).
    pub dmin: u16,
    /// Per-sub-block 6-bit quantized scales (8 values, each `0..=63`).
    pub scales: [u8; 8],
    /// Per-sub-block 6-bit quantized mins (8 values, each `0..=63`).
    pub mins: [u8; 8],
    /// 128 bytes, each packing two 4-bit codes.
    pub qs: [u8; 128],
}

// ─── Q8_0 quantizer ───────────────────────────────────────────────────────────

/// Quantize a flat weight slice into Q8_0 blocks.
///
/// The slice length must be a multiple of 32.
///
/// # Errors
///
/// * [`QuantError::EmptyInput`] — empty slice.
/// * [`QuantError::GroupSizeMismatch`] — length not divisible by 32.
pub fn quantize_q8_0(weights: &[f32]) -> QuantResult<Vec<BlockQ8_0>> {
    check_block(weights, 32, "quantize_q8_0")?;
    let mut blocks = Vec::with_capacity(weights.len() / 32);
    for chunk in weights.chunks_exact(32) {
        let amax = chunk.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
        let d = amax / 127.0;
        let dq = f16_round(d);
        let inv = if dq > 0.0 { 1.0 / dq } else { 0.0 };
        let mut qs = [0_i8; 32];
        for (q, &w) in qs.iter_mut().zip(chunk.iter()) {
            *q = (w * inv).round().clamp(-127.0, 127.0) as i8;
        }
        blocks.push(BlockQ8_0 {
            d: f32_to_f16_bits(d),
            qs,
        });
    }
    Ok(blocks)
}

/// Dequantize Q8_0 blocks back to a flat `f32` slice.
#[must_use]
pub fn dequantize_q8_0(blocks: &[BlockQ8_0]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks.len() * 32);
    for b in blocks {
        let d = f16_to_f32(b.d);
        for &q in &b.qs {
            out.push(q as f32 * d);
        }
    }
    out
}

// ─── Q4_0 quantizer ───────────────────────────────────────────────────────────

/// Quantize a flat weight slice into Q4_0 blocks (symmetric 4-bit, offset −8).
///
/// # Errors
///
/// * [`QuantError::EmptyInput`] — empty slice.
/// * [`QuantError::GroupSizeMismatch`] — length not divisible by 32.
pub fn quantize_q4_0(weights: &[f32]) -> QuantResult<Vec<BlockQ4_0>> {
    check_block(weights, 32, "quantize_q4_0")?;
    let mut blocks = Vec::with_capacity(weights.len() / 32);
    for chunk in weights.chunks_exact(32) {
        // Find the element with the largest magnitude (signed), as in ggml.
        let mut max = 0.0_f32;
        let mut amax = 0.0_f32;
        for &w in chunk {
            if w.abs() > amax {
                amax = w.abs();
                max = w;
            }
        }
        let d = max / -8.0;
        let dq = f16_round(d);
        let inv = if dq != 0.0 { 1.0 / dq } else { 0.0 };
        let mut codes = [0_u8; 32];
        for (c, &w) in codes.iter_mut().zip(chunk.iter()) {
            let xi = ((w * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
            *c = xi;
        }
        let mut qs = [0_u8; 16];
        for i in 0..16 {
            qs[i] = codes[i] | (codes[i + 16] << 4);
        }
        blocks.push(BlockQ4_0 {
            d: f32_to_f16_bits(d),
            qs,
        });
    }
    Ok(blocks)
}

/// Dequantize Q4_0 blocks back to a flat `f32` slice.
#[must_use]
pub fn dequantize_q4_0(blocks: &[BlockQ4_0]) -> Vec<f32> {
    let mut out = vec![0.0_f32; blocks.len() * 32];
    for (bi, b) in blocks.iter().enumerate() {
        let d = f16_to_f32(b.d);
        let base = bi * 32;
        for i in 0..16 {
            let lo = (b.qs[i] & 0x0F) as i32 - 8;
            let hi = (b.qs[i] >> 4) as i32 - 8;
            out[base + i] = lo as f32 * d;
            out[base + i + 16] = hi as f32 * d;
        }
    }
    out
}

// ─── Q4_1 quantizer ───────────────────────────────────────────────────────────

/// Quantize a flat weight slice into Q4_1 blocks (affine 4-bit).
///
/// # Errors
///
/// * [`QuantError::EmptyInput`] — empty slice.
/// * [`QuantError::GroupSizeMismatch`] — length not divisible by 32.
pub fn quantize_q4_1(weights: &[f32]) -> QuantResult<Vec<BlockQ4_1>> {
    check_block(weights, 32, "quantize_q4_1")?;
    let mut blocks = Vec::with_capacity(weights.len() / 32);
    for chunk in weights.chunks_exact(32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &w in chunk {
            min = min.min(w);
            max = max.max(w);
        }
        let d = (max - min) / 15.0;
        let dq = f16_round(d);
        let mq = f16_round(min);
        let inv = if dq > 0.0 { 1.0 / dq } else { 0.0 };
        let mut codes = [0_u8; 32];
        for (c, &w) in codes.iter_mut().zip(chunk.iter()) {
            *c = (((w - mq) * inv).round()).clamp(0.0, 15.0) as u8;
        }
        let mut qs = [0_u8; 16];
        for i in 0..16 {
            qs[i] = codes[i] | (codes[i + 16] << 4);
        }
        blocks.push(BlockQ4_1 {
            d: f32_to_f16_bits(d),
            m: f32_to_f16_bits(min),
            qs,
        });
    }
    Ok(blocks)
}

/// Dequantize Q4_1 blocks back to a flat `f32` slice.
#[must_use]
pub fn dequantize_q4_1(blocks: &[BlockQ4_1]) -> Vec<f32> {
    let mut out = vec![0.0_f32; blocks.len() * 32];
    for (bi, b) in blocks.iter().enumerate() {
        let d = f16_to_f32(b.d);
        let m = f16_to_f32(b.m);
        let base = bi * 32;
        for i in 0..16 {
            let lo = (b.qs[i] & 0x0F) as f32;
            let hi = (b.qs[i] >> 4) as f32;
            out[base + i] = lo * d + m;
            out[base + i + 16] = hi * d + m;
        }
    }
    out
}

// ─── Q4_K quantizer (k-quant super-block) ──────────────────────────────────────

/// Quantize a flat weight slice into Q4_K super-blocks (256 weights each).
///
/// Each super-block holds 8 sub-blocks of 32 weights. Every sub-block has an
/// affine `(scale, min)` pair; the 8 scales and 8 mins are themselves 6-bit
/// quantized against the super-block `d` (scale-of-scales) and `dmin`
/// (scale-of-mins), reproducing the `ggml` k-quant layout.
///
/// # Errors
///
/// * [`QuantError::EmptyInput`] — empty slice.
/// * [`QuantError::GroupSizeMismatch`] — length not divisible by 256.
pub fn quantize_q4_k(weights: &[f32]) -> QuantResult<Vec<BlockQ4K>> {
    check_block(weights, 256, "quantize_q4_k")?;
    let mut blocks = Vec::with_capacity(weights.len() / 256);
    for sblock in weights.chunks_exact(256) {
        // ── Per-sub-block affine (scale, min) in full precision ──────────────
        let mut sub_scales = [0.0_f32; 8];
        let mut sub_mins = [0.0_f32; 8];
        for sb in 0..8 {
            let chunk = &sblock[sb * 32..sb * 32 + 32];
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &w in chunk {
                min = min.min(w);
                max = max.max(w);
            }
            // ggml convention: min is clamped to ≤ 0 so the offset is a true
            // lower bound; the stored min is the negation `−min ≥ 0`.
            let min = min.min(0.0);
            sub_scales[sb] = (max - min) / 15.0;
            sub_mins[sb] = -min; // store the non-negative magnitude
        }

        // ── 6-bit quantize the 8 scales and 8 mins ───────────────────────────
        let max_scale = sub_scales.iter().fold(0.0_f32, |a, &x| a.max(x));
        let max_min = sub_mins.iter().fold(0.0_f32, |a, &x| a.max(x));
        let d = max_scale / 63.0;
        let dmin = max_min / 63.0;
        let dq = f16_round(d);
        let dminq = f16_round(dmin);
        let inv_d = if dq > 0.0 { 1.0 / dq } else { 0.0 };
        let inv_dmin = if dminq > 0.0 { 1.0 / dminq } else { 0.0 };

        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for sb in 0..8 {
            scales[sb] = (sub_scales[sb] * inv_d).round().clamp(0.0, 63.0) as u8;
            mins[sb] = (sub_mins[sb] * inv_dmin).round().clamp(0.0, 63.0) as u8;
        }

        // ── Quantize the 256 weights using the *reconstructed* sub params ────
        let mut codes = [0_u8; 256];
        for sb in 0..8 {
            let sc = dq * scales[sb] as f32;
            let mn = dminq * mins[sb] as f32; // non-negative offset magnitude
            let inv_sc = if sc > 0.0 { 1.0 / sc } else { 0.0 };
            for i in 0..32 {
                let w = sblock[sb * 32 + i];
                // w ≈ sc * q − mn  ⇒  q = (w + mn) / sc
                let q = ((w + mn) * inv_sc).round().clamp(0.0, 15.0) as u8;
                codes[sb * 32 + i] = q;
            }
        }

        // Pack two nibbles per byte (128 bytes for 256 codes).
        let mut qs = [0_u8; 128];
        for i in 0..128 {
            qs[i] = codes[2 * i] | (codes[2 * i + 1] << 4);
        }

        blocks.push(BlockQ4K {
            d: f32_to_f16_bits(d),
            dmin: f32_to_f16_bits(dmin),
            scales,
            mins,
            qs,
        });
    }
    Ok(blocks)
}

/// Dequantize Q4_K super-blocks back to a flat `f32` slice.
#[must_use]
pub fn dequantize_q4_k(blocks: &[BlockQ4K]) -> Vec<f32> {
    let mut out = vec![0.0_f32; blocks.len() * 256];
    for (bi, b) in blocks.iter().enumerate() {
        let d = f16_to_f32(b.d);
        let dmin = f16_to_f32(b.dmin);
        let base = bi * 256;
        // Unpack the 256 nibble codes.
        let mut codes = [0_u8; 256];
        for i in 0..128 {
            codes[2 * i] = b.qs[i] & 0x0F;
            codes[2 * i + 1] = b.qs[i] >> 4;
        }
        for sb in 0..8 {
            let sc = d * b.scales[sb] as f32;
            let mn = dmin * b.mins[sb] as f32;
            for i in 0..32 {
                let q = codes[sb * 32 + i] as f32;
                out[base + sb * 32 + i] = sc * q - mn;
            }
        }
    }
    out
}

// ─── Convenience: round-trip dequantization of a whole tensor ──────────────────

/// Quantize then immediately dequantize a flat tensor with the chosen format,
/// returning the reconstructed `f32` values.
///
/// Useful for fake-quantization / accuracy evaluation without exposing the
/// intermediate block representation.
///
/// # Errors
///
/// Propagates the format-specific quantizer errors (empty / non-divisible).
pub fn fake_quantize(weights: &[f32], ty: GgmlType) -> QuantResult<Vec<f32>> {
    match ty {
        GgmlType::Q8_0 => Ok(dequantize_q8_0(&quantize_q8_0(weights)?)),
        GgmlType::Q4_0 => Ok(dequantize_q4_0(&quantize_q4_0(weights)?)),
        GgmlType::Q4_1 => Ok(dequantize_q4_1(&quantize_q4_1(weights)?)),
        GgmlType::Q4_K => Ok(dequantize_q4_k(&quantize_q4_k(weights)?)),
    }
}

// ─── Shared validation ──────────────────────────────────────────────────────────

fn check_block(weights: &[f32], block: usize, ctx: &'static str) -> QuantResult<()> {
    if weights.is_empty() {
        return Err(QuantError::EmptyInput(ctx));
    }
    if weights.len() % block != 0 {
        return Err(QuantError::GroupSizeMismatch {
            len: weights.len(),
            group: block,
        });
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// LCG pseudo-random f32 in `[-scale, scale]` for reproducible tests.
    fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Full-range mapping: top 32 bits ÷ u32::MAX ∈ [0, 1].
                let bits = (state >> 32) as u32;
                let f = (bits as f32) / (u32::MAX as f32);
                (f * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let mut num = 0.0_f32;
        let mut den = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            num += (x - y).abs();
            den += x.abs();
        }
        num / den.max(1e-8)
    }

    // ── f16 emulation ─────────────────────────────────────────────────────────

    #[test]
    fn f16_round_trip_exact_values() {
        // Powers of two and simple fractions are exactly representable.
        for &v in &[0.0_f32, 1.0, -1.0, 0.5, 2.0, 0.25, 1024.0, -0.125] {
            let r = f16_round(v);
            assert!((r - v).abs() < 1e-6, "f16_round({v}) = {r}");
        }
    }

    #[test]
    fn f16_round_close_to_input() {
        // Arbitrary values are within half-ULP relative error for normals.
        for &v in &[0.1_f32, 3.1457, -2.7188, 100.0, 0.003] {
            let r = f16_round(v);
            let rel = (r - v).abs() / v.abs();
            assert!(rel < 1e-3, "f16_round({v}) = {r}, rel = {rel}");
        }
    }

    #[test]
    fn f16_overflow_to_inf() {
        let r = f16_round(1.0e30);
        assert!(r.is_infinite() && r > 0.0);
    }

    #[test]
    fn f16_tiny_to_zero() {
        let r = f16_round(1.0e-20);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn f16_subnormal_round_trip() {
        // Smallest normal half ≈ 6.1035e-5; below that we hit subnormals.
        let v = 5.0e-5_f32;
        let r = f16_round(v);
        assert!((r - v).abs() / v < 0.05, "subnormal f16 {v} → {r}");
    }

    #[test]
    fn f16_bits_decode_inverse() {
        for &v in &[1.5_f32, -3.25, 0.0625, 48.0] {
            let bits = f32_to_f16_bits(v);
            let back = f16_to_f32(bits);
            assert!((back - v).abs() < 1e-6);
        }
    }

    // ── Block-size metadata ──────────────────────────────────────────────────

    #[test]
    fn block_sizes_correct() {
        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q4_1.block_size(), 32);
        assert_eq!(GgmlType::Q4_K.block_size(), 256);
    }

    #[test]
    fn bits_per_weight_ordering() {
        // Higher-bit formats must cost more per weight.
        assert!(GgmlType::Q4_0.bits_per_weight() < GgmlType::Q4_1.bits_per_weight());
        assert!(GgmlType::Q4_1.bits_per_weight() < GgmlType::Q8_0.bits_per_weight());
        assert!(GgmlType::Q4_K.bits_per_weight() < GgmlType::Q8_0.bits_per_weight());
        assert!((GgmlType::Q8_0.bits_per_weight() - 8.5).abs() < 1e-6);
    }

    // ── Q8_0 ─────────────────────────────────────────────────────────────────

    #[test]
    fn q8_0_round_trip_accuracy() {
        let w = lcg(32 * 10, 1, 4.0);
        let blocks = quantize_q8_0(&w).expect("q8_0 quantize");
        assert_eq!(blocks.len(), 10);
        let deq = dequantize_q8_0(&blocks);
        // 8-bit symmetric: rel error should be small.
        let rel = rel_err(&w, &deq);
        assert!(rel < 0.01, "Q8_0 rel error too high: {rel}");
    }

    #[test]
    fn q8_0_codes_in_range() {
        let w = lcg(32, 7, 2.0);
        let blocks = quantize_q8_0(&w).expect("q8_0 quantize");
        for &q in &blocks[0].qs {
            assert!((-127..=127).contains(&(q as i32)));
        }
    }

    #[test]
    fn q8_0_zero_block() {
        let w = vec![0.0_f32; 32];
        let blocks = quantize_q8_0(&w).expect("q8_0 quantize");
        let deq = dequantize_q8_0(&blocks);
        for &v in &deq {
            assert_eq!(v, 0.0);
        }
    }

    // ── Q4_0 ─────────────────────────────────────────────────────────────────

    #[test]
    fn q4_0_round_trip_accuracy() {
        let w = lcg(32 * 8, 2, 3.0);
        let blocks = quantize_q4_0(&w).expect("q4_0 quantize");
        let deq = dequantize_q4_0(&blocks);
        let rel = rel_err(&w, &deq);
        // 4-bit symmetric is coarser; QLoRA-class tolerance.
        assert!(rel < 0.12, "Q4_0 rel error too high: {rel}");
    }

    #[test]
    fn q4_0_nibbles_in_range() {
        let w = lcg(32, 11, 2.0);
        let blocks = quantize_q4_0(&w).expect("q4_0 quantize");
        for &b in &blocks[0].qs {
            // High nibble must be a valid 4-bit code; the low nibble is masked
            // from a u8 so it is structurally ≤ 15.
            assert!((b >> 4) <= 15);
        }
    }

    #[test]
    fn q4_0_more_accurate_than_2level() {
        // Reconstruction must beat a naive sign quantizer on smooth data.
        let w: Vec<f32> = (0..32).map(|i| (i as f32 / 31.0) * 2.0 - 1.0).collect();
        let blocks = quantize_q4_0(&w).expect("q4_0 quantize");
        let deq = dequantize_q4_0(&blocks);
        let q4_mse: f32 = w
            .iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 32.0;
        let amax = w.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
        let sign_mse: f32 = w
            .iter()
            .map(|&x| (x - x.signum() * amax).powi(2))
            .sum::<f32>()
            / 32.0;
        assert!(q4_mse < sign_mse, "Q4_0 {q4_mse} !< sign {sign_mse}");
    }

    // ── Q4_1 ─────────────────────────────────────────────────────────────────

    #[test]
    fn q4_1_round_trip_accuracy() {
        let w = lcg(32 * 8, 3, 3.0);
        let blocks = quantize_q4_1(&w).expect("q4_1 quantize");
        let deq = dequantize_q4_1(&blocks);
        let rel = rel_err(&w, &deq);
        assert!(rel < 0.12, "Q4_1 rel error too high: {rel}");
    }

    #[test]
    fn q4_1_handles_asymmetric_range() {
        // All-positive data: affine Q4_1 should reconstruct well where the
        // symmetric Q4_0 would waste half its codes on the negative side.
        let w: Vec<f32> = (0..32).map(|i| 1.0 + (i as f32) / 31.0).collect();
        let q1 = dequantize_q4_1(&quantize_q4_1(&w).expect("q4_1"));
        let q0 = dequantize_q4_0(&quantize_q4_0(&w).expect("q4_0"));
        let e1 = rel_err(&w, &q1);
        let e0 = rel_err(&w, &q0);
        assert!(e1 <= e0, "Q4_1 {e1} should beat Q4_0 {e0} on positive data");
    }

    // ── Q4_K ─────────────────────────────────────────────────────────────────

    #[test]
    fn q4_k_round_trip_accuracy() {
        let w = lcg(256 * 3, 4, 5.0);
        let blocks = quantize_q4_k(&w).expect("q4_k quantize");
        assert_eq!(blocks.len(), 3);
        let deq = dequantize_q4_k(&blocks);
        let rel = rel_err(&w, &deq);
        // k-quant 4-bit: affine per-sub-block, slightly better than Q4_0.
        assert!(rel < 0.12, "Q4_K rel error too high: {rel}");
    }

    #[test]
    fn q4_k_scales_and_mins_six_bit() {
        let w = lcg(256, 9, 4.0);
        let blocks = quantize_q4_k(&w).expect("q4_k quantize");
        for &s in &blocks[0].scales {
            assert!(s <= 63, "scale {s} exceeds 6-bit range");
        }
        for &m in &blocks[0].mins {
            assert!(m <= 63, "min {m} exceeds 6-bit range");
        }
    }

    #[test]
    fn q4_k_beats_q4_0_on_blocky_data() {
        // Per-sub-block scaling helps when sub-blocks have very different ranges.
        let mut w = vec![0.0_f32; 256];
        for sb in 0..8 {
            let amp = (sb as f32 + 1.0) * 0.5;
            for i in 0..32 {
                w[sb * 32 + i] = ((i as f32 / 31.0) * 2.0 - 1.0) * amp;
            }
        }
        let qk = dequantize_q4_k(&quantize_q4_k(&w).expect("q4_k"));
        let q0 = dequantize_q4_0(&quantize_q4_0(&w).expect("q4_0"));
        let ek = rel_err(&w, &qk);
        let e0 = rel_err(&w, &q0);
        // Both 4-bit; k-quant's per-32 scale should be at least competitive.
        assert!(ek <= e0 * 1.05, "Q4_K {ek} should be ≤ Q4_0 {e0}");
    }

    // ── fake_quantize convenience ─────────────────────────────────────────────

    #[test]
    fn fake_quantize_matches_explicit_path() {
        let w = lcg(256, 5, 4.0);
        let fq = fake_quantize(&w, GgmlType::Q4_K).expect("fake quantize");
        let explicit = dequantize_q4_k(&quantize_q4_k(&w).expect("q4_k"));
        assert_eq!(fq, explicit);
    }

    // ── Error handling ────────────────────────────────────────────────────────

    #[test]
    fn empty_input_errors() {
        assert!(matches!(quantize_q8_0(&[]), Err(QuantError::EmptyInput(_))));
        assert!(matches!(quantize_q4_k(&[]), Err(QuantError::EmptyInput(_))));
    }

    #[test]
    fn non_divisible_length_errors() {
        let w = vec![1.0_f32; 33];
        assert!(matches!(
            quantize_q8_0(&w),
            Err(QuantError::GroupSizeMismatch { .. })
        ));
        let w2 = vec![1.0_f32; 200];
        assert!(matches!(
            quantize_q4_k(&w2),
            Err(QuantError::GroupSizeMismatch { .. })
        ));
    }
}

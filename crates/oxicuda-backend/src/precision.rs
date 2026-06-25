//! Pure-Rust reduced-precision storage emulation (no external crates).
//!
//! Mixed-precision GEMM and other Tensor-Core paths store operands in a
//! 16-bit format (IEEE-754 **binary16** or Google **bfloat16**) but
//! accumulate in `f32`. To give the [`CpuBackend`](crate::cpu::CpuBackend) a
//! faithful *numerical* reference for those paths — without pulling in the
//! `half` crate, which this zero-dependency crate must not do — this module
//! implements the exact bit-level rounding of an `f32` down to each 16-bit
//! format and back, using **round-to-nearest, ties-to-even** (the IEEE-754
//! default and what every GPU uses for the down-cast).
//!
//! Both functions return the value re-expanded to `f32`, i.e. the `f32` that
//! the hardware would actually hold after `f32 → {bf16, f16} → f32`. That is
//! precisely what a mixed-precision kernel sees as its inputs, so feeding
//! these rounded values into an `f32`-accumulating dot product reproduces the
//! storage error of a real Tensor-Core GEMM while keeping the accumulator at
//! full `f32` width.
//!
//! # Why this is not a simple truncation
//!
//! * **bfloat16** shares `f32`'s 8-bit exponent, so it is the top 16 bits of
//!   the `f32` — but a correct down-cast still has to *round* the discarded
//!   low 16 bits to nearest-even, not merely drop them (and that rounding can
//!   carry all the way into the exponent, e.g. just below a power of two).
//! * **binary16** has a 5-bit exponent with bias 15 (vs. `f32`'s bias 127),
//!   so the exponent must be re-biased, overflow must saturate to ±∞, and
//!   small magnitudes must round into **subnormal** binary16 (or to zero),
//!   which truncation would get wrong.

/// Round-to-nearest-even helper shared by both formats.
///
/// Given the *kept* high bits `kept`, the position-weight of the rounding
/// bit, and whether any bit strictly below the rounding bit is set
/// (`sticky`), decide whether to round the kept mantissa up by one ULP.
///
/// `round_bit` is the value of the single bit just below the retained
/// precision; `sticky` is the OR of everything below that. Ties (round bit
/// set, nothing sticky) round to make the result even.
#[inline]
const fn round_to_even(kept: u32, round_bit: bool, sticky: bool) -> u32 {
    if round_bit && (sticky || (kept & 1 == 1)) {
        kept + 1
    } else {
        kept
    }
}

/// Round an `f32` to **bfloat16** storage and return the resulting `f32`.
///
/// bfloat16 keeps the sign, the full 8-bit exponent, and the top 7 bits of
/// the mantissa; the low 16 bits of the `f32` are rounded to nearest-even
/// into bit 16. `NaN` is preserved as a quiet `NaN`; ±∞ is preserved.
#[must_use]
pub fn round_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    // Propagate NaN (any payload) as a canonical quiet bf16 NaN.
    if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
        // exponent all ones, mantissa non-zero ⇒ NaN.
        return f32::from_bits(bits | 0x0040_0000); // force quiet bit.
    }
    // Round the discarded low 16 bits (bit 15 is the round bit) to even.
    let kept = bits >> 16; // sign(1) | exp(8) | mant_hi(7)
    let round_bit = (bits & 0x0000_8000) != 0;
    let sticky = (bits & 0x0000_7FFF) != 0;
    let rounded = round_to_even(kept, round_bit, sticky);
    // A mantissa carry can ripple into the exponent; since `rounded` is the
    // top 16 bits, shifting back left by 16 reconstitutes the f32 exactly.
    f32::from_bits(rounded << 16)
}

/// Round an `f32` to IEEE-754 **binary16** storage and return the resulting
/// `f32`.
///
/// Handles exponent re-bias (127 → 15), overflow saturation to ±∞, gradual
/// underflow into binary16 subnormals (and flush to ±0 below the subnormal
/// range), and round-to-nearest-even throughout. `NaN` maps to a quiet
/// binary16 `NaN`. The returned `f32` is the exact value of that binary16
/// bit pattern.
#[must_use]
pub fn round_to_f16(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

/// Encode an `f32` into the 16 bits of an IEEE-754 binary16, with
/// round-to-nearest-even. Returned as a `u16` bit pattern.
#[must_use]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16; // sign in bit 15.
    let exp = ((bits >> 23) & 0xFF) as i32; // unbiased f32 exponent field.
    let mant = bits & 0x007F_FFFF; // 23-bit mantissa.

    // NaN / Inf: f32 exponent field all ones.
    if exp == 0xFF {
        return if mant != 0 {
            // NaN → quiet binary16 NaN (preserve sign, set MSB of mantissa).
            sign | 0x7E00
        } else {
            // ±∞.
            sign | 0x7C00
        };
    }

    // Re-bias: f32 bias 127 → f16 bias 15.
    let unbiased = exp - 127;
    let f16_exp = unbiased + 15;

    if f16_exp >= 0x1F {
        // Overflow the binary16 exponent range ⇒ saturate to ±∞.
        return sign | 0x7C00;
    }

    if f16_exp <= 0 {
        // Subnormal (or zero) binary16. Build the implicit-1 mantissa and
        // shift it right so the value is represented as a 10-bit subnormal,
        // rounding the shifted-out bits to nearest-even.
        if f16_exp < -10 {
            // Even the round bit is below the subnormal range ⇒ ±0.
            return sign;
        }
        // 24-bit significand with the implicit leading 1.
        let significand = mant | 0x0080_0000;
        // Number of bits to discard: 13 (the f32→f16 mantissa-width gap) plus
        // the extra right-shift to reach the subnormal exponent.
        let shift = 14 - f16_exp; // f16_exp in [-10, 0] ⇒ shift in [14, 24].
        debug_assert!((14..=24).contains(&shift));
        let kept = significand >> shift;
        let round_bit = (significand >> (shift - 1)) & 1 == 1;
        let sticky = (significand & ((1u32 << (shift - 1)) - 1)) != 0;
        let mant16 = round_to_even(kept, round_bit, sticky) as u16;
        // `mant16` may have rounded up into the implicit bit (0x400),
        // promoting the value to the smallest normal — its bit pattern is
        // exactly `sign | 0x0400`, which the encoding below produces for
        // free, so no special-case is required.
        return sign | mant16;
    }

    // Normal binary16. Keep the top 10 mantissa bits, round the low 13 to
    // even. A mantissa carry that overflows 10 bits ripples into the
    // exponent automatically because we add the carry to the packed value.
    let kept = mant >> 13;
    let round_bit = (mant & 0x0000_1000) != 0;
    let sticky = (mant & 0x0000_0FFF) != 0;
    let mant16 = round_to_even(kept, round_bit, sticky);
    // Pack exponent and mantissa, then add: a rounded mantissa of 0x400
    // (carry out of 10 bits) correctly increments the exponent field.
    let packed = ((f16_exp as u16) << 10) | (mant16 as u16 & 0x03FF);
    let carry = (mant16 >> 10) as u16; // 0 or 1.
    sign | (packed + (carry << 10))
}

/// Decode a binary16 bit pattern back into the `f32` it represents (exactly).
#[must_use]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // ±0.
            return f32::from_bits(sign);
        }
        // Subnormal binary16: value = mant * 2^-24. Normalize into an f32
        // normal by left-shifting the mantissa until the implicit leading 1
        // reaches bit 10 (0x400), counting the shifts `s`. After `s` shifts
        // the value is `1.frac * 2^(-14 - s)`, so the f32 biased exponent is
        // `(-14 - s) + 127 = 113 - s`.
        let mut m = mant;
        let mut s: u32 = 0;
        while (m & 0x0400) == 0 {
            m <<= 1;
            s += 1;
        }
        m &= 0x03FF; // drop the now-explicit leading 1.
        let f32_exp = 113 - s; // s ∈ [1, 10] ⇒ f32_exp ∈ [103, 112].
        return f32::from_bits(sign | (f32_exp << 23) | (m << 13));
    }

    if exp == 0x1F {
        // Inf / NaN.
        return if mant == 0 {
            f32::from_bits(sign | 0x7F80_0000)
        } else {
            f32::from_bits(sign | 0x7FC0_0000) // quiet NaN.
        };
    }

    // Normal binary16 → f32: re-bias exponent 15 → 127 and widen mantissa.
    let f32_exp = exp + (127 - 15);
    f32::from_bits(sign | (f32_exp << 23) | (mant << 13))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_exact_for_representable_values() {
        // Values whose low 16 mantissa bits are zero survive bf16 exactly.
        for &v in &[0.0f32, 1.0, -1.0, 2.0, 0.5, -0.25, 256.0, 1.5, 3.0, 100.0] {
            assert_eq!(round_to_bf16(v), v, "bf16 round-trip changed {v}");
        }
    }

    #[test]
    fn bf16_is_top_16_bits_with_rounding() {
        // 1.0 + 2^-8 has its set bit at mantissa position 15 (the bf16 round
        // bit) → ties-to-even rounds it to 1.0 (mantissa even) here, because
        // the kept mantissa LSB is 0 and nothing is sticky.
        let x = f32::from_bits(0x3F80_0000 | 0x0000_8000); // 1.0 + ulp_bf16/2
        let r = round_to_bf16(x);
        assert_eq!(r, 1.0, "exact tie should round to even (1.0)");

        // One above the tie (sticky set) must round up to 1 + 2^-7.
        let x2 = f32::from_bits(0x3F80_0000 | 0x0000_8001);
        let up = round_to_bf16(x2);
        assert_eq!(up, f32::from_bits(0x3F81_0000));
        assert!(up > 1.0);
    }

    #[test]
    fn bf16_rounding_can_carry_into_exponent() {
        // Just below 2.0, rounding up the mantissa must carry into the
        // exponent and produce exactly 2.0.
        let almost_two = f32::from_bits(0x3FFF_FFFF); // largest f32 < 2.0
        assert_eq!(round_to_bf16(almost_two), 2.0);
    }

    #[test]
    fn bf16_preserves_inf_and_nan() {
        assert_eq!(round_to_bf16(f32::INFINITY), f32::INFINITY);
        assert_eq!(round_to_bf16(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(round_to_bf16(f32::NAN).is_nan());
    }

    #[test]
    fn f16_exact_for_representable_values() {
        // Small integers and simple fractions are exactly representable in
        // binary16, so the round-trip must be the identity.
        for &v in &[
            0.0f32, 1.0, -1.0, 2.0, 0.5, -0.25, 0.125, 1.5, 3.0, 10.0, 1024.0, -2048.0,
        ] {
            assert_eq!(round_to_f16(v), v, "f16 round-trip changed {v}");
        }
    }

    #[test]
    fn f16_known_bit_patterns() {
        // 1.0 = 0x3C00, 2.0 = 0x4000, -2.0 = 0xC000, 0.5 = 0x3800.
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(-2.0), 0xC000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        // Round-trip the bit patterns back.
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0x3800), 0.5);
    }

    #[test]
    fn f16_overflow_saturates_to_infinity() {
        // 70000 exceeds the binary16 max (~65504) ⇒ +∞.
        assert_eq!(round_to_f16(70_000.0), f32::INFINITY);
        assert_eq!(round_to_f16(-70_000.0), f32::NEG_INFINITY);
        // Largest finite binary16 is 65504, exactly representable.
        assert_eq!(round_to_f16(65_504.0), 65_504.0);
    }

    #[test]
    fn f16_smallest_normal_and_subnormals() {
        // Smallest positive normal binary16 = 2^-14 (0x0400).
        let min_normal = f32::from_bits(0x3880_0000); // 2^-14
        assert_eq!(round_to_f16(min_normal), min_normal);
        assert_eq!(f32_to_f16_bits(min_normal), 0x0400);

        // Smallest positive subnormal binary16 = 2^-24 (0x0001).
        let min_sub = (2.0f32).powi(-24);
        assert_eq!(f32_to_f16_bits(min_sub), 0x0001);
        assert_eq!(round_to_f16(min_sub), min_sub);

        // A value well below the subnormal range flushes to zero.
        let tiny = (2.0f32).powi(-30);
        assert_eq!(round_to_f16(tiny), 0.0);
    }

    #[test]
    fn f16_subnormal_round_to_even() {
        // 2^-25 is exactly half of the smallest subnormal (2^-24); ties to
        // even ⇒ rounds to zero (even).
        let half_ulp = (2.0f32).powi(-25);
        assert_eq!(round_to_f16(half_ulp), 0.0);
        // Just above the half-ulp rounds up to the smallest subnormal.
        let above = f32::from_bits(half_ulp.to_bits() + 1);
        assert_eq!(round_to_f16(above), (2.0f32).powi(-24));
    }

    #[test]
    fn f16_rounding_carry_into_exponent() {
        // Largest f32 strictly below 1.0 rounds up to 1.0 in binary16.
        let almost_one = f32::from_bits(0x3F7F_FFFF);
        assert_eq!(round_to_f16(almost_one), 1.0);
    }

    #[test]
    fn f16_preserves_inf_and_nan() {
        assert_eq!(round_to_f16(f32::INFINITY), f32::INFINITY);
        assert_eq!(round_to_f16(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(round_to_f16(f32::NAN).is_nan());
    }

    #[test]
    fn f16_negative_subnormal_round_trip() {
        // Negative smallest subnormal must keep its sign.
        let v = -(2.0f32).powi(-24);
        assert_eq!(round_to_f16(v), v);
        assert_eq!(f32_to_f16_bits(v), 0x8001);
    }
}

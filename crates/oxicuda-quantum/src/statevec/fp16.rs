//! FP16 / BF16 amplitude storage for memory-bound large-qubit simulation.
//!
//! A dense state vector dominates memory: `n` qubits need `2^n` complex
//! amplitudes, and at `f32` that is `8 · 2^n` bytes. Halving the amplitude width
//! to 16 bits (per real/imag component) halves the footprint, letting one extra
//! qubit fit in the same memory budget — the standard memory-bound trade in
//! large-scale state-vector simulation, accepting reduced precision.
//!
//! This module provides pure-Rust, dependency-free conversions for the two
//! 16-bit formats used on modern accelerators:
//!
//! * **IEEE-754 binary16** ([`f32_to_f16`] / [`f16_to_f32`]) — 1 sign, 5
//!   exponent, 10 mantissa bits; correctly handles subnormals, overflow to ±∞,
//!   and round-to-nearest-even.
//! * **bfloat16** ([`f32_to_bf16`] / [`bf16_to_f32`]) — 1 sign, 8 exponent, 7
//!   mantissa bits; the truncated top 16 bits of an `f32` with round-to-nearest.
//!
//! [`HalfStateVector`] stores complex amplitudes as packed `(re16, im16)` pairs
//! in either format and converts to/from a full-precision [`StateVector`].

use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

// ─── IEEE-754 binary16 ────────────────────────────────────────────────────────

/// Convert an `f32` to IEEE-754 binary16 bits (round-to-nearest-even).
#[must_use]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    // Unbiased exponent and mantissa of the f32.
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf / NaN.
        return if mant != 0 {
            sign | 0x7e00 // quiet NaN
        } else {
            sign | 0x7c00 // ±inf
        };
    }

    // Rebias exponent from 127 (f32) to 15 (f16).
    let new_exp = exp - 127 + 15;

    if new_exp >= 0x1f {
        // Overflow → ±inf.
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        // Subnormal or zero in f16.
        if new_exp < -10 {
            // Too small even for the smallest subnormal → signed zero.
            return sign;
        }
        // Add the implicit leading 1, then shift into subnormal position.
        let mant_with_implicit = mant | 0x0080_0000;
        let shift = (14 - new_exp) as u32; // 1 - new_exp + 13
        let mut half_mant = mant_with_implicit >> shift;
        // Round-to-nearest-even using the bits shifted out.
        let remainder = mant_with_implicit & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half_mant & 1) == 1) {
            half_mant += 1;
        }
        return sign | (half_mant as u16);
    }

    // Normal f16.
    let mut half_mant = (mant >> 13) as u16;
    let remainder = mant & 0x0000_1fff;
    let halfway = 0x0000_1000u32;
    let mut he = new_exp as u16;
    if remainder > halfway || (remainder == halfway && (half_mant & 1) == 1) {
        half_mant += 1;
        if half_mant == 0x400 {
            // Mantissa overflow → bump exponent, mantissa wraps to 0.
            half_mant = 0;
            he += 1;
            if he >= 0x1f {
                return sign | 0x7c00; // overflowed to inf
            }
        }
    }
    sign | (he << 10) | half_mant
}

/// Convert IEEE-754 binary16 bits to `f32` (exact).
#[must_use]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            // Signed zero.
            return f32::from_bits(sign);
        }
        // Subnormal: normalize.
        let mut e = -1i32;
        let mut m = mant;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03ff; // drop the now-explicit leading bit
        let f32_exp = ((e + 1 - 15 + 127) as u32) << 23;
        let f32_mant = m << 13;
        return f32::from_bits(sign | f32_exp | f32_mant);
    }
    if exp == 0x1f {
        // Inf / NaN.
        let f32_mant = mant << 13;
        return f32::from_bits(sign | 0x7f80_0000 | f32_mant);
    }
    // Normal.
    let f32_exp = (exp + 127 - 15) << 23;
    let f32_mant = mant << 13;
    f32::from_bits(sign | f32_exp | f32_mant)
}

// ─── bfloat16 ─────────────────────────────────────────────────────────────────

/// Convert an `f32` to bfloat16 bits (round-to-nearest-even).
#[must_use]
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Preserve NaN (set a mantissa bit in the high half).
        return ((bits >> 16) as u16) | 0x0040;
    }
    // Round-to-nearest-even on the discarded low 16 bits.
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

/// Convert bfloat16 bits to `f32` (exact; bf16 is the top 16 bits of an f32).
#[must_use]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Selects the 16-bit amplitude format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfFormat {
    /// IEEE-754 binary16.
    Ieee,
    /// bfloat16.
    Bfloat,
}

/// A complex amplitude stored as two packed 16-bit reals `(re, im)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalfComplex {
    pub re: u16,
    pub im: u16,
}

/// A state vector whose amplitudes are stored in a 16-bit format.
///
/// Uses half the memory of an `f32` [`StateVector`]; conversions go through
/// [`HalfStateVector::from_dense`] / [`HalfStateVector::to_dense`].
#[derive(Debug, Clone)]
pub struct HalfStateVector {
    pub amps: Vec<HalfComplex>,
    pub n_qubits: usize,
    pub format: HalfFormat,
}

impl HalfStateVector {
    /// Pack a dense `f32` state vector into the chosen 16-bit format.
    #[must_use]
    pub fn from_dense(sv: &StateVector, format: HalfFormat) -> Self {
        let amps = sv.amps.iter().map(|a| pack(*a, format)).collect::<Vec<_>>();
        Self {
            amps,
            n_qubits: sv.n_qubits,
            format,
        }
    }

    /// Unpack back to a full-precision `f32` [`StateVector`].
    ///
    /// # Errors
    /// Propagates the dense constructor's qubit-count validation.
    pub fn to_dense(&self) -> QuantumResult<StateVector> {
        if self.n_qubits == 0 || self.n_qubits > 30 {
            return Err(QuantumError::InvalidQubitCount { n: self.n_qubits });
        }
        let amps = self
            .amps
            .iter()
            .map(|h| unpack(*h, self.format))
            .collect::<Vec<_>>();
        Ok(StateVector {
            amps,
            n_qubits: self.n_qubits,
        })
    }

    /// Memory footprint in bytes (4 bytes per amplitude vs 8 for `f32`).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.amps.len() * std::mem::size_of::<HalfComplex>()
    }
}

fn pack(z: Complex32, format: HalfFormat) -> HalfComplex {
    match format {
        HalfFormat::Ieee => HalfComplex {
            re: f32_to_f16(z.re),
            im: f32_to_f16(z.im),
        },
        HalfFormat::Bfloat => HalfComplex {
            re: f32_to_bf16(z.re),
            im: f32_to_bf16(z.im),
        },
    }
}

fn unpack(h: HalfComplex, format: HalfFormat) -> Complex32 {
    match format {
        HalfFormat::Ieee => Complex32::new(f16_to_f32(h.re), f16_to_f32(h.im)),
        HalfFormat::Bfloat => Complex32::new(bf16_to_f32(h.re), bf16_to_f32(h.im)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_exact_powers_of_two_roundtrip() {
        for &v in &[0.0_f32, 1.0, 2.0, 0.5, -1.0, -0.25, 4.0, 0.125] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() < 1e-6, "v={v} back={back}");
        }
    }

    #[test]
    fn f16_known_bit_patterns() {
        // 1.0 → 0x3c00, -2.0 → 0xc000, 0.0 → 0x0000.
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert!((f16_to_f32(0x3c00) - 1.0).abs() < 1e-9);
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn f16_overflow_to_inf() {
        let h = f32_to_f16(1.0e30);
        assert!(f16_to_f32(h).is_infinite());
    }

    #[test]
    fn f16_subnormal_roundtrip_small() {
        // Smallest f16 subnormal ≈ 5.96e-8.
        let tiny = 6.0e-8_f32;
        let back = f16_to_f32(f32_to_f16(tiny));
        assert!(back > 0.0 && back < 1e-6, "back={back}");
    }

    #[test]
    fn f16_rounds_to_nearest() {
        // 1/3 ≈ 0.333…; nearest f16 should be within one ULP (~0.0005 near 0.33).
        let v = 1.0_f32 / 3.0;
        let back = f16_to_f32(f32_to_f16(v));
        assert!((back - v).abs() < 5e-4, "v={v} back={back}");
    }

    #[test]
    fn bf16_roundtrip_keeps_exponent_range() {
        for &v in &[1.0_f32, -1.0, 1.0e20, 1.0e-20, 0.0, 123.5, -0.001] {
            let back = bf16_to_f32(f32_to_bf16(v));
            // bf16 keeps the full exponent, so relative error ≤ ~2^-7.
            if v == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                let rel = ((back - v) / v).abs();
                assert!(rel < 0.01, "v={v} back={back} rel={rel}");
            }
        }
    }

    #[test]
    fn bf16_is_truncated_top_half() {
        // bf16 of 1.0 is the top 16 bits of f32(1.0)=0x3f800000 → 0x3f80.
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert!((bf16_to_f32(0x3f80) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn state_vector_pack_unpack_ieee() {
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let sv = StateVector {
            amps: vec![
                Complex32::new(inv_sqrt2, 0.0),
                Complex32::new(0.0, inv_sqrt2),
            ],
            n_qubits: 1,
        };
        let half = HalfStateVector::from_dense(&sv, HalfFormat::Ieee);
        let back = half.to_dense().expect("dense");
        for (a, b) in back.amps.iter().zip(sv.amps.iter()) {
            assert!((a.re - b.re).abs() < 1e-3, "re {a:?} vs {b:?}");
            assert!((a.im - b.im).abs() < 1e-3, "im {a:?} vs {b:?}");
        }
        // Memory: 2 amps × 4 bytes = 8 (vs 16 for f32).
        assert_eq!(half.bytes(), 8);
    }

    #[test]
    fn state_vector_pack_unpack_bf16_preserves_norm() {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(2024);
        let mut amps = Vec::with_capacity(8);
        for _ in 0..8 {
            amps.push(Complex32::new(rng.next_normal(), rng.next_normal()));
        }
        let mut sv = StateVector { amps, n_qubits: 3 };
        sv.normalize_inplace();
        let half = HalfStateVector::from_dense(&sv, HalfFormat::Bfloat);
        let back = half.to_dense().expect("dense");
        let norm: f32 = back.amps.iter().map(|a| a.norm_sqr()).sum();
        // bf16 has ~3 decimal digits; norm stays close to 1.
        assert!((norm - 1.0).abs() < 5e-2, "norm={norm}");
    }
}

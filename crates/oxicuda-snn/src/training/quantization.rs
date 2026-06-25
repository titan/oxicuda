//! Quantisation-aware SNN training: fake-quant with a straight-through estimator.
//!
//! Deploying spiking networks on low-precision accelerators requires simulating
//! quantisation during training so the model learns weights that survive being
//! rounded. This module provides three building blocks.
//!
//! # Symmetric INT8 weight quantisation
//!
//! Per-tensor symmetric quantisation maps a real weight `w` to an 8-bit signed
//! integer and back:
//!
//! ```text
//! scale = max|w| / 127
//! q     = clamp(round(w / scale), −127, 127)
//! ŵ     = q · scale
//! ```
//!
//! The range is `[−127, 127]` (not `−128`) to keep the quantiser symmetric about
//! zero, which matches how most inference kernels handle signed activations.
//!
//! # Straight-Through Estimator (STE)
//!
//! Rounding has zero derivative almost everywhere, so the forward pass returns
//! the fake-quantised value `ŵ` while the backward pass copies the incoming
//! gradient through **unchanged inside the representable range** and zeroes it
//! where the value saturated (Bengio et al. 2013):
//!
//! ```text
//! ∂ŵ/∂w ≈ 1   if −127 ≤ round(w/scale) ≤ 127,
//!         0   otherwise.
//! ```
//!
//! # FP8 E4M3 simulation
//!
//! [`crate::training::quantization::Fp8E4M3Quantizer`] rounds an `f32` to the nearest value representable in the
//! OCP `e4m3` format: 1 sign bit, 4 exponent bits (bias 7) and 3 mantissa bits.
//! Normals use exponent fields `1..=15`, subnormals use field `0`, the
//! all-ones-mantissa/all-ones-exponent pattern is reserved for NaN, and the
//! maximum finite magnitude is `448 = 2⁸ · 1.75`. Rounding is round-to-nearest,
//! ties-to-even on the 3-bit mantissa; overflow and non-finite inputs saturate
//! to `±448`.

use crate::error::{SnnError, SnnResult};

/// Quantisation bound for symmetric signed INT8 (`±127`, not `±128`).
pub const INT8_QMAX: f32 = 127.0;

/// Maximum finite magnitude representable in FP8 E4M3 (`2⁸ · 1.75`).
pub const FP8_E4M3_MAX: f32 = 448.0;

/// Smallest positive normal in FP8 E4M3: `2^(1−7) · (1 + 0) = 2⁻⁶`.
pub const FP8_E4M3_MIN_NORMAL: f32 = 0.015_625;

/// Smallest positive subnormal in FP8 E4M3: `2⁻⁶ · (1/8) = 2⁻⁹`.
pub const FP8_E4M3_MIN_SUBNORMAL: f32 = 0.001_953_125;

/// Per-tensor symmetric INT8 weight quantiser.
///
/// The scale is derived from the tensor being quantised (`max|w| / 127`); the
/// quantiser itself is stateless beyond an optional cached scale recorded by
/// [`Int8Quantizer::quantize`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Int8Quantizer {
    /// Most recently computed per-tensor scale (`0.0` until first use).
    last_scale: f32,
}

impl Int8Quantizer {
    /// Create a fresh quantiser with no cached scale.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the most recently computed per-tensor scale.
    #[must_use]
    #[inline]
    pub fn last_scale(&self) -> f32 {
        self.last_scale
    }

    /// Compute the symmetric per-tensor scale `max|w| / 127` for a weight slice.
    ///
    /// A slice that is exactly zero everywhere yields a unit scale so that the
    /// round-trip is the identity (all quantised values are zero) instead of a
    /// division by zero.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `weights` is empty.
    pub fn compute_scale(weights: &[f32]) -> SnnResult<f32> {
        if weights.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        let max_abs = weights.iter().fold(0.0_f32, |m, &w| m.max(w.abs()));
        if max_abs <= 0.0 || !max_abs.is_finite() {
            Ok(1.0)
        } else {
            Ok(max_abs / INT8_QMAX)
        }
    }

    /// Quantise a weight tensor to fake-INT8, returning `(integers, dequantised)`.
    ///
    /// `integers[i] = clamp(round(w_i / scale), −127, 127)` (stored as `f32`),
    /// and `dequantised[i] = integers[i] · scale`. The computed scale is cached
    /// in `self`.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `weights` is empty.
    pub fn quantize(&mut self, weights: &[f32]) -> SnnResult<(Vec<f32>, Vec<f32>)> {
        let scale = Self::compute_scale(weights)?;
        self.last_scale = scale;
        let inv = 1.0 / scale;
        let mut q = Vec::with_capacity(weights.len());
        let mut deq = Vec::with_capacity(weights.len());
        for &w in weights {
            let qi = round_half_even(w * inv).clamp(-INT8_QMAX, INT8_QMAX);
            q.push(qi);
            deq.push(qi * scale);
        }
        Ok((q, deq))
    }

    /// Unsigned activation / spike-rate quantiser to `n_levels` uniform bins.
    ///
    /// Spike rates live in `[0, 1]`; this maps each value to one of `n_levels`
    /// reconstruction points `k / (n_levels − 1)` via round-to-nearest, clamping
    /// inputs to `[0, 1]` first. With `n_levels = 2` this is a hard threshold at
    /// `0.5`, recovering binary spikes.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `acts` is empty, or
    /// [`SnnError::OutOfRange`] if `n_levels < 2`.
    pub fn quantize_activations(acts: &[f32], n_levels: u32) -> SnnResult<Vec<f32>> {
        if acts.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if n_levels < 2 {
            return Err(SnnError::OutOfRange {
                name: "n_levels".into(),
                val: n_levels as f32,
            });
        }
        let steps = (n_levels - 1) as f32;
        let out = acts
            .iter()
            .map(|&a| {
                let c = a.clamp(0.0, 1.0);
                let level = (c * steps).round();
                level / steps
            })
            .collect();
        Ok(out)
    }
}

/// Round-to-nearest, ties-to-even of an `f32` to an integral `f32`.
///
/// `f32::round` rounds halves away from zero; this helper instead breaks ties
/// toward the even integer, matching the rounding convention used by hardware
/// quantisers and by the FP8 mantissa rounding below.
#[must_use]
#[inline]
pub fn round_half_even(x: f32) -> f32 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        // Exactly halfway: pick the even neighbour.
        let half = floor * 0.5;
        if (half - half.floor()).abs() < 1e-6 {
            floor // floor is even
        } else {
            floor + 1.0
        }
    }
}

/// Fake-quantise a weight tensor to symmetric INT8 and dequantise back.
///
/// Convenience free function returning only the dequantised tensor `ŵ` (the
/// forward output of an INT8 fake-quant node), discarding the integer codes.
///
/// # Errors
///
/// Returns [`SnnError::EmptyInput`] if `weights` is empty.
pub fn fake_quantize_weights(weights: &[f32]) -> SnnResult<Vec<f32>> {
    let mut q = Int8Quantizer::new();
    let (_codes, deq) = q.quantize(weights)?;
    Ok(deq)
}

/// Straight-through estimator backward mask for symmetric INT8.
///
/// Returns a multiplicative mask (`1.0` pass / `0.0` block) per weight: the
/// gradient passes through where `round(w/scale)` lies inside `[−127, 127]` and
/// is zeroed where the value would saturate. The `scale` is the per-tensor scale
/// used in the forward pass.
///
/// # Errors
///
/// Returns [`SnnError::EmptyInput`] if `weights` is empty, or
/// [`SnnError::OutOfRange`] if `scale` is non-finite or non-positive.
pub fn ste_backward_mask(weights: &[f32], scale: f32) -> SnnResult<Vec<f32>> {
    if weights.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "scale".into(),
            val: scale,
        });
    }
    let inv = 1.0 / scale;
    let out = weights
        .iter()
        .map(|&w| {
            let q = round_half_even(w * inv);
            if (-INT8_QMAX..=INT8_QMAX).contains(&q) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    Ok(out)
}

/// Apply the STE: forward fake-quant output paired with a backward gradient mask.
///
/// Returns `(w_hat, mask)`: `w_hat` is the dequantised forward value and `mask`
/// is the per-element straight-through multiplier (see [`ste_backward_mask`]).
///
/// # Errors
///
/// Returns [`SnnError::EmptyInput`] if `weights` is empty.
pub fn ste_fake_quant(weights: &[f32]) -> SnnResult<(Vec<f32>, Vec<f32>)> {
    let scale = Int8Quantizer::compute_scale(weights)?;
    let mut q = Int8Quantizer::new();
    let (_codes, deq) = q.quantize(weights)?;
    let mask = ste_backward_mask(weights, scale)?;
    Ok((deq, mask))
}

/// Simulated FP8 E4M3 quantiser (1 sign / 4 exponent / 3 mantissa, bias 7).
///
/// Performs the round-to-nearest-even of an `f32` to the nearest `e4m3` value
/// entirely in software, handling normals, subnormals and saturation. Stateless.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8E4M3Quantizer;

impl Fp8E4M3Quantizer {
    /// FP8 E4M3 exponent bias.
    const BIAS: i32 = 7;
    /// Number of explicit mantissa bits.
    const MANT_BITS: i32 = 3;
    /// Number of mantissa reconstruction steps (`2³ = 8`).
    const MANT_DIV: f32 = 8.0;
    /// Minimum unbiased exponent of a normal (`1 − bias`).
    const MIN_NORMAL_EXP: i32 = 1 - Self::BIAS; // −6
    /// Maximum unbiased exponent of a normal (`15 − bias`).
    const MAX_NORMAL_EXP: i32 = 15 - Self::BIAS; // 8

    /// Construct a stateless FP8 E4M3 quantiser.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Round a single `f32` to the nearest representable FP8 E4M3 value.
    ///
    /// Zero (either sign) maps to `+0.0`. Non-finite inputs and any magnitude
    /// above `448` saturate to `±448`. Subnormals are rounded against the
    /// fixed subnormal step `2⁻⁹`.
    #[must_use]
    pub fn quantize_scalar(&self, x: f32) -> f32 {
        if x == 0.0 {
            return 0.0;
        }
        let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
        if !x.is_finite() {
            // NaN/Inf have no e4m3 finite encoding → saturate.
            return sign * FP8_E4M3_MAX;
        }
        let mag = x.abs();

        // Saturate above the largest finite magnitude.
        if mag >= FP8_E4M3_MAX {
            // Round values in the top bin: anything at/over the midpoint between
            // 448 and the next (absent) step saturates; below it rounds to 448.
            return sign * FP8_E4M3_MAX;
        }

        // Decompose: find the unbiased exponent e such that 2^e ≤ mag < 2^{e+1}.
        let mut exp = mag.log2().floor() as i32;

        if exp < Self::MIN_NORMAL_EXP {
            // Subnormal region: fixed step 2^{min_normal_exp} / 8 = 2⁻⁹.
            let step = (Self::MIN_NORMAL_EXP as f32).exp2() / Self::MANT_DIV;
            let q = round_half_even(mag / step);
            let val = q * step;
            // A subnormal can round up into the smallest normal (mag == min_normal).
            return sign * val;
        }

        // Normal region: clamp exponent to the representable maximum.
        if exp > Self::MAX_NORMAL_EXP {
            return sign * FP8_E4M3_MAX;
        }

        // Represent mag = 2^exp · (1 + frac), frac ∈ [0,1). Round the 3-bit
        // mantissa: m = round_even(frac · 8) ∈ {0,…,8}.
        let scale = (exp as f32).exp2();
        let frac = mag / scale - 1.0;
        let mut m = round_half_even(frac * Self::MANT_DIV);
        if m >= Self::MANT_DIV {
            // Mantissa overflowed (rounded up to 1.0 significand): carry into exp.
            m = 0.0;
            exp += 1;
            if exp > Self::MAX_NORMAL_EXP {
                return sign * FP8_E4M3_MAX;
            }
        }
        let new_scale = (exp as f32).exp2();
        let val = new_scale * (1.0 + m / Self::MANT_DIV);
        // Final saturation guard (e.g. exp == max with the +0.875 step).
        if val > FP8_E4M3_MAX {
            return sign * FP8_E4M3_MAX;
        }
        sign * val
    }

    /// Round every element of a tensor to the nearest FP8 E4M3 value.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `values` is empty.
    pub fn quantize(&self, values: &[f32]) -> SnnResult<Vec<f32>> {
        if values.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        Ok(values.iter().map(|&v| self.quantize_scalar(v)).collect())
    }

    /// Largest absolute round-trip error a single quantisation can incur for a
    /// magnitude in `[lo, hi]`, useful for error-bound assertions in callers.
    ///
    /// This equals half the local quantisation step, which for a normal value at
    /// exponent `e` is `2^e · 2^{−mant_bits − 1}`.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::OutOfRange`] if `hi` is non-finite or `hi <= 0`.
    pub fn max_abs_error_in(hi: f32) -> SnnResult<f32> {
        if !hi.is_finite() || hi <= 0.0 {
            return Err(SnnError::OutOfRange {
                name: "hi".into(),
                val: hi,
            });
        }
        let exp = hi.log2().floor() as i32;
        let exp = exp.clamp(Self::MIN_NORMAL_EXP, Self::MAX_NORMAL_EXP);
        let half_step = (exp as f32).exp2() * 2.0_f32.powi(-(Self::MANT_BITS + 1));
        Ok(half_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_scale_matches_definition() {
        let w = vec![-2.0_f32, 0.5, 1.0, 1.5];
        let scale = Int8Quantizer::compute_scale(&w).expect("ok");
        assert!((scale - 2.0 / 127.0).abs() < 1e-9, "scale={scale}");
    }

    #[test]
    fn int8_zero_tensor_uses_unit_scale() {
        let w = vec![0.0_f32; 5];
        let scale = Int8Quantizer::compute_scale(&w).expect("ok");
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn int8_roundtrip_error_bounded_by_half_step() {
        let w = vec![-2.0_f32, -1.3, -0.1, 0.0, 0.4, 1.1, 1.9, 2.0];
        let mut q = Int8Quantizer::new();
        let (_codes, deq) = q.quantize(&w).expect("ok");
        let half_step = q.last_scale() * 0.5;
        for (orig, d) in w.iter().zip(deq.iter()) {
            assert!(
                (orig - d).abs() <= half_step + 1e-6,
                "err={} > {half_step}",
                (orig - d).abs()
            );
        }
    }

    #[test]
    fn int8_codes_within_range() {
        let w: Vec<f32> = (0..256).map(|i| (i as f32) - 128.0).collect();
        let mut q = Int8Quantizer::new();
        let (codes, _deq) = q.quantize(&w).expect("ok");
        for &c in &codes {
            assert!((-127.0..=127.0).contains(&c), "code out of range: {c}");
            assert_eq!(c, c.round(), "code not integral: {c}");
        }
    }

    #[test]
    fn int8_extremes_map_to_qmax() {
        let w = vec![-3.0_f32, 3.0, 0.0];
        let mut q = Int8Quantizer::new();
        let (codes, _deq) = q.quantize(&w).expect("ok");
        assert_eq!(codes[0], -127.0);
        assert_eq!(codes[1], 127.0);
        assert_eq!(codes[2], 0.0);
    }

    #[test]
    fn fake_quantize_weights_matches_quantizer() {
        let w = vec![0.1_f32, -0.4, 0.9, -1.0, 0.55];
        let deq = fake_quantize_weights(&w).expect("ok");
        let mut q = Int8Quantizer::new();
        let (_c, expect) = q.quantize(&w).expect("ok");
        for (a, b) in deq.iter().zip(expect.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn activation_quantizer_levels() {
        let acts = vec![0.0_f32, 0.24, 0.5, 0.76, 1.0, -0.3, 1.4];
        let q = Int8Quantizer::quantize_activations(&acts, 5).expect("ok");
        // Levels: 0, 0.25, 0.5, 0.75, 1.0
        assert!((q[0] - 0.0).abs() < 1e-6);
        assert!((q[1] - 0.25).abs() < 1e-6);
        assert!((q[2] - 0.5).abs() < 1e-6);
        assert!((q[3] - 0.75).abs() < 1e-6);
        assert!((q[4] - 1.0).abs() < 1e-6);
        // Clamped inputs.
        assert!((q[5] - 0.0).abs() < 1e-6);
        assert!((q[6] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn activation_binary_threshold() {
        let acts = vec![0.3_f32, 0.49, 0.5, 0.8];
        let q = Int8Quantizer::quantize_activations(&acts, 2).expect("ok");
        assert_eq!(q[0], 0.0);
        assert_eq!(q[1], 0.0);
        assert_eq!(q[2], 1.0);
        assert_eq!(q[3], 1.0);
    }

    #[test]
    fn activation_rejects_too_few_levels() {
        let acts = vec![0.5_f32];
        assert!(matches!(
            Int8Quantizer::quantize_activations(&acts, 1),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn ste_mask_passes_inside_blocks_outside() {
        // scale = 2/127 ≈ 0.01575. round(w/scale) saturates for |w| > 2.
        let w = vec![0.0_f32, 1.0, 2.0, 2.5, -2.5];
        let scale = Int8Quantizer::compute_scale(&w).expect("ok");
        let mask = ste_backward_mask(&w, scale).expect("ok");
        // The max-magnitude elements (±2.5) define the scale → land exactly at
        // ±127, which is inside the inclusive range → pass.
        for &m in &mask {
            assert!(m == 0.0 || m == 1.0);
        }
        // An element that would exceed 127 after the fact must be blocked: use a
        // smaller scale so 2.5/scale > 127.
        let small_scale = 1.0 / 127.0; // max|w| pretended = 1.0
        let mask2 = ste_backward_mask(&w, small_scale).expect("ok");
        assert_eq!(mask2[0], 1.0); // 0 → pass
        assert_eq!(mask2[1], 1.0); // 1.0 → exactly 127 → pass
        assert_eq!(mask2[2], 0.0); // 2.0 → 254 → block
        assert_eq!(mask2[3], 0.0); // 2.5 → block
        assert_eq!(mask2[4], 0.0); // −2.5 → block
    }

    #[test]
    fn ste_fake_quant_pairs_value_and_mask() {
        let w = vec![0.2_f32, -0.5, 1.0, -1.0];
        let (deq, mask) = ste_fake_quant(&w).expect("ok");
        assert_eq!(deq.len(), w.len());
        assert_eq!(mask.len(), w.len());
        for &m in &mask {
            assert!(m == 0.0 || m == 1.0);
        }
    }

    #[test]
    fn ste_mask_rejects_bad_scale() {
        let w = vec![1.0_f32];
        assert!(matches!(
            ste_backward_mask(&w, 0.0),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            ste_backward_mask(&w, -1.0),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn round_half_even_breaks_ties_to_even() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(-0.5), 0.0);
        assert_eq!(round_half_even(-1.5), -2.0);
        assert_eq!(round_half_even(-2.5), -2.0);
        assert_eq!(round_half_even(2.3), 2.0);
        assert_eq!(round_half_even(2.7), 3.0);
    }

    #[test]
    fn fp8_exact_powers_of_two() {
        let q = Fp8E4M3Quantizer::new();
        for &v in &[1.0_f32, 2.0, 0.5, 4.0, 0.25, 8.0, 0.125] {
            let r = q.quantize_scalar(v);
            assert!((r - v).abs() < 1e-9, "{v} -> {r}");
            let neg = q.quantize_scalar(-v);
            assert!((neg + v).abs() < 1e-9, "-{v} -> {neg}");
        }
    }

    #[test]
    fn fp8_exact_simple_mantissas() {
        let q = Fp8E4M3Quantizer::new();
        // 1.25 = 1·(1 + 2/8), 1.5 = 1·(1 + 4/8), 1.75, 3.0 = 2·(1+4/8).
        for &v in &[1.25_f32, 1.5, 1.75, 1.125, 3.0, 6.0] {
            let r = q.quantize_scalar(v);
            assert!((r - v).abs() < 1e-6, "{v} -> {r}");
        }
    }

    #[test]
    fn fp8_saturates_to_448() {
        let q = Fp8E4M3Quantizer::new();
        assert_eq!(q.quantize_scalar(448.0), 448.0);
        assert_eq!(q.quantize_scalar(500.0), 448.0);
        assert_eq!(q.quantize_scalar(1e6), 448.0);
        assert_eq!(q.quantize_scalar(-1e6), -448.0);
        assert_eq!(q.quantize_scalar(f32::INFINITY), 448.0);
        assert_eq!(q.quantize_scalar(f32::NEG_INFINITY), -448.0);
        assert_eq!(q.quantize_scalar(f32::NAN), 448.0);
    }

    #[test]
    fn fp8_max_is_exactly_representable() {
        // 448 = 2^8 · (1 + 6/8) = 256 · 1.75.
        let q = Fp8E4M3Quantizer::new();
        let r = q.quantize_scalar(448.0);
        assert!((r - 448.0).abs() < 1e-3);
        // 416 = 2^8 · (1 + 5/8) is the step just below; check it round-trips.
        let r2 = q.quantize_scalar(416.0);
        assert!((r2 - 416.0).abs() < 1e-3, "416 -> {r2}");
    }

    #[test]
    fn fp8_subnormals_round_to_step() {
        let q = Fp8E4M3Quantizer::new();
        // Smallest subnormal 2⁻⁹ = 0.001953125 is exact.
        let r = q.quantize_scalar(FP8_E4M3_MIN_SUBNORMAL);
        assert!((r - FP8_E4M3_MIN_SUBNORMAL).abs() < 1e-9, "got {r}");
        // 2·step exact.
        let two = 2.0 * FP8_E4M3_MIN_SUBNORMAL;
        let r2 = q.quantize_scalar(two);
        assert!((r2 - two).abs() < 1e-9, "got {r2}");
        // A value below half the step rounds to zero.
        let tiny = 0.4 * FP8_E4M3_MIN_SUBNORMAL;
        assert_eq!(q.quantize_scalar(tiny), 0.0);
    }

    #[test]
    fn fp8_min_normal_is_representable() {
        let q = Fp8E4M3Quantizer::new();
        let r = q.quantize_scalar(FP8_E4M3_MIN_NORMAL);
        assert!((r - FP8_E4M3_MIN_NORMAL).abs() < 1e-9, "got {r}");
    }

    #[test]
    fn fp8_zero_and_sign() {
        let q = Fp8E4M3Quantizer::new();
        assert_eq!(q.quantize_scalar(0.0), 0.0);
        assert_eq!(q.quantize_scalar(-0.0), 0.0);
    }

    #[test]
    fn fp8_roundtrip_error_within_bound() {
        // For any value in a normal bin, the round-trip error must not exceed
        // half the local step reported by max_abs_error_in.
        let q = Fp8E4M3Quantizer::new();
        let samples = [0.3_f32, 0.7, 1.1, 2.6, 5.5, 13.0, 27.0, 100.0, 300.0];
        for &v in &samples {
            let r = q.quantize_scalar(v);
            let bound = Fp8E4M3Quantizer::max_abs_error_in(v).expect("ok");
            assert!(
                (r - v).abs() <= bound + 1e-4,
                "v={v} r={r} err={} bound={bound}",
                (r - v).abs()
            );
        }
    }

    #[test]
    fn fp8_mantissa_rounds_to_nearest_even() {
        let q = Fp8E4M3Quantizer::new();
        // Between 1.0 (m=0) and 1.125 (m=1): exact midpoint 1.0625 ties to even
        // mantissa (m=0) → 1.0.
        let r = q.quantize_scalar(1.0625);
        assert!((r - 1.0).abs() < 1e-6, "1.0625 -> {r}");
        // Between 1.125 (m=1) and 1.25 (m=2): midpoint 1.1875 ties to even m=2 → 1.25.
        let r2 = q.quantize_scalar(1.1875);
        assert!((r2 - 1.25).abs() < 1e-6, "1.1875 -> {r2}");
    }

    #[test]
    fn fp8_quantize_batch() {
        let q = Fp8E4M3Quantizer::new();
        let v = vec![1.0_f32, 2.0, 0.5, 448.0, -100.0];
        let out = q.quantize(&v).expect("ok");
        assert_eq!(out.len(), v.len());
        assert_eq!(out[0], 1.0);
        assert_eq!(out[3], 448.0);
    }

    #[test]
    fn fp8_rejects_empty() {
        let q = Fp8E4M3Quantizer::new();
        assert!(matches!(q.quantize(&[]), Err(SnnError::EmptyInput)));
    }

    #[test]
    fn fp8_error_bound_rejects_bad_hi() {
        assert!(matches!(
            Fp8E4M3Quantizer::max_abs_error_in(0.0),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn int8_rejects_empty() {
        assert!(matches!(
            Int8Quantizer::compute_scale(&[]),
            Err(SnnError::EmptyInput)
        ));
        assert!(matches!(
            fake_quantize_weights(&[]),
            Err(SnnError::EmptyInput)
        ));
    }
}

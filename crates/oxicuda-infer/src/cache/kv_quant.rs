//! # Quantized KV-Cache Integration Logic
//!
//! Per-token affine integer quantization of the key/value vectors stored in the
//! paged KV cache. KV-cache quantization (e.g. KIVI, Liu et al. 2024; the FP8/
//! INT8 KV caches in vLLM and TensorRT-LLM) roughly halves or quarters cache
//! memory — the dominant cost for long-context serving — at a small, usually
//! negligible, accuracy hit.
//!
//! This module provides the *math and bookkeeping* of the scheme in pure Rust:
//! the actual reduced-precision storage is the caller's concern (downstream GPU
//! kernels keep the INT codes), but the quantize / dequantize transforms, the
//! per-token scale/zero-point derivation, and the round-trip error accounting
//! all live here and are exhaustively tested on the CPU.
//!
//! ## Scheme
//!
//! For each token's K (or V) vector `x` of length `d` we compute a **per-token,
//! symmetric or asymmetric** affine map:
//!
//! ```text
//!   asymmetric:  q = round((x - zp) / scale),  x ≈ scale·q + zp
//!   symmetric:   q = round(x / scale),          x ≈ scale·q       (zp = 0)
//! ```
//!
//! with `scale` and `zp` chosen from the token's own min/max so the full integer
//! range `[qmin, qmax]` is used. Per-token granularity (rather than per-tensor)
//! is what keeps KV quantization accurate, because the dynamic range of K/V
//! varies sharply along the sequence.

use crate::error::{InferError, InferResult};

// ─── KvQuantConfig ───────────────────────────────────────────────────────────

/// Configuration for KV-cache quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvQuantConfig {
    /// Bits per quantized element. One of `{4, 8}`.
    pub bits: u32,
    /// Symmetric (`zp = 0`) vs. asymmetric (zero-point from the token minimum).
    pub symmetric: bool,
}

impl KvQuantConfig {
    /// Construct a validated config.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if `bits` is not 4 or 8.
    pub fn new(bits: u32, symmetric: bool) -> InferResult<Self> {
        if bits != 4 && bits != 8 {
            return Err(InferError::InvalidConfig("KV quant bits must be 4 or 8"));
        }
        Ok(Self { bits, symmetric })
    }

    /// 8-bit asymmetric default (the common, robust choice).
    #[must_use]
    pub fn int8() -> Self {
        Self {
            bits: 8,
            symmetric: false,
        }
    }

    /// Inclusive integer range `[qmin, qmax]` for this configuration.
    ///
    /// Symmetric uses a signed range centred on zero; asymmetric uses an
    /// unsigned range `[0, 2^bits − 1]`.
    #[must_use]
    pub fn q_range(&self) -> (i32, i32) {
        if self.symmetric {
            let half = 1_i32 << (self.bits - 1);
            (-(half - 1), half - 1) // e.g. 8-bit → [-127, 127]
        } else {
            (0, (1_i32 << self.bits) - 1) // e.g. 8-bit → [0, 255]
        }
    }
}

// ─── QuantizedToken ──────────────────────────────────────────────────────────

/// One quantized K (or V) vector: integer codes plus its affine parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedToken {
    /// Integer codes, one per channel.
    pub codes: Vec<i32>,
    /// Per-token scale (`> 0`).
    pub scale: f32,
    /// Per-token zero-point (`0` for symmetric).
    pub zero_point: f32,
}

impl QuantizedToken {
    /// Dequantize back to floating point: `scale·code + zero_point`.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        self.codes
            .iter()
            .map(|&q| self.scale * q as f32 + self.zero_point)
            .collect()
    }
}

// ─── Quantization ────────────────────────────────────────────────────────────

/// Quantize a single token vector `x` under `config`.
///
/// Derives the per-token scale/zero-point from the data's own range so the full
/// integer span is used, then maps and clamps each channel.
///
/// # Errors
/// * [`InferError::InvalidConfig`] if `x` is empty.
pub fn quantize_token(x: &[f32], config: KvQuantConfig) -> InferResult<QuantizedToken> {
    if x.is_empty() {
        return Err(InferError::InvalidConfig("quantize_token: empty vector"));
    }
    let (qmin, qmax) = config.q_range();
    let qmin_f = qmin as f32;
    let qmax_f = qmax as f32;

    let x_min = x.iter().copied().fold(f32::INFINITY, f32::min);
    let x_max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let (scale, zero_point) = if config.symmetric {
        // Symmetric: scale from the max absolute value.
        let amax = x_min.abs().max(x_max.abs());
        let scale = if amax > 0.0 { amax / qmax_f } else { 1.0 };
        (scale, 0.0_f32)
    } else {
        // Asymmetric: span the [x_min, x_max] interval over [qmin, qmax].
        let range = x_max - x_min;
        let scale = if range > 0.0 {
            range / (qmax_f - qmin_f)
        } else {
            1.0
        };
        // zero_point is the float value mapped to the integer qmin.
        let zero_point = x_min - qmin_f * scale;
        (scale, zero_point)
    };

    let codes: Vec<i32> = x
        .iter()
        .map(|&v| {
            let q = ((v - zero_point) / scale).round() as i32;
            q.clamp(qmin, qmax)
        })
        .collect();

    Ok(QuantizedToken {
        codes,
        scale,
        zero_point,
    })
}

/// Quantize then immediately dequantize, returning the reconstructed vector.
/// Useful for simulating the precision loss a quantized cache would incur.
///
/// # Errors
/// * Propagates [`quantize_token`] errors.
pub fn quantize_dequantize_token(x: &[f32], config: KvQuantConfig) -> InferResult<Vec<f32>> {
    Ok(quantize_token(x, config)?.dequantize())
}

/// Mean-squared reconstruction error of quantizing `x` under `config`.
///
/// # Errors
/// * Propagates [`quantize_token`] errors.
pub fn quantization_mse(x: &[f32], config: KvQuantConfig) -> InferResult<f64> {
    let recon = quantize_dequantize_token(x, config)?;
    let n = x.len() as f64;
    let sse: f64 = x
        .iter()
        .zip(recon.iter())
        .map(|(&a, &b)| {
            let d = (a - b) as f64;
            d * d
        })
        .sum();
    Ok(sse / n)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_bad_bits() {
        assert!(KvQuantConfig::new(3, false).is_err());
        assert!(KvQuantConfig::new(8, false).is_ok());
        assert!(KvQuantConfig::new(4, true).is_ok());
    }

    #[test]
    fn q_range_signed_and_unsigned() {
        let asym = KvQuantConfig::new(8, false).expect("ok");
        assert_eq!(asym.q_range(), (0, 255));
        let sym = KvQuantConfig::new(8, true).expect("ok");
        assert_eq!(sym.q_range(), (-127, 127));
        let asym4 = KvQuantConfig::new(4, false).expect("ok");
        assert_eq!(asym4.q_range(), (0, 15));
    }

    #[test]
    fn empty_vector_rejected() {
        assert!(quantize_token(&[], KvQuantConfig::int8()).is_err());
    }

    #[test]
    fn asymmetric_endpoints_exact() {
        // The extreme values map to the integer endpoints and reconstruct exactly.
        let x = vec![-1.0_f32, 0.0, 1.0, 2.0];
        let q = quantize_token(&x, KvQuantConfig::int8()).expect("ok");
        assert_eq!(q.zero_point as f64, -1.0, "min maps to qmin=0");
        // min and max should round-trip to within one quantization step.
        let r = q.dequantize();
        assert!((r[0] - (-1.0)).abs() < 1e-4, "min reconstructs: {}", r[0]);
        assert!((r[3] - 2.0).abs() < 1e-4, "max reconstructs: {}", r[3]);
    }

    #[test]
    fn symmetric_zero_point_is_zero() {
        let x = vec![-2.0_f32, -1.0, 1.0, 2.0];
        let cfg = KvQuantConfig::new(8, true).expect("ok");
        let q = quantize_token(&x, cfg).expect("ok");
        assert_eq!(q.zero_point, 0.0);
        // Symmetric int8 should reconstruct a symmetric signal very accurately.
        let mse = quantization_mse(&x, cfg).expect("ok");
        assert!(mse < 1e-3, "int8 symmetric mse too large: {mse}");
    }

    #[test]
    fn int8_more_accurate_than_int4() {
        // A vector whose values do NOT land on the coarse int4 grid: 8-bit must
        // beat 4-bit on reconstruction error. (Avoid arithmetic progressions,
        // which a matching number of levels can reproduce nearly exactly.)
        let x: Vec<f32> = (0..40)
            .map(|i| {
                let t = i as f32;
                (t * 0.613).sin() * 2.7 + (t * 0.27).cos() * 1.3
            })
            .collect();
        let mse8 = quantization_mse(&x, KvQuantConfig::new(8, false).expect("ok")).expect("ok");
        let mse4 = quantization_mse(&x, KvQuantConfig::new(4, false).expect("ok")).expect("ok");
        assert!(mse8 < mse4, "int8 mse {mse8} should be < int4 mse {mse4}");
        assert!(mse4 > 0.0, "non-grid-aligned int4 must lose precision");
    }

    #[test]
    fn constant_vector_zero_error() {
        // All-equal input → degenerate range; must not divide by zero and must
        // reconstruct exactly.
        let x = vec![3.5_f32; 8];
        let q = quantize_token(&x, KvQuantConfig::int8()).expect("ok");
        let r = q.dequantize();
        for &v in &r {
            assert!((v - 3.5).abs() < 1e-5, "constant reconstructs exactly: {v}");
        }
    }

    #[test]
    fn codes_within_range() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32).sin() * 5.0).collect();
        let cfg = KvQuantConfig::int8();
        let (qmin, qmax) = cfg.q_range();
        let q = quantize_token(&x, cfg).expect("ok");
        assert!(q.codes.iter().all(|&c| c >= qmin && c <= qmax));
        assert_eq!(q.codes.len(), x.len());
    }

    #[test]
    fn dequantize_matches_round_trip_helper() {
        let x = vec![0.1_f32, -0.4, 2.2, -1.7, 0.0];
        let cfg = KvQuantConfig::int8();
        let a = quantize_token(&x, cfg).expect("ok").dequantize();
        let b = quantize_dequantize_token(&x, cfg).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn mse_decreases_with_precision_on_smooth_signal() {
        let x: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).cos()).collect();
        let mse4 = quantization_mse(&x, KvQuantConfig::new(4, false).expect("ok")).expect("ok");
        let mse8 = quantization_mse(&x, KvQuantConfig::new(8, false).expect("ok")).expect("ok");
        assert!(mse8 <= mse4);
        // 8-bit on a [-1,1] signal should be quite tight.
        assert!(mse8 < 1e-3, "mse8 = {mse8}");
    }
}

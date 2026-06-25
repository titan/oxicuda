//! # KV-Cache Quantization
//!
//! Quantizes the key/value caches that dominate memory during autoregressive
//! LLM inference. For a sequence of `n_tokens` and head dimension `head_dim`,
//! the K and V tensors are `(n_tokens × head_dim)` per attention head; storing
//! them in INT8/INT4 instead of FP16 cuts cache memory 2–4×.
//!
//! ## Axis choice (KIVI-style)
//!
//! Liu et al. (2024), "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV
//! Cache" <https://arxiv.org/abs/2402.02750>, observes:
//!
//! * **Keys** have strong *per-channel* outliers (a few head-dimension columns
//!   carry large magnitudes, largely from rotary position embeddings). Keys are
//!   therefore quantized **per-channel** (one scale/zero-point per column).
//! * **Values** have no such channel structure and are quantized **per-token**
//!   (one scale/zero-point per row), which is also friendlier to the streaming
//!   append pattern of decoding.
//!
//! Both use *asymmetric affine* quantization
//! `q = round((x − zp_real)/scale)`, `x ≈ q·scale + zp_real`, where the
//! integer codes live in `[0, 2^bits − 1]` and the real-valued offset
//! `zp_real = min` is recorded per group.
//!
//! All math is performed on flat `&[f32]` slices and is fully CPU-testable.

use crate::error::{QuantError, QuantResult};

// ─── Axis ──────────────────────────────────────────────────────────────────────

/// Quantization grouping axis for a 2-D `(n_tokens × head_dim)` tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvAxis {
    /// One `(scale, zero-point)` per token (row). Recommended for **values**.
    PerToken,
    /// One `(scale, zero-point)` per channel (column). Recommended for **keys**.
    PerChannel,
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// KV-cache quantization configuration.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Bit-width (2, 4, or 8 are typical). Must be in `[2, 8]`.
    pub bits: u32,
    /// Grouping axis.
    pub axis: KvAxis,
}

impl KvCacheConfig {
    /// Create a configuration, validating the bit-width.
    ///
    /// # Errors
    ///
    /// * [`QuantError::InvalidBitWidth`] — `bits` outside `[2, 8]`.
    pub fn new(bits: u32, axis: KvAxis) -> QuantResult<Self> {
        if !(2..=8).contains(&bits) {
            return Err(QuantError::InvalidBitWidth { bits });
        }
        Ok(Self { bits, axis })
    }

    /// INT8 per-channel key cache (KIVI default for keys).
    #[must_use]
    pub fn keys_int8() -> Self {
        Self {
            bits: 8,
            axis: KvAxis::PerChannel,
        }
    }

    /// INT8 per-token value cache (KIVI default for values).
    #[must_use]
    pub fn values_int8() -> Self {
        Self {
            bits: 8,
            axis: KvAxis::PerToken,
        }
    }
}

// ─── Quantized cache ───────────────────────────────────────────────────────────

/// A quantized KV-cache tensor: integer codes plus per-group affine params.
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    /// Integer codes in `[0, 2^bits − 1]`, row-major `(n_tokens × head_dim)`.
    pub codes: Vec<u8>,
    /// Per-group scale factors (length `n_groups`).
    pub scales: Vec<f32>,
    /// Per-group real-valued offsets (`min`; length `n_groups`).
    pub zero_points: Vec<f32>,
    /// Number of token rows.
    pub n_tokens: usize,
    /// Head dimension (column count).
    pub head_dim: usize,
    /// Bit-width used.
    pub bits: u32,
    /// Grouping axis.
    pub axis: KvAxis,
}

impl QuantizedKvCache {
    /// Number of distinct quantization groups.
    #[must_use]
    pub fn n_groups(&self) -> usize {
        match self.axis {
            KvAxis::PerToken => self.n_tokens,
            KvAxis::PerChannel => self.head_dim,
        }
    }

    /// Total bytes of integer payload (codes are packed to `bits` per element).
    #[must_use]
    pub fn packed_bytes(&self) -> usize {
        let total_bits = self.codes.len() * self.bits as usize;
        total_bits.div_ceil(8)
    }
}

// ─── Quantizer ─────────────────────────────────────────────────────────────────

/// KV-cache quantizer.
#[derive(Debug, Clone)]
pub struct KvCacheQuantizer {
    config: KvCacheConfig,
}

impl KvCacheQuantizer {
    /// Create a new quantizer.
    #[must_use]
    pub fn new(config: KvCacheConfig) -> Self {
        Self { config }
    }

    /// Quantize a `(n_tokens × head_dim)` row-major K or V tensor.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`] — empty input or zero dimension.
    /// * [`QuantError::DimensionMismatch`] — length ≠ `n_tokens·head_dim`.
    pub fn quantize(
        &self,
        tensor: &[f32],
        n_tokens: usize,
        head_dim: usize,
    ) -> QuantResult<QuantizedKvCache> {
        if tensor.is_empty() || n_tokens == 0 || head_dim == 0 {
            return Err(QuantError::EmptyInput("KvCacheQuantizer::quantize"));
        }
        if tensor.len() != n_tokens * head_dim {
            return Err(QuantError::DimensionMismatch {
                expected: n_tokens * head_dim,
                got: tensor.len(),
            });
        }
        let bits = self.config.bits;
        let q_max = ((1u32 << bits) - 1) as f32;

        let n_groups = match self.config.axis {
            KvAxis::PerToken => n_tokens,
            KvAxis::PerChannel => head_dim,
        };

        // ── Compute per-group (min, max) ─────────────────────────────────────
        let mut gmin = vec![f32::INFINITY; n_groups];
        let mut gmax = vec![f32::NEG_INFINITY; n_groups];
        for t in 0..n_tokens {
            for c in 0..head_dim {
                let v = tensor[t * head_dim + c];
                let g = match self.config.axis {
                    KvAxis::PerToken => t,
                    KvAxis::PerChannel => c,
                };
                if v < gmin[g] {
                    gmin[g] = v;
                }
                if v > gmax[g] {
                    gmax[g] = v;
                }
            }
        }

        // ── Derive affine params ─────────────────────────────────────────────
        let mut scales = vec![0.0_f32; n_groups];
        let mut zero_points = vec![0.0_f32; n_groups];
        for g in 0..n_groups {
            let range = (gmax[g] - gmin[g]).max(1e-8);
            scales[g] = range / q_max;
            zero_points[g] = gmin[g];
        }

        // ── Quantize ─────────────────────────────────────────────────────────
        let mut codes = vec![0_u8; n_tokens * head_dim];
        for t in 0..n_tokens {
            for c in 0..head_dim {
                let g = match self.config.axis {
                    KvAxis::PerToken => t,
                    KvAxis::PerChannel => c,
                };
                let v = tensor[t * head_dim + c];
                let q = ((v - zero_points[g]) / scales[g]).round().clamp(0.0, q_max);
                codes[t * head_dim + c] = q as u8;
            }
        }

        Ok(QuantizedKvCache {
            codes,
            scales,
            zero_points,
            n_tokens,
            head_dim,
            bits,
            axis: self.config.axis,
        })
    }

    /// Dequantize a previously quantized KV-cache back to `f32`.
    #[must_use]
    pub fn dequantize(cache: &QuantizedKvCache) -> Vec<f32> {
        let mut out = vec![0.0_f32; cache.n_tokens * cache.head_dim];
        for t in 0..cache.n_tokens {
            for c in 0..cache.head_dim {
                let g = match cache.axis {
                    KvAxis::PerToken => t,
                    KvAxis::PerChannel => c,
                };
                let q = cache.codes[t * cache.head_dim + c] as f32;
                out[t * cache.head_dim + c] = q * cache.scales[g] + cache.zero_points[g];
            }
        }
        out
    }

    /// Append a single new token's row to an existing per-token cache.
    ///
    /// This mirrors the decoding-time streaming pattern: the new token gets its
    /// own quantization group without re-quantizing the whole cache. Only valid
    /// for [`KvAxis::PerToken`] caches (per-channel scales would otherwise need
    /// a global recompute).
    ///
    /// # Errors
    ///
    /// * [`QuantError::InvalidConfig`] — cache axis is not `PerToken`.
    /// * [`QuantError::DimensionMismatch`] — `row.len() != head_dim`.
    pub fn append_token(&self, cache: &mut QuantizedKvCache, row: &[f32]) -> QuantResult<()> {
        if cache.axis != KvAxis::PerToken {
            return Err(QuantError::InvalidConfig(
                "append_token requires a PerToken cache".to_string(),
            ));
        }
        if row.len() != cache.head_dim {
            return Err(QuantError::DimensionMismatch {
                expected: cache.head_dim,
                got: row.len(),
            });
        }
        let q_max = ((1u32 << cache.bits) - 1) as f32;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in row {
            min = min.min(v);
            max = max.max(v);
        }
        let scale = (max - min).max(1e-8) / q_max;
        for &v in row {
            let q = ((v - min) / scale).round().clamp(0.0, q_max);
            cache.codes.push(q as u8);
        }
        cache.scales.push(scale);
        cache.zero_points.push(min);
        cache.n_tokens += 1;
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// LCG pseudo-random f32 in `[-scale, scale]`.
    fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
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

    // ── Config ─────────────────────────────────────────────────────────────────

    #[test]
    fn config_validates_bits() {
        assert!(KvCacheConfig::new(8, KvAxis::PerToken).is_ok());
        assert!(KvCacheConfig::new(4, KvAxis::PerChannel).is_ok());
        assert!(matches!(
            KvCacheConfig::new(1, KvAxis::PerToken),
            Err(QuantError::InvalidBitWidth { .. })
        ));
        assert!(matches!(
            KvCacheConfig::new(16, KvAxis::PerToken),
            Err(QuantError::InvalidBitWidth { .. })
        ));
    }

    #[test]
    fn preset_axes_correct() {
        assert_eq!(KvCacheConfig::keys_int8().axis, KvAxis::PerChannel);
        assert_eq!(KvCacheConfig::values_int8().axis, KvAxis::PerToken);
    }

    // ── Round-trip accuracy ────────────────────────────────────────────────────

    #[test]
    fn int8_per_token_round_trip() {
        let n_tokens = 12;
        let head_dim = 16;
        let v = lcg(n_tokens * head_dim, 1, 3.0);
        let q = KvCacheQuantizer::new(KvCacheConfig::values_int8());
        let cache = q.quantize(&v, n_tokens, head_dim).expect("quantize");
        assert_eq!(cache.n_groups(), n_tokens);
        let deq = KvCacheQuantizer::dequantize(&cache);
        let rel = rel_err(&v, &deq);
        assert!(rel < 0.01, "INT8 per-token rel err {rel} too high");
    }

    #[test]
    fn int8_per_channel_round_trip() {
        let n_tokens = 16;
        let head_dim = 8;
        let k = lcg(n_tokens * head_dim, 2, 3.0);
        let q = KvCacheQuantizer::new(KvCacheConfig::keys_int8());
        let cache = q.quantize(&k, n_tokens, head_dim).expect("quantize");
        assert_eq!(cache.n_groups(), head_dim);
        let deq = KvCacheQuantizer::dequantize(&cache);
        let rel = rel_err(&k, &deq);
        assert!(rel < 0.01, "INT8 per-channel rel err {rel} too high");
    }

    #[test]
    fn int4_round_trip_coarser_but_bounded() {
        let n_tokens = 10;
        let head_dim = 12;
        let v = lcg(n_tokens * head_dim, 3, 2.0);
        let cfg = KvCacheConfig::new(4, KvAxis::PerToken).expect("cfg");
        let q = KvCacheQuantizer::new(cfg);
        let cache = q.quantize(&v, n_tokens, head_dim).expect("quantize");
        let deq = KvCacheQuantizer::dequantize(&cache);
        let rel = rel_err(&v, &deq);
        assert!(rel < 0.1, "INT4 per-token rel err {rel} too high");
    }

    // ── Axis choice matters on channel outliers ────────────────────────────────

    #[test]
    fn per_channel_beats_per_token_on_key_outliers() {
        // Simulate RoPE-like per-channel key outliers: column 3 is huge.
        let n_tokens = 20;
        let head_dim = 8;
        let mut k = lcg(n_tokens * head_dim, 4, 1.0);
        for t in 0..n_tokens {
            k[t * head_dim + 3] = 50.0 + (t as f32) * 0.1; // outlier channel
        }

        let per_chan = KvCacheQuantizer::new(KvCacheConfig::keys_int8());
        let chan_cache = per_chan.quantize(&k, n_tokens, head_dim).expect("q");
        let chan_deq = KvCacheQuantizer::dequantize(&chan_cache);
        let err_chan = rel_err(&k, &chan_deq);

        let per_tok = KvCacheQuantizer::new(KvCacheConfig::values_int8());
        let tok_cache = per_tok.quantize(&k, n_tokens, head_dim).expect("q");
        let tok_deq = KvCacheQuantizer::dequantize(&tok_cache);
        let err_tok = rel_err(&k, &tok_deq);

        assert!(
            err_chan < err_tok,
            "per-channel {err_chan} should beat per-token {err_tok} on channel outliers"
        );
    }

    // ── Streaming append ───────────────────────────────────────────────────────

    #[test]
    fn append_token_grows_cache() {
        let n_tokens = 4;
        let head_dim = 6;
        let v = lcg(n_tokens * head_dim, 5, 2.0);
        let q = KvCacheQuantizer::new(KvCacheConfig::values_int8());
        let mut cache = q.quantize(&v, n_tokens, head_dim).expect("quantize");

        let new_row = lcg(head_dim, 99, 2.0);
        q.append_token(&mut cache, &new_row).expect("append");
        assert_eq!(cache.n_tokens, n_tokens + 1);
        assert_eq!(cache.scales.len(), n_tokens + 1);
        assert_eq!(cache.codes.len(), (n_tokens + 1) * head_dim);

        // The appended row must dequantize back close to the original.
        let deq = KvCacheQuantizer::dequantize(&cache);
        let last = &deq[n_tokens * head_dim..(n_tokens + 1) * head_dim];
        assert!(rel_err(&new_row, last) < 0.01);
    }

    #[test]
    fn append_token_rejects_per_channel() {
        let k = lcg(4 * 6, 6, 2.0);
        let q = KvCacheQuantizer::new(KvCacheConfig::keys_int8());
        let mut cache = q.quantize(&k, 4, 6).expect("quantize");
        let row = lcg(6, 7, 2.0);
        assert!(matches!(
            q.append_token(&mut cache, &row),
            Err(QuantError::InvalidConfig(_))
        ));
    }

    #[test]
    fn append_token_dimension_error() {
        let v = lcg(4 * 6, 8, 2.0);
        let q = KvCacheQuantizer::new(KvCacheConfig::values_int8());
        let mut cache = q.quantize(&v, 4, 6).expect("quantize");
        let bad_row = vec![1.0_f32; 5];
        assert!(matches!(
            q.append_token(&mut cache, &bad_row),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    // ── Memory accounting ──────────────────────────────────────────────────────

    #[test]
    fn packed_bytes_reflects_bit_width() {
        let v = lcg(4 * 8, 9, 1.0);
        let cfg4 = KvCacheConfig::new(4, KvAxis::PerToken).expect("cfg");
        let cache4 = KvCacheQuantizer::new(cfg4).quantize(&v, 4, 8).expect("q");
        // 32 elements × 4 bits = 128 bits = 16 bytes.
        assert_eq!(cache4.packed_bytes(), 16);

        let cfg8 = KvCacheConfig::new(8, KvAxis::PerToken).expect("cfg");
        let cache8 = KvCacheQuantizer::new(cfg8).quantize(&v, 4, 8).expect("q");
        assert_eq!(cache8.packed_bytes(), 32);
    }

    // ── Error handling ─────────────────────────────────────────────────────────

    #[test]
    fn quantize_empty_errors() {
        let q = KvCacheQuantizer::new(KvCacheConfig::values_int8());
        assert!(matches!(
            q.quantize(&[], 0, 0),
            Err(QuantError::EmptyInput(_))
        ));
    }

    #[test]
    fn quantize_dimension_error() {
        let q = KvCacheQuantizer::new(KvCacheConfig::values_int8());
        let v = vec![1.0_f32; 10];
        assert!(matches!(
            q.quantize(&v, 3, 4),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }
}

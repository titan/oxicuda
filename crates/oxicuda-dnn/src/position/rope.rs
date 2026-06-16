//! Rotary Position Embedding (RoPE) — Su et al. 2021, CPU reference.
//!
//! "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al.,
//! 2021) encodes absolute position by *rotating* consecutive feature pairs of
//! the query/key vectors by an angle proportional to the token position. Pair
//! `i ∈ [0, d_head/2)` uses inverse frequency
//!
//! ```text
//! freq_i = 1 / base^(2i / d_head)
//! ```
//!
//! and at position `p` the rotation by `θ = p · freq_i` maps
//!
//! ```text
//! (x_{2i}, x_{2i+1}) ↦ (x_{2i}·cosθ − x_{2i+1}·sinθ,
//!                       x_{2i}·sinθ + x_{2i+1}·cosθ).
//! ```
//!
//! The dot product of two rotated vectors depends only on their *relative*
//! offset, which is what gives RoPE its relative-position and length
//! extrapolation properties. Each rotation is orthogonal, so it preserves the
//! per-pair (and hence the whole-vector) Euclidean norm.
//!
//! This module pre-computes the `cos`/`sin` tables once in
//! [`Rope::new`] and applies them in [`Rope::apply`]. It is a pure-CPU `f32`
//! reference complementing the device kernel in [`crate::attn::rope`].

use crate::error::{DnnError, DnnResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`Rope`].
#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    /// Per-head feature dimension. Must be even (rotation acts on pairs).
    pub d_head: usize,
    /// Maximum sequence length the cos/sin caches cover.
    pub max_seq_len: usize,
    /// Frequency base (typically `10000.0`).
    pub base: f32,
}

// ─── Rope ────────────────────────────────────────────────────────────────────

/// Rotary position embedding with cached rotation tables.
#[derive(Debug, Clone)]
pub struct Rope {
    /// Cosine cache, flat `[max_seq_len × (d_head / 2)]`.
    cos_cache: Vec<f32>,
    /// Sine cache, flat `[max_seq_len × (d_head / 2)]`.
    sin_cache: Vec<f32>,
    /// Validated configuration.
    config: RopeConfig,
}

impl Rope {
    /// Build the cos/sin caches.
    ///
    /// # Errors
    /// - [`DnnError::InvalidArgument`] if `d_head` is odd or zero, if
    ///   `max_seq_len == 0`, or if `base <= 1` / non-finite.
    pub fn new(config: RopeConfig) -> DnnResult<Self> {
        if config.d_head == 0 || config.d_head % 2 != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "RoPE d_head must be even and > 0, got {}",
                config.d_head
            )));
        }
        if config.max_seq_len == 0 {
            return Err(DnnError::InvalidArgument(
                "RoPE max_seq_len must be > 0".into(),
            ));
        }
        if !config.base.is_finite() || config.base <= 1.0 {
            return Err(DnnError::InvalidArgument(format!(
                "RoPE base must be finite and > 1, got {}",
                config.base
            )));
        }

        let half = config.d_head / 2;
        let mut cos_cache = vec![0.0_f32; config.max_seq_len * half];
        let mut sin_cache = vec![0.0_f32; config.max_seq_len * half];

        // freq_i = base^(-2i / d_head)
        let inv_freqs: Vec<f32> = (0..half)
            .map(|i| {
                let exponent = (2 * i) as f32 / config.d_head as f32;
                config.base.powf(-exponent)
            })
            .collect();

        for pos in 0..config.max_seq_len {
            let row = pos * half;
            for (i, &freq) in inv_freqs.iter().enumerate() {
                let angle = pos as f32 * freq;
                cos_cache[row + i] = angle.cos();
                sin_cache[row + i] = angle.sin();
            }
        }

        Ok(Self {
            cos_cache,
            sin_cache,
            config,
        })
    }

    /// Per-head feature dimension.
    #[must_use]
    #[inline]
    pub fn d_head(&self) -> usize {
        self.config.d_head
    }

    /// Maximum cached sequence length.
    #[must_use]
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    /// Apply RoPE to `x` of shape `[seq_len × n_heads × d_head]`.
    ///
    /// Every head at every position is rotated by that position's cached angle.
    /// Returns a new tensor of identical shape.
    ///
    /// # Errors
    /// - [`DnnError::InvalidArgument`] if `seq_len == 0`, `n_heads == 0`, or
    ///   `seq_len > max_seq_len`.
    /// - [`DnnError::InvalidDimension`] if `x.len() != seq_len · n_heads ·
    ///   d_head`.
    pub fn apply(&self, x: &[f32], seq_len: usize, n_heads: usize) -> DnnResult<Vec<f32>> {
        if seq_len == 0 || n_heads == 0 {
            return Err(DnnError::InvalidArgument(format!(
                "RoPE apply: seq_len and n_heads must be > 0, got {seq_len} and {n_heads}"
            )));
        }
        if seq_len > self.config.max_seq_len {
            return Err(DnnError::InvalidArgument(format!(
                "RoPE apply: seq_len {seq_len} exceeds max_seq_len {}",
                self.config.max_seq_len
            )));
        }
        let d_head = self.config.d_head;
        let expected = seq_len * n_heads * d_head;
        if x.len() != expected {
            return Err(DnnError::InvalidDimension(format!(
                "RoPE apply: expected {expected} elements, got {}",
                x.len()
            )));
        }

        let half = d_head / 2;
        let mut out = x.to_vec();

        for pos in 0..seq_len {
            let cache_row = pos * half;
            for head in 0..n_heads {
                let base = (pos * n_heads + head) * d_head;
                for i in 0..half {
                    let cos = self.cos_cache[cache_row + i];
                    let sin = self.sin_cache[cache_row + i];
                    let lo = base + 2 * i;
                    let hi = lo + 1;
                    let a = x[lo];
                    let b = x[hi];
                    out[lo] = a * cos - b * sin;
                    out[hi] = a * sin + b * cos;
                }
            }
        }

        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::DnnRng;

    fn rope(d_head: usize, max_seq_len: usize) -> Rope {
        Rope::new(RopeConfig {
            d_head,
            max_seq_len,
            base: 10000.0,
        })
        .expect("valid config")
    }

    #[test]
    fn apply_shape() {
        let r = rope(8, 16);
        let seq_len = 4;
        let n_heads = 2;
        let x = vec![0.5_f32; seq_len * n_heads * 8];
        let out = r.apply(&x, seq_len, n_heads).expect("ok");
        assert_eq!(out.len(), seq_len * n_heads * 8);
    }

    #[test]
    fn apply_finite() {
        let r = rope(8, 16);
        let mut rng = DnnRng::new(1);
        let seq_len = 5;
        let n_heads = 3;
        let mut x = vec![0.0_f32; seq_len * n_heads * 8];
        rng.fill_normal(&mut x);
        let out = r.apply(&x, seq_len, n_heads).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn position_0_identity() {
        // At position 0, all angles are 0 ⇒ cos = 1, sin = 0 ⇒ x unchanged.
        let r = rope(8, 16);
        let mut rng = DnnRng::new(2);
        let n_heads = 2;
        let mut x = vec![0.0_f32; n_heads * 8];
        rng.fill_normal(&mut x);
        let out = r.apply(&x, 1, n_heads).expect("ok");
        for (a, b) in x.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "position 0 must be identity");
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        // Each pair rotation is orthogonal ⇒ the per-token norm is preserved.
        let r = rope(8, 16);
        let mut rng = DnnRng::new(3);
        let seq_len = 4;
        let n_heads = 1;
        let mut x = vec![0.0_f32; seq_len * n_heads * 8];
        rng.fill_normal(&mut x);
        let out = r.apply(&x, seq_len, n_heads).expect("ok");
        for t in 0..seq_len {
            let xs = &x[t * 8..(t + 1) * 8];
            let os = &out[t * 8..(t + 1) * 8];
            let nx: f32 = xs.iter().map(|v| v * v).sum::<f32>().sqrt();
            let no: f32 = os.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((nx - no).abs() < 1e-4, "norm changed: {nx} vs {no}");
        }
    }

    #[test]
    fn different_positions_different_rotation() {
        // A non-trivial vector rotated at position 1 vs position 2 must differ.
        let r = rope(8, 16);
        let n_heads = 1;
        let seq_len = 3;
        // Same per-token content so any difference is purely positional.
        let mut x = vec![0.0_f32; seq_len * n_heads * 8];
        for t in 0..seq_len {
            for d in 0..8 {
                x[t * 8 + d] = (d as f32) * 0.1 + 1.0;
            }
        }
        let out = r.apply(&x, seq_len, n_heads).expect("ok");
        let row1 = &out[8..16];
        let row2 = &out[16..24];
        let diff: f32 = row1
            .iter()
            .zip(row2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-4, "rows at different positions must differ");
    }

    #[test]
    fn d_head_odd_error() {
        let r = Rope::new(RopeConfig {
            d_head: 7,
            max_seq_len: 16,
            base: 10000.0,
        });
        assert!(matches!(r, Err(DnnError::InvalidArgument(_))));
    }

    #[test]
    fn seq_gt_max_error() {
        let r = rope(8, 4);
        let x = vec![0.0_f32; 5 * 8];
        let out = r.apply(&x, 5, 1);
        assert!(matches!(out, Err(DnnError::InvalidArgument(_))));
    }

    #[test]
    fn n_heads_invariant() {
        // The rotation depends only on (position, dim), not on which head; the
        // same per-head content at the same position must rotate identically.
        let r = rope(8, 16);
        let mut rng = DnnRng::new(4);
        let mut head = vec![0.0_f32; 8];
        rng.fill_normal(&mut head);
        // Build a [seq=2 × heads=2 × 8] tensor where both heads share content.
        let mut x = vec![0.0_f32; 2 * 2 * 8];
        for pos in 0..2 {
            for h in 0..2 {
                let base = (pos * 2 + h) * 8;
                x[base..base + 8].copy_from_slice(&head);
            }
        }
        let out = r.apply(&x, 2, 2).expect("ok");
        // For each position, head 0 and head 1 outputs must match.
        for pos in 0..2 {
            let h0 = &out[(pos * 2) * 8..(pos * 2) * 8 + 8];
            let h1 = &out[(pos * 2 + 1) * 8..(pos * 2 + 1) * 8 + 8];
            for (a, b) in h0.iter().zip(h1.iter()) {
                assert!((a - b).abs() < 1e-6, "head invariance violated");
            }
        }
    }

    #[test]
    fn cache_shape() {
        let r = rope(8, 16);
        assert_eq!(r.cos_cache.len(), 16 * 4);
        assert_eq!(r.sin_cache.len(), 16 * 4);
        // Position 0 cos == 1, sin == 0.
        for i in 0..4 {
            assert!((r.cos_cache[i] - 1.0).abs() < 1e-6);
            assert!(r.sin_cache[i].abs() < 1e-6);
        }
    }

    #[test]
    fn base_affects_freq() {
        // Two different bases produce different rotations for the same input at
        // a non-zero position.
        let r1 = Rope::new(RopeConfig {
            d_head: 8,
            max_seq_len: 16,
            base: 10000.0,
        })
        .expect("ok");
        let r2 = Rope::new(RopeConfig {
            d_head: 8,
            max_seq_len: 16,
            base: 500.0,
        })
        .expect("ok");
        let mut x = vec![0.0_f32; 2 * 8];
        for d in 0..8 {
            x[8 + d] = 1.0 + d as f32; // position 1 content
        }
        let o1 = r1.apply(&x, 2, 1).expect("ok");
        let o2 = r2.apply(&x, 2, 1).expect("ok");
        let diff: f32 = o1[8..16]
            .iter()
            .zip(o2[8..16].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-4, "base should change rotation, diff={diff}");
    }

    #[test]
    fn d_head_zero_error() {
        let r = Rope::new(RopeConfig {
            d_head: 0,
            max_seq_len: 16,
            base: 10000.0,
        });
        assert!(matches!(r, Err(DnnError::InvalidArgument(_))));
    }

    #[test]
    fn apply_wrong_len_error() {
        let r = rope(8, 16);
        let x = vec![0.0_f32; 10]; // not seq·heads·d_head
        let out = r.apply(&x, 2, 2);
        assert!(matches!(out, Err(DnnError::InvalidDimension(_))));
    }
}

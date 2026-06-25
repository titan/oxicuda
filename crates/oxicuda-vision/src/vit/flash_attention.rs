//! FlashAttention-2 online-softmax CPU reference.
//!
//! FlashAttention (Dao 2022) and FlashAttention-2 (Dao 2023) fuse the scaled
//! dot-product attention `softmax(QKᵀ / √d) V` into a single streaming pass that
//! never materialises the full `[N, N]` score matrix. The numerical trick is the
//! **online softmax** (Milakov & Gimelshein 2018): the running output, running
//! row-max `m`, and running normaliser `ℓ` are rescaled incrementally as each
//! block of keys is consumed, so the result is bit-for-bit the *same* value a
//! three-pass stable softmax would produce, but with `O(N)` rather than `O(N²)`
//! extra memory.
//!
//! On a GPU the win is memory-bandwidth (the scores stay in SRAM); on the CPU
//! the win is the same `O(N)` working set and a single fused loop. This module
//! provides the CPU reference so that the algorithm can be validated for
//! numerical agreement against the plain `crate::vit::vit_block::mhsa` path
//! before being lowered to a fused PTX/`oxicuda-dnn` kernel. The GPU-side fused
//! kernel (Tensor-Core / `wgmma` / warp-specialised) remains hardware-gated.
//!
//! All tensors are flat row-major `f32`. The attention is **causal-optional**:
//! with `causal = true`, query `i` attends only to keys `j ≤ i` (decoder-style),
//! which is exactly how text/decoder transformers mask the future.

use crate::error::{VisionError, VisionResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for FlashAttention-2 online-softmax attention.
#[derive(Debug, Clone, PartialEq)]
pub struct FlashAttnConfig {
    /// Number of attention heads. Must divide `embed_dim`.
    pub n_heads: usize,
    /// Model embedding dimension (`head_dim = embed_dim / n_heads`).
    pub embed_dim: usize,
    /// Key/value tile (block) length used by the streaming accumulation.
    ///
    /// This only affects the *order* of accumulation, never the result; it is
    /// the CPU analogue of the GPU `Bc` block size. Must be `> 0`.
    pub block_kv: usize,
    /// Apply a causal (lower-triangular) attention mask when `true`.
    pub causal: bool,
}

impl FlashAttnConfig {
    /// Create and validate a `FlashAttnConfig`.
    ///
    /// # Errors
    /// - [`VisionError::InvalidEmbedDim`] if `embed_dim == 0`.
    /// - [`VisionError::InvalidNumHeads`] if `n_heads == 0`.
    /// - [`VisionError::HeadDimMismatch`] if `embed_dim % n_heads != 0`.
    /// - [`VisionError::Internal`] if `block_kv == 0`.
    pub fn new(
        n_heads: usize,
        embed_dim: usize,
        block_kv: usize,
        causal: bool,
    ) -> VisionResult<Self> {
        if embed_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(embed_dim));
        }
        if n_heads == 0 {
            return Err(VisionError::InvalidNumHeads(n_heads));
        }
        if embed_dim % n_heads != 0 {
            return Err(VisionError::HeadDimMismatch { n_heads, embed_dim });
        }
        if block_kv == 0 {
            return Err(VisionError::Internal("flash block_kv must be > 0".into()));
        }
        Ok(Self {
            n_heads,
            embed_dim,
            block_kv,
            causal,
        })
    }

    /// Per-head dimension.
    #[must_use]
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.n_heads
    }

    /// Softmax temperature scaling `1 / √head_dim`.
    #[must_use]
    #[inline]
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim() as f32).sqrt()
    }
}

// ─── FlashAttention-2 ─────────────────────────────────────────────────────────

/// FlashAttention-2 multi-head attention via fused online softmax.
///
/// Inputs `q`, `k`, `v` are flat `[seq, embed_dim]` row-major tensors (the
/// heads are interleaved within the embedding dimension, identical to the
/// layout consumed by `crate::vit::vit_block::mhsa`). The keys/values may have
/// a different sequence length than the queries (cross-attention).
///
/// Returns the attention output `[seq_q, embed_dim]`.
///
/// # Errors
/// - [`VisionError::EmptyInput`] if any sequence length is 0.
/// - [`VisionError::DimensionMismatch`] if a tensor length is inconsistent with
///   the supplied sequence lengths and `embed_dim`.
/// - [`VisionError::NonFinite`] if the result contains a NaN/inf.
pub fn flash_attention(
    cfg: &FlashAttnConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_kv: usize,
) -> VisionResult<Vec<f32>> {
    let e = cfg.embed_dim;
    let n_heads = cfg.n_heads;
    let hd = cfg.head_dim();
    let scale = cfg.scale();

    if seq_q == 0 || seq_kv == 0 {
        return Err(VisionError::EmptyInput("flash attention sequence"));
    }
    if q.len() != seq_q * e {
        return Err(VisionError::DimensionMismatch {
            expected: seq_q * e,
            got: q.len(),
        });
    }
    if k.len() != seq_kv * e {
        return Err(VisionError::DimensionMismatch {
            expected: seq_kv * e,
            got: k.len(),
        });
    }
    if v.len() != seq_kv * e {
        return Err(VisionError::DimensionMismatch {
            expected: seq_kv * e,
            got: v.len(),
        });
    }

    let mut out = vec![0.0f32; seq_q * e];

    // Per-(head, query) streaming accumulator. We iterate keys in tiles of
    // `block_kv`; within each tile the running statistics (m, ℓ) and the
    // un-normalised output accumulator `acc` are updated by the FlashAttention-2
    // recurrence:
    //   m_new   = max(m, max_j s_ij)
    //   p_ij    = exp(s_ij - m_new)
    //   correct = exp(m - m_new)
    //   ℓ_new   = correct·ℓ + Σ_j p_ij
    //   acc_new = correct·acc + Σ_j p_ij · v_j
    // The final output is acc / ℓ.
    let mut acc = vec![0.0f32; hd];
    for h in 0..n_heads {
        let col0 = h * hd;
        for qi in 0..seq_q {
            // Keys this query may attend to.
            let kv_limit = if cfg.causal {
                // Align the causal boundary to the diagonal when seq_q == seq_kv;
                // for cross-attention the mask still uses absolute key index.
                (qi + 1).min(seq_kv)
            } else {
                seq_kv
            };
            if kv_limit == 0 {
                // No visible key (only possible for causal with qi+1==0, which
                // cannot happen) — leave output row zero defensively.
                continue;
            }

            let q_row = &q[qi * e + col0..qi * e + col0 + hd];

            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for a in acc.iter_mut() {
                *a = 0.0;
            }

            let mut kj = 0usize;
            while kj < kv_limit {
                let tile_end = (kj + cfg.block_kv).min(kv_limit);

                // Compute the block of scores and its local max.
                let tile_len = tile_end - kj;
                let mut scores = vec![0.0f32; tile_len];
                let mut tile_max = f32::NEG_INFINITY;
                for (t, kk) in (kj..tile_end).enumerate() {
                    let k_row = &k[kk * e + col0..kk * e + col0 + hd];
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q_row[d] * k_row[d];
                    }
                    let s = dot * scale;
                    scores[t] = s;
                    if s > tile_max {
                        tile_max = s;
                    }
                }

                // Online-softmax merge of this tile into the running stats.
                let m_new = m.max(tile_max);
                let correction = if m == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m - m_new).exp()
                };

                // Rescale the existing accumulator and normaliser.
                l *= correction;
                for a in acc.iter_mut() {
                    *a *= correction;
                }

                // Add this tile's contribution.
                for (t, kk) in (kj..tile_end).enumerate() {
                    let p = (scores[t] - m_new).exp();
                    l += p;
                    let v_row = &v[kk * e + col0..kk * e + col0 + hd];
                    for d in 0..hd {
                        acc[d] += p * v_row[d];
                    }
                }

                m = m_new;
                kj = tile_end;
            }

            // Normalise and write the output row for this head.
            let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
            let o_row = &mut out[qi * e + col0..qi * e + col0 + hd];
            for d in 0..hd {
                o_row[d] = acc[d] * inv_l;
            }
        }
    }

    if out.iter().any(|v| !v.is_finite()) {
        return Err(VisionError::NonFinite("flash attention output"));
    }
    Ok(out)
}

// ─── Reference (three-pass) attention for validation ───────────────────────────

/// Plain three-pass stable-softmax attention used as a numerical oracle.
///
/// Computes the full `[seq_q, seq_kv]` score matrix per head and applies a
/// max-subtracting softmax. This is `O(seq_q · seq_kv)` memory and exists so
/// that [`flash_attention`] can be verified to agree to within floating-point
/// tolerance. Same arguments and layout as [`flash_attention`].
///
/// # Errors
/// Mirrors [`flash_attention`]'s validation.
pub fn reference_attention(
    cfg: &FlashAttnConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_kv: usize,
) -> VisionResult<Vec<f32>> {
    let e = cfg.embed_dim;
    let n_heads = cfg.n_heads;
    let hd = cfg.head_dim();
    let scale = cfg.scale();

    if seq_q == 0 || seq_kv == 0 {
        return Err(VisionError::EmptyInput("reference attention sequence"));
    }
    if q.len() != seq_q * e || k.len() != seq_kv * e || v.len() != seq_kv * e {
        return Err(VisionError::DimensionMismatch {
            expected: seq_q * e,
            got: q.len(),
        });
    }

    let mut out = vec![0.0f32; seq_q * e];
    for h in 0..n_heads {
        let col0 = h * hd;
        for qi in 0..seq_q {
            let kv_limit = if cfg.causal {
                (qi + 1).min(seq_kv)
            } else {
                seq_kv
            };
            let q_row = &q[qi * e + col0..qi * e + col0 + hd];

            // Pass 1: scores + max.
            let mut scores = vec![0.0f32; kv_limit];
            let mut mx = f32::NEG_INFINITY;
            for (kk, sc) in scores.iter_mut().enumerate() {
                let k_row = &k[kk * e + col0..kk * e + col0 + hd];
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q_row[d] * k_row[d];
                }
                *sc = dot * scale;
                if *sc > mx {
                    mx = *sc;
                }
            }
            // Pass 2: exp + sum.
            let mut sum = 0.0f32;
            for sc in &mut scores {
                *sc = (*sc - mx).exp();
                sum += *sc;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            // Pass 3: weighted sum of V.
            let o_row = &mut out[qi * e + col0..qi * e + col0 + hd];
            for (kk, &p) in scores.iter().enumerate() {
                let w = p * inv;
                let v_row = &v[kk * e + col0..kk * e + col0 + hd];
                for d in 0..hd {
                    o_row[d] += w * v_row[d];
                }
            }
        }
    }
    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rand_vec(n: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        rng.fill_normal(&mut v);
        v
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn config_validation() {
        assert!(FlashAttnConfig::new(0, 64, 8, false).is_err());
        assert!(FlashAttnConfig::new(4, 0, 8, false).is_err());
        assert!(FlashAttnConfig::new(3, 64, 8, false).is_err()); // 64 % 3 != 0
        assert!(FlashAttnConfig::new(4, 64, 0, false).is_err());
        let cfg = FlashAttnConfig::new(4, 64, 8, false).expect("ok");
        assert_eq!(cfg.head_dim(), 16);
        assert!((cfg.scale() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn matches_reference_non_causal() {
        let cfg = FlashAttnConfig::new(4, 32, 5, false).expect("ok");
        let seq = 13;
        let mut rng = LcgRng::new(1);
        let q = rand_vec(seq * 32, &mut rng);
        let k = rand_vec(seq * 32, &mut rng);
        let v = rand_vec(seq * 32, &mut rng);
        let flash = flash_attention(&cfg, &q, &k, &v, seq, seq).expect("ok");
        let refr = reference_attention(&cfg, &q, &k, &v, seq, seq).expect("ok");
        let d = max_abs_diff(&flash, &refr);
        assert!(d < 1e-4, "flash vs reference max diff {d}");
    }

    #[test]
    fn matches_reference_causal() {
        let cfg = FlashAttnConfig::new(2, 16, 3, true).expect("ok");
        let seq = 17;
        let mut rng = LcgRng::new(2);
        let q = rand_vec(seq * 16, &mut rng);
        let k = rand_vec(seq * 16, &mut rng);
        let v = rand_vec(seq * 16, &mut rng);
        let flash = flash_attention(&cfg, &q, &k, &v, seq, seq).expect("ok");
        let refr = reference_attention(&cfg, &q, &k, &v, seq, seq).expect("ok");
        let d = max_abs_diff(&flash, &refr);
        assert!(d < 1e-4, "causal flash vs reference max diff {d}");
    }

    #[test]
    fn block_size_invariance() {
        // Different block sizes must produce (numerically) identical results.
        let seq = 11;
        let mut rng = LcgRng::new(3);
        let q = rand_vec(seq * 32, &mut rng);
        let k = rand_vec(seq * 32, &mut rng);
        let v = rand_vec(seq * 32, &mut rng);
        let cfg_a = FlashAttnConfig::new(4, 32, 1, false).expect("ok");
        let cfg_b = FlashAttnConfig::new(4, 32, 7, false).expect("ok");
        let cfg_c = FlashAttnConfig::new(4, 32, 64, false).expect("ok");
        let a = flash_attention(&cfg_a, &q, &k, &v, seq, seq).expect("ok");
        let b = flash_attention(&cfg_b, &q, &k, &v, seq, seq).expect("ok");
        let c = flash_attention(&cfg_c, &q, &k, &v, seq, seq).expect("ok");
        assert!(max_abs_diff(&a, &b) < 1e-5);
        assert!(max_abs_diff(&a, &c) < 1e-5);
    }

    #[test]
    fn cross_attention_shapes() {
        let cfg = FlashAttnConfig::new(2, 16, 4, false).expect("ok");
        let seq_q = 5;
        let seq_kv = 9;
        let mut rng = LcgRng::new(4);
        let q = rand_vec(seq_q * 16, &mut rng);
        let k = rand_vec(seq_kv * 16, &mut rng);
        let v = rand_vec(seq_kv * 16, &mut rng);
        let flash = flash_attention(&cfg, &q, &k, &v, seq_q, seq_kv).expect("ok");
        let refr = reference_attention(&cfg, &q, &k, &v, seq_q, seq_kv).expect("ok");
        assert_eq!(flash.len(), seq_q * 16);
        assert!(max_abs_diff(&flash, &refr) < 1e-4);
    }

    #[test]
    fn uniform_values_average() {
        // If all V rows are identical, attention (any weights) returns that row.
        let cfg = FlashAttnConfig::new(1, 8, 4, false).expect("ok");
        let seq = 6;
        let mut rng = LcgRng::new(5);
        let q = rand_vec(seq * 8, &mut rng);
        let k = rand_vec(seq * 8, &mut rng);
        let mut v = vec![0.0f32; seq * 8];
        let token = [1.0f32, -2.0, 3.0, 0.5, -1.5, 2.0, 0.25, -0.75];
        for r in 0..seq {
            v[r * 8..(r + 1) * 8].copy_from_slice(&token);
        }
        let out = flash_attention(&cfg, &q, &k, &v, seq, seq).expect("ok");
        for r in 0..seq {
            let row = &out[r * 8..(r + 1) * 8];
            for (o, t) in row.iter().zip(token.iter()) {
                assert!((o - t).abs() < 1e-4, "got {o} expected {t}");
            }
        }
    }

    #[test]
    fn causal_first_query_attends_only_to_first_key() {
        // Query 0 (causal) attends only to key 0 → output row 0 == V row 0.
        let cfg = FlashAttnConfig::new(1, 4, 2, true).expect("ok");
        let seq = 4;
        let mut rng = LcgRng::new(6);
        let q = rand_vec(seq * 4, &mut rng);
        let k = rand_vec(seq * 4, &mut rng);
        let v = rand_vec(seq * 4, &mut rng);
        let out = flash_attention(&cfg, &q, &k, &v, seq, seq).expect("ok");
        let row0 = &out[0..4];
        let v0 = &v[0..4];
        assert!(max_abs_diff(row0, v0) < 1e-5);
    }

    #[test]
    fn errors_on_bad_shapes() {
        let cfg = FlashAttnConfig::new(2, 16, 4, false).expect("ok");
        let q = vec![0.0f32; 5 * 16];
        let k = vec![0.0f32; 5 * 16];
        let v = vec![0.0f32; 5 * 16];
        // Wrong seq_q.
        assert!(flash_attention(&cfg, &q, &k, &v, 6, 5).is_err());
        // Empty.
        assert!(flash_attention(&cfg, &[], &k, &v, 0, 5).is_err());
    }

    #[test]
    fn deterministic() {
        let cfg = FlashAttnConfig::new(4, 32, 8, false).expect("ok");
        let seq = 9;
        let mut rng = LcgRng::new(7);
        let q = rand_vec(seq * 32, &mut rng);
        let k = rand_vec(seq * 32, &mut rng);
        let v = rand_vec(seq * 32, &mut rng);
        let a = flash_attention(&cfg, &q, &k, &v, seq, seq).expect("ok");
        let b = flash_attention(&cfg, &q, &k, &v, seq, seq).expect("ok");
        assert_eq!(a, b);
    }
}

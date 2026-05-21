//! FlashAttention CPU reference: tiled, online-softmax scaled dot-product
//! attention that never materialises the full `seq × seq` score matrix.
//!
//! # Reference
//!
//! Dao, Fu, Ermon, Rudra & Ré, *"FlashAttention: Fast and Memory-Efficient
//! Exact Attention with IO-Awareness"* (2022), arXiv:2205.14135.
//!
//! This is a numerically-exact, single-precision reference of the forward
//! pass. It processes queries in blocks of [`FlashAttentionConfig::block_size_q`]
//! and keys/values in blocks of [`FlashAttentionConfig::block_size_k`], keeping
//! per-row running statistics `(m_i, l_i)` (running max and running
//! denominator) and a running output accumulator `o_i` that are rescaled as
//! each new key block is folded in. The result is bit-comparable (within fp
//! rounding) to a standard softmax attention while using only `O(block)` score
//! storage.
//!
//! Layout: `q`, `k`, `v` are row-major `[seq × (n_heads · head_dim)]`; the
//! output has the same shape. Heads are processed independently. With
//! `causal == true`, key position `> query position` is masked to `−∞`.

use crate::error::{LmError, LmResult};

// ─── FlashAttentionConfig ──────────────────────────────────────────────────────

/// Configuration for [`flash_attention`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashAttentionConfig {
    /// Number of attention heads.
    pub n_heads: usize,
    /// Per-head feature dimension.
    pub head_dim: usize,
    /// Query block (tile) size along the sequence axis.
    pub block_size_q: usize,
    /// Key/value block (tile) size along the sequence axis.
    pub block_size_k: usize,
    /// Apply causal (lower-triangular) masking when `true`.
    pub causal: bool,
}

impl FlashAttentionConfig {
    /// Construct a configuration, validating that head/block sizes are positive.
    pub fn new(
        n_heads: usize,
        head_dim: usize,
        block_size_q: usize,
        block_size_k: usize,
        causal: bool,
    ) -> LmResult<Self> {
        if n_heads == 0 {
            return Err(LmError::InvalidConfig {
                msg: "FlashAttentionConfig: n_heads must be > 0".into(),
            });
        }
        if head_dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "FlashAttentionConfig: head_dim must be > 0".into(),
            });
        }
        if block_size_q == 0 || block_size_k == 0 {
            return Err(LmError::InvalidConfig {
                msg: "FlashAttentionConfig: block sizes must be > 0".into(),
            });
        }
        Ok(Self {
            n_heads,
            head_dim,
            block_size_q,
            block_size_k,
            causal,
        })
    }
}

// ─── flash_attention ───────────────────────────────────────────────────────────

/// Compute tiled online-softmax attention.
///
/// `q`, `k`, `v` are each row-major `[seq × (n_heads · head_dim)]`. Returns a
/// buffer of the same shape. `scale = 1/√head_dim` is applied to the logits.
pub fn flash_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cfg: FlashAttentionConfig,
) -> LmResult<Vec<f32>> {
    let model_dim = cfg.n_heads * cfg.head_dim;
    if model_dim == 0 {
        return Err(LmError::InvalidConfig {
            msg: "flash_attention: n_heads · head_dim must be > 0".into(),
        });
    }
    if q.is_empty() {
        return Err(LmError::EmptyInput {
            context: "flash_attention q",
        });
    }
    if q.len() % model_dim != 0 {
        return Err(LmError::DimensionMismatch {
            expected: model_dim,
            got: q.len(),
        });
    }
    // Keys and values must have identical length to queries (same seq, same
    // feature dim) for self-attention as specified.
    if k.len() != q.len() {
        return Err(LmError::DimensionMismatch {
            expected: q.len(),
            got: k.len(),
        });
    }
    if v.len() != q.len() {
        return Err(LmError::DimensionMismatch {
            expected: q.len(),
            got: v.len(),
        });
    }

    let seq = q.len() / model_dim;
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; q.len()];

    // Per-row running statistics, reused across heads to avoid reallocation.
    let bq = cfg.block_size_q.min(seq).max(1);
    let mut m_run = vec![f32::NEG_INFINITY; bq];
    let mut l_run = vec![0.0_f32; bq];
    // Output accumulator for the current query block: [bq × head_dim].
    let mut o_acc = vec![0.0_f32; bq * cfg.head_dim];

    for h in 0..cfg.n_heads {
        let head_off = h * cfg.head_dim;

        // Iterate query blocks.
        let mut qi = 0;
        while qi < seq {
            let q_end = (qi + cfg.block_size_q).min(seq);
            let q_rows = q_end - qi;

            // Reset running stats for this query block.
            for r in 0..q_rows {
                m_run[r] = f32::NEG_INFINITY;
                l_run[r] = 0.0;
                let base = r * cfg.head_dim;
                for d in 0..cfg.head_dim {
                    set(&mut o_acc, base + d, 0.0)?;
                }
            }

            // Iterate key blocks.
            let mut kj = 0;
            while kj < seq {
                let k_end = (kj + cfg.block_size_k).min(seq);

                // Causal pruning: if the entire key block lies strictly past the
                // last query row of this block, every entry is masked → skip.
                if cfg.causal && kj > (q_end - 1) {
                    break;
                }

                for r in 0..q_rows {
                    let q_pos = qi + r;
                    let q_base = q_pos * model_dim + head_off;

                    // S_ij row over this key block, with within-block causal mask.
                    // First pass: compute scores and the block row-max.
                    let mut block_max = f32::NEG_INFINITY;
                    let k_count = k_end - kj;
                    // Scratch scores for this key block (size ≤ block_size_k).
                    let mut scores = vec![f32::NEG_INFINITY; k_count];
                    for (c, score) in scores.iter_mut().enumerate() {
                        let k_pos = kj + c;
                        if cfg.causal && k_pos > q_pos {
                            // Masked → leave at −∞.
                            continue;
                        }
                        let k_base = k_pos * model_dim + head_off;
                        let mut dot = 0.0_f32;
                        for d in 0..cfg.head_dim {
                            dot += get(q, q_base + d)? * get(k, k_base + d)?;
                        }
                        let s = dot * scale;
                        *score = s;
                        if s > block_max {
                            block_max = s;
                        }
                    }

                    // If the whole block is masked for this row, nothing to fold.
                    if block_max == f32::NEG_INFINITY {
                        continue;
                    }

                    let m_old = m_run[r];
                    let m_new = m_old.max(block_max);
                    // Correction factor for previously-accumulated stats.
                    let corr = if m_old == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (m_old - m_new).exp()
                    };

                    // P̃ = exp(S − m_new); accumulate row-sum and o.
                    let mut p_sum = 0.0_f32;
                    let o_base = r * cfg.head_dim;
                    // Rescale the existing output accumulator by `corr`.
                    for d in 0..cfg.head_dim {
                        let prev = get(&o_acc, o_base + d)?;
                        set(&mut o_acc, o_base + d, prev * corr)?;
                    }
                    for (c, &s) in scores.iter().enumerate() {
                        if s == f32::NEG_INFINITY {
                            continue;
                        }
                        let p = (s - m_new).exp();
                        p_sum += p;
                        let k_pos = kj + c;
                        let v_base = k_pos * model_dim + head_off;
                        for d in 0..cfg.head_dim {
                            let add = p * get(v, v_base + d)?;
                            let prev = get(&o_acc, o_base + d)?;
                            set(&mut o_acc, o_base + d, prev + add)?;
                        }
                    }

                    l_run[r] = corr * l_run[r] + p_sum;
                    m_run[r] = m_new;
                }

                kj = k_end;
            }

            // Finalise: o_i /= l_i (guard fully-masked rows where l_i == 0).
            for (r, &denom) in l_run.iter().take(q_rows).enumerate() {
                let q_pos = qi + r;
                let out_base = q_pos * model_dim + head_off;
                let o_base = r * cfg.head_dim;
                if denom > 0.0 {
                    let inv = 1.0 / denom;
                    for d in 0..cfg.head_dim {
                        set(&mut out, out_base + d, get(&o_acc, o_base + d)? * inv)?;
                    }
                } else {
                    // Fully-masked row → output stays zero.
                    for d in 0..cfg.head_dim {
                        set(&mut out, out_base + d, 0.0)?;
                    }
                }
            }

            qi = q_end;
        }
    }

    Ok(out)
}

/// Bounds-checked read.
#[inline]
fn get(buf: &[f32], idx: usize) -> LmResult<f32> {
    buf.get(idx).copied().ok_or_else(|| LmError::Internal {
        msg: "flash_attention: buffer read out of range".into(),
    })
}

/// Bounds-checked write.
#[inline]
fn set(buf: &mut [f32], idx: usize, val: f32) -> LmResult<()> {
    let slot = buf.get_mut(idx).ok_or_else(|| LmError::Internal {
        msg: "flash_attention: buffer write out of range".into(),
    })?;
    *slot = val;
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic naive softmax attention used as the reference oracle.
    /// `q`/`k`/`v`: `[seq × (n_heads · head_dim)]` row-major.
    fn naive_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Vec<f32> {
        let model_dim = n_heads * head_dim;
        let seq = q.len() / model_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0_f32; q.len()];
        for h in 0..n_heads {
            let ho = h * head_dim;
            for i in 0..seq {
                let q_base = i * model_dim + ho;
                // Scores over all keys.
                let mut scores = vec![f32::NEG_INFINITY; seq];
                for (j, sc) in scores.iter_mut().enumerate() {
                    if causal && j > i {
                        continue;
                    }
                    let k_base = j * model_dim + ho;
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[k_base + d];
                    }
                    *sc = dot * scale;
                }
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if max_s == f32::NEG_INFINITY {
                    continue;
                }
                let mut sum = 0.0_f32;
                let exps: Vec<f32> = scores
                    .iter()
                    .map(|&s| {
                        if s == f32::NEG_INFINITY {
                            0.0
                        } else {
                            let e = (s - max_s).exp();
                            sum += e;
                            e
                        }
                    })
                    .collect();
                let out_base = i * model_dim + ho;
                if sum > 0.0 {
                    for (j, &e) in exps.iter().enumerate() {
                        if e == 0.0 {
                            continue;
                        }
                        let w = e / sum;
                        let v_base = j * model_dim + ho;
                        for d in 0..head_dim {
                            out[out_base + d] += w * v[v_base + d];
                        }
                    }
                }
            }
        }
        out
    }

    /// Small deterministic pseudo-data generator (no randomness crate).
    fn fill(n: usize, seed: u32) -> Vec<f32> {
        // Cheap deterministic hash → spread values in roughly [−1, 1].
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                let frac = (x % 10_007) as f32 / 10_007.0;
                frac * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn output_shape_matches_q() {
        let cfg = FlashAttentionConfig::new(2, 4, 2, 2, true).expect("valid cfg");
        let q = fill(5 * 8, 1);
        let k = fill(5 * 8, 2);
        let v = fill(5 * 8, 3);
        let out = flash_attention(&q, &k, &v, cfg).expect("flash ok");
        assert_eq!(out.len(), q.len());
    }

    #[test]
    fn matches_naive_causal_various_shapes() {
        // LOAD-BEARING: flash ≈ naive softmax within 1e-4 across configs.
        let cases = [
            (1usize, 4usize, 6usize, 2usize, 3usize),
            (2, 8, 7, 3, 2),
            (3, 4, 5, 2, 4),
            (4, 16, 9, 4, 4),
        ];
        for (n_heads, head_dim, seq, bq, bk) in cases {
            let model_dim = n_heads * head_dim;
            let q = fill(seq * model_dim, 11);
            let k = fill(seq * model_dim, 22);
            let v = fill(seq * model_dim, 33);
            let cfg =
                FlashAttentionConfig::new(n_heads, head_dim, bq, bk, true).expect("valid cfg");
            let flash = flash_attention(&q, &k, &v, cfg).expect("flash");
            let naive = naive_attention(&q, &k, &v, n_heads, head_dim, true);
            for (a, b) in flash.iter().zip(naive.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "mismatch h={n_heads} d={head_dim} seq={seq}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn matches_naive_non_causal() {
        let (n_heads, head_dim, seq) = (2usize, 8usize, 6usize);
        let model_dim = n_heads * head_dim;
        let q = fill(seq * model_dim, 5);
        let k = fill(seq * model_dim, 6);
        let v = fill(seq * model_dim, 7);
        let cfg = FlashAttentionConfig::new(n_heads, head_dim, 3, 2, false).expect("cfg");
        let flash = flash_attention(&q, &k, &v, cfg).expect("flash");
        let naive = naive_attention(&q, &k, &v, n_heads, head_dim, false);
        for (a, b) in flash.iter().zip(naive.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn causal_future_does_not_affect_past() {
        // Changing future K/V must leave earlier-position outputs unchanged.
        let (n_heads, head_dim, seq) = (1usize, 4usize, 6usize);
        let model_dim = n_heads * head_dim;
        let q = fill(seq * model_dim, 9);
        let mut k = fill(seq * model_dim, 10);
        let mut v = fill(seq * model_dim, 11);
        let cfg = FlashAttentionConfig::new(n_heads, head_dim, 2, 2, true).expect("cfg");
        let base = flash_attention(&q, &k, &v, cfg).expect("base");
        // Perturb the LAST token's K and V.
        let last = (seq - 1) * model_dim;
        for d in 0..model_dim {
            k[last + d] += 3.5;
            v[last + d] -= 2.0;
        }
        let perturbed = flash_attention(&q, &k, &v, cfg).expect("perturbed");
        // Positions 0..seq-1 must be byte-identical (they never see the last key).
        let cutoff = (seq - 1) * model_dim;
        for i in 0..cutoff {
            assert!(
                (base[i] - perturbed[i]).abs() < 1e-6,
                "position {i} changed: {} vs {}",
                base[i],
                perturbed[i]
            );
        }
    }

    #[test]
    fn single_block_when_block_larger_than_seq() {
        let (n_heads, head_dim, seq) = (2usize, 4usize, 5usize);
        let model_dim = n_heads * head_dim;
        let q = fill(seq * model_dim, 1);
        let k = fill(seq * model_dim, 2);
        let v = fill(seq * model_dim, 3);
        // Block sizes far larger than seq → single tile.
        let cfg = FlashAttentionConfig::new(n_heads, head_dim, 999, 999, true).expect("cfg");
        let flash = flash_attention(&q, &k, &v, cfg).expect("flash");
        let naive = naive_attention(&q, &k, &v, n_heads, head_dim, true);
        for (a, b) in flash.iter().zip(naive.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn block_size_one_equals_block_size_seq() {
        let (n_heads, head_dim, seq) = (2usize, 4usize, 7usize);
        let model_dim = n_heads * head_dim;
        let q = fill(seq * model_dim, 4);
        let k = fill(seq * model_dim, 8);
        let v = fill(seq * model_dim, 12);
        let cfg1 = FlashAttentionConfig::new(n_heads, head_dim, 1, 1, true).expect("cfg1");
        let cfg_seq =
            FlashAttentionConfig::new(n_heads, head_dim, seq, seq, true).expect("cfg_seq");
        let a = flash_attention(&q, &k, &v, cfg1).expect("bs=1");
        let b = flash_attention(&q, &k, &v, cfg_seq).expect("bs=seq");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn numerical_stability_large_logits() {
        // Scale Q,K by 1e3 → logits ~1e6; online softmax must stay finite and
        // still match a numerically-stable naive softmax.
        let (n_heads, head_dim, seq) = (1usize, 4usize, 8usize);
        let model_dim = n_heads * head_dim;
        let q: Vec<f32> = fill(seq * model_dim, 1).iter().map(|x| x * 1e3).collect();
        let k: Vec<f32> = fill(seq * model_dim, 2).iter().map(|x| x * 1e3).collect();
        let v = fill(seq * model_dim, 3);
        let cfg = FlashAttentionConfig::new(n_heads, head_dim, 3, 2, true).expect("cfg");
        let flash = flash_attention(&q, &k, &v, cfg).expect("flash");
        assert!(flash.iter().all(|x| x.is_finite()), "must be finite");
        let naive = naive_attention(&q, &k, &v, n_heads, head_dim, true);
        for (a, b) in flash.iter().zip(naive.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn multi_head_independence() {
        // Head outputs must not bleed: zeroing head-1's V leaves head-0 intact.
        let (n_heads, head_dim, seq) = (2usize, 4usize, 5usize);
        let model_dim = n_heads * head_dim;
        let q = fill(seq * model_dim, 7);
        let k = fill(seq * model_dim, 8);
        let mut v = fill(seq * model_dim, 9);
        let cfg = FlashAttentionConfig::new(n_heads, head_dim, 2, 2, true).expect("cfg");
        let base = flash_attention(&q, &k, &v, cfg).expect("base");
        // Zero out head 1's V across all positions.
        for i in 0..seq {
            for d in 0..head_dim {
                v[i * model_dim + head_dim + d] = 0.0;
            }
        }
        let modified = flash_attention(&q, &k, &v, cfg).expect("modified");
        // Head 0 slice (first head_dim of each row) must be unchanged.
        for i in 0..seq {
            for d in 0..head_dim {
                let idx = i * model_dim + d;
                assert!(
                    (base[idx] - modified[idx]).abs() < 1e-6,
                    "head 0 changed at ({i},{d})"
                );
            }
        }
    }

    #[test]
    fn single_query_single_key() {
        let cfg = FlashAttentionConfig::new(1, 4, 4, 4, true).expect("cfg");
        let q = fill(4, 1);
        let k = fill(4, 2);
        let v = fill(4, 3);
        let flash = flash_attention(&q, &k, &v, cfg).expect("flash");
        // seq=1 → softmax over a single key → output == v.
        for (a, b) in flash.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn scale_is_inv_sqrt_head_dim() {
        // With seq=1 the softmax is trivially 1, so the result equals v
        // regardless of scale — instead verify scale via a 2-key case against
        // a hand-rolled computation that explicitly divides by √head_dim.
        let head_dim = 4usize;
        let n_heads = 1usize;
        let q = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let k = vec![2.0_f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let v = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let cfg = FlashAttentionConfig::new(n_heads, head_dim, 2, 2, false).expect("cfg");
        let flash = flash_attention(&q, &k, &v, cfg).expect("flash");
        // Row 0: q·k0 = 2, q·k1 = 0; scale = 1/2. logits = [1, 0].
        let scale = 1.0 / (head_dim as f32).sqrt();
        let s0 = 2.0 * scale;
        let s1 = 0.0 * scale;
        let e0 = (s0 - s0).exp();
        let e1 = (s1 - s0).exp();
        let denom = e0 + e1;
        let w0 = e0 / denom;
        let w1 = e1 / denom;
        // out row0 dim0 = w0·v0[0] + w1·v1[0] = w0·1 + w1·0.
        assert!((flash[0] - w0).abs() < 1e-5, "{} vs {w0}", flash[0]);
        assert!((flash[1] - w1).abs() < 1e-5, "{} vs {w1}", flash[1]);
    }

    #[test]
    fn larger_sequence_matches_naive() {
        let (n_heads, head_dim, seq) = (3usize, 8usize, 33usize);
        let model_dim = n_heads * head_dim;
        let q = fill(seq * model_dim, 101);
        let k = fill(seq * model_dim, 202);
        let v = fill(seq * model_dim, 303);
        let cfg = FlashAttentionConfig::new(n_heads, head_dim, 8, 8, true).expect("cfg");
        let flash = flash_attention(&q, &k, &v, cfg).expect("flash");
        let naive = naive_attention(&q, &k, &v, n_heads, head_dim, true);
        for (a, b) in flash.iter().zip(naive.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn uneven_blocks_match_naive() {
        // seq not divisible by block sizes → ragged final tiles.
        let (n_heads, head_dim, seq) = (2usize, 4usize, 10usize);
        let model_dim = n_heads * head_dim;
        let q = fill(seq * model_dim, 13);
        let k = fill(seq * model_dim, 14);
        let v = fill(seq * model_dim, 15);
        let cfg = FlashAttentionConfig::new(n_heads, head_dim, 3, 4, true).expect("cfg");
        let flash = flash_attention(&q, &k, &v, cfg).expect("flash");
        let naive = naive_attention(&q, &k, &v, n_heads, head_dim, true);
        for (a, b) in flash.iter().zip(naive.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    // ── Error paths ───────────────────────────────────────────────────────

    #[test]
    fn err_head_dim_zero() {
        assert!(FlashAttentionConfig::new(2, 0, 2, 2, true).is_err());
    }

    #[test]
    fn err_n_heads_zero() {
        assert!(FlashAttentionConfig::new(0, 4, 2, 2, true).is_err());
    }

    #[test]
    fn err_block_size_zero() {
        assert!(FlashAttentionConfig::new(2, 4, 0, 2, true).is_err());
        assert!(FlashAttentionConfig::new(2, 4, 2, 0, true).is_err());
    }

    #[test]
    fn err_empty_input() {
        let cfg = FlashAttentionConfig::new(1, 4, 2, 2, true).expect("cfg");
        assert!(matches!(
            flash_attention(&[], &[], &[], cfg),
            Err(LmError::EmptyInput { .. })
        ));
    }

    #[test]
    fn err_feature_dim_mismatch() {
        // q length not a multiple of n_heads·head_dim.
        let cfg = FlashAttentionConfig::new(2, 4, 2, 2, true).expect("cfg");
        let q = vec![0.0_f32; 8 + 1];
        let k = vec![0.0_f32; 8 + 1];
        let v = vec![0.0_f32; 8 + 1];
        assert!(matches!(
            flash_attention(&q, &k, &v, cfg),
            Err(LmError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_kv_seq_mismatch() {
        let cfg = FlashAttentionConfig::new(1, 4, 2, 2, true).expect("cfg");
        let q = vec![0.0_f32; 2 * 4];
        let k = vec![0.0_f32; 3 * 4]; // different seq
        let v = vec![0.0_f32; 2 * 4];
        assert!(matches!(
            flash_attention(&q, &k, &v, cfg),
            Err(LmError::DimensionMismatch { .. })
        ));
        let v_bad = vec![0.0_f32; 3 * 4];
        assert!(matches!(
            flash_attention(&q, &k, &v_bad, cfg),
            Err(LmError::DimensionMismatch { .. })
        ));
    }
}

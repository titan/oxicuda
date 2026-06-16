//! Multi-head scaled dot-product attention with an optional causal mask that
//! also returns the per-query attention weight matrix.
//!
//! This complements [`crate::cross_attn::cross_attention::CrossAttention`],
//! which neither masks future positions nor exposes its attention weights.
//! The vision-language models (LLaVA-NeXT causal LM, Qwen-VL resampler) and the
//! Grounding-DINO bidirectional fusion all need at least one of those two extra
//! capabilities, so they share this single primitive rather than each
//! re-deriving the attention math. It reuses the existing
//! [`CrossAttnWeights`] / [`CrossAttnConfig`] types and the crate's
//! `softmax_rows_inplace` so the numerics stay consistent with the rest of the
//! crate.
//!
//! `Attn(Q, K, V) = softmax(mask(Q·Kᵀ / √d_k)) · V`, projected by `W_o`.

use crate::cross_attn::cross_attention::{CrossAttnConfig, CrossAttnWeights, softmax_rows_inplace};
use crate::error::{MmResult, MultiModalError};

/// Inputs to [`mha_with_weights`], grouped so the call stays within the
/// argument-count budget and reads clearly at every call site.
pub(crate) struct MhaArgs<'a> {
    /// Query rows, `[q_len × d_model]` row-major.
    pub query: &'a [f32],
    /// Key rows, `[kv_len × d_model]` row-major.
    pub key: &'a [f32],
    /// Value rows, `[kv_len × d_model]` row-major.
    pub value: &'a [f32],
    /// Number of query positions.
    pub q_len: usize,
    /// Number of key/value positions.
    pub kv_len: usize,
    /// Apply a causal (lower-triangular) mask — query `i` may not attend to any
    /// key `j > i`. Requires `q_len == kv_len` (self-attention).
    pub causal: bool,
}

/// Multi-head attention returning `(output, attention_weights)`.
///
/// - `output`: `[q_len × d_model]` after the output projection `W_o`.
/// - `attention_weights`: `[q_len × kv_len]`, the softmax weights **averaged
///   over heads**. Each row sums to 1; under a causal mask every entry with
///   `j > i` is exactly 0 (the average of head matrices that are each
///   lower-triangular and row-stochastic is itself lower-triangular and
///   row-stochastic).
pub(crate) fn mha_with_weights(
    args: &MhaArgs<'_>,
    cfg: &CrossAttnConfig,
    weights: &CrossAttnWeights,
) -> MmResult<(Vec<f32>, Vec<f32>)> {
    let d = cfg.d_model;
    let h = cfg.n_heads;
    let d_k = cfg.d_k;
    let d_v = cfg.d_v;
    let q_len = args.q_len;
    let kv_len = args.kv_len;

    if h == 0 || d == 0 {
        return Err(MultiModalError::InvalidHeads {
            heads: h,
            d_model: d,
        });
    }
    if args.query.len() != q_len * d {
        return Err(MultiModalError::DimensionMismatch {
            expected: q_len * d,
            got: args.query.len(),
        });
    }
    if args.key.len() != kv_len * d {
        return Err(MultiModalError::DimensionMismatch {
            expected: kv_len * d,
            got: args.key.len(),
        });
    }
    if args.value.len() != kv_len * d {
        return Err(MultiModalError::DimensionMismatch {
            expected: kv_len * d,
            got: args.value.len(),
        });
    }
    if kv_len == 0 {
        return Err(MultiModalError::MismatchedSeqLens { q_len, kv_len });
    }
    if args.causal && q_len != kv_len {
        return Err(MultiModalError::MismatchedSeqLens { q_len, kv_len });
    }

    // Linear projections Q/K/V → [seq × d_model].
    let proj_q = matmul(args.query, &weights.w_q, q_len, d, d)?;
    let proj_k = matmul(args.key, &weights.w_k, kv_len, d, d)?;
    let proj_v = matmul(args.value, &weights.w_v, kv_len, d, d)?;

    let scale = 1.0 / (d_k as f32).sqrt();
    let mut head_outputs = vec![0.0_f32; q_len * d];
    let mut weights_avg = vec![0.0_f32; q_len * kv_len];
    let inv_h = 1.0 / h as f32;

    for head in 0..h {
        let q_col = head * d_k;
        let v_col = head * d_v;

        // Per-head scores [q_len × kv_len].
        let mut scores = vec![0.0_f32; q_len * kv_len];
        for qi in 0..q_len {
            for ki in 0..kv_len {
                if args.causal && ki > qi {
                    scores[qi * kv_len + ki] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0_f32;
                for di in 0..d_k {
                    dot += proj_q[qi * d + q_col + di] * proj_k[ki * d + q_col + di];
                }
                scores[qi * kv_len + ki] = dot * scale;
            }
        }

        // Row-softmax over the key axis (masked entries → exp(−∞) = 0).
        softmax_rows_inplace(&mut scores, q_len, kv_len);

        // Accumulate the head-averaged weights and the value-weighted output.
        for qi in 0..q_len {
            for ki in 0..kv_len {
                weights_avg[qi * kv_len + ki] += inv_h * scores[qi * kv_len + ki];
            }
            for vi in 0..d_v {
                let mut s = 0.0_f32;
                for ki in 0..kv_len {
                    s += scores[qi * kv_len + ki] * proj_v[ki * d + v_col + vi];
                }
                head_outputs[qi * d + v_col + vi] = s;
            }
        }
    }

    let output = matmul(&head_outputs, &weights.w_o, q_len, d, d)?;
    Ok((output, weights_avg))
}

/// `A [rows × in_dim] · W [in_dim × out_dim]` → `[rows × out_dim]`, `W` row-major.
fn matmul(a: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> MmResult<Vec<f32>> {
    if w.len() != in_dim * out_dim {
        return Err(MultiModalError::DimensionMismatch {
            expected: in_dim * out_dim,
            got: w.len(),
        });
    }
    let mut out = vec![0.0_f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0.0_f32;
            for i in 0..in_dim {
                acc += a[r * in_dim + i] * w[i * out_dim + o];
            }
            out[r * out_dim + o] = acc;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn cfg() -> CrossAttnConfig {
        CrossAttnConfig::tiny()
    }

    #[test]
    fn weights_sum_to_one_per_query() {
        let cfg = cfg();
        let d = cfg.d_model;
        let mut rng = LcgRng::new(1);
        let w = CrossAttnWeights::random(&cfg, &mut rng);
        let q_len = 3;
        let kv_len = 5;
        let query: Vec<f32> = (0..q_len * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let kv: Vec<f32> = (0..kv_len * d).map(|i| (i as f32 * 0.07).cos()).collect();
        let args = MhaArgs {
            query: &query,
            key: &kv,
            value: &kv,
            q_len,
            kv_len,
            causal: false,
        };
        let (_, attn) = mha_with_weights(&args, &cfg, &w).expect("mha_with_weights should succeed");
        for qi in 0..q_len {
            let s: f32 = attn[qi * kv_len..(qi + 1) * kv_len].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "query {qi} sum {s}");
        }
    }

    #[test]
    fn causal_mask_is_lower_triangular() {
        let cfg = cfg();
        let d = cfg.d_model;
        let mut rng = LcgRng::new(2);
        let w = CrossAttnWeights::random(&cfg, &mut rng);
        let seq = 4;
        let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.05).sin()).collect();
        let args = MhaArgs {
            query: &x,
            key: &x,
            value: &x,
            q_len: seq,
            kv_len: seq,
            causal: true,
        };
        let (_, attn) = mha_with_weights(&args, &cfg, &w).expect("mha_with_weights should succeed");
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    assert_eq!(attn[i * seq + j], 0.0, "future leak at ({i},{j})");
                }
            }
            let s: f32 = attn[i * seq..(i + 1) * seq].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} sum {s}");
        }
    }

    #[test]
    fn causal_requires_equal_lengths() {
        let cfg = cfg();
        let d = cfg.d_model;
        let w = CrossAttnWeights::zeros(&cfg);
        let q = vec![0.0_f32; 3 * d];
        let kv = vec![0.0_f32; 5 * d];
        let args = MhaArgs {
            query: &q,
            key: &kv,
            value: &kv,
            q_len: 3,
            kv_len: 5,
            causal: true,
        };
        let err = mha_with_weights(&args, &cfg, &w).unwrap_err();
        assert!(matches!(err, MultiModalError::MismatchedSeqLens { .. }));
    }

    #[test]
    fn output_shape_and_finite() {
        let cfg = cfg();
        let d = cfg.d_model;
        let mut rng = LcgRng::new(3);
        let w = CrossAttnWeights::random(&cfg, &mut rng);
        let q_len = 2;
        let kv_len = 6;
        let query = vec![0.3_f32; q_len * d];
        let kv = vec![0.2_f32; kv_len * d];
        let args = MhaArgs {
            query: &query,
            key: &kv,
            value: &kv,
            q_len,
            kv_len,
            causal: false,
        };
        let (out, attn) =
            mha_with_weights(&args, &cfg, &w).expect("mha_with_weights should succeed");
        assert_eq!(out.len(), q_len * d);
        assert_eq!(attn.len(), q_len * kv_len);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

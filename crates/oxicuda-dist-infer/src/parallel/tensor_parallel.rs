//! Tensor-parallel collective simulation for column-sharded matmuls.
//!
//! Megatron-style tensor parallelism shards a weight matrix `W ∈ ℝ^{d_in×d_out}`
//! across `n_shards` ranks by *columns*: rank `s` owns the contiguous column
//! block `W[:, s·(d_out/n_shards) : (s+1)·(d_out/n_shards)]`. Each rank computes
//! a partial output `X · W_s`, and a collective combines the partials:
//!
//! * **all-gather** — concatenate the column blocks back into the full
//!   `[n_tokens × d_out]` activation (column-parallel linear).
//! * **all-reduce (sum)** — element-wise sum of equally-shaped partials
//!   (row-parallel linear, where each rank holds a full-width partial sum).
//!
//! These are *in-process simulations* of the corresponding NCCL collectives:
//! the math is identical to a multi-GPU run, executed sequentially on host.
//!
//! # Reference
//! - Shoeybi, Patwary, Puri, LeGresley, Casper, Catanzaro (2019) "Megatron-LM:
//!   Training Multi-Billion Parameter Language Models Using Model Parallelism."
//!   arXiv:1909.08053.

use crate::error::{DistInferError, DistInferResult};

/// Shard a column-parallel matmul `X · W` across `n_shards` ranks.
///
/// * `x` — input activations, row-major `[n_tokens × d_in]`.
/// * `w` — weight matrix, row-major `[d_in × d_out]`.
/// * Partitions `W` into `n_shards` contiguous column blocks of width
///   `d_out / n_shards`.
/// * Returns `n_shards` partial outputs, each row-major
///   `[n_tokens × (d_out / n_shards)]`.
///
/// # Errors
///
/// * [`DistInferError::InvalidWorldSize`] if `n_shards == 0`.
/// * [`DistInferError::TpFeaturesMisaligned`] if `d_out` is not divisible by
///   `n_shards`.
/// * [`DistInferError::DimensionMismatch`] if `x`/`w` lengths disagree with the
///   declared dimensions.
pub fn shard_matmul(
    x: &[f32],
    w: &[f32],
    n_tokens: usize,
    d_in: usize,
    d_out: usize,
    n_shards: usize,
) -> DistInferResult<Vec<Vec<f32>>> {
    if n_shards == 0 {
        return Err(DistInferError::InvalidWorldSize {
            world_size: 0,
            reason: "n_shards must be ≥ 1",
        });
    }
    if d_out % n_shards != 0 {
        return Err(DistInferError::TpFeaturesMisaligned {
            features: d_out,
            degree: n_shards,
        });
    }
    if x.len() != n_tokens * d_in {
        return Err(DistInferError::DimensionMismatch {
            expected: n_tokens * d_in,
            got: x.len(),
        });
    }
    if w.len() != d_in * d_out {
        return Err(DistInferError::DimensionMismatch {
            expected: d_in * d_out,
            got: w.len(),
        });
    }

    let shard_cols = d_out / n_shards;
    let mut shards = Vec::with_capacity(n_shards);
    for s in 0..n_shards {
        let col_start = s * shard_cols;
        let mut partial = vec![0.0_f32; n_tokens * shard_cols];
        for t in 0..n_tokens {
            let x_row = &x[t * d_in..t * d_in + d_in];
            let out_row = &mut partial[t * shard_cols..t * shard_cols + shard_cols];
            for (k, &xk) in x_row.iter().enumerate() {
                // W is [d_in × d_out] row-major: W[k, col] = w[k*d_out + col].
                let w_row_base = k * d_out + col_start;
                for (c, out) in out_row.iter_mut().enumerate() {
                    *out += xk * w[w_row_base + c];
                }
            }
        }
        shards.push(partial);
    }
    Ok(shards)
}

/// Element-wise sum across shards (simulated all-reduce, sum op).
///
/// Every shard must have the same length; the result is their element-wise sum
/// — the value every rank holds after an all-reduce in a row-parallel linear.
///
/// # Errors
///
/// * [`DistInferError::TooFewRanks`] if `shards` is empty.
/// * [`DistInferError::DimensionMismatch`] if the shards differ in length.
pub fn all_reduce_sum(shards: &[Vec<f32>]) -> DistInferResult<Vec<f32>> {
    let first = shards.first().ok_or(DistInferError::TooFewRanks {
        needed: 1,
        world_size: 0,
    })?;
    let len = first.len();
    let mut acc = first.clone();
    for shard in &shards[1..] {
        if shard.len() != len {
            return Err(DistInferError::DimensionMismatch {
                expected: len,
                got: shard.len(),
            });
        }
        for (a, &s) in acc.iter_mut().zip(shard.iter()) {
            *a += s;
        }
    }
    Ok(acc)
}

/// Concatenate shard outputs in rank order (simulated all-gather).
///
/// Note: this is a flat concatenation. For a column-parallel matmul whose
/// shards are `[n_tokens × shard_cols]`, the flat concatenation interleaves
/// tokens; use [`all_gather_columns`] to reconstruct the row-major
/// `[n_tokens × d_out]` activation.
#[must_use]
pub fn all_gather(shards: &[Vec<f32>]) -> Vec<f32> {
    let total: usize = shards.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total);
    for shard in shards {
        out.extend_from_slice(shard);
    }
    out
}

/// Reconstruct the full row-major `[n_tokens × d_out]` activation from the
/// per-shard column blocks produced by [`shard_matmul`].
///
/// This is the column-parallel all-gather: shard `s` supplied columns
/// `[s·shard_cols, (s+1)·shard_cols)` of every token row, and this stitches
/// them back together in column order.
///
/// # Errors
///
/// * [`DistInferError::TooFewRanks`] if `shards` is empty.
/// * [`DistInferError::DimensionMismatch`] if a shard length is inconsistent
///   with `n_tokens`.
pub fn all_gather_columns(shards: &[Vec<f32>], n_tokens: usize) -> DistInferResult<Vec<f32>> {
    if shards.is_empty() {
        return Err(DistInferError::TooFewRanks {
            needed: 1,
            world_size: 0,
        });
    }
    if n_tokens == 0 {
        return Err(DistInferError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    // Per-shard column widths.
    let mut shard_cols = Vec::with_capacity(shards.len());
    for shard in shards {
        if shard.len() % n_tokens != 0 {
            return Err(DistInferError::DimensionMismatch {
                expected: n_tokens,
                got: shard.len(),
            });
        }
        shard_cols.push(shard.len() / n_tokens);
    }
    let d_out: usize = shard_cols.iter().sum();
    let mut out = vec![0.0_f32; n_tokens * d_out];
    let mut col_offset = 0usize;
    for (shard, &cols) in shards.iter().zip(shard_cols.iter()) {
        for t in 0..n_tokens {
            let src = &shard[t * cols..t * cols + cols];
            let dst = &mut out[t * d_out + col_offset..t * d_out + col_offset + cols];
            dst.copy_from_slice(src);
        }
        col_offset += cols;
    }
    Ok(out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference dense `X · W`, row-major, for cross-checking the sharded path.
    fn full_matmul(x: &[f32], w: &[f32], n_tokens: usize, d_in: usize, d_out: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; n_tokens * d_out];
        for t in 0..n_tokens {
            for c in 0..d_out {
                let mut acc = 0.0_f32;
                for k in 0..d_in {
                    acc += x[t * d_in + k] * w[k * d_out + c];
                }
                out[t * d_out + c] = acc;
            }
        }
        out
    }

    #[test]
    fn shard_matmul_n_shards() {
        let x = vec![1.0_f32; 2 * 3];
        let w = vec![1.0_f32; 3 * 8];
        let shards = shard_matmul(&x, &w, 2, 3, 8, 4).expect("shard");
        assert_eq!(shards.len(), 4, "must produce n_shards partials");
    }

    #[test]
    fn shard_output_shape() {
        let x = vec![0.5_f32; 4 * 6];
        let w = vec![0.25_f32; 6 * 12];
        let shards = shard_matmul(&x, &w, 4, 6, 12, 3).expect("shard");
        // Each shard: [n_tokens × (d_out/n_shards)] = [4 × 4] = 16 elems.
        for s in &shards {
            assert_eq!(s.len(), 4 * 4);
        }
    }

    #[test]
    fn all_reduce_sum_correct() {
        let s0 = vec![1.0_f32, 2.0, 3.0];
        let s1 = vec![10.0_f32, 20.0, 30.0];
        let s2 = vec![100.0_f32, 200.0, 300.0];
        let reduced = all_reduce_sum(&[s0, s1, s2]).expect("reduce");
        assert_eq!(reduced, vec![111.0, 222.0, 333.0]);
    }

    #[test]
    fn all_gather_concat_len() {
        let s0 = vec![1.0_f32; 6];
        let s1 = vec![2.0_f32; 6];
        let g = all_gather(&[s0, s1]);
        assert_eq!(g.len(), 12);
    }

    #[test]
    fn shard_then_gather_matches_full_matmul() {
        // X is [3 × 4], W is [4 × 8], 2 shards.
        let x: Vec<f32> = (0..3 * 4).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let w: Vec<f32> = (0..4 * 8).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let shards = shard_matmul(&x, &w, 3, 4, 8, 2).expect("shard");
        let gathered = all_gather_columns(&shards, 3).expect("gather");
        let full = full_matmul(&x, &w, 3, 4, 8);
        assert_eq!(gathered.len(), full.len());
        for (g, f) in gathered.iter().zip(full.iter()) {
            assert!((g - f).abs() < 1e-4, "gathered {g} vs full {f}");
        }
    }

    #[test]
    fn n_shards_not_dividing_d_out_error() {
        let x = vec![1.0_f32; 2 * 3];
        let w = vec![1.0_f32; 3 * 7]; // d_out = 7, n_shards = 2 → 7 % 2 != 0
        let err = shard_matmul(&x, &w, 2, 3, 7, 2);
        assert!(matches!(
            err,
            Err(DistInferError::TpFeaturesMisaligned { .. })
        ));
    }

    #[test]
    fn all_reduce_mismatched_len_error() {
        let s0 = vec![1.0_f32, 2.0, 3.0];
        let s1 = vec![1.0_f32, 2.0]; // shorter
        let err = all_reduce_sum(&[s0, s1]);
        assert!(matches!(err, Err(DistInferError::DimensionMismatch { .. })));
    }

    #[test]
    fn n_shards_1_trivial() {
        // 1 shard reproduces the full matmul directly.
        let x: Vec<f32> = (0..2 * 3).map(|i| i as f32).collect();
        let w: Vec<f32> = (0..3 * 4).map(|i| i as f32).collect();
        let shards = shard_matmul(&x, &w, 2, 3, 4, 1).expect("shard");
        assert_eq!(shards.len(), 1);
        let full = full_matmul(&x, &w, 2, 3, 4);
        for (g, f) in shards[0].iter().zip(full.iter()) {
            assert!((g - f).abs() < 1e-4);
        }
    }

    #[test]
    fn output_finite() {
        let x: Vec<f32> = (0..5 * 6).map(|i| (i as f32).sin()).collect();
        let w: Vec<f32> = (0..6 * 9).map(|i| (i as f32).cos()).collect();
        let shards = shard_matmul(&x, &w, 5, 6, 9, 3).expect("shard");
        for s in &shards {
            for &v in s {
                assert!(v.is_finite(), "non-finite shard value {v}");
            }
        }
        let gathered = all_gather_columns(&shards, 5).expect("gather");
        for &v in &gathered {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn n_shards_0_error() {
        let x = vec![1.0_f32; 6];
        let w = vec![1.0_f32; 24];
        let err = shard_matmul(&x, &w, 2, 3, 8, 0);
        assert!(matches!(err, Err(DistInferError::InvalidWorldSize { .. })));
    }

    #[test]
    fn x_dim_mismatch_error() {
        let x = vec![1.0_f32; 5]; // should be 2*3 = 6
        let w = vec![1.0_f32; 24];
        let err = shard_matmul(&x, &w, 2, 3, 8, 2);
        assert!(matches!(err, Err(DistInferError::DimensionMismatch { .. })));
    }

    #[test]
    fn all_reduce_empty_error() {
        let shards: Vec<Vec<f32>> = Vec::new();
        let err = all_reduce_sum(&shards);
        assert!(matches!(err, Err(DistInferError::TooFewRanks { .. })));
    }
}

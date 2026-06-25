//! Token Merging (ToMe) for Vision Transformers.
//!
//! Bolya et al. "Token Merging: Your ViT But Faster." ICLR 2023.
//!
//! ToMe speeds up a ViT at inference *without retraining* by merging the `r`
//! most-similar token pairs after the attention step of each block, gradually
//! shrinking the sequence. The matching is **bipartite soft matching**:
//!
//! 1. Partition the `n` tokens into two alternating sets `A` (even indices) and
//!    `B` (odd indices).
//! 2. For every token in `A`, find its single most-similar token in `B` (cosine
//!    similarity over the attention *key* vectors, per the paper).
//! 3. Keep the `r` edges with the highest similarity; merge each chosen `A`
//!    token into its `B` partner by **size-weighted averaging** (so a merged
//!    token represents the mean of all original tokens it absorbed).
//! 4. Unmerged `A` tokens and all `B` tokens survive; the new sequence length is
//!    `n - r`.
//!
//! The per-token "size" (number of originals it represents) is tracked so that
//! later merges and the final attention remain a correct weighted mean — this is
//! the *proportional attention* bookkeeping from the paper. The implementation
//! is deterministic and key-driven, matching the reference algorithm.

use crate::error::{MmResult, MultiModalError};

/// Result of one [`merge_tokens`] step.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeResult {
    /// Merged token features `[(n - r) × dim]` row-major.
    pub tokens: Vec<f32>,
    /// Per-surviving-token size (number of originals represented), length
    /// `n - r`. Sums to the original token count.
    pub sizes: Vec<f32>,
}

/// Cosine similarity between two `dim`-length rows.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for k in 0..a.len() {
        dot += a[k] * b[k];
        na += a[k] * a[k];
        nb += b[k] * b[k];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / denom
}

/// Bipartite soft matching + merging of `r` token pairs.
///
/// - `tokens`: `[n × dim]` token features to be merged (row-major).
/// - `keys`: `[n × dim]` similarity keys used only for matching (typically the
///   attention key vectors). Pass `tokens` itself to match on features directly.
/// - `sizes`: per-token sizes, length `n`. Pass all-ones for the first merge.
/// - `r`: number of pairs to merge; clamped to `[0, |A|]` where `|A|` is the
///   size of the even-index partition.
///
/// Returns the merged tokens and their updated sizes (length `n - r_eff`).
///
/// # Errors
/// - [`MultiModalError::InvalidPatchCount`] when `n == 0`.
/// - [`MultiModalError::InvalidFeatureDim`] when `dim == 0`.
/// - [`MultiModalError::DimensionMismatch`] when any buffer length is
///   inconsistent with `n` / `dim`.
pub fn merge_tokens(
    tokens: &[f32],
    keys: &[f32],
    sizes: &[f32],
    n: usize,
    dim: usize,
    r: usize,
) -> MmResult<MergeResult> {
    if n == 0 {
        return Err(MultiModalError::InvalidPatchCount { n_patches: n });
    }
    if dim == 0 {
        return Err(MultiModalError::InvalidFeatureDim);
    }
    if tokens.len() != n * dim {
        return Err(MultiModalError::DimensionMismatch {
            expected: n * dim,
            got: tokens.len(),
        });
    }
    if keys.len() != n * dim {
        return Err(MultiModalError::DimensionMismatch {
            expected: n * dim,
            got: keys.len(),
        });
    }
    if sizes.len() != n {
        return Err(MultiModalError::DimensionMismatch {
            expected: n,
            got: sizes.len(),
        });
    }

    // Alternating bipartite partition: A = even indices, B = odd indices.
    let set_a: Vec<usize> = (0..n).step_by(2).collect();
    let set_b: Vec<usize> = (1..n).step_by(2).collect();
    let r_eff = r.min(set_a.len());

    // If nothing to merge (or B is empty), return the inputs unchanged.
    if r_eff == 0 || set_b.is_empty() {
        return Ok(MergeResult {
            tokens: tokens.to_vec(),
            sizes: sizes.to_vec(),
        });
    }

    // For each A token, its best B partner and the edge similarity.
    struct Edge {
        a: usize,
        b: usize,
        sim: f32,
    }
    let mut edges: Vec<Edge> = Vec::with_capacity(set_a.len());
    for &a in &set_a {
        let ka = &keys[a * dim..(a + 1) * dim];
        let mut best_b = set_b[0];
        let mut best_sim = f32::NEG_INFINITY;
        for &b in &set_b {
            let kb = &keys[b * dim..(b + 1) * dim];
            let s = cosine(ka, kb);
            if s > best_sim {
                best_sim = s;
                best_b = b;
            }
        }
        edges.push(Edge {
            a,
            b: best_b,
            sim: best_sim,
        });
    }

    // Keep the r_eff highest-similarity edges. Tie-break on A index for
    // determinism (stable sort by descending sim).
    edges.sort_by(|x, y| {
        y.sim
            .partial_cmp(&x.sim)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.a.cmp(&y.a))
    });

    // Mark which A tokens are merged and into which B token.
    let mut merged_a = vec![false; n];
    // Accumulate size-weighted sums into the destination B tokens.
    let mut acc = tokens.to_vec();
    let mut acc_size = sizes.to_vec();
    // First scale each token by its own size so the merge is a weighted sum.
    for i in 0..n {
        let sz = acc_size[i];
        for k in 0..dim {
            acc[i * dim + k] *= sz;
        }
    }

    for e in edges.iter().take(r_eff) {
        merged_a[e.a] = true;
        // Fold A's (already size-scaled) contribution into B.
        for k in 0..dim {
            acc[e.b * dim + k] += acc[e.a * dim + k];
        }
        acc_size[e.b] += acc_size[e.a];
    }

    // Build the surviving sequence in original index order: unmerged A then
    // interleaved with B (we simply iterate 0..n and skip merged A tokens),
    // dividing each survivor by its accumulated size to recover a mean.
    let survivors = n - r_eff;
    let mut out = Vec::with_capacity(survivors * dim);
    let mut out_sizes = Vec::with_capacity(survivors);
    for i in 0..n {
        if merged_a[i] {
            continue;
        }
        let sz = acc_size[i].max(1e-12);
        for k in 0..dim {
            out.push(acc[i * dim + k] / sz);
        }
        out_sizes.push(acc_size[i]);
    }

    if out.iter().any(|v| !v.is_finite()) {
        return Err(MultiModalError::NanEncountered {
            location: "merge_tokens",
        });
    }
    Ok(MergeResult {
        tokens: out,
        sizes: out_sizes,
    })
}

/// Apply [`merge_tokens`] repeatedly, removing `r` tokens per step, until the
/// sequence reaches `target_len` (or no further merge is possible).
///
/// This mirrors stacking ToMe across `L` transformer blocks. `keys` is reused as
/// the matcher at every step (in a real model fresh keys come from each block;
/// for a pure merging utility we reuse the supplied features).
///
/// # Errors
/// Propagates [`merge_tokens`]; additionally returns
/// [`MultiModalError::InvalidPatchCount`] when `target_len == 0`.
pub fn merge_to_length(
    tokens: &[f32],
    sizes: &[f32],
    n: usize,
    dim: usize,
    r: usize,
    target_len: usize,
) -> MmResult<MergeResult> {
    if target_len == 0 {
        return Err(MultiModalError::InvalidPatchCount {
            n_patches: target_len,
        });
    }
    let mut cur = MergeResult {
        tokens: tokens.to_vec(),
        sizes: sizes.to_vec(),
    };
    let mut cur_n = n;
    while cur_n > target_len {
        let step_r = r.min(cur_n - target_len);
        if step_r == 0 {
            break;
        }
        let next = merge_tokens(&cur.tokens, &cur.tokens, &cur.sizes, cur_n, dim, step_r)?;
        let new_n = next.sizes.len();
        if new_n == cur_n {
            // No progress possible (e.g. B partition empty) — stop.
            break;
        }
        cur_n = new_n;
        cur = next;
    }
    Ok(cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn reduces_length_by_r() {
        let n = 8;
        let dim = 4;
        let mut rng = LcgRng::new(1);
        let mut tokens = vec![0.0_f32; n * dim];
        rng.fill_normal(&mut tokens);
        let sizes = vec![1.0_f32; n];
        let res = merge_tokens(&tokens, &tokens, &sizes, n, dim, 3).expect("merge");
        assert_eq!(res.sizes.len(), n - 3);
        assert_eq!(res.tokens.len(), (n - 3) * dim);
    }

    #[test]
    fn sizes_conserve_token_mass() {
        // The total represented count must equal the original token count.
        let n = 10;
        let dim = 6;
        let mut rng = LcgRng::new(2);
        let mut tokens = vec![0.0_f32; n * dim];
        rng.fill_normal(&mut tokens);
        let sizes = vec![1.0_f32; n];
        let res = merge_tokens(&tokens, &tokens, &sizes, n, dim, 4).expect("merge");
        let total: f32 = res.sizes.iter().sum();
        assert!((total - n as f32).abs() < 1e-4, "mass {total} != {n}");
    }

    #[test]
    fn merges_most_similar_pair_first() {
        // Construct tokens where A-token 0 (index 0) is identical to B-token at
        // index 1, and everything else is far apart. With r=1 the merged token
        // representing both must be their mean (= the shared vector) and carry
        // size 2.
        let dim = 3;
        let n = 4;
        let mut tokens = vec![0.0_f32; n * dim];
        // index 0 (A) and index 1 (B) identical.
        tokens[0] = 1.0;
        tokens[dim] = 1.0;
        // index 2 (A) and index 3 (B) distinct, orthogonal.
        tokens[2 * dim + 1] = 1.0;
        tokens[3 * dim + 2] = 1.0;
        let sizes = vec![1.0_f32; n];
        let res = merge_tokens(&tokens, &tokens, &sizes, n, dim, 1).expect("merge");
        assert_eq!(res.sizes.len(), 3);
        // The merged survivor (originally index 1) should have size 2.
        let max_size = res.sizes.iter().cloned().fold(0.0_f32, f32::max);
        assert!((max_size - 2.0).abs() < 1e-5, "expected a size-2 token");
    }

    #[test]
    fn merged_token_is_size_weighted_mean() {
        // Two identical A/B tokens with value v merge to v (mean of [v, v]).
        let dim = 2;
        let n = 2;
        let tokens = vec![3.0, 4.0, 3.0, 4.0]; // index 0 (A), index 1 (B) identical
        let sizes = vec![1.0_f32; n];
        let res = merge_tokens(&tokens, &tokens, &sizes, n, dim, 1).expect("merge");
        assert_eq!(res.sizes.len(), 1);
        assert!((res.tokens[0] - 3.0).abs() < 1e-5);
        assert!((res.tokens[1] - 4.0).abs() < 1e-5);
        assert!((res.sizes[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_mean_respects_prior_sizes() {
        // A token of size 3 (value 0) merged with a B token of size 1 (value 4):
        // weighted mean = (3*0 + 1*4) / 4 = 1.0, size = 4.
        let dim = 1;
        let n = 2;
        let tokens = vec![0.0, 4.0];
        let sizes = vec![3.0, 1.0];
        let res = merge_tokens(&tokens, &tokens, &sizes, n, dim, 1).expect("merge");
        assert!((res.tokens[0] - 1.0).abs() < 1e-5, "got {}", res.tokens[0]);
        assert!((res.sizes[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn r_zero_is_identity() {
        let n = 5;
        let dim = 3;
        let mut rng = LcgRng::new(3);
        let mut tokens = vec![0.0_f32; n * dim];
        rng.fill_normal(&mut tokens);
        let sizes = vec![1.0_f32; n];
        let res = merge_tokens(&tokens, &tokens, &sizes, n, dim, 0).expect("merge");
        assert_eq!(res.tokens, tokens);
        assert_eq!(res.sizes, sizes);
    }

    #[test]
    fn r_clamped_to_partition_size() {
        // n=4 → |A| = 2; asking for r=10 merges at most 2.
        let n = 4;
        let dim = 2;
        let tokens = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let sizes = vec![1.0_f32; n];
        let res = merge_tokens(&tokens, &tokens, &sizes, n, dim, 10).expect("merge");
        assert_eq!(res.sizes.len(), n - 2);
    }

    #[test]
    fn merge_to_length_reaches_target() {
        let n = 16;
        let dim = 4;
        let mut rng = LcgRng::new(7);
        let mut tokens = vec![0.0_f32; n * dim];
        rng.fill_normal(&mut tokens);
        let sizes = vec![1.0_f32; n];
        let res = merge_to_length(&tokens, &sizes, n, dim, 4, 6).expect("merge_to_length");
        assert!(
            res.sizes.len() <= 6,
            "expected ≤ 6 tokens, got {}",
            res.sizes.len()
        );
        let total: f32 = res.sizes.iter().sum();
        assert!((total - n as f32).abs() < 1e-3, "mass {total} != {n}");
    }

    #[test]
    fn deterministic() {
        let n = 12;
        let dim = 5;
        let mut rng = LcgRng::new(11);
        let mut tokens = vec![0.0_f32; n * dim];
        rng.fill_normal(&mut tokens);
        let sizes = vec![1.0_f32; n];
        let a = merge_tokens(&tokens, &tokens, &sizes, n, dim, 3).expect("a");
        let b = merge_tokens(&tokens, &tokens, &sizes, n, dim, 3).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn zero_tokens_errors() {
        assert!(matches!(
            merge_tokens(&[], &[], &[], 0, 4, 1),
            Err(MultiModalError::InvalidPatchCount { .. })
        ));
    }

    #[test]
    fn shape_mismatch_errors() {
        let tokens = vec![0.0_f32; 3 * 4];
        let keys = vec![0.0_f32; 3 * 4];
        let sizes = vec![1.0_f32; 2]; // wrong
        assert!(matches!(
            merge_tokens(&tokens, &keys, &sizes, 3, 4, 1),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }
}

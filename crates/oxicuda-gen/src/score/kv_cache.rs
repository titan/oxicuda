//! Cross-attention key/value cache for fast conditional sampling.
//!
//! In text-conditioned diffusion (e.g. Stable Diffusion's UNet), every
//! cross-attention layer attends image-feature *queries* to *keys* and
//! *values* projected from a **fixed** text-embedding context. During an
//! `N`-step sampling loop the context never changes, so recomputing the
//! `K = context · W_Kᵀ` and `V = context · W_Vᵀ` projections at every step
//! is pure waste.
//!
//! [`CrossAttentionKvCache`] projects the context **once** at construction
//! and stores `K` / `V` as `[ctx_len × embed_dim]`. Each subsequent denoising
//! step calls [`CrossAttentionKvCache::attend`] with only the freshly
//! projected queries, reusing the cached tensors. For an `N`-step sample this
//! turns `N` context projections into `1`.
//!
//! The cache is layout-compatible with
//! [`crate::score::unet_block::CrossAttentionBlock`]: heads are interleaved on
//! the feature axis and the per-head softmax uses the same `1/√head_dim`
//! scaling, so a cached run is numerically identical to the uncached block.

use crate::error::{GenError, GenResult};

/// Row-major matmul `C[m, n] = A[m, k] · B[k, n]`.
fn matmul_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Numerically-stable softmax over a single row, in place.
fn softmax_row(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - mx).exp();
        sum += *v;
    }
    let inv = 1.0 / sum.max(1e-20);
    for v in row.iter_mut() {
        *v *= inv;
    }
}

// ─── CrossAttentionKvCache ──────────────────────────────────────────────────

/// Precomputed key/value cache for one cross-attention layer.
///
/// Construct once with the fixed context, then call [`Self::attend`] every
/// sampling step.
#[derive(Debug, Clone)]
pub struct CrossAttentionKvCache {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    ctx_len: usize,
    scale: f32,
    /// Cached keys `[ctx_len × embed_dim]`.
    k: Vec<f32>,
    /// Cached values `[ctx_len × embed_dim]`.
    v: Vec<f32>,
}

impl CrossAttentionKvCache {
    /// Build the cache by projecting `context` once.
    ///
    /// * `context`     — fixed conditioning, `[ctx_len × context_dim]`.
    /// * `kv_weight`   — fused key/value projection
    ///   `[2·embed_dim × context_dim]` (rows `0..embed_dim` → K,
    ///   rows `embed_dim..2·embed_dim` → V), matching
    ///   [`crate::score::unet_block::CrossAttentionBlock`].
    /// * `embed_dim`   — attention feature width (`num_heads · head_dim`).
    /// * `context_dim` — context feature width.
    /// * `num_heads`   — head count (`embed_dim % num_heads == 0`).
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] on a zero dimension or empty context.
    /// * [`GenError::DimensionMismatch`] if `embed_dim % num_heads != 0` or
    ///   `context.len() != ctx_len · context_dim`.
    /// * [`GenError::WeightShapeMismatch`] if `kv_weight` is the wrong size.
    pub fn build(
        context: &[f32],
        kv_weight: &[f32],
        embed_dim: usize,
        context_dim: usize,
        num_heads: usize,
    ) -> GenResult<Self> {
        if embed_dim == 0 || context_dim == 0 || num_heads == 0 {
            return Err(GenError::EmptyInput("dimensions must be > 0"));
        }
        if context.is_empty() {
            return Err(GenError::EmptyInput("context is empty"));
        }
        if embed_dim % num_heads != 0 {
            return Err(GenError::DimensionMismatch {
                expected: embed_dim - embed_dim % num_heads,
                got: embed_dim,
            });
        }
        if context.len() % context_dim != 0 {
            return Err(GenError::DimensionMismatch {
                expected: context.len() - context.len() % context_dim,
                got: context.len(),
            });
        }
        let ctx_len = context.len() / context_dim;
        if kv_weight.len() != 2 * embed_dim * context_dim {
            return Err(GenError::WeightShapeMismatch {
                weight: vec![2 * embed_dim, context_dim],
                input: vec![kv_weight.len()],
            });
        }
        // KV = context · W_KV  →  flat `[ctx_len × 2·embed_dim]`.
        //
        // The split below mirrors `CrossAttentionBlock::forward` exactly: the
        // first `ctx_len · embed_dim` flat elements are taken as K and the
        // remainder as V, so a cached run is bit-for-bit identical to the
        // uncached block (this cache is the cache *of that block*).
        let kv = matmul_nn(context, kv_weight, ctx_len, context_dim, 2 * embed_dim);
        let k = kv[..ctx_len * embed_dim].to_vec();
        let v = kv[ctx_len * embed_dim..].to_vec();
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            embed_dim,
            ctx_len,
            scale: 1.0 / (head_dim as f32).sqrt(),
            k,
            v,
        })
    }

    /// Attend pre-projected queries to the cached keys/values.
    ///
    /// * `q`       — queries `[q_len × embed_dim]` (already `x · W_Qᵀ`).
    /// * `q_len`   — number of query rows.
    ///
    /// Returns the attended output `[q_len × embed_dim]` (no residual or
    /// output projection — the caller applies those).
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] / [`GenError::EmptyInput`] on shape
    /// mismatch.
    pub fn attend(&self, q: &[f32], q_len: usize) -> GenResult<Vec<f32>> {
        if q.is_empty() {
            return Err(GenError::EmptyInput("q is empty"));
        }
        if q.len() != q_len * self.embed_dim {
            return Err(GenError::DimensionMismatch {
                expected: q_len * self.embed_dim,
                got: q.len(),
            });
        }
        let mut out = vec![0.0_f32; q_len * self.embed_dim];
        for h in 0..self.num_heads {
            let off = h * self.head_dim;
            for i in 0..q_len {
                // logits over the cached context for this query/head.
                let mut logits = vec![0.0_f32; self.ctx_len];
                for (j, logit) in logits.iter_mut().enumerate() {
                    let mut dot = 0.0_f32;
                    for d in 0..self.head_dim {
                        dot +=
                            q[i * self.embed_dim + off + d] * self.k[j * self.embed_dim + off + d];
                    }
                    *logit = dot * self.scale;
                }
                softmax_row(&mut logits);
                for d in 0..self.head_dim {
                    let mut acc = 0.0_f32;
                    for (j, &p) in logits.iter().enumerate() {
                        acc += p * self.v[j * self.embed_dim + off + d];
                    }
                    out[i * self.embed_dim + off + d] = acc;
                }
            }
        }
        Ok(out)
    }

    /// Context sequence length captured at build time.
    pub fn ctx_len(&self) -> usize {
        self.ctx_len
    }

    /// Attention feature width.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Number of heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Per-head feature dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Read-only view of the cached keys `[ctx_len × embed_dim]`.
    pub fn keys(&self) -> &[f32] {
        &self.k
    }

    /// Read-only view of the cached values `[ctx_len × embed_dim]`.
    pub fn values(&self) -> &[f32] {
        &self.v
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::score::unet_block::CrossAttentionBlock;

    const EPS: f32 = 1e-4;

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn build_rejects_bad_shapes() {
        let ctx = vec![0.0_f32; 4 * 8];
        let kv = vec![0.0_f32; 2 * 16 * 8];
        assert!(CrossAttentionKvCache::build(&ctx, &kv, 0, 8, 4).is_err());
        assert!(CrossAttentionKvCache::build(&ctx, &kv, 16, 8, 3).is_err()); // 16%3
        assert!(CrossAttentionKvCache::build(&ctx, &[0.0; 5], 16, 8, 4).is_err());
        assert!(CrossAttentionKvCache::build(&[], &kv, 16, 8, 4).is_err());
    }

    #[test]
    fn cache_matches_uncached_cross_attention() {
        // The cached attend() + output projection + residual must equal the
        // monolithic CrossAttentionBlock::forward().
        let (embed, ctx_dim, heads) = (16, 32, 4);
        let (seq, ctx_len) = (5, 7);
        let mut rng = LcgRng::new(42);
        let x = randn(&mut rng, seq * embed);
        let context = randn(&mut rng, ctx_len * ctx_dim);
        let q_w = randn(&mut rng, embed * embed);
        let kv_w = randn(&mut rng, 2 * embed * ctx_dim);
        let out_w = randn(&mut rng, embed * embed);

        // Reference: full block.
        let block = CrossAttentionBlock::new(embed, ctx_dim, heads).expect("new");
        let want = block
            .forward(&x, &context, seq, ctx_len, &q_w, &kv_w, &out_w)
            .expect("forward");

        // Cached path: project Q, attend to cached KV, project out, residual.
        let cache =
            CrossAttentionKvCache::build(&context, &kv_w, embed, ctx_dim, heads).expect("build");
        let q = matmul_nn(&x, &q_w, seq, embed, embed);
        let attended = cache.attend(&q, seq).expect("attend");
        let proj = matmul_nn(&attended, &out_w, seq, embed, embed);
        let got: Vec<f32> = x.iter().zip(&proj).map(|(&a, &b)| a + b).collect();

        assert!(
            max_abs_diff(&got, &want) < EPS,
            "cached vs block diff = {}",
            max_abs_diff(&got, &want)
        );
    }

    #[test]
    fn reuse_across_steps_is_deterministic() {
        // Same cache, different query batches → independent correct results,
        // and re-attending the same queries yields identical output (the cache
        // is immutable / side-effect free).
        let (embed, ctx_dim, heads) = (8, 8, 2);
        let ctx_len = 4;
        let mut rng = LcgRng::new(99);
        let context = randn(&mut rng, ctx_len * ctx_dim);
        let kv_w = randn(&mut rng, 2 * embed * ctx_dim);
        let cache =
            CrossAttentionKvCache::build(&context, &kv_w, embed, ctx_dim, heads).expect("build");

        let q1 = randn(&mut rng, 3 * embed);
        let a = cache.attend(&q1, 3).expect("attend");
        let b = cache.attend(&q1, 3).expect("attend");
        assert_eq!(a, b, "cache must be side-effect free across steps");

        let q2 = randn(&mut rng, embed);
        let c = cache.attend(&q2, 1).expect("attend");
        assert_eq!(c.len(), embed);
        assert!(c.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn cached_kv_shapes() {
        let (embed, ctx_dim, heads) = (12, 6, 3);
        let ctx_len = 5;
        let mut rng = LcgRng::new(1);
        let context = randn(&mut rng, ctx_len * ctx_dim);
        let kv_w = randn(&mut rng, 2 * embed * ctx_dim);
        let cache =
            CrossAttentionKvCache::build(&context, &kv_w, embed, ctx_dim, heads).expect("build");
        assert_eq!(cache.ctx_len(), ctx_len);
        assert_eq!(cache.embed_dim(), embed);
        assert_eq!(cache.num_heads(), heads);
        assert_eq!(cache.head_dim(), embed / heads);
        assert_eq!(cache.keys().len(), ctx_len * embed);
        assert_eq!(cache.values().len(), ctx_len * embed);
    }

    #[test]
    fn attend_dim_mismatch_errors() {
        let (embed, ctx_dim, heads) = (8, 8, 2);
        let ctx_len = 4;
        let mut rng = LcgRng::new(3);
        let context = randn(&mut rng, ctx_len * ctx_dim);
        let kv_w = randn(&mut rng, 2 * embed * ctx_dim);
        let cache =
            CrossAttentionKvCache::build(&context, &kv_w, embed, ctx_dim, heads).expect("build");
        let q = vec![0.0_f32; embed]; // 1 row
        assert!(cache.attend(&q, 2).is_err()); // claims 2 rows
        assert!(cache.attend(&[], 1).is_err());
    }
}

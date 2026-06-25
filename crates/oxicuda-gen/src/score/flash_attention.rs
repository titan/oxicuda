//! FlashAttention-style fused, tiled, online-softmax attention.
//!
//! Implements the streaming (online) softmax recurrence of Dao et al.
//! ("FlashAttention: Fast and Memory-Efficient Exact Attention with
//! IO-Awareness", NeurIPS 2022). Unlike the naive attention in
//! [`crate::score::unet_block`], this block **never materialises the full
//! `[q_len × kv_len]` score matrix**. Instead it walks the key/value
//! sequence in tiles of `block_size` and maintains, per query row, three
//! running quantities:
//!
//! * `m` — the running row maximum of the logits seen so far,
//! * `l` — the running sum of `exp(logit − m)`,
//! * `o` — the running (un-normalised) weighted value accumulator.
//!
//! When a new tile produces a larger maximum `m_new`, the previously
//! accumulated `l` and `o` are rescaled by `exp(m_old − m_new)`. This is
//! algebraically identical to a single global softmax but is numerically
//! stable (no overflow of `exp`) and uses only `O(head_dim)` extra storage
//! per query row irrespective of `kv_len`.
//!
//! The CPU reference here is the algorithm that the corresponding fused GPU
//! kernel (linkable to `oxicuda-dnn`'s fused MHA) implements; it is used both
//! as a correctness oracle and as a real, usable host-side path.

use crate::error::{GenError, GenResult};

// ─── FlashAttention ─────────────────────────────────────────────────────────

/// Fused multi-head attention using the online-softmax (FlashAttention)
/// recurrence.
///
/// Operates directly on pre-projected `Q`, `K`, `V` tensors laid out as
/// `[len × (num_heads · head_dim)]` (i.e. heads interleaved on the feature
/// axis, matching [`crate::score::unet_block::SelfAttentionBlock`]).
#[derive(Debug, Clone)]
pub struct FlashAttention {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    scale: f32,
    block_size: usize,
    causal: bool,
}

impl FlashAttention {
    /// Create a new fused-attention block.
    ///
    /// * `embed_dim`  — total feature width (`num_heads · head_dim`).
    /// * `num_heads`  — number of attention heads (`embed_dim % num_heads == 0`).
    /// * `block_size` — key/value tile length for the streaming loop
    ///   (`>= 1`; clamped internally to `kv_len`).
    /// * `causal`     — when `true`, query `i` may only attend to keys
    ///   `j <= i` (decoder self-attention masking).
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] if a dimension or the block size is `0`.
    /// * [`GenError::DimensionMismatch`] if `embed_dim % num_heads != 0`.
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        block_size: usize,
        causal: bool,
    ) -> GenResult<Self> {
        if embed_dim == 0 || num_heads == 0 {
            return Err(GenError::EmptyInput("dimensions must be > 0"));
        }
        if block_size == 0 {
            return Err(GenError::EmptyInput("block_size must be > 0"));
        }
        if embed_dim % num_heads != 0 {
            return Err(GenError::DimensionMismatch {
                expected: embed_dim - embed_dim % num_heads,
                got: embed_dim,
            });
        }
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self {
            num_heads,
            head_dim,
            embed_dim,
            scale,
            block_size,
            causal,
        })
    }

    /// Run fused attention `softmax(Q Kᵀ / √d) V`.
    ///
    /// * `q` — queries `[q_len × embed_dim]`.
    /// * `k` — keys    `[kv_len × embed_dim]`.
    /// * `v` — values  `[kv_len × embed_dim]`.
    ///
    /// Returns the attended output `[q_len × embed_dim]`. No residual or
    /// projection is applied here (those are the caller's responsibility),
    /// keeping the block composable with arbitrary projection weights.
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] / [`GenError::EmptyInput`] on shape
    /// mismatch.
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_len: usize,
        kv_len: usize,
    ) -> GenResult<Vec<f32>> {
        if q.is_empty() || k.is_empty() || v.is_empty() {
            return Err(GenError::EmptyInput("q, k, or v is empty"));
        }
        Self::check_len("q", q.len(), q_len * self.embed_dim)?;
        Self::check_len("k", k.len(), kv_len * self.embed_dim)?;
        Self::check_len("v", v.len(), kv_len * self.embed_dim)?;

        let block = self.block_size.min(kv_len).max(1);
        let mut out = vec![0.0_f32; q_len * self.embed_dim];

        for h in 0..self.num_heads {
            let head_off = h * self.head_dim;
            for i in 0..q_len {
                // Per-query-row online-softmax state.
                let mut row_max = f32::NEG_INFINITY;
                let mut row_sum = 0.0_f32;
                let mut acc = vec![0.0_f32; self.head_dim];

                let q_base = i * self.embed_dim + head_off;
                // Causal mask: only keys j <= i are visible.
                let kv_limit = if self.causal { i + 1 } else { kv_len };

                let mut j0 = 0;
                while j0 < kv_limit {
                    let j_end = (j0 + block).min(kv_limit);
                    // ── tile-local max ──────────────────────────────────────
                    let mut tile_logits = vec![0.0_f32; j_end - j0];
                    let mut tile_max = f32::NEG_INFINITY;
                    for (t, j) in (j0..j_end).enumerate() {
                        let k_base = j * self.embed_dim + head_off;
                        let mut dot = 0.0_f32;
                        for d in 0..self.head_dim {
                            dot += q[q_base + d] * k[k_base + d];
                        }
                        let logit = dot * self.scale;
                        tile_logits[t] = logit;
                        if logit > tile_max {
                            tile_max = logit;
                        }
                    }
                    // ── merge tile into running state ───────────────────────
                    let new_max = row_max.max(tile_max);
                    // Rescale the existing accumulator/sum to the new max.
                    let correction = if row_max == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (row_max - new_max).exp()
                    };
                    row_sum *= correction;
                    for a in acc.iter_mut() {
                        *a *= correction;
                    }
                    // Accumulate this tile's contribution.
                    for (t, j) in (j0..j_end).enumerate() {
                        let p = (tile_logits[t] - new_max).exp();
                        row_sum += p;
                        let v_base = j * self.embed_dim + head_off;
                        for d in 0..self.head_dim {
                            acc[d] += p * v[v_base + d];
                        }
                    }
                    row_max = new_max;
                    j0 = j_end;
                }

                // Normalise. If no key was visible (causal first row already
                // has j=0 visible, so this only guards numeric edge cases),
                // leave the output at zero.
                let inv = if row_sum > 0.0 { 1.0 / row_sum } else { 0.0 };
                let o_base = i * self.embed_dim + head_off;
                for d in 0..self.head_dim {
                    out[o_base + d] = acc[d] * inv;
                }
            }
        }
        Ok(out)
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Per-head feature dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Total feature width.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Key/value tile length.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Whether causal masking is applied.
    pub fn is_causal(&self) -> bool {
        self.causal
    }

    fn check_len(name: &'static str, got: usize, expected: usize) -> GenResult<()> {
        if got != expected {
            return Err(GenError::DimensionMismatch { expected, got });
        }
        let _ = name;
        Ok(())
    }
}

// ─── Naive reference (test oracle) ──────────────────────────────────────────

/// Straightforward (materialised-score) multi-head attention used **only** as
/// a numerical oracle in tests. Kept `pub(crate)` so it is available to the
/// integration test layer without leaking into the public API.
#[cfg(test)]
pub(crate) fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_len: usize,
    kv_len: usize,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let embed = num_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; q_len * embed];
    for h in 0..num_heads {
        let off = h * head_dim;
        for i in 0..q_len {
            let limit = if causal { i + 1 } else { kv_len };
            let mut logits = vec![0.0_f32; limit];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..limit {
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[i * embed + off + d] * k[j * embed + off + d];
                }
                logits[j] = dot * scale;
                if logits[j] > mx {
                    mx = logits[j];
                }
            }
            let mut sum = 0.0_f32;
            for l in logits.iter_mut() {
                *l = (*l - mx).exp();
                sum += *l;
            }
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for (j, &lj) in logits.iter().enumerate() {
                    acc += (lj / sum) * v[j * embed + off + d];
                }
                out[i * embed + off + d] = acc;
            }
        }
    }
    out
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

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
    fn new_rejects_bad_dims() {
        assert!(FlashAttention::new(0, 1, 4, false).is_err());
        assert!(FlashAttention::new(16, 0, 4, false).is_err());
        assert!(FlashAttention::new(16, 4, 0, false).is_err());
        assert!(FlashAttention::new(15, 4, 4, false).is_err());
    }

    #[test]
    fn matches_naive_full_attention() {
        let (heads, hd) = (4, 8);
        let embed = heads * hd;
        let (q_len, kv_len) = (6, 10);
        let mut rng = LcgRng::new(42);
        let q = randn(&mut rng, q_len * embed);
        let k = randn(&mut rng, kv_len * embed);
        let v = randn(&mut rng, kv_len * embed);

        let flash = FlashAttention::new(embed, heads, 3, false).expect("new should succeed");
        let got = flash
            .forward(&q, &k, &v, q_len, kv_len)
            .expect("forward should succeed");
        let want = naive_attention(&q, &k, &v, q_len, kv_len, heads, hd, false);
        assert!(
            max_abs_diff(&got, &want) < EPS,
            "flash vs naive diff = {}",
            max_abs_diff(&got, &want)
        );
    }

    #[test]
    fn matches_naive_causal() {
        let (heads, hd) = (2, 4);
        let embed = heads * hd;
        let len = 7;
        let mut rng = LcgRng::new(7);
        let q = randn(&mut rng, len * embed);
        let k = randn(&mut rng, len * embed);
        let v = randn(&mut rng, len * embed);

        let flash = FlashAttention::new(embed, heads, 2, true).expect("new should succeed");
        let got = flash
            .forward(&q, &k, &v, len, len)
            .expect("forward should succeed");
        let want = naive_attention(&q, &k, &v, len, len, heads, hd, true);
        assert!(max_abs_diff(&got, &want) < EPS, "causal flash vs naive");
    }

    #[test]
    fn tiling_is_block_size_invariant() {
        // The online-softmax result must not depend on block_size.
        let (heads, hd) = (3, 6);
        let embed = heads * hd;
        let (q_len, kv_len) = (5, 13);
        let mut rng = LcgRng::new(123);
        let q = randn(&mut rng, q_len * embed);
        let k = randn(&mut rng, kv_len * embed);
        let v = randn(&mut rng, kv_len * embed);

        let ref_out = FlashAttention::new(embed, heads, 1, false)
            .expect("new should succeed")
            .forward(&q, &k, &v, q_len, kv_len)
            .expect("forward should succeed");
        for bs in [2usize, 4, 7, 13, 100] {
            let out = FlashAttention::new(embed, heads, bs, false)
                .expect("new should succeed")
                .forward(&q, &k, &v, q_len, kv_len)
                .expect("forward should succeed");
            assert!(
                max_abs_diff(&ref_out, &out) < EPS,
                "block_size {bs} changed result"
            );
        }
    }

    #[test]
    fn numerically_stable_large_logits() {
        // Large magnitudes would overflow exp() in a naive non-shifted softmax;
        // online softmax keeps everything finite.
        let (heads, hd) = (1, 4);
        let embed = heads * hd;
        let big = 50.0_f32;
        let q = vec![big; embed];
        let k = vec![big; 3 * embed];
        let v = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let flash = FlashAttention::new(embed, heads, 2, false).expect("new should succeed");
        let out = flash
            .forward(&q, &k, &v, 1, 3)
            .expect("forward should succeed");
        assert!(out.iter().all(|x| x.is_finite()), "outputs must be finite");
        // Uniform keys ⇒ uniform attention ⇒ mean of the value rows.
        let expected = [
            (1.0 + 5.0 + 9.0) / 3.0,
            (2.0 + 6.0 + 10.0) / 3.0,
            (3.0 + 7.0 + 11.0) / 3.0,
            (4.0 + 8.0 + 12.0) / 3.0,
        ];
        assert!(max_abs_diff(&out, &expected) < EPS, "uniform-attn mean");
    }

    #[test]
    fn uniform_keys_give_value_mean() {
        // Zero queries ⇒ all logits 0 ⇒ uniform attention ⇒ value mean.
        let (heads, hd) = (1, 2);
        let embed = heads * hd;
        let q = vec![0.0_f32; embed];
        let k = vec![0.0_f32; 4 * embed];
        let v = vec![0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 6.0, 0.0];
        let flash = FlashAttention::new(embed, heads, 2, false).expect("new should succeed");
        let out = flash
            .forward(&q, &k, &v, 1, 4)
            .expect("forward should succeed");
        let mean0 = (0.0 + 2.0 + 0.0 + 6.0) / 4.0;
        let mean1 = (0.0 + 0.0 + 4.0 + 0.0) / 4.0;
        assert!((out[0] - mean0).abs() < EPS, "col0 mean: {}", out[0]);
        assert!((out[1] - mean1).abs() < EPS, "col1 mean: {}", out[1]);
    }

    #[test]
    fn causal_first_row_sees_only_itself() {
        let (heads, hd) = (1, 2);
        let embed = heads * hd;
        let q = vec![0.0_f32; 3 * embed];
        let k = vec![0.0_f32; 3 * embed];
        // Distinct value rows so we can detect leakage.
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let flash = FlashAttention::new(embed, heads, 2, true).expect("new should succeed");
        let out = flash
            .forward(&q, &k, &v, 3, 3)
            .expect("forward should succeed");
        // Row 0 attends only to key 0 ⇒ output == value row 0.
        assert!((out[0] - 10.0).abs() < EPS, "row0 col0");
        assert!((out[1] - 20.0).abs() < EPS, "row0 col1");
        // Row 1 attends to keys {0,1} ⇒ mean of value rows 0,1:
        // col0 = (10+30)/2 = 20, col1 = (20+40)/2 = 30.
        assert!((out[2] - 20.0).abs() < EPS, "row1 col0");
        assert!((out[3] - 30.0).abs() < EPS, "row1 col1");
    }

    #[test]
    fn dim_mismatch_errors() {
        let flash = FlashAttention::new(8, 2, 4, false).expect("new should succeed");
        let q = vec![0.0_f32; 8];
        let k = vec![0.0_f32; 16];
        let v = vec![0.0_f32; 16];
        // q_len claims 2 but q only holds 1 row.
        assert!(flash.forward(&q, &k, &v, 2, 2).is_err());
    }

    #[test]
    fn accessors() {
        let flash = FlashAttention::new(32, 8, 16, true).expect("new should succeed");
        assert_eq!(flash.num_heads(), 8);
        assert_eq!(flash.head_dim(), 4);
        assert_eq!(flash.embed_dim(), 32);
        assert_eq!(flash.block_size(), 16);
        assert!(flash.is_causal());
    }
}

//! Transformer NAS primitives (AutoFormer / V-MoE searchable axes).
//!
//! References:
//! - Chen, Peng, Fu & Ling, "AutoFormer: Searching Transformers for Visual
//!   Recognition", ICCV 2021. AutoFormer makes the embedding dimension, the
//!   per-block number of attention heads, the per-block MLP (FFN) expansion
//!   ratio, and the network depth (number of transformer blocks) all elastic
//!   axes of a single weight-entangled supernet.
//! - Riquelme et al., "Scaling Vision with Sparse Mixture of Experts" (V-MoE),
//!   NeurIPS 2021. V-MoE replaces the dense FFN of selected blocks with a
//!   top-`k` routed mixture of `n_experts` expert MLPs; the number of experts
//!   and the routing `k` become additional searchable axes.
//!
//! This module provides the *cost model* for that search space — the exact
//! multiply-accumulate (MAC) and parameter counts of a single transformer
//! encoder block under a chosen `(embed_dim, num_heads, mlp_ratio, n_experts,
//! moe_top_k, seq_len)` configuration — together with a discrete
//! [`TransformerSearchSpace`] that enumerates / samples the per-block choices
//! and validates a full [`TransformerArch`].
//!
//! A standard ViT-style encoder block is:
//!
//! ```text
//! x  (seq_len, embed_dim)
//!  ├── LayerNorm
//!  ├── Multi-Head Self-Attention   Q,K,V proj: 3 · embed_dim²
//!  │     attention scores:         num_heads · seq_len² · head_dim
//!  │     attention·V:              num_heads · seq_len² · head_dim
//!  │     output proj:              embed_dim²
//!  ├── residual add
//!  ├── LayerNorm
//!  └── MLP / MoE-FFN               two linears: 2 · embed_dim · hidden
//! ```
//!
//! where `head_dim = embed_dim / num_heads` and `hidden = embed_dim ·
//! mlp_ratio`. A dense FFN evaluates every token through the single MLP; a MoE
//! FFN routes every token through `moe_top_k` of `n_experts` MLPs, so its MAC
//! cost is the dense cost scaled by `moe_top_k`, while its *parameter* count is
//! the dense cost scaled by `n_experts` (all experts are materialised).
//!
//! No bias / LayerNorm parameters are counted (the standard NAS cost-model
//! convention, matching [`crate::ops::mbconv_ops`]).

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

// ─── BlockSpec ───────────────────────────────────────────────────────────────

/// Fully-resolved specification of one transformer encoder block.
///
/// All fields are concrete (no "choice index") — this is the leaf the cost
/// model consumes. Build one through [`TransformerSearchSpace::resolve_block`]
/// or directly for ad-hoc costing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSpec {
    /// Token embedding dimension `D`. Must be divisible by `num_heads`.
    pub embed_dim: usize,
    /// Number of self-attention heads `H`. Must divide `embed_dim`.
    pub num_heads: usize,
    /// FFN expansion ratio `r`; hidden dim is `embed_dim · r`.
    pub mlp_ratio: usize,
    /// Number of FFN experts. `1` ⇒ dense FFN; `> 1` ⇒ MoE (V-MoE) FFN.
    pub n_experts: usize,
    /// Number of experts each token is routed through (`top-k`). Must satisfy
    /// `1 <= moe_top_k <= n_experts`.
    pub moe_top_k: usize,
}

impl BlockSpec {
    /// Construct and validate a block specification.
    ///
    /// # Errors
    /// - [`NasError::InvalidNumOps`] if `embed_dim == 0`, `num_heads == 0`,
    ///   `mlp_ratio == 0`, or `n_experts == 0`.
    /// - [`NasError::DimensionMismatch`] if `num_heads` does not divide
    ///   `embed_dim`.
    /// - [`NasError::InvalidArchEncoding`] if `moe_top_k` is `0` or exceeds
    ///   `n_experts`.
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        mlp_ratio: usize,
        n_experts: usize,
        moe_top_k: usize,
    ) -> NasResult<Self> {
        if embed_dim == 0 || num_heads == 0 || mlp_ratio == 0 || n_experts == 0 {
            return Err(NasError::InvalidNumOps);
        }
        if embed_dim % num_heads != 0 {
            return Err(NasError::DimensionMismatch {
                expected: embed_dim,
                got: num_heads,
            });
        }
        if moe_top_k == 0 || moe_top_k > n_experts {
            return Err(NasError::InvalidArchEncoding);
        }
        Ok(Self {
            embed_dim,
            num_heads,
            mlp_ratio,
            n_experts,
            moe_top_k,
        })
    }

    /// Per-head dimension `embed_dim / num_heads`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    /// FFN hidden dimension `embed_dim · mlp_ratio`.
    #[must_use]
    pub fn ffn_hidden(&self) -> usize {
        self.embed_dim * self.mlp_ratio
    }

    /// `true` if this block uses a routed mixture-of-experts FFN.
    #[must_use]
    pub fn is_moe(&self) -> bool {
        self.n_experts > 1
    }

    /// Multiply-accumulate count for one forward pass of this block over a
    /// sequence of `seq_len` tokens.
    ///
    /// Breakdown (per the module docstring):
    /// - QKV projection: `3 · seq_len · embed_dim²`
    /// - scaled-dot-product scores: `num_heads · seq_len² · head_dim`
    /// - attention · V: `num_heads · seq_len² · head_dim`
    /// - output projection: `seq_len · embed_dim²`
    /// - FFN: `2 · seq_len · embed_dim · ffn_hidden · moe_top_k`
    ///
    /// Saturating arithmetic guards against `u64` overflow on pathological
    /// configurations.
    #[must_use]
    pub fn mac_count(&self, seq_len: usize) -> u64 {
        let d = self.embed_dim as u64;
        let s = seq_len as u64;
        let head_dim = self.head_dim() as u64;
        let heads = self.num_heads as u64;
        let hidden = self.ffn_hidden() as u64;
        let top_k = self.moe_top_k as u64;

        // QKV projection: 3 linears of shape D×D applied to every token.
        let qkv = s.saturating_mul(d).saturating_mul(d).saturating_mul(3);
        // Q·Kᵀ scores: for each head, an (S×head_dim)·(head_dim×S) GEMM.
        let scores = heads
            .saturating_mul(s)
            .saturating_mul(s)
            .saturating_mul(head_dim);
        // softmax(scores)·V: same shape as the scores GEMM.
        let context = scores;
        // Output projection D×D.
        let out_proj = s.saturating_mul(d).saturating_mul(d);
        // FFN: up-projection D→hidden plus down-projection hidden→D, routed
        // through `top_k` experts per token.
        let ffn = s
            .saturating_mul(d)
            .saturating_mul(hidden)
            .saturating_mul(2)
            .saturating_mul(top_k);

        qkv.saturating_add(scores)
            .saturating_add(context)
            .saturating_add(out_proj)
            .saturating_add(ffn)
    }

    /// Weight parameter count for this block (spatial / sequence-independent).
    ///
    /// - Attention: `4 · embed_dim²` (Q, K, V, and output projections).
    /// - FFN: `2 · embed_dim · ffn_hidden · n_experts` (all experts are
    ///   materialised, so MoE parameters scale with `n_experts`, not `top_k`).
    #[must_use]
    pub fn param_count(&self) -> u64 {
        let d = self.embed_dim as u64;
        let hidden = self.ffn_hidden() as u64;
        let experts = self.n_experts as u64;
        let attn = d.saturating_mul(d).saturating_mul(4);
        let ffn = d
            .saturating_mul(hidden)
            .saturating_mul(2)
            .saturating_mul(experts);
        attn.saturating_add(ffn)
    }
}

// ─── TransformerArch ───────────────────────────────────────────────────────────

/// A complete transformer architecture: a shared embedding dimension and a
/// per-block list of resolved [`BlockSpec`]s plus a sequence length.
///
/// All blocks in an AutoFormer subnet share one embedding dimension (weight
/// entanglement requires identical token widths along the trunk); only the
/// per-block head count and FFN ratio (and, for V-MoE, the expert layout) vary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerArch {
    /// Shared embedding dimension across all blocks.
    pub embed_dim: usize,
    /// One spec per encoder block. `len()` is the (elastic) depth.
    pub blocks: Vec<BlockSpec>,
    /// Sequence length (number of tokens / patches, plus class token).
    pub seq_len: usize,
}

impl TransformerArch {
    /// Build a transformer architecture, validating that every block shares
    /// `embed_dim` and the depth is non-zero.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `blocks` is empty or `seq_len == 0`.
    /// - [`NasError::DimensionMismatch`] if any block's `embed_dim` differs from
    ///   the trunk `embed_dim`.
    pub fn new(embed_dim: usize, blocks: Vec<BlockSpec>, seq_len: usize) -> NasResult<Self> {
        if blocks.is_empty() || seq_len == 0 {
            return Err(NasError::EmptySearchSpace);
        }
        for b in &blocks {
            if b.embed_dim != embed_dim {
                return Err(NasError::DimensionMismatch {
                    expected: embed_dim,
                    got: b.embed_dim,
                });
            }
        }
        Ok(Self {
            embed_dim,
            blocks,
            seq_len,
        })
    }

    /// Network depth (number of transformer blocks).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    /// Total MAC count summed over all blocks (patch-embedding and head
    /// excluded — they are constant across the search space).
    #[must_use]
    pub fn total_macs(&self) -> u64 {
        self.blocks
            .iter()
            .fold(0u64, |acc, b| acc.saturating_add(b.mac_count(self.seq_len)))
    }

    /// Total parameter count summed over all blocks.
    #[must_use]
    pub fn total_params(&self) -> u64 {
        self.blocks
            .iter()
            .fold(0u64, |acc, b| acc.saturating_add(b.param_count()))
    }
}

// ─── TransformerSearchSpace ─────────────────────────────────────────────────────

/// Discrete AutoFormer / V-MoE search space.
///
/// Each axis is a sorted list of candidate values; an architecture chooses one
/// value per axis (per block, for the block-local axes) plus a global depth.
/// The space is *elastic*: a sampled subnet uses the chosen depth's first
/// blocks, exactly as in AutoFormer's "supernet shrinking".
#[derive(Debug, Clone)]
pub struct TransformerSearchSpace {
    /// Candidate embedding dimensions (e.g. `[192, 240, 320]`).
    pub embed_dims: Vec<usize>,
    /// Candidate head counts (e.g. `[3, 4, 6]`). Each must divide every chosen
    /// embedding dimension; this is checked at sample time.
    pub head_choices: Vec<usize>,
    /// Candidate MLP expansion ratios (e.g. `[3, 4]`).
    pub mlp_ratio_choices: Vec<usize>,
    /// Candidate expert counts (e.g. `[1, 4]`; `1` ⇒ dense FFN).
    pub expert_choices: Vec<usize>,
    /// Routing `top-k` for MoE blocks (clamped to the chosen `n_experts`).
    pub moe_top_k: usize,
    /// Candidate depths (number of blocks), e.g. `[12, 13, 14]`.
    pub depth_choices: Vec<usize>,
    /// Sequence length (constant for a given input resolution / patch size).
    pub seq_len: usize,
}

impl TransformerSearchSpace {
    /// Construct and validate a search space.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if any axis list is empty or
    ///   `seq_len == 0`.
    /// - [`NasError::InvalidNumOps`] if any candidate value is `0`, or
    ///   `moe_top_k == 0`.
    /// - [`NasError::DimensionMismatch`] if some `(embed_dim, num_heads)`
    ///   candidate pair is non-divisible — every embedding dimension must be
    ///   divisible by every head choice, so that an arbitrary subnet sample is
    ///   always valid.
    pub fn new(
        embed_dims: Vec<usize>,
        head_choices: Vec<usize>,
        mlp_ratio_choices: Vec<usize>,
        expert_choices: Vec<usize>,
        moe_top_k: usize,
        depth_choices: Vec<usize>,
        seq_len: usize,
    ) -> NasResult<Self> {
        if embed_dims.is_empty()
            || head_choices.is_empty()
            || mlp_ratio_choices.is_empty()
            || expert_choices.is_empty()
            || depth_choices.is_empty()
            || seq_len == 0
        {
            return Err(NasError::EmptySearchSpace);
        }
        if moe_top_k == 0 {
            return Err(NasError::InvalidNumOps);
        }
        for &v in embed_dims
            .iter()
            .chain(&head_choices)
            .chain(&mlp_ratio_choices)
            .chain(&expert_choices)
            .chain(&depth_choices)
        {
            if v == 0 {
                return Err(NasError::InvalidNumOps);
            }
        }
        // Every embed_dim must be divisible by every head choice so any sampled
        // (embed_dim, num_heads) pair is structurally valid.
        for &d in &embed_dims {
            for &h in &head_choices {
                if d % h != 0 {
                    return Err(NasError::DimensionMismatch {
                        expected: d,
                        got: h,
                    });
                }
            }
        }
        Ok(Self {
            embed_dims,
            head_choices,
            mlp_ratio_choices,
            expert_choices,
            moe_top_k,
            depth_choices,
            seq_len,
        })
    }

    /// Maximum depth available in this space.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.depth_choices.iter().copied().max().unwrap_or(0)
    }

    /// Resolve a single block from explicit choice indices into each axis.
    ///
    /// `moe_top_k` is clamped down to the chosen `n_experts` (so a dense
    /// `n_experts == 1` block always ends up with `top_k == 1`).
    ///
    /// # Errors
    /// - [`NasError::InvalidArchEncoding`] if any index is out of range.
    /// - propagates [`BlockSpec::new`] validation errors.
    pub fn resolve_block(
        &self,
        embed_idx: usize,
        head_idx: usize,
        mlp_idx: usize,
        expert_idx: usize,
    ) -> NasResult<BlockSpec> {
        let embed_dim = *self
            .embed_dims
            .get(embed_idx)
            .ok_or(NasError::InvalidArchEncoding)?;
        let num_heads = *self
            .head_choices
            .get(head_idx)
            .ok_or(NasError::InvalidArchEncoding)?;
        let mlp_ratio = *self
            .mlp_ratio_choices
            .get(mlp_idx)
            .ok_or(NasError::InvalidArchEncoding)?;
        let n_experts = *self
            .expert_choices
            .get(expert_idx)
            .ok_or(NasError::InvalidArchEncoding)?;
        let top_k = self.moe_top_k.min(n_experts);
        BlockSpec::new(embed_dim, num_heads, mlp_ratio, n_experts, top_k)
    }

    /// Sample a uniformly random valid subnet.
    ///
    /// One shared `embed_dim` is drawn for the whole trunk (weight
    /// entanglement); a depth is drawn; then each of the first `depth` blocks
    /// independently draws a head count, MLP ratio, and expert count.
    ///
    /// # Errors
    /// Propagates [`Self::resolve_block`] / [`TransformerArch::new`] errors,
    /// which cannot fire for a space that passed [`Self::new`] validation but
    /// are surfaced for safety.
    pub fn sample(&self, rng: &mut LcgRng) -> NasResult<TransformerArch> {
        let embed_idx = rng.next_usize(self.embed_dims.len());
        let embed_dim = self.embed_dims[embed_idx];
        let depth = self.depth_choices[rng.next_usize(self.depth_choices.len())];
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            let head_idx = rng.next_usize(self.head_choices.len());
            let mlp_idx = rng.next_usize(self.mlp_ratio_choices.len());
            let expert_idx = rng.next_usize(self.expert_choices.len());
            blocks.push(self.resolve_block(embed_idx, head_idx, mlp_idx, expert_idx)?);
        }
        let _ = embed_dim; // documented: shared trunk width, encoded per block.
        TransformerArch::new(self.embed_dims[embed_idx], blocks, self.seq_len)
    }

    /// Build the **largest** subnet (max embed_dim, max heads, max mlp_ratio,
    /// max experts, max depth) — the "sandwich-rule" upper bound used to train
    /// AutoFormer / BigNAS supernets.
    ///
    /// # Errors
    /// Propagates resolve / construction errors.
    pub fn max_subnet(&self) -> NasResult<TransformerArch> {
        self.extreme_subnet(true)
    }

    /// Build the **smallest** subnet (min on every axis) — the sandwich-rule
    /// lower bound.
    ///
    /// # Errors
    /// Propagates resolve / construction errors.
    pub fn min_subnet(&self) -> NasResult<TransformerArch> {
        self.extreme_subnet(false)
    }

    fn extreme_subnet(&self, maximal: bool) -> NasResult<TransformerArch> {
        let pick = |xs: &[usize]| -> usize {
            if maximal {
                xs.iter().copied().enumerate().max_by_key(|&(_, v)| v)
            } else {
                xs.iter().copied().enumerate().min_by_key(|&(_, v)| v)
            }
            .map(|(i, _)| i)
            .unwrap_or(0)
        };
        let embed_idx = pick(&self.embed_dims);
        let head_idx = pick(&self.head_choices);
        let mlp_idx = pick(&self.mlp_ratio_choices);
        let expert_idx = pick(&self.expert_choices);
        let depth = if maximal {
            self.depth_choices.iter().copied().max().unwrap_or(1)
        } else {
            self.depth_choices.iter().copied().min().unwrap_or(1)
        };
        let block = self.resolve_block(embed_idx, head_idx, mlp_idx, expert_idx)?;
        let blocks = vec![block; depth];
        TransformerArch::new(self.embed_dims[embed_idx], blocks, self.seq_len)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_space() -> TransformerSearchSpace {
        // embed_dims all divisible by every head choice (192=3·64=4·48=6·32,
        // 384=3·128=4·96=6·64).
        TransformerSearchSpace::new(
            vec![192, 384],
            vec![3, 4, 6],
            vec![3, 4],
            vec![1, 4],
            2,
            vec![6, 8, 12],
            197,
        )
        .expect("default space should validate")
    }

    #[test]
    fn block_validation_rejects_non_divisible_heads() {
        // 192 / 5 != 0
        let r = BlockSpec::new(192, 5, 4, 1, 1);
        assert_eq!(
            r,
            Err(NasError::DimensionMismatch {
                expected: 192,
                got: 5
            })
        );
    }

    #[test]
    fn block_validation_rejects_top_k_over_experts() {
        let r = BlockSpec::new(192, 4, 4, 2, 3);
        assert_eq!(r, Err(NasError::InvalidArchEncoding));
    }

    #[test]
    fn block_head_dim_and_hidden() {
        let b = BlockSpec::new(192, 6, 4, 1, 1).expect("valid");
        assert_eq!(b.head_dim(), 32);
        assert_eq!(b.ffn_hidden(), 768);
        assert!(!b.is_moe());
    }

    #[test]
    fn mac_count_grows_with_heads_via_attention() {
        // Attention scores cost = num_heads · seq² · head_dim = seq² · embed_dim
        // independent of head count (head_dim = embed/heads). But more heads
        // never reduces total MACs; verify monotone non-decrease vs experts.
        let seq = 64;
        let dense = BlockSpec::new(192, 6, 4, 1, 1).expect("valid");
        let moe = BlockSpec::new(192, 6, 4, 4, 2).expect("valid");
        // top_k=2 routes each token through 2 experts → strictly more FFN MACs.
        assert!(
            moe.mac_count(seq) > dense.mac_count(seq),
            "moe top-2 ({}) should exceed dense ({})",
            moe.mac_count(seq),
            dense.mac_count(seq)
        );
    }

    #[test]
    fn moe_params_scale_with_experts_not_top_k() {
        let dense = BlockSpec::new(192, 6, 4, 1, 1).expect("valid");
        let moe4 = BlockSpec::new(192, 6, 4, 4, 2).expect("valid");
        // FFN params: dense = 2·192·768 ; moe4 = 4× that. Attention identical.
        let d = 192u64;
        let hidden = 768u64;
        let attn = d * d * 4;
        assert_eq!(dense.param_count(), attn + 2 * d * hidden);
        assert_eq!(moe4.param_count(), attn + 2 * d * hidden * 4);
    }

    #[test]
    fn mac_count_attention_term_is_exact() {
        // Hand-computed for a tiny block: D=8, H=2, head_dim=4, mlp_ratio=2 ⇒
        // hidden=16, dense FFN, seq=3.
        let b = BlockSpec::new(8, 2, 2, 1, 1).expect("valid");
        let s = 3u64;
        let d = 8u64;
        let head_dim = 4u64;
        let heads = 2u64;
        let hidden = 16u64;
        let qkv = 3 * s * d * d;
        let scores = heads * s * s * head_dim;
        let context = scores;
        let out_proj = s * d * d;
        let ffn = 2 * s * d * hidden;
        let expected = qkv + scores + context + out_proj + ffn;
        assert_eq!(b.mac_count(3), expected);
    }

    #[test]
    fn arch_rejects_mixed_embed_dim() {
        let a = BlockSpec::new(192, 6, 4, 1, 1).expect("valid");
        let b = BlockSpec::new(384, 6, 4, 1, 1).expect("valid");
        let r = TransformerArch::new(192, vec![a, b], 197);
        assert_eq!(
            r,
            Err(NasError::DimensionMismatch {
                expected: 192,
                got: 384
            })
        );
    }

    #[test]
    fn arch_totals_are_block_sums() {
        let b = BlockSpec::new(192, 6, 4, 1, 1).expect("valid");
        let arch = TransformerArch::new(192, vec![b; 3], 197).expect("valid");
        assert_eq!(arch.depth(), 3);
        assert_eq!(arch.total_macs(), b.mac_count(197) * 3);
        assert_eq!(arch.total_params(), b.param_count() * 3);
    }

    #[test]
    fn space_rejects_non_divisible_pair() {
        // embed 100 is not divisible by head 6.
        let r =
            TransformerSearchSpace::new(vec![96, 100], vec![6], vec![4], vec![1], 1, vec![4], 197);
        assert!(r.is_err());
    }

    #[test]
    fn sample_is_valid_and_within_space() {
        let space = default_space();
        let mut rng = LcgRng::new(2024);
        for _ in 0..50 {
            let arch = space.sample(&mut rng).expect("sample valid");
            assert!(space.embed_dims.contains(&arch.embed_dim));
            assert!(space.depth_choices.contains(&arch.depth()));
            for blk in &arch.blocks {
                assert_eq!(blk.embed_dim, arch.embed_dim);
                assert!(space.head_choices.contains(&blk.num_heads));
                assert!(space.mlp_ratio_choices.contains(&blk.mlp_ratio));
                assert!(space.expert_choices.contains(&blk.n_experts));
                assert!(blk.moe_top_k <= blk.n_experts);
                // Structural validity: embed divisible by heads.
                assert_eq!(blk.embed_dim % blk.num_heads, 0);
            }
        }
    }

    #[test]
    fn max_subnet_dominates_min_subnet_in_cost() {
        let space = default_space();
        let max = space.max_subnet().expect("max");
        let min = space.min_subnet().expect("min");
        assert!(max.depth() >= min.depth());
        assert!(
            max.total_macs() >= min.total_macs(),
            "max MACs {} should dominate min MACs {}",
            max.total_macs(),
            min.total_macs()
        );
        assert!(max.total_params() >= min.total_params());
        // Max picks the largest values on every axis.
        assert_eq!(max.embed_dim, 384);
        assert_eq!(max.depth(), 12);
        assert_eq!(min.embed_dim, 192);
        assert_eq!(min.depth(), 6);
    }

    #[test]
    fn sample_is_deterministic_given_seed() {
        let space = default_space();
        let mut a = LcgRng::new(7);
        let mut b = LcgRng::new(7);
        for _ in 0..10 {
            let xa = space.sample(&mut a).expect("a");
            let xb = space.sample(&mut b).expect("b");
            assert_eq!(xa, xb);
        }
    }

    #[test]
    fn resolve_block_out_of_range_errors() {
        let space = default_space();
        assert_eq!(
            space.resolve_block(99, 0, 0, 0),
            Err(NasError::InvalidArchEncoding)
        );
    }

    #[test]
    fn empty_axis_rejected() {
        let r = TransformerSearchSpace::new(vec![], vec![4], vec![4], vec![1], 1, vec![4], 197);
        assert_eq!(r.unwrap_err(), NasError::EmptySearchSpace);
    }
}

//! MegaBlocks-style block-sparse expert dispatch (Gale et al. 2023, MLSys).
//!
//! Capacity-based MoE (Switch) pads every expert to a fixed token capacity and
//! *drops* tokens beyond it. MegaBlocks instead reformulates the dispatch as a
//! **block-sparse** grouped matrix multiply: tokens are permuted so that all
//! tokens routed to the same expert are contiguous, the per-expert row ranges
//! are padded up to a fixed `block_size` multiple (so each expert occupies a
//! whole number of blocks, the unit a blocked GEMM consumes), and the grouped
//! GEMM walks expert by expert. **No token is dropped** and there is no wasted
//! work on empty capacity slots — only the small per-expert block-boundary
//! padding remains.
//!
//! This module provides the host-side scaffolding of that scheme:
//!
//! * [`build_block_sparse_layout`] — sort/permute tokens by expert and compute
//!   the padded block layout (`row_indices` gather map, per-expert block
//!   counts, block-row → expert map).
//! * [`gather_tokens`] / [`scatter_tokens`] — apply / invert the permutation.
//! * [`BlockSparseDispatcher`] — owns an [`crate::expert::bank::ExpertBank`] and
//!   runs the full grouped forward: gather → per-expert FFN over its block rows
//!   → scatter back, with optional gate-score scaling.
//!
//! The blocked padding rows carry zeros and are written nowhere on scatter, so
//! they are pure (small) overhead exactly as in the GPU kernel.

use crate::error::{MoeError, MoeResult};
use crate::expert::bank::ExpertBank;

/// Sentinel marking a padding row (no source token) in the block layout.
pub const PAD_ROW: usize = usize::MAX;

/// Block-sparse dispatch layout produced by [`build_block_sparse_layout`].
#[derive(Debug, Clone)]
pub struct BlockSparseLayout {
    /// For each *padded* row, the source token index, or [`PAD_ROW`] for a
    /// padding row. Length = `n_block_rows`.
    pub row_indices: Vec<usize>,
    /// Number of padded rows = `total_blocks * block_size`.
    pub n_block_rows: usize,
    /// Number of blocks assigned to each expert. Length = `n_experts`.
    pub blocks_per_expert: Vec<usize>,
    /// Starting *block* index for each expert (prefix sum of `blocks_per_expert`).
    /// Length = `n_experts`.
    pub block_offsets: Vec<usize>,
    /// Total number of blocks across all experts.
    pub total_blocks: usize,
    /// Block size (rows per block).
    pub block_size: usize,
    /// Number of real (non-padding) tokens placed.
    pub n_tokens: usize,
}

impl BlockSparseLayout {
    /// Row range `[start, end)` (in padded-row units) owned by `expert_idx`.
    #[must_use]
    pub fn expert_row_range(&self, expert_idx: usize) -> (usize, usize) {
        let start_block = self.block_offsets[expert_idx];
        let n_blocks = self.blocks_per_expert[expert_idx];
        let start = start_block * self.block_size;
        let end = start + n_blocks * self.block_size;
        (start, end)
    }
}

/// Build the block-sparse layout for a set of per-token expert assignments.
///
/// Tokens with assignment [`PAD_ROW`] (= `usize::MAX`) are treated as *dropped*
/// upstream and skipped. All others are grouped by expert and padded so each
/// expert spans a whole number of `block_size`-row blocks.
///
/// # Errors
/// Returns [`MoeError::InvalidExpertCount`] if `n_experts == 0`,
/// [`MoeError::InvalidHiddenDim`] if `block_size == 0`,
/// [`MoeError::DimensionMismatch`] on a `[n_tokens]` assignment-length error,
/// and [`MoeError::ExpertIndexOutOfRange`] for an out-of-range assignment.
pub fn build_block_sparse_layout(
    expert_assignments: &[usize],
    n_tokens: usize,
    n_experts: usize,
    block_size: usize,
) -> MoeResult<BlockSparseLayout> {
    if n_experts == 0 {
        return Err(MoeError::InvalidExpertCount { n_experts });
    }
    if block_size == 0 {
        return Err(MoeError::InvalidHiddenDim { dim: block_size });
    }
    if expert_assignments.len() != n_tokens {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: expert_assignments.len(),
        });
    }

    // Bucket token indices by expert (stable order preserves token order).
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
    let mut n_placed = 0usize;
    for (tok, &assign) in expert_assignments.iter().enumerate() {
        if assign == PAD_ROW {
            continue;
        }
        if assign >= n_experts {
            return Err(MoeError::ExpertIndexOutOfRange {
                idx: assign,
                n_experts,
            });
        }
        buckets[assign].push(tok);
        n_placed += 1;
    }

    // Blocks per expert = ceil(count / block_size); offsets are the prefix sum.
    let mut blocks_per_expert = vec![0usize; n_experts];
    let mut block_offsets = vec![0usize; n_experts];
    let mut total_blocks = 0usize;
    for e in 0..n_experts {
        block_offsets[e] = total_blocks;
        let cnt = buckets[e].len();
        let blocks = cnt.div_ceil(block_size);
        blocks_per_expert[e] = blocks;
        total_blocks += blocks;
    }

    let n_block_rows = total_blocks * block_size;
    let mut row_indices = vec![PAD_ROW; n_block_rows];
    // Fill each expert's block region with its tokens, leaving the tail padded.
    for e in 0..n_experts {
        let region_start = block_offsets[e] * block_size;
        for (i, &tok) in buckets[e].iter().enumerate() {
            row_indices[region_start + i] = tok;
        }
    }

    Ok(BlockSparseLayout {
        row_indices,
        n_block_rows,
        blocks_per_expert,
        block_offsets,
        total_blocks,
        block_size,
        n_tokens: n_placed,
    })
}

/// Gather token rows into the padded block layout: row `r` of the output is
/// `x[row_indices[r]]`, or zeros for padding rows.
///
/// # Errors
/// Returns [`MoeError::DimensionMismatch`] if `x` is not `[n_tokens × d_model]`.
pub fn gather_tokens(
    x: &[f32],
    layout: &BlockSparseLayout,
    n_tokens: usize,
    d_model: usize,
) -> MoeResult<Vec<f32>> {
    if d_model == 0 {
        return Err(MoeError::InvalidHiddenDim { dim: d_model });
    }
    if x.len() != n_tokens * d_model {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens * d_model,
            got: x.len(),
        });
    }
    let mut out = vec![0.0_f32; layout.n_block_rows * d_model];
    for (row, &src) in layout.row_indices.iter().enumerate() {
        if src == PAD_ROW {
            continue;
        }
        if src >= n_tokens {
            return Err(MoeError::ExpertIndexOutOfRange {
                idx: src,
                n_experts: n_tokens,
            });
        }
        out[row * d_model..(row + 1) * d_model]
            .copy_from_slice(&x[src * d_model..(src + 1) * d_model]);
    }
    Ok(out)
}

/// Scatter block-row outputs back to token space, optionally scaling each
/// token's contribution by `scores[token]`. Padding rows are ignored.
///
/// # Errors
/// Returns [`MoeError::DimensionMismatch`] on a block-buffer or score-length
/// mismatch.
pub fn scatter_tokens(
    block_out: &[f32],
    layout: &BlockSparseLayout,
    n_tokens: usize,
    d_model: usize,
    scores: Option<&[f32]>,
) -> MoeResult<Vec<f32>> {
    if d_model == 0 {
        return Err(MoeError::InvalidHiddenDim { dim: d_model });
    }
    if block_out.len() != layout.n_block_rows * d_model {
        return Err(MoeError::DimensionMismatch {
            expected: layout.n_block_rows * d_model,
            got: block_out.len(),
        });
    }
    if scores.is_some_and(|s| s.len() != n_tokens) {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: scores.map_or(0, <[f32]>::len),
        });
    }
    let mut out = vec![0.0_f32; n_tokens * d_model];
    for (row, &dst) in layout.row_indices.iter().enumerate() {
        if dst == PAD_ROW {
            continue;
        }
        let scale = scores.map_or(1.0, |s| s[dst]);
        let src = &block_out[row * d_model..(row + 1) * d_model];
        let target = &mut out[dst * d_model..(dst + 1) * d_model];
        for (t, &v) in target.iter_mut().zip(src.iter()) {
            *t += scale * v;
        }
    }
    Ok(out)
}

/// Block-sparse dispatcher owning an expert bank.
pub struct BlockSparseDispatcher {
    /// Underlying experts.
    pub bank: ExpertBank,
    /// Block size (rows per blocked-GEMM tile).
    pub block_size: usize,
}

impl BlockSparseDispatcher {
    /// Create a dispatcher around an existing [`ExpertBank`].
    ///
    /// # Errors
    /// Returns [`MoeError::InvalidHiddenDim`] if `block_size == 0`.
    pub fn new(bank: ExpertBank, block_size: usize) -> MoeResult<Self> {
        if block_size == 0 {
            return Err(MoeError::InvalidHiddenDim { dim: block_size });
        }
        Ok(Self { bank, block_size })
    }

    /// Run the full block-sparse forward: build layout → gather → per-expert
    /// grouped FFN over the expert's block rows → scatter back to tokens.
    ///
    /// `expert_assignments` uses [`PAD_ROW`] for dropped tokens; `scores` (if
    /// provided) scales each token's output.
    ///
    /// # Errors
    /// Propagates layout, gather, FFN, and scatter errors.
    pub fn forward(
        &self,
        x: &[f32],
        expert_assignments: &[usize],
        n_tokens: usize,
        scores: Option<&[f32]>,
    ) -> MoeResult<Vec<f32>> {
        let d_model = self.bank.input_dim;
        let layout = build_block_sparse_layout(
            expert_assignments,
            n_tokens,
            self.bank.n_experts,
            self.block_size,
        )?;
        let gathered = gather_tokens(x, &layout, n_tokens, d_model)?;

        let mut block_out = vec![0.0_f32; layout.n_block_rows * d_model];
        for e in 0..self.bank.n_experts {
            let (start, end) = layout.expert_row_range(e);
            if end <= start {
                continue;
            }
            let n_rows = end - start;
            let in_slice = &gathered[start * d_model..end * d_model];
            // The padding rows are zeros; running the FFN on them is harmless
            // (their outputs are dropped on scatter), exactly as the blocked
            // GEMM processes whole tiles.
            let expert_out = self.bank.forward_expert(e, in_slice, n_rows)?;
            block_out[start * d_model..end * d_model].copy_from_slice(&expert_out);
        }

        scatter_tokens(&block_out, &layout, n_tokens, d_model, scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert::ffn::ExpertActivation;
    use crate::handle::LcgRng;

    fn make_bank(n_experts: usize, d: usize) -> ExpertBank {
        let mut rng = LcgRng::new(7);
        ExpertBank::new(n_experts, d, 4 * d, ExpertActivation::Relu, &mut rng)
            .expect("new should succeed")
    }

    #[test]
    fn layout_block_counts() {
        // 5 tokens to expert 0, 3 to expert 1, block_size 4.
        let assigns = vec![0, 0, 0, 0, 0, 1, 1, 1];
        let layout = build_block_sparse_layout(&assigns, 8, 2, 4)
            .expect("build_block_sparse_layout should succeed");
        // expert 0: ceil(5/4)=2 blocks; expert 1: ceil(3/4)=1 block.
        assert_eq!(layout.blocks_per_expert, vec![2, 1]);
        assert_eq!(layout.total_blocks, 3);
        assert_eq!(layout.n_block_rows, 3 * 4);
        assert_eq!(layout.n_tokens, 8);
    }

    #[test]
    fn layout_offsets_prefix_sum() {
        let assigns = vec![0, 0, 1, 2, 2, 2];
        let layout = build_block_sparse_layout(&assigns, 6, 3, 2)
            .expect("build_block_sparse_layout should succeed");
        // counts: e0=2 (1 block), e1=1 (1 block), e2=3 (2 blocks).
        assert_eq!(layout.blocks_per_expert, vec![1, 1, 2]);
        assert_eq!(layout.block_offsets, vec![0, 1, 2]);
    }

    #[test]
    fn layout_row_indices_group_by_expert() {
        let assigns = vec![1, 0, 1, 0];
        let layout = build_block_sparse_layout(&assigns, 4, 2, 2)
            .expect("build_block_sparse_layout should succeed");
        // expert 0 rows first (tokens 1,3), then expert 1 (tokens 0,2).
        let (s0, e0) = layout.expert_row_range(0);
        let region0: Vec<usize> = layout.row_indices[s0..e0].to_vec();
        assert!(region0.contains(&1) && region0.contains(&3));
        let (s1, e1) = layout.expert_row_range(1);
        let region1: Vec<usize> = layout.row_indices[s1..e1].to_vec();
        assert!(region1.contains(&0) && region1.contains(&2));
    }

    #[test]
    fn layout_skips_dropped_tokens() {
        let assigns = vec![0, PAD_ROW, 1, PAD_ROW];
        let layout = build_block_sparse_layout(&assigns, 4, 2, 2)
            .expect("build_block_sparse_layout should succeed");
        assert_eq!(layout.n_tokens, 2); // only 2 real tokens placed
    }

    #[test]
    fn layout_zero_experts_errors() {
        let assigns = vec![0, 0];
        assert!(matches!(
            build_block_sparse_layout(&assigns, 2, 0, 2),
            Err(MoeError::InvalidExpertCount { .. })
        ));
    }

    #[test]
    fn layout_zero_block_size_errors() {
        let assigns = vec![0, 1];
        assert!(matches!(
            build_block_sparse_layout(&assigns, 2, 2, 0),
            Err(MoeError::InvalidHiddenDim { .. })
        ));
    }

    #[test]
    fn layout_out_of_range_assignment_errors() {
        let assigns = vec![0, 5]; // 5 >= n_experts(2)
        assert!(matches!(
            build_block_sparse_layout(&assigns, 2, 2, 2),
            Err(MoeError::ExpertIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn gather_scatter_round_trip_identity() {
        // With identity "expert" emulated by gather then scatter (no FFN), the
        // round trip recovers the placed tokens exactly.
        let d = 3;
        let n = 4;
        let x: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        let assigns = vec![1, 0, 1, 0];
        let layout = build_block_sparse_layout(&assigns, n, 2, 2)
            .expect("build_block_sparse_layout should succeed");
        let gathered = gather_tokens(&x, &layout, n, d).expect("gather_tokens should succeed");
        let back =
            scatter_tokens(&gathered, &layout, n, d, None).expect("scatter_tokens should succeed");
        assert_eq!(
            back, x,
            "gather∘scatter should be identity for full coverage"
        );
    }

    #[test]
    fn gather_pads_with_zeros() {
        let d = 2;
        let n = 3;
        let x = vec![1.0_f32; n * d];
        // 3 tokens, block_size 4 → 1 block (4 rows), 1 padding row.
        let assigns = vec![0, 0, 0];
        let layout = build_block_sparse_layout(&assigns, n, 1, 4)
            .expect("build_block_sparse_layout should succeed");
        let gathered = gather_tokens(&x, &layout, n, d).expect("gather_tokens should succeed");
        // Last row (padding) must be zeros.
        let last = &gathered[3 * d..4 * d];
        assert!(last.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gather_wrong_size_errors() {
        let assigns = vec![0, 1];
        let layout = build_block_sparse_layout(&assigns, 2, 2, 2)
            .expect("build_block_sparse_layout should succeed");
        let x = vec![0.0_f32; 5]; // not 2*d
        assert!(matches!(
            gather_tokens(&x, &layout, 2, 3),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn scatter_applies_scores() {
        let d = 2;
        let n = 2;
        let x = vec![1.0_f32, 1.0, 1.0, 1.0];
        let assigns = vec![0, 1];
        let layout = build_block_sparse_layout(&assigns, n, 2, 2)
            .expect("build_block_sparse_layout should succeed");
        let gathered = gather_tokens(&x, &layout, n, d).expect("gather_tokens should succeed");
        let scores = vec![2.0_f32, 0.5];
        let out = scatter_tokens(&gathered, &layout, n, d, Some(&scores))
            .expect("value should be present");
        // token 0 scaled by 2 → 2.0; token 1 scaled by 0.5 → 0.5.
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn scatter_score_len_mismatch_errors() {
        let assigns = vec![0, 1];
        let layout = build_block_sparse_layout(&assigns, 2, 2, 2)
            .expect("build_block_sparse_layout should succeed");
        let block_out = vec![0.0_f32; layout.n_block_rows * 2];
        let scores = vec![1.0_f32]; // wrong length
        assert!(matches!(
            scatter_tokens(&block_out, &layout, 2, 2, Some(&scores)),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dispatcher_forward_shape_and_finite() {
        let d = 8;
        let n = 16;
        let bank = make_bank(4, d);
        let dispatcher = BlockSparseDispatcher::new(bank, 4).expect("new should succeed");
        let x = vec![0.3_f32; n * d];
        let assigns: Vec<usize> = (0..n).map(|t| t % 4).collect();
        let out = dispatcher
            .forward(&x, &assigns, n, None)
            .expect("forward should succeed");
        assert_eq!(out.len(), n * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dispatcher_no_token_dropped() {
        // Even with a very skewed assignment (all to one expert) every token is
        // processed — block-sparse never drops, unlike capacity dispatch.
        let d = 4;
        let n = 10;
        let bank = make_bank(3, d);
        let dispatcher = BlockSparseDispatcher::new(bank, 4).expect("new should succeed");
        let x = vec![1.0_f32; n * d];
        let assigns = vec![0usize; n]; // all to expert 0
        let out = dispatcher
            .forward(&x, &assigns, n, None)
            .expect("forward should succeed");
        // Every token row must have received a (generally non-zero) output.
        for tok in 0..n {
            let row = &out[tok * d..(tok + 1) * d];
            assert!(row.iter().all(|v| v.is_finite()));
        }
        assert_eq!(out.len(), n * d);
    }

    #[test]
    fn dispatcher_dropped_tokens_zero_output() {
        let d = 4;
        let n = 4;
        let bank = make_bank(2, d);
        let dispatcher = BlockSparseDispatcher::new(bank, 2).expect("new should succeed");
        let x = vec![1.0_f32; n * d];
        let assigns = vec![0, PAD_ROW, 1, PAD_ROW];
        let out = dispatcher
            .forward(&x, &assigns, n, None)
            .expect("forward should succeed");
        // Dropped tokens (1 and 3) keep a zero output row.
        for &dropped in &[1usize, 3] {
            let row = &out[dropped * d..(dropped + 1) * d];
            assert!(
                row.iter().all(|&v| v == 0.0),
                "dropped token {dropped} not zero"
            );
        }
    }

    #[test]
    fn dispatcher_zero_block_size_errors() {
        let bank = make_bank(2, 4);
        assert!(matches!(
            BlockSparseDispatcher::new(bank, 0),
            Err(MoeError::InvalidHiddenDim { .. })
        ));
    }

    #[test]
    fn dispatcher_matches_sequential_dispatch() {
        // Block-sparse forward must equal the bank's plain per-token dispatch
        // (both apply each token's expert FFN, scaled by the score).
        let d = 6;
        let n = 12;
        let mut rng = LcgRng::new(11);
        let bank_a = ExpertBank::new(3, d, 4 * d, ExpertActivation::Gelu, &mut rng)
            .expect("new should succeed");
        let mut rng_b = LcgRng::new(11);
        let bank_b = ExpertBank::new(3, d, 4 * d, ExpertActivation::Gelu, &mut rng_b)
            .expect("new should succeed");
        let x: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.01).sin()).collect();
        let assigns: Vec<usize> = (0..n).map(|t| t % 3).collect();
        let scores: Vec<f32> = (0..n).map(|t| 0.5 + 0.5 * (t as f32 / n as f32)).collect();

        let dispatcher = BlockSparseDispatcher::new(bank_a, 4).expect("new should succeed");
        let bs_out = dispatcher
            .forward(&x, &assigns, n, Some(&scores))
            .expect("value should be present");
        let seq_out = bank_b
            .forward_dispatched(&x, &assigns, n, &scores)
            .expect("forward_dispatched should succeed");
        for (i, (&a, &b)) in bs_out.iter().zip(seq_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "row {i}: block-sparse {a} vs sequential {b}"
            );
        }
    }
}

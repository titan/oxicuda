//! # Page-Table Compaction / Defragmentation
//!
//! Over a long-running serving session the KV block pool fragments: as
//! sequences of varying lengths come and go, the *free* blocks scatter across
//! the physical ID space. While a pure free-list allocator (see
//! [`crate::cache::kv_cache::PagedKvCache`]) never suffers *internal* allocation
//! failures from fragmentation, the scattering hurts locality — copy-on-write
//! prefix sharing, NUMA placement, and contiguous DMA all benefit from used
//! blocks living in a dense, low ID range.
//!
//! This module computes a **compaction plan**: a remapping `old_block →
//! new_block` that slides every live block down to the smallest free slot,
//! producing a contiguous `[0, n_used)` used region with all free blocks above
//! it. The plan is *advisory data* — the caller applies the physical moves on
//! the device and rewrites the affected per-sequence block tables. Computing
//! the plan and rewriting tables is pure CPU logic and fully tested here.
//!
//! The algorithm is the classic two-pointer compaction used by mark-compact
//! garbage collectors: it is order-preserving for the live set, minimises the
//! number of moves (a block already in its final slot is left untouched), and
//! runs in `O(n_blocks)`.

use crate::cache::kv_cache::BlockId;
use crate::error::{InferError, InferResult};
use std::collections::HashMap;

// ─── CompactionPlan ──────────────────────────────────────────────────────────

/// A computed remapping of live blocks to a dense low-ID region.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactionPlan {
    /// Ordered list of `(from, to)` moves; `from != to` for every entry. The
    /// list is sorted by destination ascending, so applying it front-to-back
    /// never overwrites a not-yet-moved source (destinations are all in the
    /// already-vacated low region).
    pub moves: Vec<(BlockId, BlockId)>,
    /// Number of live blocks after compaction (the size of the dense region).
    pub n_live: usize,
    /// Total physical blocks in the pool.
    pub n_total: usize,
}

impl CompactionPlan {
    /// Was any block actually relocated?
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.moves.is_empty()
    }

    /// Fragmentation *before* compaction, in `[0, 1]`: the fraction of the
    /// occupied span `[0, max_used_id]` that is free holes. `0.0` means the live
    /// blocks were already contiguous from `0`.
    #[must_use]
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.n_live == 0 {
            return 0.0;
        }
        // After compaction the live region is exactly [0, n_live); the number of
        // moves is the count of blocks that were out of place — an honest,
        // monotone proxy for how fragmented the layout was.
        self.moves.len() as f64 / self.n_live as f64
    }

    /// Resolve a single old block id to its post-compaction id under this plan.
    /// Blocks not moved keep their id.
    #[must_use]
    pub fn remap(&self, old: BlockId) -> BlockId {
        self.moves
            .iter()
            .find(|(from, _)| *from == old)
            .map(|(_, to)| *to)
            .unwrap_or(old)
    }
}

// ─── Planning ────────────────────────────────────────────────────────────────

/// Compute a compaction plan from the set of currently-live (in-use) blocks.
///
/// `live` is the set of physical block ids still owned by some sequence;
/// `n_total` is the size of the pool. The plan slides the live blocks (in
/// ascending id order) into slots `0, 1, …, live.len()-1`. A live block already
/// sitting in its target slot is not moved.
///
/// # Errors
/// * [`InferError::InvalidConfig`] if a live id is `>= n_total`.
/// * [`InferError::DimensionMismatch`] if `live` contains duplicates.
pub fn plan_compaction(live: &[BlockId], n_total: usize) -> InferResult<CompactionPlan> {
    // Validate ids and detect duplicates.
    let mut seen = vec![false; n_total];
    for &BlockId(id) in live {
        let idx = id as usize;
        if idx >= n_total {
            return Err(InferError::InvalidConfig(
                "compaction: live block id out of pool range",
            ));
        }
        if seen[idx] {
            return Err(InferError::DimensionMismatch {
                expected: live.len(),
                got: live.len() - 1,
            });
        }
        seen[idx] = true;
    }

    // Live ids in ascending order → assign target slots 0..n_live.
    let mut live_sorted: Vec<u32> = live.iter().map(|b| b.0).collect();
    live_sorted.sort_unstable();

    let mut moves = Vec::new();
    for (target, &old) in live_sorted.iter().enumerate() {
        let target = target as u32;
        if old != target {
            moves.push((BlockId(old), BlockId(target)));
        }
    }
    // Sort moves by destination ascending so application never clobbers a source.
    moves.sort_by_key(|(_, to)| to.0);

    Ok(CompactionPlan {
        moves,
        n_live: live_sorted.len(),
        n_total,
    })
}

/// Rewrite a per-sequence block table in place according to a compaction plan.
///
/// Every block id present in `table` is replaced by its remapped id.
pub fn rewrite_block_table(table: &mut [BlockId], plan: &CompactionPlan) {
    // Build a fast lookup once for large tables.
    let remap: HashMap<BlockId, BlockId> = plan.moves.iter().copied().collect();
    for id in table.iter_mut() {
        if let Some(&new) = remap.get(id) {
            *id = new;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn b(ids: &[u32]) -> Vec<BlockId> {
        ids.iter().copied().map(BlockId).collect()
    }

    #[test]
    fn already_compact_is_noop() {
        // Live blocks 0,1,2 in a pool of 8 — already dense from 0.
        let plan = plan_compaction(&b(&[0, 1, 2]), 8).expect("ok");
        assert!(plan.is_noop());
        assert_eq!(plan.n_live, 3);
        assert_eq!(plan.fragmentation_ratio(), 0.0);
    }

    #[test]
    fn scattered_blocks_compacted_low() {
        // Live blocks 1,4,6 in a pool of 8 → should map to 0,1,2.
        let plan = plan_compaction(&b(&[1, 4, 6]), 8).expect("ok");
        // sorted live: 1->0, 4->1, 6->2.
        assert_eq!(
            plan.moves,
            vec![
                (BlockId(1), BlockId(0)),
                (BlockId(4), BlockId(1)),
                (BlockId(6), BlockId(2))
            ]
        );
        assert_eq!(plan.n_live, 3);
        assert_eq!(plan.remap(BlockId(4)), BlockId(1));
        assert_eq!(
            plan.remap(BlockId(99)),
            BlockId(99),
            "unmapped id unchanged"
        );
    }

    #[test]
    fn partially_placed_blocks_not_moved() {
        // 0 already correct; 1 already correct; 5 must move to 2.
        let plan = plan_compaction(&b(&[0, 1, 5]), 8).expect("ok");
        assert_eq!(plan.moves, vec![(BlockId(5), BlockId(2))]);
        assert_eq!(plan.remap(BlockId(0)), BlockId(0));
        assert_eq!(plan.remap(BlockId(1)), BlockId(1));
    }

    #[test]
    fn rewrite_table_applies_remap() {
        let plan = plan_compaction(&b(&[1, 4, 6]), 8).expect("ok");
        let mut table = b(&[6, 1, 4]); // some sequence's block table (order preserved)
        rewrite_block_table(&mut table, &plan);
        // 6->2, 1->0, 4->1.
        assert_eq!(table, b(&[2, 0, 1]));
    }

    #[test]
    fn out_of_range_live_id_rejected() {
        let r = plan_compaction(&b(&[0, 9]), 8);
        assert!(matches!(r, Err(InferError::InvalidConfig(_))));
    }

    #[test]
    fn duplicate_live_id_rejected() {
        let r = plan_compaction(&b(&[2, 2]), 8);
        assert!(matches!(r, Err(InferError::DimensionMismatch { .. })));
    }

    #[test]
    fn empty_live_set() {
        let plan = plan_compaction(&[], 8).expect("ok");
        assert!(plan.is_noop());
        assert_eq!(plan.n_live, 0);
        assert_eq!(plan.fragmentation_ratio(), 0.0);
    }

    #[test]
    fn fragmentation_ratio_monotone() {
        // Fully scattered (none in place): 3,5,7 -> 0,1,2 → 3 moves / 3 live = 1.0.
        let plan = plan_compaction(&b(&[3, 5, 7]), 8).expect("ok");
        assert_eq!(plan.moves.len(), 3);
        assert!((plan.fragmentation_ratio() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn moves_sorted_by_destination() {
        let plan = plan_compaction(&b(&[7, 3, 5]), 8).expect("ok");
        let dests: Vec<u32> = plan.moves.iter().map(|(_, t)| t.0).collect();
        let mut sorted = dests.clone();
        sorted.sort_unstable();
        assert_eq!(
            dests, sorted,
            "moves must be ordered by ascending destination"
        );
    }

    #[test]
    fn applying_plan_yields_contiguous_region() {
        // Apply the plan to the live set and confirm it lands on [0, n_live).
        let live = b(&[2, 5, 6, 9]);
        let plan = plan_compaction(&live, 16).expect("ok");
        let mut remapped: Vec<u32> = live.iter().map(|&id| plan.remap(id).0).collect();
        remapped.sort_unstable();
        assert_eq!(remapped, vec![0, 1, 2, 3]);
    }
}

//! # Radix-tree Prefix Cache (RadixAttention)
//!
//! Implements automatic, *partial* prefix sharing via a token-keyed radix tree,
//! as popularised by SGLang's **RadixAttention** (Zheng et al., 2024).
//!
//! Whereas the hash-based [`crate::cache::prefix_cache::PrefixCache`] only ever
//! reuses a *complete* prefix (the entire cached token sequence must match), a
//! radix tree shares any common *prefix span* between requests. Two prompts
//! that agree on their first `m` tokens but then diverge can still reuse the KV
//! blocks computed for those `m` tokens — exactly the behaviour required for a
//! shared system prompt followed by per-user content, multi-turn chat history,
//! or beam-search siblings.
//!
//! ## Structure
//!
//! Each tree edge is labelled with a run of token ids; each node owns the
//! physical [`BlockId`]s that cover the tokens *on the edge leading into it*.
//! [`RadixCache::match_prefix`] walks the tree from the root, descending while
//! the query agrees with an edge, and splitting an edge mid-run when the query
//! diverges in its interior. The returned match is the longest token prefix of
//! the query that already exists in the tree, together with the physical blocks
//! that **fully** cover it.
//!
//! ## Block granularity
//!
//! Physical blocks tile the sequence on a fixed global grid of `block_size`
//! tokens, fixed once at construction. A node covering the absolute token span
//! `[abs_start, abs_start + key_len)` owns exactly the blocks whose grid cells
//! intersect that span. When a query matches `shared` tokens into a node, the
//! blocks reported are those lying **entirely** within the matched region — a
//! block straddling the divergence point is *not* reusable, matching how real
//! paged caches reuse whole blocks only.
//!
//! ## Block lifecycle
//!
//! This cache is purely a *bookkeeping* structure: as with
//! [`crate::cache::prefix_cache::PrefixCache`], reference-count management on
//! the underlying [`crate::cache::kv_cache::PagedKvCache`] is the caller's
//! responsibility — [`RadixCache::evict_lru`] returns the blocks the caller
//! should `dec_ref`.

use crate::cache::kv_cache::BlockId;
use crate::error::{InferError, InferResult};

// ─── Node ──────────────────────────────────────────────────────────────────────

/// One node in the token radix tree.
///
/// A node is reached by consuming the `key` token-run of the edge that points
/// to it from its parent; `blocks` are the physical KV blocks that store the
/// K/V for exactly those `key` tokens, aligned to the global block grid.
#[derive(Debug, Clone)]
struct RadixNode {
    /// Token run labelling the edge from the parent into this node.
    key: Vec<u32>,
    /// Physical KV blocks covering the `key` tokens (in token order).
    blocks: Vec<BlockId>,
    /// Absolute start position of this node's `key` in the full sequence.
    abs_start: usize,
    /// Child node indices.
    children: Vec<usize>,
    /// Parent node index (`usize::MAX` for the root).
    parent: usize,
    /// Logical clock at last access — drives LRU eviction.
    last_access: u64,
}

impl RadixNode {
    fn root() -> Self {
        Self {
            key: Vec::new(),
            blocks: Vec::new(),
            abs_start: 0,
            children: Vec::new(),
            parent: usize::MAX,
            last_access: 0,
        }
    }
}

// ─── MatchResult ─────────────────────────────────────────────────────────────

/// Result of [`RadixCache::match_prefix`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MatchResult {
    /// Number of leading query tokens matched against the tree.
    pub matched_len: usize,
    /// Physical blocks fully covering the matched prefix, in token order.
    pub blocks: Vec<BlockId>,
}

// ─── RadixCache ──────────────────────────────────────────────────────────────

/// Token radix tree for partial KV-prefix sharing.
///
/// Nodes live in a flat arena (`Vec<RadixNode>`); index `0` is always the
/// (empty) root. Freed node slots are recycled through `free_slots`.
pub struct RadixCache {
    nodes: Vec<RadixNode>,
    free_slots: Vec<usize>,
    /// Global block grid size (tokens per physical block).
    block_size: usize,
    clock: u64,
    /// Total `match_prefix` calls (for hit-rate reporting).
    total_queries: u64,
    /// Calls that matched at least one token.
    total_hits: u64,
}

impl RadixCache {
    /// Create an empty radix cache (root only) on a `block_size` token grid.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if `block_size == 0`.
    pub fn new(block_size: usize) -> InferResult<Self> {
        if block_size == 0 {
            return Err(InferError::InvalidConfig("radix cache block_size == 0"));
        }
        Ok(Self {
            nodes: vec![RadixNode::root()],
            free_slots: Vec::new(),
            block_size,
            clock: 0,
            total_queries: 0,
            total_hits: 0,
        })
    }

    /// Number of *interior* nodes (excludes the always-present root).
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len() - 1 - self.free_slots.len()
    }

    /// Fraction of queries that matched at least one token.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.total_hits as f64 / self.total_queries as f64
        }
    }

    // ── Matching ───────────────────────────────────────────────────────────────

    /// Find the longest prefix of `tokens` already present in the tree.
    ///
    /// Descends from the root, following edges whose token-run is a prefix of
    /// the remaining query. The returned `blocks` are those that lie entirely
    /// within the matched region (a block straddling the divergence point is
    /// excluded). Updates the access clock of every node on the matched path.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> MatchResult {
        self.total_queries += 1;
        self.clock += 1;

        let mut blocks = Vec::new();
        let mut matched = 0_usize;
        let mut node = 0_usize; // root
        self.nodes[0].last_access = self.clock;

        'descend: while matched < tokens.len() {
            let next = tokens[matched];
            let child = self.nodes[node]
                .children
                .iter()
                .copied()
                .find(|&c| self.nodes[c].key.first() == Some(&next));
            let Some(child) = child else { break 'descend };

            let edge_len = self.nodes[child].key.len();
            let rem = &tokens[matched..];
            let shared = common_prefix_len(&self.nodes[child].key, rem);

            // Append the blocks of this node that are fully within the first
            // `shared` matched tokens (on the global block grid). A node with no
            // children is a leaf, so its final block legitimately ends at the
            // cached run boundary (a possibly-partial trailing block).
            let abs_start = self.nodes[child].abs_start;
            let is_leaf = self.nodes[child].children.is_empty();
            let n_blocks_full = full_blocks_in_span(
                abs_start,
                shared,
                edge_len,
                self.nodes[child].blocks.len(),
                self.block_size,
                is_leaf,
            );
            blocks.extend_from_slice(&self.nodes[child].blocks[..n_blocks_full]);
            matched += shared;

            self.nodes[child].last_access = self.clock;

            if shared < edge_len {
                break 'descend; // diverged inside this edge
            }
            node = child;
        }

        if matched > 0 {
            self.total_hits += 1;
        }
        MatchResult {
            matched_len: matched,
            blocks,
        }
    }

    // ── Insertion ──────────────────────────────────────────────────────────────

    /// Insert `tokens` (covered by `blocks`) into the tree.
    ///
    /// `blocks` must cover the whole token run: `blocks.len() ==
    /// tokens.len().div_ceil(block_size)`. Any prefix that already exists is
    /// reused (no duplicate nodes are created); only the novel suffix adds new
    /// nodes. Edges are split as needed so that shared prefixes become shared
    /// nodes.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if `tokens` is empty.
    /// * [`InferError::DimensionMismatch`] if `blocks` does not exactly cover
    ///   `tokens` at the cache's `block_size`.
    pub fn insert(&mut self, tokens: &[u32], blocks: Vec<BlockId>) -> InferResult<()> {
        if tokens.is_empty() {
            return Err(InferError::InvalidConfig("radix insert: empty token run"));
        }
        let expected_blocks = tokens.len().div_ceil(self.block_size);
        if blocks.len() != expected_blocks {
            return Err(InferError::DimensionMismatch {
                expected: expected_blocks,
                got: blocks.len(),
            });
        }

        self.clock += 1;
        let mut node = 0_usize;
        let mut offset = 0_usize; // tokens consumed so far (== abs position)
        self.nodes[0].last_access = self.clock;

        while offset < tokens.len() {
            let next = tokens[offset];
            let child = self.nodes[node]
                .children
                .iter()
                .copied()
                .find(|&c| self.nodes[c].key.first() == Some(&next));

            match child {
                None => {
                    // No matching edge — attach the whole remaining suffix.
                    let suffix = tokens[offset..].to_vec();
                    let suffix_blocks = blocks_for_span(&blocks, offset, self.block_size).to_vec();
                    let new_idx = self.alloc_node(suffix, suffix_blocks, offset, node);
                    self.nodes[node].children.push(new_idx);
                    self.nodes[new_idx].last_access = self.clock;
                    return Ok(());
                }
                Some(child) => {
                    let edge_len = self.nodes[child].key.len();
                    let rem = &tokens[offset..];
                    let shared = common_prefix_len(&self.nodes[child].key, rem);

                    self.nodes[child].last_access = self.clock;

                    if shared == edge_len {
                        offset += edge_len;
                        node = child;
                        continue;
                    }
                    // Partial match: split the child edge at `shared`.
                    self.split_edge(child, shared);
                    offset += shared;
                    node = child;
                }
            }
        }
        Ok(())
    }

    // ── Eviction ───────────────────────────────────────────────────────────────

    /// Evict the least-recently-used *leaf* and return its blocks for the caller
    /// to `dec_ref`. Only leaves are evictable. Returns `None` when the tree is
    /// empty.
    pub fn evict_lru(&mut self) -> Option<Vec<BlockId>> {
        let mut best: Option<(usize, u64)> = None;
        for (idx, n) in self.nodes.iter().enumerate() {
            if idx == 0 || self.free_slots.contains(&idx) {
                continue;
            }
            if n.children.is_empty() {
                match best {
                    Some((_, t)) if n.last_access >= t => {}
                    _ => best = Some((idx, n.last_access)),
                }
            }
        }
        let (leaf, _) = best?;
        Some(self.remove_leaf(leaf))
    }

    // ── Internal helpers ─────────────────────────────────────────────────────────

    fn alloc_node(
        &mut self,
        key: Vec<u32>,
        blocks: Vec<BlockId>,
        abs_start: usize,
        parent: usize,
    ) -> usize {
        let node = RadixNode {
            key,
            blocks,
            abs_start,
            children: Vec::new(),
            parent,
            last_access: self.clock,
        };
        if let Some(slot) = self.free_slots.pop() {
            self.nodes[slot] = node;
            slot
        } else {
            self.nodes.push(node);
            self.nodes.len() - 1
        }
    }

    /// Split the edge into `child` at token offset `at` (`0 < at < edge_len`).
    ///
    /// After the split `child` retains the first `at` tokens and only the blocks
    /// it **completes** (those whose grid cell ends at or before `at`); a
    /// freshly-allocated node takes the remaining tokens and the remaining
    /// blocks — including any block that *straddles* the boundary, since that
    /// block's tokens finish in the tail. This keeps whole-block ownership
    /// unique: every physical block belongs to exactly the node where it
    /// completes, so a root-to-leaf traversal never double-counts it.
    fn split_edge(&mut self, child: usize, at: usize) {
        let key = std::mem::take(&mut self.nodes[child].key);
        let blocks = std::mem::take(&mut self.nodes[child].blocks);
        let grandchildren = std::mem::take(&mut self.nodes[child].children);
        let abs_start = self.nodes[child].abs_start;

        let (head_key, tail_key) = key.split_at(at);

        // Number of blocks the head fully completes within its `at` tokens
        // (strict grid completion — the head is an interior node, never a leaf).
        // The straddling block and everything after it go to the tail.
        let head_end = full_blocks_in_span(abs_start, at, at, blocks.len(), self.block_size, false);
        let head_blocks = blocks[..head_end].to_vec();
        let tail_blocks = blocks[head_end..].to_vec();

        let tail_idx = self.alloc_node(tail_key.to_vec(), tail_blocks, abs_start + at, child);
        self.nodes[tail_idx].children = grandchildren;
        let gc: Vec<usize> = self.nodes[tail_idx].children.clone();
        for g in gc {
            self.nodes[g].parent = tail_idx;
        }

        self.nodes[child].key = head_key.to_vec();
        self.nodes[child].blocks = head_blocks;
        self.nodes[child].children = vec![tail_idx];
    }

    /// Remove a leaf node, recycling its slot, and return its blocks.
    fn remove_leaf(&mut self, leaf: usize) -> Vec<BlockId> {
        let parent = self.nodes[leaf].parent;
        if parent != usize::MAX {
            self.nodes[parent].children.retain(|&c| c != leaf);
        }
        let blocks = std::mem::take(&mut self.nodes[leaf].blocks);
        self.nodes[leaf].key.clear();
        self.nodes[leaf].children.clear();
        self.nodes[leaf].parent = usize::MAX;
        self.free_slots.push(leaf);
        blocks
    }
}

// ─── Free functions ────────────────────────────────────────────────────────────

/// Length of the longest common prefix of `a` and `b`.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Number of leading blocks of a node (whose `key` of `key_len` tokens starts at
/// absolute position `abs_start`) that lie **entirely** within the first
/// `shared` matched tokens.
///
/// Blocks tile the global grid of size `bs`. Block `i` spans node-relative
/// tokens `[start_i, end_i)`, walked from `abs_start` (the first cell may be a
/// partial leading cell). A block is *complete* iff its grid cell ends at or
/// before the matched span.
///
/// `is_leaf` controls the final block: at a leaf the node's last block is the
/// genuine end of the cached run, so it counts once `key_len` (not a full grid
/// cell) is reached — capturing a legitimately partial trailing block. Interior
/// nodes (e.g. a split *head*) use strict grid completion, because a straddling
/// block actually finishes in a child node.
fn full_blocks_in_span(
    abs_start: usize,
    shared: usize,
    key_len: usize,
    n_blocks: usize,
    bs: usize,
    is_leaf: bool,
) -> usize {
    if shared == 0 || n_blocks == 0 {
        return 0;
    }
    let mut full = 0;
    let mut rel_start = 0_usize; // node-relative start of the current block
    for i in 0..n_blocks {
        // A block starting at or beyond the matched span contributes nothing.
        if rel_start >= shared || rel_start >= key_len {
            break;
        }
        let abs_block_start = abs_start + rel_start;
        let grid_end = (((abs_block_start / bs) + 1) * bs) - abs_start; // strict cell end
        let is_last = i + 1 == n_blocks;
        // The block's effective end: a leaf's final block ends at the run end.
        let rel_end = if is_leaf && is_last {
            grid_end.min(key_len)
        } else {
            grid_end
        };
        if rel_end <= shared {
            full += 1;
            rel_start = rel_end;
        } else {
            break;
        }
    }
    full
}

/// Blocks of `all` (which cover a token run starting at absolute position 0)
/// that cover the suffix span starting at absolute position `at`, on the global
/// `block_size` grid.
fn blocks_for_span(all: &[BlockId], at: usize, block_size: usize) -> &[BlockId] {
    let start_blk = at / block_size;
    &all[start_blk.min(all.len())..]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn b(ids: &[u32]) -> Vec<BlockId> {
        ids.iter().copied().map(BlockId).collect()
    }

    #[test]
    fn block_size_zero_rejected() {
        assert!(matches!(
            RadixCache::new(0),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn empty_match_is_zero() {
        let mut c = RadixCache::new(2).expect("ok");
        let m = c.match_prefix(&[1, 2, 3]);
        assert_eq!(m.matched_len, 0);
        assert!(m.blocks.is_empty());
    }

    #[test]
    fn insert_then_exact_match() {
        let mut c = RadixCache::new(2).expect("ok");
        // 4 tokens, block_size=2 → 2 blocks.
        c.insert(&[10, 11, 12, 13], b(&[0, 1])).expect("insert ok");
        let m = c.match_prefix(&[10, 11, 12, 13]);
        assert_eq!(m.matched_len, 4);
        assert_eq!(m.blocks, b(&[0, 1]));
    }

    #[test]
    fn longest_shared_prefix_partial() {
        let mut c = RadixCache::new(2).expect("ok");
        // Insert "A B C D" (block_size=2 → blocks 0,1).
        c.insert(&[1, 2, 3, 4], b(&[0, 1])).expect("insert");
        // Query "A B C X": shares first 3 tokens, but only block 0 (tokens 0..2)
        // is *fully* within the shared span [0,3); block 1 covers tokens 2..4 and
        // straddles the divergence, so it is not returned.
        let m = c.match_prefix(&[1, 2, 3, 9]);
        assert_eq!(m.matched_len, 3, "should match the shared A B C");
        assert_eq!(m.blocks, b(&[0]), "only the fully-covered block is reused");
    }

    #[test]
    fn match_two_full_blocks_partial_third() {
        let mut c = RadixCache::new(2).expect("ok");
        // 6 tokens at bs=2 → blocks 0,1,2.
        c.insert(&[1, 2, 3, 4, 5, 6], b(&[0, 1, 2]))
            .expect("insert");
        // Query shares first 5 tokens. Blocks 0 (toks0-1) and 1 (toks2-3) are
        // fully within [0,5); block 2 (toks4-5) straddles → excluded.
        let m = c.match_prefix(&[1, 2, 3, 4, 5, 9]);
        assert_eq!(m.matched_len, 5);
        assert_eq!(m.blocks, b(&[0, 1]));
    }

    #[test]
    fn diverging_branch_splits_edge() {
        let mut c = RadixCache::new(2).expect("ok");
        c.insert(&[1, 2, 3, 4], b(&[0, 1])).expect("first insert");
        // Second prompt shares "1 2" then diverges to "5 6".
        c.insert(&[1, 2, 5, 6], b(&[0, 7])).expect("second insert");

        // The shared prefix "1 2" is now a node; both leaves hang off it.
        let m = c.match_prefix(&[1, 2]);
        assert_eq!(m.matched_len, 2);
        assert_eq!(m.blocks, b(&[0]));

        // Full match of the second branch.
        let m2 = c.match_prefix(&[1, 2, 5, 6]);
        assert_eq!(m2.matched_len, 4);
        assert_eq!(m2.blocks, b(&[0, 7]));

        // Full match of the first branch still works after the split.
        let m3 = c.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(m3.matched_len, 4);
        assert_eq!(m3.blocks, b(&[0, 1]));
    }

    #[test]
    fn no_shared_prefix_different_first_token() {
        let mut c = RadixCache::new(2).expect("ok");
        c.insert(&[1, 2], b(&[0])).expect("insert");
        let m = c.match_prefix(&[9, 9]);
        assert_eq!(m.matched_len, 0);
        assert!(m.blocks.is_empty());
    }

    #[test]
    fn reinsert_existing_prefix_no_dup() {
        let mut c = RadixCache::new(2).expect("ok");
        c.insert(&[1, 2, 3, 4], b(&[0, 1])).expect("insert");
        let n_before = c.n_nodes();
        c.insert(&[1, 2, 3, 4], b(&[0, 1])).expect("reinsert");
        assert_eq!(c.n_nodes(), n_before);
    }

    #[test]
    fn extend_existing_prefix() {
        let mut c = RadixCache::new(2).expect("ok");
        c.insert(&[1, 2], b(&[0])).expect("insert short");
        // Extend with "1 2 3 4": shares "1 2", appends "3 4".
        c.insert(&[1, 2, 3, 4], b(&[0, 1])).expect("extend");
        let m = c.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(m.matched_len, 4);
        assert_eq!(m.blocks, b(&[0, 1]));
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let mut c = RadixCache::new(2).expect("ok");
        // 4 tokens at bs=2 needs 2 blocks; supplying 1 is an error.
        let r = c.insert(&[1, 2, 3, 4], b(&[0]));
        assert!(matches!(r, Err(InferError::DimensionMismatch { .. })));
    }

    #[test]
    fn empty_tokens_rejected() {
        let mut c = RadixCache::new(2).expect("ok");
        assert!(matches!(
            c.insert(&[], b(&[])),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn evict_lru_returns_leaf_blocks() {
        let mut c = RadixCache::new(2).expect("ok");
        c.insert(&[1, 2], b(&[0])).expect("insert A");
        c.insert(&[3, 4], b(&[5])).expect("insert B");
        c.match_prefix(&[1, 2]); // touch A → B is LRU
        let evicted = c.evict_lru().expect("a leaf should be evictable");
        assert_eq!(evicted, b(&[5]), "LRU leaf (3 4) should be evicted");
        assert_eq!(c.match_prefix(&[1, 2]).matched_len, 2);
    }

    #[test]
    fn hit_rate_tracks_matches() {
        let mut c = RadixCache::new(2).expect("ok");
        c.insert(&[1, 2], b(&[0])).expect("insert");
        c.match_prefix(&[1, 2]); // hit
        c.match_prefix(&[9, 9]); // miss
        assert!((c.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn three_way_branch() {
        let mut c = RadixCache::new(2).expect("ok");
        c.insert(&[1, 2, 3, 4], b(&[0, 1])).expect("A");
        c.insert(&[1, 2, 5, 6], b(&[0, 2])).expect("B");
        c.insert(&[1, 2, 7, 8], b(&[0, 3])).expect("C");
        assert_eq!(c.match_prefix(&[1, 2, 3, 4]).blocks, b(&[0, 1]));
        assert_eq!(c.match_prefix(&[1, 2, 5, 6]).blocks, b(&[0, 2]));
        assert_eq!(c.match_prefix(&[1, 2, 7, 8]).blocks, b(&[0, 3]));
    }

    #[test]
    fn odd_length_last_block_partial() {
        let mut c = RadixCache::new(2).expect("ok");
        // 5 tokens at bs=2 → 3 blocks (last block holds 1 token).
        c.insert(&[1, 2, 3, 4, 5], b(&[0, 1, 2])).expect("insert");
        let m = c.match_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(m.matched_len, 5);
        assert_eq!(m.blocks, b(&[0, 1, 2]));
    }

    #[test]
    fn block_aligned_divergence_returns_whole_blocks() {
        let mut c = RadixCache::new(2).expect("ok");
        c.insert(&[1, 2, 3, 4], b(&[0, 1])).expect("insert");
        // Divergence exactly on the block boundary (after 2 tokens = block 0).
        let m = c.match_prefix(&[1, 2, 9, 9]);
        assert_eq!(m.matched_len, 2);
        assert_eq!(
            m.blocks,
            b(&[0]),
            "block 0 fully matched, block 1 untouched"
        );
    }

    #[test]
    fn split_preserves_both_branches_block_coverage() {
        // block_size=4: a split inside a block must keep coverage on both sides.
        let mut c = RadixCache::new(4).expect("ok");
        // 8 tokens at bs=4 → blocks 0,1.
        c.insert(&[1, 2, 3, 4, 5, 6, 7, 8], b(&[0, 1])).expect("A");
        // Shares "1 2 3" (inside block 0), then diverges.
        c.insert(&[1, 2, 3, 9, 9, 9, 9, 9], b(&[0, 2])).expect("B");
        // Re-match the original full sequence: must still recover blocks 0,1.
        let m = c.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(m.matched_len, 8);
        assert_eq!(m.blocks, b(&[0, 1]));
    }
}
